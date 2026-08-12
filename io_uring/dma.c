// SPDX-License-Identifier: GPL-2.0

#include <linux/io_uring_types.h>
#include <linux/io_uring.h>
#include <linux/delay.h>
#include <linux/uio.h>
#include <linux/dma-mapping.h>
#include <linux/spinlock.h>
#include <linux/xarray.h>
#include <linux/pagemap.h>
#include <linux/folio_batch.h>
#include <linux/swap.h>
#include <linux/fs.h>
#include <linux/scatterlist.h>
#include <linux/timekeeping.h>
#include <linux/debugfs.h>
#include <linux/seq_file.h>
#include <linux/kthread.h>
#include <linux/pm_qos.h>
#include <linux/task_work.h>
#include "io_uring.h"
#include "rsrc.h"
#include "refs.h"

#ifndef pr_fmt
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt
#endif

struct kmem_cache *dma_cachep;

/*
 * DSA cache control: when enabled (1, the default), destination writes are
 * cache-allocating (IDXD_OP_FLAG_CC) so data the application reads back
 * immediately is warm in cache; when disabled (0), writes bypass cache.
 * Configurable via /proc/sys/kernel/io_uring_dma_cache_control.
 */
unsigned int io_dma_cache_control __read_mostly = 1;

static inline unsigned long io_dma_prep_flags(void)
{
	return READ_ONCE(io_dma_cache_control) ? DMA_PREP_CACHE_CONTROL : 0;
}

/*
 * PFN-keyed persistent source-mapping cache.
 *
 * Under translated IOMMU domains the dominant recoverable submit cost
 * on the DMA source paths is per-chunk dma_map/dma_unmap (measured via
 * the temporary iommu=pt experiment: buffered reads 10.7 -> 53.7 GB/s).
 * Destinations are already persistent (registered buffers map once at
 * registration); this gives sources the same discipline, lazily: the
 * first chunk touching a folio maps its io_dma_map_quantum()-sized
 * segment DMA_TO_DEVICE and caches {segment-head PFN -> dma_addr} in a
 * per-device xarray; later chunks on that segment are pure arithmetic.
 * dma_unmap runs only on CLOCK eviction (bytes-capped) and explicit
 * flush.
 *
 * Correctness: struct page <-> physical address is immutable, and
 * descriptors are only issued against pages held live by the I/O being
 * processed (folio ref on filemap), so a stale cached
 * translation can never misdirect DMA.  Staleness costs IOVA space and
 * a lingering device *read* window, bounded by the cap -- the trade
 * page_pool's persistent NIC mappings also make.
 *
 * Entry lifetime: refs = 1 cache bias + one per in-flight batch entry.
 * Lookup takes a ref with atomic_inc_not_zero() under RCU; eviction
 * erases the entry and drops the bias, so a mapping survives until its
 * last in-flight user completes.
 *
 * Mappings are keyed to the DSA struct device and are not torn down on
 * driver unbind -- flush via debugfs before unbinding idxd.
 */

struct io_pfn_map {
	unsigned long		pfn;		/* segment-head PFN: cache key */
	dma_addr_t		dma_base;	/* segment DMA_TO_DEVICE mapping */
	unsigned int		size;		/* mapped bytes (<= cache quantum) */
	atomic_t		refs;		/* cache bias + in-flight users */
	bool			referenced;	/* CLOCK second-chance bit */
	struct device		*dev;		/* unmap handle */
	struct rcu_head		rcu;
};

struct io_pfn_cache {
	struct xarray		xa;
	spinlock_t		lock;		/* serializes the CLOCK sweep */
	unsigned long		hand;		/* next PFN the sweep visits */
	struct device		*dev;
	size_t			quantum;	/* pow2 segment size entries are
						 * carved into (io_dma_map_quantum) */
	atomic64_t		covered;	/* bytes mapped through the tree */
	atomic64_t		hits;
	atomic64_t		misses;
	atomic64_t		inserts;
	atomic64_t		insert_fails;	/* alloc/map/xa failure -> plain map */
	atomic64_t		range_fallbacks;/* chunk not coverable by one entry */
	atomic64_t		evictions;
	atomic64_t		ref_skips;	/* sweep passed an in-flight entry */
};

#define IO_PFN_CACHE_DEVS	16
static struct io_pfn_cache *io_pfn_caches[IO_PFN_CACHE_DEVS];
static struct device *io_pfn_cache_devs[IO_PFN_CACHE_DEVS];
static DEFINE_SPINLOCK(io_pfn_cache_reg_lock);

/*
 * Covered-bytes cap in MiB; 0 disables the cache entirely.  Generous by
 * default: oversizing only widens the exposure window, while a cap under
 * a cycling working set makes the sweep thrash (evict+remap per I/O,
 * worse than no cache).  Size it above the source working set -- the
 * hot file set for reads.
 */
static u32 io_dma_pfn_cache_cap_mb __read_mostly = 4096;

/* Sweep visit budget per eviction call: bounds datapath latency when the
 * table is large and mostly referenced/in-flight (soft cap). */
#define IO_PFN_EVICT_BUDGET	1024

/*
 * Standing mappings are carved into power-of-two segments no larger than
 * dma_opt_mapping_size(): IOVA allocations above that limit bypass the
 * IOMMU's per-CPU rcaches and fall to the domain rbtree under its lock
 * (measured at 12-48% of node cycles under shuffle load when whole 2MB
 * folios were mapped as single entries).  With the stock rcache ceiling
 * this yields 128KB segments; capped at 2MB so a no-IOMMU SIZE_MAX
 * answer degenerates to whole-folio behaviour.
 */
static size_t io_dma_map_quantum(struct device *dev)
{
	size_t q = dma_opt_mapping_size(dev);

	if (!q || q > SZ_2M)
		q = SZ_2M;
	return rounddown_pow_of_two(q);
}

static struct io_pfn_cache *io_pfn_cache_get(struct device *dev)
{
	struct io_pfn_cache *c;
	int i;

	for (i = 0; i < IO_PFN_CACHE_DEVS; i++) {
		if (smp_load_acquire(&io_pfn_cache_devs[i]) == dev)
			return io_pfn_caches[i];
		if (!READ_ONCE(io_pfn_cache_devs[i]))
			break;
	}

	c = kzalloc(sizeof(*c), GFP_NOWAIT | __GFP_NOWARN);
	if (!c)
		return NULL;
	xa_init(&c->xa);
	spin_lock_init(&c->lock);
	c->dev = dev;
	c->quantum = io_dma_map_quantum(dev);

	spin_lock(&io_pfn_cache_reg_lock);
	for (i = 0; i < IO_PFN_CACHE_DEVS; i++) {
		if (io_pfn_cache_devs[i] == dev) {	/* insert race */
			spin_unlock(&io_pfn_cache_reg_lock);
			kfree(c);
			return io_pfn_caches[i];
		}
		if (!io_pfn_cache_devs[i]) {
			io_pfn_caches[i] = c;
			/* pairs with the lockless load above */
			smp_store_release(&io_pfn_cache_devs[i], dev);
			spin_unlock(&io_pfn_cache_reg_lock);
			return c;
		}
	}
	spin_unlock(&io_pfn_cache_reg_lock);
	kfree(c);	/* registry full: this device runs uncached */
	return NULL;
}

/* Drop one reference; the last dropper unmaps and frees. Safe from any
 * context (IRQ-mode completions run in dmaengine callback context). */
static void io_pfn_map_put(struct io_pfn_map *pm)
{
	if (!atomic_dec_and_test(&pm->refs))
		return;
	dma_unmap_page(pm->dev, pm->dma_base, pm->size, DMA_TO_DEVICE);
	kfree_rcu(pm, rcu);
}

/*
 * CLOCK sweep: advance the hand from where it last stopped, giving
 * referenced entries a second chance (scan resistance: a streaming
 * pattern cannot flush the recycling working set, whose entries keep
 * their bit set) and skipping entries with in-flight users.  Runs on
 * the submit path after an insert pushes covered past the cap, so both
 * the trylock (another submitter is already sweeping) and the visit
 * budget bound the added latency.
 */
static void io_pfn_cache_evict(struct io_pfn_cache *c, u64 cap)
{
	struct io_pfn_map *pm;
	unsigned long index;
	int budget = IO_PFN_EVICT_BUDGET;
	int pass;

	if (!spin_trylock(&c->lock))
		return;

	for (pass = 0; pass < 2 && atomic64_read(&c->covered) > cap; pass++) {
		unsigned long start = pass ? 0 : c->hand;

		xa_for_each_start(&c->xa, index, pm, start) {
			if (--budget <= 0) {
				c->hand = index + 1;
				goto out;
			}
			if (READ_ONCE(pm->referenced)) {
				WRITE_ONCE(pm->referenced, false);
			} else if (atomic_read(&pm->refs) > 1) {
				atomic64_inc(&c->ref_skips);
			} else {
				xa_erase(&c->xa, index);
				atomic64_sub(pm->size, &c->covered);
				atomic64_inc(&c->evictions);
				io_pfn_map_put(pm);	/* drop cache bias */
				if (atomic64_read(&c->covered) <= cap) {
					c->hand = index + 1;
					goto out;
				}
			}
		}
		c->hand = 0;
	}
out:
	spin_unlock(&c->lock);
}

/*
 * Look up (or create) the persistent mapping covering [offset, offset+len)
 * from the head of @folio and return it with an in-flight reference taken;
 * *dma is set to the chunk's device address.  NULL means the caller should
 * fall back to a plain per-chunk map (never an error).
 *
 * Entries are quantized: the folio (or run) is carved into c->quantum
 * segments from its head, each cached and mapped independently, so every
 * IOVA allocation stays inside the IOMMU's per-CPU rcache size classes.
 * The chunk must lie within one segment -- the filemap-read path splits
 * chunks at segment boundaries to guarantee it; a chunk that crosses
 * one falls back to a plain map.
 *
 * @map_len is the known-physically-contiguous extent from the folio
 * head: folio_size() for page-cache folios.
 */
static struct io_pfn_map *io_pfn_map_lookup(struct io_pfn_cache *c,
					    struct folio *folio,
					    size_t offset, size_t len,
					    size_t map_len, dma_addr_t *dma)
{
	u64 cap = (u64)READ_ONCE(io_dma_pfn_cache_cap_mb) << 20;
	size_t seg_base, seg_len, rel;
	struct io_pfn_map *pm, *old;
	unsigned long pfn;
	dma_addr_t base;

	if (!c || !cap)
		return NULL;

	seg_base = offset & ~(c->quantum - 1);
	seg_len = min_t(size_t, c->quantum, map_len - seg_base);
	rel = offset - seg_base;
	if (unlikely(rel + len > seg_len)) {
		atomic64_inc(&c->range_fallbacks);
		return NULL;
	}
	pfn = folio_pfn(folio) + (seg_base >> PAGE_SHIFT);

	rcu_read_lock();
	pm = xa_load(&c->xa, pfn);
	if (pm && atomic_inc_not_zero(&pm->refs)) {
		rcu_read_unlock();
		if (unlikely(rel + len > pm->size)) {
			/*
			 * The run outgrew the cached region (fused striding
			 * runs vary in length, and a recycled PFN may carry a
			 * shorter mapping).  Displace the entry and fall
			 * through to remap the larger run under the same key;
			 * in-flight users of the old mapping stay safe via
			 * the bias protocol.
			 */
			if (xa_cmpxchg(&c->xa, pfn, pm, NULL,
				       GFP_NOWAIT | __GFP_NOWARN) == pm) {
				atomic64_sub(pm->size, &c->covered);
				io_pfn_map_put(pm);	/* cache bias */
			}
			io_pfn_map_put(pm);		/* lookup ref */
			atomic64_inc(&c->range_fallbacks);
			goto miss;
		}
		WRITE_ONCE(pm->referenced, true);
		atomic64_inc(&c->hits);
		*dma = pm->dma_base + rel;
		return pm;
	}
	rcu_read_unlock();
miss:
	atomic64_inc(&c->misses);

	pm = kmalloc(sizeof(*pm), GFP_NOWAIT | __GFP_NOWARN);
	if (!pm)
		goto fail;
	base = dma_map_page(c->dev, folio_page(folio, 0), seg_base,
			    seg_len, DMA_TO_DEVICE);
	if (dma_mapping_error(c->dev, base)) {
		kfree(pm);
		goto fail;
	}
	pm->pfn = pfn;
	pm->dma_base = base;
	pm->size = seg_len;
	pm->dev = c->dev;
	pm->referenced = true;
	atomic_set(&pm->refs, 2);	/* cache bias + this I/O */

	rcu_read_lock();
	old = xa_cmpxchg(&c->xa, pfn, NULL, pm, GFP_NOWAIT | __GFP_NOWARN);
	if (old) {
		/* Lost an insert race, or xarray node allocation failed. */
		dma_unmap_page(c->dev, base, pm->size, DMA_TO_DEVICE);
		kfree(pm);
		if (!xa_is_err(old) && atomic_inc_not_zero(&old->refs)) {
			rcu_read_unlock();
			if (unlikely(rel + len > old->size)) {
				atomic64_inc(&c->range_fallbacks);
				io_pfn_map_put(old);
				return NULL;
			}
			WRITE_ONCE(old->referenced, true);
			atomic64_inc(&c->hits);
			*dma = old->dma_base + rel;
			return old;
		}
		rcu_read_unlock();
		goto fail;
	}
	rcu_read_unlock();
	atomic64_inc(&c->inserts);
	if (atomic64_add_return(pm->size, &c->covered) > cap)
		io_pfn_cache_evict(c, cap);
	*dma = pm->dma_base + rel;
	return pm;
fail:
	atomic64_inc(&c->insert_fails);
	return NULL;
}

/* Erase everything; in-flight users keep their mappings alive until
 * their references drop. */
static void io_pfn_cache_flush(struct io_pfn_cache *c)
{
	struct io_pfn_map *pm;
	unsigned long index;

	spin_lock(&c->lock);
	xa_for_each(&c->xa, index, pm) {
		xa_erase(&c->xa, index);
		atomic64_sub(pm->size, &c->covered);
		io_pfn_map_put(pm);
	}
	c->hand = 0;
	spin_unlock(&c->lock);
}

static bool io_pfn_cache_usable(struct io_ring_ctx *ctx)
{
	return !ctx->dma.use_phys_addrs &&
	       READ_ONCE(io_dma_pfn_cache_cap_mb) != 0;
}

static int io_pfn_cache_stats_show(struct seq_file *m, void *p)
{
	int i;

	seq_printf(m, "cap_mb %u\n", READ_ONCE(io_dma_pfn_cache_cap_mb));
	for (i = 0; i < IO_PFN_CACHE_DEVS; i++) {
		struct io_pfn_cache *c;

		if (!smp_load_acquire(&io_pfn_cache_devs[i]))
			break;
		c = io_pfn_caches[i];
		seq_printf(m,
			   "dev %s quantum_kb %zu covered_kb %lld hits %lld misses %lld inserts %lld insert_fails %lld range_fallbacks %lld evictions %lld ref_skips %lld\n",
			   dev_name(c->dev),
			   c->quantum >> 10,
			   atomic64_read(&c->covered) >> 10,
			   atomic64_read(&c->hits),
			   atomic64_read(&c->misses),
			   atomic64_read(&c->inserts),
			   atomic64_read(&c->insert_fails),
			   atomic64_read(&c->range_fallbacks),
			   atomic64_read(&c->evictions),
			   atomic64_read(&c->ref_skips));
	}
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(io_pfn_cache_stats);

static ssize_t io_pfn_cache_flush_write(struct file *file,
					const char __user *ubuf,
					size_t len, loff_t *ppos)
{
	int i;

	for (i = 0; i < IO_PFN_CACHE_DEVS; i++) {
		if (!smp_load_acquire(&io_pfn_cache_devs[i]))
			break;
		io_pfn_cache_flush(io_pfn_caches[i]);
	}
	return len;
}

static const struct file_operations io_pfn_cache_flush_fops = {
	.owner	= THIS_MODULE,
	.write	= io_pfn_cache_flush_write,
};

/*
 * Busy-poll budget (in microseconds) for draining in-flight DMA completions
 * from the CQ-wait path (io_dma_cq_wait_poll()) before the waiting task
 * commits to sleeping. A DSA transfer for a typical read chunk completes
 * in ~4-6us plus queueing; sleeping instead costs a kworker schedule_work +
 * wakeup (~+10us when the poller is idle) just to be woken again. 0
 * disables the poll. Tunable via debugfs io_uring_dma_cq_poll_us.
 */
static unsigned int io_dma_cq_poll_us __read_mostly = 20;

/*
 * Inline spin budget (microseconds) for the DMA filemap-write wait. The
 * write path is synchronous per request; a typical request's batch
 * completes in tens of microseconds, so sleeping immediately trades a
 * ~100us usleep wakeup for ~35us of engine time and halves throughput
 * at small request sizes. Spin (with cond_resched) for up to this long
 * before backing off to sleeping -- the backoff still protects the
 * many-rings contention case. 0 sleeps immediately. Tunable via
 * debugfs io_uring_dma_fmw_spin_us.
 */
static unsigned int io_dma_fmw_spin_us __read_mostly = 60;

/*
 * Per-transaction DMA latency tracking, binned by transfer size. A
 * "transaction" is one io_dma_task (descriptor or batch): stamped at
 * hardware submit, sampled when the poller (or IRQ callback) observes
 * completion, so the bins measure detection-inclusive latency.
 */
static const struct {
	size_t		max;	/* exclusive upper bound in bytes; 0 == catch-all */
	const char	*label;
} io_dma_lat_bins[] = {
	{    4096, "<4KB"       },
	{    8192, "4-8KB"      },
	{   16384, "8-16KB"     },
	{   32768, "16-32KB"    },
	{   65536, "32-64KB"    },
	{  131072, "64-128KB"   },
	{  262144, "128-256KB"  },
	{  524288, "256-512KB"  },
	{ 1048576, "512KB-1MB"  },
	{       0, ">=1MB"      },
};
#define IO_DMA_LAT_NBINS	ARRAY_SIZE(io_dma_lat_bins)

struct io_dma_lat_stats {
	atomic64_t	count[IO_DMA_LAT_NBINS];
	atomic64_t	sum_ns[IO_DMA_LAT_NBINS];
};

static struct io_dma_lat_stats io_dma_lat_dma;	/* DSA transactions (per task) */
/*
 * Filemap DMA-read gate/result counters. The read path falls back to the
 * normal buffered read on ANY failure, silently, so a per-reason count is
 * the only way to see whether it engages at all. Recorded from the io_read
 * gate (io_uring/rw.c) and the submit loop.
 */
static const char * const io_dma_fm_names[IO_DMA_FM_NR] = {
	"engaged", "shmem", "not_bvec", "direct", "no_dma_addrs",
	"eagain", "enomem", "efault", "other",
};
static atomic64_t io_dma_fm[IO_DMA_FM_NR];

void io_dma_fm_record(unsigned int reason)
{
	atomic64_inc(&io_dma_fm[reason]);
}

static unsigned int io_dma_lat_bin(size_t len)
{
	unsigned int i;

	for (i = 0; i < IO_DMA_LAT_NBINS - 1; i++)
		if (len < io_dma_lat_bins[i].max)
			return i;
	return IO_DMA_LAT_NBINS - 1;	/* catch-all */
}

static void io_dma_lat_record(struct io_dma_lat_stats *s, size_t len, u64 ns)
{
	unsigned int b = io_dma_lat_bin(len);

	atomic64_inc(&s->count[b]);
	atomic64_add(ns, &s->sum_ns[b]);
}

/*
 * Per-channel queueing diagnostics for the channel-sharing design work:
 * seen_kb = bytes already in flight on the channel at each submit (the
 * queue the op joins, log2 KB buckets), lat_us = the submit->complete
 * latency it then experienced (log2 us buckets; includes service time
 * and detection), plus an in-flight high-water mark.  Bucket 0 is "<1",
 * bucket k is [2^(k-1), 2^k), the last bucket is the catch-all.
 * Lockless registry in submission order (like io_pfn_cache_devs).
 * Dumped via io_uring_dma_chan_qstat; counters (not live in-flight)
 * cleared by latency_reset.
 */
#define IO_DMA_QSTAT_CHANS	64
#define IO_DMA_QSTAT_NBUCKETS	16

struct io_dma_chan_qstat {
	struct dma_chan	*chan;		/* identity ONLY -- may be stale after
					 * a WQ reconfig; never dereference */
	char		name[24];	/* captured at registration */
	atomic64_t	inflight;	/* bytes submitted, not yet completed */
	atomic64_t	hwm;		/* max in-flight bytes seen */
	atomic64_t	submits;
	atomic64_t	bytes;
	atomic64_t	lat_us[IO_DMA_QSTAT_NBUCKETS];
	atomic64_t	seen_kb[IO_DMA_QSTAT_NBUCKETS];
};
static struct io_dma_chan_qstat io_dma_qstats[IO_DMA_QSTAT_CHANS];

static struct io_dma_chan_qstat *io_dma_qstat_get(struct dma_chan *chan)
{
	int i;

	for (i = 0; i < IO_DMA_QSTAT_CHANS; i++) {
		struct dma_chan *c = READ_ONCE(io_dma_qstats[i].chan);

		if (c == chan)
			return &io_dma_qstats[i];
		if (!c) {
			if (!cmpxchg(&io_dma_qstats[i].chan, NULL, chan)) {
				/* claimed: chan is live here (we are on its
				 * submit path); racing readers may see a
				 * partial name, never a stale pointer */
				strscpy(io_dma_qstats[i].name,
					dma_chan_name(chan),
					sizeof(io_dma_qstats[i].name));
				return &io_dma_qstats[i];
			}
			if (READ_ONCE(io_dma_qstats[i].chan) == chan)
				return &io_dma_qstats[i];
		}
	}
	return NULL;	/* registry full: op uncounted */
}

static unsigned int io_dma_qstat_bucket(u64 v)
{
	return v ? min_t(unsigned int, ilog2(v) + 1,
			 IO_DMA_QSTAT_NBUCKETS - 1) : 0;
}

/*
 * Per-DEVICE in-flight byte budget: the queueing-delay bound for the
 * channel-sharing design.  One shared WQ per device serves every ring;
 * delay for a new descriptor is (bytes ahead)/(engine bandwidth), so
 * bounding admitted bytes bounds delay by construction:
 *
 *	budget_kb ~= target_delay * engine_BW   (3MB ~= 100us at 30GB/s)
 *
 * Classes: the write sink (bulk, waits inline on its cookies) may hold
 * at most budget*wr_pct/100 -- refused chunks fall to the CPU-redo pass.
 * File reads are admitted while total in-flight <= budget, so the
 * (100-wr_pct)% is their guaranteed headroom (fixes the writer-crowds-
 * out-reader coverage loss measured in the 32/16/8-channel sweep).
 * Small ops (recv) always ride: their footprint is noise.
 * budget_kb 0 (default) disables all checks.  Accounting shares the
 * qstat submit/complete hooks; per-device lines appear in chan_qstat.
 */
static u32 io_dma_budget_kb;			/* 0 = off */
static u32 io_dma_budget_wr_pct = 75;

/*
 * Short DMA writes: write_begin() failures under reclaim surface as a
 * short return here, where generic_perform_write() would block and
 * retry. Counted for visibility (debugfs latency file).
 */
static atomic64_t io_dma_fmw_short;

struct io_dma_devload {
	void		*key;		/* dma_device*, identity only --
					 * never dereferenced */
	char		name[24];	/* captured at registration */
	atomic64_t	inflight;	/* all classes, bytes */
	atomic64_t	inflight_wr;	/* write-sink bytes */
	atomic64_t	rej_rd;
	atomic64_t	rej_wr;
};
#define IO_DMA_DEVLOADS	16
static struct io_dma_devload io_dma_devloads[IO_DMA_DEVLOADS];

static struct io_dma_devload *io_dma_devload_get(struct dma_chan *chan)
{
	void *key = chan->device;
	int i;

	for (i = 0; i < IO_DMA_DEVLOADS; i++) {
		void *k = READ_ONCE(io_dma_devloads[i].key);

		if (k == key)
			return &io_dma_devloads[i];
		if (!k) {
			if (!cmpxchg(&io_dma_devloads[i].key, NULL, key)) {
				strscpy(io_dma_devloads[i].name,
					dev_name(chan->device->dev),
					sizeof(io_dma_devloads[i].name));
				return &io_dma_devloads[i];
			}
			if (READ_ONCE(io_dma_devloads[i].key) == key)
				return &io_dma_devloads[i];
		}
	}
	return NULL;
}

/* Write-sink chunk admission; refusal sends the chunk to the CPU-redo
 * pass.  Checked per chunk so a draining queue readmits mid-write. */
static bool io_dma_budget_refuse_wr(struct dma_chan *chan, u32 len)
{
	u64 budget = (u64)READ_ONCE(io_dma_budget_kb) << 10;
	struct io_dma_devload *d;

	if (!budget)
		return false;
	d = io_dma_devload_get(chan);
	if (!d)
		return false;
	if (atomic64_read(&d->inflight) + len > budget ||
	    atomic64_read(&d->inflight_wr) + len >
			div_u64(budget * READ_ONCE(io_dma_budget_wr_pct), 100)) {
		atomic64_inc(&d->rej_wr);
		return true;
	}
	return false;
}

/* Whole-read admission at io_dma_filemap_read() entry; refusal falls
 * back to the buffered read (fm reason: eagain). */
static bool io_dma_budget_refuse_rd(struct dma_chan *chan)
{
	u64 budget = (u64)READ_ONCE(io_dma_budget_kb) << 10;
	struct io_dma_devload *d;

	if (!budget)
		return false;
	d = io_dma_devload_get(chan);
	if (!d)
		return false;
	if (atomic64_read(&d->inflight) > budget) {
		atomic64_inc(&d->rej_rd);
		return true;
	}
	return false;
}

static void io_dma_qstat_submit(struct dma_chan *chan, u32 len, bool wr)
{
	struct io_dma_chan_qstat *q = io_dma_qstat_get(chan);
	struct io_dma_devload *d = io_dma_devload_get(chan);
	u64 now, hwm;

	if (d) {
		atomic64_add(len, &d->inflight);
		if (wr)
			atomic64_add(len, &d->inflight_wr);
	}
	if (!q)
		return;
	atomic64_inc(&q->seen_kb[io_dma_qstat_bucket(
					atomic64_read(&q->inflight) >> 10)]);
	atomic64_inc(&q->submits);
	atomic64_add(len, &q->bytes);
	now = atomic64_add_return(len, &q->inflight);
	hwm = atomic64_read(&q->hwm);
	while (now > hwm) {
		u64 old = atomic64_cmpxchg(&q->hwm, hwm, now);

		if (old == hwm)
			break;
		hwm = old;
	}
}

static void io_dma_qstat_complete(struct dma_chan *chan, u32 len, u64 submit_ns,
				  bool wr)
{
	struct io_dma_chan_qstat *q = io_dma_qstat_get(chan);
	struct io_dma_devload *d = io_dma_devload_get(chan);

	if (d) {
		atomic64_sub(len, &d->inflight);
		if (wr)
			atomic64_sub(len, &d->inflight_wr);
	}
	if (!q)
		return;
	atomic64_sub(len, &q->inflight);
	if (!submit_ns)
		return;
	atomic64_inc(&q->lat_us[io_dma_qstat_bucket(
			div_u64(ktime_get_ns() - submit_ns, NSEC_PER_USEC))]);
}

static int io_dma_chan_qstat_show(struct seq_file *m, void *v)
{
	int i, b;

	for (i = 0; i < IO_DMA_QSTAT_CHANS; i++) {
		struct io_dma_chan_qstat *q = &io_dma_qstats[i];
		struct dma_chan *chan = READ_ONCE(q->chan);

		if (!chan)
			break;
		if (!atomic64_read(&q->submits))
			continue;	/* idle or pre-reconfig leftover */
		seq_printf(m, "%s submits %llu bytes_mb %llu inflight_kb %llu hwm_kb %llu\n",
			   q->name,
			   (u64)atomic64_read(&q->submits),
			   (u64)atomic64_read(&q->bytes) >> 20,
			   (u64)atomic64_read(&q->inflight) >> 10,
			   (u64)atomic64_read(&q->hwm) >> 10);
		seq_puts(m, "  lat_us");
		for (b = 0; b < IO_DMA_QSTAT_NBUCKETS; b++)
			seq_printf(m, " %llu", (u64)atomic64_read(&q->lat_us[b]));
		seq_puts(m, "\n  seen_kb");
		for (b = 0; b < IO_DMA_QSTAT_NBUCKETS; b++)
			seq_printf(m, " %llu", (u64)atomic64_read(&q->seen_kb[b]));
		seq_puts(m, "\n");
	}
	for (i = 0; i < IO_DMA_DEVLOADS; i++) {
		struct io_dma_devload *d = &io_dma_devloads[i];

		if (!READ_ONCE(d->key))
			break;
		seq_printf(m, "device %s inflight_kb %llu wr_kb %llu rej_rd %llu rej_wr %llu budget_kb %u wr_pct %u\n",
			   d->name,
			   (u64)atomic64_read(&d->inflight) >> 10,
			   (u64)atomic64_read(&d->inflight_wr) >> 10,
			   (u64)atomic64_read(&d->rej_rd),
			   (u64)atomic64_read(&d->rej_wr),
			   READ_ONCE(io_dma_budget_kb),
			   READ_ONCE(io_dma_budget_wr_pct));
	}
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(io_dma_chan_qstat);

static void io_dma_qstat_reset(void)
{
	int i, b;

	for (i = 0; i < IO_DMA_QSTAT_CHANS; i++) {
		struct io_dma_chan_qstat *q = &io_dma_qstats[i];

		if (!READ_ONCE(q->chan))
			break;
		atomic64_set(&q->submits, 0);
		atomic64_set(&q->bytes, 0);
		/* keep live inflight; restart the high-water mark from it */
		atomic64_set(&q->hwm, atomic64_read(&q->inflight));
		for (b = 0; b < IO_DMA_QSTAT_NBUCKETS; b++) {
			atomic64_set(&q->lat_us[b], 0);
			atomic64_set(&q->seen_kb[b], 0);
		}
	}
	for (i = 0; i < IO_DMA_DEVLOADS; i++) {
		if (!READ_ONCE(io_dma_devloads[i].key))
			break;
		/* live inflight/inflight_wr stay */
		atomic64_set(&io_dma_devloads[i].rej_rd, 0);
		atomic64_set(&io_dma_devloads[i].rej_wr, 0);
	}
}

static void io_dma_lat_show_one(struct seq_file *m, const char *name,
				struct io_dma_lat_stats *s)
{
	unsigned int i;

	seq_printf(m, "%s:\n", name);
	seq_printf(m, "  %-12s %12s %16s\n", "bin", "count", "avg_ns");
	for (i = 0; i < IO_DMA_LAT_NBINS; i++) {
		u64 count = atomic64_read(&s->count[i]);
		u64 sum = atomic64_read(&s->sum_ns[i]);
		u64 avg = count ? div64_u64(sum, count) : 0;

		seq_printf(m, "  %-12s %12llu %16llu\n",
			   io_dma_lat_bins[i].label, count, avg);
	}
}

static void io_dma_lat_reset(struct io_dma_lat_stats *s)
{
	unsigned int i;

	for (i = 0; i < IO_DMA_LAT_NBINS; i++) {
		atomic64_set(&s->count[i], 0);
		atomic64_set(&s->sum_ns[i], 0);
	}
}

/*
 * Filemap DMA-write gate/result counters, surfaced through the
 * io_uring_dma_latency debugfs stats file (and zeroed by its _reset
 * companion).
 */
static const char * const io_dma_fmw_names[IO_DMA_FMW_NR] = {
	"engaged", "no_aops", "not_bvec", "direct", "no_dma_addrs",
	"eagain", "cpu_redo", "error",
};
static atomic64_t io_dma_fmw[IO_DMA_FMW_NR];

void io_dma_fmw_record(unsigned int reason)
{
	atomic64_inc(&io_dma_fmw[reason]);
}

static int io_dma_lat_show(struct seq_file *m, void *v)
{
	unsigned int i;

	io_dma_lat_show_one(m, "dma", &io_dma_lat_dma);
	seq_puts(m, "filemap_write:\n");
	for (i = 0; i < IO_DMA_FMW_NR; i++)
		seq_printf(m, "  %-12s %12llu\n", io_dma_fmw_names[i],
			   atomic64_read(&io_dma_fmw[i]));
	seq_printf(m, "  %-12s %12llu\n", "short_write",
		   (u64)atomic64_read(&io_dma_fmw_short));
	seq_puts(m, "filemap:\n");
	for (i = 0; i < IO_DMA_FM_NR; i++)
		seq_printf(m, "  %-12s %12llu\n", io_dma_fm_names[i],
			   atomic64_read(&io_dma_fm[i]));
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(io_dma_lat);

static ssize_t io_dma_lat_reset_write(struct file *file,
				      const char __user *ubuf,
				      size_t count, loff_t *ppos)
{
	unsigned int i;

	io_dma_lat_reset(&io_dma_lat_dma);
	for (i = 0; i < IO_DMA_FMW_NR; i++)
		atomic64_set(&io_dma_fmw[i], 0);
	for (i = 0; i < IO_DMA_FM_NR; i++)
		atomic64_set(&io_dma_fm[i], 0);
	for (i = 0; i < IO_DMA_FMW_NR; i++)
		atomic64_set(&io_dma_fmw[i], 0);
	atomic64_set(&io_dma_fmw_short, 0);
	io_dma_qstat_reset();
	return count;
}

static const struct file_operations io_dma_lat_reset_fops = {
	.owner		= THIS_MODULE,
	.open		= simple_open,
	.write		= io_dma_lat_reset_write,
	.llseek		= noop_llseek,
};

/*
 * Optional global CPU latency QoS request. DMA completion latency includes
 * waking a kworker on an idle CPU; on a
 * large idle machine those wakeups pay deep C-state exit latency. Writing
 * N >= 0 (microseconds) installs/updates a cpu_latency_qos request bounding
 * C-state exit latency machine-wide; writing a negative value removes it.
 * Default: no request (-1), so power policy is unchanged until opted in.
 */
static DEFINE_MUTEX(io_dma_cpu_lat_lock);
static struct pm_qos_request io_dma_cpu_lat_qos;
static int io_dma_cpu_lat_us = -1;

static int io_dma_cpu_lat_show(struct seq_file *m, void *v)
{
	seq_printf(m, "%d\n", READ_ONCE(io_dma_cpu_lat_us));
	return 0;
}

static int io_dma_cpu_lat_open(struct inode *inode, struct file *file)
{
	return single_open(file, io_dma_cpu_lat_show, NULL);
}

static ssize_t io_dma_cpu_lat_write(struct file *file, const char __user *ubuf,
				    size_t len, loff_t *ppos)
{
	int val, ret;

	ret = kstrtoint_from_user(ubuf, len, 0, &val);
	if (ret)
		return ret;

	mutex_lock(&io_dma_cpu_lat_lock);
	if (val < 0) {
		if (cpu_latency_qos_request_active(&io_dma_cpu_lat_qos))
			cpu_latency_qos_remove_request(&io_dma_cpu_lat_qos);
		val = -1;
	} else if (cpu_latency_qos_request_active(&io_dma_cpu_lat_qos)) {
		cpu_latency_qos_update_request(&io_dma_cpu_lat_qos, val);
	} else {
		cpu_latency_qos_add_request(&io_dma_cpu_lat_qos, val);
	}
	WRITE_ONCE(io_dma_cpu_lat_us, val);
	mutex_unlock(&io_dma_cpu_lat_lock);
	return len;
}

static const struct file_operations io_dma_cpu_lat_fops = {
	.owner		= THIS_MODULE,
	.open		= io_dma_cpu_lat_open,
	.read		= seq_read,
	.write		= io_dma_cpu_lat_write,
	.llseek		= seq_lseek,
	.release	= single_release,
};

void io_dma_debugfs_init(void)
{
	debugfs_create_file("io_uring_dma_chan_qstat", 0444, NULL, NULL,
			    &io_dma_chan_qstat_fops);
	debugfs_create_u32("io_uring_dma_budget_kb", 0644, NULL,
			   &io_dma_budget_kb);
	debugfs_create_u32("io_uring_dma_budget_wr_pct", 0644, NULL,
			   &io_dma_budget_wr_pct);
	debugfs_create_u32("io_uring_dma_max_clients_per_chan", 0644, NULL,
			   &io_dma_max_clients_per_chan);
	debugfs_create_u32("io_uring_dma_admission_rejects", 0444, NULL,
			   &io_dma_admission_rejects);
	debugfs_create_u32("io_uring_dma_cq_poll_us", 0644, NULL,
			   &io_dma_cq_poll_us);
	debugfs_create_u32("io_uring_dma_fmw_spin_us", 0644, NULL,
			   &io_dma_fmw_spin_us);
	debugfs_create_file("io_uring_dma_cpu_latency_us", 0644, NULL, NULL,
			    &io_dma_cpu_lat_fops);
	debugfs_create_file("io_uring_dma_latency", 0444, NULL, NULL,
			    &io_dma_lat_fops);
	debugfs_create_file("io_uring_dma_latency_reset", 0200, NULL, NULL,
			    &io_dma_lat_reset_fops);
	debugfs_create_file("io_uring_dma_pfn_cache", 0444, NULL, NULL,
			    &io_pfn_cache_stats_fops);
	debugfs_create_u32("io_uring_dma_pfn_cache_cap_mb", 0644, NULL,
			   &io_dma_pfn_cache_cap_mb);
	debugfs_create_file("io_uring_dma_pfn_cache_flush", 0200, NULL, NULL,
			    &io_pfn_cache_flush_fops);
}

/* Datapath alloc: pool first, then non-blocking slab, never sleeps. */
static struct io_dma_task *io_dma_task_alloc(struct io_ring_ctx *ctx)
{
	struct io_dma_channel *d = &ctx->dma;
	struct io_dma_task *t;
	unsigned long flags;

	spin_lock_irqsave(&d->free_lock, flags);
	t = d->free_list;
	if (t) {
		d->free_list = t->next;
		d->free_count--;
	}
	spin_unlock_irqrestore(&d->free_lock, flags);

	if (t) {
		memset(t, 0, sizeof(*t));	/* match old kmem_cache_zalloc */
		return t;
	}

	/*
	 * Pool exhausted: NOWAIT so we never enter reclaim on the datapath.
	 * NOWARN because failure is expected and handled (CPU-copy fallback).
	 */
	return kmem_cache_zalloc(dma_cachep, GFP_NOWAIT | __GFP_NOWARN);
}

/* Datapath free: park back into the pool up to the cap, else slab. */
static void io_dma_task_free(struct io_ring_ctx *ctx, struct io_dma_task *t)
{
	struct io_dma_channel *d = &ctx->dma;
	unsigned long flags;

	spin_lock_irqsave(&d->free_lock, flags);
	if (d->free_count < d->free_max) {
		t->next = d->free_list;
		d->free_list = t;
		d->free_count++;
		t = NULL;
	}
	spin_unlock_irqrestore(&d->free_lock, flags);

	if (t)
		kmem_cache_free(dma_cachep, t);
}

/*
 * Prefill the per-ctx io_dma_task pool, sized to the ring's CQ depth.
 * Called from io_allocate_dma_chan() at ring setup (process context, no
 * locks held), so GFP_KERNEL is fine here.
 */
void io_dma_init_freelist(struct io_ring_ctx *ctx, struct io_uring_params *p)
{
	struct io_dma_channel *d = &ctx->dma;
	unsigned int n, i;

	spin_lock_init(&d->free_lock);
	d->free_list = NULL;
	d->free_count = 0;

	/*
	 * Cover roughly the in-flight CQ depth; clamp to a sane range.
	 * io_dma_task is tiny (~96B), so even 8192 is ~768KB worst case.
	 */
	n = clamp(p->cq_entries * 2u, 256u, 8192u);
	d->free_max = n;

	for (i = 0; i < n; i++) {
		struct io_dma_task *t = kmem_cache_alloc(dma_cachep, GFP_KERNEL);

		if (!t)
			break;		/* partial prefill OK -- NOWAIT covers rest */
		t->next = d->free_list;
		d->free_list = t;
		d->free_count++;
	}
}

void io_dma_poll_workfn(struct work_struct *w)
{
	/* work_struct is embedded in ctx->dma (struct io_dma_channel) */
	struct io_dma_channel *d = container_of(w, struct io_dma_channel, poll_work);
	/* io_dma_channel is embedded in io_ring_ctx as 'dma' */
	struct io_ring_ctx *ctx = container_of(d, struct io_ring_ctx, dma);

	/* Drain until the list is empty */
	do {
		__io_dma_poll(ctx);
		cpu_relax();
	} while (io_dma_pending(ctx));
}

/* TODO: Laid out like io_rw because we have to get back from this kiocb to the io_kiocb. */
struct io_dma {
	/* NOTE: kiocb has the file as the first member, so don't do it here */
	struct kiocb			kiocb;
	u64				addr;
	u32				len;
	rwf_t				flags;
};

void io_uring_dma_prep(struct io_kiocb *req)
{
	if (IS_ERR_OR_NULL(req->ctx->dma.chan))
		return;

	req->dma.dma_active = true;
	/*
	 * dma_ref_held is deliberately NOT reset here: it is set/cleared
	 * only under ctx->dma.lock by the submit/complete ref protocol, and
	 * a reissue can reach this prep while the previous cycle's
	 * completer may still be about to drop the previous in-flight ref.
	 * Clearing the flag here would erase that pending drop and leak one
	 * req reference per race.  A fresh req needs no init: the submit
	 * path sets it.
	 */
	req->dma.dma_refcnt = 0;
	req->dma.dma_result = 0;
	req->dma.dma_tasks = NULL;
	req->dma.dma_tasks_tail = NULL;
	req->dma.dst_user_addr = 0;
	req->dma.saved_res = 0;
	req->dma.saved_cflags = 0;
}

/*
 * Submit a batch of DMA copy operations as a single DSA batch descriptor.
 * Takes one io_dma_task for the entire batch instead of one per chunk.
 * The entries[] array is copied to a heap allocation for deferred cleanup.
 */
static ssize_t io_dma_submit_batch(struct io_kiocb *req,
				   struct device *dev, struct dma_chan *chan,
				   struct io_dma_batch_entry *entries,
				   unsigned int nr_entries)
{
	struct dma_async_tx_descriptor *tx;
	struct io_dma_batch_entry *heap_entries;
	struct scatterlist *sgls, *src_sgl, *dst_sgl;
	struct io_dma_task *dma;
	u32 total_len = 0;
	int i;

	/* Allocate src + dst scatterlists together.
	 * Initialize SG tables so sg_next()/sg_is_last() work correctly,
	 * then populate DMA addresses from the entries array.
	 */
	sgls = kmalloc_array(nr_entries * 2, sizeof(*sgls),
			     GFP_NOWAIT | __GFP_NOWARN);
	if (!sgls)
		return -ENOMEM;
	src_sgl = sgls;
	dst_sgl = sgls + nr_entries;

	sg_init_table(src_sgl, nr_entries);
	sg_init_table(dst_sgl, nr_entries);
	for (i = 0; i < nr_entries; i++) {
		sg_dma_address(&src_sgl[i]) = entries[i].src_dma;
		sg_dma_len(&src_sgl[i]) = entries[i].src_len;
		sg_dma_address(&dst_sgl[i]) = entries[i].dst_dma;
		sg_dma_len(&dst_sgl[i]) = entries[i].src_len;
		total_len += entries[i].src_len;
	}

	tx = dmaengine_prep_dma_memcpy_sg(chan, dst_sgl, nr_entries,
					   src_sgl, nr_entries,
					   io_dma_prep_flags());
	if (!tx) {
		kfree(sgls);
		return -EAGAIN;
	}

	/* SG arrays are consumed by dmaengine_prep_dma_memcpy_sg —
	 * the driver copies what it needs into batch descriptors.
	 */
	kfree(sgls);

	heap_entries = kmalloc_array(nr_entries, sizeof(*heap_entries),
				     GFP_NOWAIT | __GFP_NOWARN);
	if (!heap_entries)
		return -ENOMEM;
	memcpy(heap_entries, entries, nr_entries * sizeof(*heap_entries));

	dma = io_dma_task_alloc(req->ctx);
	if (!dma) {
		kfree(heap_entries);
		return -ENOMEM;
	}

	dma->req = req;
	dma->next = NULL;
	dma->len = total_len;
	dma->is_batch = true;
	dma->batch_nr = nr_entries;
	dma->batch_entries = heap_entries;

	/* Capture before the doorbell: dmaengine_submit() makes the descriptor
	 * eligible for a concurrent reap-and-recycle. */
	dma->submit_ns = ktime_get_ns();
	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		kfree(heap_entries);
		io_dma_task_free(req->ctx, dma);
		return -EAGAIN;	/* full shared WQ etc.: CPU-copy fallback */
	}
	io_dma_qstat_submit(req->ctx->dma.chan, dma->len, false);

	/* Take folio refs for DMA duration */
	for (i = 0; i < nr_entries; i++)
		folio_get(entries[i].folio);

	req->dma.dma_refcnt++;

	if (!req->dma.dma_tasks) {
		req->dma.dma_tasks = dma;
		req->dma.dma_tasks_tail = dma;
	} else {
		req->dma.dma_tasks_tail->next = dma;
		req->dma.dma_tasks_tail = dma;
	}

	return total_len;
}

#define IO_DMA_BATCH_MIN	8

/*
 * Submit a single DMA descriptor for one batch entry.
 * Used when nr_entries < IO_DMA_BATCH_MIN to avoid batch overhead.
 * The entry already has DMA-mapped src_dma from the caller.
 */
static ssize_t io_dma_submit_single_entry(struct io_kiocb *req,
					  struct dma_chan *chan,
					  struct io_dma_batch_entry *entry)
{
	struct dma_async_tx_descriptor *tx;
	struct io_dma_task *dma;

	dma = io_dma_task_alloc(req->ctx);
	if (!dma)
		return -ENOMEM;

	/* Prep after the task alloc: an abandoned prep would orphan an
	 * idxd descriptor from the channel pool. */
	tx = dmaengine_prep_dma_memcpy(chan, entry->dst_dma, entry->src_dma,
				       entry->src_len, io_dma_prep_flags());
	if (!tx) {
		io_dma_task_free(req->ctx, dma);
		return -EAGAIN;
	}

	folio_get(entry->folio);

	dma->req = req;
	dma->next = NULL;
	dma->src_dma = entry->src_dma;
	dma->dst_dma = entry->dst_dma;
	dma->len = entry->src_len;
	dma->src_map_addr = entry->src_dma;
	dma->src_map_len = entry->src_len;
	dma->src_pfn_map = entry->pfn_map;
	dma->src_folio = entry->folio;
	dma->src_is_page = true;
	dma->is_batch = false;

	/* Capture before the doorbell: dmaengine_submit() makes the descriptor
	 * eligible for a concurrent reap-and-recycle. */
	dma->submit_ns = ktime_get_ns();
	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		folio_put(entry->folio);
		io_dma_task_free(req->ctx, dma);
		return -EAGAIN;	/* full shared WQ etc.: CPU-copy fallback */
	}
	io_dma_qstat_submit(req->ctx->dma.chan, dma->len, false);

	req->dma.dma_refcnt++;

	if (!req->dma.dma_tasks) {
		req->dma.dma_tasks = dma;
		req->dma.dma_tasks_tail = dma;
	} else {
		req->dma.dma_tasks_tail->next = dma;
		req->dma.dma_tasks_tail = dma;
	}

	return entry->src_len;
}

/*
 * Unmap source DMA mappings for batch entries on error paths.
 */
static void io_dma_unmap_batch_entries(struct io_kiocb *req,
				       struct device *dev,
				       struct io_dma_batch_entry *entries,
				       unsigned int nr)
{
	unsigned int i;

	if (req->ctx->dma.use_phys_addrs)
		return;

	for (i = 0; i < nr; i++) {
		struct io_dma_batch_entry *e = &entries[i];

		if (e->pfn_map) {
			/* Cached mapping: drop the in-flight reference; the
			 * unmap belongs to eviction/flush, not this I/O. */
			io_pfn_map_put(e->pfn_map);
		} else {
			dma_unmap_page(dev, e->src_dma, e->src_len,
				       DMA_TO_DEVICE);
		}
	}
}

/*
 * Flush collected batch entries.  Uses individual descriptors when below
 * IO_DMA_BATCH_MIN to avoid DSA batch descriptor overhead, and a single
 * batch descriptor otherwise.
 */
static ssize_t io_dma_flush_batch(struct io_kiocb *req,
				  struct device *dev, struct dma_chan *chan,
				  struct io_dma_batch_entry *entries,
				  unsigned int nr_entries)
{
	ssize_t total = 0;
	unsigned int i;
	ssize_t ret;

	if (!nr_entries)
		return 0;

	if (nr_entries < IO_DMA_BATCH_MIN) {
		for (i = 0; i < nr_entries; i++) {
			ret = io_dma_submit_single_entry(req, chan, &entries[i]);
			if (ret < 0) {
				/* Unmap remaining entries */
				io_dma_unmap_batch_entries(req, dev,
							  entries + i + 1,
							  nr_entries - i - 1);
				return total > 0 ? total : ret;
			}
			total += ret;
		}
		return total;
	}

	ret = io_dma_submit_batch(req, dev, chan, entries, nr_entries);
	if (ret < 0)
		io_dma_unmap_batch_entries(req, dev, entries, nr_entries);
	return ret;
}

/*
 * DMA-offloaded page cache read for io_uring registered buffers.
 * Mirrors filemap_read() but replaces copy_folio_to_iter() with DMA
 * batch submissions from page cache folios to pre-mapped registered
 * buffer pages.  Uses dmaengine_prep_dma_memcpy_sg() to submit all
 * chunks in a folio batch as a single DSA batch descriptor.
 *
 * Returns bytes read (>0), 0 at EOF, or negative error.
 * Caller must call io_dma_submit_queued_tasks() after this returns >0
 * with dma_refcnt > 0 to drain the queued DMA tasks.
 */
ssize_t io_dma_filemap_read(struct io_kiocb *req, struct kiocb *iocb,
			    u64 dst_user_addr, size_t want)
{
	struct io_ring_ctx *ctx = req->ctx;
	struct file *filp = iocb->ki_filp;
	struct address_space *mapping = filp->f_mapping;
	struct inode *inode = mapping->host;
	struct io_mapped_ubuf *imu = req->buf_node->buf;
	struct device *dev = ctx->dma.chan->device->dev;
	struct dma_chan *chan = ctx->dma.chan;
	struct io_pfn_cache *pfn_cache =
		io_pfn_cache_usable(ctx) ? io_pfn_cache_get(dev) : NULL;
	size_t map_quantum = io_dma_map_quantum(dev);
	struct folio_batch fbatch;
	struct io_dma_batch_entry *entries;
	unsigned int nr_entries = 0;
	ssize_t total_read = 0;
	ssize_t submitted = 0;
	size_t batch_bytes = 0;
	size_t dst_offset = 0;
	loff_t start_pos = iocb->ki_pos;
	loff_t isize;
	int i, error = 0;
	bool writably_mapped;

	if (unlikely(iocb->ki_pos < 0))
		return -EINVAL;
	if (unlikely(iocb->ki_pos >= inode->i_sb->s_maxbytes))
		return 0;
	/* Device byte-budget admission: reads are refused only when even
	 * their full-budget headroom is gone (writers are capped at
	 * wr_pct below it).  Falls back to the buffered read. */
	if (io_dma_budget_refuse_rd(chan))
		return -EAGAIN;

	/*
	 * NOWAIT: never block on the read datapath. On failure the caller
	 * (io_read) falls through to the normal buffered-read path.
	 */
	entries = kmalloc_array(IO_DMA_BATCH_MAX, sizeof(*entries),
				GFP_NOWAIT | __GFP_NOWARN);
	if (!entries)
		return -ENOMEM;

	isize = i_size_read(inode);
	if (unlikely(iocb->ki_pos >= isize)) {
		kfree(entries);
		return 0;
	}

	folio_batch_init(&fbatch);

	do {
		size_t count;
		loff_t end_offset;

		cond_resched();

		if (unlikely(iocb->ki_pos >= i_size_read(inode)))
			break;

		/*
		 * How many bytes remain in the REQUEST.  The registered
		 * buffer (imu) is usually larger than the read: clamping to
		 * imu->len instead of the requested count made a short
		 * READ_FIXED overrun its length and fill the whole buffer
		 * (then trip -EFAULT at the buffer-end dst lookup).
		 */
		count = want - dst_offset;
		if (!count)
			break;

		error = filemap_get_pages(iocb, count, &fbatch, false);
		if (error < 0)
			break;

		isize = i_size_read(inode);
		if (unlikely(iocb->ki_pos >= isize))
			goto put_folios;
		end_offset = min_t(loff_t, isize, iocb->ki_pos + count);

		writably_mapped = mapping_writably_mapped(mapping);

		if (folio_batch_count(&fbatch))
			folio_mark_accessed(fbatch.folios[0]);

		for (i = 0; i < folio_batch_count(&fbatch); i++) {
			struct folio *folio = fbatch.folios[i];
			size_t fsize = folio_size(folio);
			size_t offset = iocb->ki_pos & (fsize - 1);
			size_t bytes = min_t(loff_t, end_offset - iocb->ki_pos,
					     fsize - offset);
			size_t copied = 0;

			if (end_offset < folio_pos(folio))
				break;
			if (i > 0)
				folio_mark_accessed(folio);
			if (writably_mapped)
				flush_dcache_folio(folio);

			/* Collect DMA entries for this folio, splitting
			 * at destination registered buffer folio boundaries.
			 */
			while (copied < bytes) {
				struct io_pfn_map *pm = NULL;
				dma_addr_t dst_dma, src_dma;
				size_t dst_folio_remain;
				size_t chunk;

				if (ctx->dma.use_phys_addrs) {
					dst_dma = io_reg_buf_phys_addr(imu,
							dst_user_addr + dst_offset);
				} else {
					dst_dma = io_reg_buf_dma_addr(imu,
							dst_user_addr + dst_offset);
				}
				if (!dst_dma) {
					error = -EFAULT;
					goto flush_and_put;
				}

				dst_folio_remain = (1UL << imu->folio_shift) -
					((dst_user_addr + dst_offset) &
					 ((1UL << imu->folio_shift) - 1));

				chunk = min_t(size_t, bytes - copied,
					      dst_folio_remain);
				/* Split at source map-quantum boundaries so
				 * one cache segment covers each entry and no
				 * transient map exceeds the rcache classes. */
				chunk = min_t(size_t, chunk, map_quantum -
					((offset + copied) & (map_quantum - 1)));

				if (ctx->dma.use_phys_addrs) {
					src_dma = page_to_phys(folio_page(folio, 0)) +
						  offset + copied;
				} else {
					pm = io_pfn_map_lookup(pfn_cache, folio,
							       offset + copied,
							       chunk,
							       folio_size(folio),
							       &src_dma);
					if (!pm) {
						src_dma = dma_map_page(dev, &folio->page,
								       offset + copied,
								       chunk, DMA_TO_DEVICE);
						if (dma_mapping_error(dev, src_dma)) {
							error = -EFAULT;
							goto flush_and_put;
						}
					}
				}

				/* Collect entry for batch submission */
				entries[nr_entries].src_dma = src_dma;
				entries[nr_entries].pfn_map = pm;
				entries[nr_entries].dst_dma = dst_dma;
				entries[nr_entries].src_len = chunk;
				entries[nr_entries].folio = folio;
				nr_entries++;

				copied += chunk;
				dst_offset += chunk;
				batch_bytes += chunk;

				/* Flush batch if full */
				if (nr_entries == IO_DMA_BATCH_MAX) {
					ssize_t ret;

					ret = io_dma_flush_batch(req, dev, chan,
						entries, nr_entries);
					nr_entries = 0;
					if (ret < 0) {
						error = ret;
						goto put_folios;
					}
					submitted += batch_bytes;
					batch_bytes = 0;
				}
			}

			total_read += copied;
			iocb->ki_pos += copied;

			if (copied < bytes) {
				error = -EFAULT;
				break;
			}
		}
flush_and_put:
		/* Flush any remaining entries from this folio batch */
		if (nr_entries > 0) {
			ssize_t ret;

			ret = io_dma_flush_batch(req, dev, chan,
				entries, nr_entries);
			nr_entries = 0;
			if (ret < 0) {
				if (!error)
					error = ret;
			} else {
				submitted += batch_bytes;
				batch_bytes = 0;
			}
		}
put_folios:
		for (i = 0; i < folio_batch_count(&fbatch); i++)
			folio_put(fbatch.folios[i]);
		folio_batch_init(&fbatch);
	} while (dst_offset < want && iocb->ki_pos < isize && !error);

	file_accessed(filp);

	if (req->dma.dma_refcnt > 0)
		dma_async_issue_pending(chan);

	kfree(entries);

	/*
	 * Only bytes whose batch was successfully handed to the DMA engine
	 * may be claimed.  total_read (and ki_pos) run ahead of the flushes
	 * during collection; a failed flush leaves those trailing bytes
	 * uncopied, so clamp the result -- and the file position -- to what
	 * was actually submitted.  Claiming unsubmitted bytes returns
	 * uninitialized destination memory to userspace.
	 */
	if (unlikely(submitted != total_read))
		iocb->ki_pos = start_pos + submitted;
	return submitted ? submitted : error;
}

/*
 * DMA-offloaded buffered write for registered buffers (WRITE_FIXED):
 * replaces generic_perform_write()'s copy_folio_from_iter() with DSA
 * copies from the pre-mapped registered buffer into the page-cache
 * folios obtained from aops->write_begin().
 *
 * DELIBERATELY SYNCHRONOUS: the caller path runs in io-wq process
 * context (buffered regular-file writes always punt there), and
 * inode_lock plus the per-folio write_begin/write_end protocol make a
 * deferred-CQE design a lifetime minefield.  Waiting inline keeps every
 * VFS invariant identical to the generic path; the win is the removed
 * CPU memcpy, not latency.
 *
 * Failure ladder:
 *  - prep/submit failure or a DMA_ERROR completion, with every submitted
 *    cookie reaped: CPU-RE-COPY every chunk (same src -> same dst is
 *    idempotent) and commit normally ("cpu_redo").
 *  - completion timeout (device wedged): unmap the dst ranges so a late
 *    DMA write faults in the IOMMU instead of hitting reclaimed memory,
 *    skip write_end, and deliberately LEAK the affected folios (locked +
 *    referenced) -- a wedged range beats silent corruption.  Returns -EIO.
 */
#define IO_DMA_FMW_WAIT_MS	5000

struct io_dma_fmw_folio {
	struct folio *folio;
	void *fsdata;
	loff_t pos;
	unsigned int len;
	dma_addr_t dst_dma;	/* 0: phys-addr mode, nothing to unmap */
	unsigned int map_len;
};

/* One in-flight write descriptor, carrying what qstat accounting needs. */
struct io_dma_fmw_ck {
	dma_cookie_t ck;
	unsigned int len;
	u64 submit_ns;
};

/*
 * Poll every outstanding cookie to completion.  Sets *redo on any
 * DMA_ERROR.  Returns 0, or -ETIMEDOUT with cookies possibly still in
 * flight (the caller must treat every dst as poisoned).
 *
 * Spin only briefly: DSA transfers normally complete in single-digit
 * microseconds, but under contention (many rings sharing the WQ
 * descriptor pools) completion can take milliseconds -- and hundreds of
 * io-wq workers busy-polling here saturated whole sockets and starved
 * application heartbeats at O(100) shared rings.  Back off to sleeping
 * once the fast path misses.
 */
static int io_dma_fmw_wait(struct dma_chan *chan, struct io_dma_fmw_ck *cookies,
			   unsigned int *nr, bool *redo)
{
	unsigned long deadline = jiffies + msecs_to_jiffies(IO_DMA_FMW_WAIT_MS);
	u64 spin_end = ktime_get_ns() +
		READ_ONCE(io_dma_fmw_spin_us) * NSEC_PER_USEC;
	unsigned int i, j, spins;

	for (i = 0; i < *nr; i++) {
		enum dma_status st;

		spins = 0;
		while ((st = dmaengine_async_is_tx_complete(chan, cookies[i].ck))
		       == DMA_IN_PROGRESS) {
			if (time_after(jiffies, deadline)) {
				/* Wedged: retire the qstat in-flight bytes
				 * (the descriptors are leaked/fenced by the
				 * caller), no latency samples. */
				for (j = i; j < *nr; j++)
					io_dma_qstat_complete(chan,
							cookies[j].len, 0,
							true);
				*nr = 0;
				return -ETIMEDOUT;
			}
			if (ktime_get_ns() < spin_end) {
				cpu_relax();
				if (!(++spins & 63))
					cond_resched();
			} else {
				usleep_range(50, 150);
			}
		}
		if (st != DMA_COMPLETE)
			*redo = true;
		/* Latency is poll-observation time: cookies are polled in
		 * order, so a later cookie's sample can be inflated by up
		 * to the earlier polls -- fine at histogram resolution. */
		io_dma_qstat_complete(chan, cookies[i].len,
				      st == DMA_COMPLETE ?
						cookies[i].submit_ns : 0, true);
	}
	*nr = 0;
	return 0;
}

ssize_t io_dma_filemap_write(struct io_kiocb *req, struct kiocb *iocb,
			     struct iov_iter *from, u64 src_user_addr)
{
	struct io_ring_ctx *ctx = req->ctx;
	struct file *file = iocb->ki_filp;
	struct address_space *mapping = file->f_mapping;
	const struct address_space_operations *aops = mapping->a_ops;
	struct inode *inode = mapping->host;
	struct io_mapped_ubuf *imu = req->buf_node->buf;
	struct dma_chan *chan = ctx->dma.chan;
	struct device *dev = chan->device->dev;
	/* Clamp write chunks (and so the transient dst-folio maps) to the
	 * IOVA-rcache-served quantum. */
	size_t max_chunk = min_t(size_t, mapping_max_folio_size(mapping),
				 io_dma_map_quantum(chan->device->dev));
	struct io_dma_fmw_folio *fol = NULL;
	struct io_dma_fmw_ck *cookies = NULL;
	unsigned int nr_fol = 0, nr_cookies = 0, max_fol, max_cookies, i;
	ssize_t want, written = 0, err = 0;
	bool redo = false;
	bool surrendered = false;
	unsigned int prep_fails = 0;
	int wedged = 0;

	inode_lock(inode);
	want = generic_write_checks(iocb, from);
	if (want <= 0) {
		inode_unlock(inode);
		return want;
	}
	err = file_remove_privs(file);
	if (!err)
		err = file_update_time(file);
	if (err)
		goto out_unlock;

	max_fol = DIV_ROUND_UP(want, PAGE_SIZE) + 1;
	/* one entry per src-folio crossing per chunk; PAGE_SIZE granularity
	 * over-provisions safely */
	max_cookies = max_fol + DIV_ROUND_UP(want, 1UL << imu->folio_shift) + 8;
	fol = kvmalloc_array(max_fol, sizeof(*fol), GFP_KERNEL);
	cookies = kvmalloc_array(max_cookies, sizeof(*cookies), GFP_KERNEL);
	if (!fol || !cookies) {
		err = -EAGAIN;	/* fall back to the normal write path */
		goto out_unlock;
	}

	while (written < want && nr_fol < max_fol) {
		loff_t pos = iocb->ki_pos + written;
		size_t bytes = min_t(size_t,
				     max_chunk - (pos & (max_chunk - 1)),
				     want - written);
		size_t offset, sub;
		dma_addr_t dst_dma;
		struct folio *folio;
		void *fsdata;
		int status;

		status = aops->write_begin(iocb, mapping, pos, bytes,
					   &folio, &fsdata);
		if (unlikely(status < 0)) {
			/* Unlike generic_perform_write(), which blocks in
			 * reclaim and retries, fail into a short write;
			 * fmw_short counts these. */
			if (!written)
				err = status;
			break;
		}
		offset = offset_in_folio(folio, pos);
		/*
		 * Cover the locked folio to its end (or the write's end):
		 * ending a chunk mid-folio would make the next iteration's
		 * write_begin() wait forever on the folio lock this batch
		 * already holds.  Large folios (e.g. a rewrite of ranges
		 * cached by earlier big writes) exceed the quantum-sized
		 * request hint, so the returned folio, not the hint, decides
		 * the chunk; descriptors below still split at source-folio
		 * boundaries and the quantum.
		 */
		bytes = min_t(size_t, want - written,
			      folio_size(folio) - offset);

		if (surrendered) {
			/* Sustained descriptor exhaustion: stop touching the
			 * DMA engine for the rest of this write and let the
			 * CPU-redo pass commit everything. Beats paying a
			 * drain-wait per chunk while dozens of other rings
			 * hold the pools empty. */
			dst_dma = 0;
			redo = true;
			goto record;
		}
		if (io_dma_budget_refuse_wr(chan, bytes)) {
			/* Over the device in-flight byte budget: this chunk
			 * goes to the CPU-redo pass.  Re-checked per chunk so
			 * a draining queue readmits mid-write. */
			dst_dma = 0;
			redo = true;
			goto record;
		}
		if (ctx->dma.use_phys_addrs) {
			dst_dma = 0;
		} else {
			dst_dma = dma_map_page(dev, folio_page(folio, 0),
					       offset, bytes, DMA_FROM_DEVICE);
			if (dma_mapping_error(dev, dst_dma)) {
				/* commit this chunk via CPU-redo instead */
				dst_dma = 0;
				redo = true;
				goto record;
			}
		}

		/* split at source registered-buffer folio boundaries */
		for (sub = 0; sub < bytes; ) {
			u64 uaddr = src_user_addr + written + sub;
			size_t src_remain = (1UL << imu->folio_shift) -
				(uaddr & ((1UL << imu->folio_shift) - 1));
			size_t len = min3(bytes - sub, src_remain, max_chunk);
			struct dma_async_tx_descriptor *tx;
			dma_addr_t src_dma, dst;
			dma_cookie_t ck;

			if (ctx->dma.use_phys_addrs) {
				src_dma = io_reg_buf_phys_addr(imu, uaddr);
				dst = page_to_phys(folio_page(folio, 0)) +
					offset + sub;
			} else {
				src_dma = io_reg_buf_dma_addr(imu, uaddr);
				dst = dst_dma + sub;
			}
			if (unlikely(!src_dma)) {
				redo = true;	/* CPU-redo covers the chunk */
				break;
			}

			tx = dmaengine_prep_dma_memcpy(chan, dst, src_dma, len,
						       io_dma_prep_flags());
			if (!tx) {
				/* pool exhausted: drain in-flight, retry once */
				dma_async_issue_pending(chan);
				wedged = io_dma_fmw_wait(chan, cookies,
							 &nr_cookies, &redo);
				if (wedged)
					goto collect_done;
				tx = dmaengine_prep_dma_memcpy(chan, dst,
						src_dma, len,
						io_dma_prep_flags());
			}
			if (!tx) {
				io_dma_fmw_record(IO_DMA_FMW_EAGAIN);
				redo = true;
				/* Two drain-and-retry failures in one write
				 * means the pools are held empty by other
				 * rings: surrender the remainder to CPU. */
				if (++prep_fails >= 2) {
					surrendered = true;
					break;	/* redo covers this chunk */
				}
			} else {
				ck = dmaengine_submit(tx);
				if (dma_submit_error(ck)) {
					redo = true;
				} else {
					cookies[nr_cookies++] =
						(struct io_dma_fmw_ck){
							.ck = ck, .len = len,
							.submit_ns = ktime_get_ns(),
						};
					io_dma_qstat_submit(chan, len, true);
				}
				if (nr_cookies == max_cookies) {
					dma_async_issue_pending(chan);
					wedged = io_dma_fmw_wait(chan, cookies,
							&nr_cookies, &redo);
					if (wedged)
						goto collect_done;
				}
			}
			sub += len;
		}
record:
		fol[nr_fol++] = (struct io_dma_fmw_folio){
			.folio = folio, .fsdata = fsdata, .pos = pos,
			.len = bytes, .dst_dma = dst_dma, .map_len = bytes,
		};
		written += bytes;
	}

collect_done:
	if (!wedged) {
		dma_async_issue_pending(chan);
		wedged = io_dma_fmw_wait(chan, cookies, &nr_cookies, &redo);
	}

	/* Unmap dst IOVAs. After a timeout this also FENCES late DMA writes:
	 * they fault in the IOMMU instead of landing in reclaimed memory. */
	for (i = 0; i < nr_fol; i++)
		if (fol[i].dst_dma)
			dma_unmap_page(dev, fol[i].dst_dma, fol[i].map_len,
				       DMA_FROM_DEVICE);

	if (unlikely(wedged)) {
		/* Folio contents unknown and a stray write may still land:
		 * leak the locked folios rather than expose them. */
		pr_warn_ratelimited("io_uring DMA write: wedged after %dms, leaking %u folios (%s)\n",
				    IO_DMA_FMW_WAIT_MS, nr_fol,
				    dma_chan_name(chan));
		io_dma_fmw_record(IO_DMA_FMW_ERROR);
		written = 0;
		err = -EIO;
		goto out_unlock;
	}

	if (unlikely(redo)) {
		/* Re-copy every chunk by CPU: same source bytes to the same
		 * folio ranges, so overlap with completed DMA is idempotent.
		 * The iter was never advanced during collection. */
		for (i = 0; i < nr_fol; i++) {
			size_t n = copy_folio_from_iter(fol[i].folio,
					offset_in_folio(fol[i].folio, fol[i].pos),
					fol[i].len, from);
			if (unlikely(n != fol[i].len)) {
				written = fol[i].pos - iocb->ki_pos + n;
				break;
			}
		}
		io_dma_fmw_record(IO_DMA_FMW_CPU_REDO);
	} else {
		iov_iter_advance(from, written);
	}

	/* Commit in ascending order: dirty + unlock + i_size updates. */
	{
		ssize_t committed = 0;

		for (i = 0; i < nr_fol; i++) {
			size_t claim = min_t(size_t, fol[i].len,
					     written - committed);
			int done;

			done = aops->write_end(iocb, mapping, fol[i].pos,
					       fol[i].len, claim,
					       fol[i].folio, fol[i].fsdata);
			if (unlikely(done < 0)) {
				if (!committed)
					err = done;
				break;
			}
			committed += done;
			if ((size_t)done < fol[i].len)
				break;
			balance_dirty_pages_ratelimited(mapping);
		}
		written = committed;
	}

	if (written > 0)
		iocb->ki_pos += written;
out_unlock:
	inode_unlock(inode);
	kvfree(fol);
	kvfree(cookies);
	if (written > 0) {
		if (written < want)
			atomic64_inc(&io_dma_fmw_short);
		return generic_write_sync(iocb, written) ?: written;
	}
	return err;
}

/*
 * Release a completed (or aborted) task's source resources: DMA unmaps --
 * IOVA frees and IOTLB work under an IOMMU -- and folio references.  This
 * is the expensive half of completion and needs no ctx->dma.lock; callers
 * run it before taking the lock for __io_dma_task_complete() so the lock
 * hold shrinks to the refcount handshake.
 */
void io_dma_task_release_res(struct io_ring_ctx *ctx, struct device *dev,
			     struct io_dma_task *dma)
{
	if (dma->is_batch) {
		int i;

		for (i = 0; i < dma->batch_nr; i++) {
			struct io_dma_batch_entry *e = &dma->batch_entries[i];

			if (e->pfn_map) {
				/* Cached mapping: drop the in-flight
				 * reference; the unmap belongs to
				 * eviction/flush, not this I/O. */
				io_pfn_map_put(e->pfn_map);
			} else if (!ctx->dma.use_phys_addrs) {
				dma_unmap_page(dev, e->src_dma, e->src_len,
					       DMA_TO_DEVICE);
			}
			folio_put(e->folio);
		}
		kfree(dma->batch_entries);
	} else {
		if (dma->src_pfn_map) {
			io_pfn_map_put(dma->src_pfn_map);
		} else if (dma->src_map_len && !ctx->dma.use_phys_addrs) {
			if (dma->src_is_page)
				dma_unmap_page(dev, dma->src_map_addr,
					       dma->src_map_len,
					       DMA_TO_DEVICE);
			else
				dma_unmap_single(dev, dma->src_map_addr,
						 dma->src_map_len,
						 DMA_TO_DEVICE);
		}
		if (dma->src_folio)
			folio_put(dma->src_folio);
	}
}

/*
 * Caller must hold ctx->dma.lock and must have already released the task's
 * source resources via io_dma_task_release_res().
 */
static void __io_dma_task_complete(struct device *dev, struct io_dma_task *dma,
				   int ret)
{
	struct io_kiocb *req;
	u32 task_len;

	req = dma->req;
	task_len = dma->len;

	if (ret == DMA_COMPLETE) {
		if (dma->submit_ns)
			io_dma_lat_record(&io_dma_lat_dma, task_len,
					  ktime_get_ns() - dma->submit_ns);
		pr_debug("dma task complete: len=%u result=%d\n",
			 task_len, req->dma.dma_result);
		if (req->dma.dma_result >= 0)
			req->dma.dma_result += task_len;
	} else {
		pr_debug("dma task failed: len=%u ret=%d\n", task_len, ret);
		req->dma.dma_result = -EFAULT;
	}

	io_dma_qstat_complete(req->ctx->dma.chan, task_len,
			      dma->submit_ns, false);

	/* Free the task before touching refcnt -- task_len saved above */
	io_dma_task_free(req->ctx, dma);
	req->dma.dma_refcnt--;

	if (req->dma.dma_refcnt == 0) {
		pr_debug("dma req done: opcode=%d result=%d\n",
			 req->opcode, req->dma.dma_result);

		if (req->dma.dma_result < 0) {
			req_set_fail(req);
			io_req_set_res(req, req->dma.dma_result,
				       req->dma.saved_cflags);
			req->io_task_work.func = io_req_task_complete;
			io_req_task_work_add(req);
		} else {
			/*
			 * Terminal completion via task_work: __io_dma_poll()
			 * may run from a workqueue without ctx->uring_lock,
			 * so io_req_task_complete() defers the CQE post until
			 * the task_work loop acquires it.
			 */
			io_req_set_res(req, req->dma.saved_res,
				       req->dma.saved_cflags);
			req->io_task_work.func = io_req_task_complete;
			io_req_task_work_add(req);
		}

		/*
		 * Drop the in-flight DMA reference taken in
		 * io_dma_submit_queued_tasks(). The completion handling above
		 * only queues task_work — it does not free the req inline —
		 * so the req is still valid here. If we held the last
		 * reference (the req was already terminally
		 * completed/cancelled, e.g. by teardown), free it now;
		 * otherwise the owner frees it once it drops its reference.
		 */
		if (req->dma.dma_ref_held) {
			req->dma.dma_ref_held = false;
			atomic_inc(&req->ctx->dma.diag_refs_dropped);
			if (req_ref_put_and_test(req))
				io_free_req(req);
		}
	}
}

int io_dma_submit_queued_tasks(struct io_kiocb *req)
{
	struct io_ring_ctx *ctx = req->ctx;
	int ret = 0;

	if (IS_ERR_OR_NULL(ctx->dma.chan))
		return 0;

	if (req->dma.dma_active) {
		pr_debug("submit_queued: refcnt=%d\n", req->dma.dma_refcnt);
		if (req->dma.dma_refcnt > 0) {
			unsigned long flags;

			/*
			 * Take the in-flight DMA reference BEFORE the tasks
			 * become reachable by any completer. The hardware
			 * doorbell was already rung by the submit path
			 * (io_dma_filemap_read()), so the instant the tasks
			 * are published to the submit_list below a
			 * concurrent drain -- the poll_work kworker or
			 * io_ring_exit_work -- can reach dma_refcnt == 0 in
			 * __io_dma_task_complete(). Were the ref taken after
			 * publishing, that drain would complete the req, see
			 * dma_ref_held == false and drop nothing, and this path
			 * would then set refcount/dma_ref_held on an
			 * already-completed req: double-complete / orphaned ref /
			 * use-after-free. Taking it here (before publish)
			 * closes the window; dropped in
			 * __io_dma_task_complete() at dma_refcnt == 0.
			 * Mirrors io_wq_submit_work().
			 *
			 * The take must ALSO be under ctx->dma.lock, the lock
			 * every completer holds through its dma_refcnt == 0
			 * block. A reissue can reach here while the PREVIOUS
			 * cycle's completer may still be between completing the
			 * request and dropping the previous in-flight ref.
			 * Taken unlocked, this ref/flag write interleaves into
			 * that window: the completer then either skips its drop
			 * or donates it to the new cycle, and one reference
			 * leaks each time. The lock orders this take strictly
			 * after that drop.
			 */
			spin_lock_irqsave(&ctx->dma.lock, flags);
			if (!(req->flags & REQ_F_REFCOUNT))
				__io_req_set_refcount(req, 2);
			else
				req_ref_get(req);
			req->dma.dma_ref_held = true;
			atomic_inc(&ctx->dma.diag_refs_taken);
			spin_unlock_irqrestore(&ctx->dma.lock, flags);

			/*
			 * Publish the req's tasks to the lock-free
			 * submit_list -- AFTER the ref-take above, so no
			 * drain can reach dma_refcnt == 0 before the
			 * in-flight ref exists.  Every task was fully
			 * submitted with a valid cookie (tasks are linked
			 * only after dmaengine_submit() succeeds), so the
			 * poller can consume them as-is.  Publishing in
			 * chain order makes the llist a LIFO of a FIFO; the
			 * consumer reverses it back (submissions are
			 * serialized by uring_lock, so no other producer
			 * interleaves).
			 */
			struct io_dma_task *t = req->dma.dma_tasks;

			while (t) {
				/* read ->next BEFORE publishing: a
				 * published task can complete and be
				 * freed immediately */
				struct io_dma_task *nxt = t->next;

				llist_add(&t->llnode, &ctx->dma.submit_list);
				t = nxt;
			}

			req->dma.dma_tasks = NULL;
			req->dma.dma_tasks_tail = NULL;
			ret = -EIOCBQUEUED;
		}

		req->dma.dma_active = false;
	}

	/*
	 * Use ctx (not req) below — __io_dma_poll may complete the
	 * request and free it, so req must not be dereferenced after.
	 */
	if (ret == -EIOCBQUEUED) {
		/*
		 * This req's tasks were just queued and the issuer is about to
		 * return with the req handed off. Do NOT complete them inline
		 * here: a fast transfer (especially a single batched
		 * descriptor) can finish synchronously, and
		 * __io_dma_task_complete() would then complete this same req
		 * while the issuer is still unwinding. Defer all completion to
		 * a clean context: the poll_work kworker.
		 */
		if (io_dma_pending(ctx)) {
			/*
			 * Unbound, NOT schedule_work(): the per-CPU
			 * pool would run the poller on THIS (the
			 * submitter's) CPU, so detection while the app
			 * computes depends on the kworker winning a
			 * wakeup-preemption fight with the app thread.
			 * That fight is fragile -- measured: an inline
			 * CQ-wait drain shifting the kworkers' runtime
			 * profile was enough to make preemption stop
			 * happening, leaving completions undetected for
			 * the whole app compute slice (~83us) instead
			 * of ~10us. An unbound worker lands on an idle
			 * CPU and detects in parallel with the app.
			 */
			queue_work(system_unbound_wq,
				   &ctx->dma.poll_work);
		}
	} else if (atomic_read(&ctx->dma.poll_armed) == 0) {
		/*
		 * No task was queued for this req (e.g. CPU fallback). Draining
		 * other reqs' already-queued tasks inline is safe — completion
		 * for a different req is ordinary async wakeup, not re-entrancy
		 * on the req currently being issued.
		 */
		__io_dma_poll(ctx);
	}

	return ret;
}

int __io_dma_poll(struct io_ring_ctx *ctx)
{
	struct io_dma_task *dma, *next;
	int ret;
	struct device *dev;
	int count;
	unsigned long flags;

	if (atomic_cmpxchg(&ctx->dma.poll_armed, 0, 1) != 0)
		return 0;

	if (IS_ERR_OR_NULL(ctx->dma.chan))
		goto out_disarm;

	/*
	 * No doorbell here: pollable descriptors reach the hardware at
	 * dmaengine_submit() time (see idxd_dma_tx_submit), and every
	 * submitter rings for its own tasks from the issue path, under
	 * uring_lock. Nothing on the pending lists needs a kick.
	 */
	dev = ctx->dma.chan->device->dev;

	/*
	 * Splice newly submitted tasks onto the consumer-owned poll_list.
	 * Only the armed poller (poll_armed) touches poll_list, so no lock:
	 * the llist arrives newest-first, reversing it restores submission
	 * order.  (Completion is NOT in-order -- DSA WQs are fed by multiple
	 * engines -- so the walk below scans the whole list, not just the
	 * head.)
	 */
	{
		struct llist_node *node = llist_del_all(&ctx->dma.submit_list);
		struct io_dma_task *fifo_head = NULL, *fifo_tail = NULL;

		while (node) {
			struct io_dma_task *t =
				llist_entry(node, struct io_dma_task, llnode);

			node = node->next;
			t->next = fifo_head;	/* prepend reverses LIFO->FIFO */
			if (!fifo_head)
				fifo_tail = t;
			fifo_head = t;
		}
		if (fifo_head) {
			if (ctx->dma.poll_list_tail)
				ctx->dma.poll_list_tail->next = fifo_head;
			else
				WRITE_ONCE(ctx->dma.poll_list, fifo_head);
			ctx->dma.poll_list_tail = fifo_tail;
		}
	}

	{
		struct io_dma_task *prev = NULL;

		dma = ctx->dma.poll_list;
		count = 0;
		pr_debug("poll: head=%p\n", dma);
		while (dma != NULL) {
			next = dma->next;

			/* No lock around the hardware poll: the cookie state
			 * is the dmaengine's own (safe lockless), and holding
			 * ctx->dma.lock across the whole walk is what made
			 * submitters fight the poller for it (measured 3.6% of
			 * node cycles in queued_spin_lock_slowpath under 60KB
			 * sets). */
			ret = dmaengine_async_is_tx_complete(ctx->dma.chan,
							     dma->cookie);
			if (ret == DMA_IN_PROGRESS) {
				/*
				 * Keep walking: DSA groups feed each WQ from
				 * multiple engines, so completion is NOT
				 * in-order.  Stopping at the first in-flight
				 * entry head-of-line-blocked every completed
				 * task behind it -- reaped chunks (and the
				 * netty DMA slab they pin) stayed held for
				 * the head's full latency.  Each extra check
				 * is one completion-record read.
				 */
				prev = dma;
				dma = next;
				continue;
			}

			/* Unlink before completing (complete may free dma) */
			if (prev)
				prev->next = next;
			else
				WRITE_ONCE(ctx->dma.poll_list, next);
			if (!next)
				ctx->dma.poll_list_tail = prev;

			/* Heavy resource release (IOMMU unmaps, folio puts)
			 * runs unlocked; ctx->dma.lock then covers only the
			 * refcount handshake with the submitter's ref-take
			 * and the IRQ completion path. */
			io_dma_task_release_res(ctx, dev, dma);
			spin_lock_irqsave(&ctx->dma.lock, flags);
			__io_dma_task_complete(dev, dma, ret);
			spin_unlock_irqrestore(&ctx->dma.lock, flags);

			count++;
			dma = next;
		}
	}

	pr_debug("poll: completed=%d remaining=%s\n",
		 count, io_dma_pending(ctx) ? "yes" : "no");

out_disarm:
	/* Release ordering hands poll_list (written lock-free above) to
	 * whichever thread arms the poller next. */
	atomic_set_release(&ctx->dma.poll_armed, 0);
	return io_dma_pending(ctx) ? 1 : 0;
}

/*
 * Called from io_cqring_wait_schedule() right before the task would block.
 * If this ring has pollable (non-IRQ) DMA tasks in flight, spin on their
 * completion records for at most io_dma_cq_poll_us. Completions found here
 * run __io_dma_task_complete() in this task's context; the CQE itself is
 * posted by the poll task_work that queues (io_poll_kick -> this task), so
 * report progress via the pending-work checks and let the wait loop run it.
 *
 * Returns true if the caller should skip sleeping and re-run its wait loop
 * (task_work/CQEs are ready), false to fall through to schedule(). IRQ-mode
 * tasks never appear on the pending lists, so this is a no-op for them, as it
 * for rings with no DMA channel.
 */
/*

 * Teardown diagnostic, called from io_ring_exit_work() when the ring's
 * references fail to drop within the timeout. The dump distinguishes the
 * wedge classes before teardown proceeds anyway:
 *
 *  - task(s) on the pending lists: a poll-mode descriptor stuck DMA_IN_PROGRESS.
 *    The head task is the wedged one (__io_dma_poll() stops at the first
 *    incomplete cookie); anything behind it may just be queued behind the
 *    wedge.
 *  - empty list but nr_req_allocated > 0: either an IRQ-mode descriptor
 *    whose interrupt never arrived (interrupt cookies are invisible to
 *    polling by design -- idxd_dma_tx_status() returns DMA_IN_PROGRESS for
 *    them until the IRQ handler invalidates the cookie), or a ctx reference
 *    held by something other than DMA.
 */
void io_dma_dump_stuck(struct io_ring_ctx *ctx)
{
	struct io_dma_task *dma;
	int count = 0, shown = 0;

	pr_err("io_uring DMA teardown stuck: ctx=%p nr_req_allocated=%u chan=%s poll_armed=%d\n",
	       ctx, ctx->nr_req_allocated,
	       IS_ERR_OR_NULL(ctx->dma.chan) ? "none" :
			dma_chan_name(ctx->dma.chan),
	       atomic_read(&ctx->dma.poll_armed));

	/*
	 * refs_taken vs refs_dropped says whether every in-flight DMA req
	 * ref was released; an imbalance locates the wedge in the ref-drop
	 * path of __io_dma_task_complete().
	 */
	pr_err("io_uring DMA teardown stuck: refs_taken=%d refs_dropped=%d\n",
	       atomic_read(&ctx->dma.diag_refs_taken),
	       atomic_read(&ctx->dma.diag_refs_dropped));

	if (IS_ERR_OR_NULL(ctx->dma.chan))
		return;

	/* poll_list belongs to the (quiesced-by-now) poller and submit_list
	 * is lock-free; this walk is racy by design -- it only feeds a
	 * diagnostic dump on an already-wedged teardown. */
	for (dma = READ_ONCE(ctx->dma.poll_list); dma; dma = dma->next) {
		count++;
		if (shown < 8) {
			struct io_kiocb *req = dma->req;

			pr_err("  task=%p cookie=0x%08x len=%u batch=%d req=%p refcnt=%u ref_held=%d\n",
			       dma, dma->cookie, dma->len, dma->is_batch, req,
			       req->dma.dma_refcnt, req->dma.dma_ref_held);
			shown++;
		}
	}

	pr_err("io_uring DMA teardown stuck: %d task(s) on poll_list, submit_list %sempty\n",
	       count, llist_empty(&ctx->dma.submit_list) ? "" : "NON-");
}

bool io_dma_cq_wait_poll(struct io_ring_ctx *ctx, struct io_wait_queue *iowq)
{
	unsigned int budget_us = READ_ONCE(io_dma_cq_poll_us);
	u64 end_ns;

	if (!budget_us || IS_ERR_OR_NULL(ctx->dma.chan))
		return false;
	if (!io_dma_pending(ctx))
		return false;

	end_ns = ktime_get_ns() + (u64)budget_us * NSEC_PER_USEC;
	for (;;) {
		__io_dma_poll(ctx);

		if (task_work_pending(current) || io_local_work_pending(ctx) ||
		    io_should_wake(iowq))
			return true;
		if (!io_dma_pending(ctx))
			return false;
		if (need_resched() || task_sigpending(current))
			return false;
		if (ktime_get_ns() >= end_ns)
			return false;
		cpu_relax();
	}
}

