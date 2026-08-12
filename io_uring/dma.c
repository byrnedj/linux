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
#include <linux/sizes.h>
#include <linux/timekeeping.h>
#include <linux/debugfs.h>
#include <linux/task_work.h>
#include "io_uring.h"
#include "rsrc.h"
#include "refs.h"

struct kmem_cache *dma_cachep;

/*
 * DSA cache control. When enabled (1, the default), destination writes
 * are cache-allocating (IDXD_OP_FLAG_CC) so that data the application
 * reads back immediately is warm in cache. When disabled (0), writes
 * bypass the cache. This is configurable via
 * /proc/sys/kernel/io_uring_dma_cache_control.
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
 * on the DMA source paths is the per-chunk dma_map and dma_unmap.
 * Destinations are already persistent since registered buffers map
 * once at registration.  This gives sources the same discipline,
 * lazily.  The first chunk touching a folio maps its
 * io_dma_map_quantum() sized segment DMA_TO_DEVICE and caches the
 * segment-head PFN to dma_addr translation in a per-device xarray.
 * Later chunks on that segment are pure arithmetic.  dma_unmap runs
 * only on the bytes-capped CLOCK eviction and on explicit flush.
 *
 * This is correct because the struct page to physical address relation
 * is immutable and descriptors are only issued against pages held live
 * by the I/O being processed through the filemap folio reference, so a
 * stale cached translation can never misdirect DMA.  Staleness costs
 * IOVA space and a lingering device read window, bounded by the cap.
 * This is the same trade that page_pool's persistent NIC mappings
 * make.
 *
 * The entry lifetime is refs = 1 cache bias plus one per in-flight
 * batch entry.  Lookup takes a ref with atomic_inc_not_zero() under
 * RCU.  Eviction erases the entry and drops the bias, so a mapping
 * survives until its last in-flight user completes.
 *
 * Mappings are keyed to the DSA struct device and are not torn down on
 * driver unbind, so flush via debugfs before unbinding idxd.
 */

struct io_pfn_map {
	unsigned long		pfn;		/* segment-head PFN, the cache key */
	dma_addr_t		dma_base;	/* segment DMA_TO_DEVICE mapping */
	unsigned int		size;		/* mapped bytes (<= cache quantum) */
	atomic_t		refs;		/* cache bias + in-flight users */
	bool			referenced;	/* CLOCK second-chance bit */
	unsigned long		last_used;	/* jiffies, for age-based retire */
	struct device		*dev;		/* unmap handle */
	struct rcu_head		rcu;
};

struct io_pfn_cache {
	struct xarray		xa;
	spinlock_t		lock;		/* serializes the CLOCK sweep */
	unsigned long		hand;		/* next PFN the sweep visits */
	unsigned long		next_age;	/* jiffies, next age sweep due */
	struct device		*dev;
	size_t			quantum;	/* pow2 segment size entries are
						 * carved into (io_dma_map_quantum)
						 */
	atomic64_t		covered;	/* bytes mapped through the tree */
	atomic64_t		hits;
	atomic64_t		misses;
	atomic64_t		inserts;
	atomic64_t		insert_fails;	/* alloc/map/xa failure, plain map */
	atomic64_t		range_fallbacks;/* chunk not coverable by one entry */
	atomic64_t		evictions;
	atomic64_t		age_evictions;	/* retired for idleness */
	atomic64_t		ref_skips;	/* sweep passed an in-flight entry */
};

#define IO_PFN_CACHE_DEVS	16
static struct io_pfn_cache *io_pfn_caches[IO_PFN_CACHE_DEVS];
static struct device *io_pfn_cache_devs[IO_PFN_CACHE_DEVS];
static DEFINE_SPINLOCK(io_pfn_cache_reg_lock);

/*
 * Covered-bytes cap in MiB, where 0 disables the cache entirely.  A cap
 * under a cycling working set makes the sweep thrash with an evict and
 * remap per I/O, which is worse than no cache, so a deployment that
 * wants the cache should size this above its source working set.  The
 * default is deliberately modest rather than generous, because every
 * mapped byte is a byte the device can read until the entry is retired.
 */
static u32 io_dma_pfn_cache_cap_mb __read_mostly = 1024;

/*
 * Age at which an idle entry is retired even though the cache is under
 * its cap, in milliseconds, where 0 keeps entries until the cap forces
 * them out.  The cap alone bounds the standing device-readable window
 * in bytes and says nothing about how long any one mapping persists,
 * so this bounds it in time: a segment that no I/O has touched for
 * this long is unmapped.
 */
static u32 io_dma_pfn_cache_max_age_ms __read_mostly = 60000;

/* Sweep visit budget per eviction call.  This bounds the datapath
 * latency when the table is large and mostly referenced or in flight.
 */
#define IO_PFN_EVICT_BUDGET	1024

/* Unmaps performed inline per eviction call. An insert overshoots the
 * cap by at most one segment, so the steady state evicts about one
 * entry. Lowering the cap at runtime is the case that would otherwise
 * unmap the whole excess in one sweep, with preemption disabled, so
 * cap the inline work and leave the rest to the following inserts.
 */
#define IO_PFN_EVICT_UNMAP_MAX	64

/*
 * Standing mappings are carved into power-of-two segments no larger
 * than dma_opt_mapping_size().  IOVA allocations above that limit
 * bypass the IOMMU's per-CPU rcaches and fall to the domain rbtree
 * under its lock.  We cap at 2MB so that a no-IOMMU SIZE_MAX answer
 * degenerates to whole-folio behaviour.
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
		/* acquire pairs with the release publishing the slot below */
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
		if (io_pfn_cache_devs[i] == dev) {	/* We lost an insert race. */
			spin_unlock(&io_pfn_cache_reg_lock);
			kfree(c);
			return io_pfn_caches[i];
		}
		if (!io_pfn_cache_devs[i]) {
			/*
			 * Pin the device for the registry's machine
			 * lifetime. Slots are never released, so this
			 * reference is deliberately never dropped; it
			 * turns the flush-before-hot-remove discipline
			 * into an enforced invariant.
			 */
			get_device(dev);
			io_pfn_caches[i] = c;
			/* pairs with the lockless load above */
			smp_store_release(&io_pfn_cache_devs[i], dev);
			spin_unlock(&io_pfn_cache_reg_lock);
			return c;
		}
	}
	spin_unlock(&io_pfn_cache_reg_lock);
	kfree(c);	/* The registry is full, so this device runs uncached. */
	return NULL;
}

/* Drop one reference. The last dropper unmaps and frees. */
static void io_pfn_map_put(struct io_pfn_map *pm)
{
	if (!atomic_dec_and_test(&pm->refs))
		return;
	dma_unmap_page(pm->dev, pm->dma_base, pm->size, DMA_TO_DEVICE);
	kfree_rcu(pm, rcu);
}

/*
 * The CLOCK sweep advances the hand from where it last stopped.  It
 * gives referenced entries a second chance and skips entries with
 * in-flight users.  The second chance provides scan resistance since a
 * streaming pattern cannot flush the recycling working set, whose
 * entries keep their bit set.  The sweep runs on the submit path after
 * an insert pushes covered past the cap, so both the trylock, which
 * fails when another submitter is already sweeping, and the visit
 * budget bound the added latency.
 */
/*
 * Retire entries, under two policies that share one walk.
 *
 * Over the cap, the CLOCK hand advances from where it stopped and
 * gives referenced entries a second chance, which is what keeps a
 * streaming pattern from flushing a recycling working set.
 *
 * Independently of the cap, an entry that no I/O has touched for
 * max_age is retired on sight. Age takes precedence over the second
 * chance, since an idle entry's reference bit only records that it was
 * used at some point in the past, not recently.
 */
static void io_pfn_cache_evict(struct io_pfn_cache *c, u64 cap)
{
	unsigned long max_age = READ_ONCE(io_dma_pfn_cache_max_age_ms);
	struct io_pfn_map *pm;
	unsigned long index, age_before = 0;
	int budget = IO_PFN_EVICT_BUDGET;
	int unmaps = IO_PFN_EVICT_UNMAP_MAX;
	bool over, aging;
	int pass;

	over = atomic64_read(&c->covered) > cap;
	aging = max_age && time_after(jiffies, READ_ONCE(c->next_age));
	if (!over && !aging)
		return;

	if (!spin_trylock(&c->lock))
		return;

	if (aging) {
		age_before = jiffies - msecs_to_jiffies(max_age);
		/* Rate-limit the age walk; the cap walk runs on demand. */
		WRITE_ONCE(c->next_age, jiffies + HZ);
	}

	for (pass = 0; pass < 2; pass++) {
		unsigned long start = pass ? 0 : c->hand;

		if (!over && !aging)
			break;

		xa_for_each_start(&c->xa, index, pm, start) {
			bool aged = aging &&
				time_before(READ_ONCE(pm->last_used), age_before);

			if (--budget <= 0) {
				c->hand = index + 1;
				goto out;
			}
			if (atomic_read(&pm->refs) > 1) {
				atomic64_inc(&c->ref_skips);
				continue;
			}
			if (!aged) {
				if (!over)
					continue;
				if (READ_ONCE(pm->referenced)) {
					WRITE_ONCE(pm->referenced, false);
					continue;
				}
			}
			xa_erase(&c->xa, index);
			atomic64_sub(pm->size, &c->covered);
			atomic64_inc(&c->evictions);
			if (aged)
				atomic64_inc(&c->age_evictions);
			io_pfn_map_put(pm);	/* Drop the cache bias. */
			over = atomic64_read(&c->covered) > cap;
			if (--unmaps <= 0) {
				c->hand = index + 1;
				goto out;
			}
			if (!over && !aging) {
				c->hand = index + 1;
				goto out;
			}
		}
		c->hand = 0;
		/* One full pass is enough to retire every aged entry. */
		aging = false;
	}
out:
	spin_unlock(&c->lock);
}

/*
 * Look up or create the persistent mapping covering [offset,
 * offset+len) from the head of @folio and return it with an in-flight
 * reference taken.  *dma is set to the chunk's device address.  NULL
 * means the caller should fall back to a plain per-chunk map and is
 * never an error.
 *
 * Entries are quantized.  The folio is carved into c->quantum segments
 * from its head, each cached and mapped independently, so that every
 * IOVA allocation stays inside the IOMMU's per-CPU rcache size
 * classes.  The chunk must lie within one segment.  The filemap-read
 * path splits chunks at segment boundaries to guarantee this and a
 * chunk that crosses one falls back to a plain map.
 *
 * @map_len is the known physically contiguous extent from the folio
 * head, which is folio_size() for page-cache folios.
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
			 * The run outgrew the cached region.  Fused striding
			 * runs vary in length and a recycled PFN may carry a
			 * shorter mapping.  We displace the entry and fall
			 * through to remap the larger run under the same
			 * key.  In-flight users of the old mapping stay safe
			 * via the bias protocol.
			 */
			if (xa_cmpxchg(&c->xa, pfn, pm, NULL,
				       GFP_NOWAIT | __GFP_NOWARN) == pm) {
				atomic64_sub(pm->size, &c->covered);
				io_pfn_map_put(pm);	/* the cache bias */
			}
			io_pfn_map_put(pm);		/* the lookup ref */
			atomic64_inc(&c->range_fallbacks);
			goto miss;
		}
		WRITE_ONCE(pm->referenced, true);
		WRITE_ONCE(pm->last_used, jiffies);
		atomic64_inc(&c->hits);
		*dma = pm->dma_base + rel;
		return pm;
	}
	rcu_read_unlock();
miss:
	atomic64_inc(&c->misses);

	pm = kmalloc_obj(*pm, GFP_NOWAIT | __GFP_NOWARN);
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
	pm->last_used = jiffies;
	atomic_set(&pm->refs, 2);	/* the cache bias plus this I/O */

	rcu_read_lock();
	old = xa_cmpxchg(&c->xa, pfn, NULL, pm, GFP_NOWAIT | __GFP_NOWARN);
	if (old) {
		/* We lost an insert race or the xarray node allocation
		 * failed.
		 */
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
			WRITE_ONCE(old->last_used, jiffies);
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

/* Erase everything.  In-flight users keep their mappings alive until
 * their references drop.
 */
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

static bool io_pfn_cache_usable(void)
{
	return READ_ONCE(io_dma_pfn_cache_cap_mb) != 0;
}

static int io_pfn_cache_stats_show(struct seq_file *m, void *p)
{
	int i;

	seq_printf(m, "cap_mb %u max_age_ms %u\n",
		   READ_ONCE(io_dma_pfn_cache_cap_mb),
		   READ_ONCE(io_dma_pfn_cache_max_age_ms));
	for (i = 0; i < IO_PFN_CACHE_DEVS; i++) {
		struct io_pfn_cache *c;

		/* acquire pairs with the slot-publishing release in
		 * io_pfn_cache_get()
		 */
		if (!smp_load_acquire(&io_pfn_cache_devs[i]))
			break;
		c = io_pfn_caches[i];
		seq_printf(m,
			   "dev %s quantum_kb %zu covered_kb %lld hits %lld misses %lld inserts %lld insert_fails %lld range_fallbacks %lld evictions %lld age_evictions %lld ref_skips %lld\n",
			   dev_name(c->dev),
			   c->quantum >> 10,
			   atomic64_read(&c->covered) >> 10,
			   atomic64_read(&c->hits),
			   atomic64_read(&c->misses),
			   atomic64_read(&c->inserts),
			   atomic64_read(&c->insert_fails),
			   atomic64_read(&c->range_fallbacks),
			   atomic64_read(&c->evictions),
			   atomic64_read(&c->age_evictions),
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
		/* acquire pairs with the slot-publishing release in
		 * io_pfn_cache_get()
		 */
		if (!smp_load_acquire(&io_pfn_cache_devs[i]))
			break;
		io_pfn_cache_flush(io_pfn_caches[i]);
	}
	return len;
}

static const struct file_operations io_pfn_cache_flush_fops = {
	.owner		= THIS_MODULE,
	.open		= simple_open,
	.write		= io_pfn_cache_flush_write,
	.llseek		= noop_llseek,
};

/*
 * Busy-poll budget in microseconds for draining in-flight DMA
 * completions from the CQ-wait path (io_dma_cq_wait_poll()) before the
 * waiting task commits to sleeping. A DSA transfer for a typical read
 * chunk completes in about 4 to 6us plus queueing. Sleeping instead
 * costs a kworker schedule_work plus a wakeup, which is about 10us
 * more when the poller is idle, just to be woken again. 0 disables
 * the poll. This is tunable via debugfs io_uring_dma_cq_poll_us.
 */
static unsigned int io_dma_cq_poll_us __read_mostly = 20;

/*
 * Inline spin budget in microseconds for the DMA filemap-write wait.
 * The write path is synchronous per request. A typical request's batch
 * completes in tens of microseconds, so sleeping immediately trades a
 * usleep wakeup of about 100us for about 35us of engine time and
 * halves throughput at small request sizes. We spin with cond_resched
 * for up to this long before backing off to sleeping. The backoff
 * still protects the many-rings contention case. 0 sleeps immediately.
 * This is tunable via debugfs io_uring_dma_fmw_spin_us.
 */
static unsigned int io_dma_fmw_spin_us __read_mostly = 60;

/*
 * How long the DMA filemap-write wait gives the device before it
 * declares the transfer wedged, fences the destination ranges and
 * fails the write. Tunable via debugfs io_uring_dma/fmw_wait_ms.
 */
static unsigned int io_dma_fmw_wait_ms __read_mostly = 5000;

/*
 * Per-transaction DMA latency tracking, binned by transfer size. A
 * transaction is one io_dma_task, either a descriptor or a batch. It
 * is stamped at hardware submit and sampled when the poller observes
 * completion, so the bins measure detection-inclusive latency.
 *
 * Sampling costs two clock reads per transaction, which is noise
 * against a multi-kilobyte transfer but is unconditional on the
 * datapath. Setting io_uring_dma/stats to 0 skips both reads and
 * leaves the histogram frozen.
 */
static unsigned int io_dma_stats_enabled __read_mostly = 1;

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
 * Filemap DMA-read gate and result counters. The read path falls back
 * to the normal buffered read silently on any failure, so a per-reason
 * count is the only way to see whether it engages at all. These are
 * recorded from the io_read gate in io_uring/rw.c and from the submit
 * loop.
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
 * Per-channel queueing diagnostics for the channel-sharing design
 * work.  seen_kb is the number of bytes already in flight on the
 * channel at each submit, meaning the queue the op joins, in log2 KB
 * buckets.  lat_us is the submit-to-complete latency the op then
 * experienced, in log2 us buckets, and it includes service time and
 * detection.  There is also an in-flight high-water mark.  Bucket 0 is
 * "<1", bucket k is [2^(k-1), 2^k), and the last bucket is the
 * catch-all.  The registry is lockless and in submission order, like
 * io_pfn_cache_devs.  The stats are dumped via io_uring_dma_chan_qstat
 * and the counters, but not the live in-flight bytes, are cleared by
 * latency_reset.
 */
#define IO_DMA_STAT_DEVS	16
#define IO_DMA_STAT_CHANS	8
#define IO_DMA_QSTAT_NBUCKETS	16

struct io_dma_chan_qstat {
	struct dma_chan	*chan;		/* identity only, may be stale after
					 * a WQ reconfig, never dereference
					 */
	char		name[24];	/* captured at registration */
	atomic64_t	inflight;	/* bytes submitted, not yet completed */
	atomic64_t	hwm;		/* max in-flight bytes seen */
	atomic64_t	submits;
	atomic64_t	bytes;
	atomic64_t	lat_us[IO_DMA_QSTAT_NBUCKETS];
	atomic64_t	seen_kb[IO_DMA_QSTAT_NBUCKETS];
};
/*
 * One registry, keyed by device, holding that device's channels. A
 * channel knows its device, so the submit and complete hooks resolve
 * the device slot once and then index the channel within it, instead
 * of scanning two independent tables. Both keys are stored for
 * identity only and are never dereferenced after registration.
 */
struct io_dma_dev_stat {
	void		*key;		/* dma_device, identity only */
	char		name[24];	/* captured at registration */
	atomic64_t	inflight;	/* all classes, descriptors */
	atomic64_t	inflight_wr;	/* write-sink descriptors */
	atomic64_t	rej_rd;
	atomic64_t	rej_wr;
	struct io_dma_chan_qstat chans[IO_DMA_STAT_CHANS];
};
static struct io_dma_dev_stat io_dma_dev_stats[IO_DMA_STAT_DEVS];

static struct io_dma_dev_stat *io_dma_dev_stat_get(struct dma_chan *chan)
{
	void *key = chan->device;
	int i;

	for (i = 0; i < IO_DMA_STAT_DEVS; i++) {
		void *k = READ_ONCE(io_dma_dev_stats[i].key);

		if (k == key)
			return &io_dma_dev_stats[i];
		if (!k) {
			if (!cmpxchg(&io_dma_dev_stats[i].key, NULL, key)) {
				/* We claimed the slot and the device is live
				 * here since we are on its submit path.
				 * Racing readers may see a partial name but
				 * never a stale pointer.
				 */
				strscpy(io_dma_dev_stats[i].name,
					dev_name(chan->device->dev),
					sizeof(io_dma_dev_stats[i].name));
				return &io_dma_dev_stats[i];
			}
			if (READ_ONCE(io_dma_dev_stats[i].key) == key)
				return &io_dma_dev_stats[i];
		}
	}
	return NULL;	/* The registry is full, so the op is uncounted. */
}

static struct io_dma_chan_qstat *io_dma_qstat_get(struct io_dma_dev_stat *d,
						  struct dma_chan *chan)
{
	int i;

	for (i = 0; i < IO_DMA_STAT_CHANS; i++) {
		struct dma_chan *c = READ_ONCE(d->chans[i].chan);

		if (c == chan)
			return &d->chans[i];
		if (!c) {
			if (!cmpxchg(&d->chans[i].chan, NULL, chan)) {
				strscpy(d->chans[i].name,
					dma_chan_name(chan),
					sizeof(d->chans[i].name));
				return &d->chans[i];
			}
			if (READ_ONCE(d->chans[i].chan) == chan)
				return &d->chans[i];
		}
	}
	return NULL;
}

static unsigned int io_dma_qstat_bucket(u64 v)
{
	return v ? min_t(unsigned int, ilog2(v) + 1,
			 IO_DMA_QSTAT_NBUCKETS - 1) : 0;
}

/*
 * Per-device in-flight descriptor budget, the queueing bound for the
 * channel-sharing design.  A device's work queues hold a fixed number
 * of descriptor slots and the queueing delay for a new descriptor
 * grows with the descriptors ahead of it.  Therefore bounding admitted
 * descriptors bounds both by construction.  A budget at or below the
 * configured WQ size means submissions never find the queue full.
 *
 * There are two classes.  The write sink is bulk and waits inline on
 * its cookies, so it may hold at most budget*wr_pct/100 descriptors
 * and refused chunks fall to the CPU-redo pass.  File reads are
 * admitted while the total in flight is within the budget, so the
 * remaining share is their guaranteed headroom.
 *
 * budget_descs 0, the default, disables all checks.  The counters live
 * in the same per-device slot the queueing histograms use, so the
 * submit and complete hooks resolve one registry entry and serve both,
 * and per-device lines appear in chan_qstat.
 */
static u32 io_dma_budget_descs;			/* 0 = off */
static u32 io_dma_budget_wr_pct = 75;


/*
 * Write-sink chunk admission.  A refusal sends the chunk to the
 * CPU-redo pass.  This is checked per chunk so that a draining queue
 * readmits mid-write.
 */
static bool io_dma_budget_refuse_wr(struct dma_chan *chan)
{
	u64 budget = READ_ONCE(io_dma_budget_descs);
	struct io_dma_dev_stat *d;

	if (!budget)
		return false;
	d = io_dma_dev_stat_get(chan);
	if (!d)
		return false;
	if (atomic64_read(&d->inflight) >= budget ||
	    atomic64_read(&d->inflight_wr) >=
			div_u64(budget * min(READ_ONCE(io_dma_budget_wr_pct), 100U), 100)) {
		atomic64_inc(&d->rej_wr);
		return true;
	}
	return false;
}

/*
 * Whole-read admission at io_dma_filemap_read() entry.  A refusal
 * falls back to the buffered read and is counted under the eagain
 * reason.
 */
static bool io_dma_budget_refuse_rd(struct dma_chan *chan)
{
	u64 budget = READ_ONCE(io_dma_budget_descs);
	struct io_dma_dev_stat *d;

	if (!budget)
		return false;
	d = io_dma_dev_stat_get(chan);
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
	struct io_dma_dev_stat *d = io_dma_dev_stat_get(chan);
	struct io_dma_chan_qstat *q;
	u64 now, hwm;

	if (!d)
		return;
	atomic64_inc(&d->inflight);
	if (wr)
		atomic64_inc(&d->inflight_wr);
	q = io_dma_qstat_get(d, chan);
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
	struct io_dma_dev_stat *d = io_dma_dev_stat_get(chan);
	struct io_dma_chan_qstat *q;

	if (!d)
		return;
	atomic64_dec(&d->inflight);
	if (wr)
		atomic64_dec(&d->inflight_wr);
	q = io_dma_qstat_get(d, chan);
	if (!q)
		return;
	atomic64_sub(len, &q->inflight);
	if (!submit_ns)
		return;
	atomic64_inc(&q->lat_us[io_dma_qstat_bucket(
			div_u64(ktime_get_ns() - submit_ns, NSEC_PER_USEC))]);
}

/*
 * Retire the qstat and devload accounting for a task that ring teardown
 * abandons without a hardware completion. Without this, the machine
 * global in-flight counters drift upward on every hung-hardware drain
 * and the per-device budget eventually refuses all DMA.
 */
void io_dma_qstat_task_abandon(struct io_ring_ctx *ctx, struct io_dma_task *dma)
{
	if (!IS_ERR_OR_NULL(ctx->dma.chan))
		io_dma_qstat_complete(ctx->dma.chan, dma->len, 0, false);
}

static int io_dma_chan_qstat_show(struct seq_file *m, void *v)
{
	int i, j, b;

	for (i = 0; i < IO_DMA_STAT_DEVS; i++) {
		struct io_dma_dev_stat *d = &io_dma_dev_stats[i];

		if (!READ_ONCE(d->key))
			break;
		for (j = 0; j < IO_DMA_STAT_CHANS; j++) {
			struct io_dma_chan_qstat *q = &d->chans[j];
			struct dma_chan *chan = READ_ONCE(q->chan);

			if (!chan)
				break;
			if (!atomic64_read(&q->submits))
				continue;	/* idle or a pre-reconfig leftover */
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
	}
	for (i = 0; i < IO_DMA_STAT_DEVS; i++) {
		struct io_dma_dev_stat *d = &io_dma_dev_stats[i];

		if (!READ_ONCE(d->key))
			break;
		seq_printf(m, "device %s inflight_descs %llu wr_descs %llu rej_rd %llu rej_wr %llu budget_descs %u wr_pct %u\n",
			   d->name,
			   (u64)atomic64_read(&d->inflight),
			   (u64)atomic64_read(&d->inflight_wr),
			   (u64)atomic64_read(&d->rej_rd),
			   (u64)atomic64_read(&d->rej_wr),
			   READ_ONCE(io_dma_budget_descs),
			   READ_ONCE(io_dma_budget_wr_pct));
	}
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(io_dma_chan_qstat);

static void io_dma_qstat_reset(void)
{
	int i, j, b;

	for (i = 0; i < IO_DMA_STAT_DEVS; i++) {
		struct io_dma_dev_stat *d = &io_dma_dev_stats[i];

		if (!READ_ONCE(d->key))
			break;
		for (j = 0; j < IO_DMA_STAT_CHANS; j++) {
			struct io_dma_chan_qstat *q = &d->chans[j];

			if (!READ_ONCE(q->chan))
				break;
			atomic64_set(&q->submits, 0);
			atomic64_set(&q->bytes, 0);
			/* Keep the live in-flight bytes and restart the high-water
			 * mark from them.
			 */
			atomic64_set(&q->hwm, atomic64_read(&q->inflight));
			for (b = 0; b < IO_DMA_QSTAT_NBUCKETS; b++) {
				atomic64_set(&q->lat_us[b], 0);
				atomic64_set(&q->seen_kb[b], 0);
			}
		}
	}
	for (i = 0; i < IO_DMA_STAT_DEVS; i++) {
		if (!READ_ONCE(io_dma_dev_stats[i].key))
			break;
		/* The live inflight and inflight_wr counts stay. */
		atomic64_set(&io_dma_dev_stats[i].rej_rd, 0);
		atomic64_set(&io_dma_dev_stats[i].rej_wr, 0);
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
 * Filemap DMA-write gate and result counters. These are surfaced
 * through the io_uring_dma_latency debugfs stats file and zeroed by
 * its _reset companion.
 */
static const char * const io_dma_fmw_names[IO_DMA_FMW_NR] = {
	"engaged", "no_aops", "not_bvec", "direct", "no_dma_addrs",
	"eagain", "cpu_redo", "error", "short_write",
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
	io_dma_qstat_reset();
	return count;
}

static const struct file_operations io_dma_lat_reset_fops = {
	.owner		= THIS_MODULE,
	.open		= simple_open,
	.write		= io_dma_lat_reset_write,
	.llseek		= noop_llseek,
};

void io_dma_debugfs_init(void)
{
	struct dentry *dir;

	dir = debugfs_create_dir("io_uring_dma", NULL);

	debugfs_create_file("chan_qstat", 0444, dir, NULL,
			    &io_dma_chan_qstat_fops);
	debugfs_create_u32("budget_descs", 0644, dir, &io_dma_budget_descs);
	debugfs_create_u32("budget_wr_pct", 0644, dir, &io_dma_budget_wr_pct);
	debugfs_create_u32("cq_poll_us", 0644, dir, &io_dma_cq_poll_us);
	debugfs_create_u32("fmw_spin_us", 0644, dir, &io_dma_fmw_spin_us);
	debugfs_create_u32("fmw_wait_ms", 0644, dir, &io_dma_fmw_wait_ms);
	debugfs_create_u32("stats", 0644, dir, &io_dma_stats_enabled);
	debugfs_create_file("latency", 0444, dir, NULL, &io_dma_lat_fops);
	debugfs_create_file("latency_reset", 0200, dir, NULL,
			    &io_dma_lat_reset_fops);
	debugfs_create_file("pfn_cache", 0444, dir, NULL,
			    &io_pfn_cache_stats_fops);
	debugfs_create_u32("pfn_cache_cap_mb", 0644, dir,
			   &io_dma_pfn_cache_cap_mb);
	debugfs_create_u32("pfn_cache_max_age_ms", 0644, dir,
			   &io_dma_pfn_cache_max_age_ms);
	debugfs_create_file("pfn_cache_flush", 0200, dir, NULL,
			    &io_pfn_cache_flush_fops);
}

/* Datapath allocation takes from the pool first and then falls back
 * to the non-blocking slab. It never sleeps.
 */
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
		memset(t, 0, sizeof(*t));	/* Match the old kmem_cache_zalloc. */
		return t;
	}

	/*
	 * The pool is exhausted. We use NOWAIT so that we never enter
	 * reclaim on the datapath and NOWARN because failure is expected
	 * and handled by the CPU-copy fallback.
	 */
	return kmem_cache_zalloc(dma_cachep, GFP_NOWAIT | __GFP_NOWARN);
}

/* Datapath free parks the task back into the pool up to the cap and
 * otherwise frees it to the slab.
 */
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
 * This is called from io_allocate_dma_chan() at ring setup in process
 * context with no locks held, so GFP_KERNEL is fine here.
 */
void io_dma_init_freelist(struct io_ring_ctx *ctx, struct io_uring_params *p)
{
	struct io_dma_channel *d = &ctx->dma;
	unsigned int n, i;

	spin_lock_init(&d->free_lock);
	d->free_list = NULL;
	d->free_count = 0;

	/*
	 * Cover roughly the in-flight CQ depth and clamp to a sane range.
	 * Since io_dma_task is small (about 96 bytes), even 8192 entries
	 * is about 768KB in the worst case.
	 */
	n = clamp(p->cq_entries * 2u, 256u, 8192u);
	d->free_max = n;

	for (i = 0; i < n; i++) {
		struct io_dma_task *t = kmem_cache_alloc(dma_cachep, GFP_KERNEL);

		if (!t)
			break;		/* A partial prefill is fine. NOWAIT covers the rest. */
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

	unsigned int passes = 0;

	/* Drain until the list is empty, but stay cancelable. A descriptor
	 * stuck DMA_IN_PROGRESS on halted hardware would otherwise spin
	 * this worker forever and deadlock the cancel_work_sync() in ring
	 * teardown. Bound the inline passes and requeue instead, so the
	 * detection latency is unchanged while cancellation can win
	 * between requeues.
	 */
	do {
		__io_dma_poll(ctx);
		cpu_relax();
		if (++passes >= 1024) {
			if (io_dma_pending(ctx))
				queue_work(system_unbound_wq, &d->poll_work);
			return;
		}
	} while (io_dma_pending(ctx));
}

void io_uring_dma_prep(struct io_kiocb *req)
{
	if (IS_ERR_OR_NULL(req->ctx->dma.chan))
		return;

	req->dma.dma_active = true;
	/*
	 * dma_ref_held is deliberately not reset here. It is set and
	 * cleared only under ctx->dma.lock by the submit and complete ref
	 * protocol, and a reissue can reach this prep while the previous
	 * cycle's completer is still about to drop the previous in-flight
	 * ref. Clearing the flag here would erase that pending drop and
	 * leak one req reference per race. A fresh req needs no
	 * initialization because the submit path sets the flag.
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

	/* All fallible allocations come before the prep. A prepped but
	 * never submitted descriptor cannot be returned to the driver
	 * pool, so abandoning one here would orphan an idxd descriptor.
	 */
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

	/* Allocate the src and dst scatterlists together. We initialize
	 * the SG tables so that sg_next() and sg_is_last() work correctly
	 * and then populate the DMA addresses from the entries array.
	 */
	sgls = kmalloc_array(nr_entries * 2, sizeof(*sgls),
			     GFP_NOWAIT | __GFP_NOWARN);
	if (!sgls) {
		io_dma_task_free(req->ctx, dma);
		kfree(heap_entries);
		return -ENOMEM;
	}
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
		io_dma_task_free(req->ctx, dma);
		kfree(heap_entries);
		return -EAGAIN;
	}

	/* The SG arrays are consumed by dmaengine_prep_dma_memcpy_sg().
	 * The driver copies what it needs into batch descriptors.
	 */
	kfree(sgls);

	dma->req = req;
	dma->next = NULL;
	dma->len = total_len;
	dma->is_batch = true;
	dma->batch_nr = nr_entries;
	dma->batch_entries = heap_entries;

	/* Stamp before the doorbell so the sample covers the whole
	 * hardware transfer. A zero stamp means sampling is off.
	 */
	dma->submit_ns = READ_ONCE(io_dma_stats_enabled) ? ktime_get_ns() : 0;
	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		kfree(heap_entries);
		io_dma_task_free(req->ctx, dma);
		return -EAGAIN;	/* The WQ may be full. Fall back to CPU copy. */
	}
	io_dma_qstat_submit(req->ctx->dma.chan, dma->len, false);

	/* Take folio references for the duration of the DMA. */
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

	/*
	 * We prep after the task allocation because an abandoned prep
	 * would orphan an idxd descriptor from the channel pool.
	 */
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

	/* Stamp before the doorbell so the sample covers the whole
	 * hardware transfer. A zero stamp means sampling is off.
	 */
	dma->submit_ns = READ_ONCE(io_dma_stats_enabled) ? ktime_get_ns() : 0;
	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		folio_put(entry->folio);
		io_dma_task_free(req->ctx, dma);
		return -EAGAIN;	/* The WQ may be full. Fall back to CPU copy. */
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

	for (i = 0; i < nr; i++) {
		struct io_dma_batch_entry *e = &entries[i];

		if (e->pfn_map) {
			/*
			 * This is a cached mapping, so we drop the in-flight
			 * reference. The unmap belongs to eviction or flush
			 * and not to this I/O.
			 */
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
				/* A failed submit never consumes the entry's
				 * source mapping, so release this entry and
				 * every remaining one.
				 */
				io_dma_unmap_batch_entries(req, dev,
							  entries + i,
							  nr_entries - i);
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

/* Reads at or below this size fall back to the CPU copy path */
#define IO_DMA_MIN_READ_BYTES	SZ_16K

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
		io_pfn_cache_usable() ? io_pfn_cache_get(dev) : NULL;
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

	/* At or below this size the descriptor setup and completion
	 * detection cost more than the copy itself, so we take the CPU
	 * path.
	 */
	if (want <= IO_DMA_MIN_READ_BYTES)
		return -EAGAIN;

	if (unlikely(iocb->ki_pos < 0))
		return -EINVAL;
	if (unlikely(iocb->ki_pos >= inode->i_sb->s_maxbytes))
		return 0;
	/* Device budget admission. Reads are refused only when even their
	 * full-budget headroom is gone, since writers are capped at
	 * wr_pct below it. A refusal falls back to the buffered read.
	 */
	if (io_dma_budget_refuse_rd(chan))
		return -EAGAIN;

	/*
	 * We never block on the read datapath. On failure the caller
	 * (io_read) continues to the normal buffered-read path.
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
		 * count is the number of bytes remaining in the request.
		 * The registered buffer (imu) is usually larger than the
		 * read. Clamping to imu->len instead of the requested count
		 * made a short READ_FIXED overrun its length and fill the
		 * whole buffer, which then tripped -EFAULT at the
		 * buffer-end destination lookup.
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
				size_t dst_seg_remain;
				size_t chunk;

				dst_dma = io_reg_buf_dma_addr(imu,
						dst_user_addr + dst_offset,
						&dst_seg_remain);
				if (!dst_dma) {
					error = -EFAULT;
					goto flush_and_put;
				}

				chunk = min_t(size_t, bytes - copied,
					      dst_seg_remain);
				/* Split at source map-quantum boundaries so
				 * that one cache segment covers each entry
				 * and no transient map exceeds the rcache
				 * classes.
				 */
				chunk = min_t(size_t, chunk, map_quantum -
					((offset + copied) & (map_quantum - 1)));

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

				/* Flush the batch if it is full. */
				if (nr_entries == IO_DMA_BATCH_MAX) {
					ssize_t ret;

					ret = io_dma_flush_batch(req, dev, chan,
						entries, nr_entries);
					nr_entries = 0;
					if (ret < 0) {
						error = ret;
						goto put_folios;
					}
					/*
					 * Count only what the flush actually
					 * submitted and stop on a short
					 * flush. The claim must stay a
					 * contiguous prefix. Collecting past
					 * a gap would claim bytes that were
					 * never copied.
					 */
					submitted += ret;
					if (ret < (ssize_t)batch_bytes) {
						batch_bytes = 0;
						goto put_folios;
					}
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
		/* Flush any remaining entries from this folio batch. */
		if (nr_entries > 0) {
			ssize_t ret;

			ret = io_dma_flush_batch(req, dev, chan,
				entries, nr_entries);
			nr_entries = 0;
			if (ret < 0) {
				if (!error)
					error = ret;
			} else {
				/* Partial flushes count only submitted bytes. */
				submitted += ret;
				if (ret < (ssize_t)batch_bytes)
					error = error ? error : -EAGAIN;
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
	 * may be claimed.  total_read and ki_pos run ahead of the flushes
	 * during collection.  A failed flush leaves those trailing bytes
	 * uncopied, so we clamp the result and the file position to what
	 * was actually submitted.  Claiming unsubmitted bytes returns
	 * uninitialized destination memory to userspace.
	 */
	if (unlikely(submitted != total_read))
		iocb->ki_pos = start_pos + submitted;
	return submitted ? submitted : error;
}

/*
 * DMA-offloaded buffered write for registered buffers (WRITE_FIXED).
 * This replaces generic_perform_write()'s copy_folio_from_iter() with
 * DSA copies from the pre-mapped registered buffer into the page-cache
 * folios obtained from aops->write_begin().
 *
 * The write path is deliberately synchronous. The caller path runs in
 * io-wq process context because buffered regular-file writes always
 * punt there, and inode_lock plus the per-folio write_begin and
 * write_end protocol make a deferred-CQE design a lifetime minefield.
 * Waiting inline keeps every VFS invariant identical to the generic
 * path. The win is the removed CPU memcpy and not latency.
 *
 * The failure ladder is as follows. On a prep or submit failure or a
 * DMA_ERROR completion, with every submitted cookie reaped, we CPU
 * re-copy every chunk and commit normally (counted as "cpu_redo").
 * This is safe because copying the same src to the same dst is
 * idempotent. On a completion timeout the device is wedged. We unmap
 * the dst ranges so that a late DMA write faults in the IOMMU instead
 * of hitting reclaimed memory, skip write_end, and deliberately leak
 * the affected folios locked and referenced. A wedged range beats
 * silent corruption. This case returns -EIO.
 */

struct io_dma_fmw_folio {
	struct folio *folio;
	void *fsdata;
	loff_t pos;
	unsigned int len;
	dma_addr_t dst_dma;	/* 0 means nothing to unmap */
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
 * flight, in which case the caller must treat every dst as poisoned.
 *
 * We spin only briefly.  DSA transfers normally complete in
 * single-digit microseconds, but under contention with many rings
 * sharing the WQ descriptor pools completion can take milliseconds.
 * Hundreds of io-wq workers busy-polling here saturated whole sockets
 * and starved application heartbeats at O(100) shared rings.
 * Therefore we back off to sleeping once the fast path misses.
 */
static int io_dma_fmw_wait(struct dma_chan *chan, struct io_dma_fmw_ck *cookies,
			   unsigned int *nr, bool *redo)
{
	unsigned int wait_ms = READ_ONCE(io_dma_fmw_wait_ms);
	unsigned long deadline = jiffies + msecs_to_jiffies(wait_ms);
	u64 spin_end = ktime_get_ns() +
		READ_ONCE(io_dma_fmw_spin_us) * NSEC_PER_USEC;
	unsigned int i, j, spins;

	for (i = 0; i < *nr; i++) {
		enum dma_status st;

		spins = 0;
		while ((st = dmaengine_async_is_tx_complete(chan, cookies[i].ck))
		       == DMA_IN_PROGRESS) {
			if (time_after(jiffies, deadline)) {
				/* We are wedged, so we retire the qstat
				 * in-flight bytes with no latency samples.
				 * The descriptors are leaked and fenced by
				 * the caller.
				 */
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
		/* Latency is poll-observation time. Cookies are polled in
		 * order, so a later cookie's sample can be inflated by up
		 * to the earlier polls. This is fine at histogram
		 * resolution.
		 */
		io_dma_qstat_complete(chan, cookies[i].len,
				      st == DMA_COMPLETE ?
						cookies[i].submit_ns : 0, true);
	}
	*nr = 0;
	return 0;
}

/* Writes at or below this size fall back to the CPU copy path */
#define IO_DMA_MIN_WRITE_BYTES	SZ_64K

/*
 * Folios collected, submitted and committed as one group. Capping the
 * group bounds the folio array, the number of folios held locked at
 * once, and the range a wedged device can strand. None of those would
 * otherwise be bounded by anything smaller than the request itself.
 */
#define IO_DMA_FMW_GROUP_FOLIOS	256

/*
 * Run one group of a DMA write: collect folios starting @base bytes
 * into the write until the group is full or the write ends, submit the
 * copies, wait for them, and commit. Returns the bytes committed, or a
 * negative error when nothing was committed. On a device timeout the
 * group's folios are deliberately leaked and -EIO is returned, so the
 * caller must stop.
 */
static ssize_t io_dma_fmw_group(struct io_kiocb *req, struct kiocb *iocb,
				struct iov_iter *from, u64 src_user_addr,
				size_t base, size_t room,
				struct io_dma_fmw_folio *fol,
				unsigned int max_fol,
				struct io_dma_fmw_ck *cookies,
				unsigned int max_cookies)
{
	struct io_ring_ctx *ctx = req->ctx;
	struct file *file = iocb->ki_filp;
	struct address_space *mapping = file->f_mapping;
	const struct address_space_operations *aops = mapping->a_ops;
	struct io_mapped_ubuf *imu = req->buf_node->buf;
	struct dma_chan *chan = ctx->dma.chan;
	struct device *dev = chan->device->dev;
	/* Clamp the write chunks, and so the transient dst-folio maps, to
	 * the IOVA-rcache-served quantum.
	 */
	size_t max_chunk = min_t(size_t, mapping_max_folio_size(mapping),
				 io_dma_map_quantum(dev));
	unsigned int nr_fol = 0, nr_cookies = 0, i;
	unsigned int prep_fails = 0;
	bool redo = false;
	bool surrendered = false;
	ssize_t collected = 0, committed = 0;
	ssize_t err = 0;
	int wedged = 0;

	while (collected < room && nr_fol < max_fol) {
		loff_t pos = iocb->ki_pos + base + collected;
		size_t bytes = min_t(size_t,
				     max_chunk - (pos & (max_chunk - 1)),
				     room - collected);
		size_t offset, sub;
		dma_addr_t dst_dma;
		struct folio *folio;
		void *fsdata;
		int status;

		status = aops->write_begin(iocb, mapping, pos, bytes,
					   &folio, &fsdata);
		if (unlikely(status < 0)) {
			/* Unlike generic_perform_write(), which blocks in
			 * reclaim and retries, we fail into a short write.
			 * fmw_short counts these.
			 */
			if (!collected)
				err = status;
			break;
		}
		/*
		 * A write_begin that leaves a journal handle open (ext4
		 * without delalloc, or delalloc falling back under low
		 * free space) expects its matching write_end before the
		 * next write_begin. Batching would nest handles with
		 * h_ref only and no credits, and a wedge would leak the
		 * references and stall the journal. Hand such
		 * filesystems back to the CPU path.
		 */
		if (unlikely(!nr_fol && current->journal_info)) {
			aops->write_end(iocb, mapping, pos, bytes, 0,
					folio, fsdata);
			return -EAGAIN;
		}
		offset = offset_in_folio(folio, pos);
		/*
		 * Cover the locked folio to its end or to the write's end.
		 * Ending a chunk mid-folio would make the next iteration's
		 * write_begin() wait forever on the folio lock this batch
		 * already holds.  Large folios, for example on a rewrite of
		 * ranges cached by earlier big writes, exceed the
		 * request-size hint.  Therefore the returned folio and not
		 * the hint decides the chunk.  Descriptors below still
		 * split at source-folio boundaries.
		 */
		bytes = min_t(size_t, room - collected,
			      folio_size(folio) - offset);

		if (surrendered) {
			/* This is sustained descriptor exhaustion. We stop
			 * touching the DMA engine for the rest of this write
			 * and let the CPU-redo pass commit everything. This
			 * beats paying a drain-wait per chunk while dozens
			 * of other rings hold the pools empty.
			 */
			dst_dma = 0;
			redo = true;
			goto record;
		}
		if (io_dma_budget_refuse_wr(chan)) {
			/*
			 * We are over the device in-flight budget, so this
			 * chunk goes to the CPU-redo pass.  The budget is
			 * re-checked per chunk so that a draining queue
			 * readmits mid-write.
			 */
			dst_dma = 0;
			redo = true;
			goto record;
		}
		dst_dma = dma_map_page(dev, folio_page(folio, 0),
				       offset, bytes, DMA_FROM_DEVICE);
		if (dma_mapping_error(dev, dst_dma)) {
			/* Commit this chunk via the CPU-redo pass instead. */
			dst_dma = 0;
			redo = true;
			goto record;
		}

		/* Split at the source registered-buffer folio boundaries. */
		for (sub = 0; sub < bytes; ) {
			u64 uaddr = src_user_addr + base + collected + sub;
			size_t src_seg_remain, len;
			struct dma_async_tx_descriptor *tx;
			dma_addr_t src_dma, dst;
			dma_cookie_t ck;

			src_dma = io_reg_buf_dma_addr(imu, uaddr,
						      &src_seg_remain);
			dst = dst_dma + sub;
			if (unlikely(!src_dma)) {
				redo = true;	/* The CPU-redo pass covers the chunk. */
				break;
			}
			len = min3(bytes - sub, src_seg_remain, max_chunk);

			tx = dmaengine_prep_dma_memcpy(chan, dst, src_dma, len,
						       io_dma_prep_flags());
			if (!tx) {
				/* The pool is exhausted. Drain in-flight
				 * work and retry once.
				 */
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
				 * rings, so we surrender the remainder to
				 * the CPU.
				 */
				if (++prep_fails >= 2) {
					surrendered = true;
					break;	/* redo covers this chunk. */
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
		collected += bytes;
	}

collect_done:
	if (!wedged) {
		dma_async_issue_pending(chan);
		wedged = io_dma_fmw_wait(chan, cookies, &nr_cookies, &redo);
	}

	/* Unmap the dst IOVAs. After a timeout this also fences late DMA
	 * writes: under a strict IOMMU they fault, and under a flush-queue
	 * domain they land in the folios leaked below, which stay locked
	 * and referenced. Either way they never reach reclaimed memory.
	 */
	for (i = 0; i < nr_fol; i++)
		if (fol[i].dst_dma)
			dma_unmap_page(dev, fol[i].dst_dma, fol[i].map_len,
				       DMA_FROM_DEVICE);

	if (unlikely(wedged)) {
		/* The folio contents are unknown and a stray write may
		 * still land, so we leak the locked folios rather than
		 * expose them.
		 */
		pr_warn_ratelimited("io_uring DMA write: wedged after %dms, leaking %u folios (%s)\n",
				    READ_ONCE(io_dma_fmw_wait_ms), nr_fol,
				    dma_chan_name(chan));
		io_dma_fmw_record(IO_DMA_FMW_ERROR);
		return -EIO;
	}

	if (unlikely(redo)) {
		/* Re-copy every chunk with the CPU. The same source bytes
		 * go to the same folio ranges, so overlap with completed
		 * DMA is idempotent. The iter was never advanced during
		 * collection.
		 */
		for (i = 0; i < nr_fol; i++) {
			size_t n = copy_folio_from_iter(fol[i].folio,
					offset_in_folio(fol[i].folio, fol[i].pos),
					fol[i].len, from);
			if (unlikely(n != fol[i].len)) {
				collected = fol[i].pos -
					(iocb->ki_pos + base) + n;
				break;
			}
		}
		io_dma_fmw_record(IO_DMA_FMW_CPU_REDO);
	}

	/* Commit in ascending order with the dirty, unlock, and i_size
	 * updates. Every collected folio must pass through write_end even
	 * after a failure, with a zero claim, or the tail folios stay
	 * locked and referenced forever.
	 */
	{
		bool commit_failed = false;

		for (i = 0; i < nr_fol; i++) {
			size_t claim = commit_failed ? 0 :
				min_t(size_t, fol[i].len,
				      collected - committed);
			int done;

			done = aops->write_end(iocb, mapping, fol[i].pos,
					       fol[i].len, claim,
					       fol[i].folio, fol[i].fsdata);
			if (commit_failed)
				continue;
			if (unlikely(done < 0)) {
				if (!committed)
					err = done;
				commit_failed = true;
				continue;
			}
			committed += done;
			if ((size_t)done < fol[i].len) {
				commit_failed = true;
				continue;
			}
			balance_dirty_pages_ratelimited(mapping);
		}
	}

	/* The CPU re-copy consumes the iter when it runs; otherwise
	 * advance it over exactly what was committed.
	 */
	if (!redo)
		iov_iter_advance(from, committed);
	return committed ? committed : err;
}

ssize_t io_dma_filemap_write(struct io_kiocb *req, struct kiocb *iocb,
			     struct iov_iter *from, u64 src_user_addr)
{
	struct io_mapped_ubuf *imu = req->buf_node->buf;
	struct file *file = iocb->ki_filp;
	struct inode *inode = file->f_mapping->host;
	struct io_dma_fmw_folio *fol = NULL;
	struct io_dma_fmw_ck *cookies = NULL;
	unsigned int max_fol, max_cookies;
	ssize_t want, written = 0, err = 0;

	/* At or below this size the per-folio write_begin and write_end
	 * machinery plus the descriptor overhead exceeds the memcpy it
	 * replaces, so we take the CPU path.
	 */
	if (iov_iter_count(from) <= IO_DMA_MIN_WRITE_BYTES)
		return -EAGAIN;

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

	max_fol = min_t(unsigned long, IO_DMA_FMW_GROUP_FOLIOS,
			DIV_ROUND_UP(want, PAGE_SIZE) + 1);
	/* One entry per src-folio crossing per chunk. The collection loop
	 * flushes whenever the array fills, so this sets the wait
	 * granularity rather than bounding the transfer.
	 */
	max_cookies = max_fol + DIV_ROUND_UP(want, 1UL << imu->folio_shift) + 8;
	max_cookies = min_t(unsigned int, max_cookies, 4 * max_fol + 8);
	fol = kvmalloc_array(max_fol, sizeof(*fol), GFP_KERNEL);
	cookies = kvmalloc_array(max_cookies, sizeof(*cookies), GFP_KERNEL);
	if (!fol || !cookies) {
		err = -EAGAIN;	/* Fall back to the normal write path. */
		goto out_unlock;
	}

	/* One group at a time, so a large write neither allocates an array
	 * proportional to its length nor holds every one of its folios
	 * locked at once. Groups commit as they go, so a failure part way
	 * through is a short write over the groups that already landed.
	 */
	while (written < want) {
		ssize_t done = io_dma_fmw_group(req, iocb, from, src_user_addr,
						written, want - written, fol,
						max_fol, cookies, max_cookies);

		if (done <= 0) {
			if (!written)
				err = done;
			break;
		}
		written += done;
	}

	if (written > 0)
		iocb->ki_pos += written;
out_unlock:
	inode_unlock(inode);
	kvfree(fol);
	kvfree(cookies);
	if (written > 0) {
		if (written < want)
			io_dma_fmw_record(IO_DMA_FMW_SHORT);
		return generic_write_sync(iocb, written) ?: written;
	}
	return err;
}

/*
 * Release a completed or aborted task's source resources.  These are
 * the DMA unmaps, which involve IOVA frees and IOTLB work under an
 * IOMMU, and the folio references.  This is the expensive half of
 * completion and needs no ctx->dma.lock.  Callers run it before taking
 * the lock for __io_dma_task_complete() so that the lock hold shrinks
 * to the refcount handshake.
 */
void io_dma_task_release_res(struct io_ring_ctx *ctx, struct device *dev,
			     struct io_dma_task *dma)
{
	if (dma->is_batch) {
		int i;

		for (i = 0; i < dma->batch_nr; i++) {
			struct io_dma_batch_entry *e = &dma->batch_entries[i];

			if (e->pfn_map) {
				/*
				 * This is a cached mapping, so we drop the
				 * in-flight reference. The unmap belongs to
				 * eviction or flush and not to this I/O.
				 */
				io_pfn_map_put(e->pfn_map);
			} else {
				dma_unmap_page(dev, e->src_dma, e->src_len,
					       DMA_TO_DEVICE);
			}
			folio_put(e->folio);
		}
		kfree(dma->batch_entries);
	} else {
		if (dma->src_pfn_map) {
			io_pfn_map_put(dma->src_pfn_map);
		} else if (dma->src_map_len) {
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

	/* Free the task before touching the refcnt. task_len was saved above. */
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
			 * Terminal completion goes through task_work.
			 * __io_dma_poll() may run from a workqueue without
			 * ctx->uring_lock, so io_req_task_complete() defers
			 * the CQE post until the task_work loop acquires it.
			 */
			io_req_set_res(req, req->dma.saved_res,
				       req->dma.saved_cflags);
			req->io_task_work.func = io_req_task_complete;
			io_req_task_work_add(req);
		}

		/*
		 * Drop the in-flight DMA reference taken in
		 * io_dma_submit_queued_tasks(). The completion handling
		 * above only queues task_work and does not free the req
		 * inline, so the req is still valid here. If we held the
		 * last reference, meaning the req was already terminally
		 * completed or cancelled by teardown, we free it now.
		 * Otherwise the owner frees it once it drops its reference.
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
			 * Take the in-flight reference before the tasks are
			 * published. The doorbell is already rung, so once
			 * the tasks appear on the submit_list a completer
			 * can reach dma_refcnt == 0, see dma_ref_held ==
			 * false, and complete the req. Stamping the ref
			 * afterwards would double-complete an already
			 * completed req. The reference is dropped in
			 * __io_dma_task_complete() at dma_refcnt == 0,
			 * which mirrors io_wq_submit_work().
			 *
			 * The take must also be under ctx->dma.lock. On
			 * reissue the previous cycle's completer may still
			 * sit between completing the request and dropping
			 * the old ref. An unlocked take interleaves into
			 * that window and leaks one reference each time.
			 * The lock orders this take after that drop.
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
			 * submit_list after the ref-take above so that no
			 * drain can reach dma_refcnt == 0 before the
			 * in-flight ref exists.  Every task was fully
			 * submitted with a valid cookie because tasks are
			 * linked only after dmaengine_submit() succeeds, so
			 * the poller can consume them as-is.  Publishing in
			 * chain order makes the llist a LIFO of a FIFO and
			 * the consumer reverses it back.  Submissions are
			 * serialized by uring_lock, so no other producer
			 * interleaves.
			 */
			struct io_dma_task *t = req->dma.dma_tasks;

			while (t) {
				/* We read ->next before publishing because
				 * a published task can complete and be
				 * freed immediately.
				 */
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
	 * Use ctx and not req below. __io_dma_poll() may complete the
	 * request and free it, so req must not be dereferenced after
	 * this point.
	 */
	if (ret == -EIOCBQUEUED) {
		/*
		 * Tasks were just queued and the issuer is handing the req
		 * off. Completing them inline could complete this same req
		 * while the issuer is still unwinding, so we defer to the
		 * poller.
		 */
		if (io_dma_pending(ctx)) {
			/*
			 * We use the unbound workqueue rather than
			 * schedule_work(). A per-CPU kworker must win a
			 * wakeup-preemption fight with the submitting
			 * thread to run, and this shows up as completions
			 * sitting undetected for the application's whole
			 * compute slice. An unbound worker lands on an
			 * idle CPU and detects completions in parallel.
			 */
			queue_work(system_unbound_wq,
				   &ctx->dma.poll_work);
		}
	} else if (atomic_read(&ctx->dma.poll_armed) == 0) {
		/*
		 * Nothing was queued for this req, for example on a CPU
		 * fallback. Draining other reqs' tasks inline is ordinary
		 * async completion and not re-entrancy on the req being
		 * issued.
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
	 * There is no doorbell here. Pollable descriptors reach the
	 * hardware at dmaengine_submit() time (see idxd_dma_tx_submit)
	 * and every submitter rings for its own tasks from the issue
	 * path under uring_lock. Nothing on the pending lists needs a
	 * kick.
	 */
	dev = ctx->dma.chan->device->dev;

	/*
	 * Splice newly submitted tasks onto the consumer-owned poll_list.
	 * Only the armed poller touches poll_list, so no lock is needed.
	 * The llist arrives newest-first and reversing it restores
	 * submission order.  Completion is not in-order because DSA WQs
	 * are fed by multiple engines, so the walk below scans the whole
	 * list and not just the head.
	 */
	{
		struct llist_node *node = llist_del_all(&ctx->dma.submit_list);
		struct io_dma_task *fifo_head = NULL, *fifo_tail = NULL;

		while (node) {
			struct io_dma_task *t =
				llist_entry(node, struct io_dma_task, llnode);

			node = node->next;
			t->next = fifo_head;	/* Prepending reverses LIFO to FIFO. */
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

			/* There is no lock around the hardware poll. The
			 * cookie state is the dmaengine's own and is safe to
			 * read locklessly. Holding ctx->dma.lock across the
			 * whole walk made submitters fight the poller for
			 * it.
			 */
			ret = dmaengine_async_is_tx_complete(ctx->dma.chan,
							     dma->cookie);
			if (ret == DMA_IN_PROGRESS) {
				/*
				 * Keep walking. DSA groups feed each WQ from
				 * multiple engines, so completion is not
				 * in-order and stopping at the first
				 * in-flight entry would block every
				 * completed task behind it.  Each extra
				 * check is one completion-record read.
				 */
				prev = dma;
				dma = next;
				continue;
			}

			/* Unlink before completing since complete may free dma. */
			if (prev)
				prev->next = next;
			else
				WRITE_ONCE(ctx->dma.poll_list, next);
			if (!next)
				ctx->dma.poll_list_tail = prev;

			/* The heavy resource release of IOMMU unmaps and
			 * folio puts runs unlocked. ctx->dma.lock then
			 * covers only the refcount handshake with the
			 * submitter's ref-take.
			 */
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
	/* Release ordering hands poll_list, which was written lock-free
	 * above, to whichever thread arms the poller next.
	 */
	atomic_set_release(&ctx->dma.poll_armed, 0);
	return io_dma_pending(ctx) ? 1 : 0;
}

/*
 * Called from io_cqring_wait_schedule() right before the task would
 * block. If this ring has DMA tasks in flight, spin on their
 * completion records for at most io_dma_cq_poll_us. Completions found
 * here run __io_dma_task_complete() in this task's context. The CQE
 * itself is posted by the poll task_work that this queues, so we
 * report progress via the pending-work checks and let the wait loop
 * run it.
 *
 * Returns true if the caller should skip sleeping and re-run its wait
 * loop because task_work or CQEs are ready. Returns false to fall back
 * to schedule(). This is a no-op for rings with no DMA channel.
 */
/*
 * Teardown diagnostic, called from io_ring_exit_work() when the ring's
 * references fail to drop within the timeout. The dump distinguishes
 * the wedge classes before teardown proceeds anyway. Tasks on the
 * pending lists mean a descriptor is stuck DMA_IN_PROGRESS. Every
 * listed task is polled each pass, so any entry may be the wedged one.
 * An empty list with nr_req_allocated > 0 means a ctx reference is
 * held by something other than DMA.
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
	 * Comparing refs_taken with refs_dropped says whether every
	 * in-flight DMA req ref was released. An imbalance locates the
	 * wedge in the ref-drop path of __io_dma_task_complete().
	 */
	pr_err("io_uring DMA teardown stuck: refs_taken=%d refs_dropped=%d\n",
	       atomic_read(&ctx->dma.diag_refs_taken),
	       atomic_read(&ctx->dma.diag_refs_dropped));

	if (IS_ERR_OR_NULL(ctx->dma.chan))
		return;

	/*
	 * poll_list belongs to the poller, which is quiesced by now, and
	 * submit_list is lock-free. This walk is racy by design since it
	 * feeds only a diagnostic dump on an already-wedged teardown.
	 */
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
