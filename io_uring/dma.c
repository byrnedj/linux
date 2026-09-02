// SPDX-License-Identifier: GPL-2.0

#include <linux/io_uring_types.h>
#include <linux/io_uring.h>
#include <linux/delay.h>
#include <linux/uio.h>
#include <linux/dma-mapping.h>
#include <linux/spinlock.h>
#include <linux/xarray.h>
#include <linux/hash.h>
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
 * io_dma_map_quantum() sized segment DMA_BIDIRECTIONAL and caches the
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
	dma_addr_t		dma_base;	/* segment mapping, bidirectional
						 * so reads (src) and writes
						 * (dst) share entries
						 */
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

	/*
	 * Ghost list: the keys, and only the keys, of segments the cap
	 * recently forced out, each stored as an xarray value holding
	 * its eviction time. A miss that hits the ghost is an eviction
	 * the workload paid for with a remap, which is the one signal
	 * that separates "the cap is trimming dead streaming entries"
	 * from "the cap is thrashing a live working set". Ghost hits
	 * grow the adaptive target below; their absence decays it. Age
	 * evictions never enter the ghost, idleness is not undersizing.
	 */
	struct xarray		ghost;
	unsigned long		ghost_hand;	/* purge walk cursor */
	atomic64_t		ghost_count;
	atomic64_t		ghost_hits;
	u64			eff_cap;	/* adaptive target, bytes;
						 * 0 uninitialized, 1 parked
						 */
	u64			ghost_hits_snap;/* at the last decay tick */
	u64			hits_snap;	/* ditto, for the utility check */
	u64			misses_snap;
	u32			prev_ratio;	/* hit % at the last tick */
	u32			parked_ticks;	/* ticks spent parked */
	unsigned long		next_adapt;	/* jiffies */
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

/*
 * Adaptive sizing.  When set, cap_mb is a ceiling rather than the
 * standing target: the cache aims for an effective cap that ghost
 * hits grow toward the ceiling and quiet intervals decay toward a
 * floor of an eighth of the ceiling.  A streaming phase then keeps
 * the standing device-readable window small, and a re-read phase
 * whose working set the floor thrashes earns its way back up at one
 * step per paid-for eviction.  0 pins the target at cap_mb, which is
 * the historical behaviour.
 */
static u32 io_dma_pfn_cache_auto __read_mostly = 1;

/*
 * Machine-wide standing-mapping budget for adaptive mode, in MiB.
 * Striping gives every device its own cache and the devices duplicate
 * coverage of a shared working set, so per-device targets multiply
 * into total standing bytes. Past roughly 20GB total on this class of
 * machine the IOVA rcaches deplete and every transient mapping falls
 * to the domain rbtree: throughput collapses by an order of magnitude
 * while the caches report healthy hit rates. The adaptive ceiling is
 * this budget split across the registered caches, so the sum stays
 * under the cliff no matter how generous cap_mb is.
 */
static u32 io_dma_pfn_cache_auto_budget_mb __read_mostly = 16384;

/* Registered per-device caches; slots are never released. */
static atomic_t io_pfn_cache_nr;

/* Ghost growth per hit and decay per quiet 2s tick (eff >> shift). */
#define IO_PFN_GHOST_GROW_SEGS	8
#define IO_PFN_ADAPT_DECAY_SHIFT 3
#define IO_PFN_ADAPT_TICK	(2 * HZ)

/*
 * Per-cache adaptive ceiling: the machine budget's fair share among
 * the caches currently holding mappings. Registration is forever, but
 * a device that a past workload touched and this one does not must
 * not dilute the budget: with rings acquiring channels from a shared
 * pool, the registry accretes devices over time, and dividing by the
 * all-time count starved every active cache. An empty cache counts
 * itself the moment it inserts, and the per-tick recount converges as
 * traffic shifts.
 */
static u64 io_pfn_cache_ceiling(u64 hard)
{
	u64 budget = (u64)READ_ONCE(io_dma_pfn_cache_auto_budget_mb) << 20;
	unsigned int i, n = 0;

	for (i = 0; i < IO_PFN_CACHE_DEVS; i++) {
		struct io_pfn_cache *pc;

		if (!READ_ONCE(io_pfn_cache_devs[i]))
			break;
		pc = io_pfn_caches[i];
		if (pc && atomic64_read(&pc->covered) > 0)
			n++;
	}
	if (!n)
		n = 1;
	return min(hard, budget) / n;
}

static u64 io_pfn_cache_floor(u64 ceil)
{
	return min(ceil, clamp(ceil >> 3, (u64)SZ_64M, (u64)SZ_2G));
}

/*
 * The eviction target: the adaptive effective cap, clamped to the
 * per-cache ceiling, or the hard cap itself when auto sizing is off.
 */
#define IO_PFN_EFF_PARKED	1
#define IO_PFN_PARK_PROBE_TICKS	8

static u64 io_pfn_cache_target(struct io_pfn_cache *c, u64 hard)
{
	u64 eff, ceil;

	if (!READ_ONCE(io_dma_pfn_cache_auto))
		return hard;
	ceil = io_pfn_cache_ceiling(hard);
	eff = READ_ONCE(c->eff_cap);
	if (eff == IO_PFN_EFF_PARKED)
		return 0;
	if (!eff)
		eff = io_pfn_cache_floor(ceil);
	return min(eff, ceil);
}

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
	xa_init(&c->ghost);
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
			atomic_inc(&io_pfn_cache_nr);
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
	dma_unmap_page(pm->dev, pm->dma_base, pm->size, DMA_BIDIRECTIONAL);
	kfree_rcu(pm, rcu);
}

/*
 * Displace an entry from the cache so its mapping dies with its last
 * reference. The wedge path uses this on cached write destinations,
 * where the standing mapping is device-writable and must not outlive
 * the failed write; the unmap happens on the final put, immediately
 * unless a concurrent I/O still holds the segment.
 */
static void io_pfn_map_displace(struct io_pfn_cache *c, struct io_pfn_map *pm)
{
	if (!c || !pm)
		return;
	if (xa_cmpxchg(&c->xa, pm->pfn, pm, NULL,
		       GFP_NOWAIT | __GFP_NOWARN) == pm) {
		atomic64_sub(pm->size, &c->covered);
		io_pfn_map_put(pm);	/* the cache bias */
	}
}

/*
 * The CLOCK sweep advances the hand from where it last stopped.  It
 * gives referenced entries a second chance and skips entries with
 * in-flight users.  The second chance provides scan resistance since a
 * streaming pattern cannot flush the recycling working set, whose
 * entries keep their bit set.  The sweep runs on the submit path, from
 * every lookup and again when an insert pushes covered past the cap.
 * The not-due early-out, the trylock, which fails when another
 * submitter is already sweeping, and the visit budget bound the added
 * latency.  A cache that sees no lookups at all does not age; there is
 * no timer, and retirement rides the datapath.
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
	u64 target = io_pfn_cache_target(c, cap);
	struct io_pfn_map *pm;
	unsigned long index, age_before = 0;
	int budget = IO_PFN_EVICT_BUDGET;
	int unmaps = IO_PFN_EVICT_UNMAP_MAX;
	bool over, aging;
	int pass;

	/* A dead-band of a few segments over the target parks the sweep
	 * when a fitting working set sits at its converged size; without
	 * it every insert at the boundary evicts one entry and the sweep
	 * stays hot on the datapath.
	 */
	over = atomic64_read(&c->covered) > target + ((u64)c->quantum << 2);
	aging = max_age && time_after(jiffies, READ_ONCE(c->next_age));
	if (!over && !aging)
		return;

	if (!spin_trylock(&c->lock))
		return;

	if (aging) {
		age_before = jiffies - msecs_to_jiffies(max_age);
		/* Rate-limit the age walk; the cap walk runs on demand. */
		WRITE_ONCE(c->next_age, jiffies + HZ);

		if (READ_ONCE(io_dma_pfn_cache_auto)) {
			/*
			 * Decay the adaptive target when a whole tick has
			 * passed without a single ghost hit: nothing the
			 * cap evicted was missed, so the cache can stand
			 * to be smaller. Growth happens on the miss path.
			 */
			if (time_after(jiffies, c->next_adapt)) {
				u64 gh = atomic64_read(&c->ghost_hits);
				u64 h = atomic64_read(&c->hits);
				u64 m = atomic64_read(&c->misses);
				u64 dh = h - c->hits_snap;
				u64 dm = m - c->misses_snap;
				u64 floor = io_pfn_cache_floor(
						io_pfn_cache_ceiling(cap));
				u64 eff = io_pfn_cache_target(c, cap);

				if (READ_ONCE(c->eff_cap) == IO_PFN_EFF_PARKED) {
					/*
					 * Parked: the cache proved useless
					 * for the current pattern and every
					 * lookup bypasses it. Probe again
					 * periodically; access patterns
					 * change with workload phases.
					 */
					if (++c->parked_ticks >=
					    IO_PFN_PARK_PROBE_TICKS) {
						c->parked_ticks = 0;
						WRITE_ONCE(c->eff_cap, floor);
					}
				} else {
					u32 ratio = (dh + dm) ?
						(u32)div64_u64(dh * 100,
							       dh + dm) : 100;
					bool ghosting =
						gh != c->ghost_hits_snap;

					if (ghosting && ratio < 50 &&
					    ratio <= c->prev_ratio + 2 &&
					    eff >= io_pfn_cache_ceiling(cap)) {
						/*
						 * Gate only once growth is
						 * exhausted: below the
						 * ceiling a low, flat ratio
						 * is what mid-fill looks
						 * like for a set that will
						 * fit, and growth resolves
						 * it. At the ceiling it
						 * cannot.
						 */
						/*
						 * Ghost hits keep demanding
						 * growth but the hit ratio
						 * is low and no longer
						 * improving: a cyclic set
						 * larger than the ceiling,
						 * where growth cannot help
						 * until the whole set fits,
						 * which it never will. A set
						 * still growing toward a fit
						 * shows a climbing ratio and
						 * is spared. Halve, and once
						 * at the floor park: a
						 * floor-sized cache under
						 * cyclic access is pure
						 * mapping churn.
						 */
						if (eff <= floor) {
							WRITE_ONCE(c->eff_cap,
							    IO_PFN_EFF_PARKED);
							c->parked_ticks = 0;
						} else {
							WRITE_ONCE(c->eff_cap,
							    max(eff >> 1,
								floor));
						}
					} else if (!ghosting) {
						eff -= eff >>
						    IO_PFN_ADAPT_DECAY_SHIFT;
						WRITE_ONCE(c->eff_cap,
							   max(eff, floor));
					}
					c->prev_ratio = ratio;
				}
				c->ghost_hits_snap = gh;
				c->hits_snap = h;
				c->misses_snap = m;
				c->next_adapt = jiffies + IO_PFN_ADAPT_TICK;
			}

			/*
			 * Purge stale ghosts: entries older than twice
			 * max_age can no longer say anything about the
			 * present working set. Bounded walk from a hand.
			 */
			{
				unsigned long gidx = 0;
				void *gv;
				int gbudget = IO_PFN_EVICT_BUDGET;
				unsigned long ghost_before = jiffies -
					2 * msecs_to_jiffies(max_age ? max_age : 60000);

				xa_for_each_start(&c->ghost, gidx, gv,
						  c->ghost_hand) {
					if (--gbudget <= 0)
						break;
					if (time_before((unsigned long)xa_to_value(gv),
							ghost_before)) {
						xa_erase(&c->ghost, gidx);
						atomic64_dec(&c->ghost_count);
					}
				}
				c->ghost_hand = gbudget <= 0 ? gidx + 1 : 0;
			}
		}
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
			if (aged) {
				atomic64_inc(&c->age_evictions);
			} else if (READ_ONCE(io_dma_pfn_cache_auto)) {
				/*
				 * Cap pressure took a live-looking entry.
				 * Remember its key so a near-term re-read
				 * can prove the target too small. Values
				 * carry the eviction time for the purge.
				 */
				void *gv = xa_mk_value(jiffies &
						       (LONG_MAX >> 1));

				if (!xa_is_err(xa_store(&c->ghost, index, gv,
						GFP_NOWAIT | __GFP_NOWARN)))
					atomic64_inc(&c->ghost_count);
			}
			io_pfn_map_put(pm);	/* Drop the cache bias. */
			over = atomic64_read(&c->covered) >
				target + ((u64)c->quantum << 2);
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
		/* Retiring every aged entry needs a pass that began at
		 * index zero. Pass 0 starts at the hand, so when the hand
		 * was ahead, aging stays on through pass 1 and the range
		 * above the hand is visited twice, bounded by the budget.
		 */
		if (!start)
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

	/* Give the age sweep a chance on every lookup. Below the cap
	 * with no aging due this is three reads and a compare. Without
	 * this an idle entry would only ever be visited once an insert
	 * pushed the cache over the cap.
	 */
	io_pfn_cache_evict(c, cap);

	/*
	 * Parked: the utility check found the pattern uncacheable, so
	 * skip straight to the caller's plain per-chunk map instead of
	 * inserting entries the sweep immediately evicts. The sweep
	 * call above keeps draining what is left and runs the adapt
	 * tick that eventually un-parks for a fresh look.
	 */
	if (!io_pfn_cache_target(c, cap))
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

	if (READ_ONCE(io_dma_pfn_cache_auto) &&
	    xa_load(&c->ghost, pfn)) {
		if (xa_erase(&c->ghost, pfn)) {
			u64 eff;

			/*
			 * This segment was evicted by cap pressure and is
			 * being remapped: the effective cap is thrashing
			 * the working set. Step the target up toward the
			 * hard cap, one increment per paid-for eviction.
			 */
			atomic64_dec(&c->ghost_count);
			atomic64_inc(&c->ghost_hits);
			/*
			 * Step up, and never converge to exactly the
			 * resident bytes: a target equal to the working
			 * set keeps the evict sweep hot on every insert.
			 * An eighth of headroom over what is currently
			 * mapped parks the sweep once the set fits.
			 */
			eff = io_pfn_cache_target(c, cap);
			eff += (u64)c->quantum * IO_PFN_GHOST_GROW_SEGS;
			eff = max(eff, (u64)atomic64_read(&c->covered) +
				       ((u64)atomic64_read(&c->covered) >> 3));
			WRITE_ONCE(c->eff_cap,
				   min(eff, io_pfn_cache_ceiling(cap)));
		}
	}

	pm = kmalloc_obj(*pm, GFP_NOWAIT | __GFP_NOWARN);
	if (!pm)
		goto fail;
	base = dma_map_page(c->dev, folio_page(folio, 0), seg_base,
			    seg_len, DMA_BIDIRECTIONAL);
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
		dma_unmap_page(c->dev, base, pm->size, DMA_BIDIRECTIONAL);
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
	if (atomic64_add_return(pm->size, &c->covered) >
			io_pfn_cache_target(c, cap))
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
	{
		void *gv;

		xa_for_each(&c->ghost, index, gv)
			xa_erase(&c->ghost, index);
		atomic64_set(&c->ghost_count, 0);
		c->ghost_hand = 0;
		c->eff_cap = 0;	/* re-derive the floor on next use */
		c->hits_snap = atomic64_read(&c->hits);
		c->misses_snap = atomic64_read(&c->misses);
		c->ghost_hits_snap = atomic64_read(&c->ghost_hits);
		c->prev_ratio = 0;
		c->parked_ticks = 0;
	}
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
			   "dev %s quantum_kb %zu covered_kb %lld hits %lld misses %lld inserts %lld insert_fails %lld range_fallbacks %lld evictions %lld age_evictions %lld ref_skips %lld ghost_hits %lld ghost_count %lld eff_cap_mb %llu\n",
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
			   atomic64_read(&c->ref_skips),
			   atomic64_read(&c->ghost_hits),
			   atomic64_read(&c->ghost_count),
			   io_pfn_cache_target(c,
				(u64)READ_ONCE(io_dma_pfn_cache_cap_mb) << 20)
					>> 20);
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
/* DEBUG experiment: force singles to test batch-serialization theory */
static unsigned int io_dma_batch_min __read_mostly = 8;
/*
 * Bounded wait for a descriptor slot on a failed read prep, in
 * microseconds; 0 restores the old shatter-into-CPU-tails behavior.
 * The budget should cover one descriptor service time under load.
 * Default off: measured on saturated 16-job reads, waiting for slots
 * more than halved throughput, because the CPU tails it eliminates
 * are a hybrid copy mode that adds CPU bandwidth on top of the
 * slot-limited device. The knob remains for latency-sensitive or
 * CPU-scarce experiments.
 */
static unsigned int io_dma_slot_wait_us __read_mostly;
/*
 * Channels acquired per ring for read striping. One channel reaches one
 * device, four engines of sixteen; a single-stream op striped across
 * distinct devices reaches them all. Read at ring creation.
 */
unsigned int io_dma_stripe_chans __read_mostly = IO_DMA_RING_CHANS;
/*
 * Bounded outstanding: past this many unreaped descriptors on one ring,
 * a nowait-pass read defers to io-wq instead of engaging, so excess
 * submissions wait rather than deepening the poll list and the
 * detection latency with it. Throughput against queue depth is not
 * monotonic; the peak sits near two thousand descriptors machine-wide,
 * and an application driving past it loses close to a third. 0 is off.
 */
unsigned int io_dma_ring_max_descs __read_mostly;

/*
 * The outstanding-task cap. A nowait-pass submission over the cap
 * defers to io-wq. The io-wq pass may sleep, so instead of engaging
 * over the cap it waits for the reaper to drain below it; the wait is
 * time-bounded so a missed wakeup degrades to a timed engage, never a
 * wedge. Returns true when the caller should defer with -EAGAIN.
 */
bool io_dma_cap_defer(struct io_ring_ctx *ctx, bool nonblock)
{
	unsigned int cap = READ_ONCE(io_dma_ring_max_descs);

	if (!cap)
		return false;
	if (atomic_read(&ctx->dma.tasks_pending) < cap)
		return false;
	if (nonblock)
		return true;
	wait_event_timeout(ctx->dma.inflight_wq,
			   atomic_read(&ctx->dma.tasks_pending) < cap,
			   msecs_to_jiffies(20));
	return false;
}

/* DEBUG experiment: flush threshold, caps entries per batch descriptor */
static unsigned int io_dma_batch_max __read_mostly = IO_DMA_BATCH_MAX;

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
	"eagain", "enomem", "efault", "other", "cpu_tail", "slot_wait",
	"deferred",
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
	bool		released;	/* ghost: keeps its counters readable
					 * until the slot is reused or reset
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

static DEFINE_SPINLOCK(io_dma_stat_slot_lock);

static void io_dma_qstat_zero(struct io_dma_chan_qstat *q)
{
	int b;

	atomic64_set(&q->submits, 0);
	atomic64_set(&q->bytes, 0);
	atomic64_set(&q->inflight, 0);
	atomic64_set(&q->hwm, 0);
	for (b = 0; b < IO_DMA_QSTAT_NBUCKETS; b++) {
		atomic64_set(&q->lat_us[b], 0);
		atomic64_set(&q->seen_kb[b], 0);
	}
}

/*
 * A released channel's slot stays behind as a ghost so its counters
 * remain readable after the last ring exits, which is when they are
 * usually read. The slot is reclaimed when a new channel needs one and
 * no empty slot remains, and freed by latency_reset. A recycled
 * dma_chan allocation that lands on its old ghost is a new channel, so
 * the ghost's counters are cleared rather than inherited. Slot state
 * transitions are rare and take a lock; the fast path is the lock-free
 * pointer match on a live slot.
 */
static struct io_dma_chan_qstat *io_dma_qstat_get(struct io_dma_dev_stat *d,
						  struct dma_chan *chan)
{
	struct io_dma_chan_qstat *hole, *ghost, *q;
	int i;

	for (i = 0; i < IO_DMA_STAT_CHANS; i++) {
		struct io_dma_chan_qstat *c = &d->chans[i];

		if (READ_ONCE(c->chan) == chan && !READ_ONCE(c->released))
			return c;
	}

	spin_lock(&io_dma_stat_slot_lock);
	hole = ghost = NULL;
	q = NULL;
	for (i = 0; i < IO_DMA_STAT_CHANS; i++) {
		struct io_dma_chan_qstat *c = &d->chans[i];
		struct dma_chan *cc = READ_ONCE(c->chan);

		if (cc == chan) {
			q = c;
			break;
		}
		if (!cc && !hole)
			hole = c;
		if (cc && READ_ONCE(c->released) && !ghost)
			ghost = c;
	}
	if (q) {
		/* A recycled allocation reusing its old address. */
		if (READ_ONCE(q->released)) {
			io_dma_qstat_zero(q);
			WRITE_ONCE(q->released, false);
		}
	} else if (hole || ghost) {
		q = hole ? hole : ghost;
		io_dma_qstat_zero(q);
		WRITE_ONCE(q->released, false);
		strscpy(q->name, dma_chan_name(chan), sizeof(q->name));
		WRITE_ONCE(q->chan, chan);
	}
	spin_unlock(&io_dma_stat_slot_lock);
	return q;
}

/*
 * Called when the last ring sharing a channel releases it, with the
 * channel drained and still valid. The slot becomes a ghost rather
 * than being cleared, so a run's counters survive the run's rings.
 */
void io_dma_qstat_forget(struct dma_chan *chan)
{
	int i, j;

	for (i = 0; i < IO_DMA_STAT_DEVS; i++) {
		struct io_dma_dev_stat *d = &io_dma_dev_stats[i];

		if (READ_ONCE(d->key) != chan->device)
			continue;
		for (j = 0; j < IO_DMA_STAT_CHANS; j++) {
			struct io_dma_chan_qstat *q = &d->chans[j];

			if (READ_ONCE(q->chan) == chan)
				WRITE_ONCE(q->released, true);
		}
		return;
	}
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
/*
 * Memo for the registry lookups. The submit and complete hooks run per
 * descriptor and the two linear scans behind them showed up as several
 * percent of submit-path CPU. The memo is validated against the
 * authoritative fields, so a torn or stale entry can never return the
 * wrong slot, only fall back to the scans.
 */
struct io_dma_stat_memo {
	struct dma_chan		*chan;
	struct io_dma_dev_stat	*d;
	struct io_dma_chan_qstat *q;
};
/*
 * Per CPU, not hashed and shared: a global memo array written on every
 * miss by every submitting CPU was a cacheline storm that cost more
 * than the scans it saved. Each CPU tends to feed one ring and so one
 * channel, so a single private entry hits almost always and a miss
 * only rewrites a line this CPU owns.
 */
static DEFINE_PER_CPU(struct io_dma_stat_memo, io_dma_stat_memo_pcpu);

static void io_dma_stat_lookup(struct dma_chan *chan,
			       struct io_dma_dev_stat **dp,
			       struct io_dma_chan_qstat **qp)
{
	struct io_dma_stat_memo *m = get_cpu_ptr(&io_dma_stat_memo_pcpu);
	struct io_dma_dev_stat *d = m->d;
	struct io_dma_chan_qstat *q = m->q;

	if (m->chan == chan && d && q &&
	    READ_ONCE(d->key) == chan->device &&
	    READ_ONCE(q->chan) == chan && !READ_ONCE(q->released)) {
		put_cpu_ptr(&io_dma_stat_memo_pcpu);
		*dp = d;
		*qp = q;
		return;
	}
	d = io_dma_dev_stat_get(chan);
	q = d ? io_dma_qstat_get(d, chan) : NULL;
	if (d && q) {
		m->d = d;
		m->q = q;
		m->chan = chan;
	}
	put_cpu_ptr(&io_dma_stat_memo_pcpu);
	*dp = d;
	*qp = q;
}

static bool io_dma_budget_refuse_wr(struct dma_chan *chan)
{
	u64 budget = READ_ONCE(io_dma_budget_descs);
	struct io_dma_dev_stat *d;

	if (!budget)
		return false;
	{
		struct io_dma_chan_qstat *q;

		io_dma_stat_lookup(chan, &d, &q);
	}
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
	{
		struct io_dma_chan_qstat *q;

		io_dma_stat_lookup(chan, &d, &q);
	}
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
	struct io_dma_dev_stat *d;
	struct io_dma_chan_qstat *q;
	u64 now, hwm;

	io_dma_stat_lookup(chan, &d, &q);
	if (!d)
		return;
	atomic64_inc(&d->inflight);
	if (wr)
		atomic64_inc(&d->inflight_wr);
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
	struct io_dma_dev_stat *d;
	struct io_dma_chan_qstat *q;

	io_dma_stat_lookup(chan, &d, &q);
	if (!d)
		return;
	atomic64_dec(&d->inflight);
	if (wr)
		atomic64_dec(&d->inflight_wr);
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
	if (dma->chan)
		io_dma_qstat_complete(dma->chan, dma->len, 0, false);
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

			/* Released channels leave holes. */
			if (!chan)
				continue;
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
				continue;
			/* Reset frees the ghosts of released channels. */
			if (READ_ONCE(q->released)) {
				spin_lock(&io_dma_stat_slot_lock);
				if (READ_ONCE(q->released)) {
					io_dma_qstat_zero(q);
					q->name[0] = '\0';
					WRITE_ONCE(q->released, false);
					WRITE_ONCE(q->chan, NULL);
				}
				spin_unlock(&io_dma_stat_slot_lock);
				continue;
			}
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
	debugfs_create_u32("batch_min", 0644, dir, &io_dma_batch_min);
	debugfs_create_u32("batch_max", 0644, dir, &io_dma_batch_max);
	debugfs_create_u32("slot_wait_us", 0644, dir, &io_dma_slot_wait_us);
	debugfs_create_u32("stripe_chans", 0644, dir, &io_dma_stripe_chans);
	debugfs_create_u32("ring_max_descs", 0644, dir, &io_dma_ring_max_descs);
	debugfs_create_u32("stats", 0644, dir, &io_dma_stats_enabled);
	debugfs_create_file("latency", 0444, dir, NULL, &io_dma_lat_fops);
	debugfs_create_file("latency_reset", 0200, dir, NULL,
			    &io_dma_lat_reset_fops);
	debugfs_create_file("pfn_cache", 0444, dir, NULL,
			    &io_pfn_cache_stats_fops);
	debugfs_create_u32("pfn_cache_cap_mb", 0644, dir,
			   &io_dma_pfn_cache_cap_mb);
	debugfs_create_u32("pfn_cache_auto", 0644, dir,
			   &io_dma_pfn_cache_auto);
	debugfs_create_u32("pfn_cache_auto_budget_mb", 0644, dir,
			   &io_dma_pfn_cache_auto_budget_mb);
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
	req->dma.min_fail_off = U32_MAX;
	req->dma.claim_len = 0;
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
				   unsigned int nr_entries, u32 off,
				   bool can_wait)
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
		/* Same bounded wait-for-slot as the single path. */
		u64 us = READ_ONCE(io_dma_slot_wait_us);
		u64 end;

		if (!can_wait)
			us /= 8;
		end = ktime_get_ns() + us * NSEC_PER_USEC;
		while (!tx && us && ktime_get_ns() < end) {
			__io_dma_poll(req->ctx);
			if (can_wait)
				cond_resched();
			tx = dmaengine_prep_dma_memcpy_sg(chan, dst_sgl,
							  nr_entries, src_sgl,
							  nr_entries,
							  io_dma_prep_flags());
		}
		if (tx)
			io_dma_fm_record(IO_DMA_FM_SLOT_WAIT);
	}
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
	dma->off = off;
	dma->chan = chan;
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
	io_dma_qstat_submit(chan, dma->len, false);

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

#define IO_DMA_BATCH_MIN	READ_ONCE(io_dma_batch_min)

/*
 * Submit a single DMA descriptor for one batch entry.
 * Used when nr_entries < IO_DMA_BATCH_MIN to avoid batch overhead.
 * The entry already has DMA-mapped src_dma from the caller.
 */
static ssize_t io_dma_submit_single_entry(struct io_kiocb *req,
					  struct dma_chan *chan,
					  struct io_dma_batch_entry *entry,
					  u32 off, bool can_wait)
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
		/*
		 * The channel pool is exhausted. The old behavior took a
		 * short claim and the CPU copied the tail, which at
		 * saturation turned a quarter to half of all large reads
		 * partly back into memcpy. Polling completions retires
		 * finished descriptors and frees their slots, so a
		 * bounded wait converts those CPU tails into a little
		 * submission latency. The nowait pass, where most warm
		 * reads complete, gets an eighth of the budget; the
		 * poll loop never sleeps, so spinning there is legal.
		 */
		u64 us = READ_ONCE(io_dma_slot_wait_us);
		u64 end;

		if (!can_wait)
			us /= 8;
		end = ktime_get_ns() + us * NSEC_PER_USEC;
		while (!tx && us && ktime_get_ns() < end) {
			__io_dma_poll(req->ctx);
			if (can_wait)
				cond_resched();
			tx = dmaengine_prep_dma_memcpy(chan, entry->dst_dma,
						       entry->src_dma,
						       entry->src_len,
						       io_dma_prep_flags());
		}
		if (tx)
			io_dma_fm_record(IO_DMA_FM_SLOT_WAIT);
	}
	if (!tx) {
		io_dma_task_free(req->ctx, dma);
		return -EAGAIN;
	}

	folio_get(entry->folio);

	dma->req = req;
	dma->next = NULL;
	dma->chan = chan;
	dma->src_dma = entry->src_dma;
	dma->dst_dma = entry->dst_dma;
	dma->len = entry->src_len;
	dma->off = off;
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
	io_dma_qstat_submit(chan, dma->len, false);

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
				  unsigned int nr_entries, u32 off,
				  bool can_wait)
{
	ssize_t total = 0;
	unsigned int i;
	ssize_t ret;

	if (!nr_entries)
		return 0;

	if (nr_entries < IO_DMA_BATCH_MIN) {
		for (i = 0; i < nr_entries; i++) {
			ret = io_dma_submit_single_entry(req, chan, &entries[i],
							 off + total, can_wait);
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

	ret = io_dma_submit_batch(req, dev, chan, entries, nr_entries, off,
				  can_wait);
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
	/*
	 * Stripe state. One channel reaches one device, four engines of
	 * sixteen, so a single stream's batches stripe across the ring's
	 * channels. The stripe is deterministic in the file position, at
	 * a 1MB granule, rather than round-robin: every pass over a
	 * region then lands its batches on the same devices, so each
	 * device's PFN cache holds only its own share of the working set
	 * instead of every device converging on a full duplicate copy.
	 * Round-robin cost four times the standing mappings for the same
	 * data, which is what pushed moderate working sets over the IOVA
	 * rcache depletion cliff. Sequential streams and parallel jobs
	 * spread over the granule exactly as they did over the rotation.
	 * Everything device-scoped, meaning the registered-buffer
	 * addresses, the PFN cache instance, and the flush target,
	 * follows the current stripe.
	 */
#define IO_DMA_STRIPE_SHIFT	20
	unsigned int nr_chans = ctx->dma.nr_chans ? ctx->dma.nr_chans : 1;
	/*
	 * Position-consistent mapping only pays when mappings persist:
	 * uncached or parked, it concentrates convoyed readers' copies
	 * on one device for no dedup benefit, so rotate instead. The
	 * regime is sampled per call; a flip mid-workload only changes
	 * which device new batches land on.
	 */
	bool det_stripe;
	unsigned int stripe;
	struct dma_chan *chan;
	struct device *dev;
	struct io_pfn_cache *pfn_cache;
	bool can_wait = !(iocb->ki_flags & IOCB_NOWAIT);
	size_t map_quantum;
	struct folio_batch fbatch;
	struct io_dma_batch_entry *entries;
	unsigned int nr_entries = 0;
	ssize_t total_read = 0;
	ssize_t submitted = 0;
	size_t batch_bytes = 0;
	size_t dst_offset = 0;
	loff_t start_pos = iocb->ki_pos;

	det_stripe = false;
	if (io_pfn_cache_usable() && nr_chans > 1) {
		u64 hard = (u64)READ_ONCE(io_dma_pfn_cache_cap_mb) << 20;
		struct io_pfn_cache *pc =
			io_pfn_cache_get(ctx->dma.chans[0]->device->dev);

		if (pc && io_pfn_cache_target(pc, hard))
			det_stripe = true;
	}
	if (det_stripe)
		stripe = ((u64)iocb->ki_pos >> IO_DMA_STRIPE_SHIFT) %
			 nr_chans;
	else
		stripe = ctx->dma.stripe_rr % nr_chans;
	chan = ctx->dma.nr_chans ? ctx->dma.chans[stripe] : ctx->dma.chan;
	dev = chan->device->dev;
	pfn_cache = io_pfn_cache_usable() ? io_pfn_cache_get(dev) : NULL;
	map_quantum = io_dma_map_quantum(dev);
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
		/* Persist the rotation for the uncached regime's spread. */
	if (!det_stripe)
		ctx->dma.stripe_rr = stripe + 1;

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
						&dst_seg_remain, dev);
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

				/* Flush when the batch is entry-full or
				 * carries this op's share of one engine.
				 * A batch descriptor executes serially on
				 * one engine, so a multi-megabyte op
				 * folded into a single batch runs at
				 * single-engine speed while the rest of
				 * the group idles: 4M reads plateaued at
				 * a third of their rate. Splitting each
				 * op into about four batches, bounded to
				 * [256K, 1M] apiece, spreads it across
				 * engines; the bounds keep small ops from
				 * shattering into per-entry descriptors
				 * and huge ops from oversplitting.
				 */
				if (nr_entries >= min_t(unsigned int,
						READ_ONCE(io_dma_batch_max),
						IO_DMA_BATCH_MAX) ||
				    batch_bytes >= clamp(want / 4,
						(size_t)SZ_256K,
						(size_t)SZ_1M)) {
					ssize_t ret;

					ret = io_dma_flush_batch(req, dev, chan,
						entries, nr_entries,
						dst_offset - batch_bytes,
						can_wait);
					nr_entries = 0;
					if (ret < 0) {
						error = ret;
						goto put_folios;
					}
					/*
					 * Re-derive the stripe for the next
					 * batch from its file position. The
					 * reload keeps every device-scoped
					 * hand, the dst addresses and the
					 * PFN cache, consistent with the
					 * new target.
					 */
					if (nr_chans > 1 &&
					    ret == (ssize_t)batch_bytes) {
						if (det_stripe)
							stripe =
							  ((u64)(start_pos +
							  copied) >>
							  IO_DMA_STRIPE_SHIFT)
							  % nr_chans;
						else
							stripe = (stripe + 1) %
								 nr_chans;
						chan = ctx->dma.chans[stripe];
						dev = chan->device->dev;
						pfn_cache =
						    io_pfn_cache_usable() ?
						    io_pfn_cache_get(dev) :
						    NULL;
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
				entries, nr_entries,
				dst_offset - batch_bytes, can_wait);
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

	if (req->dma.dma_refcnt > 0) {
		unsigned int c;

		/* Descriptors reach hardware at submit time; this is the
		 * belt over every channel a stripe may have touched.
		 */
		for (c = 0; c < nr_chans; c++)
			dma_async_issue_pending(ctx->dma.nr_chans ?
						ctx->dma.chans[c] :
						ctx->dma.chan);
	}
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
};

/* One destination mapping piece, bounded by the map quantum so cached
 * segments serve it. A folio chunk larger than the quantum spans
 * several pieces; the release and the wedge fence walk this array
 * rather than per-chunk mappings.
 */
struct io_dma_fmw_dst {
	dma_addr_t dma;
	unsigned int len;
	unsigned int off;		/* offset in its folio */
	struct folio *folio;
	struct io_pfn_map *pm;		/* NULL = plain per-piece mapping */
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
				unsigned int max_cookies,
				struct io_dma_fmw_dst *dsts,
				unsigned int max_dst)
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
	struct io_pfn_cache *pfn_cache =
		io_pfn_cache_usable() ? io_pfn_cache_get(dev) : NULL;
	struct io_dma_fmw_dst *cur_dst = NULL;
	size_t cur_dst_remain = 0;
	unsigned int nr_dst = 0;
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
		 *
		 * The check runs on every chunk because the fallback is
		 * per write_begin. A concurrent writer can push free
		 * space under the delalloc watermark mid-write, so a
		 * later chunk can be the first to open a handle. That
		 * chunk's write_end below closes it, the collected
		 * prefix commits, and the reissue of the remainder hits
		 * this check on its first chunk.
		 */
		if (unlikely(current->journal_info)) {
			aops->write_end(iocb, mapping, pos, bytes, 0,
					folio, fsdata);
			if (!nr_fol)
				return -EAGAIN;
			break;
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
			redo = true;
			goto record;
		}
		/* Split at source registered-buffer folio boundaries and
		 * at map-quantum boundaries of the destination folio. The
		 * dst folios are the same page-cache folios reads source
		 * from and cache entries are bidirectional, so quantum
		 * sized dst pieces let the cache serve the write side.
		 * A folio chunk mapped whole would exceed the IOVA
		 * rcache class under a strict IOMMU and pay the
		 * allocator slow path on every chunk, which is most of
		 * the write path's gap to passthrough.
		 */
		for (sub = 0; sub < bytes; ) {
			u64 uaddr = src_user_addr + base + collected + sub;
			size_t src_seg_remain, len, dst_remain;
			struct dma_async_tx_descriptor *tx;
			dma_addr_t src_dma, dst;
			dma_cookie_t ck;

			src_dma = io_reg_buf_dma_addr(imu, uaddr,
						      &src_seg_remain,
						      dev);
			if (unlikely(!src_dma)) {
				redo = true;	/* The CPU-redo pass covers the chunk. */
				break;
			}

			/* Acquire the dst piece covering offset + sub. */
			if (!cur_dst || !cur_dst_remain) {
				size_t doff = offset + sub;
				size_t plen = min_t(size_t, bytes - sub,
						max_chunk -
						(doff & (max_chunk - 1)));
				struct io_pfn_map *pm = NULL;
				dma_addr_t d;

				if (nr_dst == max_dst) {
					redo = true;
					break;
				}
				pm = pfn_cache ?
					io_pfn_map_lookup(pfn_cache, folio,
							  doff, plen,
							  folio_size(folio),
							  &d) : NULL;
				if (!pm) {
					d = dma_map_page(dev,
							 folio_page(folio, 0),
							 doff, plen,
							 DMA_FROM_DEVICE);
					if (dma_mapping_error(dev, d)) {
						redo = true;
						break;
					}
				}
				dsts[nr_dst] = (struct io_dma_fmw_dst){
					.dma = d, .len = plen, .off = doff,
					.folio = folio, .pm = pm,
				};
				cur_dst = &dsts[nr_dst++];
				cur_dst_remain = plen;
			}
			dst = cur_dst->dma +
			      (cur_dst->len - cur_dst_remain);
			dst_remain = cur_dst_remain;
			len = min3(bytes - sub, src_seg_remain, dst_remain);

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
			cur_dst_remain -= len;
		}
		cur_dst = NULL;
		cur_dst_remain = 0;
record:
		fol[nr_fol++] = (struct io_dma_fmw_folio){
			.folio = folio, .fsdata = fsdata, .pos = pos,
			.len = bytes,
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
	for (i = 0; i < nr_dst; i++) {
		if (dsts[i].pm) {
			/* On a wedge the cached mapping must not stay
			 * device-writable, so displace it; the unmap
			 * happens when the last reference drops, which
			 * is this put unless a concurrent I/O holds the
			 * same segment.
			 */
			if (unlikely(wedged))
				io_pfn_map_displace(pfn_cache, dsts[i].pm);
			io_pfn_map_put(dsts[i].pm);
		} else {
			dma_unmap_page(dev, dsts[i].dma, dsts[i].len,
				       DMA_FROM_DEVICE);
		}
	}

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
	 * advance it over exactly what was committed. The two counts
	 * cannot diverge: the source is a pinned registered-buffer
	 * bvec, so copy_folio_from_iter() never copies short, every
	 * claim therefore equals its chunk length, and write_end
	 * with a full claim returns it whole or fails outright.
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
	struct io_dma_fmw_dst *dsts = NULL;
	unsigned int max_fol, max_cookies, max_dst;
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
	/* One dst piece per map-quantum crossing per folio chunk. */
	max_dst = max_fol * (1 + DIV_ROUND_UP(
			mapping_max_folio_size(file->f_mapping),
			io_dma_map_quantum(req->ctx->dma.chan->device->dev)));
	fol = kvmalloc_array(max_fol, sizeof(*fol), GFP_KERNEL);
	cookies = kvmalloc_array(max_cookies, sizeof(*cookies), GFP_KERNEL);
	dsts = kvmalloc_array(max_dst, sizeof(*dsts), GFP_KERNEL);
	if (!fol || !cookies || !dsts) {
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
						max_fol, cookies, max_cookies,
						dsts, max_dst);

		if (done <= 0) {
			/* A zero-progress group falls back to the normal
			 * write path rather than completing as a zero-byte
			 * write.
			 */
			if (!written)
				err = done ? done : -EAGAIN;
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
	kvfree(dsts);
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
		/* Everything below the lowest failed offset was delivered
		 * by tasks that are disjoint from this one, so it stays
		 * claimable as a contiguous prefix.
		 */
		req->dma.min_fail_off = min(req->dma.min_fail_off, dma->off);
	}

	io_dma_qstat_complete(dma->chan, task_len,
			      dma->submit_ns, false);

	/* Free the task before touching the refcnt. task_len was saved above. */
	atomic_dec(&req->ctx->dma.tasks_pending);
	if (READ_ONCE(io_dma_ring_max_descs)) {
		/*
		 * Order the dec against the waitqueue_active() read; the
		 * waiter's prepare_to_wait() barrier pairs with it. Plain
		 * atomic_dec() has no return and so no implicit ordering.
		 */
		smp_mb__after_atomic();
		if (waitqueue_active(&req->ctx->dma.inflight_wq))
			wake_up(&req->ctx->dma.inflight_wq);
	}
	io_dma_task_free(req->ctx, dma);
	req->dma.dma_refcnt--;

	if (req->dma.dma_refcnt == 0) {
		pr_debug("dma req done: opcode=%d result=%d\n",
			 req->opcode, req->dma.dma_result);

		if (req->dma.dma_result < 0) {
			int res = req->dma.dma_result;

			/*
			 * Completion is out of order, so the bytes that
			 * completed need not form a contiguous prefix. The
			 * bytes below the lowest failed offset do, since
			 * every task lying wholly below it succeeded, so a
			 * failure mid-request reports a short read up to
			 * that point plus whatever earlier issues already
			 * delivered (saved_res minus this issue's claim).
			 * Only a request with nothing deliverable at all
			 * fails whole.
			 */
			if (req->dma.min_fail_off != U32_MAX) {
				int prior = req->dma.saved_res -
					    req->dma.claim_len;
				int short_res = prior +
						(int)req->dma.min_fail_off;

				if (short_res > 0)
					res = short_res;
			}
			if (res < 0)
				req_set_fail(req);
			io_req_set_res(req, res, req->dma.saved_cflags);
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
				 * freed immediately. The pending count
				 * rises before the task is visible, so no
				 * observer can see the task without the
				 * count.
				 */
				struct io_dma_task *nxt = t->next;

				atomic_inc(&ctx->dma.tasks_pending);
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
			ret = dmaengine_async_is_tx_complete(dma->chan,
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
			io_dma_task_release_res(ctx,
					dma->chan->device->dev, dma);
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
			break;
		if (ktime_get_ns() >= end_ns)
			break;
		cpu_relax();
	}

	/*
	 * Abandoning the spin with tasks still pending must hand them to
	 * the poll worker, because this caller is about to block and the
	 * worker's own pending check can misread during another poller's
	 * splice window: llist_del_all() has emptied the submit list
	 * while the spliced tasks are still in walker-local variables,
	 * so both lists look empty and the worker exits. With both
	 * pollers gone nothing ever retires the remaining descriptors
	 * and the waiter above sleeps forever.
	 */
	queue_work(system_unbound_wq, &ctx->dma.poll_work);
	return false;
}
