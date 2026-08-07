// SPDX-License-Identifier: GPL-2.0

#include <linux/io_uring_types.h>
#include <linux/io_uring.h>
#include <linux/backing-dev.h>
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
#include <linux/kmemleak.h>
#include "io_uring.h"
#include "rw.h"
#include "pfn_mrc.h"
#include <linux/bitmap.h>
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
 * IOVA space and a lingering window in which the device can read and
 * write the mapped memory, bounded by the cap.  That memory may be
 * freed and reused while it stays mapped, so the cache relies on the
 * device being programmed by the kernel alone.
 *
 * A standing mapping outlives the I/O that made it, so the ownership
 * transfers a per-chunk map and unmap perform at their ends have no
 * place here: segments are mapped with DMA_ATTR_SKIP_CPU_SYNC and the
 * datapath transfers explicitly, handing a source chunk to the device
 * before its descriptor is issued and a destination piece to the
 * device before its write and back to the CPU once the write has
 * landed. On a coherent device each transfer is a flag test.
 *
 * The entry lifetime is refs = 1 cache bias plus one per in-flight
 * batch entry.  Lookup takes a ref with atomic_inc_not_zero() under
 * RCU.  Eviction erases the entry and drops the bias, so a mapping
 * survives until its last in-flight user completes.
 *
 * Mappings are keyed to the DSA struct device and live in its IOMMU
 * domain, so the cache gives them and its reservation back when the
 * device's driver unbinds, and reserves again for the next binding.
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
	struct io_pfn_cache	*cache;
	int			slot;		/* slot in the cache's reserved
						 * IOVA range, or -1 for a plain
						 * per-entry mapping
						 */
	struct rcu_head		rcu;
};

struct io_pfn_cache {
	/*
	 * Every lookup reads the fields up to covered, and every hit on
	 * every CPU writes the counters from hits on, so the counters start
	 * a cache line of their own. With covered in their line, 128KB
	 * reads lost 11 to 14 percent of their throughput. A new field read
	 * on lookup goes in a hole above.
	 */
	struct xarray		xa;
	spinlock_t		lock;		/* serializes the CLOCK sweep */
	bool			dead;		/* the binding or the device is
						 * going away: no lookup or
						 * insert uses the cache
						 */
	bool			gone;		/* retired: reservation freed,
						 * collector held until a
						 * revive
						 */
	u8			nr_sharing;	/* caches the budget is split
						 * across, this one included
						 * (io_pfn_cache_ceiling())
						 */
	unsigned long		hand;		/* next PFN the sweep visits */
	unsigned long		next_age;	/* jiffies, next age walk due */
	struct device		*dev;
	size_t			quantum;	/* pow2 segment size entries are
						 * carved into (io_dma_map_quantum)
						 */
	atomic64_t		covered;	/* bytes mapped through the tree */
	atomic64_t		hits ____cacheline_aligned_in_smp;
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
	 * its eviction time under the key scaled down by the quantum
	 * (io_pfn_ghost_index()). A miss that hits the ghost is an eviction
	 * the workload paid for with a remap, which is the one signal
	 * that separates "the cap is trimming dead streaming entries"
	 * from "the cap is thrashing a live working set". Ghost hits
	 * grow the adaptive target below; their absence decays it. Age
	 * evictions never enter the ghost, idleness is not undersizing.
	 */
	struct xarray		ghost;
	unsigned long		ghost_hand;	/* purge and replace hand */
	atomic64_t		ghost_count;
	atomic64_t		ghost_hits;
	u64			eff_cap;	/* adaptive target, bytes;
						 * 0 uninitialized, 1 parked
						 */
	u64			ghost_hits_snap;/* at the last decay tick */
	u64			hits_snap;	/* ditto, hits plus waits */
	u64			misses_snap;
	u32			prev_ratio;	/* hit % at the last tick */
	u32			parked_ticks;	/* ticks spent parked */
	u64			inserts_snap;	/* at the last tick */
	u64			evictions_snap;	/* cap evictions only */
	u32			dead_ticks;	/* ticks that only churned */
	u32			cyclic_turnover;/* cap evictions in a row of
						 * cyclic ticks
						 */
	unsigned long		next_adapt;	/* jiffies */

	struct io_pfn_mrc	*mrc;		/* reuse-time sampler, or NULL */

	/*
	 * Reserved IOVA range, allocated at creation or by a revive and
	 * freed by the retire, so no range carries over from one binding of
	 * the device to the next. Entries are linked in at slot * quantum
	 * and unlinked once retired.
	 * Standing mappings then never touch the IOVA allocator: no fill
	 * ever walks its rbtree, and no flush or eviction parks a range in
	 * its rcache depot, which is where a burst of frees left a million
	 * free-but-cached nodes for every later allocation to walk. With
	 * no reservation (no IOMMU, or the range could not be had) entries
	 * fall back to per-entry maps.
	 */
	struct dma_iova_state	iova;
	unsigned int		nslots;		/* 0: no reservation */
	unsigned int		quantum_shift;
	unsigned long		*slots;		/* used-slot bitmap */
	unsigned long		*dirty;		/* retired, still linked */
	struct io_pfn_map	**by_slot;	/* the entry holding each slot */
	unsigned int		nr_dirty;
	unsigned int		slot_hint;	/* next slot to try */
	unsigned int		slot_hand;	/* CLOCK hand over the ring */
	spinlock_t		slot_lock;
	atomic_t		gc_busy;
	bool			gc_rescan;	/* a caller left its slots
						 * to the collector at work,
						 * under slot_lock
						 */
	bool			age_walking;	/* an age walk has retired
						 * and not yet collected
						 */
	atomic64_t		slot_links;	/* entries linked into the range */
	atomic64_t		slot_fails;	/* range full or link failed */
	atomic64_t		unlink_runs;	/* dma_iova_unlink() calls */
	atomic64_t		unlink_slots;	/* slots those covered */
	atomic64_t		waits;		/* hits taken after waiting for
						 * another CPU's insert
						 */
	atomic64_t		wait_timeouts;	/* waits that ran out */
	atomic64_t		short_fallbacks;/* segment shorter than the
						 * quantum: not cached
						 */
	struct mutex		life_lock;	/* orders retire and revive */
	struct work_struct	age_work;	/* io_pfn_cache_age_work() */
	unsigned int		nr_used;	/* bits set in slots, under
						 * slot_lock
						 */
	bool			removed;	/* the device is gone for good,
						 * under life_lock
						 */
};

/*
 * A miss claims its key with this marker before it maps, so the
 * followers of a convoy find the claim and wait for the entry instead
 * of each mapping the segment, losing the insert race to the leader and
 * unmapping again: a fifth of all touches did that under four jobs. The
 * marker holds no bytes and no reference; walkers skip it.
 */
#define IO_PFN_PENDING		xa_mk_value(0)

#define IO_PFN_CACHE_DEVS	16
static struct io_pfn_cache *io_pfn_caches[IO_PFN_CACHE_DEVS];
static struct device *io_pfn_cache_devs[IO_PFN_CACHE_DEVS];
static DEFINE_SPINLOCK(io_pfn_cache_reg_lock);
static void io_pfn_cache_kill(struct io_pfn_cache *c, unsigned long action);
static struct io_pfn_cache *io_pfn_cache_get(struct device *dev);
static void io_pfn_cache_age_work(struct work_struct *work);

/*
 * Covered-bytes cap in MiB, where 0 disables the cache entirely.  This
 * is the hard bound on the standing window, the bytes the device can
 * reach until an entry is retired; the adaptive target sizes itself
 * below it from the workload's reuse and parks on streaming, so the
 * cap need not be tuned to the working set.  It defaults to the IOVA
 * reservation, which is what the adaptive ceiling is anyway: a cap
 * above the reservation only spills entries to per-entry maps, and one
 * below it forgoes coverage the reservation already paid for.
 */
static u32 io_dma_pfn_cache_cap_mb __read_mostly = 65536;

/*
 * Clears of the cap whose first round of flushes has yet to finish. A
 * put at cap 0 that collects leaves its slot to such a clear, which
 * flushes every cache once more after the count drops
 * (io_pfn_slots_gc_cleared()).
 */
static atomic_t io_pfn_cache_clearing;

/*
 * Age at which an idle entry is retired even though the cache is under
 * its cap, in milliseconds, where 0 keeps entries until the cap forces
 * them out.  The cap alone bounds the standing device-reachable window
 * in bytes and says nothing about how long any one mapping persists,
 * so this bounds it in time: a segment that no I/O has touched for
 * this long is unmapped, no more than a quarter of this later, or a
 * second when that is longer, plus the walk itself, as long as the cache
 * still sees lookups (io_pfn_cache_age_period()).  A write re-arms every
 * cache, so a lowered age applies from the next lookup.
 */
static u32 io_dma_pfn_cache_max_age_ms __read_mostly = 60000;

/*
 * Adaptive sizing.  When set, cap_mb is a ceiling rather than the
 * standing target: the cache aims for an effective cap that ghost
 * hits grow toward the ceiling and quiet intervals decay toward a
 * floor of an eighth of the ceiling.  A streaming phase then keeps
 * the standing device-reachable window small, and a re-read phase
 * whose working set the floor thrashes earns its way back up at one
 * step per paid-for eviction.  0 pins the target at cap_mb, which is
 * the historical behaviour.
 */
static u32 io_dma_pfn_cache_auto __read_mostly = 1;

/*
 * Machine-wide standing-mapping budget for adaptive mode, in MiB, or 0
 * for none, in which case a cache's ceiling is its IOVA reservation.
 * The budget existed to keep the caches under an IOVA allocator cliff
 * that the reserved ranges have since removed: standing mappings no
 * longer touch the allocator, and what a large cache costs is its
 * IOMMU page tables (a fraction of a percent of the bytes) and a
 * longer standing window before an idle entry ages out. A host that
 * wants to bound the total anyway sets this; it is split across the
 * caches currently holding mappings.
 */
static u32 io_dma_pfn_cache_auto_budget_mb __read_mostly;

/*
 * Registry slots taken. A removed device's slot passes to a new device
 * and is not counted again (__io_pfn_cache_get()).
 */
static atomic_t io_pfn_cache_nr;

/*
 * Reuse-time sampling for the miss ratio curve: 1 in 2^shift segment
 * touches per cache, 0 off. Measurement only; the histogram is read
 * through pfn_cache_rtd and solved offline.
 */
static u32 io_dma_pfn_mrc_shift __read_mostly;

/*
 * IOVA reserved per cache, in MiB, taken when the cache is created; 0
 * leaves the cache on per-entry maps. Reservation is address space
 * only, page tables are built as entries link in, so on a 56-bit device
 * the default costs nothing until used. A cap above the reservation
 * still works: entries past the range take plain maps. A cache that an
 * unbind retired reserves the size it took at creation again when a ring
 * on the rebound device revives it, or nothing if it took none, whatever
 * the knob says then.
 */
static u32 io_dma_pfn_iova_reserve_mb __read_mostly = 65536;

/*
 * How long a lookup waits for another CPU's insert of the same segment,
 * in microseconds, before mapping for itself. A link and its sync take
 * a few microseconds; the wait costs nothing when no insert is pending.
 */
static u32 io_dma_pfn_wait_us __read_mostly = 20;

/* Ghost growth per hit and decay per quiet 2s tick (eff >> shift). */
#define IO_PFN_GHOST_GROW_SEGS	64
#define IO_PFN_ADAPT_DECAY_SHIFT 3
#define IO_PFN_ADAPT_TICK	(2 * HZ)

/*
 * The caches the machine budget is split across: those currently holding
 * mappings, and @c, which asks for its ceiling only to hold some.
 * Registration outlives a device's use, but a device that a past workload
 * touched and this one does not must not dilute the budget: with rings
 * acquiring channels from a shared pool, the registry accretes devices
 * over time, and dividing by the all-time count starved every active
 * cache.
 */
static u8 io_pfn_cache_sharing(struct io_pfn_cache *c)
{
	unsigned int i;
	u8 n = 1;

	for (i = 0; i < IO_PFN_CACHE_DEVS; i++) {
		struct io_pfn_cache *pc;

		if (!READ_ONCE(io_pfn_cache_devs[i]))
			break;
		pc = io_pfn_caches[i];
		if (pc && pc != c && atomic64_read(&pc->covered) > 0)
			n++;
	}
	return n;
}

/*
 * Per-cache adaptive ceiling: the machine budget's fair share among the
 * caches that share it. The share count is taken when the cache is
 * created and on every adaptive tick, and the per-tick recount converges
 * as traffic shifts. Counting on every call would read every cache's
 * covered count, whose line every claim, publish and retire writes, on
 * every lookup.
 */
static u64 io_pfn_cache_ceiling(struct io_pfn_cache *c, u64 hard)
{
	u64 budget = (u64)READ_ONCE(io_dma_pfn_cache_auto_budget_mb) << 20;
	u64 ceil = hard;

	if (c->nslots)
		ceil = min(ceil, (u64)c->nslots << c->quantum_shift);
	if (!budget)
		return ceil;
	return min(ceil, div_u64(budget, READ_ONCE(c->nr_sharing)));
}

static u64 io_pfn_cache_floor(u64 ceil)
{
	return min(ceil, clamp(ceil >> 3, (u64)SZ_64M, (u64)SZ_8G));
}

/*
 * The eviction target: the adaptive effective cap, clamped to the
 * per-cache ceiling, or the hard cap itself when auto sizing is off.
 */
#define IO_PFN_EFF_PARKED	1
#define IO_PFN_PARK_PROBE_TICKS	8
/*
 * A tick with this many inserts, no ghost hit, about as many evictions
 * as inserts, and a low chunk hit ratio is a streaming pattern: little
 * of what it inserted was reused before it left. Two in a row park the
 * cache. The ratio guard keeps a cache that serves thirty-one of every
 * thirty-two chunk lookups of a small-block stream engaged even though
 * no segment ever repeats.
 * The margin under three quarters keeps a four-job convoy engaged: it
 * reads three of every four lookups as hits, less its followers'
 * timed-out waits, its dropped claims and the segments whose miss and
 * follower hits fall on either side of a tick boundary.
 */
#define IO_PFN_DEAD_MIN_INSERTS	1024
#define IO_PFN_DEAD_MAX_RATIO	70
/*
 * A tick at a full ceiling with ghost hits and a chunk hit ratio under
 * this that has stopped improving looks like a cyclic set that will never
 * fit. The margin under one half keeps a two-job convoy engaged: it reads
 * half its lookups as hits, less its followers' timed-out waits and the
 * segments whose miss and follower hit fall on either side of a tick
 * boundary.
 */
#define IO_PFN_CYCLIC_MAX_RATIO	45

static u64 io_pfn_cache_target(struct io_pfn_cache *c, u64 hard)
{
	u64 eff, ceil;

	if (!READ_ONCE(io_dma_pfn_cache_auto))
		return hard;
	ceil = io_pfn_cache_ceiling(c, hard);
	eff = READ_ONCE(c->eff_cap);
	if (eff == IO_PFN_EFF_PARKED)
		return 0;
	if (!eff)
		eff = io_pfn_cache_floor(ceil);
	return min(eff, ceil);
}

/* Sweep visit budget per eviction call.  This bounds the datapath
 * latency when the table is large and mostly referenced or in flight.
 * The age walk visits as many entries per RCU read-side batch.
 */
#define IO_PFN_EVICT_BUDGET	1024

/* Unmaps performed inline per eviction call. An insert overshoots the
 * cap by at most one segment, so the steady state evicts about one
 * entry. Lowering the cap at runtime is the case that would otherwise
 * unmap the whole excess in one sweep, with preemption disabled, so
 * cap the inline work and leave the rest to the following inserts.
 * The age walk and a flush end an RCU read-side batch after as many
 * retirements.
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

static void io_pfn_cache_reserve(struct io_pfn_cache *c, gfp_t gfp)
{
	u64 bytes = (u64)READ_ONCE(io_dma_pfn_iova_reserve_mb) << 20;
	u64 nslots = bytes >> c->quantum_shift;

	/*
	 * dma_iova_try_alloc() takes a size_t. On 32-bit a size of 4 GiB or
	 * more would be truncated, to nothing or to less than what the slots
	 * span, so such a cache keeps to per-entry maps.
	 */
	if (!nslots || nslots > UINT_MAX || bytes != (size_t)bytes)
		return;
	c->slots = bitmap_zalloc(nslots, gfp);
	c->dirty = bitmap_zalloc(nslots, gfp);
	c->by_slot = kvcalloc(nslots, sizeof(*c->by_slot), gfp);
	if (!c->slots || !c->dirty || !c->by_slot ||
	    !dma_iova_try_alloc(c->dev, &c->iova, 0, bytes)) {
		bitmap_free(c->slots);
		bitmap_free(c->dirty);
		kvfree(c->by_slot);
		c->slots = c->dirty = NULL;
		c->by_slot = NULL;
		return;
	}
	spin_lock_init(&c->slot_lock);
	c->nslots = nslots;
}

static void io_pfn_cache_unreserve(struct io_pfn_cache *c)
{
	if (c->nslots)
		dma_iova_free(c->dev, &c->iova);
	bitmap_free(c->slots);
	bitmap_free(c->dirty);
	kvfree(c->by_slot);
}

/*
 * Give back the slot arrays of a removed device's cache. A removed device
 * never binds again, so its cache is never revived, and once every slot
 * that a transfer held at the retire has had its last put and been marked,
 * nothing touches the arrays again: the retire holds the collector,
 * inserts refuse, and a sweep on the dead cache leaves the ring
 * (io_pfn_cache_evict()). Until then the arrays stay, and ring setup for
 * a device without a cache tries again whenever its walk of the registry
 * passes this one (io_pfn_cache_full()). The caller holds c->life_lock.
 */
static void io_pfn_cache_release(struct io_pfn_cache *c)
{
	bool clean;

	lockdep_assert_held(&c->life_lock);
	if (!c->slots || !READ_ONCE(c->gone))
		return;
	spin_lock(&c->slot_lock);
	clean = bitmap_equal(c->slots, c->dirty, c->nslots);
	spin_unlock(&c->slot_lock);
	if (!clean)
		return;
	bitmap_free(c->dirty);
	kvfree(c->by_slot);
	c->dirty = NULL;
	c->by_slot = NULL;
	bitmap_free(c->slots);
	WRITE_ONCE(c->slots, NULL);
}

/*
 * Whether @c's registry slot can pass to a new device: its device was
 * removed and its slot arrays are released. The cache itself stays
 * allocated, since lookups and in-flight entries may still hold it.
 */
static bool io_pfn_cache_reusable(struct io_pfn_cache *c)
{
	return READ_ONCE(c->removed) && !READ_ONCE(c->slots);
}

/*
 * Whether every registry slot holds another device whose cache is still
 * in use. Such a device runs uncached, which is reported once. A removed
 * device's cache whose arrays a transfer still held at the removal is
 * given another chance to release them here. Each slot is read once. A
 * second read could find the slot that @dev took after the first, and so
 * report a full registry for a device that has a cache.
 */
static bool io_pfn_cache_full(struct device *dev)
{
	int i;

	for (i = 0; i < IO_PFN_CACHE_DEVS; i++) {
		/* Pairs with the release in __io_pfn_cache_get(). */
		struct device *d = smp_load_acquire(&io_pfn_cache_devs[i]);
		struct io_pfn_cache *c;

		if (!d || d == dev)
			return false;
		c = READ_ONCE(io_pfn_caches[i]);
		if (READ_ONCE(c->removed) && READ_ONCE(c->slots)) {
			mutex_lock(&c->life_lock);
			io_pfn_cache_release(c);
			mutex_unlock(&c->life_lock);
		}
		if (io_pfn_cache_reusable(c))
			return false;
	}
	dev_warn_once(dev, "io_uring DMA: mapping cache registry full, device runs uncached\n");
	return true;
}

static struct io_pfn_cache *__io_pfn_cache_get(struct device *dev, gfp_t gfp)
{
	struct io_pfn_cache *c, *old;
	int i, slot = -1;

	c = io_pfn_cache_get(dev);
	if (c)
		return c;

	c = kzalloc(sizeof(*c), gfp);
	if (!c)
		return NULL;
	xa_init(&c->xa);
	xa_init(&c->ghost);
	spin_lock_init(&c->lock);
	INIT_WORK(&c->age_work, io_pfn_cache_age_work);
	c->dev = dev;
	c->quantum = io_dma_map_quantum(dev);
	c->quantum_shift = ilog2(c->quantum);
	c->nr_sharing = io_pfn_cache_sharing(c);
	mutex_init(&c->life_lock);
	io_pfn_cache_reserve(c, gfp);
	if (READ_ONCE(io_dma_pfn_mrc_shift))
		c->mrc = io_pfn_mrc_alloc(gfp);

	/*
	 * A removed device's slot can sit before @dev's, so the whole
	 * prefix is searched for @dev before a slot is taken.
	 */
	spin_lock(&io_pfn_cache_reg_lock);
	for (i = 0; i < IO_PFN_CACHE_DEVS; i++) {
		if (io_pfn_cache_devs[i] == dev) {	/* We lost an insert race. */
			spin_unlock(&io_pfn_cache_reg_lock);
			io_pfn_mrc_free(c->mrc);
			io_pfn_cache_unreserve(c);
			kfree(c);
			return io_pfn_caches[i];
		}
		if (!io_pfn_cache_devs[i])
			break;
		if (slot < 0 && io_pfn_cache_reusable(io_pfn_caches[i]))
			slot = i;
	}
	if (slot < 0 && i < IO_PFN_CACHE_DEVS)
		slot = i;
	if (slot >= 0) {
		/*
		 * Pin the device for good. A cache that gives its slot up
		 * stays allocated and names its device, through which the
		 * entries that transfers still hold unmap, so the reference
		 * is never dropped, and a device that comes back after
		 * removal is a new one, which the retired cache is never
		 * handed. A lookup that read the slot's old device can then
		 * load the new cache, so lookups check the cache's device.
		 */
		get_device(dev);
		old = io_pfn_caches[slot];
		if (old)
			kmemleak_not_leak(old);
		else
			atomic_inc(&io_pfn_cache_nr);
		/*
		 * Orders the new cache's initialization before a lookup
		 * that loads it without having seen the new device.
		 */
		smp_store_release(&io_pfn_caches[slot], c);
		/* pairs with the lockless loads in io_pfn_cache_get() */
		smp_store_release(&io_pfn_cache_devs[slot], dev);
		spin_unlock(&io_pfn_cache_reg_lock);
		return c;
	}
	spin_unlock(&io_pfn_cache_reg_lock);
	io_pfn_mrc_free(c->mrc);
	io_pfn_cache_unreserve(c);
	kfree(c);	/* The registry is full, so this device runs uncached. */
	return NULL;
}

/*
 * Datapath lookup. It never creates a cache: only io_pfn_cache_prepare()
 * does, where the reservation can be allocated and the device's unbind
 * and removal watched.
 */
static struct io_pfn_cache *io_pfn_cache_get(struct device *dev)
{
	int i;

	for (i = 0; i < IO_PFN_CACHE_DEVS; i++) {
		/* Pairs with the slot-publishing release in __io_pfn_cache_get(). */
		struct device *d = smp_load_acquire(&io_pfn_cache_devs[i]);

		if (d == dev) {
			struct io_pfn_cache *c = READ_ONCE(io_pfn_caches[i]);

			/* The slot may have passed to a new device since. */
			return c->dev == dev ? c : NULL;
		}
		if (!d)
			break;
	}
	return NULL;
}

/*
 * Everything a cache holds lives in its device's IOMMU domain, and every
 * IOVA call dereferences the domain the device has at the time. Once the
 * driver has unbound, the group's domain type can be changed, which frees
 * the domain, and removal frees it too. dma-debug also reports whatever
 * the device still has mapped when the unbind completes. So the cache is
 * retired when the driver starts to unbind, while the domain cannot be
 * replaced, and a slot that a transfer still in flight holds is unlinked
 * once the unbind completes and the driver has stopped the device. PCI
 * unbinds the driver before it removes the device. A bus that removes a
 * bound device sends BUS_NOTIFY_DEL_DEVICE first, which comes before the
 * IOMMU's BUS_NOTIFY_REMOVED_DEVICE, so the cache is retired there while
 * the domain still exists. Such a bus can report the removal while
 * another task unbinds the driver, so each cache's life_lock serializes
 * its retires. The lock is per cache: the retire that completes an
 * unbind runs after the driver core has dropped the binding's hold on
 * the domain, and a wait there for another device's retire would widen
 * the window in which a type change can free the domain. The removal
 * sends BUS_NOTIFY_REMOVED_DEVICE once the driver has unbound, and the
 * device can no longer bind, so a removed device's cache gives back its
 * slot arrays there, and with them its registry slot.
 */
static int io_pfn_cache_bus_notify(struct notifier_block *nb,
				   unsigned long action, void *data)
{
	struct device *dev = data;
	int i;

	if (action != BUS_NOTIFY_UNBIND_DRIVER &&
	    action != BUS_NOTIFY_UNBOUND_DRIVER &&
	    action != BUS_NOTIFY_DEL_DEVICE &&
	    action != BUS_NOTIFY_REMOVED_DEVICE)
		return NOTIFY_DONE;
	for (i = 0; i < IO_PFN_CACHE_DEVS; i++) {
		/* Pairs with the slot-publishing release in __io_pfn_cache_get(). */
		struct device *d = smp_load_acquire(&io_pfn_cache_devs[i]);

		if (!d)
			break;
		if (d == dev) {
			struct io_pfn_cache *c = READ_ONCE(io_pfn_caches[i]);

			if (c->dev != dev)
				continue;
			mutex_lock(&c->life_lock);
			if (action == BUS_NOTIFY_REMOVED_DEVICE) {
				WRITE_ONCE(c->removed, true);
				io_pfn_cache_release(c);
			} else {
				io_pfn_cache_kill(c, action);
			}
			mutex_unlock(&c->life_lock);
		}
	}
	return NOTIFY_OK;
}

#define IO_PFN_CACHE_BUSES	4
static const struct bus_type *io_pfn_cache_buses[IO_PFN_CACHE_BUSES];
static struct notifier_block io_pfn_cache_bus_nb[IO_PFN_CACHE_BUSES];
static DEFINE_MUTEX(io_pfn_cache_bus_mutex);

/*
 * Watch @dev's bus for driver unbind and device removal. Returns false
 * when that cannot be had.
 */
static bool io_pfn_cache_watch(struct device *dev)
{
	const struct bus_type *bus = dev->bus;
	bool watched = false;
	int i;

	if (!bus)
		return false;
	mutex_lock(&io_pfn_cache_bus_mutex);
	for (i = 0; i < IO_PFN_CACHE_BUSES; i++) {
		if (io_pfn_cache_buses[i] == bus) {
			watched = true;
			break;
		}
		if (io_pfn_cache_buses[i])
			continue;
		io_pfn_cache_bus_nb[i].notifier_call = io_pfn_cache_bus_notify;
		/*
		 * The notifier runs ahead of dma-debug's check for mappings
		 * a driver left behind, which runs at the default priority
		 * when the unbind completes, so a retire finished there is
		 * not reported.
		 */
		io_pfn_cache_bus_nb[i].priority = 1;
		if (!bus_register_notifier(bus, &io_pfn_cache_bus_nb[i])) {
			io_pfn_cache_buses[i] = bus;
			watched = true;
		}
		break;
	}
	mutex_unlock(&io_pfn_cache_bus_mutex);
	return watched;
}

/*
 * Bring back a cache that an unbind retired, for its device's new
 * binding. The caller holds the device lock and c->life_lock. The old
 * range went with the old binding, so the range is reserved again in
 * whatever domain the device has now, with the cache's own geometry,
 * which lockless lookups read and so never changes. Nothing from before
 * the retire may carry over. An insert that mapped before the retire
 * keeps its claim, and then its entry, in the xarray until it sees the
 * mark and takes the entry back out, and its mapping is in the old
 * domain. The retire's grace period orders that claim before the retire,
 * so the test below finds it. The test looks for a present entry, claims
 * included, instead of testing the head, since a claim whose node
 * allocations failed can leave an empty node linked, which no insert on a
 * dead cache would ever free. An entry that a transfer from before the
 * retire still holds keeps its slot until its last put marks it, so the
 * slot state is reset only once every slot the old range showed is
 * marked. Until then, while the quantum does not fit the domain, and
 * while no range can be had, the cache stays retired and the device runs
 * uncached. The marks are tested before anything is reserved. Once
 * complete they stay so, since a retired cache hands out no slot and
 * holds its collector. A lookup takes the slot lock before it links, so
 * the unlock after the reset orders the new range before any lookup that
 * finds the cache live.
 */
static void io_pfn_cache_revive(struct io_pfn_cache *c)
{
	unsigned long index = 0;
	bool clean;

	if (xa_find(&c->xa, &index, ULONG_MAX, XA_PRESENT) ||
	    c->quantum != io_dma_map_quantum(c->dev))
		return;
	if (c->nslots) {
		spin_lock(&c->slot_lock);
		clean = bitmap_equal(c->slots, c->dirty, c->nslots);
		spin_unlock(&c->slot_lock);
		if (!clean)
			return;
		if (!dma_iova_try_alloc(c->dev, &c->iova, 0,
					(size_t)c->nslots << c->quantum_shift))
			return;
		spin_lock(&c->slot_lock);
		bitmap_zero(c->slots, c->nslots);
		bitmap_zero(c->dirty, c->nslots);
		c->nr_dirty = 0;
		c->nr_used = 0;
		c->gc_rescan = false;
		c->slot_hint = 0;
		spin_unlock(&c->slot_lock);
	}
	atomic_set(&c->gc_busy, 0);
	WRITE_ONCE(c->gone, false);
	WRITE_ONCE(c->dead, false);
	dev_info(c->dev, "io_uring DMA: mapping cache revived\n");
}

/*
 * Create the device's cache from a context that may sleep, so its
 * reservation and bitmap are allocated where they can be. Ring setup
 * calls this for every channel in the ring's stripe set, and a device
 * whose cache is live needs nothing more. A device whose bus cannot be
 * watched runs uncached, since its cache could not be retired before its
 * domain can go. So does a device that finds the registry full, which is
 * turned away before the device lock is taken and a cache is built for it
 * only to be freed. A cache is created, or revived, only while the binding
 * that registered @chan is in place. That binding holds the device's
 * default domain, which a driver bound since may leave free to change,
 * as vfio-pci does, and the ring's channel may come from a binding that
 * has ended since. An unbind of idxd orphans every channel its clients
 * hold inside remove(), under the device lock, so under that lock a
 * channel still on its device's list proves the binding, and a later
 * unbind retires the cache again. A signal ends the wait for the lock,
 * as does the task_work that completions on the caller's other rings
 * queue, unless those rings use DEFER_TASKRUN or SQPOLL. A cache this
 * call would have created or revived then waits for a later ring.
 */
void io_pfn_cache_prepare(struct dma_chan *chan)
{
	struct device *dev = chan->device->dev;
	struct io_pfn_cache *c = io_pfn_cache_get(dev);

	/*
	 * Only a watched device gets a cache. A binding that began after
	 * the cache's last retire registered its channels after that
	 * retire, and a ring acquires a channel only once it is registered,
	 * so a stale read here can only miss the retire that ends the
	 * binding of @chan, and the test below would skip that channel too.
	 */
	if (c && !READ_ONCE(c->gone))
		return;
	if (!c && io_pfn_cache_full(dev))
		return;
	/*
	 * A retired cache whose quantum does not fit the domain stays
	 * retired until the group's type changes back, since the revive
	 * refuses it, so it is not worth the device lock, which an FLR or a
	 * reset holds for up to seconds. While the binding of @chan is in
	 * place the domain cannot be replaced, so this read is exact, and
	 * the test below skips any other channel anyway.
	 */
	if (c && c->quantum != io_dma_map_quantum(dev))
		return;
	if (!io_pfn_cache_watch(dev))
		return;
	if (device_lock_interruptible(dev))
		return;
	if (dev->driver && !dma_chan_orphaned(chan)) {
		c = __io_pfn_cache_get(dev, GFP_KERNEL);
		if (c) {
			mutex_lock(&c->life_lock);
			if (READ_ONCE(c->gone))
				io_pfn_cache_revive(c);
			mutex_unlock(&c->life_lock);
		}
	}
	device_unlock(dev);
}

/*
 * Retired slots stay linked until this collector unlinks them, in runs
 * of adjacent slots with one IOTLB flush per run. dma_iova_unlink()
 * cannot queue its flush the way a plain unmap on a flush-queue domain
 * does, so unlinking every retirement on its own cost the thrashing
 * cache a third of its throughput. Slots are handed out sequentially
 * and the sweep retires them in ring order, so runs are long.
 */
#define IO_PFN_GC_MIN_DIRTY	64
#define IO_PFN_GC_RUNS		16
#define IO_PFN_GC_RUN_MAX	1024	/* slots per unlink: 128 MB */

/*
 * min_dirty of a caller that needs every retired slot unlinked and has no
 * later collector to count on: a put after the cap was cleared, which no
 * lookup will follow, and the end of the age walk (IO_PFN_GC_WALK). Such
 * a caller that finds another collector at work leaves it a request
 * instead of returning with its slots still linked. The walk, which runs
 * in process context, waits for the collector instead.
 */
#define IO_PFN_GC_ALL		0
/*
 * min_dirty of the age walk's final collect: an IO_PFN_GC_ALL caller in
 * process context. The slots the walk retired can take thousands of
 * passes to unlink, so it yields between them, and it waits for a
 * collector at work rather than leave them to it.
 */
#define IO_PFN_GC_WALK		UINT_MAX
/*
 * min_dirty of an insert that found the ring full: a threshold of one
 * whose first pass does not stand aside for an age walk. Until the walk
 * has collected, nothing else would free a slot, and every insert would
 * map outside the ring. The pass-end test stops it after that pass while
 * the walk runs, and an insert that finds another collector at work maps
 * outside the ring meanwhile.
 */
#define IO_PFN_GC_SLOT		(UINT_MAX - 1)

static void __io_pfn_slots_gc(struct io_pfn_cache *c, unsigned int min_dirty)
{
	struct { unsigned int start, len; } runs[IO_PFN_GC_RUNS];
	bool walk = min_dirty == IO_PFN_GC_WALK;
	bool all = walk || min_dirty == IO_PFN_GC_ALL;
	bool for_slot = min_dirty == IO_PFN_GC_SLOT;
	unsigned int n, i, pos;
	bool more;

	if (all || for_slot)
		min_dirty = 1;
	/*
	 * The age walk retires in PFN order, which scatters its slots over
	 * the ring until it is done. A collector alongside it would keep
	 * finding full passes of short runs and unlink them at an IOTLB sync
	 * each for as long as the walk lasts, on a submitter. So until the
	 * walk has retired and collected, only a caller that must see its
	 * slots go collects, and a datapath collector already at work stops
	 * after its current pass unless it takes such a caller's request.
	 * A flush's walk and its retry loop stand aside too, and the loop
	 * waits for the walk's collect. The walk collects at its end, from
	 * process context, once its slots have joined into what runs they
	 * can. An insert that found the ring full still collects one pass.
	 */
	if (!all && !for_slot && READ_ONCE(c->age_walking))
		return;
	while (atomic_cmpxchg(&c->gc_busy, 0, 1)) {
		bool busy, done;

		if (!all)
			return;		/* another collector is at it */
		if (walk) {
			/*
			 * A request would hand the walk's scattered slots to
			 * the collector at work, which may be a submitter, so
			 * the walk waits for it instead. Once the cache is
			 * dead, kill holds the collector for good and unlinks
			 * every slot itself.
			 */
			if (READ_ONCE(c->dead))
				return;
			cond_resched();
			continue;
		}
		/*
		 * The collector at work may already be past this caller's
		 * slots. It lets go only under slot_lock, after it has
		 * taken any request left there, so either it collects them
		 * or this caller finds it gone and collects them itself.
		 * A holder found gone with nothing dirty has collected them.
		 */
		spin_lock(&c->slot_lock);
		busy = atomic_read(&c->gc_busy);
		if (busy)
			c->gc_rescan = true;
		done = busy || !c->nr_dirty;
		spin_unlock(&c->slot_lock);
		if (done)
			return;
	}
	do {
		n = 0;
		pos = 0;
		spin_lock(&c->slot_lock);
		while (n < IO_PFN_GC_RUNS) {
			unsigned int start, end;

			start = find_next_bit(c->dirty, c->nslots, pos);
			if (start >= c->nslots)
				break;
			end = find_next_zero_bit(c->dirty, c->nslots, start);
			if (end - start > IO_PFN_GC_RUN_MAX)
				end = start + IO_PFN_GC_RUN_MAX;
			bitmap_clear(c->dirty, start, end - start);
			c->nr_dirty -= end - start;
			runs[n].start = start;
			runs[n].len = end - start;
			n++;
			pos = end;
		}
		spin_unlock(&c->slot_lock);
		for (i = 0; i < n; i++) {
			dma_iova_unlink(c->dev, &c->iova,
					(size_t)runs[i].start << c->quantum_shift,
					(size_t)runs[i].len << c->quantum_shift,
					DMA_BIDIRECTIONAL, DMA_ATTR_SKIP_CPU_SYNC);
			atomic64_inc(&c->unlink_runs);
			atomic64_add(runs[i].len, &c->unlink_slots);
		}
		/* Only now are the slots free to hand out again. */
		spin_lock(&c->slot_lock);
		for (i = 0; i < n; i++) {
			bitmap_clear(c->slots, runs[i].start, runs[i].len);
			c->nr_used -= runs[i].len;
		}
		if (c->gc_rescan) {
			/*
			 * A caller retired slots this pass may have missed
			 * and counts on this collector to unlink them, so from
			 * here on it collects as that caller would have.
			 */
			c->gc_rescan = false;
			all = true;
			min_dirty = 1;
			more = c->nr_dirty != 0;
		} else {
			more = n == IO_PFN_GC_RUNS &&
			       c->nr_dirty >= min_dirty &&
			       (all || !READ_ONCE(c->age_walking));
		}
		if (!more)
			atomic_set(&c->gc_busy, 0);
		spin_unlock(&c->slot_lock);
		if (more && walk)
			cond_resched();
	} while (more);
}

/*
 * Every lookup calls the collector, so the test that it has nothing to do
 * stays inline in the caller.
 */
static __always_inline void io_pfn_slots_gc(struct io_pfn_cache *c,
					    unsigned int min_dirty)
{
	bool one = min_dirty == IO_PFN_GC_ALL || min_dirty == IO_PFN_GC_WALK ||
		   min_dirty == IO_PFN_GC_SLOT;

	if (c->nslots && READ_ONCE(c->nr_dirty) >= (one ? 1 : min_dirty))
		__io_pfn_slots_gc(c, min_dirty);
}

static int io_pfn_slot_get(struct io_pfn_cache *c)
{
	unsigned int slot;
	bool collected = false;

	if (!c->nslots)
		return -1;
again:
	/*
	 * A full ring fails on the count, without taking the lock. An age
	 * walk keeps the ring full until a collect frees its slots, and
	 * inserts that scanned the whole bitmap under slot_lock to find that
	 * out kept the walk's own retirements, which take the lock too,
	 * queued behind them. The count is tested again under the lock, since
	 * inserts that passed the first test together race for the last free
	 * slots.
	 */
	if (READ_ONCE(c->nr_used) >= c->nslots)
		goto full;
	spin_lock(&c->slot_lock);
	slot = c->nslots;
	if (c->nr_used < c->nslots) {
		slot = find_next_zero_bit(c->slots, c->nslots, c->slot_hint);
		if (slot >= c->nslots)
			slot = find_first_zero_bit(c->slots, c->nslots);
	}
	if (slot >= c->nslots) {
		spin_unlock(&c->slot_lock);
		goto full;
	}
	__set_bit(slot, c->slots);
	c->nr_used++;
	/* Sequential handout keeps entries inserted together adjacent in
	 * the range: IOMMU page-table locality, and the runs the
	 * collector unlinks.
	 */
	c->slot_hint = slot + 1 < c->nslots ? slot + 1 : 0;
	spin_unlock(&c->slot_lock);
	return slot;
full:
	if (!collected) {
		collected = true;
		io_pfn_slots_gc(c, IO_PFN_GC_SLOT);
		goto again;
	}
	return -1;
}

/* A slot whose link failed: nothing to unlink, free it outright. */
static void io_pfn_slot_put(struct io_pfn_cache *c, int slot)
{
	spin_lock(&c->slot_lock);
	__clear_bit(slot, c->slots);
	c->nr_used--;
	spin_unlock(&c->slot_lock);
}

/* A retired entry's slot: mark it for the collector. */
static void io_pfn_slot_retire(struct io_pfn_cache *c, int slot)
{
	spin_lock(&c->slot_lock);
	WRITE_ONCE(c->by_slot[slot], NULL);
	__set_bit(slot, c->dirty);
	c->nr_dirty++;
	spin_unlock(&c->slot_lock);
}

/* Map a segment for a new entry: into the reserved range when a slot is
 * free, else a plain per-entry mapping. Returns the device address or 0.
 */
static dma_addr_t io_pfn_map_segment(struct io_pfn_cache *c,
				     struct io_pfn_map *pm,
				     struct folio *folio, size_t seg_base,
				     size_t seg_len)
{
	dma_addr_t base;
	int slot;

	/*
	 * io_pfn_cache_kill() marks a cache whose binding or device is going
	 * away and waits a grace period before it touches the IOVA space,
	 * so a mapping made here either completes first or sees the mark.
	 * Every call below is safe in atomic context.
	 */
	rcu_read_lock();
	if (unlikely(READ_ONCE(c->dead))) {
		rcu_read_unlock();
		return 0;
	}
	slot = io_pfn_slot_get(c);
	if (slot >= 0) {
		size_t off = (size_t)slot << c->quantum_shift;
		int err;

		err = dma_iova_link(c->dev, &c->iova,
				    PFN_PHYS(folio_pfn(folio)) + seg_base, off,
				    seg_len, DMA_BIDIRECTIONAL,
				    DMA_ATTR_SKIP_CPU_SYNC);
		if (!err) {
			err = dma_iova_sync(c->dev, &c->iova, off, seg_len);
			if (err)
				dma_iova_unlink(c->dev, &c->iova, off, seg_len,
						DMA_BIDIRECTIONAL,
						DMA_ATTR_SKIP_CPU_SYNC);
		}
		if (!err) {
			pm->slot = slot;
			WRITE_ONCE(c->by_slot[slot], pm);
			atomic64_inc(&c->slot_links);
			rcu_read_unlock();
			return c->iova.addr + off;
		}
		io_pfn_slot_put(c, slot);
	}
	if (c->nslots)
		atomic64_inc(&c->slot_fails);
	pm->slot = -1;
	base = dma_map_page_attrs(c->dev, folio_page(folio, 0), seg_base,
				  seg_len, DMA_BIDIRECTIONAL,
				  DMA_ATTR_SKIP_CPU_SYNC);
	rcu_read_unlock();
	if (dma_mapping_error(c->dev, base))
		return 0;
	return base;
}

/*
 * The collect of a slot-backed put that found the cap cleared. While the
 * clear's first round of flushes runs, the put leaves its slot to the
 * clear, which flushes every cache again after that round. The flush's
 * walk can retire slots scattered over the ring, and a collector that
 * took them here would follow the walk on the put's datapath.
 */
static void io_pfn_slots_gc_cleared(struct io_pfn_cache *c)
{
	/* Pairs with the barrier before the clear's store. */
	smp_rmb();
	if (atomic_read(&io_pfn_cache_clearing))
		return;
	io_pfn_slots_gc(c, IO_PFN_GC_ALL);
}

/*
 * @collect is false when the caller collects after the put: the walk of
 * a flush or of the age retirement.
 */
static void io_pfn_map_unmap(struct io_pfn_map *pm, bool collect)
{
	struct io_pfn_cache *c = pm->cache;

	if (pm->slot >= 0) {
		io_pfn_slot_retire(c, pm->slot);
		/*
		 * A walk that collects after its puts has them only mark
		 * their slots. Each collect scans the whole dirty bitmap and
		 * unlinks with an IOTLB sync of its own, which a flush at a
		 * cleared cap would otherwise pay once for every entry.
		 */
		if (!collect)
			return;
		/*
		 * With the cache switched off no lookup runs, so no sweep
		 * would ever collect this slot. A put after the cap was
		 * cleared therefore collects it, or leaves it to the collector
		 * at work or to the clear. The barrier pairs with those in
		 * io_pfn_cache_cap_set(), so either this put sees the cleared
		 * cap and the clear over, or a flush of the clear finds the
		 * slot dirty.
		 */
		smp_mb();
		if (unlikely(!READ_ONCE(io_dma_pfn_cache_cap_mb)))
			io_pfn_slots_gc_cleared(c);
	} else {
		dma_unmap_page_attrs(pm->dev, pm->dma_base, pm->size,
				     DMA_BIDIRECTIONAL, DMA_ATTR_SKIP_CPU_SYNC);
	}
}

/*
 * Drop one reference. The last dropper unmaps and frees, and collects
 * once the cap is cleared unless @collect is false (io_pfn_map_unmap()).
 */
static void __io_pfn_map_put(struct io_pfn_map *pm, bool collect)
{
	if (!atomic_dec_and_test(&pm->refs))
		return;
	io_pfn_map_unmap(pm, collect);
	kfree_rcu(pm, rcu);
}

/* Drop one reference. The last dropper unmaps and frees. */
static void io_pfn_map_put(struct io_pfn_map *pm)
{
	__io_pfn_map_put(pm, true);
}

/*
 * Displace an entry from the cache so its mapping dies with its last
 * reference. The wedge path uses this on cached write destinations,
 * whose standing mapping a stalled device may still write through, so
 * no later I/O is handed it. That path keeps its own reference, so the
 * mapping and its slot stay with the folio it leaks.
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
 * Wait for a pending insert of @pfn. Returns the entry with a reference
 * taken, or NULL when the claim was dropped or the wait ran out.
 */
static struct io_pfn_map *io_pfn_map_wait(struct io_pfn_cache *c,
					  unsigned long pfn)
{
	u64 end = ktime_get_ns() +
		  (u64)READ_ONCE(io_dma_pfn_wait_us) * NSEC_PER_USEC;
	struct io_pfn_map *pm;
	void *e;

	do {
		cpu_relax();
		rcu_read_lock();
		e = xa_load(&c->xa, pfn);
		if (e && !xa_is_value(e)) {
			pm = e;
			if (atomic_inc_not_zero(&pm->refs)) {
				rcu_read_unlock();
				return pm;
			}
		}
		rcu_read_unlock();
		if (!e)
			return NULL;	/* the claimant gave up */
	} while (ktime_get_ns() < end);
	atomic64_inc(&c->wait_timeouts);
	return NULL;
}

/*
 * Finish a hit on @pm, which the caller holds a reference to. Every
 * cached entry covers a whole quantum and the caller has already bounded
 * the span to one, so a span past the entry cannot happen; the check
 * guards that invariant rather than a case. Every hit ends here, so it
 * stays inline in the lookup.
 */
static __always_inline struct io_pfn_map *
io_pfn_map_hit(struct io_pfn_cache *c, struct io_pfn_map *pm, size_t rel,
	       size_t len, dma_addr_t *dma, atomic64_t *counter)
{
	if (WARN_ON_ONCE(rel + len > pm->size)) {
		atomic64_inc(&c->range_fallbacks);
		io_pfn_map_put(pm);
		return NULL;
	}
	WRITE_ONCE(pm->referenced, true);
	WRITE_ONCE(pm->last_used, jiffies);
	atomic64_inc(counter);
	*dma = pm->dma_base + rel;
	return pm;
}

/* A ghost's value is its eviction time in jiffies, cut to fit a value. */
#define IO_PFN_GHOST_TIME_MASK	(LONG_MAX >> 1)

/*
 * A ghost older than twice max_age, or two minutes when aging is off,
 * can no longer say anything about the present working set.
 */
static bool io_pfn_ghost_stale(void *gv, unsigned long max_age)
{
	unsigned long age = (jiffies - xa_to_value(gv)) &
			    IO_PFN_GHOST_TIME_MASK;

	return age > 2 * msecs_to_jiffies(max_age ? max_age : 60000);
}

/*
 * A ghost's index is its segment-head PFN scaled down by the quantum.
 * Every cached segment is a whole quantum of a folio at least that large,
 * and folios are naturally aligned, so the scaling loses nothing, and an
 * xarray leaf holds up to 64 ghosts instead of two. Were two segments to
 * share an index, they would only share a sizing hint.
 */
static unsigned long io_pfn_ghost_index(struct io_pfn_cache *c,
					unsigned long pfn)
{
	unsigned int shift = c->quantum_shift > PAGE_SHIFT ?
			     c->quantum_shift - PAGE_SHIFT : 0;

	return pfn >> shift;
}

/*
 * Remember the key of an entry that cap pressure took. The list holds no
 * more keys than the cache could hold entries, under its hard cap and
 * its reservation. A key that more evictions than that have followed
 * would not have stayed even in a cache of that size, so it cannot show
 * the target too small, and without the bound a list that only the purge
 * trims grows with every segment the cache ever evicts. At the bound the
 * new ghost replaces the first one at or past the purge hand, so the list
 * follows the present evictions instead of keeping the first ones. That
 * is random in time only while evictions come in no PFN order. Evictions
 * in ascending PFN order can bring the hand up behind the newest ghosts,
 * which it then erases as they come, so a pass of a sequential re-read
 * can lose most of its ghosts before the next pass reaches them. Each
 * purge moves the hand past the fresh ghosts it visits, which keeps those
 * until the hand comes round again. A list that a lowered cap left over
 * the bound loses two ghosts for every one added, so it shrinks back with
 * the evictions instead of staying as long as they keep its ghosts fresh.
 * Runs under c->lock, which the hand needs.
 */
static void io_pfn_ghost_add(struct io_pfn_cache *c, unsigned long pfn)
{
	u64 max = ((u64)READ_ONCE(io_dma_pfn_cache_cap_mb) << 20) >>
		  c->quantum_shift;
	unsigned int i;
	void *old;

	lockdep_assert_held(&c->lock);
	if (c->nslots)
		max = min_t(u64, max, c->nslots);
	if (!max)
		return;
	for (i = 0; i < 2 && atomic64_read(&c->ghost_count) >= (s64)max; i++) {
		unsigned long gidx = c->ghost_hand;

		old = xa_find(&c->ghost, &gidx, ULONG_MAX, XA_PRESENT);
		if (!old) {
			gidx = 0;
			old = xa_find(&c->ghost, &gidx, ULONG_MAX, XA_PRESENT);
		}
		if (!old)
			break;
		if (xa_erase(&c->ghost, gidx))
			atomic64_dec(&c->ghost_count);
		c->ghost_hand = gidx + 1;
	}
	/*
	 * Counted before the store, so that a hit or a flush that takes the
	 * new ghost out at once cannot drive the count below the list, and
	 * taken back when the store added no key: a key ghosted again unhit,
	 * or a failed allocation.
	 */
	atomic64_inc(&c->ghost_count);
	old = xa_store(&c->ghost, io_pfn_ghost_index(c, pfn),
		       xa_mk_value(jiffies & IO_PFN_GHOST_TIME_MASK),
		       GFP_NOWAIT | __GFP_NOWARN);
	if (old)
		atomic64_dec(&c->ghost_count);
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
 * latency.  A lookup also starts the age walk, which runs from a work
 * item.  A cache that sees no lookups at all does not age.  There is
 * no timer: the CLOCK sweep rides the datapath and the age walk is
 * started from it.
 */
/*
 * Retire entries, under two policies.
 *
 * Over the cap, the CLOCK hand advances from where it stopped and
 * gives referenced entries a second chance, which is what keeps a
 * streaming pattern from flushing a recycling working set.
 *
 * Independently of the cap, the age walk retires every entry that no
 * I/O has touched for max_age. Age takes precedence over the second
 * chance, since an idle entry's reference bit only records that it was
 * used at some point in the past, not recently.
 */
/*
 * Take @pm out of the cache and drop its bias. Identity-checked: the
 * flush, the age walk and the write wedge's displacement remove entries
 * without the cache lock, so between a walk handing us @pm and the
 * removal another CPU can take it out, drop the bias and insert a
 * replacement under the same key. Only the remover that actually took an
 * entry out may retire it. Returns false when someone else got there
 * first. An aged retirement needs no lock. A capped one runs under
 * c->lock, which its ghost needs.
 */
static bool io_pfn_cache_retire(struct io_pfn_cache *c, struct io_pfn_map *pm,
				bool aged)
{
	if (xa_cmpxchg(&c->xa, pm->pfn, pm, NULL,
		       GFP_NOWAIT | __GFP_NOWARN) != pm)
		return false;
	atomic64_sub(pm->size, &c->covered);
	atomic64_inc(&c->evictions);
	if (aged) {
		atomic64_inc(&c->age_evictions);
	} else if (READ_ONCE(io_dma_pfn_cache_auto)) {
		/*
		 * Cap pressure took a live-looking entry. Remember its key
		 * so a near-term re-read can prove the target too small.
		 * Values carry the eviction time, for the purge and for the
		 * staleness test of a hit.
		 */
		io_pfn_ghost_add(c, pm->pfn);
	}
	/*
	 * Drop the cache bias. An aged retirement comes from the age walk,
	 * which collects what it retired, so its put leaves the slot to it.
	 */
	__io_pfn_map_put(pm, !aged);
	return true;
}

/*
 * The cap walk over the reserved range: the CLOCK hand goes round the
 * slot ring in insertion order, which is the canonical CLOCK and what
 * makes retirements adjacent for the collector. Called under c->lock
 * with RCU held. Returns with *over updated.
 */
static void io_pfn_cache_sweep_ring(struct io_pfn_cache *c, u64 target,
				    bool *over, int *budget, int *unmaps)
{
	unsigned int slot = c->slot_hand;
	bool wrapped = false;

	while (*over) {
		struct io_pfn_map *pm;

		slot = find_next_bit(c->slots, c->nslots, slot);
		if (slot >= c->nslots) {
			if (wrapped)
				break;
			wrapped = true;
			slot = 0;
			continue;
		}
		pm = READ_ONCE(c->by_slot[slot]);
		slot++;
		/*
		 * A retired slot counts against the budget too. The age walk
		 * leaves what it retired uncollected until its collect ends,
		 * and a sweep that passed over that for free could cross the
		 * whole stretch the walk had retired, on a submitter.
		 */
		if (--*budget <= 0)
			break;
		if (!pm)
			continue;	/* retired, or still linking */
		if (atomic_read(&pm->refs) > 1) {
			atomic64_inc(&c->ref_skips);
			continue;
		}
		if (READ_ONCE(pm->referenced)) {
			WRITE_ONCE(pm->referenced, false);
			continue;
		}
		if (!io_pfn_cache_retire(c, pm, false))
			continue;
		*over = atomic64_read(&c->covered) >
			(s64)(target + ((u64)c->quantum << 2));
		if (--*unmaps <= 0)
			break;
	}
	c->slot_hand = slot < c->nslots ? slot : 0;
}

/*
 * The adaptive tick. Decay the target when a whole tick has passed
 * without a single ghost hit: nothing the cap evicted was missed, so the
 * cache can stand to be smaller. Growth happens on the miss path. Runs
 * under c->lock. The evictions it counts are the cap's alone: the age walk
 * retires idle entries in bursts, which say nothing about the pattern. An
 * aged retirement counts in both counters, so those between the two reads
 * can put the difference off either way, even below zero. The difference
 * and its change since the last tick are both clamped at zero, which keeps
 * every tick sane. A lookup that waited for another CPU's insert counts as
 * a hit, since the entry served it: a convoy's followers are reuse.
 * Without them four jobs reading a large file once would read as
 * streaming, and two re-reading a set larger than the ceiling as a cyclic
 * set. The tick also recounts the caches the budget is split across.
 */
static void io_pfn_cache_adapt_tick(struct io_pfn_cache *c, u64 cap)
{
	u64 gh = atomic64_read(&c->ghost_hits);
	u64 h = atomic64_read(&c->hits) + atomic64_read(&c->waits);
	u64 m = atomic64_read(&c->misses);
	u64 ins = atomic64_read(&c->inserts);
	u64 ev = max_t(s64, atomic64_read(&c->evictions) -
			    atomic64_read(&c->age_evictions), 0);
	u64 dh = h - c->hits_snap;
	u64 dm = m - c->misses_snap;
	u64 di = ins - c->inserts_snap;
	u64 de = ev > c->evictions_snap ? ev - c->evictions_snap : 0;
	u64 floor, eff;

	WRITE_ONCE(c->nr_sharing, io_pfn_cache_sharing(c));
	floor = io_pfn_cache_floor(io_pfn_cache_ceiling(c, cap));
	eff = io_pfn_cache_target(c, cap);

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
			/* One dead probe tick re-parks. */
			c->dead_ticks = 1;
			WRITE_ONCE(c->eff_cap, floor);
		}
	} else {
		u32 ratio = (dh + dm) ?
			(u32)div64_u64(dh * 100,
				       dh + dm) : 100;
		bool ghosting =
			gh != c->ghost_hits_snap;
		u64 turned = c->cyclic_turnover + de;

		/* Only a row of cyclic ticks adds up. */
		c->cyclic_turnover = 0;
		if (ghosting && ratio < IO_PFN_CYCLIC_MAX_RATIO &&
		    ratio <= c->prev_ratio + 2 &&
		    eff >= io_pfn_cache_ceiling(c, cap) &&
		    atomic64_read(&c->covered) >= (s64)eff && de) {
			/*
			 * Gate only once growth is
			 * exhausted: the target is at
			 * the ceiling, and the cache
			 * has filled it and evicts.
			 * Before that a low, flat
			 * ratio is what mid-fill
			 * looks like for a set that
			 * will fit, and growth
			 * resolves it. The target
			 * alone does not say, since
			 * ghost hits carry it to the
			 * ceiling long before the
			 * cache fills it.
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
			 * is spared. Park, not halve:
			 * a smaller cache misses such
			 * a set as often, and a halved
			 * target would only invite
			 * the next ghost hit to grow
			 * it back to the ceiling.
			 * Park only once a row of such
			 * ticks has evicted more than
			 * the ceiling holds. A switch
			 * between two phases that each
			 * fit evicts no more than that
			 * before the new phase hits,
			 * and a set that will never
			 * fit gets there within one
			 * turnover.
			 */
			if (turned <= eff >> c->quantum_shift) {
				c->cyclic_turnover =
					min_t(u64, turned, U32_MAX);
				c->dead_ticks = 0;
			} else {
				WRITE_ONCE(c->eff_cap,
					   IO_PFN_EFF_PARKED);
				c->parked_ticks = 0;
			}
		} else if (!ghosting &&
			   di >= IO_PFN_DEAD_MIN_INSERTS &&
			   de * 2 >= di &&
			   ratio < IO_PFN_DEAD_MAX_RATIO &&
			   ++c->dead_ticks >= 2) {
			/*
			 * Streaming: the tick retired
			 * about what it inserted and
			 * missed none of it. The cache
			 * only taxes such a pattern
			 * with an insert, an eviction
			 * and a flush share per touch.
			 */
			WRITE_ONCE(c->eff_cap,
				   IO_PFN_EFF_PARKED);
			c->parked_ticks = 0;
			c->dead_ticks = 0;
		} else if (!ghosting) {
			if (di < IO_PFN_DEAD_MIN_INSERTS ||
			    de * 2 < di ||
			    ratio >= IO_PFN_DEAD_MAX_RATIO)
				c->dead_ticks = 0;
			eff -= eff >>
			    IO_PFN_ADAPT_DECAY_SHIFT;
			WRITE_ONCE(c->eff_cap,
				   max(eff, floor));
		} else {
			c->dead_ticks = 0;
		}
		c->prev_ratio = ratio;
	}
	c->ghost_hits_snap = gh;
	c->hits_snap = h;
	c->misses_snap = m;
	c->inserts_snap = ins;
	c->evictions_snap = ev;
	c->next_adapt = jiffies + IO_PFN_ADAPT_TICK;
}

/*
 * Purge stale ghosts (io_pfn_ghost_stale()). Bounded walk from a hand.
 * Runs under c->lock. A ghost hit may take a ghost out concurrently, so
 * only the eraser that got it counts it.
 */
static void io_pfn_cache_purge_ghosts(struct io_pfn_cache *c,
				      unsigned long max_age)
{
	unsigned long gidx = 0;
	void *gv;
	int gbudget = IO_PFN_EVICT_BUDGET;

	xa_for_each_start(&c->ghost, gidx, gv,
			  c->ghost_hand) {
		if (--gbudget <= 0)
			break;
		if (io_pfn_ghost_stale(gv, max_age) &&
		    xa_erase(&c->ghost, gidx))
			atomic64_dec(&c->ghost_count);
	}
	c->ghost_hand = gbudget <= 0 ? gidx + 1 : 0;
}

/*
 * How long until the next age walk. Each walk visits every entry, so a
 * larger cache is walked less often, at about IO_PFN_AGE_VISITS entries a
 * second, but at least once per quarter of max_age and at most once a
 * second. An idle entry is therefore unmapped no later than a quarter of
 * max_age after it ages, or a second when that is longer, plus the walk
 * itself and the wait for the lookup that starts it.
 */
#define IO_PFN_AGE_VISITS	4096

static unsigned long io_pfn_cache_age_period(struct io_pfn_cache *c,
					     unsigned long max_age)
{
	u64 n = max_t(s64, atomic64_read(&c->covered), 0) >> c->quantum_shift;
	unsigned long most = max_t(unsigned long,
				   msecs_to_jiffies(max_age) / 4, HZ);

	return clamp_t(unsigned long, div_u64(n * HZ, IO_PFN_AGE_VISITS),
		       HZ, most);
}

/*
 * The age walk. A lookup starts it once per period and it runs here, off
 * the datapath, so that it can visit every entry and retire every idle
 * one. A walk on the datapath, under the cache lock, has to stop after a
 * bounded number of visits and retirements, and a large idle population
 * would leave at a few dozen entries a second. Retirement is
 * identity-checked, so the walk needs no cache lock against the CLOCK
 * sweep or a flush. Entries are freed with kfree_rcu(), so each batch of
 * visits holds the read lock, and the walk yields between batches.
 */
static void io_pfn_cache_age_work(struct work_struct *work)
{
	struct io_pfn_cache *c = container_of(work, struct io_pfn_cache,
					      age_work);
	unsigned long max_age = READ_ONCE(io_dma_pfn_cache_max_age_ms);
	unsigned int visits = 0, retired = 0;
	unsigned long index, age_before;
	struct io_pfn_map *pm;
	bool walking = false;

	if (max_age) {
		unsigned long next = READ_ONCE(c->next_age);
		unsigned long due = jiffies +
				    io_pfn_cache_age_period(c, max_age);

		/*
		 * A write of pfn_cache_max_age_ms in the jiffy a walk falls
		 * due re-arms next_age with the value it already had, so a
		 * lookup that read the old age can still win its exchange and
		 * arm the next walk a quarter of the old age out. Such a
		 * lookup has queued this walk, which reads the new age, so the
		 * walk pulls the next start in to its own period.
		 */
		if (time_after(next, due))
			cmpxchg(&c->next_age, next, due);
		age_before = jiffies - msecs_to_jiffies(max_age);
		rcu_read_lock();
		xa_for_each(&c->xa, index, pm) {
			bool aged;

			/*
			 * io_pfn_cache_kill() waits a grace period after the
			 * mark, so a retirement here, and the unmap that a
			 * plain-mapped one does, finishes before it touches
			 * the device's IOVA space. A cleared cap is followed
			 * by a flush that revokes every entry, so the walk
			 * leaves the rest to it.
			 */
			if (READ_ONCE(c->dead) ||
			    !READ_ONCE(io_dma_pfn_cache_cap_mb))
				break;
			if (xa_is_value(pm))
				continue;	/* an insert's claim */
			/*
			 * Age first, so that ref_skips counts only an aged
			 * entry that an I/O still holds, which is rare, and
			 * not every in-flight entry the walk passes.
			 */
			aged = time_before(READ_ONCE(pm->last_used),
					   age_before);
			if (aged && atomic_read(&pm->refs) > 1) {
				atomic64_inc(&c->ref_skips);
			} else if (aged) {
				/* Datapath collectors stand aside from here. */
				if (!walking) {
					WRITE_ONCE(c->age_walking, true);
					walking = true;
				}
				if (io_pfn_cache_retire(c, pm, true))
					retired++;
			}
			if (++visits >= IO_PFN_EVICT_BUDGET ||
			    retired >= IO_PFN_EVICT_UNMAP_MAX) {
				visits = 0;
				retired = 0;
				rcu_read_unlock();
				cond_resched();
				rcu_read_lock();
			}
		}
		rcu_read_unlock();
	}
	/*
	 * Retirement only marks a slot, and the datapath collector waits
	 * for a run of them. An idle cache would otherwise leave its last
	 * retired slots linked indefinitely, so the walk collects whatever
	 * it retired. It does so before it lets the datapath collectors
	 * back in, since one of them would otherwise take the whole
	 * collect, at an IOTLB sync per short run, on a submitter. A walk
	 * that retired nothing owes nothing and has kept no collector out,
	 * so it only collects what the datapath left under its threshold,
	 * and not while another collector is at work.
	 */
	io_pfn_slots_gc(c, walking ? IO_PFN_GC_WALK : 1);
	if (walking)
		WRITE_ONCE(c->age_walking, false);
}

static void io_pfn_cache_evict(struct io_pfn_cache *c, u64 cap)
{
	unsigned long next_age = READ_ONCE(c->next_age);
	unsigned long max_age;
	u64 target = io_pfn_cache_target(c, cap);
	struct io_pfn_map *pm;
	unsigned long index;
	int budget = IO_PFN_EVICT_BUDGET;
	int unmaps = IO_PFN_EVICT_UNMAP_MAX;
	bool over, aging, adapt_due, xa_over;
	int pass;

	/*
	 * Pairs with the barrier in io_pfn_cache_max_age_set(): a lookup
	 * that finds the walk re-armed also finds the age it was re-armed
	 * for, so it cannot arm the next walk a quarter of the old age out.
	 */
	smp_rmb();
	max_age = READ_ONCE(io_dma_pfn_cache_max_age_ms);

	io_pfn_slots_gc(c, IO_PFN_GC_MIN_DIRTY);

	/* A dead-band of a few segments over the target parks the sweep
	 * when a fitting working set sits at its converged size; without
	 * it every insert at the boundary evicts one entry and the sweep
	 * stays hot on the datapath.
	 */
	over = atomic64_read(&c->covered) >
		(s64)(target + ((u64)c->quantum << 2));
	aging = max_age && time_after(jiffies, next_age);
	/* The adaptive tick is independent of aging, so a cache with
	 * max_age_ms at 0 still decays, parks and unparks.
	 */
	adapt_due = READ_ONCE(io_dma_pfn_cache_auto) &&
		    time_after(jiffies, READ_ONCE(c->next_adapt));
	if (!over && !aging && !adapt_due)
		return;

	/*
	 * The age walk visits every entry, which no lookup can afford
	 * under the cache lock, so the lookup only starts it. The exchange
	 * picks one lookup per period.
	 */
	if (aging) {
		unsigned long due = jiffies +
				    io_pfn_cache_age_period(c, max_age);

		if (cmpxchg(&c->next_age, next_age, due) == next_age)
			queue_work(system_dfl_long_wq, &c->age_work);
	}

	if (!spin_trylock(&c->lock))
		return;

	/*
	 * The entries this walk examines are freed with kfree_rcu(), and
	 * the xarray iterators hand them back outside any read-side
	 * section. Under preemptible RCU c->lock is not one, so a grace
	 * period elapsing mid-sweep would leave it reading freed entries.
	 */
	rcu_read_lock();

	if (adapt_due || (aging && READ_ONCE(io_dma_pfn_cache_auto))) {
		if (time_after(jiffies, c->next_adapt))
			io_pfn_cache_adapt_tick(c, cap);
		io_pfn_cache_purge_ghosts(c, max_age);
	}

	/*
	 * A lookup that passed the dead test before io_pfn_cache_kill() set
	 * the mark can still get here. The retire's grace period orders the
	 * mark before any read-side section that it does not wait for, so
	 * such a sweep leaves the ring, whose arrays a removed device's cache
	 * gives back (io_pfn_cache_release()).
	 */
	if (over && c->nslots && !READ_ONCE(c->dead))
		io_pfn_cache_sweep_ring(c, target, &over, &budget, &unmaps);
	/*
	 * The xarray walk carries cap pressure only for a cache without a
	 * ring: mixing the ring's insertion order with the tree's PFN order
	 * would undo the runs. A ring cache's plain-mapped remainder,
	 * entries that found the ring full, is retired by age, and by this
	 * walk while the cache is parked. When the ceiling is the whole
	 * ring, as at the defaults, a cyclic park comes at a full ring, so
	 * there is such a remainder, and left in place it would keep
	 * covered over the dead band and send every parked lookup through a
	 * ring sweep that scans the empty slot bitmap twice. The walk
	 * leaves slot-backed entries to the ring sweep.
	 */
	xa_over = over && (!c->nslots || !target);
	for (pass = 0; pass < 2 && xa_over; pass++) {
		unsigned long start = pass ? 0 : c->hand;

		if (budget <= 0 || unmaps <= 0)
			break;

		xa_for_each_start(&c->xa, index, pm, start) {
			if (xa_is_value(pm))
				continue;	/* an insert's claim */
			if (--budget <= 0) {
				c->hand = index + 1;
				goto out;
			}
			if (c->nslots && pm->slot >= 0)
				continue;	/* the ring sweep's */
			if (atomic_read(&pm->refs) > 1) {
				atomic64_inc(&c->ref_skips);
				continue;
			}
			if (READ_ONCE(pm->referenced)) {
				WRITE_ONCE(pm->referenced, false);
				continue;
			}
			if (!io_pfn_cache_retire(c, pm, false))
				continue;
			xa_over = atomic64_read(&c->covered) >
				(s64)(target + ((u64)c->quantum << 2));
			if (--unmaps <= 0 || !xa_over) {
				c->hand = index + 1;
				goto out;
			}
		}
		c->hand = 0;
	}
out:
	rcu_read_unlock();
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
 *
 * @touch says this chunk is the op's first on its segment; later
 * chunks of the same segment are the same access for the reuse-time
 * sampler and are not counted again.
 */
static struct io_pfn_map *io_pfn_map_lookup(struct io_pfn_cache *c,
					    struct folio *folio,
					    size_t offset, size_t len,
					    size_t map_len, bool touch,
					    dma_addr_t *dma)
{
	u64 cap = (u64)READ_ONCE(io_dma_pfn_cache_cap_mb) << 20;
	size_t seg_base, seg_len, rel;
	struct io_pfn_map *pm, *old;
	unsigned long pfn;
	dma_addr_t base;

	if (!c || !cap || READ_ONCE(c->dead))
		return NULL;

	/* Give the sweep a chance, and the age walk a start, on every
	 * lookup. Below the cap with nothing due this is a few reads and
	 * compares. Without this an idle entry would only ever be visited
	 * once an insert pushed the cache over the cap.
	 */
	io_pfn_cache_evict(c, cap);

	seg_base = offset & ~(c->quantum - 1);
	seg_len = min_t(size_t, c->quantum, map_len - seg_base);
	rel = offset - seg_base;
	pfn = folio_pfn(folio) + (seg_base >> PAGE_SHIFT);

	/*
	 * A segment shorter than the quantum is a small folio or a folio
	 * tail. Each would take a whole slot of the reserved range for a
	 * fraction of its bytes - small-folio files exhausted a 64 GB
	 * range with 2 GB of data - and a mapping that small saves the
	 * plain path almost nothing. Leave them to per-chunk maps, which
	 * recycle through the allocator's per-CPU cache.
	 */
	if (seg_len < c->quantum) {
		atomic64_inc(&c->short_fallbacks);
		return NULL;
	}

	/*
	 * Sample the access before the cache decides anything about
	 * it: the curve describes the stream, parked or not.
	 */
	if (touch) {
		struct io_pfn_mrc *mrc = READ_ONCE(c->mrc);
		unsigned int shift = READ_ONCE(io_dma_pfn_mrc_shift);

		if (mrc && shift)
			io_pfn_mrc_touch(mrc, pfn, shift);
	}

	/*
	 * Parked: the utility check found the pattern uncacheable, so
	 * skip straight to the caller's plain per-chunk map instead of
	 * inserting entries the sweep immediately evicts. The sweep
	 * call above keeps draining what is left and runs the adapt
	 * tick that eventually un-parks for a fresh look.
	 */
	if (!io_pfn_cache_target(c, cap))
		return NULL;
	if (unlikely(rel + len > seg_len)) {
		atomic64_inc(&c->range_fallbacks);
		return NULL;
	}

	rcu_read_lock();
	pm = xa_load(&c->xa, pfn);
	if (pm && !xa_is_value(pm) && atomic_inc_not_zero(&pm->refs)) {
		rcu_read_unlock();
		return io_pfn_map_hit(c, pm, rel, len, dma, &c->hits);
	}
	rcu_read_unlock();
	if (xa_is_value(pm)) {
		/* Another CPU is inserting this segment; take its entry. */
		pm = io_pfn_map_wait(c, pfn);
		if (!pm)
			return NULL;
		return io_pfn_map_hit(c, pm, rel, len, dma, &c->waits);
	}
	/*
	 * Claim the key before mapping. A claim already there is a
	 * concurrent insert to wait for; an entry already there is a hit.
	 */
	/*
	 * An entry the exchange hands back is only pinned once its
	 * reference is taken, and it is freed with kfree_rcu(); the
	 * read-side section covers the gap between the two.
	 */
	rcu_read_lock();
	old = xa_cmpxchg(&c->xa, pfn, NULL, IO_PFN_PENDING,
			 GFP_NOWAIT | __GFP_NOWARN);
	if (old) {
		bool got = false;

		if (!xa_is_err(old) && !xa_is_value(old))
			got = atomic_inc_not_zero(&old->refs);
		rcu_read_unlock();
		if (xa_is_err(old))
			goto fail;
		if (xa_is_value(old)) {
			pm = io_pfn_map_wait(c, pfn);
			if (!pm)
				return NULL;
			return io_pfn_map_hit(c, pm, rel, len, dma, &c->waits);
		}
		if (!got)
			goto fail;
		return io_pfn_map_hit(c, old, rel, len, dma, &c->hits);
	}
	rcu_read_unlock();
	atomic64_inc(&c->misses);

	if (READ_ONCE(io_dma_pfn_cache_auto) &&
	    xa_load(&c->ghost, io_pfn_ghost_index(c, pfn))) {
		void *gv = xa_erase(&c->ghost, io_pfn_ghost_index(c, pfn));

		if (gv)
			atomic64_dec(&c->ghost_count);
		/*
		 * A stale ghost the purge has yet to reach says nothing
		 * about the present working set. With ageing on, an entry
		 * idle that long would have aged out at any size, so the
		 * miss cannot show the target too small. With ageing off
		 * the list speaks only for the last two minutes.
		 */
		if (gv && !io_pfn_ghost_stale(gv,
				READ_ONCE(io_dma_pfn_cache_max_age_ms))) {
			u64 eff;

			/*
			 * This segment was evicted by cap pressure and is
			 * being remapped: the effective cap is thrashing
			 * the working set. Step the target up toward the
			 * hard cap, one increment per paid-for eviction.
			 */
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
			eff = min(eff, io_pfn_cache_ceiling(c, cap));
			/*
			 * A tick may have parked the cache since this lookup
			 * checked, and growing the target now would undo the
			 * park at once, with the cache still full.
			 */
			if (READ_ONCE(c->eff_cap) != IO_PFN_EFF_PARKED)
				WRITE_ONCE(c->eff_cap, eff);
		}
	}

	pm = kmalloc_obj(*pm, GFP_NOWAIT | __GFP_NOWARN);
	if (!pm)
		goto unclaim;
	/*
	 * Fill the entry before mapping it: a slot-backed mapping is
	 * published in by_slot[] as it is made, and the ring sweep reads
	 * the entry from there.
	 */
	pm->cache = c;
	pm->pfn = pfn;
	pm->size = seg_len;
	pm->dev = c->dev;
	pm->referenced = true;
	pm->last_used = jiffies;
	atomic_set(&pm->refs, 2);	/* the cache bias plus this I/O */
	base = io_pfn_map_segment(c, pm, folio, seg_base, seg_len);
	if (!base) {
		kfree(pm);
		goto unclaim;
	}
	pm->dma_base = base;

	/*
	 * Charge before publishing: once the entry is visible a flush, or
	 * a lookup displacing it for a longer run under the same key, may
	 * remove it and subtract its bytes, and a remover that subtracts
	 * what was never added drives the counter negative. covered is
	 * signed and the cap is not, so a negative count compared unsigned
	 * reads as enormous and the sweep spends its whole budget on every
	 * call until the add lands.
	 */
	atomic64_add(seg_len, &c->covered);

	/*
	 * Publish over the claim. Nothing else takes a claim out, so this
	 * cannot fail; the fallback below only guards the invariant.
	 */
	old = xa_cmpxchg(&c->xa, pfn, IO_PFN_PENDING, pm,
			 GFP_NOWAIT | __GFP_NOWARN);
	if (unlikely(old != IO_PFN_PENDING)) {
		WARN_ON_ONCE(1);
		atomic64_sub(seg_len, &c->covered);
		io_pfn_map_unmap(pm, true);
		kfree_rcu(pm, rcu);	/* by_slot[] may have shown it */
		goto fail;
	}
	/*
	 * Pairs with the barrier in io_pfn_cache_cap_set(), and with the
	 * grace period in io_pfn_cache_kill(): if the cap was cleared or
	 * the device marked dead under this insert, either the revoking
	 * flush's walk finds the entry now that it is published, or the
	 * change is visible here and this insert takes its own entry back
	 * out. Nothing else would ever retire it.
	 */
	smp_mb();
	if (unlikely(!READ_ONCE(io_dma_pfn_cache_cap_mb) ||
		     READ_ONCE(c->dead)))
		io_pfn_map_displace(c, pm);
	atomic64_inc(&c->inserts);
	if (atomic64_read(&c->covered) > (s64)io_pfn_cache_target(c, cap))
		io_pfn_cache_evict(c, cap);
	*dma = pm->dma_base + rel;
	return pm;
unclaim:
	xa_cmpxchg(&c->xa, pfn, IO_PFN_PENDING, NULL, GFP_NOWAIT | __GFP_NOWARN);
fail:
	atomic64_inc(&c->insert_fails);
	return NULL;
}

/* Erase everything.  In-flight users keep their mappings alive until
 * their references drop.  @retire is set for the flush of
 * io_pfn_cache_kill(), the one flush that collects on a dead cache.
 */
static void io_pfn_cache_flush(struct io_pfn_cache *c, bool retire)
{
	unsigned int n = 0, retired = 0;
	struct io_pfn_map *pm;
	unsigned long index;
	void *gv;

	if (READ_ONCE(c->gone))
		return;	/* nothing is mapped and the collector is held */

	/* The walk runs without c->lock: removals are identity-checked,
	 * so a concurrent sweep and this flush retire disjoint entries,
	 * and a spinlock held across an unmap per entry would put the
	 * whole cache's unmaps in one non-preemptible section. Entries
	 * are freed with kfree_rcu() and the walk hands them back outside
	 * any read-side section, so take the read lock, which under
	 * preemptible RCU the spinlock never was. The walk yields between
	 * batches, so that a large cache's flush neither holds off grace
	 * periods nor, without preemptible RCU, runs as one non-preemptible
	 * section.
	 */
	rcu_read_lock();
	xa_for_each(&c->xa, index, pm) {
		if (xa_is_value(pm))
			continue;	/* an insert's claim; it publishes later */
		if (xa_cmpxchg(&c->xa, index, pm, NULL,
			       GFP_NOWAIT | __GFP_NOWARN) != pm)
			continue;	/* a sweep got there first */
		atomic64_sub(pm->size, &c->covered);
		/*
		 * The loop below collects whatever this walk retires, so a
		 * put here only marks its slot. A collect from each would
		 * unlink the walk's slots one at a time, and a request from
		 * each would keep a busy collector, perhaps a submitter or a
		 * ring's completion poller, at work for the rest of the walk.
		 */
		__io_pfn_map_put(pm, false);
		if (++retired >= IO_PFN_EVICT_UNMAP_MAX) {
			retired = 0;
			rcu_read_unlock();
			cond_resched();
			rcu_read_lock();
		}
	}
	rcu_read_unlock();
	/*
	 * A collector already running on the datapath stops once fewer
	 * than its run threshold are dirty, so retry until every retired
	 * slot is unlinked: a flush that returns with slots still linked
	 * has not revoked anything. On a dead cache the retire collects in
	 * its own flush, then holds the collector and unlinks every slot
	 * left, at the latest once the unbind completes, so any other
	 * flush leaves the slots to it rather than spin until then.
	 */
	while (READ_ONCE(c->nr_dirty) && !READ_ONCE(c->gone) &&
	       (retire || !READ_ONCE(c->dead))) {
		io_pfn_slots_gc(c, 1);
		if (READ_ONCE(c->nr_dirty))
			cond_resched();
	}

	/*
	 * Ghosts are plain values that need neither the cache lock nor RCU,
	 * so a long list is erased with scheduling points rather than in one
	 * non-preemptible section. Only the eraser that got a ghost counts
	 * it, since a concurrent ghost hit or eviction may race this walk.
	 */
	xa_for_each(&c->ghost, index, gv) {
		if (xa_erase(&c->ghost, index))
			atomic64_dec(&c->ghost_count);
		if (!(++n % IO_PFN_EVICT_BUDGET))
			cond_resched();
	}

	spin_lock(&c->lock);
	c->hand = 0;
	c->ghost_hand = 0;
	c->eff_cap = 0;	/* re-derive the floor on next use */
	c->hits_snap = atomic64_read(&c->hits) + atomic64_read(&c->waits);
	c->misses_snap = atomic64_read(&c->misses);
	c->ghost_hits_snap = atomic64_read(&c->ghost_hits);
	c->prev_ratio = 0;
	c->parked_ticks = 0;
	c->dead_ticks = 0;
	c->cyclic_turnover = 0;
	c->inserts_snap = atomic64_read(&c->inserts);
	c->evictions_snap = max_t(s64, atomic64_read(&c->evictions) -
				       atomic64_read(&c->age_evictions), 0);
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

		/* Pairs with the slot-publishing release in io_pfn_cache_get(). */
		if (!smp_load_acquire(&io_pfn_cache_devs[i]))
			break;
		c = io_pfn_caches[i];
		seq_printf(m,
			   "dev %s quantum_kb %zu iova_mb %llu slots_used %u dirty %u links %lld link_fails %lld unlink_runs %lld unlink_slots %lld covered_kb %lld hits %lld waits %lld wait_timeouts %lld misses %lld inserts %lld insert_fails %lld range_fallbacks %lld short_fallbacks %lld evictions %lld age_evictions %lld ref_skips %lld ghost_hits %lld ghost_count %lld eff_cap_mb %llu dead %d\n",
			   dev_name(c->dev),
			   c->quantum >> 10,
			   ((u64)c->nslots << c->quantum_shift) >> 20,
			   READ_ONCE(c->nr_used),
			   READ_ONCE(c->nr_dirty),
			   atomic64_read(&c->slot_links),
			   atomic64_read(&c->slot_fails),
			   atomic64_read(&c->unlink_runs),
			   atomic64_read(&c->unlink_slots),
			   atomic64_read(&c->covered) >> 10,
			   atomic64_read(&c->hits),
			   atomic64_read(&c->waits),
			   atomic64_read(&c->wait_timeouts),
			   atomic64_read(&c->misses),
			   atomic64_read(&c->inserts),
			   atomic64_read(&c->insert_fails),
			   atomic64_read(&c->range_fallbacks),
			   atomic64_read(&c->short_fallbacks),
			   atomic64_read(&c->evictions),
			   atomic64_read(&c->age_evictions),
			   atomic64_read(&c->ref_skips),
			   atomic64_read(&c->ghost_hits),
			   atomic64_read(&c->ghost_count),
			   io_pfn_cache_target(c,
				(u64)READ_ONCE(io_dma_pfn_cache_cap_mb) << 20)
					>> 20,
			   READ_ONCE(c->dead));
	}
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(io_pfn_cache_stats);

/*
 * Retire a cache whose binding or device is going away, while the binding
 * holds the device's IOMMU domain, or just after the hold is dropped for
 * the part the start of an unbind deferred. After the mark, lookups miss and
 * inserts refuse, and after the grace period none is still mapping. The
 * flush drops every entry, then the collector is taken. When the
 * driver is only starting to unbind, its transfers may still be in
 * flight, so a slot that one of them holds keeps the range linked until
 * the unbind completes. Otherwise whatever the reserved range still links
 * is unlinked, including the slots of entries a retired transfer has yet
 * to put, and the range is given back. The cache stays registered, since
 * its device is pinned there, and dead until a ring on the rebound device
 * revives it. A device that comes back after removal is a new one, and
 * the removed device's cache gives its registry slot up once it has given
 * back its slot arrays (io_pfn_cache_release()). The
 * caller holds c->life_lock, so a cache found dead and not gone is one
 * whose retire the start of an unbind left for its completion.
 */
static void io_pfn_cache_kill(struct io_pfn_cache *c, unsigned long action)
{
	unsigned int start, end;

	if (READ_ONCE(c->gone))
		return;
	if (!READ_ONCE(c->dead)) {
		WRITE_ONCE(c->dead, true);
		synchronize_rcu();
		io_pfn_cache_flush(c, true);
		while (atomic_cmpxchg(&c->gc_busy, 0, 1))
			cond_resched();
		if (action == BUS_NOTIFY_UNBIND_DRIVER && c->nslots &&
		    find_first_bit(c->slots, c->nslots) < c->nslots)
			return;
	}
	/* Nothing hands out or collects slots now, so the bitmap is stable. */
	for (start = c->nslots ? find_first_bit(c->slots, c->nslots) : 0;
	     start < c->nslots;
	     start = find_next_bit(c->slots, c->nslots, end)) {
		end = find_next_zero_bit(c->slots, c->nslots, start);
		end = min(end, start + IO_PFN_GC_RUN_MAX);
		dma_iova_unlink(c->dev, &c->iova,
				(size_t)start << c->quantum_shift,
				(size_t)(end - start) << c->quantum_shift,
				DMA_BIDIRECTIONAL, DMA_ATTR_SKIP_CPU_SYNC);
		cond_resched();
	}
	if (c->nslots)
		dma_iova_free(c->dev, &c->iova);
	WRITE_ONCE(c->gone, true);
	dev_info(c->dev, "io_uring DMA: mapping cache retired %s\n",
		 action == BUS_NOTIFY_DEL_DEVICE ? "for device removal" :
		 action == BUS_NOTIFY_UNBOUND_DRIVER ?
		 "once the driver unbound" : "for driver unbind");
}

static void io_pfn_cache_flush_all(void)
{
	int i;

	for (i = 0; i < IO_PFN_CACHE_DEVS; i++) {
		/* Pairs with the slot-publishing release in io_pfn_cache_get(). */
		if (!smp_load_acquire(&io_pfn_cache_devs[i]))
			break;
		io_pfn_cache_flush(io_pfn_caches[i], false);
	}
}

static ssize_t io_pfn_cache_flush_write(struct file *file,
					const char __user *ubuf,
					size_t len, loff_t *ppos)
{
	io_pfn_cache_flush_all();
	return len;
}

static const struct file_operations io_pfn_cache_flush_fops = {
	.owner		= THIS_MODULE,
	.open		= simple_open,
	.write		= io_pfn_cache_flush_write,
	.llseek		= noop_llseek,
};

/*
 * Capping the cache at nothing has to revoke the mappings it already
 * holds. With the cap at zero the datapath never asks for the cache,
 * so no lookup runs and no sweep with it, neither the CLOCK walk nor
 * the age retirement: "off" would otherwise leave standing device
 * access to every folio the cache had touched, past even the age
 * bound, until the cap was raised again.
 */
static int io_pfn_cache_cap_get(void *data, u64 *val)
{
	*val = READ_ONCE(io_dma_pfn_cache_cap_mb);
	return 0;
}

static int io_pfn_cache_cap_set(void *data, u64 val)
{
	if (val > U32_MAX)
		return -ERANGE;
	if (!val) {
		atomic_inc(&io_pfn_cache_clearing);
		/*
		 * The count is raised before the store, so that a put that
		 * sees the zero also sees the clear under way
		 * (io_pfn_slots_gc_cleared()).
		 */
		smp_mb__after_atomic();
	}
	WRITE_ONCE(io_dma_pfn_cache_cap_mb, val);
	if (!val) {
		/*
		 * Pairs with the barrier after the publish in
		 * io_pfn_map_lookup(): an insert that read the old cap
		 * either lands where this flush's walk still finds it,
		 * or sees the zero and takes its own entry back out.
		 * Pairs too with the barrier in io_pfn_map_unmap(): a put
		 * that retires a slot either sees the zero and has the slot
		 * collected, or the flush finds the slot dirty.
		 */
		smp_mb();
		io_pfn_cache_flush_all();
		atomic_dec(&io_pfn_cache_clearing);
		/*
		 * A put during the flushes left its slot to this clear, so
		 * every cache is flushed once more after the clear is seen
		 * to be over. The barrier pairs with the one in
		 * io_pfn_map_unmap(): a put either sees the clear over and
		 * collects, or this flush finds its slot dirty. With the cap
		 * still at zero the caches are empty by then, so the second
		 * flush only collects what those puts left.
		 */
		smp_mb__after_atomic();
		io_pfn_cache_flush_all();
	}
	return 0;
}
DEFINE_DEBUGFS_ATTRIBUTE(io_pfn_cache_cap_fops, io_pfn_cache_cap_get,
			 io_pfn_cache_cap_set, "%llu\n");

/*
 * A walk armed under the old age can lie a quarter of that age ahead,
 * which would hold a lowered bound off for as long, so a write re-arms
 * every cache and the next lookup starts a walk under the new age.
 */
static int io_pfn_cache_max_age_get(void *data, u64 *val)
{
	*val = READ_ONCE(io_dma_pfn_cache_max_age_ms);
	return 0;
}

static int io_pfn_cache_max_age_set(void *data, u64 val)
{
	int i;

	if (val > U32_MAX)
		return -ERANGE;
	WRITE_ONCE(io_dma_pfn_cache_max_age_ms, val);
	/* Pairs with the barrier in io_pfn_cache_evict(). */
	smp_wmb();
	for (i = 0; i < IO_PFN_CACHE_DEVS; i++) {
		/* Pairs with the release in __io_pfn_cache_get(). */
		if (!smp_load_acquire(&io_pfn_cache_devs[i]))
			break;
		WRITE_ONCE(io_pfn_caches[i]->next_age, jiffies);
	}
	return 0;
}
DEFINE_DEBUGFS_ATTRIBUTE(io_pfn_cache_max_age_fops, io_pfn_cache_max_age_get,
			 io_pfn_cache_max_age_set, "%llu\n");

/*
 * Reuse-time export: one section per cache, a comment line of cache
 * state followed by a libmrc .rtd histogram. A write resets every
 * sampler, so a measurement covers exactly one workload.
 */
static int io_pfn_cache_rtd_show(struct seq_file *m, void *p)
{
	unsigned int shift = READ_ONCE(io_dma_pfn_mrc_shift);
	u64 hard = (u64)READ_ONCE(io_dma_pfn_cache_cap_mb) << 20;
	int i;

	seq_printf(m, "# io_uring pfn_cache rtd shift %u cap_mb %u auto %u budget_mb %u\n",
		   shift, READ_ONCE(io_dma_pfn_cache_cap_mb),
		   READ_ONCE(io_dma_pfn_cache_auto),
		   READ_ONCE(io_dma_pfn_cache_auto_budget_mb));
	for (i = 0; i < IO_PFN_CACHE_DEVS; i++) {
		struct io_pfn_cache *c;
		struct io_pfn_mrc *mrc;

		/* Pairs with the slot-publishing release in io_pfn_cache_get(). */
		if (!smp_load_acquire(&io_pfn_cache_devs[i]))
			break;
		c = io_pfn_caches[i];
		seq_printf(m, "# dev %s quantum_kb %zu covered_kb %lld target_mb %llu hits %lld waits %lld misses %lld inserts %lld evictions %lld ghost_hits %lld\n",
			   dev_name(c->dev), c->quantum >> 10,
			   atomic64_read(&c->covered) >> 10,
			   io_pfn_cache_target(c, hard) >> 20,
			   atomic64_read(&c->hits), atomic64_read(&c->waits),
			   atomic64_read(&c->misses),
			   atomic64_read(&c->inserts),
			   atomic64_read(&c->evictions),
			   atomic64_read(&c->ghost_hits));
		mrc = READ_ONCE(c->mrc);
		if (mrc)
			io_pfn_mrc_show(m, mrc, shift);
		else
			seq_puts(m, "# no sampler\n");
	}
	return 0;
}

static int io_pfn_cache_rtd_open(struct inode *inode, struct file *file)
{
	return single_open(file, io_pfn_cache_rtd_show, NULL);
}

static ssize_t io_pfn_cache_rtd_write(struct file *file,
				      const char __user *ubuf,
				      size_t len, loff_t *ppos)
{
	int i;

	for (i = 0; i < IO_PFN_CACHE_DEVS; i++) {
		struct io_pfn_mrc *mrc;

		/* Pairs with the slot-publishing release in io_pfn_cache_get(). */
		if (!smp_load_acquire(&io_pfn_cache_devs[i]))
			break;
		mrc = READ_ONCE(io_pfn_caches[i]->mrc);
		if (mrc)
			io_pfn_mrc_reset(mrc);
	}
	return len;
}

static const struct file_operations io_pfn_cache_rtd_fops = {
	.owner		= THIS_MODULE,
	.open		= io_pfn_cache_rtd_open,
	.read		= seq_read,
	.write		= io_pfn_cache_rtd_write,
	.llseek		= seq_lseek,
	.release	= single_release,
};

static int io_pfn_mrc_shift_get(void *data, u64 *val)
{
	*val = READ_ONCE(io_dma_pfn_mrc_shift);
	return 0;
}

/*
 * Turning sampling on gives every registered cache a sampler here,
 * where allocation may sleep; caches registered later allocate their
 * own at creation, non-blocking, and run unsampled if that fails.
 */
static int io_pfn_mrc_shift_set(void *data, u64 val)
{
	int i;

	if (val > 40)
		return -ERANGE;
	WRITE_ONCE(io_dma_pfn_mrc_shift, val);
	if (!val)
		return 0;
	for (i = 0; i < IO_PFN_CACHE_DEVS; i++) {
		struct io_pfn_cache *c;
		struct io_pfn_mrc *m;

		/* Pairs with the slot-publishing release in io_pfn_cache_get(). */
		if (!smp_load_acquire(&io_pfn_cache_devs[i]))
			break;
		c = io_pfn_caches[i];
		if (READ_ONCE(c->mrc))
			continue;
		m = io_pfn_mrc_alloc(GFP_KERNEL);
		if (!m)
			return -ENOMEM;
		if (cmpxchg(&c->mrc, NULL, m))
			io_pfn_mrc_free(m);
	}
	return 0;
}
DEFINE_DEBUGFS_ATTRIBUTE(io_pfn_mrc_shift_fops, io_pfn_mrc_shift_get,
			 io_pfn_mrc_shift_set, "%llu\n");

/*
 * Busy-poll budget in microseconds for draining in-flight DMA
 * completions from the CQ-wait path (io_dma_cq_wait_poll()) before the
 * waiting task commits to sleeping. A DSA transfer for a typical read
 * chunk completes in about 4 to 6us plus queueing. Sleeping instead
 * leaves the completion to the poll worker that the publish queued,
 * plus a wakeup of this task, about 10us more, just to be woken again.
 * 0 disables the poll. This is tunable via debugfs
 * io_uring_dma/cq_poll_us.
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
 * This is tunable via debugfs io_uring_dma/fmw_spin_us.
 */
static unsigned int io_dma_fmw_spin_us __read_mostly = 60;

/*
 * How long the DMA filemap-write wait gives the device before it
 * declares the transfer wedged, leaks the destination mappings and
 * fails the write. Tunable via debugfs io_uring_dma/fmw_wait_ms, which
 * refuses values below IO_DMA_FMW_WAIT_MIN_MS. A wedge leaks the folios
 * the device may still write and loses what they held, so the timeout
 * must stay well above what a healthy but contended queue takes. At 0
 * the deadline is the next tick, and at a few milliseconds the sleep
 * back-off and the queueing of many rings on one work queue already
 * reach it.
 */
#define IO_DMA_FMW_WAIT_MIN_MS	100
static unsigned int io_dma_fmw_wait_ms __read_mostly = 5000;

static int io_dma_fmw_wait_ms_get(void *data, u64 *val)
{
	*val = READ_ONCE(io_dma_fmw_wait_ms);
	return 0;
}

static int io_dma_fmw_wait_ms_set(void *data, u64 val)
{
	if (val > U32_MAX)
		return -ERANGE;
	if (val < IO_DMA_FMW_WAIT_MIN_MS)
		return -EINVAL;
	WRITE_ONCE(io_dma_fmw_wait_ms, val);
	return 0;
}
DEFINE_DEBUGFS_ATTRIBUTE(io_dma_fmw_wait_ms_fops, io_dma_fmw_wait_ms_get,
			 io_dma_fmw_wait_ms_set, "%llu\n");

/*
 * Filemap DMA-write gate and result counters. These are surfaced
 * through the io_uring_dma/latency debugfs stats file and zeroed by
 * its _reset companion.
 */
static const char * const io_dma_fmw_names[IO_DMA_FMW_NR] = {
	[IO_DMA_FMW_ENGAGED] = "engaged",
	[IO_DMA_FMW_UNSUPPORTED] = "unsupported",
	[IO_DMA_FMW_NOT_BVEC] = "not_bvec",
	[IO_DMA_FMW_DIRECT] = "direct",
	[IO_DMA_FMW_NO_DMA_ADDRS] = "no_dma_addrs",
	[IO_DMA_FMW_EAGAIN] = "eagain",
	[IO_DMA_FMW_CPU_REDO] = "cpu_redo",
	[IO_DMA_FMW_ERROR] = "error",
	[IO_DMA_FMW_FS_DECLINE] = "fs_decline",
};
static atomic64_t io_dma_fmw[IO_DMA_FMW_NR];

void io_dma_fmw_record(unsigned int reason)
{
	atomic64_inc(&io_dma_fmw[reason]);
}

static int io_dma_lat_show(struct seq_file *m, void *v)
{
	unsigned int i;

	seq_puts(m, "filemap_write:\n");
	for (i = 0; i < IO_DMA_FMW_NR; i++)
		seq_printf(m, "  %-12s %12llu\n", io_dma_fmw_names[i],
			   atomic64_read(&io_dma_fmw[i]));
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(io_dma_lat);

static ssize_t io_dma_lat_reset_write(struct file *file,
				      const char __user *ubuf,
				      size_t count, loff_t *ppos)
{
	unsigned int i;

	for (i = 0; i < IO_DMA_FMW_NR; i++)
		atomic64_set(&io_dma_fmw[i], 0);
	return count;
}

static const struct file_operations io_dma_lat_reset_fops = {
	.owner		= THIS_MODULE,
	.open		= simple_open,
	.write		= io_dma_lat_reset_write,
	.llseek		= noop_llseek,
};

/*
 * Devices a ring stripes over, one channel each. One channel reaches one
 * device, four engines of sixteen; a single-stream op striped across the
 * devices reaches them all. A ring takes no more devices than a buffer
 * is mapped on (IO_REG_BUF_DEVS, four), so the default takes every
 * device on the ring's node, or, when none of those can give a channel,
 * every device of the nearest distance that can, up to that cap. A ring
 * offered more devices than it takes starts at a rotating one, so a
 * value below the count, or a node with more devices than the cap,
 * trades the cross-ring cache dedup for spread. Values below the default
 * exist for experiments, and 0 takes the default. Read at ring creation.
 */
unsigned int io_dma_stripe_chans __read_mostly = IO_DMA_RING_CHANS;

void io_dma_debugfs_init(void)
{
	struct dentry *dir;

	dir = debugfs_create_dir("io_uring_dma", NULL);

	debugfs_create_u32("cq_poll_us", 0644, dir, &io_dma_cq_poll_us);
	debugfs_create_u32("fmw_spin_us", 0644, dir, &io_dma_fmw_spin_us);
	debugfs_create_file_unsafe("fmw_wait_ms", 0644, dir, NULL,
				   &io_dma_fmw_wait_ms_fops);
	debugfs_create_u32("stripe_chans", 0644, dir, &io_dma_stripe_chans);
	debugfs_create_file("latency", 0444, dir, NULL, &io_dma_lat_fops);
	debugfs_create_file("latency_reset", 0200, dir, NULL,
			    &io_dma_lat_reset_fops);
	debugfs_create_file("pfn_cache", 0444, dir, NULL,
			    &io_pfn_cache_stats_fops);
	debugfs_create_file_unsafe("pfn_cache_cap_mb", 0644, dir, NULL,
				   &io_pfn_cache_cap_fops);
	debugfs_create_u32("pfn_cache_auto", 0644, dir,
			   &io_dma_pfn_cache_auto);
	debugfs_create_u32("pfn_cache_auto_budget_mb", 0644, dir,
			   &io_dma_pfn_cache_auto_budget_mb);
	debugfs_create_file_unsafe("pfn_cache_max_age_ms", 0644, dir, NULL,
				   &io_pfn_cache_max_age_fops);
	debugfs_create_file("pfn_cache_flush", 0200, dir, NULL,
			    &io_pfn_cache_flush_fops);
	debugfs_create_file("pfn_cache_rtd", 0644, dir, NULL,
			    &io_pfn_cache_rtd_fops);
	debugfs_create_file_unsafe("pfn_cache_mrc_shift", 0644, dir, NULL,
				   &io_pfn_mrc_shift_fops);
	debugfs_create_u32("pfn_cache_iova_reserve_mb", 0644, dir,
			   &io_dma_pfn_iova_reserve_mb);
	debugfs_create_u32("pfn_cache_wait_us", 0644, dir, &io_dma_pfn_wait_us);
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
				queue_work(system_dfl_wq, &d->poll_work);
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
				   unsigned int nr_entries, u32 off)
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
	dma->off = off;
	dma->chan = chan;
	dma->is_batch = true;
	dma->batch_nr = nr_entries;
	dma->batch_entries = heap_entries;

	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		kfree(heap_entries);
		io_dma_task_free(req->ctx, dma);
		return -EAGAIN;	/* The WQ may be full. Fall back to CPU copy. */
	}

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
					  struct io_dma_batch_entry *entry,
					  u32 off)
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

	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		folio_put(entry->folio);
		io_dma_task_free(req->ctx, dma);
		return -EAGAIN;	/* The WQ may be full. Fall back to CPU copy. */
	}

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
				  unsigned int nr_entries, u32 off)
{
	ssize_t total = 0;
	unsigned int i;
	ssize_t ret;

	if (!nr_entries)
		return 0;

	if (nr_entries < IO_DMA_BATCH_MIN) {
		for (i = 0; i < nr_entries; i++) {
			ret = io_dma_submit_single_entry(req, chan, &entries[i],
							 off + total);
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

	ret = io_dma_submit_batch(req, dev, chan, entries, nr_entries, off);
	if (ret < 0)
		io_dma_unmap_batch_entries(req, dev, entries, nr_entries);
	return ret;
}

/*
 * File-position striping. A region's transfers go to the same device
 * from every ring that holds the same devices in the same order, at
 * this granule, so the per-device PFN caches each hold one share of a
 * shared working set instead of a full copy each.
 * The granule index is hashed so block-aligned op starts do not
 * resonate with the stripe modulus. Position-consistent placement only
 * pays when mappings persist: uncached or parked it would concentrate
 * convoyed readers on one device for no dedup benefit, so a ring
 * rotates instead. Reads and writes share both rules; that is what
 * lets the entry a write inserts serve the read that follows it.
 */
#define IO_DMA_STRIPE_SHIFT	20

/*
 * Return the first stripe at or after @stripe, wrapping, whose channel
 * its provider has not orphaned, or @stripe when every one is. @stripe
 * is below the ring's channel count. An orphaned channel completes
 * nothing for the rest of the ring's life, so the rotation passes over
 * it, and a lost device narrows the stripe instead of taking every op
 * that would start on it to the CPU.
 */
static unsigned int io_dma_stripe_live(struct io_ring_ctx *ctx,
				       unsigned int stripe)
{
	unsigned int i;

	for (i = 0; i < ctx->dma.nr_chans; i++) {
		if (!dma_chan_orphaned(ctx->dma.chans[stripe]))
			break;
		if (++stripe == ctx->dma.nr_chans)
			stripe = 0;
	}
	return stripe;
}

/*
 * The stripe of @granule when placement follows position. A region whose
 * channel its provider orphaned goes to a live channel picked by the hash
 * bits the modulus left over, so that the live channels share the
 * orphan's regions evenly instead of the next one taking them all. Only
 * the orphan's regions move, so every other region stays on the device
 * that holds its standing mappings, and rings that hold the same
 * channels, with the same ones orphaned, still agree on placement.
 */
static unsigned int io_dma_stripe_hashed(struct io_ring_ctx *ctx,
					 u64 granule)
{
	unsigned int nr_chans = ctx->dma.nr_chans;
	u32 hash = hash_64(granule, 32);
	unsigned int stripe, nr_live = 0, k, i;

	if (nr_chans <= 1)
		return 0;
	stripe = hash % nr_chans;
	if (!dma_chan_orphaned(ctx->dma.chans[stripe]))
		return stripe;
	for (i = 0; i < nr_chans; i++)
		nr_live += !dma_chan_orphaned(ctx->dma.chans[i]);
	if (!nr_live)
		return stripe;
	k = (hash / nr_chans) % nr_live;
	for (i = 0; i < nr_chans; i++) {
		if (dma_chan_orphaned(ctx->dma.chans[i]))
			continue;
		if (!k)
			return i;
		k--;
	}
	/*
	 * The walk runs out only when a channel was orphaned after the
	 * count. The region then takes the next live channel, so that a
	 * pick returns an orphan only when every channel is one.
	 */
	return io_dma_stripe_live(ctx, stripe);
}

static bool io_dma_stripe_deterministic(struct io_ring_ctx *ctx)
{
	u64 hard = (u64)READ_ONCE(io_dma_pfn_cache_cap_mb) << 20;
	struct io_pfn_cache *pc;

	if (!io_pfn_cache_usable() || ctx->dma.nr_chans <= 1)
		return false;
	/*
	 * The cache of the first channel not orphaned decides, one that
	 * this ring's lookups, which run a cache's adapt tick, still
	 * drive. They no longer reach the cache of an orphaned channel's
	 * device, which after a removal nothing drives at all.
	 */
	pc = io_pfn_cache_get(
		ctx->dma.chans[io_dma_stripe_live(ctx, 0)]->device->dev);
	return pc && io_pfn_cache_target(pc, hard);
}

/*
 * The stripe for @pos: by position when deterministic, else the first
 * live one at or after @rr, where the rotation stands. A write group's
 * later granules step on from there, see io_dma_piece_stripe().
 */
static unsigned int io_dma_stripe_index(struct io_ring_ctx *ctx, loff_t pos,
					bool det, unsigned int rr)
{
	unsigned int nr_chans = ctx->dma.nr_chans ? ctx->dma.nr_chans : 1;
	u64 granule = (u64)pos >> IO_DMA_STRIPE_SHIFT;

	if (det)
		return io_dma_stripe_hashed(ctx, granule);
	return io_dma_stripe_live(ctx, rr % nr_chans);
}

/* The stripe of the read batch starting at @pos: by position while the
 * cache is engaged, the next live one in the rotation otherwise.
 */
static unsigned int io_dma_next_stripe(struct io_ring_ctx *ctx, bool det,
				       unsigned int stripe, loff_t pos)
{
	unsigned int nr_chans = ctx->dma.nr_chans ? ctx->dma.nr_chans : 1;

	if (det)
		return io_dma_stripe_index(ctx, pos, true, 0);
	return io_dma_stripe_live(ctx, (stripe + 1) % nr_chans);
}

/*
 * The stripe of the write piece at @pos in a group placed by @det, whose
 * rotation starts at @rr. Every piece of a granule goes to the same
 * stripe, so the group derives it again only when a piece starts in
 * another granule, which keeps the hash and the divisions of a
 * derivation off the per-piece path. @granule and @stripe hold the last
 * derivation, and @granule is U64_MAX before the group's first. In the
 * rotation the first granule takes the first live stripe at or after
 * @rr, and each later one the next live stripe after the last, so a
 * group's granules take its live channels in turn and a group of one
 * granule runs where the rotation stands. Counting from the absolute
 * granule instead would step a stream of one-granule groups by two per
 * group, against the rotation's one, and leave half of a four-channel
 * ring idle. A channel orphaned within a granule leaves the granule's
 * remaining pieces to the CPU redo, through the orphan check of each
 * piece.
 */
static unsigned int io_dma_piece_stripe(struct io_ring_ctx *ctx, loff_t pos,
					bool det, unsigned int rr,
					u64 *granule, unsigned int *stripe)
{
	unsigned int nr_chans = ctx->dma.nr_chans ? ctx->dma.nr_chans : 1;
	u64 g = (u64)pos >> IO_DMA_STRIPE_SHIFT;

	if (g == *granule)
		return *stripe;
	if (det || *granule == U64_MAX)
		*stripe = io_dma_stripe_index(ctx, pos, det, rr);
	else
		*stripe = io_dma_stripe_live(ctx, (*stripe + 1) % nr_chans);
	*granule = g;
	return *stripe;
}

/* The cache instance the datapath uses for @dev, or NULL when it is off. */
static struct io_pfn_cache *io_pfn_cache_for(struct device *dev)
{
	return io_pfn_cache_usable() ? io_pfn_cache_get(dev) : NULL;
}

static struct dma_chan *io_dma_stripe_chan(struct io_ring_ctx *ctx,
					   unsigned int stripe)
{
	return ctx->dma.nr_chans ? ctx->dma.chans[stripe] : ctx->dma.chan;
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
	size_t map_quantum;
	struct folio_batch fbatch;
	struct io_dma_batch_entry *entries;
	unsigned int nr_entries = 0;
	ssize_t total_read = 0;
	ssize_t submitted = 0;
	size_t batch_bytes = 0;
	size_t dst_offset = 0;
	loff_t start_pos = iocb->ki_pos;

	det_stripe = io_dma_stripe_deterministic(ctx);
	/*
	 * Hash the granule index rather than using it raw: block-aligned
	 * op starts otherwise resonate with the stripe modulus, opening
	 * every op on index zero. With rings agreeing on index zero's
	 * device, that device saturates, its batches flush short, the
	 * claims end before the rotation to the next granule ever runs,
	 * and the whole workload converges on one device with CPU tails.
	 * The hash keeps the mapping deterministic, so the dedup holds,
	 * while decorrelating it from any alignment.
	 */
	if (det_stripe)
		stripe = io_dma_stripe_hashed(ctx,
				(u64)iocb->ki_pos >> IO_DMA_STRIPE_SHIFT);
	else
		stripe = io_dma_stripe_live(ctx,
				READ_ONCE(ctx->dma.stripe_rr) % nr_chans);
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
	/*
	 * A channel its provider orphaned at unbind completes nothing any
	 * more, and its device may be on its way out. A read that starts
	 * on its stripe takes the CPU path without mapping anything. The
	 * pick above passes over such channels, so a read gets here only
	 * when every channel of the ring is orphaned, or when its channel
	 * was orphaned since the pick.
	 */
	if (dma_chan_orphaned(chan))
		return -EAGAIN;

	if (unlikely(iocb->ki_pos < 0))
		return -EINVAL;
	if (unlikely(iocb->ki_pos >= inode->i_sb->s_maxbytes))
		return 0;

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
						       !copied ||
						       !((offset + copied) &
							 (map_quantum - 1)),
						       &src_dma);
				if (pm) {
					/* A standing mapping is synced by
					 * nobody but us: hand the chunk to
					 * the device, since the CPU may have
					 * written the folio since the mapping
					 * was made.
					 */
					dma_sync_single_for_device(dev, src_dma,
								   chunk,
								   DMA_TO_DEVICE);
				} else {
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
				if (nr_entries == IO_DMA_BATCH_MAX ||
				    batch_bytes >= clamp(want / 4,
						(size_t)SZ_256K,
						(size_t)SZ_1M)) {
					ssize_t ret;
					loff_t next;

					ret = io_dma_flush_batch(req, dev, chan,
						entries, nr_entries,
						dst_offset - batch_bytes);
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
					 * new target. ki_pos still points at
					 * this folio's start, and copied is
					 * the part of it collected so far.
					 * A read with nothing left to collect
					 * has no next batch. A stripe derived
					 * for one anyway would move the
					 * rotation on twice and leave a
					 * device out of it.
					 */
					next = iocb->ki_pos + copied;
					if (nr_chans > 1 &&
					    ret == (ssize_t)batch_bytes &&
					    next < end_offset) {
						stripe = io_dma_next_stripe(ctx,
							det_stripe, stripe, next);
						chan = ctx->dma.chans[stripe];
						dev = chan->device->dev;
						pfn_cache = io_pfn_cache_for(dev);
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
						/* Ending the loop, not just
						 * the folio batch: another
						 * pass would refetch from a
						 * stale ki_pos and land its
						 * data past the gap.
						 */
						batch_bytes = 0;
						error = -EAGAIN;
						goto put_folios;
					}
					batch_bytes = 0;
					/* The pick and the re-derive above
					 * pass over orphaned channels, so this
					 * one was orphaned since its pick, or
					 * every channel is. End the claim at
					 * what was submitted and leave any
					 * rest to the CPU tail. Without a
					 * re-derive, on the read's last batch
					 * or on a ring of one channel, this is
					 * the channel just used.
					 */
					if (dma_chan_orphaned(chan)) {
						error = -EAGAIN;
						goto put_folios;
					}
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
				dst_offset - batch_bytes);
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
			/*
			 * The next pass starts a new batch, which takes its
			 * stripe from its own position as a batch after the
			 * in-loop flush does. A folio batch of small folios
			 * never reaches the in-loop flush, and without this
			 * a whole read of them would stay on the stripe of
			 * its first byte. A batch that crosses into the next
			 * granule still runs on the stripe of its first
			 * byte, since ending batches at granule boundaries
			 * would cost an unaligned read one more descriptor.
			 */
			if (nr_chans > 1 && !error && dst_offset < want &&
			    iocb->ki_pos < isize) {
				stripe = io_dma_next_stripe(ctx, det_stripe,
							    stripe,
							    iocb->ki_pos);
				chan = ctx->dma.chans[stripe];
				dev = chan->device->dev;
				pfn_cache = io_pfn_cache_for(dev);
				/* See the orphan check of the in-loop flush. */
				if (dma_chan_orphaned(chan))
					error = -EAGAIN;
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
	/* Persist the rotation for the uncached regime's spread, past the
	 * last stripe this read used.
	 */
	if (!det_stripe)
		WRITE_ONCE(ctx->dma.stripe_rr, stripe + 1);
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
 * idempotent. On a completion timeout the device is wedged. We skip
 * write_end for the folios the device may write and deliberately leak
 * them referenced, together with their dst mappings, so that a late
 * DMA write can only land in memory the kernel still owns. Unmapping
 * would free the IOVA for the next mapping to reuse. Where the
 * filesystem can take a folio out of its page cache, as memory failure
 * does, the leaked folios are taken out and unlocked, so no task waits
 * on them and later I/O of the range goes to the disk instead of the
 * torn copy. tmpfs keeps them locked. The write then returns -EIO, or
 * a short write when groups before the wedged one committed.
 */

struct io_dma_fmw_folio {
	struct folio *folio;
	void *fsdata;
	loff_t pos;
	unsigned int len;
	bool redo;		/* some of it was not handed to the device */
	bool targeted;		/* a descriptor for it was submitted */
	bool lost;		/* the disk may lack what it held */
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
	struct dma_chan *chan;		/* the piece's stripe */
	struct device *dev;
	unsigned int stripe;
};

/* One in-flight write descriptor and the channel it was submitted on. */
struct io_dma_fmw_ck {
	dma_cookie_t ck;
	struct dma_chan *chan;
};

/*
 * Poll every outstanding cookie to completion.  Sets *redo on any
 * DMA_ERROR.  Returns 0, or -ETIMEDOUT with cookies possibly still in
 * flight, in which case the caller must treat every dst as poisoned.
 * On -ETIMEDOUT *stuck is the channel of the cookie that ran out of
 * time. A group's cookies can be on several devices, so the caller
 * names that channel and not the one it submitted to last.
 *
 * We spin only briefly.  DSA transfers normally complete in
 * single-digit microseconds, but under contention with many rings
 * sharing the WQ descriptor pools completion can take milliseconds.
 * Hundreds of io-wq workers busy-polling here saturated whole sockets
 * and starved application heartbeats at O(100) shared rings.
 * Therefore we back off to sleeping once the fast path misses.
 */
static int io_dma_fmw_wait(struct io_dma_fmw_ck *cookies, unsigned int *nr,
			   bool *redo, struct dma_chan **stuck)
{
	unsigned int wait_ms = READ_ONCE(io_dma_fmw_wait_ms);
	unsigned long deadline = jiffies + msecs_to_jiffies(wait_ms);
	u64 spin_end = ktime_get_ns() +
		READ_ONCE(io_dma_fmw_spin_us) * NSEC_PER_USEC;
	unsigned int i, j, spins;

	for (i = 0; i < *nr; i++) {
		enum dma_status st;

		struct dma_chan *chan = cookies[i].chan;

		spins = 0;
		while ((st = dmaengine_async_is_tx_complete(chan, cookies[i].ck))
		       == DMA_IN_PROGRESS) {
			if (time_after(jiffies, deadline)) {
				/* The group is wedged on this cookie. A polled
				 * descriptor's only completion path is this
				 * status query, so an abandoned cookie keeps
				 * its descriptor until its work queue is
				 * disabled and no client holds the channel any
				 * more, which frees the pool. Freeing the
				 * descriptor earlier would let the device write
				 * a completion record into recycled memory. The
				 * group's pieces may be placed on several
				 * devices, and the later cookies on healthy
				 * ones have most likely finished, so each is
				 * queried once, which hands a finished
				 * descriptor back to its pool. Only the
				 * descriptors that query does not retire are
				 * leaked, and the caller fences them.
				 */
				for (j = i + 1; j < *nr; j++)
					dmaengine_async_is_tx_complete(
						cookies[j].chan, cookies[j].ck);
				*stuck = chan;
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
 * On a filesystem that truncates failed writes a chunk that reaches past
 * EOF is a group of its own, see the loop in io_dma_fmw_group().
 * Folios can reach the mapping's maximum folio size, so a group also
 * ends with the folio that brings it to IO_DMA_FMW_GROUP_BYTES or beyond.
 * Without that, 256 folios of 2MB would hold 512MB locked and not yet
 * dirty, which reclaim cannot free and the dirty throttle does not see.
 * Each group end waits for the copies still in flight, so every group
 * adds the latency of its last copies to the write.
 */
#define IO_DMA_FMW_GROUP_FOLIOS	256
#define IO_DMA_FMW_GROUP_BYTES	SZ_32M

/*
 * Copy one chunk with the CPU and commit it, as generic_perform_write()
 * does, while no other folio of the file is held. That is a chunk whose
 * write_begin() opened a journal handle, which the write_end() closes,
 * or a small chunk of a group capped at one folio. The source is a
 * pinned registered-buffer bvec, so the copy cannot fault and needs no
 * atomic variant. A committed chunk counts as eagain. A negative error
 * ends the write, which counts as error when nothing was written and is
 * a short write otherwise. Returns the bytes committed, 0 when
 * write_end() took nothing, or a negative error.
 */
static ssize_t io_dma_fmw_cpu_chunk(struct kiocb *iocb,
				    struct address_space *mapping,
				    struct iov_iter *from, loff_t pos,
				    size_t bytes, struct folio *folio,
				    void *fsdata)
{
	size_t copied;
	long status;

	if (mapping_writably_mapped(mapping))
		flush_dcache_folio(folio);
	copied = copy_folio_from_iter(folio, offset_in_folio(folio, pos),
				      bytes, from);
	flush_dcache_folio(folio);
	status = mapping->a_ops->write_end(iocb, mapping, pos, bytes, copied,
					   folio, fsdata);
	if (unlikely(status != copied))
		iov_iter_revert(from, copied - max(status, 0L));
	if (status > 0) {
		balance_dirty_pages_ratelimited(mapping);
		io_dma_fmw_record(IO_DMA_FMW_EAGAIN);
	}
	cond_resched();
	return status;
}

/* Ring the doorbell of every stripe with submitted, unissued work. */
static void io_dma_fmw_issue(struct io_ring_ctx *ctx, unsigned long *issued)
{
	unsigned int i;

	for_each_set_bit(i, issued, IO_DMA_RING_CHANS)
		dma_async_issue_pending(io_dma_stripe_chan(ctx, i));
	*issued = 0;
}

/*
 * Take a folio that a wedged group leaks out of the page cache, as memory
 * failure takes out a poisoned one, then unlock it. The caller keeps the
 * folio's reference and its destination mapping, so a late write still
 * lands in memory the kernel owns, and no lookup can find the folio any
 * more. A task asleep on its lock wakes and finds it truncated:
 * writeback skips it, and readers, faults and page_mkwrite look the range
 * up again and read it from the disk. What it held that the disk lacks
 * is lost, which @lost says, and fsync() reports -EIO for that. The loss
 * is recorded before the folio leaves the cache, as memory failure does,
 * so that an fsync() that no longer finds the folio still reports it.
 * The truncation needs the I/O of an earlier writeback ended. Returns
 * false, with the folio still locked, where the filesystem keeps such a
 * folio in its cache, as tmpfs does.
 */
static bool io_dma_fmw_drop(struct address_space *mapping,
			    struct folio *folio, bool lost)
{
	const struct address_space_operations *aops = mapping->a_ops;

	if (!aops->error_remove_folio)
		return false;
	folio_wait_writeback(folio);
	if (lost)
		mapping_set_error(mapping, -EIO);
	if (aops->error_remove_folio(mapping, folio) || folio->mapping)
		return false;
	folio_unlock(folio);
	return true;
}

/*
 * Run one group of a DMA write: collect folios starting @base bytes
 * into the write until the group is full or the write ends, submit the
 * copies, wait for them, and commit. Returns the bytes committed, or a
 * negative error when nothing was committed. On a device timeout the
 * folios the device may still write are deliberately leaked, the others
 * released, and -EIO is returned, so the caller must stop.
 */
static ssize_t io_dma_fmw_group(struct io_kiocb *req, struct kiocb *iocb,
				struct iov_iter *from, u64 src_user_addr,
				size_t base, size_t room,
				struct io_dma_fmw_folio *fol,
				unsigned int *fol_cap,
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
	size_t max_chunk = mapping_max_folio_size(mapping);
	/*
	 * A group collects at most max_fol folios. The loop lowers the cap
	 * to 1 for the rest of the write once a write_begin() opens a
	 * journal handle, or fails a later chunk with -EAGAIN.
	 */
	unsigned int max_fol = *fol_cap;
	/*
	 * Only this write's write_end() moves i_size under inode_lock. The
	 * ones that run before the group commits lie inside EOF, so the
	 * value holds while the group collects.
	 */
	loff_t isize = i_size_read(mapping->host);
	/*
	 * A folio that was already dirty ends the group only on a mapping
	 * that writes back, for the reason the loop gives. That depends on
	 * the mapping alone, so it is read once.
	 */
	bool dirty_ends = mapping_can_writeback(mapping);
	/*
	 * ext4 answers a failed write past EOF by truncating the page cache
	 * from EOF, which locks every folio there and so would wait forever
	 * on one this group holds. On such a filesystem a chunk that reaches
	 * past EOF is a group of its own, so that no folio of the file is
	 * held while its write_begin() runs or while a journal handle it
	 * opened is closed. tmpfs never truncates, and keeps batching its
	 * appends.
	 */
	bool eof_alone = file->f_op->fop_flags & FOP_TRUNCATE_FAILED_WRITE;
	/*
	 * Each destination piece is placed by file position like a read
	 * batch, so the entry a write inserts is on the device the read
	 * of that region will use. The pieces are split at quantum
	 * boundaries aligned to file position, so none crosses a stripe
	 * granule. Rotation, when the cache is parked, advances per group.
	 * A rotating group starts at the first live stripe, as a read does,
	 * and moves the rotation on past it.
	 */
	bool det = io_dma_stripe_deterministic(ctx);
	unsigned int rr = det ? 0 : io_dma_stripe_index(ctx, 0, false,
					READ_ONCE(ctx->dma.stripe_rr));
	/* The granule of the last piece placed, and its stripe. */
	u64 piece_granule = U64_MAX;
	unsigned int piece_stripe = 0;
	unsigned long issued = 0;
	/*
	 * Destination folios are mapped in pieces of at most one
	 * IOVA-rcache-served quantum, split at quantum boundaries, and only
	 * whole-quantum segments are cached. The quantum splits only the
	 * pieces. It never shortens a chunk, whose length must stay what
	 * write_begin() prepared.
	 */
	size_t map_quantum = io_dma_map_quantum(dev);
	struct io_dma_fmw_dst *cur_dst = NULL;
	size_t cur_dst_remain = 0;
	unsigned int nr_dst = 0;
	unsigned int nr_fol = 0, nr_cookies = 0, i;
	unsigned int prep_fails = 0;
	/* redo: a completion could not be trusted, so every folio is
	 * re-copied. any_redo: at least one folio was not fully handed to
	 * the device and is re-copied on its own.
	 */
	bool redo = false, any_redo = false;
	bool surrendered = false;
	ssize_t collected = 0, committed = 0;
	ssize_t err = 0;
	int wedged = 0;
	/* The channel a wedge timed out on, which the warning names. */
	struct dma_chan *stuck = NULL;

	if (!det)
		WRITE_ONCE(ctx->dma.stripe_rr, rr + 1);

	while (collected < room && nr_fol < max_fol &&
	       collected < IO_DMA_FMW_GROUP_BYTES) {
		loff_t pos = iocb->ki_pos + base + collected;
		size_t bytes = min_t(size_t,
				     max_chunk - (pos & (max_chunk - 1)),
				     room - collected);
		size_t offset, sub;
		struct folio *folio;
		void *fsdata;
		bool chunk_redo = false, targeted = false, was_dirty, lost;
		int status;

		/*
		 * The length is the hint and so at least the chunk, which
		 * keeps the test conservative.
		 */
		if (eof_alone && nr_fol && pos + bytes > isize)
			break;
		/*
		 * Once the group holds a folio, a write_begin() that starts
		 * a journal handle, pushes writeback or forces a commit
		 * would wait with that folio locked, against ext4's order
		 * of transaction start before folio lock. A task that
		 * sleeps on a held folio while it holds up a commit would
		 * then deadlock with the write. Writeback does that when
		 * another writeback cleaned a folio it looked up dirty and
		 * this group took it clean, and so does a page fault in
		 * ext4's non-delalloc branch. So a later chunk's
		 * write_begin() runs under IOCB_FOLIOS_LOCKED, under which
		 * ext4 skips its writeback push and fails the call where
		 * it would start a handle or retry an allocation. The first
		 * chunk of a group runs with nothing of the file held and
		 * may do all three, as under generic_perform_write().
		 */
		if (nr_fol)
			iocb->ki_flags |= IOCB_FOLIOS_LOCKED;
		status = aops->write_begin(iocb, mapping, pos, bytes,
					   &folio, &fsdata);
		iocb->ki_flags &= ~IOCB_FOLIOS_LOCKED;
		if (unlikely(status < 0)) {
			/*
			 * A failure on a later chunk ends the group, which
			 * commits the folios collected before it, and the
			 * next group retries the chunk with nothing held,
			 * where ext4 falls back or retries the allocation.
			 * A failure there ends the write short, as on the
			 * stock path. Until then the group's earlier folios
			 * stay locked, so reclaim has less to work with. On
			 * a filesystem that truncates failed writes those
			 * folios all lie inside EOF, so the truncation from
			 * EOF never waits on one. ext4 returns -EAGAIN under
			 * the flag only where it would have opened a handle,
			 * so that lowers the cap as a handle does below.
			 */
			if (status == -EAGAIN && nr_fol)
				*fol_cap = 1;
			if (!collected)
				err = status;
			break;
		}
		/*
		 * Cover the locked folio to its end or to the write's end.
		 * That keeps the chunk inside the range write_begin() was
		 * asked to prepare, which runs to the write's end or to the
		 * next boundary of the mapping's maximum folio size,
		 * whichever comes first. No folio crosses that boundary.
		 * The chunk must not reach past the range. ext4 maps or
		 * reserves blocks only inside it, and its writeback treats
		 * a dirty block that was neither mapped nor reserved as a
		 * hole and never writes it. tmpfs checks F_SEAL_GROW
		 * against the same range. Short of the write's end the
		 * chunk must not stop inside the folio either, or the next
		 * iteration's write_begin() would wait forever on the
		 * folio lock this group already holds. Descriptors below
		 * still split at source-folio boundaries.
		 */
		offset = offset_in_folio(folio, pos);
		bytes = min_t(size_t, room - collected,
			      folio_size(folio) - offset);
		/*
		 * On a mapping that writes back, a folio that was already
		 * dirty is tagged for writeback, and the flusher locks
		 * tagged folios. Another write_begin() while the group
		 * holds it locked can wait on that flusher (ext4 under low
		 * free space, or a journal commit), which is a deadlock, so
		 * such a folio ends its group below. shmem keeps its folios
		 * dirty from their first write, fallocate() or swap-in until
		 * swap-out without ever tagging them, and nothing writes
		 * them back but reclaim, which skips locked folios. Testing
		 * the flag alone would end a group at every folio a tmpfs
		 * overwrite reaches.
		 */
		was_dirty = folio_test_dirty(folio) && dirty_ends;
		/*
		 * Whether the disk may lack what the folio holds, which a
		 * wedge that drops the folio would lose. Writeback cannot
		 * start while the group holds the lock, but one already in
		 * flight can end during the wait with torn bytes on the
		 * disk, so the state is taken now and not at the wedge.
		 * Only a mapping that writes back has a disk to lack it.
		 */
		lost = dirty_ends &&
		       (folio_test_dirty(folio) || folio_test_writeback(folio));
		/*
		 * A write_begin that leaves a journal handle open (ext4
		 * delalloc falling back under low free space) expects its
		 * matching write_end before the next write_begin. Batching
		 * would nest handles with h_ref only and no credits, and a
		 * wedge would leak the references and stall the journal.
		 * So the handle is closed at once. ext4 declines a write up
		 * front when its write_begin would open a handle, so in
		 * practice this catches a fallback that starts during the
		 * write. The check runs on every chunk because the fallback
		 * is per write_begin, and this write's own delayed
		 * allocations, or a concurrent writer, can bring free space
		 * below dirty space plus the watermark mid-write.
		 *
		 * The first chunk of a group is copied with the CPU and
		 * committed, as the stock path does, with no other folio of
		 * the file held. ext4 fails the write_begin() of a later
		 * chunk rather than open a handle there, see
		 * IOCB_FOLIOS_LOCKED above, so a handle on a later chunk
		 * comes only from a filesystem that ignores the flag. Such
		 * a chunk ends its group. A write_end that copies nothing
		 * closes its handle, with the chunk, the range the
		 * filesystem prepared, as its length. The group commits the
		 * folios collected before it, and the next group starts
		 * with this chunk. On a filesystem that truncates failed
		 * writes a chunk that reaches past EOF never joins a group
		 * that holds folios, so this one lies inside EOF and that
		 * write_end neither moves i_size nor truncates. Past EOF it
		 * would truncate the file, fallocate preallocation
		 * included.
		 *
		 * Either way the write goes on, so a fallback never ends it
		 * short, but it runs one folio per group from here on. A
		 * write that has met the fallback runs near the watermark,
		 * where ext4 fails the write_begin() of a later chunk of a
		 * batched group whenever the fallback holds, which would
		 * end that group early.
		 */
		if (unlikely(current->journal_info)) {
			*fol_cap = 1;
			if (!nr_fol)
				return io_dma_fmw_cpu_chunk(iocb, mapping, from,
							    pos, bytes, folio,
							    fsdata);
			aops->write_end(iocb, mapping, pos, bytes, 0,
					folio, fsdata);
			break;
		}
		/*
		 * Only the caps above make max_fol 1, since
		 * io_dma_filemap_write() starts it at two folios or more. A
		 * group of one folio pays a device round trip for that
		 * folio alone, which at or below IO_DMA_MIN_WRITE_BYTES
		 * costs more than the copy it replaces, so such a chunk is
		 * copied with the CPU, as a write of that size is.
		 */
		if (unlikely(max_fol == 1) && bytes <= IO_DMA_MIN_WRITE_BYTES)
			return io_dma_fmw_cpu_chunk(iocb, mapping, from, pos,
						    bytes, folio, fsdata);

		if (surrendered) {
			/*
			 * This is sustained descriptor exhaustion. We stop
			 * touching the DMA engine for the rest of this
			 * group, and the CPU-redo pass copies its remaining
			 * chunks. This beats paying a drain-wait per chunk
			 * while dozens of other rings hold the pools empty.
			 */
			chunk_redo = true;
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

			/* Acquire the dst piece covering offset + sub; it
			 * picks the stripe every descriptor into it uses.
			 */
			if (!cur_dst || !cur_dst_remain) {
				size_t doff = offset + sub;
				size_t plen = min_t(size_t, bytes - sub,
						map_quantum -
						(doff & (map_quantum - 1)));
				unsigned int stripe = io_dma_piece_stripe(ctx,
							pos + sub, det, rr,
							&piece_granule,
							&piece_stripe);
				struct dma_chan *pchan =
					io_dma_stripe_chan(ctx, stripe);
				struct device *pdev = pchan->device->dev;
				struct io_pfn_cache *pc = io_pfn_cache_usable() ?
					io_pfn_cache_get(pdev) : NULL;
				struct io_pfn_map *pm = NULL;
				dma_addr_t d;

				if (nr_dst == max_dst) {
					chunk_redo = true;
					break;
				}
				/* An orphaned channel completes nothing: leave
				 * this piece to the CPU-redo pass, mapping
				 * nothing.
				 */
				if (dma_chan_orphaned(pchan)) {
					chunk_redo = true;
					break;
				}
				pm = pc ?
					io_pfn_map_lookup(pc, folio,
							  doff, plen,
							  folio_size(folio),
							  (!collected && !sub) ||
							  !(doff & (map_quantum - 1)),
							  &d) : NULL;
				if (pm) {
					/* The device is about to write it;
					 * it owns the piece until the wait
					 * below hands it back.
					 */
					dma_sync_single_for_device(pdev, d, plen,
								   DMA_FROM_DEVICE);
				} else {
					d = dma_map_page(pdev,
							 folio_page(folio, 0),
							 doff, plen,
							 DMA_FROM_DEVICE);
					if (dma_mapping_error(pdev, d)) {
						chunk_redo = true;
						break;
					}
				}
				dsts[nr_dst] = (struct io_dma_fmw_dst){
					.dma = d, .len = plen, .off = doff,
					.folio = folio, .pm = pm,
					.chan = pchan, .dev = pdev,
					.stripe = stripe,
				};
				cur_dst = &dsts[nr_dst++];
				cur_dst_remain = plen;
			}
			chan = cur_dst->chan;
			src_dma = io_reg_buf_dma_addr(imu, uaddr,
						      &src_seg_remain,
						      cur_dst->dev);
			if (unlikely(!src_dma)) {
				chunk_redo = true; /* The CPU re-copies this folio. */
				break;
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
				io_dma_fmw_issue(ctx, &issued);
				wedged = io_dma_fmw_wait(cookies, &nr_cookies,
							 &redo, &stuck);
				if (wedged)
					goto record;
				tx = dmaengine_prep_dma_memcpy(chan, dst,
						src_dma, len,
						io_dma_prep_flags());
			}
			if (!tx) {
				io_dma_fmw_record(IO_DMA_FMW_EAGAIN);
				chunk_redo = true;
				/*
				 * Two drain-and-retry failures in one group
				 * mean the pools are held empty by other
				 * rings, so the rest of the group goes to
				 * the CPU.
				 */
				if (++prep_fails >= 2) {
					surrendered = true;
					break;	/* redo covers this chunk. */
				}
			} else {
				ck = dmaengine_submit(tx);
				if (dma_submit_error(ck)) {
					chunk_redo = true;
				} else {
					cookies[nr_cookies++] =
						(struct io_dma_fmw_ck){
							.ck = ck, .chan = chan,
						};
					issued |= BIT(cur_dst->stripe);
					targeted = true;
				}
				if (nr_cookies == max_cookies) {
					io_dma_fmw_issue(ctx, &issued);
					wedged = io_dma_fmw_wait(cookies,
							&nr_cookies, &redo,
							&stuck);
					if (wedged)
						goto record;
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
			.redo = chunk_redo, .targeted = targeted, .lost = lost,
		};
		/*
		 * A wedge found while this folio's copies were submitted ends
		 * the group here, with the folio recorded, so the wedge path
		 * below sees every folio the group holds.
		 */
		if (unlikely(wedged))
			break;
		any_redo |= chunk_redo;
		collected += bytes;
		if (was_dirty || (eof_alone && pos + bytes > isize))
			break;
	}

	if (!wedged) {
		io_dma_fmw_issue(ctx, &issued);
		wedged = io_dma_fmw_wait(cookies, &nr_cookies, &redo,
					 &stuck);
	}

	/* Unmap the dst IOVAs, except after a timeout. An unmap frees the
	 * IOVA, at once under a strict IOMMU and at the next flush-queue
	 * drain otherwise, and the next mapping for the device can reuse
	 * it. A late write from an abandoned descriptor would then go
	 * through that valid translation into an unrelated page. So a
	 * wedge leaks the mappings with the folios below, which stay
	 * referenced, and a late write can only land there.
	 */
	for (i = 0; i < nr_dst; i++) {
		if (dsts[i].pm) {
			/* The device wrote it, or on a wedge may have:
			 * hand the piece back to the CPU before the folio
			 * is unlocked or leaked.
			 */
			dma_sync_single_for_cpu(dsts[i].dev, dsts[i].dma,
						dsts[i].len, DMA_FROM_DEVICE);
			/* On a wedge the cached mapping leaves the cache,
			 * so no other I/O is handed it, and keeps this
			 * write's reference. The mapping is then never
			 * unmapped, and a slot-backed one's slot is never
			 * collected or linked to another folio.
			 */
			if (unlikely(wedged)) {
				io_pfn_map_displace(dsts[i].pm->cache,
						    dsts[i].pm);
				continue;
			}
			io_pfn_map_put(dsts[i].pm);
		} else if (likely(!wedged)) {
			dma_unmap_page(dsts[i].dev, dsts[i].dma, dsts[i].len,
				       DMA_FROM_DEVICE);
		}
	}

	if (unlikely(wedged)) {
		unsigned int leaked = 0, dropped = 0;

		/*
		 * No descriptor of this write was submitted for such a folio,
		 * so the device cannot write it. It is released as after a
		 * failed copy, with nothing claimed, and keeps whatever it
		 * held, but only where a zero claim cannot move i_size. On a
		 * filesystem that truncates failed writes every such folio
		 * lies inside EOF, since a chunk that reaches past EOF is a
		 * group of its own and a group with nothing submitted cannot
		 * wedge, so its write_end() neither starts a handle nor waits.
		 * tmpfs batches appends past EOF, and its write_end() moves
		 * i_size to the position even with nothing copied, which would
		 * put the leaked folios below it inside EOF. A folio that
		 * starts past EOF is leaked with them instead.
		 */
		for (i = 0; i < nr_fol; i++)
			if (!fol[i].targeted && fol[i].pos <= isize)
				aops->write_end(iocb, mapping, fol[i].pos,
						fol[i].len, 0, fol[i].folio,
						fol[i].fsdata);
		/*
		 * The others keep their reference and their mapping for good,
		 * so that a late write can only land in memory the kernel
		 * still owns. Their contents are unknown. A task can sleep on
		 * the lock of any of them: writeback that looked it up, a
		 * read of a range not uptodate, a fault, compaction, and
		 * every later truncate or write of the file. And readers of
		 * an uptodate one would copy the torn bytes. Where the
		 * filesystem takes such a folio out of its page cache, see
		 * io_dma_fmw_drop(), it is unlocked as well, and every task
		 * asleep on it wakes and finds it gone. A reader or a pin that
		 * took the folio before can still see the torn bytes, as after
		 * a truncate. The highest goes first. ext4 writeback can hold
		 * the bio of a lower folio unsubmitted while it sleeps on a
		 * higher one, so waiting on a folio's writeback with a higher
		 * folio still held could wait forever. A mapping that writes
		 * back must drop the folio, see FOP_DMA_WRITE, or the next
		 * writeback that looks it up would sleep on it for good.
		 */
		for (i = nr_fol; i-- > 0; ) {
			if (!fol[i].targeted && fol[i].pos <= isize)
				continue;
			leaked++;
			if (io_dma_fmw_drop(mapping, fol[i].folio, fol[i].lost))
				dropped++;
			else
				WARN_ON_ONCE(dirty_ends);
		}
		pr_warn_ratelimited("io_uring DMA write: wedged after %ums, leaking %u folios, %u of them dropped from the page cache (%s)\n",
				    READ_ONCE(io_dma_fmw_wait_ms), leaked, dropped,
				    dma_chan_name(stuck));
		io_dma_fmw_record(IO_DMA_FMW_ERROR);
		return -EIO;
	}

	if (unlikely(redo || any_redo)) {
		/* Re-copy with the CPU: every folio after an untrusted
		 * completion, otherwise only the folios the device was not
		 * handed in full. The same source bytes go to the same
		 * folio ranges, so overlap with completed DMA is
		 * idempotent. The iter was never advanced during
		 * collection, so it is walked over the folios in order.
		 */
		for (i = 0; i < nr_fol; i++) {
			size_t n;

			if (!redo && !fol[i].redo) {
				iov_iter_advance(from, fol[i].len);
				continue;
			}
			n = copy_folio_from_iter(fol[i].folio,
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
		}
	}
	/*
	 * Throttle once per group, with every folio unlocked. Inside the
	 * loop the later folios of a rewrite are dirty and locked, and
	 * the flusher this waits for would block on them.
	 */
	if (committed)
		balance_dirty_pages_ratelimited(mapping);

	/* The CPU re-copy consumes the iter when it runs; otherwise
	 * advance it over exactly what was committed. The two counts
	 * cannot diverge: the source is a pinned registered-buffer
	 * bvec, so copy_folio_from_iter() never copies short, every
	 * claim therefore equals its chunk length, and write_end
	 * with a full claim returns it whole or fails outright.
	 */
	if (!redo && !any_redo)
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
	size_t max_folio, quantum;
	ssize_t want, written = 0, err = 0;

	/* At or below this size the per-folio write_begin and write_end
	 * machinery plus the descriptor overhead exceeds the memcpy it
	 * replaces, so we take the CPU path.
	 */
	if (iov_iter_count(from) <= IO_DMA_MIN_WRITE_BYTES)
		return -EAGAIN;

	inode_lock(inode);
	/*
	 * The prologue of the filesystem's write_iter. A filesystem whose
	 * checks go beyond the generic ones, such as ext4 with its immutable
	 * test, its bitmap-format size limit and the zeroing of the old last
	 * block on a write past EOF, runs them through ->dma_copy_checks();
	 * the others get the generic sequence.
	 */
	if (file->f_op->dma_copy_checks) {
		want = file->f_op->dma_copy_checks(iocb, from, WRITE);
	} else {
		want = generic_write_checks(iocb, from);
		if (want > 0) {
			err = file_modified(file);
			if (err)
				want = err;
		}
	}
	if (want <= 0) {
		inode_unlock(inode);
		/*
		 * The write path runs without IOCB_NOWAIT, so only a
		 * filesystem declining the offload returns -EAGAIN here.
		 */
		if (want == -EAGAIN)
			io_dma_fmw_record(IO_DMA_FMW_FS_DECLINE);
		return want;
	}

	max_fol = min_t(unsigned long, IO_DMA_FMW_GROUP_FOLIOS,
			DIV_ROUND_UP(want, PAGE_SIZE) + 1);
	/* One entry per src-folio crossing per chunk. The collection loop
	 * flushes whenever the array fills, so this sets the wait
	 * granularity rather than bounding the transfer.
	 */
	max_cookies = max_fol + DIV_ROUND_UP(want, 1UL << imu->folio_shift) + 8;
	max_cookies = min_t(unsigned int, max_cookies, 4 * max_fol + 8);
	/*
	 * One dst piece per folio chunk, plus one per map-quantum boundary
	 * inside a chunk. A folio is aligned to its size, so in a folio
	 * larger than the quantum those boundaries are quantum multiples of
	 * the file position, and a group spans at most the write, or
	 * IO_DMA_FMW_GROUP_BYTES plus its last folio. Bounding by that span
	 * rather than by a maximum-size folio per chunk keeps the array,
	 * which every write allocates and frees, small enough for the
	 * per-CPU page lists.
	 */
	max_folio = mapping_max_folio_size(file->f_mapping);
	quantum = io_dma_map_quantum(req->ctx->dma.chan->device->dev);
	max_dst = min_t(size_t,
			max_fol * (1 + DIV_ROUND_UP(max_folio, quantum)),
			max_fol + DIV_ROUND_UP(min_t(size_t, want,
					IO_DMA_FMW_GROUP_BYTES + max_folio),
					quantum));
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
		ssize_t done;

		/*
		 * A fatal signal ends the write before the next group, as
		 * it ends generic_perform_write() before the next chunk. A
		 * write that starts with one pending copies nothing, and
		 * one that runs a folio per group stops after the folio in
		 * progress.
		 */
		if (fatal_signal_pending(current)) {
			if (!written)
				err = -EINTR;
			break;
		}
		done = io_dma_fmw_group(req, iocb, from, src_user_addr,
					written, want - written, fol,
					&max_fol, cookies, max_cookies,
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
	if (written > 0)
		return generic_write_sync(iocb, written) ?: written;
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
static void io_dma_task_release_res(struct io_ring_ctx *ctx, struct device *dev,
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

	/* Free the task before touching the refcnt. task_len was saved above. */
	atomic_dec(&req->ctx->dma.tasks_pending);
	io_dma_task_free(req->ctx, dma);
	req->dma.dma_refcnt--;

	if (req->dma.dma_refcnt == 0) {
		pr_debug("dma req done: opcode=%d result=%d\n",
			 req->opcode, req->dma.dma_result);

		if (req->dma.dma_result < 0 ||
		    req->dma.min_fail_off != U32_MAX) {
			/*
			 * A failed task, a device that went away under the
			 * ring for instance, or a claim the submitter could
			 * not finish without blocking. The bytes below the
			 * lowest failed or unread offset were delivered,
			 * since every task lying wholly below it succeeded;
			 * the rest is read again from the page cache, which
			 * still holds it. io_rw_dma_redo() reissues the
			 * request from that offset, with the DMA gate closed
			 * after a failure.
			 */
			req->io_task_work.func = io_rw_dma_redo;
			io_req_task_work_add(req);
		} else {
			/*
			 * Terminal completion goes through task_work.
			 * __io_dma_poll() may run from a workqueue without
			 * ctx->uring_lock, so io_req_task_complete() defers
			 * the CQE post until the task_work loop acquires it.
			 *
			 * A short result marks the request failed as a short
			 * CPU read does, so that an IOSQE_IO_LINK successor
			 * does not run on a partly filled buffer and an
			 * IOSQE_CQE_SKIP_SUCCESS read still posts its CQE.
			 * The issuer writes no req->flags once the tasks are
			 * published, and the flags it still tests are ones
			 * this completion never changes.
			 */
			if (req->dma.short_res)
				req_set_fail(req);
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
		struct io_dma_task *t = req->dma.dma_tasks;

		pr_debug("submit_queued: refcnt=%d\n", req->dma.dma_refcnt);
		/*
		 * Take the chain off the req and clear its DMA state before
		 * any task is published. Once the tasks are visible a poller
		 * can complete the req, and that completion can queue a
		 * reissue that arms new DMA state on the req from another
		 * thread when the publisher is an io-wq worker. So the req
		 * is not written after the publish below.
		 */
		req->dma.dma_tasks = NULL;
		req->dma.dma_tasks_tail = NULL;
		req->dma.dma_active = false;
		if (req->dma.dma_refcnt > 0) {
			struct llist_node *first = NULL, *last = NULL;
			unsigned int n = 0;
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
			spin_unlock_irqrestore(&ctx->dma.lock, flags);

			/*
			 * Link the chain newest first, as one llist_add() per
			 * task in chain order would, so the consumer's
			 * reversal gives chain order back. The next pointers
			 * are read here, before any task is published,
			 * because a published task can complete and be freed
			 * at once. Every task was fully submitted with a valid
			 * cookie, since tasks are linked only after
			 * dmaengine_submit() succeeds, so the poller can
			 * consume them as they are.
			 */
			for (; t; t = t->next) {
				t->llnode.next = first;
				first = &t->llnode;
				if (!last)
					last = first;
				n++;
			}
			/*
			 * Publish after the ref take above, so that no drain
			 * can reach dma_refcnt == 0 before the in-flight ref
			 * exists. The pending count rises before the tasks are
			 * visible, so no observer can see a task without the
			 * count. The cmpxchg in llist_add_batch() orders the
			 * two. Several producers can publish at once: the one
			 * uring_lock holder, which is the SQPOLL thread on an
			 * SQPOLL ring, and any number of io-wq workers, which
			 * publish without the lock. llist_add_batch() is safe
			 * for that, and each producer's tasks keep their order
			 * after the reversal.
			 *
			 * The poll worker is queued in the same section with
			 * preemption off. A publisher preempted between the
			 * publish and the queue_work() could leave its tasks
			 * without a poller, and their CQEs held back, until it
			 * ran again. The worker goes on system_dfl_wq, which
			 * is unbound, rather than through schedule_work(),
			 * whose system_percpu_wq is per-CPU. A per-CPU kworker
			 * must win a wakeup-preemption fight with the
			 * submitting thread to run, and this shows up as
			 * completions sitting undetected for the application's
			 * whole compute slice. An unbound worker lands on an
			 * idle CPU and detects completions in parallel.
			 */
			preempt_disable();
			atomic_add(n, &ctx->dma.tasks_pending);
			llist_add_batch(first, last, &ctx->dma.submit_list);
			if (io_dma_pending(ctx))
				queue_work(system_dfl_wq, &ctx->dma.poll_work);
			preempt_enable();
			ret = -EIOCBQUEUED;
		}
	}

	/*
	 * Nothing here touches the req after the publish. The caller may
	 * still read fields that neither a completion nor a reissue
	 * changes, such as req->file. The req stays allocated for that:
	 * an io-wq worker holds its own reference, and an inline or
	 * SQPOLL issuer runs the completion's task_work only after it
	 * returns.
	 */
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
	 * and every submitter rings for its own tasks from its issue
	 * path. Nothing on the pending lists needs a kick.
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
 * @deadline is the first expiry of the caller's wait timer in the ring's
 * clock. That is the end of min_wait when one is set and the timeout
 * otherwise, or KTIME_MAX when the wait has neither. The spin starts no
 * poll pass once it has passed and stops at the first check at or after
 * it.
 *
 * Returns true if the caller should skip sleeping and re-run its wait
 * loop because task_work or CQEs are ready. Returns false to fall back
 * to schedule(). This is a no-op for rings with no DMA channel.
 */
bool io_dma_cq_wait_poll(struct io_ring_ctx *ctx, struct io_wait_queue *iowq,
			 ktime_t deadline)
{
	unsigned int budget_us = READ_ONCE(io_dma_cq_poll_us);
	u64 spin_ns, end_ns;

	if (!budget_us || IS_ERR_OR_NULL(ctx->dma.chan))
		return false;
	if (!io_dma_pending(ctx))
		return false;

	/*
	 * Only the caller's timed sleep applies the deadline, so the spin
	 * ends at the deadline and does not start once it has passed. The
	 * ring's clock is read only here and the spin is timed with
	 * ktime_get_ns(), as napi's busy poll does. IORING_REGISTER_CLOCK
	 * can switch the ring's clock meanwhile, and comparing readings of
	 * two clocks could leave the spin without a time bound.
	 */
	spin_ns = (u64)budget_us * NSEC_PER_USEC;
	if (deadline != KTIME_MAX) {
		ktime_t now = io_get_time(ctx);

		if (now >= deadline)
			return false;
		spin_ns = min_t(u64, spin_ns, ktime_sub(deadline, now));
	}
	end_ns = ktime_get_ns() + spin_ns;
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
	 * The caller is about to block with tasks still pending, and the
	 * poll worker is their backstop. No exit of the spin queues it.
	 * Every publish queues the worker after it raises tasks_pending,
	 * and the worker polls again or requeues itself while
	 * io_dma_pending(), which reads tasks_pending first, is true, so
	 * the worker does not exit while a task is pending. Queueing it
	 * here as well would find it running nearly every time, and
	 * queueing a running work item wakes an idle kworker only to hand
	 * the item back to the busy one. A task whose publisher has raised
	 * tasks_pending but not yet reached its queue_work() is picked up
	 * once the publisher gets there.
	 */
	return false;
}
