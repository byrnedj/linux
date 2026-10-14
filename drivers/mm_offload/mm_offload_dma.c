// SPDX-License-Identifier: GPL-2.0-only
/*
 * mm_offload_dma - shared DMAEngine channel pool for mm_offload providers.
 *
 * The pool holds every kernel channel it can get, up to its cap, and
 * releases them only when the last user goes away, so a provider
 * enabled while another one is active joins the existing channels
 * instead of finding them all taken, and every node's devices are there
 * for the operations that target it. Growth appends channels and groups
 * behind a release barrier; claimers read the published counts with an
 * acquire load and never see a half-initialised entry.
 */

#include <linux/atomic.h>
#include <linux/bitops.h>
#include <linux/device.h>
#include <linux/dma-mapping.h>
#include <linux/dmaengine.h>
#include <linux/idxd_dma.h>
#include <linux/kobject.h>
#include <linux/ktime.h>
#include <linux/mm.h>
#include <linux/perf_event.h>
#include <linux/mm_offload_dma.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/rcupdate.h>
#include <linux/sizes.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/sysfs.h>
#include <linux/wait.h>
#include <linux/xarray.h>
#include <linux/jiffies.h>

static void map_cache_free_all(void);
static void hw_events_create(void);
static void hw_events_release(void);

static DEFINE_MUTEX(pool_mutex);
static unsigned int pool_users;

/*
 * Shared channels: any number of operations submit to one channel, and
 * the budget below is what bounds the queue in front of each descriptor.
 * Exclusive channels: one operation per channel under a trylock, others
 * fall back to the CPU. Fixed at module load so a claim and its release
 * always agree on which regime they are in.
 */
static bool shared = true;
module_param(shared, bool, 0444);
MODULE_PARM_DESC(shared, "Share channels between concurrent operations (Y) or hand each to one operation under a trylock (N)");

/*
 * Descriptors admitted per channel. A dedicated DSA work queue of 64
 * entries holds 64 descriptors; staying below that means a submission
 * never finds the ring full, so the wait happens here, bounded, rather
 * than as a failed prep that fails the whole batch.
 */
static unsigned int chan_budget = 48;
module_param(chan_budget, uint, 0644);
MODULE_PARM_DESC(chan_budget, "Descriptors admitted per channel before submitters wait");

static unsigned int admit_timeout_ms = 100;
module_param(admit_timeout_ms, uint, 0644);
MODULE_PARM_DESC(admit_timeout_ms, "How long a submitter waits for channel budget before giving the work to the CPU (0: never wait)");

/*
 * Which devices an operation may use, relative to the node it writes to.
 * Local devices allocate their cache-controlled writes in the LLC the
 * consumer will read from; a remote device can still win on raw
 * bandwidth where the interconnect is asymmetric, so the choice is a
 * knob. A node without devices of its own always falls back to the
 * others, whatever the setting.
 */
enum {
	MM_OFFLOAD_DMA_LOCAL,		/* the node's own devices only */
	MM_OFFLOAD_DMA_LOCAL_FIRST,	/* local, then the rest for what remains */
	MM_OFFLOAD_DMA_REMOTE,		/* other nodes' devices only (diagnostic) */
};
static unsigned int cross_node = MM_OFFLOAD_DMA_LOCAL;
module_param(cross_node, uint, 0644);
MODULE_PARM_DESC(cross_node, "Devices for an operation: 0 its node's only, 1 its node's first then others, 2 other nodes' only");

static struct {
	struct dma_chan *chan;
	struct device *dev;
	struct mutex lock;		/* exclusive mode only */
	atomic_t inflight;		/* admitted, not yet retired */
	atomic_t completions;		/* retires that came from the engine */
	wait_queue_head_t waitq;	/* woken by every retire */
	atomic_long_t admits;
	atomic_long_t waits;
	atomic_long_t timeouts;
} channels[MM_OFFLOAD_DMA_MAX_CHANNELS];
static unsigned int nr_channels;

static struct mm_offload_dma_group groups[MM_OFFLOAD_DMA_MAX_CHANNELS];
static unsigned int nr_groups;
static atomic_t group_cursor;
/* One channel cursor per group, so groups rotate independently of each other. */
static atomic_t group_chan_cursor[MM_OFFLOAD_DMA_MAX_CHANNELS];

unsigned int mm_offload_dma_nr_channels(void)
{
	return smp_load_acquire(&nr_channels);
}
EXPORT_SYMBOL_GPL(mm_offload_dma_nr_channels);

struct dma_chan *mm_offload_dma_chan(unsigned int idx)
{
	return channels[idx].chan;
}
EXPORT_SYMBOL_GPL(mm_offload_dma_chan);

struct device *mm_offload_dma_chan_dev(unsigned int idx)
{
	return channels[idx].dev;
}
EXPORT_SYMBOL_GPL(mm_offload_dma_chan_dev);

/* Called with pool_mutex held. */
static void pool_grow(void)
{
	dma_cap_mask_t mask;

	dma_cap_zero(mask);
	dma_cap_set(DMA_MEMCPY, mask);

	while (nr_channels < MM_OFFLOAD_DMA_MAX_CHANNELS) {
		struct dma_chan *chan = dma_request_chan_by_mask(&mask);
		struct device *dev;

		if (IS_ERR(chan))
			break;
		dev = dmaengine_get_dma_device(chan);
		if (!dev) {
			dma_release_channel(chan);
			break;
		}
		channels[nr_channels].chan = chan;
		channels[nr_channels].dev = dev;
		mutex_init(&channels[nr_channels].lock);
		atomic_set(&channels[nr_channels].inflight, 0);
		atomic_set(&channels[nr_channels].completions, 0);
		init_waitqueue_head(&channels[nr_channels].waitq);
		atomic_long_set(&channels[nr_channels].admits, 0);
		atomic_long_set(&channels[nr_channels].waits, 0);
		atomic_long_set(&channels[nr_channels].timeouts, 0);
		if (!nr_groups || groups[nr_groups - 1].dev != dev) {
			groups[nr_groups].dev = dev;
			groups[nr_groups].node = dev_to_node(dev);
			groups[nr_groups].first = nr_channels;
			groups[nr_groups].nr = 0;
			/* Publish the group before the channel count. */
			smp_store_release(&nr_groups, nr_groups + 1);
		}
		groups[nr_groups - 1].nr++;
		smp_store_release(&nr_channels, nr_channels + 1);
	}
}

/**
 * mm_offload_dma_pool_get - become a user of the pool.
 *
 * Acquires every kernel DMA_MEMCPY channel the pool does not hold yet,
 * up to MM_OFFLOAD_DMA_MAX_CHANNELS. Return: 0 if the pool holds at
 * least one channel afterwards, -ENODEV otherwise (the caller is then
 * not a user).
 */
int mm_offload_dma_pool_get(void)
{
	int ret = 0;

	mutex_lock(&pool_mutex);
	pool_grow();
	if (nr_channels) {
		pool_users++;
		hw_events_create();
	} else {
		ret = -ENODEV;
	}
	mutex_unlock(&pool_mutex);
	if (!ret)
		pr_info("mm_offload_dma: %u channel(s) in %u group(s), %u user(s)\n",
			nr_channels, nr_groups, pool_users);
	return ret;
}
EXPORT_SYMBOL_GPL(mm_offload_dma_pool_get);

/**
 * mm_offload_dma_pool_put - drop one user; free the channels with the last.
 *
 * The caller guarantees none of its operations are still in flight
 * (mm_offload_unregister() has waited for them).
 */
void mm_offload_dma_pool_put(void)
{
	mutex_lock(&pool_mutex);
	if (WARN_ON(!pool_users))
		goto out;
	if (--pool_users == 0) {
		/* Standing mappings and counters go before the devices they belong to. */
		map_cache_free_all();
		hw_events_release();
		while (nr_channels) {
			nr_channels--;
			dma_release_channel(channels[nr_channels].chan);
			channels[nr_channels].chan = NULL;
		}
		nr_groups = 0;
		pr_info("mm_offload_dma: channels released\n");
	}
out:
	mutex_unlock(&pool_mutex);
}
EXPORT_SYMBOL_GPL(mm_offload_dma_pool_put);

/* Take channel @idx for one operation: always, or if nobody holds it. */
static bool take_channel(unsigned int idx)
{
	return shared || mutex_trylock(&channels[idx].lock);
}

static unsigned long claim_group(struct mm_offload_dma_group *grp,
				 unsigned int want, unsigned int limit)
{
	unsigned long mask = 0;
	unsigned int i, got = 0, start = 0;

	/*
	 * Shared channels: start each claim at a different channel of the
	 * group so concurrent operations spread over its engines instead of
	 * all queueing on the first one.
	 */
	if (shared && grp->nr)
		start = (unsigned int)atomic_inc_return(
				&group_chan_cursor[grp - groups]) % grp->nr;

	for (i = 0; i < grp->nr && got < want; i++) {
		unsigned int idx = grp->first + (start + i) % grp->nr;

		if (idx >= limit)
			continue;
		if (take_channel(idx)) {
			mask |= BIT(idx);
			got++;
		}
	}
	return mask;
}

/*
 * One selection pass: collect the groups whose node does (or does not)
 * match @nid and try them from a rotating offset within that selection,
 * so rotation is fair however the eligible groups are laid out.
 */
static unsigned long claim_pass(int nid, bool match_node, unsigned int want,
				struct mm_offload_dma_group **grpp)
{
	unsigned int sel[MM_OFFLOAD_DMA_MAX_CHANNELS];
	unsigned int ngroups = smp_load_acquire(&nr_groups);
	unsigned int limit = smp_load_acquire(&nr_channels);
	unsigned int n = 0, g, i;

	for (g = 0; g < ngroups; g++)
		if ((groups[g].node == nid) == match_node)
			sel[n++] = g;
	if (!n)
		return 0;

	g = (unsigned int)atomic_inc_return(&group_cursor) % n;
	for (i = 0; i < n; i++) {
		struct mm_offload_dma_group *grp = &groups[sel[(g + i) % n]];
		unsigned long mask = claim_group(grp, want, limit);

		if (mask) {
			*grpp = grp;
			return mask;
		}
	}
	return 0;
}

/**
 * mm_offload_dma_claim - claim up to @want channels of one device group.
 * @want: channels wanted.
 * @nid: NUMA node of the destination; groups on it are tried first
 *       (a cross-socket copy pays remote-link bandwidth on every written
 *       line plus a remote completion interrupt, so locality beats
 *       spreading).
 * @grpp: the group the channels belong to, on success.
 *
 * Return: bitmask of claimed channel indices, 0 if every channel of
 * every group is busy (the caller then falls back to the CPU rather
 * than queueing).
 */
unsigned long mm_offload_dma_claim(unsigned int want, int nid,
				   struct mm_offload_dma_group **grpp)
{
	bool local_first = READ_ONCE(cross_node) != MM_OFFLOAD_DMA_REMOTE;
	unsigned long mask = claim_pass(nid, local_first, want, grpp);

	if (!mask)
		mask = claim_pass(nid, !local_first, want, grpp);
	return mask;
}
EXPORT_SYMBOL_GPL(mm_offload_dma_claim);

/*
 * As claim_pass(), but the group is chosen by @key instead of by
 * rotation: the same key always lands on the same device.
 */
static unsigned long claim_pass_keyed(int nid, bool match_node,
				      unsigned int want, unsigned long key,
				      struct mm_offload_dma_group **grpp)
{
	unsigned int sel[MM_OFFLOAD_DMA_MAX_CHANNELS];
	unsigned int ngroups = smp_load_acquire(&nr_groups);
	unsigned int limit = smp_load_acquire(&nr_channels);
	unsigned int n = 0, g, i;

	for (g = 0; g < ngroups; g++)
		if ((groups[g].node == nid) == match_node)
			sel[n++] = g;
	if (!n)
		return 0;

	g = key % n;
	for (i = 0; i < n; i++) {
		struct mm_offload_dma_group *grp = &groups[sel[(g + i) % n]];
		unsigned long mask = claim_group(grp, want, limit);

		if (mask) {
			*grpp = grp;
			return mask;
		}
	}
	return 0;
}

/**
 * mm_offload_dma_claim_keyed - claim channels of the device @key selects.
 * @want: channels wanted.
 * @nid: NUMA node of the destination; its groups are tried first.
 * @key: anything stable for the memory being copied, e.g. the PFN of
 *       the destination at huge-page granularity. The same key always
 *       maps to the same device among a node's, so a device's standing
 *       mappings (mm_offload_dma_map_get()) only ever have to cover its
 *       own share of a recycling pool instead of all of it.
 * @grpp: the group the channels belong to, on success.
 *
 * Return: bitmask of claimed channel indices, 0 if every channel of
 * every group is busy.
 */
unsigned long mm_offload_dma_claim_keyed(unsigned int want, int nid,
					 unsigned long key,
					 struct mm_offload_dma_group **grpp)
{
	bool local_first = READ_ONCE(cross_node) != MM_OFFLOAD_DMA_REMOTE;
	unsigned long mask = claim_pass_keyed(nid, local_first, want, key, grpp);

	if (!mask)
		mask = claim_pass_keyed(nid, !local_first, want, key, grpp);
	return mask;
}
EXPORT_SYMBOL_GPL(mm_offload_dma_claim_keyed);

/* One breadth-first pass over the groups that do (not) match @nid. */
static unsigned long spread_pass(unsigned int want, int nid, bool match_node,
				 unsigned long mask, unsigned int *gotp)
{
	unsigned int sel[MM_OFFLOAD_DMA_MAX_CHANNELS];
	unsigned int ngroups = smp_load_acquire(&nr_groups);
	unsigned int limit = smp_load_acquire(&nr_channels);
	unsigned int maxnr = 0, n = 0, r, k, gs = 0;

	for (k = 0; k < ngroups; k++) {
		if ((groups[k].node == nid) != match_node || !groups[k].nr)
			continue;
		sel[n++] = k;
		maxnr = max(maxnr, groups[k].nr);
	}
	if (!n)
		return mask;
	if (shared)
		gs = (unsigned int)atomic_inc_return(&group_cursor);

	for (r = 0; r < maxnr && *gotp < want; r++) {
		for (k = 0; k < n && *gotp < want; k++) {
			struct mm_offload_dma_group *grp = &groups[sel[(gs + k) % n]];
			unsigned int idx;

			if (r >= grp->nr)
				continue;
			/*
			 * Shared channels are never "busy", so without rotation
			 * every spread claim would land on the first channel of
			 * each group and the rest of the device would idle. A
			 * group's cursor advances each time a channel is TAKEN
			 * from it, so groups that are used at different rates
			 * never march in step and skip the same channels.
			 */
			if (shared)
				idx = grp->first + (unsigned int)atomic_inc_return(
					&group_chan_cursor[grp - groups]) % grp->nr;
			else
				idx = grp->first + r;
			if (idx >= limit || (mask & BIT(idx)))
				continue;
			if (take_channel(idx)) {
				mask |= BIT(idx);
				(*gotp)++;
			}
		}
	}
	return mask;
}

/**
 * mm_offload_dma_claim_spread - claim channels across distinct devices.
 * @want: channels wanted.
 * @nid: NUMA node the copy is written to; groups on it are taken first,
 *       since a stripe on a remote device makes that share of the copy
 *       wait on remote-link bandwidth.
 *
 * Breadth-first: one channel from every eligible group, then a second
 * from each, ... so @want channels land on as many distinct devices as
 * possible. Used to stripe one large copy over several devices.
 *
 * The devices of @nid are the only ones used while any of them can be
 * had, even when they hold fewer channels than @want: a share of the
 * copy on a remote device not only waits on the remote link, its
 * cache-allocating writes land in the remote socket's LLC, where the
 * consumer of the data is not. Only a node without a device of its own
 * - a CPU-less memory node such as a CXL tier, where a migration
 * destination often lives - falls back to the groups on other nodes
 * rather than refusing: every device is remote to that node, and
 * spreading over all of them still beats using one.
 *
 * Return: bitmask of claimed channel indices, 0 if every channel is
 * busy.
 */
unsigned long mm_offload_dma_claim_spread(unsigned int want, int nid)
{
	unsigned int policy = READ_ONCE(cross_node);
	bool local_first = policy != MM_OFFLOAD_DMA_REMOTE;
	unsigned long mask;
	unsigned int got = 0;

	mask = spread_pass(want, nid, local_first, 0, &got);
	if (!got || (policy == MM_OFFLOAD_DMA_LOCAL_FIRST && got < want))
		mask = spread_pass(want, nid, !local_first, mask, &got);
	return mask;
}
EXPORT_SYMBOL_GPL(mm_offload_dma_claim_spread);

void mm_offload_dma_release(unsigned long mask)
{
	unsigned int i;

	if (shared)
		return;
	for_each_set_bit(i, &mask, MM_OFFLOAD_DMA_MAX_CHANNELS)
		mutex_unlock(&channels[i].lock);
}
EXPORT_SYMBOL_GPL(mm_offload_dma_release);

bool mm_offload_dma_shared(void)
{
	return shared;
}
EXPORT_SYMBOL_GPL(mm_offload_dma_shared);

/**
 * mm_offload_dma_admit - reserve @nr descriptor slots on channel @idx.
 * @wait: sleep for budget, up to admit_timeout_ms in total, or fail at once.
 *
 * Call before preparing a descriptor, from process context. Return: 0
 * with the slots held (retire them when the descriptor completes, or at
 * once if it is never submitted), -EBUSY if there is no budget and the
 * caller would not wait, -ETIMEDOUT if it did not free up in time - the
 * caller then does the work on the CPU. A caller that holds a lock others
 * need (a page fault under mmap_lock) should not wait.
 */
int mm_offload_dma_admit(unsigned int idx, unsigned int nr, bool wait)
{
	atomic_t *inflight = &channels[idx].inflight;
	unsigned int timeout = READ_ONCE(admit_timeout_ms);
	unsigned long deadline = jiffies + msecs_to_jiffies(timeout);

	for (;;) {
		int budget = (int)clamp_t(unsigned int, READ_ONCE(chan_budget),
					  1u, (unsigned int)INT_MAX - 1);
		int cur = atomic_read(inflight);
		long left;

		while (cur + (int)nr <= budget) {
			int old = atomic_cmpxchg(inflight, cur, cur + nr);

			if (old == cur) {
				atomic_long_inc(&channels[idx].admits);
				return 0;
			}
			cur = old;
		}
		if (!wait || !timeout)
			return -EBUSY;
		/* One deadline for the whole attempt, however often it retries. */
		left = (long)deadline - (long)jiffies;
		if (left <= 0) {
			atomic_long_inc(&channels[idx].timeouts);
			return -ETIMEDOUT;
		}
		atomic_long_inc(&channels[idx].waits);
		wait_event_timeout(channels[idx].waitq,
				   atomic_read(inflight) + (int)nr <=
					(int)clamp_t(unsigned int,
						     READ_ONCE(chan_budget), 1u,
						     (unsigned int)INT_MAX - 1),
				   left);
	}
}
EXPORT_SYMBOL_GPL(mm_offload_dma_admit);

/**
 * mm_offload_dma_prep_wait - the engine refused an admitted descriptor.
 *
 * The budget is set below the work queue's ring, but the ring holds a
 * slot until the completion callback - which retires the budget first -
 * has returned, and the ring may be smaller than the budget on another
 * configuration. So a refused prep is transient whenever the channel has
 * work in flight. Called after the caller returned its slot: waits, up
 * to admit_timeout_ms, for the in-flight count to drop below what it
 * saw. Return: 0 to admit and prepare again, -ETIMEDOUT to give up.
 */
int mm_offload_dma_prep_wait(unsigned int idx)
{
	int seq = atomic_read(&channels[idx].completions);
	unsigned int timeout = READ_ONCE(admit_timeout_ms);

	if (atomic_read(&channels[idx].inflight) <= 0 || !timeout)
		return -ETIMEDOUT;
	atomic_long_inc(&channels[idx].waits);
	if (!wait_event_timeout(channels[idx].waitq,
				atomic_read(&channels[idx].completions) != seq,
				msecs_to_jiffies(timeout))) {
		atomic_long_inc(&channels[idx].timeouts);
		return -ETIMEDOUT;
	}
	return 0;
}
EXPORT_SYMBOL_GPL(mm_offload_dma_prep_wait);

/**
 * mm_offload_dma_retire - return @nr slots on channel @idx.
 *
 * For a descriptor that was never queued (a refused prep, a failed
 * submit). Completions use mm_offload_dma_complete() instead.
 */
void mm_offload_dma_retire(unsigned int idx, unsigned int nr)
{
	atomic_sub(nr, &channels[idx].inflight);
	wake_up(&channels[idx].waitq);
}
EXPORT_SYMBOL_GPL(mm_offload_dma_retire);

/**
 * mm_offload_dma_complete - return @nr slots the engine has finished with.
 *
 * Safe from the completion callback. Counted separately from retires so a
 * submitter waiting for the ring to drain sees engine progress, not a
 * peer's own refused prep bouncing the count.
 */
void mm_offload_dma_complete(unsigned int idx, unsigned int nr)
{
	atomic_sub(nr, &channels[idx].inflight);
	atomic_add(nr, &channels[idx].completions);
	wake_up(&channels[idx].waitq);
}
EXPORT_SYMBOL_GPL(mm_offload_dma_complete);

static int stats_get(char *buf, const struct kernel_param *kp)
{
	/* Acquire: pairs with the release store that publishes the pool. */
	unsigned int n = smp_load_acquire(&nr_channels), i;
	int pos = 0;

	for (i = 0; i < n; i++)
		pos += sysfs_emit_at(buf, pos,
				     "chan%u %s inflight %d completions %d admits %ld waits %ld timeouts %ld\n",
				     i, dev_name(channels[i].dev),
				     atomic_read(&channels[i].inflight),
				     atomic_read(&channels[i].completions),
				     atomic_long_read(&channels[i].admits),
				     atomic_long_read(&channels[i].waits),
				     atomic_long_read(&channels[i].timeouts));
	return pos;
}

static const struct kernel_param_ops stats_ops = { .get = stats_get };
module_param_cb(stats, &stats_ops, NULL, 0444);
MODULE_PARM_DESC(stats, "Per-channel admission counters");

/*
 * PFN-keyed standing mapping cache, shared by the providers.
 *
 * Every copy a provider offloads maps its folios, hands them to the
 * engines and unmaps them again, so each one pays for an IOVA allocation
 * and its teardown. Under a translated IOMMU domain that is not merely
 * CPU cost: an allocation larger than the per-CPU IOVA rcache ceiling
 * (dma_opt_mapping_size(), 128K on a stock kernel) is served from the
 * domain rbtree, the addresses it hands back drift as that space
 * fragments, and the IOMMU stops holding the translations the engines
 * are about to use. Measured on two DSA devices, offloaded migration
 * starts at ~60 GiB/s and settles at ~30 after an hour of traffic;
 * iommu.passthrough=1 removes the loss entirely.
 *
 * So keep the mappings. A per-device xarray keyed by segment-head PFN
 * holds standing mappings of the folios the providers have touched. The
 * folios come from bounded sets - an hstate pool, a guest's memory, the
 * pages a restore populates - so the same pages come back, and with
 * them the same device addresses. dma_unmap leaves the copy path
 * entirely: it runs on CLOCK eviction under a byte cap, or on a flush.
 *
 * Entries are mapped DMA_BIDIRECTIONAL because a folio that is a source
 * this time is a destination the next, and one entry serving both roles
 * halves both the entry count and the IOVA footprint.
 *
 * Correctness: struct page to physical address is immutable, and the
 * providers only issue descriptors against folios they hold - a
 * migration source unmapped and locked, a destination allocated and not
 * yet visible to anyone - so a stale entry cannot misdirect a copy.
 * What staleness costs is IOVA space and a standing window in which the
 * device could reach a page since released: the trade page_pool already
 * makes for its persistent NIC mappings, bounded here by the cap and
 * revocable by writing map_cache_stats. Pages a provider does not own
 * for the life of the entry - user memory it has merely pinned - must
 * not go through the cache.
 *
 * A standing mapping is not the device's to keep between copies, so the
 * ownership transfers the per-copy map and unmap used to perform happen
 * explicitly in the caller instead: the source is handed to the device
 * (dma_sync_single_for_device) before its descriptors are submitted and
 * the destination handed back (dma_sync_single_for_cpu) once they have
 * landed. On a coherent device both compile away.
 *
 * Entry lifetime is a bias refcount: refs = 1 for the cache plus one
 * per in-flight copy. Lookup takes its reference with
 * atomic_inc_not_zero() under RCU, and eviction erases the entry and
 * drops the bias, so a mapping outlives eviction until the last copy
 * using it completes and the last dropper unmaps.
 *
 * The caches live as long as the pool has a user: the devices they map
 * against are the pool's, and the last mm_offload_dma_pool_put() frees
 * them before it releases the channels.
 */

/* Largest segment an entry covers; also the largest descriptor. */
#define MAP_QUANTUM_MAX		SZ_2M
/* Entries one eviction sweep may visit, bounding submit-path latency. */
#define MAP_EVICT_BUDGET	64
/* Devices that can have a cache; the pool holds at most this many. */
#define MAP_CACHE_DEVS		(MM_OFFLOAD_DMA_MAX_CHANNELS / 2)

/* One standing mapping: a quantum-sized segment of one folio. */
struct mm_offload_dma_map {
	unsigned long		pfn;		/* segment-head PFN: the key */
	dma_addr_t		dma;
	unsigned int		size;
	atomic_t		refs;		/* cache bias + in-flight copies */
	bool			referenced;	/* CLOCK second chance */
	struct device		*dev;
	struct rcu_head		rcu;
};

struct map_cache {
	struct xarray		xa;
	spinlock_t		lock;		/* serialises the CLOCK sweep */
	unsigned long		hand;		/* where the sweep resumes */
	struct device		*dev;
	size_t			quantum;
	atomic64_t		covered;	/* bytes mapped through the tree */
	atomic64_t		hits;
	atomic64_t		misses;
	atomic64_t		inserts;
	atomic64_t		insert_fails;
	atomic64_t		range_fallbacks;
	atomic64_t		evictions;
	atomic64_t		ref_skips;	/* sweep passed an in-flight entry */
};

static bool map_cache = true;
static unsigned long map_cache_cap_mb = 4096;
static unsigned long map_quantum_kb = 128;	/* 0: dma_opt_mapping_size() */
static unsigned long map_cache_min_kb = 64;

static struct map_cache *map_caches[MAP_CACHE_DEVS];
static struct device *map_cache_devs[MAP_CACHE_DEVS];
static DEFINE_SPINLOCK(map_cache_reg_lock);
/* Flush and free against the stats walk. */
static DEFINE_MUTEX(map_mutex);

/*
 * Segment size entries are carved into. Keeping it at or below
 * dma_opt_mapping_size() puts every IOVA allocation the cache does make
 * in a per-CPU rcache size class; raising it to the folio size trades
 * that for fewer, larger entries and larger descriptors, which pays
 * once the working set fits the cap and the allocations stop.
 */
static size_t map_quantum_of(struct device *dev)
{
	size_t q = (size_t)READ_ONCE(map_quantum_kb) << 10;

	if (!q)
		q = dma_opt_mapping_size(dev);
	if (!q || q > MAP_QUANTUM_MAX)
		q = MAP_QUANTUM_MAX;
	if (q < PAGE_SIZE)
		q = PAGE_SIZE;

	return 1UL << (fls_long(q) - 1);	/* power of two, rounded down */
}

/* The cache of @dev, created on first use. NULL: caching is off. */
static struct map_cache *map_cache_get(struct device *dev)
{
	struct map_cache *c;
	int i;

	if (!READ_ONCE(map_cache))
		return NULL;

	for (i = 0; i < MAP_CACHE_DEVS; i++) {
		/*
		 * Pairs with the store-release below: a published device
		 * means its cache is fully initialised.
		 */
		if (smp_load_acquire(&map_cache_devs[i]) == dev)
			return map_caches[i];
		if (!READ_ONCE(map_cache_devs[i]))
			break;
	}

	c = kzalloc(sizeof(*c), GFP_KERNEL);
	if (!c)
		return NULL;
	xa_init(&c->xa);
	spin_lock_init(&c->lock);
	/*
	 * Entries outlive any one copy and unmap against this device, so
	 * hold it for as long as the cache does.
	 */
	c->dev = get_device(dev);
	c->quantum = map_quantum_of(dev);

	spin_lock(&map_cache_reg_lock);
	for (i = 0; i < MAP_CACHE_DEVS; i++) {
		if (map_cache_devs[i] == dev) {		/* lost the race */
			spin_unlock(&map_cache_reg_lock);
			put_device(c->dev);
			kfree(c);
			return map_caches[i];
		}
		if (!map_cache_devs[i]) {
			map_caches[i] = c;
			/* Publish the cache before the device it answers to. */
			smp_store_release(&map_cache_devs[i], dev);
			spin_unlock(&map_cache_reg_lock);
			return c;
		}
	}
	spin_unlock(&map_cache_reg_lock);
	put_device(c->dev);
	kfree(c);
	return NULL;
}

/**
 * mm_offload_dma_map_quantum - segment size of @dev's standing mappings.
 *
 * Return: the size a provider should cut its ranges at so each one is a
 * whole cache entry, or 0 when the cache is off and ranges are mapped
 * per copy.
 */
size_t mm_offload_dma_map_quantum(struct device *dev)
{
	struct map_cache *c = map_cache_get(dev);

	return c ? c->quantum : 0;
}
EXPORT_SYMBOL_GPL(mm_offload_dma_map_quantum);

/**
 * mm_offload_dma_map_put - drop one reference on a standing mapping.
 *
 * The last dropper unmaps and frees, which is safe from any context the
 * copy path completes in.
 */
void mm_offload_dma_map_put(struct mm_offload_dma_map *e)
{
	if (!atomic_dec_and_test(&e->refs))
		return;
	dma_unmap_page_attrs(e->dev, e->dma, e->size, DMA_BIDIRECTIONAL,
			     DMA_ATTR_SKIP_CPU_SYNC);
	kfree_rcu(e, rcu);
}
EXPORT_SYMBOL_GPL(mm_offload_dma_map_put);

/*
 * CLOCK sweep: resume the hand where it stopped, give referenced
 * entries a second chance so a one-pass scan cannot flush the recycling
 * working set, and skip entries a copy is still using. Runs on the
 * submit path once an insert pushes the cache past its cap, so the
 * trylock - another submitter is already sweeping - and the visit
 * budget both bound the latency it adds.
 */
static void map_cache_evict(struct map_cache *c, u64 cap)
{
	/*
	 * Evict down to a low-water mark rather than exactly to the cap.
	 * Stopping at the cap leaves the next insert over it again, so a
	 * working set that happens to sit at the cap pays a sweep and an
	 * unmap per insert and thrashes: measured at 775k evictions and
	 * barely half the throughput, against none and full speed once the
	 * cap was raised past the same working set.
	 */
	u64 low = cap - (cap >> 3);
	struct mm_offload_dma_map *e;
	unsigned long index;
	int budget = MAP_EVICT_BUDGET;
	int pass;

	if (!spin_trylock(&c->lock))
		return;

	/*
	 * Entries are freed with kfree_rcu(), and the iterator hands them
	 * back outside any read-side section, so hold one across the sweep:
	 * under preemptible RCU c->lock does not stand in for it, and a
	 * grace period elapsing mid-sweep would leave us reading freed
	 * entries.
	 */
	rcu_read_lock();
	for (pass = 0; pass < 2 && atomic64_read(&c->covered) > (s64)low; pass++) {
		unsigned long start = pass ? 0 : c->hand;

		xa_for_each_start(&c->xa, index, e, start) {
			if (--budget <= 0) {
				c->hand = index + 1;
				goto out;
			}
			if (READ_ONCE(e->referenced)) {
				WRITE_ONCE(e->referenced, false);
			} else if (atomic_read(&e->refs) > 1) {
				atomic64_inc(&c->ref_skips);
			} else if (xa_cmpxchg(&c->xa, index, e, NULL,
					      GFP_NOWAIT | __GFP_NOWARN) == e) {
				/*
				 * Identity-checked: the displace path in
				 * mm_offload_dma_map_get() removes entries
				 * without this lock, so only the remover that
				 * actually took *this* entry out may drop its
				 * bias.
				 */
				atomic64_sub(e->size, &c->covered);
				atomic64_inc(&c->evictions);
				mm_offload_dma_map_put(e);	/* the cache bias */
				if (atomic64_read(&c->covered) <= (s64)low) {
					c->hand = index + 1;
					goto out;
				}
			}
		}
		c->hand = 0;
	}
out:
	rcu_read_unlock();
	spin_unlock(&c->lock);
}

/*
 * Erase every entry, dropping the cache's bias; in-flight copies keep
 * their own reference and unmap when they finish.
 */
static void map_cache_flush(struct map_cache *c)
{
	struct mm_offload_dma_map *e;
	unsigned long index;

	/* As in the sweep: the entries are only safe to touch under RCU. */
	rcu_read_lock();
	xa_for_each(&c->xa, index, e) {
		if (xa_cmpxchg(&c->xa, index, e, NULL,
			       GFP_NOWAIT | __GFP_NOWARN) != e)
			continue;	/* a sweep got there first */
		atomic64_sub(e->size, &c->covered);
		mm_offload_dma_map_put(e);
	}
	rcu_read_unlock();

	spin_lock(&c->lock);
	c->hand = 0;
	spin_unlock(&c->lock);
}

static void map_cache_flush_all(void)
{
	int i;

	for (i = 0; i < MAP_CACHE_DEVS; i++) {
		/* Pairs with the store-release in map_cache_get(). */
		if (!smp_load_acquire(&map_cache_devs[i]))
			break;
		map_cache_flush(map_caches[i]);
	}
}

/**
 * mm_offload_dma_map_flush - revoke every standing mapping, all devices.
 *
 * In-flight copies keep theirs until they complete.
 */
void mm_offload_dma_map_flush(void)
{
	mutex_lock(&map_mutex);
	map_cache_flush_all();
	mutex_unlock(&map_mutex);
}
EXPORT_SYMBOL_GPL(mm_offload_dma_map_flush);

/*
 * The pool's last user is gone: no copy can be in flight, so the caches
 * go too, before the channels - and with them the pool's handle on the
 * devices - are released.
 */
static void map_cache_free_all(void)
{
	int i;

	mutex_lock(&map_mutex);
	map_cache_flush_all();
	rcu_barrier();		/* the entries' kfree_rcu() */

	spin_lock(&map_cache_reg_lock);
	for (i = 0; i < MAP_CACHE_DEVS; i++) {
		struct map_cache *c = map_caches[i];

		if (!c)
			continue;
		/* flush emptied it; anything left would be an unmapped leak */
		WARN_ON_ONCE(!xa_empty(&c->xa));
		xa_destroy(&c->xa);
		put_device(c->dev);
		kfree(c);
		map_caches[i] = NULL;
		map_cache_devs[i] = NULL;
	}
	spin_unlock(&map_cache_reg_lock);
	mutex_unlock(&map_mutex);
}

/**
 * mm_offload_dma_map_get - device address of a folio range, cached.
 * @dev: the DMA device the copy will run on.
 * @folio: a folio the caller holds for the life of the copy.
 * @offset: start within the folio.
 * @len: bytes; must not cross a quantum boundary (see _quantum()).
 * @dma: set to the device address of @offset.
 *
 * Resolve [offset, offset + len) of @folio to a device address, mapping
 * and caching the segment that covers it if this is the first time.
 * Return: the entry with an in-flight reference taken (drop it with
 * mm_offload_dma_map_put() once the descriptors have landed), or NULL
 * when the caller should map the range itself - never an error, only
 * slower. The entry is mapped bidirectional without CPU sync: the
 * caller syncs for the device before submitting and for the CPU after.
 */
struct mm_offload_dma_map *mm_offload_dma_map_get(struct device *dev,
						  struct folio *folio,
						  size_t offset, size_t len,
						  dma_addr_t *dma)
{
	u64 cap = (u64)READ_ONCE(map_cache_cap_mb) << 20;
	struct map_cache *c = map_cache_get(dev);
	size_t seg_base, seg_len, rel, map_len;
	struct mm_offload_dma_map *e, *old;
	unsigned long pfn;
	dma_addr_t base;

	if (!c || !cap)
		return NULL;

	map_len = folio_size(folio);
	seg_base = offset & ~(c->quantum - 1);
	seg_len = min_t(size_t, c->quantum, map_len - seg_base);
	rel = offset - seg_base;
	if (unlikely(rel + len > seg_len)) {
		atomic64_inc(&c->range_fallbacks);
		return NULL;
	}
	/*
	 * Below the IOVA rcache ceiling a mapping is served from a per-CPU
	 * magazine and costs little to make and unmake, so small folios are
	 * left to map themselves rather than filling the cache with entries
	 * that save nothing.
	 */
	if (seg_len < ((size_t)READ_ONCE(map_cache_min_kb) << 10))
		return NULL;

	pfn = folio_pfn(folio) + (seg_base >> PAGE_SHIFT);

	rcu_read_lock();
	e = xa_load(&c->xa, pfn);
	if (e && atomic_inc_not_zero(&e->refs)) {
		rcu_read_unlock();
		if (unlikely(rel + len > e->size)) {
			/*
			 * The PFN came back under a larger folio than the one
			 * the entry was mapped for. Displace it and remap
			 * under the same key; anyone still copying through the
			 * old mapping is held up by the bias protocol.
			 */
			if (xa_cmpxchg(&c->xa, pfn, e, NULL,
				       GFP_NOWAIT | __GFP_NOWARN) == e) {
				atomic64_sub(e->size, &c->covered);
				mm_offload_dma_map_put(e);	/* the cache bias */
			}
			mm_offload_dma_map_put(e);		/* the lookup */
			atomic64_inc(&c->range_fallbacks);
			goto miss;
		}
		WRITE_ONCE(e->referenced, true);
		atomic64_inc(&c->hits);
		*dma = e->dma + rel;
		return e;
	}
	rcu_read_unlock();

miss:
	atomic64_inc(&c->misses);
	e = kmalloc_obj(*e, GFP_NOWAIT | __GFP_NOWARN);
	if (!e)
		goto fail;
	base = dma_map_page_attrs(c->dev, folio_page(folio, seg_base >> PAGE_SHIFT),
				  0, seg_len, DMA_BIDIRECTIONAL,
				  DMA_ATTR_SKIP_CPU_SYNC);
	if (dma_mapping_error(c->dev, base)) {
		kfree(e);
		goto fail;
	}
	e->pfn = pfn;
	e->dma = base;
	e->size = seg_len;
	e->dev = c->dev;
	e->referenced = true;
	atomic_set(&e->refs, 2);	/* the cache bias + this copy */

	/*
	 * Charge before publishing: once the entry is visible a sweep may
	 * remove it, and a remover that subtracted bytes never added would
	 * drive the counter negative and the cap comparison with it.
	 */
	atomic64_add(seg_len, &c->covered);
	rcu_read_lock();
	old = xa_cmpxchg(&c->xa, pfn, NULL, e, GFP_NOWAIT | __GFP_NOWARN);
	if (old) {
		/* Lost an insert race, or the xarray could not allocate. */
		atomic64_sub(seg_len, &c->covered);
		dma_unmap_page_attrs(c->dev, base, seg_len, DMA_BIDIRECTIONAL,
				     DMA_ATTR_SKIP_CPU_SYNC);
		kfree(e);
		if (!xa_is_err(old) && atomic_inc_not_zero(&old->refs)) {
			rcu_read_unlock();
			if (unlikely(rel + len > old->size)) {
				atomic64_inc(&c->range_fallbacks);
				mm_offload_dma_map_put(old);
				return NULL;
			}
			WRITE_ONCE(old->referenced, true);
			atomic64_inc(&c->hits);
			*dma = old->dma + rel;
			return old;
		}
		rcu_read_unlock();
		goto fail;
	}
	rcu_read_unlock();
	atomic64_inc(&c->inserts);

	/*
	 * Pairs with the barrier in the setters that revoke the cache:
	 * either the flush that follows the revoke sees this entry, or this
	 * sees the revoke and takes the entry back out itself. The copy's
	 * own reference keeps the mapping until the copy completes.
	 */
	smp_mb();
	if (unlikely(!READ_ONCE(map_cache) || !READ_ONCE(map_cache_cap_mb))) {
		if (xa_cmpxchg(&c->xa, pfn, e, NULL, GFP_NOWAIT) == e) {
			atomic64_sub(seg_len, &c->covered);
			mm_offload_dma_map_put(e);
		}
		*dma = e->dma + rel;
		return e;
	}
	if (atomic64_read(&c->covered) > (s64)cap)
		map_cache_evict(c, cap);
	*dma = e->dma + rel;
	return e;
fail:
	atomic64_inc(&c->insert_fails);
	return NULL;
}
EXPORT_SYMBOL_GPL(mm_offload_dma_map_get);

/*
 * Turning the cache off, or capping it at nothing, has to revoke the
 * mappings it is already holding - otherwise "off" leaves standing
 * device access to every folio it ever touched. A copy that read the
 * old setting may still be inserting, so the barrier orders the new
 * setting before the flush and pairs with the check an inserter makes
 * after publishing: one of the two sees the other's entry.
 */
static int map_cache_off_set(const char *val, const struct kernel_param *kp)
{
	int ret = param_set_bool(val, kp);

	if (!ret && !READ_ONCE(map_cache)) {
		smp_mb();
		mm_offload_dma_map_flush();
	}
	return ret;
}

static const struct kernel_param_ops map_cache_ops = {
	.set = map_cache_off_set,
	.get = param_get_bool,
};
module_param_cb(map_cache, &map_cache_ops, &map_cache, 0644);
MODULE_PARM_DESC(map_cache,
		 "Keep DMA mappings of the folios providers copy instead of remapping each copy");

static int map_cache_cap_set(const char *val, const struct kernel_param *kp)
{
	int ret = param_set_ulong(val, kp);

	if (!ret && !READ_ONCE(map_cache_cap_mb)) {
		smp_mb();	/* as in map_cache_off_set() */
		mm_offload_dma_map_flush();
	}
	return ret;
}

static const struct kernel_param_ops map_cache_cap_ops = {
	.set = map_cache_cap_set,
	.get = param_get_ulong,
};
module_param_cb(map_cache_cap_mb, &map_cache_cap_ops, &map_cache_cap_mb, 0644);
MODULE_PARM_DESC(map_cache_cap_mb,
		 "Bytes of standing mappings to keep per device, MiB (0 disables and revokes)");

module_param(map_cache_min_kb, ulong, 0644);
MODULE_PARM_DESC(map_cache_min_kb,
		 "Smallest mapping worth caching, KiB (below the IOVA rcache ceiling it saves nothing)");

module_param(map_quantum_kb, ulong, 0644);
MODULE_PARM_DESC(map_quantum_kb,
		 "Segment size cached mappings are carved into, KiB (default 128, the IOVA rcache ceiling; 0: dma_opt_mapping_size; fixed per cache, takes effect when the pool is next acquired)");

/* map_cache_stats: per-device cache counters; any write flushes every cache */
static int map_cache_stats_get(char *buffer, const struct kernel_param *kp)
{
	int len = 0, i;

	/* Against map_cache_free_all(), which frees what we walk. */
	mutex_lock(&map_mutex);
	for (i = 0; i < MAP_CACHE_DEVS; i++) {
		struct map_cache *c;

		/* Pairs with the store-release in map_cache_get(). */
		if (!smp_load_acquire(&map_cache_devs[i]))
			break;
		c = map_caches[i];
		len += sysfs_emit_at(buffer, len,
				     "%s quantum_kb %zu covered_kb %lld hits %lld misses %lld inserts %lld insert_fails %lld range_fallbacks %lld evictions %lld ref_skips %lld\n",
				     dev_name(c->dev), c->quantum >> 10,
				     atomic64_read(&c->covered) >> 10,
				     atomic64_read(&c->hits),
				     atomic64_read(&c->misses),
				     atomic64_read(&c->inserts),
				     atomic64_read(&c->insert_fails),
				     atomic64_read(&c->range_fallbacks),
				     atomic64_read(&c->evictions),
				     atomic64_read(&c->ref_skips));
	}
	if (!len)
		len = sysfs_emit(buffer, "none\n");
	mutex_unlock(&map_mutex);
	return len;
}

static int map_cache_stats_set(const char *val, const struct kernel_param *kp)
{
	mm_offload_dma_map_flush();
	return 0;
}

static const struct kernel_param_ops map_cache_stats_ops = {
	.set = map_cache_stats_set,
	.get = map_cache_stats_get,
};
module_param_cb(map_cache_stats, &map_cache_stats_ops, NULL, 0644);
MODULE_PARM_DESC(map_cache_stats, "Standing mapping cache counters per device (write to flush)");


/*
 * ---- usage counters ----
 *
 * Two views of the same traffic. The software view is per operation
 * class, the mm_offload function the bytes served (folio clears,
 * migration copies, UFFDIO_COPY fills), counted by the providers at
 * submission and completion with the submit-to-completion time summed,
 * so a mean device latency per class falls out. The hardware view is the
 * devices' own perfmon counters, read through the idxd PMU as kernel
 * perf events: data processed and written per device (the engine
 * category, which has no work-queue filter, so it includes user-space
 * submitters such as DTO), and descriptor counts per operation type on
 * the pool's own work queues and on every queue. When nothing else uses
 * a device the two views agree on bytes; the difference is other
 * submitters' share. Both are running totals from the pool's first
 * acquisition; `window` reports the deltas and the rates since it was
 * last written to.
 */
struct op_stat {
	atomic64_t submitted;
	atomic64_t completed;
	atomic64_t failed;
	atomic64_t bytes_submitted;
	atomic64_t bytes_completed;
	atomic64_t busy_ns;		/* submit to completion, summed */
	atomic_t inflight;
};

static struct op_stat op_stats[MM_OFFLOAD_DMA_OP_MAX];
static const char *const op_names[MM_OFFLOAD_DMA_OP_MAX] = {
	[MM_OFFLOAD_DMA_OP_CLEAR] = "clear",
	[MM_OFFLOAD_DMA_OP_MIGRATE] = "migrate",
	[MM_OFFLOAD_DMA_OP_UFFD] = "uffd_copy",
};

void mm_offload_dma_stat_submit(enum mm_offload_dma_op op, size_t bytes)
{
	struct op_stat *s = &op_stats[op];

	atomic64_inc(&s->submitted);
	atomic64_add(bytes, &s->bytes_submitted);
	atomic_inc(&s->inflight);
}
EXPORT_SYMBOL_GPL(mm_offload_dma_stat_submit);

void mm_offload_dma_stat_complete(enum mm_offload_dma_op op, size_t bytes,
				  ktime_t submitted, bool ok)
{
	struct op_stat *s = &op_stats[op];

	atomic_dec(&s->inflight);
	if (ok) {
		atomic64_inc(&s->completed);
		atomic64_add(bytes, &s->bytes_completed);
	} else {
		atomic64_inc(&s->failed);
	}
	atomic64_add(ktime_to_ns(ktime_sub(ktime_get(), submitted)), &s->busy_ns);
}
EXPORT_SYMBOL_GPL(mm_offload_dma_stat_complete);

/*
 * Hardware counters, one set per device the pool holds a channel on.
 * perf attr.config is the idxd PMU's format: event category in bits
 * 0-3, event mask from bit 4; config1 bits 0-31 are the WQ filter.
 */
enum hw_ev {
	HW_CL_PROCESSED,	/* EV_CL_PROCESSED: input data, 32-byte units */
	HW_CL_WRITTEN,		/* EV_CL_WRITE: data written, 32-byte units */
	HW_MOVE_POOL,		/* EV_MEM_MOVE descriptors on the pool's WQs */
	HW_FILL_POOL,		/* EV_FILL descriptors on the pool's WQs */
	HW_CPL_POOL,		/* EV_NUM_CPL_SUCC on the pool's WQs */
	HW_MOVE_ALL,		/* EV_MEM_MOVE, every WQ */
	HW_FILL_ALL,		/* EV_FILL, every WQ */
	HW_EV_MAX,
};

static const struct {
	const char *name;
	u8 category;
	u32 event;
	bool wq_filter;
	bool cachelines;	/* counts 32-byte units, reported as bytes */
} hw_evs[HW_EV_MAX] = {
	[HW_CL_PROCESSED] = { "processed_bytes", 1, 0x1, false, true },
	[HW_CL_WRITTEN] = { "written_bytes", 1, 0x2, false, true },
	[HW_MOVE_POOL] = { "memmove_pool", 3, 0x8, true, false },
	[HW_FILL_POOL] = { "fill_pool", 3, 0x10, true, false },
	[HW_CPL_POOL] = { "completions_pool", 4, 0x10, true, false },
	[HW_MOVE_ALL] = { "memmove_all", 3, 0x8, false, false },
	[HW_FILL_ALL] = { "fill_all", 3, 0x10, false, false },
};

#define HW_MAX_DEVS	(MM_OFFLOAD_DMA_MAX_CHANNELS / 2)

struct hw_dev {
	struct device *dev;
	int pmu_type;
	u32 wq_mask;		/* the pool's WQs on this device */
	struct perf_event *ev[HW_EV_MAX];
};

/* Under pool_mutex. */
static struct hw_dev hw_devs[HW_MAX_DEVS];
static unsigned int nr_hw_devs;
static bool hw_create_failed_once;

static void hw_events_create(void)
{
	unsigned int i, d, e;

	for (i = 0; i < nr_channels; i++) {
		struct dma_chan *chan = channels[i].chan;
		int type = idxd_dma_chan_pmu_type(chan);
		int wq = idxd_dma_chan_wq_id(chan);
		struct hw_dev *hd = NULL;

		if (type < 0)
			continue;
		for (d = 0; d < nr_hw_devs; d++)
			if (hw_devs[d].dev == channels[i].dev)
				hd = &hw_devs[d];
		if (!hd) {
			if (nr_hw_devs == HW_MAX_DEVS)
				continue;
			hd = &hw_devs[nr_hw_devs++];
			memset(hd, 0, sizeof(*hd));
			hd->dev = channels[i].dev;
			hd->pmu_type = type;
		}
		if (wq >= 0 && wq < 32)
			hd->wq_mask |= BIT(wq);
	}

	for (d = 0; d < nr_hw_devs; d++) {
		struct hw_dev *hd = &hw_devs[d];

		if (hd->ev[0])
			continue;	/* created on an earlier acquisition */
		for (e = 0; e < HW_EV_MAX; e++) {
			struct perf_event_attr attr = {
				.type = hd->pmu_type,
				.size = sizeof(attr),
				.config = hw_evs[e].category |
					  ((u64)hw_evs[e].event << 4),
				.config1 = hw_evs[e].wq_filter ? hd->wq_mask : 0,
				.pinned = 1,
			};
			struct perf_event *ev;

			/* The PMU's cpumask is CPU 0: uncore-style, counted there. */
			ev = perf_event_create_kernel_counter(&attr, 0, NULL, NULL, NULL);
			if (IS_ERR(ev)) {
				if (!hw_create_failed_once)
					pr_warn("mm_offload_dma: %s: perfmon event %s unavailable (%ld)\n",
						dev_name(hd->dev), hw_evs[e].name, PTR_ERR(ev));
				hw_create_failed_once = true;
				ev = NULL;
			}
			hd->ev[e] = ev;
		}
	}
}

static void hw_events_release(void)
{
	unsigned int d, e;

	for (d = 0; d < nr_hw_devs; d++)
		for (e = 0; e < HW_EV_MAX; e++)
			if (hw_devs[d].ev[e]) {
				perf_event_release_kernel(hw_devs[d].ev[e]);
				hw_devs[d].ev[e] = NULL;
			}
	nr_hw_devs = 0;
}

static u64 hw_read(const struct hw_dev *hd, unsigned int e)
{
	u64 enabled, running, v;

	if (!hd->ev[e])
		return 0;
	v = perf_event_read_value(hd->ev[e], &enabled, &running);
	return hw_evs[e].cachelines ? v * 32 : v;
}

/* Everything the window is measured against, taken under pool_mutex. */
struct usage_snap {
	ktime_t t;
	u64 ops[MM_OFFLOAD_DMA_OP_MAX];
	u64 bytes[MM_OFFLOAD_DMA_OP_MAX];
	u64 busy_ns[MM_OFFLOAD_DMA_OP_MAX];
	u64 hw[HW_MAX_DEVS][HW_EV_MAX];
};

static struct usage_snap window_start;

static void usage_snap(struct usage_snap *snap)
{
	unsigned int o, d, e;

	snap->t = ktime_get();
	for (o = 0; o < MM_OFFLOAD_DMA_OP_MAX; o++) {
		snap->ops[o] = atomic64_read(&op_stats[o].completed);
		snap->bytes[o] = atomic64_read(&op_stats[o].bytes_completed);
		snap->busy_ns[o] = atomic64_read(&op_stats[o].busy_ns);
	}
	for (d = 0; d < nr_hw_devs; d++)
		for (e = 0; e < HW_EV_MAX; e++)
			snap->hw[d][e] = hw_read(&hw_devs[d], e);
}

/* MiB/s and ops/s from a byte or count delta over @ns. */
static u64 per_second(u64 delta, u64 ns)
{
	return ns ? div64_u64(delta * NSEC_PER_SEC, ns) : 0;
}

static ssize_t totals_show(struct kobject *kobj, struct kobj_attribute *attr,
			   char *buf)
{
	unsigned int o, d, e;
	int len = 0;

	mutex_lock(&pool_mutex);
	for (o = 0; o < MM_OFFLOAD_DMA_OP_MAX; o++) {
		struct op_stat *s = &op_stats[o];
		u64 ops = atomic64_read(&s->completed);
		u64 busy = atomic64_read(&s->busy_ns);

		len += sysfs_emit_at(buf, len,
				     "%-10s ops %llu bytes %llu failed %llu inflight %d mean_us %llu\n",
				     op_names[o], ops, (u64)atomic64_read(&s->bytes_completed),
				     (u64)atomic64_read(&s->failed), atomic_read(&s->inflight),
				     ops ? div64_u64(busy, ops * 1000) : 0);
	}
	for (d = 0; d < nr_hw_devs; d++) {
		len += sysfs_emit_at(buf, len, "%s wq_mask 0x%x", dev_name(hw_devs[d].dev),
				     hw_devs[d].wq_mask);
		for (e = 0; e < HW_EV_MAX; e++)
			len += sysfs_emit_at(buf, len, " %s %llu", hw_evs[e].name,
					     hw_read(&hw_devs[d], e));
		len += sysfs_emit_at(buf, len, "\n");
	}
	if (!nr_hw_devs)
		len += sysfs_emit_at(buf, len, "hw none (pool not acquired or no idxd PMU)\n");
	mutex_unlock(&pool_mutex);
	return len;
}

static ssize_t window_show(struct kobject *kobj, struct kobj_attribute *attr,
			   char *buf)
{
	struct usage_snap *now;
	unsigned int o, d;
	u64 ns;
	int len = 0;

	now = kzalloc(sizeof(*now), GFP_KERNEL);
	if (!now)
		return -ENOMEM;
	mutex_lock(&pool_mutex);
	usage_snap(now);
	ns = ktime_to_ns(ktime_sub(now->t, window_start.t));
	len += sysfs_emit_at(buf, len, "window_ms %llu\n", div64_u64(ns, NSEC_PER_MSEC));
	for (o = 0; o < MM_OFFLOAD_DMA_OP_MAX; o++) {
		u64 ops = now->ops[o] - window_start.ops[o];
		u64 bytes = now->bytes[o] - window_start.bytes[o];
		u64 busy = now->busy_ns[o] - window_start.busy_ns[o];

		len += sysfs_emit_at(buf, len,
				     "%-10s ops %llu bytes %llu MiB/s %llu ops/s %llu mean_us %llu\n",
				     op_names[o], ops, bytes, per_second(bytes, ns) >> 20,
				     per_second(ops, ns), ops ? div64_u64(busy, ops * 1000) : 0);
	}
	for (d = 0; d < nr_hw_devs; d++) {
		u64 *a = now->hw[d], *b = window_start.hw[d];

		len += sysfs_emit_at(buf, len,
				     "%s processed MiB/s %llu written MiB/s %llu memmove_pool %llu fill_pool %llu completions_pool %llu memmove_all %llu fill_all %llu\n",
				     dev_name(hw_devs[d].dev),
				     per_second(a[HW_CL_PROCESSED] - b[HW_CL_PROCESSED], ns) >> 20,
				     per_second(a[HW_CL_WRITTEN] - b[HW_CL_WRITTEN], ns) >> 20,
				     a[HW_MOVE_POOL] - b[HW_MOVE_POOL],
				     a[HW_FILL_POOL] - b[HW_FILL_POOL],
				     a[HW_CPL_POOL] - b[HW_CPL_POOL],
				     a[HW_MOVE_ALL] - b[HW_MOVE_ALL],
				     a[HW_FILL_ALL] - b[HW_FILL_ALL]);
	}
	mutex_unlock(&pool_mutex);
	kfree(now);
	return len;
}

/* Any write starts a new window. */
static ssize_t window_store(struct kobject *kobj, struct kobj_attribute *attr,
			    const char *buf, size_t count)
{
	mutex_lock(&pool_mutex);
	usage_snap(&window_start);
	mutex_unlock(&pool_mutex);
	return count;
}

static struct kobj_attribute totals_attr = __ATTR_RO(totals);
static struct kobj_attribute window_attr = __ATTR_RW(window);

static struct attribute *usage_attrs[] = {
	&totals_attr.attr,
	&window_attr.attr,
	NULL
};

static const struct attribute_group usage_attr_group = {
	.attrs = usage_attrs,
};

static struct kobject *usage_kobj;

static int __init mm_offload_dma_init(void)
{
	int ret;

	usage_kobj = kobject_create_and_add("mm_offload_dma", kernel_kobj);
	if (!usage_kobj)
		return -ENOMEM;
	ret = sysfs_create_group(usage_kobj, &usage_attr_group);
	if (ret) {
		kobject_put(usage_kobj);
		return ret;
	}
	window_start.t = ktime_get();
	return 0;
}
module_init(mm_offload_dma_init);

static void __exit mm_offload_dma_exit(void)
{
	map_cache_free_all();
	sysfs_remove_group(usage_kobj, &usage_attr_group);
	kobject_put(usage_kobj);
}
module_exit(mm_offload_dma_exit);

MODULE_IMPORT_NS("IDXD");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Shared DMA channel pool for mm_offload providers");
