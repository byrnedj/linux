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
#include <linux/dmaengine.h>
#include <linux/mm_offload_dma.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/wait.h>
#include <linux/jiffies.h>

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
	if (nr_channels)
		pool_users++;
	else
		ret = -ENODEV;
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

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Shared DMA channel pool for mm_offload providers");
