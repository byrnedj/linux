// SPDX-License-Identifier: GPL-2.0-only
/*
 * mm_offload_dma - shared DMAEngine channel pool for mm_offload providers.
 *
 * The pool is sized by the largest request among its current users and
 * only released when the last user goes away, so a provider enabled
 * while another one is active joins the existing channels instead of
 * finding them all taken. Growth appends channels and groups behind a
 * release barrier; claimers read the published counts with an acquire
 * load and never see a half-initialised entry.
 */

#include <linux/atomic.h>
#include <linux/bitops.h>
#include <linux/dmaengine.h>
#include <linux/mm_offload_dma.h>
#include <linux/module.h>
#include <linux/mutex.h>

static DEFINE_MUTEX(pool_mutex);
static unsigned int pool_users;

static struct {
	struct dma_chan *chan;
	struct mutex lock;
} channels[MM_OFFLOAD_DMA_MAX_CHANNELS];
static unsigned int nr_channels;

static struct mm_offload_dma_group groups[MM_OFFLOAD_DMA_MAX_CHANNELS];
static unsigned int nr_groups;
static atomic_t group_cursor;

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

/* Called with pool_mutex held. */
static void pool_grow(unsigned int nr_want)
{
	dma_cap_mask_t mask;

	dma_cap_zero(mask);
	dma_cap_set(DMA_MEMCPY, mask);

	while (nr_channels < min_t(unsigned int, nr_want,
				   MM_OFFLOAD_DMA_MAX_CHANNELS)) {
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
		mutex_init(&channels[nr_channels].lock);
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
 * @nr_want: channels this user would like the pool to hold.
 *
 * Acquires channels up to @nr_want if the pool holds fewer. Return: 0
 * if the pool holds at least one channel afterwards, -ENODEV otherwise
 * (the caller is then not a user).
 */
int mm_offload_dma_pool_get(unsigned int nr_want)
{
	int ret = 0;

	mutex_lock(&pool_mutex);
	pool_grow(nr_want);
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

static unsigned long claim_group(struct mm_offload_dma_group *grp,
				 unsigned int want, unsigned int limit)
{
	unsigned long mask = 0;
	unsigned int i, got = 0;

	for (i = 0; i < grp->nr && got < want; i++) {
		unsigned int idx = grp->first + i;

		if (idx >= limit)
			break;
		if (mutex_trylock(&channels[idx].lock)) {
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
	unsigned long mask = claim_pass(nid, true, want, grpp);

	if (!mask)
		mask = claim_pass(nid, false, want, grpp);
	return mask;
}
EXPORT_SYMBOL_GPL(mm_offload_dma_claim);

void mm_offload_dma_release(unsigned long mask)
{
	unsigned int i;

	for_each_set_bit(i, &mask, MM_OFFLOAD_DMA_MAX_CHANNELS)
		mutex_unlock(&channels[i].lock);
}
EXPORT_SYMBOL_GPL(mm_offload_dma_release);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Shared DMA channel pool for mm_offload providers");
