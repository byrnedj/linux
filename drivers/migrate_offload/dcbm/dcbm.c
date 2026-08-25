// SPDX-License-Identifier: GPL-2.0-only
/*
 * DMA Core Batch Migrator (DCBM)
 *
 * Uses DMAEngine memcpy channels to offload batch folio copies during
 * page migration. Reference driver meant for testing the offload
 * infrastructure.
 *
 * Copyright (C) 2024-26 Advanced Micro Devices, Inc.
 */

#include <linux/bitops.h>
#include <linux/module.h>
#include <linux/dma-mapping.h>
#include <linux/dmaengine.h>
#include <linux/migrate.h>
#include <linux/migrate_copy_offload.h>
#include <linux/mutex.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>

#define MAX_DMA_CHANNELS	16
/*
 * Folios per scatter-gather transaction. A provider that supports
 * DMA_MEMCPY_SG turns one transaction into one hardware batch, so
 * this bounds the batch element count it must accept.
 */
#define DCBM_SG_ELEMS_DEFAULT	32

static atomic_long_t folios_migrated;
static atomic_long_t folios_failures;
static atomic_long_t batches_refused;

static bool offloading_enabled;
static unsigned int nr_dma_channels = 1;
static unsigned int sg_elems = DCBM_SG_ELEMS_DEFAULT;
static bool cache_ctrl = true;
static DEFINE_MUTEX(dcbm_mutex);

/*
 * Channels are acquired once when offloading is enabled and shared by
 * concurrent batches through per-channel trylocks. When every channel
 * is busy the batch is refused and the move phase copies on the CPU,
 * instead of queueing behind other batches.
 */
static struct dcbm_chan {
	struct dma_chan *chan;
	struct mutex lock;
} channels[MAX_DMA_CHANNELS];
static unsigned int nr_channels;

/*
 * Channels grouped by DMA device. One batch maps its folios against a
 * single device and uses only that group's channels. Concurrent
 * batches rotate over the groups so every device contributes.
 */
static struct dcbm_group {
	struct device *dev;
	int node;
	unsigned int first;
	unsigned int nr;
} groups[MAX_DMA_CHANNELS];
static unsigned int nr_groups;
static atomic_t group_cursor;

struct dcbm_copy {
	dma_addr_t src;
	dma_addr_t dst;
	size_t len;
};

struct dma_work {
	struct dma_chan *chan;
	struct device *dev;
	struct completion done;
	atomic_t pending;
	atomic_t error;
	struct dcbm_copy *copies;
	unsigned int nr_copies;
	bool submitted;
};

/*
 * Every descriptor carries its own completion. Engines such as Intel
 * DSA complete the descriptors of one channel out of order when the
 * work queue is served by several engines, so a callback on the last
 * submitted descriptor does not mean the earlier ones have landed.
 * The pending count starts at one for the submitter, so the work
 * cannot complete before every descriptor has been submitted.
 *
 * A descriptor the hardware rejected or aborted completes with an
 * error result; it must fail the batch, or a folio that was never
 * written would be reported as copied.
 */
static void dma_completion_callback(void *data,
				    const struct dmaengine_result *result)
{
	struct dma_work *work = data;

	if (!result || result->result != DMA_TRANS_NOERROR)
		atomic_set(&work->error, -EIO);

	if (atomic_dec_and_test(&work->pending))
		complete(&work->done);
}

static void dma_work_done_submitting(struct dma_work *work)
{
	if (atomic_dec_and_test(&work->pending))
		complete(&work->done);
}

static unsigned long dcbm_claim_group(struct dcbm_group *grp,
				      unsigned int want)
{
	unsigned long mask = 0;
	unsigned int i, got = 0;

	for (i = 0; i < grp->nr && got < want; i++) {
		unsigned int idx = grp->first + i;

		if (mutex_trylock(&channels[idx].lock)) {
			mask |= BIT(idx);
			got++;
		}
	}
	return mask;
}

/*
 * One selection pass: collect the groups whose node does (or does
 * not) match @nid and try them starting from a rotating offset within
 * that selection, so rotation is fair however the eligible groups are
 * laid out in the global array.
 */
static unsigned long dcbm_claim_pass(int nid, bool match_node,
				     unsigned int want,
				     struct dcbm_group **grpp)
{
	unsigned int sel[MAX_DMA_CHANNELS];
	unsigned int n = 0, g, i;

	for (g = 0; g < nr_groups; g++)
		if ((groups[g].node == nid) == match_node)
			sel[n++] = g;
	if (!n)
		return 0;

	g = (unsigned int)atomic_inc_return(&group_cursor) % n;
	for (i = 0; i < n; i++) {
		struct dcbm_group *grp = &groups[sel[(g + i) % n]];
		unsigned long mask = dcbm_claim_group(grp, want);

		if (mask) {
			*grpp = grp;
			return mask;
		}
	}
	return 0;
}

/*
 * Claim up to @want channels from one device group. Groups on the
 * destination node are tried first: a cross-socket copy pays
 * remote-link bandwidth on every written line plus a remote
 * completion interrupt, so locality beats spreading. The first group
 * with any free channel wins. If every channel of every group is busy
 * the batch is refused.
 */
static unsigned long dcbm_claim_channels(unsigned int want, int nid,
					 struct dcbm_group **grpp)
{
	unsigned long mask = dcbm_claim_pass(nid, true, want, grpp);

	if (!mask)
		mask = dcbm_claim_pass(nid, false, want, grpp);
	return mask;
}

static void dcbm_release_channels(unsigned long mask)
{
	unsigned int i;

	for_each_set_bit(i, &mask, nr_channels)
		mutex_unlock(&channels[i].lock);
}

/*
 * Map each folio of the slice on its own. A folio is physically
 * contiguous, so it is one DMA segment, and the source and
 * destination of a copy pair up by folio rather than by whatever
 * segments an IOMMU would merge two scatterlists into.
 */
static int map_folios(struct dma_work *work, struct list_head **src_pos,
		      struct list_head **dst_pos, unsigned int nr)
{
	struct device *dev = work->dev;
	unsigned int i;

	work->copies = kcalloc(nr, sizeof(*work->copies), GFP_KERNEL);
	if (!work->copies)
		return -ENOMEM;

	for (i = 0; i < nr; i++) {
		struct folio *src = list_entry(*src_pos, struct folio, lru);
		struct folio *dst = list_entry(*dst_pos, struct folio, lru);
		struct dcbm_copy *copy = &work->copies[i];

		copy->len = folio_size(src);
		copy->src = dma_map_page_attrs(dev, folio_page(src, 0), 0,
					       copy->len, DMA_TO_DEVICE, 0);
		if (dma_mapping_error(dev, copy->src))
			goto err;
		copy->dst = dma_map_page_attrs(dev, folio_page(dst, 0), 0,
					       copy->len, DMA_FROM_DEVICE, 0);
		if (dma_mapping_error(dev, copy->dst)) {
			dma_unmap_page_attrs(dev, copy->src, copy->len,
					     DMA_TO_DEVICE, 0);
			goto err;
		}
		work->nr_copies++;

		*src_pos = (*src_pos)->next;
		*dst_pos = (*dst_pos)->next;
	}
	return 0;
err:
	/* The mapped prefix is undone by cleanup_dma_work(). */
	return -EIO;
}

static void unmap_folios(struct dma_work *work)
{
	unsigned int i;

	for (i = 0; i < work->nr_copies; i++) {
		struct dcbm_copy *copy = &work->copies[i];

		dma_unmap_page_attrs(work->dev, copy->dst, copy->len,
				     DMA_FROM_DEVICE, 0);
		dma_unmap_page_attrs(work->dev, copy->src, copy->len,
				     DMA_TO_DEVICE, 0);
	}
	work->nr_copies = 0;
	kfree(work->copies);
	work->copies = NULL;
}

/*
 * Wait for every descriptor of a slice that was handed to the engine.
 * Nothing is ever terminated: the descriptors reference the folio
 * mappings and the on-stack work until they complete, dmaengine has
 * no per-descriptor abort, and a channel-wide terminate on an engine
 * such as DSA tears down the channel's interrupt handle, after which
 * no later descriptor on that channel completes. Anything submitted
 * runs to completion; a failure only decides what the caller does
 * afterwards.
 */
static int dma_work_wait(struct dma_work *work)
{
	if (!work->submitted)
		return 0;
	wait_for_completion(&work->done);
	return atomic_read(&work->error);
}

static void cleanup_dma_work(struct dma_work *works, int actual_channels)
{
	int i;

	if (!works)
		return;

	for (i = 0; i < actual_channels; i++) {
		if (!works[i].chan)
			continue;
		unmap_folios(&works[i]);
	}
	kfree(works);
}

static int submit_one(struct dma_work *work, struct dma_async_tx_descriptor *tx)
{
	dma_cookie_t cookie;

	tx->callback_result = dma_completion_callback;
	tx->callback_param = work;
	atomic_inc(&work->pending);

	cookie = dmaengine_submit(tx);
	if (dma_submit_error(cookie)) {
		atomic_dec(&work->pending);
		return -EIO;
	}
	return 0;
}

/*
 * Hand the copies to the provider as scatter-gather transactions of
 * up to sg_elems folios each. A batch-capable engine such as DSA
 * executes one transaction as one hardware batch descriptor, so the
 * submission cost is paid once per sg_elems folios instead of once
 * per folio. The scatterlists carry the DMA addresses mapped by
 * map_folios(); they are not mapped again.
 */
static int submit_sg_transfers(struct dma_work *work, unsigned long flags)
{
	struct scatterlist *src_sg, *dst_sg;
	unsigned int elems = READ_ONCE(sg_elems);
	unsigned int done = 0;
	int ret = 0;

	src_sg = kmalloc_array(2 * elems, sizeof(*src_sg), GFP_KERNEL);
	if (!src_sg)
		return -ENOMEM;
	dst_sg = src_sg + elems;

	while (done < work->nr_copies) {
		unsigned int n = min(elems, work->nr_copies - done);
		struct dma_async_tx_descriptor *tx;
		unsigned int i;

		sg_init_table(src_sg, n);
		sg_init_table(dst_sg, n);
		for (i = 0; i < n; i++) {
			struct dcbm_copy *copy = &work->copies[done + i];

			sg_dma_address(&src_sg[i]) = copy->src;
			sg_dma_len(&src_sg[i]) = copy->len;
			sg_dma_address(&dst_sg[i]) = copy->dst;
			sg_dma_len(&dst_sg[i]) = copy->len;
		}

		tx = dmaengine_prep_dma_memcpy_sg(work->chan, dst_sg, n,
						  src_sg, n, flags);
		if (!tx) {
			ret = -EIO;
			break;
		}
		ret = submit_one(work, tx);
		if (ret)
			break;
		done += n;
	}

	kfree(src_sg);
	return ret;
}

static int submit_dma_transfers(struct dma_work *work)
{
	struct dma_async_tx_descriptor *tx;
	unsigned long flags = DMA_CTRL_ACK | DMA_PREP_INTERRUPT;
	unsigned int i;
	int ret;

	/*
	 * Cache-allocating writes leave the copied data LLC-warm for the
	 * first access after remapping, at the cost of cache footprint
	 * for folios that are not touched soon.
	 */
	if (READ_ONCE(cache_ctrl))
		flags |= DMA_PREP_CACHE_CONTROL;

	/* Submission reference, dropped by dma_work_done_submitting(). */
	atomic_set(&work->pending, 1);
	atomic_set(&work->error, 0);

	if (dma_has_cap(DMA_MEMCPY_SG, work->chan->device->cap_mask))
		return submit_sg_transfers(work, flags);

	for (i = 0; i < work->nr_copies; i++) {
		struct dcbm_copy *copy = &work->copies[i];

		tx = dmaengine_prep_dma_memcpy(work->chan, copy->dst, copy->src,
					       copy->len, flags);
		if (!tx)
			return -EIO;

		ret = submit_one(work, tx);
		if (ret)
			return ret;
	}
	return 0;
}

/**
 * folios_copy_dma - copy a batch of folios via DMA memcpy
 * @dst_list: destination folio list
 * @src_list: source folio list
 * @nr_folios: number of folios in each list
 *
 * Return: 0 on success, negative errno on failure.
 */
static int folios_copy_dma(struct list_head *dst_list,
			   struct list_head *src_list, unsigned int nr_folios)
{
	struct folio *dst;
	struct dma_work *works;
	struct dcbm_group *grp;
	struct list_head *src_pos = src_list->next;
	struct list_head *dst_pos = dst_list->next;
	unsigned long chan_mask;
	int i, folios_per_chan, ret;
	int actual_channels = 0;
	unsigned int max_channels, idx;

	max_channels = min3(READ_ONCE(nr_dma_channels), nr_folios,
			    (unsigned int)MAX_DMA_CHANNELS);

	/* Prefer the device closest to where the copies are written. */
	dst = list_first_entry(dst_list, struct folio, lru);
	chan_mask = dcbm_claim_channels(max_channels, folio_nid(dst), &grp);
	if (!chan_mask) {
		atomic_long_inc(&batches_refused);
		return -EBUSY;
	}

	works = kcalloc(hweight_long(chan_mask), sizeof(*works), GFP_KERNEL);
	if (!works) {
		dcbm_release_channels(chan_mask);
		return -ENOMEM;
	}

	for_each_set_bit(idx, &chan_mask, nr_channels) {
		works[actual_channels].chan = channels[idx].chan;
		works[actual_channels].dev = grp->dev;
		init_completion(&works[actual_channels].done);
		actual_channels++;
	}

	for (i = 0; i < actual_channels; i++) {
		folios_per_chan = nr_folios * (i + 1) / actual_channels -
				(nr_folios * i) / actual_channels;
		if (folios_per_chan == 0)
			continue;

		ret = map_folios(&works[i], &src_pos, &dst_pos,
				 folios_per_chan);
		if (ret)
			goto err_cleanup;
	}

	for (i = 0; i < actual_channels; i++) {
		if (!works[i].copies)
			continue;
		ret = submit_dma_transfers(&works[i]);
		dma_async_issue_pending(works[i].chan);
		dma_work_done_submitting(&works[i]);
		works[i].submitted = true;
		if (ret)
			goto err_wait;
	}

	ret = 0;
	for (i = 0; i < actual_channels; i++)
		ret |= dma_work_wait(&works[i]);
	if (ret)
		goto err_cleanup;

	/*
	 * All folios copied; mark each dst with FOLIO_CONTENT_COPIED so
	 * __migrate_folio() skips the per-folio copy in the move phase.
	 */
	list_for_each_entry(dst, dst_list, lru)
		dst->migrate_info |= FOLIO_CONTENT_COPIED;

	cleanup_dma_work(works, actual_channels);
	dcbm_release_channels(chan_mask);

	atomic_long_add(nr_folios, &folios_migrated);
	return 0;

err_wait:
	/* Whatever was queued before the failure runs out. */
	for (i = 0; i < actual_channels; i++)
		dma_work_wait(&works[i]);
err_cleanup:
	pr_warn_ratelimited("dcbm: DMA copy failed (%d), falling back to CPU\n",
			    ret);
	cleanup_dma_work(works, actual_channels);
	dcbm_release_channels(chan_mask);

	atomic_long_add(nr_folios, &folios_failures);
	return ret;
}

static void dcbm_put_channels(void)
{
	while (nr_channels) {
		nr_channels--;
		dma_release_channel(channels[nr_channels].chan);
		channels[nr_channels].chan = NULL;
	}
	nr_groups = 0;
}

static int dcbm_get_channels(void)
{
	dma_cap_mask_t mask;

	dma_cap_zero(mask);
	dma_cap_set(DMA_MEMCPY, mask);

	while (nr_channels < nr_dma_channels) {
		struct dma_chan *chan = dma_request_chan_by_mask(&mask);
		struct device *dev;

		if (IS_ERR(chan))
			break;

		/*
		 * The descriptors of one batch may spread over several
		 * channels but its folios are mapped only once, so a
		 * batch must stay within one DMA device: group the
		 * channels by device as they enumerate.
		 */
		dev = dmaengine_get_dma_device(chan);
		if (!dev) {
			dma_release_channel(chan);
			break;
		}
		if (!nr_groups || groups[nr_groups - 1].dev != dev) {
			groups[nr_groups].dev = dev;
			groups[nr_groups].node = dev_to_node(dev);
			groups[nr_groups].first = nr_channels;
			groups[nr_groups].nr = 0;
			nr_groups++;
		}
		groups[nr_groups - 1].nr++;

		channels[nr_channels].chan = chan;
		mutex_init(&channels[nr_channels].lock);
		nr_channels++;
	}
	return nr_channels ? 0 : -ENODEV;
}

static const struct migrator dma_migrator = {
	.name = "DCBM",
	.offload_copy = folios_copy_dma,
	.owner = THIS_MODULE,
};

static unsigned long dcbm_reason_mask = MIGRATE_OFFLOAD_REASONS_ALLOWED;

/* offloading: enable/disable DMA migration offload */
static int offloading_param_set(const char *val, const struct kernel_param *kp)
{
	bool enable;
	int ret;

	ret = kstrtobool(val, &enable);
	if (ret)
		return ret;

	mutex_lock(&dcbm_mutex);
	if (enable == offloading_enabled) {
		mutex_unlock(&dcbm_mutex);
		return 0;
	}
	if (enable) {
		ret = dcbm_get_channels();
		if (ret) {
			mutex_unlock(&dcbm_mutex);
			return ret;
		}
		ret = migrate_offload_register(&dma_migrator,
					       READ_ONCE(dcbm_reason_mask));
		if (ret) {
			dcbm_put_channels();
			mutex_unlock(&dcbm_mutex);
			return ret;
		}
		WRITE_ONCE(offloading_enabled, true);
	} else {
		migrate_offload_unregister(&dma_migrator);
		/* No batch is in flight past unregister; channels are idle. */
		dcbm_put_channels();
		WRITE_ONCE(offloading_enabled, false);
	}
	mutex_unlock(&dcbm_mutex);
	return 0;
}

static int offloading_param_get(char *buffer, const struct kernel_param *kp)
{
	return sysfs_emit(buffer, "%d\n", READ_ONCE(offloading_enabled));
}

static const struct kernel_param_ops offloading_param_ops = {
	.set = offloading_param_set,
	.get = offloading_param_get,
};
module_param_cb(offloading, &offloading_param_ops, NULL, 0644);
MODULE_PARM_DESC(offloading, "Enable DMA migration offload (0/1)");

/* nr_dma_chan: max DMA channels to use per batch */
static int nr_dma_chan_param_set(const char *val, const struct kernel_param *kp)
{
	unsigned int new_val;
	int ret;

	ret = kstrtouint(val, 0, &new_val);
	if (ret)
		return ret;
	if (new_val < 1 || new_val > MAX_DMA_CHANNELS)
		return -EINVAL;

	mutex_lock(&dcbm_mutex);
	if (offloading_enabled) {
		mutex_unlock(&dcbm_mutex);
		return -EBUSY;
	}
	WRITE_ONCE(nr_dma_channels, new_val);
	mutex_unlock(&dcbm_mutex);
	return 0;
}

static int nr_dma_chan_param_get(char *buffer, const struct kernel_param *kp)
{
	return sysfs_emit(buffer, "%u\n", READ_ONCE(nr_dma_channels));
}

static const struct kernel_param_ops nr_dma_chan_param_ops = {
	.set = nr_dma_chan_param_set,
	.get = nr_dma_chan_param_get,
};
module_param_cb(nr_dma_chan, &nr_dma_chan_param_ops, NULL, 0644);
MODULE_PARM_DESC(nr_dma_chan, "DMA channels to acquire when enabling (1..16)");

/* reason_mask: set of MR_* reasons this migrator handles */
static int reason_mask_param_set(const char *val, const struct kernel_param *kp)
{
	unsigned long mask;
	int ret;

	ret = migrate_offload_reason_mask_parse(val, &mask);
	if (ret)
		return ret;

	mutex_lock(&dcbm_mutex);
	WRITE_ONCE(dcbm_reason_mask, mask);
	if (offloading_enabled)
		migrate_offload_set_reason_mask(&dma_migrator, mask);
	mutex_unlock(&dcbm_mutex);
	return 0;
}

static int reason_mask_param_get(char *buffer, const struct kernel_param *kp)
{
	return migrate_offload_reason_mask_format(buffer, READ_ONCE(dcbm_reason_mask));
}

static const struct kernel_param_ops reason_mask_param_ops = {
	.set = reason_mask_param_set,
	.get = reason_mask_param_get,
};
module_param_cb(reason_mask, &reason_mask_param_ops, NULL, 0644);
MODULE_PARM_DESC(reason_mask,
		 "Reasons to offload: comma-separated names (e.g. compaction,demotion), 'all', 'none', or a raw hex mask");

/* folios_migrated / folios_failures: counters; any write resets to 0 */
static int folios_migrated_param_set(const char *val, const struct kernel_param *kp)
{
	atomic_long_set(&folios_migrated, 0);
	return 0;
}

static int folios_migrated_param_get(char *buffer, const struct kernel_param *kp)
{
	return sysfs_emit(buffer, "%ld\n", atomic_long_read(&folios_migrated));
}

static const struct kernel_param_ops folios_migrated_param_ops = {
	.set = folios_migrated_param_set,
	.get = folios_migrated_param_get,
};
module_param_cb(folios_migrated, &folios_migrated_param_ops, NULL, 0644);
MODULE_PARM_DESC(folios_migrated, "Folios DMA-copied (write to reset)");

static int folios_failures_param_set(const char *val, const struct kernel_param *kp)
{
	atomic_long_set(&folios_failures, 0);
	return 0;
}

static int folios_failures_param_get(char *buffer, const struct kernel_param *kp)
{
	return sysfs_emit(buffer, "%ld\n", atomic_long_read(&folios_failures));
}

static const struct kernel_param_ops folios_failures_param_ops = {
	.set = folios_failures_param_set,
	.get = folios_failures_param_get,
};
module_param_cb(folios_failures, &folios_failures_param_ops, NULL, 0644);
MODULE_PARM_DESC(folios_failures, "DMA-copy failure count (write to reset)");

static int batches_refused_param_set(const char *val, const struct kernel_param *kp)
{
	atomic_long_set(&batches_refused, 0);
	return 0;
}

static int batches_refused_param_get(char *buffer, const struct kernel_param *kp)
{
	return sysfs_emit(buffer, "%ld\n", atomic_long_read(&batches_refused));
}

static const struct kernel_param_ops batches_refused_param_ops = {
	.set = batches_refused_param_set,
	.get = batches_refused_param_get,
};
module_param_cb(batches_refused, &batches_refused_param_ops, NULL, 0644);
MODULE_PARM_DESC(batches_refused, "Batches refused because all channels were busy (write to reset)");

module_param(cache_ctrl, bool, 0644);
MODULE_PARM_DESC(cache_ctrl, "Request cache-allocating writes (DMA_PREP_CACHE_CONTROL)");

module_param(sg_elems, uint, 0644);
MODULE_PARM_DESC(sg_elems, "Folios per scatter-gather transaction on DMA_MEMCPY_SG providers");

static int __init dcbm_init(void)
{
	pr_info("dcbm: DMA Core Batch Migrator initialized\n");
	return 0;
}

static void __exit dcbm_exit(void)
{
	mutex_lock(&dcbm_mutex);
	if (offloading_enabled) {
		migrate_offload_unregister(&dma_migrator);
		dcbm_put_channels();
		offloading_enabled = false;
	}
	mutex_unlock(&dcbm_mutex);

	pr_info("dcbm: DMA Core Batch Migrator unloaded\n");
}

module_init(dcbm_init);
module_exit(dcbm_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Shivank Garg");
MODULE_DESCRIPTION("DMA Core Batch Migrator");
