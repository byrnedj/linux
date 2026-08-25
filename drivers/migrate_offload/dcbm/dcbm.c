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

#define MAX_DMA_CHANNELS	16

static atomic_long_t folios_migrated;
static atomic_long_t folios_failures;
static atomic_long_t batches_refused;

static bool offloading_enabled;
static unsigned int nr_dma_channels = 1;
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

struct dma_work {
	struct dma_chan *chan;
	struct device *dev;
	struct completion done;
	atomic_t pending;
	atomic_t error;
	struct sg_table *src_sgt;
	struct sg_table *dst_sgt;
	bool mapped;
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

static int setup_sg_tables(struct dma_work *work, struct list_head **src_pos,
			   struct list_head **dst_pos, int nr)
{
	struct scatterlist *sg_src, *sg_dst;
	struct device *dev;
	int i, ret;

	work->src_sgt = kmalloc_obj(*work->src_sgt, GFP_KERNEL);
	if (!work->src_sgt)
		return -ENOMEM;
	work->dst_sgt = kmalloc_obj(*work->dst_sgt, GFP_KERNEL);
	if (!work->dst_sgt) {
		ret = -ENOMEM;
		goto err_free_src;
	}

	ret = sg_alloc_table(work->src_sgt, nr, GFP_KERNEL);
	if (ret)
		goto err_free_dst;
	ret = sg_alloc_table(work->dst_sgt, nr, GFP_KERNEL);
	if (ret)
		goto err_free_src_table;

	sg_src = work->src_sgt->sgl;
	sg_dst = work->dst_sgt->sgl;
	for (i = 0; i < nr; i++) {
		struct folio *src = list_entry(*src_pos, struct folio, lru);
		struct folio *dst = list_entry(*dst_pos, struct folio, lru);

		sg_set_folio(sg_src, src, folio_size(src), 0);
		sg_set_folio(sg_dst, dst, folio_size(dst), 0);

		*src_pos = (*src_pos)->next;
		*dst_pos = (*dst_pos)->next;

		if (i < nr - 1) {
			sg_src = sg_next(sg_src);
			sg_dst = sg_next(sg_dst);
		}
	}

	dev = work->dev;
	ret = dma_map_sgtable(dev, work->src_sgt, DMA_TO_DEVICE,
			      DMA_ATTR_SKIP_CPU_SYNC | DMA_ATTR_NO_KERNEL_MAPPING);
	if (ret)
		goto err_free_dst_table;
	ret = dma_map_sgtable(dev, work->dst_sgt, DMA_FROM_DEVICE,
			      DMA_ATTR_SKIP_CPU_SYNC | DMA_ATTR_NO_KERNEL_MAPPING);
	if (ret)
		goto err_unmap_src;

	/*
	 * TODO: IOMMU may merge segments unevenly on the two sides, fall back
	 * bail to CPU copy. In practice, I have not observed merging in tests.
	 * Handling unequal nents is left for follow-up.
	 */
	if (work->src_sgt->nents != work->dst_sgt->nents) {
		ret = -EINVAL;
		goto err_unmap_dst;
	}
	work->mapped = true;
	return 0;

err_unmap_dst:
	dma_unmap_sgtable(dev, work->dst_sgt, DMA_FROM_DEVICE,
			  DMA_ATTR_SKIP_CPU_SYNC | DMA_ATTR_NO_KERNEL_MAPPING);
err_unmap_src:
	dma_unmap_sgtable(dev, work->src_sgt, DMA_TO_DEVICE,
			  DMA_ATTR_SKIP_CPU_SYNC | DMA_ATTR_NO_KERNEL_MAPPING);
err_free_dst_table:
	sg_free_table(work->dst_sgt);
err_free_src_table:
	sg_free_table(work->src_sgt);
err_free_dst:
	kfree(work->dst_sgt);
	work->dst_sgt = NULL;
err_free_src:
	kfree(work->src_sgt);
	work->src_sgt = NULL;
	return ret;
}

static void cleanup_dma_work(struct dma_work *works, int actual_channels)
{
	struct device *dev;
	int i;

	if (!works)
		return;

	for (i = 0; i < actual_channels; i++) {
		if (!works[i].chan)
			continue;

		dev = works[i].dev;

		if (works[i].mapped)
			dmaengine_terminate_sync(works[i].chan);

		if (dev && works[i].mapped) {
			if (works[i].src_sgt) {
				dma_unmap_sgtable(dev, works[i].src_sgt,
						  DMA_TO_DEVICE,
						  DMA_ATTR_SKIP_CPU_SYNC |
						  DMA_ATTR_NO_KERNEL_MAPPING);
				sg_free_table(works[i].src_sgt);
				kfree(works[i].src_sgt);
			}
			if (works[i].dst_sgt) {
				dma_unmap_sgtable(dev, works[i].dst_sgt,
						  DMA_FROM_DEVICE,
						  DMA_ATTR_SKIP_CPU_SYNC |
						  DMA_ATTR_NO_KERNEL_MAPPING);
				sg_free_table(works[i].dst_sgt);
				kfree(works[i].dst_sgt);
			}
		}
	}
	kfree(works);
}

static int submit_dma_transfers(struct dma_work *work)
{
	struct scatterlist *sg_src, *sg_dst;
	struct dma_async_tx_descriptor *tx;
	unsigned long flags = DMA_CTRL_ACK | DMA_PREP_INTERRUPT;
	dma_cookie_t cookie;
	int i;

	/* Submission reference, dropped by dma_work_done_submitting(). */
	atomic_set(&work->pending, 1);
	atomic_set(&work->error, 0);

	sg_src = work->src_sgt->sgl;
	sg_dst = work->dst_sgt->sgl;
	for_each_sgtable_dma_sg(work->src_sgt, sg_src, i) {
		tx = dmaengine_prep_dma_memcpy(work->chan,
					       sg_dma_address(sg_dst),
					       sg_dma_address(sg_src),
					       sg_dma_len(sg_src), flags);
		if (!tx)
			return -EIO;

		tx->callback_result = dma_completion_callback;
		tx->callback_param = work;
		atomic_inc(&work->pending);

		cookie = dmaengine_submit(tx);
		if (dma_submit_error(cookie)) {
			atomic_dec(&work->pending);
			return -EIO;
		}
		sg_dst = sg_next(sg_dst);
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

		ret = setup_sg_tables(&works[i], &src_pos, &dst_pos,
				      folios_per_chan);
		if (ret)
			goto err_cleanup;
	}

	for (i = 0; i < actual_channels; i++) {
		if (!works[i].mapped)
			continue;
		ret = submit_dma_transfers(&works[i]);
		if (ret) {
			dma_work_done_submitting(&works[i]);
			goto err_cleanup;
		}
		dma_async_issue_pending(works[i].chan);
		dma_work_done_submitting(&works[i]);
	}

	for (i = 0; i < actual_channels; i++) {
		if (!works[i].mapped)
			continue;
		if (!wait_for_completion_timeout(&works[i].done,
						 msecs_to_jiffies(10000))) {
			ret = -ETIMEDOUT;
			goto err_cleanup;
		}
		ret = atomic_read(&works[i].error);
		if (ret)
			goto err_cleanup;
	}

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
