// SPDX-License-Identifier: GPL-2.0-only
/*
 *
 * DMA batch-offloading interface driver
 *
 * Copyright (C) 2024-25 Advanced Micro Devices, Inc.
 */

#include <linux/module.h>
#include <linux/dma-mapping.h>
#include <linux/dmaengine.h>
#include <linux/migrate.h>
#include <linux/migrate_offc.h>

#define MAX_DMA_CHANNELS	16

static unsigned long long pages_migrated;
static unsigned long long pages_failures;
static unsigned long long folios_cow_copied;
static unsigned long long folios_cow_failures;
static unsigned long long folios_zeroed;
static unsigned long long folios_zero_failures;

static bool offloading_enabled;
static unsigned int nr_dma_channels = 1;
static DEFINE_MUTEX(dcbm_mutex);

struct dma_work {
	struct dma_chan *chan;
	struct completion done;
	atomic_t pending;
	struct sg_table *src_sgt;
	struct sg_table *dst_sgt;
	bool mapped;
};

static void dma_completion_callback(void *data)
{
	struct dma_work *work = data;

	if (atomic_dec_and_test(&work->pending))
		complete(&work->done);
}

static int setup_sg_tables(struct dma_work *work, struct list_head **src_pos,
		struct list_head **dst_pos, int nr)
{
	struct scatterlist *sg_src, *sg_dst;
	struct device *dev;
	int i, ret;

	work->src_sgt = kmalloc(sizeof(*work->src_sgt), GFP_KERNEL);
	if (!work->src_sgt)
		return -ENOMEM;
	work->dst_sgt = kmalloc(sizeof(*work->dst_sgt), GFP_KERNEL);
	if (!work->dst_sgt)
		goto err_free_src;

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

	dev = dmaengine_get_dma_device(work->chan);
	if (!dev) {
		ret = -ENODEV;
		goto err_free_dst_table;
	}
	ret = dma_map_sgtable(dev, work->src_sgt, DMA_TO_DEVICE,
			      DMA_ATTR_SKIP_CPU_SYNC | DMA_ATTR_NO_KERNEL_MAPPING);
	if (ret)
		goto err_free_dst_table;
	ret = dma_map_sgtable(dev, work->dst_sgt, DMA_FROM_DEVICE,
			      DMA_ATTR_SKIP_CPU_SYNC | DMA_ATTR_NO_KERNEL_MAPPING);
	if (ret)
		goto err_unmap_src;
	/* Verify mapping produced same number of entries */
	if (work->src_sgt->nents != work->dst_sgt->nents) {
		pr_err("Mismatched SG entries after mapping: src=%d dst=%d\n",
			work->src_sgt->nents, work->dst_sgt->nents);
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
	pr_err("DCBM: Failed to setup SG tables\n");
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

		dev = dmaengine_get_dma_device(works[i].chan);

		/* Terminate any pending transfers */
		if (atomic_read(&works[i].pending) > 0)
			dmaengine_terminate_sync(works[i].chan);

		if (dev && works[i].mapped) {
			if (works[i].src_sgt) {
				dma_unmap_sgtable(dev, works[i].src_sgt, DMA_TO_DEVICE,
					DMA_ATTR_SKIP_CPU_SYNC | DMA_ATTR_NO_KERNEL_MAPPING);
				sg_free_table(works[i].src_sgt);
				kfree(works[i].src_sgt);
			}
			if (works[i].dst_sgt) {
				dma_unmap_sgtable(dev, works[i].dst_sgt, DMA_FROM_DEVICE,
					DMA_ATTR_SKIP_CPU_SYNC | DMA_ATTR_NO_KERNEL_MAPPING);
				sg_free_table(works[i].dst_sgt);
				kfree(works[i].dst_sgt);
			}
		}
		dma_release_channel(works[i].chan);
	}
	kfree(works);
}

static int submit_dma_transfers(struct dma_work *work)
{
	struct scatterlist *sg_src, *sg_dst;
	struct dma_async_tx_descriptor *tx;
	unsigned long flags = DMA_CTRL_ACK;
	dma_cookie_t cookie;
	int i;

	/* Logic: send single callback for the entire batch */
	atomic_set(&work->pending, 1);

	sg_src = work->src_sgt->sgl;
	sg_dst = work->dst_sgt->sgl;
	/* Iterate over DMA-mapped entries */
	for_each_sgtable_dma_sg(work->src_sgt, sg_src, i) {
		/* Only interrupt on the last transfer */
		if (i == work->src_sgt->nents - 1)
			flags |= DMA_PREP_INTERRUPT;

		tx = dmaengine_prep_dma_memcpy(work->chan,
					       sg_dma_address(sg_dst),
					       sg_dma_address(sg_src),
					       sg_dma_len(sg_src),
					       flags);
		if (!tx) {
			atomic_set(&work->pending, 0);
			return -EIO;
		}

		/* Only set callback on last transfer */
		if (i == work->src_sgt->nents - 1) {
			tx->callback = dma_completion_callback;
			tx->callback_param = work;
		}

		cookie = dmaengine_submit(tx);
		if (dma_submit_error(cookie)) {
			atomic_set(&work->pending, 0);
			return -EIO;
		}
		sg_dst = sg_next(sg_dst);
	}
	return 0;
}

/**
 * folios_copy_dma - Copy folios using DMA engine
 * @dst_list: Destination folio list
 * @src_list: Source folio list
 * @nr_folios: Number of folios to copy
 *
 * Return: 0. Fallback to CPU copy on any error.
 */
static int folios_copy_dma(struct list_head *dst_list,
			   struct list_head *src_list,
			   unsigned int nr_folios)
{
	struct dma_work *works;
	struct list_head *src_pos = src_list->next;
	struct list_head *dst_pos = dst_list->next;
	int i, folios_per_chan, ret;
	dma_cap_mask_t mask;
	int actual_channels = 0;
	int max_channels;

	max_channels = min3(nr_dma_channels, nr_folios, MAX_DMA_CHANNELS);

	works = kcalloc(max_channels, sizeof(*works), GFP_KERNEL);
	if (!works)
		goto fallback;

	dma_cap_zero(mask);
	dma_cap_set(DMA_MEMCPY, mask);

	for (i = 0; i < max_channels; i++) {
		works[actual_channels].chan = dma_request_chan_by_mask(&mask);
		if (IS_ERR(works[actual_channels].chan))
			break;
		init_completion(&works[actual_channels].done);
		actual_channels++;
	}

	if (actual_channels == 0) {
		kfree(works);
		goto fallback;
	}

	for (i = 0; i < actual_channels; i++) {
		folios_per_chan = nr_folios * (i + 1) / actual_channels -
					(nr_folios * i) / actual_channels;
		if (folios_per_chan == 0)
			continue;

		ret = setup_sg_tables(&works[i], &src_pos, &dst_pos, folios_per_chan);
		if (ret) {
			cleanup_dma_work(works, actual_channels);
			goto fallback;
		}
	}

	for (i = 0; i < actual_channels; i++) {
		ret = submit_dma_transfers(&works[i]);
		if (ret) {
			dev_err(dmaengine_get_dma_device(works[i].chan),
				"Failed to submit transfers for channel %d\n", i);
			cleanup_dma_work(works, actual_channels);
			goto fallback;
		}
	}

	for (i = 0; i < actual_channels; i++) {
		if (atomic_read(&works[i].pending) > 0)
			dma_async_issue_pending(works[i].chan);
	}

	for (i = 0; i < actual_channels; i++) {
		if (atomic_read(&works[i].pending) == 0)
			continue;
		if (!wait_for_completion_timeout(&works[i].done, msecs_to_jiffies(10000))) {
			dev_err(dmaengine_get_dma_device(works[i].chan),
				"DMA timeout on channel %d\n", i);
			ret = -ETIMEDOUT;
			cleanup_dma_work(works, actual_channels);
			goto fallback;
		}
	}

	cleanup_dma_work(works, actual_channels);

	mutex_lock(&dcbm_mutex);
	pages_migrated += nr_folios;
	mutex_unlock(&dcbm_mutex);

	return 0;
fallback:
	mutex_lock(&dcbm_mutex);
	pages_failures += nr_folios;
	mutex_unlock(&dcbm_mutex);

	/* Fall back to CPU copy */
	pr_err("DCBM: Falling back to CPU copy\n");
	folios_mc_copy(dst_list, src_list, nr_folios);
	return 0;
}

/*
 * folio_dma_completion - DMA completion callback for single-folio operations.
 */
static void folio_dma_completion(void *data)
{
	complete((struct completion *)data);
}

/**
 * folio_copy_dma - DMA-offloaded copy of a single large/gigantic folio pair.
 * @dst: Destination folio (newly allocated, not yet mapped to userspace)
 * @src: Source folio (currently mapped; CPU cache must be flushed by DMA map)
 *
 * Used on the CoW (copy-on-write) fault path for huge/gigantic pages.
 * Large folios are always physically contiguous, so a single DMA transfer
 * covers the entire folio without scatter-gather splitting.
 *
 * Returns 0 on success, negative to signal the caller to use CPU copy.
 */
static int folio_copy_dma(struct folio *dst, struct folio *src)
{
	struct completion done;
	struct dma_chan *chan;
	struct device *dev;
	dma_addr_t src_dma, dst_dma;
	size_t size = folio_size(src);
	struct dma_async_tx_descriptor *tx;
	dma_cookie_t cookie;
	dma_cap_mask_t mask;
	int ret = 0;

	dma_cap_zero(mask);
	dma_cap_set(DMA_MEMCPY, mask);
	chan = dma_request_chan_by_mask(&mask);
	if (IS_ERR(chan))
		return PTR_ERR(chan);

	dev = dmaengine_get_dma_device(chan);
	if (!dev) {
		ret = -ENODEV;
		goto release;
	}

	/*
	 * Map without DMA_ATTR_SKIP_CPU_SYNC: the source page is actively used
	 * by the CPU so we need the cache flush, and the destination needs
	 * cache invalidation after DMA writes.
	 */
	src_dma = dma_map_page(dev, &src->page, 0, size, DMA_TO_DEVICE);
	if (dma_mapping_error(dev, src_dma)) {
		ret = -ENOMEM;
		goto release;
	}
	dst_dma = dma_map_page(dev, &dst->page, 0, size, DMA_FROM_DEVICE);
	if (dma_mapping_error(dev, dst_dma)) {
		ret = -ENOMEM;
		goto unmap_src;
	}

	init_completion(&done);

	tx = dmaengine_prep_dma_memcpy(chan, dst_dma, src_dma, size,
				       DMA_CTRL_ACK | DMA_PREP_INTERRUPT);
	if (!tx) {
		ret = -EIO;
		goto unmap_dst;
	}
	tx->callback = folio_dma_completion;
	tx->callback_param = &done;

	cookie = dmaengine_submit(tx);
	if (dma_submit_error(cookie)) {
		ret = -EIO;
		goto unmap_dst;
	}

	dma_async_issue_pending(chan);

	if (!wait_for_completion_timeout(&done, msecs_to_jiffies(10000))) {
		dmaengine_terminate_sync(chan);
		ret = -ETIMEDOUT;
	}

unmap_dst:
	dma_unmap_page(dev, dst_dma, size, DMA_FROM_DEVICE);
unmap_src:
	dma_unmap_page(dev, src_dma, size, DMA_TO_DEVICE);
release:
	dma_release_channel(chan);
	mutex_lock(&dcbm_mutex);
	if (ret)
		folios_cow_failures++;
	else
		folios_cow_copied++;
	mutex_unlock(&dcbm_mutex);
	return ret;
}

/**
 * folio_zero_dma - DMA-offloaded zero of a single large/gigantic folio.
 * @folio: Folio to zero (newly allocated, about to be mapped to userspace)
 *
 * Used on the huge-page anonymous fault path to zero a freshly allocated
 * large folio before mapping it.  Requires a DMA engine with DMA_MEMSET
 * capability (e.g. Intel DSA/IDXD).
 *
 * Returns 0 on success, negative to signal the caller to use CPU zero.
 */
static int folio_zero_dma(struct folio *folio)
{
	struct completion done;
	struct dma_chan *chan;
	struct device *dev;
	dma_addr_t dst_dma;
	size_t size = folio_size(folio);
	struct dma_async_tx_descriptor *tx;
	dma_cookie_t cookie;
	dma_cap_mask_t mask;
	int ret = 0;

	dma_cap_zero(mask);
	dma_cap_set(DMA_MEMSET, mask);
	chan = dma_request_chan_by_mask(&mask);
	if (IS_ERR(chan))
		return PTR_ERR(chan);

	dev = dmaengine_get_dma_device(chan);
	if (!dev) {
		ret = -ENODEV;
		goto release;
	}

	dst_dma = dma_map_page(dev, &folio->page, 0, size, DMA_FROM_DEVICE);
	if (dma_mapping_error(dev, dst_dma)) {
		ret = -ENOMEM;
		goto release;
	}

	init_completion(&done);

	/* value=0: DMA engine writes 0x00 across the entire folio */
	tx = dmaengine_prep_dma_memset(chan, dst_dma, 0, size,
				       DMA_CTRL_ACK | DMA_PREP_INTERRUPT);
	if (!tx) {
		ret = -EIO;
		goto unmap;
	}
	tx->callback = folio_dma_completion;
	tx->callback_param = &done;

	cookie = dmaengine_submit(tx);
	if (dma_submit_error(cookie)) {
		ret = -EIO;
		goto unmap;
	}

	dma_async_issue_pending(chan);

	if (!wait_for_completion_timeout(&done, msecs_to_jiffies(10000))) {
		dmaengine_terminate_sync(chan);
		ret = -ETIMEDOUT;
	}

unmap:
	dma_unmap_page(dev, dst_dma, size, DMA_FROM_DEVICE);
release:
	dma_release_channel(chan);
	mutex_lock(&dcbm_mutex);
	if (ret)
		folios_zero_failures++;
	else
		folios_zeroed++;
	mutex_unlock(&dcbm_mutex);
	return ret;
}

/**
 * dma_should_handle - Determine if DMA backend should handle this batch
 * @src_list: Source folio list
 * @nr_folios: Number of folios to migrate
 * @reason: Migration reason
 *
 * Returns: true if DMA should handle this batch, false for CPU fallback
 *
 * Heuristics:
 * - MR_SYSCALL: always use DMA (move_pages(2) etc.)
 * - MR_DEMOTION, MR_NUMA_MISPLACED: use DMA for batches of 4+
 *   (smaller batches: CPU overhead dominates)
 * - TODO: Heuristics/Tuning need more thought.
 */
static bool dma_should_handle(struct list_head *src_list, unsigned int nr_folios,
			      int reason)
{
	if (reason == MR_SYSCALL)
		return true;
	if (reason == MR_COMPACTION || reason == MR_DEMOTION ||
	    reason == MR_NUMA_MISPLACED)
		return nr_folios >= 4;
	return false;
}

static struct migrator dma_migrator = {
	.name = "DCBM",
	.should_handle = dma_should_handle,
	.migrate_offload_copy = folios_copy_dma,
	.copy_large_folio = folio_copy_dma,
	.zero_folio = folio_zero_dma,
	.owner = THIS_MODULE,
};

static ssize_t offloading_show(struct kobject *kobj,
			       struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%d\n", offloading_enabled);
}

static ssize_t offloading_store(struct kobject *kobj,
				struct kobj_attribute *attr,
				const char *buf, size_t count)
{
	bool enable;
	int ret;

	ret = kstrtobool(buf, &enable);
	if (ret)
		return ret;

	mutex_lock(&dcbm_mutex);

	if (enable == offloading_enabled) {
		pr_err("migration offloading is already %s\n",
			 enable ? "ON" : "OFF");
		goto out;
	}

	if (enable) {
		start_offloading(&dma_migrator);
		offloading_enabled = true;
		pr_info("migration offloading is now ON\n");
	} else {
		stop_offloading();
		offloading_enabled = false;
		pr_info("migration offloading is now OFF\n");
	}
out:
	mutex_unlock(&dcbm_mutex);
	return count;
}


static ssize_t pages_migrated_show(struct kobject *kobj,
				   struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%llu\n", pages_migrated);
}

static ssize_t pages_migrated_store(struct kobject *kobj,
				    struct kobj_attribute *attr,
				    const char *buf, size_t count)
{
	/* reset the counter */
	mutex_lock(&dcbm_mutex);
	pages_migrated = 0;
	mutex_unlock(&dcbm_mutex);

	return count;
}

static ssize_t pages_failures_show(struct kobject *kobj,
				   struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%llu\n", pages_failures);
}

static ssize_t pages_failures_store(struct kobject *kobj,
				    struct kobj_attribute *attr,
				    const char *buf, size_t count)
{
	/* reset the counter */
	mutex_lock(&dcbm_mutex);
	pages_failures = 0;
	mutex_unlock(&dcbm_mutex);

	return count;
}

static ssize_t nr_dma_chan_show(struct kobject *kobj,
				struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", nr_dma_channels);
}

static ssize_t nr_dma_chan_store(struct kobject *kobj,
				 struct kobj_attribute *attr,
				 const char *buf, size_t count)
{
	unsigned int val;
	int ret;

	ret = kstrtouint(buf, 0, &val);
	if (ret)
		return ret;

	if (val < 1 || val > MAX_DMA_CHANNELS)
		return -EINVAL;

	mutex_lock(&dcbm_mutex);
	nr_dma_channels = val;
	mutex_unlock(&dcbm_mutex);

	return count;
}

static ssize_t folios_cow_copied_show(struct kobject *kobj,
				      struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%llu\n", folios_cow_copied);
}

static ssize_t folios_cow_copied_store(struct kobject *kobj,
				       struct kobj_attribute *attr,
				       const char *buf, size_t count)
{
	mutex_lock(&dcbm_mutex);
	folios_cow_copied = 0;
	mutex_unlock(&dcbm_mutex);
	return count;
}

static ssize_t folios_cow_failures_show(struct kobject *kobj,
					struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%llu\n", folios_cow_failures);
}

static ssize_t folios_cow_failures_store(struct kobject *kobj,
					 struct kobj_attribute *attr,
					 const char *buf, size_t count)
{
	mutex_lock(&dcbm_mutex);
	folios_cow_failures = 0;
	mutex_unlock(&dcbm_mutex);
	return count;
}

static ssize_t folios_zeroed_show(struct kobject *kobj,
				  struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%llu\n", folios_zeroed);
}

static ssize_t folios_zeroed_store(struct kobject *kobj,
				   struct kobj_attribute *attr,
				   const char *buf, size_t count)
{
	mutex_lock(&dcbm_mutex);
	folios_zeroed = 0;
	mutex_unlock(&dcbm_mutex);
	return count;
}

static ssize_t folios_zero_failures_show(struct kobject *kobj,
					 struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%llu\n", folios_zero_failures);
}

static ssize_t folios_zero_failures_store(struct kobject *kobj,
					  struct kobj_attribute *attr,
					  const char *buf, size_t count)
{
	mutex_lock(&dcbm_mutex);
	folios_zero_failures = 0;
	mutex_unlock(&dcbm_mutex);
	return count;
}

static struct kobj_attribute offloading_attr = __ATTR_RW(offloading);
static struct kobj_attribute nr_dma_chan_attr = __ATTR_RW(nr_dma_chan);
static struct kobj_attribute pages_migrated_attr = __ATTR_RW(pages_migrated);
static struct kobj_attribute pages_failures_attr = __ATTR_RW(pages_failures);
static struct kobj_attribute folios_cow_copied_attr = __ATTR_RW(folios_cow_copied);
static struct kobj_attribute folios_cow_failures_attr = __ATTR_RW(folios_cow_failures);
static struct kobj_attribute folios_zeroed_attr = __ATTR_RW(folios_zeroed);
static struct kobj_attribute folios_zero_failures_attr = __ATTR_RW(folios_zero_failures);

static struct attribute *dcbm_attrs[] = {
	&offloading_attr.attr,
	&nr_dma_chan_attr.attr,
	&pages_migrated_attr.attr,
	&pages_failures_attr.attr,
	&folios_cow_copied_attr.attr,
	&folios_cow_failures_attr.attr,
	&folios_zeroed_attr.attr,
	&folios_zero_failures_attr.attr,
	NULL
};
ATTRIBUTE_GROUPS(dcbm);

static struct kobject *dcbm_kobj;

static int __init dcbm_init(void)
{
	int ret;

	dcbm_kobj = kobject_create_and_add("dcbm", kernel_kobj);
	if (!dcbm_kobj)
		return -ENOMEM;

	ret = sysfs_create_groups(dcbm_kobj, dcbm_groups);
	if (ret) {
		kobject_put(dcbm_kobj);
		return ret;
	}

	pr_info("DMA Core Batch Migrator initialized\n");
	return 0;
}

static void __exit dcbm_exit(void)
{
	/* Ensure offloading is stopped before module unload */
	mutex_lock(&dcbm_mutex);
	if (offloading_enabled) {
		stop_offloading();
		offloading_enabled = false;
	}
	mutex_unlock(&dcbm_mutex);

	sysfs_remove_groups(dcbm_kobj, dcbm_groups);
	kobject_put(dcbm_kobj);

	pr_info("DMA Core Batch Migrator unloaded\n");
}

module_init(dcbm_init);
module_exit(dcbm_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Shivank Garg");
MODULE_DESCRIPTION("DMA Core Batch Migrator");
