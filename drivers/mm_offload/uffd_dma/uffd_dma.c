// SPDX-License-Identifier: GPL-2.0-only
/*
 * UFFD_DMA - DMA user page copy offload for UFFDIO_COPY.
 *
 * Copies user memory into freshly allocated, not yet mapped pages
 * through DMAEngine memcpy channels (Intel DSA MEMMOVE). mm_offload
 * provider for the copy_user_pages op; the destination is a
 * physically contiguous page run (a hugetlb folio, or a batch of
 * order-0 pages carved from one allocation by mm/userfaultfd.c), the
 * source is user memory pinned for the duration of the copy.
 *
 * Channels come from the mm_offload_dma pool shared with the other
 * providers (grouped by DMA device, per-channel trylocks); an all-busy
 * engine refuses the copy so the caller falls back to the CPU rather
 * than queueing.
 */

#include <linux/atomic.h>
#include <linux/bitops.h>
#include <linux/completion.h>
#include <linux/dma-mapping.h>
#include <linux/dmaengine.h>
#include <linux/highmem.h>
#include <linux/kobject.h>
#include <linux/ktime.h>
#include <linux/mm.h>
#include <linux/mm_offload.h>
#include <linux/mm_offload_dma.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/scatterlist.h>
#include <linux/sizes.h>
#include <linux/slab.h>
#include <linux/sysfs.h>
#include <linux/uaccess.h>

#define UFFD_DMA_TIMEOUT_MS	10000
/* One descriptor per this many bytes when spreading over channels */
#define UFFD_DMA_CHUNK_BYTES	SZ_2M

enum uffd_dma_wait { UFFD_DMA_WAIT_IRQ = 0, UFFD_DMA_WAIT_SPIN_IRQ = 1 };

static atomic_long_t copies_done;
static atomic_long_t copies_failed;
static atomic_long_t copies_refused;
static atomic_long_t copies_efault;
static atomic_long_t bytes_dma;
static atomic_long_t bytes_cpu;
static atomic_long_t wait_ns_total;	/* submit -> completion, microseconds */
static atomic_long_t pin_us_total;	/* source pinning */
static atomic_long_t map_us_total;	/* DMA mapping of source + destination */
static atomic_long_t unmap_us_total;	/* unmap + unpin */
static atomic_long_t src_contig;
static atomic_long_t sg_txns;	/* DMA_MEMCPY_SG transactions submitted */	/* copies whose source was one contiguous run */

static bool offloading_enabled;
static unsigned int nr_dma_chan = 4;
static unsigned long min_bytes = SZ_64K;
static unsigned int cpu_pct;		/* tail fraction copied by the CPU */
static bool cache_ctrl = true;
static unsigned int wait_mode = UFFD_DMA_WAIT_IRQ;
static unsigned int spin_us = 60;
/* Source segments per DMA_MEMCPY_SG transaction; 0 disables the batch path */
static unsigned int sg_elems = 32;
static bool util_gate;

static DEFINE_MUTEX(uffd_dma_mutex);

static struct kobject *uffd_dma_kobj;

struct uffd_dma_req {
	struct completion done;
	atomic_t pending;
	atomic_t error;
};

static void uffd_dma_callback(void *data, const struct dmaengine_result *result)
{
	struct uffd_dma_req *req = data;

	if (!result || result->result != DMA_TRANS_NOERROR)
		atomic_set(&req->error, -EIO);
	if (atomic_dec_and_test(&req->pending))
		complete(&req->done);
}


/* ---- source pinning ---- */

/*
 * Without page faults allowed only pages that are already present may
 * be used (lockless GUP, no fallback); with faults allowed the source
 * is pinned properly, which may take mmap_lock - the caller has
 * dropped its own locks in that case.
 */
static int uffd_dma_pin_src(unsigned long src, unsigned long nr,
			    struct page **pages, bool allow_pagefault,
			    bool *pinned)
{
	long got;

	if (!allow_pagefault) {
		got = get_user_pages_fast_only(src, nr, 0, pages);
		*pinned = false;
	} else {
		got = pin_user_pages_fast(src, nr, 0, pages);
		*pinned = true;
	}
	if (got == nr)
		return 0;
	if (got > 0) {
		if (*pinned)
			unpin_user_pages(pages, got);
		else
			release_pages(pages, got);
	}
	return -EFAULT;
}

/*
 * Folio-aware release: consecutive pages of the same folio (a THP or
 * hugetlb source) drop their references with one atomic per folio
 * instead of one per page.
 */
static void uffd_dma_unpin_src(struct page **pages, unsigned long nr,
			       bool pinned)
{
	unsigned long i = 0;

	while (i < nr) {
		struct folio *folio = page_folio(pages[i]);
		unsigned long j = i + 1;

		while (j < nr && pages[j] == pages[j - 1] + 1 &&
		       page_folio(pages[j]) == folio)
			j++;
		if (pinned)
			unpin_user_folio(folio, j - i);
		else
			folio_put_refs(folio, j - i);
		i = j;
	}
}

/* Is the pinned source one physically contiguous run? */
static bool uffd_dma_src_contiguous(struct page **pages, unsigned long nr)
{
	unsigned long i;

	for (i = 1; i < nr; i++)
		if (pages[i] != pages[0] + i)
			return false;
	return true;
}

/* ---- submission ---- */

struct uffd_dma_cursor {
	unsigned long chan_mask;
	unsigned int idx;
	unsigned long flags;
};

static int uffd_dma_submit_one(struct uffd_dma_req *req,
			       struct uffd_dma_cursor *cur,
			       dma_addr_t dst, dma_addr_t src, size_t len)
{
	struct dma_async_tx_descriptor *tx;
	dma_cookie_t cookie;

	tx = dmaengine_prep_dma_memcpy(mm_offload_dma_chan(cur->idx), dst, src, len,
				       cur->flags);
	if (!tx)
		return -EIO;
	tx->callback_result = uffd_dma_callback;
	tx->callback_param = req;
	atomic_inc(&req->pending);
	cookie = dmaengine_submit(tx);
	if (dma_submit_error(cookie)) {
		atomic_dec(&req->pending);
		return -EIO;
	}
	cur->idx = find_next_bit(&cur->chan_mask, MM_OFFLOAD_DMA_MAX_CHANNELS,
				 cur->idx + 1);
	if (cur->idx >= MM_OFFLOAD_DMA_MAX_CHANNELS)
		cur->idx = find_first_bit(&cur->chan_mask, MM_OFFLOAD_DMA_MAX_CHANNELS);
	return 0;
}

/*
 * Walk the source scatterlist and emit one memcpy per DMA segment
 * (further split into UFFD_DMA_CHUNK_BYTES pieces rotated over the
 * claimed channels), covering destination bytes [0, dma_len).
 */
static int uffd_dma_submit(struct uffd_dma_req *req,
			   struct uffd_dma_cursor *cur, struct sg_table *sgt,
			   dma_addr_t dst, size_t dma_len)
{
	struct scatterlist *sg;
	size_t off = 0;
	int i, ret;

	for_each_sgtable_dma_sg(sgt, sg, i) {
		dma_addr_t saddr = sg_dma_address(sg);
		size_t slen = sg_dma_len(sg);

		while (slen && off < dma_len) {
			size_t this = min3(slen, dma_len - off,
					   (size_t)UFFD_DMA_CHUNK_BYTES);

			ret = uffd_dma_submit_one(req, cur, dst + off, saddr, this);
			if (ret)
				return ret;
			off += this;
			saddr += this;
			slen -= this;
		}
		if (off >= dma_len)
			break;
	}
	return 0;
}

/*
 * Batch path: pack the scattered source into DMA_MEMCPY_SG transactions
 * of up to sg_elems segments against one contiguous destination window.
 * A batch-capable engine such as DSA executes one transaction as one
 * hardware BATCH descriptor, so a 512-page copy occupies a handful of
 * ring slots instead of one per segment (which could exhaust the ring,
 * the -EIO fallback) and the submission cost is paid once per
 * transaction. Transactions rotate over the claimed channels.
 */
static int uffd_dma_submit_sg(struct uffd_dma_req *req,
			      struct uffd_dma_cursor *cur,
			      struct sg_table *sgt, dma_addr_t dst,
			      size_t dma_len)
{
	unsigned int elems = clamp(READ_ONCE(sg_elems), 1U, 1024U);
	struct scatterlist *src_win, dst_sg;
	struct scatterlist *sg = sgt->sgl;
	unsigned int nents = sgt->nents;
	size_t sg_off = 0, off = 0;
	int ret = 0;

	src_win = kmalloc_array(elems, sizeof(*src_win), GFP_KERNEL);
	if (!src_win)
		return -ENOMEM;

	while (off < dma_len && nents) {
		struct dma_async_tx_descriptor *tx;
		size_t win_len = 0;
		unsigned int n = 0;
		dma_cookie_t cookie;

		sg_init_table(src_win, elems);
		while (n < elems && nents && off + win_len < dma_len) {
			size_t seg = min(sg_dma_len(sg) - sg_off,
					 dma_len - off - win_len);

			sg_dma_address(&src_win[n]) = sg_dma_address(sg) + sg_off;
			sg_dma_len(&src_win[n]) = seg;
			win_len += seg;
			n++;
			sg_off += seg;
			if (sg_off == sg_dma_len(sg)) {
				sg = sg_next(sg);
				sg_off = 0;
				nents--;
			}
		}
		if (!n)
			break;

		sg_init_table(&dst_sg, 1);
		sg_dma_address(&dst_sg) = dst + off;
		sg_dma_len(&dst_sg) = win_len;

		tx = dmaengine_prep_dma_memcpy_sg(mm_offload_dma_chan(cur->idx),
						  &dst_sg, 1, src_win, n,
						  cur->flags);
		if (!tx) {
			ret = -EIO;
			break;
		}
		tx->callback_result = uffd_dma_callback;
		tx->callback_param = req;
		atomic_inc(&req->pending);
		cookie = dmaengine_submit(tx);
		if (dma_submit_error(cookie)) {
			atomic_dec(&req->pending);
			ret = -EIO;
			break;
		}
		atomic_long_inc(&sg_txns);
		off += win_len;
		cur->idx = find_next_bit(&cur->chan_mask,
					 MM_OFFLOAD_DMA_MAX_CHANNELS,
					 cur->idx + 1);
		if (cur->idx >= MM_OFFLOAD_DMA_MAX_CHANNELS)
			cur->idx = find_first_bit(&cur->chan_mask,
						  MM_OFFLOAD_DMA_MAX_CHANNELS);
	}
	kfree(src_win);
	return ret;
}

/* Contiguous source: chunks of UFFD_DMA_CHUNK_BYTES over the claimed channels. */
static int uffd_dma_submit_contig(struct uffd_dma_req *req,
				  struct uffd_dma_cursor *cur, dma_addr_t dst,
				  dma_addr_t src, size_t dma_len)
{
	size_t off = 0;
	int ret;

	while (off < dma_len) {
		size_t this = min_t(size_t, dma_len - off, UFFD_DMA_CHUNK_BYTES);

		ret = uffd_dma_submit_one(req, cur, dst + off, src + off, this);
		if (ret)
			return ret;
		off += this;
	}
	return 0;
}

static int uffd_dma_wait(struct uffd_dma_req *req)
{
	if (READ_ONCE(wait_mode) == UFFD_DMA_WAIT_SPIN_IRQ) {
		ktime_t deadline = ktime_add_us(ktime_get(), READ_ONCE(spin_us));

		while (!completion_done(&req->done)) {
			if (ktime_after(ktime_get(), deadline))
				break;
			cpu_relax();
		}
	}
	if (!wait_for_completion_timeout(&req->done,
					 msecs_to_jiffies(UFFD_DMA_TIMEOUT_MS)))
		return -ETIMEDOUT;
	return atomic_read(&req->error);
}

static int uffd_dma_copy_pages(struct page *dst, unsigned long nr_pages,
			       const void __user *usrc, bool allow_pagefault)
{
	const size_t size = nr_pages << PAGE_SHIFT;
	unsigned long src = (unsigned long)usrc;
	struct mm_offload_dma_group *grp;
	struct uffd_dma_cursor cur;
	struct uffd_dma_req req;
	struct page **pages;
	struct sg_table sgt;
	struct device *dev;
	dma_addr_t dst_dma, src_dma = 0;
	size_t dma_len, cpu_off;
	ktime_t t0, t1;
	bool pinned, contig = false, cpu_fault = false;
	unsigned int i;
	int ret;

	if (size < READ_ONCE(min_bytes) || (src & ~PAGE_MASK)) {
		atomic_long_inc(&copies_refused);
		return -ENODEV;
	}
	if (IS_ENABLED(CONFIG_PAGE_CLEAR_OFFLOAD) && READ_ONCE(util_gate) &&
	    mm_offload_cpus_saturated()) {
		atomic_long_inc(&copies_refused);
		return -EBUSY;
	}

	cur.chan_mask = mm_offload_dma_claim(
			clamp_t(size_t, size / UFFD_DMA_CHUNK_BYTES, 1,
				READ_ONCE(nr_dma_chan)),
			page_to_nid(dst), &grp);
	if (!cur.chan_mask) {
		atomic_long_inc(&copies_refused);
		return -EBUSY;
	}
	cur.idx = find_first_bit(&cur.chan_mask, MM_OFFLOAD_DMA_MAX_CHANNELS);
	cur.flags = DMA_PREP_INTERRUPT | DMA_CTRL_ACK;
	if (READ_ONCE(cache_ctrl))
		cur.flags |= DMA_PREP_CACHE_CONTROL;
	dev = grp->dev;

	pages = kvmalloc_array(nr_pages, sizeof(*pages), GFP_KERNEL);
	if (!pages) {
		ret = -ENOMEM;
		goto out_release;
	}
	t0 = ktime_get();
	ret = uffd_dma_pin_src(src, nr_pages, pages, allow_pagefault, &pinned);
	if (ret) {
		atomic_long_inc(&copies_efault);
		goto out_free;
	}
	t1 = ktime_get();
	atomic_long_add(ktime_us_delta(t1, t0), &pin_us_total);

	/*
	 * A THP or hugetlb-backed source is one physical run: map it as a
	 * single range and skip the scatterlist entirely.
	 */
	contig = uffd_dma_src_contiguous(pages, nr_pages);
	if (contig) {
		src_dma = dma_map_page_attrs(dev, pages[0], 0, size,
					     DMA_TO_DEVICE, 0);
		if (dma_mapping_error(dev, src_dma)) {
			ret = -EIO;
			goto out_unpin;
		}
		atomic_long_inc(&src_contig);
	} else {
		ret = sg_alloc_table_from_pages(&sgt, pages, nr_pages, 0, size,
						GFP_KERNEL);
		if (ret)
			goto out_unpin;
		ret = dma_map_sgtable(dev, &sgt, DMA_TO_DEVICE, 0);
		if (ret)
			goto out_sgfree;
	}
	dst_dma = dma_map_page_attrs(dev, dst, 0, size, DMA_FROM_DEVICE, 0);
	if (dma_mapping_error(dev, dst_dma)) {
		ret = -EIO;
		goto out_unmap_src;
	}
	t0 = ktime_get();
	atomic_long_add(ktime_us_delta(t0, t1), &map_us_total);

	/* The CPU copies the tail cpu_pct% concurrently with the engine. */
	cpu_off = size - ALIGN_DOWN(size * min(READ_ONCE(cpu_pct), 100u) / 100,
				    PAGE_SIZE);
	dma_len = cpu_off;

	init_completion(&req.done);
	atomic_set(&req.pending, 1);	/* submission reference */
	atomic_set(&req.error, 0);

	if (dma_len) {
		if (contig)
			ret = uffd_dma_submit_contig(&req, &cur, dst_dma, src_dma,
						     dma_len);
		else
			if (READ_ONCE(sg_elems) &&
			    dma_has_cap(DMA_MEMCPY_SG,
					mm_offload_dma_chan(cur.idx)->device->cap_mask))
				ret = uffd_dma_submit_sg(&req, &cur, &sgt,
							 dst_dma, dma_len);
			else
				ret = uffd_dma_submit(&req, &cur, &sgt, dst_dma,
						      dma_len);
		if (ret)
			goto out_terminate;
		for_each_set_bit(i, &cur.chan_mask, MM_OFFLOAD_DMA_MAX_CHANNELS)
			dma_async_issue_pending(mm_offload_dma_chan(i));
	}

	for (i = cpu_off >> PAGE_SHIFT; i < nr_pages; i++) {
		void *kaddr = kmap_local_page(dst + i);
		unsigned long rc;

		if (!allow_pagefault)
			pagefault_disable();
		rc = copy_from_user(kaddr, usrc + i * PAGE_SIZE, PAGE_SIZE);
		if (!allow_pagefault)
			pagefault_enable();
		kunmap_local(kaddr);
		if (rc) {
			cpu_fault = true;
			break;
		}
		flush_dcache_page(dst + i);
	}

	if (atomic_dec_and_test(&req.pending))
		complete(&req.done);
	ret = uffd_dma_wait(&req);
	if (ret == -ETIMEDOUT)
		goto out_terminate;
	t1 = ktime_get();
	atomic_long_add(ktime_us_delta(t1, t0), &wait_ns_total);

	dma_unmap_page_attrs(dev, dst_dma, size, DMA_FROM_DEVICE, 0);
	if (contig) {
		dma_unmap_page_attrs(dev, src_dma, size, DMA_TO_DEVICE, 0);
	} else {
		dma_unmap_sgtable(dev, &sgt, DMA_TO_DEVICE, 0);
		sg_free_table(&sgt);
	}
	uffd_dma_unpin_src(pages, nr_pages, pinned);
	kvfree(pages);
	mm_offload_dma_release(cur.chan_mask);
	atomic_long_add(ktime_us_delta(ktime_get(), t1), &unmap_us_total);

	if (ret) {
		atomic_long_inc(&copies_failed);
		pr_warn_ratelimited("uffd_dma: DMA copy failed (%d), falling back to CPU\n",
				    ret);
		return ret;
	}
	if (cpu_fault) {
		atomic_long_inc(&copies_efault);
		return -EFAULT;
	}
	atomic_long_inc(&copies_done);
	atomic_long_add(dma_len, &bytes_dma);
	atomic_long_add(size - dma_len, &bytes_cpu);
	return 0;

out_terminate:
	for_each_set_bit(i, &cur.chan_mask, MM_OFFLOAD_DMA_MAX_CHANNELS)
		dmaengine_terminate_sync(mm_offload_dma_chan(i));
	dma_unmap_page_attrs(dev, dst_dma, size, DMA_FROM_DEVICE, 0);
out_unmap_src:
	if (contig) {
		dma_unmap_page_attrs(dev, src_dma, size, DMA_TO_DEVICE, 0);
		goto out_unpin;
	}
	dma_unmap_sgtable(dev, &sgt, DMA_TO_DEVICE, 0);
out_sgfree:
	sg_free_table(&sgt);
out_unpin:
	uffd_dma_unpin_src(pages, nr_pages, pinned);
out_free:
	kvfree(pages);
out_release:
	mm_offload_dma_release(cur.chan_mask);
	if (ret != -EFAULT) {
		atomic_long_inc(&copies_failed);
		pr_warn_ratelimited("uffd_dma: copy setup failed (%d), falling back to CPU\n",
				    ret);
	}
	return ret;
}

static const struct mm_offload_provider uffd_dma_copier = {
	.name = "UFFD_DMA",
	.copy_user_pages = uffd_dma_copy_pages,
	.owner = THIS_MODULE,
};


/* ---- sysfs ---- */

static ssize_t offloading_show(struct kobject *kobj,
			       struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%d\n", offloading_enabled);
}

static ssize_t offloading_store(struct kobject *kobj,
				struct kobj_attribute *attr, const char *buf,
				size_t count)
{
	bool enable;
	int ret;

	ret = kstrtobool(buf, &enable);
	if (ret)
		return ret;

	mutex_lock(&uffd_dma_mutex);
	if (enable == offloading_enabled)
		goto out;
	if (enable) {
		ret = mm_offload_dma_pool_get(READ_ONCE(nr_dma_chan));
		if (ret)
			goto err;
		ret = mm_offload_register(&uffd_dma_copier, 0);
		if (ret) {
			mm_offload_dma_pool_put();
			goto err;
		}
		offloading_enabled = true;
	} else {
		mm_offload_unregister(&uffd_dma_copier);
		mm_offload_dma_pool_put();
		offloading_enabled = false;
	}
out:
	mutex_unlock(&uffd_dma_mutex);
	return count;
err:
	mutex_unlock(&uffd_dma_mutex);
	return ret;
}

#define UFFD_DMA_UINT_ATTR(name, lo, hi, needs_disabled)			\
static ssize_t name##_show(struct kobject *kobj, struct kobj_attribute *attr,	\
			   char *buf)						\
{										\
	return sysfs_emit(buf, "%u\n", READ_ONCE(name));			\
}										\
static ssize_t name##_store(struct kobject *kobj, struct kobj_attribute *attr,	\
			    const char *buf, size_t count)			\
{										\
	unsigned int val;							\
	int ret = kstrtouint(buf, 0, &val);					\
										\
	if (ret)								\
		return ret;							\
	if (val < (lo) || val > (hi))						\
		return -EINVAL;							\
	if (needs_disabled) {							\
		mutex_lock(&uffd_dma_mutex);					\
		if (offloading_enabled) {					\
			mutex_unlock(&uffd_dma_mutex);				\
			return -EBUSY;						\
		}								\
		name = val;							\
		mutex_unlock(&uffd_dma_mutex);					\
	} else {								\
		WRITE_ONCE(name, val);						\
	}									\
	return count;								\
}										\
static struct kobj_attribute name##_attr = __ATTR_RW(name)

#define UFFD_DMA_BOOL_ATTR(name)						\
static ssize_t name##_show(struct kobject *kobj, struct kobj_attribute *attr,	\
			   char *buf)						\
{										\
	return sysfs_emit(buf, "%d\n", READ_ONCE(name));			\
}										\
static ssize_t name##_store(struct kobject *kobj, struct kobj_attribute *attr,	\
			    const char *buf, size_t count)			\
{										\
	bool val;								\
	int ret = kstrtobool(buf, &val);					\
										\
	if (ret)								\
		return ret;							\
	WRITE_ONCE(name, val);							\
	return count;								\
}										\
static struct kobj_attribute name##_attr = __ATTR_RW(name)

#define UFFD_DMA_COUNTER_ATTR(name)						\
static ssize_t name##_show(struct kobject *kobj, struct kobj_attribute *attr,	\
			   char *buf)						\
{										\
	return sysfs_emit(buf, "%ld\n", atomic_long_read(&name));		\
}										\
static ssize_t name##_store(struct kobject *kobj, struct kobj_attribute *attr,	\
			    const char *buf, size_t count)			\
{										\
	atomic_long_set(&name, 0);						\
	return count;								\
}										\
static struct kobj_attribute name##_attr = __ATTR_RW(name)

UFFD_DMA_UINT_ATTR(nr_dma_chan, 1, MM_OFFLOAD_DMA_MAX_CHANNELS, true);
UFFD_DMA_UINT_ATTR(cpu_pct, 0, 100, false);
UFFD_DMA_UINT_ATTR(wait_mode, 0, 1, false);
UFFD_DMA_UINT_ATTR(spin_us, 0, 100000, false);
UFFD_DMA_UINT_ATTR(sg_elems, 0, 1024, false);
UFFD_DMA_BOOL_ATTR(cache_ctrl);
UFFD_DMA_BOOL_ATTR(util_gate);
UFFD_DMA_COUNTER_ATTR(copies_done);
UFFD_DMA_COUNTER_ATTR(copies_failed);
UFFD_DMA_COUNTER_ATTR(copies_refused);
UFFD_DMA_COUNTER_ATTR(copies_efault);
UFFD_DMA_COUNTER_ATTR(bytes_dma);
UFFD_DMA_COUNTER_ATTR(bytes_cpu);
UFFD_DMA_COUNTER_ATTR(wait_ns_total);
UFFD_DMA_COUNTER_ATTR(pin_us_total);
UFFD_DMA_COUNTER_ATTR(map_us_total);
UFFD_DMA_COUNTER_ATTR(unmap_us_total);
UFFD_DMA_COUNTER_ATTR(src_contig);
UFFD_DMA_COUNTER_ATTR(sg_txns);

static ssize_t min_bytes_show(struct kobject *kobj,
			      struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%lu\n", READ_ONCE(min_bytes));
}

static ssize_t min_bytes_store(struct kobject *kobj,
			       struct kobj_attribute *attr, const char *buf,
			       size_t count)
{
	unsigned long val;
	int ret = kstrtoul(buf, 0, &val);

	if (ret)
		return ret;
	WRITE_ONCE(min_bytes, val);
	return count;
}

static struct kobj_attribute offloading_attr = __ATTR_RW(offloading);
static struct kobj_attribute min_bytes_attr = __ATTR_RW(min_bytes);

static struct attribute *uffd_dma_attrs[] = {
	&offloading_attr.attr,
	&nr_dma_chan_attr.attr,
	&min_bytes_attr.attr,
	&cpu_pct_attr.attr,
	&cache_ctrl_attr.attr,
	&wait_mode_attr.attr,
	&spin_us_attr.attr,
	&sg_elems_attr.attr,
	&util_gate_attr.attr,
	&copies_done_attr.attr,
	&copies_failed_attr.attr,
	&copies_refused_attr.attr,
	&copies_efault_attr.attr,
	&bytes_dma_attr.attr,
	&bytes_cpu_attr.attr,
	&wait_ns_total_attr.attr,
	&pin_us_total_attr.attr,
	&map_us_total_attr.attr,
	&unmap_us_total_attr.attr,
	&src_contig_attr.attr,
	&sg_txns_attr.attr,
	NULL
};

static const struct attribute_group uffd_dma_attr_group = {
	.attrs = uffd_dma_attrs,
};

static int __init uffd_dma_init(void)
{
	int ret;

	uffd_dma_kobj = kobject_create_and_add("uffd_dma", kernel_kobj);
	if (!uffd_dma_kobj)
		return -ENOMEM;
	ret = sysfs_create_group(uffd_dma_kobj, &uffd_dma_attr_group);
	if (ret) {
		kobject_put(uffd_dma_kobj);
		return ret;
	}
	pr_info("uffd_dma: DMA user page copy offload initialized\n");
	return 0;
}

static void __exit uffd_dma_exit(void)
{
	mutex_lock(&uffd_dma_mutex);
	if (offloading_enabled) {
		mm_offload_unregister(&uffd_dma_copier);
		mm_offload_dma_pool_put();
		offloading_enabled = false;
	}
	mutex_unlock(&uffd_dma_mutex);
	sysfs_remove_group(uffd_dma_kobj, &uffd_dma_attr_group);
	kobject_put(uffd_dma_kobj);
	pr_info("uffd_dma: DMA user page copy offload unloaded\n");
}

module_init(uffd_dma_init);
module_exit(uffd_dma_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Daniel Byrne");
MODULE_DESCRIPTION("DMA user page copy offload for UFFDIO_COPY");
