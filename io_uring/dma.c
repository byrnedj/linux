// SPDX-License-Identifier: GPL-2.0

#include <linux/io_uring_types.h>
#include <linux/io_uring.h>
#include <linux/uio.h>
#include <linux/dma-mapping.h>
#include <linux/spinlock.h>
#include <linux/xarray.h>
#include <linux/pagemap.h>
#include <linux/pagevec.h>
#include <linux/swap.h>
#include <linux/fs.h>
#include <linux/scatterlist.h>
#include "io_uring.h"
#include "kbuf.h"
#include "rsrc.h"
#include "net.h"
#include "poll.h"

#ifndef pr_fmt
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt
#endif

struct kmem_cache *dma_cachep;

void io_dma_poll_workfn(struct work_struct *w)
{
	/* work_struct is embedded in ctx->dma (struct io_dma_channel) */
	struct io_dma_channel *d = container_of(w, struct io_dma_channel, poll_work);
	/* io_dma_channel is embedded in io_ring_ctx as 'dma' */
	struct io_ring_ctx *ctx = container_of(d, struct io_ring_ctx, dma);

	/* Drain until the list is empty */
	do {
		__io_dma_poll(ctx);
		cpu_relax();
	} while (READ_ONCE(ctx->dma.head));
}

static int __io_dma_task_submit(struct dma_chan *chan, struct io_dma_task *dma)
{
	struct dma_async_tx_descriptor *tx;

	tx = dmaengine_prep_dma_memcpy(chan, dma->dst_dma, dma->src_dma,
				       dma->len, 0);
	if (!tx) {
		pr_err("dma prep failed: len=%u src=0x%llx dst=0x%llx\n",
		       dma->len, (u64)dma->src_dma, (u64)dma->dst_dma);
		return -EAGAIN;
	}

	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		pr_debug("dma submit error: cookie=%d len=%u\n",
			 dma->cookie, dma->len);
		return -EFAULT;
	}

	pr_debug("dma task submitted: cookie=%d len=%u src=0x%llx dst=0x%llx\n",
		 dma->cookie, dma->len,
		 (u64)dma->src_dma, (u64)dma->dst_dma);
	return 0;
}

/* TODO: Laid out like io_rw because we have to get back from this kiocb to the io_kiocb. */
struct io_dma {
	/* NOTE: kiocb has the file as the first member, so don't do it here */
	struct kiocb			kiocb;
	u64				addr;
	u32				len;
	rwf_t				flags;
};

void io_uring_dma_prep(struct io_kiocb *req)
{
	if (IS_ERR_OR_NULL(req->ctx->dma.chan))
		return;

	req->dma.dma_active = true;
	req->dma.dma_refcnt = 0;
	req->dma.dma_result = 0;
	req->dma.dma_tasks = NULL;
	req->dma.dma_tasks_tail = NULL;
	req->dma.dst_user_addr = 0;
	req->dma.saved_res = 0;
	req->dma.saved_cflags = 0;
	req->dma.cb_fn = NULL;
	req->dma.cb_arg = NULL;
	/*
	 * io_recv guards on mshot_in_flight BEFORE calling dma_prep, so by
	 * the time we get here it must already be false. Reset anyway to
	 * cover fresh requests pulled from the slab cache whose prior user
	 * may have set it.
	 */
	req->dma.mshot_in_flight = false;
}

/*
 * Threshold below which we fall back to CPU copy instead of DMA offload.
 * Set to 0 to always use DMA; tune based on DSA descriptor overhead.
 * Configurable via /proc/sys/kernel/io_uring_dma_cpu_threshold.
 */
unsigned int io_dma_cpu_threshold __read_mostly = 16384;

/*
 * CPU copy helper: copies from kvec source to user destination.
 * Returns bytes copied, or negative error.
 */
static ssize_t io_dma_cpu_copy(struct iov_iter *dst_iter,
			       struct iov_iter *src_iter, size_t len)
{
	size_t left = len;
	ssize_t copied_total = 0;

	while (left > 0) {
		const size_t seg_avail = min_t(size_t, left,
			iter_iov_len(src_iter));
		size_t copied;
		const void *base;

		if (!seg_avail)
			break;
		base = iter_iov_addr(src_iter);
		copied = copy_to_iter(base, seg_avail, dst_iter);
		if (!copied)
			return copied_total ? copied_total : -EFAULT;
		iov_iter_advance(src_iter, copied);
		copied_total += copied;
		left -= copied;
		if (copied < seg_avail)
			break;
	}
	return copied_total;
}

/*
 * Resolve destination DMA address for a given offset from the base.
 * Returns the DMA address, or 0 on failure.
 * Sets *folio_remain to bytes remaining in the folio from that address.
 */
static dma_addr_t io_dma_dst_addr(struct io_kiocb *req,
				  struct iov_iter *dst_iter,
				  u64 dst_base_addr, size_t offset,
				  size_t *folio_remain)
{
	struct io_ring_ctx *ctx = req->ctx;
	dma_addr_t dst_dma;
	unsigned int folio_shift;

	if (iov_iter_is_bvec(dst_iter)) {
		struct io_mapped_ubuf *imu;

		if (!(req->flags & REQ_F_BUF_NODE) || !req->buf_node)
			return 0;
		imu = req->buf_node->buf;
		dst_dma = io_reg_buf_dma_addr(imu, dst_base_addr + offset);
		folio_shift = imu->folio_shift;
	} else {
		struct io_buffer_list *bl;

		bl = xa_load(&ctx->io_bl_xa, req->dma.buf_group);
		if (!bl || !bl->dma_addrs)
			return 0;
		dst_dma = io_kbuf_dma_addr(bl, dst_base_addr + offset);
		folio_shift = bl->dma_folio_shift;
	}

	if (!dst_dma)
		return 0;

	*folio_remain = (1UL << folio_shift) -
			(dst_dma & ((1UL << folio_shift) - 1));
	return dst_dma;
}

/*
 * Submit a single DMA chunk: map source, allocate task, submit descriptor.
 * Returns bytes submitted (>0), or 0 on failure (caller should CPU-copy).
 */
static size_t io_dma_submit_chunk(struct io_kiocb *req,
				  struct device *dev, struct dma_chan *chan,
				  void *src_kaddr, dma_addr_t dst_dma,
				  size_t chunk_len)
{
	struct io_dma_task *dma;
	dma_addr_t src_dma;
	int rc;

	src_dma = dma_map_single(dev, src_kaddr, chunk_len, DMA_TO_DEVICE);
	if (dma_mapping_error(dev, src_dma))
		return 0;

	dma = kmem_cache_zalloc(dma_cachep, GFP_KERNEL);
	if (!dma) {
		dma_unmap_single(dev, src_dma, chunk_len, DMA_TO_DEVICE);
		return 0;
	}

	dma->req = req;
	dma->next = NULL;
	dma->src_dma = src_dma;
	dma->dst_dma = dst_dma;
	dma->len = chunk_len;
	dma->src_map_addr = src_dma;
	dma->src_map_len = chunk_len;

	rc = __io_dma_task_submit(chan, dma);
	if (rc == -EAGAIN) {
		dma->cookie = 0;
	} else if (rc != 0) {
		dma_unmap_single(dev, src_dma, chunk_len, DMA_TO_DEVICE);
		kmem_cache_free(dma_cachep, dma);
		return 0;
	}

	req->dma.dma_refcnt++;

	/* O(1) append using tail pointer */
	if (!req->dma.dma_tasks) {
		req->dma.dma_tasks = dma;
		req->dma.dma_tasks_tail = dma;
	} else {
		req->dma.dma_tasks_tail->next = dma;
		req->dma.dma_tasks_tail = dma;
	}

	return chunk_len;
}

/*
 * Submit a single DMA chunk from a page cache folio.
 * Uses dma_map_page() for the source and takes an extra folio reference
 * so the folio stays pinned until DMA completes.
 * Returns bytes submitted (>0), or 0 on failure.
 */
static size_t __maybe_unused io_dma_submit_folio_chunk(struct io_kiocb *req,
					struct device *dev,
					struct dma_chan *chan,
					struct folio *folio,
					size_t folio_offset,
					dma_addr_t dst_dma,
					size_t chunk_len)
{
	struct io_dma_task *dma;
	dma_addr_t src_dma;
	int rc;

	src_dma = dma_map_page(dev, &folio->page, folio_offset,
			       chunk_len, DMA_TO_DEVICE);
	if (dma_mapping_error(dev, src_dma))
		return 0;

	dma = kmem_cache_zalloc(dma_cachep, GFP_KERNEL);
	if (!dma) {
		dma_unmap_page(dev, src_dma, chunk_len, DMA_TO_DEVICE);
		return 0;
	}

	/* Take an extra folio ref for the DMA task lifetime */
	folio_get(folio);

	dma->req = req;
	dma->next = NULL;
	dma->src_dma = src_dma;
	dma->dst_dma = dst_dma;
	dma->len = chunk_len;
	dma->src_map_addr = src_dma;
	dma->src_map_len = chunk_len;
	dma->src_folio = folio;
	dma->src_is_page = true;

	rc = __io_dma_task_submit(chan, dma);
	if (rc == -EAGAIN) {
		dma->cookie = 0;
	} else if (rc != 0) {
		dma_unmap_page(dev, src_dma, chunk_len, DMA_TO_DEVICE);
		folio_put(folio);
		kmem_cache_free(dma_cachep, dma);
		return 0;
	}

	req->dma.dma_refcnt++;

	if (!req->dma.dma_tasks) {
		req->dma.dma_tasks = dma;
		req->dma.dma_tasks_tail = dma;
	} else {
		req->dma.dma_tasks_tail->next = dma;
		req->dma.dma_tasks_tail = dma;
	}

	return chunk_len;
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

	/* Allocate src + dst scatterlists together.
	 * Initialize SG tables so sg_next()/sg_is_last() work correctly,
	 * then populate DMA addresses from the entries array.
	 */
	sgls = kmalloc_array(nr_entries * 2, sizeof(*sgls), GFP_KERNEL);
	if (!sgls)
		return -ENOMEM;
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
					   src_sgl, nr_entries, 0);
	if (!tx) {
		kfree(sgls);
		return -EAGAIN;
	}

	/* SG arrays are consumed by dmaengine_prep_dma_memcpy_sg —
	 * the driver copies what it needs into batch descriptors.
	 */
	kfree(sgls);

	heap_entries = kmalloc_array(nr_entries, sizeof(*heap_entries),
				     GFP_KERNEL);
	if (!heap_entries)
		return -ENOMEM;
	memcpy(heap_entries, entries, nr_entries * sizeof(*heap_entries));

	dma = kmem_cache_zalloc(dma_cachep, GFP_KERNEL);
	if (!dma) {
		kfree(heap_entries);
		return -ENOMEM;
	}

	dma->req = req;
	dma->next = NULL;
	dma->len = total_len;
	dma->is_batch = true;
	dma->batch_nr = nr_entries;
	dma->batch_entries = heap_entries;

	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		kfree(heap_entries);
		kmem_cache_free(dma_cachep, dma);
		return -EFAULT;
	}

	/* Take folio refs for DMA duration */
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

ssize_t io_uring_copy_to_iter(struct kiocb *kiocb, struct iov_iter *dst_iter,
			struct iov_iter *src_iter,
			void (*cb_fn)(struct kiocb *, void *, int), void *cb_arg,
			unsigned long flags)
{
	struct io_dma *cmd;
	struct io_kiocb *req;
	struct io_ring_ctx *ctx;
	struct device *dev;
	size_t len, total_dma;
	unsigned int threshold;
	u64 dst_base_addr;

	cmd = container_of(kiocb, struct io_dma, kiocb);
	req = cmd_to_io_kiocb(cmd);
	ctx = req->ctx;

	len = min(iov_iter_count(src_iter), iov_iter_count(dst_iter));
	threshold = READ_ONCE(io_dma_cpu_threshold);

	/*
	 * Full CPU copy for small transfers or non-kvec sources.
	 * Still store cb_fn on the request so io_dma_submit_queued_tasks
	 * can release the source data after TCP has unlinked the SKB.
	 */
	if (!len || len < threshold || !iov_iter_is_kvec(src_iter)) {
		req->dma.cb_fn = cb_fn;
		req->dma.cb_arg = cb_arg;
		return io_dma_cpu_copy(dst_iter, src_iter, len);
	}

	/*
	 * Compute destination base address once.  DMA addresses are
	 * resolved arithmetically as dst_base_addr + offset, independent
	 * of iterator state, so the inner loop doesn't need to track
	 * iterator positions.
	 */
	if (iov_iter_is_bvec(dst_iter)) {
		if (!(req->flags & REQ_F_BUF_NODE) || !req->buf_node)
			goto cpu_fallback;
		dst_base_addr = req->dma.dst_user_addr;
	} else if (iter_is_ubuf(dst_iter)) {
		dst_base_addr = (u64)dst_iter->ubuf + dst_iter->iov_offset;
	} else if (iter_is_iovec(dst_iter)) {
		dst_base_addr = (u64)iter_iov_addr(dst_iter);
	} else {
		goto cpu_fallback;
	}

	dev = ctx->dma.chan->device->dev;

	/* Store cb_fn/cb_arg at request level — called once when all DMA done */
	req->dma.cb_fn = cb_fn;
	req->dma.cb_arg = cb_arg;

	total_dma = 0;

	/*
	 * Walk all source kvec segments.  For each segment, split into
	 * DMA tasks at destination folio boundaries so no single transfer
	 * crosses an IOMMU mapping.
	 */
	while (total_dma < len) {
		void *seg_base;
		size_t seg_avail, seg_remaining, seg_off;

		seg_avail = iter_iov_len(src_iter);
		if (!seg_avail)
			break;

		seg_base = (void *)iter_iov_addr(src_iter);
		seg_remaining = min_t(size_t, seg_avail, len - total_dma);
		seg_off = 0;

		/* Inner loop: split this source segment at folio boundaries */
		while (seg_off < seg_remaining) {
			size_t folio_remain, chunk_len, submitted;
			dma_addr_t dst_dma;

			dst_dma = io_dma_dst_addr(req, dst_iter,
						  dst_base_addr, total_dma,
						  &folio_remain);
			if (!dst_dma)
				goto cpu_fallback_rest;

			chunk_len = min_t(size_t, seg_remaining - seg_off,
					  folio_remain);

			submitted = io_dma_submit_chunk(req, dev,
							ctx->dma.chan,
							seg_base + seg_off,
							dst_dma, chunk_len);
			if (!submitted)
				goto cpu_fallback_rest;

			seg_off += submitted;
			total_dma += submitted;
		}

		/* Advance source iterator past this segment */
		iov_iter_advance(src_iter, seg_off);
	}

	/* Advance destination iterator past all DMA'd bytes */
	iov_iter_advance(dst_iter, total_dma);

	/* Kick the DMA engine once after all tasks are submitted */
	dma_async_issue_pending(ctx->dma.chan);

	return total_dma;

cpu_fallback_rest:
	/*
	 * DMA submission failed mid-way.  Advance iterators past
	 * already-DMA'd bytes, then CPU-copy the rest.
	 */
	if (total_dma > 0) {
		iov_iter_advance(src_iter, total_dma);
		iov_iter_advance(dst_iter, total_dma);
	}
	{
		size_t remain = len - total_dma;
		ssize_t cpu_ret = io_dma_cpu_copy(dst_iter, src_iter, remain);

		if (cpu_ret > 0)
			total_dma += cpu_ret;
	}
	if (req->dma.dma_refcnt > 0)
		dma_async_issue_pending(ctx->dma.chan);
	return total_dma;

cpu_fallback:
	/*
	 * Store cb_fn so io_dma_submit_queued_tasks can release the
	 * source data after TCP has unlinked the SKB from the recv queue.
	 */
	req->dma.cb_fn = cb_fn;
	req->dma.cb_arg = cb_arg;
	return io_dma_cpu_copy(dst_iter, src_iter, len);
}
EXPORT_SYMBOL_GPL(io_uring_copy_to_iter);

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

	tx = dmaengine_prep_dma_memcpy(chan, entry->dst_dma, entry->src_dma,
				       entry->src_len, 0);
	if (!tx)
		return -EAGAIN;

	dma = kmem_cache_zalloc(dma_cachep, GFP_KERNEL);
	if (!dma)
		return -ENOMEM;

	folio_get(entry->folio);

	dma->req = req;
	dma->next = NULL;
	dma->src_dma = entry->src_dma;
	dma->dst_dma = entry->dst_dma;
	dma->len = entry->src_len;
	dma->src_map_addr = entry->src_dma;
	dma->src_map_len = entry->src_len;
	dma->src_folio = entry->folio;
	dma->src_is_page = true;
	dma->is_batch = false;

	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		folio_put(entry->folio);
		kmem_cache_free(dma_cachep, dma);
		return -EFAULT;
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

	if (req->ctx->dma.use_phys_addrs)
		return;

	for (i = 0; i < nr; i++)
		dma_unmap_page(dev, entries[i].src_dma,
			       entries[i].src_len, DMA_TO_DEVICE);
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
				/* Unmap remaining entries */
				io_dma_unmap_batch_entries(req, dev,
							  entries + i + 1,
							  nr_entries - i - 1);
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
			    u64 dst_user_addr)
{
	struct io_ring_ctx *ctx = req->ctx;
	struct file *filp = iocb->ki_filp;
	struct address_space *mapping = filp->f_mapping;
	struct inode *inode = mapping->host;
	struct io_mapped_ubuf *imu = req->buf_node->buf;
	struct device *dev = ctx->dma.chan->device->dev;
	struct dma_chan *chan = ctx->dma.chan;
	struct folio_batch fbatch;
	struct io_dma_batch_entry *entries;
	unsigned int nr_entries = 0;
	ssize_t total_read = 0;
	size_t dst_offset = 0;
	loff_t isize;
	int i, error = 0;
	bool writably_mapped;

	if (unlikely(iocb->ki_pos < 0))
		return -EINVAL;
	if (unlikely(iocb->ki_pos >= inode->i_sb->s_maxbytes))
		return 0;

	entries = kmalloc_array(IO_DMA_BATCH_MAX, sizeof(*entries), GFP_KERNEL);
	if (!entries)
		return -ENOMEM;

	isize = i_size_read(inode);
	if (unlikely(iocb->ki_pos >= isize))
		return 0;

	folio_batch_init(&fbatch);

	do {
		size_t count;
		loff_t end_offset;

		cond_resched();

		if (unlikely(iocb->ki_pos >= i_size_read(inode)))
			break;

		/* How many bytes remain in the destination buffer */
		count = imu->len - dst_offset;
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
				dma_addr_t dst_dma, src_dma;
				size_t dst_folio_remain;
				size_t chunk;

				if (ctx->dma.use_phys_addrs) {
					dst_dma = io_reg_buf_phys_addr(imu,
							dst_user_addr + dst_offset);
				} else {
					dst_dma = io_reg_buf_dma_addr(imu,
							dst_user_addr + dst_offset);
				}
				if (!dst_dma) {
					error = -EFAULT;
					goto flush_and_put;
				}

				dst_folio_remain = (1UL << imu->folio_shift) -
					((dst_user_addr + dst_offset) &
					 ((1UL << imu->folio_shift) - 1));

				chunk = min_t(size_t, bytes - copied,
					      dst_folio_remain);

				if (ctx->dma.use_phys_addrs) {
					src_dma = page_to_phys(folio_page(folio, 0)) +
						  offset + copied;
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
				entries[nr_entries].dst_dma = dst_dma;
				entries[nr_entries].src_len = chunk;
				entries[nr_entries].folio = folio;
				nr_entries++;

				copied += chunk;
				dst_offset += chunk;

				/* Flush batch if full */
				if (nr_entries == IO_DMA_BATCH_MAX) {
					ssize_t ret;

					ret = io_dma_flush_batch(req, dev, chan,
						entries, nr_entries);
					nr_entries = 0;
					if (ret < 0) {
						error = ret;
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
		/* Flush any remaining entries from this folio batch */
		if (nr_entries > 0) {
			ssize_t ret;

			ret = io_dma_flush_batch(req, dev, chan,
				entries, nr_entries);
			nr_entries = 0;
			if (ret < 0 && !error)
				error = ret;
		}
put_folios:
		for (i = 0; i < folio_batch_count(&fbatch); i++)
			folio_put(fbatch.folios[i]);
		folio_batch_init(&fbatch);
	} while (dst_offset < imu->len && iocb->ki_pos < isize && !error);

	file_accessed(filp);

	if (req->dma.dma_refcnt > 0)
		dma_async_issue_pending(chan);

	kfree(entries);
	return total_read ? total_read : error;
}

/*
 * Task_work fired when a DMA-offloaded multishot recv completes.
 * Posts an intermediate CQE with IORING_CQE_F_MORE and resets the
 * multishot bookkeeping so the next poll wakeup can dispatch another
 * recv into the same request. If posting the aux CQE fails (CQ full),
 * fall back to a terminal completion and let userspace re-arm.
 */
static void io_dma_mshot_retry_tw(struct io_tw_req tw_req, io_tw_token_t tw)
{
	struct io_kiocb *req = tw_req.req;
	s32 res = req->dma.saved_res;
	u32 cflags = req->dma.saved_cflags;
	bool posted;

	io_tw_lock(req->ctx, tw);

	pr_debug("mshot_retry_tw: opcode=%d res=%d cflags=0x%x user_data=0x%llx flags=0x%llx\n",
		 req->opcode, res, cflags,
		 (unsigned long long)req->cqe.user_data,
		 (unsigned long long)req->flags);

	posted = io_req_post_cqe(req, res, cflags | IORING_CQE_F_MORE);
	if (!posted) {
		/* CQ ring is full — terminate multishot with a final CQE. */
		pr_debug("mshot_retry_tw: io_req_post_cqe FAILED, terminating multishot\n");
		io_req_set_res(req, res, cflags);
		io_req_complete_defer(req);
		return;
	}

	io_recv_mshot_dma_retry(req);

	/* Clear per-recv DMA scratch so the next io_recv starts clean. */
	req->dma.saved_res = 0;
	req->dma.saved_cflags = 0;
	req->dma.mshot_in_flight = false;
	pr_debug("mshot_retry_tw: posted aux CQE F_MORE, mshot_in_flight cleared\n");

	/*
	 * If the socket still had data buffered when the recv returned
	 * (F_SOCK_NONEMPTY), no new waker will fire for that already-
	 * queued data. Kick the poll state machine to re-issue io_recv
	 * and drain it — mirrors the goto retry_multishot path in
	 * io_recv_finish.
	 */
	if (cflags & IORING_CQE_F_SOCK_NONEMPTY) {
		pr_debug("mshot_retry_tw: F_SOCK_NONEMPTY set, kicking poll\n");
		io_poll_kick(req);
	}
}

static void __io_dma_task_complete(struct device *dev, struct io_dma_task *dma,
				   int ret)
{
	struct io_kiocb *req;
	u32 task_len;

	req = dma->req;
	task_len = dma->len;

	/* Clean up source DMA mappings and folio references */
	if (dma->is_batch) {
		int i;

		for (i = 0; i < dma->batch_nr; i++) {
			if (!req->ctx->dma.use_phys_addrs)
				dma_unmap_page(dev,
					       dma->batch_entries[i].src_dma,
					       dma->batch_entries[i].src_len,
					       DMA_TO_DEVICE);
			folio_put(dma->batch_entries[i].folio);
		}
		kfree(dma->batch_entries);
	} else {
		if (dma->src_map_len && !req->ctx->dma.use_phys_addrs) {
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

	if (ret == DMA_COMPLETE) {
		pr_debug("dma task complete: len=%u result=%d\n",
			 task_len, req->dma.dma_result);
		if (req->dma.dma_result >= 0)
			req->dma.dma_result += task_len;
	} else {
		pr_debug("dma task failed: len=%u ret=%d\n", task_len, ret);
		req->dma.dma_result = -EFAULT;
	}

	/* Free the task before touching refcnt -- task_len saved above */
	kmem_cache_free(dma_cachep, dma);
	req->dma.dma_refcnt--;

	if (req->dma.dma_refcnt == 0) {
		pr_debug("dma req done: opcode=%d result=%d\n",
			 req->opcode, req->dma.dma_result);

		/* Release source data (e.g. SKB) now that all DMA is done */
		if (req->dma.cb_fn) {
			struct io_dma *iod = io_kiocb_to_cmd(req, struct io_dma);

			req->dma.cb_fn(&iod->kiocb, req->dma.cb_arg,
				       req->dma.dma_result);
			req->dma.cb_fn = NULL;
			req->dma.cb_arg = NULL;
		}

		if (req->dma.dma_result < 0) {
			req_set_fail(req);
			io_req_set_res(req, req->dma.dma_result,
				       req->dma.saved_cflags);
			req->io_task_work.func = io_req_task_complete;
		} else if (req->flags & REQ_F_APOLL_MULTISHOT) {
			/*
			 * Multishot recv via DMA: post an intermediate CQE
			 * with IORING_CQE_F_MORE and leave the request in
			 * flight under the poll machinery. The saved res/cflags
			 * are consumed by io_dma_mshot_retry_tw.
			 */
			req->io_task_work.func = io_dma_mshot_retry_tw;
		} else {
			io_req_set_res(req, req->dma.saved_res,
				       req->dma.saved_cflags);
			req->io_task_work.func = io_req_task_complete;
		}
		/*
		 * Complete via task_work rather than io_req_complete_defer(),
		 * because __io_dma_poll() runs from a work queue without
		 * ctx->uring_lock held.  io_req_task_complete() will call
		 * io_req_complete_defer() once the lock is acquired in the
		 * task_work run loop.
		 */
		io_req_task_work_add(req);
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
			struct io_dma_task *t = req->dma.dma_tasks;
			struct io_dma_task *next;
			unsigned long flags;

			spin_lock_irqsave(&ctx->dma.lock, flags);
			while (t) {
				next = t->next;

				if (t->cookie == 0 && ctx->dma.head == NULL) {
					spin_unlock_irqrestore(&ctx->dma.lock, flags);
					pr_err("dma_prep failed unexpectedly\n");
					__io_dma_task_complete(
						ctx->dma.chan->device->dev,
						t, DMA_ERROR);
					spin_lock_irqsave(&ctx->dma.lock, flags);
					t = next;
					continue;
				}

				t->next = NULL;
				if (ctx->dma.tail == NULL)
					ctx->dma.head = t;
				else
					ctx->dma.tail->next = t;
				ctx->dma.tail = t;

				t = next;
			}
			spin_unlock_irqrestore(&ctx->dma.lock, flags);

			req->dma.dma_tasks = NULL;
			req->dma.dma_tasks_tail = NULL;
			ret = -EIOCBQUEUED;
		} else if (req->dma.cb_fn) {
			/*
			 * CPU fallback path: no DMA tasks were submitted but
			 * TCP already unlinked the SKB from the receive queue
			 * (tcp_eat_recv_skb with free=false).  Release it now.
			 */
			struct io_dma *iod = io_kiocb_to_cmd(req, struct io_dma);

			req->dma.cb_fn(&iod->kiocb, req->dma.cb_arg, 0);
			req->dma.cb_fn = NULL;
		}

		req->dma.dma_active = false;
	}

	/*
	 * Use ctx (not req) below — __io_dma_poll may complete the
	 * request and free it, so req must not be dereferenced after.
	 */
	if (atomic_read(&ctx->dma.poll_armed) == 0)
		__io_dma_poll(ctx);

	/* If DMA tasks are still pending after the synchronous poll,
	 * schedule the poll work to keep draining completions.
	 */
	if (ret == -EIOCBQUEUED && READ_ONCE(ctx->dma.head))
		schedule_work(&ctx->dma.poll_work);

	return ret;
}

int __io_dma_poll(struct io_ring_ctx *ctx)
{
	struct io_dma_task *dma, *next, *prev;
	struct io_kiocb *req;
	int ret;
	struct device *dev;
	int count;
	unsigned long flags;

	if (atomic_cmpxchg(&ctx->dma.poll_armed, 0, 1) != 0)
		return 0;

	if (IS_ERR_OR_NULL(ctx->dma.chan))
		goto out_disarm;

	dma_async_issue_pending(ctx->dma.chan);

	dev = ctx->dma.chan->device->dev;

	spin_lock_irqsave(&ctx->dma.lock, flags);
	dma = ctx->dma.head;
	count = 0;
	pr_debug("poll: head=%p\n", dma);
	while (dma != NULL) {
		next = dma->next;

		if (dma->cookie == 0)
			break;

		ret = dmaengine_async_is_tx_complete(ctx->dma.chan,
						     dma->cookie);
		if (ret == DMA_IN_PROGRESS)
			break;

		/* Unlink before completing (complete may free dma) */
		__io_dma_task_complete(dev, dma, ret);

		count++;
		dma = next;
	}

	/* Remove all the entries we've processed */
	ctx->dma.head = dma;
	if (!dma)
		ctx->dma.tail = NULL;

	/* Try to submit any entries that were queued with cookie==0 */
	prev = NULL;
	while (dma && count > 0) {
		next = dma->next;

		if (dma->cookie != 0) {
			prev = dma;
			dma = next;
			continue;
		}

		req = dma->req;

		ret = __io_dma_task_submit(ctx->dma.chan, dma);
		if (ret != 0) {
			if (prev)
				prev->next = next;
			else
				ctx->dma.head = next;

			if (ctx->dma.tail == dma)
				ctx->dma.tail = prev;

			__io_dma_task_complete(dev, dma, DMA_ERROR);
		} else {
			prev = dma;
			count--;
		}

		dma = next;
	}
	spin_unlock_irqrestore(&ctx->dma.lock, flags);

	pr_debug("poll: completed=%d remaining=%s\n",
		 count, ctx->dma.head ? "yes" : "no");

out_disarm:
	atomic_set(&ctx->dma.poll_armed, 0);
	return ctx->dma.head ? 1 : 0;
}
