// SPDX-License-Identifier: GPL-2.0

#include <linux/io_uring_types.h>
#include <linux/io_uring.h>
#include <linux/uio.h>
#include <linux/dma-mapping.h>
#include <linux/spinlock.h>
#include <linux/xarray.h>
#include "io_uring.h"
#include "kbuf.h"
#include "rsrc.h"

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
	if (req->ctx->dma.chan == NULL)
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

static void __io_dma_task_complete(struct device *dev, struct io_dma_task *dma,
				   int ret)
{
	struct io_kiocb *req;
	u32 task_len;

	req = dma->req;
	task_len = dma->len;

	/* Unmap the source DMA mapping */
	if (dma->src_map_len)
		dma_unmap_single(dev, dma->src_map_addr,
				 dma->src_map_len, DMA_TO_DEVICE);

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
		}

		if (req->dma.dma_result < 0) {
			req_set_fail(req);
			io_req_set_res(req, req->dma.dma_result,
				       req->dma.saved_cflags);
		} else {
			io_req_set_res(req, req->dma.saved_res,
				       req->dma.saved_cflags);
		}
		/*
		 * Complete via task_work rather than io_req_complete_defer(),
		 * because __io_dma_poll() runs from a work queue without
		 * ctx->uring_lock held.  io_req_task_complete() will call
		 * io_req_complete_defer() once the lock is acquired in the
		 * task_work run loop.
		 */
		req->io_task_work.func = io_req_task_complete;
		io_req_task_work_add(req);
	}
}

int io_dma_submit_queued_tasks(struct io_kiocb *req)
{
	struct io_ring_ctx *ctx = req->ctx;
	int ret = 0;

	if (!ctx->dma.chan)
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

	if (!ctx->dma.chan)
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
