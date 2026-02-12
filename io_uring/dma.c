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

	pr_debug("dma task submitted: cookie=%d len=%u\n",
		 dma->cookie, dma->len);
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
}

/*
 * Threshold below which we fall back to CPU copy instead of DMA offload.
 * Set to 0 to always use DMA; tune based on DSA descriptor overhead.
 */
#define IO_DMA_CPU_THRESHOLD 8192

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

ssize_t io_uring_copy_to_iter(struct kiocb *kiocb, struct iov_iter *dst_iter,
			struct iov_iter *src_iter,
			void (*cb_fn)(struct kiocb *, void *, int), void *cb_arg,
			unsigned long flags)
{
	struct io_dma *cmd;
	struct io_kiocb *req;
	struct io_ring_ctx *ctx;
	struct io_dma_task *dma;
	struct io_buffer_list *bl;
	struct device *dev;
	size_t len, cpu_bytes, dma_bytes;
	ssize_t cpu_copied;
	void *src_kaddr;
	dma_addr_t src_dma, dst_dma;
	u64 dst_user_addr;
	int rc;

	cmd = container_of(kiocb, struct io_dma, kiocb);
	req = cmd_to_io_kiocb(cmd);
	ctx = req->ctx;

	len = min(iov_iter_count(src_iter), iov_iter_count(dst_iter));

	/* Full CPU copy for small transfers or non-kvec sources */
	if (!len || len < IO_DMA_CPU_THRESHOLD || !iov_iter_is_kvec(src_iter))
		return io_dma_cpu_copy(dst_iter, src_iter, len);

	dev = ctx->dma.chan->device->dev;

	/*
	 * Split copy: CPU handles the first portion synchronously,
	 * DMA handles the remainder asynchronously.
	 */
	cpu_bytes = min_t(size_t, len, IO_DMA_CPU_THRESHOLD);
	if (cpu_bytes > 0) {
		cpu_copied = io_dma_cpu_copy(dst_iter, src_iter, cpu_bytes);
		if (cpu_copied <= 0)
			return cpu_copied;
		cpu_bytes = cpu_copied;
	}

	dma_bytes = len - cpu_bytes;
	if (dma_bytes == 0)
		return cpu_bytes;

	/* Get source kernel address for DMA portion */
	if (!iov_iter_count(src_iter))
		return cpu_bytes;
	src_kaddr = (void *)iter_iov_addr(src_iter);

	/* Determine DMA destination address based on iter type.
	 * After cpu_copy, dst_iter has advanced past cpu_bytes.
	 */
	if (iov_iter_is_bvec(dst_iter)) {
		/* Registered buffer path */
		struct io_mapped_ubuf *imu;

		if (!(req->flags & REQ_F_BUF_NODE) || !req->buf_node)
			goto cpu_fallback_rest;
		imu = req->buf_node->buf;
		dst_dma = io_reg_buf_dma_addr(imu,
				req->dma.dst_user_addr + cpu_bytes);
	} else {
		/* Provided buffer path */
		bl = xa_load(&ctx->io_bl_xa, req->dma.buf_group);
		if (!bl || !bl->dma_addrs)
			goto cpu_fallback_rest;

		if (iter_is_ubuf(dst_iter))
			dst_user_addr = (u64)dst_iter->ubuf + dst_iter->iov_offset;
		else if (iter_is_iovec(dst_iter))
			dst_user_addr = (u64)iter_iov_addr(dst_iter);
		else
			goto cpu_fallback_rest;

		dst_dma = io_kbuf_dma_addr(bl, dst_user_addr);
	}
	if (!dst_dma)
		goto cpu_fallback_rest;

	/* Clamp dma_bytes to what's available in this source segment */
	dma_bytes = min_t(size_t, dma_bytes, iter_iov_len(src_iter));

	/* Map source kernel memory for DMA */
	src_dma = dma_map_single(dev, src_kaddr, dma_bytes, DMA_TO_DEVICE);
	if (dma_mapping_error(dev, src_dma)) {
		pr_debug("dma_map_single failed for src %p len %zu\n",
			 src_kaddr, dma_bytes);
		goto cpu_fallback_rest;
	}

	/* Allocate and populate DMA task */
	dma = kmem_cache_zalloc(dma_cachep, GFP_KERNEL);
	if (!dma) {
		dma_unmap_single(dev, src_dma, dma_bytes, DMA_TO_DEVICE);
		goto cpu_fallback_rest;
	}

	dma->req = req;
	dma->next = NULL;
	dma->src_dma = src_dma;
	dma->dst_dma = dst_dma;
	dma->len = dma_bytes;
	dma->src_map_addr = src_dma;
	dma->src_map_len = dma_bytes;
	dma->cb_fn = cb_fn;
	dma->cb_arg = cb_arg;

	rc = __io_dma_task_submit(ctx->dma.chan, dma);
	if (rc == -EAGAIN) {
		dma->cookie = 0;
	} else if (rc != 0) {
		dma_unmap_single(dev, src_dma, dma_bytes, DMA_TO_DEVICE);
		kmem_cache_free(dma_cachep, dma);
		goto cpu_fallback_rest;
	}

	/*
	 * Issue DMA immediately so hardware starts reading source data
	 * before the skb is freed after sock_recvmsg returns.
	 */
	dma_async_issue_pending(ctx->dma.chan);

	req->dma.dma_refcnt++;

	/* O(1) append using tail pointer */
	if (!req->dma.dma_tasks) {
		req->dma.dma_tasks = dma;
		req->dma.dma_tasks_tail = dma;
	} else {
		req->dma.dma_tasks_tail->next = dma;
		req->dma.dma_tasks_tail = dma;
	}

	/* Advance iterators past the DMA portion */
	iov_iter_advance(src_iter, dma_bytes);
	iov_iter_advance(dst_iter, dma_bytes);

	return cpu_bytes + dma_bytes;

cpu_fallback_rest:
	{
		ssize_t more = io_dma_cpu_copy(dst_iter, src_iter, dma_bytes);
		return cpu_bytes + (more > 0 ? more : 0);
	}
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

	if (dma->cb_fn) {
		struct io_dma *iod = io_kiocb_to_cmd(req, struct io_dma);

		dma->cb_fn(&iod->kiocb, dma->cb_arg,
			   req->dma.dma_result >= 0 ?
			   task_len : req->dma.dma_result);
	}

	/* Free the task before touching refcnt -- task_len saved above */
	kmem_cache_free(dma_cachep, dma);
	req->dma.dma_refcnt--;

	if (req->dma.dma_refcnt == 0) {
		pr_debug("dma req done: opcode=%d result=%d\n",
			 req->opcode, req->dma.dma_result);
		io_req_set_res(req, req->dma.saved_res,
			       req->dma.saved_cflags);
		io_req_complete_defer(req);
	}
}

int io_dma_submit_queued_tasks(struct io_kiocb *req)
{
	int ret = 0;

	if (!req->ctx->dma.chan)
		return 0;

	if (req->dma.dma_active) {
		pr_debug("submit_queued: refcnt=%d\n", req->dma.dma_refcnt);
		if (req->dma.dma_refcnt > 0) {
			struct io_dma_task *t = req->dma.dma_tasks;
			struct io_dma_task *next;
			unsigned long flags;

			spin_lock_irqsave(&req->ctx->dma.lock, flags);
			while (t) {
				next = t->next;

				if (t->cookie == 0 && req->ctx->dma.head == NULL) {
					spin_unlock_irqrestore(&req->ctx->dma.lock, flags);
					pr_err("dma_prep failed unexpectedly\n");
					__io_dma_task_complete(
						req->ctx->dma.chan->device->dev,
						t, DMA_ERROR);
					spin_lock_irqsave(&req->ctx->dma.lock, flags);
					t = next;
					continue;
				}

				t->next = NULL;
				if (req->ctx->dma.tail == NULL)
					req->ctx->dma.head = t;
				else
					req->ctx->dma.tail->next = t;
				req->ctx->dma.tail = t;

				t = next;
			}
			spin_unlock_irqrestore(&req->ctx->dma.lock, flags);

			req->dma.dma_tasks = NULL;
			req->dma.dma_tasks_tail = NULL;
			ret = -EIOCBQUEUED;
		}

		req->dma.dma_active = false;
	}

	if (atomic_read(&req->ctx->dma.poll_armed) == 0)
		__io_dma_poll(req->ctx);

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

	io_submit_flush_completions(ctx);

out_disarm:
	atomic_set(&ctx->dma.poll_armed, 0);
	return ctx->dma.head ? 1 : 0;
}
