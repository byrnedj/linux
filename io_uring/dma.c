// SPDX-License-Identifier: GPL-2.0

#include <linux/io_uring_types.h>
#include <linux/io_uring.h>
#include <linux/uio.h>
#include <linux/dma-mapping.h>
#include <linux/spinlock.h>
#include "io_uring.h"

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
	struct iov_iter src;
	struct iov_iter dst;

	iov_iter_kvec(&src, WRITE, dma->src, IO_DMA_MAX_ELEMENTS, dma->len);
	iov_iter_init(&dst, READ, dma->dst, IO_DMA_MAX_ELEMENTS, dma->len);

	tx = dmaengine_prep_memcpy_sva_kernel_user(chan,
		&dst, &src, dma->flags);
	if (!tx) {
		pr_err("dma prep failed: len=%zu flags=0x%lx\n",
		       (size_t)dma->len, dma->flags);
		/*
		 * We don't actually know why the prep step failed, so
		 * just pick an error code for the most likely reason.
		 */
		return -EAGAIN;
	}

	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		/*
		 * This failure is never due to lack of resources, so we
		 * can really fail the request.
		 */
		pr_debug("dma submit error: cookie=%d len=%zu flags=0x%lx\n",
			 dma->cookie, (size_t)dma->len, dma->flags);
		return -EFAULT;
	}

	pr_debug("dma task submitted: cookie=%d len=%zu flags=0x%lx\n",
		 dma->cookie, (size_t)dma->len, dma->flags);
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
	struct io_dma *dma;

	if (req->ctx->dma.chan == NULL)
		return;

	dma = io_kiocb_to_cmd(req, struct io_dma);
	/* DANGER
	 * for io_recv this field overlaps with msg_flags
	 */
	dma->kiocb.ki_flags |= IOCB_DMA_COPY;
	req->dma.dma_refcnt = 0;
	req->dma.dma_result = 0;
	req->dma.dma_tasks = NULL;
	req->dma.dma_tasks_tail = NULL;
}

/*
 * Threshold below which we fall back to CPU copy instead of DMA offload.
 * Set to 0 to always use DMA; tune based on DSA descriptor overhead.
 */
#define IO_DMA_CPU_THRESHOLD 8192

ssize_t io_uring_copy_to_iter(struct kiocb *kiocb, struct iov_iter *dst_iter,
			struct iov_iter *src_iter,
			void (*cb_fn)(struct kiocb *, void *, int), void *cb_arg,
			unsigned long flags)
{
	struct io_dma *cmd;
	struct io_kiocb *req;
	struct io_ring_ctx *ctx;
	struct io_dma_task *dma;
	int rc, i;
	size_t len, bytes;
	struct iov_iter_state dst_state, src_state;

	/* TODO: Really do not want to assume this is io_rw because we
	 * want to do recv too. */
	cmd = container_of(kiocb, struct io_dma, kiocb);
	req = cmd_to_io_kiocb(cmd);

	ctx = req->ctx;

	len = min(iov_iter_count(src_iter), iov_iter_count(dst_iter));

	if (len && len < IO_DMA_CPU_THRESHOLD && iov_iter_is_kvec(src_iter)) {
		pr_debug("cpu copy fallback: len=%zu threshold=%d\n",
			 len, IO_DMA_CPU_THRESHOLD);
		/* For small copies, just do it on the CPU */
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

	iov_iter_save_state(dst_iter, &dst_state);
	iov_iter_save_state(src_iter, &src_state);

	iov_iter_truncate(dst_iter, len);
	iov_iter_truncate(src_iter, len);

	/* Remove the interrupt flag. We'll poll for completions. */
	flags &= ~(unsigned long)DMA_PREP_INTERRUPT;

	dma = kmem_cache_zalloc(dma_cachep, GFP_KERNEL);
	if (!dma) {
		rc = -ENOMEM;
		goto error_restore;
	}

	dma->req = req;
	dma->next = NULL;

	/* src_iter is really kvecs, but they're the same as iovecs */
	i = 0;
	bytes = 0;
	while (iov_iter_count(src_iter) > 0) {
		dma->src[i].iov_base = (void *)iter_iov_addr(src_iter);
		dma->src[i].iov_len = iter_iov_len(src_iter);
		bytes += dma->src[i].iov_len;
		iov_iter_advance(src_iter, dma->src[i].iov_len);
		if (dma->src[i].iov_len == 0) {
			pr_warn("iov_len is zero, iter count %zu\n",
				iov_iter_count(src_iter));
		}

		i++;

		if (i == IO_DMA_MAX_ELEMENTS) {
			pr_warn("reached max src elements\n");
			len = bytes;
			iov_iter_truncate(dst_iter, len);
			iov_iter_truncate(src_iter, len);
			break;
		}
	}

	i = 0;
	bytes = 0;
	while (iov_iter_count(dst_iter) > 0) {
		if (iter_is_iovec(dst_iter)) {
			dma->dst[i].iov_base = (void __user *)iter_iov_addr(dst_iter);
			dma->dst[i].iov_len = iter_iov_len(dst_iter);
		} else if (iter_is_ubuf(dst_iter)) {
			dma->dst[i].iov_base = dst_iter->ubuf;
			dma->dst[i].iov_len = dst_iter->count;
		} else {
			WARN_ONCE(1, "Unknown user iterator type %d\n",
				  dst_iter->iter_type);
		}
		bytes += dma->dst[i].iov_len;
		iov_iter_advance(dst_iter, dma->dst[i].iov_len);
		i++;

		if (i == IO_DMA_MAX_ELEMENTS) {
			pr_warn("reached max dst elements\n");
			len = bytes;
			iov_iter_truncate(dst_iter, len);
			iov_iter_truncate(src_iter, len);
			break;
		}
	}

	dma->flags = flags;
	dma->len = len;
	dma->cb_fn = cb_fn;
	dma->cb_arg = cb_arg;

	pr_debug("copy_to_iter: submitting dma task len=%zu flags=0x%lx\n",
		 len, flags);
	rc = __io_dma_task_submit(ctx->dma.chan, dma);

	if (rc == -EAGAIN) {
		/*
		 * Continue on and resubmit this operation when another
		 * one completes.
		 */
		dma->cookie = 0;
	} else if (rc != 0) {
		goto error_free;
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

	iov_iter_restore(dst_iter, &dst_state);
	iov_iter_restore(src_iter, &src_state);
	return len;

error_free:
	kmem_cache_free(dma_cachep, dma);
error_restore:
	iov_iter_restore(dst_iter, &dst_state);
	iov_iter_restore(src_iter, &src_state);
	return rc;
}
EXPORT_SYMBOL_GPL(io_uring_copy_to_iter);

static void __io_dma_task_complete(struct device *dev, struct io_dma_task *dma,
				   int ret)
{
	struct io_kiocb *req;
	u32 task_len;

	req = dma->req;
	task_len = dma->len;

	if (ret == DMA_COMPLETE) {
		pr_debug("dma task complete: len=%u result=%d\n",
			 task_len, req->dma.dma_result);
		/*
		 * If this DMA was successful and no earlier DMA failed,
		 * we increment the total amount copied. Preserve
		 * earlier failures otherwise.
		 */
		if (req->dma.dma_result >= 0)
			req->dma.dma_result += task_len;
	} else {
		pr_debug("dma task failed: len=%u ret=%d\n", task_len, ret);
		/*
		 * If this DMA failed, report the whole operation
		 * as a failure. Some data may have been copied
		 * as part of an earlier DMA operation that will
		 * be ignored.
		 */
		req->dma.dma_result = -EFAULT;
	}

	if (dma->cb_fn) {
		struct io_dma *iod = io_kiocb_to_cmd(req, struct io_dma);

		dma->cb_fn(&iod->kiocb, dma->cb_arg,
			   req->dma.dma_result >= 0 ?
			   task_len : req->dma.dma_result);
	}

	/* Free the task before touching refcnt — task_len saved above */
	kmem_cache_free(dma_cachep, dma);
	req->dma.dma_refcnt--;

	if (req->dma.dma_refcnt == 0) {
		pr_debug("dma req done: opcode=%d result=%d\n",
			 req->opcode, req->dma.dma_result);
		if (req->opcode == IORING_OP_RECV)
			io_req_set_res(req, task_len, 0 /*cflags*/);
		else
			io_req_set_res(req, req->dma.dma_result, 0);
		/* Queue the task for processing completion later */
		io_req_complete_defer(req);
	}
}

int io_dma_submit_queued_tasks(struct io_kiocb *req)
{
	struct io_dma *dma = io_kiocb_to_cmd(req, struct io_dma);
	struct kiocb *kiocb = &dma->kiocb;
	int ret = 0;

	if (!req->ctx->dma.chan)
		return 0;

	if ((kiocb->ki_flags & IOCB_DMA_COPY) != 0) {
		pr_debug("submit_queued: refcnt=%d\n", req->dma.dma_refcnt);
		if (req->dma.dma_refcnt > 0) {
			struct io_dma_task *t = req->dma.dma_tasks;
			struct io_dma_task *next;
			unsigned long flags;

			spin_lock_irqsave(&req->ctx->dma.lock, flags);
			while (t) {
				next = t->next;

				if (t->cookie == 0 && req->ctx->dma.head == NULL) {
					/*
					 * If the list is empty and cookie is 0,
					 * the dma task failed to submit for some
					 * reason other than out of resources.
					 */
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

		kiocb->ki_flags &= ~IOCB_DMA_COPY;
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
		if (ret == DMA_IN_PROGRESS) {
			/*
			 * Stop polling here. We rely on completing operations
			 * in submission order for error handling below to be
			 * correct. Later entries in this list may well be
			 * complete at this point, but we cannot process
			 * them yet.
			 */
			break;
		}

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
			/*
			 * If submission fails here, we know we're not out
			 * of resources. Always fail the task.
			 */
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
