// SPDX-License-Identifier: GPL-2.0

#include <linux/io_uring_types.h>
#include <linux/io_uring.h>
#include <linux/dma-mapping.h>
#include "io_uring.h"

struct kmem_cache *dma_cachep;

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
		/* We don't actually know why the prep step failed, so
		* just pick an error code for the most likely reason.
		*/
		return -EAGAIN;
	}

	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		/*
		* This failure is never due to lack of resources, so we can really fail
		* the request.
		*/
		return -EFAULT;
	}

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

	dma->kiocb.ki_flags |= IOCB_DMA_COPY;
	req->dma.dma_refcnt = 0;
	req->dma.dma_result = 0;
	req->dma.dma_tasks = NULL;
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
	struct io_dma_task *dma, *tmp;
	int rc, len, i, bytes;
	struct iov_iter_state dst_state, src_state;

	/* TODO: Really do not want to assume this is io_rw because we
	 * want to do recv too. */
	cmd = container_of(kiocb, struct io_dma, kiocb);
	req = cmd_to_io_kiocb(cmd);

	ctx = req->ctx;
	dev = ctx->dma.chan->device->dev;

	iov_iter_save_state(dst_iter, &dst_state);
	iov_iter_save_state(src_iter, &src_state);

	len = (iov_iter_count(src_iter) > iov_iter_count(dst_iter)) ?
		iov_iter_count(dst_iter) : iov_iter_count(src_iter);
	len = (len > req->cqe.res) ? req->cqe.res : len;

	iov_iter_truncate(dst_iter, len);
	iov_iter_truncate(src_iter, len);

	/* Remove the interrupt flag. We'll poll for completions. */
	flags &= ~(unsigned long)DMA_PREP_INTERRUPT;

	dma = kmem_cache_zalloc(dma_cachep, GFP_KERNEL);
	if (!dma) {
		rc = -ENOMEM;
		goto error_unmap;
	}

	dma->req = req;

	/* src_iter is really kvecs, but they're the same as iovecs */
	i = 0;
	bytes = 0;
	while (iov_iter_count(src_iter) > 0) {
		dma->src[i].iov_base = (void *)iter_iov_addr(src_iter);
		dma->src[i].iov_len = iter_iov_len(src_iter);
		bytes += dma->src[i].iov_len;
		iov_iter_advance(src_iter, dma->src[i].iov_len);
		i++;

		if (i == IO_DMA_MAX_ELEMENTS) {
			pr_warn("reached max elements\n");
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
			dma->dst[i].iov_base = (void *)iter_iov_addr(dst_iter);
			dma->dst[i].iov_len = iter_iov_len(dst_iter);
		} else if (iter_is_ubuf(dst_iter)) {
			dma->dst[i].iov_base = dst_iter->ubuf;
			dma->dst[i].iov_len = dst_iter->count;
		} else {
			WARN_ONCE(1, "Unknown user iterator type %d\n", dst_iter->iter_type);
		}
		bytes += dma->dst[i].iov_len;
		iov_iter_advance(dst_iter, dma->dst[i].iov_len);
		i++;

		if (i == IO_DMA_MAX_ELEMENTS) {
			pr_warn("reached max elements\n");
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

	rc = __io_dma_task_submit(ctx->dma.chan, dma);
	if (rc == -EAGAIN) {
		/*
		* Continue on and resubmit this operation when another one completes.
		*/
		dma->cookie = 0;
	} else if (rc != 0) {
		goto error_free;
	}

	req->dma.dma_refcnt++;

	if (!req->dma.dma_tasks)
		req->dma.dma_tasks = dma;
	else {
		tmp = req->dma.dma_tasks;
		while (tmp->next)
			tmp = tmp->next;
		tmp->next = dma;
	}

	iov_iter_restore(dst_iter, &dst_state);
	iov_iter_restore(src_iter, &src_state);

	return len;

error_free:
	kmem_cache_free(dma_cachep, dma);
error_unmap:
	iov_iter_restore(dst_iter, &dst_state);
	iov_iter_restore(src_iter, &src_state);

	return rc;
}
EXPORT_SYMBOL_GPL(io_uring_copy_to_iter);

static void __io_dma_task_complete(struct device *dev, struct io_dma_task *dma, int ret)
{
	struct io_kiocb *req;
	struct iov_iter src;
	struct iov_iter dst;

	iov_iter_kvec(&src, WRITE, dma->src, IO_DMA_MAX_ELEMENTS, dma->len);
	iov_iter_init(&dst, READ, dma->dst, IO_DMA_MAX_ELEMENTS, dma->len);

	req = dma->req;

	if (ret == DMA_COMPLETE) {
		/*
		* If this DMA was successful and no earlier DMA failed,
		* we increment the total amount copied. Preserve
		* earlier failures otherwise.
		*/
		if (req->dma.dma_result >= 0)
			req->dma.dma_result += dma->len;
	} else {
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
		dma->cb_fn(&iod->kiocb, dma->cb_arg, req->dma.dma_result >= 0 ?
						dma->len : req->dma.dma_result);
	}

	kmem_cache_free(dma_cachep, dma);
	req->dma.dma_refcnt--;

	if (req->dma.dma_refcnt == 0)
		kiocb_done(req, req->dma.dma_result, NULL, IO_URING_F_COMPLETE_DEFER);
}

int io_dma_submit_queued_tasks(struct io_kiocb *req)
{
	struct io_dma *dma = io_kiocb_to_cmd(req, struct io_dma);
	struct kiocb *kiocb = &dma->kiocb;
	int ret = 0;

	if ((kiocb->ki_flags & IOCB_DMA_COPY) != 0) {
		if (req->dma.dma_refcnt > 0) {
			struct io_dma_task *dma = req->dma.dma_tasks;
			struct io_dma_task *next;

			while (dma) {
				next = dma->next;

				if (dma->cookie == 0 && req->ctx->dma.head == NULL) {
					/* It's ok if the dma task has not been submitted yet, as
					* long as it isn't being placed at the head of the list.
					* If the list is empty, then this dma task failed to
					* submit for some other reason.
					*/
					pr_err("dma_prep failed for some reason other than out "
						"of resources!\n");
					__io_dma_task_complete(req->ctx->dma.chan->device->dev,
								dma, DMA_ERROR);
					dma = next;
					continue;
				}

				if (req->ctx->dma.tail == NULL)
					req->ctx->dma.head = dma;
				else
					req->ctx->dma.tail->next = dma;
				req->ctx->dma.tail = dma;

				dma = next;
			}

			req->dma.dma_tasks = NULL;
			ret = -EIOCBQUEUED;
			/* Queue the task for processing completion later */
			io_req_complete_defer(req);
		}

		kiocb->ki_flags &= ~IOCB_DMA_COPY;
	}

	return ret;
}

int __io_dma_poll(struct io_ring_ctx *ctx)
{
	struct io_dma_task *dma, *next, *prev;
	struct io_kiocb *req;
	int ret;
	struct device *dev;
	int count;

	if (!ctx->dma.chan)
		return 0;

	dma_async_issue_pending(ctx->dma.chan);

	dev = ctx->dma.chan->device->dev;

	dma = ctx->dma.head;
	count = 0;
	while (dma != NULL) {
		next = dma->next;

		if (dma->cookie == 0)
			break;

		ret = dmaengine_async_is_tx_complete(ctx->dma.chan, dma->cookie);

		if (ret == DMA_IN_PROGRESS) {
			/*
			* Stop polling here. We rely on completing operations
			* in submission order for error handling below to be
			* correct. Later entries in this list may well be
			* complete at this point, but we cannot process
			* them yet. Re-ordering, fortunately, is rare.
			*/
			break;
		}

		__io_dma_task_complete(dev, dma, ret);

		count++;
		dma = next;
	}

	/* Remove all the entries we've processed */
	ctx->dma.head = dma;
	if (!dma)
		ctx->dma.tail = NULL;

	io_submit_flush_completions(ctx);

	/* Try to submit any entries that were queued */
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
			* If submission fails here, we know we're not out of resources.
			* Always fail the task.
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

	return ctx->dma.head ? 1 : 0;
}
