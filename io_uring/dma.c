// SPDX-License-Identifier: GPL-2.0

#include <linux/io_uring_types.h>
#include <linux/io_uring.h>
#include <linux/uio.h>
#include <linux/dma-mapping.h>
#include "io_uring.h"

#ifndef pr_fmt
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt
#endif

static const char *dma_status_str(int s)
{
	switch (s) {
	case DMA_COMPLETE:    return "DMA_COMPLETE";
	case DMA_IN_PROGRESS: return "DMA_IN_PROGRESS";
	case DMA_PAUSED:      return "DMA_PAUSED";
	case DMA_ERROR:       return "DMA_ERROR";
	default:              return "DMA_?";
	}
}


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
		/* We don't actually know why the prep step failed, so
		* just pick an error code for the most likely reason.
		*/
		return -EAGAIN;
	}
	pr_debug("dma prep OK: len=%zu flags=0x%lx tx=%px\n",
	 (size_t)dma->len, dma->flags, tx);

	dma->cookie = dmaengine_submit(tx);

        pr_debug("dma submit cookie=%d (submit_error=%d)\n",
	dma->cookie, dma_submit_error(dma->cookie));

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
	/* DANGER
	 * for io_recv this field overlaps with msg_flags
	 */
	dma->kiocb.ki_flags |= IOCB_DMA_COPY;
	req->dma.dma_refcnt = 0;
	req->dma.dma_result = 0;
	//req->dma.remaining = req->rw.len;
	req->dma.dma_tasks = NULL;
}

#define IO_DMA_CPU_THRESHOLD 0

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


        pr_debug("io_uring_copy_to_iter: enter to=%px from=%px dst=%zu src=%zu\n",dst_iter, src_iter,
				               iov_iter_count(dst_iter), iov_iter_count(src_iter));
        if (len && len < IO_DMA_CPU_THRESHOLD && iov_iter_is_kvec(src_iter)) {
                /* For small copies, just do it on the CPU */
            size_t left = len;
            ssize_t copied_total = 0;
            while (left > 0) {
                const size_t seg_avail = min_t(size_t, left, iter_iov_len(src_iter));
                size_t copied;
                const void *base;
                if (!seg_avail)
                    break;
                base = iter_iov_addr(src_iter);
                copied = copy_to_iter(base, seg_avail, dst_iter);
                if (!copied) {
                    return copied_total ? copied_total : -EFAULT;
                }
                iov_iter_advance(src_iter, copied);
                copied_total += copied;
                left -= copied;
                if (copied < seg_avail)
                    break;
            }
            pr_debug("io_uring_copy_to_iter: CPU fallback ret=%zd dst_cnt=%zu src_cnt=%zu\n",
                         copied_total, iov_iter_count(dst_iter), iov_iter_count(src_iter));
            return copied_total;
        }

        iov_iter_save_state(dst_iter, &dst_state);
        iov_iter_save_state(src_iter, &src_state);
                   
	//if (len > req->cqe.res) {
	    //pr_err("len %d cqe.res %d\n", len, req->cqe.res);
	//}
	//len = (len > req->cqe.res) ? req->cqe.res : len;

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
		if (dma->src[i].iov_len == 0)
			pr_warn("iov_len is zero, iter count %ld\n", iov_iter_count(src_iter));

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

pr_debug("copy_to_iter (%px <- %px): len=%zu (dst_cnt=%zu src_cnt=%zu) cqe->res %d\n", dst_iter, src_iter,
	 (size_t)len, iov_iter_count(dst_iter), iov_iter_count(src_iter), req->cqe.res);
	 rc = __io_dma_task_submit(ctx->dma.chan, dma);
        //
        // we want sync mode

	if (rc == -EAGAIN) {
	pr_debug("submit returns EAGAIN; deferring (cookie=0)\n");
		/*
		* Continue on and resubmit this operation when another one completes.
		*/
		dma->cookie = 0;
	} else if (rc != 0) {
	pr_err("submit failed: rc=%d\n", rc);
		goto error_free;
	}

	req->dma.dma_refcnt++;
pr_debug("queued dma task %px cookie=%d refcnt=%d\n",
	 dma, dma->cookie, req->dma.dma_refcnt);

	if (!req->dma.dma_tasks)
		req->dma.dma_tasks = dma;
	else {
		tmp = req->dma.dma_tasks;
		while (tmp->next)
			tmp = tmp->next;
		tmp->next = dma;
	}

        //do {
        //        __io_dma_poll(ctx);
        //        cpu_relax();
        //} while (READ_ONCE(ctx->dma.head));
        //__
        //while (atomic_read(&ctx->dma.poll_armed) == 1) {
        //    //pr_debug("polling: dma=%px\n",ctx->dma.chan);
        //    
        //}

	iov_iter_restore(dst_iter, &dst_state);
	iov_iter_restore(src_iter, &src_state);
        pr_debug("io_uring_copy_to_iter: exit  ret=%d dst_cnt=%zu src_cnt=%zu\n",
			                 len, iov_iter_count(dst_iter), iov_iter_count(src_iter));
	return len;

error_free:
	        pr_debug("error free io_uring_copy_to_iter: exit  ret=%d dst_cnt=%zu src_cnt=%zu\n",
			                 len, iov_iter_count(dst_iter), iov_iter_count(src_iter));
	kmem_cache_free(dma_cachep, dma);
error_unmap:
	        pr_debug("error unmap io_uring_copy_to_iter: exit  ret=%d dst_cnt=%zu src_cnt=%zu\n",
			                 len, iov_iter_count(dst_iter), iov_iter_count(src_iter));
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
pr_debug("task_complete: dma=%px ret=%s(%d) len=%zu prev_total=%d\n",
	 dma, dma_status_str(ret), ret, (size_t)dma->len, req->dma.dma_result);
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
pr_debug("task_complete: refcnt=%d after free\n", req->dma.dma_refcnt);

	if (req->dma.dma_refcnt == 0) {
	pr_debug("finalize: opcode=%d total_res=%d (recv? %d) -> defer CQE\n",
		 req->opcode, req->dma.dma_result, req->opcode == IORING_OP_RECV);
		if (req->opcode != IORING_OP_RECV)
			kiocb_done(req, req->dma.dma_result, NULL, IO_URING_F_COMPLETE_DEFER);
		else
			io_req_set_res(req, dma->len, 0 /*cflags*/);
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
				pr_debug("io_submit_queued: %d cookie=%d\n",
					  dma->len, dma->cookie);

				if (req->ctx->dma.tail == NULL)
					req->ctx->dma.head = dma;
				else
					req->ctx->dma.tail->next = dma;
				req->ctx->dma.tail = dma;

				dma = next;
			}

			req->dma.dma_tasks = NULL;
			ret = -EIOCBQUEUED;
		}

		kiocb->ki_flags &= ~IOCB_DMA_COPY;
	}

	if (atomic_read(&req->ctx->dma.poll_armed) == 0) {
            pr_debug("queueing the poll: dma=%px\n",req->ctx->dma.chan);
	    //queue_work(system_unbound_wq, &req->ctx->dma.poll_work);
            __io_dma_poll(req->ctx);
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
        if (atomic_cmpxchg(&ctx->dma.poll_armed, 0, 1) != 0) {
            return 0;
        }
//pr_info("poll: poller entered ctx=%px chan=%px\n",
//		        ctx, ctx->dma.chan);

	if (!ctx->dma.chan)
		return 0;

	dma_async_issue_pending(ctx->dma.chan);
pr_debug("poll: issue_pending; head=%px tail=%px\n", ctx->dma.head, ctx->dma.tail);

	dev = ctx->dma.chan->device->dev;
    //pr_info("DMA dev=%s copy_align=%u residue_granularity=%u max_sg_burst=%u\n",
///		                dev_driver_string(dev), dev->copy_align, dev->residue_granularity,
//				            dev->max_sg_burst);

	dma = ctx->dma.head;
	count = 0;
	while (dma != NULL) {
		next = dma->next;

		if (dma->cookie == 0)
			break;
pr_debug("poll: check cookie=%d\n", dma->cookie);
		ret = dmaengine_async_is_tx_complete(ctx->dma.chan, dma->cookie);
pr_debug("poll: cookie=%d status=%s(%d)\n",
	 dma->cookie, dma_status_str(ret), ret);
        /* If the operation is still in progress, stop checking */  
                uint64_t a = 0;
		while (ret == DMA_IN_PROGRESS) {
                    if (a % 100000 == 0) {
	                //pr_debug("poll: in progress; stop at this cookie\n");
	                pr_debug("poll: in progress\n");
                    }
                    a++;
    
			/*
			* Stop polling here. We rely on completing operations
			* in submission order for error handling below to be
			* correct. Later entries in this list may well be
			* complete at this point, but we cannot process
			* them yet. Re-ordering, fortunately, is rare.
			*/
			//break;
		    ret = dmaengine_async_is_tx_complete(ctx->dma.chan, dma->cookie);
		}
pr_debug("poll: complete cookie=%d; calling task_complete\n", dma->cookie);
		__io_dma_task_complete(dev, dma, ret);

		count++;
		dma = next;
	}

	/* Remove all the entries we've processed */
	ctx->dma.head = dma;
	if (!dma)
		ctx->dma.tail = NULL;
pr_debug("poll: flushing io_uring completions\n");
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

        atomic_set(&ctx->dma.poll_armed, 0);
	return ctx->dma.head ? 1 : 0;
}
