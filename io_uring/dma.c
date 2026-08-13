// SPDX-License-Identifier: GPL-2.0

#include <linux/io_uring_types.h>
#include <linux/io_uring.h>
#include <linux/uio.h>
#include <linux/dma-mapping.h>
#include <linux/spinlock.h>
#include <linux/xarray.h>
#include <linux/pagemap.h>
#include <linux/folio_batch.h>
#include <linux/swap.h>
#include <linux/fs.h>
#include <linux/scatterlist.h>
#include <linux/sizes.h>
#include "io_uring.h"
#include "rsrc.h"
#include "refs.h"

struct kmem_cache *dma_cachep;

/* Datapath allocation takes from the pool first and then falls back
 * to the non-blocking slab. It never sleeps.
 */
static struct io_dma_task *io_dma_task_alloc(struct io_ring_ctx *ctx)
{
	struct io_dma_channel *d = &ctx->dma;
	struct io_dma_task *t;
	unsigned long flags;

	spin_lock_irqsave(&d->free_lock, flags);
	t = d->free_list;
	if (t) {
		d->free_list = t->next;
		d->free_count--;
	}
	spin_unlock_irqrestore(&d->free_lock, flags);

	if (t) {
		memset(t, 0, sizeof(*t));	/* Match the old kmem_cache_zalloc. */
		return t;
	}

	/*
	 * The pool is exhausted. We use NOWAIT so that we never enter
	 * reclaim on the datapath and NOWARN because failure is expected
	 * and handled by the CPU-copy fallback.
	 */
	return kmem_cache_zalloc(dma_cachep, GFP_NOWAIT | __GFP_NOWARN);
}

/* Datapath free parks the task back into the pool up to the cap and
 * otherwise frees it to the slab.
 */
static void io_dma_task_free(struct io_ring_ctx *ctx, struct io_dma_task *t)
{
	struct io_dma_channel *d = &ctx->dma;
	unsigned long flags;

	spin_lock_irqsave(&d->free_lock, flags);
	if (d->free_count < d->free_max) {
		t->next = d->free_list;
		d->free_list = t;
		d->free_count++;
		t = NULL;
	}
	spin_unlock_irqrestore(&d->free_lock, flags);

	if (t)
		kmem_cache_free(dma_cachep, t);
}

/*
 * Prefill the per-ctx io_dma_task pool, sized to the ring's CQ depth.
 * This is called from io_allocate_dma_chan() at ring setup in process
 * context with no locks held, so GFP_KERNEL is fine here.
 */
void io_dma_init_freelist(struct io_ring_ctx *ctx, struct io_uring_params *p)
{
	struct io_dma_channel *d = &ctx->dma;
	unsigned int n, i;

	spin_lock_init(&d->free_lock);
	d->free_list = NULL;
	d->free_count = 0;

	/*
	 * Cover roughly the in-flight CQ depth and clamp to a sane range.
	 * Since io_dma_task is small (about 96 bytes), even 8192 entries
	 * is about 768KB in the worst case.
	 */
	n = clamp(p->cq_entries * 2u, 256u, 8192u);
	d->free_max = n;

	for (i = 0; i < n; i++) {
		struct io_dma_task *t = kmem_cache_alloc(dma_cachep, GFP_KERNEL);

		if (!t)
			break;		/* A partial prefill is fine. NOWAIT covers the rest. */
		t->next = d->free_list;
		d->free_list = t;
		d->free_count++;
	}
}

void io_dma_poll_workfn(struct work_struct *w)
{
	/* work_struct is embedded in ctx->dma (struct io_dma_channel) */
	struct io_dma_channel *d = container_of(w, struct io_dma_channel, poll_work);
	/* io_dma_channel is embedded in io_ring_ctx as 'dma' */
	struct io_ring_ctx *ctx = container_of(d, struct io_ring_ctx, dma);

	unsigned int passes = 0;

	/* Drain until the list is empty, but stay cancelable. A descriptor
	 * stuck DMA_IN_PROGRESS on halted hardware would otherwise spin
	 * this worker forever and deadlock the cancel_work_sync() in ring
	 * teardown. Bound the inline passes and requeue instead, so the
	 * detection latency is unchanged while cancellation can win
	 * between requeues.
	 */
	do {
		__io_dma_poll(ctx);
		cpu_relax();
		if (++passes >= 1024) {
			if (io_dma_pending(ctx))
				queue_work(system_unbound_wq, &d->poll_work);
			return;
		}
	} while (io_dma_pending(ctx));
}

void io_uring_dma_prep(struct io_kiocb *req)
{
	if (IS_ERR_OR_NULL(req->ctx->dma.chan))
		return;

	req->dma.dma_active = true;
	/*
	 * dma_ref_held is deliberately not reset here. It is set and
	 * cleared only under ctx->dma.lock by the submit and complete ref
	 * protocol, and a reissue can reach this prep while the previous
	 * cycle's completer is still about to drop the previous in-flight
	 * ref. Clearing the flag here would erase that pending drop and
	 * leak one req reference per race. A fresh req needs no
	 * initialization because the submit path sets the flag.
	 */
	req->dma.dma_refcnt = 0;
	req->dma.dma_result = 0;
	req->dma.min_fail_off = U32_MAX;
	req->dma.claim_len = 0;
	req->dma.dma_tasks = NULL;
	req->dma.dma_tasks_tail = NULL;
	req->dma.dst_user_addr = 0;
	req->dma.saved_res = 0;
	req->dma.saved_cflags = 0;
}

/*
 * Submit a batch of DMA copy operations as a single DSA batch descriptor.
 * Takes one io_dma_task for the entire batch instead of one per chunk.
 * The entries[] array is copied to a heap allocation for deferred cleanup.
 */
static ssize_t io_dma_submit_batch(struct io_kiocb *req,
				   struct device *dev, struct dma_chan *chan,
				   struct io_dma_batch_entry *entries,
				   unsigned int nr_entries, u32 off)
{
	struct dma_async_tx_descriptor *tx;
	struct io_dma_batch_entry *heap_entries;
	struct scatterlist *sgls, *src_sgl, *dst_sgl;
	struct io_dma_task *dma;
	u32 total_len = 0;
	int i;

	/* All fallible allocations come before the prep. A prepped but
	 * never submitted descriptor cannot be returned to the driver
	 * pool, so abandoning one here would orphan an idxd descriptor.
	 */
	heap_entries = kmalloc_array(nr_entries, sizeof(*heap_entries),
				     GFP_NOWAIT | __GFP_NOWARN);
	if (!heap_entries)
		return -ENOMEM;
	memcpy(heap_entries, entries, nr_entries * sizeof(*heap_entries));

	dma = io_dma_task_alloc(req->ctx);
	if (!dma) {
		kfree(heap_entries);
		return -ENOMEM;
	}

	/* Allocate the src and dst scatterlists together. We initialize
	 * the SG tables so that sg_next() and sg_is_last() work correctly
	 * and then populate the DMA addresses from the entries array.
	 */
	sgls = kmalloc_array(nr_entries * 2, sizeof(*sgls),
			     GFP_NOWAIT | __GFP_NOWARN);
	if (!sgls) {
		io_dma_task_free(req->ctx, dma);
		kfree(heap_entries);
		return -ENOMEM;
	}
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
		io_dma_task_free(req->ctx, dma);
		kfree(heap_entries);
		return -EAGAIN;
	}

	/* The SG arrays are consumed by dmaengine_prep_dma_memcpy_sg().
	 * The driver copies what it needs into batch descriptors.
	 */
	kfree(sgls);

	dma->req = req;
	dma->next = NULL;
	dma->len = total_len;
	dma->off = off;
	dma->is_batch = true;
	dma->batch_nr = nr_entries;
	dma->batch_entries = heap_entries;

	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		kfree(heap_entries);
		io_dma_task_free(req->ctx, dma);
		return -EAGAIN;	/* The WQ may be full. Fall back to CPU copy. */
	}

	/* Take folio references for the duration of the DMA. */
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

#define IO_DMA_BATCH_MIN	8

/*
 * Submit a single DMA descriptor for one batch entry.
 * Used when nr_entries < IO_DMA_BATCH_MIN to avoid batch overhead.
 * The entry already has DMA-mapped src_dma from the caller.
 */
static ssize_t io_dma_submit_single_entry(struct io_kiocb *req,
					  struct dma_chan *chan,
					  struct io_dma_batch_entry *entry,
					  u32 off)
{
	struct dma_async_tx_descriptor *tx;
	struct io_dma_task *dma;

	dma = io_dma_task_alloc(req->ctx);
	if (!dma)
		return -ENOMEM;

	/*
	 * We prep after the task allocation because an abandoned prep
	 * would orphan an idxd descriptor from the channel pool.
	 */
	tx = dmaengine_prep_dma_memcpy(chan, entry->dst_dma, entry->src_dma,
				       entry->src_len, 0);
	if (!tx) {
		io_dma_task_free(req->ctx, dma);
		return -EAGAIN;
	}

	folio_get(entry->folio);

	dma->req = req;
	dma->next = NULL;
	dma->src_dma = entry->src_dma;
	dma->dst_dma = entry->dst_dma;
	dma->len = entry->src_len;
	dma->off = off;
	dma->src_map_addr = entry->src_dma;
	dma->src_map_len = entry->src_len;
	dma->src_folio = entry->folio;
	dma->src_is_page = true;
	dma->is_batch = false;

	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		folio_put(entry->folio);
		io_dma_task_free(req->ctx, dma);
		return -EAGAIN;	/* The WQ may be full. Fall back to CPU copy. */
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
				  unsigned int nr_entries, u32 off)
{
	ssize_t total = 0;
	unsigned int i;
	ssize_t ret;

	if (!nr_entries)
		return 0;

	if (nr_entries < IO_DMA_BATCH_MIN) {
		for (i = 0; i < nr_entries; i++) {
			ret = io_dma_submit_single_entry(req, chan, &entries[i],
							 off + total);
			if (ret < 0) {
				/* A failed submit never consumes the entry's
				 * source mapping, so release this entry and
				 * every remaining one.
				 */
				io_dma_unmap_batch_entries(req, dev,
							  entries + i,
							  nr_entries - i);
				return total > 0 ? total : ret;
			}
			total += ret;
		}
		return total;
	}

	ret = io_dma_submit_batch(req, dev, chan, entries, nr_entries, off);
	if (ret < 0)
		io_dma_unmap_batch_entries(req, dev, entries, nr_entries);
	return ret;
}

/* Reads at or below this size fall back to the CPU copy path */
#define IO_DMA_MIN_READ_BYTES	SZ_16K

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
			    u64 dst_user_addr, size_t want)
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
	ssize_t submitted = 0;
	size_t batch_bytes = 0;
	size_t dst_offset = 0;
	loff_t start_pos = iocb->ki_pos;
	loff_t isize;
	int i, error = 0;
	bool writably_mapped;

	/* At or below this size the descriptor setup and completion
	 * detection cost more than the copy itself, so we take the CPU
	 * path.
	 */
	if (want <= IO_DMA_MIN_READ_BYTES)
		return -EAGAIN;

	if (unlikely(iocb->ki_pos < 0))
		return -EINVAL;
	if (unlikely(iocb->ki_pos >= inode->i_sb->s_maxbytes))
		return 0;

	/*
	 * We never block on the read datapath. On failure the caller
	 * (io_read) continues to the normal buffered-read path.
	 */
	entries = kmalloc_array(IO_DMA_BATCH_MAX, sizeof(*entries),
				GFP_NOWAIT | __GFP_NOWARN);
	if (!entries)
		return -ENOMEM;

	isize = i_size_read(inode);
	if (unlikely(iocb->ki_pos >= isize)) {
		kfree(entries);
		return 0;
	}

	folio_batch_init(&fbatch);

	do {
		size_t count;
		loff_t end_offset;

		cond_resched();

		if (unlikely(iocb->ki_pos >= i_size_read(inode)))
			break;

		/*
		 * count is the number of bytes remaining in the request.
		 * The registered buffer (imu) is usually larger than the
		 * read. Clamping to imu->len instead of the requested count
		 * made a short READ_FIXED overrun its length and fill the
		 * whole buffer, which then tripped -EFAULT at the
		 * buffer-end destination lookup.
		 */
		count = want - dst_offset;
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
				size_t dst_seg_remain;
				size_t chunk;

				dst_dma = io_reg_buf_dma_addr(imu,
						dst_user_addr + dst_offset,
						&dst_seg_remain);
				if (!dst_dma) {
					error = -EFAULT;
					goto flush_and_put;
				}

				chunk = min_t(size_t, bytes - copied,
					      dst_seg_remain);

				src_dma = dma_map_page(dev, &folio->page,
						       offset + copied,
						       chunk, DMA_TO_DEVICE);
				if (dma_mapping_error(dev, src_dma)) {
					error = -EFAULT;
					goto flush_and_put;
				}

				/* Collect entry for batch submission */
				entries[nr_entries].src_dma = src_dma;
				entries[nr_entries].dst_dma = dst_dma;
				entries[nr_entries].src_len = chunk;
				entries[nr_entries].folio = folio;
				nr_entries++;

				copied += chunk;
				dst_offset += chunk;
				batch_bytes += chunk;

				/* Flush the batch if it is full. */
				if (nr_entries == IO_DMA_BATCH_MAX) {
					ssize_t ret;

					ret = io_dma_flush_batch(req, dev, chan,
						entries, nr_entries,
						dst_offset - batch_bytes);
					nr_entries = 0;
					if (ret < 0) {
						error = ret;
						goto put_folios;
					}
					/*
					 * Count only what the flush actually
					 * submitted and stop on a short
					 * flush. The claim must stay a
					 * contiguous prefix. Collecting past
					 * a gap would claim bytes that were
					 * never copied.
					 */
					submitted += ret;
					if (ret < (ssize_t)batch_bytes) {
						batch_bytes = 0;
						goto put_folios;
					}
					batch_bytes = 0;
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
		/* Flush any remaining entries from this folio batch. */
		if (nr_entries > 0) {
			ssize_t ret;

			ret = io_dma_flush_batch(req, dev, chan,
				entries, nr_entries,
				dst_offset - batch_bytes);
			nr_entries = 0;
			if (ret < 0) {
				if (!error)
					error = ret;
			} else {
				/* Partial flushes count only submitted bytes. */
				submitted += ret;
				if (ret < (ssize_t)batch_bytes)
					error = error ? error : -EAGAIN;
				batch_bytes = 0;
			}
		}
put_folios:
		for (i = 0; i < folio_batch_count(&fbatch); i++)
			folio_put(fbatch.folios[i]);
		folio_batch_init(&fbatch);
	} while (dst_offset < want && iocb->ki_pos < isize && !error);

	file_accessed(filp);

	if (req->dma.dma_refcnt > 0)
		dma_async_issue_pending(chan);

	kfree(entries);

	/*
	 * Only bytes whose batch was successfully handed to the DMA engine
	 * may be claimed.  total_read and ki_pos run ahead of the flushes
	 * during collection.  A failed flush leaves those trailing bytes
	 * uncopied, so we clamp the result and the file position to what
	 * was actually submitted.  Claiming unsubmitted bytes returns
	 * uninitialized destination memory to userspace.
	 */
	if (unlikely(submitted != total_read))
		iocb->ki_pos = start_pos + submitted;
	return submitted ? submitted : error;
}

/*
 * Release a completed or aborted task's source resources.  These are
 * the DMA unmaps, which involve IOVA frees and IOTLB work under an
 * IOMMU, and the folio references.  This is the expensive half of
 * completion and needs no ctx->dma.lock.  Callers run it before taking
 * the lock for __io_dma_task_complete() so that the lock hold shrinks
 * to the refcount handshake.
 */
void io_dma_task_release_res(struct io_ring_ctx *ctx, struct device *dev,
			     struct io_dma_task *dma)
{
	if (dma->is_batch) {
		int i;

		for (i = 0; i < dma->batch_nr; i++) {
			dma_unmap_page(dev,
				       dma->batch_entries[i].src_dma,
				       dma->batch_entries[i].src_len,
				       DMA_TO_DEVICE);
			folio_put(dma->batch_entries[i].folio);
		}
		kfree(dma->batch_entries);
	} else {
		if (dma->src_map_len) {
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
}

/*
 * Caller must hold ctx->dma.lock and must have already released the task's
 * source resources via io_dma_task_release_res().
 */
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
		if (req->dma.dma_result >= 0)
			req->dma.dma_result += task_len;
	} else {
		pr_debug("dma task failed: len=%u ret=%d\n", task_len, ret);
		req->dma.dma_result = -EFAULT;
		/* Everything below the lowest failed offset was delivered
		 * by tasks that are disjoint from this one, so it stays
		 * claimable as a contiguous prefix.
		 */
		req->dma.min_fail_off = min(req->dma.min_fail_off, dma->off);
	}

	/* Free the task before touching the refcnt. task_len was saved above. */
	io_dma_task_free(req->ctx, dma);
	req->dma.dma_refcnt--;

	if (req->dma.dma_refcnt == 0) {
		pr_debug("dma req done: opcode=%d result=%d\n",
			 req->opcode, req->dma.dma_result);

		if (req->dma.dma_result < 0) {
			int res = req->dma.dma_result;

			/*
			 * Completion is out of order, so the bytes that
			 * completed need not form a contiguous prefix. The
			 * bytes below the lowest failed offset do, since
			 * every task lying wholly below it succeeded, so a
			 * failure mid-request reports a short read up to
			 * that point plus whatever earlier issues already
			 * delivered (saved_res minus this issue's claim).
			 * Only a request with nothing deliverable at all
			 * fails whole.
			 */
			if (req->dma.min_fail_off != U32_MAX) {
				int prior = req->dma.saved_res -
					    req->dma.claim_len;
				int short_res = prior +
						(int)req->dma.min_fail_off;

				if (short_res > 0)
					res = short_res;
			}
			if (res < 0)
				req_set_fail(req);
			io_req_set_res(req, res, req->dma.saved_cflags);
			req->io_task_work.func = io_req_task_complete;
			io_req_task_work_add(req);
		} else {
			/*
			 * Terminal completion goes through task_work.
			 * __io_dma_poll() may run from a workqueue without
			 * ctx->uring_lock, so io_req_task_complete() defers
			 * the CQE post until the task_work loop acquires it.
			 */
			io_req_set_res(req, req->dma.saved_res,
				       req->dma.saved_cflags);
			req->io_task_work.func = io_req_task_complete;
			io_req_task_work_add(req);
		}

		/*
		 * Drop the in-flight DMA reference taken in
		 * io_dma_submit_queued_tasks(). The completion handling
		 * above only queues task_work and does not free the req
		 * inline, so the req is still valid here. If we held the
		 * last reference, meaning the req was already terminally
		 * completed or cancelled by teardown, we free it now.
		 * Otherwise the owner frees it once it drops its reference.
		 */
		if (req->dma.dma_ref_held) {
			req->dma.dma_ref_held = false;
			if (req_ref_put_and_test(req))
				io_free_req(req);
		}
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
			unsigned long flags;

			/*
			 * Take the in-flight reference before the tasks are
			 * published. The doorbell is already rung, so once
			 * the tasks appear on the submit_list a completer
			 * can reach dma_refcnt == 0, see dma_ref_held ==
			 * false, and complete the req. Stamping the ref
			 * afterwards would double-complete an already
			 * completed req. The reference is dropped in
			 * __io_dma_task_complete() at dma_refcnt == 0,
			 * which mirrors io_wq_submit_work().
			 *
			 * The take must also be under ctx->dma.lock. On
			 * reissue the previous cycle's completer may still
			 * sit between completing the request and dropping
			 * the old ref. An unlocked take interleaves into
			 * that window and leaks one reference each time.
			 * The lock orders this take after that drop.
			 */
			spin_lock_irqsave(&ctx->dma.lock, flags);
			if (!(req->flags & REQ_F_REFCOUNT))
				__io_req_set_refcount(req, 2);
			else
				req_ref_get(req);
			req->dma.dma_ref_held = true;
			spin_unlock_irqrestore(&ctx->dma.lock, flags);

			/*
			 * Publish the req's tasks to the lock-free
			 * submit_list after the ref-take above so that no
			 * drain can reach dma_refcnt == 0 before the
			 * in-flight ref exists.  Every task was fully
			 * submitted with a valid cookie because tasks are
			 * linked only after dmaengine_submit() succeeds, so
			 * the poller can consume them as-is.  Publishing in
			 * chain order makes the llist a LIFO of a FIFO and
			 * the consumer reverses it back.  Submissions are
			 * serialized by uring_lock, so no other producer
			 * interleaves.
			 */
			struct io_dma_task *t = req->dma.dma_tasks;

			while (t) {
				/* We read ->next before publishing because
				 * a published task can complete and be
				 * freed immediately.
				 */
				struct io_dma_task *nxt = t->next;

				llist_add(&t->llnode, &ctx->dma.submit_list);
				t = nxt;
			}

			req->dma.dma_tasks = NULL;
			req->dma.dma_tasks_tail = NULL;
			ret = -EIOCBQUEUED;
		}

		req->dma.dma_active = false;
	}

	/*
	 * Use ctx and not req below. __io_dma_poll() may complete the
	 * request and free it, so req must not be dereferenced after
	 * this point.
	 */
	if (ret == -EIOCBQUEUED) {
		/*
		 * Tasks were just queued and the issuer is handing the req
		 * off. Completing them inline could complete this same req
		 * while the issuer is still unwinding, so we defer to the
		 * poller.
		 */
		if (io_dma_pending(ctx)) {
			/*
			 * We use the unbound workqueue rather than
			 * schedule_work(). A per-CPU kworker must win a
			 * wakeup-preemption fight with the submitting
			 * thread to run, and this shows up as completions
			 * sitting undetected for the application's whole
			 * compute slice. An unbound worker lands on an
			 * idle CPU and detects completions in parallel.
			 */
			queue_work(system_unbound_wq,
				   &ctx->dma.poll_work);
		}
	} else if (atomic_read(&ctx->dma.poll_armed) == 0) {
		/*
		 * Nothing was queued for this req, for example on a CPU
		 * fallback. Draining other reqs' tasks inline is ordinary
		 * async completion and not re-entrancy on the req being
		 * issued.
		 */
		__io_dma_poll(ctx);
	}

	return ret;
}

int __io_dma_poll(struct io_ring_ctx *ctx)
{
	struct io_dma_task *dma, *next;
	int ret;
	struct device *dev;
	int count;
	unsigned long flags;

	if (atomic_cmpxchg(&ctx->dma.poll_armed, 0, 1) != 0)
		return 0;

	if (IS_ERR_OR_NULL(ctx->dma.chan))
		goto out_disarm;

	/*
	 * There is no doorbell here. Pollable descriptors reach the
	 * hardware at dmaengine_submit() time (see idxd_dma_tx_submit)
	 * and every submitter rings for its own tasks from the issue
	 * path under uring_lock. Nothing on the pending lists needs a
	 * kick.
	 */
	dev = ctx->dma.chan->device->dev;

	/*
	 * Splice newly submitted tasks onto the consumer-owned poll_list.
	 * Only the armed poller touches poll_list, so no lock is needed.
	 * The llist arrives newest-first and reversing it restores
	 * submission order.  Completion is not in-order because DSA WQs
	 * are fed by multiple engines, so the walk below scans the whole
	 * list and not just the head.
	 */
	{
		struct llist_node *node = llist_del_all(&ctx->dma.submit_list);
		struct io_dma_task *fifo_head = NULL, *fifo_tail = NULL;

		while (node) {
			struct io_dma_task *t =
				llist_entry(node, struct io_dma_task, llnode);

			node = node->next;
			t->next = fifo_head;	/* Prepending reverses LIFO to FIFO. */
			if (!fifo_head)
				fifo_tail = t;
			fifo_head = t;
		}
		if (fifo_head) {
			if (ctx->dma.poll_list_tail)
				ctx->dma.poll_list_tail->next = fifo_head;
			else
				WRITE_ONCE(ctx->dma.poll_list, fifo_head);
			ctx->dma.poll_list_tail = fifo_tail;
		}
	}

	{
		struct io_dma_task *prev = NULL;

		dma = ctx->dma.poll_list;
		count = 0;
		pr_debug("poll: head=%p\n", dma);
		while (dma != NULL) {
			next = dma->next;

			/* There is no lock around the hardware poll. The
			 * cookie state is the dmaengine's own and is safe to
			 * read locklessly. Holding ctx->dma.lock across the
			 * whole walk made submitters fight the poller for
			 * it.
			 */
			ret = dmaengine_async_is_tx_complete(ctx->dma.chan,
							     dma->cookie);
			if (ret == DMA_IN_PROGRESS) {
				/*
				 * Keep walking. DSA groups feed each WQ from
				 * multiple engines, so completion is not
				 * in-order and stopping at the first
				 * in-flight entry would block every
				 * completed task behind it.  Each extra
				 * check is one completion-record read.
				 */
				prev = dma;
				dma = next;
				continue;
			}

			/* Unlink before completing since complete may free dma. */
			if (prev)
				prev->next = next;
			else
				WRITE_ONCE(ctx->dma.poll_list, next);
			if (!next)
				ctx->dma.poll_list_tail = prev;

			/* The heavy resource release of IOMMU unmaps and
			 * folio puts runs unlocked. ctx->dma.lock then
			 * covers only the refcount handshake with the
			 * submitter's ref-take.
			 */
			io_dma_task_release_res(ctx, dev, dma);
			spin_lock_irqsave(&ctx->dma.lock, flags);
			__io_dma_task_complete(dev, dma, ret);
			spin_unlock_irqrestore(&ctx->dma.lock, flags);

			count++;
			dma = next;
		}
	}

	pr_debug("poll: completed=%d remaining=%s\n",
		 count, io_dma_pending(ctx) ? "yes" : "no");

out_disarm:
	/* Release ordering hands poll_list, which was written lock-free
	 * above, to whichever thread arms the poller next.
	 */
	atomic_set_release(&ctx->dma.poll_armed, 0);
	return io_dma_pending(ctx) ? 1 : 0;
}
