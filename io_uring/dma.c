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
#include "io_uring.h"
#include "rsrc.h"

#ifndef pr_fmt
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt
#endif

struct kmem_cache *dma_cachep;

/* Datapath alloc: pool first, then non-blocking slab, never sleeps. */
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
		memset(t, 0, sizeof(*t));	/* match old kmem_cache_zalloc */
		return t;
	}

	/*
	 * Pool exhausted: NOWAIT so we never enter reclaim on the datapath.
	 * NOWARN because failure is expected and handled (CPU-copy fallback).
	 */
	return kmem_cache_zalloc(dma_cachep, GFP_NOWAIT | __GFP_NOWARN);
}

/* Datapath free: park back into the pool up to the cap, else slab. */
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
 * Called from io_allocate_dma_chan() at ring setup (process context, no
 * locks held), so GFP_KERNEL is fine here.
 */
void io_dma_init_freelist(struct io_ring_ctx *ctx, struct io_uring_params *p)
{
	struct io_dma_channel *d = &ctx->dma;
	unsigned int n, i;

	spin_lock_init(&d->free_lock);
	d->free_list = NULL;
	d->free_count = 0;

	/*
	 * Cover roughly the in-flight CQ depth; clamp to a sane range.
	 * io_dma_task is tiny (~96B), so even 8192 is ~768KB worst case.
	 */
	n = clamp(p->cq_entries * 2u, 256u, 8192u);
	d->free_max = n;

	for (i = 0; i < n; i++) {
		struct io_dma_task *t = kmem_cache_alloc(dma_cachep, GFP_KERNEL);

		if (!t)
			break;		/* partial prefill OK -- NOWAIT covers rest */
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
	sgls = kmalloc_array(nr_entries * 2, sizeof(*sgls),
			     GFP_NOWAIT | __GFP_NOWARN);
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
				     GFP_NOWAIT | __GFP_NOWARN);
	if (!heap_entries)
		return -ENOMEM;
	memcpy(heap_entries, entries, nr_entries * sizeof(*heap_entries));

	dma = io_dma_task_alloc(req->ctx);
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
		io_dma_task_free(req->ctx, dma);
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

	dma = io_dma_task_alloc(req->ctx);
	if (!dma)
		return -ENOMEM;

	/* Prep after the task alloc: an abandoned prep would orphan an
	 * idxd descriptor from the channel pool. */
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
	dma->src_map_addr = entry->src_dma;
	dma->src_map_len = entry->src_len;
	dma->src_folio = entry->folio;
	dma->src_is_page = true;
	dma->is_batch = false;

	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		folio_put(entry->folio);
		io_dma_task_free(req->ctx, dma);
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

	if (unlikely(iocb->ki_pos < 0))
		return -EINVAL;
	if (unlikely(iocb->ki_pos >= inode->i_sb->s_maxbytes))
		return 0;

	/*
	 * NOWAIT: never block on the read datapath. On failure the caller
	 * (io_read) falls through to the normal buffered-read path.
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
		 * How many bytes remain in the REQUEST.  The registered
		 * buffer (imu) is usually larger than the read: clamping to
		 * imu->len instead of the requested count made a short
		 * READ_FIXED overrun its length and fill the whole buffer
		 * (then trip -EFAULT at the buffer-end dst lookup).
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
				batch_bytes += chunk;

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
					submitted += batch_bytes;
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
		/* Flush any remaining entries from this folio batch */
		if (nr_entries > 0) {
			ssize_t ret;

			ret = io_dma_flush_batch(req, dev, chan,
				entries, nr_entries);
			nr_entries = 0;
			if (ret < 0) {
				if (!error)
					error = ret;
			} else {
				submitted += batch_bytes;
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
	 * may be claimed.  total_read (and ki_pos) run ahead of the flushes
	 * during collection; a failed flush leaves those trailing bytes
	 * uncopied, so clamp the result -- and the file position -- to what
	 * was actually submitted.  Claiming unsubmitted bytes returns
	 * uninitialized destination memory to userspace.
	 */
	if (unlikely(submitted != total_read))
		iocb->ki_pos = start_pos + submitted;
	return submitted ? submitted : error;
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
	io_dma_task_free(req->ctx, dma);
	req->dma.dma_refcnt--;

	if (req->dma.dma_refcnt == 0) {
		pr_debug("dma req done: opcode=%d result=%d\n",
			 req->opcode, req->dma.dma_result);

		if (req->dma.dma_result < 0) {
			req_set_fail(req);
			io_req_set_res(req, req->dma.dma_result,
				       req->dma.saved_cflags);
			req->io_task_work.func = io_req_task_complete;
			io_req_task_work_add(req);
		} else {
			/*
			 * Terminal completion via task_work: __io_dma_poll()
			 * may run from a workqueue without ctx->uring_lock,
			 * so io_req_task_complete() defers the CQE post until
			 * the task_work loop acquires it.
			 */
			io_req_set_res(req, req->dma.saved_res,
				       req->dma.saved_cflags);
			req->io_task_work.func = io_req_task_complete;
			io_req_task_work_add(req);
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
