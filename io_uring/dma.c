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
#include <linux/timekeeping.h>
#include <linux/debugfs.h>
#include <linux/seq_file.h>
#include <linux/kthread.h>
#include <linux/pm_qos.h>
#include <linux/task_work.h>
#include "io_uring.h"
#include "rsrc.h"
#include "refs.h"

#ifndef pr_fmt
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt
#endif

struct kmem_cache *dma_cachep;

/*
 * DSA cache control: when enabled (1, the default), destination writes are
 * cache-allocating (IDXD_OP_FLAG_CC) so data the application reads back
 * immediately is warm in cache; when disabled (0), writes bypass cache.
 * Configurable via /proc/sys/kernel/io_uring_dma_cache_control.
 */
unsigned int io_dma_cache_control __read_mostly = 1;

static inline unsigned long io_dma_prep_flags(void)
{
	return READ_ONCE(io_dma_cache_control) ? DMA_PREP_CACHE_CONTROL : 0;
}

/*
 * Busy-poll budget (in microseconds) for draining in-flight DMA completions
 * from the CQ-wait path (io_dma_cq_wait_poll()) before the waiting task
 * commits to sleeping. A DSA transfer for a typical read chunk completes
 * in ~4-6us plus queueing; sleeping instead costs a kworker schedule_work +
 * wakeup (~+10us when the poller is idle) just to be woken again. 0
 * disables the poll. Tunable via debugfs io_uring_dma_cq_poll_us.
 */
static unsigned int io_dma_cq_poll_us __read_mostly = 20;

/*
 * Optional global CPU latency QoS request. DMA completion latency includes
 * waking a kworker on an idle CPU; on a
 * large idle machine those wakeups pay deep C-state exit latency. Writing
 * N >= 0 (microseconds) installs/updates a cpu_latency_qos request bounding
 * C-state exit latency machine-wide; writing a negative value removes it.
 * Default: no request (-1), so power policy is unchanged until opted in.
 */
static DEFINE_MUTEX(io_dma_cpu_lat_lock);
static struct pm_qos_request io_dma_cpu_lat_qos;
static int io_dma_cpu_lat_us = -1;

static int io_dma_cpu_lat_show(struct seq_file *m, void *v)
{
	seq_printf(m, "%d\n", READ_ONCE(io_dma_cpu_lat_us));
	return 0;
}

static int io_dma_cpu_lat_open(struct inode *inode, struct file *file)
{
	return single_open(file, io_dma_cpu_lat_show, NULL);
}

static ssize_t io_dma_cpu_lat_write(struct file *file, const char __user *ubuf,
				    size_t len, loff_t *ppos)
{
	int val, ret;

	ret = kstrtoint_from_user(ubuf, len, 0, &val);
	if (ret)
		return ret;

	mutex_lock(&io_dma_cpu_lat_lock);
	if (val < 0) {
		if (cpu_latency_qos_request_active(&io_dma_cpu_lat_qos))
			cpu_latency_qos_remove_request(&io_dma_cpu_lat_qos);
		val = -1;
	} else if (cpu_latency_qos_request_active(&io_dma_cpu_lat_qos)) {
		cpu_latency_qos_update_request(&io_dma_cpu_lat_qos, val);
	} else {
		cpu_latency_qos_add_request(&io_dma_cpu_lat_qos, val);
	}
	WRITE_ONCE(io_dma_cpu_lat_us, val);
	mutex_unlock(&io_dma_cpu_lat_lock);
	return len;
}

static const struct file_operations io_dma_cpu_lat_fops = {
	.owner		= THIS_MODULE,
	.open		= io_dma_cpu_lat_open,
	.read		= seq_read,
	.write		= io_dma_cpu_lat_write,
	.llseek		= seq_lseek,
	.release	= single_release,
};

void io_dma_debugfs_init(void)
{
	debugfs_create_u32("io_uring_dma_cq_poll_us", 0644, NULL,
			   &io_dma_cq_poll_us);
	debugfs_create_file("io_uring_dma_cpu_latency_us", 0644, NULL, NULL,
			    &io_dma_cpu_lat_fops);
}

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
	} while (io_dma_pending(ctx));
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
	/*
	 * dma_ref_held is deliberately NOT reset here: it is set/cleared
	 * only under ctx->dma.lock by the submit/complete ref protocol, and
	 * a reissue can reach this prep while the previous cycle's
	 * completer may still be about to drop the previous in-flight ref.
	 * Clearing the flag here would erase that pending drop and leak one
	 * req reference per race.  A fresh req needs no init: the submit
	 * path sets it.
	 */
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
					   src_sgl, nr_entries,
					   io_dma_prep_flags());
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
				       entry->src_len, io_dma_prep_flags());
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

		/*
		 * Drop the in-flight DMA reference taken in
		 * io_dma_submit_queued_tasks(). The completion handling above
		 * only queues task_work — it does not free the req inline —
		 * so the req is still valid here. If we held the last
		 * reference (the req was already terminally
		 * completed/cancelled, e.g. by teardown), free it now;
		 * otherwise the owner frees it once it drops its reference.
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
			 * Take the in-flight DMA reference BEFORE the tasks
			 * become reachable by any completer. The hardware
			 * doorbell was already rung by the submit path
			 * (io_dma_filemap_read()), so the instant the tasks
			 * are published to the submit_list below a
			 * concurrent drain -- the poll_work kworker or
			 * io_ring_exit_work -- can reach dma_refcnt == 0 in
			 * __io_dma_task_complete(). Were the ref taken after
			 * publishing, that drain would complete the req, see
			 * dma_ref_held == false and drop nothing, and this path
			 * would then set refcount/dma_ref_held on an
			 * already-completed req: double-complete / orphaned ref /
			 * use-after-free. Taking it here (before publish)
			 * closes the window; dropped in
			 * __io_dma_task_complete() at dma_refcnt == 0.
			 * Mirrors io_wq_submit_work().
			 *
			 * The take must ALSO be under ctx->dma.lock, the lock
			 * every completer holds through its dma_refcnt == 0
			 * block. A reissue can reach here while the PREVIOUS
			 * cycle's completer may still be between completing the
			 * request and dropping the previous in-flight ref.
			 * Taken unlocked, this ref/flag write interleaves into
			 * that window: the completer then either skips its drop
			 * or donates it to the new cycle, and one reference
			 * leaks each time. The lock orders this take strictly
			 * after that drop.
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
			 * submit_list -- AFTER the ref-take above, so no
			 * drain can reach dma_refcnt == 0 before the
			 * in-flight ref exists.  Every task was fully
			 * submitted with a valid cookie (tasks are linked
			 * only after dmaengine_submit() succeeds), so the
			 * poller can consume them as-is.  Publishing in
			 * chain order makes the llist a LIFO of a FIFO; the
			 * consumer reverses it back (submissions are
			 * serialized by uring_lock, so no other producer
			 * interleaves).
			 */
			struct io_dma_task *t = req->dma.dma_tasks;

			while (t) {
				/* read ->next BEFORE publishing: a
				 * published task can complete and be
				 * freed immediately */
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
	 * Use ctx (not req) below — __io_dma_poll may complete the
	 * request and free it, so req must not be dereferenced after.
	 */
	if (ret == -EIOCBQUEUED) {
		/*
		 * This req's tasks were just queued and the issuer is about to
		 * return with the req handed off. Do NOT complete them inline
		 * here: a fast transfer (especially a single batched
		 * descriptor) can finish synchronously, and
		 * __io_dma_task_complete() would then complete this same req
		 * while the issuer is still unwinding. Defer all completion to
		 * a clean context: the poll_work kworker.
		 */
		if (io_dma_pending(ctx)) {
			/*
			 * Unbound, NOT schedule_work(): the per-CPU
			 * pool would run the poller on THIS (the
			 * submitter's) CPU, so detection while the app
			 * computes depends on the kworker winning a
			 * wakeup-preemption fight with the app thread.
			 * That fight is fragile -- measured: an inline
			 * CQ-wait drain shifting the kworkers' runtime
			 * profile was enough to make preemption stop
			 * happening, leaving completions undetected for
			 * the whole app compute slice (~83us) instead
			 * of ~10us. An unbound worker lands on an idle
			 * CPU and detects in parallel with the app.
			 */
			queue_work(system_unbound_wq,
				   &ctx->dma.poll_work);
		}
	} else if (atomic_read(&ctx->dma.poll_armed) == 0) {
		/*
		 * No task was queued for this req (e.g. CPU fallback). Draining
		 * other reqs' already-queued tasks inline is safe — completion
		 * for a different req is ordinary async wakeup, not re-entrancy
		 * on the req currently being issued.
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
	 * No doorbell here: pollable descriptors reach the hardware at
	 * dmaengine_submit() time (see idxd_dma_tx_submit), and every
	 * submitter rings for its own tasks from the issue path, under
	 * uring_lock. Nothing on the pending lists needs a kick.
	 */
	dev = ctx->dma.chan->device->dev;

	/*
	 * Splice newly submitted tasks onto the consumer-owned poll_list.
	 * Only the armed poller (poll_armed) touches poll_list, so no lock:
	 * the llist arrives newest-first, reversing it restores submission
	 * order, and appending at the tail keeps the whole list in cookie
	 * order -- which the early-break below relies on (per-channel
	 * completion is in-order).
	 */
	{
		struct llist_node *node = llist_del_all(&ctx->dma.submit_list);
		struct io_dma_task *fifo_head = NULL, *fifo_tail = NULL;

		while (node) {
			struct io_dma_task *t =
				llist_entry(node, struct io_dma_task, llnode);

			node = node->next;
			t->next = fifo_head;	/* prepend reverses LIFO->FIFO */
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

	dma = ctx->dma.poll_list;
	count = 0;
	pr_debug("poll: head=%p\n", dma);
	while (dma != NULL) {
		next = dma->next;

		/* No lock around the hardware poll: the cookie state is the
		 * dmaengine's own (safe lockless), and holding ctx->dma.lock
		 * across the whole walk is what made submitters fight the
		 * poller for it (measured 3.6% of node cycles in
		 * queued_spin_lock_slowpath under 60KB sets). */
		ret = dmaengine_async_is_tx_complete(ctx->dma.chan,
						     dma->cookie);
		if (ret == DMA_IN_PROGRESS)
			break;

		/* Unlink before completing (complete may free dma) */
		WRITE_ONCE(ctx->dma.poll_list, next);
		if (!next)
			ctx->dma.poll_list_tail = NULL;

		/* ctx->dma.lock still serializes the completion itself:
		 * the non-atomic dma_refcnt / dma_ref_held handshake with
		 * the submitter's ref-take and the IRQ completion path
		 * depends on it.  The hold is now just this one call. */
		spin_lock_irqsave(&ctx->dma.lock, flags);
		__io_dma_task_complete(dev, dma, ret);
		spin_unlock_irqrestore(&ctx->dma.lock, flags);

		count++;
		dma = next;
	}

	pr_debug("poll: completed=%d remaining=%s\n",
		 count, io_dma_pending(ctx) ? "yes" : "no");

out_disarm:
	/* Release ordering hands poll_list (written lock-free above) to
	 * whichever thread arms the poller next. */
	atomic_set_release(&ctx->dma.poll_armed, 0);
	return io_dma_pending(ctx) ? 1 : 0;
}

/*
 * Called from io_cqring_wait_schedule() right before the task would block.
 * If this ring has pollable (non-IRQ) DMA tasks in flight, spin on their
 * completion records for at most io_dma_cq_poll_us. Completions found here
 * run __io_dma_task_complete() in this task's context; the CQE itself is
 * posted by the poll task_work that queues (io_poll_kick -> this task), so
 * report progress via the pending-work checks and let the wait loop run it.
 *
 * Returns true if the caller should skip sleeping and re-run its wait loop
 * (task_work/CQEs are ready), false to fall through to schedule(). IRQ-mode
 * tasks never appear on the pending lists, so this is a no-op for them, as it
 * for rings with no DMA channel.
 */
bool io_dma_cq_wait_poll(struct io_ring_ctx *ctx, struct io_wait_queue *iowq)
{
	unsigned int budget_us = READ_ONCE(io_dma_cq_poll_us);
	u64 end_ns;

	if (!budget_us || IS_ERR_OR_NULL(ctx->dma.chan))
		return false;
	if (!io_dma_pending(ctx))
		return false;

	end_ns = ktime_get_ns() + (u64)budget_us * NSEC_PER_USEC;
	for (;;) {
		__io_dma_poll(ctx);

		if (task_work_pending(current) || io_local_work_pending(ctx) ||
		    io_should_wake(iowq))
			return true;
		if (!io_dma_pending(ctx))
			return false;
		if (need_resched() || task_sigpending(current))
			return false;
		if (ktime_get_ns() >= end_ns)
			return false;
		cpu_relax();
	}
}

