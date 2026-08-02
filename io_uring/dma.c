// SPDX-License-Identifier: GPL-2.0

#include <linux/io_uring_types.h>
#include <linux/io_uring.h>
#include <linux/backing-dev.h>
#include <linux/delay.h>
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
#include <linux/timekeeping.h>
#include <linux/debugfs.h>
#include <linux/task_work.h>
#include "io_uring.h"
#include "rw.h"
#include "rsrc.h"
#include "refs.h"

struct kmem_cache *dma_cachep;

/*
 * DSA cache control. When enabled (1, the default), destination writes
 * are cache-allocating (IDXD_OP_FLAG_CC) so that data the application
 * reads back immediately is warm in cache. When disabled (0), writes
 * bypass the cache. This is configurable via
 * /proc/sys/kernel/io_uring_dma_cache_control.
 */
unsigned int io_dma_cache_control __read_mostly = 1;

static inline unsigned long io_dma_prep_flags(void)
{
	return READ_ONCE(io_dma_cache_control) ? DMA_PREP_CACHE_CONTROL : 0;
}

/*
 * Busy-poll budget in microseconds for draining in-flight DMA
 * completions from the CQ-wait path (io_dma_cq_wait_poll()) before the
 * waiting task commits to sleeping. A DSA transfer for a typical read
 * chunk completes in about 4 to 6us plus queueing. Sleeping instead
 * leaves the completion to the poll worker that the publish queued,
 * plus a wakeup of this task, about 10us more, just to be woken again.
 * 0 disables the poll. This is tunable via debugfs
 * io_uring_dma/cq_poll_us.
 */
static unsigned int io_dma_cq_poll_us __read_mostly = 20;

/*
 * Inline spin budget in microseconds for the DMA filemap-write wait.
 * The write path is synchronous per request. A typical request's batch
 * completes in tens of microseconds, so sleeping immediately trades a
 * usleep wakeup of about 100us for about 35us of engine time and
 * halves throughput at small request sizes. We spin with cond_resched
 * for up to this long before backing off to sleeping. The backoff
 * still protects the many-rings contention case. 0 sleeps immediately.
 * This is tunable via debugfs io_uring_dma/fmw_spin_us.
 */
static unsigned int io_dma_fmw_spin_us __read_mostly = 60;

/*
 * How long the DMA filemap-write wait gives the device before it
 * declares the transfer wedged, leaks the destination mappings and
 * fails the write. Tunable via debugfs io_uring_dma/fmw_wait_ms, which
 * refuses values below IO_DMA_FMW_WAIT_MIN_MS. A wedge leaks the folios
 * the device may still write and loses what they held, so the timeout
 * must stay well above what a healthy but contended queue takes. At 0
 * the deadline is the next tick, and at a few milliseconds the sleep
 * back-off and the queueing of many rings on one work queue already
 * reach it.
 */
#define IO_DMA_FMW_WAIT_MIN_MS	100
static unsigned int io_dma_fmw_wait_ms __read_mostly = 5000;

static int io_dma_fmw_wait_ms_get(void *data, u64 *val)
{
	*val = READ_ONCE(io_dma_fmw_wait_ms);
	return 0;
}

static int io_dma_fmw_wait_ms_set(void *data, u64 val)
{
	if (val > U32_MAX)
		return -ERANGE;
	if (val < IO_DMA_FMW_WAIT_MIN_MS)
		return -EINVAL;
	WRITE_ONCE(io_dma_fmw_wait_ms, val);
	return 0;
}
DEFINE_DEBUGFS_ATTRIBUTE(io_dma_fmw_wait_ms_fops, io_dma_fmw_wait_ms_get,
			 io_dma_fmw_wait_ms_set, "%llu\n");

/*
 * Filemap DMA-write gate and result counters. These are surfaced
 * through the io_uring_dma/latency debugfs stats file and zeroed by
 * its _reset companion.
 */
static const char * const io_dma_fmw_names[IO_DMA_FMW_NR] = {
	[IO_DMA_FMW_ENGAGED] = "engaged",
	[IO_DMA_FMW_UNSUPPORTED] = "unsupported",
	[IO_DMA_FMW_NOT_BVEC] = "not_bvec",
	[IO_DMA_FMW_DIRECT] = "direct",
	[IO_DMA_FMW_NO_DMA_ADDRS] = "no_dma_addrs",
	[IO_DMA_FMW_EAGAIN] = "eagain",
	[IO_DMA_FMW_CPU_REDO] = "cpu_redo",
	[IO_DMA_FMW_ERROR] = "error",
	[IO_DMA_FMW_FS_DECLINE] = "fs_decline",
};
static atomic64_t io_dma_fmw[IO_DMA_FMW_NR];

void io_dma_fmw_record(unsigned int reason)
{
	atomic64_inc(&io_dma_fmw[reason]);
}

static int io_dma_lat_show(struct seq_file *m, void *v)
{
	unsigned int i;

	seq_puts(m, "filemap_write:\n");
	for (i = 0; i < IO_DMA_FMW_NR; i++)
		seq_printf(m, "  %-12s %12llu\n", io_dma_fmw_names[i],
			   atomic64_read(&io_dma_fmw[i]));
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(io_dma_lat);

static ssize_t io_dma_lat_reset_write(struct file *file,
				      const char __user *ubuf,
				      size_t count, loff_t *ppos)
{
	unsigned int i;

	for (i = 0; i < IO_DMA_FMW_NR; i++)
		atomic64_set(&io_dma_fmw[i], 0);
	return count;
}

static const struct file_operations io_dma_lat_reset_fops = {
	.owner		= THIS_MODULE,
	.open		= simple_open,
	.write		= io_dma_lat_reset_write,
	.llseek		= noop_llseek,
};

/*
 * Devices a ring stripes over, one channel each. One channel reaches one
 * device, four engines of sixteen; a single-stream op striped across the
 * devices reaches them all. A ring takes no more devices than a buffer
 * is mapped on (IO_REG_BUF_DEVS, four), so the default takes every
 * device on the ring's node, or, when none of those can give a channel,
 * every device of the nearest distance that can, up to that cap. A ring
 * offered more devices than it takes starts at a rotating one, so a
 * value below the count, or a node with more devices than the cap,
 * trades the cross-ring cache dedup for spread. Values below the default
 * exist for experiments, and 0 takes the default. Read at ring creation.
 */
unsigned int io_dma_stripe_chans __read_mostly = IO_DMA_RING_CHANS;

void io_dma_debugfs_init(void)
{
	struct dentry *dir;

	dir = debugfs_create_dir("io_uring_dma", NULL);

	debugfs_create_u32("cq_poll_us", 0644, dir, &io_dma_cq_poll_us);
	debugfs_create_u32("fmw_spin_us", 0644, dir, &io_dma_fmw_spin_us);
	debugfs_create_file_unsafe("fmw_wait_ms", 0644, dir, NULL,
				   &io_dma_fmw_wait_ms_fops);
	debugfs_create_u32("stripe_chans", 0644, dir, &io_dma_stripe_chans);
	debugfs_create_file("latency", 0444, dir, NULL, &io_dma_lat_fops);
	debugfs_create_file("latency_reset", 0200, dir, NULL,
			    &io_dma_lat_reset_fops);
}

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
				queue_work(system_dfl_wq, &d->poll_work);
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
					   src_sgl, nr_entries,
					   io_dma_prep_flags());
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
	dma->chan = chan;
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
				       entry->src_len, io_dma_prep_flags());
	if (!tx) {
		io_dma_task_free(req->ctx, dma);
		return -EAGAIN;
	}

	folio_get(entry->folio);

	dma->req = req;
	dma->next = NULL;
	dma->chan = chan;
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
	/*
	 * Stripe state. One channel reaches one device, four engines of
	 * sixteen, so a single stream's batches stripe across the ring's
	 * channels. The stripe is deterministic in the file position, at
	 * a 1MB granule, rather than round-robin: every pass over a
	 * region then lands its batches on the same devices, so each
	 * device's PFN cache holds only its own share of the working set
	 * instead of every device converging on a full duplicate copy.
	 * Round-robin cost four times the standing mappings for the same
	 * data, which is what pushed moderate working sets over the IOVA
	 * rcache depletion cliff. Sequential streams and parallel jobs
	 * spread over the granule exactly as they did over the rotation.
	 * Everything device-scoped, meaning the registered-buffer
	 * addresses, the PFN cache instance, and the flush target,
	 * follows the current stripe.
	 */
#define IO_DMA_STRIPE_SHIFT	20
	unsigned int nr_chans = ctx->dma.nr_chans ? ctx->dma.nr_chans : 1;
	/*
	 * Hash the granule index rather than using it raw: block-aligned
	 * op starts otherwise resonate with the stripe modulus, opening
	 * every op on index zero. With rings agreeing on index zero's
	 * device, that device saturates, its batches flush short, the
	 * claims end before the rotation to the next granule ever runs,
	 * and the whole workload converges on one device with CPU tails.
	 * The hash keeps the mapping deterministic, so the dedup holds,
	 * while decorrelating it from any alignment.
	 */
	unsigned int stripe = hash_64((u64)iocb->ki_pos >> IO_DMA_STRIPE_SHIFT,
				      32) % nr_chans;
	struct dma_chan *chan = ctx->dma.nr_chans ?
		ctx->dma.chans[stripe] : ctx->dma.chan;
	struct device *dev = chan->device->dev;
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
	/*
	 * A channel its provider orphaned at unbind completes nothing any
	 * more, and its device may be on its way out. A read that starts
	 * on its stripe takes the CPU path without mapping anything.
	 */
	if (dma_chan_orphaned(chan))
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
						&dst_seg_remain, dev);
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

				/* Flush when the batch is entry-full or
				 * carries this op's share of one engine.
				 * A batch descriptor executes serially on
				 * one engine, so a multi-megabyte op
				 * folded into a single batch runs at
				 * single-engine speed while the rest of
				 * the group idles: 4M reads plateaued at
				 * a third of their rate. Splitting each
				 * op into about four batches, bounded to
				 * [256K, 1M] apiece, spreads it across
				 * engines; the bounds keep small ops from
				 * shattering into per-entry descriptors
				 * and huge ops from oversplitting.
				 */
				if (nr_entries == IO_DMA_BATCH_MAX ||
				    batch_bytes >= clamp(want / 4,
						(size_t)SZ_256K,
						(size_t)SZ_1M)) {
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
					 * Re-derive the stripe for the next
					 * batch from its file position. The
					 * reload keeps every device-scoped
					 * hand, the dst addresses and the
					 * PFN cache, consistent with the
					 * new target.
					 */
					if (nr_chans > 1 &&
					    ret == (ssize_t)batch_bytes) {
						/* ki_pos still points at this
						 * folio's start; copied is the
						 * part of it collected so far.
						 */
						stripe = hash_64((u64)(iocb->ki_pos +
							copied) >>
							IO_DMA_STRIPE_SHIFT,
							32) % nr_chans;
						chan = ctx->dma.chans[stripe];
						dev = chan->device->dev;
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
						/* Ending the loop, not just
						 * the folio batch: another
						 * pass would refetch from a
						 * stale ki_pos and land its
						 * data past the gap.
						 */
						batch_bytes = 0;
						error = -EAGAIN;
						goto put_folios;
					}
					batch_bytes = 0;
					/* The next batch's channel was orphaned:
					 * end the claim at what was submitted
					 * and leave the rest to the CPU tail.
					 */
					if (dma_chan_orphaned(chan)) {
						error = -EAGAIN;
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
			/*
			 * The next pass starts a new batch, which takes its
			 * stripe from its own position as a batch after the
			 * in-loop flush does. A folio batch of small folios
			 * never reaches the in-loop flush, and without this
			 * a whole read of them would stay on the stripe of
			 * its first byte. A batch that crosses into the next
			 * granule still runs on the stripe of its first
			 * byte, since ending batches at granule boundaries
			 * would cost an unaligned read one more descriptor.
			 */
			if (nr_chans > 1 && !error && dst_offset < want &&
			    iocb->ki_pos < isize) {
				stripe = hash_64((u64)iocb->ki_pos >>
						 IO_DMA_STRIPE_SHIFT,
						 32) % nr_chans;
				chan = ctx->dma.chans[stripe];
				dev = chan->device->dev;
				/* See the orphan check of the in-loop flush. */
				if (dma_chan_orphaned(chan))
					error = -EAGAIN;
			}
		}
put_folios:
		for (i = 0; i < folio_batch_count(&fbatch); i++)
			folio_put(fbatch.folios[i]);
		folio_batch_init(&fbatch);
	} while (dst_offset < want && iocb->ki_pos < isize && !error);

	file_accessed(filp);

	if (req->dma.dma_refcnt > 0) {
		unsigned int c;

		/* Descriptors reach hardware at submit time; this is the
		 * belt over every channel a stripe may have touched.
		 */
		for (c = 0; c < nr_chans; c++)
			dma_async_issue_pending(ctx->dma.nr_chans ?
						ctx->dma.chans[c] :
						ctx->dma.chan);
	}
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
 * DMA-offloaded buffered write for registered buffers (WRITE_FIXED).
 * This replaces generic_perform_write()'s copy_folio_from_iter() with
 * DSA copies from the pre-mapped registered buffer into the page-cache
 * folios obtained from aops->write_begin().
 *
 * The write path is deliberately synchronous. The caller path runs in
 * io-wq process context because buffered regular-file writes always
 * punt there, and inode_lock plus the per-folio write_begin and
 * write_end protocol make a deferred-CQE design a lifetime minefield.
 * Waiting inline keeps every VFS invariant identical to the generic
 * path. The win is the removed CPU memcpy and not latency.
 *
 * The failure ladder is as follows. On a prep or submit failure or a
 * DMA_ERROR completion, with every submitted cookie reaped, we CPU
 * re-copy every chunk and commit normally (counted as "cpu_redo").
 * This is safe because copying the same src to the same dst is
 * idempotent. On a completion timeout the device is wedged. We skip
 * write_end for the folios the device may write and deliberately leak
 * them referenced, together with their dst mappings, so that a late
 * DMA write can only land in memory the kernel still owns. Unmapping
 * would free the IOVA for the next mapping to reuse. Where the
 * filesystem can take a folio out of its page cache, as memory failure
 * does, the leaked folios are taken out and unlocked, so no task waits
 * on them and later I/O of the range goes to the disk instead of the
 * torn copy. tmpfs keeps them locked. The write then returns -EIO, or
 * a short write when groups before the wedged one committed.
 */

struct io_dma_fmw_folio {
	struct folio *folio;
	void *fsdata;
	loff_t pos;
	unsigned int len;
	bool redo;		/* some of it was not handed to the device */
	bool targeted;		/* a descriptor for it was submitted */
	bool lost;		/* the disk may lack what it held */
	dma_addr_t dst_dma;	/* 0 means nothing to unmap */
	unsigned int map_len;
};

/*
 * Poll every outstanding cookie to completion.  Sets *redo on any
 * DMA_ERROR.  Returns 0, or -ETIMEDOUT with cookies possibly still in
 * flight, in which case the caller must treat every dst as poisoned.
 *
 * We spin only briefly.  DSA transfers normally complete in
 * single-digit microseconds, but under contention with many rings
 * sharing the WQ descriptor pools completion can take milliseconds.
 * Hundreds of io-wq workers busy-polling here saturated whole sockets
 * and starved application heartbeats at O(100) shared rings.
 * Therefore we back off to sleeping once the fast path misses.
 */
static int io_dma_fmw_wait(struct dma_chan *chan, dma_cookie_t *cookies,
			   unsigned int *nr, bool *redo)
{
	unsigned int wait_ms = READ_ONCE(io_dma_fmw_wait_ms);
	unsigned long deadline = jiffies + msecs_to_jiffies(wait_ms);
	u64 spin_end = ktime_get_ns() +
		READ_ONCE(io_dma_fmw_spin_us) * NSEC_PER_USEC;
	unsigned int i, spins;

	for (i = 0; i < *nr; i++) {
		enum dma_status st;

		spins = 0;
		while ((st = dmaengine_async_is_tx_complete(chan, cookies[i]))
		       == DMA_IN_PROGRESS) {
			if (time_after(jiffies, deadline)) {
				*nr = 0;
				return -ETIMEDOUT;
			}
			if (ktime_get_ns() < spin_end) {
				cpu_relax();
				if (!(++spins & 63))
					cond_resched();
			} else {
				usleep_range(50, 150);
			}
		}
		if (st != DMA_COMPLETE)
			*redo = true;
	}
	*nr = 0;
	return 0;
}

/* Writes at or below this size fall back to the CPU copy path */
#define IO_DMA_MIN_WRITE_BYTES	SZ_64K

/*
 * Folios collected, submitted and committed as one group. Capping the
 * group bounds the folio array, the number of folios held locked at
 * once, and the range a wedged device can strand. None of those would
 * otherwise be bounded by anything smaller than the request itself.
 * On a filesystem that truncates failed writes a chunk that reaches past
 * EOF is a group of its own, see the loop in io_dma_fmw_group().
 * Folios can reach the mapping's maximum folio size, so a group also
 * ends with the folio that brings it to IO_DMA_FMW_GROUP_BYTES or beyond.
 * Without that, 256 folios of 2MB would hold 512MB locked and not yet
 * dirty, which reclaim cannot free and the dirty throttle does not see.
 * Each group end waits for the copies still in flight, so every group
 * adds the latency of its last copies to the write.
 */
#define IO_DMA_FMW_GROUP_FOLIOS	256
#define IO_DMA_FMW_GROUP_BYTES	SZ_32M

/*
 * Copy one chunk with the CPU and commit it, as generic_perform_write()
 * does, while no other folio of the file is held. That is a chunk whose
 * write_begin() opened a journal handle, which the write_end() closes,
 * or a small chunk of a group capped at one folio. The source is a
 * pinned registered-buffer bvec, so the copy cannot fault and needs no
 * atomic variant. A committed chunk counts as eagain. A negative error
 * ends the write, which counts as error when nothing was written and is
 * a short write otherwise. Returns the bytes committed, 0 when
 * write_end() took nothing, or a negative error.
 */
static ssize_t io_dma_fmw_cpu_chunk(struct kiocb *iocb,
				    struct address_space *mapping,
				    struct iov_iter *from, loff_t pos,
				    size_t bytes, struct folio *folio,
				    void *fsdata)
{
	size_t copied;
	long status;

	if (mapping_writably_mapped(mapping))
		flush_dcache_folio(folio);
	copied = copy_folio_from_iter(folio, offset_in_folio(folio, pos),
				      bytes, from);
	flush_dcache_folio(folio);
	status = mapping->a_ops->write_end(iocb, mapping, pos, bytes, copied,
					   folio, fsdata);
	if (unlikely(status != copied))
		iov_iter_revert(from, copied - max(status, 0L));
	if (status > 0) {
		balance_dirty_pages_ratelimited(mapping);
		io_dma_fmw_record(IO_DMA_FMW_EAGAIN);
	}
	cond_resched();
	return status;
}

/*
 * Take a folio that a wedged group leaks out of the page cache, as memory
 * failure takes out a poisoned one, then unlock it. The caller keeps the
 * folio's reference and its destination mapping, so a late write still
 * lands in memory the kernel owns, and no lookup can find the folio any
 * more. A task asleep on its lock wakes and finds it truncated:
 * writeback skips it, and readers, faults and page_mkwrite look the range
 * up again and read it from the disk. What it held that the disk lacks
 * is lost, which @lost says, and fsync() reports -EIO for that. The loss
 * is recorded before the folio leaves the cache, as memory failure does,
 * so that an fsync() that no longer finds the folio still reports it.
 * The truncation needs the I/O of an earlier writeback ended. Returns
 * false, with the folio still locked, where the filesystem keeps such a
 * folio in its cache, as tmpfs does.
 */
static bool io_dma_fmw_drop(struct address_space *mapping,
			    struct folio *folio, bool lost)
{
	const struct address_space_operations *aops = mapping->a_ops;

	if (!aops->error_remove_folio)
		return false;
	folio_wait_writeback(folio);
	if (lost)
		mapping_set_error(mapping, -EIO);
	if (aops->error_remove_folio(mapping, folio) || folio->mapping)
		return false;
	folio_unlock(folio);
	return true;
}

/*
 * Run one group of a DMA write: collect folios starting @base bytes
 * into the write until the group is full or the write ends, submit the
 * copies, wait for them, and commit. Returns the bytes committed, or a
 * negative error when nothing was committed. On a device timeout the
 * folios the device may still write are deliberately leaked, the others
 * released, and -EIO is returned, so the caller must stop.
 */
static ssize_t io_dma_fmw_group(struct io_kiocb *req, struct kiocb *iocb,
				struct iov_iter *from, u64 src_user_addr,
				size_t base, size_t room,
				struct io_dma_fmw_folio *fol,
				unsigned int *fol_cap,
				dma_cookie_t *cookies,
				unsigned int max_cookies)
{
	struct io_ring_ctx *ctx = req->ctx;
	struct file *file = iocb->ki_filp;
	struct address_space *mapping = file->f_mapping;
	const struct address_space_operations *aops = mapping->a_ops;
	struct io_mapped_ubuf *imu = req->buf_node->buf;
	struct dma_chan *chan = ctx->dma.chan;
	struct device *dev = chan->device->dev;
	size_t max_chunk = mapping_max_folio_size(mapping);
	/*
	 * A group collects at most max_fol folios. The loop lowers the cap
	 * to 1 for the rest of the write once a write_begin() opens a
	 * journal handle, or fails a later chunk with -EAGAIN.
	 */
	unsigned int max_fol = *fol_cap;
	/*
	 * Only this write's write_end() moves i_size under inode_lock. The
	 * ones that run before the group commits lie inside EOF, so the
	 * value holds while the group collects.
	 */
	loff_t isize = i_size_read(mapping->host);
	/*
	 * A folio that was already dirty ends the group only on a mapping
	 * that writes back, for the reason the loop gives. That depends on
	 * the mapping alone, so it is read once.
	 */
	bool dirty_ends = mapping_can_writeback(mapping);
	/*
	 * ext4 answers a failed write past EOF by truncating the page cache
	 * from EOF, which locks every folio there and so would wait forever
	 * on one this group holds. On such a filesystem a chunk that reaches
	 * past EOF is a group of its own, so that no folio of the file is
	 * held while its write_begin() runs or while a journal handle it
	 * opened is closed. tmpfs never truncates, and keeps batching its
	 * appends.
	 */
	bool eof_alone = file->f_op->fop_flags & FOP_TRUNCATE_FAILED_WRITE;
	unsigned int nr_fol = 0, nr_cookies = 0, i;
	unsigned int prep_fails = 0;
	/* redo: a completion could not be trusted, so every folio is
	 * re-copied. any_redo: at least one folio was not fully handed to
	 * the device and is re-copied on its own.
	 */
	bool redo = false, any_redo = false;
	bool surrendered = false;
	ssize_t collected = 0, committed = 0;
	ssize_t err = 0;
	int wedged = 0;

	while (collected < room && nr_fol < max_fol &&
	       collected < IO_DMA_FMW_GROUP_BYTES) {
		loff_t pos = iocb->ki_pos + base + collected;
		size_t bytes = min_t(size_t,
				     max_chunk - (pos & (max_chunk - 1)),
				     room - collected);
		size_t offset, sub;
		dma_addr_t dst_dma;
		struct folio *folio;
		void *fsdata;
		bool chunk_redo = false, targeted = false, was_dirty, lost;
		int status;

		/*
		 * The length is the hint and so at least the chunk, which
		 * keeps the test conservative.
		 */
		if (eof_alone && nr_fol && pos + bytes > isize)
			break;
		/*
		 * Once the group holds a folio, a write_begin() that starts
		 * a journal handle, pushes writeback or forces a commit
		 * would wait with that folio locked, against ext4's order
		 * of transaction start before folio lock. A task that
		 * sleeps on a held folio while it holds up a commit would
		 * then deadlock with the write. Writeback does that when
		 * another writeback cleaned a folio it looked up dirty and
		 * this group took it clean, and so does a page fault in
		 * ext4's non-delalloc branch. So a later chunk's
		 * write_begin() runs under IOCB_FOLIOS_LOCKED, under which
		 * ext4 skips its writeback push and fails the call where
		 * it would start a handle or retry an allocation. The first
		 * chunk of a group runs with nothing of the file held and
		 * may do all three, as under generic_perform_write().
		 */
		if (nr_fol)
			iocb->ki_flags |= IOCB_FOLIOS_LOCKED;
		status = aops->write_begin(iocb, mapping, pos, bytes,
					   &folio, &fsdata);
		iocb->ki_flags &= ~IOCB_FOLIOS_LOCKED;
		if (unlikely(status < 0)) {
			/*
			 * A failure on a later chunk ends the group, which
			 * commits the folios collected before it, and the
			 * next group retries the chunk with nothing held,
			 * where ext4 falls back or retries the allocation.
			 * A failure there ends the write short, as on the
			 * stock path. Until then the group's earlier folios
			 * stay locked, so reclaim has less to work with. On
			 * a filesystem that truncates failed writes those
			 * folios all lie inside EOF, so the truncation from
			 * EOF never waits on one. ext4 returns -EAGAIN under
			 * the flag only where it would have opened a handle,
			 * so that lowers the cap as a handle does below.
			 */
			if (status == -EAGAIN && nr_fol)
				*fol_cap = 1;
			if (!collected)
				err = status;
			break;
		}
		/*
		 * Cover the locked folio to its end or to the write's end.
		 * That keeps the chunk inside the range write_begin() was
		 * asked to prepare, which runs to the write's end or to the
		 * next boundary of the mapping's maximum folio size,
		 * whichever comes first. No folio crosses that boundary.
		 * The chunk must not reach past the range. ext4 maps or
		 * reserves blocks only inside it, and its writeback treats
		 * a dirty block that was neither mapped nor reserved as a
		 * hole and never writes it. tmpfs checks F_SEAL_GROW
		 * against the same range. Short of the write's end the
		 * chunk must not stop inside the folio either, or the next
		 * iteration's write_begin() would wait forever on the
		 * folio lock this group already holds. Descriptors below
		 * still split at source-folio boundaries.
		 */
		offset = offset_in_folio(folio, pos);
		bytes = min_t(size_t, room - collected,
			      folio_size(folio) - offset);
		/*
		 * On a mapping that writes back, a folio that was already
		 * dirty is tagged for writeback, and the flusher locks
		 * tagged folios. Another write_begin() while the group
		 * holds it locked can wait on that flusher (ext4 under low
		 * free space, or a journal commit), which is a deadlock, so
		 * such a folio ends its group below. shmem keeps its folios
		 * dirty from their first write, fallocate() or swap-in until
		 * swap-out without ever tagging them, and nothing writes
		 * them back but reclaim, which skips locked folios. Testing
		 * the flag alone would end a group at every folio a tmpfs
		 * overwrite reaches.
		 */
		was_dirty = folio_test_dirty(folio) && dirty_ends;
		/*
		 * Whether the disk may lack what the folio holds, which a
		 * wedge that drops the folio would lose. Writeback cannot
		 * start while the group holds the lock, but one already in
		 * flight can end during the wait with torn bytes on the
		 * disk, so the state is taken now and not at the wedge.
		 * Only a mapping that writes back has a disk to lack it.
		 */
		lost = dirty_ends &&
		       (folio_test_dirty(folio) || folio_test_writeback(folio));
		/*
		 * A write_begin that leaves a journal handle open (ext4
		 * delalloc falling back under low free space) expects its
		 * matching write_end before the next write_begin. Batching
		 * would nest handles with h_ref only and no credits, and a
		 * wedge would leak the references and stall the journal.
		 * So the handle is closed at once. ext4 declines a write up
		 * front when its write_begin would open a handle, so in
		 * practice this catches a fallback that starts during the
		 * write. The check runs on every chunk because the fallback
		 * is per write_begin, and this write's own delayed
		 * allocations, or a concurrent writer, can bring free space
		 * below dirty space plus the watermark mid-write.
		 *
		 * The first chunk of a group is copied with the CPU and
		 * committed, as the stock path does, with no other folio of
		 * the file held. ext4 fails the write_begin() of a later
		 * chunk rather than open a handle there, see
		 * IOCB_FOLIOS_LOCKED above, so a handle on a later chunk
		 * comes only from a filesystem that ignores the flag. Such
		 * a chunk ends its group. A write_end that copies nothing
		 * closes its handle, with the chunk, the range the
		 * filesystem prepared, as its length. The group commits the
		 * folios collected before it, and the next group starts
		 * with this chunk. On a filesystem that truncates failed
		 * writes a chunk that reaches past EOF never joins a group
		 * that holds folios, so this one lies inside EOF and that
		 * write_end neither moves i_size nor truncates. Past EOF it
		 * would truncate the file, fallocate preallocation
		 * included.
		 *
		 * Either way the write goes on, so a fallback never ends it
		 * short, but it runs one folio per group from here on. A
		 * write that has met the fallback runs near the watermark,
		 * where ext4 fails the write_begin() of a later chunk of a
		 * batched group whenever the fallback holds, which would
		 * end that group early.
		 */
		if (unlikely(current->journal_info)) {
			*fol_cap = 1;
			if (!nr_fol)
				return io_dma_fmw_cpu_chunk(iocb, mapping, from,
							    pos, bytes, folio,
							    fsdata);
			aops->write_end(iocb, mapping, pos, bytes, 0,
					folio, fsdata);
			break;
		}
		/*
		 * Only the caps above make max_fol 1, since
		 * io_dma_filemap_write() starts it at two folios or more. A
		 * group of one folio pays a device round trip for that
		 * folio alone, which at or below IO_DMA_MIN_WRITE_BYTES
		 * costs more than the copy it replaces, so such a chunk is
		 * copied with the CPU, as a write of that size is.
		 */
		if (unlikely(max_fol == 1) && bytes <= IO_DMA_MIN_WRITE_BYTES)
			return io_dma_fmw_cpu_chunk(iocb, mapping, from, pos,
						    bytes, folio, fsdata);

		if (surrendered) {
			/*
			 * This is sustained descriptor exhaustion. We stop
			 * touching the DMA engine for the rest of this
			 * group, and the CPU-redo pass copies its remaining
			 * chunks. This beats paying a drain-wait per chunk
			 * while dozens of other rings hold the pools empty.
			 */
			dst_dma = 0;
			chunk_redo = true;
			goto record;
		}
		dst_dma = dma_map_page(dev, folio_page(folio, 0),
				       offset, bytes, DMA_FROM_DEVICE);
		if (dma_mapping_error(dev, dst_dma)) {
			/* Commit this chunk via the CPU-redo pass instead. */
			dst_dma = 0;
			chunk_redo = true;
			goto record;
		}

		/* Split at the source registered-buffer folio boundaries. */
		for (sub = 0; sub < bytes; ) {
			u64 uaddr = src_user_addr + base + collected + sub;
			size_t src_seg_remain, len;
			struct dma_async_tx_descriptor *tx;
			dma_addr_t src_dma, dst;
			dma_cookie_t ck;

			src_dma = io_reg_buf_dma_addr(imu, uaddr,
						      &src_seg_remain,
						      dev);
			dst = dst_dma + sub;
			if (unlikely(!src_dma)) {
				chunk_redo = true; /* The CPU re-copies this folio. */
				break;
			}
			len = min3(bytes - sub, src_seg_remain, max_chunk);

			tx = dmaengine_prep_dma_memcpy(chan, dst, src_dma, len,
						       io_dma_prep_flags());
			if (!tx) {
				/* The pool is exhausted. Drain in-flight
				 * work and retry once.
				 */
				dma_async_issue_pending(chan);
				wedged = io_dma_fmw_wait(chan, cookies,
							 &nr_cookies, &redo);
				if (wedged)
					goto record;
				tx = dmaengine_prep_dma_memcpy(chan, dst,
						src_dma, len,
						io_dma_prep_flags());
			}
			if (!tx) {
				io_dma_fmw_record(IO_DMA_FMW_EAGAIN);
				chunk_redo = true;
				/*
				 * Two drain-and-retry failures in one group
				 * mean the pools are held empty by other
				 * rings, so the rest of the group goes to
				 * the CPU.
				 */
				if (++prep_fails >= 2) {
					surrendered = true;
					break;	/* redo covers this chunk. */
				}
			} else {
				ck = dmaengine_submit(tx);
				if (dma_submit_error(ck)) {
					chunk_redo = true;
				} else {
					cookies[nr_cookies++] = ck;
					targeted = true;
				}
				if (nr_cookies == max_cookies) {
					dma_async_issue_pending(chan);
					wedged = io_dma_fmw_wait(chan, cookies,
							&nr_cookies, &redo);
					if (wedged)
						goto record;
				}
			}
			sub += len;
		}
record:
		fol[nr_fol++] = (struct io_dma_fmw_folio){
			.folio = folio, .fsdata = fsdata, .pos = pos,
			.len = bytes, .dst_dma = dst_dma, .map_len = bytes,
			.redo = chunk_redo, .targeted = targeted, .lost = lost,
		};
		/*
		 * A wedge found while this folio's copies were submitted ends
		 * the group here, with the folio recorded, so the wedge path
		 * below sees every folio the group holds.
		 */
		if (unlikely(wedged))
			break;
		any_redo |= chunk_redo;
		collected += bytes;
		if (was_dirty || (eof_alone && pos + bytes > isize))
			break;
	}

	if (!wedged) {
		dma_async_issue_pending(chan);
		wedged = io_dma_fmw_wait(chan, cookies, &nr_cookies, &redo);
	}

	/* Unmap the dst IOVAs, except after a timeout. An unmap frees the
	 * IOVA, at once under a strict IOMMU and at the next flush-queue
	 * drain otherwise, and the next mapping for the device can reuse
	 * it. A late write from an abandoned descriptor would then go
	 * through that valid translation into an unrelated page. So a
	 * wedge leaks the mappings with the folios below, which stay
	 * referenced, and a late write can only land there.
	 */
	for (i = 0; i < nr_fol; i++)
		if (fol[i].dst_dma && likely(!wedged))
			dma_unmap_page(dev, fol[i].dst_dma, fol[i].map_len,
				       DMA_FROM_DEVICE);

	if (unlikely(wedged)) {
		unsigned int leaked = 0, dropped = 0;

		/*
		 * No descriptor of this write was submitted for such a folio,
		 * so the device cannot write it. It is released as after a
		 * failed copy, with nothing claimed, and keeps whatever it
		 * held, but only where a zero claim cannot move i_size. On a
		 * filesystem that truncates failed writes every such folio
		 * lies inside EOF, since a chunk that reaches past EOF is a
		 * group of its own and a group with nothing submitted cannot
		 * wedge, so its write_end() neither starts a handle nor waits.
		 * tmpfs batches appends past EOF, and its write_end() moves
		 * i_size to the position even with nothing copied, which would
		 * put the leaked folios below it inside EOF. A folio that
		 * starts past EOF is leaked with them instead.
		 */
		for (i = 0; i < nr_fol; i++)
			if (!fol[i].targeted && fol[i].pos <= isize)
				aops->write_end(iocb, mapping, fol[i].pos,
						fol[i].len, 0, fol[i].folio,
						fol[i].fsdata);
		/*
		 * The others keep their reference and their mapping for good,
		 * so that a late write can only land in memory the kernel
		 * still owns. Their contents are unknown. A task can sleep on
		 * the lock of any of them: writeback that looked it up, a
		 * read of a range not uptodate, a fault, compaction, and
		 * every later truncate or write of the file. And readers of
		 * an uptodate one would copy the torn bytes. Where the
		 * filesystem takes such a folio out of its page cache, see
		 * io_dma_fmw_drop(), it is unlocked as well, and every task
		 * asleep on it wakes and finds it gone. A reader or a pin that
		 * took the folio before can still see the torn bytes, as after
		 * a truncate. The highest goes first. ext4 writeback can hold
		 * the bio of a lower folio unsubmitted while it sleeps on a
		 * higher one, so waiting on a folio's writeback with a higher
		 * folio still held could wait forever. A mapping that writes
		 * back must drop the folio, see FOP_DMA_WRITE, or the next
		 * writeback that looks it up would sleep on it for good.
		 */
		for (i = nr_fol; i-- > 0; ) {
			if (!fol[i].targeted && fol[i].pos <= isize)
				continue;
			leaked++;
			if (io_dma_fmw_drop(mapping, fol[i].folio, fol[i].lost))
				dropped++;
			else
				WARN_ON_ONCE(dirty_ends);
		}
		pr_warn_ratelimited("io_uring DMA write: wedged after %ums, leaking %u folios, %u of them dropped from the page cache (%s)\n",
				    READ_ONCE(io_dma_fmw_wait_ms), leaked, dropped,
				    dma_chan_name(chan));
		io_dma_fmw_record(IO_DMA_FMW_ERROR);
		return -EIO;
	}

	if (unlikely(redo || any_redo)) {
		/* Re-copy with the CPU: every folio after an untrusted
		 * completion, otherwise only the folios the device was not
		 * handed in full. The same source bytes go to the same
		 * folio ranges, so overlap with completed DMA is
		 * idempotent. The iter was never advanced during
		 * collection, so it is walked over the folios in order.
		 */
		for (i = 0; i < nr_fol; i++) {
			size_t n;

			if (!redo && !fol[i].redo) {
				iov_iter_advance(from, fol[i].len);
				continue;
			}
			n = copy_folio_from_iter(fol[i].folio,
					offset_in_folio(fol[i].folio, fol[i].pos),
					fol[i].len, from);
			if (unlikely(n != fol[i].len)) {
				collected = fol[i].pos -
					(iocb->ki_pos + base) + n;
				break;
			}
		}
		io_dma_fmw_record(IO_DMA_FMW_CPU_REDO);
	}

	/* Commit in ascending order with the dirty, unlock, and i_size
	 * updates. Every collected folio must pass through write_end even
	 * after a failure, with a zero claim, or the tail folios stay
	 * locked and referenced forever.
	 */
	{
		bool commit_failed = false;

		for (i = 0; i < nr_fol; i++) {
			size_t claim = commit_failed ? 0 :
				min_t(size_t, fol[i].len,
				      collected - committed);
			int done;

			done = aops->write_end(iocb, mapping, fol[i].pos,
					       fol[i].len, claim,
					       fol[i].folio, fol[i].fsdata);
			if (commit_failed)
				continue;
			if (unlikely(done < 0)) {
				if (!committed)
					err = done;
				commit_failed = true;
				continue;
			}
			committed += done;
			if ((size_t)done < fol[i].len) {
				commit_failed = true;
				continue;
			}
		}
	}
	/*
	 * Throttle once per group, with every folio unlocked. Inside the
	 * loop the later folios of a rewrite are dirty and locked, and
	 * the flusher this waits for would block on them.
	 */
	if (committed)
		balance_dirty_pages_ratelimited(mapping);

	/* The CPU re-copy consumes the iter when it runs; otherwise
	 * advance it over exactly what was committed. The two counts
	 * cannot diverge: the source is a pinned registered-buffer
	 * bvec, so copy_folio_from_iter() never copies short, every
	 * claim therefore equals its chunk length, and write_end
	 * with a full claim returns it whole or fails outright.
	 */
	if (!redo && !any_redo)
		iov_iter_advance(from, committed);
	return committed ? committed : err;
}

ssize_t io_dma_filemap_write(struct io_kiocb *req, struct kiocb *iocb,
			     struct iov_iter *from, u64 src_user_addr)
{
	struct io_mapped_ubuf *imu = req->buf_node->buf;
	struct file *file = iocb->ki_filp;
	struct inode *inode = file->f_mapping->host;
	struct io_dma_fmw_folio *fol = NULL;
	dma_cookie_t *cookies = NULL;
	unsigned int max_fol, max_cookies;
	ssize_t want, written = 0, err = 0;

	/* At or below this size the per-folio write_begin and write_end
	 * machinery plus the descriptor overhead exceeds the memcpy it
	 * replaces, so we take the CPU path.
	 */
	if (iov_iter_count(from) <= IO_DMA_MIN_WRITE_BYTES)
		return -EAGAIN;

	inode_lock(inode);
	/*
	 * The prologue of the filesystem's write_iter. A filesystem whose
	 * checks go beyond the generic ones, such as ext4 with its immutable
	 * test, its bitmap-format size limit and the zeroing of the old last
	 * block on a write past EOF, runs them through ->dma_copy_checks();
	 * the others get the generic sequence.
	 */
	if (file->f_op->dma_copy_checks) {
		want = file->f_op->dma_copy_checks(iocb, from, WRITE);
	} else {
		want = generic_write_checks(iocb, from);
		if (want > 0) {
			err = file_modified(file);
			if (err)
				want = err;
		}
	}
	if (want <= 0) {
		inode_unlock(inode);
		/*
		 * The write path runs without IOCB_NOWAIT, so only a
		 * filesystem declining the offload returns -EAGAIN here.
		 */
		if (want == -EAGAIN)
			io_dma_fmw_record(IO_DMA_FMW_FS_DECLINE);
		return want;
	}

	max_fol = min_t(unsigned long, IO_DMA_FMW_GROUP_FOLIOS,
			DIV_ROUND_UP(want, PAGE_SIZE) + 1);
	/* One entry per src-folio crossing per chunk. The collection loop
	 * flushes whenever the array fills, so this sets the wait
	 * granularity rather than bounding the transfer.
	 */
	max_cookies = max_fol + DIV_ROUND_UP(want, 1UL << imu->folio_shift) + 8;
	max_cookies = min_t(unsigned int, max_cookies, 4 * max_fol + 8);
	fol = kvmalloc_array(max_fol, sizeof(*fol), GFP_KERNEL);
	cookies = kvmalloc_array(max_cookies, sizeof(*cookies), GFP_KERNEL);
	if (!fol || !cookies) {
		err = -EAGAIN;	/* Fall back to the normal write path. */
		goto out_unlock;
	}

	/* One group at a time, so a large write neither allocates an array
	 * proportional to its length nor holds every one of its folios
	 * locked at once. Groups commit as they go, so a failure part way
	 * through is a short write over the groups that already landed.
	 */
	while (written < want) {
		ssize_t done;

		/*
		 * A fatal signal ends the write before the next group, as
		 * it ends generic_perform_write() before the next chunk. A
		 * write that starts with one pending copies nothing, and
		 * one that runs a folio per group stops after the folio in
		 * progress.
		 */
		if (fatal_signal_pending(current)) {
			if (!written)
				err = -EINTR;
			break;
		}
		done = io_dma_fmw_group(req, iocb, from, src_user_addr,
					written, want - written, fol,
					&max_fol, cookies, max_cookies);
		if (done <= 0) {
			/* A zero-progress group falls back to the normal
			 * write path rather than completing as a zero-byte
			 * write.
			 */
			if (!written)
				err = done ? done : -EAGAIN;
			break;
		}
		written += done;
	}

	if (written > 0)
		iocb->ki_pos += written;
out_unlock:
	inode_unlock(inode);
	kvfree(fol);
	kvfree(cookies);
	if (written > 0)
		return generic_write_sync(iocb, written) ?: written;
	return err;
}

/*
 * Release a completed or aborted task's source resources.  These are
 * the DMA unmaps, which involve IOVA frees and IOTLB work under an
 * IOMMU, and the folio references.  This is the expensive half of
 * completion and needs no ctx->dma.lock.  Callers run it before taking
 * the lock for __io_dma_task_complete() so that the lock hold shrinks
 * to the refcount handshake.
 */
static void io_dma_task_release_res(struct io_ring_ctx *ctx, struct device *dev,
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
	atomic_dec(&req->ctx->dma.tasks_pending);
	io_dma_task_free(req->ctx, dma);
	req->dma.dma_refcnt--;

	if (req->dma.dma_refcnt == 0) {
		pr_debug("dma req done: opcode=%d result=%d\n",
			 req->opcode, req->dma.dma_result);

		if (req->dma.dma_result < 0 ||
		    req->dma.min_fail_off != U32_MAX) {
			/*
			 * A failed task, a device that went away under the
			 * ring for instance, or a claim the submitter could
			 * not finish without blocking. The bytes below the
			 * lowest failed or unread offset were delivered,
			 * since every task lying wholly below it succeeded;
			 * the rest is read again from the page cache, which
			 * still holds it. io_rw_dma_redo() reissues the
			 * request from that offset, with the DMA gate closed
			 * after a failure.
			 */
			req->io_task_work.func = io_rw_dma_redo;
			io_req_task_work_add(req);
		} else {
			/*
			 * Terminal completion goes through task_work.
			 * __io_dma_poll() may run from a workqueue without
			 * ctx->uring_lock, so io_req_task_complete() defers
			 * the CQE post until the task_work loop acquires it.
			 *
			 * A short result marks the request failed as a short
			 * CPU read does, so that an IOSQE_IO_LINK successor
			 * does not run on a partly filled buffer and an
			 * IOSQE_CQE_SKIP_SUCCESS read still posts its CQE.
			 * The issuer writes no req->flags once the tasks are
			 * published, and the flags it still tests are ones
			 * this completion never changes.
			 */
			if (req->dma.short_res)
				req_set_fail(req);
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
		struct io_dma_task *t = req->dma.dma_tasks;

		pr_debug("submit_queued: refcnt=%d\n", req->dma.dma_refcnt);
		/*
		 * Take the chain off the req and clear its DMA state before
		 * any task is published. Once the tasks are visible a poller
		 * can complete the req, and that completion can queue a
		 * reissue that arms new DMA state on the req from another
		 * thread when the publisher is an io-wq worker. So the req
		 * is not written after the publish below.
		 */
		req->dma.dma_tasks = NULL;
		req->dma.dma_tasks_tail = NULL;
		req->dma.dma_active = false;
		if (req->dma.dma_refcnt > 0) {
			struct llist_node *first = NULL, *last = NULL;
			unsigned int n = 0;
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
			 * Link the chain newest first, as one llist_add() per
			 * task in chain order would, so the consumer's
			 * reversal gives chain order back. The next pointers
			 * are read here, before any task is published,
			 * because a published task can complete and be freed
			 * at once. Every task was fully submitted with a valid
			 * cookie, since tasks are linked only after
			 * dmaengine_submit() succeeds, so the poller can
			 * consume them as they are.
			 */
			for (; t; t = t->next) {
				t->llnode.next = first;
				first = &t->llnode;
				if (!last)
					last = first;
				n++;
			}
			/*
			 * Publish after the ref take above, so that no drain
			 * can reach dma_refcnt == 0 before the in-flight ref
			 * exists. The pending count rises before the tasks are
			 * visible, so no observer can see a task without the
			 * count. The cmpxchg in llist_add_batch() orders the
			 * two. Several producers can publish at once: the one
			 * uring_lock holder, which is the SQPOLL thread on an
			 * SQPOLL ring, and any number of io-wq workers, which
			 * publish without the lock. llist_add_batch() is safe
			 * for that, and each producer's tasks keep their order
			 * after the reversal.
			 *
			 * The poll worker is queued in the same section with
			 * preemption off. A publisher preempted between the
			 * publish and the queue_work() could leave its tasks
			 * without a poller, and their CQEs held back, until it
			 * ran again. The worker goes on system_dfl_wq, which
			 * is unbound, rather than through schedule_work(),
			 * whose system_percpu_wq is per-CPU. A per-CPU kworker
			 * must win a wakeup-preemption fight with the
			 * submitting thread to run, and this shows up as
			 * completions sitting undetected for the application's
			 * whole compute slice. An unbound worker lands on an
			 * idle CPU and detects completions in parallel.
			 */
			preempt_disable();
			atomic_add(n, &ctx->dma.tasks_pending);
			llist_add_batch(first, last, &ctx->dma.submit_list);
			if (io_dma_pending(ctx))
				queue_work(system_dfl_wq, &ctx->dma.poll_work);
			preempt_enable();
			ret = -EIOCBQUEUED;
		}
	}

	/*
	 * Nothing here touches the req after the publish. The caller may
	 * still read fields that neither a completion nor a reissue
	 * changes, such as req->file. The req stays allocated for that:
	 * an io-wq worker holds its own reference, and an inline or
	 * SQPOLL issuer runs the completion's task_work only after it
	 * returns.
	 */
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
	 * and every submitter rings for its own tasks from its issue
	 * path. Nothing on the pending lists needs a kick.
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
			ret = dmaengine_async_is_tx_complete(dma->chan,
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
			io_dma_task_release_res(ctx,
					dma->chan->device->dev, dma);
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

/*
 * Called from io_cqring_wait_schedule() right before the task would
 * block. If this ring has DMA tasks in flight, spin on their
 * completion records for at most io_dma_cq_poll_us. Completions found
 * here run __io_dma_task_complete() in this task's context. The CQE
 * itself is posted by the poll task_work that this queues, so we
 * report progress via the pending-work checks and let the wait loop
 * run it.
 *
 * @deadline is the first expiry of the caller's wait timer in the ring's
 * clock. That is the end of min_wait when one is set and the timeout
 * otherwise, or KTIME_MAX when the wait has neither. The spin starts no
 * poll pass once it has passed and stops at the first check at or after
 * it.
 *
 * Returns true if the caller should skip sleeping and re-run its wait
 * loop because task_work or CQEs are ready. Returns false to fall back
 * to schedule(). This is a no-op for rings with no DMA channel.
 */
bool io_dma_cq_wait_poll(struct io_ring_ctx *ctx, struct io_wait_queue *iowq,
			 ktime_t deadline)
{
	unsigned int budget_us = READ_ONCE(io_dma_cq_poll_us);
	u64 spin_ns, end_ns;

	if (!budget_us || IS_ERR_OR_NULL(ctx->dma.chan))
		return false;
	if (!io_dma_pending(ctx))
		return false;

	/*
	 * Only the caller's timed sleep applies the deadline, so the spin
	 * ends at the deadline and does not start once it has passed. The
	 * ring's clock is read only here and the spin is timed with
	 * ktime_get_ns(), as napi's busy poll does. IORING_REGISTER_CLOCK
	 * can switch the ring's clock meanwhile, and comparing readings of
	 * two clocks could leave the spin without a time bound.
	 */
	spin_ns = (u64)budget_us * NSEC_PER_USEC;
	if (deadline != KTIME_MAX) {
		ktime_t now = io_get_time(ctx);

		if (now >= deadline)
			return false;
		spin_ns = min_t(u64, spin_ns, ktime_sub(deadline, now));
	}
	end_ns = ktime_get_ns() + spin_ns;
	for (;;) {
		__io_dma_poll(ctx);

		if (task_work_pending(current) || io_local_work_pending(ctx) ||
		    io_should_wake(iowq))
			return true;
		if (!io_dma_pending(ctx))
			return false;
		if (need_resched() || task_sigpending(current))
			break;
		if (ktime_get_ns() >= end_ns)
			break;
		cpu_relax();
	}

	/*
	 * The caller is about to block with tasks still pending, and the
	 * poll worker is their backstop. No exit of the spin queues it.
	 * Every publish queues the worker after it raises tasks_pending,
	 * and the worker polls again or requeues itself while
	 * io_dma_pending(), which reads tasks_pending first, is true, so
	 * the worker does not exit while a task is pending. Queueing it
	 * here as well would find it running nearly every time, and
	 * queueing a running work item wakes an idle kworker only to hand
	 * the item back to the busy one. A task whose publisher has raised
	 * tasks_pending but not yet reached its queue_work() is picked up
	 * once the publisher gets there.
	 */
	return false;
}
