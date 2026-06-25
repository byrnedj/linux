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
#include <linux/timekeeping.h>
#include <linux/debugfs.h>
#include <linux/seq_file.h>
#include "io_uring.h"
#include "kbuf.h"
#include "rsrc.h"
#include "poll.h"
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
 * Per-transaction latency tracking, split by copy engine (DMA vs CPU) and
 * binned by transfer size. A "transaction" is one io_dma_task for the DMA
 * side (descriptor or batch) and one io_dma_cpu_copy() call for the CPU side.
 *
 * Bins are contiguous so no transaction is uncounted -- this fills in the
 * <4KB and 32-64KB bins that complete the requested set
 * (4-8/8-16/16-32/64-128/128-256/256-512/512-1024KB and 1024KB+).
 */
static const struct {
	size_t		max;	/* exclusive upper bound in bytes; 0 == catch-all */
	const char	*label;
} io_dma_lat_bins[] = {
	{    4096, "<4KB"        },
	{    8192, "4-8KB"      },
	{   16384, "8-16KB"     },
	{   32768, "16-32KB"    },
	{   65536, "32-64KB"    },
	{  131072, "64-128KB"   },
	{  262144, "128-256KB"  },
	{  524288, "256-512KB"  },
	{ 1048576, "512KB-1MB"  },
	{       0, ">=1MB"      },
};
#define IO_DMA_LAT_NBINS	ARRAY_SIZE(io_dma_lat_bins)

struct io_dma_lat_stats {
	atomic64_t	count[IO_DMA_LAT_NBINS];
	atomic64_t	sum_ns[IO_DMA_LAT_NBINS];
};

static struct io_dma_lat_stats io_dma_lat_dma;	/* DSA transactions (per task) */
static struct io_dma_lat_stats io_dma_lat_cpu;	/* CPU-copy transactions */
static struct io_dma_lat_stats io_dma_lat_call;	/* DSA, per copy_to_iter call */

/*
 * Diagnostic: how many DMA descriptors (chunks) each io_uring_copy_to_iter()
 * call emitted. Index 1..7 exact, index 8 = "8 or more"; index 0 unused. If
 * this is overwhelmingly 1, each recv is a single source segment and there is
 * nothing for batching to aggregate.
 */
#define IO_DMA_CHUNKS_HIST	9
static atomic64_t io_dma_chunks_hist[IO_DMA_CHUNKS_HIST];

static void io_dma_chunks_record(unsigned int n)
{
	if (n >= IO_DMA_CHUNKS_HIST)
		n = IO_DMA_CHUNKS_HIST - 1;
	atomic64_inc(&io_dma_chunks_hist[n]);
}

static unsigned int io_dma_lat_bin(size_t len)
{
	unsigned int i;

	for (i = 0; i < IO_DMA_LAT_NBINS - 1; i++)
		if (len < io_dma_lat_bins[i].max)
			return i;
	return IO_DMA_LAT_NBINS - 1;	/* catch-all */
}

static void io_dma_lat_record(struct io_dma_lat_stats *s, size_t len, u64 ns)
{
	unsigned int b = io_dma_lat_bin(len);

	atomic64_inc(&s->count[b]);
	atomic64_add(ns, &s->sum_ns[b]);
}

static void io_dma_lat_show_one(struct seq_file *m, const char *name,
				struct io_dma_lat_stats *s)
{
	unsigned int i;

	seq_printf(m, "%s:\n", name);
	seq_printf(m, "  %-12s %12s %16s\n", "bin", "count", "avg_ns");
	for (i = 0; i < IO_DMA_LAT_NBINS; i++) {
		u64 count = atomic64_read(&s->count[i]);
		u64 sum = atomic64_read(&s->sum_ns[i]);
		u64 avg = count ? div64_u64(sum, count) : 0;

		seq_printf(m, "  %-12s %12llu %16llu\n",
			   io_dma_lat_bins[i].label, count, avg);
	}
}

static int io_dma_lat_show(struct seq_file *m, void *v)
{
	unsigned int i;

	io_dma_lat_show_one(m, "dma", &io_dma_lat_dma);
	io_dma_lat_show_one(m, "cpu", &io_dma_lat_cpu);
	io_dma_lat_show_one(m, "dma_call", &io_dma_lat_call);

	seq_puts(m, "dma_chunks_per_call:\n");
	for (i = 1; i < IO_DMA_CHUNKS_HIST; i++)
		seq_printf(m, "  %u%-10s %12llu\n", i,
			   i == IO_DMA_CHUNKS_HIST - 1 ? "+" : "",
			   atomic64_read(&io_dma_chunks_hist[i]));
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(io_dma_lat);

static void io_dma_lat_reset(struct io_dma_lat_stats *s)
{
	unsigned int i;

	for (i = 0; i < IO_DMA_LAT_NBINS; i++) {
		atomic64_set(&s->count[i], 0);
		atomic64_set(&s->sum_ns[i], 0);
	}
}

static ssize_t io_dma_lat_reset_write(struct file *file,
				      const char __user *ubuf,
				      size_t count, loff_t *ppos)
{
	unsigned int i;

	io_dma_lat_reset(&io_dma_lat_dma);
	io_dma_lat_reset(&io_dma_lat_cpu);
	io_dma_lat_reset(&io_dma_lat_call);
	for (i = 0; i < IO_DMA_CHUNKS_HIST; i++)
		atomic64_set(&io_dma_chunks_hist[i], 0);
	return count;
}

static const struct file_operations io_dma_lat_reset_fops = {
	.owner		= THIS_MODULE,
	.open		= simple_open,
	.write		= io_dma_lat_reset_write,
	.llseek		= noop_llseek,
};

void io_dma_debugfs_init(void)
{
	debugfs_create_file("io_uring_dma_latency", 0444, NULL, NULL,
			    &io_dma_lat_fops);
	debugfs_create_file("io_uring_dma_latency_reset", 0200, NULL, NULL,
			    &io_dma_lat_reset_fops);
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
	} while (READ_ONCE(ctx->dma.head));
}

static int __io_dma_task_submit(struct dma_chan *chan, struct io_dma_task *dma)
{
	struct dma_async_tx_descriptor *tx;

	tx = dmaengine_prep_dma_memcpy(chan, dma->dst_dma, dma->src_dma,
				       dma->len, io_dma_prep_flags());
	if (!tx) {
		pr_err("dma prep failed: len=%u src=0x%llx dst=0x%llx\n",
		       dma->len, (u64)dma->src_dma, (u64)dma->dst_dma);
		return -EAGAIN;
	}

	dma->submit_ns = ktime_get_ns();
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

	/*
	 * Mark this request as non-lazy for poll task_work. When a DMA
	 * completion from an IRQ/workqueue kicks the poll, the task_work it
	 * adds MUST force-wake the owning user task — the aux CQE that would
	 * otherwise wake the task is produced by io_recv's prelude AFTER that
	 * task_work runs (classic deferred-wake deadlock if lazy).
	 *
	 * It's not sufficient to set this flag inside io_poll_kick: an earlier
	 * task_work may already be queued (from the initial poll wake / inline
	 * issue) with LAZY_WAKE baked into its nr_tw. Setting the flag here,
	 * before any task_work is queued for the multishot-DMA recv, ensures
	 * every subsequent __io_poll_execute for this req uses non-lazy wake.
	 */
	req->flags |= REQ_F_POLL_NO_LAZY;

	req->dma.dma_active = true;
	req->dma.dma_refcnt = 0;
	req->dma.dma_ref_held = false;
	req->dma.dma_terminal = false;
	req->dma.dma_result = 0;
	req->dma.dma_tasks = NULL;
	req->dma.dma_tasks_tail = NULL;
	req->dma.dst_user_addr = 0;
	req->dma.saved_res = 0;
	req->dma.saved_cflags = 0;
	req->dma.cb_fn = NULL;
	req->dma.cb_arg = NULL;
	/*
	 * mshot_in_flight and pending_aux_cqe are multishot-DMA lifecycle
	 * flags. io_recv clears them at its prelude / after dispatch, so
	 * by the time dma_prep is called again for the same req they must
	 * already be false. Reset defensively in case a fresh io_kiocb
	 * comes from the slab cache with stale bytes.
	 */
	req->dma.mshot_in_flight = false;
	req->dma.pending_aux_cqe = false;
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
 *
 * Each copy_to_iter() is capped to stay within a single page so that
 * __check_object_size() / check_heap_object() in usercopy hardening
 * does not abort: for a compound source page, the check rejects any
 * copy whose length exceeds page_size(compound_head) - offset. Skb
 * linear segments (e.g., GRO-aggregated frames) can legitimately be
 * larger than a single compound's page_size, so we slice at page
 * boundaries rather than hand the full kvec segment to copy_to_iter.
 */
static ssize_t io_dma_cpu_copy(struct iov_iter *dst_iter,
			       struct iov_iter *src_iter, size_t len)
{
	size_t left = len;
	ssize_t copied_total = 0;
	u64 t0 = ktime_get_ns();

	while (left > 0) {
		size_t seg_avail = min_t(size_t, left,
			iter_iov_len(src_iter));
		const void *base;
		size_t seg_off = 0;

		if (!seg_avail)
			break;
		base = iter_iov_addr(src_iter);

		while (seg_off < seg_avail) {
			size_t page_off = offset_in_page(base + seg_off);
			size_t chunk = min_t(size_t, seg_avail - seg_off,
					     PAGE_SIZE - page_off);
			size_t copied = copy_to_iter(base + seg_off, chunk,
						     dst_iter);

			if (!copied) {
				if (seg_off)
					iov_iter_advance(src_iter, seg_off);
				return copied_total ? copied_total : -EFAULT;
			}
			seg_off += copied;
			copied_total += copied;
			left -= copied;
			if (copied < chunk)
				break;
		}
		iov_iter_advance(src_iter, seg_off);
		if (seg_off < seg_avail)
			break;
	}
	if (copied_total > 0)
		io_dma_lat_record(&io_dma_lat_cpu, copied_total,
				  ktime_get_ns() - t0);
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
 * Append a submitted DMA task to the req's pending list (O(1) via tail
 * pointer) and account it. Caller has filled in the task and obtained a
 * valid cookie.
 */
static void io_dma_task_link(struct io_kiocb *req, struct io_dma_task *dma)
{
	dma->next = NULL;
	req->dma.dma_refcnt++;
	if (!req->dma.dma_tasks) {
		req->dma.dma_tasks = dma;
		req->dma.dma_tasks_tail = dma;
	} else {
		req->dma.dma_tasks_tail->next = dma;
		req->dma.dma_tasks_tail = dma;
	}
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

	/*
	 * Request cache-control as usual: idxd applies it to the MEMMOVE
	 * sub-descriptors (the data movement, so the destination is left
	 * cache-warm for the app), and strips it from the DSA_OPCODE_BATCH
	 * dispatch descriptor where it is invalid (see idxd_dma_prep_memcpy_sg).
	 */
	tx = dmaengine_prep_dma_memcpy_sg(chan, dst_sgl, nr_entries,
					   src_sgl, nr_entries, io_dma_prep_flags());
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

	dma->submit_ns = ktime_get_ns();
	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		kfree(heap_entries);
		io_dma_task_free(req->ctx, dma);
		return -EFAULT;
	}

	/* Take folio refs for DMA duration (page-cache sources only; skb kvec
	 * sources are kept alive by the deferred cb_fn at dma_refcnt == 0).
	 */
	for (i = 0; i < nr_entries; i++)
		if (entries[i].src_is_page)
			folio_get(entries[i].folio);

	io_dma_task_link(req, dma);

	return total_len;
}

static ssize_t io_dma_submit_single_entry(struct io_kiocb *req,
					  struct dma_chan *chan,
					  struct io_dma_batch_entry *entry);

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
	struct io_dma_batch_entry *entries = NULL;
	unsigned int nr_entries = 0;
	unsigned int call_chunks = 0;
	u64 call_t0;
	ssize_t batch_ret;

	cmd = container_of(kiocb, struct io_dma, kiocb);
	req = cmd_to_io_kiocb(cmd);
	ctx = req->ctx;

	len = min(iov_iter_count(src_iter), iov_iter_count(dst_iter));

	/*
	 * No DSA channel: do a pure CPU copy so a CPU-copy latency baseline
	 * is still recorded (io_dma_cpu_copy() bins it) for comparison against
	 * the DSA path. Store cb_fn so io_dma_submit_queued_tasks() can release
	 * the source skb — it must run AFTER tcp_eat_recv_skb() unlinks the skb
	 * from the receive queue (TCP ate it with free=false because
	 * msg_io_iocb was set), so releasing it synchronously here would free a
	 * still-linked skb and corrupt the queue.
	 */
	if (IS_ERR_OR_NULL(ctx->dma.chan)) {
		req->dma.cb_fn = cb_fn;
		req->dma.cb_arg = cb_arg;
		return io_dma_cpu_copy(dst_iter, src_iter, len);
	}

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
	call_t0 = ktime_get_ns();

	/*
	 * Small source segments (< threshold) are deferred to a CPU-copy pass:
	 * per-descriptor DMA overhead loses to a cache-hot memcpy, and the
	 * copies overlap with in-flight DMA. Each records the source address
	 * and destination offset; they are copied after the DMA is submitted.
	 */
#define IO_DMA_CPU_DEFER_MAX	8
	struct io_dma_cpu_seg {
		void *src;
		size_t len;
		size_t dst_off;
	};
	struct io_dma_cpu_seg cpu_segs[IO_DMA_CPU_DEFER_MAX];
	unsigned int nr_cpu_segs = 0;
	unsigned int i;

	/*
	 * Folio-bounded source chunks collected across ALL source segments of
	 * this transfer. A TCP skb's linear region and each page frag is a
	 * separate kvec segment; collecting across segments lets one DSA batch
	 * descriptor (dmaengine_prep_dma_memcpy_sg) span many frags/folios, so
	 * the whole recv is one transaction — larger transfer, one completion,
	 * one latency sample — instead of one descriptor per frag. The batch is
	 * flushed when full (IO_DMA_BATCH_MAX) and once at the end.
	 *
	 * dma_src[] mirrors the still-unflushed entries with the CPU-reachable
	 * source address + destination offset, so the error path can CPU-copy a
	 * chunk that was collected but not yet submitted to hardware (its length
	 * is entries[i].src_len).
	 */
	struct io_dma_src_ref {
		void *src;
		size_t dst_off;
	};
	struct io_dma_src_ref dma_src[IO_DMA_BATCH_MAX];

	entries = kmalloc_array(IO_DMA_BATCH_MAX, sizeof(*entries),
				GFP_NOWAIT | __GFP_NOWARN);
	if (!entries)
		goto cpu_fallback;

	/*
	 * Walk the source one chunk at a time. src_iter is advanced per chunk
	 * (and per deferred small segment), so it always points at the next
	 * byte not yet claimed for DMA — which keeps the error path's CPU copy
	 * of the remainder trivial.
	 */
	while (total_dma < len) {
		size_t seg_avail, folio_remain, chunk_len;
		void *src_kaddr;
		dma_addr_t dst_dma, src_dma;

		seg_avail = iter_iov_len(src_iter);
		if (!seg_avail)
			break;
		seg_avail = min_t(size_t, seg_avail, len - total_dma);
		src_kaddr = (void *)iter_iov_addr(src_iter);

		/* Defer a small segment (or the small tail of one) to CPU. */
		if (seg_avail < threshold &&
		    nr_cpu_segs < IO_DMA_CPU_DEFER_MAX) {
			cpu_segs[nr_cpu_segs].src = src_kaddr;
			cpu_segs[nr_cpu_segs].len = seg_avail;
			cpu_segs[nr_cpu_segs].dst_off = total_dma;
			nr_cpu_segs++;
			total_dma += seg_avail;
			iov_iter_advance(src_iter, seg_avail);
			continue;
		}

		/* Resolve the destination folio and clamp the chunk to it so no
		 * single transfer crosses an IOMMU mapping.
		 */
		dst_dma = io_dma_dst_addr(req, dst_iter, dst_base_addr,
					  total_dma, &folio_remain);
		if (!dst_dma)
			goto cpu_fallback_rest;
		chunk_len = min_t(size_t, seg_avail, folio_remain);

		src_dma = dma_map_single(dev, src_kaddr, chunk_len,
					 DMA_TO_DEVICE);
		if (dma_mapping_error(dev, src_dma))
			goto cpu_fallback_rest;

		entries[nr_entries].src_dma = src_dma;
		entries[nr_entries].dst_dma = dst_dma;
		entries[nr_entries].src_len = chunk_len;
		entries[nr_entries].folio = NULL;
		entries[nr_entries].src_is_page = false;
		dma_src[nr_entries].src = src_kaddr;
		dma_src[nr_entries].dst_off = total_dma;
		nr_entries++;
		call_chunks++;

		total_dma += chunk_len;
		iov_iter_advance(src_iter, chunk_len);

		/* Flush a full batch and keep collecting. */
		if (nr_entries == IO_DMA_BATCH_MAX) {
			batch_ret = io_dma_submit_batch(req, dev, ctx->dma.chan,
							entries, nr_entries);
			if (batch_ret < 0)
				goto cpu_fallback_rest;
			nr_entries = 0;
		}
	}

	/*
	 * Flush the final partial batch. Same single-vs-batch cutoff as
	 * io_dma_flush_batch() (1 -> memcpy, >=2 -> one batch descriptor), but
	 * open-coded because the recv path recovers unsubmitted chunks via a CPU
	 * copy on failure (cpu_fallback_rest, using dma_src[]), rather than
	 * io_dma_flush_batch()'s unmap-and-return-error contract.
	 */
	if (nr_entries) {
		if (nr_entries == 1)
			batch_ret = io_dma_submit_single_entry(req,
					ctx->dma.chan, &entries[0]);
		else
			batch_ret = io_dma_submit_batch(req, dev,
					ctx->dma.chan, entries, nr_entries);
		if (batch_ret < 0)
			goto cpu_fallback_rest;
		nr_entries = 0;
	}

	kfree(entries);
	entries = NULL;

	/* Advance destination iterator past all processed bytes. */
	iov_iter_advance(dst_iter, total_dma);

	/*
	 * Kick the engine before the CPU pass so hardware starts immediately;
	 * the CPU copies land in disjoint byte ranges and overlap the in-flight
	 * DMA writes.
	 */
	if (req->dma.dma_refcnt > 0)
		dma_async_issue_pending(ctx->dma.chan);

	for (i = 0; i < nr_cpu_segs; i++) {
		void __user *dst_ua =
			(void __user *)(dst_base_addr + cpu_segs[i].dst_off);
		unsigned long unc = copy_to_user(dst_ua, cpu_segs[i].src,
						 cpu_segs[i].len);

		if (unlikely(unc)) {
			/*
			 * Destination became unwritable. Back the undelivered
			 * bytes out of total_dma so the recv reports a short
			 * transfer rather than counting bytes never written.
			 */
			WARN_ON_ONCE(1);
			total_dma -= unc;
		}
	}

	/* Diagnostic: per-call DSA descriptor count + aggregate transfer size. */
	if (call_chunks) {
		io_dma_chunks_record(call_chunks);
		io_dma_lat_record(&io_dma_lat_call, total_dma,
				  ktime_get_ns() - call_t0);
	}

	return total_dma;

cpu_fallback_rest:
	/*
	 * A chunk failed to map, a destination folio was unmapped, or a batch
	 * failed to submit. Recover without losing data:
	 *   - unmap and CPU-copy the chunks collected but not yet submitted
	 *     (their CPU-reachable sources are in dma_src[]);
	 *   - CPU-copy the deferred small segments;
	 *   - CPU-copy the remainder [total_dma, len), whose source is exactly
	 *     where src_iter now points (advanced per claimed chunk).
	 * Already-submitted batches are left to complete in hardware; their
	 * destination ranges are disjoint from everything copied here.
	 */
	if (nr_entries) {
		io_dma_unmap_batch(req->ctx, dev, entries, nr_entries, false);
		for (i = 0; i < nr_entries; i++) {
			void __user *dst_ua = (void __user *)
				(dst_base_addr + dma_src[i].dst_off);
			unsigned long unc = copy_to_user(dst_ua, dma_src[i].src,
							 entries[i].src_len);

			if (unlikely(unc)) {
				WARN_ON_ONCE(1);
				total_dma -= unc;
			}
		}
		nr_entries = 0;
	}
	kfree(entries);
	entries = NULL;

	for (i = 0; i < nr_cpu_segs; i++) {
		void __user *dst_ua =
			(void __user *)(dst_base_addr + cpu_segs[i].dst_off);
		unsigned long unc = copy_to_user(dst_ua, cpu_segs[i].src,
						 cpu_segs[i].len);

		if (unlikely(unc)) {
			WARN_ON_ONCE(1);
			total_dma -= unc;
		}
	}
	nr_cpu_segs = 0;

	if (total_dma < len) {
		ssize_t cpu_ret;

		iov_iter_advance(dst_iter, total_dma);
		cpu_ret = io_dma_cpu_copy(dst_iter, src_iter, len - total_dma);
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

/*
 * Submit a single DMA descriptor for one batch entry. Used for a lone chunk,
 * where a 1-entry DSA batch descriptor would be pure overhead. The entry
 * already has DMA-mapped src_dma from the caller.
 */
static ssize_t io_dma_submit_single_entry(struct io_kiocb *req,
					  struct dma_chan *chan,
					  struct io_dma_batch_entry *entry)
{
	struct dma_async_tx_descriptor *tx;
	struct io_dma_task *dma;

	tx = dmaengine_prep_dma_memcpy(chan, entry->dst_dma, entry->src_dma,
				       entry->src_len, io_dma_prep_flags());
	if (!tx)
		return -EAGAIN;

	dma = io_dma_task_alloc(req->ctx);
	if (!dma)
		return -ENOMEM;

	if (entry->src_is_page)
		folio_get(entry->folio);

	dma->req = req;
	dma->next = NULL;
	dma->src_dma = entry->src_dma;
	dma->dst_dma = entry->dst_dma;
	dma->len = entry->src_len;
	dma->src_map_addr = entry->src_dma;
	dma->src_map_len = entry->src_len;
	dma->src_folio = entry->src_is_page ? entry->folio : NULL;
	dma->src_is_page = entry->src_is_page;
	dma->is_batch = false;

	dma->submit_ns = ktime_get_ns();
	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		if (entry->src_is_page)
			folio_put(entry->folio);
		io_dma_task_free(req->ctx, dma);
		return -EFAULT;
	}

	io_dma_task_link(req, dma);

	return entry->src_len;
}

/*
 * Release the source side of batch entries: unmap each DMA mapping (skipped
 * under IOMMU passthrough) and, when put_folios is set, drop the folio ref a
 * page-cache source took. Used by the error path (folio refs not yet taken ->
 * put_folios=false), task completion, and the teardown drain.
 */
void io_dma_unmap_batch(struct io_ring_ctx *ctx, struct device *dev,
			struct io_dma_batch_entry *entries, unsigned int nr,
			bool put_folios)
{
	unsigned int i;

	for (i = 0; i < nr; i++) {
		struct io_dma_batch_entry *e = &entries[i];

		if (!ctx->dma.use_phys_addrs) {
			if (e->src_is_page)
				dma_unmap_page(dev, e->src_dma, e->src_len,
					       DMA_TO_DEVICE);
			else
				dma_unmap_single(dev, e->src_dma, e->src_len,
						 DMA_TO_DEVICE);
		}
		if (put_folios && e->src_is_page)
			folio_put(e->folio);
	}
}

/*
 * Flush collected batch entries: a single entry uses a plain memcpy
 * descriptor (a 1-entry batch descriptor is pure overhead); two or more are
 * submitted as one DSA batch descriptor. On submit failure the entries are
 * unmapped (folio refs aren't taken until submit succeeds, so put_folios is
 * false).
 */
static ssize_t io_dma_flush_batch(struct io_kiocb *req,
				  struct device *dev, struct dma_chan *chan,
				  struct io_dma_batch_entry *entries,
				  unsigned int nr_entries)
{
	ssize_t ret;

	if (!nr_entries)
		return 0;

	if (nr_entries == 1)
		ret = io_dma_submit_single_entry(req, chan, &entries[0]);
	else
		ret = io_dma_submit_batch(req, dev, chan, entries, nr_entries);

	if (ret < 0)
		io_dma_unmap_batch(req->ctx, dev, entries, nr_entries, false);
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

	/*
	 * NOWAIT: never block on the read datapath. On failure the caller
	 * (io_read) falls through to the normal buffered-read path.
	 */
	entries = kmalloc_array(IO_DMA_BATCH_MAX, sizeof(*entries),
				GFP_NOWAIT | __GFP_NOWARN);
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
				entries[nr_entries].src_is_page = true;
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

static void __io_dma_task_complete(struct device *dev, struct io_dma_task *dma,
				   int ret)
{
	struct io_kiocb *req;
	u32 task_len;

	req = dma->req;
	task_len = dma->len;

	/* Clean up source DMA mappings and folio references */
	if (dma->is_batch) {
		io_dma_unmap_batch(req->ctx, dev, dma->batch_entries,
				   dma->batch_nr, true);
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
		if (dma->submit_ns)
			io_dma_lat_record(&io_dma_lat_dma, task_len,
					  ktime_get_ns() - dma->submit_ns);
		pr_debug("dma task complete: len=%u result=%d\n",
			 task_len, req->dma.dma_result);
		if (req->dma.dma_result >= 0)
			req->dma.dma_result += task_len;
	} else {
		pr_debug("dma task failed: len=%u ret=%d is_batch=%d\n",
			 task_len, ret, dma->is_batch);
		req->dma.dma_result = -EFAULT;
	}

	/* Free the task before touching refcnt -- task_len saved above */
	io_dma_task_free(req->ctx, dma);
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

		/*
		 * The recv only reaches the DMA path while poll-armed (io_recv
		 * gates on REQ_F_POLLED), so the req is always in the poll
		 * cancel-hash here. Drive every completion through poll ownership
		 * (io_poll_kick) so io_poll_task_func does the single hash_del:
		 *
		 *  - multishot success: post an aux CQE and stay armed. io_recv's
		 *    prelude flushes pending_aux_cqe before the next sock_recvmsg.
		 *  - terminal (one-shot recv, or a multishot DMA error): set the
		 *    saved result and tear the poll down via dma_terminal, which
		 *    io_poll_check_events() turns into IOU_POLL_REMOVE_POLL_USE_RES.
		 */
		if (req->dma.dma_result >= 0 &&
		    (req->flags & REQ_F_APOLL_MULTISHOT)) {
			req->dma.pending_aux_cqe = true;
		} else {
			if (req->dma.dma_result < 0) {
				req_set_fail(req);
				req->dma.saved_res = req->dma.dma_result;
			}
			req->dma.dma_terminal = true;
		}
		io_poll_kick(req);

		/*
		 * Drop the in-flight DMA reference taken in
		 * io_dma_submit_queued_tasks(). The completion handling above
		 * only queues task_work (re-arm via io_poll_kick, or terminal
		 * via io_req_task_complete) — it does not free the req inline —
		 * so the req is still valid here. If we held the last reference
		 * (the req was already terminally completed/cancelled, e.g. by
		 * teardown), free it now; otherwise the owner frees it once it
		 * drops its reference.
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

	if (IS_ERR_OR_NULL(ctx->dma.chan)) {
		/*
		 * CPU baseline path (no DSA): the copy already happened
		 * synchronously in io_uring_copy_to_iter(). TCP has now
		 * unlinked the source skb (tcp_eat_recv_skb ran with
		 * free=false), so release it here.
		 */
		if (req->dma.cb_fn) {
			struct io_dma *iod = io_kiocb_to_cmd(req, struct io_dma);

			req->dma.cb_fn(&iod->kiocb, req->dma.cb_arg, 0);
			req->dma.cb_fn = NULL;
		}
		return 0;
	}

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

			/*
			 * Hold a reference for the in-flight DMA. io_recv is
			 * about to return IOU_ISSUE_SKIP_COMPLETE, handing the
			 * req off to the DMA engine, but the req remains in the
			 * poll cancel-hash. Without this ref, ring-teardown
			 * cancellation (io_poll_remove_all) could free the req
			 * while DMA still references it (and while it is still
			 * hashed) -> use-after-free. The ref keeps it alive until
			 * the last task completes; dropped in
			 * __io_dma_task_complete() at dma_refcnt == 0. Mirrors the
			 * io-wq reference pattern in io_wq_submit_work().
			 */
			if (!(req->flags & REQ_F_REFCOUNT))
				__io_req_set_refcount(req, 2);
			else
				req_ref_get(req);
			req->dma.dma_ref_held = true;

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
	if (ret == -EIOCBQUEUED) {
		/*
		 * This req's tasks were just queued and io_recv will return
		 * IOU_ISSUE_SKIP_COMPLETE. Do NOT complete them inline here: a
		 * fast transfer (especially a single batched descriptor) can
		 * finish synchronously, and __io_dma_task_complete() would then
		 * re-enter io_poll_kick() for this same req while io_recv is
		 * still unwinding — double-completing it (manifests as an
		 * imbalanced file-ref put / req double-free at exit). Defer all
		 * completion to poll_work, which runs from a clean context.
		 */
		if (READ_ONCE(ctx->dma.head))
			schedule_work(&ctx->dma.poll_work);
	} else if (atomic_read(&ctx->dma.poll_armed) == 0) {
		/*
		 * No task was queued for this req (e.g. CPU fallback). Draining
		 * other reqs' already-queued tasks inline is safe — io_poll_kick
		 * for a different req is ordinary async wakeup, not re-entrancy
		 * on the req currently being issued.
		 */
		__io_dma_poll(ctx);
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
