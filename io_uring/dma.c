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
 * DMA completion-detection mode, switchable at runtime via
 * /sys/kernel/debug/io_uring_dma_completion_mode:
 *
 *  - kworker:  schedule_work() busy-poll on a shared kworker.
 *  - mwait:    per-ctx kthread UMONITOR/UMWAITs on the completion record.
 *  - irq:      interrupt-driven, no completion thread.
 */
enum {
	IO_DMA_COMP_KWORKER,
	IO_DMA_COMP_MWAIT,
	IO_DMA_COMP_IRQ,
	IO_DMA_COMP_NR,
};
static const char * const io_dma_comp_mode_names[IO_DMA_COMP_NR] = {
	[IO_DMA_COMP_KWORKER]  = "kworker",
	[IO_DMA_COMP_MWAIT]    = "mwait",
	[IO_DMA_COMP_IRQ]      = "irq",
};
/* Default to the original kworker path so behaviour is unchanged until set. */
static unsigned int io_dma_completion_mode __read_mostly = IO_DMA_COMP_KWORKER;

/* For io_recv() (net.c) to capture the recv-only IRQ mode at prep time. */
bool io_dma_irq_mode(void)
{
	return READ_ONCE(io_dma_completion_mode) == IO_DMA_COMP_IRQ;
}

/* dmaengine completion callback for IRQ mode; defined after the completion
 * helpers it calls.
 */
static void io_dma_irq_complete(void *param, const struct dmaengine_result *res);

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

static int io_dma_comp_mode_show(struct seq_file *m, void *v)
{
	unsigned int cur = READ_ONCE(io_dma_completion_mode);
	unsigned int i;

	for (i = 0; i < IO_DMA_COMP_NR; i++)
		seq_printf(m, "%s%s%s ", i == cur ? "[" : "",
			   io_dma_comp_mode_names[i], i == cur ? "]" : "");
	seq_putc(m, '\n');
	return 0;
}

static int io_dma_comp_mode_open(struct inode *inode, struct file *file)
{
	return single_open(file, io_dma_comp_mode_show, NULL);
}

static ssize_t io_dma_comp_mode_write(struct file *file, const char __user *ubuf,
				      size_t len, loff_t *ppos)
{
	char buf[16], *mode;
	unsigned int i;

	if (len == 0 || len >= sizeof(buf))
		return -EINVAL;
	if (copy_from_user(buf, ubuf, len))
		return -EFAULT;
	buf[len] = '\0';
	mode = strim(buf);

	for (i = 0; i < IO_DMA_COMP_NR; i++) {
		if (strcmp(mode, io_dma_comp_mode_names[i]))
			continue;
		/* mwait is not wired up yet. */
		if (i == IO_DMA_COMP_MWAIT)
			return -EOPNOTSUPP;
		WRITE_ONCE(io_dma_completion_mode, i);
		return len;
	}
	return -EINVAL;
}

static const struct file_operations io_dma_comp_mode_fops = {
	.owner		= THIS_MODULE,
	.open		= io_dma_comp_mode_open,
	.read		= seq_read,
	.write		= io_dma_comp_mode_write,
	.llseek		= seq_lseek,
	.release	= single_release,
};

void io_dma_debugfs_init(void)
{
	debugfs_create_file("io_uring_dma_latency", 0444, NULL, NULL,
			    &io_dma_lat_fops);
	debugfs_create_file("io_uring_dma_latency_reset", 0200, NULL, NULL,
			    &io_dma_lat_reset_fops);
	debugfs_create_file("io_uring_dma_completion_mode", 0644, NULL, NULL,
			    &io_dma_comp_mode_fops);
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

/*
 * Per-ctx completion kthread for the mwait mode. Sleeps
 * until a submission wakes it (io_dma_submit_queued_tasks -> wake_up), then
 * drains __io_dma_poll() until the in-flight list empties. __io_dma_poll() is
 * serialized against the teardown drain via ctx->dma.poll_armed, so it is safe
 * to run here concurrently with io_ring_exit_work().
 */
static int io_dma_compl_thread_fn(void *data)
{
	struct io_ring_ctx *ctx = data;

	while (!kthread_should_stop()) {
		wait_event_interruptible(ctx->dma.compl_wait,
					 READ_ONCE(ctx->dma.head) ||
					 kthread_should_stop());

		while (READ_ONCE(ctx->dma.head) && !kthread_should_stop()) {
			__io_dma_poll(ctx);
			if (!READ_ONCE(ctx->dma.head))
				break;
			/*
			 * Wait between drains. mwait (UMONITOR/UMWAIT on the
			 * head completion record) lands in a later step; until
			 * then the thread spins between passes.
			 */
			cpu_relax();
		}
	}
	return 0;
}

/*
 * Start the per-ctx completion kthread, pinned to a CPU on the DMA device's
 * NUMA node (required for mwait to monitor the device's writes).
 * On failure compl_thread stays NULL and the submit path falls back to the
 * poll_work kworker, so this never fails channel setup.
 */
void io_dma_compl_thread_start(struct io_ring_ctx *ctx)
{
	struct task_struct *t;
	int node, cpu;

	init_waitqueue_head(&ctx->dma.compl_wait);
	ctx->dma.compl_thread = NULL;

	t = kthread_create(io_dma_compl_thread_fn, ctx, "iou-dmac/%d",
			   task_pid_nr(current));
	if (IS_ERR(t)) {
		pr_warn("io_uring DMA: completion kthread create failed (%pe); "
			"using kworker fallback\n", t);
		return;
	}

	node = dev_to_node(ctx->dma.chan->device->dev);
	cpu = (node != NUMA_NO_NODE) ? cpumask_first(cpumask_of_node(node))
				     : nr_cpu_ids;
	if (cpu >= nr_cpu_ids)
		cpu = cpumask_first(cpu_online_mask);
	kthread_bind(t, cpu);

	ctx->dma.compl_thread = t;
	wake_up_process(t);
}

/* Stop the completion kthread. Safe to call when it was never started. */
void io_dma_compl_thread_stop(struct io_ring_ctx *ctx)
{
	if (ctx->dma.compl_thread) {
		kthread_stop(ctx->dma.compl_thread);
		ctx->dma.compl_thread = NULL;
	}
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
	 * task_work runs (deferred-wake deadlock if lazy).
	 *
	 * It's not sufficient to set this flag inside io_poll_kick: an earlier
	 * task_work may already be queued (from the initial poll wake / inline
	 * issue) with LAZY_WAKE baked into its nr_tw. Setting the flag here,
	 * before any task_work is queued for the multishot-DMA recv, ensures
	 * every subsequent __io_poll_execute for this req uses non-lazy wake.
	 */
	req->flags |= REQ_F_POLL_NO_LAZY;

	req->dma.dma_active = true;
	/*
	 * Default to the poll/kworker completion path. IRQ mode is opt-in
	 * and recv-only: io_recv() sets irq_mode after this, capturing the
	 * knob once so the interrupt arming, the publish decision, and the
	 * doorbell all agree even if the knob flips mid-recv.
	 *
	 * dma_ref_held is deliberately NOT reset here: it is set/cleared
	 * only under ctx->dma.lock by the submit/complete ref protocol, and
	 * a multishot reissue reaches this prep while the previous cycle's
	 * completer may still be about to drop the previous in-flight ref.
	 * Clearing the flag here would erase that pending drop and leak one
	 * req reference per race.  A fresh req needs no init: the submit
	 * path sets it.
	 */
	req->dma.irq_mode = false;
	req->dma.dma_refcnt = 0;
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
	 * Allocate everything that can fail BEFORE preparing the descriptor.
	 * A prepared dmaengine descriptor has no unprepare API: once
	 * dmaengine_prep_*() returns one it must be submitted or it leaks from
	 * the engine's descriptor pool. So prep is the last fallible step
	 * before dmaengine_submit().
	 */
	heap_entries = kmalloc_array(nr_entries, sizeof(*heap_entries),
				     GFP_NOWAIT | __GFP_NOWARN);
	if (!heap_entries) {
		kfree(sgls);
		return -ENOMEM;
	}
	memcpy(heap_entries, entries, nr_entries * sizeof(*heap_entries));

	dma = io_dma_task_alloc(req->ctx);
	if (!dma) {
		kfree(heap_entries);
		kfree(sgls);
		return -ENOMEM;
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
		io_dma_task_free(req->ctx, dma);
		kfree(heap_entries);
		kfree(sgls);
		return -EAGAIN;
	}

	/* SG arrays are consumed by dmaengine_prep_dma_memcpy_sg —
	 * the driver copies what it needs into batch descriptors.
	 */
	kfree(sgls);

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
static ssize_t io_dma_submit_batch_sg(struct io_kiocb *req, struct device *dev,
				      struct dma_chan *chan,
				      struct io_dma_batch_entry *entries,
				      unsigned int nr_entries);
static ssize_t io_dma_recv_flush(struct io_kiocb *req, struct device *dev,
				 struct dma_chan *chan,
				 struct io_dma_batch_entry *entries,
				 unsigned int nr_entries);

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
	unsigned int nr_entries = 0;
	unsigned int call_chunks = 0;
	u64 call_t0;
	ssize_t batch_ret;

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
	call_t0 = ktime_get_ns();

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
	 * Each entry also carries its CPU-reachable source (src_kaddr) and
	 * destination offset (dst_off) so the error path can CPU-copy a chunk
	 * that was collected but not yet submitted to hardware. At
	 * IO_DMA_BATCH_MAX == 16 the array is small enough to live on the stack,
	 * which avoids a per-recv allocation on the hot path.
	 */
	struct io_dma_batch_entry entries[IO_DMA_BATCH_MAX];

	/*
	 * Walk the source one chunk at a time. src_iter is advanced per chunk,
	 * so it always points at the next byte not yet claimed for DMA — which
	 * keeps the error path's CPU copy of the remainder trivial.
	 *
	 * Every segment of a DMA'd transfer rides in the batch, including ones
	 * smaller than the threshold: the whole-transfer threshold gate above
	 * already routes small recvs to a full CPU copy, so a transfer that
	 * reaches here is worth DSA, and a sub-threshold segment is just one more
	 * sub-descriptor in a batch we are already paying to build. Deferring
	 * such segments to a CPU copy_to_user() pass (as an earlier design did)
	 * burned ~2us of submit-thread time per recv copying data the engine
	 * could move for ~free — the dominant cost in the submit path.
	 */
	while (total_dma < len) {
		size_t seg_avail, folio_remain, chunk_len;
		void *src_kaddr;
		dma_addr_t dst_dma;

		seg_avail = iter_iov_len(src_iter);
		if (!seg_avail)
			break;
		seg_avail = min_t(size_t, seg_avail, len - total_dma);
		src_kaddr = (void *)iter_iov_addr(src_iter);

		/* Resolve the destination folio and clamp the chunk to it so no
		 * single transfer crosses an IOMMU mapping.
		 */
		dst_dma = io_dma_dst_addr(req, dst_iter, dst_base_addr,
					  total_dma, &folio_remain);
		if (!dst_dma)
			goto cpu_fallback_rest;
		chunk_len = min_t(size_t, seg_avail, folio_remain);

		/*
		 * Record the chunk; the source is mapped later, once per batch
		 * via dma_map_sg() in io_dma_recv_flush(). Deferring the map
		 * means a chunk that is collected but never submitted (error
		 * path) was never mapped, so it just needs a CPU copy.
		 */
		entries[nr_entries].dst_dma = dst_dma;
		entries[nr_entries].src_kaddr = src_kaddr;
		entries[nr_entries].dst_off = total_dma;
		entries[nr_entries].src_len = chunk_len;
		entries[nr_entries].folio = NULL;
		entries[nr_entries].src_is_page = false;
		nr_entries++;
		call_chunks++;

		total_dma += chunk_len;
		iov_iter_advance(src_iter, chunk_len);

		/* Flush a full batch and keep collecting. */
		if (nr_entries == IO_DMA_BATCH_MAX) {
			batch_ret = io_dma_recv_flush(req, dev, ctx->dma.chan,
						      entries, nr_entries);
			if (batch_ret < 0)
				goto cpu_fallback_rest;
			nr_entries = 0;
		}
	}

	/* Flush the final partial batch (single memcpy or one SG batch). */
	if (nr_entries) {
		batch_ret = io_dma_recv_flush(req, dev, ctx->dma.chan,
					      entries, nr_entries);
		if (batch_ret < 0)
			goto cpu_fallback_rest;
		nr_entries = 0;
	}

	/* Advance destination iterator past all processed bytes. */
	iov_iter_advance(dst_iter, total_dma);

	/*
	 * Kick the engine so the hardware copies start immediately — except in
	 * IRQ mode, where the doorbell is deferred to io_dma_submit_queued_tasks
	 * so the in-flight ref is taken before any completion interrupt fires.
	 */
	if (req->dma.dma_refcnt > 0 && !req->dma.irq_mode)
		dma_async_issue_pending(ctx->dma.chan);


	return total_dma;

cpu_fallback_rest:
	/*
	 * A destination folio was unmapped or a batch failed to submit. Recover
	 * without losing data:
	 *   - CPU-copy the chunks collected but not yet submitted. Their source
	 *     mapping is deferred to io_dma_recv_flush(), so these were never
	 *     mapped (a failed flush unmaps its own batch); just copy from
	 *     entries[].src_kaddr.
	 *   - CPU-copy the remainder [total_dma, len), whose source is exactly
	 *     where src_iter now points (advanced per claimed chunk).
	 * Already-submitted batches are left to complete in hardware; their
	 * destination ranges are disjoint from everything copied here.
	 */
	if (nr_entries) {
		for (i = 0; i < nr_entries; i++) {
			void __user *dst_ua = (void __user *)
				(dst_base_addr + entries[i].dst_off);
			unsigned long unc = copy_to_user(dst_ua,
							 entries[i].src_kaddr,
							 entries[i].src_len);

			if (unlikely(unc)) {
				WARN_ON_ONCE(1);
				total_dma -= unc;
			}
		}
		nr_entries = 0;
	}

	if (total_dma < len) {
		ssize_t cpu_ret;

		iov_iter_advance(dst_iter, total_dma);
		cpu_ret = io_dma_cpu_copy(dst_iter, src_iter, len - total_dma);
		if (cpu_ret > 0)
			total_dma += cpu_ret;
	}
	/* IRQ mode defers the doorbell to io_dma_submit_queued_tasks (see above). */
	if (req->dma.dma_refcnt > 0 && !req->dma.irq_mode)
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
 * Hand the recv copy path a scratch kvec that stays live and privately owned
 * for the whole (sleepable) io_uring_copy_to_iter() call. The buffer is cached
 * on the request and reused across multishot recv reissues, so the steady
 * state costs no allocation; it is freed when the req leaves the slab cache
 * (io_dma_free_recv_kvec(), called from __io_req_caches_free()).
 *
 * A per-CPU buffer cannot serve this: io_uring_copy_to_iter() reads the kvec
 * throughout its body, including its CPU-copy fallback which can sleep, so once
 * preemption is re-enabled another recvmsg on the same CPU would clobber a
 * shared buffer mid-copy. Returns NULL if the (first-use) allocation fails; the
 * caller then falls back to a plain CPU copy.
 */
struct kvec *io_uring_recv_kvec(struct kiocb *kiocb, unsigned int nr)
{
	struct io_dma *cmd = container_of(kiocb, struct io_dma, kiocb);
	struct io_kiocb *req = cmd_to_io_kiocb(cmd);

	if (req->dma.recv_kvec_nr < nr) {
		/*
		 * The only caller is tcp_recvmsg_locked(), sleepable process
		 * context, so reclaim is allowed: the NULL return (and the
		 * caller's CPU-copy fallback) is all but unreachable.
		 */
		struct kvec *kv = kmalloc_array(nr, sizeof(*kv), GFP_KERNEL);
		if (!kv)
			return NULL;
		kfree(req->dma.recv_kvec);
		req->dma.recv_kvec = kv;
		req->dma.recv_kvec_nr = nr;
	}
	return req->dma.recv_kvec;
}
EXPORT_SYMBOL_GPL(io_uring_recv_kvec);

/* Release a request's cached recv kvec. Called for every req as it leaves the
 * slab cache; a no-op for reqs that never took the DMA recv path.
 */
void io_dma_free_recv_kvec(struct io_kiocb *req)
{
	if (req->dma.recv_kvec) {
		kfree(req->dma.recv_kvec);
		req->dma.recv_kvec = NULL;
		req->dma.recv_kvec_nr = 0;
	}
}

/*
 * Submit a recv batch (>=2 chunks), mapping all sources with a single
 * dma_map_sg() instead of one dma_map_single() per chunk. dma_map_sg() may
 * coalesce physically-contiguous sources into fewer segments; that is safe
 * because idxd_dma_prep_memcpy_sg() walks the source and destination lists
 * independently and re-splits at the finer boundary, so each pre-mapped
 * destination chunk still gets its own MEMMOVE sub-descriptor. The mapped
 * source SG is the unmap handle, held on the task until completion. recv
 * sources are skb kvecs (no folio refs), so no batch_entries copy is kept.
 */
static ssize_t io_dma_submit_batch_sg(struct io_kiocb *req, struct device *dev,
				      struct dma_chan *chan,
				      struct io_dma_batch_entry *entries,
				      unsigned int nr_entries)
{
	struct dma_async_tx_descriptor *tx;
	struct scatterlist *src_sg, *dst_sg;
	struct io_dma_task *dma;
	size_t total_len = 0;
	unsigned int dma_nents;
	unsigned int i;

	src_sg = kmalloc_array(nr_entries, sizeof(*src_sg),
			       GFP_NOWAIT | __GFP_NOWARN);
	if (!src_sg)
		return -ENOMEM;
	dst_sg = kmalloc_array(nr_entries, sizeof(*dst_sg),
			       GFP_NOWAIT | __GFP_NOWARN);
	if (!dst_sg) {
		kfree(src_sg);
		return -ENOMEM;
	}

	/* Map all source chunks in one shot. */
	sg_init_table(src_sg, nr_entries);
	for (i = 0; i < nr_entries; i++)
		sg_set_buf(&src_sg[i], entries[i].src_kaddr, entries[i].src_len);

	dma_nents = dma_map_sg(dev, src_sg, nr_entries, DMA_TO_DEVICE);
	if (!dma_nents) {
		kfree(dst_sg);
		kfree(src_sg);
		return -EFAULT;
	}

	/*
	 * The destination is a pre-mapped registered/provided buffer: carry its
	 * DMA addresses directly, uncoalesced (one segment per chunk).
	 */
	sg_init_table(dst_sg, nr_entries);
	for (i = 0; i < nr_entries; i++) {
		sg_dma_address(&dst_sg[i]) = entries[i].dst_dma;
		sg_dma_len(&dst_sg[i]) = entries[i].src_len;
		total_len += entries[i].src_len;
	}

	/*
	 * Allocate the task BEFORE preparing the descriptor: a prepared
	 * dmaengine descriptor has no unprepare API and must be submitted or
	 * it leaks, so prep is the last fallible step before submit.
	 */
	dma = io_dma_task_alloc(req->ctx);
	if (!dma) {
		dma_unmap_sg(dev, src_sg, nr_entries, DMA_TO_DEVICE);
		kfree(dst_sg);
		kfree(src_sg);
		return -ENOMEM;
	}

	tx = dmaengine_prep_dma_memcpy_sg(chan, dst_sg, nr_entries, src_sg,
			dma_nents,
			io_dma_prep_flags() |
			(req->dma.irq_mode ? DMA_PREP_INTERRUPT : 0));
	if (!tx) {
		io_dma_task_free(req->ctx, dma);
		dma_unmap_sg(dev, src_sg, nr_entries, DMA_TO_DEVICE);
		kfree(dst_sg);
		kfree(src_sg);
		return -EAGAIN;
	}

	/* dst_sg is consumed by prep; src_sg is held for unmap at completion. */
	kfree(dst_sg);

	dma->req = req;
	dma->next = NULL;
	dma->len = total_len;
	dma->is_batch = true;
	dma->batch_nr = nr_entries;
	dma->batch_entries = NULL;
	dma->batch_src_sg = src_sg;
	dma->batch_src_nents = nr_entries;

	/* IRQ mode: completion arrives via io_dma_irq_complete(); interrupt
	 * descriptors can't be polled (idxd returns DMA_IN_PROGRESS for them).
	 */
	if (req->dma.irq_mode) {
		tx->callback_result = io_dma_irq_complete;
		tx->callback_param = dma;
	}

	dma->cookie = dmaengine_submit(tx);
	if (dma_submit_error(dma->cookie)) {
		dma_unmap_sg(dev, src_sg, nr_entries, DMA_TO_DEVICE);
		kfree(src_sg);
		io_dma_task_free(req->ctx, dma);
		return -EFAULT;
	}

	io_dma_task_link(req, dma);

	return total_len;
}

/*
 * Flush a collected recv batch. A lone chunk uses a plain memcpy descriptor
 * (mapped here; a 1-entry SG batch is pure overhead); two or more go through
 * io_dma_submit_batch_sg(). On failure the sources are left unmapped (the
 * single map is undone here, the SG map is undone inside io_dma_submit_batch_sg)
 * so the caller can CPU-copy them from src_kaddr.
 */
static ssize_t io_dma_recv_flush(struct io_kiocb *req, struct device *dev,
				 struct dma_chan *chan,
				 struct io_dma_batch_entry *entries,
				 unsigned int nr_entries)
{
	ssize_t ret;
	dma_addr_t src_dma;

	if (nr_entries != 1)
		return io_dma_submit_batch_sg(req, dev, chan, entries,
					      nr_entries);

	src_dma = dma_map_single(dev, entries[0].src_kaddr, entries[0].src_len,
				 DMA_TO_DEVICE);
	if (dma_mapping_error(dev, src_dma))
		return -EFAULT;
	entries[0].src_dma = src_dma;

	ret = io_dma_submit_single_entry(req, chan, &entries[0]);
	if (ret < 0 && !req->ctx->dma.use_phys_addrs)
		dma_unmap_single(dev, src_dma, entries[0].src_len, DMA_TO_DEVICE);
	return ret;
}

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

	/*
	 * Allocate the task BEFORE preparing the descriptor: a prepared
	 * dmaengine descriptor has no unprepare API and must be submitted or
	 * it leaks, so prep is the last fallible step before submit.
	 */
	dma = io_dma_task_alloc(req->ctx);
	if (!dma)
		return -ENOMEM;

	tx = dmaengine_prep_dma_memcpy(chan, entry->dst_dma, entry->src_dma,
			entry->src_len,
			io_dma_prep_flags() |
			(req->dma.irq_mode ? DMA_PREP_INTERRUPT : 0));
	if (!tx) {
		io_dma_task_free(req->ctx, dma);
		return -EAGAIN;
	}

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

	/* IRQ mode (recv only): complete via io_dma_irq_complete(). */
	if (req->dma.irq_mode) {
		tx->callback_result = io_dma_irq_complete;
		tx->callback_param = dma;
	}

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
 * Release a completed (or torn-down) batch task's source side. A recv batch
 * carries a single dma_map_sg() mapping (batch_src_sg); everything else carries
 * a per-entry batch_entries array. Called from task completion and the teardown
 * drain, which must stay in sync.
 */
void io_dma_batch_cleanup(struct io_ring_ctx *ctx, struct device *dev,
			  struct io_dma_task *dma)
{
	if (dma->batch_src_sg) {
		if (!ctx->dma.use_phys_addrs)
			dma_unmap_sg(dev, dma->batch_src_sg,
				     dma->batch_src_nents, DMA_TO_DEVICE);
		kfree(dma->batch_src_sg);
	} else {
		io_dma_unmap_batch(ctx, dev, dma->batch_entries, dma->batch_nr,
				   true);
	}
	kfree(dma->batch_entries);	/* NULL for SG-mapped recv batches */
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
				entries[nr_entries].src_is_page = true;
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
		io_dma_batch_cleanup(req->ctx, dev, dma);
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
		 * Complete according to whether the req is still poll-armed.
		 *
		 * Multishot recv stays poll-armed across reissues (its apoll is
		 * not EPOLLONESHOT; io_poll_check_events() reissues it in place),
		 * so it is still in the poll cancel-hash. Drive completion through
		 * poll ownership (io_poll_kick) so io_poll_task_func() does the
		 * single hash_del() and we do not race poll's task_work llist:
		 *  - success: post an aux CQE and stay armed (pending_aux_cqe);
		 *    io_recv's prelude flushes it before the next sock_recvmsg.
		 *  - DMA error: terminate via dma_terminal, which
		 *    io_poll_check_events() turns into REMOVE_POLL_USE_RES.
		 *
		 * One-shot recv: its apoll is EPOLLONESHOT, so the first poll
		 * event already tore it down (io_poll_task_func(): hash_del() +
		 * io_req_task_submit()) before this reissue offloaded to DMA. The
		 * req is no longer poll-armed and io_poll_kick() cannot reacquire
		 * ownership, so complete it directly -- there is no poll task_work
		 * to race. saved_res/saved_cflags were captured in io_recv().
		 */
		if (req->flags & REQ_F_APOLL_MULTISHOT) {
			if (req->dma.dma_result >= 0) {
				req->dma.pending_aux_cqe = true;
			} else {
				req_set_fail(req);
				req->dma.saved_res = req->dma.dma_result;
				req->dma.dma_terminal = true;
			}
			io_poll_kick(req);
		} else {
			if (req->dma.dma_result < 0) {
				req_set_fail(req);
				io_req_set_res(req, req->dma.dma_result, 0);
			} else {
				io_req_set_res(req, req->dma.saved_res,
					       req->dma.saved_cflags);
			}
			req->io_task_work.func = io_req_task_complete;
			io_req_task_work_add(req);
		}

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

/*
 * Release a task's resources without completing the req. Used by the IRQ
 * callback during channel release (ctx->dma.releasing): idxd aborts any
 * still-in-flight descriptor and invokes the callback, but completing the req
 * that late would need io_free_req() whose task_work falls back onto the
 * tearing-down ctx -> UAF (see io_release_dma_chan). So unmap + free the task
 * and leave the req's in-flight ref leaked, exactly as io_release_dma_chan
 * does for a hung task. Confined to the already-degraded teardown path.
 */
static void io_dma_task_release_orphan(struct io_ring_ctx *ctx,
				       struct device *dev,
				       struct io_dma_task *dma)
{
	if (dma->is_batch) {
		io_dma_batch_cleanup(ctx, dev, dma);
	} else {
		if (dma->src_map_len && !ctx->dma.use_phys_addrs) {
			if (dma->src_is_page)
				dma_unmap_page(dev, dma->src_map_addr,
					       dma->src_map_len, DMA_TO_DEVICE);
			else
				dma_unmap_single(dev, dma->src_map_addr,
						 dma->src_map_len, DMA_TO_DEVICE);
		}
		if (dma->src_folio)
			folio_put(dma->src_folio);
	}
	kmem_cache_free(dma_cachep, dma);
}

/*
 * IRQ-mode completion. Runs in idxd's threaded IRQ (process context, no idxd
 * lock held). Interrupt descriptors can't be polled, so this is the only
 * completion path for them. Take ctx->dma.lock so __io_dma_task_complete()
 * runs under the same serialization the poll path gives it (notably the
 * non-atomic dma_refcnt, and against any concurrent drain). If the channel is
 * being released, orphan the task instead of completing (see above).
 */
static void io_dma_irq_complete(void *param, const struct dmaengine_result *res)
{
	struct io_dma_task *dma = param;
	struct io_kiocb *req = dma->req;
	struct io_ring_ctx *ctx = req->ctx;
	struct device *dev = ctx->dma.chan->device->dev;
	int status = (res && res->result == DMA_TRANS_NOERROR) ?
		     DMA_COMPLETE : DMA_ERROR;
	unsigned long flags;

	spin_lock_irqsave(&ctx->dma.lock, flags);
	if (ctx->dma.releasing)
		io_dma_task_release_orphan(ctx, dev, dma);
	else
		__io_dma_task_complete(dev, dma, status);
	spin_unlock_irqrestore(&ctx->dma.lock, flags);
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
			 * become reachable by any completer. In non-IRQ mode the
			 * hardware doorbell was already rung in
			 * io_uring_copy_to_iter(), so the instant the tasks are
			 * published to ctx->dma.head below a concurrent drain --
			 * the poll_work kworker, the busypoll kthread, or
			 * io_ring_exit_work -- can reach dma_refcnt == 0 in
			 * __io_dma_task_complete(). Were the ref taken after
			 * publishing, that drain would complete the req, see
			 * dma_ref_held == false and drop nothing, and this path
			 * would then set refcount/dma_ref_held on an
			 * already-completed req: double-complete / orphaned ref /
			 * use-after-free. Taking it here (before publish, and
			 * before IRQ mode's deferred doorbell) closes the window;
			 * dropped in __io_dma_task_complete() at dma_refcnt == 0.
			 * Mirrors io_wq_submit_work().
			 *
			 * The take must ALSO be under ctx->dma.lock, the lock
			 * every completer holds through its dma_refcnt == 0
			 * block. A multishot reissue reaches here while the
			 * PREVIOUS cycle's completer may still be between
			 * publishing pending_aux_cqe / kicking the poll (which
			 * is what let this reissue run) and dropping the
			 * previous in-flight ref. Taken unlocked, this
			 * ref/flag write interleaves into that window: the
			 * completer then either skips its drop or donates it to
			 * the new cycle, and one reference leaks each time --
			 * accumulating on the long-lived multishot req until it
			 * can never be freed and ring teardown wedges. The lock
			 * orders this take strictly after that drop.
			 *
			 * io_recv is about to return IOU_ISSUE_SKIP_COMPLETE,
			 * handing the req to the DMA engine while it stays in the
			 * poll cancel-hash; without this ref, teardown
			 * cancellation (io_poll_remove_all) could free the still-
			 * hashed req while DMA references it.
			 */
			spin_lock_irqsave(&ctx->dma.lock, flags);
			if (!(req->flags & REQ_F_REFCOUNT))
				__io_req_set_refcount(req, 2);
			else
				req_ref_get(req);
			req->dma.dma_ref_held = true;

			if (!req->dma.irq_mode) {
				/*
				 * Splice the req's task list onto ctx->dma.head.
				 * Every task on the list was fully submitted with
				 * a valid cookie (io_dma_task_link() runs only
				 * after dmaengine_submit() succeeds), so the
				 * pollers can consume the list as-is.
				 *
				 * IRQ-mode tasks are not published: each one
				 * completes via its dmaengine callback
				 * (io_dma_irq_complete), and the doorbell is
				 * deferred to the -EIOCBQUEUED branch below so no
				 * callback can fire before the ref taken above
				 * exists.
				 */
				if (ctx->dma.tail == NULL)
					ctx->dma.head = req->dma.dma_tasks;
				else
					ctx->dma.tail->next = req->dma.dma_tasks;
				ctx->dma.tail = req->dma.dma_tasks_tail;
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
	if (ret == -EIOCBQUEUED) {
		/*
		 * This req's tasks were just queued and io_recv will return
		 * IOU_ISSUE_SKIP_COMPLETE. Do NOT complete them inline here: a
		 * fast transfer (especially a single batched descriptor) can
		 * finish synchronously, and __io_dma_task_complete() would then
		 * re-enter io_poll_kick() for this same req while io_recv is
		 * still unwinding — double-completing it (manifests as an
		 * imbalanced file-ref put / req double-free at exit). Defer all
		 * completion to a clean context: the per-ctx kthread in mwait
		 * mode (woken here), otherwise the poll_work kworker. A NULL
		 * compl_thread (create failed) also falls back to the kworker.
		 */
		if (req->dma.irq_mode) {
			/*
			 * IRQ mode: ring the doorbell now, after the in-flight
			 * ref was taken above, so a completion callback can never
			 * run __io_dma_task_complete() before the ref exists.
			 * Tasks are not on ctx->dma.head; the callbacks complete
			 * them.
			 */
			dma_async_issue_pending(ctx->dma.chan);
		} else if (READ_ONCE(ctx->dma.head)) {
			if (READ_ONCE(io_dma_completion_mode) == IO_DMA_COMP_MWAIT &&
			    ctx->dma.compl_thread)
				wake_up(&ctx->dma.compl_wait);
			else
				schedule_work(&ctx->dma.poll_work);
		}
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

	/*
	 * Do NOT ring the channel doorbell here. dma_async_issue_pending() is
	 * channel-global, and an IRQ-mode request deliberately leaves its
	 * descriptors unissued between dmaengine_submit() in
	 * io_uring_copy_to_iter() and the ref-then-doorbell sequence in
	 * io_dma_submit_queued_tasks(): a ring from this context — which runs
	 * with no uring_lock serialization against that window — would let a
	 * completion interrupt fire before the in-flight ref exists and before
	 * TCP has unlinked the source skb (double-complete, receive-queue
	 * corruption). Nothing on ctx->dma.head needs a kick anyway: pollable
	 * (non-interrupt) descriptors reach the hardware at dmaengine_submit()
	 * time (see idxd_dma_tx_submit), and every submitter rings for its own
	 * tasks from the issue path, under uring_lock.
	 */
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
