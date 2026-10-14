// SPDX-License-Identifier: GPL-2.0-only
/*
 * DMA Core Batch Migrator (DCBM)
 *
 * Uses DMAEngine memcpy channels to offload batch folio copies during
 * page migration. Reference driver meant for testing the offload
 * infrastructure.
 *
 * Copyright (C) 2024-26 Advanced Micro Devices, Inc.
 */

#include <linux/bitops.h>
#include <linux/module.h>
#include <linux/dma-mapping.h>
#include <linux/dmaengine.h>
#include <linux/highmem.h>
#include <linux/ktime.h>
#include <linux/migrate.h>
#include <linux/mm_offload.h>
#include <linux/mm_offload_dma.h>
#include <linux/mutex.h>
#include <linux/rcupdate.h>
#include <linux/scatterlist.h>
#include <linux/sizes.h>
#include <linux/slab.h>
#include <linux/wait.h>
#include <linux/xarray.h>

/*
 * Copies per scatter-gather transaction. A provider that supports
 * DMA_MEMCPY_SG turns one transaction into one hardware batch, so
 * this bounds the batch element count it must accept.
 */
#define DCBM_SG_ELEMS_DEFAULT	32
/*
 * Descriptors in flight per channel. A DSA work queue owns a fixed
 * descriptor pool (its queue depth, 64 in a typical configuration)
 * and prep returns NULL once it is exhausted, so a slice submits at
 * most this many descriptors before waiting for completions.
 */
#define DCBM_MAX_INFLIGHT_DEFAULT	32
#define DCBM_SG_ELEMS_MAX		1024
/* A refused prep is retried this often while the channel drains. */
#define DCBM_PREP_RETRIES		3
/* Batches at least this large wait for channel budget instead of falling back. */
#define DCBM_ADMIT_WAIT_BYTES		SZ_8M
/* How often to say so while a completion is overdue. */
#define DCBM_WAIT_WARN_MS		10000
/* Matches FOLIO_ZERO_LOCALITY_RADIUS in mm/memory.c */
#define DCBM_WARM_RADIUS	2
/*
 * Smallest share of a copy to give a channel. A PMD folio spread over
 * eight channels lands on 256K each, the split DSA-2LM found best for
 * migrating a 2M page (Liu et al., ATC'25).
 */
#define DCBM_MIN_CHUNK_DEFAULT	SZ_256K
/* Largest single descriptor; longer slices are cut into several. */
#define DCBM_MAX_CHUNK_DEFAULT	SZ_2M
/* Below this an operation is not worth a descriptor at all. */
#define DCBM_MIN_BYTES_DEFAULT	SZ_32K
/*
 * Largest fill descriptor. A DSA work queue transfers at most 2M per
 * descriptor at its default configuration, and the prep refuses more.
 */
#define DCBM_CLEAR_CHUNK_BYTES	SZ_2M
/*
 * Segments one clear may hold references on. A PMD folio needs sixteen
 * at the default segment size; a folio needing more than this maps
 * itself rather than growing the fault path an allocation.
 */
#define DCBM_CLEAR_MAX_SEGS	32

static atomic_long_t folios_migrated;
static atomic_long_t folios_failures;
static atomic_long_t batches_refused;
static atomic_long_t admit_failures;	/* channel budget never freed up */
static atomic_long_t folios_cleared;
static atomic_long_t clear_failures;
static atomic_long_t folios_gated;

static bool offloading_enabled;
/*
 * Channels one operation may claim. The claim is confined to the devices
 * of the operation's node, so this only bounds how far a copy is spread
 * there: 16 is every channel of a four-device socket at four kernel work
 * queues per device.
 */
static unsigned int nr_dma_channels = 16;
static unsigned int sg_elems = DCBM_SG_ELEMS_DEFAULT;
static unsigned int max_inflight = DCBM_MAX_INFLIGHT_DEFAULT;
static bool cache_ctrl = true;
static unsigned long min_clear_bytes = SZ_2M;
static bool cpu_warm = true;
static bool util_gate = true;
static bool map_cache_clear = true;
static unsigned long min_chunk_bytes = DCBM_MIN_CHUNK_DEFAULT;
static unsigned long max_chunk_bytes = DCBM_MAX_CHUNK_DEFAULT;
static unsigned long min_bytes = DCBM_MIN_BYTES_DEFAULT;
static DEFINE_MUTEX(dcbm_mutex);


struct dcbm_copy {
	dma_addr_t src;
	dma_addr_t dst;
	size_t len;
};

/*
 * What a completion callback gets: the work it belongs to and the channel
 * whose admission slot it returns. Copy works keep an array of these in
 * the plan; the clear path allocates one per descriptor and the callback
 * frees it.
 */
struct dcbm_desc {
	struct dma_work *work;
	unsigned int chan;
	bool free_me;
	/* usage accounting: what this descriptor does and when it went out */
	enum mm_offload_dma_op op;
	size_t bytes;
	ktime_t submitted;
};

struct dma_work {
	struct dma_chan *chan;
	unsigned int chan_idx;
	struct device *dev;
	wait_queue_head_t waitq;	/* throttle only */
	struct completion done;		/* the last descriptor landed */
	atomic_t pending;		/* lifetime: submitter + every descriptor */
	atomic_t inflight;		/* throttle: descriptors not yet completed */
	atomic_t error;
	/* Slice of the plan's descriptor array belonging to this channel. */
	struct dcbm_copy *copies;
	unsigned int nr_copies;
	struct dcbm_desc *descs;	/* one per submission, same slice */
	unsigned int nr_descs;
	bool admit_wait;		/* sleep for channel budget, or give up */
	bool submitted;
};

/* One folio pair to copy. */
struct dcbm_pair {
	struct folio *src;
	struct folio *dst;
};

/*
 * One range of one folio pair mapped against one device. A folio is
 * physically contiguous, so a range of it is one DMA segment; the
 * descriptors that cover it are offsets into this mapping, which keeps
 * the number of IOMMU mappings at one per folio per device however
 * finely the copy is sliced.
 */
struct dcbm_map {
	struct device *dev;
	/*
	 * Cache entries when the range came from the mapping cache, NULL
	 * when it was mapped for this copy alone and must be unmapped.
	 */
	struct mm_offload_dma_map *src_ent;
	struct mm_offload_dma_map *dst_ent;
	dma_addr_t src;
	dma_addr_t dst;
	size_t off;			/* start within the whole copy */
	size_t len;
};

/* The channels claimed on one device, and the bytes they carry. */
struct dcbm_dev {
	struct device *dev;
	unsigned long chans;
	unsigned int nr_chan;
	size_t start;
	size_t end;
};

/*
 * A planned copy: every folio pair cut into per-device mappings and
 * per-channel descriptors.
 */
struct dcbm_plan {
	struct dma_work *works;
	unsigned int nr_works;
	struct dcbm_map *maps;
	unsigned int nr_maps;
	unsigned int max_maps;
	struct dcbm_copy *copies;
	unsigned int nr_copies;
	unsigned int max_copies;
	struct dcbm_desc *descs;	/* parallel to copies */
	struct dcbm_dev *devs;		/* planning scratch, one per channel */
	unsigned long chan_mask;
};

/*
 * Every descriptor carries its own completion. Engines such as Intel
 * DSA complete the descriptors of one channel out of order when the
 * work queue is served by several engines, so a callback on the last
 * submitted descriptor does not mean the earlier ones have landed.
 * The pending count starts at one for the submitter, so the work
 * cannot complete before every descriptor has been submitted.
 *
 * A descriptor the hardware rejected or aborted completes with an
 * error result; it must fail the batch, or a folio that was never
 * written would be reported as copied.
 */
static void dma_completion_callback(void *data,
				    const struct dmaengine_result *result)
{
	struct dcbm_desc *desc = data;
	struct dma_work *work = desc->work;
	unsigned int chan = desc->chan;
	bool ok = result && result->result == DMA_TRANS_NOERROR;

	mm_offload_dma_stat_complete(desc->op, desc->bytes, desc->submitted, ok);
	if (desc->free_me)
		kfree(desc);
	mm_offload_dma_complete(chan, 1);

	if (!ok)
		atomic_set(&work->error, -EIO);

	/*
	 * Two counts. The throttle count is what the submitter sleeps on,
	 * and it is woken here while this callback still holds a lifetime
	 * reference, so the work cannot go away under the wake. The
	 * lifetime reference is dropped last: with several channels
	 * completing into one work, another completer (or the submitter's
	 * own release) may take the count to zero the moment this one lets
	 * go, and the waiter then frees the work - nothing below the drop
	 * may touch it.
	 */
	atomic_dec(&work->inflight);
	wake_up(&work->waitq);
	if (atomic_dec_and_test(&work->pending))
		complete(&work->done);
}

static void dma_work_init(struct dma_work *work)
{
	init_waitqueue_head(&work->waitq);
	init_completion(&work->done);
	/* Submission reference, dropped by dma_work_done_submitting(). */
	atomic_set(&work->pending, 1);
	atomic_set(&work->inflight, 0);
	atomic_set(&work->error, 0);
}

static void dma_work_done_submitting(struct dma_work *work)
{
	if (atomic_dec_and_test(&work->pending))
		complete(&work->done);
}

/*
 * Throttle submission to max_inflight descriptors per channel, so a
 * slice never outruns the channel's descriptor pool.
 */
static void dma_work_throttle(struct dma_work *work)
{
	/* Signed and clamped so the condition is reachable for any setting. */
	int limit = (int)clamp_t(unsigned int, READ_ONCE(max_inflight),
				 1u, (unsigned int)INT_MAX - 1);
	unsigned int overdue = 0;

	while (!wait_event_timeout(work->waitq,
				   atomic_read(&work->inflight) < limit,
				   msecs_to_jiffies(DCBM_WAIT_WARN_MS))) {
		overdue++;
		pr_warn_ratelimited("dcbm: throttled %u s waiting for completions, still waiting\n",
				    overdue * (DCBM_WAIT_WARN_MS / 1000));
	}
}


/*
 * Map [off, off + len) of one folio pair against @dev. A folio is
 * physically contiguous, so any range of it is a single DMA segment,
 * and every descriptor covering that range is an offset into this one
 * mapping - the copy can be sliced as finely as the engines want
 * without paying an extra IOMMU mapping per slice.
 */
static int plan_map(struct dcbm_plan *plan, struct device *dev,
		    struct folio *src, struct folio *dst,
		    size_t in_folio, size_t off, size_t len)
{
	struct dcbm_map *m = &plan->maps[plan->nr_maps];
	unsigned long pgoff = in_folio >> PAGE_SHIFT;

	/*
	 * The arrays are sized from the segment size the caches reported
	 * before planning began; a flush can change it under us, so refuse
	 * rather than run off the end. The batch falls back to the CPU.
	 */
	if (unlikely(plan->nr_maps >= plan->max_maps))
		return -ENOSPC;

	m->dev = dev;
	m->off = off;
	m->len = len;

	m->src_ent = mm_offload_dma_map_get(dev, src, in_folio, len, &m->src);
	if (!m->src_ent) {
		m->src = dma_map_page_attrs(dev, folio_page(src, pgoff), 0, len,
					    DMA_TO_DEVICE, 0);
		if (dma_mapping_error(dev, m->src))
			return -EIO;
	}

	if (m->src_ent) {
		/* The CPU wrote this folio since the mapping was made. */
		dma_sync_single_for_device(dev, m->src, len, DMA_TO_DEVICE);
	}

	m->dst_ent = mm_offload_dma_map_get(dev, dst, in_folio, len, &m->dst);
	if (!m->dst_ent) {
		m->dst = dma_map_page_attrs(dev, folio_page(dst, pgoff), 0, len,
					    DMA_FROM_DEVICE, 0);
		if (dma_mapping_error(dev, m->dst)) {
			if (m->src_ent)
				mm_offload_dma_map_put(m->src_ent);
			else
				dma_unmap_page_attrs(dev, m->src, len,
						     DMA_TO_DEVICE, 0);
			return -EIO;
		}
	} else {
		/* The device is about to write it; it owns it until then. */
		dma_sync_single_for_device(dev, m->dst, len, DMA_FROM_DEVICE);
	}
	plan->nr_maps++;
	return 0;
}

static void plan_unmap(struct dcbm_plan *plan)
{
	unsigned int i;

	for (i = 0; i < plan->nr_maps; i++) {
		struct dcbm_map *m = &plan->maps[i];

		if (m->dst_ent) {
			/* The device wrote it; hand it back to the CPU. */
			dma_sync_single_for_cpu(m->dev, m->dst, m->len,
						DMA_FROM_DEVICE);
			mm_offload_dma_map_put(m->dst_ent);
		} else
			dma_unmap_page_attrs(m->dev, m->dst, m->len,
					     DMA_FROM_DEVICE, 0);
		if (m->src_ent)
			mm_offload_dma_map_put(m->src_ent);
		else
			dma_unmap_page_attrs(m->dev, m->src, m->len,
					     DMA_TO_DEVICE, 0);
	}
	plan->nr_maps = 0;
}

/*
 * Wait for every descriptor of a slice that was handed to the engine.
 * Nothing is ever terminated: the descriptors reference the folio
 * mappings and the on-stack work until they complete, dmaengine has
 * no per-descriptor abort, and a channel-wide terminate on an engine
 * such as DSA tears down the channel's interrupt handle, after which
 * no later descriptor on that channel completes. Anything submitted
 * runs to completion; a failure only decides what the caller does
 * afterwards.
 */
static int dma_work_wait(struct dma_work *work)
{
	unsigned int overdue = 0;

	if (!work->submitted)
		return 0;
	/*
	 * Untimed by design (see above), but not silent: a descriptor the
	 * engine dropped without a callback would otherwise be an invisible
	 * D-state task.
	 */
	while (!wait_for_completion_timeout(&work->done,
					    msecs_to_jiffies(DCBM_WAIT_WARN_MS))) {
		overdue++;
		pr_warn_ratelimited("dcbm: DMA completion overdue (%u s), still waiting\n",
				    overdue * (DCBM_WAIT_WARN_MS / 1000));
	}
	return atomic_read(&work->error);
}

static void plan_free(struct dcbm_plan *plan)
{
	plan_unmap(plan);
	kfree(plan->works);
	kvfree(plan->maps);
	kvfree(plan->copies);
	kvfree(plan->descs);
	kfree(plan->devs);
	plan->works = NULL;
	plan->maps = NULL;
	plan->copies = NULL;
	plan->descs = NULL;
	plan->devs = NULL;
}

/*
 * Submit a prepared descriptor whose channel slot @desc already holds.
 * On failure the slot is returned here; on success the completion
 * callback returns it.
 */
static int submit_one(struct dma_work *work, struct dma_async_tx_descriptor *tx,
		      struct dcbm_desc *desc)
{
	dma_cookie_t cookie;

	tx->callback_result = dma_completion_callback;
	tx->callback_param = desc;
	atomic_inc(&work->inflight);
	atomic_inc(&work->pending);
	desc->submitted = ktime_get();
	mm_offload_dma_stat_submit(desc->op, desc->bytes);

	cookie = dmaengine_submit(tx);
	if (dma_submit_error(cookie)) {
		/* Never queued, so no callback will run for it. */
		mm_offload_dma_stat_complete(desc->op, desc->bytes, desc->submitted, false);
		atomic_dec(&work->inflight);
		atomic_dec(&work->pending);
		mm_offload_dma_retire(desc->chan, 1);
		if (desc->free_me)
			kfree(desc);
		return -EIO;
	}
	/*
	 * Start it now. The throttle sleeps for completions before the
	 * next descriptor is prepared, and on a provider whose submit only
	 * queues (a virt-dma driver such as ptdma) nothing completes until
	 * issue_pending: deferring it to the end of the slice would wait
	 * on descriptors that never ran. The descriptor's own channel,
	 * because a work may spread its descriptors over several.
	 */
	dma_async_issue_pending(tx->chan);
	return 0;
}

/*
 * Take one admission slot on the work's channel and the context that
 * will return it. NULL if the channel stayed over budget for
 * admit_timeout_ms: the rest of the batch is the CPU's.
 */
static struct dcbm_desc *work_admit(struct dma_work *work)
{
	struct dcbm_desc *desc;

	if (mm_offload_dma_admit(work->chan_idx, 1, work->admit_wait)) {
		atomic_long_inc(&admit_failures);
		return NULL;
	}
	desc = &work->descs[work->nr_descs++];
	desc->work = work;
	desc->chan = work->chan_idx;
	desc->free_me = false;
	return desc;
}

/*
 * Hand the copies to the provider as scatter-gather transactions of
 * up to sg_elems folios each. A batch-capable engine such as DSA
 * executes one transaction as one hardware batch descriptor, so the
 * submission cost is paid once per sg_elems folios instead of once
 * per folio. The scatterlists carry the DMA addresses mapped by
 * map_folios(); they are not mapped again.
 */
static int submit_sg_transfers(struct dma_work *work, unsigned long flags)
{
	struct scatterlist *src_sg, *dst_sg;
	struct dcbm_desc *desc;
	unsigned int elems = clamp(READ_ONCE(sg_elems), 1u, DCBM_SG_ELEMS_MAX);
	unsigned int retries = 0;
	unsigned int done = 0;
	size_t bytes;
	int ret = 0;

	/* Up to 64 KB, and the device never reads it: no need for it contiguous. */
	src_sg = kvmalloc_array(2 * elems, sizeof(*src_sg), GFP_KERNEL);
	if (!src_sg)
		return -ENOMEM;
	dst_sg = src_sg + elems;

	while (done < work->nr_copies) {
		unsigned int n = min(elems, work->nr_copies - done);
		struct dma_async_tx_descriptor *tx;
		unsigned int i;

		sg_init_table(src_sg, n);
		sg_init_table(dst_sg, n);
		bytes = 0;
		for (i = 0; i < n; i++) {
			struct dcbm_copy *copy = &work->copies[done + i];

			sg_dma_address(&src_sg[i]) = copy->src;
			sg_dma_len(&src_sg[i]) = copy->len;
			sg_dma_address(&dst_sg[i]) = copy->dst;
			sg_dma_len(&dst_sg[i]) = copy->len;
			bytes += copy->len;
		}

		dma_work_throttle(work);
		desc = work_admit(work);
		if (!desc) {
			ret = -EBUSY;
			break;
		}
		desc->op = MM_OFFLOAD_DMA_OP_MIGRATE;
		desc->bytes = bytes;
		tx = dmaengine_prep_dma_memcpy_sg(work->chan, dst_sg, n,
						  src_sg, n, flags);
		if (!tx) {
			mm_offload_dma_retire(desc->chan, 1);
			work->nr_descs--;
			/* Ring full behind the budget: let it drain, try again. */
			if (work->admit_wait && retries++ < DCBM_PREP_RETRIES &&
			    !mm_offload_dma_prep_wait(work->chan_idx))
				continue;
			ret = -EIO;
			break;
		}
		ret = submit_one(work, tx, desc);
		if (ret)
			break;
		done += n;
		retries = 0;
	}

	kvfree(src_sg);
	return ret;
}

static int submit_dma_transfers(struct dma_work *work)
{
	struct dma_async_tx_descriptor *tx;
	unsigned long flags = DMA_CTRL_ACK | DMA_PREP_INTERRUPT;
	unsigned int i;
	int ret;

	/*
	 * Cache-allocating writes leave the copied data LLC-warm for the
	 * first access after remapping, at the cost of cache footprint
	 * for folios that are not touched soon.
	 */
	if (READ_ONCE(cache_ctrl))
		flags |= DMA_PREP_CACHE_CONTROL;

	dma_work_init(work);

	if (dma_has_cap(DMA_MEMCPY_SG, work->chan->device->cap_mask))
		return submit_sg_transfers(work, flags);

	for (i = 0; i < work->nr_copies; i++) {
		struct dcbm_copy *copy = &work->copies[i];
		struct dcbm_desc *desc;
		unsigned int retries = 0;

		for (;;) {
			dma_work_throttle(work);
			desc = work_admit(work);
			if (!desc)
				return -EBUSY;
			desc->op = MM_OFFLOAD_DMA_OP_MIGRATE;
			desc->bytes = copy->len;
			tx = dmaengine_prep_dma_memcpy(work->chan, copy->dst,
						       copy->src, copy->len,
						       flags);
			if (tx)
				break;
			mm_offload_dma_retire(desc->chan, 1);
			work->nr_descs--;
			if (!work->admit_wait || retries++ >= DCBM_PREP_RETRIES ||
			    mm_offload_dma_prep_wait(work->chan_idx))
				return -EIO;
		}

		ret = submit_one(work, tx, desc);
		if (ret)
			return ret;
	}
	return 0;
}

/*
 * Cut the copy into per-device mappings and per-channel descriptors.
 *
 * Each device gets a byte-balanced share of the whole copy, so a folio
 * needs mapping at most once per device; that share is then split
 * evenly over the device's channels, cutting inside a folio where a
 * channel boundary falls there. A lone PMD folio handed to eight
 * channels therefore becomes eight 256K descriptors rather than one 2M
 * descriptor on one channel, which is the split DSA-2LM measured as
 * best for migrating a 2M page.
 *
 * Which device a folio goes to decides whether its standing mapping
 * is ever found again: the mapping cache is per device. A batch with
 * enough folios to balance by count therefore deals them out by the
 * PFN of the folio that lives on the devices' own node, so the same
 * folio meets the same device every time it is copied - as the
 * destination of one migration and the source of the next, or as a
 * destination recycled through its node's free lists. A batch too
 * small for that, a lone gigantic folio above all, is cut into
 * contiguous shares in a fixed device order instead, which repeats for
 * a repeated batch.
 */
static int plan_build(struct dcbm_plan *plan, struct dcbm_pair *pairs,
		      unsigned int nr, size_t total)
{
	struct dcbm_dev *devs = plan->devs;
	size_t max_chunk = max_t(size_t, READ_ONCE(max_chunk_bytes), PAGE_SIZE);
	unsigned int ndev = 0, nwork = 0, idx, d, c, m;
	unsigned int cur = 0;
	size_t folio_start = 0, off = 0;
	bool by_pfn, key_dst = false;
	int ret;

	/*
	 * Group the claimed channels by the device they sit on. Channel
	 * order is pool order, so the devices come out in the same order
	 * for every batch that claims the same set.
	 */
	for_each_set_bit(idx, &plan->chan_mask, MM_OFFLOAD_DMA_MAX_CHANNELS) {
		struct device *dev = mm_offload_dma_chan_dev(idx);

		for (d = 0; d < ndev; d++)
			if (devs[d].dev == dev)
				break;
		if (d == ndev) {
			devs[ndev].dev = dev;
			devs[ndev].chans = 0;
			devs[ndev].nr_chan = 0;
			ndev++;
		}
		devs[d].chans |= BIT(idx);
		devs[d].nr_chan++;
	}

	by_pfn = ndev > 1 && nr >= 2 * ndev;
	if (by_pfn) {
		/*
		 * The folio on the devices' own node names the device. With
		 * the devices on the destination node that is the
		 * destination; with a destination node that has none (a
		 * CPU-less tier) and the source node's devices doing the
		 * work, it is the source.
		 */
		key_dst = dev_to_node(devs[0].dev) == folio_nid(pairs[0].dst);
	} else {
		for (d = 0; d < ndev; d++) {
			devs[d].start = off;
			off = (d == ndev - 1) ? total :
				min(total, ALIGN(total * (d + 1) / ndev, PAGE_SIZE));
			devs[d].end = off;
		}
	}

	for (d = 0; d < ndev; d++) {
		size_t dstart, dend;
		unsigned int first_map = plan->nr_maps;
		unsigned int p;
		size_t o, fs, quantum;

		/*
		 * Entries are per device, so the cache follows the device
		 * whose share is being planned, and mappings are cut at its
		 * segment boundaries so each one is a whole entry.
		 */
		quantum = mm_offload_dma_map_quantum(devs[d].dev);

		if (by_pfn) {
			/*
			 * This device's folios, concatenated: the offsets
			 * below are positions in that concatenation, which is
			 * all the channel split needs.
			 */
			dstart = 0;
			o = 0;
			for (p = 0; p < nr; p++) {
				struct folio *key = key_dst ? pairs[p].dst :
							      pairs[p].src;
				size_t fsize, in_folio = 0;

				if ((folio_pfn(key) >> (PMD_SHIFT - PAGE_SHIFT)) %
				    ndev != d)
					continue;
				fsize = folio_size(pairs[p].src);
				while (in_folio < fsize) {
					size_t len = fsize - in_folio;

					if (quantum)
						len = min(len, quantum -
							  (in_folio & (quantum - 1)));
					ret = plan_map(plan, devs[d].dev,
						       pairs[p].src, pairs[p].dst,
						       in_folio, o, len);
					if (ret)
						return ret;
					in_folio += len;
					o += len;
				}
			}
			dend = o;
		} else {
			dstart = devs[d].start;
			dend = devs[d].end;

			/* Walk the cursor to the folio holding the first byte. */
			while (cur < nr &&
			       folio_start + folio_size(pairs[cur].src) <= dstart) {
				folio_start += folio_size(pairs[cur].src);
				cur++;
			}

			/* One mapping per folio range this device touches. */
			for (o = dstart, p = cur, fs = folio_start;
			     o < dend && p < nr; ) {
				size_t fsize = folio_size(pairs[p].src);
				size_t in_folio = o - fs;
				size_t len = min(fsize - in_folio, dend - o);

				if (quantum)
					len = min(len, quantum -
						  (in_folio & (quantum - 1)));

				ret = plan_map(plan, devs[d].dev, pairs[p].src,
					       pairs[p].dst, in_folio, o, len);
				if (ret)
					return ret;
				o += len;
				if (o >= fs + fsize) {
					fs += fsize;
					p++;
				}
			}
		}

		/* Split the device's share evenly over its channels. */
		c = 0;
		for_each_set_bit(idx, &devs[d].chans, MM_OFFLOAD_DMA_MAX_CHANNELS) {
			struct dma_work *work = &plan->works[nwork];
			size_t span = dend - dstart;
			size_t cstart, cend;

			cstart = dstart + ALIGN_DOWN(span * c / devs[d].nr_chan,
						     PAGE_SIZE);
			cend = (c == devs[d].nr_chan - 1) ? dend :
				dstart + ALIGN_DOWN(span * (c + 1) /
						    devs[d].nr_chan, PAGE_SIZE);
			c++;

			work->chan = mm_offload_dma_chan(idx);
			work->chan_idx = idx;
			work->dev = devs[d].dev;
			work->copies = &plan->copies[plan->nr_copies];
			work->nr_copies = 0;
			work->descs = &plan->descs[plan->nr_copies];
			work->nr_descs = 0;
			nwork++;

			for (m = first_map; m < plan->nr_maps && cstart < cend; m++) {
				struct dcbm_map *mp = &plan->maps[m];
				size_t s, e;

				if (mp->off + mp->len <= cstart)
					continue;
				if (mp->off >= cend)
					break;
				s = max(cstart, mp->off);
				e = min(cend, mp->off + mp->len);
				while (s < e) {
					struct dcbm_copy *cp;
					size_t len = min(e - s, max_chunk);

					if (unlikely(plan->nr_copies >=
						     plan->max_copies))
						return -ENOSPC;
					cp = &plan->copies[plan->nr_copies++];
					cp->src = mp->src + (s - mp->off);
					cp->dst = mp->dst + (s - mp->off);
					cp->len = len;
					work->nr_copies++;
					s += len;
				}
			}
		}
	}
	plan->nr_works = nwork;
	return 0;
}

/*
 * Copy @nr folio pairs. Returns 0 only when every byte was copied; the
 * caller then marks the destinations so the move phase skips them.
 */
static int copy_pairs_dma(struct dcbm_pair *pairs, unsigned int nr)
{
	struct dcbm_plan plan = {};
	struct mm_offload_dma_group *grp;
	unsigned int want, nchan, i;
	size_t total = 0, max_maps, max_copies, unit;
	int nid, ret = 0;

	for (i = 0; i < nr; i++) {
		if (folio_size(pairs[i].dst) != folio_size(pairs[i].src))
			return -EINVAL;
		total += folio_size(pairs[i].src);
	}
	if (total < READ_ONCE(min_bytes))
		return -EINVAL;

	/*
	 * Ask for as many channels as the copy can keep usefully busy:
	 * one per min_chunk_bytes of work, so a small batch is not spread
	 * so thin that each channel gets a descriptor too short to be
	 * worth its completion.
	 */
	want = clamp_t(size_t, total / max_t(size_t, READ_ONCE(min_chunk_bytes),
					     PAGE_SIZE),
		       1, READ_ONCE(nr_dma_channels));

	/* Prefer the device closest to where the copies are written. */
	nid = folio_nid(pairs[0].dst);
	plan.chan_mask = mm_offload_dma_claim_spread(want, nid);
	if (!plan.chan_mask)
		plan.chan_mask = mm_offload_dma_claim(want, nid, &grp);
	if (!plan.chan_mask) {
		atomic_long_inc(&batches_refused);
		return -EBUSY;
	}
	nchan = hweight_long(plan.chan_mask);

	/*
	 * A folio is cut at most once per device boundary, and a channel
	 * slice at most once per mapping it crosses and once per
	 * max_chunk_bytes it spans.
	 */
	/*
	 * Segments cut mappings and mappings cut descriptors, so the
	 * smallest segment any claimed device uses bounds both counts.
	 */
	unit = max_t(size_t, READ_ONCE(max_chunk_bytes), PAGE_SIZE);
	for_each_set_bit(i, &plan.chan_mask, MM_OFFLOAD_DMA_MAX_CHANNELS) {
		size_t q = mm_offload_dma_map_quantum(mm_offload_dma_chan_dev(i));

		if (q)
			unit = min(unit, q);
	}
	max_maps = total / unit + nr + 2 * nchan + 2;
	max_copies = max_maps + 2 * nchan + 2;
	plan.max_maps = max_maps;
	plan.max_copies = max_copies;
	plan.works = kcalloc(nchan, sizeof(*plan.works), GFP_KERNEL);
	/*
	 * Segment-sized mappings make these arrays large and the device
	 * never reads them, so they need not be physically contiguous.
	 */
	plan.maps = kvcalloc(max_maps, sizeof(*plan.maps), GFP_KERNEL);
	plan.copies = kvcalloc(max_copies, sizeof(*plan.copies), GFP_KERNEL);
	plan.descs = kvcalloc(max_copies, sizeof(*plan.descs), GFP_KERNEL);
	plan.devs = kcalloc(nchan, sizeof(*plan.devs), GFP_KERNEL);
	if (!plan.works || !plan.maps || !plan.copies || !plan.descs ||
	    !plan.devs) {
		ret = -ENOMEM;
		goto out;
	}

	ret = plan_build(&plan, pairs, nr, total);
	if (ret)
		goto out;

	for (i = 0; i < plan.nr_works; i++) {
		/*
		 * A bulk migration is worth sleeping for channel budget; a
		 * single folio moved from a fault (NUMA hinting) is not, and
		 * its caller may hold locks - it goes to the CPU at once.
		 */
		plan.works[i].admit_wait = total >= DCBM_ADMIT_WAIT_BYTES;
	}

	for (i = 0; i < plan.nr_works; i++) {
		if (!plan.works[i].nr_copies)
			continue;
		ret = submit_dma_transfers(&plan.works[i]);
		dma_work_done_submitting(&plan.works[i]);
		plan.works[i].submitted = true;
		if (ret)
			break;
	}
	for (i = 0; i < plan.nr_works; i++)
		ret = dma_work_wait(&plan.works[i]) ? : ret;

out:
	plan_free(&plan);
	mm_offload_dma_release(plan.chan_mask);

	if (ret == -EBUSY) {
		/*
		 * Refused, not failed: no channel (batches_refused)
		 * or no budget (admit_failures).
		 */
	} else if (ret) {
		atomic_long_add(nr, &folios_failures);
		pr_warn_ratelimited("dcbm: DMA copy failed (%d), falling back to CPU\n",
				    ret);
	} else {
		atomic_long_add(nr, &folios_migrated);
	}
	return ret;
}

/**
 * folios_copy_dma - copy a batch of folios via DMA memcpy
 * @dst_list: destination folio list
 * @src_list: source folio list
 * @nr_folios: number of folios in each list
 *
 * Return: 0 on success, negative errno on failure.
 */
static int folios_copy_dma(struct list_head *dst_list,
			   struct list_head *src_list, unsigned int nr_folios)
{
	struct list_head *src_pos = src_list->next;
	struct list_head *dst_pos = dst_list->next;
	struct dcbm_pair *pairs;
	unsigned int i;
	int ret;

	pairs = kcalloc(nr_folios, sizeof(*pairs), GFP_KERNEL);
	if (!pairs)
		return -ENOMEM;

	for (i = 0; i < nr_folios; i++) {
		pairs[i].src = list_entry(src_pos, struct folio, lru);
		pairs[i].dst = list_entry(dst_pos, struct folio, lru);
		src_pos = src_pos->next;
		dst_pos = dst_pos->next;
	}

	ret = copy_pairs_dma(pairs, nr_folios);
	if (!ret) {
		for (i = 0; i < nr_folios; i++)
			folio_set_migrate_copied(pairs[i].dst);
	}
	kfree(pairs);
	return ret;
}

/**
 * folio_pairs_copy_dma - copy folio pairs given as arrays
 * @dst: destination folios
 * @src: source folios
 * @nr: number of pairs
 *
 * Return: 0 on success, negative errno on failure.
 */
static int folio_pairs_copy_dma(struct folio **dst, struct folio **src,
				unsigned int nr)
{
	struct dcbm_pair *pairs;
	unsigned int i;
	int ret;

	pairs = kcalloc(nr, sizeof(*pairs), GFP_KERNEL);
	if (!pairs)
		return -ENOMEM;

	for (i = 0; i < nr; i++) {
		pairs[i].src = src[i];
		pairs[i].dst = dst[i];
	}

	ret = copy_pairs_dma(pairs, nr);
	if (!ret) {
		for (i = 0; i < nr; i++)
			folio_set_migrate_copied(dst[i]);
	}
	kfree(pairs);
	return ret;
}

/*
 * Split [addr, addr + len) into chunks of at most DCBM_CLEAR_CHUNK_BYTES,
 * dealt round-robin over the claimed channels, and submit one memset
 * per chunk, each with its own completion. A 1G folio is 512 chunks
 * over however many channels were claimed; the throttle keeps the
 * number in flight bounded.
 */
static int submit_clear_range(struct dma_work *work, unsigned long chan_mask,
			      dma_addr_t addr, size_t len, unsigned long flags)
{
	unsigned int nchunks = DIV_ROUND_UP(len, DCBM_CLEAR_CHUNK_BYTES);
	size_t chunk = DIV_ROUND_UP(len, nchunks);
	unsigned int idx = find_first_bit(&chan_mask, MM_OFFLOAD_DMA_MAX_CHANNELS);

	while (len) {
		size_t this_len = min(chunk, len);
		struct dma_async_tx_descriptor *tx;
		int ret;

		struct dcbm_desc *desc;

		dma_work_throttle(work);
		desc = kmalloc(sizeof(*desc), GFP_KERNEL);
		if (!desc)
			return -ENOMEM;
		/*
		 * A clear runs in the fault path under mmap_lock; better to
		 * clear on the CPU than to sleep for budget holding it.
		 */
		if (mm_offload_dma_admit(idx, 1, false)) {
			atomic_long_inc(&admit_failures);
			kfree(desc);
			return -EBUSY;
		}
		desc->work = work;
		desc->chan = idx;
		desc->free_me = true;
		desc->op = MM_OFFLOAD_DMA_OP_CLEAR;
		desc->bytes = this_len;
		tx = dmaengine_prep_dma_memset(mm_offload_dma_chan(idx), addr, 0,
					       this_len, flags);
		if (!tx) {
			mm_offload_dma_retire(idx, 1);
			kfree(desc);
			return -EIO;
		}

		ret = submit_one(work, tx, desc);
		if (ret)
			return ret;

		addr += this_len;
		len -= this_len;

		idx = find_next_bit(&chan_mask, MM_OFFLOAD_DMA_MAX_CHANNELS, idx + 1);
		if (idx >= MM_OFFLOAD_DMA_MAX_CHANNELS)
			idx = find_first_bit(&chan_mask, MM_OFFLOAD_DMA_MAX_CHANNELS);
	}
	return 0;
}

/*
 * Fill [off, off + len) of the folio. Without the cache one mapping
 * covers the whole folio and this is a single range; with it the folio
 * is covered by segment-sized standing mappings, so the range is walked
 * a segment at a time.
 */
static int clear_folio_range(struct dma_work *work, unsigned long chan_mask,
			     dma_addr_t base, const dma_addr_t *segdma,
			     size_t quantum, size_t off, size_t len,
			     unsigned long flags)
{
	while (len) {
		size_t this_len;
		dma_addr_t addr;
		int ret;

		if (segdma) {
			this_len = min(len, quantum - (off & (quantum - 1)));
			addr = segdma[off / quantum] + (off & (quantum - 1));
		} else {
			this_len = len;
			addr = base + off;
		}

		ret = submit_clear_range(work, chan_mask, addr, this_len, flags);
		if (ret)
			return ret;
		off += this_len;
		len -= this_len;
	}
	return 0;
}

/*
 * Resolve every segment of @folio through the cache, so the clear can
 * run on standing mappings. All or nothing: a folio only partly covered
 * would need both paths at once for no gain, so on any refusal the
 * references taken so far go back and the caller maps the folio itself.
 */
static unsigned int clear_map_segments(struct device *dev, size_t quantum,
				       struct folio *folio, size_t size,
				       struct mm_offload_dma_map **ents,
				       dma_addr_t *segdma)
{
	unsigned int nsegs = DIV_ROUND_UP(size, quantum);
	unsigned int i;

	if (nsegs > DCBM_CLEAR_MAX_SEGS)
		return 0;

	for (i = 0; i < nsegs; i++) {
		size_t soff = i * quantum;
		size_t slen = min(quantum, size - soff);

		ents[i] = mm_offload_dma_map_get(dev, folio, soff, slen, &segdma[i]);
		if (!ents[i])
			break;
		/* The device is about to write it; it owns it until then. */
		dma_sync_single_for_device(dev, segdma[i], slen,
					   DMA_FROM_DEVICE);
	}
	if (i == nsegs)
		return nsegs;

	while (i--)
		mm_offload_dma_map_put(ents[i]);
	return 0;
}

/**
 * folio_clear_dma - zero a folio via DMA memset
 * @folio: folio to zero, not yet visible to anyone
 * @addr_hint: user address expected to be touched first, or 0
 *
 * The folio is mapped once against one device and its fills spread
 * over that device's channels. With cpu_warm the pages around
 * @addr_hint are cleared on the CPU concurrently, so the first touch
 * after the fault hits cache.
 *
 * Return: 0 on success, negative errno on failure (the caller clears
 * on the CPU).
 */
static int folio_clear_dma(struct folio *folio, unsigned long addr_hint)
{
	const size_t size = folio_size(folio);
	const long nr_pages = folio_nr_pages(folio);
	const unsigned long base_addr = ALIGN_DOWN(addr_hint, size);
	unsigned long flags = DMA_CTRL_ACK | DMA_PREP_INTERRUPT;
	struct mm_offload_dma_map *ents[DCBM_CLEAR_MAX_SEGS];
	dma_addr_t segdma[DCBM_CLEAR_MAX_SEGS];
	struct dma_work work = {};
	struct mm_offload_dma_group *grp;
	unsigned long chan_mask;
	dma_addr_t dma_base = 0;
	unsigned int nsegs = 0;
	size_t quantum = 0;
	long ws = 0, we = -1, pg;
	unsigned int i;
	int ret;

	if (size < READ_ONCE(min_clear_bytes))
		return -ENODEV;

	/*
	 * On a fully busy machine the cycles an offloaded clear frees
	 * cannot run anything, while the interrupts and context
	 * switches still cost; clear synchronously instead.
	 */
	if (READ_ONCE(util_gate) && mm_offload_cpus_saturated()) {
		atomic_long_inc(&folios_gated);
		return -EBUSY;
	}

	chan_mask = mm_offload_dma_claim(
			clamp_t(size_t, size / DCBM_CLEAR_CHUNK_BYTES, 1,
				READ_ONCE(nr_dma_channels)), folio_nid(folio), &grp);
	if (!chan_mask)
		return -EBUSY;

	i = find_first_bit(&chan_mask, MM_OFFLOAD_DMA_MAX_CHANNELS);
	if (!dma_has_cap(DMA_MEMSET, mm_offload_dma_chan(i)->device->cap_mask)) {
		ret = -EOPNOTSUPP;
		goto out_release;
	}

	work.dev = grp->dev;

	/*
	 * A cleared folio is usually cleared again - the pool it came from
	 * hands the same pages back - so take the standing mappings if the
	 * cache will give them, and map the folio for this clear alone if
	 * it will not.
	 */
	if (READ_ONCE(map_cache_clear))
		quantum = mm_offload_dma_map_quantum(work.dev);
	if (quantum)
		nsegs = clear_map_segments(work.dev, quantum, folio, size, ents,
					   segdma);
	if (!nsegs) {
		dma_base = dma_map_page_attrs(work.dev, folio_page(folio, 0), 0,
					      size, DMA_FROM_DEVICE, 0);
		if (dma_mapping_error(work.dev, dma_base)) {
			ret = -EIO;
			goto out_release;
		}
	}

	dma_work_init(&work);

	if (READ_ONCE(cache_ctrl))
		flags |= DMA_PREP_CACHE_CONTROL;

	/*
	 * Leave the faulting neighbourhood to the CPU: it overlaps with
	 * the fills and its cachelines stay hot for the first touch.
	 */
	if (READ_ONCE(cpu_warm) && addr_hint) {
		long fault_idx = (addr_hint - base_addr) >> PAGE_SHIFT;

		ws = max(fault_idx - DCBM_WARM_RADIUS, 0L);
		we = min(fault_idx + DCBM_WARM_RADIUS, nr_pages - 1);
	}

	ret = 0;
	if (ws > 0)
		ret = clear_folio_range(&work, chan_mask, dma_base,
					nsegs ? segdma : NULL, quantum,
					0, ws * PAGE_SIZE, flags);
	if (!ret && we < nr_pages - 1)
		ret = clear_folio_range(&work, chan_mask, dma_base,
					nsegs ? segdma : NULL, quantum,
					(we + 1) * PAGE_SIZE,
					(nr_pages - 1 - we) * PAGE_SIZE, flags);

	if (!ret)
		for (pg = ws; pg <= we; pg++)
			clear_user_highpage(folio_page(folio, pg),
					    base_addr + pg * PAGE_SIZE);

	/*
	 * Whatever was queued runs out before the folio is unmapped and
	 * handed back for CPU clearing; see dma_work_wait().
	 */
	dma_work_done_submitting(&work);
	work.submitted = true;
	ret = dma_work_wait(&work) ? : ret;

	if (nsegs) {
		for (i = 0; i < nsegs; i++) {
			size_t soff = i * quantum;

			/* The device has written it; hand it back. */
			dma_sync_single_for_cpu(work.dev, segdma[i],
						min(quantum, size - soff),
						DMA_FROM_DEVICE);
			mm_offload_dma_map_put(ents[i]);
		}
	} else {
		dma_unmap_page_attrs(work.dev, dma_base, size,
				     DMA_FROM_DEVICE, 0);
	}
	if (ret)
		goto out_release;

	mm_offload_dma_release(chan_mask);
	atomic_long_inc(&folios_cleared);
	return 0;

out_release:
	mm_offload_dma_release(chan_mask);
	/* No channel budget is a refusal (counted in admit_failures), not a failure. */
	if (ret != -EBUSY) {
		atomic_long_inc(&clear_failures);
		pr_warn_ratelimited("dcbm: DMA clear failed (%d), falling back to CPU\n",
				    ret);
	}
	return ret;
}

static const struct mm_offload_provider dma_migrator = {
	.name = "DCBM",
	.copy_folios = folios_copy_dma,
	.copy_folio_pairs = folio_pairs_copy_dma,
	.clear_folio = folio_clear_dma,
	.owner = THIS_MODULE,
};

static unsigned long dcbm_reason_mask = MIGRATE_OFFLOAD_REASONS_ALLOWED;

/* offloading: enable/disable DMA migration offload */
static int offloading_param_set(const char *val, const struct kernel_param *kp)
{
	bool enable;
	int ret;

	ret = kstrtobool(val, &enable);
	if (ret)
		return ret;

	mutex_lock(&dcbm_mutex);
	if (enable == offloading_enabled) {
		mutex_unlock(&dcbm_mutex);
		return 0;
	}
	if (enable) {
		ret = mm_offload_dma_pool_get();
		if (ret) {
			mutex_unlock(&dcbm_mutex);
			return ret;
		}
		ret = mm_offload_register(&dma_migrator,
					       READ_ONCE(dcbm_reason_mask));
		if (ret) {
			mm_offload_dma_pool_put();
			mutex_unlock(&dcbm_mutex);
			return ret;
		}
		WRITE_ONCE(offloading_enabled, true);
	} else {
		mm_offload_unregister(&dma_migrator);
		mm_offload_dma_pool_put();
		WRITE_ONCE(offloading_enabled, false);
	}
	mutex_unlock(&dcbm_mutex);
	return 0;
}

static int offloading_param_get(char *buffer, const struct kernel_param *kp)
{
	return sysfs_emit(buffer, "%d\n", READ_ONCE(offloading_enabled));
}

static const struct kernel_param_ops offloading_param_ops = {
	.set = offloading_param_set,
	.get = offloading_param_get,
};
module_param_cb(offloading, &offloading_param_ops, NULL, 0644);
MODULE_PARM_DESC(offloading, "Enable DMA migration offload (0/1)");

/* nr_dma_chan: max DMA channels to use per batch */
static int nr_dma_chan_param_set(const char *val, const struct kernel_param *kp)
{
	unsigned int new_val;
	int ret;

	ret = kstrtouint(val, 0, &new_val);
	if (ret)
		return ret;
	if (new_val < 1 || new_val > MM_OFFLOAD_DMA_MAX_CHANNELS)
		return -EINVAL;

	/* Bounds the next claim only; the pool's size does not depend on it. */
	WRITE_ONCE(nr_dma_channels, new_val);
	return 0;
}

static int nr_dma_chan_param_get(char *buffer, const struct kernel_param *kp)
{
	return sysfs_emit(buffer, "%u\n", READ_ONCE(nr_dma_channels));
}

static const struct kernel_param_ops nr_dma_chan_param_ops = {
	.set = nr_dma_chan_param_set,
	.get = nr_dma_chan_param_get,
};
module_param_cb(nr_dma_chan, &nr_dma_chan_param_ops, NULL, 0644);
MODULE_PARM_DESC(nr_dma_chan, "DMA channels to acquire when enabling (1..16)");

/* reason_mask: set of MR_* reasons this migrator handles */
static int reason_mask_param_set(const char *val, const struct kernel_param *kp)
{
	unsigned long mask;
	int ret;

	ret = mm_offload_reason_mask_parse(val, &mask);
	if (ret)
		return ret;

	mutex_lock(&dcbm_mutex);
	WRITE_ONCE(dcbm_reason_mask, mask);
	if (offloading_enabled)
		mm_offload_set_migrate_reason_mask(&dma_migrator, mask);
	mutex_unlock(&dcbm_mutex);
	return 0;
}

static int reason_mask_param_get(char *buffer, const struct kernel_param *kp)
{
	return mm_offload_reason_mask_format(buffer, READ_ONCE(dcbm_reason_mask));
}

static const struct kernel_param_ops reason_mask_param_ops = {
	.set = reason_mask_param_set,
	.get = reason_mask_param_get,
};
module_param_cb(reason_mask, &reason_mask_param_ops, NULL, 0644);
MODULE_PARM_DESC(reason_mask,
		 "Reasons to offload: comma-separated names (e.g. compaction,demotion), 'all', 'none', or a raw hex mask");

/* folios_migrated / folios_failures: counters; any write resets to 0 */
static int folios_migrated_param_set(const char *val, const struct kernel_param *kp)
{
	atomic_long_set(&folios_migrated, 0);
	return 0;
}

static int folios_migrated_param_get(char *buffer, const struct kernel_param *kp)
{
	return sysfs_emit(buffer, "%ld\n", atomic_long_read(&folios_migrated));
}

static const struct kernel_param_ops folios_migrated_param_ops = {
	.set = folios_migrated_param_set,
	.get = folios_migrated_param_get,
};
module_param_cb(folios_migrated, &folios_migrated_param_ops, NULL, 0644);
MODULE_PARM_DESC(folios_migrated, "Folios DMA-copied (write to reset)");

static int folios_failures_param_set(const char *val, const struct kernel_param *kp)
{
	atomic_long_set(&folios_failures, 0);
	return 0;
}

static int folios_failures_param_get(char *buffer, const struct kernel_param *kp)
{
	return sysfs_emit(buffer, "%ld\n", atomic_long_read(&folios_failures));
}

static const struct kernel_param_ops folios_failures_param_ops = {
	.set = folios_failures_param_set,
	.get = folios_failures_param_get,
};
module_param_cb(folios_failures, &folios_failures_param_ops, NULL, 0644);
MODULE_PARM_DESC(folios_failures, "DMA-copy failure count (write to reset)");

static int batches_refused_param_set(const char *val, const struct kernel_param *kp)
{
	atomic_long_set(&batches_refused, 0);
	return 0;
}

static int batches_refused_param_get(char *buffer, const struct kernel_param *kp)
{
	return sysfs_emit(buffer, "%ld\n", atomic_long_read(&batches_refused));
}

static const struct kernel_param_ops batches_refused_param_ops = {
	.set = batches_refused_param_set,
	.get = batches_refused_param_get,
};
module_param_cb(batches_refused, &batches_refused_param_ops, NULL, 0644);
MODULE_PARM_DESC(batches_refused, "Batches refused because all channels were busy (write to reset)");

static int admit_failures_param_set(const char *val, const struct kernel_param *kp)
{
	atomic_long_set(&admit_failures, 0);
	return 0;
}

static int admit_failures_param_get(char *buffer, const struct kernel_param *kp)
{
	return sysfs_emit(buffer, "%ld\n", atomic_long_read(&admit_failures));
}

static const struct kernel_param_ops admit_failures_param_ops = {
	.set = admit_failures_param_set,
	.get = admit_failures_param_get,
};
module_param_cb(admit_failures, &admit_failures_param_ops, NULL, 0644);
MODULE_PARM_DESC(admit_failures, "Submissions handed to the CPU because a channel stayed over budget (write to reset)");

static int folios_cleared_param_set(const char *val, const struct kernel_param *kp)
{
	atomic_long_set(&folios_cleared, 0);
	return 0;
}

static int folios_cleared_param_get(char *buffer, const struct kernel_param *kp)
{
	return sysfs_emit(buffer, "%ld\n", atomic_long_read(&folios_cleared));
}

static const struct kernel_param_ops folios_cleared_param_ops = {
	.set = folios_cleared_param_set,
	.get = folios_cleared_param_get,
};
module_param_cb(folios_cleared, &folios_cleared_param_ops, NULL, 0644);
MODULE_PARM_DESC(folios_cleared, "Folios DMA-cleared (write to reset)");

static int folios_gated_param_set(const char *val, const struct kernel_param *kp)
{
	atomic_long_set(&folios_gated, 0);
	return 0;
}

static int folios_gated_param_get(char *buffer, const struct kernel_param *kp)
{
	return sysfs_emit(buffer, "%ld\n", atomic_long_read(&folios_gated));
}

static const struct kernel_param_ops folios_gated_param_ops = {
	.set = folios_gated_param_set,
	.get = folios_gated_param_get,
};
module_param_cb(folios_gated, &folios_gated_param_ops, NULL, 0644);
MODULE_PARM_DESC(folios_gated, "Clears refused by the CPU saturation gate (write to reset)");

static int clear_failures_param_set(const char *val, const struct kernel_param *kp)
{
	atomic_long_set(&clear_failures, 0);
	return 0;
}

static int clear_failures_param_get(char *buffer, const struct kernel_param *kp)
{
	return sysfs_emit(buffer, "%ld\n", atomic_long_read(&clear_failures));
}

static const struct kernel_param_ops clear_failures_param_ops = {
	.set = clear_failures_param_set,
	.get = clear_failures_param_get,
};
module_param_cb(clear_failures, &clear_failures_param_ops, NULL, 0644);
MODULE_PARM_DESC(clear_failures, "DMA-clear failure count (write to reset)");

module_param(min_clear_bytes, ulong, 0644);
MODULE_PARM_DESC(min_clear_bytes, "Smallest folio to clear by DMA (bytes)");

module_param(cpu_warm, bool, 0644);
MODULE_PARM_DESC(cpu_warm, "Clear the faulting neighbourhood on the CPU while DMA clears the rest");

module_param(util_gate, bool, 0644);
MODULE_PARM_DESC(util_gate, "Refuse clears while every CPU is busy");

module_param(cache_ctrl, bool, 0644);
MODULE_PARM_DESC(cache_ctrl, "Request cache-allocating writes (DMA_PREP_CACHE_CONTROL)");

module_param(max_inflight, uint, 0644);
MODULE_PARM_DESC(max_inflight, "Descriptors in flight per channel before waiting for completions");

module_param(sg_elems, uint, 0644);
MODULE_PARM_DESC(sg_elems, "Folios per scatter-gather transaction on DMA_MEMCPY_SG providers");
module_param(map_cache_clear, bool, 0644);
MODULE_PARM_DESC(map_cache_clear,
		 "Let folio clears use the standing mapping cache too (they stream over fresh pages and can displace the migration set)");

module_param(min_chunk_bytes, ulong, 0644);
MODULE_PARM_DESC(min_chunk_bytes,
		 "Smallest share of a copy to give one channel (bytes)");

module_param(max_chunk_bytes, ulong, 0644);
MODULE_PARM_DESC(max_chunk_bytes, "Largest single copy descriptor (bytes)");

module_param(min_bytes, ulong, 0644);
MODULE_PARM_DESC(min_bytes, "Smallest copy worth a descriptor at all (bytes)");

static int __init dcbm_init(void)
{
	pr_info("dcbm: DMA Core Batch Migrator initialized\n");
	return 0;
}

static void __exit dcbm_exit(void)
{
	mutex_lock(&dcbm_mutex);
	if (offloading_enabled) {
		mm_offload_unregister(&dma_migrator);
		mm_offload_dma_pool_put();
		offloading_enabled = false;
	}
	mutex_unlock(&dcbm_mutex);

	pr_info("dcbm: DMA Core Batch Migrator unloaded\n");
}

module_init(dcbm_init);
module_exit(dcbm_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Shivank Garg");
MODULE_DESCRIPTION("DMA Core Batch Migrator");
