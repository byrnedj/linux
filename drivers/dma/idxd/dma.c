// SPDX-License-Identifier: GPL-2.0
/* Copyright(c) 2019 Intel Corporation. All rights rsvd. */
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/device.h>
#include <linux/io-64-nonatomic-lo-hi.h>
#include <linux/dmaengine.h>
#include <uapi/linux/idxd.h>
#include "../dmaengine.h"
#include "registers.h"
#include "idxd.h"

#define DMA_COOKIE_BITS (sizeof(dma_cookie_t) * 8)
/*
 * The descriptor id takes the lower 16 bits of the cookie.
 */
#define DESC_ID_BITS 16
#define DESC_ID_MASK ((1 << DESC_ID_BITS) - 1)
/*
 * The generation is stored in the upper half of the cookie. Since
 * dma_cookie_t is signed, we leave the upper-most bit for the sign.
 */
#define DESC_GEN_MAX ((1 << (DMA_COOKIE_BITS - DESC_ID_BITS - 1)) - 1)

static inline struct idxd_wq *to_idxd_wq(struct dma_chan *c)
{
	struct idxd_dma_chan *idxd_chan;

	idxd_chan = container_of(c, struct idxd_dma_chan, chan);
	return idxd_chan->wq;
}

void idxd_dma_complete_txd(struct idxd_desc *desc,
			   enum idxd_complete_type comp_type,
			   bool free_desc, void *ctx, u32 *status)
{
	struct idxd_device *idxd = desc->wq->idxd;
	struct dma_async_tx_descriptor *tx;
	struct dmaengine_result res;
	int complete = 1;

	if (desc->completion->status == DSA_COMP_SUCCESS) {
		res.result = DMA_TRANS_NOERROR;
	} else if (desc->completion->status) {
		if (idxd->request_int_handles && comp_type != IDXD_COMPLETE_ABORT &&
		    desc->completion->status == DSA_COMP_INT_HANDLE_INVAL &&
		    idxd_queue_int_handle_resubmit(desc))
			return;
		res.result = DMA_TRANS_WRITE_FAILED;
	} else if (comp_type == IDXD_COMPLETE_ABORT) {
		res.result = DMA_TRANS_ABORTED;
	} else {
		complete = 0;
	}

	tx = &desc->txd;
	if (complete && tx->cookie) {
		dma_cookie_complete(tx);
		dma_descriptor_unmap(tx);
		dmaengine_desc_get_callback_invoke(tx, &res);
		tx->callback = NULL;
		tx->callback_result = NULL;
	}

	if (free_desc)
		idxd_free_desc(desc->wq, desc);
}

static void op_flag_setup(unsigned long flags, u32 *desc_flags)
{
	*desc_flags = IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR;
	if (flags & DMA_PREP_INTERRUPT)
		*desc_flags |= IDXD_OP_FLAG_RCI;
}

/*
 * Cache-Control is valid only on data-mover descriptors and only when
 * GENCAP reports Cache Control Support for memory destinations. On a
 * device without the capability the flag encoding is reserved and the
 * descriptor fails with DSA_COMP_INVALID_FLAGS, so honour the advisory
 * prep flag by dropping it there.
 */
static inline u32 idxd_cc_flag(struct idxd_wq *wq, unsigned long flags)
{
	return (flags & DMA_PREP_CACHE_CONTROL) &&
	       wq->idxd->hw.gen_cap.cache_control_mem ? IDXD_OP_FLAG_CC : 0;
}

static inline void idxd_prep_desc_common(struct idxd_wq *wq,
					 struct dsa_hw_desc *hw, char opcode,
					 u64 addr_f1, u64 addr_f2, u64 len,
					 u64 compl, u32 flags)
{
	hw->flags = flags;
	hw->opcode = opcode;
	hw->src_addr = addr_f1;
	hw->dst_addr = addr_f2;
	hw->xfer_size = len;
	/*
	 * For dedicated WQ, this field is ignored and HW will use the WQCFG.priv
	 * field instead. This field should be set to 0 for kernel descriptors
	 * since kernel DMA on VT-d supports "user" privilege only.
	 */
	hw->priv = 0;
	hw->completion_addr = compl;
}

static struct dma_async_tx_descriptor *
idxd_dma_prep_interrupt(struct dma_chan *c, unsigned long flags)
{
	struct idxd_wq *wq = to_idxd_wq(c);
	u32 desc_flags;
	struct idxd_desc *desc;

	if (wq->state != IDXD_WQ_ENABLED)
		return NULL;

	op_flag_setup(flags, &desc_flags);
	desc = idxd_alloc_desc(wq, IDXD_OP_NONBLOCK);
	if (IS_ERR(desc))
		return NULL;

	idxd_prep_desc_common(wq, desc->hw, DSA_OPCODE_NOOP,
			      0, 0, 0, desc->compl_dma, desc_flags);
	desc->txd.flags = flags;
	return &desc->txd;
}

static struct dma_async_tx_descriptor *
idxd_dma_submit_memcpy(struct dma_chan *c, dma_addr_t dma_dest,
		       dma_addr_t dma_src, size_t len, unsigned long flags)
{
	struct idxd_wq *wq = to_idxd_wq(c);
	u32 desc_flags;
	struct idxd_device *idxd = wq->idxd;
	struct idxd_desc *desc;

	if (wq->state != IDXD_WQ_ENABLED)
		return NULL;

	if (len > idxd->max_xfer_bytes)
		return NULL;

	op_flag_setup(flags, &desc_flags);
	desc_flags |= idxd_cc_flag(wq, flags);
	desc = idxd_alloc_desc(wq, IDXD_OP_NONBLOCK);
	if (IS_ERR(desc))
		return NULL;

	idxd_prep_desc_common(wq, desc->hw, DSA_OPCODE_MEMMOVE,
			      dma_src, dma_dest, len, desc->compl_dma,
			      desc_flags);

	desc->txd.flags = flags;

	return &desc->txd;
}

static inline int fetch_sg_and_pos(struct scatterlist **sg, size_t *remain,
				   unsigned int len)
{
	struct scatterlist *next = *sg;
	int count = 0;

	*remain -= len;

	while (*remain == 0 && next && !sg_is_last(next)) {
		next = sg_next(next);
		*remain = sg_dma_len(next);
		count++;
	}

	*sg = next;

	return count;
}

/*
 * idxd_dma_prep_memcpy_sg - prepare a memcpy_sg transaction
 *
 * @chan: DMA channel
 * @dst_sg: Destination scatter list
 * @dst_nents: Number of entries in destination scatter list
 * @src_sg: Source scatter list
 * @src_nents: Number of entries in source scatter list
 * @flags: DMA transaction flags
 *
 * Return: Async transaction descriptor on success and NULL on failure.
 *
 * The scatter lists are folded into one hardware BATCH descriptor of
 * DSA_OPCODE_MEMMOVE elements. A single-element copy degrades to a
 * plain MEMMOVE descriptor.
 */
static struct dma_async_tx_descriptor *
idxd_dma_prep_memcpy_sg(struct dma_chan *chan,
			struct scatterlist *dst_sg, unsigned int dst_nents,
			struct scatterlist *src_sg, unsigned int src_nents,
			unsigned long flags)
{
	struct idxd_wq *wq = to_idxd_wq(chan);
	struct idxd_desc *desc;
	struct idxd_batch *batch;
	dma_addr_t dma_dst, dma_src;
	size_t dst_avail, src_avail, len;
	u32 desc_flags;
	int i;

	if (unlikely(!dst_sg || !src_sg))
		return NULL;
	if (unlikely(dst_nents == 0 || src_nents == 0))
		return NULL;

	if (wq->state != IDXD_WQ_ENABLED)
		return NULL;

	if (min(dst_nents, src_nents) > wq->max_batch_size)
		return NULL;

	dst_avail = sg_dma_len(dst_sg);
	src_avail = sg_dma_len(src_sg);

	if (dst_nents == 1 && src_nents == 1) {
		if (unlikely(dst_avail != src_avail))
			return NULL;

		return idxd_dma_submit_memcpy(chan, sg_dma_address(dst_sg),
				sg_dma_address(src_sg), dst_avail, flags);
	}

	desc = idxd_alloc_desc(wq, IDXD_OP_NONBLOCK);
	if (IS_ERR(desc))
		return NULL;

	/*
	 * Fill the batch with DSA_OPCODE_MEMMOVE elements until
	 * max_batch_size or a scatter list is consumed.
	 */
	batch = desc->batch;
	/*
	 * Bound the fill on the alloc-time capacity. wq->max_batch_size
	 * can be reset to the default by device disable or HALT recovery
	 * while a prep is in flight, and a larger live value would let
	 * the loop write past the coherent descriptor array.
	 */
	for (i = 0; i < batch->max; i++) {
		dma_dst = sg_dma_address(dst_sg) + sg_dma_len(dst_sg) -
			dst_avail;
		dma_src = sg_dma_address(src_sg) + sg_dma_len(src_sg) -
			src_avail;

		len = min_t(size_t, dst_avail, src_avail);
		len = min_t(size_t, len, wq->idxd->max_xfer_bytes);

		memset(batch->descs + i, 0, sizeof(struct dsa_hw_desc));
		idxd_prep_desc_common(wq, batch->descs + i, DSA_OPCODE_MEMMOVE,
				dma_src, dma_dst, len, 0,
				idxd_cc_flag(wq, flags));
		batch->num++;

		dst_nents -= fetch_sg_and_pos(&dst_sg, &dst_avail, len);
		src_nents -= fetch_sg_and_pos(&src_sg, &src_avail, len);

		/* Stop when either scatter list is consumed. */
		if (!dst_nents || !src_nents ||
				!min_t(size_t, dst_avail, src_avail)) {
			break;
		}
	}

	/*
	 * The element count can exceed min(dst_nents, src_nents) when
	 * boundaries interleave, so the admission check does not bound
	 * it. Submitting a capacity-exhausted batch would complete
	 * successfully while covering only a prefix of the transfer.
	 * Refuse instead and let the caller fall back.
	 */
	if (dst_nents && src_nents &&
	    min_t(size_t, dst_avail, src_avail)) {
		idxd_free_desc(wq, desc);
		return NULL;
	}

	op_flag_setup(flags, &desc_flags);
	if (batch->num == 1) {
		/*
		 * A one element batch is an invalid Descriptor Count on
		 * hardware without Batch1 support, and GENCAP has no bit
		 * to probe for it. Dispatch the single element as a plain
		 * MEMMOVE instead. Cache-Control is valid there, so it
		 * rides along.
		 */
		idxd_prep_desc_common(wq, desc->hw, DSA_OPCODE_MEMMOVE,
				batch->descs[0].src_addr,
				batch->descs[0].dst_addr,
				batch->descs[0].xfer_size,
				desc->compl_dma,
				desc_flags | idxd_cc_flag(wq, flags));
	} else {
		/*
		 * The BATCH descriptor itself moves no data, so it never
		 * carries Cache-Control; the elements above do.
		 */
		idxd_prep_desc_common(wq, desc->hw, DSA_OPCODE_BATCH,
				batch->dma_descs, 0, batch->num,
				desc->compl_dma, desc_flags);
	}

	desc->txd.flags = flags;
	return &desc->txd;
}

static int idxd_dma_alloc_chan_resources(struct dma_chan *chan)
{
	struct idxd_wq *wq = to_idxd_wq(chan);
	struct device *dev = &wq->idxd->pdev->dev;

	idxd_wq_get(wq);
	dev_dbg(dev, "%s: client_count: %d\n", __func__,
		idxd_wq_refcount(wq));
	return 0;
}

static void idxd_dma_free_chan_resources(struct dma_chan *chan)
{
	struct idxd_wq *wq = to_idxd_wq(chan);
	struct device *dev = &wq->idxd->pdev->dev;

	idxd_wq_put(wq);
	dev_dbg(dev, "%s: client_count: %d\n", __func__,
		idxd_wq_refcount(wq));
}

static enum dma_status idxd_dma_tx_status(struct dma_chan *dma_chan,
					  dma_cookie_t cookie,
					  struct dma_tx_state *txstate)
{
	struct idxd_wq *wq;
	struct idxd_desc *desc;
	u32 idx;
	u8 status;

	if (txstate)
		memset(txstate, 0, sizeof(*txstate));

	if (dma_submit_error(cookie))
		return DMA_ERROR;

	wq = to_idxd_wq(dma_chan);

	idx = cookie & DESC_ID_MASK;
	if (idx >= wq->num_descs)
		return DMA_ERROR;

	desc = wq->descs[idx];

	if (desc->txd.cookie != cookie) {
		/*
		 * The cookie belongs to an old transaction that has
		 * already completed and been recycled.
		 */
		return DMA_COMPLETE;
	}

	/*
	 * Descriptors with a completion interrupt armed (RCI) are
	 * completed and recycled by the IRQ handler. Reading their
	 * completion record here would race with the handler. Since
	 * the handler invalidates the cookie on completion, a matching
	 * cookie means the command is still in progress.
	 */
	if (desc->hw->flags & IDXD_OP_FLAG_RCI)
		return DMA_IN_PROGRESS;

	status = desc->completion->status & DSA_COMP_STATUS_MASK;

	if (status) {
		/* Capture completion record fields before desc is freed below. */
		u8 fault_info = desc->completion->fault_info;
		u64 fault_addr = desc->completion->fault_addr;

		/*
		 * Check against the original status because ABORT is
		 * software defined as 0xff, which DSA_COMP_STATUS_MASK
		 * can mask out.
		 */
		if (unlikely(desc->completion->status == IDXD_COMP_DESC_ABORT))
			idxd_dma_complete_txd(desc, IDXD_COMPLETE_ABORT, true, NULL, NULL);
		else
			idxd_dma_complete_txd(desc, IDXD_COMPLETE_NORMAL, true, NULL, NULL);

		pr_debug("dsa completion: status=0x%02x fault_info=0x%02x fault_addr=0x%llx\n",
			 status, fault_info, fault_addr);

		if (status == DSA_COMP_SUCCESS)
			return DMA_COMPLETE;

		return DMA_ERROR;
	}

	return DMA_IN_PROGRESS;
}

/*
 * issue_pending() does not need to do anything since tx_submit() does the job
 * already.
 */
static void idxd_dma_issue_pending(struct dma_chan *dma_chan)
{
}

static dma_cookie_t idxd_dma_tx_submit(struct dma_async_tx_descriptor *tx)
{
	struct dma_chan *c = tx->chan;
	struct idxd_wq *wq = to_idxd_wq(c);
	dma_cookie_t cookie;
	int rc;
	struct idxd_desc *desc = container_of(tx, struct idxd_desc, txd);

	cookie = (desc->gen << DESC_ID_BITS) | (desc->id & DESC_ID_MASK);

	if (desc->gen == DESC_GEN_MAX)
		desc->gen = 1;
	else
		desc->gen++;

	tx->cookie = cookie;

	rc = idxd_submit_desc(wq, desc);
	if (rc < 0) {
		idxd_free_desc(wq, desc);
		return rc;
	}

	return cookie;
}

static void idxd_dma_release(struct dma_device *device)
{
	struct idxd_dma_dev *idxd_dma = container_of(device, struct idxd_dma_dev, dma);

	kfree(idxd_dma);
}

static int idxd_dma_terminate_all(struct dma_chan *c)
{
	struct idxd_wq *wq = to_idxd_wq(c);

	idxd_wq_flush_descs(wq);

	return 0;
}

static void idxd_dma_synchronize(struct dma_chan *c)
{
	struct idxd_wq *wq = to_idxd_wq(c);

	idxd_wq_drain(wq);
}

int idxd_register_dma_device(struct idxd_device *idxd)
{
	struct idxd_dma_dev *idxd_dma;
	struct dma_device *dma;
	struct device *dev = &idxd->pdev->dev;
	int rc;

	idxd_dma = kzalloc_node(sizeof(*idxd_dma), GFP_KERNEL, dev_to_node(dev));
	if (!idxd_dma)
		return -ENOMEM;

	dma = &idxd_dma->dma;
	INIT_LIST_HEAD(&dma->channels);
	dma->dev = dev;

	dma_cap_set(DMA_INTERRUPT, dma->cap_mask);
	dma_cap_set(DMA_PRIVATE, dma->cap_mask);
	dma->device_release = idxd_dma_release;

	dma->device_prep_dma_interrupt = idxd_dma_prep_interrupt;
	if (idxd->hw.opcap.bits[0] & IDXD_OPCAP_MEMMOVE) {
		dma_cap_set(DMA_MEMCPY, dma->cap_mask);
		dma->device_prep_dma_memcpy = idxd_dma_submit_memcpy;
	}

	if ((idxd->hw.opcap.bits[0] & IDXD_OPCAP_BATCH) &&
	    (idxd->hw.opcap.bits[0] & IDXD_OPCAP_MEMMOVE)) {
		dma_cap_set(DMA_MEMCPY_SG, dma->cap_mask);
		dma->device_prep_dma_memcpy_sg = idxd_dma_prep_memcpy_sg;
	}

	dma->device_tx_status = idxd_dma_tx_status;
	dma->device_issue_pending = idxd_dma_issue_pending;
	dma->device_alloc_chan_resources = idxd_dma_alloc_chan_resources;
	dma->device_free_chan_resources = idxd_dma_free_chan_resources;
	dma->device_terminate_all = idxd_dma_terminate_all;
	dma->device_synchronize = idxd_dma_synchronize;

	rc = dma_async_device_register(dma);
	if (rc < 0) {
		kfree(idxd_dma);
		return rc;
	}

	idxd_dma->idxd = idxd;
	/*
	 * This pointer is protected by the refs taken by the dma_chan. It will remain valid
	 * as long as there are outstanding channels.
	 */
	idxd->idxd_dma = idxd_dma;
	return 0;
}

void idxd_unregister_dma_device(struct idxd_device *idxd)
{
	dma_async_device_unregister(&idxd->idxd_dma->dma);
}

static int idxd_register_dma_channel(struct idxd_wq *wq)
{
	struct idxd_device *idxd = wq->idxd;
	struct dma_device *dma = &idxd->idxd_dma->dma;
	struct device *dev = &idxd->pdev->dev;
	struct idxd_dma_chan *idxd_chan;
	struct dma_chan *chan;
	int rc, i;

	idxd_chan = kzalloc_node(sizeof(*idxd_chan), GFP_KERNEL, dev_to_node(dev));
	if (!idxd_chan)
		return -ENOMEM;

	chan = &idxd_chan->chan;
	chan->device = dma;
	list_add_tail(&chan->device_node, &dma->channels);

	for (i = 0; i < wq->num_descs; i++) {
		struct idxd_desc *desc = wq->descs[i];

		dma_async_tx_descriptor_init(&desc->txd, chan);
		desc->txd.tx_submit = idxd_dma_tx_submit;
	}

	rc = dma_async_device_channel_register(dma, chan, NULL);
	if (rc < 0) {
		kfree(idxd_chan);
		return rc;
	}

	wq->idxd_chan = idxd_chan;
	idxd_chan->wq = wq;
	get_device(wq_confdev(wq));

	return 0;
}


static int idxd_dmaengine_drv_probe(struct idxd_dev *idxd_dev)
{
	struct device *dev = &idxd_dev->conf_dev;
	struct idxd_wq *wq = idxd_dev_to_wq(idxd_dev);
	struct idxd_device *idxd = wq->idxd;
	int rc;

	if (idxd->state != IDXD_DEV_ENABLED)
		return -ENXIO;

	mutex_lock(&wq->wq_lock);
	if (!idxd_wq_driver_name_match(wq, dev)) {
		idxd->cmd_status = IDXD_SCMD_WQ_NO_DRV_NAME;
		rc = -ENODEV;
		goto err;
	}

	wq->type = IDXD_WQT_KERNEL;

	rc = idxd_drv_enable_wq(wq);
	if (rc < 0) {
		dev_dbg(dev, "Enable wq %d failed: %d\n", wq->id, rc);
		rc = -ENXIO;
		goto err;
	}

	rc = idxd_register_dma_channel(wq);
	if (rc < 0) {
		idxd->cmd_status = IDXD_SCMD_DMA_CHAN_ERR;
		dev_dbg(dev, "Failed to register dma channel\n");
		goto err_dma;
	}

	idxd->cmd_status = 0;
	mutex_unlock(&wq->wq_lock);
	return 0;

err_dma:
	idxd_drv_disable_wq(wq);
err:
	wq->type = IDXD_WQT_NONE;
	mutex_unlock(&wq->wq_lock);
	return rc;
}

static void idxd_dmaengine_drv_remove(struct idxd_dev *idxd_dev)
{
	struct idxd_wq *wq = idxd_dev_to_wq(idxd_dev);

	mutex_lock(&wq->wq_lock);
	__idxd_wq_quiesce(wq);
	if (wq->idxd_chan &&
	    dma_async_device_channel_unregister_if_unused(
			&wq->idxd->idxd_dma->dma, &wq->idxd_chan->chan)) {
		/*
		 * Live clients such as io_uring rings hold pointers into
		 * the channel and poll wq->descs. Freeing any of that
		 * state here is a use-after-free. This can happen when a
		 * device HALT's FLR recovery unbinds the driver under
		 * load. The reference check and the channel unregister
		 * run atomically under dma_list_mutex, so a client
		 * acquiring the channel concurrently either makes this
		 * unregister fail or finds the channel already gone.
		 * On failure we leak the wq's dmaengine state instead.
		 * Submissions already fail cleanly through the killed
		 * wq_active percpu ref and clients fall back to CPU
		 * copies.
		 */
		dev_warn(&wq->idxd->pdev->dev,
			 "wq %d unbound with live DMA clients; leaking channel state\n",
			 wq->id);
		mutex_unlock(&wq->wq_lock);
		return;
	}
	if (wq->idxd_chan) {
		kfree(wq->idxd_chan);
		wq->idxd_chan = NULL;
		put_device(wq_confdev(wq));
	}
	idxd_drv_disable_wq(wq);
	mutex_unlock(&wq->wq_lock);
}

static enum idxd_dev_type dev_types[] = {
	IDXD_DEV_WQ,
	IDXD_DEV_NONE,
};

struct idxd_device_driver idxd_dmaengine_drv = {
	.probe = idxd_dmaengine_drv_probe,
	.remove = idxd_dmaengine_drv_remove,
	.desc_complete = idxd_dma_complete_txd,
	.name = "dmaengine",
	.type = dev_types,
};
EXPORT_SYMBOL_GPL(idxd_dmaengine_drv);
