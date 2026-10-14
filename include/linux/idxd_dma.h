/* SPDX-License-Identifier: GPL-2.0 */
/*
 * For users of Intel DSA channels through the dmaengine API that want the
 * device's own (perfmon) view of their traffic: which work queue a channel
 * submits to, so a counter can be filtered to it, and the device's PMU.
 */
#ifndef _LINUX_IDXD_DMA_H
#define _LINUX_IDXD_DMA_H

#include <linux/errno.h>

struct dma_chan;

#if IS_ENABLED(CONFIG_INTEL_IDXD)
/* The WQ index of @chan on its device, or -ENODEV if not an idxd channel. */
int idxd_dma_chan_wq_id(struct dma_chan *chan);
/* The perf PMU type of @chan's device, or -ENODEV without a PMU. */
int idxd_dma_chan_pmu_type(struct dma_chan *chan);
#else
static inline int idxd_dma_chan_wq_id(struct dma_chan *chan) { return -ENODEV; }
static inline int idxd_dma_chan_pmu_type(struct dma_chan *chan) { return -ENODEV; }
#endif

#endif /* _LINUX_IDXD_DMA_H */
