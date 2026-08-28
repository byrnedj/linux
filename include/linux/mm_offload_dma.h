/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Shared DMAEngine channel pool for mm_offload providers.
 *
 * Providers (migration/clear, user copy, ...) are separate modules with
 * separate enable knobs, but they compete for the same kernel work
 * queues. The pool acquires the channels once, refcounted across
 * providers, groups them by DMA device and NUMA node, and hands them
 * out for the duration of one operation through per-channel trylocks.
 */
#ifndef _LINUX_MM_OFFLOAD_DMA_H
#define _LINUX_MM_OFFLOAD_DMA_H

#include <linux/types.h>

struct dma_chan;
struct device;

#define MM_OFFLOAD_DMA_MAX_CHANNELS	16

/*
 * Channels grouped by DMA device. One operation maps its pages against
 * a single device and uses only that group's channels; concurrent
 * operations rotate over the groups so every device contributes.
 */
struct mm_offload_dma_group {
	struct device *dev;
	int node;
	unsigned int first;
	unsigned int nr;
};

int mm_offload_dma_pool_get(unsigned int nr_want);
void mm_offload_dma_pool_put(void);
unsigned int mm_offload_dma_nr_channels(void);
struct dma_chan *mm_offload_dma_chan(unsigned int idx);
unsigned long mm_offload_dma_claim(unsigned int want, int nid,
				   struct mm_offload_dma_group **grpp);
void mm_offload_dma_release(unsigned long mask);

#endif /* _LINUX_MM_OFFLOAD_DMA_H */
