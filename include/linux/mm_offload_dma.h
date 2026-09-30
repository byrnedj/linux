/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Shared DMAEngine channel pool for mm_offload providers.
 *
 * Providers (migration/clear, user copy, ...) are separate modules with
 * separate enable knobs, but they compete for the same kernel work
 * queues. The pool acquires every kernel channel once, refcounted
 * across providers, groups them by DMA device and NUMA node, and hands
 * them out for the duration of one operation. It holds them all so
 * that an operation can always be served from the devices of its own
 * node; how many an operation takes is the provider's business.
 *
 * Channels are shared: several operations submit to the same channel at
 * once and a per-channel in-flight budget, taken with mm_offload_dma_admit()
 * before each submission and returned with mm_offload_dma_complete() from
 * the completion callback (mm_offload_dma_retire() for a descriptor that
 * was never queued), bounds the queue ahead of every descriptor
 * (the design io_uring's DMA path uses). With shared=N at module load the
 * pool falls back to handing each channel to one operation under a
 * trylock, which sends concurrent operations to the CPU instead.
 */
#ifndef _LINUX_MM_OFFLOAD_DMA_H
#define _LINUX_MM_OFFLOAD_DMA_H

#include <linux/types.h>

struct dma_chan;
struct device;

/*
 * Channels the pool can hold. The channels are taken in the order the
 * dmaengine core hands them out, which follows device enumeration, so a
 * cap below the number of configured work queues leaves the devices
 * enumerated last - typically a whole socket - out of the pool. Eight
 * DSA devices with four kernel work queues each fit.
 */
#define MM_OFFLOAD_DMA_MAX_CHANNELS	32

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

int mm_offload_dma_pool_get(void);
void mm_offload_dma_pool_put(void);
unsigned int mm_offload_dma_nr_channels(void);
struct dma_chan *mm_offload_dma_chan(unsigned int idx);
unsigned long mm_offload_dma_claim(unsigned int want, int nid,
				   struct mm_offload_dma_group **grpp);
unsigned long mm_offload_dma_claim_spread(unsigned int want, int nid);
struct device *mm_offload_dma_chan_dev(unsigned int idx);
void mm_offload_dma_release(unsigned long mask);

#endif /* _LINUX_MM_OFFLOAD_DMA_H */
