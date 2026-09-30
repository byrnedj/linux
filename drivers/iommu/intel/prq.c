// SPDX-License-Identifier: GPL-2.0-only
/*
 * Copyright (C) 2015 Intel Corporation
 *
 * Originally split from drivers/iommu/intel/svm.c
 */

#include <linux/pci.h>
#include <linux/pci-ats.h>
#include <linux/kernel_stat.h>

#include "iommu.h"
#include "pasid.h"
#include "../iommu-pages.h"
#include "trace.h"

/* Page request queue descriptor */
struct page_req_dsc {
	union {
		struct {
			u64 type:8;
			u64 pasid_present:1;
			u64 rsvd:7;
			u64 rid:16;
			u64 pasid:20;
			u64 exe_req:1;
			u64 pm_req:1;
			u64 rsvd2:10;
		};
		u64 qw_0;
	};
	union {
		struct {
			u64 rd_req:1;
			u64 wr_req:1;
			u64 lpig:1;
			u64 prg_index:9;
			u64 addr:52;
		};
		u64 qw_1;
	};
	u64 qw_2;
	u64 qw_3;
};

/**
 * intel_iommu_drain_pasid_prq - Drain page requests and responses for a pasid
 * @dev: target device
 * @pasid: pasid for draining
 *
 * Drain all pending page requests and responses related to @pasid in both
 * software and hardware. This is supposed to be called after the device
 * driver has stopped DMA, the pasid entry has been cleared, and both IOTLB
 * and DevTLB have been invalidated.
 *
 * It waits until all pending page requests for @pasid in the page fault
 * queue are completed by the prq handling thread. Then follow the steps
 * described in VT-d spec CH7.10 to drain all page requests and page
 * responses pending in the hardware.
 */
void intel_iommu_drain_pasid_prq(struct device *dev, u32 pasid)
{
	struct device_domain_info *info;
	struct dmar_domain *domain;
	struct intel_iommu *iommu;
	struct qi_desc desc[3];
	int head, tail;
	u16 sid, did;

	info = dev_iommu_priv_get(dev);
	if (!info->iopf_refcount)
		return;

	iommu = info->iommu;
	domain = info->domain;
	sid = PCI_DEVID(info->bus, info->devfn);
	did = domain ? domain_id_iommu(domain, iommu) : FLPT_DEFAULT_DID;

	/*
	 * Check and wait until all pending page requests in the queue are
	 * handled by the prq handling thread.
	 */
prq_retry:
	reinit_completion(&iommu->prq_complete);
	tail = readq(iommu->reg + DMAR_PQT_REG) & PRQ_RING_MASK;
	head = readq(iommu->reg + DMAR_PQH_REG) & PRQ_RING_MASK;
	while (head != tail) {
		struct page_req_dsc *req;

		req = &iommu->prq[head / sizeof(*req)];
		if (req->rid != sid ||
		    (req->pasid_present && pasid != req->pasid) ||
		    (!req->pasid_present && pasid != IOMMU_NO_PASID)) {
			head = (head + sizeof(*req)) & PRQ_RING_MASK;
			continue;
		}

		wait_for_completion(&iommu->prq_complete);
		goto prq_retry;
	}

	iopf_queue_flush_dev(dev);

	/*
	 * Perform steps described in VT-d spec CH7.10 to drain page
	 * requests and responses in hardware.
	 */
	memset(desc, 0, sizeof(desc));
	desc[0].qw0 = QI_IWD_STATUS_DATA(QI_DONE) |
			QI_IWD_FENCE |
			QI_IWD_TYPE;
	if (pasid == IOMMU_NO_PASID) {
		qi_desc_iotlb(iommu, did, 0, 0, DMA_TLB_DSI_FLUSH, &desc[1]);
		qi_desc_dev_iotlb(sid, info->pfsid, info->ats_qdep, 0,
				  MAX_AGAW_PFN_WIDTH, &desc[2]);
	} else {
		qi_desc_piotlb_all(did, pasid, &desc[1]);
		qi_desc_dev_iotlb_pasid(sid, info->pfsid, pasid, info->ats_qdep,
					0, MAX_AGAW_PFN_WIDTH, &desc[2]);
	}
qi_retry:
	reinit_completion(&iommu->prq_complete);
	qi_submit_sync(iommu, desc, 3, QI_OPT_WAIT_DRAIN);
	if (readl(iommu->reg + DMAR_PRS_REG) & DMA_PRS_PRO) {
		wait_for_completion(&iommu->prq_complete);
		goto qi_retry;
	}
}

static bool is_canonical_address(u64 addr)
{
	int shift = 64 - (__VIRTUAL_MASK_SHIFT + 1);
	long saddr = (long)addr;

	return (((saddr << shift) >> shift) == saddr);
}

static void handle_bad_prq_event(struct intel_iommu *iommu,
				 struct page_req_dsc *req, int result)
{
	struct qi_desc desc = { };

	pr_err("%s: Invalid page request: %08llx %08llx\n",
	       iommu->name, ((unsigned long long *)req)[0],
	       ((unsigned long long *)req)[1]);

	if (!req->lpig)
		return;

	desc.qw0 = QI_PGRP_PASID(req->pasid) |
			QI_PGRP_DID(req->rid) |
			QI_PGRP_PASID_P(req->pasid_present) |
			QI_PGRP_RESP_CODE(result) |
			QI_PGRP_RESP_TYPE;
	desc.qw1 = QI_PGRP_IDX(req->prg_index);

	qi_submit_sync(iommu, &desc, 1, 0);
}

static int prq_to_iommu_prot(struct page_req_dsc *req)
{
	int prot = 0;

	if (req->rd_req)
		prot |= IOMMU_FAULT_PERM_READ;
	if (req->wr_req)
		prot |= IOMMU_FAULT_PERM_WRITE;
	if (req->exe_req)
		prot |= IOMMU_FAULT_PERM_EXEC;
	if (req->pm_req)
		prot |= IOMMU_FAULT_PERM_PRIV;

	return prot;
}

static void intel_prq_report(struct intel_iommu *iommu, struct device *dev,
			     struct page_req_dsc *desc)
{
	struct iopf_fault event = { };

	/* Fill in event data for device specific processing */
	event.fault.type = IOMMU_FAULT_PAGE_REQ;
	event.fault.prm.addr = (u64)desc->addr << VTD_PAGE_SHIFT;
	event.fault.prm.pasid = desc->pasid;
	event.fault.prm.grpid = desc->prg_index;
	event.fault.prm.perm = prq_to_iommu_prot(desc);

	if (desc->lpig)
		event.fault.prm.flags |= IOMMU_FAULT_PAGE_REQUEST_LAST_PAGE;
	if (desc->pasid_present) {
		event.fault.prm.flags |= IOMMU_FAULT_PAGE_REQUEST_PASID_VALID;
		event.fault.prm.flags |= IOMMU_FAULT_PAGE_RESPONSE_NEEDS_PASID;
	}

	iommu_report_device_fault(dev, &event);
}

static irqreturn_t prq_event_thread(int irq, void *d)
{
	struct intel_iommu *iommu = d;
	struct page_req_dsc *req;
	int head, tail, handled;
	struct device *dev;
	u64 address;

	/*
	 * Clear PPR bit before reading head/tail registers, to ensure that
	 * we get a new interrupt if needed.
	 */
	writel(DMA_PRS_PPR, iommu->reg + DMAR_PRS_REG);

	tail = readq(iommu->reg + DMAR_PQT_REG) & PRQ_RING_MASK;
	head = readq(iommu->reg + DMAR_PQH_REG) & PRQ_RING_MASK;
	handled = (head != tail);
	while (head != tail) {
		req = &iommu->prq[head / sizeof(*req)];
		address = (u64)req->addr << VTD_PAGE_SHIFT;

		if (unlikely(!is_canonical_address(address))) {
			pr_err("IOMMU: %s: Address is not canonical\n",
			       iommu->name);
bad_req:
			handle_bad_prq_event(iommu, req, QI_RESP_INVALID);
			goto prq_advance;
		}

		if (unlikely(req->pm_req && (req->rd_req | req->wr_req))) {
			pr_err("IOMMU: %s: Page request in Privilege Mode\n",
			       iommu->name);
			goto bad_req;
		}

		if (unlikely(req->exe_req && req->rd_req)) {
			pr_err("IOMMU: %s: Execution request not supported\n",
			       iommu->name);
			goto bad_req;
		}

		/* Drop Stop Marker message. No need for a response. */
		if (unlikely(req->lpig && !req->rd_req && !req->wr_req))
			goto prq_advance;

		/*
		 * If prq is to be handled outside iommu driver via receiver of
		 * the fault notifiers, we skip the page response here.
		 */
		mutex_lock(&iommu->iopf_lock);
		dev = device_rbtree_find(iommu, req->rid);
		if (!dev) {
			mutex_unlock(&iommu->iopf_lock);
			goto bad_req;
		}

		intel_prq_report(iommu, dev, req);
		trace_prq_report(iommu, dev, req->qw_0, req->qw_1,
				 req->qw_2, req->qw_3,
				 iommu->prq_seq_number++);
		mutex_unlock(&iommu->iopf_lock);
prq_advance:
		head = (head + sizeof(*req)) & PRQ_RING_MASK;
	}

	writeq(tail, iommu->reg + DMAR_PQH_REG);

	/*
	 * Clear the page request overflow bit and wake up all threads that
	 * are waiting for the completion of this handling.
	 */
	if (readl(iommu->reg + DMAR_PRS_REG) & DMA_PRS_PRO) {
		pr_info_ratelimited("IOMMU: %s: PRQ overflow detected\n",
				    iommu->name);
		head = readq(iommu->reg + DMAR_PQH_REG) & PRQ_RING_MASK;
		tail = readq(iommu->reg + DMAR_PQT_REG) & PRQ_RING_MASK;
		if (head == tail) {
			iopf_queue_discard_partial(iommu->iopf_queue);
			writel(DMA_PRS_PRO, iommu->reg + DMAR_PRS_REG);
			pr_info_ratelimited("IOMMU: %s: PRQ overflow cleared",
					    iommu->name);
		}
	}

	if (!completion_done(&iommu->prq_complete))
		complete(&iommu->prq_complete);

	return IRQ_RETVAL(handled);
}

#define PRQ_WATCHDOG_INTERVAL	HZ

/*
 * Count every watchdog-detected stall. The log message is ratelimited, so
 * counting log lines badly undercounts; this is the number to trust for a
 * soak test. /sys/module/kernel/parameters/prq_wd_stalls (writable to reset).
 */
static unsigned long prq_wd_stalls;
core_param(prq_wd_stalls, prq_wd_stalls, ulong, 0644);

/*
 * A stall means the handler did not run. Two very different causes:
 *
 *   _lost    - no interrupt was delivered in the interval either, so the
 *              message the IOMMU believes it sent never reached a CPU.
 *   _blocked - interrupts WERE delivered but the threaded handler made no
 *              progress, i.e. it is stuck (iopf_lock, iopf workqueue, ...).
 *
 * For a threaded irq the primary handler increments the delivery count before
 * waking the thread, so the count advancing while prq_seq_number does not is
 * exactly the "delivered but stuck" case.
 */
static unsigned long prq_wd_stalls_lost;
core_param(prq_wd_stalls_lost, prq_wd_stalls_lost, ulong, 0644);
static unsigned long prq_wd_stalls_blocked;
core_param(prq_wd_stalls_blocked, prq_wd_stalls_blocked, ulong, 0644);

static unsigned long prq_irq_count(unsigned int irq)
{
	unsigned long sum = 0;
	int cpu;

	for_each_possible_cpu(cpu)
		sum += kstat_irqs_cpu(irq, cpu);
	return sum;
}

static void prq_watchdog_fn(struct work_struct *work)
{
	struct intel_iommu *iommu = container_of(work, struct intel_iommu,
						 prq_watchdog.work);
	u64 head = readq(iommu->reg + DMAR_PQH_REG) & PRQ_RING_MASK;
	u64 tail = readq(iommu->reg + DMAR_PQT_REG) & PRQ_RING_MASK;
	unsigned long irqs = prq_irq_count(iommu->pr_irq);
	unsigned long irq_delta = irqs - iommu->prq_wd_last_irqs;

	/*
	 * Detect lack of progress with prq_seq_number, not the head pointer:
	 * the head wraps the ring many times a second under load, so it can
	 * legitimately hold the same value across an interval. The sequence
	 * number increments once per request actually handled, so it advances
	 * if and only if the handler ran.
	 */
	if (head != tail && iommu->prq_seq_number == iommu->prq_wd_last_seq) {
		prq_wd_stalls++;
		if (irq_delta)
			prq_wd_stalls_blocked++;
		else
			prq_wd_stalls_lost++;
		/*
		 * The queue has been non-empty with no handler progress for a
		 * full interval. Either the page request event interrupt was
		 * lost or cannot be delivered (e.g. a stale or unreachable MSI
		 * destination), or it was delivered and the handler thread is
		 * stuck. Only the first case is helped by draining inline:
		 * prq_event_thread() clears PPR first, which re-arms the event
		 * edge for future requests, and disable_irq() excludes a
		 * concurrently running irq thread. In the second case that
		 * same disable_irq() would park this work behind the stuck
		 * thread for as long as it stays stuck, so only report it.
		 */
		pr_warn_ratelimited("IOMMU: %s: page request queue stalled [%s: irqs_delta=%lu] (head %llx tail %llx PRS %x PECTL %x PEDATA %x PEADDR %x PEUADDR %x)%s\n",
				    iommu->name,
				    irq_delta ? "handler blocked" : "NO INTERRUPT DELIVERED",
				    irq_delta, head, tail,
				    readl(iommu->reg + DMAR_PRS_REG),
				    readl(iommu->reg + DMAR_PECTL_REG),
				    readl(iommu->reg + DMAR_PEDATA_REG),
				    readl(iommu->reg + DMAR_PEADDR_REG),
				    readl(iommu->reg + DMAR_PEUADDR_REG),
				    irq_delta ? "" : ", draining inline");
		if (!irq_delta) {
			disable_irq(iommu->pr_irq);
			prq_event_thread(iommu->pr_irq, iommu);
			enable_irq(iommu->pr_irq);
			head = readq(iommu->reg + DMAR_PQH_REG) & PRQ_RING_MASK;
		}
	}
	iommu->prq_wd_last_head = head;
	iommu->prq_wd_last_seq = iommu->prq_seq_number;
	iommu->prq_wd_last_irqs = irqs;
	schedule_delayed_work(&iommu->prq_watchdog, PRQ_WATCHDOG_INTERVAL);
}

int intel_iommu_enable_prq(struct intel_iommu *iommu)
{
	struct iopf_queue *iopfq;
	int irq, ret;
	u32 prs;

	/*
	 * Observe the page request status register before touching anything:
	 * a PPR latched here (carried across a crash/kexec, or left by
	 * firmware) is what silently kills the first interrupt.
	 */
	prs = readl(iommu->reg + DMAR_PRS_REG);
	pr_info("IOMMU: %s: prq: entry PRS %08x PECTL %08x PQH %llx PQT %llx\n",
		iommu->name, prs, readl(iommu->reg + DMAR_PECTL_REG),
		readq(iommu->reg + DMAR_PQH_REG),
		readq(iommu->reg + DMAR_PQT_REG));

	iommu->prq =
		iommu_alloc_pages_node_sz(iommu->node, GFP_KERNEL, PRQ_SIZE);
	if (!iommu->prq) {
		pr_warn("IOMMU: %s: Failed to allocate page request queue\n",
			iommu->name);
		return -ENOMEM;
	}

	irq = dmar_alloc_hwirq(IOMMU_IRQ_ID_OFFSET_PRQ + iommu->seq_id, iommu->node, iommu);
	if (irq <= 0) {
		pr_err("IOMMU: %s: Failed to create IRQ vector for page request queue\n",
		       iommu->name);
		ret = -EINVAL;
		goto free_prq;
	}
	iommu->pr_irq = irq;

	snprintf(iommu->iopfq_name, sizeof(iommu->iopfq_name),
		 "dmar%d-iopfq", iommu->seq_id);
	iopfq = iopf_queue_alloc(iommu->iopfq_name);
	if (!iopfq) {
		pr_err("IOMMU: %s: Failed to allocate iopf queue\n", iommu->name);
		ret = -ENOMEM;
		goto free_hwirq;
	}
	iommu->iopf_queue = iopfq;

	snprintf(iommu->prq_name, sizeof(iommu->prq_name), "dmar%d-prq", iommu->seq_id);

	ret = request_threaded_irq(irq, NULL, prq_event_thread, IRQF_ONESHOT,
				   iommu->prq_name, iommu);
	if (ret) {
		pr_err("IOMMU: %s: Failed to request IRQ for page request queue\n",
		       iommu->name);
		goto free_iopfq;
	}
	writeq(0ULL, iommu->reg + DMAR_PQH_REG);
	writeq(0ULL, iommu->reg + DMAR_PQT_REG);

	/*
	 * Clear any latched page request status before the queue goes live.
	 * The page request event interrupt is generated only on the 0->1
	 * transition of PPR, so a stale PPR - left behind by a previous PRQ
	 * session torn down with requests outstanding, by kexec, or by
	 * firmware - means the first request after enable never generates an
	 * interrupt. The queue then fills with the head parked at 0 and no
	 * interrupt can ever arrive, because the transition already happened.
	 */
	prs = readl(iommu->reg + DMAR_PRS_REG);
	if (prs & (DMA_PRS_PPR | DMA_PRS_PRO))
		pr_warn("IOMMU: %s: stale page request status at enable (PRS %x), clearing\n",
			iommu->name, prs);
	else
		pr_info("IOMMU: %s: prq: PRS clean at enable (%08x)\n",
			iommu->name, prs);
	writel(DMA_PRS_PPR | DMA_PRS_PRO, iommu->reg + DMAR_PRS_REG);

	writeq(virt_to_phys(iommu->prq) | PRQ_ORDER, iommu->reg + DMAR_PQA_REG);

	init_completion(&iommu->prq_complete);

	iommu->prq_wd_last_head = 0;
	iommu->prq_wd_last_seq = 0;
	iommu->prq_wd_last_irqs = 0;
	INIT_DELAYED_WORK(&iommu->prq_watchdog, prq_watchdog_fn);
	schedule_delayed_work(&iommu->prq_watchdog, PRQ_WATCHDOG_INTERVAL);

	return 0;

free_iopfq:
	iopf_queue_free(iommu->iopf_queue);
	iommu->iopf_queue = NULL;
free_hwirq:
	dmar_free_hwirq(irq);
	iommu->pr_irq = 0;
free_prq:
	iommu_free_pages(iommu->prq);
	iommu->prq = NULL;

	return ret;
}

int intel_iommu_finish_prq(struct intel_iommu *iommu)
{
	/* The watchdog exists only once the queue was enabled. */
	if (iommu->prq)
		cancel_delayed_work_sync(&iommu->prq_watchdog);
	writeq(0ULL, iommu->reg + DMAR_PQH_REG);
	writeq(0ULL, iommu->reg + DMAR_PQT_REG);
	writeq(0ULL, iommu->reg + DMAR_PQA_REG);

	/* Leave no latched status behind for the next enable. */
	writel(DMA_PRS_PPR | DMA_PRS_PRO, iommu->reg + DMAR_PRS_REG);

	if (iommu->pr_irq) {
		free_irq(iommu->pr_irq, iommu);
		dmar_free_hwirq(iommu->pr_irq);
		iommu->pr_irq = 0;
	}

	if (iommu->iopf_queue) {
		iopf_queue_free(iommu->iopf_queue);
		iommu->iopf_queue = NULL;
	}

	iommu_free_pages(iommu->prq);
	iommu->prq = NULL;

	return 0;
}

void intel_iommu_page_response(struct device *dev, struct iopf_fault *evt,
			       struct iommu_page_response *msg)
{
	struct device_domain_info *info = dev_iommu_priv_get(dev);
	struct intel_iommu *iommu = info->iommu;
	u8 bus = info->bus, devfn = info->devfn;
	struct iommu_fault_page_request *prm;
	struct qi_desc desc;
	bool pasid_present;
	u16 sid;

	prm = &evt->fault.prm;
	sid = PCI_DEVID(bus, devfn);
	pasid_present = prm->flags & IOMMU_FAULT_PAGE_REQUEST_PASID_VALID;

	desc.qw0 = QI_PGRP_PASID(prm->pasid) | QI_PGRP_DID(sid) |
			QI_PGRP_PASID_P(pasid_present) |
			QI_PGRP_RESP_CODE(msg->code) |
			QI_PGRP_RESP_TYPE;
	desc.qw1 = QI_PGRP_IDX(prm->grpid);
	desc.qw2 = 0;
	desc.qw3 = 0;

	qi_submit_sync(iommu, &desc, 1, 0);
}
