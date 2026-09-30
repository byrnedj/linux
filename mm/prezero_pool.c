// SPDX-License-Identifier: GPL-2.0
/*
 * Per-node pool of pre-zeroed PMD-order folios.
 *
 * A background worker allocates folios, zeroes them through the
 * clearing offload provider and parks them; the PMD anonymous fault
 * path consumes them, skipping fault-time zeroing entirely. Refill
 * runs off the critical path, where offloaded zeroing is strictly
 * better than CPU zeroing: with no consumer adjacent there is no
 * cache-warmth to lose, so the freed cycles are pure profit.
 *
 * The pool is filled only for nodes that faults have asked for, so a
 * node without CPUs never parks memory nobody will take. Parked folios
 * are given back under memory pressure through a shrinker, and ahead
 * of a memory offline through a notifier, since a folio held here is
 * neither on an LRU nor movable.
 *
 * Disabled by default; /sys/kernel/mm/prezero_pool/enabled toggles.
 */
#include <linux/mm_offload.h>
#include <linux/cpuset.h>
#include <linux/gfp.h>
#include <linux/kobject.h>
#include <linux/list.h>
#include <linux/memory.h>
#include <linux/mempolicy.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/nodemask.h>
#include <linux/shrinker.h>
#include <linux/spinlock.h>
#include <linux/workqueue.h>

#include "internal.h"

#ifdef CONFIG_TRANSPARENT_HUGEPAGE

static bool pool_enabled;
static unsigned int max_per_node = 256;
static atomic_long_t pool_hits;
static atomic_long_t pool_misses;
/* Nodes a fault has asked for: the only ones worth filling. */
static nodemask_t pool_wanted;

static struct pz_node {
	spinlock_t lock;
	struct list_head folios;
	unsigned int count;
} pz_nodes[MAX_NUMNODES];

static void prezero_refill_fn(struct work_struct *work);
static DECLARE_WORK(refill_work, prezero_refill_fn);

static void prezero_refill_fn(struct work_struct *work)
{
	int nid;

	for_each_node_mask(nid, pool_wanted) {
		struct pz_node *pz = &pz_nodes[nid];

		if (!node_online(nid))
			continue;

		while (READ_ONCE(pool_enabled) &&
		       mm_offload_clear_available() &&
		       READ_ONCE(pz->count) < READ_ONCE(max_per_node)) {
			struct folio *folio;
			struct page *page;

			page = alloc_pages_node(nid, GFP_TRANSHUGE_LIGHT |
						__GFP_THISNODE | __GFP_NOWARN,
						HPAGE_PMD_ORDER);
			if (!page)
				break;
			folio = page_rmappable_folio(page);

			if (mm_offload_clear_folio(folio, 0)) {
				folio_put(folio);
				break;
			}

			spin_lock(&pz->lock);
			list_add(&folio->lru, &pz->folios);
			pz->count++;
			spin_unlock(&pz->lock);
		}
	}
}

/*
 * The pool is per node and knows nothing of policy, so it may only
 * serve a fault the allocator would have placed on @nid anyway.
 */
static bool prezero_pool_policy_allows(struct vm_area_struct *vma,
				       unsigned long addr, gfp_t gfp, int nid)
{
	bool allowed = true;
#ifdef CONFIG_NUMA
	struct mempolicy *pol;
	pgoff_t ilx;

	pol = get_vma_policy(vma, addr, HPAGE_PMD_ORDER, &ilx);
	if (pol) {
		switch (pol->mode) {
		case MPOL_DEFAULT:
		case MPOL_LOCAL:
			break;
		case MPOL_PREFERRED:
		case MPOL_PREFERRED_MANY:
		case MPOL_BIND:
			allowed = node_isset(nid, pol->nodes);
			break;
		default:
			/* The interleaves place by address; leave them to it. */
			allowed = false;
			break;
		}
	}
	mpol_cond_put(pol);
#endif
	return allowed && cpuset_current_node_allowed(nid, gfp);
}

/**
 * prezero_pool_get - take a pre-zeroed PMD-order folio for a fault.
 * @vma: the faulting VMA
 * @addr: the faulting address
 * @gfp: the flags the fault would allocate with
 *
 * Return: a fully zeroed, rmappable folio with reference held, or
 * NULL when the pool is disabled or empty for the local node, or the
 * fault's memory policy or cpuset would not have placed it there.
 */
struct folio *prezero_pool_get(struct vm_area_struct *vma, unsigned long addr,
			       gfp_t gfp)
{
	int nid = numa_mem_id();
	struct pz_node *pz;
	struct folio *folio = NULL;

	if (!READ_ONCE(pool_enabled))
		return NULL;
	if (!prezero_pool_policy_allows(vma, addr, gfp, nid))
		return NULL;

	if (!node_isset(nid, pool_wanted))
		node_set(nid, pool_wanted);

	pz = &pz_nodes[nid];
	spin_lock(&pz->lock);
	folio = list_first_entry_or_null(&pz->folios, struct folio, lru);
	if (folio) {
		list_del(&folio->lru);
		pz->count--;
	}
	spin_unlock(&pz->lock);

	if (folio) {
		atomic_long_inc(&pool_hits);
		if (READ_ONCE(pz->count) < READ_ONCE(max_per_node) / 2)
			schedule_work(&refill_work);
	} else {
		atomic_long_inc(&pool_misses);
		schedule_work(&refill_work);
	}
	return folio;
}

/* Give back parked folios of @nid until at most @keep remain. */
static unsigned long prezero_pool_trim(int nid, unsigned int keep)
{
	struct pz_node *pz = &pz_nodes[nid];
	unsigned long freed = 0;
	struct folio *folio;

	spin_lock(&pz->lock);
	while (pz->count > keep &&
	       (folio = list_first_entry_or_null(&pz->folios,
						 struct folio, lru))) {
		list_del(&folio->lru);
		pz->count--;
		spin_unlock(&pz->lock);
		folio_put(folio);
		freed++;
		spin_lock(&pz->lock);
	}
	spin_unlock(&pz->lock);
	return freed;
}

static void prezero_pool_drain(void)
{
	int nid;

	for (nid = 0; nid < MAX_NUMNODES; nid++)
		prezero_pool_trim(nid, 0);
}

/*
 * Parked folios are invisible to reclaim, so hand them back when the
 * system is short: each one is a PMD's worth of free memory.
 */
static unsigned long prezero_pool_count(struct shrinker *shrinker,
					struct shrink_control *sc)
{
	unsigned long nr = READ_ONCE(pz_nodes[sc->nid].count) * HPAGE_PMD_NR;

	return nr ? nr : SHRINK_EMPTY;
}

static unsigned long prezero_pool_scan(struct shrinker *shrinker,
				       struct shrink_control *sc)
{
	unsigned int nr = DIV_ROUND_UP(sc->nr_to_scan, HPAGE_PMD_NR);
	unsigned int count = READ_ONCE(pz_nodes[sc->nid].count);

	return prezero_pool_trim(sc->nid, count > nr ? count - nr : 0) *
		HPAGE_PMD_NR;
}

/*
 * A parked folio is neither on an LRU nor movable, so it would stop a
 * memory offline cold. Empty the node's pool before the offline starts;
 * the refill avoids the range because the memory is gone by then.
 */
static int prezero_pool_memory_notify(struct notifier_block *nb,
				      unsigned long action, void *data)
{
	struct memory_notify *mn = data;

	if (action == MEM_GOING_OFFLINE)
		prezero_pool_trim(pfn_to_nid(mn->start_pfn), 0);
	return NOTIFY_OK;
}

static ssize_t enabled_show(struct kobject *kobj, struct kobj_attribute *attr,
			    char *buf)
{
	return sysfs_emit(buf, "%d\n", pool_enabled);
}

static ssize_t enabled_store(struct kobject *kobj, struct kobj_attribute *attr,
			     const char *buf, size_t count)
{
	bool val;
	int ret;

	ret = kstrtobool(buf, &val);
	if (ret)
		return ret;

	WRITE_ONCE(pool_enabled, val);
	if (val) {
		schedule_work(&refill_work);
	} else {
		flush_work(&refill_work);
		prezero_pool_drain();
	}
	return count;
}

static ssize_t max_per_node_show(struct kobject *kobj,
				 struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", max_per_node);
}

static ssize_t max_per_node_store(struct kobject *kobj,
				  struct kobj_attribute *attr, const char *buf,
				  size_t count)
{
	unsigned int val;
	int ret, nid;

	ret = kstrtouint(buf, 0, &val);
	if (ret)
		return ret;

	WRITE_ONCE(max_per_node, val);
	if (READ_ONCE(pool_enabled))
		schedule_work(&refill_work);
	/* A lower limit applies to what is already parked, too. */
	for (nid = 0; nid < MAX_NUMNODES; nid++)
		prezero_pool_trim(nid, val);
	return count;
}

static ssize_t size_show(struct kobject *kobj, struct kobj_attribute *attr,
			 char *buf)
{
	unsigned long total = 0;
	int nid;

	for_each_online_node(nid)
		total += READ_ONCE(pz_nodes[nid].count);
	return sysfs_emit(buf, "%lu\n", total);
}

static ssize_t hits_show(struct kobject *kobj, struct kobj_attribute *attr,
			 char *buf)
{
	return sysfs_emit(buf, "%lu\n", atomic_long_read(&pool_hits));
}

static ssize_t misses_show(struct kobject *kobj, struct kobj_attribute *attr,
			   char *buf)
{
	return sysfs_emit(buf, "%lu\n", atomic_long_read(&pool_misses));
}

static struct kobj_attribute enabled_attr = __ATTR_RW(enabled);
static struct kobj_attribute max_per_node_attr = __ATTR_RW(max_per_node);
static struct kobj_attribute size_attr = __ATTR_RO(size);
static struct kobj_attribute hits_attr = __ATTR_RO(hits);
static struct kobj_attribute misses_attr = __ATTR_RO(misses);

static struct attribute *prezero_pool_attrs[] = {
	&enabled_attr.attr,
	&max_per_node_attr.attr,
	&size_attr.attr,
	&hits_attr.attr,
	&misses_attr.attr,
	NULL
};

static const struct attribute_group prezero_pool_attr_group = {
	.name = "prezero_pool",
	.attrs = prezero_pool_attrs,
};

static int __init prezero_pool_init(void)
{
	struct shrinker *shrinker;
	int nid;

	for (nid = 0; nid < MAX_NUMNODES; nid++) {
		spin_lock_init(&pz_nodes[nid].lock);
		INIT_LIST_HEAD(&pz_nodes[nid].folios);
	}

	shrinker = shrinker_alloc(SHRINKER_NUMA_AWARE, "mm-prezero-pool");
	if (!shrinker)
		return -ENOMEM;
	shrinker->count_objects = prezero_pool_count;
	shrinker->scan_objects = prezero_pool_scan;
	shrinker_register(shrinker);

	hotplug_memory_notifier(prezero_pool_memory_notify, 0);

	return sysfs_create_group(mm_kobj, &prezero_pool_attr_group);
}
subsys_initcall(prezero_pool_init);

#else /* CONFIG_TRANSPARENT_HUGEPAGE */

struct folio *prezero_pool_get(struct vm_area_struct *vma, unsigned long addr,
			       gfp_t gfp)
{
	return NULL;
}

#endif
