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
 * Disabled by default; /sys/kernel/mm/prezero_pool/enabled toggles.
 */
#include <linux/mm_offload.h>
#include <linux/gfp.h>
#include <linux/kobject.h>
#include <linux/list.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/spinlock.h>
#include <linux/workqueue.h>

#include "internal.h"

#ifdef CONFIG_TRANSPARENT_HUGEPAGE

static bool pool_enabled;
static unsigned int max_per_node = 256;
static atomic_long_t pool_hits;
static atomic_long_t pool_misses;

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

	for_each_online_node(nid) {
		struct pz_node *pz = &pz_nodes[nid];

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

/**
 * prezero_pool_get - take a pre-zeroed PMD-order folio for @nid.
 *
 * Return: a fully zeroed, rmappable folio with reference held, or
 * NULL when the pool is disabled or empty for this node.
 */
struct folio *prezero_pool_get(int nid)
{
	struct pz_node *pz;
	struct folio *folio = NULL;

	if (!READ_ONCE(pool_enabled))
		return NULL;
	if (nid < 0 || nid >= MAX_NUMNODES)
		return NULL;

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

static void prezero_pool_drain(void)
{
	int nid;

	for_each_online_node(nid) {
		struct pz_node *pz = &pz_nodes[nid];
		struct folio *folio;

		spin_lock(&pz->lock);
		while ((folio = list_first_entry_or_null(&pz->folios,
							 struct folio, lru))) {
			list_del(&folio->lru);
			pz->count--;
			spin_unlock(&pz->lock);
			folio_put(folio);
			spin_lock(&pz->lock);
		}
		spin_unlock(&pz->lock);
	}
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
	int ret;

	ret = kstrtouint(buf, 0, &val);
	if (ret)
		return ret;

	WRITE_ONCE(max_per_node, val);
	if (READ_ONCE(pool_enabled))
		schedule_work(&refill_work);
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
	int nid;

	for (nid = 0; nid < MAX_NUMNODES; nid++) {
		spin_lock_init(&pz_nodes[nid].lock);
		INIT_LIST_HEAD(&pz_nodes[nid].folios);
	}
	return sysfs_create_group(mm_kobj, &prezero_pool_attr_group);
}
subsys_initcall(prezero_pool_init);

#else /* CONFIG_TRANSPARENT_HUGEPAGE */

struct folio *prezero_pool_get(int nid)
{
	return NULL;
}

#endif
