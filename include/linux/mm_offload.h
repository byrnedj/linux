/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _LINUX_MM_OFFLOAD_H
#define _LINUX_MM_OFFLOAD_H

#include <linux/bits.h>
#include <linux/errno.h>
#include <linux/jump_label.h>
#include <linux/migrate_mode.h>

struct folio;
struct list_head;
struct module;

#define MM_OFFLOAD_NAME_LEN 32

/*
 * Migration reasons that may ever be offloaded. A provider's mask is
 * clamped to this. Allow all reasons by default.
 */
#define MIGRATE_OFFLOAD_REASONS_ALLOWED		(BIT(MR_TYPES) - 1)

/**
 * struct mm_offload_provider - accelerator for memory management data paths.
 * @name: name of the provider.
 * @copy_folios: copy @folio_cnt folios from @src_list to @dst_list for
 *	page migration.
 *
 *	The provider may inspect @folio_cnt to decide whether the batch
 *	is worth offloading, e.g. skip when the batch is too small to
 *	amortize setup cost.
 *	The callback must set FOLIO_CONTENT_COPIED in dst->migrate_info
 *	for each successfully copied destination folio.
 *	Folios without this marker are copied via per-folio CPU copy in
 *	the move phase.
 *
 * @clear_folio: zero the contents of @folio.
 *
 *	@addr_hint is the user address most likely to be touched first
 *	when the folio is a fault-path allocation, or 0 when there is no
 *	meaningful hint. The provider may inspect folio_size() to decide
 *	whether the folio is worth offloading, e.g. refuse when it is too
 *	small to amortize setup cost. May sleep. On any non-zero return
 *	the core zeroes the folio on the CPU; the provider must not leave
 *	the folio partially cleared and return success.
 *
 * @owner: module providing the operations. NULL for built-in (=y) drivers.
 *
 * Each operation is optional. The core only takes the offload path for
 * an operation the active provider implements; the others keep their
 * CPU implementation.
 */
struct mm_offload_provider {
	char name[MM_OFFLOAD_NAME_LEN];
	int (*copy_folios)(struct list_head *dst_list, struct list_head *src_list,
			unsigned int folio_cnt);
	int (*clear_folio)(struct folio *folio, unsigned long addr_hint);
	struct module *owner;
};

#ifdef CONFIG_MM_OFFLOAD
DECLARE_STATIC_KEY_FALSE(mm_offload_clear_folio_enabled);
int mm_offload_register(const struct mm_offload_provider *p,
		unsigned long migrate_reason_mask);
int mm_offload_unregister(const struct mm_offload_provider *p);
int mm_offload_set_migrate_reason_mask(const struct mm_offload_provider *p,
		unsigned long mask);
int mm_offload_reason_mask_parse(const char *buf, unsigned long *maskp);
int mm_offload_reason_mask_format(char *buf, unsigned long mask);
bool migrate_should_offload(int reason);
int migrate_offload_batch_copy(struct list_head *dst_batch,
		struct list_head *src_batch, unsigned int nr_batch);
int mm_offload_clear_folio(struct folio *folio, unsigned long addr_hint);
bool mm_offload_cpus_saturated(void);
struct folio *prezero_pool_get(int nid);

/*
 * Cheap availability test for hot paths: a static branch, patched only
 * when a provider with the operation (un)registers.
 */
static inline bool mm_offload_clear_available(void)
{
	return static_branch_unlikely(&mm_offload_clear_folio_enabled);
}
#else
static inline int mm_offload_register(const struct mm_offload_provider *p,
		unsigned long migrate_reason_mask) { return -EOPNOTSUPP; }
static inline int mm_offload_unregister(const struct mm_offload_provider *p)
{
	return -EOPNOTSUPP;
}
static inline int mm_offload_set_migrate_reason_mask(
		const struct mm_offload_provider *p, unsigned long mask)
{
	return -EOPNOTSUPP;
}
static inline int mm_offload_reason_mask_parse(const char *buf,
		unsigned long *maskp) { return -EOPNOTSUPP; }
static inline int mm_offload_reason_mask_format(char *buf,
		unsigned long mask) { return -EOPNOTSUPP; }
static inline bool migrate_should_offload(int reason) { return false; }
static inline int migrate_offload_batch_copy(struct list_head *dst_batch,
		struct list_head *src_batch, unsigned int nr_batch)
{
	return -EOPNOTSUPP;
}
static inline int mm_offload_clear_folio(struct folio *folio,
		unsigned long addr_hint) { return -EOPNOTSUPP; }
static inline bool mm_offload_clear_available(void) { return false; }
static inline bool mm_offload_cpus_saturated(void) { return false; }
static inline struct folio *prezero_pool_get(int nid) { return NULL; }
#endif

#endif /* _LINUX_MM_OFFLOAD_H */
