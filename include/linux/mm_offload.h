/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _LINUX_MM_OFFLOAD_H
#define _LINUX_MM_OFFLOAD_H

#include <linux/bits.h>
#include <linux/errno.h>
#include <linux/migrate_mode.h>

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
	struct module *owner;
};

#ifdef CONFIG_MM_OFFLOAD
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
#endif

#endif /* _LINUX_MM_OFFLOAD_H */
