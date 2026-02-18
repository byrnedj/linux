/* SPDX-License-Identifier: GPL-2.0 */

#ifndef _MIGRATE_OFFC_H
#define _MIGRATE_OFFC_H
#include <linux/atomic.h>
#include <linux/migrate_mode.h>
#include <linux/static_call.h>

struct folio;

#define MIGRATOR_NAME_LEN 32
struct migrator {
	char name[MIGRATOR_NAME_LEN];
	int (*migrate_offload_copy)(struct list_head *dst_list, struct list_head *src_list,
			    unsigned int folio_cnt);
	bool (*should_handle)(struct list_head *src_list, unsigned int nr_folios,
			      int reason);
	/* DMA-offloaded copy of a single large/gigantic folio pair (CoW path).
	 * Returns 0 on success, negative to fall back to CPU copy.
	 */
	int (*copy_large_folio)(struct folio *dst, struct folio *src);
	/* DMA-offloaded zero of a single large/gigantic folio (page-fault path).
	 * Returns 0 on success, negative to fall back to CPU zero.
	 */
	int (*zero_folio)(struct folio *folio);
	struct rcu_head srcu_head;
	struct module *owner;
};

extern struct migrator migrator;
extern struct mutex migrator_mut;
extern struct srcu_struct mig_srcu;
extern atomic_t dispatch_to_offc;

#ifdef CONFIG_OFFC_MIGRATION
void srcu_mig_cb(struct rcu_head *head);
int offc_update_migrator(struct migrator *mig);
unsigned char *get_active_migrator_name(void);
int start_offloading(struct migrator *migrator);
int stop_offloading(void);
#else
static inline void srcu_mig_cb(struct rcu_head *head) { };
static inline int offc_update_migrator(struct migrator *mig) { return 0; };
static inline unsigned char *get_active_migrator_name(void) { return NULL; };
static inline void start_offloading(struct migrator *migrator) { return 0; };
static inline void stop_offloading(void) { return 0; };
#endif /* CONFIG_OFFC_MIGRATION */

#ifdef CONFIG_HAVE_STATIC_CALL
int copy_large_folio_notsupp(struct folio *dst, struct folio *src);
int zero_folio_notsupp(struct folio *folio);
DECLARE_STATIC_CALL(_copy_large_folio, copy_large_folio_notsupp);
DECLARE_STATIC_CALL(_zero_folio, zero_folio_notsupp);
#endif /* CONFIG_HAVE_STATIC_CALL */

#endif /* _MIGRATE_OFFC_H */
