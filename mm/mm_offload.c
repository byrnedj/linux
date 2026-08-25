// SPDX-License-Identifier: GPL-2.0
#include <linux/bitops.h>
#include <linux/jump_label.h>
#include <linux/kstrtox.h>
#include <linux/migrate.h>
#include <linux/mm_offload.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/srcu.h>
#include <linux/static_call.h>
#include <linux/string.h>
#include <linux/sysfs.h>

DEFINE_STATIC_KEY_FALSE(mm_offload_copy_folios_enabled);
DEFINE_STATIC_KEY_FALSE(mm_offload_clear_folio_enabled);
EXPORT_SYMBOL_GPL(mm_offload_clear_folio_enabled);
DEFINE_SRCU(mm_offload_srcu);
DEFINE_STATIC_CALL(mm_offload_copy_folios_fn, migrate_folios_mc_copy);

/*
 * Default clear target. Only reachable in the narrow window where a
 * caller saw the static branch enabled while the provider was being
 * torn down; the caller falls back to CPU clearing.
 */
static int mm_offload_clear_folio_null(struct folio *folio,
				       unsigned long addr_hint)
{
	return -ENXIO;
}
DEFINE_STATIC_CALL(mm_offload_clear_folio_fn, mm_offload_clear_folio_null);

static DEFINE_MUTEX(provider_mutex);
static const struct mm_offload_provider *active_provider;

/*
 * The active provider's migration reason mask. This is the single source of truth
 * for which reasons are offloaded.
 */
static unsigned long active_reason_mask;

bool migrate_should_offload(int reason)
{
	if (!static_branch_unlikely(&mm_offload_copy_folios_enabled))
		return false;

	return READ_ONCE(active_reason_mask) & BIT(reason);
}

/**
 * mm_offload_reason_mask_parse - parse migration reason mask.
 * @buf: input string, either a number (hex/decimal, e.g. "0x101") or a
 *	comma-separated list of reason names (see migrate_reason_names[]),
 *	e.g. "compaction,demotion". The tokens "all" and "none" select
 *	every or no reason.
 * @maskp: parsed mask, clamped to MIGRATE_OFFLOAD_REASONS_ALLOWED.
 *
 * Return: 0 on success, -EINVAL on an unknown name, -ENOMEM on OOM.
 */
int mm_offload_reason_mask_parse(const char *buf, unsigned long *maskp)
{
	unsigned long mask;
	char *copy, *p, *tok;
	int ret = 0;

	/* A plain number is treated as a raw mask. */
	if (!kstrtoul(buf, 0, &mask))
		goto done;

	copy = kstrdup(buf, GFP_KERNEL);
	if (!copy)
		return -ENOMEM;
	mask = 0;
	p = strim(copy);

	while ((tok = strsep(&p, ",")) != NULL) {
		int i;

		tok = strim(tok);
		if (!*tok)
			continue;
		if (!strcmp(tok, "none")) {
			mask = 0;
			continue;
		}
		if (!strcmp(tok, "all")) {
			mask = MIGRATE_OFFLOAD_REASONS_ALLOWED;
			continue;
		}
		i = match_string(migrate_reason_names, MR_TYPES, tok);
		if (i < 0) {
			ret = -EINVAL;
			break;
		}
		mask |= BIT(i);
	}
	kfree(copy);
	if (ret)
		return ret;
done:
	*maskp = mask & MIGRATE_OFFLOAD_REASONS_ALLOWED;
	return 0;
}
EXPORT_SYMBOL_GPL(mm_offload_reason_mask_parse);

/**
 * mm_offload_reason_mask_format - render a reason mask as names.
 * @buf: output buffer from kernel_param_ops get.
 * @mask: reason mask to render.
 *
 * Emits a comma-separated list of reason names, or "none".
 *
 * Return: number of bytes written to @buf.
 */
int mm_offload_reason_mask_format(char *buf, unsigned long mask)
{
	int len = 0, i;

	for_each_set_bit(i, &mask, MR_TYPES)
		len += sysfs_emit_at(buf, len, "%s%s",
				     len ? "," : "", migrate_reason_names[i]);
	if (!len)
		return sysfs_emit(buf, "none\n");

	len += sysfs_emit_at(buf, len, "\n");
	return len;
}
EXPORT_SYMBOL_GPL(mm_offload_reason_mask_format);

/*
 * Hand the batch to the registered provider. The provider may decline
 * (typically based on batch size), in which case the move phase falls
 * back to per-folio CPU copy.
 */
int migrate_offload_batch_copy(struct list_head *dst_batch,
		struct list_head *src_batch, unsigned int nr_batch)
{
	int idx, rc;

	idx = srcu_read_lock(&mm_offload_srcu);
	rc = static_call(mm_offload_copy_folios_fn)(dst_batch, src_batch, nr_batch);
	srcu_read_unlock(&mm_offload_srcu, idx);
	return rc;
}

/**
 * mm_offload_clear_folio - zero a folio through the registered provider.
 * @folio: folio to zero.
 * @addr_hint: user address expected to be touched first, or 0.
 *
 * May sleep. The caller must hold a reference on @folio and the folio
 * must not be visible to any other user (no page table entries, not on
 * the LRU). On failure the folio contents are unspecified and the
 * caller must zero it by other means before exposing it.
 *
 * Return: 0 when the folio is fully zeroed, negative errno otherwise.
 */
int mm_offload_clear_folio(struct folio *folio, unsigned long addr_hint)
{
	int idx, rc;

	might_sleep();

	idx = srcu_read_lock(&mm_offload_srcu);
	rc = static_call(mm_offload_clear_folio_fn)(folio, addr_hint);
	srcu_read_unlock(&mm_offload_srcu, idx);
	return rc;
}

/**
 * mm_offload_set_migrate_reason_mask - update the active provider's migration reason mask.
 * @p: provider (must be the currently active one).
 * @mask: new reason mask.
 *
 * Return: 0 on success, -EINVAL if @p is NULL or not the active provider.
 */
int mm_offload_set_migrate_reason_mask(const struct mm_offload_provider *p, unsigned long mask)
{
	if (!p)
		return -EINVAL;

	mask &= MIGRATE_OFFLOAD_REASONS_ALLOWED;

	mutex_lock(&provider_mutex);
	if (active_provider != p) {
		mutex_unlock(&provider_mutex);
		return -EINVAL;
	}
	WRITE_ONCE(active_reason_mask, mask);
	mutex_unlock(&provider_mutex);
	return 0;
}
EXPORT_SYMBOL_GPL(mm_offload_set_migrate_reason_mask);

/**
 * mm_offload_register - register an offload provider.
 * @p: provider to install.
 * @migrate_reason_mask: initial set of BIT(MR_*) migration reasons to
 *	offload, can be changed later via mm_offload_set_migrate_reason_mask().
 *
 * Only one provider can be active at a time, returns -EBUSY if another
 * provider is already registered.
 *
 * Return: 0 on success, negative errno on failure.
 */
int mm_offload_register(const struct mm_offload_provider *p,
			unsigned long migrate_reason_mask)
{
	unsigned long mask;
	int ret = 0;

	if (!p || (!p->copy_folios && !p->clear_folio))
		return -EINVAL;

	mask = migrate_reason_mask & MIGRATE_OFFLOAD_REASONS_ALLOWED;

	mutex_lock(&provider_mutex);
	if (active_provider) {
		ret = -EBUSY;
		goto unlock;
	}

	/* @owner is NULL for built-in (=y) drivers; nothing to ref. */
	if (p->owner && !try_module_get(p->owner)) {
		ret = -ENODEV;
		goto unlock;
	}

	active_provider = p;
	if (p->copy_folios) {
		WRITE_ONCE(active_reason_mask, mask);
		static_call_update(mm_offload_copy_folios_fn, p->copy_folios);
		static_branch_enable(&mm_offload_copy_folios_enabled);
	}
	if (p->clear_folio) {
		static_call_update(mm_offload_clear_folio_fn, p->clear_folio);
		static_branch_enable(&mm_offload_clear_folio_enabled);
	}

unlock:
	mutex_unlock(&provider_mutex);

	if (ret)
		pr_err("mm_offload: %s: failed to register (%d)\n", p->name, ret);
	else
		pr_info("mm_offload: enabled by %s (reason_mask=0x%lx)\n",
			p->name, mask);
	return ret;
}
EXPORT_SYMBOL_GPL(mm_offload_register);

/**
 * mm_offload_unregister - unregister the active offload provider.
 * @p: provider to remove (must be the currently active one).
 *
 * Reverts static_call targets and waits for SRCU grace period so that
 * no in-flight migration is still calling the driver functions before
 * releasing the module.
 *
 * Return: 0 on success, negative errno on failure.
 */
int mm_offload_unregister(const struct mm_offload_provider *p)
{
	struct module *owner;

	if (!p)
		return -EINVAL;

	mutex_lock(&provider_mutex);
	if (active_provider != p) {
		mutex_unlock(&provider_mutex);
		return -EINVAL;
	}

	/*
	 * Disable the static branches first so new callers take the CPU
	 * paths.
	 */
	if (p->copy_folios) {
		static_branch_disable(&mm_offload_copy_folios_enabled);
		WRITE_ONCE(active_reason_mask, 0);
		static_call_update(mm_offload_copy_folios_fn,
				   migrate_folios_mc_copy);
	}
	if (p->clear_folio) {
		static_branch_disable(&mm_offload_clear_folio_enabled);
		static_call_update(mm_offload_clear_folio_fn,
				   mm_offload_clear_folio_null);
	}
	owner = active_provider->owner;
	active_provider = NULL;
	mutex_unlock(&provider_mutex);

	/* Wait for all in-flight callers to finish before module_put(). */
	synchronize_srcu(&mm_offload_srcu);
	if (owner)
		module_put(owner);

	pr_info("mm_offload: disabled by %s\n", p->name);
	return 0;
}
EXPORT_SYMBOL_GPL(mm_offload_unregister);
