// SPDX-License-Identifier: GPL-2.0-or-later

#include <linux/ethtool_netlink.h>
#include <linux/netdevice.h>
#include <linux/printk.h>
#include <net/netdev_lock.h>
#include <net/netdev_queues.h>
#include <net/netdev_rx_queue.h>
#include <net/page_pool/memory_provider.h>

#include "page_pool_priv.h"

static void net_mp_log(struct net_device *dev, unsigned int rxq_idx,
                      const char *fmt, ...)
{
        struct va_format vaf;
        va_list args;

        va_start(args, fmt);
        vaf.fmt = fmt;
        vaf.va = &args;
        netdev_info(dev, "mp_open_rxq[%u]: %pV", rxq_idx, &vaf);
        va_end(args);
}

int netdev_rx_queue_restart(struct net_device *dev, unsigned int rxq_idx)
{
	struct netdev_rx_queue *rxq = __netif_get_rx_queue(dev, rxq_idx);
	const struct netdev_queue_mgmt_ops *qops = dev->queue_mgmt_ops;
	void *new_mem, *old_mem;
	int err;

	if (!qops || !qops->ndo_queue_stop || !qops->ndo_queue_mem_free ||
	    !qops->ndo_queue_mem_alloc || !qops->ndo_queue_start)
		return -EOPNOTSUPP;

	netdev_assert_locked(dev);

	new_mem = kvzalloc(qops->ndo_queue_mem_size, GFP_KERNEL);
	if (!new_mem)
		return -ENOMEM;

	old_mem = kvzalloc(qops->ndo_queue_mem_size, GFP_KERNEL);
	if (!old_mem) {
		err = -ENOMEM;
		goto err_free_new_mem;
	}

	err = qops->ndo_queue_mem_alloc(dev, new_mem, rxq_idx);
	if (err)
		goto err_free_old_mem;

	err = page_pool_check_memory_provider(dev, rxq);
	if (err)
		goto err_free_new_queue_mem;

	if (netif_running(dev)) {
		err = qops->ndo_queue_stop(dev, old_mem, rxq_idx);
		if (err)
			goto err_free_new_queue_mem;

		err = qops->ndo_queue_start(dev, new_mem, rxq_idx);
		if (err)
			goto err_start_queue;
	} else {
		swap(new_mem, old_mem);
	}

	qops->ndo_queue_mem_free(dev, old_mem);

	kvfree(old_mem);
	kvfree(new_mem);

	return 0;

err_start_queue:
	/* Restarting the queue with old_mem should be successful as we haven't
	 * changed any of the queue configuration, and there is not much we can
	 * do to recover from a failure here.
	 *
	 * WARN if we fail to recover the old rx queue, and at least free
	 * old_mem so we don't also leak that.
	 */
	if (qops->ndo_queue_start(dev, old_mem, rxq_idx)) {
		WARN(1,
		     "Failed to restart old queue in error path. RX queue %d may be unhealthy.",
		     rxq_idx);
		qops->ndo_queue_mem_free(dev, old_mem);
	}

err_free_new_queue_mem:
	qops->ndo_queue_mem_free(dev, new_mem);

err_free_old_mem:
	kvfree(old_mem);

err_free_new_mem:
	kvfree(new_mem);

	return err;
}
EXPORT_SYMBOL_NS_GPL(netdev_rx_queue_restart, "NETDEV_INTERNAL");

int __net_mp_open_rxq(struct net_device *dev, unsigned int rxq_idx,
                      const struct pp_memory_provider_params *p,
                      struct netlink_ext_ack *extack)
{
        unsigned int xdp_prog_cnt;
        struct netdev_rx_queue *rxq;
        int ret;

        net_mp_log(dev, rxq_idx,
                   "attempt: mp_ops=%p mp_priv=%p",
                   p ? p->mp_ops : NULL, p ? p->mp_priv : NULL);

        if (!netdev_need_ops_lock(dev)) {
                net_mp_log(dev, rxq_idx,
                           "reject: device does not require ops lock (unsupported)");
                return -EOPNOTSUPP;
        }

        if (rxq_idx >= dev->real_num_rx_queues) {
                NL_SET_ERR_MSG(extack, "rx queue index out of range");
                net_mp_log(dev, rxq_idx,
                           "reject: queue index %u >= real_num_rx_queues %u",
                           rxq_idx, dev->real_num_rx_queues);
                return -ERANGE;
        }
        rxq_idx = array_index_nospec(rxq_idx, dev->real_num_rx_queues);

        if (dev->cfg->hds_config != ETHTOOL_TCP_DATA_SPLIT_ENABLED) {
                NL_SET_ERR_MSG(extack, "tcp-data-split is disabled");
                net_mp_log(dev, rxq_idx,
                           "reject: tcp-data-split disabled (cfg=%u)",
                           dev->cfg->hds_config);
                return -EINVAL;
        }
        if (dev->cfg->hds_thresh) {
                NL_SET_ERR_MSG(extack, "hds-thresh is not zero");
                net_mp_log(dev, rxq_idx,
                           "reject: hds_thresh=%u (expected zero)",
                           dev->cfg->hds_thresh);
                return -EINVAL;
        }
        xdp_prog_cnt = dev_xdp_prog_count(dev);
        if (xdp_prog_cnt) {
                NL_SET_ERR_MSG(extack, "unable to custom memory provider to device with XDP program attached");
                net_mp_log(dev, rxq_idx,
                           "reject: device has %u XDP programs attached",
                           xdp_prog_cnt);
                return -EEXIST;
        }

        rxq = __netif_get_rx_queue(dev, rxq_idx);
        if (rxq->mp_params.mp_ops) {
                NL_SET_ERR_MSG(extack, "designated queue already memory provider bound");
                net_mp_log(dev, rxq_idx,
                           "reject: queue already bound to mp_ops=%p mp_priv=%p",
                           rxq->mp_params.mp_ops, rxq->mp_params.mp_priv);
                return -EEXIST;
        }
#ifdef CONFIG_XDP_SOCKETS
        if (rxq->pool) {
                NL_SET_ERR_MSG(extack, "designated queue already in use by AF_XDP");
                net_mp_log(dev, rxq_idx,
                           "reject: queue has AF_XDP pool=%p", rxq->pool);
                return -EBUSY;
        }
#endif

        net_mp_log(dev, rxq_idx,
                   "binding provider: mp_ops=%p mp_priv=%p",
                   p ? p->mp_ops : NULL, p ? p->mp_priv : NULL);
        rxq->mp_params = *p;
        ret = netdev_rx_queue_restart(dev, rxq_idx);
        if (ret) {
                net_mp_log(dev, rxq_idx,
                           "restart failed with %d -- rolling back", ret);
                rxq->mp_params.mp_ops = NULL;
                rxq->mp_params.mp_priv = NULL;
        } else {
                net_mp_log(dev, rxq_idx, "success");
        }
        return ret;
}

int net_mp_open_rxq(struct net_device *dev, unsigned int rxq_idx,
                    struct pp_memory_provider_params *p)
{
        int ret;

        netdev_lock(dev);
        net_mp_log(dev, rxq_idx, "invocation from unlocked context");
        ret = __net_mp_open_rxq(dev, rxq_idx, p, NULL);
        netdev_unlock(dev);
        net_mp_log(dev, rxq_idx, "returning %d", ret);
        return ret;
}

void __net_mp_close_rxq(struct net_device *dev, unsigned int ifq_idx,
			const struct pp_memory_provider_params *old_p)
{
	struct netdev_rx_queue *rxq;
	int err;

	if (WARN_ON_ONCE(ifq_idx >= dev->real_num_rx_queues))
		return;

	rxq = __netif_get_rx_queue(dev, ifq_idx);

	/* Callers holding a netdev ref may get here after we already
	 * went thru shutdown via dev_memory_provider_uninstall().
	 */
	if (dev->reg_state > NETREG_REGISTERED &&
	    !rxq->mp_params.mp_ops)
		return;

	if (WARN_ON_ONCE(rxq->mp_params.mp_ops != old_p->mp_ops ||
			 rxq->mp_params.mp_priv != old_p->mp_priv))
		return;

	rxq->mp_params.mp_ops = NULL;
	rxq->mp_params.mp_priv = NULL;
	err = netdev_rx_queue_restart(dev, ifq_idx);
	WARN_ON(err && err != -ENETDOWN);
}

void net_mp_close_rxq(struct net_device *dev, unsigned ifq_idx,
		      struct pp_memory_provider_params *old_p)
{
	netdev_lock(dev);
	__net_mp_close_rxq(dev, ifq_idx, old_p);
	netdev_unlock(dev);
}
