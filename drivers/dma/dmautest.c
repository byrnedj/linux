// SPDX-License-Identifier: GPL-2.0-only
/*
 * DMA Engine test module
 *
 * Copyright (C) 2021 Intel Corporation
 */
#include <linux/dma-mapping.h>
#include <linux/dmaengine.h>
#include <linux/fs.h>
#include <linux/iommu.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/io_uring.h>

static unsigned int max_iovecs = 32;
module_param(max_iovecs, uint, 0644);
MODULE_PARM_DESC(max_iovecs, "maximum kernel buffers ready for copy to user");

static unsigned int buf_size = 0x1000;
module_param(buf_size, uint, 0644);
MODULE_PARM_DESC(buf_size, "Size of each kernel buffer");

static unsigned int pool_size = 1;
module_param(pool_size, uint, 0644);
MODULE_PARM_DESC(pool_size, "Number of source buffers to rotate through");


#define ONLY_IO_URING 1

struct dmautest_ctx {
	struct dma_chan		*chan;
	struct iommu_sva	*sva;
	unsigned int		pasid;
	struct kvec             **k_vecs;
	unsigned int		cur_element;
	ssize_t			total_len;
};

static int dmautest_open(struct inode *inode, struct file *file)
{
	struct dmautest_ctx *ctx;
	int rc = 0, i, j;
	unsigned int aligned_buf_size = (buf_size + 0xFFF) & ~0xFFF;
#ifndef ONLY_IO_URING
	dma_cap_mask_t mask;
	struct device *dev;
	struct dma_chan_attr_params param;
	int flags = IOMMU_SVA_BIND_KERNEL;
#endif

	ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
	if (!ctx)
		return -ENOMEM;

	ctx->total_len = 0;

	ctx->k_vecs = kcalloc(pool_size, sizeof(struct kvec *), GFP_KERNEL);
	if (!ctx->k_vecs) {
		rc = -ENOMEM;
		goto failed;
	}

	for (i = 0; i < pool_size; i++) {
		ctx->k_vecs[i] = kcalloc(max_iovecs, sizeof(struct kvec), GFP_KERNEL);
		if (!ctx->k_vecs[i]) {
			rc = -ENOMEM;
			goto failed;
		}

		for (j = 0; j < max_iovecs; j++) {
			void *buf;

			buf = kzalloc(aligned_buf_size, GFP_KERNEL);
			if (!buf) {
				rc = -ENOMEM;
				goto failed;
			}

			/* Initialize the buffer content */
			memset(buf, 0xb, aligned_buf_size);

			ctx->k_vecs[i][j].iov_base = buf;
			ctx->k_vecs[i][j].iov_len = buf_size;
		}
	}

	ctx->total_len = buf_size * max_iovecs;
	ctx->cur_element = 0;

#ifndef ONLY_IO_URING
	dma_cap_zero(mask);
	dma_cap_set(DMA_MEMCPY, mask);
	dma_cap_set(DMA_KERNEL_USER, mask);

	ctx->chan = dma_request_chan_by_mask(&mask);
	if (IS_ERR(ctx->chan)) {
		pr_warn("Failed to allocate dma channel!");
		rc = PTR_ERR(ctx->chan);
		goto failed;
	}

	dev = ctx->chan->device->dev;

	ctx->sva = iommu_sva_bind_device(dev, current->mm, flags);
	if (IS_ERR(ctx->sva)) {
		pr_warn("Failed to perform SVA bind\n");
		rc = PTR_ERR(ctx->sva);
		goto failed;
	}

	ctx->pasid = iommu_sva_get_pasid(ctx->sva);
	if (ctx->pasid == IOMMU_PASID_INVALID) {
		rc = -EINVAL;
		goto failed;
	}

	param.p.pasid = ctx->pasid;
	param.p.priv = true;

	if (dmaengine_chan_set_attr(ctx->chan, DMA_CHAN_SET_PASID, &param)) {
		rc = -EINVAL;
		goto failed;
	}
#endif
	file->private_data = ctx;

	return 0;

failed:
	if (ctx) {
		if (ctx->k_vecs) {
			for (i = 0; i < pool_size; i++) {
				for (j = 0; j < max_iovecs; j++)
					kfree(ctx->k_vecs[i][j].iov_base);

				kfree(ctx->k_vecs[i]);
			}
			kfree(ctx->k_vecs);
		}
		if (ctx->sva && !IS_ERR(ctx->sva))
			iommu_sva_unbind_device(ctx->sva);
		if (ctx->chan && !IS_ERR(ctx->chan))
			dma_release_channel(ctx->chan);
		kfree(ctx);
	}

	return rc;
}

static int dmautest_release(struct inode *inodep, struct file *file)
{
	struct dmautest_ctx *ctx = file->private_data;
	int i, j;

	if (!ctx)
		return 0;

#ifndef ONLY_IO_URING
	dma_release_channel(ctx->chan);
	if (ctx->sva)
		iommu_sva_unbind_device(ctx->sva);
#endif
	if (ctx->k_vecs) {
		for (i = 0; i < pool_size; i++) {
			for (j = 0; j < max_iovecs; j++)
				kfree(ctx->k_vecs[i][j].iov_base);

			kfree(ctx->k_vecs[i]);
		}
		kfree(ctx->k_vecs);
	}
	kfree(ctx);

	return 0;
}

static ssize_t dmautest_read_iter_uring_cpu(struct kiocb *kiocb,
					struct iov_iter *dst)
{
	ssize_t total;
	struct file *file = kiocb->ki_filp;
	struct dmautest_ctx *ctx = file->private_data;
	int n, i;
	unsigned int idx;
	struct kvec *k_vecs;

	total = iov_iter_count(dst);

	idx = ctx->cur_element;
	k_vecs = ctx->k_vecs[idx++];
	if (idx >= pool_size)
		idx = 0;
	ctx->cur_element = idx;

	i = 0;
	while (iov_iter_count(dst) > 0) {
		void *buf = k_vecs[i].iov_base;
		int len = k_vecs[i].iov_len;

		n = copy_to_iter(buf, len, dst);

		i++;

		if (i == max_iovecs)
			i = 0;
	}

	return total;
}

static ssize_t dmautest_read_iter_uring(struct kiocb *kiocb,
					struct iov_iter *dst)
{
	ssize_t len;
	struct file *file = kiocb->ki_filp;
	struct dmautest_ctx *ctx = file->private_data;
	int rc;
	unsigned int idx;
	struct iov_iter src;
	struct kvec *k_vecs;

	len = 0;

	idx = ctx->cur_element;
	k_vecs = ctx->k_vecs[idx++];
	if (idx >= pool_size)
		idx = 0;
	ctx->cur_element = idx;

	while (iov_iter_count(dst) > 0) {
		iov_iter_kvec(&src, READ, k_vecs, max_iovecs, ctx->total_len);
		rc = io_uring_copy_to_iter(kiocb, dst, &src, NULL, NULL, 0);
		if (rc <= 0)
			break;

		len += rc;

		iov_iter_advance(dst, rc);
	}

	return len;
}

static ssize_t dmautest_read_iter(struct kiocb *kiocb, struct iov_iter *u)
{
	struct file *file = kiocb->ki_filp;
	struct dmautest_ctx *ctx = file->private_data;
	struct iov_iter k;
	ssize_t offset, len;

#ifdef ONLY_IO_URING
	if (kiocb->ki_flags & IOCB_DMA_COPY) {
		/* io_uring has already set up an offload context for this
		 * operation, so use that.
		 */
		return dmautest_read_iter_uring(kiocb, u);
	} else 
		return dmautest_read_iter_uring_cpu(kiocb, u);

#endif

	iov_iter_kvec(&k, READ, ctx->k_vecs[0], max_iovecs, ctx->total_len);

	offset = 0;
	while (offset < len) {
		dma_cookie_t cookie;
		size_t tx_len;
		int status;
		struct dma_async_tx_descriptor *tx;
		unsigned long dma_sync_wait_timeout = jiffies + msecs_to_jiffies(5000);

		iov_iter_kvec(&k, READ, ctx->k_vecs[0], max_iovecs, ctx->total_len);

		tx_len = len - offset;
		if (tx_len > max_iovecs * buf_size)
			tx_len = max_iovecs * buf_size;

		tx = dmaengine_prep_memcpy_sva_kernel_user(ctx->chan,
				u, &k, 0);

		if (!tx)
			return -EFAULT;

		cookie = dmaengine_submit(tx);
		if (dma_submit_error(cookie))
			return offset;

		dma_async_issue_pending(ctx->chan);
		do {
			status = dmaengine_async_is_tx_complete(ctx->chan, cookie);
			if (time_after_eq(jiffies, dma_sync_wait_timeout))
				return -ETIMEDOUT;

			if (status == DMA_COMPLETE || status == DMA_ERROR)
				break;
			cpu_relax();
		} while (1);

		if (status == DMA_ERROR)
			break;

		offset += tx_len;
	}

	iov_iter_kvec(&k, READ, ctx->k_vecs[0], max_iovecs, ctx->total_len);

	return offset;
}

static ssize_t dmautest_read(struct file *file, char __user *buf,
			     size_t len, loff_t *ppos)
{
#ifdef ONLY_IO_URING
	return len;
#else
	void *src;
	void __user *dst;
	size_t offset;
	struct dmautest_ctx *ctx = file->private_data;
	struct device *dev = ctx->chan->device->dev;

	/* TODO: ppos needs to be taken into account */
	offset = 0;
	while (offset < len) {
		dma_cookie_t cookie;
		size_t tx_len;
		int status;
		struct dma_async_tx_descriptor *tx;
		unsigned long dma_sync_wait_timeout = jiffies + msecs_to_jiffies(5000);

		tx_len = len - offset;
		if (tx_len > buf_size)
			tx_len = buf_size;

		tx = dmaengine_prep_memcpy_sva_single_kernel_user(ctx->chan,
				dst + offset, src, tx_len, 0);

		if (!tx)
			return -EFAULT;

		cookie = dmaengine_submit(tx);
		if (dma_submit_error(cookie))
			return offset;

		dma_async_issue_pending(ctx->chan);
		do {
			status = dmaengine_async_is_tx_complete(ctx->chan, cookie);
			if (time_after_eq(jiffies, dma_sync_wait_timeout))
				return -ETIMEDOUT;

			if (status == DMA_COMPLETE || status == DMA_ERROR)
				break;
			cpu_relax();
		} while (1);

		if (status == DMA_ERROR)
			break;

		offset += tx_len;
	}

	return offset;
#endif
}

static ssize_t dmautest_write(struct file *file, const char __user *buf,
			      size_t len, loff_t *ppos)
{
	return len;
}

static ssize_t dmautest_write_iter(struct kiocb *kiocb, struct iov_iter *u)
{
	return iov_iter_count(u);
}

static const struct file_operations dmautest_fops = {
	.owner			= THIS_MODULE,
	.write			= dmautest_write,
	.read			= dmautest_read,
	.write_iter		= dmautest_write_iter,
	.read_iter		= dmautest_read_iter,
	.open			= dmautest_open,
	.release		= dmautest_release,
	.llseek			= no_llseek,
};

struct miscdevice dmautest_device = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "dmautest",
	.fops = &dmautest_fops,
};

static int __init dmautest_init(void)
{
	int error;

	error = misc_register(&dmautest_device);
	if (error)
		return error;

	return 0;
}

static void __exit dmautest_exit(void)
{
	misc_deregister(&dmautest_device);
}

module_init(dmautest_init)
module_exit(dmautest_exit)

MODULE_DESCRIPTION("DMA engine copy-to-user tester");
MODULE_AUTHOR("Ben Walker <benjamin.walker@intel.com>");
MODULE_LICENSE("GPL");
