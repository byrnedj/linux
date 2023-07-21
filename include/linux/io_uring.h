/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef _LINUX_IO_URING_H
#define _LINUX_IO_URING_H

#include <linux/sched.h>
#include <linux/xarray.h>
#include <uapi/linux/io_uring.h>

enum io_uring_cmd_flags {
	IO_URING_F_COMPLETE_DEFER	= 1,
	IO_URING_F_UNLOCKED		= 2,
	/* the request is executed from poll, it should not be freed */
	IO_URING_F_MULTISHOT		= 4,
	/* executed by io-wq */
	IO_URING_F_IOWQ			= 8,
	/* int's last bit, sign checks are usually faster than a bit test */
	IO_URING_F_NONBLOCK		= INT_MIN,

	/* ctx state flags, for URING_CMD */
	IO_URING_F_SQE128		= (1 << 8),
	IO_URING_F_CQE32		= (1 << 9),
	IO_URING_F_IOPOLL		= (1 << 10),
};

struct io_uring_cmd {
	struct file	*file;
	const void	*cmd;
	union {
		/* callback to defer completions to task context */
		void (*task_work_cb)(struct io_uring_cmd *cmd, unsigned);
		/* used for polled completion */
		void *cookie;
	};
	u32		cmd_op;
	u32		flags;
	u8		pdu[32]; /* available inline for free use */
};

typedef void (*io_uring_copy_to_iter_cb)(struct kiocb *, void *, int);

#if defined(CONFIG_IO_URING)
void __io_uring_cancel(bool cancel_all);
void __io_uring_free(struct task_struct *tsk);
void io_uring_unreg_ringfd(void);
const char *io_uring_get_opcode(u8 opcode);
bool io_is_uring_fops(struct file *file);

static inline void io_uring_files_cancel(void)
{
	if (current->io_uring)
		__io_uring_cancel(false);
}
static inline void io_uring_task_cancel(void)
{
	if (current->io_uring)
		__io_uring_cancel(true);
}
static inline void io_uring_free(struct task_struct *tsk)
{
	if (tsk->io_uring)
		__io_uring_free(tsk);
}

ssize_t io_uring_copy_to_iter(struct kiocb *iocb, struct iov_iter *dst_iter,
			struct iov_iter *src_iter,
			io_uring_copy_to_iter_cb cb_fn, void *cb_arg,
			unsigned long flags);
#else
static inline void io_uring_task_cancel(void)
{
}
static inline void io_uring_files_cancel(void)
{
}
static inline void io_uring_free(struct task_struct *tsk)
{
}
static inline const char *io_uring_get_opcode(u8 opcode)
{
	return "";
}
static inline bool io_is_uring_fops(struct file *file)
{
	return false;
}
static inline ssize_t io_uring_copy_to_iter(struct kiocb *iocb, struct iov_iter *dst_iter,
			struct iov_iter *src_iter,
			io_uring_copy_to_iter_cb cb_fn, void *cb_arg,
			unsigned long flags)
{
	return -EOPNOTSUPP;
}
#endif

#endif
