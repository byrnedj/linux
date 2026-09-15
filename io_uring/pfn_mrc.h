/* SPDX-License-Identifier: GPL-2.0 */
#ifndef IOU_PFN_MRC_H
#define IOU_PFN_MRC_H

#include <linux/types.h>
#include <linux/gfp_types.h>

struct seq_file;
struct io_pfn_mrc;

struct io_pfn_mrc *io_pfn_mrc_alloc(gfp_t gfp);
void io_pfn_mrc_free(struct io_pfn_mrc *m);
void io_pfn_mrc_touch(struct io_pfn_mrc *m, unsigned long key,
		      unsigned int shift);
void io_pfn_mrc_reset(struct io_pfn_mrc *m);
void io_pfn_mrc_show(struct seq_file *s, struct io_pfn_mrc *m,
		     unsigned int shift);

#endif
