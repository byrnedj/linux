// SPDX-License-Identifier: GPL-2.0
/*
 * Reuse-time sampler for the PFN cache.
 *
 * The input a miss ratio curve needs, gathered on the datapath and
 * exported for an offline AET solve; the first stage of sizing the
 * cache from a curve instead of a heuristic. It measures, it decides
 * nothing.
 *
 * An access is one segment touch: the first lookup an op makes on a
 * segment. Later chunks of the same segment in the same op are the
 * same cache access, and counting them would pile the histogram onto
 * reuse time 1 and hide the knee. Keys are sampled by hash, 1 in
 * 2^shift, so both ends of a reuse pair are seen without a table
 * lookup on the unsampled majority; the clock counts every touch.
 * Reuse times land in a domain-compressed histogram, 256 linear
 * buckets per octave, the bucketing libmrc uses, so the export is
 * an .rtd file it can read. The first sighting of a sampled key is a
 * cold miss in bucket 0; sample_rate * rtd[0] estimates the unique
 * segments.
 */
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/hash.h>
#include <linux/xarray.h>
#include <linux/seq_file.h>
#include <linux/atomic.h>
#include <linux/bitops.h>
#include <linux/bits.h>
#include <linux/limits.h>
#include "pfn_mrc.h"

#define IO_PFN_MRC_DOMAIN	256
#define IO_PFN_MRC_OCTAVES	32
#define IO_PFN_MRC_BUCKETS	(IO_PFN_MRC_DOMAIN * IO_PFN_MRC_OCTAVES + 1)

struct io_pfn_mrc {
	atomic64_t	clock;		/* segment touches */
	atomic64_t	samples;	/* sampled touches, reuse and cold */
	atomic64_t	cold;		/* first sightings of sampled keys */
	atomic64_t	dropped;	/* sampled touches the table lost */
	atomic64_t	max_rt;
	struct xarray	tab;		/* sampled key -> clock at last touch */
	atomic_t	rtd[IO_PFN_MRC_BUCKETS];
};

/*
 * Reuse-time value to bucket, libmrc's domain compression: bucket 0
 * is the cold miss, values up to the domain are linear, then each
 * octave gets IO_PFN_MRC_DOMAIN buckets of doubling width. Closed
 * form of the reference loop, as in libmrc's value_to_bucket().
 */
static unsigned int io_pfn_mrc_bucket(u64 value)
{
	const u64 d = IO_PFN_MRC_DOMAIN;
	u64 q, loc, step;
	unsigned int k;

	if (value <= d)
		return value;
	q = (value + d - 1) / d;
	k = fls64(q) - 1;
	while (k > 0 && d * ((2ULL << (k - 1)) - 1) + (1ULL << (k - 1)) * d >= value)
		k--;
	while (d * ((2ULL << k) - 1) < value)
		k++;
	if (k >= IO_PFN_MRC_OCTAVES)
		return IO_PFN_MRC_BUCKETS - 1;
	loc = d * ((1ULL << k) - 1);
	step = 1ULL << k;
	return k * d + (value - loc + step - 1) / step;
}

/* Bucket to its upper-edge value, the inverse libmrc prints per row. */
static u64 io_pfn_mrc_bucket_value(unsigned int index)
{
	const u64 d = IO_PFN_MRC_DOMAIN;
	unsigned int j;

	if (index <= d)
		return index;
	j = (index - 1) / d;
	return d * ((1ULL << j) - 1) + (1ULL << j) * (index - d * j);
}

struct io_pfn_mrc *io_pfn_mrc_alloc(gfp_t gfp)
{
	struct io_pfn_mrc *m = kzalloc(sizeof(*m), gfp);

	if (!m)
		return NULL;
	xa_init(&m->tab);
	return m;
}

void io_pfn_mrc_free(struct io_pfn_mrc *m)
{
	if (!m)
		return;
	xa_destroy(&m->tab);
	kfree(m);
}

void io_pfn_mrc_touch(struct io_pfn_mrc *m, unsigned long key,
		      unsigned int shift)
{
	u64 now = atomic64_inc_return(&m->clock);
	u64 last, rt, max;
	void *v;

	if (hash_64(key, shift))
		return;

	rcu_read_lock();
	v = xa_load(&m->tab, key);
	rcu_read_unlock();
	if (v) {
		last = xa_to_value(v);
		rt = now > last ? now - last : 1;
		atomic_inc(&m->rtd[io_pfn_mrc_bucket(rt)]);
		max = atomic64_read(&m->max_rt);
		while (rt > max &&
		       !atomic64_try_cmpxchg(&m->max_rt, &max, rt))
			;
	} else {
		atomic_inc(&m->rtd[0]);
		atomic64_inc(&m->cold);
	}
	/*
	 * Values carry the clock; a lost store just loses this key's
	 * next reuse, counted so the export can say how many.
	 */
	if (xa_is_err(xa_store(&m->tab, key,
			       xa_mk_value(now & (LONG_MAX >> 1)),
			       GFP_NOWAIT | __GFP_NOWARN)))
		atomic64_inc(&m->dropped);
	atomic64_inc(&m->samples);
}

void io_pfn_mrc_reset(struct io_pfn_mrc *m)
{
	unsigned int i;

	xa_destroy(&m->tab);
	for (i = 0; i < IO_PFN_MRC_BUCKETS; i++)
		atomic_set(&m->rtd[i], 0);
	atomic64_set(&m->clock, 0);
	atomic64_set(&m->samples, 0);
	atomic64_set(&m->cold, 0);
	atomic64_set(&m->dropped, 0);
	atomic64_set(&m->max_rt, 0);
}

/*
 * libmrc .rtd layout after one comment line of sampler state:
 *   header "N,m,writes,max_rt,max_dt,nlow,nhigh"
 *   rows   "value,rtd,wrtd,brtd,avgbwrt,avglow,avghigh"
 * N counts every sampled touch, cold ones included, and m is the
 * unique-segment estimate; the write columns are zero, the sampler
 * does not distinguish directions.
 */
void io_pfn_mrc_show(struct seq_file *s, struct io_pfn_mrc *m,
		     unsigned int shift)
{
	u64 cold = atomic64_read(&m->cold);
	u64 samples = atomic64_read(&m->samples);
	u64 max_rt = atomic64_read(&m->max_rt);
	unsigned int max_dt = 0, i;

	for (i = 1; i < IO_PFN_MRC_BUCKETS; i++)
		if (atomic_read(&m->rtd[i]))
			max_dt = i;

	seq_printf(s, "# mrc shift %u clock %llu samples %llu cold %llu dropped %llu max_rt %llu max_dt %u\n",
		   shift, (u64)atomic64_read(&m->clock), samples, cold,
		   (u64)atomic64_read(&m->dropped), max_rt, max_dt);
	seq_printf(s, "%llu,%llu,0,%llu,%u,0,0\n",
		   samples, cold << shift, max_rt, max_dt);
	for (i = 1; i <= max_dt; i++)
		seq_printf(s, "%llu,%u,0,0,0,0,0\n",
			   io_pfn_mrc_bucket_value(i),
			   (unsigned int)atomic_read(&m->rtd[i]));
}
