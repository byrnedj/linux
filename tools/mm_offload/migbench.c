// SPDX-License-Identifier: GPL-2.0
/*
 * migbench - move_pages(2) microbenchmark for page migration offload.
 *
 * Allocates a buffer on one NUMA node, fills it with a pattern and
 * migrates it to another node with move_pages(2), timing the syscall
 * only. Optionally verifies the contents afterwards and moves it back
 * for the next iteration.
 *
 *   migbench -s <MiB> -f <src node> -t <dst node> [-T threads] [-i iters]
 *            [-H] [-o order] [-v] [-b batch]
 *
 *   -H   MADV_HUGEPAGE (PMD folios); default is MADV_NOHUGEPAGE
 *   -o   folio order to hand to move_pages (address stride = 4K << order)
 *   -b   addresses per move_pages call (default: all)
 *   -v   verify the pattern after each migration
 */
#define _GNU_SOURCE
#include <errno.h>
#include <numaif.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static size_t size_mb = 1024;
static int src_node = 0, dst_node = 1, nthreads = 1, iters = 3;
static int hugepage, order, verify;
static unsigned long batch;

struct thr {
	int id;
	char *buf;
	size_t len;
	void **pages;
	int *nodes, *status;
	unsigned long npages;
	double secs;
	long fails;
	int to_node;
};

static double now(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void fill(char *buf, size_t len, unsigned long seed)
{
	uint64_t *p = (uint64_t *)buf;
	size_t i;

	for (i = 0; i < len / 8; i++)
		p[i] = seed ^ (i * 0x9E3779B97F4A7C15ULL);
}

static long check(const char *buf, size_t len, unsigned long seed)
{
	const uint64_t *p = (const uint64_t *)buf;
	size_t i;
	long bad = 0;

	for (i = 0; i < len / 8; i++)
		if (p[i] != (seed ^ (i * 0x9E3779B97F4A7C15ULL)))
			bad++;
	return bad;
}

static void *worker(void *arg)
{
	struct thr *t = arg;
	unsigned long done = 0, n;
	double t0;
	int rc;

	for (n = 0; n < t->npages; n++)
		t->nodes[n] = t->to_node;

	t0 = now();
	while (done < t->npages) {
		n = t->npages - done;
		if (batch && n > batch)
			n = batch;
		rc = syscall(SYS_move_pages, 0, n, t->pages + done,
			     t->nodes + done, t->status + done, MPOL_MF_MOVE);
		if (rc < 0) {
			t->fails += n;
			break;
		}
		done += n;
	}
	t->secs = now() - t0;
	for (n = 0; n < t->npages; n++)
		if (t->status[n] != t->to_node && t->status[n] != -EBUSY)
			t->fails++;
	return NULL;
}

int main(int argc, char **argv)
{
	size_t len, stride, per_thread;
	struct thr *thr;
	pthread_t *tid;
	char *buf;
	int c, it, i;
	unsigned long nodemask;
	struct rusage ru0, ru1;

	while ((c = getopt(argc, argv, "s:f:t:T:i:Ho:vb:")) != -1) {
		switch (c) {
		case 's': size_mb = strtoul(optarg, NULL, 0); break;
		case 'f': src_node = atoi(optarg); break;
		case 't': dst_node = atoi(optarg); break;
		case 'T': nthreads = atoi(optarg); break;
		case 'i': iters = atoi(optarg); break;
		case 'H': hugepage = 1; break;
		case 'o': order = atoi(optarg); break;
		case 'v': verify = 1; break;
		case 'b': batch = strtoul(optarg, NULL, 0); break;
		default:
			fprintf(stderr, "usage: %s -s MiB -f src -t dst [-T thr] [-i iters] [-H] [-o order] [-v] [-b batch]\n", argv[0]);
			return 1;
		}
	}
	if (hugepage && !order)
		order = 9;

	len = size_mb << 20;
	stride = 4096UL << order;
	buf = mmap(NULL, len, PROT_READ | PROT_WRITE,
		   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (buf == MAP_FAILED) {
		perror("mmap");
		return 1;
	}
	madvise(buf, len, hugepage ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
	nodemask = 1UL << src_node;
	if (mbind(buf, len, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, 0)) {
		perror("mbind");
		return 1;
	}
	fill(buf, len, 0x1234);

	thr = calloc(nthreads, sizeof(*thr));
	tid = calloc(nthreads, sizeof(*tid));
	per_thread = (len / nthreads) & ~(stride - 1);
	for (i = 0; i < nthreads; i++) {
		unsigned long n;

		thr[i].id = i;
		thr[i].buf = buf + i * per_thread;
		thr[i].len = per_thread;
		thr[i].npages = per_thread / stride;
		thr[i].pages = calloc(thr[i].npages, sizeof(void *));
		thr[i].nodes = calloc(thr[i].npages, sizeof(int));
		thr[i].status = calloc(thr[i].npages, sizeof(int));
		for (n = 0; n < thr[i].npages; n++)
			thr[i].pages[n] = thr[i].buf + n * stride;
	}

	printf("migbench: %zu MiB node %d -> %d, %d threads, order %d, stride %zu KiB, %lu addrs/thread%s\n",
	       size_mb, src_node, dst_node, nthreads, order, stride >> 10,
	       thr[0].npages, hugepage ? ", THP" : "");

	for (it = 0; it < iters; it++) {
		int from = it & 1 ? dst_node : src_node;
		int to = it & 1 ? src_node : dst_node;
		double wall, maxs = 0, sys, usr;
		long fails = 0, bad = 0;

		for (i = 0; i < nthreads; i++)
			thr[i].to_node = to;
		getrusage(RUSAGE_SELF, &ru0);
		wall = now();
		for (i = 0; i < nthreads; i++)
			pthread_create(&tid[i], NULL, worker, &thr[i]);
		for (i = 0; i < nthreads; i++) {
			pthread_join(tid[i], NULL);
			if (thr[i].secs > maxs)
				maxs = thr[i].secs;
			fails += thr[i].fails;
			thr[i].fails = 0;
		}
		wall = now() - wall;
		getrusage(RUSAGE_SELF, &ru1);
		sys = (ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec) +
		      (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec) / 1e6;
		usr = (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec) +
		      (ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) / 1e6;
		if (verify)
			bad = check(buf, len, 0x1234);
		printf("iter %d: %d->%d wall %.3f s (max thread %.3f) = %.2f GiB/s, sys %.3f s usr %.3f s, fails %ld, bad %ld\n",
		       it, from, to, wall, maxs,
		       (double)len / (1UL << 30) / maxs, sys, usr, fails, bad);
	}
	return 0;
}
