/* thp_zero_bench - measure THP/hugetlb fault-time zeroing.
 *
 * Each thread mmaps its own anonymous region, madvises it to huge
 * pages (or uses MAP_HUGETLB with -H), then touches one byte per 2MB
 * block: every touch is a huge page fault whose dominant cost is
 * zeroing the folio. Reports wall time, user/sys CPU time, and
 * per-fault latency. With -v every byte of the region is checked to
 * be zero afterwards (except the touched offsets).
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

static unsigned long BLOCK = 2UL << 20;

static unsigned long mb_per_thread = 4096;
static int nthreads = 1;
static int use_hugetlb;
static int use_1g;
static int verify;
static unsigned long touch_off = 64 << 10;  /* offset of touch in block */

static pthread_barrier_t barrier;

struct thr {
	pthread_t tid;
	double wall_s;
	unsigned long bad_bytes;
	unsigned long huge_faults;
};

static double now_s(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void *worker(void *arg)
{
	struct thr *t = arg;
	size_t size = mb_per_thread << 20;
	unsigned long nblocks = size / BLOCK;
	int flags = MAP_PRIVATE | MAP_ANONYMOUS;
	unsigned char *p;
	double t0;

	if (use_hugetlb)
		flags |= MAP_HUGETLB;
	if (use_1g)
		flags |= MAP_HUGE_1GB;

	p = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, -1, 0);
	if (p == MAP_FAILED) {
		perror("mmap");
		exit(1);
	}
	if (!use_hugetlb && madvise(p, size, MADV_HUGEPAGE)) {
		perror("madvise");
		exit(1);
	}

	pthread_barrier_wait(&barrier);
	t0 = now_s();
	for (unsigned long b = 0; b < nblocks; b++)
		p[b * BLOCK + touch_off] = 0xAA;
	t->wall_s = now_s() - t0;
	t->huge_faults = nblocks;

	if (verify) {
		for (unsigned long i = 0; i < size; i++) {
			unsigned long boff = i % BLOCK;
			unsigned char want = (boff == touch_off) ? 0xAA : 0;

			if (p[i] != want)
				t->bad_bytes++;
		}
	}

	/* Keep mappings alive until main has sampled THP residency. */
	pthread_barrier_wait(&barrier);
	pthread_barrier_wait(&barrier);
	munmap(p, size);
	return NULL;
}

static unsigned long anon_huge_kb(void)
{
	char line[128];
	unsigned long kb = 0;
	FILE *f = fopen("/proc/self/smaps_rollup", "r");

	if (!f)
		return 0;
	while (fgets(line, sizeof(line), f))
		if (sscanf(line, "AnonHugePages: %lu kB", &kb) == 1)
			break;
	fclose(f);
	return kb;
}

int main(int argc, char **argv)
{
	struct rusage ru;
	double wall_max = 0, user_s, sys_s;
	unsigned long faults = 0, bad = 0;
	struct thr *thr;
	int opt;

	while ((opt = getopt(argc, argv, "s:t:o:HGv")) != -1) {
		switch (opt) {
		case 's': mb_per_thread = strtoul(optarg, NULL, 0); break;
		case 't': nthreads = atoi(optarg); break;
		case 'o': touch_off = strtoul(optarg, NULL, 0); break;
		case 'H': use_hugetlb = 1; break;
		case 'G': use_hugetlb = 1; use_1g = 1; BLOCK = 1UL << 30; break;
		case 'v': verify = 1; break;
		default:
			fprintf(stderr,
				"usage: %s [-s mb/thread] [-t threads] [-o touch_off] [-H hugetlb] [-G 1G-hugetlb] [-v verify]\n",
				argv[0]);
			return 1;
		}
	}

	unsigned long huge_kb;

	thr = calloc(nthreads, sizeof(*thr));
	pthread_barrier_init(&barrier, NULL, nthreads + 1);

	for (int i = 0; i < nthreads; i++)
		pthread_create(&thr[i].tid, NULL, worker, &thr[i]);

	pthread_barrier_wait(&barrier);	/* start */
	pthread_barrier_wait(&barrier);	/* workers done touching+verify */
	huge_kb = anon_huge_kb();
	pthread_barrier_wait(&barrier);	/* let workers unmap */

	for (int i = 0; i < nthreads; i++) {
		pthread_join(thr[i].tid, NULL);
		if (thr[i].wall_s > wall_max)
			wall_max = thr[i].wall_s;
		faults += thr[i].huge_faults;
		bad += thr[i].bad_bytes;
	}

	getrusage(RUSAGE_SELF, &ru);
	user_s = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6;
	sys_s = ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1e6;

	printf("threads=%d mb/thread=%lu faults=%lu thp_mb=%lu wall=%.3fs "
	       "user=%.2fs sys=%.2fs us/fault=%.1f GB/s=%.2f%s\n",
	       nthreads, mb_per_thread, faults, huge_kb >> 10,
	       wall_max, user_s, sys_s,
	       wall_max * 1e6 / faults,
	       (double)faults * BLOCK / wall_max / (1 << 30),
	       verify ? (bad ? " VERIFY-FAIL" : " verify-ok") : "");
	if (bad) {
		fprintf(stderr, "!! %lu unexpected bytes\n", bad);
		return 2;
	}
	return 0;
}
