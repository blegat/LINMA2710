/*
 * Benchmark for Q3: R = Q * K^T   with Q, K, R in R^{n x n}
 *
 * Tests:
 *   - IJK  (as printed in q3.tex)
 *   - IKJ  (as printed in q3.tex)
 *   - Tiling  (as printed in q3.tex, AND a fixed version)
 *
 * Build:
 *   gcc -O3 -std=gnu11 -Wall -o bench bench.c -lm
 *   (Use -O3 — at -O2 gcc fails to scalar-promote BR[i*n+j] across the
 *    innermost k loop in the tiling kernel, which artificially hides the
 *    cache-tiling speedup behind a memory-dependency chain.)
 *
 * Run:
 *   ./bench           # default n = 1024, b = 64
 *   ./bench 2048 64   # n = 2048, b = 64
 *
 * The benchmark also (a) verifies correctness against a reference,
 * and (b) sweeps the tile size b to confirm the prediction from part (b).
 *
 * NOTE on the exam code:
 *  - q3.tex says "32-bit floating point" but the printed C uses `double`.
 *    This benchmark uses `float` to match the question wording. The cache
 *    analysis (16 floats per line) only matches with `float`.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------- kernels ---------- */

#include "ijk.c"
#include "ikj.c"
#include "block.c"

/* ---------- helpers ---------- */

static void fill_random(float *A, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n * n; ++i)
        A[i] = (float)rand() / RAND_MAX - 0.5f;
}

static double max_abs_diff(const float *A, const float *B, int n) {
    double m = 0.0;
    for (int i = 0; i < n * n; ++i) {
        double d = fabs((double)A[i] - (double)B[i]);
        if (d > m) m = d;
    }
    return m;
}

static double time_kernel(void (*f)(const float*, const float*, float*, int),
                          const float *Q, const float *K, float *R, int n) {
    /* Warm-up + one timed run. For a real measurement you'd want multiple
       runs and a median; one run is enough to see the order-of-magnitude
       differences this exercise is about. */
    f(Q, K, R, n);
    double t0 = now_sec();
    f(Q, K, R, n);
    double t1 = now_sec();
    return t1 - t0;
}

static double time_block(void (*f)(const float*, const float*, float*, int, int),
                         const float *Q, const float *K, float *R, int n, int b) {
    f(Q, K, R, n, b);
    double t0 = now_sec();
    f(Q, K, R, n, b);
    double t1 = now_sec();
    return t1 - t0;
}

/* ---------- main ---------- */

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 1024;
    int b = (argc > 2) ? atoi(argv[2]) : 64;

    if (n % b != 0) {
        fprintf(stderr, "n (%d) must be a multiple of b (%d)\n", n, b);
        return 1;
    }

    size_t bytes = (size_t)n * n * sizeof(float);
    float *Q  = aligned_alloc(64, bytes);
    float *K  = aligned_alloc(64, bytes);
    float *R0 = aligned_alloc(64, bytes);  /* reference (IJK) */
    float *R1 = aligned_alloc(64, bytes);
    float *R2 = aligned_alloc(64, bytes);
    if (!Q || !K || !R0 || !R1 || !R2) {
        fprintf(stderr, "alloc failed (n=%d => %.1f MiB per matrix)\n",
                n, bytes / (1024.0 * 1024.0));
        return 1;
    }

    fill_random(Q, n, 1);
    fill_random(K, n, 2);

    double gflops = 2.0 * (double)n * n * n / 1e9;

    printf("n = %d, b = %d, 2n^3 = %.2f Gflop\n", n, b, gflops);
    printf("kernel             time (s)   Gflop/s   speedup vs IJK   max|diff|\n");
    printf("------------------------------------------------------------------\n");

    /* IJK (reference) */
    double t_ijk = time_kernel(ijk, Q, K, R0, n);
    printf("IJK              %9.3f  %8.2f   %14s  %10s\n",
           t_ijk, gflops / t_ijk, "1.00x", "(ref)");

    /* IKJ */
    double t_ikj = time_kernel(ikj, Q, K, R1, n);
    double err_ikj = max_abs_diff(R0, R1, n);
    printf("IKJ              %9.3f  %8.2f   %13.2fx  %10.2e\n",
           t_ikj, gflops / t_ikj, t_ijk / t_ikj, err_ikj);

    /* Tiling */
    double t_b = time_block(block_tile, Q, K, R2, n, b);
    double err_b = max_abs_diff(R0, R2, n);
    printf("Tiling           %9.3f  %8.2f   %13.2fx  %10.2e\n",
           t_b, gflops / t_b, t_ijk / t_b, err_b);

    /* Sweep b to confirm the L1-fitting prediction. We expect a U-shape
       around b ~ 64 (since 3*b^2*4 <= 64 KiB means b <= 73). */
    printf("\nTile-size sweep:\n");
    printf("   b   3*b^2*4 (KiB)    time (s)   Gflop/s\n");
    int bs[] = {8, 16, 32, 48, 64, 80, 128, 256};
    for (size_t i = 0; i < sizeof(bs)/sizeof(bs[0]); ++i) {
        int bb = bs[i];
        if (n % bb != 0) continue;
        double kib = 3.0 * bb * bb * 4.0 / 1024.0;
        double t = time_block(block_tile, Q, K, R2, n, bb);
        printf("  %3d   %12.1f   %9.3f  %8.2f\n", bb, kib, t, gflops / t);
    }

    free(Q); free(K); free(R0); free(R1); free(R2);
    return 0;
}
