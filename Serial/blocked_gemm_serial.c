/* blocked_gemm_serial.c
 * Serial (single-threaded) baseline for dense matrix multiplication.
 *
 * Compile:
 *   gcc -O2 -std=c11 -o blocked_gemm_serial blocked_gemm_serial.c -lm
 * Run:
 *   ./blocked_gemm_serial <N> <block_size>
 *   e.g. ./blocked_gemm_serial 512 64
 * If block_size <= 0, the code runs naive triple-loop multiply.
 */

#ifndef _MSC_VER
#define _POSIX_C_SOURCE 199309L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>

#ifdef _MSC_VER
#include <malloc.h> /* for _aligned_malloc / _aligned_free */
#include <windows.h> /* for QueryPerformanceCounter */
#endif

/*
 * Cross-platform aligned matrix allocation.
 * 
 * Allocates an n×n matrix with 64-byte alignment for optimal performance.
 * 
 * Why 64-byte alignment?
 * - Matches typical cache line size (prevents false sharing in parallel code)
 * - Enables efficient SIMD operations (AVX-512 requires 64-byte alignment)
 * - Improves hardware prefetching behavior
 * - Reduces cache misses from unaligned access
 * 
 * Platform-specific:
 * - Windows (MSVC): Uses _aligned_malloc / _aligned_free
 * - POSIX: Uses standard malloc (typically 16-byte aligned for double)
 */
static inline double *alloc_matrix(int n) {
    size_t bytes = sizeof(double) * (size_t)n * n;
    double *m = NULL;

#ifdef _MSC_VER
    m = (double*)_aligned_malloc(bytes, 64);
    if (!m) {
        fprintf(stderr, "aligned malloc failed\n");
        exit(EXIT_FAILURE);
    }
#else
    /* On non-Windows, fall back to malloc (alignment usually sufficient for doubles) */
    m = (double*)malloc(bytes);
    if (!m) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
#endif

    return m;
}

/*
 * Cross-platform high-resolution timer.
 * 
 * Returns wall-clock time in seconds with nanosecond precision.
 * 
 * Platform-specific:
 * - Windows: Uses QueryPerformanceCounter (~100 ns resolution)
 * - POSIX: Uses clock_gettime(CLOCK_MONOTONIC) (~1 ns resolution)
 */
static inline double wall_time_seconds(void) {
#ifdef _MSC_VER
    static LARGE_INTEGER frequency;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#endif
}

/*
 * Deterministic matrix initialization using Linear Congruential Generator (LCG).
 * 
 * Why deterministic?
 * - Ensures reproducibility across runs for verification
 * - Enables exact comparison between serial/OpenMP/MPI/CUDA implementations
 * - No need to save/load large matrix files for testing
 * 
 * LCG parameters (from glibc rand()):
 * - Multiplier a = 1103515245
 * - Increment c = 12345
 * - Produces values in [0.0, 1.0) range
 * 
 * seed_offset allows generating different matrices:
 * - A uses seed_offset=1
 * - B uses seed_offset=2
 * - Ensures A ≠ B while maintaining determinism
 */
void init_matrix(double *M, int n, int seed_offset) {
    uint32_t state = 1234567u + (uint32_t)seed_offset;
    for (int i = 0; i < n * n; ++i) {
        state = state * 1103515245u + 12345u;
        M[i] = (double)(state % 1000) / 1000.0;
    }
}

/*
 * Naive triple-loop matrix multiplication: C = A × B
 * 
 * Algorithm: Standard ijk ordering
 * - Computes each C[i,j] as dot product of A[i,:] and B[:,j]
 * - O(N³) floating-point operations (2N³ FLOPs for multiply-add)
 * 
 * Performance characteristics:
 * - Poor cache locality: B accessed in column-major order (strided)
 * - Each B element loaded N times (no temporal reuse)
 * - High cache miss rate for large N
 * - Memory bandwidth bound rather than compute bound
 * 
 * Used as:
 * - Baseline for performance comparison
 * - Reference for correctness verification (N ≤ 256)
 * - Demonstrates need for blocking optimization
 */
void matmul_naive(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            // Compute dot product: C[i,j] = A[i,:] · B[:,j]
            for (int k = 0; k < n; ++k) {
                sum += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

/*
 * Blocked (tiled) matrix multiplication for improved cache locality.
 * 
 * Strategy:
 * 1. Partition N×N matrices into bs×bs blocks (tiles)
 * 2. Process one tile at a time to maximize cache reuse
 * 3. Perform mini-GEMM on each tile pair
 * 
 * Cache optimization:
 * - Tiles fit in L1/L2 cache (bs² × sizeof(double) bytes per tile)
 * - Each tile element reused bs times (vs. N times in naive)
 * - Reduces memory traffic by factor of bs/cache_line_size
 * - Typical speedup: 2-5× for N=512-2048, bs=32-128
 * 
 * Loop structure (outer to inner):
 * - ii, kk, jj: Tile indices (step by bs)
 * - i, k, j: Element indices within tiles
 * - ikj ordering chosen for:
 *   * Good spatial locality in C (row-major writes)
 *   * Register reuse of A[i,k] across j loop
 *   * Sequential B[k,j] access (prefetcher-friendly)
 * 
 * Complexity:
 * - Still O(N³) FLOPs, but with O(N³/bs) cache misses instead of O(N³)
 * - Performance limited by memory bandwidth for large N
 */
void matmul_blocked(const double *A, const double *B, double *C, int n, int bs) {
    // Zero-initialize output matrix
    for (int i = 0; i < n * n; ++i) C[i] = 0.0;

    // Outer loops: iterate over bs×bs tiles
    for (int ii = 0; ii < n; ii += bs) {          // Row tiles of A and C
        int iimax = (ii + bs < n) ? ii + bs : n;  // Handle edge case (N % bs != 0)
        for (int kk = 0; kk < n; kk += bs) {      // Column tiles of A, row tiles of B
            int kkmax = (kk + bs < n) ? kk + bs : n;
            for (int jj = 0; jj < n; jj += bs) {  // Column tiles of B and C
                int jjmax = (jj + bs < n) ? jj + bs : n;
                
                // Inner loops: process bs×bs tile (micro-kernel)
                for (int i = ii; i < iimax; ++i) {
                    for (int k = kk; k < kkmax; ++k) {
                        // Load A[i,k] once and reuse across entire row of B
                        // This exploits register reuse and reduces memory bandwidth
                        double a_ik = A[i*n + k];
                        
                        // Innermost loop: good spatial locality in B and C
                        // Sequential access pattern enables hardware prefetching
                        for (int j = jj; j < jjmax; ++j) {
                            C[i*n + j] += a_ik * B[k*n + j];
                        }
                    }
                }
            }
        }
    }
}

/*
 * Simple checksum for correctness verification.
 * 
 * Computes sum of all matrix elements. Not cryptographically secure,
 * but sufficient for detecting implementation errors.
 * 
 * Properties:
 * - Deterministic for given input
 * - Commutative (order-independent for exact arithmetic)
 * - Must match across all implementations (serial/OpenMP/MPI/CUDA)
 */
double checksum(const double *M, int n) {
    double s = 0.0;
    for (int i = 0; i < n * n; ++i) s += M[i];
    return s;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <N> <block_size (<=0 for naive)>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int bs = atoi(argv[2]);
    if (N <= 0) {
        fprintf(stderr, "Invalid N\n");
        return 1;
    }

    double *A = alloc_matrix(N), *B = alloc_matrix(N), *C = alloc_matrix(N);
    double *C_ref = alloc_matrix(N);

    init_matrix(A, N, 1);
    init_matrix(B, N, 2);

    double t0 = wall_time_seconds();

    if (bs <= 0)
        matmul_naive(A, B, C, N);
    else
        matmul_blocked(A, B, C, N, bs);

    double elapsed = wall_time_seconds() - t0;
    printf("N=%d  block=%d  time=%.6f sec  checksum=%.6f\n", N, bs, elapsed, checksum(C, N));

    if (N <= 256) {
        matmul_naive(A, B, C_ref, N);
        double diff = 0.0;
        for (int i = 0; i < N*N; ++i)
            diff += fabs(C_ref[i] - C[i]);
        printf("verification l1-diff = %.12f\n", diff);
        if (diff > 1e-8)
            fprintf(stderr, "Verification failed (difference %g)\n", diff);
        else
            printf("Verification OK\n");
    } else {
        printf("Skipping verification for large N\n");
    }

#ifdef _MSC_VER
    _aligned_free(A);
    _aligned_free(B);
    _aligned_free(C);
    _aligned_free(C_ref);
#else
    free(A);
    free(B);
    free(C);
    free(C_ref);
#endif

    return 0;
}
