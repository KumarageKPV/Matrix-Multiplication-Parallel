/* blocked_gemm_omp.c
 * OpenMP-parallel dense matrix multiplication with optional blocking.
 *
 * Provides two paths:
 *   - Naive triple-loop GEMM using OpenMP (if block_size <= 0)
 *   - Blocked (tiled) GEMM with OpenMP parallelization for improved locality
 *
 * Includes:
 *   - Cross-platform high-resolution timers
 *   - 64-byte aligned allocation for better SIMD/cache behavior
 *   - Deterministic matrix initialization and lightweight checksum
 *
 * Compile:
 *   gcc -O2 -fopenmp -std=c11 -o blocked_gemm_omp blocked_gemm_omp.c -lm
 *
 * Run:
 *   ./blocked_gemm_omp <N> <block_size> [num_threads]
 *   e.g. ./blocked_gemm_omp 1024 64 8
 *
 * Notes:
 *   - If block_size > N, it is clamped to N.
 *   - If num_threads is omitted, OpenMP uses its default thread count.
 *   - If block_size <= 0, the program switches to naive OpenMP GEMM.
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif
#include <omp.h>

static double wall_time_seconds(void) {
    /*
     * Cross-platform high-resolution wall-clock timer.
     * 
     * Windows: Uses QueryPerformanceCounter (microsecond precision)
     * POSIX:   Uses clock_gettime with CLOCK_MONOTONIC (nanosecond precision)
     * 
     * Returns elapsed time in seconds as a double for easy manipulation.
     */
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#endif
}

static double *alloc_matrix(int n) {
    /*
     * Allocates an n×n matrix with 64-byte alignment.
     * 
     * Why 64-byte alignment?
     * - Matches typical cache line size (prevents false sharing)
     * - Enables efficient SIMD operations (AVX-512 requires 64-byte alignment)
     * - Improves prefetching behavior
     * 
     * Platform-specific:
     * - Windows: _aligned_malloc / _aligned_free
     * - POSIX:   posix_memalign / free
     */
    size_t bytes = (size_t)n * (size_t)n * sizeof(double);
#ifdef _WIN32
    void *ptr = _aligned_malloc(bytes, 64);
    if (!ptr) return NULL;
    return (double*)ptr;
#else
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, bytes) != 0) return NULL;
    return (double*)ptr;
#endif
}

static void free_matrix(double *m) {
#ifdef _WIN32
    _aligned_free(m);
#else
    free(m);
#endif
}

static void init_matrix(double *M, int n, int seed_offset) {
    /*
     * Deterministic matrix initialization using Linear Congruential Generator (LCG).
     * 
     * Why deterministic?
     * - Ensures reproducibility across runs for verification
     * - Matches serial baseline exactly (same seed = same data)
     * - LCG parameters: a=1103515245, c=12345 (from glibc rand())
     * 
     * seed_offset allows generating different A and B matrices:
     * - A uses seed_offset=1
     * - B uses seed_offset=2
     */
    uint32_t state = 1234567u + (uint32_t)seed_offset;
    for (int i = 0; i < n * n; ++i) {
        state = state * 1103515245u + 12345u;
        M[i] = (double)(state % 1000) / 1000.0;  // Values in [0.0, 1.0)
    }
}

static double checksum(const double *M, int n) {
    /*
     * Simple checksum for correctness verification.
     * 
     * Computes the sum of all matrix elements. While not cryptographically
     * secure, it's sufficient for detecting implementation errors.
     * 
     * Must match the serial baseline's checksum exactly for validation.
     */
    double s = 0.0;
    for (int i = 0; i < n * n; ++i) s += M[i];
    return s;
}

static void matmul_blocked_omp(const double *A, const double *B, double *C, int n, int bs) {
    /*
     * Blocked (tiled) GEMM with OpenMP parallelization.
     * 
     * Strategy:
     * 1. Initialize output matrix C to zero using parallel loops
     * 2. Partition the N×N matrices into blocks of size bs×bs
     * 3. Parallelize over tile indices (i0, j0) for load balancing
     * 4. Keep k0 loop sequential within each thread to accumulate partial results
     * 
     * Benefits of blocking:
     * - Improves cache locality by keeping tiles in L1/L2 cache
     * - Reduces main memory traffic (critical for large N)
     * - Allows better prefetching and pipeline utilization
     * 
     * Parallelization approach:
     * - collapse(2) combines i0 and j0 loops into one work pool
     * - Each thread processes independent (i0,j0) tile pairs
     * - No race conditions since each C[i,j] is updated by only one thread
     * - Static scheduling for predictable load distribution
     */

    // Step 1: Zero-initialize output matrix C in parallel
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i*n + j] = 0.0;
        }
    }

    // Step 2: Blocked multiplication with OpenMP parallelization over output tiles
    // collapse(2) merges i0 and j0 loops for better work distribution across threads
#pragma omp parallel for collapse(2) schedule(static)
    for (int i0 = 0; i0 < n; i0 += bs) {          // Outer row tile loop
        for (int j0 = 0; j0 < n; j0 += bs) {      // Outer column tile loop
            for (int k0 = 0; k0 < n; k0 += bs) {  // Outer reduction tile loop (sequential per thread)
                // Compute tile bounds (handle edge cases where N % bs != 0)
                int iMax = i0 + bs < n ? i0 + bs : n;
                int jMax = j0 + bs < n ? j0 + bs : n;
                int kMax = k0 + bs < n ? k0 + bs : n;
                
                // Inner loops process bs×bs tiles (cache-friendly micro-kernel)
                for (int i = i0; i < iMax; ++i) {
                    for (int k = k0; k < kMax; ++k) {
                        // Load A[i,k] once and reuse for entire row of B
                        // This improves register reuse and reduces memory bandwidth
                        double aik = A[i*n + k];
                        
                        // Innermost loop: accumulate into C[i,j] for all j in current tile
                        // Good spatial locality in B and C (row-major access)
                        for (int j = j0; j < jMax; ++j) {
                            C[i*n + j] += aik * B[k*n + j];
                        }
                    }
                }
            }
        }
    }
}

static void matmul_naive_omp(const double *A, const double *B, double *C, int n) {
    /*
     * Naive triple-loop GEMM with OpenMP parallelization.
     * 
     * Strategy:
     * - Parallelize over the outer i loop (rows of C)
     * - Each thread computes independent rows, avoiding race conditions
     * - Inner k loop performs the dot product C[i,j] = sum(A[i,k] * B[k,j])
     * 
     * Performance notes:
     * - Poor cache locality due to column-major access of B (strided memory)
     * - O(N³) work with minimal data reuse
     * - Primarily used as baseline for comparison with blocked version
     */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            // Accumulate dot product: C[i,j] = A[i,:] · B[:,j]
            for (int k = 0; k < n; ++k) {
                sum += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <N> <block_size> [num_threads]\n", argv[0]);
        fprintf(stderr, "If block_size <= 0, runs naive OpenMP GEMM.\n");
        return 1;
    }
    int N = atoi(argv[1]);
    int bs = atoi(argv[2]);
    if (argc >= 4) {
        int nt = atoi(argv[3]);
        if (nt > 0) omp_set_num_threads(nt);
    }

    double *A = alloc_matrix(N);
    double *B = alloc_matrix(N);
    double *C = alloc_matrix(N);
    if (!A || !B || !C) {
        fprintf(stderr, "Allocation failed for N=%d\n", N);
        free_matrix(A); free_matrix(B); free_matrix(C);
        return 2;
    }

    init_matrix(A, N, 1);
    init_matrix(B, N, 2);

    double t0 = wall_time_seconds();
    if (bs <= 0) {
        matmul_naive_omp(A, B, C, N);
    } else {
        // Ensure block size is reasonable
        if (bs > N) bs = N;
        matmul_blocked_omp(A, B, C, N, bs);
    }
    double t1 = wall_time_seconds();

    double elapsed = t1 - t0;
    double chk = checksum(C, N);

    int num_threads = omp_get_max_threads();
    printf("OpenMP GEMM: N=%d block=%d threads=%d time=%.6f sec checksum=%.10f\n", N, bs, num_threads, elapsed, chk);

    free_matrix(A); free_matrix(B); free_matrix(C);
    return 0;
}
