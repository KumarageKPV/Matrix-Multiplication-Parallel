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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>

#ifdef _MSC_VER
#include <malloc.h> /* for _aligned_malloc / _aligned_free */
#endif

/* Cross-platform aligned allocation */
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

/* initialize matrix with deterministic pseudo-random values */
void init_matrix(double *M, int n, int seed_offset) {
    uint32_t state = 1234567u + (uint32_t)seed_offset;
    for (int i = 0; i < n * n; ++i) {
        state = state * 1103515245u + 12345u;
        M[i] = (double)(state % 1000) / 1000.0;
    }
}

/* naive matrix multiply */
void matmul_naive(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

/* blocked (tiled) multiply */
void matmul_blocked(const double *A, const double *B, double *C, int n, int bs) {
    for (int i = 0; i < n * n; ++i) C[i] = 0.0;

    for (int ii = 0; ii < n; ii += bs) {
        int iimax = (ii + bs < n) ? ii + bs : n;
        for (int kk = 0; kk < n; kk += bs) {
            int kkmax = (kk + bs < n) ? kk + bs : n;
            for (int jj = 0; jj < n; jj += bs) {
                int jjmax = (jj + bs < n) ? jj + bs : n;
                for (int i = ii; i < iimax; ++i) {
                    for (int k = kk; k < kkmax; ++k) {
                        double a_ik = A[i*n + k];
                        for (int j = jj; j < jjmax; ++j) {
                            C[i*n + j] += a_ik * B[k*n + j];
                        }
                    }
                }
            }
        }
    }
}

/* quick checksum */
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

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (bs <= 0)
        matmul_naive(A, B, C, N);
    else
        matmul_blocked(A, B, C, N, bs);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
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
