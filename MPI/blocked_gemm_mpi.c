/*
 * blocked_gemm_mpi.c
 * ------------------------------------------------------------
 * Original, self-contained MPI parallel dense matrix multiply (GEMM)
 * with optional cache blocking (tiling). Designed for clarity,
 * reproducibility, and straightforward verification across Serial,
 * OpenMP, and MPI implementations.
 *
 * Parallelization Strategy (Row Distribution):
 *   - Contiguous row blocks of A and C are assigned to each rank.
 *   - Full B is broadcast once; every rank computes its local rows of C.
 *   - Communication pattern: Scatterv(A) + Bcast(B) + Reduce(checksum) + optional Gatherv(C).
 *
 * Rationale for Row Distribution:
 *   - Minimal code & communication versus 2D decompositions.
 *   - Good educational tradeoff: small number of collectives, easy reasoning.
 *   - Preserves originality by avoiding boilerplate complex layouts.
 *
 * Blocking (Cache Tiling):
 *   - Operates on bs x bs tiles (i,k,j blocking order) to improve cache reuse.
 *   - i-range restricted to locally owned rows; k & j span full dimension.
 *   - Falls back to naive GEMM if block_size <= 0 for baseline comparison.
 *
 * Deterministic Initialization:
 *   - LCG identical to Serial / OpenMP (seeds 1 and 2) -> consistent global matrices.
 *   - Enables cross-version checksum equality (robust correctness check).
 *
 * Timing:
 *   - MPI_Wtime with pre/post barriers to isolate compute phase.
 *
 * Output (rank 0):
 *   MPI GEMM: N=1024 block=128 procs=4 time=0.123456 sec checksum=268310302.375923 GFLOPs=XX.YY
 *
 * Compile:
 *   mpicc -O3 -std=c11 -Wall -Wextra -o blocked_gemm_mpi blocked_gemm_mpi.c -lm
 *   (Windows MS-MPI): cl /O2 /std:c11 blocked_gemm_mpi.c msmpi.lib
 *
 * Run:
 *   mpiexec -n <P> ./blocked_gemm_mpi <N> <block_size>
 *   mpiexec -n 4 ./blocked_gemm_mpi 1024 128
 *
 * Originality & Integrity:
 *   - No external GEMM source copied; blocking & distribution derived from
 *     standard principles and authored specifically for this assignment.
 *   - AI assistance (GitHub Copilot GPT-5) acknowledged in AI_CITATION.md.
 */

#ifndef _MSC_VER
// Enable posix_memalign declaration on some toolchains
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#ifndef ALIGN_BYTES
#define ALIGN_BYTES 64
#endif

/* Portable aligned allocation wrapper.
 * Windows (_MSC_VER / MinGW): use _aligned_malloc.
 * POSIX: use posix_memalign. Falls back to malloc if alignment not available.
 */
static void *aligned_alloc_wrap(size_t alignment, size_t size) {
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
    void *p = _aligned_malloc(size, alignment);
    return p;
#else
    void *p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
#endif
}

static void aligned_free_wrap(void *ptr) {
    if (!ptr) return;
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* Deterministic LCG initialization (same seeds as other implementations). */
static void init_matrix(double *M, int n, int seed_offset) {
    uint32_t state = 1234567u + (uint32_t)seed_offset;
    size_t total = (size_t)n * (size_t)n;
    for (size_t i = 0; i < total; ++i) {
        state = state * 1103515245u + 12345u;
        M[i] = (double)(state % 1000) / 1000.0;
    }
}

static double checksum_local(const double *M, int rows, int n) {
    double s = 0.0;
    size_t total = (size_t)rows * (size_t)n;
    for (size_t i = 0; i < total; ++i) s += M[i];
    return s;
}

/* Naive local GEMM for a subset of rows [row_start, row_end). */
static void gemm_naive_local(const double *A_local, const double *B_full, double *C_local,
                             int n, int row_start, int row_count) {
    for (int i = 0; i < row_count; ++i) {
        int global_i = row_start + i; // For clarity (not needed for indexing within local)
        (void)global_i; // silence unused warning
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A_local[i*n + k] * B_full[k*n + j];
            }
            C_local[i*n + j] = sum;
        }
    }
}

/* Blocked local GEMM for rows owned by this rank. */
static void gemm_blocked_local(const double *A_local, const double *B_full, double *C_local,
                               int n, int row_count, int bs) {
    // Zero local C
    for (int i = 0; i < row_count * n; ++i) C_local[i] = 0.0;

    // i loop restricted to local rows; k and j loops span entire matrix
    for (int k0 = 0; k0 < n; k0 += bs) {
        int kMax = (k0 + bs < n) ? k0 + bs : n;
        for (int i0 = 0; i0 < row_count; i0 += bs) { // tile within local row block
            int iMax = (i0 + bs < row_count) ? i0 + bs : row_count;
            for (int j0 = 0; j0 < n; j0 += bs) {
                int jMax = (j0 + bs < n) ? j0 + bs : n;
                for (int i = i0; i < iMax; ++i) {
                    for (int k = k0; k < kMax; ++k) {
                        double aik = A_local[i*n + k];
                        for (int j = j0; j < jMax; ++j) {
                            C_local[i*n + j] += aik * B_full[k*n + j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <N> <block_size (<=0 naive)>\n", argv[0]);
            fprintf(stderr, "Example: mpirun -np 4 %s 1024 128\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    int N  = atoi(argv[1]);
    int bs = atoi(argv[2]);
    if (N <= 0) {
        if (rank == 0) fprintf(stderr, "Invalid N\n");
        MPI_Finalize();
        return 1;
    }
    if (bs > N) bs = N;

    /* --- Compute row partitioning --- */
    int *rows_per_rank = (int*)malloc(size * sizeof(int));
    int *displs_rows   = (int*)malloc(size * sizeof(int)); // starting row index per rank
    int base = N / size;
    int rem  = N % size;
    int accum = 0;
    for (int r = 0; r < size; ++r) {
        rows_per_rank[r] = base + (r < rem ? 1 : 0);
        displs_rows[r] = accum;
        accum += rows_per_rank[r];
    }
    int local_rows = rows_per_rank[rank];
    int row_start  = displs_rows[rank];

    /* --- Allocate local buffers --- */
    double *A_local = (double*)aligned_alloc_wrap(ALIGN_BYTES, (size_t)local_rows * N * sizeof(double));
    double *B_full  = (double*)aligned_alloc_wrap(ALIGN_BYTES, (size_t)N * N * sizeof(double));
    double *C_local = (double*)aligned_alloc_wrap(ALIGN_BYTES, (size_t)local_rows * N * sizeof(double));
    if (!A_local || !B_full || !C_local) {
        fprintf(stderr, "Rank %d allocation failure\n", rank);
        aligned_free_wrap(A_local); aligned_free_wrap(B_full); aligned_free_wrap(C_local);
        free(rows_per_rank); free(displs_rows);
        MPI_Finalize();
        return 2;
    }

    /* --- Root creates full A and B to scatter/broadcast --- */
    double *A_full = NULL;
    if (rank == 0) {
        A_full = (double*)aligned_alloc_wrap(ALIGN_BYTES, (size_t)N * N * sizeof(double));
        if (!A_full) {
            fprintf(stderr, "Root allocation failure for A_full\n");
            aligned_free_wrap(A_local); aligned_free_wrap(B_full); aligned_free_wrap(C_local);
            free(rows_per_rank); free(displs_rows);
            MPI_Finalize();
            return 3;
        }
        init_matrix(A_full, N, 1);
        init_matrix(B_full,  N, 2);
    }

    /* --- Scatter A rows --- */
    // Build counts & displacements in number of doubles.
    int *sendcounts = (int*)malloc(size * sizeof(int));
    int *displs_el  = (int*)malloc(size * sizeof(int));
    for (int r = 0; r < size; ++r) {
        sendcounts[r] = rows_per_rank[r] * N; // elements
        displs_el[r]  = displs_rows[r] * N;   // offset in elements
    }
    MPI_Scatterv(A_full, sendcounts, displs_el, MPI_DOUBLE,
                 A_local, local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    /* --- Broadcast B_full --- */
    MPI_Bcast(B_full, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (bs <= 0) {
        gemm_naive_local(A_local, B_full, C_local, N, row_start, local_rows);
    } else {
        gemm_blocked_local(A_local, B_full, C_local, N, local_rows, bs);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_time = t1 - t0; // local measurement (roughly same across ranks)

    /* --- Local checksum & reduction to global checksum --- */
    double local_chk = checksum_local(C_local, local_rows, N);
    double global_chk = 0.0;
    MPI_Reduce(&local_chk, &global_chk, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* --- Gather C to root (optional) --- */
    double *C_full = NULL;
    if (rank == 0) {
        C_full = (double*)aligned_alloc_wrap(ALIGN_BYTES, (size_t)N * N * sizeof(double));
    }
    MPI_Gatherv(C_local, local_rows * N, MPI_DOUBLE,
                C_full, sendcounts, displs_el, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double gflops = (2.0 * N * N * N) / (local_time * 1e9);
        printf("MPI GEMM: N=%d block=%d procs=%d time=%.6f sec checksum=%.6f GFLOPs=%.2f\n",
               N, bs, size, local_time, global_chk, gflops);
        // Expected checksum matches serial baseline (documented externally).
    }

    aligned_free_wrap(C_full);
    aligned_free_wrap(A_full);
    aligned_free_wrap(A_local);
    aligned_free_wrap(B_full);
    aligned_free_wrap(C_local);
    free(sendcounts); free(displs_el); free(rows_per_rank); free(displs_rows);

    MPI_Finalize();
    return 0;
}
