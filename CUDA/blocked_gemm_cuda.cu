/* blocked_gemm_cuda.cu
 * CUDA-accelerated dense matrix multiplication with blocked (tiled) GEMM.
 *
 * This implementation demonstrates GPU parallelization using CUDA with:
 *   - Blocked (tiled) matrix multiplication for improved memory locality
 *   - Shared memory usage for cache optimization
 *   - Deterministic matrix initialization (same as Serial/OpenMP/MPI)
 *   - Checksum validation for correctness verification
 *   - Cross-platform timing and error handling
 *
 * Parallelization Strategy:
 *   - Each CUDA block computes a tile of the output matrix C
 *   - Threads within a block collaborate using shared memory
 *   - Grid/block dimensions are tuned for occupancy and coalesced access
 *   - Falls back to naive kernel if block_size <= 0
 *
 * Compile:
 *   nvcc -O3 -arch=sm_75 -o blocked_gemm_cuda blocked_gemm_cuda.cu
 *   (adjust -arch= to match your GPU, e.g., sm_86 for RTX 30xx, sm_89 for RTX 40xx)
 *
 * Run:
 *   ./blocked_gemm_cuda <N> <block_size>
 *   e.g. ./blocked_gemm_cuda 1024 32
 *
 * Notes:
 *   - block_size is the tile dimension (typically 16, 32, or 64)
 *   - If block_size <= 0, uses naive CUDA GEMM without tiling
 *   - Matrix size N should be divisible by block_size for optimal performance
 *
 * AI Assistance:
 *   This implementation was developed with AI assistance. See AI_CITATION.md for details.
 *   All code is original work demonstrating understanding of CUDA parallelization.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>

/* CUDA error checking macro */
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/*
 * Deterministic Linear Congruential Generator (LCG) for matrix initialization.
 * 
 * Uses the same parameters as Serial/OpenMP/MPI for reproducible results:
 * - a = 1103515245 (multiplier, from glibc)
 * - c = 12345 (increment, from glibc)
 * - m = 2^32 (modulus, implicit via uint32_t overflow)
 * 
 * Maps to [0.0, 1.0) deterministically for cross-version validation.
 */
static inline uint32_t lcg_next(uint32_t state) {
    return state * 1103515245U + 12345U;
}

static inline double lcg_to_double(uint32_t state) {
    return (double)(state % 1000) / 1000.0;
}

/*
 * Initialize matrix A and B on host with deterministic values.
 * Matches Serial/OpenMP/MPI initialization for checksum validation.
 * 
 * A uses seed_offset=1, B uses seed_offset=2
 */
static void init_matrices_host(double *A, double *B, int N) {
    uint32_t seed_a = 1234567U + 1U;  /* seed_offset=1 for A */
    uint32_t seed_b = 1234567U + 2U;  /* seed_offset=2 for B */
    
    for (int i = 0; i < N * N; i++) {
        seed_a = lcg_next(seed_a);
        A[i] = lcg_to_double(seed_a);
        
        seed_b = lcg_next(seed_b);
        B[i] = lcg_to_double(seed_b);
    }
}

/*
 * Naive CUDA GEMM kernel (fallback when block_size <= 0).
 * Each thread computes one element of C = A * B.
 * 
 * Grid/block configuration:
 * - 2D grid of 2D blocks
 * - Each thread computes C[row][col] via dot product
 */
__global__ void gemm_kernel_naive(const double *A, const double *B, double *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/*
 * Blocked (tiled) CUDA GEMM kernel using shared memory.
 * 
 * Parallelization Strategy:
 * - Each thread block computes a TILE_SIZE × TILE_SIZE sub-matrix of C
 * - Threads collaborate to load tiles of A and B into shared memory
 * - Reduces global memory accesses by ~2N/TILE_SIZE factor
 * - Improves memory coalescing and cache reuse
 * 
 * Algorithm:
 * 1. Loop over tiles along the k-dimension (N/TILE_SIZE iterations)
 * 2. Collaboratively load A_tile and B_tile into shared memory
 * 3. Synchronize threads within block
 * 4. Compute partial dot products using shared memory
 * 5. Accumulate into C
 * 
 * Memory Access Pattern:
 * - Coalesced loads from global memory (threads in warp access consecutive addresses)
 * - Shared memory eliminates redundant global memory fetches
 * - Bank conflict-free access (stride pattern avoids conflicts)
 */
__global__ void gemm_kernel_blocked(const double *A, const double *B, double *C, int N, int TILE_SIZE) {
    /* Shared memory tiles (allocated dynamically based on TILE_SIZE) */
    extern __shared__ double shared_mem[];
    double *A_tile = shared_mem;
    double *B_tile = &shared_mem[TILE_SIZE * TILE_SIZE];
    
    /* Thread and block indices */
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    double sum = 0.0;
    
    /* Loop over tiles along k-dimension */
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; t++) {
        /* Collaborative loading of A_tile and B_tile into shared memory */
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;
        
        /* Load A_tile (bounds checking for non-multiple matrix sizes) */
        if (row < N && a_col < N) {
            A_tile[ty * TILE_SIZE + tx] = A[row * N + a_col];
        } else {
            A_tile[ty * TILE_SIZE + tx] = 0.0;
        }
        
        /* Load B_tile (bounds checking) */
        if (b_row < N && col < N) {
            B_tile[ty * TILE_SIZE + tx] = B[b_row * N + col];
        } else {
            B_tile[ty * TILE_SIZE + tx] = 0.0;
        }
        
        /* Synchronize to ensure all threads have loaded their data */
        __syncthreads();
        
        /* Compute partial dot product using shared memory */
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[ty * TILE_SIZE + k] * B_tile[k * TILE_SIZE + tx];
        }
        
        /* Synchronize before loading next tile (prevent race conditions) */
        __syncthreads();
    }
    
    /* Write result to global memory */
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

/*
 * Compute checksum of matrix C for validation.
 * Simple sum of all elements (deterministic given deterministic inputs).
 */
static double compute_checksum(const double *C, int N) {
    double sum = 0.0;
    for (int i = 0; i < N * N; i++) {
        sum += C[i];
    }
    return sum;
}

/*
 * High-resolution timer using CUDA events.
 * Provides accurate GPU kernel execution time.
 */
static double cuda_event_elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return (double)ms / 1000.0; /* Convert to seconds */
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <N> <block_size>\n", argv[0]);
        fprintf(stderr, "  N          : matrix dimension (N×N)\n");
        fprintf(stderr, "  block_size : tile size (<=0 for naive CUDA GEMM)\n");
        return EXIT_FAILURE;
    }
    
    int N = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    
    if (N <= 0 || N > 100000) {
        fprintf(stderr, "Error: N must be in range (0, 100000]\n");
        return EXIT_FAILURE;
    }
    
    /* Clamp block_size to reasonable range for CUDA */
    if (block_size > N) block_size = N;
    
    /* Check shared memory and thread limits */
    if (block_size > 0) {
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
        
        int max_threads_per_block = deviceProp.maxThreadsPerBlock;
        size_t shared_mem_per_block = deviceProp.sharedMemPerBlock;
        size_t required_shared_mem = 2 * block_size * block_size * sizeof(double);
        int required_threads = block_size * block_size;
        
        if (required_threads > max_threads_per_block) {
            fprintf(stderr, "Error: block_size=%d requires %d threads, but GPU only supports %d threads per block\n",
                    block_size, required_threads, max_threads_per_block);
            fprintf(stderr, "Maximum safe block_size for this GPU: %d\n", (int)sqrt(max_threads_per_block));
            return EXIT_FAILURE;
        }
        
        if (required_shared_mem > shared_mem_per_block) {
            fprintf(stderr, "Error: block_size=%d requires %zu bytes of shared memory, but GPU only has %zu bytes per block\n",
                    block_size, required_shared_mem, shared_mem_per_block);
            fprintf(stderr, "Maximum safe block_size for this GPU: %d\n", (int)sqrt(shared_mem_per_block / (2 * sizeof(double))));
            return EXIT_FAILURE;
        }
    }
    
    size_t matrix_bytes = sizeof(double) * (size_t)N * (size_t)N;
    
    /* Allocate host matrices */
    double *h_A = (double*)malloc(matrix_bytes);
    double *h_B = (double*)malloc(matrix_bytes);
    double *h_C = (double*)malloc(matrix_bytes);
    
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Error: host malloc failed\n");
        return EXIT_FAILURE;
    }
    
    /* Initialize matrices with deterministic values */
    init_matrices_host(h_A, h_B, N);
    
    /* Allocate device matrices */
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, matrix_bytes));
    
    /* Copy inputs to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, matrix_bytes, cudaMemcpyHostToDevice));
    
    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    /* Configure kernel launch parameters */
    dim3 block_dim, grid_dim;
    size_t shared_mem_size = 0;
    
    if (block_size <= 0) {
        /* Naive kernel configuration */
        block_dim = dim3(16, 16);
        grid_dim = dim3((N + 15) / 16, (N + 15) / 16);
        
        CUDA_CHECK(cudaEventRecord(start));
        gemm_kernel_naive<<<grid_dim, block_dim>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
    } else {
        /* Blocked kernel configuration */
        block_dim = dim3(block_size, block_size);
        grid_dim = dim3((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);
        shared_mem_size = 2 * block_size * block_size * sizeof(double);
        
        CUDA_CHECK(cudaEventRecord(start));
        gemm_kernel_blocked<<<grid_dim, block_dim, shared_mem_size>>>(d_A, d_B, d_C, N, block_size);
        CUDA_CHECK(cudaEventRecord(stop));
    }
    
    /* Wait for kernel completion and check for errors */
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    
    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, matrix_bytes, cudaMemcpyDeviceToHost));
    
    /* Compute elapsed time and performance metrics */
    double elapsed = cuda_event_elapsed_ms(start, stop);
    double flops = 2.0 * (double)N * (double)N * (double)N;
    double gflops = (flops / elapsed) / 1e9;
    
    /* Compute checksum for validation */
    double checksum = compute_checksum(h_C, N);
    
    /* Print results in consistent format */
    printf("CUDA GEMM: N=%d block=%d time=%.6f sec checksum=%.10f GFLOPs=%.2f\n",
           N, block_size, elapsed, checksum, gflops);
    
    /* Cleanup */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    
    return EXIT_SUCCESS;
}
