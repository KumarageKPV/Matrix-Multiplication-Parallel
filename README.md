## Dense Matrix Multiplication (GEMM) – Parallel Versions

### Algorithm
**Title**: Blocked Dense Matrix Multiplication (GEMM)  
**Domain**: Numerical Computation and Scientific Computing  
**Problem**: Compute C = A × B where A, B, C are large dense square matrices

### Why suitable for parallelization?
- O(N³) computational intensity with independent accumulation operations
- Excellent data parallelism (each C[i][j] can be computed independently)
- High cache reuse potential with blocking/tiled approach
- Well-studied benchmark in high-performance computing (BLAS SGEMM/DGEMM)
- Scales across shared memory (OpenMP), distributed memory (MPI), and GPU (CUDA)

### Implementations
- `Serial/`       → Optimized blocked serial baseline (C)
- `OpenMP/`       → Multi-threaded OpenMP version
- `MPI/`          → Distributed-memory MPI version
- `CUDA/`         → CUDA GPU implementation
- `report/`       → Full PDF report + graphs
- `screenshots/`  → Execution screenshots and timing results
- `data/`         → Sample input matrices and verification outputs

### Build & Run: OpenMP

Windows (GCC via MSYS2/MinGW):

```powershell
gcc -O3 -std=c11 -fopenmp -o blocked_gemm_omp OpenMP/blocked_gemm_omp.c -lm
./blocked_gemm_omp 512 64 8
```

Windows (MSVC):

```powershell
cl /O2 /openmp /Fe:blocked_gemm_omp.exe OpenMP\blocked_gemm_omp.c
blocked_gemm_omp.exe 512 64 8
```

Arguments:
- `N`: matrix size (NxN)
- `block_size`: tile size (<=0 runs naive OpenMP)
- `num_threads` (optional): OpenMP threads

Output sample:
```
OpenMP GEMM: N=512 block=64 threads=8 time=0.123456 sec checksum=0.0001234567
```
