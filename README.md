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
