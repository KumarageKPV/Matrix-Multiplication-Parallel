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

#### Build
- **GCC (MSYS2/MinGW):**
	```bash
	cd OpenMP
	make clean
	make
	```
- **MSVC:**
	```powershell
	cl /O2 /openmp /Fe:blocked_gemm_omp.exe blocked_gemm_omp.c
	```

#### Run
```bash
./blocked_gemm_omp N block_size [num_threads]
# Example:
./blocked_gemm_omp 512 64 8
```
Arguments:
- `N`: matrix size (NxN)
- `block_size`: tile size (<=0 runs naive OpenMP)
- `num_threads` (optional): OpenMP threads

#### Output sample
```
OpenMP GEMM: N=512 block=64 threads=8 time=0.123456 sec checksum=0.0001234567
```

#### Makefile Targets (OpenMP)
| Target     | Purpose                                                     |
|------------|-------------------------------------------------------------|
| `make`     | Build `blocked_gemm_omp`                                    |
| `make test`| Single run (N=512, block=64, threads=8)                     |
| `make verify` | Multi-size checksum verification (256,512,1024,2048)     |
| `make sweep`  | Block size sweep (writes `block_size_analysis.csv`)      |
| `make threadscale` | Thread scaling study (writes `thread_scalability.csv`)|

#### Data Artifacts (OpenMP)
- `data/OpenMP/expected_checksums.txt` – reference checksums for correctness
- `data/OpenMP/block_size_analysis.csv` – block size vs time & GFLOPs
- `data/OpenMP/thread_scalability.csv` – threads vs time, speedup, efficiency, GFLOPs



### Build & Run: MPI

#### Build
- **Linux/WSL/MSYS2 (OpenMPI):**
	```bash
	cd MPI
	make clean
	make
	```

#### Run
```bash
mpiexec --oversubscribe -n 4 ./blocked_gemm_mpi N block_size
# Example:
mpiexec --oversubscribe -n 4 ./blocked_gemm_mpi 1024 128
```
Arguments:
- `N`: matrix size (NxN)
- `block_size`: tile size (<=0 runs naive MPI)

#### Output sample
```
MPI GEMM: N=1024 block=128 procs=4 time=0.123456 sec checksum=268310302.375923 GFLOPs=XX.YY
```

#### Makefile Targets (MPI)
| Target     | Purpose                                                     |
|------------|-------------------------------------------------------------|
| `make`     | Build `blocked_gemm_mpi`                                    |
| `make test`| Single run (N=512, block=64, procs=4)                       |
| `make verify` | Multi-size checksum verification (256,512,1024,2048)     |
| `make sweep`  | Block size sweep (writes `block_size_analysis.csv`)      |
| `make procscale` | Process scaling study (writes `proc_scaling.csv`)     |
| `make baseline` | Full baseline timings (writes `baseline_timings.txt`)  |

#### Data Artifacts (MPI)
- `data/MPI/expected_checksums.txt` – reference checksums for correctness
- `data/MPI/block_size_analysis.csv` – block size vs time & GFLOPs
- `data/MPI/proc_scaling.csv` – procs vs time, speedup, efficiency, GFLOPs
- `data/MPI/baseline_timings.txt` – matrix size × process grid baseline results
- `data/MPI/test_config.txt` – configurable sizes, block sizes, and process counts

#### Notes
- All MPI runs use `mpiexec --oversubscribe` by default for CI and local compatibility.

