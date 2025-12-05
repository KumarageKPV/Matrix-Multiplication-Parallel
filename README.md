# Parallel Matrix Multiplication (Blocked GEMM)
  
**Algorithm**: Blocked Dense Matrix Multiplication (C = A × B)

## Overview
Four implementations of cache-blocked GEMM with O(N³) complexity:
- **Serial** – Optimized blocked baseline (C)
- **OpenMP** – Shared-memory parallel (multi-threaded)
- **MPI** – Distributed-memory parallel (message passing)
- **CUDA** – GPU-accelerated (NVIDIA)

## Repository Structure
```
Matrix-Multiplication-Parallel/
├── Serial/           # Sequential baseline
├── OpenMP/           # Shared-memory (threads)
├── MPI/              # Distributed-memory (processes)
├── CUDA/             # GPU-accelerated
├── performance_analysis/  # Graphs, CSVs, analysis
├── data/             # Performance artifacts & checksums
└── screenshots/      # Execution & timing results
```

## Quick Start
See individual README files in each implementation folder for detailed instructions:
- [`Serial/README.md`](Serial/README.md)
- [`OpenMP/README.md`](OpenMP/README.md)
- [`MPI/README.md`](MPI/README.md)
- [`CUDA/README.md`](CUDA/README.md)
- [`performance_analysis/README.md`](performance_analysis/README.md)

## Performance Summary
Graphs and analysis available in `performance_analysis/graphs/`. Generate with:
```bash
cd performance_analysis
python generate_graphs.py
```

## Documentation
- **AI Citation**: See `*/AI_CITATION.md` in each implementation folder

---


## Serial Baseline

**Build**: `cd Serial && make`  
**Run**: `./blocked_gemm_serial N block_size`  
**Details**: See [`Serial/README.md`](Serial/README.md)


## OpenMP (Shared-Memory)

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
**MS-MPI (Windows):**
```bash
mpiexec -n 4 blocked_gemm_mpi.exe 1024 128
```

**Open MPI (Linux/WSL/MSYS2):**
```bash
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
- **MS-MPI** (Windows default): Use `mpiexec -n P blocked_gemm_mpi.exe`
- **Open MPI** (Linux/WSL): Use `mpiexec --oversubscribe -n P ./blocked_gemm_mpi` (--oversubscribe allows overcommitting cores)
- See `MPI/AI_CITATION.md` for AI assistance details and originality statement.
- **Details**: See [`MPI/README.md`](MPI/README.md)


## CUDA (GPU)

#### Build
- **NVIDIA CUDA Toolkit:**
  ```bash
  cd CUDA
  make clean
  make ARCH=sm_75  # Adjust ARCH for your GPU
  ```

- **Windows (PowerShell):**
  ```powershell
  cd CUDA
  .\build.ps1          # Build with default arch (sm_75)
  .\build.ps1 -Arch sm_89  # Build for RTX 40xx
  ```
  
Common GPU architectures:
- `sm_75`: RTX 20xx, Tesla T4, GTX 1650
- `sm_80`: A100
- `sm_86`: RTX 30xx
- `sm_89`: RTX 40xx, L4

Or compile directly:
```bash
nvcc -O3 -arch=sm_75 -o blocked_gemm_cuda blocked_gemm_cuda.cu
```

#### Run
```bash
./blocked_gemm_cuda N block_size
# Example:
./blocked_gemm_cuda 1024 32
```

**Windows (PowerShell):**
```powershell
.\blocked_gemm_cuda.exe 1024 32
# Or use build script
.\build.ps1 -Target test
```

Arguments:
- `N`: matrix size (NxN)
- `block_size`: tile size (8, 16, 24, 32; <=0 for naive CUDA GEMM)
  - **Note**: Max block_size = 32 for most GPUs (1024 threads/block limit)

#### Output sample
```
CUDA GEMM: N=1024 block=32 time=0.012345 sec checksum=268310302.3759 GFLOPs=1750.23
```

#### Makefile Targets (CUDA)
| Target     | Purpose                                                     |
|------------|-------------------------------------------------------------|
| `make`     | Build `blocked_gemm_cuda`                                   |
| `make test`| Single run (N=512, block=32)                                |
| `make verify` | Multi-size checksum verification (256,512,1024,2048)     |
| `make sweep`  | Block size sweep (writes `block_size_analysis.csv`)      |
| `make baseline` | Full baseline timings (writes `baseline_timings.txt`)  |

#### Data Artifacts (CUDA)
- `data/CUDA/expected_checksums.txt` – reference checksums for correctness
- `data/CUDA/block_size_analysis.csv` – block size vs time & GFLOPs
- `data/CUDA/baseline_timings.txt` – matrix size × block size baseline results
- `data/CUDA/test_config.txt` – configurable test parameters

#### Notes
- Requires NVIDIA GPU with CUDA support
- Windows users: Use `build.ps1` PowerShell script (no `make` required)
- Block size limited by GPU (typically max 32 due to 1024 threads/block)
- See `CUDA/AI_CITATION.md` for AI assistance details and originality statement
- **Details**: See [`CUDA/README.md`](CUDA/README.md)

