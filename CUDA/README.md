# CUDA Blocked Matrix Multiplication

GPU-accelerated dense matrix multiplication (GEMM) using CUDA with blocked (tiled) algorithm and shared memory optimization.

## Overview

This implementation demonstrates CUDA parallelization with:
- **Blocked (tiled) GEMM** for improved memory locality and cache reuse
- **Shared memory** to reduce global memory traffic
- **Coalesced memory access** for optimal memory bandwidth utilization
- **Deterministic initialization** (same LCG as Serial/OpenMP/MPI for validation)
- **Checksum validation** for correctness verification across implementations

## Parallelization Strategy

### Thread Hierarchy
- **Grid**: 2D grid of thread blocks covering the output matrix C
- **Block**: 2D block of threads (typically 16×16 or 32×32)
- **Thread**: Each thread computes one element of the output tile

### Blocked Algorithm
1. **Tile Loading**: Threads collaboratively load tiles of A and B into shared memory
2. **Synchronization**: `__syncthreads()` ensures all threads finish loading before computation
3. **Partial Computation**: Threads compute partial dot products using shared memory
4. **Accumulation**: Results accumulate across tiles and written to global memory

### Memory Optimization
- **Shared Memory**: Reduces global memory accesses by ~2N/TILE_SIZE factor
- **Coalesced Access**: Threads in a warp access consecutive memory addresses
- **Bank Conflict Avoidance**: Shared memory access pattern minimizes bank conflicts

### Performance Considerations
- **Tile Size**: Typically 16, 32, or 64 (limited by shared memory per block)
- **Occupancy**: Block size affects GPU occupancy and register usage
- **Architecture**: Compile with `-arch=sm_XX` matching your GPU for optimal performance

## Build & Run

### Prerequisites
- NVIDIA GPU with CUDA support (compute capability ≥3.5)
- CUDA Toolkit 11.0+ (includes `nvcc` compiler)
- Windows: Visual Studio 2019+ or PowerShell
- Linux: GCC 7.0+ or compatible C++ compiler

**Verify Installation:**
```bash
nvcc --version                                          # Check CUDA compiler
nvidia-smi                                              # Check GPU
nvidia-smi --query-gpu=name,compute_cap --format=csv   # Check compute capability
```

### Build

**Linux/WSL/MSYS2 (using Makefile):**
```bash
cd CUDA
make clean
make ARCH=sm_75  # Adjust ARCH for your GPU
```

**Windows PowerShell (using build script):**
```powershell
cd CUDA
.\build.ps1              # Build with default arch (sm_75)
.\build.ps1 -Arch sm_86  # Or specify architecture
```

**Direct compilation (any platform):**
```bash
# Linux/WSL/Mac
nvcc -O3 -arch=sm_75 -std=c++11 -o blocked_gemm_cuda blocked_gemm_cuda.cu

# Windows PowerShell
nvcc -O3 -arch=sm_75 -std=c++11 -o blocked_gemm_cuda.exe blocked_gemm_cuda.cu
```

**Common GPU architectures:**
| GPU Family | Compute Capability | Architecture Flag |
|------------|-------------------|-------------------|
| GTX 1650, RTX 20xx, Tesla T4 | 7.5 | `-arch=sm_75` |
| A100 | 8.0 | `-arch=sm_80` |
| RTX 30xx | 8.6 | `-arch=sm_86` |
| RTX 40xx, L4 | 8.9 | `-arch=sm_89` |

### Run

**Linux/WSL:**
```bash
./blocked_gemm_cuda <N> <block_size>
# Example:
./blocked_gemm_cuda 1024 32
```

**Windows:**
```powershell
.\blocked_gemm_cuda.exe <N> <block_size>
# Or use build script:
.\build.ps1 -Target test
```

**Arguments:**
- `N`: matrix size (N×N), range: 1 to 100000
- `block_size`: tile size (8, 16, 24, 32; ≤0 for naive CUDA GEMM)
  - **Maximum**: 32 for most GPUs (1024 threads/block limit)
  - Test your limit: `./blocked_gemm_cuda 1024 48` (will show error if exceeded)

**Example Output:**
```
CUDA GEMM: N=1024 block=32 time=0.031856 sec checksum=268310302.3759228587 GFLOPs=67.41
```

## Build Targets

### Makefile (Linux/WSL/MSYS2)
| Target | Purpose |
|--------|---------|
| `make` | Build `blocked_gemm_cuda` |
| `make test` | Single run (N=512, block=32) |
| `make verify` | Multi-size checksum verification (256,512,1024,2048) |
| `make sweep` | Block size sweep (writes `block_size_analysis.csv`) |
| `make baseline` | Full baseline timings (writes `baseline_timings.txt`) |
| `make clean` | Remove build artifacts |
| `make help` | Show help message |

### PowerShell Build Script (Windows)
```powershell
.\build.ps1                    # Build with default settings
.\build.ps1 -Target test       # Single test run
.\build.ps1 -Target verify     # Checksum verification
.\build.ps1 -Target sweep      # Block size analysis
.\build.ps1 -Target baseline   # Full benchmarks
.\build.ps1 -Target clean      # Remove artifacts
.\build.ps1 -Target help       # Show help
.\build.ps1 -Arch sm_86        # Build for specific GPU
```

## Data Artifacts

- `data/CUDA/expected_checksums.txt` – Reference checksums for correctness
- `data/CUDA/block_size_analysis.csv` – Block size vs time & GFLOPs
- `data/CUDA/baseline_timings.txt` – Matrix size × block size baseline results
- `data/CUDA/test_config.txt` – Configurable test parameters

## Verification

Checksums are deterministic and match Serial/OpenMP/MPI implementations (same LCG initialization):

**Linux/WSL:**
```bash
make verify
```

**Windows:**
```powershell
.\build.ps1 -Target verify
```

**Expected checksums** (compare with output):
```
N=256  block=32  checksum=4199507.5471359976
N=512  block=32  checksum=33526118.7210875191
N=1024 block=32  checksum=268310302.3759228587
N=2048 block=32  checksum=2143508340.9166996479
```

All checksums should match exactly (within floating-point precision ~1e-6).

## Performance Analysis

### Block Size Impact
```bash
make sweep
# Analyzes performance across block sizes 8,16,24,32,48,64
# Results in data/CUDA/block_size_analysis.csv
```

**Typical findings:**
- **Optimal tile size**: 16×16 to 32×32 (balances shared memory and occupancy)
- **GTX 1650 results** (sm_75): block=16 achieves 73.02 GFLOPs (best for N=1024)
- **Too small** (8×8): Underutilizes GPU
- **Too large** (>32): Exceeds 1024 threads/block limit on most GPUs

### Baseline Performance
```bash
make baseline
# Full benchmark: N=256,512,1024,2048 × block=16,32,64
# Results in data/CUDA/baseline_timings.txt
```

## Implementation Notes

### Correctness
- **Deterministic**: Same LCG seeds as Serial/OpenMP/MPI ensure reproducible checksums
- **Bounds checking**: Handles non-multiple matrix sizes gracefully
- **Error checking**: `CUDA_CHECK` macro catches runtime errors

### Code Quality
- **Well-commented**: Explains parallelization strategy, memory access patterns, and algorithm
- **Modular**: Separate naive and blocked kernels for clarity
- **Original work**: See `AI_CITATION.md` for AI assistance details

### Limitations
- **Matrix size**: Best performance when N is multiple of block_size
- **Block size**: Limited by GPU (max 32×32 = 1024 threads/block on most GPUs)
- **Shared memory**: Also limited by GPU (typically 48-96 KB per block)
- **Precision**: Uses double precision (matches Serial/OpenMP/MPI for validation)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `nvcc: command not found` | Add CUDA to PATH or use full path to nvcc |
| `no kernel image available` | Rebuild with correct `-arch=sm_XX` for your GPU |
| `make: command not found` (Windows) | Use `.\build.ps1` PowerShell script instead |
| Block size error (>32) | Use block_size ≤32 (1024 threads/block limit) |
| Checksums don't match | Rebuild: `make clean && make` |

## AI Assistance

This implementation was developed with AI assistance (GitHub Copilot). See `AI_CITATION.md` for:
- Prompts used
- Scope of AI contributions
- Human contributions and originality statement

## References

- CUDA Programming Guide: [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)
- Matrix Multiplication Optimization: [NVIDIA Blog](https://developer.nvidia.com/blog/optimizing-matrix-multiplication/)
- Shared Memory Best Practices: [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
