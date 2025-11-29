# OpenMP Implementation - Build and Run Instructions

## Overview
This directory contains the OpenMP parallel implementation of blocked dense matrix multiplication (GEMM). The implementation uses shared-memory parallelism with compiler directives to parallelize computation across multiple CPU cores.

## Files
- `blocked_gemm_omp.c` - Main OpenMP implementation with detailed comments
- `Makefile` - Comprehensive build system with test/benchmark targets
- `AI_CITATION.md` - Documentation of AI assistance (required by assignment)

## Prerequisites

### Windows (Recommended: MSYS2 MinGW64)
1. Install MSYS2 from https://www.msys2.org/
2. Open "MSYS2 MinGW 64-bit" terminal
3. Install GCC with OpenMP support:
   ```bash
   pacman -Sy --noconfirm
   pacman -S --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-libwinpthread
   ```

### Windows (Alternative: MSVC)
1. Install Visual Studio (Community Edition or higher)
2. Open "Developer Command Prompt for VS"
3. MSVC supports OpenMP via `/openmp` flag

### Linux/macOS
- GCC with OpenMP: `sudo apt install gcc` (Ubuntu/Debian) or `brew install gcc` (macOS)
- Clang with OpenMP: `sudo apt install clang libomp-dev` or `brew install libomp`

## Compilation

### Quick Build (MinGW64 Bash or Linux/macOS)
```bash
make
```

### Manual Compilation

#### GCC (MinGW64, Linux, macOS)
```bash
gcc -O3 -std=c11 -fopenmp -o blocked_gemm_omp blocked_gemm_omp.c -lm
```

#### MSVC (Windows)
```cmd
cl /O2 /openmp /std:c11 /Fe:blocked_gemm_omp.exe blocked_gemm_omp.c
```

#### Clang (with OpenMP)
```bash
clang -O3 -std=c11 -fopenmp -o blocked_gemm_omp blocked_gemm_omp.c -lm
```

## Running

### Basic Usage
```bash
./blocked_gemm_omp <N> <block_size> [num_threads]
```

**Arguments:**
- `N`: Matrix dimension (N×N square matrices)
- `block_size`: Tile size for cache blocking (use ≤0 for naive OpenMP)
- `num_threads` (optional): Number of OpenMP threads (defaults to system max)

**Examples:**
```bash
# 512×512 matrices, 64×64 blocks, 8 threads
./blocked_gemm_omp 512 64 8

# 1024×1024 matrices, 128×128 blocks, 16 threads
./blocked_gemm_omp 1024 128 16

# Naive (non-blocked) OpenMP with 4 threads
./blocked_gemm_omp 512 0 4
```

### Quick Test
```bash
make test
```

### Verify Against Serial Baseline
```bash
make verify
```
This compares the checksum against the serial implementation to ensure correctness.

## Performance Evaluation

### Thread Scalability Sweep
```bash
make sweep-threads
```
Generates `timings_threads.csv` with performance across 1, 2, 4, 8, 12, 16 threads.

### Block Size Sweep
```bash
make sweep-blocks
```
Generates `timings_blocks.csv` with performance for block sizes 16, 32, 64, 128, 256.

### Full Benchmark
```bash
make benchmark
```
Runs comprehensive tests across multiple matrix sizes, block sizes, and thread counts.
Output: `benchmark_results.csv`

### Custom Runs
```bash
# Override defaults
make test N=1024 BS=128 THREADS=16

# Test specific configuration
./blocked_gemm_omp 2048 64 8
```

## Understanding the Output

Example output:
```
OpenMP GEMM: N=512 block=64 threads=8 time=0.030691 sec checksum=33526118.7210875191
```

- `N=512`: Matrix size (512×512)
- `block=64`: Block size used (64×64 tiles)
- `threads=8`: Number of OpenMP threads
- `time=0.030691`: Wall-clock time in seconds
- `checksum=...`: Sum of all output matrix elements (for verification)

## Parallelization Strategy

### Blocked GEMM Approach
1. **Tiling**: Partition matrices into bs×bs blocks for cache locality
2. **Parallelization**: Use `#pragma omp parallel for collapse(2)` over (i0, j0) tile indices
3. **Load Balancing**: Static scheduling distributes tiles evenly across threads
4. **No Race Conditions**: Each thread updates independent output tiles

### Key OpenMP Directives
```c
// Parallelize over output tiles (i0, j0)
#pragma omp parallel for collapse(2) schedule(static)
for (int i0 = 0; i0 < n; i0 += bs) {
    for (int j0 = 0; j0 < n; j0 += bs) {
        // k0 loop sequential per thread to accumulate C[i,j]
        for (int k0 = 0; k0 < n; k0 += bs) {
            // Inner micro-kernel (cache-resident)
        }
    }
}
```

## Performance Tips

1. **Thread Count**: Use physical core count for best results (avoid hyperthreading overhead)
   ```bash
   # Linux: check core count
   nproc
   
   # Windows PowerShell
   (Get-CimInstance Win32_Processor).NumberOfCores
   ```

2. **Block Size**: Optimal size depends on CPU cache
   - L1 cache: ~32-64 for best performance
   - L2 cache: ~64-128 for larger matrices
   - Try multiple sizes and pick the fastest

3. **Environment Variables**:
   ```bash
   export OMP_NUM_THREADS=8        # Set thread count
   export OMP_PROC_BIND=true       # Pin threads to cores
   export OMP_PLACES=cores         # Thread placement policy
   ```

## Troubleshooting

### Build Issues

**"gcc: command not found"** (Windows)
- Use MSYS2 MinGW64 Bash (not regular PowerShell)
- Or install MinGW-w64 and add to PATH

**"stdio.h: no such file"** (MSVC)
- Run from "Developer Command Prompt for VS"
- Sets `INCLUDE` and `LIB` environment variables

**"undefined reference to `omp_get_max_threads`"**
- Ensure `-fopenmp` flag is present
- GCC/Clang: link with `-fopenmp`
- MSVC: use `/openmp`

### Runtime Issues

**Slow performance**
- Try different block sizes (32, 64, 128)
- Reduce thread count to physical cores only
- Check CPU frequency scaling (may throttle)

**Checksum mismatch**
- Ensure serial baseline uses same N and block size
- Floating-point precision: small differences (<1e-6) are acceptable

## Expected Results

### Speedup vs Serial (N=512, block=64)
| Threads | Time (s) | Speedup |
|---------|----------|---------|
| 1       | ~0.044   | 1.0×    |
| 2       | ~0.024   | 1.8×    |
| 4       | ~0.014   | 3.1×    |
| 8       | ~0.009   | 4.9×    |

*Actual results vary by CPU architecture and clock speed.*

## Clean Up
```bash
make clean
```
Removes compiled binaries and generated CSV files.

## References
- OpenMP 4.5 Specification: https://www.openmp.org/specifications/
- BLAS GEMM: https://netlib.org/blas/
- Cache-Oblivious Algorithms: Frigo et al., 1999

