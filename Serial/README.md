# Serial Blocked GEMM - Baseline Implementation

Single-threaded, cache-optimized matrix multiplication using blocking (tiling) for improved performance.

## Overview

This is the **baseline serial implementation** for comparing parallel versions (OpenMP, MPI, CUDA). The blocked algorithm partitions large matrices into smaller tiles that fit in cache, reducing memory bandwidth requirements.

**Key Features:**
- Cache-optimized blocking with configurable tile size
- Deterministic initialization for reproducible results
- Cross-platform compatibility (Windows, Linux, macOS)
- High-resolution timing for accurate performance measurement
- Checksum verification for correctness

## Algorithm

**Blocking Strategy:**
```
Instead of computing C = A × B in one pass (poor cache locality),
partition into bs×bs tiles and compute:

for each tile row ii
  for each tile depth kk
    for each tile column jj
      C[ii:ii+bs, jj:jj+bs] += A[ii:ii+bs, kk:kk+bs] × B[kk:kk+bs, jj:jj+bs]
```

**Cache Benefits:**
- Tiles fit in L1/L2 cache (e.g., 64×64 doubles = 32KB)
- Each element reused `bs` times (vs. `N` times in naive algorithm)
- Reduces cache misses by factor of ~`bs/cache_line_size`
- Typical speedup: 2-5× for N=512-2048

## Building

### Prerequisites

**Windows (MinGW64/MSYS2):**
```bash
# Install MSYS2 from https://www.msys2.org/
pacman -S mingw-w64-x86_64-gcc make
```

**Windows (MSVC):**
- Visual Studio 2019+ with C++ Build Tools
- Run from "Developer Command Prompt for VS"

**Linux/macOS:**
```bash
# GCC or Clang already installed on most systems
sudo apt-get install build-essential  # Ubuntu/Debian
```

### Compilation

**Method 1: Using Makefile (Recommended)**
```bash
# In MSYS2 MinGW64 Bash (Windows) or any Unix shell:
cd Serial
make
```

**Method 2: Direct Compilation**

**GCC/Clang:**
```bash
gcc -Wall -Wextra -std=c11 -O3 -march=native -o blocked_gemm_serial blocked_gemm_serial.c -lm
```

**MSVC (Developer Command Prompt):**
```cmd
cl /O2 /W4 /std:c11 /Fe:blocked_gemm_serial.exe blocked_gemm_serial.c
```

**Compiler Flags Explained:**
- `-O3` (GCC) / `/O2` (MSVC): Maximum optimization
- `-march=native`: Enable CPU-specific instructions (AVX, FMA)
- `-std=c11`: C11 standard for `<stdint.h>`, `clock_gettime`
- `-lm`: Link math library (Unix/Linux)

## Usage

### Basic Execution
```bash
./blocked_gemm_serial <N> <block_size>
```

**Parameters:**
- `N`: Matrix dimension (N×N matrices)
- `block_size`: Tile size (must divide N evenly for best performance)

**Examples:**
```bash
./blocked_gemm_serial 512 64    # 512×512 matrices, 64×64 tiles
./blocked_gemm_serial 1024 128  # 1024×1024 matrices, 128×128 tiles
./blocked_gemm_serial 2048 64   # 2048×2048 matrices, 64×64 tiles
```

### Makefile Targets

```bash
make           # Build executable
make test      # Run test: N=512, block=64
make sweep     # Block size analysis for N=1024 (generates CSV)
make verify    # Verify checksums match expected values
make clean     # Remove build artifacts
```

### Sample Output
```
Matrix size N = 512, block size = 64
Initializing...
Multiplying (blocked)...
time=0.043723 checksum=33526118.721088
```

**Interpreting Results:**
- `time`: Wall-clock seconds for multiplication
- `checksum`: Sum of all elements in C (for verification)

## Performance Notes

### Block Size Selection

**General Guidelines:**
- Start with `bs = 64` (good default for most CPUs)
- Tune based on cache hierarchy:
  - **L1 cache (32-64KB)**: Try `bs = 32-64`
  - **L2 cache (256KB-1MB)**: Try `bs = 64-128`
  - **L3 cache (>2MB)**: Try `bs = 128-256`

**Rule of Thumb:**
```
3 × bs² × sizeof(double) ≤ Cache_Size
3 × bs² × 8 bytes ≤ Cache_Size

For L1 = 32KB:  bs ≤ 53  → use bs=32 or 64
For L2 = 256KB: bs ≤ 150 → use bs=128
```

### Expected Performance

**Typical Timings (Intel i5-12400, 6 cores, 2.5 GHz base):**

| N    | Block Size | Time (s) | GFlops |
|------|------------|----------|--------|
| 256  | 64         | 0.003    | 11.3   |
| 512  | 64         | 0.044    | 6.1    |
| 1024 | 128        | 0.452    | 4.7    |
| 2048 | 128        | 4.892    | 3.5    |

**Performance degrades with N** due to increasing memory bandwidth pressure (working set exceeds cache).

### Verification

**Checksums (deterministic initialization, seeds 1 and 2):**
- N=256: `2095714.920068`
- N=512: `33526118.721088`
- N=1024: `536844513.279285`
- N=2048: `8589951418.446381`

All parallel implementations (OpenMP, MPI, CUDA) **must match these checksums** exactly.

## Troubleshooting

### Windows: "gcc not found"
- Use **MSYS2 MinGW64 Bash** (not regular PowerShell/CMD)
- Or use MSVC in Developer Command Prompt

### MSVC: "unresolved external symbol clock_gettime"
- MSVC uses `QueryPerformanceCounter` (already in code)
- Compile with `cl /O2 /W4 /Fe:blocked_gemm_serial.exe blocked_gemm_serial.c`

### Linux: "undefined reference to sqrt"
- Add `-lm` flag: `gcc ... -lm`

### Performance Lower Than Expected
1. Verify `-O3` or `/O2` optimization enabled
2. Try different block sizes: `make sweep`
3. Check CPU frequency scaling (may throttle on laptops)
4. Ensure no background processes consuming CPU

### Checksum Mismatch
- Ensure deterministic initialization (LCG seeds 1 and 2)
- Do **not** use `rand()` or time-based seeds
- Verify no numerical instabilities (e.g., accumulator overflow)

## File Structure

```
Serial/
├── blocked_gemm_serial.c  # Main implementation
├── Makefile               # Build system
└── README.md              # This file

../data/Serial/
├── expected_checksums.txt # Reference checksums for N=256-2048
├── baseline_timings.txt   # Performance baselines
├── block_size_analysis.csv # Block size sweep results
└── sample_run_*.log       # Example outputs with analysis
```

## Next Steps

After validating the serial baseline:
1. **OpenMP**: Shared-memory parallelism (multi-core CPU)
2. **MPI**: Distributed-memory parallelism (multi-node cluster)
3. **CUDA**: GPU acceleration (NVIDIA GPUs)

All implementations should produce **identical checksums** to this serial version.

## References

- **Blocked Matrix Multiplication**: [Wikipedia - Matrix Multiplication Algorithm](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Block_matrix_multiplication)
- **Cache Optimization**: [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf) by Ulrich Drepper
- **BLAS**: [Intel MKL GEMM](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/gemm-001.html) (production-quality reference)
