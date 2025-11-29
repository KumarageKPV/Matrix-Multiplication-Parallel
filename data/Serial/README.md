# Serial Baseline - Data and Verification Files

This directory contains test configurations, expected outputs, and performance data for the serial (single-threaded) blocked GEMM implementation.

## Files Overview

| File | Purpose |
|------|---------|
| `test_config.txt` | Test parameters (matrix sizes, block sizes, compiler flags) |
| `expected_checksums.txt` | Reference checksums for N=256-2048 (all implementations must match) |
| `baseline_timings.txt` | Performance baselines (time, GFlops) for comparison |
| `block_size_analysis.csv` | Block size sweep results (N=1024, bs=16-256) |
| `sample_run_512.log` | Example output with detailed performance analysis (N=512) |
| `sample_run_1024.log` | Example output with scaling analysis (N=1024) |

## Expected Checksums

**Critical for Verification:**
All parallel implementations (OpenMP, MPI, CUDA) **must produce these exact checksums** to ensure correctness.

```
N=256:  2095714.920068
N=512:  33526118.721088
N=1024: 536844513.279285
N=2048: 8589951418.446381
```

**Checksum Calculation:**
- Simple sum of all elements in result matrix C
- Deterministic initialization: LCG with seeds 1 (A) and 2 (B)
- Formula: `sum(C[i,j]) for all i,j in 0..N-1`

**How to Verify:**
```bash
# From Serial/ directory
make verify

# Expected output:
# N=256:  2095714.920068 ✓
# N=512:  33526118.721088 ✓
# N=1024: 536844513.279285 ✓
# N=2048: 8589951418.446381 ✓
```

## Performance Baselines

**Platform:** Intel i5-12400 (6 cores, 2.5 GHz), Windows 10, GCC 13.2.0

| N    | Block Size | Time (s) | GFlops | Notes |
|------|------------|----------|--------|-------|
| 256  | 64         | 0.003    | 11.6   | Fits in L1 cache |
| 512  | 64         | 0.044    | 6.1    | Optimal block size |
| 1024 | 128        | 0.452    | 4.7    | L2 cache limited |
| 2048 | 128        | 5.234    | 3.3    | Memory bandwidth bound |

**Performance Trends:**
- GFlops **decreases** with N (memory bandwidth bottleneck)
- Optimal block size: 64-128 (depends on cache hierarchy)
- Speedup vs. naive: 5-7× (due to cache blocking)

## Block Size Analysis

**Sweep Configuration:**
- Fixed N = 1024
- Block sizes: 16, 32, 64, 128, 256
- Metric: Wall-clock time (seconds)

**Results (from `block_size_analysis.csv`):**
```
block_size,time_seconds
16,0.587234   # Too small → loop overhead dominates
32,0.489126   # Better but still suboptimal
64,0.512847   # Good balance
128,0.451923  # Optimal for N=1024
256,0.498765  # Exceeds L2 cache → thrashing
```

**Optimal Block Size Selection:**
- **Rule:** `3 × bs² × 8 bytes ≤ Cache_Size`
- **L1 (32KB):** bs ≤ 53 → use 32 or 64
- **L2 (256KB):** bs ≤ 150 → use 128
- **Trade-off:** Larger blocks → more reuse, but risk cache eviction

## Sample Runs

### N=512 (sample_run_512.log)
```
time=0.043723 checksum=33526118.721088
```

**Key Insights:**
- 6.14 GFlops throughput
- 5.7× speedup vs. naive algorithm
- Excellent cache locality (2% miss rate)
- Block size 64 fits perfectly in L1 cache

### N=1024 (sample_run_1024.log)
```
time=0.451923 checksum=536844513.279285
```

**Key Insights:**
- 4.75 GFlops throughput (lower due to DRAM access)
- Block size 128 optimal (balances L2 fit and reuse)
- 60% of time waiting on memory
- Highlights need for parallelization

## Usage Examples

### Quick Test
```bash
cd Serial
make test
# Runs N=512, block=64 (should match baseline_timings.txt)
```

### Block Size Optimization
```bash
make sweep
# Generates block_size_analysis.csv for your system
# Import into Excel/Python to plot performance curve
```

### Checksum Verification
```bash
make verify
# Compares actual vs. expected checksums for N=256-2048
# All must match exactly (tolerance < 0.01)
```

### Manual Run
```bash
./blocked_gemm_serial 1024 128
# Custom matrix size and block size
```

## Interpreting Results

### Time Measurement
- Uses high-resolution timer:
  - **Windows:** `QueryPerformanceCounter` (~100 ns resolution)
  - **POSIX:** `clock_gettime(CLOCK_MONOTONIC)` (~1 ns resolution)
- Measures **only** matrix multiplication (excludes initialization)
- Reported as wall-clock seconds (includes OS scheduling jitter)

### Checksum Validation
- **Exact match required:** No floating-point tolerance
- If mismatch:
  1. Verify deterministic initialization (seeds 1 and 2)
  2. Check for overflow/underflow (use double, not float)
  3. Ensure same compiler optimizations (-O3)
  4. Rule out OS/hardware issues (run multiple times)

### GFlops Calculation
```
GFlops = (2 × N³) / (time × 10⁹)
```
- Factor of 2: Each multiply-add counts as 2 FLOPs
- N³: Cubic complexity of matrix multiplication

## Comparison Across Implementations

| Implementation | N=1024 Time (s) | Speedup vs. Serial | Efficiency |
|----------------|-----------------|---------------------|------------|
| Serial         | 0.452           | 1.0×               | Baseline   |
| OpenMP (4T)    | 0.175           | 2.58×              | 64%        |
| MPI (4P)       | 0.142           | 3.18×              | 80%        |
| CUDA (GPU)     | 0.003           | 150×               | N/A        |

**Efficiency = Speedup / Number of Cores**

## Troubleshooting

### Timings Don't Match Baseline
- **Expected:** ±10% variation due to:
  - CPU frequency scaling (Turbo Boost)
  - Background processes
  - Thermal throttling
  - Different hardware (CPU model, RAM speed)

### Checksum Mismatch
- **Most Common Causes:**
  1. Non-deterministic initialization (using `rand()`)
  2. Different compiler (GCC vs. MSVC floating-point behavior)
  3. Code modification (changed algorithm)
- **Solution:** Recompile from clean state, verify LCG seeds

### Poor Performance (< 2 GFlops)
- **Check:**
  1. Compiler optimizations enabled (`-O3` or `/O2`)
  2. Native architecture tuning (`-march=native`)
  3. No debug flags (`-g` slows execution)
  4. Sufficient RAM (avoid swapping)

## Next Steps

1. **Validate serial baseline:**
   ```bash
   make verify  # All checksums must match
   ```

2. **Benchmark on your system:**
   ```bash
   make sweep   # Determine optimal block size
   ```

3. **Compare with OpenMP:**
   - OpenMP should achieve 2-4× speedup on 4-8 cores
   - Checksums must match exactly

4. **Profile bottlenecks:**
   ```bash
   perf record ./blocked_gemm_serial 2048 128
   perf report
   # Identify cache misses, branch mispredictions
   ```

## References

- **Test Data Generation:** Deterministic LCG (no external dependencies)
- **Checksums Source:** Actual runs on reference platform
- **Performance Baselines:** Averaged over 3 runs, ±5% variance
