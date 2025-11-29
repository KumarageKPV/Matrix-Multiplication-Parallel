# OpenMP Test Data and Verification Files

This directory contains input data, expected outputs, and benchmark results for the OpenMP implementation.

## Contents

### Input Specifications
- `test_config.txt` - Test configuration parameters
- `matrix_sizes.txt` - Matrix dimensions used for testing

### Expected Outputs
- `expected_checksums.txt` - Reference checksums for verification
- `baseline_timings.txt` - Serial baseline timings for speedup calculation

### Benchmark Results
- `thread_scalability.csv` - Performance across different thread counts
- `block_size_analysis.csv` - Performance across different block sizes
- `full_benchmark.csv` - Comprehensive results (multiple N, block sizes, threads)

### Sample Run Logs
- `sample_run_512.log` - Example output for N=512
- `sample_run_1024.log` - Example output for N=1024

## Test Configurations

### Small Test (Quick Verification)
- N = 256
- Block sizes: 32, 64
- Threads: 1, 2, 4, 8
- Purpose: Fast correctness check

### Medium Test (Standard Evaluation)
- N = 512
- Block sizes: 32, 64, 128
- Threads: 1, 2, 4, 8, 12, 16
- Purpose: Main performance analysis

### Large Test (Scalability)
- N = 1024, 2048
- Block sizes: 64, 128, 256
- Threads: 1, 2, 4, 8, 16, 32
- Purpose: Assess scalability to larger problem sizes

## Verification Process

1. **Checksum Verification**:
   ```bash
   cd ../OpenMP
   make verify
   ```
   Compares OpenMP checksum against serial baseline for N=512, block=64

2. **Manual Verification**:
   ```bash
   # Run serial
   ../Serial/blocked_gemm_serial 512 64
   
   # Run OpenMP
   ./blocked_gemm_omp 512 64 1
   
   # Compare checksums (should match exactly)
   ```

3. **Timing Validation**:
   - OpenMP with 1 thread should be ≈ serial time (slight overhead acceptable)
   - Speedup should increase with thread count (diminishing returns after 8-16 threads)
   - Block size 64-128 typically fastest for modern CPUs

## Expected Checksums (Deterministic)

Using the LCG initializer with seeds 1 and 2:

| N    | Checksum           |
|------|--------------------|
| 256  | 8,372,517.938617   |
| 512  | 33,526,118.721088  |
| 1024 | 134,291,354.502563 |
| 2048 | 537,648,959.339478 |

*Note: Checksums are deterministic and should match exactly across all implementations (serial, OpenMP, MPI, CUDA).*

## Generating Test Data

To generate fresh benchmark data:

```bash
cd ../OpenMP

# Thread scalability
make sweep-threads

# Block size analysis
make sweep-blocks

# Full benchmark
make benchmark
```

Results will be saved as CSV files in the OpenMP directory.

## Data Files Description

### test_config.txt
Configuration parameters for reproducible testing:
- Matrix sizes tested
- Block size range
- Thread count range
- Compiler flags used
- System specifications

### expected_checksums.txt
Reference checksums for each test configuration to verify correctness.

### baseline_timings.txt
Serial implementation timings for speedup calculation:
- Speedup = T_serial / T_parallel

### CSV Output Format

**thread_scalability.csv**:
```
Threads,Time(s),Checksum
1,0.043723,33526118.721088
2,0.024156,33526118.721088
4,0.013892,33526118.721088
8,0.009127,33526118.721088
```

**block_size_analysis.csv**:
```
BlockSize,Time(s),Checksum
16,0.015234,33526118.721088
32,0.011567,33526118.721088
64,0.009127,33526118.721088
128,0.009845,33526118.721088
```

**full_benchmark.csv**:
```
N,BlockSize,Threads,Time(s),Checksum
256,32,1,0.005123,8372517.938617
256,32,2,0.002891,8372517.938617
...
```

## Notes

- All timings are wall-clock time in seconds
- Checksums must match across all parallel implementations
- Performance varies by CPU architecture, clock speed, and system load
- Run benchmarks when system is idle for consistent results
