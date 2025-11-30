# Screenshot Capture Guide
## Part B: Performance Evaluation

This guide helps you capture all required screenshots for the performance evaluation.

## Required Screenshots

### 1. OpenMP Evaluation (Thread Scalability)

Capture screenshots showing execution with different thread counts:

```powershell
cd OpenMP

# 1 thread
$env:OMP_NUM_THREADS=1; .\blocked_gemm_omp.exe 512 64 1
# Screenshot: openmp_1thread.png

# 2 threads
$env:OMP_NUM_THREADS=2; .\blocked_gemm_omp.exe 512 64 2
# Screenshot: openmp_2threads.png

# 4 threads
$env:OMP_NUM_THREADS=4; .\blocked_gemm_omp.exe 512 64 4
# Screenshot: openmp_4threads.png

# 8 threads
$env:OMP_NUM_THREADS=8; .\blocked_gemm_omp.exe 512 64 8
# Screenshot: openmp_8threads.png

# 16 threads
$env:OMP_NUM_THREADS=16; .\blocked_gemm_omp.exe 512 64 16
# Screenshot: openmp_16threads.png
```

### 2. MPI Evaluation (Process Scalability)

Capture screenshots showing execution with different process counts:

```bash
cd MPI

# 1 process
mpiexec --oversubscribe -n 1 ./blocked_gemm_mpi 1024 128
# Screenshot: mpi_1proc.png

# 2 processes
mpiexec --oversubscribe -n 2 ./blocked_gemm_mpi 1024 128
# Screenshot: mpi_2procs.png

# 4 processes
mpiexec --oversubscribe -n 4 ./blocked_gemm_mpi 1024 128
# Screenshot: mpi_4procs.png

# 8 processes
mpiexec --oversubscribe -n 8 ./blocked_gemm_mpi 1024 128
# Screenshot: mpi_8procs.png
```

### 3. CUDA Evaluation (Block Size Configurations)

Capture screenshots showing execution with different block sizes:

```powershell
cd CUDA

# Block size 8
.\blocked_gemm_cuda.exe 1024 8
# Screenshot: cuda_block8.png

# Block size 16
.\blocked_gemm_cuda.exe 1024 16
# Screenshot: cuda_block16.png

# Block size 24
.\blocked_gemm_cuda.exe 1024 24
# Screenshot: cuda_block24.png

# Block size 32
.\blocked_gemm_cuda.exe 1024 32
# Screenshot: cuda_block32.png
```

### 4. Comparative Runs (Same Problem Size)

Capture all implementations running N=1024:

```powershell
# Serial
cd Serial
.\blocked_gemm_serial.exe 1024 64
# Screenshot: serial_1024.png

# OpenMP (best config)
cd ../OpenMP
.\blocked_gemm_omp.exe 1024 64 8
# Screenshot: openmp_1024.png

# MPI (best config)
cd ../MPI
mpiexec --oversubscribe -n 4 ./blocked_gemm_mpi 1024 128
# Screenshot: mpi_1024.png

# CUDA (best config)
cd ../CUDA
.\blocked_gemm_cuda.exe 1024 16
# Screenshot: cuda_1024.png
```

## Screenshot Organization

Create directory structure:
```
screenshots/
├── openmp/
│   ├── openmp_1thread.png
│   ├── openmp_2threads.png
│   ├── openmp_4threads.png
│   ├── openmp_8threads.png
│   └── openmp_16threads.png
├── mpi/
│   ├── mpi_1proc.png
│   ├── mpi_2procs.png
│   ├── mpi_4procs.png
│   └── mpi_8procs.png
├── cuda/
│   ├── cuda_block8.png
│   ├── cuda_block16.png
│   ├── cuda_block24.png
│   └── cuda_block32.png
└── comparative/
    ├── serial_1024.png
    ├── openmp_1024.png
    ├── mpi_1024.png
    └── cuda_1024.png
```

## Tips for Good Screenshots

1. **Terminal Size**: Use full screen or large terminal window
2. **Font Size**: Increase terminal font for readability
3. **Clean Output**: Clear terminal before each command
4. **Highlight**: Consider highlighting key output (time, speedup)
5. **Context**: Include the command being executed
6. **Quality**: Use PNG format, high resolution
7. **Annotation**: Add labels or captions if needed

## Automated Screenshot Collection (Windows PowerShell)

```powershell
# Create screenshot directories
New-Item -ItemType Directory -Force -Path screenshots/openmp
New-Item -ItemType Directory -Force -Path screenshots/mpi
New-Item -ItemType Directory -Force -Path screenshots/cuda
New-Item -ItemType Directory -Force -Path screenshots/comparative

# Note: Screenshots must be captured manually using print screen
# or screen capture tool. This script provides the commands.

Write-Host "Execute these commands and capture screenshots:"
Write-Host "================================================"
Write-Host ""
Write-Host "OPENMP TESTS:"
Write-Host "-------------"
$threads = @(1, 2, 4, 8, 16)
foreach ($t in $threads) {
    Write-Host ".\blocked_gemm_omp.exe 512 64 $t"
    Write-Host "  -> Save as: screenshots/openmp/openmp_${t}threads.png"
    Write-Host ""
}

Write-Host "MPI TESTS:"
Write-Host "----------"
$procs = @(1, 2, 4, 8)
foreach ($p in $procs) {
    Write-Host "mpiexec --oversubscribe -n $p ./blocked_gemm_mpi 1024 128"
    Write-Host "  -> Save as: screenshots/mpi/mpi_${p}procs.png"
    Write-Host ""
}

Write-Host "CUDA TESTS:"
Write-Host "-----------"
$blocks = @(8, 16, 24, 32)
foreach ($b in $blocks) {
    Write-Host ".\blocked_gemm_cuda.exe 1024 $b"
    Write-Host "  -> Save as: screenshots/cuda/cuda_block${b}.png"
    Write-Host ""
}
```

## Checklist

- [ ] OpenMP: 5 screenshots (1, 2, 4, 8, 16 threads)
- [ ] MPI: 4 screenshots (1, 2, 4, 8 processes)
- [ ] CUDA: 4 screenshots (8, 16, 24, 32 block sizes)
- [ ] Comparative: 4 screenshots (Serial, OpenMP, MPI, CUDA at N=1024)
- [ ] All screenshots clearly show output
- [ ] All screenshots include execution time
- [ ] All screenshots show checksum for verification
- [ ] Screenshots organized in proper directories

## Total Screenshots Required: 17

After capturing all screenshots, verify they are all present:

```powershell
Get-ChildItem -Path screenshots -Recurse -Filter *.png | Measure-Object
# Should show Count: 17
```
