# CUDA Build and Test Script for Windows PowerShell
# Alternative to Makefile for systems without make

param(
    [string]$Target = "build",
    [string]$Arch = "sm_75"  # GTX 1650 default
)

$ErrorActionPreference = "Stop"
$DataDir = "..\data\CUDA"
$Executable = "blocked_gemm_cuda.exe"

function Build-CUDA {
    Write-Host "=== Building CUDA GEMM ===" -ForegroundColor Cyan
    Write-Host "Architecture: $Arch"
    
    nvcc -O3 -arch=$Arch -o $Executable blocked_gemm_cuda.cu
    
    if (Test-Path $Executable) {
        Write-Host "Build successful!" -ForegroundColor Green
    } else {
        Write-Host "Build failed!" -ForegroundColor Red
        exit 1
    }
}

function Test-CUDA {
    Write-Host "=== Single Test Run ===" -ForegroundColor Cyan
    & ".\$Executable" 512 32
}

function Verify-Checksums {
    Write-Host "=== Checksum Verification ===" -ForegroundColor Cyan
    Write-Host "Comparing against expected checksums in $DataDir\expected_checksums.txt"
    
    Write-Host "`nN=256 block=32:"
    & ".\$Executable" 256 32
    
    Write-Host "`nN=512 block=32:"
    & ".\$Executable" 512 32
    
    Write-Host "`nN=1024 block=32:"
    & ".\$Executable" 1024 32
    
    Write-Host "`nN=2048 block=32:"
    & ".\$Executable" 2048 32
    
    Write-Host "`nVerification complete. Compare checksums with $DataDir\expected_checksums.txt" -ForegroundColor Green
}

function Run-Sweep {
    Write-Host "=== Block Size Sweep (N=1024) ===" -ForegroundColor Cyan
    
    $csvFile = "$DataDir\block_size_analysis.csv"
    "N,block_size,time_sec,checksum,GFLOPs" | Out-File -FilePath $csvFile -Encoding utf8
    
    # Block sizes limited by GPU constraints (GTX 1650: max 1024 threads/block = 32x32)
    $blockSizes = @(8, 16, 24, 32)
    
    foreach ($bs in $blockSizes) {
        Write-Host "Testing block_size=$bs..."
        $output = & ".\$Executable" 1024 $bs
        
        # Parse output: "CUDA GEMM: N=1024 block=32 time=0.012345 sec checksum=268310302.3759 GFLOPs=1750.23"
        if ($output -match "N=(\d+)\s+block=(\d+)\s+time=([\d.]+)\s+sec\s+checksum=([\d.]+)\s+GFLOPs=([\d.]+)") {
            $n = $matches[1]
            $block = $matches[2]
            $time = $matches[3]
            $checksum = $matches[4]
            $gflops = $matches[5]
            
            "$n,$block,$time,$checksum,$gflops" | Out-File -FilePath $csvFile -Append -Encoding utf8
        }
    }
    
    Write-Host "Results written to $csvFile" -ForegroundColor Green
}

function Run-Baseline {
    Write-Host "=== Baseline Timings ===" -ForegroundColor Cyan
    
    $txtFile = "$DataDir\baseline_timings.txt"
    "=== Baseline Timings ===" | Out-File -FilePath $txtFile -Encoding utf8
    "Format: N block_size time(s) checksum GFLOPs" | Out-File -FilePath $txtFile -Append -Encoding utf8
    "" | Out-File -FilePath $txtFile -Append -Encoding utf8
    
    $sizes = @(256, 512, 1024, 2048)
    # Block sizes limited by GPU constraints (GTX 1650: max 1024 threads/block = 32x32)
    $blockSizes = @(16, 32)
    
    foreach ($n in $sizes) {
        foreach ($bs in $blockSizes) {
            Write-Host "Testing N=$n block=$bs..."
            $output = & ".\$Executable" $n $bs
            $output | Out-File -FilePath $txtFile -Append -Encoding utf8
        }
        "" | Out-File -FilePath $txtFile -Append -Encoding utf8
    }
    
    Write-Host "Results written to $txtFile" -ForegroundColor Green
}

function Clean-Build {
    Write-Host "=== Cleaning Build Artifacts ===" -ForegroundColor Cyan
    
    if (Test-Path $Executable) {
        Remove-Item $Executable
        Write-Host "Removed $Executable"
    }
    
    Get-ChildItem -Filter "*.exp" | Remove-Item
    Get-ChildItem -Filter "*.lib" | Remove-Item
    
    Write-Host "Clean complete!" -ForegroundColor Green
}

function Show-Help {
    Write-Host @"
CUDA Build Script for Windows PowerShell

Usage:
    .\build.ps1 [-Target <target>] [-Arch <arch>]

Targets:
    build       - Build blocked_gemm_cuda.exe (default)
    test        - Single test run (N=512, block=32)
    verify      - Verify checksums across multiple sizes
    sweep       - Block size sweep, writes block_size_analysis.csv
    baseline    - Full baseline timings, writes baseline_timings.txt
    clean       - Remove build artifacts
    help        - Show this help message

Architecture (use -Arch):
    sm_75       - RTX 20xx, Tesla T4, GTX 1650 (default)
    sm_80       - A100
    sm_86       - RTX 30xx
    sm_89       - RTX 40xx, L4

Examples:
    .\build.ps1
    .\build.ps1 -Target test
    .\build.ps1 -Target build -Arch sm_86
    .\build.ps1 -Target verify

Direct nvcc commands:
    nvcc -O3 -arch=sm_75 -o blocked_gemm_cuda.exe blocked_gemm_cuda.cu
    .\blocked_gemm_cuda.exe 1024 32
"@ -ForegroundColor Yellow
}

# Main execution
switch ($Target.ToLower()) {
    "build"    { Build-CUDA }
    "test"     { Test-CUDA }
    "verify"   { Verify-Checksums }
    "sweep"    { Run-Sweep }
    "baseline" { Run-Baseline }
    "clean"    { Clean-Build }
    "help"     { Show-Help }
    default    { 
        Write-Host "Unknown target: $Target" -ForegroundColor Red
        Write-Host "Use -Target help for available targets"
        exit 1
    }
}
