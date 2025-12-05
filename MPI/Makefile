# Makefile for MPI blocked GEMM

MPICC ?= mpicc
# Auto-detect MPI implementation and set appropriate flags
# MS-MPI (Windows) doesn't support --oversubscribe; Open MPI does
MPI_VERSION := $(shell mpiexec --version 2>&1)
ifneq (,$(findstring Microsoft MPI,$(MPI_VERSION)))
    MPIRUN ?= mpiexec
else
    MPIRUN ?= mpiexec --oversubscribe
endif
CFLAGS = -O3 -std=c11 -Wall -Wextra -march=native
TARGET = blocked_gemm_mpi
SRC = blocked_gemm_mpi.c

# Default process count for tests
PROCS ?= 4
TOL ?= 1e-4

.PHONY: all clean test verify sweep procscale help

all: $(TARGET)

$(TARGET): $(SRC)
	$(MPICC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

# Simple correctness test
# N=512, block=64
# Adjust PROCS for different process counts

test: $(TARGET)
	@echo "=== MPI Test: N=512 block=64 procs=$(PROCS) ==="
	$(MPIRUN) -n $(PROCS) ./$(TARGET) 512 64

# Verify checksums across multiple sizes (must match Serial baseline)
verify: $(TARGET)
	@echo "=== Verifying MPI checksums (tolerance $(TOL)) ==="
	@pass=1; \
	for n in 256 512 1024 2048; do \
		exp=$$(grep "^N=$$n:" ../data/MPI/expected_checksums.txt | awk '{print $$2}'); \
		line=$$($(MPIRUN) -n $(PROCS) ./$(TARGET) $$n 128); \
		actual=$$(echo $$line | grep -oP 'checksum=\K[0-9.]+' ); \
		diff=$$(awk -v a=$$actual -v e=$$exp 'BEGIN{d=a-e; if (d<0) d=-d; print d}'); \
		echo "N=$$n expected=$$exp actual=$$actual diff=$$diff"; \
		awk -v d=$$diff -v t=$(TOL) 'BEGIN{if (d>t){exit 1}}' || pass=0; \
	done; \
	if [ $$pass -eq 1 ]; then echo "\xE2\x9C\x93 All checksums within tolerance - PASS"; else echo "\xE2\x9D\x8C Checksum mismatch exceeds tolerance - FAIL"; exit 1; fi
	@echo "Reference file: ../data/MPI/expected_checksums.txt"

# Block size sweep (fixed N=1024)
# Output CSV: block_size, time_seconds
sweep: $(TARGET)
	@echo "block_size,time_seconds,GFLOPs,N,procs" > ../data/MPI/block_size_analysis.csv
	@for bs in 16 32 64 128 256; do \
		printf "Testing block size $$bs...\n"; \
		line=$$($(MPIRUN) -n $(PROCS) ./$(TARGET) 1024 $$bs); \
		time=$$(echo $$line | grep -oP 'time=\K[0-9.]+'); \
		gflops=$$(awk -v t=$$time 'BEGIN{ops=2*1024*1024*1024; print ops/(t*1e9)}'); \
		echo "$$bs,$$time,$$gflops,1024,$(PROCS)" >> ../data/MPI/block_size_analysis.csv; \
	done
	@echo "Saved to ../data/MPI/block_size_analysis.csv"

# Process scaling (N=1024, block=128)
# Vary number of processes: 1 2 4 8
# Output CSV: procs,time_seconds

baseline: $(TARGET)
	@echo "N,procs,block_size,time_seconds,checksum,GFLOPs" > ../data/MPI/baseline_timings.txt
	@for p in 1 2 4 8; do \
		for n in 256 512 1024 2048; do \
			line=$$($(MPIRUN) -n $$p ./$(TARGET) $$n 128); \
			time=$$(echo $$line | grep -oP 'time=\K[0-9.]+' ); \
			chk=$$(echo $$line | grep -oP 'checksum=\K[0-9.]+' ); \
			gflops=$$(echo $$line | grep -oP 'GFLOPs=\K[0-9.]+' ); \
			echo "$$n,$$p,128,$$time,$$chk,$$gflops" >> ../data/MPI/baseline_timings.txt; \
		done; \
	done
	@echo "Saved baseline to ../data/MPI/baseline_timings.txt"

procscale: $(TARGET)
	@echo "procs,time_seconds,speedup,efficiency,GFLOPs,N,block_size" > ../data/MPI/proc_scaling.csv
	@echo "Measuring baseline (1 proc)..."; \
	base_time=$$($(MPIRUN) -n 1 ./$(TARGET) 1024 128 | grep -oP 'time=\K[0-9.]+' ); \
	gflops_base=$$(awk -v t=$$base_time 'BEGIN{ops=2*1024*1024*1024; print ops/(t*1e9)}'); \
	echo "Baseline T1=$$base_time"; \
	echo "1,$$base_time,1.0,1.0,$$gflops_base,1024,128" >> ../data/MPI/proc_scaling.csv; \
	for p in 2 4 8; do \
		printf "Testing procs=$$p...\n"; \
		cur_time=$$($(MPIRUN) -n $$p ./$(TARGET) 1024 128 | grep -oP 'time=\K[0-9.]+' ); \
		gflops=$$(awk -v t=$$cur_time 'BEGIN{ops=2*1024*1024*1024; print ops/(t*1e9)}'); \
		speedup=$$(awk -v t1=$$base_time -v tp=$$cur_time 'BEGIN{print t1/tp}'); \
		efficiency=$$(awk -v s=$$speedup -v p=$$p 'BEGIN{print s/p}'); \
		echo "$$p,$$cur_time,$$speedup,$$efficiency,$$gflops,1024,128" >> ../data/MPI/proc_scaling.csv; \
	done; \
	echo "Saved to ../data/MPI/proc_scaling.csv"

clean:
	$(RM) $(TARGET) *.o

help:
	@echo "Targets:";
	@echo "  make            - Build binary";
	@echo "  make test       - Run single test (N=512 block=64)";
	@echo "  make verify     - Verify checksums (N=256,512,1024)";
	@echo "  make sweep      - Block size sweep for N=1024";
	@echo "  make procscale  - Process scaling study";
	@echo "  PROCS=<p> make test   # Override process count";
