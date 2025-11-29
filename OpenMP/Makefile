# OpenMP Blocked GEMM - Comprehensive Makefile
# Targets:
#   make            - Build the OpenMP binary
#   make test       - Run quick correctness test
#   make sweep      - Run timing sweep (threads and block sizes)
#   make verify     - Compare checksums against serial baseline
#   make clean      - Remove build artifacts and output files
#
# Configuration:
#   Override defaults with: make CC=clang CFLAGS="-O3 -march=native"

# Compiler and flags
CC       = gcc
CFLAGS   = -O3 -std=c11 -fopenmp -Wall -Wextra
LDFLAGS  = -lm
SRC      = blocked_gemm_omp.c
BIN      = blocked_gemm_omp
DATADIR  = ../data/OpenMP

# Default matrix size and block size for testing
N        = 512
BS       = 64
THREADS  = 8

# Build targets
.PHONY: all clean test sweep verify benchmark help

all: $(BIN)

$(BIN): $(SRC)
	@echo "Building OpenMP GEMM with $(CC)..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Build complete: $@"

# Quick correctness test
test: $(BIN)
	@echo "Running correctness test (N=$(N), block=$(BS), threads=$(THREADS))..."
	./$(BIN) $(N) $(BS) $(THREADS)

# Thread scalability sweep
sweep-threads: $(BIN)
	@echo "Thread scalability sweep (N=$(N), block=$(BS))..."
	@echo "Threads,Time(s),Checksum" > timings_threads.csv
	@for t in 1 2 4 8 12 16; do \
		output=$$(./$(BIN) $(N) $(BS) $$t); \
		time=$$(echo "$$output" | grep -oP 'time=\K[0-9.]+'); \
		chksum=$$(echo "$$output" | grep -oP 'checksum=\K[0-9.]+'); \
		echo "$$t,$$time,$$chksum" >> timings_threads.csv; \
	done
	@echo "Results saved to timings_threads.csv"

# Block size sweep
sweep-blocks: $(BIN)
	@echo "Block size sweep (N=$(N), threads=$(THREADS))..."
	@echo "BlockSize,Time(s),Checksum" > timings_blocks.csv
	@for bs in 16 32 64 128 256; do \
		output=$$(./$(BIN) $(N) $$bs $(THREADS)); \
		time=$$(echo "$$output" | grep -oP 'time=\K[0-9.]+'); \
		chksum=$$(echo "$$output" | grep -oP 'checksum=\K[0-9.]+'); \
		echo "$$bs,$$time,$$chksum" >> timings_blocks.csv; \
	done
	@echo "Results saved to timings_blocks.csv"

# Full benchmark sweep
sweep: sweep-threads sweep-blocks

# Verify against serial baseline (assumes serial binary exists)
verify: $(BIN)
	@echo "Verifying OpenMP output against serial baseline..."
	@serial_output=$$(../Serial/blocked_gemm_serial $(N) $(BS)); \
	serial_chk=$$(echo "$$serial_output" | grep -oP 'checksum=\K[0-9.]+'); \
	omp_output=$$(./$(BIN) $(N) $(BS) 1); \
	omp_chk=$$(echo "$$omp_output" | grep -oP 'checksum=\K[0-9.]+'); \
	echo "Serial checksum:  $$serial_chk"; \
	echo "OpenMP checksum:  $$omp_chk"; \
	if [ "$$(echo "$$serial_chk $$omp_chk" | awk '{d=$$1-$$2; if(d<0)d=-d; print (d<0.01)?"match":"differ"}')" = "match" ]; then \
		echo "✓ Checksums match - PASS"; \
	else \
		echo "✗ Checksums differ - FAIL"; \
		exit 1; \
	fi

# Run comprehensive benchmark with multiple sizes
benchmark: $(BIN)
	@echo "Running comprehensive benchmark..."
	@echo "N,BlockSize,Threads,Time(s),Checksum" > benchmark_results.csv
	@for n in 256 512 1024; do \
		for bs in 32 64 128; do \
			for t in 1 2 4 8; do \
				output=$$(./$(BIN) $$n $$bs $$t); \
				time=$$(echo "$$output" | grep -oP 'time=\K[0-9.]+'); \
				chksum=$$(echo "$$output" | grep -oP 'checksum=\K[0-9.]+'); \
				echo "$$n,$$bs,$$t,$$time,$$chksum" >> benchmark_results.csv; \
			done; \
		done; \
	done
	@echo "Benchmark complete: benchmark_results.csv"

# Clean build artifacts and output files
clean:
	rm -f $(BIN) *.o *.csv *.log
	@echo "Clean complete"

# Help target
help:
	@echo "OpenMP GEMM Makefile - Available targets:"
	@echo "  make              - Build the binary"
	@echo "  make test         - Run quick test (N=$(N), BS=$(BS), T=$(THREADS))"
	@echo "  make sweep        - Run timing sweeps (threads and block sizes)"
	@echo "  make verify       - Verify against serial baseline"
	@echo "  make benchmark    - Comprehensive benchmark across sizes"
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "Override defaults: make test N=1024 BS=128 THREADS=16"
