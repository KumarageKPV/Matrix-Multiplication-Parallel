# CUDA Blocked GEMM Makefile
# Provides targets for building, testing, and benchmarking CUDA matrix multiplication

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++11
ARCH = sm_75  # Default architecture (adjust for your GPU)
# Common architectures:
#   sm_75: RTX 20xx, Tesla T4
#   sm_80: A100
#   sm_86: RTX 30xx
#   sm_89: RTX 40xx, L4
TARGET = blocked_gemm_cuda

# Directories
DATA_DIR = ../data/CUDA

# Default sizes for testing
TEST_N = 512
TEST_BLOCK = 32

# Build the CUDA executable
all: $(TARGET)

$(TARGET): blocked_gemm_cuda.cu
	$(NVCC) $(NVCC_FLAGS) -arch=$(ARCH) -o $(TARGET) blocked_gemm_cuda.cu

# Single test run
test: $(TARGET)
	@echo "=== Single Test Run ==="
	./$(TARGET) $(TEST_N) $(TEST_BLOCK)

# Verify checksums across multiple matrix sizes
verify: $(TARGET)
	@echo "=== Checksum Verification ==="
	@echo "Comparing against expected checksums in $(DATA_DIR)/expected_checksums.txt"
	@echo "N=256 block=32:"
	@./$(TARGET) 256 32
	@echo "N=512 block=32:"
	@./$(TARGET) 512 32
	@echo "N=1024 block=32:"
	@./$(TARGET) 1024 32
	@echo "N=2048 block=32:"
	@./$(TARGET) 2048 32
	@echo "Verification complete. Compare checksums with $(DATA_DIR)/expected_checksums.txt"

# Block size sweep (analyze performance vs block size)
sweep: $(TARGET)
	@echo "=== Block Size Sweep (N=1024) ==="
	@echo "N,block_size,time_sec,checksum,GFLOPs" > $(DATA_DIR)/block_size_analysis.csv
	@for bs in 8 16 24 32 48 64; do \
		./$(TARGET) 1024 $$bs | \
		awk '{print "1024," $$3 "," $$5 "," $$7 "," $$9}' | \
		sed 's/block=//' | sed 's/time=//' | sed 's/checksum=//' | sed 's/GFLOPs=//' \
		>> $(DATA_DIR)/block_size_analysis.csv; \
	done
	@echo "Results written to $(DATA_DIR)/block_size_analysis.csv"

# Baseline timings (various matrix sizes)
baseline: $(TARGET)
	@echo "=== Baseline Timings ===" | tee $(DATA_DIR)/baseline_timings.txt
	@echo "Format: N block_size time(s) checksum GFLOPs" | tee -a $(DATA_DIR)/baseline_timings.txt
	@echo "" | tee -a $(DATA_DIR)/baseline_timings.txt
	@for n in 256 512 1024 2048; do \
		for bs in 16 32 64; do \
			./$(TARGET) $$n $$bs | tee -a $(DATA_DIR)/baseline_timings.txt; \
		done; \
		echo "" | tee -a $(DATA_DIR)/baseline_timings.txt; \
	done
	@echo "Results written to $(DATA_DIR)/baseline_timings.txt"

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Help target
help:
	@echo "Available targets:"
	@echo "  make           - Build blocked_gemm_cuda"
	@echo "  make test      - Single test run (N=$(TEST_N), block=$(TEST_BLOCK))"
	@echo "  make verify    - Verify checksums across multiple sizes"
	@echo "  make sweep     - Block size sweep, writes $(DATA_DIR)/block_size_analysis.csv"
	@echo "  make baseline  - Full baseline timings, writes $(DATA_DIR)/baseline_timings.txt"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  ARCH=$(ARCH) (adjust for your GPU)"
	@echo "  TEST_N=$(TEST_N)"
	@echo "  TEST_BLOCK=$(TEST_BLOCK)"

.PHONY: all test verify sweep baseline clean help
