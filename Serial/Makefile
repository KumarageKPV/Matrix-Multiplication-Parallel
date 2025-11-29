# Makefile for blocked_gemm_serial.c
# Serial (single-threaded) baseline implementation
# Compatible with GCC, Clang, MSVC

CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O3 -march=native
TARGET = blocked_gemm_serial
SRC = blocked_gemm_serial.c

# Platform detection
ifeq ($(OS),Windows_NT)
    RM = del /Q
    EXE_EXT = .exe
    MKDIR = if not exist $(subst /,\,$(1)) mkdir $(subst /,\,$(1))
else
    RM = rm -f
    EXE_EXT =
    MKDIR = mkdir -p $(1)
endif

# Data directory
DATA_DIR = ../data/Serial

.PHONY: all clean test sweep verify help

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET)$(EXE_EXT) $(SRC) -lm

# Single test run (N=512, block=64)
test: $(TARGET)
	@echo "=== Testing N=512, block=64 ==="
	./$(TARGET)$(EXE_EXT) 512 64

# Block size sweep for N=1024
sweep: $(TARGET)
	@echo "=== Block size analysis (N=1024) ==="
	@echo "block_size,time_seconds" > $(DATA_DIR)/block_size_analysis.csv
	@for bs in 16 32 64 128 256; do \
		echo "Testing block size $$bs..."; \
		./$(TARGET)$(EXE_EXT) 1024 $$bs | grep -oP 'time=\K[0-9.]+' | \
		awk -v b=$$bs '{print b "," $$0}' >> $(DATA_DIR)/block_size_analysis.csv; \
	done
	@echo "Results saved to $(DATA_DIR)/block_size_analysis.csv"

# Verify checksums against expected values
verify: $(TARGET)
	@echo "=== Verifying checksums ==="
	@echo "Expected checksums:"
	@cat $(DATA_DIR)/expected_checksums.txt
	@echo ""
	@echo "Actual checksums:"
	@for n in 256 512 1024 2048; do \
		echo -n "N=$$n: "; \
		./$(TARGET)$(EXE_EXT) $$n 64 | grep -oP 'checksum=\K[0-9.]+'; \
	done

# Clean build artifacts
clean:
	$(RM) $(TARGET)$(EXE_EXT)

# Help message
help:
	@echo "Available targets:"
	@echo "  make          - Build the executable"
	@echo "  make test     - Run single test (N=512, block=64)"
	@echo "  make sweep    - Block size analysis for N=1024"
	@echo "  make verify   - Check checksums against expected values"
	@echo "  make clean    - Remove build artifacts"
	@echo ""
	@echo "Manual usage:"
	@echo "  ./$(TARGET)$(EXE_EXT) <N> <block_size>"
	@echo "  Example: ./$(TARGET)$(EXE_EXT) 1024 64"
