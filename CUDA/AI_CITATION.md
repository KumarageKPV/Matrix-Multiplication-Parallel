# AI Assistance Citation

This CUDA implementation was developed with assistance from GitHub Copilot (Claude Sonnet 4.5) through iterative prompting. All final algorithm decisions, code structure, and documentation wording were reviewed and authored for originality and clarity.

## Prompts Used (Representative)
1. "Implement CUDA blocked matrix multiplication with shared memory tiling."
2. "Add deterministic LCG initialization consistent with Serial/OpenMP/MPI implementations."
3. "Explain shared memory benefits and coalesced memory access patterns in comments."
4. "Provide Makefile targets for test, verify, sweep, and baseline benchmarking."
5. "Add CUDA event-based timing and checksum validation for correctness."

## Scope of AI Contributions
- Skeleton generation for CUDA kernel setup (grid/block dimensions, shared memory)
- Suggestions for shared memory optimization patterns
- Formatting of documentation sections (Parallelization Strategy, Memory Access)
- Error checking macro template (CUDA_CHECK)

## Human Contributions
- Choice of tiled algorithm with shared memory (vs register-only or naive approaches)
- Blocked kernel design adapted from serial blocking approach
- Deterministic initialization and checksum integration for cross-version correctness
- Performance analysis guidance and GPU architecture notes
- Kernel launch configuration tuning

## Originality & Integrity
No external CUDA GEMM source code copied. Concepts align with standard GPU matrix multiplication best practices (tiling, shared memory, coalesced access). All code is self-contained and independently verifiable.

---
