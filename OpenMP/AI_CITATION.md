# AI Assistance Citation

This OpenMP implementation was developed with assistance from GitHub Copilot (GPT-5) through iterative prompting. All final algorithm decisions, code structure, and documentation wording were reviewed and authored for originality and clarity.

## Prompts Used (Representative)
1. "Implement Phase 2 Part A OpenMP blocked matrix multiplication with deterministic init."
2. "Add portable timer for Windows and POSIX in C."
3. "Explain blocking strategy and thread scalability in README format."
4. "Provide Makefile targets for test and verification with checksum." 
5. "Align OpenMP checksum with serial baseline deterministic LCG seeds."

## Scope of AI Contributions
- Skeleton generation for blocked OpenMP loops (`collapse(2)` tiling structure)
- Cross-platform timing snippet suggestions
- Alignment strategy recommendations (64-byte, `_aligned_malloc` / `posix_memalign`)
- Draft wording for README performance and verification sections

## Human Contributions
- Selection of tile sizes and verification matrices
- Deterministic LCG initialization integration and checksum validation
- Manual tuning of scheduling (static over tile indices) and reduction removal for deterministic sums
- Refinement of documentation to match assignment standards

## Originality & Integrity
No external GEMM source code copied. Concepts reflect standard cache-blocked matrix multiplication and OpenMP parallel for usage. Code is self-contained and verifiable against the serial baseline.

---
