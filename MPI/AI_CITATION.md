# AI Assistance Citation

This MPI implementation was developed with assistance from GitHub Copilot (GPT-5) through iterative prompting. All final algorithm decisions, code structure, and documentation wording were reviewed and authored for originality and clarity.

## Prompts Used (Representative)
1. "Implement MPI blocked matrix multiplication using row distribution and broadcast B."
2. "Add checksum reduction and gather final C to root with performance output."
3. "Explain blocking benefits and row partition strategy in README format."
4. "Provide Makefile targets for test, verify, sweep, and process scaling."
5. "Add deterministic LCG initialization consistent with Serial implementation."

## Scope of AI Contributions
- Skeleton generation for MPI setup (init, rank/size queries)
- Suggestions for collective communication patterns
- Formatting of documentation sections (Strategy, Verification, Performance)

## Human Contributions
- Choice of row distribution strategy (simplicity vs 2D decomposition)
- Blocked local kernel design & adaptation from serial approach
- Deterministic initialization and checksum integration for cross-version correctness
- Performance analysis guidance and extensibility notes

## Originality & Integrity
No external MPI GEMM source code copied. Concepts align with standard parallel matrix multiplication best practices. All code is self-contained and independently verifiable.

---
