# MPI Blocked GEMM Implementation

MPI-based parallel matrix multiplication with optional cache blocking. This implementation distributes rows of matrices across processes and broadcasts the full B matrix for local computation.

## 1. Overview

- Parallelization model: **Distributed-memory (MPI)**
- Decomposition: **Row-wise partition of A and C**, full replication of B
- Algorithmic options:
  - Naive triple-loop GEMM (if block_size <= 0)
  - Blocked (tiled) GEMM with block size `bs`
- Deterministic initialization (LCG) for reproducibility and checksum validation
- High-resolution timing using `MPI_Wtime()`

## 2. Build Instructions

### Linux / macOS (OpenMPI or MPICH)
```bash
mpicc -O3 -std=c11 -Wall -Wextra -o blocked_gemm_mpi blocked_gemm_mpi.c -lm
```
Or using the provided Makefile:
```bash
cd MPI
make
```

### Windows (MSYS2 + OpenMPI)
```bash
pacman -S mingw-w64-x86_64-openmpi
mpicc -O3 -std=c11 -Wall -Wextra -o blocked_gemm_mpi blocked_gemm_mpi.c -lm
```

### Windows (MS-MPI + Visual Studio Developer Prompt)
```cmd
cl /O2 /W4 /std:c11 blocked_gemm_mpi.c msmpi.lib
```

## 3. Running

```bash
mpiexec -n <P> ./blocked_gemm_mpi <N> <block_size>
```
Examples:
```bash
mpiexec -n 4 ./blocked_gemm_mpi 512 64
mpiexec -n 8 ./blocked_gemm_mpi 1024 128
mpiexec -n 4 ./blocked_gemm_mpi 1024 0   # naive GEMM
```

## 4. Output Format
```
MPI GEMM: N=1024 block=128 procs=4 time=0.123456 sec checksum=268310302.375923 GFLOPs=14.85
```
Fields:
- `N`: matrix dimension
- `block`: block (tile) size (0 or negative means naive)
- `procs`: number of MPI ranks
- `time`: wall-clock time (seconds)
- `checksum`: sum of all elements in result matrix C (must match Serial baseline)
- `GFLOPs`: (2 * N^3) / (time * 1e9)

## 5. Parallelization Strategy

### Row Distribution
- Partition N rows among P processes.
- Each rank receives `rows_per_rank[r]` contiguous rows of A.
- B is broadcast to all ranks: minimal communication vs. 2D decomposition.
- Each rank computes its portion of C independently.
- Final result gathered (optional) or checksum reduced.

### Why Broadcast B?
- Fewer collective operations: one `MPI_Bcast` vs. multiple gathers.
- Simplicity: each rank performs complete dot products for its rows.
- Good fit for moderate matrix sizes where memory replication is acceptable.

### Blocking Benefits
- Operate on bs×bs tiles for better cache locality.
- Reuse loaded cache lines for A_local and B_full.
- Reduces main memory traffic and improves arithmetic intensity.

## 6. Communication Pattern
| Operation | Calls | Purpose |
|-----------|-------|---------|
| `MPI_Scatterv` | 1 | Distribute rows of A to ranks |
| `MPI_Bcast` | 1 | Broadcast full B matrix |
| `MPI_Barrier` | 2 | Synchronize before/after timed computation |
| `MPI_Reduce` | 1 | Aggregate checksum (sum) |
| `MPI_Gatherv` | 1 | Collect final C (optional) |

## 7. Complexity Analysis
- Serial compute: O(N^3)
- Communication volume:
  - Scatter A: ~ (N^2) doubles total
  - Broadcast B: N^2 doubles
  - Gather C: N^2 doubles
- Total data moved: ~3N^2 doubles (independent of block size)
- Scalability limited by broadcast/gather overhead as P grows.

## 8. Block Size Selection
General guidance (same as Serial/OpenMP):
- L1 cache (32–64 KB): bs ≈ 32–64
- L2 cache (256 KB–1 MB): bs ≈ 64–128
- Avoid bs so large that single tile exceeds L2.

Use Makefile sweep:
```bash
make sweep PROCS=4
cat ../data/MPI/block_size_analysis.csv
```

## 9. Process Scaling
Study strong scaling (fixed N, vary processes):
```bash
make procscale
cat ../data/MPI/proc_scaling.csv
```
Speedup S(p) = T(1)/T(p), Efficiency E(p) = S(p)/p.
Expect diminishing returns as communication starts dominating beyond moderate P.

## 10. Verification
Expected checksums (shared with Serial & OpenMP):
```
N=256:  4199507.547136
N=512:  33526118.721088
N=1024: 268310302.375923
```
Run:
```bash
make verify PROCS=4
```
All lines must match exactly.

## 11. Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| Mismatch checksum | Wrong seeds or altered code | Rebuild; ensure LCG intact |
| Slow performance | Small block size or no -O3 | Recompile with flags; tune bs |
| mpiexec not found | MPI not installed | Install OpenMPI / MPICH / MS-MPI |
| Hang at broadcast | Mismatched collective calls | Ensure all ranks call MPI_Bcast |

## 12. Extensibility Ideas
- 2D block-cyclic distribution for better scalability.
- Overlap communication & compute using non-blocking collectives.
- Use MPI_Allreduce for distributed verification only.
- Hybrid MPI + OpenMP inside each rank.

## 13. Code Structure
```
MPI/
├── blocked_gemm_mpi.c   # Implementation
├── Makefile             # Build & analysis targets
└── README.md            # This documentation
```
Data artifacts:
```
data/MPI/
├── expected_checksums.txt
├── block_size_analysis.csv
├── proc_scaling.csv
└── test_config.txt
```

## 14. Academic Integrity
Implementation authored originally for this assignment. AI assistance cited separately (see AI_CITATION.md). No third-party GEMM code copied.

## 15. References (Conceptual Only)
- MPI Standard (message passing basics)
- Cache-oblivious & blocked algorithms (matrix multiplication literature)
- Strong vs. weak scaling principles

---
