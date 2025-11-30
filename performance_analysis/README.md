# Performance Analysis Toolkit

Performance benchmarking and comparative evaluation for the blocked GEMM implementations (Serial, OpenMP, MPI, CUDA). Produces graphs, summary CSVs, and a textual analysis report used in the Part C documentation.

## 1. Overview

This module aggregates raw timing data from the implementation directories and generates:
- Per‑technology scalability/optimization graphs (OpenMP threads, MPI processes, CUDA block sizes)
- Cross‑technology comparative graph (time and/or GFLOPs)
- Summary CSV files with computed metrics (speedup, efficiency, GFLOPs)
- Textual analysis report consolidating major findings

Deterministic initialization (shared seeds) ensures checksums are consistent across all implementations.

## 2. Contents

| File | Purpose |
|------|---------|
| `generate_graphs.py` | Main Python driver (graph + report generation) |
| `requirements.txt` | Python dependency versions |
| `graphs/` | Generated PNG/PDF graphs + CSV summaries + `analysis_report.txt` |

## 3. Prerequisites

Python 3.11+ (tested with 3.11/3.12). Install dependencies:
```bash
pip install -r requirements.txt
```

Dependencies (from `requirements.txt`): `pandas`, `numpy`, `matplotlib`.

## 4. Data Sources

Raw performance data pulled from:
- `data/Serial/baseline_timings.txt` (serial baseline times)
- `data/OpenMP/thread_scalability.csv` (thread scaling)
- `data/MPI/proc_scaling.csv` (process scaling)
- `data/CUDA/block_size_analysis.csv` (block size sweep)

If any file is missing re-run the sweeps in the corresponding implementation directory (see Troubleshooting).

## 5. Generating Artifacts

Run from repository root or inside this directory:
```bash
python performance_analysis/generate_graphs.py
```

Outputs (written to `performance_analysis/graphs/`):
- `openmp_evaluation.(png|pdf)` and `openmp_summary.csv`
- `mpi_evaluation.(png|pdf)` and `mpi_summary.csv`
- `cuda_evaluation.(png|pdf)` and `cuda_summary.csv`
- `comparative_analysis.(png|pdf)` and `comparative_summary.csv`
- `analysis_report.txt` (narrative summary)

## 6. Metric Definitions

| Metric | Formula |
|--------|---------|
| Speedup | `T_serial / T_parallel` |
| Efficiency (OpenMP/MPI) | `Speedup / P` |
| GFLOPs | `(2 * N^3) / (time * 1e9)` |

Serial baseline time comes from the matching matrix size entry in `baseline_timings.txt`.

## 7. Performance Highlights (Representative)

| Technology | Configuration (N=2048) | Time (s) | Speedup | Efficiency |
|------------|------------------------|----------|---------|-----------|
| Serial | Baseline | 13.57 | 1.00× | 100% |
| OpenMP | 16 threads, bs=64 | 0.68 | 19.94× | 124.6% |
| MPI | 8 processes, bs=128 | 4.58 | 2.96× | 37.0% |

CUDA (N=1024) block size sweep (GTX 1650):
| Block | Time (s) | GFLOPs |
|-------|---------|--------|
| 16 | 0.0295 | 73.15 |
| 24 | 0.0311 | 69.26 |
| 32 | 0.0335 | 64.34 |

## 8. Example Workflow

```bash
# 1. Ensure raw data exists (rerun sweeps if needed)
cd OpenMP && make sweep-threads && cd ..
cd MPI && make procscale && cd ..
cd CUDA && make sweep && cd ..

# 2. Install Python dependencies
pip install -r performance_analysis/requirements.txt

# 3. Generate all graphs and report
python performance_analysis/generate_graphs.py

# 4. View outputs
ls performance_analysis/graphs
```

## 9. Screenshots

Assignment requires representative screenshots of each graph. See `SCREENSHOT_GUIDE.md` for capture conventions and naming. Categories:
- OpenMP thread scalability (multiple thread counts)
- MPI process scalability (1,2,4,8 ranks)
- CUDA block size comparison
- Comparative overview

## 10. Troubleshooting

| Symptom | Cause | Resolution |
|---------|-------|------------|
| Missing CSV file | Sweep not executed | Rerun `make sweep-*` or PowerShell build script in implementation dir |
| Empty graph | Input file headers mismatched | Check original CSV formats; regenerate |
| Python import error | Dependency not installed | `pip install -r requirements.txt` |
| Inconsistent speedup | Serial baseline mismatch | Confirm baseline file has matching N and block size |
| Checksum mismatch in source runs | Modified initialization | Restore original LCG seeding |

## 11. Extensibility

Potential improvements:
- Add roofline plot (requires manual peak bandwidth/FLOPs input)
- Include confidence intervals via multiple timed repetitions
- Auto-tune optimal block size / thread count heuristics
- Export graphs in SVG for web embedding

## 12. File Structure

```
performance_analysis/
├── README.md
├── generate_graphs.py
├── requirements.txt
└── graphs/
    ├── openmp_evaluation.(png|pdf)
    ├── openmp_summary.csv
    ├── mpi_evaluation.(png|pdf)
    ├── mpi_summary.csv
    ├── cuda_evaluation.(png|pdf)
    ├── cuda_summary.csv
    ├── comparative_analysis.(png|pdf)
    ├── comparative_summary.csv
    └── analysis_report.txt
```

## 13. Academic Integrity

The analysis code and methodology were authored for this assignment. AI assistance is documented separately in `AI_CITATION.md`. No external benchmarking frameworks or third‑party GEMM kernels are embedded.

## 14. References (Conceptual)

- OpenMP Specification 5.2 (thread scalability concepts)
- MPI Standard (collective communication impacts)
- CUDA Programming Guide (shared memory & occupancy)
- Roofline Model (memory vs compute bound reasoning)

---

