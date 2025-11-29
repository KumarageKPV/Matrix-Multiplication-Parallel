# AI Assistance Citation (OpenMP Implementation)

This OpenMP implementation was developed with assistance from GitHub Copilot (using GPT-5). AI assistance was used for code authoring, debugging, portability fixes, and build/run guidance. The student reviewed, modified, and validated all code and instructions.

## Scope of AI Contributions
- Implemented blocked GEMM using OpenMP with `#pragma omp parallel for` and `collapse(2)` over tile loops.
- Added portable high-resolution timing (`QueryPerformanceCounter` on Windows, `clock_gettime` elsewhere).
- Implemented 64-byte aligned allocation with `_aligned_malloc`/`_aligned_free` (Windows) and `posix_memalign` (POSIX).
- Aligned initializer and checksum with serial baseline for correctness parity.
- Provided Windows-focused build/run instructions for MSYS2/MinGW (GCC) and MSVC.
- Suggested timing sweeps and speedup calculation approach.

## Prompts Used (abridged)
- "Now based on the above info, implement Phase 2: Part A: for OpenMP"
- "run a quick local build in my terminal"
- "try using the bash terminal"
- "Read the current MINGW64 terminal output and see whether it is correct"
- "MinGW64 Bash"
- "use of AI is permitted (but must be cited, with prompts). Do this in the OpenMP folder for the current implementations"

## Files Influenced
- `OpenMP/blocked_gemm_omp.c`
- `OpenMP/Makefile`
- `README.md` (OpenMP build/run section)

## Student Verification
- The student compiled and ran the OpenMP binary under MSYS2 MinGW64 Bash.
- Checksums matched the serial baseline for `N=512, block=64`.
- Timing results were collected for multiple thread counts and block sizes.

## Notes
- All code has been manually reviewed and adapted to the specific assignment requirements.
- No external copyrighted code was copied; logic is original and tailored to this project.
