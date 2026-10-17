# trit — packed ternary weights in C++ and Rust (P2 · P3 · L5)

[![status: pre-alpha](https://img.shields.io/badge/status-pre--alpha%20(under%20development)-orange)](STATUS.md)
[![defensive publication](https://img.shields.io/badge/defensive%20publication-TECHNICAL--DISCLOSURE.md-blue)](TECHNICAL-DISCLOSURE.md)

Status: **under development — pre-alpha** (`STATUS.md`). Defensive publication: `TECHNICAL-DISCLOSURE.md` (dated 2026-10-16; the DOI is
minted by Zenodo on the first GitHub release).

A ternary weight is −1, 0 or +1. This library stores such weights compactly and multiplies them against int8 activations with integer
arithmetic only, so every back-end (portable C, AVX2, AVX-512, NEON) gives the same integers, bit for bit. It is the weight format our
ternary models and the omnid inference daemon use.

Three formats:
- **P2 (run time):** 4 trits per byte, code = trit + 1 in 2 bits. One AVX2 register decodes 32 trits (shuffle, mask, compare, subtract).
- **P3 (storage / network):** 5 trits per byte in base 3 (3⁵ = 243 ≤ 256), decoded by a 243-entry table: 1.6 bits per weight.
- **L5 (compute on storage):** each P3 byte is used directly as an index into a 243-entry table of partial sums, for 16 queries at once.

## Run it (Linux or macOS, a C++17 compiler; Rust optional)
```bash
g++ -O3 -march=native -fopenmp -std=c++17 -Icpp tests/trit_test.cpp -o trit_test && ./trit_test                      # P2/P3/L5 exactness + speed
g++ -O3 -march=native -fopenmp -std=c++17 -Icpp tests/trit_kernel_test.cpp -o trit_kernel_test && ./trit_kernel_test  # every back-end vs scalar
cd rust && cargo test --release && RUSTFLAGS="-C target-cpu=native" cargo test --release                               # Rust: portable, then AVX2
```
Or run the whole suite with its explanations: `bash install.sh` (opens a local page; `--no-gui` for the terminal only).

## What it does today (one laptop, Intel i5-8365U, AVX2 only; files in `results/`)
| Test | Result |
|---|---|
| P2 pack → dot vs int8 reference, 8,192 × 4,096 | 0 mismatches; P3 round trip exact; L5 0 mismatches (`results/2026-10-16-trit_test.log`) |
| Back-ends vs the scalar arbiter | scalar and AVX2: 0 mismatches over 348 law cases + 2,000,000 random rows; AVX2 7.3× scalar (`results/2026-10-16-trit_kernel_test.log`) |
| Memory | P2 = ¼ of int8 (8.4 MB vs 33.6 MB for 8,192 × 4,096); P3 = 6.7 MB |
| Rust crate | portable build 2/2 tests; AVX2 build 3/3 (`results/2026-10-16-cargo-test.log`) |

Not shown here: AVX-512 (compiled, not runnable on this CPU), NEON (needs an ARM machine), WGSL (the header names a WebGPU runner that is
not in this repository). See `STATUS.md`.

## Files
| Path | What |
|---|---|
| `cpp/trit.hpp` | header-only P2 / P3 / L5 library |
| `cpp/trit_kernels.hpp` | per-back-end P2 kernels (scalar, AVX2, AVX-512, NEON) + run-time dispatch by CPUID |
| `cpp/p3_scores.c` | P3 row scoring against an int32 partial-sum table (OpenMP) |
| `rust/` | the `trit` crate (no_std core, AVX2 kernels behind `target_feature`) |
| `laws/trit.json` | the library as 11 law records (statement, assumption, C and Rust forms) |
| `emit/trit-emit.py` | writes `cpp/` and `rust/` from `laws/trit.json` (today it copies the recorded forms; see `STATUS.md`) |
| `tests/` | the two C++ harnesses above |

## Licence
Code: **Apache-2.0** (`LICENSE`). Documents (`README.md`, `STATUS.md`, `TECHNICAL-DISCLOSURE.md`, `PROVENANCE.md`, the statements in
`laws/`): **CC BY 4.0** (`LICENSE-CC-BY-4.0.md`). Free for the community, like our research corpus. Details: `LICENSING.md`.

Part of [SovRing-Labs](https://github.com/SovRing-Labs/SovRing). Cite: `CITATION.cff`.
