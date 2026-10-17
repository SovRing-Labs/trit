# AGENTS.md — trit

For coding assistants (and people) setting this repository up. Plain steps; every command runs from the repo root.

**Status: pre-alpha, under development.** Interfaces, numbers and file layout can change without notice. Read `STATUS.md` first: what works today (each line names its test and evidence file), what is not yet tested, the known failures and the corrections.

## What it is
Stage: about 30 % (under development, pre-alpha).
Packed ternary weights (P2 4 trits/byte, P3 1.6 bits/weight, L5 compute on storage) with integer-only kernels that are bit-identical on scalar, AVX2, AVX-512 and NEON, in C++ and Rust.

## Install and check (3 commands)
```bash
git clone --depth 1 https://github.com/SovRing-Labs/trit && cd trit
# nothing to install beyond: g++ or clang++ (C++17, OpenMP), cargo (optional, for the Rust crate)
g++ -O3 -march=native -fopenmp -std=c++17 -Icpp tests/trit_kernel_test.cpp -o /tmp/trit_kernel_test && /tmp/trit_kernel_test | tail -1   # expect: "pass": true
```
Needs: g++ or clang++ (C++17, OpenMP), cargo (optional, for the Rust crate).
Whole suite instead: `bash install.sh --no-gui` (python check → venv → every runnable `suite.json` test, headless; without `--no-gui` it then opens the local results page on 127.0.0.1).

GUI: `bash install.sh   # suite page on http://127.0.0.1 (localhost only)`

## What the tests mean
`suite.json` lists every claim this repository makes with its verify command, the expected output, what passing means, how the number was obtained and how to use it in your own work (5 tests, in `suite.json` at the repository root). `STATUS.md` says what works today, what is not yet tested and the known failures; the umbrella's `STATUS.md` has one row per repository.

## Where the evidence is
- `README.md`
- `TECHNICAL-DISCLOSURE.md` — the method, as dated prior art
- `CITATION.cff`

## Honest limits (from STATUS.md)
The Rust AVX2 path did not compile before 2026-10-16 (93 missing-name and 12 unsafe-call errors with `-C target-cpu=native`), so `p2_avx2_exact` had never run; the portable build compiles it out. Fixed in this release (intrinsics imported inside `mod p2`, unsafe blocks in three helpers); T4 is the check. `emit/trit-emit.py` does not generate code from `laws/trit.json` yet: it copies the hand-written `cpp/trit.hpp` and `rust/src/lib.rs` and checks they are byte-identical to themselves, so its "verify" proves determinism of a copy, not spec → code. Generation from the law records is the next …

## Licence
Apache-2.0 (code) + CC BY 4.0 (documents).

## Guided setup through MCP
The SovRing umbrella ships an MCP server that explains, checks readiness, clones and runs these tests: https://github.com/SovRing-Labs/SovRing (README → "Paste this to your assistant"). Zero-install docs endpoint: https://gitmcp.io/SovRing-Labs/trit
