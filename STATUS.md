> **Status: under development — pre-alpha.** Interfaces, numbers and file layout can change without notice.

# STATUS — trit 0.1.0 (2026-10-16)
Stage: about 30 % (J, 2026-10-16: the cores and libraries outside doctor / math / embed are at about 30 %).

## Works today (each line = a suite test that passed on 2026-10-16, with its evidence)
- **T1** P2 / P3 / L5 exactness on 8,192 × 4,096: 0 mismatches, P3 round trip exact, L5 0 mismatches — `results/2026-10-16-trit_test.log`
- **T2** back-ends bit-identical to the scalar arbiter: scalar + AVX2, 348 law cases + 2,000,000 random rows, 0 mismatches — `results/2026-10-16-trit_kernel_test.log`
- **T3** Rust crate, portable build: 2 tests pass (`p3_roundtrip`, `p2_pack_unpack_scalar`) — `results/2026-10-16-cargo-test.log`
- **T4** Rust crate, AVX2 build (`RUSTFLAGS="-C target-cpu=native"`): 3 tests pass incl. `p2_avx2_exact` — `results/2026-10-16-cargo-test.log`
- **T5** memory: P2 is a quarter of int8, P3 1.6 bits per weight — `results/2026-10-16-trit_test.log` (`bytes_int8`, `bytes_p2`, `bytes_p3`)

## Not yet tested
- AVX-512 kernel: compiled into every x86 build, not runnable on the AVX2 laptop used here (`runnable: 0` in T2's log).
- NEON kernel: needs an AArch64 machine (`compiled: 0` on x86).
- WGSL / WebGPU kernel: `cpp/trit_kernels.hpp` names `wgsl/run_wgsl.py`; that runner is not in this repository.
- Speed on any CPU other than one Intel i5-8365U laptop under normal desktop load.

## Known failures and honest negatives
- The Rust AVX2 path did not compile before 2026-10-16 (93 missing-name and 12 unsafe-call errors with `-C target-cpu=native`), so
  `p2_avx2_exact` had never run; the portable build compiles it out. Fixed in this release (intrinsics imported inside `mod p2`, unsafe
  blocks in three helpers); T4 is the check.
- `emit/trit-emit.py` does not generate code from `laws/trit.json` yet: it copies the hand-written `cpp/trit.hpp` and `rust/src/lib.rs`
  and checks they are byte-identical to themselves, so its "verify" proves determinism of a copy, not spec → code. Generation from the
  law records is the next step.
- `trit-emit.py --emit` appends one header line to `rust/Cargo.toml` on every run (not idempotent).
- Balanced-ternary arithmetic (add / sub / mul / div / compare with exact carries) and three-valued logic (Kleene K3, Łukasiewicz L3) are
  planned and not in this library yet; today it is the packed-weights layer only.

## Corrections
None recorded against this repository (first release).

## How to re-check
`python3 sovring-suite.pyz --run trit` or `bash install.sh` (runs T1–T5 with their explanations), or the three commands in `README.md`.

## Defensive publication
`TECHNICAL-DISCLOSURE.md` (P2 / P3 / L5 formats, the integer-only multi-back-end kernel with a scalar arbiter, the law-record form),
dated 2026-10-16. DOI: minted by Zenodo on the first GitHub release; `CITATION.cff` carries it after that.
