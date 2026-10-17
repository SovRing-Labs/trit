# trit changelog

## 0.1.0 (2026-10-17)
First public release (pre-alpha). P2 / P3 / L5 packed ternary weights in C++ (`cpp/trit.hpp`, `cpp/trit_kernels.hpp`) and Rust
(`rust/`), law records (`laws/trit.json`), harnesses (`tests/`) and their 2026-10-16 logs (`results/`), suite (`suite.json`, `install.sh`).
Fix in this release: the Rust AVX2 build now compiles and its exactness test passes (it never compiled before). Known gaps: `STATUS.md`.
