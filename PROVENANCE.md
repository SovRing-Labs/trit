# PROVENANCE — trit 0.1.0

- **Origin.** Written in the SovRing-Labs development tree from 2026-10-07 to 2026-10-16: the P2 / P3 / L5 code was first written as the
  weight layer of our ternary inference daemon (the `trit_pack.hpp` header published inside SovRing-Labs/omnid) and our ternary model
  runtime, then consolidated here as one library with a C++ header, a Rust crate and the law records.
- **Sources of ideas (no code taken).** Ternary / 1.58-bit weights: Ma et al., "The Era of 1-bit LLMs" (BitNet b1.58, 2024). Base-3
  packing of 5 trits per byte is the standard 3⁵ ≤ 2⁸ observation. Table-lookup inner products over packed codes follow the general
  lookup-table GEMM idea (e.g. "T-MAC", Wei et al., 2024); our L5 tables, sizes and batching are our own. SIMD decode uses the documented
  Intel / Arm intrinsics only.
- **Third-party code.** None. No file is copied from another project; ScanCode on the release tree finds only Apache-2.0 / CC BY 4.0 text
  (census record kept by the publisher).
- **AI assistance.** Code and text were written with AI coding assistants (Claude) directed and reviewed by the author; every number in
  this repository comes from a harness in `tests/` with its log in `results/`.
- **Export.** This repository is a fresh single-history export of the development tree (history squashed because the development
  history carries private paths), cut by a manifest-driven release tool that rejects private paths, secrets and databases.

Recovery 2026-10-17: the exported Rust lib.rs preserves the original AVX2 compilation repair: intrinsics imported within p2 and explicit unsafe blocks for intrinsic calls. The development source is unchanged by this export.
