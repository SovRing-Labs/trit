# TECHNICAL DISCLOSURE — trit: integer-only packed ternary weights, bit-identical across SIMD back-ends

Author: Juan Carlos De Santiago, with the Sovereign Raptor fleet (AI-assisted). Date of disclosure: 2026-10-16.
Published as defensive prior art: the author claims no patent on what is described here and asks that none be granted on it.
Licence of this text: CC BY 4.0. Code it describes: Apache-2.0 (this repository).

## 1. Problem
Ternary neural-network weights (−1, 0, +1) are usually stored at 2 bits and multiplied in floating point or through per-platform kernels
whose results differ in the last bits, so a model's output depends on the machine. We want one weight format, three storage densities,
and kernels on every instruction set that return the same integers.

## 2. Formats
- **P2, run time.** Each trit t is stored as code = t + 1 in a 2-bit field, 4 per byte; a row of n trits is padded to a multiple of 32
  (row bytes = ⌈n / 32⌉ · 8), padding = code 1 (t = 0), so a padded row needs no tail special case. Decode of trit k:
  `((p[k >> 2] >> (2 · (k & 3))) & 3) − 1`.
- **P3, storage / network.** 5 trits per byte in base 3, byte = d₁ + 3d₂ + 9d₃ + 27d₄ + 81d₅ with dᵢ = tᵢ + 1 (3⁵ = 243 ≤ 256): 1.6 bits
  per weight. Decode by a static 243 × 5 table.
- **L5, compute on storage.** A P3 byte is used directly as an index: for each group of 5 inputs, a 243-entry table holds the partial sum
  Σₖ xₖ · tₖ(c) for every code c, built once per query; a row's dot product is the sum of table reads, one per byte. Sixteen queries share
  one pass (int16 lanes, widened to int32 at block boundaries so no partial sum overflows).

## 3. Kernel: one algorithm, every back-end exact
The dot product of an int8 activation vector with a P2 row uses only integer operations: decode to int8 trits, multiply by sign
(AVX2 `sign_epi8`: x · t for t ∈ {−1, 0, 1} without a multiplier), widen to int16 and int32 and add. Because no floating point is
involved and every partial sum fits its lane, all back-ends compute the same integers:
- scalar C (the arbiter),
- AVX2: 8 packed bytes → 32 trits by a byte spread (`pshufb`), AND with a per-lane mask, compare-equal, subtract;
- AVX-512: the same with `vpermb` and 64 trits;
- NEON: `TBL` byte spread, AND, CEQ, SUB, 32 trits.
Selection is by run-time CPUID, once, cached as a function pointer; the x86 kernels carry per-function target attributes, so one binary
holds every kernel and falls back on an older CPU. Correctness is defined against the scalar arbiter on (a) law cases that cover every
code, padding and tail length and (b) random rows; a back-end ships only with 0 mismatches.

## 4. The library as law records
Each format and kernel is a record: a statement, its assumption, and its form per language (C++ and Rust source). The C++ header and the
Rust crate are written from the records, so a new back-end or language is a new form on the same law, checked by the same harness.

## 5. Measured (one Intel i5-8365U laptop, 2026-10-16; `results/`)
P2 dot on 8,192 × 4,096: 0 mismatches vs an int8 reference, P3 round trip exact, L5 0 mismatches; scalar and AVX2 0 mismatches over
348 law cases and 2,000,000 random rows, AVX2 7.3× scalar; P2 memory one quarter of int8.

## 6. What is not claimed
No speed claim beyond the one machine named; AVX-512 and NEON correctness are by construction and not yet measured on hardware;
WGSL is described in the header but not shipped; the emitter copies the recorded forms rather than generating them (see `STATUS.md`).
