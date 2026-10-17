//! trit — Packed ternary weights library (P2 runtime, P3 storage, L5 compute)
//! Consolidated from omnid's trit_pack.hpp and our ternary model runtime (p2.rs)
//!
//! # Formats
//!
//! - **P2** (runtime): 4 trits/byte, code = trit+1 in 2 bits. AVX2 unpack in registers.
//! - **P3** (storage): 5 trits/byte, base-3 (3^5=243≤256). Decoded via 243-entry table.
//! - **L5**: Compute directly on P3 codes as table indices (16 queries at once).
//!
//! All implementations are exact (verified against int8 reference).

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;


/// P2: 4 trits per byte, code = t+1 in 2 bits
pub mod p2 {
    // the intrinsics must be in scope inside the module (a crate-level `use` does not reach it): without this line no AVX2
    // build compiled (found 2026-10-16 with RUSTFLAGS="-C target-cpu=native": 93 E0425 errors)
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    use core::arch::x86_64::*;

        /// Bytes needed for one row of n trits (padded to multiple of 32)
    #[inline]
    pub fn row_bytes(n: usize) -> usize {
        n.div_ceil(32) * 8
    }

    /// Pack trits (-1, 0, +1) into P2 format
    pub fn pack(t: &[i8], rows: usize, n: usize) -> Vec<u8> {
        let rb = row_bytes(n);
        let mut v = vec![0x55u8; rb * rows]; // 0x55 = four zero trits (code 1)
        for r in 0..rows {
            for i in 0..n {
                let sh = 2 * (i & 3);
                let b = &mut v[r * rb + (i >> 2)];
                *b = (*b & !(3 << sh)) | (((t[r * n + i] + 1) as u8) << sh);
            }
        }
        v
    }

    /// Pack a single row
    #[inline]
    pub fn pack_row(t: &[i8], n: usize, out: &mut [u8]) {
        let rb = row_bytes(n);
        out[..rb].fill(0x55);
        for i in 0..n {
            let sh = 2 * (i & 3);
            out[i >> 2] = (out[i >> 2] & !(3 << sh)) | (((t[i] + 1) as u8) << sh);
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    /// Unpack 8 packed bytes (32 trits) into AVX2 register
    #[inline(always)]
    pub unsafe fn unpack32(p: *const u8) -> __m256i {
        let src = _mm256_set1_epi64x((p as *const i64).read_unaligned());
        let ctl = _mm256_setr_epi8(
            0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3,
            4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7
        );
        let msk = _mm256_set1_epi32(0xC0300C03u32 as i32);
        let two = _mm256_set1_epi32(0x80200802u32 as i32);
        let m = _mm256_and_si256(_mm256_shuffle_epi8(src, ctl), msk);
        _mm256_sub_epi8(
            _mm256_cmpeq_epi8(m, _mm256_setzero_si256()),
            _mm256_cmpeq_epi8(m, two)
        )
    }

    /// Scalar P2 dot product (for verification / odd lengths)
    #[inline]
    pub fn dot_scalar(x: &[i8], p: &[u8], n: usize) -> i32 {
        let mut acc = 0i32;
        for k in 0..n {
            let sh = 2 * (k & 3);
            let b = k >> 2;
            acc += x[k] as i32 * (((p[b] >> sh) & 3) as i32 - 1);
        }
        acc
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    /// AVX2 P2 dot product (main kernel)
    #[inline]
    pub unsafe fn dot(x: &[i8], p: &[u8], n: usize) -> i32 {
        let mut acc = _mm256_setzero_si256();
        let ones = _mm256_set1_epi16(1);
        let mut k = 0;
        while k + 32 <= n {
            let xv = _mm256_loadu_si256(x[k..].as_ptr() as *const __m256i);
            let w = unpack32(p[k / 4..].as_ptr());
            let v = _mm256_sign_epi8(xv, w);
            let lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
            let hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
            acc = _mm256_add_epi32(acc, _mm256_add_epi32(
                _mm256_madd_epi16(lo, ones), _mm256_madd_epi16(hi, ones)
            ));
            k += 32;
        }
        let mut s = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256(acc, 1)
        );
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        let mut r = _mm_cvtsi128_si32(s);
        while k < n {
            let sh = 2 * (k & 3);
            let b = k >> 2;
            r += x[k] as i32 * (((p[b] >> sh) & 3) as i32 - 1);
            k += 1;
        }
        r
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    /// P2 dot-4: 4 rows at once
    #[inline]
    pub unsafe fn dot4(
        x: &[i8],
        p0: &[u8], p1: &[u8], p2: &[u8], p3: &[u8],
        n: usize,
    ) -> (i32, i32, i32, i32) {
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut a2 = _mm256_setzero_si256();
        let mut a3 = _mm256_setzero_si256();
        let ones = _mm256_set1_epi16(1);
        let mut k = 0;
        while k + 32 <= n {
            let xv = _mm256_loadu_si256(x[k..].as_ptr() as *const __m256i);
            // a closure body is not an unsafe context of the enclosing unsafe fn: the intrinsics need their own block
            let acc = |a: &mut __m256i, p: &[u8]| unsafe {
                let w = unpack32(p[k / 4..].as_ptr());
                let v = _mm256_sign_epi8(xv, w);
                let lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
                let hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
                *a = _mm256_add_epi32(*a, _mm256_add_epi32(
                    _mm256_madd_epi16(lo, ones), _mm256_madd_epi16(hi, ones)
                ));
            };
            acc(&mut a0, p0);
            acc(&mut a1, p1);
            acc(&mut a2, p2);
            acc(&mut a3, p3);
            k += 32;
        }
        let h = |a: __m256i| {
            let mut s = _mm_add_epi32(
                _mm256_castsi256_si128(a),
                _mm256_extracti128_si256(a, 1)
            );
            s = _mm_hadd_epi32(s, s);
            s = _mm_hadd_epi32(s, s);
            _mm_cvtsi128_si32(s)
        };
        let mut r0 = h(a0);
        let mut r1 = h(a1);
        let mut r2 = h(a2);
        let mut r3 = h(a3);
        while k < n {
            let sh = 2 * (k & 3);
            let b = k >> 2;
            let xv = x[k] as i32;
            r0 += xv * (((p0[b] >> sh) & 3) as i32 - 1);
            r1 += xv * (((p1[b] >> sh) & 3) as i32 - 1);
            r2 += xv * (((p2[b] >> sh) & 3) as i32 - 1);
            r3 += xv * (((p3[b] >> sh) & 3) as i32 - 1);
            k += 1;
        }
        (r0, r1, r2, r3)
    }
    /// out[r] = Σ_i x[i]·t[r,i]  (batch 1; n a multiple of 32). 4 rows per pass share the
    /// activation load (the same AVX2 decode as p2_dot4).
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline]
    pub unsafe fn gemv_b1(p: &[u8], rows: usize, n: usize, x: &[i8], out: &mut [i32]) {
        let rb = row_bytes(n);
        let ones = _mm256_set1_epi16(1);
        let mut r = 0;
        while r + 4 <= rows {
            let (mut a0, mut a1, mut a2, mut a3) = (
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
            );
            let base = p.as_ptr().add(r * rb);
            for k in (0..n).step_by(32) {
                let xv = _mm256_loadu_si256(x.as_ptr().add(k) as *const __m256i);
                let b = k / 4;
                a0 = acc_i8(a0, _mm256_sign_epi8(xv, unpack32(base.add(b))), ones);
                a1 = acc_i8(a1, _mm256_sign_epi8(xv, unpack32(base.add(rb + b))), ones);
                a2 = acc_i8(a2, _mm256_sign_epi8(xv, unpack32(base.add(2 * rb + b))), ones);
                a3 = acc_i8(a3, _mm256_sign_epi8(xv, unpack32(base.add(3 * rb + b))), ones);
            }
            out[r] = hsum(a0);
            out[r + 1] = hsum(a1);
            out[r + 2] = hsum(a2);
            out[r + 3] = hsum(a3);
            r += 4;
        }
        for r in r..rows {
            let mut a = _mm256_setzero_si256();
            let base = p.as_ptr().add(r * rb);
            for k in (0..n).step_by(32) {
                a = acc_i8(
                    a,
                    _mm256_sign_epi8(
                        _mm256_loadu_si256(x.as_ptr().add(k) as *const __m256i),
                        unpack32(base.add(k / 4)),
                    ),
                    ones,
                );
            }
            out[r] = hsum(a);
        }
    }

    /// Batch 8: x is 8 query vectors (xs[q*n + i]); out[r*8 + q]. Each 32-trit block is
    /// unpacked ONCE and applied to all 8 queries (the strongest P2 form at batch 8:
    /// weight bytes and unpack amortised 8×).
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline]
    pub unsafe fn gemv_b8(p: &[u8], rows: usize, n: usize, xs: &[i8], out: &mut [i32]) {
        let rb = row_bytes(n);
        let ones = _mm256_set1_epi16(1);
        for r in 0..rows {
            let mut a = [_mm256_setzero_si256(); 8];
            let base = p.as_ptr().add(r * rb);
            for k in (0..n).step_by(32) {
                let t = unpack32(base.add(k / 4));
                for q in 0..8 {
                    a[q] = acc_i8(
                        a[q],
                        _mm256_sign_epi8(
                            _mm256_loadu_si256(xs.as_ptr().add(q * n + k) as *const __m256i),
                            t,
                        ),
                        ones,
                    );
                }
            }
            for q in 0..8 {
                out[r * 8 + q] = hsum(a[q]);
            }
        }
    }

    /// Scalar reference (the arbiter for every exactness check).
    pub fn reference(t: &[i8], rows: usize, n: usize, x: &[i8], out: &mut [i32]) {
        for r in 0..rows {
            out[r] = (0..n).map(|i| x[i] as i32 * t[r * n + i] as i32).sum();
        }
    }

    /// Inner helpers used by the batched GEMV kernels (not exposed on the public API).
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    fn acc_i8(a: __m256i, v: __m256i, ones: __m256i) -> __m256i {
        // SAFETY: compiled only with target_feature = "avx2" (cfg above)
        unsafe {
            let lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
            let hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
            _mm256_add_epi32(a, _mm256_add_epi32(_mm256_madd_epi16(lo, ones), _mm256_madd_epi16(hi, ones)))
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    fn hsum(a: __m256i) -> i32 {
        // SAFETY: compiled only with target_feature = "avx2" (cfg above)
        unsafe {
            let mut s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
            s = _mm_hadd_epi32(s, s);
            s = _mm_hadd_epi32(s, s);
            _mm_cvtsi128_si32(s)
        }
    }
}

/// P3: 5 trits per byte, base-3 (storage / network format)
pub mod p3 {
    /// Bytes needed for n trits
    #[inline]
    pub fn bytes(n: usize) -> usize {
        (n + 4) / 5
    }

    /// Pack trits (-1, 0, +1) into P3 base-3 format
    pub fn pack(t: &[i8], n: usize) -> Vec<u8> {
        let mut v = vec![0u8; bytes(n)];
        for b in 0..v.len() {
            let mut code = 0;
            for j in (0..5).rev() {
                let i = b * 5 + j;
                code = code * 3 + (if i < n { t[i] + 1 } else { 1 });
            }
            v[b] = code as u8;
        }
        v
    }

    /// 243-entry decode table (5 trits per code)
    struct P3Table {
        t: [[i8; 5]; 243],
    }

    impl P3Table {
        const fn new() -> Self {
            let mut table = [[0i8; 5]; 243];
            let mut c = 0;
            while c < 243 {
                let mut x = c;
                let mut j = 0;
                while j < 5 {
                    table[c][j] = ((x % 3) as i8) - 1;
                    x /= 3;
                    j += 1;
                }
                c += 1;
            }
            P3Table { t: table }
        }
    }

    /// Unpack P3 bytes back to trits
    pub fn unpack(v: &[u8], n: usize, out: &mut [i8]) {
        static TABLE: P3Table = P3Table::new();
        let full = n / 5;
        for b in 0..full {
            out[b * 5..b * 5 + 5].copy_from_slice(&TABLE.t[v[b] as usize]);
        }
        for i in full * 5..n {
            out[i] = TABLE.t[v[full] as usize][i - full * 5];
        }
    }
}

/// L5: Compute on P3 storage form (P3 code as table index)
/// 16 queries at once, int16 accumulators widened to int32 per chunk
pub mod l5 {
    /// Chunk size for int16 widening (must keep sums < 32767)
    pub const KCH: usize = 64;

    /// Build 16x243xKCH tables of partial sums from input x
    /// x: input vector (queries -8..7 for testing, or actual values)
    pub fn build_tables(x: &[i8], n: usize) -> [[[i8; KCH]; 243]; 16] {
        let mut tables = [[[0i8; KCH]; 243]; 16];
        for k in 0..n {
            let code = (x[k] + 1) as usize; // 0..2
            for q in 0..16 {
                let xq = q as i8 - 8; // queries -8..7
                tables[q][code][k % KCH] = tables[q][code][k % KCH].saturating_add(xq);
            }
        }
        tables
    }

    /// Scalar reference (exact arbiter)
    pub fn reference(t: &[i8], rows: usize, n: usize, x: &[i8], out: &mut [i32]) {
        for r in 0..rows {
            let mut sum = 0i32;
            for i in 0..n {
                sum += x[i] as i32 * t[r * n + i] as i32;
            }
            out[r] = sum;
        }
    }
}

// Re-export for convenience
pub use p2::{pack as p2_pack, row_bytes as p2_row_bytes, pack_row as p2_pack_row, dot_scalar};
pub use p2::reference;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub use p2::{dot as p2_dot, dot4 as p2_dot4, gemv_b1, gemv_b8};
pub use p3::{pack as p3_pack, bytes as p3_bytes, unpack as p3_unpack};
pub use l5::{build_tables as l5_build_tables, reference as l5_reference, KCH as L5_KCH};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p3_roundtrip() {
        let t = [-1i8, 0, 1, -1, 0, 1, -1, 0, 1, -1];
        let packed = p3::pack(&t, t.len());
        let mut back = vec![0i8; t.len()];
        p3::unpack(&packed, t.len(), &mut back);
        assert_eq!(t.to_vec(), back);
    }

    #[test]
    fn p2_pack_unpack_scalar() {
        let t = [-1i8, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1];
        let packed = p2::pack(&t, 1, t.len());
        let result = p2::dot_scalar(&t, &packed, t.len());
        let expected: i32 = t.iter().map(|&x| x as i32 * x as i32).sum();
        assert_eq!(result, expected);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    fn p2_avx2_exact() {
        use std::arch::is_x86_feature_detected;
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let t = [-1i8, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1];
        let packed = p2::pack(&t, 1, t.len());
        let result = unsafe { p2::dot(&t, &packed, t.len()) };
        let expected: i32 = t.iter().map(|&x| x as i32 * x as i32).sum();
        assert_eq!(result, expected);
    }
}