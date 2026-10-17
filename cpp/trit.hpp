// trit_pack.hpp — packed ternary weights for omnid engines (omni_infer, omni_train, farm transfer). NeuralPC idea, adapted.
// Why: our engines stored one trit per BYTE (8 bits/weight). Big models on old machines are memory-bound (law L10: RAM
// 11–12 GB/s here), so bytes per weight IS the speed. Two formats:
//   P2  runtime: 4 trits/byte (code = t+1 in 2 bits). AVX2 unpack in registers (shuffle + mask + compare), then the same
//       sign_epi8 dot as before — exact, no multiplies, 4× fewer weight bytes from RAM.
//   P3  storage/network: 5 trits/byte base 3 (3^5 = 243 ≤ 256) = 1.6 bits/weight; decoded by a 243-entry table at load.
// Field: bitnet.cpp I2_S / TL1 / TL2 kernels, T-MAC (lookup-table GEMV). Ours: exact sign_epi8 path kept, both formats in one header.
#pragma once
#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace trit {

// ---- P2: 4 trits per byte, trit i of a row at byte i/4, bits 2·(i%4); rows padded to a multiple of 32 trits
inline size_t p2_row_bytes(int n) { return (size_t)((n + 31) / 32) * 8; }
inline void p2_pack_row(const int8_t *t, int n, uint8_t *out) {
    size_t nb = p2_row_bytes(n); memset(out, 0x55, nb);              // 0x55 = four zero trits (code 1)
    for (int i = 0; i < n; i++) { int sh = 2 * (i & 3); out[i >> 2] = (uint8_t)((out[i >> 2] & ~(3 << sh)) | ((t[i] + 1) << sh)); }
}
inline std::vector<uint8_t> p2_pack(const int8_t *t, int rows, int n) {
    size_t rb = p2_row_bytes(n); std::vector<uint8_t> v(rb * rows);
    for (int r = 0; r < rows; r++) p2_pack_row(t + (size_t)r * n, n, v.data() + rb * r); return v;
}
// 8 packed bytes → 32 int8 trits in one register
inline __m256i p2_unpack32(const uint8_t *p) {
    const __m256i src = _mm256_set1_epi64x(*(const int64_t *)p);
    const __m256i ctl = _mm256_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3, 4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7);
    const __m256i msk = _mm256_set1_epi32(0xC0300C03);               // per byte position k: 0x03, 0x0C, 0x30, 0xC0
    const __m256i two = _mm256_set1_epi32((int)0x80200802);          // code 2 (= +1) at each position
    __m256i m = _mm256_and_si256(_mm256_shuffle_epi8(src, ctl), msk);
    return _mm256_sub_epi8(_mm256_cmpeq_epi8(m, _mm256_setzero_si256()), _mm256_cmpeq_epi8(m, two));   // −1 / 0 / +1
}
// Σ x·t over n (x int8, padded with zeros to a multiple of 32 is NOT required: tail handled scalar from the packed row)
inline int32_t p2_dot(const int8_t *x, const uint8_t *p, int n) {
    __m256i acc = _mm256_setzero_si256(); const __m256i ones = _mm256_set1_epi16(1); int k = 0;
    for (; k + 32 <= n; k += 32) {
        __m256i v = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(x + k)), p2_unpack32(p + k / 4));
        __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v)), hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
        acc = _mm256_add_epi32(acc, _mm256_add_epi32(_mm256_madd_epi16(lo, ones), _mm256_madd_epi16(hi, ones)));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s); int32_t r = _mm_cvtsi128_si32(s);
    for (; k < n; k++) r += (int32_t)x[k] * (((p[k >> 2] >> (2 * (k & 3))) & 3) - 1);
    return r;
}

// 4 packed rows at once (Efficiency doorway 4): the int8 activation block is loaded ONCE per 32 columns and reused for 4 rows;
// trits are unpacked in registers only (never written back). Exact: same sums as 4 × p2_dot.
inline void p2_dot4(const int8_t *x, const uint8_t *p0, const uint8_t *p1, const uint8_t *p2r, const uint8_t *p3, int n,
                    int32_t &r0, int32_t &r1, int32_t &r2, int32_t &r3) {
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0; const __m256i ones = _mm256_set1_epi16(1); int k = 0;
    auto acc = [&](__m256i &a, __m256i v) {
        __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v)), hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
        a = _mm256_add_epi32(a, _mm256_add_epi32(_mm256_madd_epi16(lo, ones), _mm256_madd_epi16(hi, ones))); };
    for (; k + 32 <= n; k += 32) {
        const __m256i xv = _mm256_loadu_si256((const __m256i *)(x + k)); const int b = k / 4;
        acc(a0, _mm256_sign_epi8(xv, p2_unpack32(p0 + b))); acc(a1, _mm256_sign_epi8(xv, p2_unpack32(p1 + b)));
        acc(a2, _mm256_sign_epi8(xv, p2_unpack32(p2r + b))); acc(a3, _mm256_sign_epi8(xv, p2_unpack32(p3 + b)));
    }
    auto h = [](__m256i a) { __m128i s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
        s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s); return _mm_cvtsi128_si32(s); };
    r0 = h(a0); r1 = h(a1); r2 = h(a2); r3 = h(a3);
    for (; k < n; k++) { const int sh = 2 * (k & 3), b = k >> 2; const int32_t xv = x[k];
        r0 += xv * (((p0[b] >> sh) & 3) - 1); r1 += xv * (((p1[b] >> sh) & 3) - 1); r2 += xv * (((p2r[b] >> sh) & 3) - 1); r3 += xv * (((p3[b] >> sh) & 3) - 1); }
}

// ---- P3: 5 trits per byte, base 3 (storage / network)
inline size_t p3_bytes(size_t n) { return (n + 4) / 5; }
inline std::vector<uint8_t> p3_pack(const int8_t *t, size_t n) {
    std::vector<uint8_t> v(p3_bytes(n), 0);
    for (size_t b = 0; b < v.size(); b++) { int code = 0; for (int j = 4; j >= 0; j--) { size_t i = b * 5 + j; code = code * 3 + (i < n ? t[i] + 1 : 1); } v[b] = (uint8_t)code; }
    return v;
}
struct P3Table { int8_t t[243][5]; P3Table() { for (int c = 0; c < 243; c++) { int x = c; for (int j = 0; j < 5; j++) { t[c][j] = (int8_t)(x % 3 - 1); x /= 3; } } } };
inline void p3_unpack(const uint8_t *v, size_t n, int8_t *out) {
    static const P3Table T;
    size_t full = n / 5;
    for (size_t b = 0; b < full; b++) memcpy(out + b * 5, T.t[v[b]], 5);
    for (size_t i = full * 5; i < n; i++) out[i] = T.t[v[full]][i - full * 5];
}

// ---- L5: the P3 code (5 trits/byte, same digit order as p3_pack) used AS a table index — compute on the storage form.
// Per 5 inputs one 243-entry table of partial sums is built for 16 queries at once (int16 lanes, built incrementally:
// 242 adds), then every row costs ONE lookup-add per 5 weights. Exact: |table| ≤ 5·128 = 640; i16 sums are widened to
// int32 after each chunk of L5_KCH blocks (≤ 4·4·640 = 10,240 < 32,767). 2-D tiled: tables of L5_KCH blocks (124 KB, L2) ×
// row tiles whose accumulators stay in L1. Measured (omni-render-jit `m1`, results/m1-*.json, TERNARY-STACK-0 M1):
// 2–3× the P2 kernel at ≥ 16 queries, 0 mismatches. Field: T-MAC (Wei et al. 2024), bitnet.cpp TL1/TL2 — clean-room.
// Layout [blk][rows][4]: a block = 20 inputs = 4 groups of 5; n padded with zero trits (code 121) to a multiple of 20.
constexpr int L5_KCH = 4, L5_RT = 512;
inline int l5_blocks(int n) { return (n + 19) / 20; }
inline std::vector<uint8_t> l5_pack(const int8_t *t, int rows, int n) {
    const int nb = l5_blocks(n); std::vector<uint8_t> v((size_t)nb * rows * 4);
    for (int b = 0; b < nb; b++) for (int r = 0; r < rows; r++) for (int g = 0; g < 4; g++) {
        int code = 0, m = 1;
        for (int j = 0; j < 5; j++) { int i = b * 20 + g * 5 + j; code += (i < n ? t[(size_t)r * n + i] + 1 : 1) * m; m *= 3; }
        v[((size_t)b * rows + r) * 4 + g] = (uint8_t)code;
    }
    return v;
}
// T[code] for 16 queries: xt = 5 consecutive inputs, each [16] int16
inline void l5_build16(__m256i *T, const int16_t *xt) {
    const __m256i x0 = _mm256_loadu_si256((const __m256i *)xt);
    T[0] = _mm256_sub_epi16(_mm256_setzero_si256(), x0); T[1] = _mm256_setzero_si256(); T[2] = x0;
    int len = 3;
    for (int j = 1; j < 5; j++) {                                   // code = low + 3^j·d, trit = d − 1
        const __m256i xj = _mm256_loadu_si256((const __m256i *)(xt + 16 * j));
        for (int c = 0; c < len; c++) { const __m256i lo = T[c]; T[c + 2 * len] = _mm256_add_epi16(lo, xj); T[c + len] = lo; T[c] = _mm256_sub_epi16(lo, xj); }
        len *= 3;
    }
}
// y[r*16 + q] = Σ_i x[q*ldx + i] · t[r, i] for q < nq ≤ 16 (lanes ≥ nq read zero inputs). Scratch is thread-local.
inline void l5_gemm16(const uint8_t *L, int rows, int n, const int8_t *x, int ldx, int nq, int32_t *y) {
    const int nb = l5_blocks(n);
    struct alignas(32) V { __m256i v; };                           // vector<__m256i> would drop the 32-B alignment (aligned loads below)
    thread_local std::vector<int16_t> xt; thread_local std::vector<V> tabv, accv;
    xt.assign((size_t)nb * 20 * 16, 0); tabv.resize((size_t)L5_KCH * 4 * 243); accv.resize(L5_RT);
    __m256i *tab = reinterpret_cast<__m256i *>(tabv.data()), *acc = reinterpret_cast<__m256i *>(accv.data());
    for (int q = 0; q < nq; q++) { const int8_t *xr = x + (size_t)q * ldx; for (int i = 0; i < n; i++) xt[(size_t)i * 16 + q] = xr[i]; }
    memset(y, 0, sizeof(int32_t) * 16 * (size_t)rows);
    const __m256i z = _mm256_setzero_si256();
    for (int b0 = 0; b0 < nb; b0 += L5_KCH) {
        const int b1 = std::min(nb, b0 + L5_KCH);
        for (int b = b0; b < b1; b++) for (int g = 0; g < 4; g++) l5_build16(tab + 243 * ((b - b0) * 4 + g), xt.data() + (size_t)(b * 4 + g) * 80);
        for (int t0 = 0; t0 < rows; t0 += L5_RT) {
            const int t1 = std::min(rows, t0 + L5_RT);
            for (int r = t0; r < t1; r++) acc[r - t0] = z;
            for (int b = b0; b < b1; b++) {
                const __m256i *T = tab + 243 * 4 * (b - b0); const uint8_t *base = L + (size_t)b * rows * 4;
                for (int r = t0; r < t1; r++) {
                    uint32_t w; memcpy(&w, base + (size_t)r * 4, 4);
                    const __m256i s = _mm256_add_epi16(_mm256_add_epi16(T[w & 255], T[243 + ((w >> 8) & 255)]),
                                                       _mm256_add_epi16(T[486 + ((w >> 16) & 255)], T[729 + (w >> 24)]));
                    acc[r - t0] = _mm256_add_epi16(acc[r - t0], s);
                }
            }
            for (int r = t0; r < t1; r++) {                            // widen this chunk's i16 sums into int32
                const __m256i v = acc[r - t0]; __m256i *o0 = (__m256i *)(y + (size_t)r * 16), *o1 = (__m256i *)(y + (size_t)r * 16 + 8);
                _mm256_storeu_si256(o0, _mm256_add_epi32(_mm256_loadu_si256(o0), _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v))));
                _mm256_storeu_si256(o1, _mm256_add_epi32(_mm256_loadu_si256(o1), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1))));
            }
        }
    }
}

}  // namespace trit
