// trit_kernels.hpp — TRIT-LIB-2: per-backend P2 kernels + run-time dispatch.
//
// One algorithm, four SIMD/scalar back-ends, all bit-identical by construction.
// The only primitive is trit = code - 1, where code is a 2-bit field holding
// trit+1 (00 -> -1, 01 -> 0, 10 -> +1, 11 unused). The decode is pure integer
// arithmetic and the accumulator is a widening integer sum, so every back-end
// reduces to the same integers: there is no floating point anywhere, which is
// what makes "bit-identical" exact rather than approximate.
//
//   trit(k) = ((p[k>>2] >> (2*(k&3))) & 3) - 1
//
// Back-ends:
//   SCALAR  portable C reference (the arbiter)
//   AVX2    x86 (pshufb byte-spread + AND + PCMPEQB + PSUBB decode, 32 trits)
//   AVX512  x86 AVX-512 (vpermb byte-spread + AND + cmpeq + vpsubb, 64 trits)
//   NEON    AArch64 (TBL byte-spread + AND + CEQ + SUB decode, 32 trits)
//
// A WGSL compute shader implements the same kernel for WebGPU; it is checked
// against this header by wgsl/run_wgsl.py on the same corpus and same oracle.
//
// Dispatch: trit::kernels::dot() picks the fastest back-end the running CPU
// actually supports, exactly once, and caches the function pointer. Selection
// is by runtime CPUID, never by #ifdef alone, so one fat binary keeps every
// kernel and still runs correctly on a machine that has only some of them.
//
// The x86 kernels carry __attribute__((target(...))) so they are compiled into
// EVERY x86 build regardless of the -m flags used. That matters: without it a
// portable build would silently lack AVX2 and the dispatcher could not fall
// back at run time on an older CPU.

#pragma once

#include <cstdint>
#include <cstddef>

// ── arch headers at GLOBAL scope (never inside a namespace) ──
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  include <immintrin.h>
#  define TRIT_HAS_X86 1
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#  include <arm_neon.h>
#  define TRIT_HAS_NEON 1
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define TRIT_TARGET(isa) __attribute__((target(isa), noinline))
#else
#  define TRIT_TARGET(isa)
#endif

// ───────────────────────── backend identification ─────────────────────────
// Fixed-width enum plus a plain string, so the C++ harness, the Rust side and
// the WGSL runner can all name "which kernel" the same way over a pipe.
enum class TritKernel : int32_t {
    SCALAR = 0,
    AVX2   = 1,
    AVX512 = 2,
    NEON   = 3,
};

static inline const char* trit_kernel_name(TritKernel k) {
    switch (k) {
        case TritKernel::SCALAR: return "scalar";
        case TritKernel::AVX2:   return "avx2";
        case TritKernel::AVX512: return "avx512";
        case TritKernel::NEON:   return "neon";
    }
    return "unknown";
}

namespace trit { namespace kernels {

// The tail every SIMD kernel shares: fewer than one full vector left over.
static inline int32_t tail_scalar(const int8_t* x, const uint8_t* p, int k, int n) {
    int32_t r = 0;
    for (; k < n; k++) {
        int sh = 2 * (k & 3);
        int code = (p[k >> 2] >> sh) & 3;
        r += (int32_t)x[k] * (int32_t)(code - 1);
    }
    return r;
}

// ═══════════════════════════════ SCALAR ═══════════════════════════════════
// The arbiter: a straight transcription of the trit() definition.
static inline int32_t dot_scalar(const int8_t* x, const uint8_t* p, int n) {
    int32_t acc = 0;
    for (int k = 0; k < n; k++) {
        int sh = 2 * (k & 3);
        int code = (p[k >> 2] >> sh) & 3;
        acc += (int32_t)x[k] * (int32_t)(code - 1);
    }
    return acc;
}

// ═══════════════════════════════ AVX2 ═════════════════════════════════════
//
// 32 trits per iteration: spread -> decode -> psignb -> pmaddwd. pmaddwd sums
// two int8-derived int16 lanes into int32; |v| <= 127 and the widened pair is at
// most 254, so there is no overflow and no wrap — the accumulator holds exact
// integers, which is what makes this bit-identical.
//
// Decode 2-bit code -> signed trit for one register of 32 (AVX2) or 64 (AVX-512)
// packed lanes that have ALREADY been spread so lane j holds byte j>>2.
//
// The field for lane j lives at bit offset 2*(j&3) of that byte, so a plain
// mask is not enough: we take four constant shifts (one per phase) and blend
// them by a fixed phase mask. Keeping lane order == trit order is deliberate —
// it means the x vector is consumed as-is, with no de-interleave.
//
//   code = select by (j&3) of { (d>>0)&3, (d>>2)&3, (d>>4)&3, (d>>6)&3 }
//   trit = (code == 0) - (code == 2)          == code-1 for codes {0,1,2}
//
// The shifts are 16-bit lane shifts; bits crossing byte boundaries are removed
// by the following &3, so nothing leaks between lanes.
#if defined(TRIT_HAS_X86)
TRIT_TARGET("avx2")
static inline __m256i decode_w_avx2(__m256i spread) {
    const __m256i c3 = _mm256_set1_epi8(0x03);
    __m256i s0 = _mm256_and_si256(spread,                                  c3);
    __m256i s1 = _mm256_and_si256(_mm256_srli_epi16(spread, 2),           c3);
    __m256i s2 = _mm256_and_si256(_mm256_srli_epi16(spread, 4),           c3);
    __m256i s3 = _mm256_and_si256(_mm256_srli_epi16(spread, 6),           c3);
    // a 32-bit lane holds phases 0,1,2,3 in its four bytes
    __m256i code = _mm256_blendv_epi8(s0, s1, _mm256_set1_epi32(0x0000FF00));
    code = _mm256_blendv_epi8(code, s2, _mm256_set1_epi32(0x00FF0000));
    code = _mm256_blendv_epi8(code, s3, _mm256_set1_epi32(0xFF000000));
    return _mm256_sub_epi8(_mm256_cmpeq_epi8(code, _mm256_setzero_si256()),
                           _mm256_cmpeq_epi8(code, _mm256_set1_epi8(2)));
}
#endif

#if defined(TRIT_HAS_X86)
TRIT_TARGET("avx2")
static int32_t dot_avx2_impl(const int8_t* x, const uint8_t* p, int n) {
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    // lane j must equal packed byte j>>2:
    //   lanes  0..15 <- bytes 0..3 (each 4x)   lanes 16..31 <- bytes 4..7
    const __m128i idxA = _mm_setr_epi8(0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3);
    const __m128i idxB = _mm_setr_epi8(4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7);
    int k = 0;
    for (; k + 32 <= n; k += 32) {
        __m128i src128 = _mm_loadu_si128((const __m128i*)(p + (k >> 2)));
        __m256i spread = _mm256_set_m128i(_mm_shuffle_epi8(src128, idxB),
                                          _mm_shuffle_epi8(src128, idxA));
        __m256i w = decode_w_avx2(spread);                     // signed trit
        __m256i xv = _mm256_loadu_si256((const __m256i*)(x + k));
        // Widen BEFORE applying the sign. psignb negates in 8 bits, so
        // x == -128 with a -1 trit would wrap to -128 instead of +128 and
        // silently disagree with the scalar definition. Widening first makes
        // the negation exact in 16 bits. (This is the one place where a "fast"
        // 8-bit form would be wrong.)
        __m256i xl = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(xv));
        __m256i xh = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv, 1));
        __m256i wl = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w));
        __m256i wh = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
        __m256i lo = _mm256_mullo_epi16(xl, wl);             // x * trit, exact
        __m256i hi = _mm256_mullo_epi16(xh, wh);
        acc = _mm256_add_epi32(acc, _mm256_add_epi32(
            _mm256_madd_epi16(lo, ones), _mm256_madd_epi16(hi, ones)));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc),
                              _mm256_extracti128_si256(acc, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s) + tail_scalar(x, p, k, n);
}
#endif // TRIT_HAS_X86

// ═══════════════════════════════ AVX-512 ══════════════════════════════════
//
// 64 trits per iteration. There is deliberately no psignb at 512 bits (AVX-512
// has no _mm512_sign_epi8), so x*w is built from the two decode masks:
//   code==0 -> +x      code==1 -> 0      code==2 -> -x
// which is exactly code-1 multiplied by x, with no multiply instruction at all.
#if defined(TRIT_HAS_X86)
TRIT_TARGET("avx512f,avx512bw")
static inline __m512i decode_w_avx512(__m512i spread) {
    const __m512i c3 = _mm512_set1_epi8(0x03);
    __m512i s0 = _mm512_and_si512(spread,                        c3);
    __m512i s1 = _mm512_and_si512(_mm512_srli_epi16(spread, 2), c3);
    __m512i s2 = _mm512_and_si512(_mm512_srli_epi16(spread, 4), c3);
    __m512i s3 = _mm512_and_si512(_mm512_srli_epi16(spread, 6), c3);
    // blend by lane phase: phase masks as __mmask64 (one bit per byte). Set the
    // bit for every byte whose lane index j satisfies j&3 == r:
    //   r=1 -> bits 1,5,9,...  = 0x2222222222222222
    //   r=2 -> bits 2,6,10,... = 0x4444444444444444
    //   r=3 -> bits 3,7,11,... = 0x8888888888888888
    const __mmask64 m1 = 0x2222222222222222ull;
    const __mmask64 m2 = 0x4444444444444444ull;
    const __mmask64 m3 = 0x8888888888888888ull;
    __m512i code = _mm512_mask_blend_epi8(m1, s0, s1);
    code = _mm512_mask_blend_epi8(m2, code, s2);
    code = _mm512_mask_blend_epi8(m3, code, s3);
    // trit = code - 1, so code 0 -> -1, code 1 -> 0, code 2 -> +1.
    // Build it from the two compare masks: -1 where code==0, +1 where code==2.
    __mmask64 is0 = _mm512_cmpeq_epi8_mask(code, _mm512_setzero_si512());
    __mmask64 is2 = _mm512_cmpeq_epi8_mask(code, _mm512_set1_epi8(2));
    __m512i w = _mm512_maskz_set1_epi8(is0, (char)-1);
    return _mm512_mask_add_epi8(w, is2, w, _mm512_set1_epi8(1)); // +1 where code==2
}

TRIT_TARGET("avx512f,avx512bw")
static int32_t dot_avx512_impl(const int8_t* x, const uint8_t* p, int n) {
    __m512i acc = _mm512_setzero_si512();
    const __m512i zero = _mm512_setzero_si512();
    const __m256i ones256 = _mm256_set1_epi16(1);   // madd operands are 256-bit
    // spread: 16 packed bytes -> 64 lanes, lane j holds byte j>>2.
    // Built from two 256-bit AVX2 spreads (pshufb) rather than vpermb, which is
    // AVX-512VBMI and absent on Skylake-SP.
    const __m128i idxA = _mm_setr_epi8(0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3);
    const __m128i idxB = _mm_setr_epi8(4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7);
    int k = 0;
    for (; k + 64 <= n; k += 64) {
        // trits 0..31 come from bytes p[k/4 .. k/4+7]
        __m128i s0_128 = _mm_loadu_si128((const __m128i*)(p + (k >> 2)));
        __m256i sp_lo = _mm256_set_m128i(_mm_shuffle_epi8(s0_128, idxB),
                                         _mm_shuffle_epi8(s0_128, idxA));
        // trits 32..63 come from bytes p[k/4+8 .. k/4+15]
        __m128i s1_128 = _mm_loadu_si128((const __m128i*)(p + (k >> 2) + 8));
        __m256i sp_hi = _mm256_set_m128i(_mm_shuffle_epi8(s1_128, idxB),
                                         _mm_shuffle_epi8(s1_128, idxA));
        __m512i spread = _mm512_castsi256_si512(sp_lo);
        spread = _mm512_inserti64x4(spread, sp_hi, 1);
        __m512i w = decode_w_avx512(spread);               // signed trit
        __m512i xv = _mm512_loadu_si512((const void*)(x + k));
        // Widen BEFORE negating, for the same reason as AVX2: an 8-bit
        // negation of -128 wraps to -128 instead of +128 and would disagree
        // with the scalar definition. The 16-bit multiply below is exact.
        // widen 4x(16 int8 -> 16 int16) for both x and the trit, multiply,
        // then pairwise-add to int32.
        __m256i c0 = _mm256_cvtepi8_epi16(_mm512_castsi512_si128(xv));
        __m256i c1 = _mm256_cvtepi8_epi16(_mm512_extracti32x4_epi32(xv, 1));
        __m256i c2 = _mm256_cvtepi8_epi16(_mm512_extracti32x4_epi32(xv, 2));
        __m256i c3 = _mm256_cvtepi8_epi16(_mm512_extracti32x4_epi32(xv, 3));
        __m256i d0 = _mm256_cvtepi8_epi16(_mm512_castsi512_si128(w));
        __m256i d1 = _mm256_cvtepi8_epi16(_mm512_extracti32x4_epi32(w, 1));
        __m256i d2 = _mm256_cvtepi8_epi16(_mm512_extracti32x4_epi32(w, 2));
        __m256i d3 = _mm256_cvtepi8_epi16(_mm512_extracti32x4_epi32(w, 3));
        __m256i pair = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_madd_epi16(_mm256_mullo_epi16(c0, d0), ones256),
                             _mm256_madd_epi16(_mm256_mullo_epi16(c1, d1), ones256)),
            _mm256_add_epi32(_mm256_madd_epi16(_mm256_mullo_epi16(c2, d2), ones256),
                             _mm256_madd_epi16(_mm256_mullo_epi16(c3, d3), ones256)));
        acc = _mm512_add_epi32(acc, _mm512_zextsi256_si512(pair));
    }
    return _mm512_reduce_add_epi32(acc) + tail_scalar(x, p, k, n);
}
#endif // TRIT_HAS_X86

// ═══════════════════════════════ NEON ═════════════════════════════════════
//
// Same phase-decode as the x86 kernels, in NEON form. lane j holds byte j>>2;
// the field for lane j is at bit 2*(j&3), so take four constant shifts and
// blend by the lane's phase. vshrq_n_u16 shifts 16-bit lanes; the &3 removes
// any bits that crossed a byte boundary.
#if defined(TRIT_HAS_NEON)
static inline int8x16_t decode_w_neon(uint8x16_t spread) {
    const uint8x16_t three = vdupq_n_u8(0x03);
    uint8x16_t s0 = vandq_u8(spread,                        three);
    uint8x16_t s1 = vandq_u8(vshrq_n_u16(vreinterpretq_u16_u8(spread), 2), three);
    uint8x16_t s2 = vandq_u8(vshrq_n_u16(vreinterpretq_u16_u8(spread), 4), three);
    uint8x16_t s3 = vandq_u8(vshrq_n_u16(vreinterpretq_u16_u8(spread), 6), three);
    // phase masks (0xFF where lane&3 == r): lanes 1,5,9,13 / 2,6,10,14 / 3,7,11,15
    const uint8x16_t m1 = {0x00,0xFF,0x00,0x00, 0x00,0xFF,0x00,0x00,
                           0x00,0xFF,0x00,0x00, 0x00,0xFF,0x00,0x00};
    const uint8x16_t m2 = {0x00,0x00,0xFF,0x00, 0x00,0x00,0xFF,0x00,
                           0x00,0x00,0xFF,0x00, 0x00,0x00,0xFF,0x00};
    const uint8x16_t m3 = {0x00,0x00,0x00,0xFF, 0x00,0x00,0x00,0xFF,
                           0x00,0x00,0x00,0xFF, 0x00,0x00,0x00,0xFF};
    uint8x16_t code = vbslq_u8(m1, s1, s0);
    code = vbslq_u8(m2, s2, code);
    code = vbslq_u8(m3, s3, code);
    return vreinterpretq_s8_u8(vsubq_u8(vceqq_u8(code, vdupq_n_u8(0)),
                                        vceqq_u8(code, vdupq_n_u8(2))));
}

// aarch64, same decode. 8 packed bytes -> 32 trits (bytes 0..3 -> lanes 0..15,
// bytes 4..7 -> lanes 16..31) via two TBL lookups out of a doubled 8-byte load.
// vpadalq_s16 accumulates int16 pairwise-sums into int32: |v| <= 127, so each
// pair is at most 254 and the widening is exact.
static inline int32_t dot_neon(const int8_t* x, const uint8_t* p, int n) {
    // 8 packed bytes -> 32 trits, so each byte must land in 4 lanes. vqtbl4q_u8
    // takes four 16-byte tables plus a 16-lane index and yields 16 lanes; put the
    // row's 16 packed bytes in all four tables, then index
    // {0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3} to repeat bytes 0..3 four times, and
    // {4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7} for bytes 4..7. Two calls give 32 trits.
    const uint8x16_t idx03 = {0,0,0,0,1,1,1,1, 2,2,2,2,3,3,3,3};
    const uint8x16_t idx47 = {4,4,4,4,5,5,5,5, 6,6,6,6,7,7,7,7};
    int32x4_t acc = vdupq_n_s32(0);
    int k = 0;
    for (; k + 32 <= n; k += 32) {
        uint8x16_t src = vld1q_u8(p + (k >> 2));        // 16 packed bytes
        uint8x16x4_t tab = {src, src, src, src};
        int8x16_t wA = decode_w_neon(vqtbl4q_u8(tab, idx03));
        int8x16_t wB = decode_w_neon(vqtbl4q_u8(tab, idx47));
        // Widen to int16 BEFORE multiplying: an 8-bit multiply wraps for
        // x == -128, which would disagree with the scalar definition.
        int8x16_t xa = vld1q_s8(x + k);
        int8x16_t xb = vld1q_s8(x + k + 16);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(xa),  vget_low_s8(wA)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(xa), vget_high_s8(wA)));
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(xb),  vget_low_s8(wB)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(xb), vget_high_s8(wB)));
    }
    return vaddvq_s32(acc) + tail_scalar(x, p, k, n);
}
#endif // TRIT_HAS_NEON

// ═════════════════════════ run-time dispatch ═════════════════════════════

using DotFn = int32_t(*)(const int8_t*, const uint8_t*, int);

#if defined(TRIT_HAS_X86)
static inline bool cpu_has_avx2()   { return __builtin_cpu_supports("avx2") != 0; }
static inline bool cpu_has_avx512() { return __builtin_cpu_supports("avx512bw") != 0; }
#endif

// Is this back-end present in the binary AND runnable on this CPU?
static inline bool kernel_available(TritKernel k) {
    switch (k) {
        case TritKernel::SCALAR: return true;   // portable, always present
#if defined(TRIT_HAS_NEON)
        case TritKernel::NEON:   return true;
#endif
#if defined(TRIT_HAS_X86)
        case TritKernel::AVX512: return cpu_has_avx512();
        case TritKernel::AVX2:   return cpu_has_avx2();
#endif
        default:                 return false;  // not in this binary
    }
}

// Fastest back-end this binary can actually run right now.
static inline TritKernel best_kernel() {
#if defined(TRIT_HAS_NEON)
    return TritKernel::NEON;
#elif defined(TRIT_HAS_X86)
    if (cpu_has_avx512()) return TritKernel::AVX512;
    if (cpu_has_avx2())   return TritKernel::AVX2;
    return TritKernel::SCALAR;
#else
    return TritKernel::SCALAR;
#endif
}

static inline DotFn kernel_fn(TritKernel k) {
    switch (k) {
#if defined(TRIT_HAS_NEON)
        case TritKernel::NEON:   return &dot_neon;
#endif
#if defined(TRIT_HAS_X86)
        case TritKernel::AVX512: return &dot_avx512_impl;
        case TritKernel::AVX2:   return &dot_avx2_impl;
#endif
        case TritKernel::SCALAR: return &dot_scalar;
        default:                 return &dot_scalar;
    }
}

static inline int32_t dot(const int8_t* x, const uint8_t* p, int n) {
    static DotFn cached = nullptr;          // resolved once, then reused
    if (!cached) cached = kernel_fn(best_kernel());
    return cached(x, p, n);
}

// Call a specific back-end. Refuses (returns the scalar answer) when the
// running CPU cannot execute it, so a forced call can never SIGILL.
static inline int32_t dot_forced(const int8_t* x, const uint8_t* p, int n,
                                 TritKernel k) {
    if (!kernel_available(k)) return dot_scalar(x, p, n);
    return kernel_fn(k)(x, p, n);
}

}} // namespace trit::kernels

// ───────────────────────── C ABI for the runners ─────────────────────────
// force_kernel: -1 = auto (best available); otherwise a TritKernel value.
// Used by the harness, the Intel-SDE runner and the qemu-aarch64 runner so all
// three drive the exact same code through one stable symbol.

extern "C" int32_t trit_dot_dispatch(const int8_t* x, const uint8_t* p, int n,
                                     int32_t force_kernel) {
    if (force_kernel < 0) return trit::kernels::dot(x, p, n);
    return trit::kernels::dot_forced(x, p, n, (TritKernel)force_kernel);
}

// Build fingerprint for the harness JSON: which back-ends are in this binary.
extern "C" int32_t trit_kernel_compiled(int32_t k) {
    switch ((TritKernel)k) {
#if defined(TRIT_HAS_NEON)
        case TritKernel::NEON:   return 1;
#endif
#if defined(TRIT_HAS_X86)
        case TritKernel::AVX512: return 1;
        case TritKernel::AVX2:   return 1;
#endif
        case TritKernel::SCALAR: return 1;
        default:                 return 0;
    }
}

// CPU capability probe, so a runner can report "compiled but not runnable here".
extern "C" int32_t trit_kernel_runnable(int32_t k) {
    return trit::kernels::kernel_available((TritKernel)k) ? 1 : 0;
}