// trit_test.cpp — Exactness tests (P2 dot == int8 dot, P3 round trip) + GEMV speed
// Build: cmake -B build && cmake --build build && ./build/trit_test [rows cols threads]
#include "trit.hpp"
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static inline int32_t tdot(const int8_t *x, const int8_t *t, int n) {
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    int k = 0;
    for (; k + 32 <= n; k += 32) {
        __m256i v = _mm256_sign_epi8(
            _mm256_loadu_si256((const __m256i *)(x + k)),
            _mm256_loadu_si256((const __m256i *)(t + k)));
        __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
        __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
        acc = _mm256_add_epi32(acc, _mm256_add_epi32(
            _mm256_madd_epi16(lo, ones), _mm256_madd_epi16(hi, ones)));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    int32_t r = _mm_cvtsi128_si32(s);
    for (; k < n; k++) r += (int32_t)x[k] * t[k];
    return r;
}

int main(int argc, char **argv) {
    int R = argc > 1 ? std::atoi(argv[1]) : 8192;
    int C = argc > 2 ? std::atoi(argv[2]) : 4096;
    int th = argc > 3 ? std::atoi(argv[3]) : 1;
    omp_set_num_threads(th);

    std::mt19937 g(1);
    std::uniform_int_distribution<int> tri(-1, 1);
    std::uniform_int_distribution<int> xs(-128, 127);

    std::vector<int8_t> W((size_t)R * C), x(C);
    for (auto &v : W) v = (int8_t)tri(g);
    for (auto &v : x) v = (int8_t)xs(g);

    // Pack to P2 and P3
    auto P2 = trit::p2_pack(W.data(), R, C);
    size_t rb = trit::p2_row_bytes(C);
    auto P3 = trit::p3_pack(W.data(), W.size());

    // P3 round-trip test
    std::vector<int8_t> back(W.size());
    trit::p3_unpack(P3.data(), W.size(), back.data());
    bool p3ok = true;
    for (size_t i = 0; i < W.size(); i++) if (back[i] != W[i]) { p3ok = false; break; }

    // ===== Exactness Tests =====
    long bad = 0;

    // 1. Single-row P2 dot vs int8 dot (single x vector)
    for (int r = 0; r < R; r++) {
        int32_t p2d = trit::p2_dot(x.data(), P2.data() + rb * r, C);
        int32_t ref = tdot(x.data(), W.data() + (size_t)r * C, C);
        if (p2d != ref) bad++;
    }

    // 2. P2 dot-4 for 16 X vectors (X16)
    std::vector<int8_t> X16(16 * C);
    for (auto &v : X16) v = (int8_t)xs(g);
    std::vector<int32_t> y16(16 * R, 0);

    for (int r = 0; r < R; r += 4) {
        for (int q = 0; q < 16; q++) {
            int32_t a, b2, c2, d2;
            trit::p2_dot4(X16.data() + (size_t)q * C,
                P2.data() + rb * r,
                P2.data() + rb * (r + 1),
                P2.data() + rb * (r + 2),
                P2.data() + rb * (r + 3), C, a, b2, c2, d2);
            y16[(size_t)r * 16 + q] = a;
            if (r + 1 < R) y16[((size_t)r + 1) * 16 + q] = b2;
            if (r + 2 < R) y16[((size_t)r + 2) * 16 + q] = c2;
            if (r + 3 < R) y16[((size_t)r + 3) * 16 + q] = d2;
        }
    }

    long bad5 = 0;
    for (int r = 0; r < R; r += 389) {
        for (int q = 0; q < 16; q++) {
            int32_t ref = tdot(X16.data() + (size_t)q * C, W.data() + (size_t)r * C, C);
            if (y16[(size_t)r * 16 + q] != ref) bad5++;
        }
    }
    bad += bad5;

    // ===== Speed Tests =====
    double ti = 1e9, tp = 1e9, t4 = 1e9, t5 = 1e9;
    std::vector<int32_t> y(R);

    for (int it = 0; it < 5; it++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < R; r++) y[r] = tdot(x.data(), W.data() + (size_t)r * C, C);
        ti = std::min(ti, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < R; r++) y[r] = trit::p2_dot(x.data(), P2.data() + rb * r, C);
        tp = std::min(tp, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < R; r += 4) {
            int32_t a, b, c, d;
            trit::p2_dot4(x.data(), P2.data() + rb * r, P2.data() + rb * (r + 1),
                P2.data() + rb * (r + 2), P2.data() + rb * (r + 3), C, a, b, c, d);
            y[r] = a; y[r+1] = b; y[r+2] = c; y[r+3] = d;
        }
        t4 = std::min(t4, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    // L5 reference benchmark
    for (int it = 0; it < 3; it++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < R; r += 389)
            for (int q = 0; q < 16; q++)
                bad5 += y16[(size_t)r * 16 + q] != tdot(X16.data() + (size_t)q * C, W.data() + (size_t)r * C, C);
        t5 = std::min(t5, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    bad += bad5;

    // Print JSON result (matches original format)
    printf("{\"l5_mismatches\": %ld, \"bytes_l5\": %zu, \"ms_l5x16_1thr\": %.3f, \"ms_p2x4x16\": %.3f, \"speedup_l5_vs_p2x4\": %.2f}\n",
           bad5, P3.size(), t5 * 1e3, t4 * 1e3, t4 / t5);
    printf("{\"rows\": %d, \"cols\": %d, \"threads\": %d, \"exact_mismatches\": %ld, \"p3_roundtrip\": %s, "
           "\"bytes_int8\": %zu, \"bytes_p2\": %zu, \"bytes_p3\": %zu, \"ms_int8\": %.3f, \"ms_p2\": %.3f, \"speedup_p2\": %.2f, "
           "\"GBps_int8\": %.2f, \"ms_p2x4\": %.3f, \"speedup_p2x4\": %.2f, \"pass\": %s}\n",
           R, C, th, bad, p3ok ? "true" : "false", W.size(), P2.size(), P3.size(),
           ti * 1e3, tp * 1e3, ti / tp, W.size() / ti / 1e9, t4 * 1e3, ti / t4, (bad == 0 && p3ok) ? "true" : "false");

    return (bad == 0 && p3ok) ? 0 : 1;
}