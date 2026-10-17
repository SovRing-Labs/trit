// trit_kernel_test.cpp — TRIT-LIB-2 verification harness.
//
// Answers one question with numbers: do every compiled back-end of the P2 dot
// kernel return BIT-IDENTICAL results to the scalar arbiter, and how fast is
// each one?
//
//   1. LAW CASES   — the exactness laws from the trit library: all-trit values,
//                    unit vectors, alternating signs, single-lane perturbations,
//                    and every width around the vector boundaries (31/32/33,
//                    63/64/65, ...). These are the cases a SIMD kernel breaks on
//                    when it mishandles a lane, a shift or a tail.
//   2. RANDOM ROWS — 1,000,000 rows of random trits x random int8 activations,
//                    every row compared against the scalar arbiter.
//   3. MB/s TABLE   — throughput per back-end over a GEMV-shaped workload,
//                    reported against the scalar baseline.
//
// Every back-end that is compiled into this binary AND runnable on the running
// CPU is exercised. Back-ends that cannot run here are reported as
// "compiled=1 runnable=0" rather than silently skipped, so the JSON is an
// honest fingerprint of what was actually proved.
//
// Output: one JSON object on stdout. Exit 0 iff every comparison matched.
//
// Build (native, exercises scalar + AVX2):
//   g++ -O2 -std=c++17 -I cpp tests/trit_kernel_test.cpp -o trit_kernel_test
// Build for AVX-512 verification (run under Intel SDE):
//   g++ -O2 -std=c++17 -I cpp tests/trit_kernel_test.cpp -o trit_kernel_test
//   sde64 -skx -- ./trit_kernel_test
// Build for NEON verification (run under qemu-aarch64):
//   zig c++ --target=aarch64-linux-musl -O2 -std=c++17 -static \
//       -I cpp tests/trit_kernel_test.cpp -o trit_kernel_test_neon
//   qemu-aarch64-static ./trit_kernel_test_neon

#include "trit_kernels.hpp"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

// ─────────────────────────── P2 helpers ───────────────────────────
// Row bytes for n trits, padded to a multiple of 32 (matches trit.hpp).
size_t row_bytes(int n) { return (size_t)((n + 31) / 32) * 8; }

// Pack trits (-1,0,+1) into P2: code = trit+1 in 2 bits, 4 trits per byte.
void pack_row(const int8_t* t, int n, uint8_t* out, size_t nb) {
    std::memset(out, 0x55, nb);                 // 0x55 = four zero trits
    for (int i = 0; i < n; i++) {
        int sh = 2 * (i & 3);
        out[i >> 2] = (uint8_t)((out[i >> 2] & ~(3 << sh)) | ((t[i] + 1) << sh));
    }
}

// ─────────────────────────── back-end registry ───────────────────────────
struct Backend { TritKernel k; const char* name; };

std::vector<Backend> all_backends() {
    return {{TritKernel::SCALAR, "scalar"},
            {TritKernel::AVX2,   "avx2"},
            {TritKernel::AVX512, "avx512"},
            {TritKernel::NEON,   "neon"}};
}

// ─────────────────────────── law cases ───────────────────────────
// Each law builds one (weights, x) pair of length n and names the property it
// pins down. The oracle is always dot_scalar, so a law case that fails means a
// back-end deviates from the definition, not from another kernel.
struct LawCase { const char* name; int n; };

std::vector<LawCase> law_widths() {
    // widths around every vector boundary in every kernel: 32 (AVX2/NEON),
    // 64 (AVX-512), plus the small and awkward sizes.
    std::vector<LawCase> v;
    for (int n : {1, 2, 3, 4, 5, 7, 8, 15, 16, 17,
                  31, 32, 33, 47, 63, 64, 65,
                  95, 96, 97, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025})
        v.push_back({"boundary", n});
    return v;
}

// Build the deterministic law inputs for a width. Six distinct laws per width.
enum Law { L_ALL_POS, L_ALL_NEG, L_ALT, L_UNIT_HEAD, L_ZERO_WEIGHTS, L_EXTREME_X };

LawCase law_of(int n, Law& out_law) {
    // cycle the six laws across widths so every width sees every law
    static const Law laws[6] = {L_ALL_POS, L_ALL_NEG, L_ALT,
                                L_UNIT_HEAD, L_ZERO_WEIGHTS, L_EXTREME_X};
    out_law = laws[n % 6];
    return {"law", n};
}

void make_law(Law law, int n, std::vector<int8_t>& w, std::vector<int8_t>& x) {
    w.assign(n, 0);
    x.assign(n, 0);
    for (int i = 0; i < n; i++) {
        switch (law) {
            case L_ALL_POS:      w[i] = +1;            x[i] = (int8_t)((i % 127) + 1); break;
            case L_ALL_NEG:      w[i] = -1;            x[i] = (int8_t)(-(i % 127 + 1)); break;
            case L_ALT:          w[i] = (i & 1) ? -1 : +1; x[i] = (int8_t)((i * 37) % 255 - 127); break;
            case L_UNIT_HEAD:    w[i] = (i == 0) ? +1 : 0; x[i] = (int8_t)((i % 255) - 127); break;
            case L_ZERO_WEIGHTS: w[i] = 0;             x[i] = (int8_t)((i % 255) - 127); break;
            case L_EXTREME_X:    w[i] = (int8_t)((i % 3) - 1);
                                 x[i] = (i & 1) ? (int8_t)-128 : (int8_t)127; break;
        }
    }
}

// ─────────────────────────── throughput ───────────────────────────
// A GEMV-shaped workload: one query, many weight rows, P2 packed. Measures
// effective bandwidth over the packed representation.
struct BenchResult { double ms; double mb_per_s; };

BenchResult bench(TritKernel k, const std::vector<std::vector<uint8_t>>& rows,
                  size_t rb, const int8_t* x, int n, int reps) {
    volatile int32_t sink = 0;
    double best = 1e30;
    std::vector<int32_t> out(rows.size());
    for (int rep = 0; rep < reps; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        for (size_t r = 0; r < rows.size(); r++)
            out[r] = trit::kernels::dot_forced(x, rows[r].data(), n, k);
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        for (int32_t v : out) sink += v;
        if (ms < best) best = ms;
    }
    (void)sink;
    // bytes touched = packed rows only; this is the P2 baseline the library
    // exists to reduce.
    double bytes = (double)rows.size() * (double)rb;
    return {best, bytes / (best * 1e-3) / 1e6};
}

} // namespace

int main(int argc, char** argv) {
    // ---- knobs (defaults chosen so a full run is a few seconds) ----
    long random_rows = 1000000;   // 1M random rows, per the verify spec
    int  bench_rows  = 4096;
    int  bench_cols  = 4096;
    int  bench_reps  = 3;
    unsigned seed    = 20261010u;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string("0"); };
        if      (a == "--random-rows") random_rows = std::stol(next());
        else if (a == "--bench-rows")  bench_rows  = std::stoi(next());
        else if (a == "--bench-cols")  bench_cols  = std::stoi(next());
        else if (a == "--bench-reps")  bench_reps  = std::stoi(next());
        else if (a == "--seed")        seed       = (unsigned)std::stoul(next());
    }

    auto backends = all_backends();
    std::mt19937 g(seed);
    std::uniform_int_distribution<int> tri(-1, 1);
    std::uniform_int_distribution<int> xs(-128, 127);

    long law_checked = 0, law_bad = 0;
    long rand_checked = 0, rand_bad = 0;
    long per_backend_bad[4] = {0, 0, 0, 0};
    long per_backend_checked[4] = {0, 0, 0, 0};

    // ─────────── 1. LAW CASES ───────────
    for (int n : {1, 2, 3, 4, 5, 7, 8, 15, 16, 17,
                  31, 32, 33, 47, 63, 64, 65,
                  95, 96, 97, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025}) {
        for (int law_i = 0; law_i < 6; law_i++) {
            Law law = (Law)law_i;
            std::vector<int8_t> w, x;
            make_law(law, n, w, x);
            size_t rb = row_bytes(n);
            std::vector<uint8_t> p(rb);
            pack_row(w.data(), n, p.data(), rb);
            int32_t ref = trit::kernels::dot_scalar(x.data(), p.data(), n);
            for (size_t bi = 0; bi < backends.size(); bi++) {
                if (!trit::kernels::kernel_available(backends[bi].k)) continue;
                int32_t got = trit::kernels::dot_forced(x.data(), p.data(), n,
                                                        backends[bi].k);
                law_checked++;
                per_backend_checked[bi]++;
                if (got != ref) { law_bad++; per_backend_bad[bi]++; }
            }
        }
    }

    // ─────────── 2. RANDOM ROWS (1M) ───────────
    // One query vector, many random weight rows — the GEMV shape the kernel
    // actually serves. Rows are generated and checked in blocks so memory stays
    // bounded regardless of the requested count.
    {
        const int n = bench_cols;
        std::vector<int8_t> x(n);
        for (auto& v : x) v = (int8_t)xs(g);
        size_t rb = row_bytes(n);
        const long BLOCK = 4096;
        std::vector<int8_t> w((size_t)BLOCK * n);
        std::vector<uint8_t> p(rb * BLOCK);
        for (long done = 0; done < random_rows; ) {
            long m = std::min(BLOCK, random_rows - done);
            for (long r = 0; r < m; r++)
                for (int i = 0; i < n; i++) w[(size_t)r * n + i] = (int8_t)tri(g);
            for (long r = 0; r < m; r++)
                pack_row(w.data() + (size_t)r * n, n, p.data() + rb * r, rb);
            for (long r = 0; r < m; r++) {
                int32_t ref = trit::kernels::dot_scalar(x.data(), p.data() + rb * r, n);
                for (size_t bi = 0; bi < backends.size(); bi++) {
                    if (!trit::kernels::kernel_available(backends[bi].k)) continue;
                    int32_t got = trit::kernels::dot_forced(x.data(), p.data() + rb * r, n,
                                                            backends[bi].k);
                    rand_checked++;
                    per_backend_checked[bi]++;
                    if (got != ref) { rand_bad++; per_backend_bad[bi]++; }
                }
            }
            done += m;
        }
    }

    // ─────────── 3. MB/s TABLE ───────────
    {
        int n = bench_cols;
        std::vector<int8_t> x(n);
        for (auto& v : x) v = (int8_t)xs(g);
        size_t rb = row_bytes(n);
        std::vector<std::vector<uint8_t>> rows;
        rows.reserve(bench_rows);
        std::vector<int8_t> w(n);
        for (int r = 0; r < bench_rows; r++) {
            for (int i = 0; i < n; i++) w[i] = (int8_t)tri(g);
            std::vector<uint8_t> p(rb);
            pack_row(w.data(), n, p.data(), rb);
            rows.push_back(std::move(p));
        }
        double base_mbps = 0;
        printf("%-8s %12s %12s %10s\n", "backend", "ms", "MB/s", "vs scalar");
        for (size_t bi = 0; bi < backends.size(); bi++) {
            if (!trit::kernels::kernel_available(backends[bi].k)) {
                printf("%-8s %12s %12s %10s\n", backends[bi].name,
                       "not-runnable", "-", "-");
                continue;
            }
            BenchResult b = bench(backends[bi].k, rows, rb, x.data(), n, bench_reps);
            if (backends[bi].k == TritKernel::SCALAR) base_mbps = b.mb_per_s;
            printf("%-8s %12.3f %12.2f", backends[bi].name, b.ms, b.mb_per_s);
            if (base_mbps > 0 && backends[bi].k != TritKernel::SCALAR)
                printf(" %9.2fx", b.mb_per_s / base_mbps);
            printf("\n");
        }
    }

    // ─────────── JSON report ───────────
    bool pass = (law_bad == 0 && rand_bad == 0);
    printf("\n{");
    printf("\"task\": \"TRIT-LIB-2\",");
    printf("\"best_kernel\": \"%s\",", trit_kernel_name(trit::kernels::best_kernel()));
    printf("\"law_cases_checked\": %ld, \"law_mismatches\": %ld,", law_checked, law_bad);
    printf("\"random_rows_requested\": %ld, \"random_rows_checked\": %ld, \"random_mismatches\": %ld,",
           random_rows, rand_checked, rand_bad);
    printf("\"bench_cols\": %d, \"bench_rows\": %d,", bench_cols, bench_rows);
    printf("\"backends\": [");
    bool first = true;
    for (size_t bi = 0; bi < backends.size(); bi++) {
        if (!first) printf(",");
        first = false;
        printf("{\"name\": \"%s\", \"compiled\": %d, \"runnable\": %d, "
               "\"checked\": %ld, \"mismatches\": %ld}",
               backends[bi].name,
               trit_kernel_compiled((int32_t)backends[bi].k),
               trit_kernel_runnable((int32_t)backends[bi].k),
               per_backend_checked[bi], per_backend_bad[bi]);
    }
    printf("], \"pass\": %s}\n", pass ? "true" : "false");
    return pass ? 0 : 1;
}