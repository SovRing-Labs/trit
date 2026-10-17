// p3_scores.c — TERNARY-IO-0 T2 (2026-10-14): asymmetric scores computed ON the P3 storage form (5 trits/byte, trit.hpp digit
// order), the trit.hpp "L5" idea for plain row-major P3 rows: per query a table T[b][c] = sum_k q8[5b+k] * trit_k(c) (int32,
// built by the caller from an int8-quantised query), then score[r] = sum_b T[b][P[r][b]]. Integer only: every build and the numpy
// reference reduce to the same int32 sums (bit-identical by construction). Rows are processed in parallel blocks (OpenMP).
// Build: cc -O3 -march=native -fopenmp -shared -fPIC p3_scores.c -o build/libp3scores.so
#include <stdint.h>
#include <stddef.h>

void p3_scores_i32(const uint8_t *P, int64_t n_rows, int nb, const int32_t *T, int32_t *out) {
    #pragma omp parallel for schedule(static, 4096)
    for (int64_t r = 0; r < n_rows; r++) {
        const uint8_t *row = P + (size_t)r * nb;
        int32_t s = 0;
        for (int b = 0; b < nb; b++) s += T[b * 243 + row[b]];
        out[r] = s;
    }
}
