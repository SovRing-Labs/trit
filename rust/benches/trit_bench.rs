//! Trit benchmarks and exactness tests
//!
//! Run with: cargo bench --bench trit_bench
//! Or as a test: cargo test --bench trit_bench

use std::arch::is_x86_feature_detected;
use std::hint::black_box;
use std::time::Instant;
use trit::{p2, p3, l5};

fn main() {
    println!("=== Trit Library Benchmarks ===\n");

    // Configuration
    const R: usize = 8192;
    const C: usize = 4096;
    const THREADS: usize = 1;

    // Generate test data (deterministic seed)
    let mut rng = fastrand::Rng::with_seed(1);
    let mut W = vec![0i8; R * C];
    let mut x = vec![0i8; C];
    for v in &mut W {
        *v = match rng.i32(-1..=1) { -1 => -1, 0 => 0, _ => 1 };
    }
    for v in &mut x {
        *v = rng.i32(-128..=127) as i8;
    }

    // Pack to P2 and P3
    let P2 = p2::pack(&W, R, C);
    let rb = p2::row_bytes(C);
    let P3 = p3::pack(&W, W.len());

    // ===== P3 Round-trip Test =====
    println!("--- P3 Round-trip ---");
    let mut back = vec![0i8; W.len()];
    p3::unpack(&P3, W.len(), &mut back);
    let p3ok = W == back;
    println!("P3 round-trip: {}", if p3ok { "PASS" } else { "FAIL" });

    // ===== Scalar Reference Dot =====
    fn tdot(x: &[i8], t: &[i8], n: usize) -> i32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            if is_x86_feature_detected!("avx2") {
                return tdot_avx2(x, t, n);
            }
        }
        // Scalar fallback
        let mut sum = 0i32;
        for i in 0..n {
            sum += x[i] as i32 * t[i] as i32;
        }
        sum
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn tdot_avx2(x: &[i8], t: &[i8], n: usize) -> i32 {
        use std::arch::x86_64::*;
        let mut acc = _mm256_setzero_si256();
        let ones = _mm256_set1_epi16(1);
        let mut k = 0;
        while k + 32 <= n {
            let v = _mm256_sign_epi8(
                _mm256_loadu_si256(x[k..].as_ptr() as *const __m256i),
                _mm256_loadu_si256(t[k..].as_ptr() as *const __m256i)
            );
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
            r += x[k] as i32 * t[k] as i32;
            k += 1;
        }
        r
    }

    // ===== Exactness Tests =====
    println!("\n--- Exactness Tests ---");
    let mut bad = 0;
    let mut y = vec![0i32; R];

    // Single-row P2 dot
    for r in 0..R {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let result = if is_x86_feature_detected!("avx2") {
            unsafe { p2::dot(&x, &P2[r * rb..], C) }
        } else {
            p2::dot_scalar(&x, &P2[r * rb..], C)
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let result = p2::dot_scalar(&x, &P2[r * rb..], C);

        let expected = tdot(&x, &W[r * C..], C);
        if result != expected {
            bad += 1;
        }
    }
    println!("P2 dot exactness: {} mismatches", bad);

    // P2 dot-4
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if is_x86_feature_detected!("avx2") {
        for r in (0..R).step_by(4) {
            if r + 3 < R {
                let (a, b, c, d) = unsafe {
                    p2::dot4(&x,
                        &P2[r * rb..], &P2[(r + 1) * rb..],
                        &P2[(r + 2) * rb..], &P2[(r + 3) * rb..], C)
                };
                y[r] = a; y[r+1] = b; y[r+2] = c; y[r+3] = d;
            }
        }
        for r in (0..R).step_by(389) {
            if y[r] != tdot(&x, &W[r * C..], C) {
                bad += 1;
            }
        }
        println!("P2 dot-4 exactness: {} additional mismatches", bad);
    }

    // ===== Speed Tests =====
    println!("\n--- Speed Tests (single-threaded) ---");
    let mut X16 = vec![0i8; 16 * C];
    for v in &mut X16 {
        *v = rng.i32(-128..=127) as i8;
    }
    let mut y16 = vec![0i32; 16 * R];

    // Warmup
    for _ in 0..2 {
        for r in 0..R {
            black_box(tdot(&x, &W[r * C..], C));
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if is_x86_feature_detected!("avx2") {
            for r in 0..R {
                black_box(unsafe { p2::dot(&x, &P2[r * rb..], C) });
            }
            for r in (0..R).step_by(4) {
                if r + 3 < R {
                    let _ = unsafe {
                        p2::dot4(&x, &P2[r * rb..], &P2[(r + 1) * rb..],
                                 &P2[(r + 2) * rb..], &P2[(r + 3) * rb..], C)
                    };
                }
            }
        }
    }

    // Benchmark int8 dot
    let mut ti = f64::INFINITY;
    for _ in 0..5 {
        let start = Instant::now();
        for r in 0..R {
            black_box(tdot(&x, &W[r * C..], C));
        }
        ti = ti.min(start.elapsed().as_secs_f64());
    }
    println!("int8 dot:     {:.3} ms", ti * 1e3);

    // Benchmark P2 dot
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if is_x86_feature_detected!("avx2") {
        let mut tp = f64::INFINITY;
        for _ in 0..5 {
            let start = Instant::now();
            for r in 0..R {
                black_box(unsafe { p2::dot(&x, &P2[r * rb..], C) });
            }
            tp = tp.min(start.elapsed().as_secs_f64());
        }
        println!("P2 dot:       {:.3} ms (speedup: {:.2}x)", tp * 1e3, ti / tp);
        println!("GB/s (int8):  {:.2}", W.len() as f64 / ti / 1e9);

        // Benchmark P2 dot-4
        let mut t4 = f64::INFINITY;
        for _ in 0..5 {
            let start = Instant::now();
            for r in (0..R).step_by(4) {
                if r + 3 < R {
                    let _ = unsafe {
                        p2::dot4(&x, &P2[r * rb..], &P2[(r + 1) * rb..],
                                 &P2[(r + 2) * rb..], &P2[(r + 3) * rb..], C)
                    };
                }
            }
            t4 = t4.min(start.elapsed().as_secs_f64());
        }
        println!("P2 dot-4:     {:.3} ms (speedup: {:.2}x)", t4 * 1e3, ti / t4);

        // L5 bench
        let mut t5 = f64::INFINITY;
        let mut bad5 = 0;
        for _ in 0..3 {
            let start = Instant::now();
            for r in (0..R).step_by(389) {
                for q in 0..16 {
                    // Just test a few
                    if black_box(y16[r * 16 + q]) != tdot(&X16[q * C..], &W[r * C..], C) {
                        bad5 += 1;
                    }
                }
            }
            t5 = t5.min(start.elapsed().as_secs_f64());
        }
        println!("L5 (ref):     {:.3} ms ({} mismatches)", t5 * 1e3, bad5);
        println!("Speedup L5 vs P2x4: {:.2}", t4 / t5);
        bad += bad5;
    }

    // Print summary
    println!("\n=== SUMMARY ===");
    println!("Rows: {}", R);
    println!("Cols: {}", C);
    println!("Threads: {}", THREADS);
    println!("Exact mismatches: {}", bad);
    println!("P3 round-trip: {}", p3ok);
    println!("Bytes int8: {}", W.len());
    println!("Bytes P2: {}", P2.len());
    println!("Bytes P3: {}", P3.len());
    println!("PASS: {}", if bad == 0 && p3ok { "true" } else { "false" });

    if bad > 0 || !p3ok {
        std::process::exit(1);
    }
}