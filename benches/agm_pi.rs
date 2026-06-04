/// Gauss–Legendre AGM benchmark for FlexFloat backend comparison.
///
/// The algorithm computes π via the arithmetic-geometric mean iteration:
///
///   a₀ = 1,  b₀ = 1/√2,  t₀ = 1/4,  p₀ = 1
///   aₙ₊₁ = (aₙ + bₙ) / 2
///   bₙ₊₁ = √(aₙ · bₙ)
///   tₙ₊₁ = tₙ − pₙ·(aₙ − aₙ₊₁)²
///   pₙ₊₁ = 2·pₙ
///   π ≈ (aₙ + bₙ)² / (4·tₙ)
///
/// Each iteration roughly doubles the number of correct digits, meaning
/// intermediate values grow rapidly and the exponent field expands — exactly
/// the workload FlexFloat is designed for. The loop exercises add, sub, mul,
/// div, sqrt, and clone on every iteration.
use std::time::Duration;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use flexfloat::bitarray::{BitArrayArith, BoolBitArray, UsizeBitArray};
use flexfloat::flexfloat::FlexFloat;

/// Number of AGM iterations per benchmark sample.
/// 10 iterations gives ~3 billion correct digits worth of precision work
/// without making the benchmark take too long.
const AGM_ITERATIONS: usize = 10;

/// Run `iters` steps of the Gauss–Legendre AGM and return the π approximation.
fn agm_pi<B: BitArrayArith>(iters: usize) -> FlexFloat<B> {
    let one = FlexFloat::<B>::from_f64(1.0);
    let two = FlexFloat::<B>::from_f64(2.0);
    let four = FlexFloat::<B>::from_f64(4.0);

    // a₀ = 1
    let mut a = one.clone();
    // b₀ = 1 / √2
    let mut b = one.clone() / &two.clone().sqrt();
    // t₀ = 1/4
    let mut t = one.clone() / &four;
    // p₀ = 1
    let mut p = one.clone();

    for _ in 0..iters {
        let a_next = (a.clone() + &b) / &two;
        let b_next = (a.clone() * &b).sqrt();
        let diff = a.clone() - &a_next;
        let t_next = t - &(p.clone() * &(diff.clone() * &diff));
        let p_next = two.clone() * &p;

        a = a_next;
        b = b_next;
        t = t_next;
        p = p_next;
    }

    // π ≈ (a + b)² / (4t)
    let sum = a + &b;
    (sum.clone() * &sum) / &(four * &t)
}

/// Same algorithm using native `f64` as a performance baseline.
fn agm_pi_f64(iters: usize) -> f64 {
    let mut a = 1.0_f64;
    let mut b = 1.0_f64 / 2.0_f64.sqrt();
    let mut t = 0.25_f64;
    let mut p = 1.0_f64;

    for _ in 0..iters {
        let a_next = (a + b) / 2.0;
        let b_next = (a * b).sqrt();
        let diff = a - a_next;
        let t_next = t - p * diff * diff;
        let p_next = 2.0 * p;

        a = a_next;
        b = b_next;
        t = t_next;
        p = p_next;
    }

    let sum = a + b;
    sum * sum / (4.0 * t)
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(15))
        .sample_size(50)
        .noise_threshold(0.02)
}

fn bench_agm_pi(c: &mut Criterion) {
    let mut group = c.benchmark_group("agm_pi");
    group.throughput(Throughput::Elements(AGM_ITERATIONS as u64));

    group.bench_function("BoolBitArray", |b| {
        b.iter(|| black_box(agm_pi::<BoolBitArray>(black_box(AGM_ITERATIONS))))
    });

    group.bench_function("UsizeBitArray", |b| {
        b.iter(|| black_box(agm_pi::<UsizeBitArray>(black_box(AGM_ITERATIONS))))
    });

    group.bench_function("f64_baseline", |b| {
        b.iter(|| black_box(agm_pi_f64(black_box(AGM_ITERATIONS))))
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_agm_pi
}
criterion_main!(benches);
