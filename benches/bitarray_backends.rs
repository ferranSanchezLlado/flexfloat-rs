use core::f64;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use flexfloat::bitarray::{BitArrayArith, BoolBitArray, UsizeBitArray};
use flexfloat::flexfloat::FlexFloat;

const BIT_SIZES: [usize; 3] = [64, 256, 1024];
const BITARRAY_BATCHES: usize = 256;
const FLEXFLOAT_BATCHES: usize = 512;
const FLOAT_INPUTS: [f64; 8] = [
    0.125,
    0.5,
    1.0,
    1.5,
    f64::consts::PI,
    10.25,
    1234.5678,
    65536.5,
];

fn patterned_bits(n_bits: usize) -> Vec<bool> {
    (0..n_bits).map(|i| ((i * 37 + 11) % 17) < 8).collect()
}

fn patterned_bytes(n_bits: usize) -> Vec<u8> {
    let n_bytes = n_bits.div_ceil(8);
    (0..n_bytes)
        .map(|i| ((i * 29 + 7) as u8).rotate_left((i % 8) as u32))
        .collect()
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(5))
        .measurement_time(Duration::from_secs(10))
        .sample_size(120)
        .noise_threshold(0.03)
}

fn bench_bitarray_backend<B: BitArrayArith + 'static>(c: &mut Criterion, backend_name: &str) {
    let mut group = c.benchmark_group(format!("bitarray/{backend_name}"));

    for n_bits in BIT_SIZES {
        let bits = patterned_bits(n_bits);
        let bytes = patterned_bytes(n_bits);
        let base = B::from_bits(&bits);
        let mid = n_bits / 4;
        let end = mid + n_bits / 2;

        group.throughput(Throughput::Elements((n_bits * BITARRAY_BATCHES) as u64));

        group.bench_with_input(BenchmarkId::new("from_bits", n_bits), &bits, |b, bits| {
            b.iter(|| {
                for _ in 0..BITARRAY_BATCHES {
                    black_box(B::from_bits(black_box(bits)));
                }
            })
        });

        group.bench_with_input(
            BenchmarkId::new("from_bytes", n_bits),
            &bytes,
            |b, bytes| {
                b.iter(|| {
                    for _ in 0..BITARRAY_BATCHES {
                        black_box(B::from_bytes(black_box(bytes), n_bits));
                    }
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("get_range", n_bits), &base, |b, base| {
            b.iter(|| {
                for _ in 0..BITARRAY_BATCHES {
                    black_box(base.get_range(mid..end).unwrap());
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("shift_fixed", n_bits), &base, |b, base| {
            b.iter(|| {
                for _ in 0..BITARRAY_BATCHES {
                    black_box(base.clone().shift_fixed(13));
                }
            })
        });

        group.bench_with_input(
            BenchmarkId::new("append_repeated", n_bits),
            &base,
            |b, base| {
                b.iter(|| {
                    for _ in 0..BITARRAY_BATCHES {
                        black_box(base.clone().append_repeated(true, 32));
                    }
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("to_bytes", n_bits), &base, |b, base| {
            b.iter(|| {
                for _ in 0..BITARRAY_BATCHES {
                    black_box(base.to_bytes());
                }
            })
        });
    }

    group.finish();
}

fn bench_flexfloat_backend<B: BitArrayArith + 'static>(c: &mut Criterion, backend_name: &str) {
    let mut group = c.benchmark_group(format!("flexfloat/{backend_name}"));
    group.throughput(Throughput::Elements(
        (FLOAT_INPUTS.len() * FLEXFLOAT_BATCHES) as u64,
    ));

    let floats: Vec<FlexFloat<B>> = FLOAT_INPUTS
        .iter()
        .copied()
        .map(FlexFloat::<B>::from_f64)
        .collect();
    let lhs = floats.clone();
    let rhs: Vec<FlexFloat<B>> = FLOAT_INPUTS
        .iter()
        .rev()
        .copied()
        .map(FlexFloat::<B>::from_f64)
        .collect();

    group.bench_function("from_f64", |b| {
        b.iter(|| {
            for _ in 0..FLEXFLOAT_BATCHES {
                for value in FLOAT_INPUTS {
                    black_box(FlexFloat::<B>::from_f64(black_box(value)));
                }
            }
        })
    });

    group.bench_function("to_f64", |b| {
        b.iter(|| {
            for _ in 0..FLEXFLOAT_BATCHES {
                for value in &floats {
                    black_box(value.to_f64().unwrap());
                }
            }
        })
    });

    group.bench_function("add", |b| {
        b.iter(|| {
            for _ in 0..FLEXFLOAT_BATCHES {
                for (left, right) in lhs.iter().zip(&rhs) {
                    black_box(left.clone() + right);
                }
            }
        })
    });

    group.bench_function("mul", |b| {
        b.iter(|| {
            for _ in 0..FLEXFLOAT_BATCHES {
                for (left, right) in lhs.iter().zip(&rhs) {
                    black_box(left.clone() * right);
                }
            }
        })
    });

    group.bench_function("sqrt", |b| {
        b.iter(|| {
            for _ in 0..FLEXFLOAT_BATCHES {
                for value in &floats {
                    black_box(value.clone().sqrt());
                }
            }
        })
    });

    group.finish();
}

fn bench_f64_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("flexfloat/f64_baseline");
    group.throughput(Throughput::Elements(
        (FLOAT_INPUTS.len() * FLEXFLOAT_BATCHES) as u64,
    ));

    let lhs: Vec<f64> = FLOAT_INPUTS.to_vec();
    let rhs: Vec<f64> = FLOAT_INPUTS.iter().rev().copied().collect();

    // "from_f64": identity cast to signal that the value is already f64.
    // This measures the loop + black_box overhead as a baseline.
    group.bench_function("from_f64", |b| {
        b.iter(|| {
            for _ in 0..FLEXFLOAT_BATCHES {
                for value in FLOAT_INPUTS {
                    black_box(black_box(value) as f64);
                }
            }
        })
    });

    // "to_f64": same — already f64, baseline cost.
    group.bench_function("to_f64", |b| {
        b.iter(|| {
            for _ in 0..FLEXFLOAT_BATCHES {
                for value in &lhs {
                    black_box(*black_box(value));
                }
            }
        })
    });

    group.bench_function("add", |b| {
        b.iter(|| {
            for _ in 0..FLEXFLOAT_BATCHES {
                for (left, right) in lhs.iter().zip(&rhs) {
                    black_box(black_box(*left) + black_box(*right));
                }
            }
        })
    });

    group.bench_function("mul", |b| {
        b.iter(|| {
            for _ in 0..FLEXFLOAT_BATCHES {
                for (left, right) in lhs.iter().zip(&rhs) {
                    black_box(black_box(*left) * black_box(*right));
                }
            }
        })
    });

    group.bench_function("sqrt", |b| {
        b.iter(|| {
            for _ in 0..FLEXFLOAT_BATCHES {
                for value in &lhs {
                    black_box(black_box(*value).sqrt());
                }
            }
        })
    });

    group.finish();
}

fn bitarray_backends(c: &mut Criterion) {
    bench_bitarray_backend::<BoolBitArray>(c, "bool");
    bench_bitarray_backend::<UsizeBitArray>(c, "usize");
    bench_flexfloat_backend::<BoolBitArray>(c, "bool");
    bench_flexfloat_backend::<UsizeBitArray>(c, "usize");
    bench_f64_baseline(c);
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bitarray_backends
}
criterion_main!(benches);
