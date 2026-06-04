//! # Test Support
//!
//! Test helpers shared across the crate test suite.
//! This module is only compiled in `#[cfg(test)]` mode.

use std::sync::OnceLock;

use num_bigint::{BigInt, BigUint, Sign};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rstest::fixture;

use crate::{FlexFloat, flexfloat::consts, prelude::BitArrayAccess};

static SEED: OnceLock<u64> = OnceLock::new();

/// Returns the number of experiments to run in tests.
#[fixture]
pub(crate) const fn n_experiments() -> usize {
    100_000
}

/// Returns a random seed for reproducible tests.
#[fixture]
pub(crate) fn seed() -> u64 {
    *SEED.get_or_init(rand::random)
}

/// Returns a random number generator for tests.
#[fixture]
pub(crate) fn rng(n_experiments: usize, seed: u64) -> impl Rng {
    println!("{} experiments with seed {}", n_experiments, seed);
    // No need for cryptographic security in tests, speed is more important
    SmallRng::seed_from_u64(seed)
}

/// Asserts that two floating-point values are almost equal, within a relative tolerance.
#[track_caller]
pub(crate) fn assert_almost_eq(result: f64, expected: f64, message: &str) {
    crate::__private::assert_almost_eq(
        result,
        expected,
        crate::__private::DEFAULT_ASSERT_FF_ALMOST_EQ_TOLERANCE,
        message,
    );
}

/// Asserts that two floating-point values are almost equal for logarithm tests with custom tolerance
#[track_caller]
pub(crate) fn assert_almost_eq_with_tolerance(
    result: f64,
    expected: f64,
    tolerance: f64,
    message: &str,
) {
    crate::__private::assert_almost_eq(result, expected, tolerance, message);
}

/// Generates a random finite f64 value using the provided RNG.
pub(crate) fn random_f64(mut rng: impl Rng) -> f64 {
    loop {
        let float = f64::from_bits(rng.random());
        if float.is_finite() {
            return float;
        }
    }
}

/// Generates a vector of random bits of the given length.
pub(crate) fn random_bits(mut rng: impl Rng, len: usize) -> Vec<bool> {
    (0..len).map(|_| rng.random_bool(0.5)).collect()
}

/// Generates a vector of random bytes of the given length.
pub(crate) fn random_bytes(mut rng: impl Rng, len: usize) -> Vec<u8> {
    (0..len).map(|_| rng.random()).collect()
}

/// Generates a random bit string of the given length.
pub(crate) fn random_bits_string(mut rng: impl Rng, len: usize) -> String {
    (0..len)
        .map(|_| if rng.random_bool(0.5) { '1' } else { '0' })
        .collect()
}

/// Generates a random BigUint with the specified number of bits.
pub(crate) fn random_biguint(mut rng: impl Rng, n_bits: usize) -> BigUint {
    let n_bytes = n_bits.div_ceil(8);
    let mut bytes = vec![0u8; n_bytes];
    rng.fill(&mut bytes[..]);
    let last_num_bits = n_bits % 8;
    if last_num_bits > 0 {
        bytes[n_bytes - 1] &= (1 << last_num_bits) - 1;
    }
    BigUint::from_bytes_le(&bytes)
}

/// Generates a random BigInt with the specified number of bits.
pub(crate) fn random_bigint(mut rng: impl Rng, n_bits: usize) -> BigInt {
    let uint = random_biguint(&mut rng, n_bits - 1);
    if rng.random_bool(0.5) {
        BigInt::from_biguint(Sign::Plus, uint)
    } else {
        BigInt::from_biguint(Sign::Minus, uint)
    }
}

/// Converts a string of '0' and '1' characters to a vector of bools.
pub(crate) fn string_to_bits(s: &str) -> Vec<bool> {
    s.chars().map(|c| c == '1').collect()
}

/// Converts a string of bits to a vector of bytes.
pub(crate) fn string_to_bytes(s: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    for bits in s.as_bytes().chunks(8) {
        let string = std::str::from_utf8(bits).unwrap();

        let mut byte = u8::from_str_radix(string, 2).unwrap();
        byte = byte.reverse_bits() >> (8 - bits.len());
        bytes.push(byte);
    }
    bytes
}

/// Converts an f64 value to a vector of bits (little-endian order).
pub(crate) fn f64_to_bits(value: f64) -> Vec<bool> {
    let bytes = value.to_le_bytes();
    let mut bits = Vec::with_capacity(64);
    for byte in &bytes {
        for i in 0..8 {
            bits.push((byte >> i) & 1 == 1);
        }
    }
    bits
}

pub(crate) fn identity_1<T>(value: T) -> Option<T> {
    Some(value)
}

pub(crate) fn identity_2<T>(a: T, b: T) -> Option<(T, T)> {
    Some((a, b))
}

#[allow(unused)]
pub(crate) fn skip_big_inputs_1(a: f64) -> Option<f64> {
    if a.abs() > 1e270 || a.abs() < 1e-270 {
        None
    } else {
        Some(a)
    }
}

#[allow(unused)]
pub(crate) fn skip_big_inputs_2(a: f64, b: f64) -> Option<(f64, f64)> {
    if a.abs().max(b.abs()) > 1e270 || a.abs().min(b.abs()) < 1e-270 {
        None
    } else {
        Some((a, b))
    }
}

#[inline(always)]
#[track_caller]
pub(crate) fn test_common_logic(
    fn_string: &str,
    result: FlexFloat,
    expected: f64,
    tolerance: Option<f64>,
) {
    // Overflow or underflow cases are still representable as FlexFloat but not as f64, so we need to handle them separately
    let Some(result_f64) = result.to_f64() else {
        assert!(
            result.exponent.len() > 11,
            "{fn_string}: result has unrepresentable shape but exponent is still 11 bits, got {result:?}",
        );

        if expected.is_nan() {
            panic!("{fn_string}: expected NaN but FlexFloat result has grown exponent: {result:?}",);
        }

        if expected.is_infinite() {
            assert!(
                !result.is_infinite() && !result.is_nan(),
                "{fn_string}: expected f64 overflow -> grown finite, got {result:?}",
            );
            assert!(
                result.abs() > consts::MAX_F64,
                "{fn_string}: expected magnitude > MAX_F64 (f64 overflow case), got {result:?}",
            );
            assert_eq!(
                result.is_negative(),
                expected.is_sign_negative(),
                "{fn_string}: sign mismatch in overflow case (expected={expected:?}, result={result:?})",
            );
            return;
        }

        if expected == 0.0 {
            assert!(
                !result.is_zero(),
                "{fn_string}: expected f64 underflow -> grown nonzero, got zero (expected={expected:?})",
            );
            assert!(
                result.abs() < consts::MIN_POSITIVE_F64,
                "{fn_string}: expected magnitude < MIN_POSITIVE_F64 (f64 underflow case), got {result:?}",
            );
            assert_eq!(
                result.is_negative(),
                expected.is_sign_negative(),
                "{fn_string}: sign mismatch in underflow case (expected={expected:?}, result={result:?})",
            );
            return;
        }

        panic!(
            "{fn_string}: expected representable f64 ({expected:?}) but FlexFloat has grown exponent: {result:?}",
        );
    };

    if expected.is_nan() {
        assert!(
            result.is_nan(),
            "{fn_string}: expected NaN, got {:?}",
            result_f64
        );
        return;
    }

    if expected.is_infinite() {
        assert!(
            result_f64.is_infinite(),
            "{fn_string}: expected infinity, got {result_f64:?}"
        );
        assert_eq!(
            result_f64.is_sign_positive(),
            expected.is_sign_positive(),
            "{fn_string}: sign mismatch"
        );
        return;
    }

    let tolerance = tolerance.unwrap_or(crate::__private::DEFAULT_ASSERT_FF_ALMOST_EQ_TOLERANCE);
    assert_almost_eq_with_tolerance(result_f64, expected, tolerance, fn_string);
}

/// Generic test function for unary FlexFloat operations (functions that take one float).
///
/// # Arguments
///
/// * `rng` - Random number generator
/// * `n_experiments` - Number of random test cases to run
/// * `flexfloat_fn` - FlexFloat function to test (e.g., `|x| x.sqrt()`)
/// * `f64_fn` - Corresponding f64 function (e.g., `|x: f64| x.sqrt()`)
/// * `fn_name` - Name of the function for error messages
/// * `value_filter` - Optional filter to preprocess/validate random values
/// * `tolerance` - Optional tolerance for floating-point comparison
#[track_caller]
pub(crate) fn test_unary_flexfloat_op<FF, F64, Filter>(
    mut rng: impl Rng,
    n_experiments: usize,
    flexfloat_fn: FF,
    f64_fn: F64,
    fn_name: &str,
    value_filter: Filter,
    tolerance: Option<f64>,
) where
    FF: Fn(FlexFloat) -> FlexFloat,
    F64: Fn(f64) -> f64,
    Filter: Fn(f64) -> Option<f64>,
{
    use crate::FlexFloat;

    for _ in 0..n_experiments {
        let value = random_f64(&mut rng);

        let value = match (value_filter)(value) {
            None => continue,
            Some(v) => v,
        };

        let ff = FlexFloat::from(value);
        let result = flexfloat_fn(ff);
        let expected = f64_fn(value);

        test_common_logic(
            &format!("{fn_name}({value:?})"),
            result,
            expected,
            tolerance,
        );
    }
}

/// Generic test function for binary FlexFloat operations (functions that take two floats).
///
/// # Arguments
///
/// * `rng` - Random number generator
/// * `n_experiments` - Number of random test cases to run
/// * `flexfloat_fn` - FlexFloat function to test (e.g., `|x, y| x.powf(&y)`)
/// * `f64_fn` - Corresponding f64 function (e.g., `|x: f64, y: f64| x.powf(y)`)
/// * `fn_name` - Name of the function for error messages
/// * `value_filter` - Optional filter to preprocess/validate random value pairs
/// * `tolerance` - Optional tolerance for floating-point comparison
#[track_caller]
pub(crate) fn test_binary_flexfloat_op<FF, F64, Filter>(
    mut rng: impl Rng,
    n_experiments: usize,
    flexfloat_fn: FF,
    f64_fn: F64,
    fn_name: &str,
    value_filter: Filter,
    tolerance: Option<f64>,
) where
    FF: Fn(FlexFloat, FlexFloat) -> FlexFloat,
    F64: Fn(f64, f64) -> f64,
    Filter: Fn(f64, f64) -> Option<(f64, f64)>,
{
    use crate::FlexFloat;

    for _ in 0..n_experiments {
        let a = random_f64(&mut rng);
        let b = random_f64(&mut rng);

        let (a, b) = match (value_filter)(a, b) {
            None => continue,
            Some((va, vb)) => (va, vb),
        };
        let fa = FlexFloat::from(a);
        let fb = FlexFloat::from(b);

        let result = flexfloat_fn(fa, fb);
        let expected = f64_fn(a, b);

        test_common_logic(
            &format!("{fn_name}({a:?}, {b:?})"),
            result,
            expected,
            tolerance,
        );
    }
}
