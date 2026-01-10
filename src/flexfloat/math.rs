use core::f64;

use crate::flexfloat::grow_exponent;
use crate::{BitArray, FlexFloat};

/// Rounds the value to the nearest integer using round half to even
fn round<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    // Handle special cases: NaN, Infinity
    if value.is_nan() || value.is_infinity() {
        value
    } else if value >= FlexFloat::zero() {
        floor(value + FlexFloat::from_f64(0.5))
    } else {
        ceil(value - FlexFloat::from_f64(0.5))
    }
}

fn floor<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    if value.is_nan() || value.is_infinity() || value.is_zero() {
        return value;
    }

    let trunc = value.clone().to_int().expect("Handled NaN/Inf above");
    let value_trunc: FlexFloat<B> = FlexFloat::from_int(trunc.clone());
    if value_trunc == value {
        return value;
    }

    if !value.sign {
        // If positive and not integral, floor = trunc (truncation toward zero is already the flooring)
        value_trunc
    } else {
        FlexFloat::from_int(trunc - 1)
    }
}

fn ceil<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    if value.is_nan() || value.is_infinity() || value.is_zero() {
        return value;
    }

    let trunc = value.clone().to_int().expect("Handled NaN/Inf above");
    let value_trunc: FlexFloat<B> = FlexFloat::from_int(trunc.clone());
    if value_trunc == value {
        return value;
    }

    if !value.sign {
        FlexFloat::from_int(trunc + 1)
    } else {
        // If negative and not integral, ceil = trunc (truncation toward zero is already the ceiling)
        value_trunc
    }
}

pub fn exp<B: BitArray + Clone>(value: FlexFloat<B>) -> FlexFloat<B> {
    if value.is_infinity() || value.is_nan() {
        if value.is_infinity() && value.sign {
            return FlexFloat::new_zero();
        }
        return value;
    }

    if value.is_zero() {
        return FlexFloat::from_f64(1.0);
    }

    let ln2 = FlexFloat::from_f64(f64::consts::LN_2);
    let tolerance: FlexFloat<B> = FlexFloat::from_f64(1e-16);
    // Range reduction:
    // exp(x) = exp(k * ln(2) + r) = 2^k * exp(r)
    // where k = round(x / ln(2)) and r = x - k * ln(2)

    // Polynomial approximation for exp(r) where r is in [-ln(2)/2, ln(2)/2]
    // Using a Taylor series expansion or minimax polynomial could be options here.
    // For now, we'll use a simple Taylor series expansion.
    // exp(r) ≈ 1 + r + r^2/2! + r^3/3! + r^4/4! (up to desired precision)

    let k = round(value.clone() / ln2.clone());
    let r = value - k.clone() * ln2;

    // First 2 iterations Taylor series
    let mut exp_r = FlexFloat::from_f64(1.0) + r.clone();
    let mut term: FlexFloat<B> = r.clone();

    // term[n+1] = term[n] * x / (n+1)
    for n in 2..=10 {
        term = term * r.clone() / FlexFloat::from_f64(n as f64);
        exp_r = exp_r + term.clone();

        if term.abs() < tolerance {
            break;
        }
    }

    fast_exp2(k) * exp_r
}

// Assumes Integer
fn fast_exp2<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    let value = value.to_int().unwrap() - 1;

    let exponent = grow_exponent(value, 11);
    FlexFloat::new(false, exponent, B::zeros(52) + B::ones(1))
}

// https://libc.llvm.org/headers/math/log.html
pub fn ln<B: BitArray + Clone>(value: FlexFloat<B>) -> FlexFloat<B> {
    // Handle special cases: negative numbers, zero, NaN, Infinity
    if value.is_nan() || value.is_infinity() {
        return value;
    }
    if value.sign {
        // Negative numbers return NaN
        return FlexFloat::new_nan();
    }
    if value.is_zero() {
        // ln(0) = -infinity
        return FlexFloat::new_infinity(true);
    }
    if value == FlexFloat::from(1.0) {
        return FlexFloat::new_zero();
    }

    // Range reduction:
    // ln(x) = ln(2^k * m) = k*ln(2) + ln(m)
    // where k is the exponent and m is the mantissa (1 <= m < 2)

    // Compute ln(m) using the more efficient series:
    // ln(x) = 2 * (t + t^3/3 + t^5/5 + t^7/7 + ...)
    // where t = (x - 1) / (x + 1)
    // This converges much faster than ln(1+x) series

    let ln2 = FlexFloat::from_f64(f64::consts::LN_2);
    let tolerance: FlexFloat<B> = FlexFloat::from_f64(1e-16);
    let one = FlexFloat::from_f64(1.0);
    let two = FlexFloat::from_f64(2.0);

    // Get the exponent value as BigInt, add 1 for the actual exponent
    let k_int = value.exponent.to_bigint() + 1;
    let k = FlexFloat::from_int(k_int);

    // Normalize to get mantissa (1 <= m < 2)
    // m = value / 2^k
    let m = value.clone() / fast_exp2(k.clone());

    let mut t = (m.clone() - one.clone()) / (m.clone() + one.clone());
    let t_squared = t.clone() * t.clone();

    let mut ln_m = FlexFloat::new_zero();
    for n in 1..=10 {
        ln_m = ln_m.clone() + t.clone() / FlexFloat::from_f64((2 * n - 1) as f64);
        t = t * t_squared.clone();

        if t.abs() < tolerance {
            break;
        }
    }

    ln_m = ln_m * two;

    // Result: k*ln(2) + ln(m)
    k * ln2 + ln_m
}

impl<B: BitArray + Clone> FlexFloat<B> {
    pub fn round(self) -> Self {
        round(self)
    }

    pub fn ceil(self) -> Self {
        ceil(self)
    }

    pub fn floor(self) -> Self {
        floor(self)
    }

    pub fn exp(self) -> Self {
        exp(self)
    }

    pub fn ln(self) -> Self {
        ln(self)
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::tests::*;

    #[rstest]
    fn test_ceil(mut rng: impl Rng, n_experiments: usize) {
        for _ in 0..n_experiments {
            let value = random_f64(&mut rng);
            let ff = FlexFloat::from(value);
            let rounded_ff = ff.ceil();
            let expected = value.ceil();
            let converted_result: f64 = rounded_ff.into();
            assert_almost_eq(converted_result, expected, &format!("ceil({value:#?})"));
        }
    }

    #[rstest]
    fn test_floor(mut rng: impl Rng, n_experiments: usize) {
        for _ in 0..n_experiments {
            let value = random_f64(&mut rng);
            let ff = FlexFloat::from(value);
            let rounded_ff = ff.floor();
            let expected = value.floor();
            let converted_result: f64 = rounded_ff.into();
            assert_almost_eq(converted_result, expected, &format!("floor({value:#?})"));
        }
    }

    #[rstest]
    fn test_round(mut rng: impl Rng, n_experiments: usize) {
        for _ in 0..n_experiments {
            let value = random_f64(&mut rng);
            let ff = FlexFloat::from(value);
            let rounded_ff = ff.round();
            let expected = value.round();
            let converted_result: f64 = rounded_ff.into();
            assert_almost_eq(converted_result, expected, &format!("round({value:#?})"));
        }
    }

    #[rstest]
    fn test_exp(mut rng: impl Rng, n_experiments: usize) {
        let special_values = [0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];

        for value in special_values {
            let ff = FlexFloat::from(value);
            let result = ff.exp();
            let expected = value.exp();
            let converted_result: f64 = result.into();

            if expected.is_nan() {
                assert!(converted_result.is_nan());
            } else if expected.is_infinite() {
                assert!(converted_result.is_infinite());
            } else {
                assert_almost_eq(converted_result, expected, &format!("exp({})", value));
            }
        }

        let value = 1e200;
        let ff = FlexFloat::from(value);
        let result = ff.exp();
        assert!(!result.is_infinity(), "Result should not overflow");
        assert!(result.exponent.len() > 11, "Exponent should have grown");

        for _ in 0..(n_experiments / 10) {
            let value = random_f64(&mut rng);
            let ff = FlexFloat::from(value);
            let result = ff.exp();
            let expected = value.exp();

            if let Some(result) = result.to_f64() {
                assert_almost_eq(
                    result,
                    expected,
                    format!("Failed on exp({value:?})").as_str(),
                );
            } else {
                // TODO: Check is valid
            }
        }
    }

    #[rstest]
    fn test_ln(mut rng: impl Rng, n_experiments: usize) {
        let special_values = [1.0, f64::consts::E, 0.0, -1.0, f64::INFINITY, f64::NAN];
        for value in special_values {
            let ff = FlexFloat::from(value);
            let result = ff.ln();
            let expected = value.ln();
            let converted_result: f64 = result.into();
            if expected.is_nan() {
                assert!(converted_result.is_nan());
            } else if expected.is_infinite() {
                assert!(converted_result.is_infinite());
                assert_eq!(
                    converted_result.is_sign_positive(),
                    expected.is_sign_positive()
                );
            } else {
                assert_almost_eq(converted_result, expected, &format!("ln({})", value));
            }
        }

        for _ in 0..(n_experiments / 10) {
            let value = random_f64(&mut rng);
            // Ensure the value is positive to avoid domain errors
            let value = value.abs() + 1e-10;

            let ff = FlexFloat::from(value);
            let result = ff.ln();
            let expected = value.ln();
            let converted_result: f64 = result.into();
            assert_almost_eq(converted_result, expected, &format!("ln({})", value));
        }
    }
}
