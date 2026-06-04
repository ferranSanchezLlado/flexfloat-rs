//! Multiplication and division kernels.

use core::cmp::max;
use core::ops::{Div, Mul};

use num_bigint::BigInt;

use crate::bitarray::traits::{BitArrayRounding, ShiftRoundingInfo};
use crate::bitarray::{BitArrayArith, BitArrayConversion};
use crate::flexfloat::FlexFloat;
use crate::flexfloat::arithmetic::normalize::{
    build_finite_result, ensure_width, extract_mantissa_and_exponent,
};

/// Multiplication operation for `FlexFloat`.
///
/// Implements the [`Mul`] trait for FlexFloat, performing floating-point multiplication
/// with automatic exponent growth and precision preservation.
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// let width = FlexFloat::from(8.5);
/// let height = FlexFloat::from(11.0);
/// let area = width * height;
/// assert_ff_almost_eq!(area, FlexFloat::from(93.5));
/// ```
impl<E1: BitArrayArith, F1: BitArrayArith, E2: BitArrayConversion, F2: BitArrayConversion>
    Mul<FlexFloat<E2, F2>> for FlexFloat<E1, F1>
{
    type Output = Self;

    fn mul(self, rhs: FlexFloat<E2, F2>) -> Self::Output {
        let rhs: FlexFloat<E1, F1> = rhs.convert_to();
        // 0. Handle special cases (NaN, Infinity).
        if self.is_nan() || rhs.is_nan() {
            return Self::nan();
        }

        if (self.is_infinite() && rhs.is_zero()) || (self.is_zero() && rhs.is_infinite()) {
            return Self::nan();
        }

        let sign = self.sign ^ rhs.sign;
        if self.is_zero() || rhs.is_zero() {
            return Self::zero_with_sign(sign);
        }

        if self.is_infinite() || rhs.is_infinite() {
            return Self::infinity(sign);
        }

        let (mant_self, exp_self) = extract_mantissa_and_exponent(&self);
        let (mant_rhs, exp_rhs) = extract_mantissa_and_exponent(&rhs);
        let mut exp_res = exp_self + exp_rhs;

        let mut mant_res = mant_self * mant_rhs;

        let msb_pos = mant_res.iter_bits().rposition(|b| b).unwrap();

        let shift_amount = (msb_pos as isize) - 52;
        exp_res += BigInt::from(msb_pos) - 104;

        let mul_round_info = if shift_amount > 0 {
            let r = mant_res.shift_right_rounded(shift_amount as usize);
            mant_res = r.value;
            r.info
        } else if shift_amount < 0 {
            mant_res = mant_res.shift_fixed(shift_amount);
            ShiftRoundingInfo::default()
        } else {
            ShiftRoundingInfo::default()
        };

        let mut mant_res_53 = ensure_width(mant_res, 53).truncate(53);

        let result_lsb = mant_res_53.get(0).unwrap_or(false);
        if mul_round_info.should_round_up(result_lsb) {
            mant_res_53.add_one_in_place();
            if mant_res_53.len() > 53 {
                mant_res_53 = mant_res_53.shift_fixed(1).truncate(53);
                exp_res += 1_u8;
            }
        }

        // 7. Grow exponent if necessary (no limit on size)
        let exp_res = exp_res - 1_u8;
        let max_exp_len = max(self.exponent.len(), rhs.exponent.len());
        build_finite_result(sign, exp_res, max_exp_len, mant_res_53)
    }
}

/// Division operation for `FlexFloat`.
///
/// Implements the [`Div`] trait for FlexFloat, performing floating-point division
/// with automatic exponent growth and precision preservation.
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// let budget = FlexFloat::from(144.0);
/// let people = FlexFloat::from(6.0);
/// let share = budget / people;
/// assert_ff_almost_eq!(share, FlexFloat::from(24.0));
/// ```
impl<E1: BitArrayArith, F1: BitArrayArith, E2: BitArrayConversion, F2: BitArrayConversion>
    Div<FlexFloat<E2, F2>> for FlexFloat<E1, F1>
{
    type Output = Self;

    fn div(self, rhs: FlexFloat<E2, F2>) -> Self::Output {
        let rhs: FlexFloat<E1, F1> = rhs.convert_to();
        // 0. Handle special cases (NaN, Infinity).
        if self.is_nan() || rhs.is_nan() {
            return Self::nan();
        }

        if rhs.is_zero() {
            if self.is_zero() {
                return Self::nan(); // 0 / 0 = NaN
            }
            return Self::infinity(self.sign ^ rhs.sign); // x / 0 = inf
        }

        let sign = self.sign ^ rhs.sign;
        if self.is_zero() {
            return Self::zero_with_sign(sign);
        }

        if self.is_infinite() {
            if rhs.is_infinite() {
                return Self::nan(); // inf / inf = NaN
            }
            return Self::infinity(sign);
        }

        if rhs.is_infinite() {
            return Self::zero_with_sign(sign); // x / inf = 0
        }

        let (mant_self, exp_self) = extract_mantissa_and_exponent(&self);
        let (mant_rhs, exp_rhs) = extract_mantissa_and_exponent(&rhs);
        // Compute a few extra quotient bits so normalization can still observe
        // guard/round/sticky information instead of rounding a pre-truncated
        // integer quotient.
        const DIV_EXTRA_BITS: isize = 3;
        let mant_self_msb = mant_self.iter_bits().rposition(|b| b).unwrap();
        let mant_rhs_msb = mant_rhs.iter_bits().rposition(|b| b).unwrap();
        let quotient_shift = 52 + DIV_EXTRA_BITS + mant_rhs_msb as isize - mant_self_msb as isize;
        let mut exp_res = exp_self - exp_rhs - BigInt::from(quotient_shift - 52);
        let mut mant_res = mant_self.shift_grow(quotient_shift) / mant_rhs;

        // If result is zero, return signed zero.
        if mant_res.is_zeros() {
            return Self::zero_with_sign(sign);
        }

        // Find MSB position and normalize so the MSB sits at index 52.
        let msb_pos = mant_res.iter_bits().rposition(|b| b).unwrap();
        let shift = msb_pos as isize - 52;
        exp_res += BigInt::from(shift);

        let div_round_info = if shift > 0 {
            let r = mant_res.shift_right_rounded(shift as usize);
            mant_res = r.value;
            r.info
        } else if shift < 0 {
            mant_res = mant_res.shift_fixed(shift);
            ShiftRoundingInfo::default()
        } else {
            ShiftRoundingInfo::default()
        };

        let mut mant_res_final = ensure_width(mant_res, 53).truncate(53);

        let result_lsb = mant_res_final.get(0).unwrap_or(false);
        if div_round_info.should_round_up(result_lsb) {
            mant_res_final.add_one_in_place();
            if mant_res_final.len() > 53 {
                mant_res_final = mant_res_final.shift_fixed(1).truncate(53);
                exp_res += 1_u8;
            }
        }

        // 7. Grow exponent if necessary (no limit on size)
        let exp_res = exp_res - 1_u8;
        let max_exp_len = max(self.exponent.len(), rhs.exponent.len());
        build_finite_result(sign, exp_res, max_exp_len, mant_res_final)
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use crate::bitarray::BitArrayAccess;
    use crate::bitarray::DefaultBitArray;
    use crate::flexfloat::FlexFloat;
    use crate::test_support::*;

    #[rstest]
    fn test_mul(mut rng: impl Rng, n_experiments: usize) {
        let a = 1.5;
        let b = 2.25;
        let c = FlexFloat::from(a) * FlexFloat::from(b);
        assert_eq!(c.to_f64(), Some(a * b));

        let a = 0.123;
        let b = 0.321;
        let c = FlexFloat::from(a) * FlexFloat::from(b);
        assert_eq!(c.to_f64(), Some(a * b));

        // Test overflow case
        let a = f64::MAX;
        let b = 100.0;
        let c = FlexFloat::from(a) * FlexFloat::from(b);
        assert!(!c.is_infinite(), "Result should not overflow");
        assert!(c.exponent.len() > 11, "Exponent should have grown");

        let a = -5.18059e300;
        let b = 1.97397e-308;
        let c = FlexFloat::from(a) * FlexFloat::from(b);
        let expected = a * b;
        let result = c.to_f64().expect("result must be representable as f64");
        let rel = (result - expected).abs() / expected.abs().max(1e-300);
        assert!(
            rel < 1e-12,
            "edge case mismatch: result={result:?} expected={expected:?}"
        );

        assert!(
            (FlexFloat::<DefaultBitArray>::zero() * FlexFloat::<DefaultBitArray>::pos_infinity())
                .is_nan()
        );
        assert!(
            (FlexFloat::<DefaultBitArray>::pos_infinity() * FlexFloat::from(-2.0)).is_infinite()
        );
        let signed_zero = (FlexFloat::<DefaultBitArray>::zero_with_sign(true)
            * FlexFloat::from(-2.0))
        .to_f64()
        .unwrap();
        assert_eq!(signed_zero.to_bits(), 0.0f64.to_bits());

        test_binary_flexfloat_op(
            &mut rng,
            n_experiments,
            |a, b| a * b,
            |a: f64, b: f64| a * b,
            "mul",
            identity_2,
            None,
        );
    }

    #[rstest]
    fn test_div(mut rng: impl Rng, n_experiments: usize) {
        let a = 1.5;
        let b = 2.25;
        let c = FlexFloat::from(a) / FlexFloat::from(b);
        assert_eq!(c.to_f64(), Some(a / b));

        let a = 0.123;
        let b = 0.321;
        let c = FlexFloat::from(a) / FlexFloat::from(b);
        assert_almost_eq(c.to_f64().unwrap(), a / b, "");

        // Test overflow case
        let a = f64::MAX;
        let b = 1e-100;
        let c = FlexFloat::from(a) / FlexFloat::from(b);
        assert!(!c.is_infinite(), "Result should not overflow");
        assert!(c.exponent.len() > 11, "Exponent should have grown");

        let a = f64::from_bits(1);
        let b = 1e-300_f64;
        let result = (FlexFloat::<DefaultBitArray>::from(a)
            / FlexFloat::<DefaultBitArray>::from(b))
        .to_f64()
        .unwrap();
        assert_eq!(result.to_bits(), (a / b).to_bits());

        let pos_inf = (FlexFloat::<DefaultBitArray>::from(1.0)
            / FlexFloat::<DefaultBitArray>::zero())
        .to_f64()
        .unwrap();
        assert_eq!(pos_inf.to_bits(), f64::INFINITY.to_bits());
        assert!(
            (FlexFloat::<DefaultBitArray>::zero() / FlexFloat::<DefaultBitArray>::zero()).is_nan()
        );
        assert!(
            (FlexFloat::<DefaultBitArray>::pos_infinity()
                / FlexFloat::<DefaultBitArray>::pos_infinity())
            .is_nan()
        );

        let signed_zero = (FlexFloat::<DefaultBitArray>::zero_with_sign(true)
            / FlexFloat::from(2.0))
        .to_f64()
        .unwrap();
        assert_eq!(signed_zero.to_bits(), (-0.0f64).to_bits());

        test_binary_flexfloat_op(
            &mut rng,
            n_experiments,
            |a, b| a / b,
            |a: f64, b: f64| a / b,
            "div",
            identity_2,
            None,
        );
    }

    #[test]
    fn test_mul_round_to_nearest() {
        let a = 1.0_f64 + 2.0_f64.powi(-40);
        let b = 1.0_f64 + 2.0_f64.powi(-40);
        let ff_a = FlexFloat::<DefaultBitArray>::from(a);
        let ff_b = FlexFloat::<DefaultBitArray>::from(b);
        let result = (ff_a * ff_b).to_f64().unwrap();
        let expected = a * b;
        assert_eq!(
            result.to_bits(),
            expected.to_bits(),
            "result={result:?} expected={expected:?}"
        );
    }

    #[test]
    fn test_div_round_to_nearest() {
        let a = 1.0_f64;
        let b = 7.0_f64;
        let ff_a = FlexFloat::<DefaultBitArray>::from(a);
        let ff_b = FlexFloat::<DefaultBitArray>::from(b);
        let result = (ff_a / ff_b).to_f64().unwrap();
        let expected = a / b;
        assert_eq!(
            result.to_bits(),
            expected.to_bits(),
            "result={result:?} expected={expected:?}"
        );
    }

    /// Regression test for the Div extraction bug (Pattern D).
    /// After normalization, the MSB sits at index 52 by construction.
    /// Extraction must use the constant range 0..53, not the stale msb_pos.
    #[rstest]
    fn test_div_normalization_extracts_correct_range(_rng: impl Rng, _n_experiments: usize) {
        let cases: &[(f64, f64)] = &[
            (1.0, 7.0),
            (1.0, 3.0),
            (1.7e10, 4.2e-10),
            (-3.5e100, 2.0e50),
            (5.0e-200, 7.0e100),
        ];
        for &(a, b) in cases {
            let ff_a = FlexFloat::from(a);
            let ff_b = FlexFloat::from(b);
            let result = (ff_a / ff_b).to_f64().unwrap();
            let expected = a / b;
            assert_eq!(
                result.to_bits(),
                expected.to_bits(),
                "div_normalization({a:?},{b:?}) result={result:?} expected={expected:?}",
            );
        }
    }

    #[rstest]
    fn test_subnormal_mul_randomized(mut rng: impl Rng, n_experiments: usize) {
        for _ in 0..(n_experiments / 50) {
            let tiny_a = f64::from_bits(rng.random_range(1..=2048));

            test_common_logic(
                &format!("subnormal_mul({tiny_a:?})"),
                FlexFloat::<DefaultBitArray>::from(tiny_a)
                    * FlexFloat::<DefaultBitArray>::from(2.0),
                tiny_a * 2.0,
                Some(1e-12),
            );
        }
    }

    #[rstest]
    fn test_subnormal_div_randomized(mut rng: impl Rng, n_experiments: usize) {
        for _ in 0..(n_experiments / 50) {
            let tiny_a = f64::from_bits(rng.random_range(1..=2048));
            let tiny_b = f64::from_bits(rng.random_range(1..=2048));
            let divisor = tiny_b.max(f64::from_bits(1));

            test_common_logic(
                &format!("subnormal_div({tiny_a:?}, {divisor:?})"),
                FlexFloat::<DefaultBitArray>::from(tiny_a)
                    / FlexFloat::<DefaultBitArray>::from(divisor),
                tiny_a / divisor,
                Some(1e-12),
            );
        }
    }
}
