//! Multiplication and division kernels.
//!
//! Both kernels widen operand mantissas to a common `mant_len` (the larger of the two
//! operand widths) via [`align_mantissa`] before computing, so grown-format values
//! retain their extra precision throughout.  The result is assembled by
//! [`build_finite_result`], which may further grow the exponent and fraction fields if
//! the output exceeds the IEEE 754 double-precision exponent range.

use core::cmp::max;
use core::ops::{Div, Mul};

use num_bigint::BigInt;

use crate::bitarray::traits::{BitArrayRounding, ShiftRoundingInfo};
use crate::bitarray::{BitArrayArith, BitArrayConversion};
use crate::flexfloat::FlexFloat;
use crate::flexfloat::arithmetic::normalize::{
    align_mantissa, build_finite_result, ensure_width, extract_mantissa_and_exponent,
};

/// Multiplication operation for `FlexFloat`.
///
/// Implements the [`Mul`] trait for FlexFloat, performing floating-point multiplication
/// with automatic exponent growth and precision preservation.  Both operands are
/// widened to a common mantissa length before multiplying, so grown-format values
/// (with more than 52 fraction bits) are not silently truncated.
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
        // Handle special cases (NaN, Infinity).
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

        // Work at the precision of the larger operand.  Both mantissas are
        // left-aligned so their implicit leading 1 sits at bit `mant_len - 1`.
        let mant_len = max(mant_self.len(), mant_rhs.len());
        let msb_idx = mant_len - 1; // index of the implicit leading 1
        let mant_self = align_mantissa(mant_self, mant_len);
        let mant_rhs = align_mantissa(mant_rhs, mant_len);

        let mut exp_res = exp_self + exp_rhs;
        let mut mant_res = mant_self * mant_rhs;

        // The product of two `mant_len`-bit numbers has up to `2*mant_len` bits.
        // Normalise so the MSB sits at `msb_idx` and adjust the exponent accordingly.
        let product_msb = mant_res.iter_bits().rposition(|b| b).unwrap();
        exp_res += BigInt::from(product_msb) - BigInt::from(2 * msb_idx);
        let shift = product_msb as isize - msb_idx as isize;

        let mul_round_info = if shift > 0 {
            let r = mant_res.shift_right_rounded(shift as usize);
            mant_res = r.value;
            r.info
        } else {
            if shift < 0 {
                mant_res = mant_res.shift_fixed(shift);
            }
            ShiftRoundingInfo::default()
        };

        let mut mant_res_norm = ensure_width(mant_res, mant_len).truncate(mant_len);

        let result_lsb = mant_res_norm.get(0).unwrap_or(false);
        if mul_round_info.should_round_up(result_lsb) {
            mant_res_norm.add_one_in_place();
            if mant_res_norm.len() > mant_len {
                mant_res_norm = mant_res_norm.shift_fixed(1).truncate(mant_len);
                exp_res += 1_u8;
            }
        }

        let max_exp_len = max(self.exponent.len(), rhs.exponent.len());
        build_finite_result(sign, exp_res - 1_u8, max_exp_len, mant_res_norm)
    }
}

/// Division operation for `FlexFloat`.
///
/// Implements the [`Div`] trait for FlexFloat, performing floating-point division
/// with automatic exponent growth and precision preservation.  Both operands are
/// widened to a common mantissa length before dividing, so grown-format values
/// retain their extra precision.  Three extra guard bits are carried through the
/// integer division to allow correct round-to-nearest-even.
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
        // Handle special cases (NaN, Infinity, zero denominator).
        if self.is_nan() || rhs.is_nan() {
            return Self::nan();
        }

        if rhs.is_zero() {
            if self.is_zero() {
                return Self::nan(); // 0 / 0 = NaN
            }
            return Self::infinity(self.sign ^ rhs.sign); // x / 0 = ±∞
        }

        let sign = self.sign ^ rhs.sign;
        if self.is_zero() {
            return Self::zero_with_sign(sign);
        }

        if self.is_infinite() {
            if rhs.is_infinite() {
                return Self::nan(); // ∞ / ∞ = NaN
            }
            return Self::infinity(sign);
        }

        if rhs.is_infinite() {
            return Self::zero_with_sign(sign); // x / ∞ = 0
        }

        let (mant_self, exp_self) = extract_mantissa_and_exponent(&self);
        let (mant_rhs, exp_rhs) = extract_mantissa_and_exponent(&rhs);

        // Work at the precision of the larger operand.  Both mantissas are
        // left-aligned so their implicit leading 1 sits at bit `mant_len - 1`.
        let mant_len = max(mant_self.len(), mant_rhs.len());
        let msb_idx = mant_len - 1; // index of the implicit leading 1
        let mant_self = align_mantissa(mant_self, mant_len);
        let mant_rhs = align_mantissa(mant_rhs, mant_len);

        // Pre-shift the numerator so the integer quotient carries a few extra
        // guard/round/sticky bits for correct rounding.
        const DIV_EXTRA_BITS: isize = 3;
        let self_msb = mant_self.iter_bits().rposition(|b| b).unwrap();
        let rhs_msb = mant_rhs.iter_bits().rposition(|b| b).unwrap();
        let quotient_shift =
            msb_idx as isize + DIV_EXTRA_BITS + rhs_msb as isize - self_msb as isize;
        let mut exp_res = exp_self - exp_rhs - BigInt::from(quotient_shift - msb_idx as isize);
        let mut mant_res = mant_self.shift_grow(quotient_shift) / mant_rhs;

        if mant_res.is_zeros() {
            return Self::zero_with_sign(sign);
        }

        // Normalise: bring the MSB to `msb_idx`.
        let product_msb = mant_res.iter_bits().rposition(|b| b).unwrap();
        let shift = product_msb as isize - msb_idx as isize;
        exp_res += BigInt::from(shift);

        let div_round_info = if shift > 0 {
            let r = mant_res.shift_right_rounded(shift as usize);
            mant_res = r.value;
            r.info
        } else {
            if shift < 0 {
                // Widen before shifting left so no bits are clipped.
                mant_res = ensure_width(mant_res, mant_len).shift_fixed(shift);
            }
            ShiftRoundingInfo::default()
        };

        let mut mant_res_final = ensure_width(mant_res, mant_len).truncate(mant_len);

        let result_lsb = mant_res_final.get(0).unwrap_or(false);
        if div_round_info.should_round_up(result_lsb) {
            mant_res_final.add_one_in_place();
            if mant_res_final.len() > mant_len {
                mant_res_final = mant_res_final.shift_fixed(1).truncate(mant_len);
                exp_res += 1_u8;
            }
        }

        let max_exp_len = max(self.exponent.len(), rhs.exponent.len());
        build_finite_result(sign, exp_res - 1_u8, max_exp_len, mant_res_final)
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

        // Test overflow case: exponent AND fraction must grow together.
        // e=12 => best_n=128 => target_frac = 128 - 12 - 1 = 115 bits.
        let a = f64::MAX;
        let b = 100.0;
        let c = FlexFloat::from(a) * FlexFloat::from(b);
        assert!(!c.is_infinite(), "Result should not overflow");
        assert_eq!(
            c.exponent.len(),
            12,
            "Exponent should have grown to 12 bits"
        );
        assert_eq!(
            c.fraction.len(),
            115,
            "Fraction should have grown to 115 bits alongside exponent"
        );

        // Precision round-trip: (f64::MAX * 100) / 100 ≈ f64::MAX within 1 ULP.
        let roundtrip = c / FlexFloat::from(100.0_f64);
        let rt_f64 = roundtrip.to_f64().expect("round-trip must be in f64 range");
        let rel = (rt_f64 - f64::MAX).abs() / f64::MAX;
        assert!(rel < 1e-14, "round-trip lost precision: rel={rel:.2e}");

        // grown * normal: fraction must remain grown (115 bits).
        let grown = FlexFloat::from(f64::MAX) * FlexFloat::from(100.0_f64);
        let grown_times_two = grown * FlexFloat::from(2.0_f64);
        assert_eq!(
            grown_times_two.fraction.len(),
            115,
            "grown * normal fraction should stay 115 bits"
        );

        // grown / normal: fraction must remain grown.
        let grown = FlexFloat::from(f64::MAX) * FlexFloat::from(100.0_f64);
        let grown_div = grown / FlexFloat::from(3.0_f64);
        assert_eq!(
            grown_div.fraction.len(),
            115,
            "grown / normal fraction should stay 115 bits"
        );

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

        // Test overflow case: exponent AND fraction must grow together.
        let a = f64::MAX;
        let b = 1e-100;
        let c = FlexFloat::from(a) / FlexFloat::from(b);
        assert!(!c.is_infinite(), "Result should not overflow");
        assert_eq!(
            c.exponent.len(),
            12,
            "Exponent should have grown to 12 bits"
        );
        assert_eq!(
            c.fraction.len(),
            115,
            "Fraction should have grown to 115 bits"
        );

        // Precision round-trip: (f64::MAX / 1e-100) * 1e-100 ≈ f64::MAX.
        let roundtrip = c / FlexFloat::from(1.0_f64 / 1e-100_f64);
        let rt_f64 = roundtrip.to_f64().expect("round-trip must be in f64 range");
        let rel = (rt_f64 - f64::MAX).abs() / f64::MAX;
        assert!(rel < 1e-14, "div round-trip lost precision: rel={rel:.2e}");

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
