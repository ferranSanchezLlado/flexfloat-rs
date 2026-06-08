//! Addition and subtraction kernels.
//!
//! Both operations share the same sign-dispatch pattern:
//!
//! - [`add`] checks whether the operand signs differ and, if so, delegates to [`sub`].
//! - [`sub`] checks whether the operand signs differ and, if so, delegates to [`add`].
//!
//! After sign normalisation each kernel handles IEEE 754 special-value cases (NaN,
//! infinity), then proceeds to the finite path:
//!
//! **Addition** (`add`):
//! 1. Extract normalised mantissas and unbiased exponents via [`extract_mantissa_and_exponent`].
//! 2. Widen both mantissas to the same bit-length with [`align_mantissa`].
//! 3. Swap operands if necessary so `lhs` carries the larger exponent.
//! 4. Right-shift the smaller mantissa by the exponent difference, collecting
//!    guard/round/sticky bits for later rounding.
//! 5. Add the aligned mantissas; repair the width with [`ensure_width`].
//! 6. If the sum overflowed by one bit, right-shift by 1 and increment the exponent.
//! 7. Apply round-to-nearest-even using the accumulated rounding bits.
//! 8. Assemble the result with [`build_finite_result`], which handles exponent growth
//!    and subnormal encoding automatically.
//!
//! **Subtraction** (`sub`):
//! 1–2. Same extraction and widening as addition.
//! 3. If `|lhs| < |rhs|`, delegate to `sub(&mut rhs, lhs)` and negate.
//! 4. Shift the smaller mantissa by the (possibly negative) exponent difference.
//! 5. Extend both mantissas by one guard bit and subtract.
//! 6. Normalise: left- or right-shift the difference to restore the leading 1,
//!    adjusting the exponent accordingly.  A zero result returns early.
//! 7. Drop the guard column and round using the residual bits from step 4.
//! 8. Assemble with [`build_finite_result`].

use core::cmp::max;
use core::ops::{Add, Sub};

use num_bigint::BigInt;

use crate::bitarray::traits::{BitArrayRounding, ShiftRoundingInfo, ShiftRoundingResult};
use crate::bitarray::{BitArrayArith, BitArrayConversion};
use crate::flexfloat::FlexFloat;
use crate::flexfloat::arithmetic::normalize::{
    align_mantissa, build_finite_result, ensure_width, extract_mantissa_and_exponent,
};

pub(super) fn add<Exp: BitArrayArith, Frac: BitArrayArith>(
    lhs: &mut FlexFloat<Exp, Frac>,
    mut rhs: FlexFloat<Exp, Frac>,
) {
    if lhs.sign != rhs.sign {
        sub(lhs, -rhs);
        return;
    }

    // Handle special cases (NaN, Infinity).
    // References:
    //   https://www.sciencedirect.com/topics/computer-science/floating-point-addition
    //   https://cse.hkust.edu.hk/~cktang/cs180/notes/lec21.pdf
    match (lhs.is_nan(), rhs.is_nan()) {
        (true, _) => return,
        (_, true) => return *lhs = rhs,
        _ => {}
    }

    match (lhs.is_infinite(), rhs.is_infinite(), lhs.sign == rhs.sign) {
        (true, true, true) => return,                          // ∞ + ∞ = ∞
        (true, true, false) => return *lhs = FlexFloat::nan(), // ∞ + -∞ = NaN
        (true, false, _) => return,                            // ∞ + x = ∞
        (false, true, _) => return *lhs = rhs,                 // x + ∞ = ∞
        _ => {}
    }

    // Extract mantissas and align to the same bit-width, with the implicit
    // leading 1 at the MSB (bit `mant_len - 1`).
    let (lhs_mantissa, mut exp_lhs) = extract_mantissa_and_exponent(lhs);
    let (rhs_mantissa, mut exp_rhs) = extract_mantissa_and_exponent(&rhs);

    let mant_len = max(lhs_mantissa.len(), rhs_mantissa.len());
    let mut mant_lhs = align_mantissa(lhs_mantissa, mant_len);
    let mut mant_rhs_aligned = align_mantissa(rhs_mantissa, mant_len);

    // Ensure lhs carries the larger exponent so we always shift rhs right.
    if exp_lhs < exp_rhs {
        core::mem::swap(&mut mant_lhs, &mut mant_rhs_aligned);
        core::mem::swap(&mut exp_lhs, &mut exp_rhs);
        // Keep `lhs` metadata (sign, exponent width) pointing at the larger operand.
        core::mem::swap(lhs, &mut rhs);
    }
    debug_assert!(exp_lhs >= exp_rhs, "lhs exponent must be >= rhs after swap");

    // Align rhs by shifting right by the exponent difference.
    let exp_diff = &exp_lhs - &exp_rhs;
    let saturation = mant_len + 64;
    let shift: usize = exp_diff.try_into().unwrap_or(saturation);
    let ShiftRoundingResult {
        value: mant_rhs,
        info: align_info,
    } = mant_rhs_aligned.shift_right_rounded(shift);

    // Add mantissas.  BitArray addition may shrink the array if the MSB is zero;
    // `ensure_width` restores the expected width.
    let mut mantissa_result = ensure_width(mant_lhs + mant_rhs, mant_len);

    // If the sum overflowed by 1 bit, right-shift and increment the exponent.
    let mut rounding_info = align_info;
    if mantissa_result.len() > mant_len {
        let ShiftRoundingResult {
            value: shifted,
            info: norm_info,
        } = mantissa_result.shift_right_rounded(1);
        mantissa_result = shifted.truncate(mant_len);
        exp_lhs += 1_u8;
        rounding_info = rounding_info.combine(norm_info);
    }

    let result_lsb = mantissa_result.get(0).unwrap_or(false);
    if rounding_info.should_round_up(result_lsb) {
        mantissa_result.add_one_in_place();
        if mantissa_result.len() > mant_len {
            mantissa_result = mantissa_result.shift_fixed(1).truncate(mant_len);
            exp_lhs += 1_u8;
        }
    }

    *lhs = build_finite_result(
        lhs.sign,
        exp_lhs - 1_u8,
        lhs.exponent.len(),
        mantissa_result,
    );
}

pub(super) fn sub<Exp: BitArrayArith, Frac: BitArrayArith>(
    lhs: &mut FlexFloat<Exp, Frac>,
    rhs: FlexFloat<Exp, Frac>,
) {
    if lhs.sign != rhs.sign {
        // a - (-b) == a + b
        add(lhs, -rhs);
        return;
    }

    // Handle special cases (NaN, Infinity).
    match (lhs.is_nan(), rhs.is_nan()) {
        (true, _) => return,
        (_, true) => return *lhs = rhs,
        _ => {}
    }

    match (lhs.is_infinite(), rhs.is_infinite(), lhs.sign == rhs.sign) {
        (true, true, true) => return *lhs = FlexFloat::nan(), // ∞ - ∞ = NaN
        (true, true, false) => return,                        // ∞ - -∞ = ∞
        (true, false, _) => return,                           // ∞ - x = ∞
        (false, true, _) => return *lhs = -rhs,               // x - ∞ = -∞
        _ => {}
    }

    // Guarantee |lhs| >= |rhs| by delegating the reversed case.
    if lhs.abs() < rhs.abs() {
        let mut tmp = rhs;
        sub(&mut tmp, lhs.clone());
        *lhs = -tmp;
        return;
    }

    let (lhs_mantissa, mut exp_lhs) = extract_mantissa_and_exponent(lhs);
    let (rhs_mantissa, exp_rhs) = extract_mantissa_and_exponent(&rhs);

    let mant_len = max(lhs_mantissa.len(), rhs_mantissa.len());
    let mant_lhs = align_mantissa(lhs_mantissa, mant_len);
    let mant_rhs_aligned = align_mantissa(rhs_mantissa, mant_len);

    // Align rhs by the exponent difference (may be negative if rhs has larger exponent,
    // but the |lhs| >= |rhs| guarantee above bounds how large the shift can be).
    let exp_diff = exp_lhs.clone() - exp_rhs;
    let saturation = (mant_len + 64) as isize;
    let exp_diff_signed: isize = exp_diff.try_into().unwrap_or(saturation);

    let (mant_rhs, sub_align_info) = if exp_diff_signed >= 0 {
        let r = mant_rhs_aligned.shift_right_rounded(exp_diff_signed as usize);
        (r.value, r.info)
    } else {
        (
            mant_rhs_aligned.shift_fixed(-exp_diff_signed),
            ShiftRoundingInfo::default(),
        )
    };

    // Subtract mantissas using a 1-bit guard column at bit 0.
    let lhs_ext = mant_lhs.append_bool_in_place(false).shift_fixed(-1);
    let rhs_ext_shifted = mant_rhs.append_bool_in_place(false).shift_fixed(-1);
    let rhs_ext = if sub_align_info.guard {
        rhs_ext_shifted + Frac::from_bits(&[true]) // propagate guard bit
    } else {
        rhs_ext_shifted
    };

    let mut mantissa_result = lhs_ext - rhs_ext;

    // Normalise: bring the MSB to bit `mant_len` (the MSB of the extended mantissa).
    let msb_pos = mantissa_result.iter_bits().rposition(|b| b);
    match msb_pos {
        Some(msb_pos) => {
            let shift = msb_pos as isize - mant_len as isize;
            mantissa_result = mantissa_result.shift_fixed(shift);
            exp_lhs += BigInt::from(shift);
        }
        None => {
            *lhs = FlexFloat::zero_with_sign(lhs.sign);
            return;
        }
    }

    // Drop the guard column by right-shifting 1, then round.
    let mut mantissa_result = mantissa_result.shift_fixed(1).truncate(mant_len);

    // The guard bit was already consumed above; the remaining round/sticky bits
    // from the alignment shift become the new guard/round inputs.
    let post_sub_info = ShiftRoundingInfo {
        guard: sub_align_info.round,
        round: sub_align_info.sticky,
        sticky: false,
    };
    let result_lsb = mantissa_result.get(0).unwrap_or(false);
    if post_sub_info.should_round_up(result_lsb) {
        mantissa_result.add_one_in_place();
        if mantissa_result.len() > mant_len {
            mantissa_result = mantissa_result.shift_fixed(1).truncate(mant_len);
            exp_lhs += 1_u8;
        }
    }

    *lhs = build_finite_result(
        lhs.sign,
        exp_lhs - 1_u8,
        lhs.exponent.len(),
        mantissa_result,
    );
}

/// Addition operation for `FlexFloat`.
///
/// Implements the [`Add`] trait for FlexFloat, performing floating-point addition
/// with automatic exponent growth and precision preservation.
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// let distance = FlexFloat::from(120.5);
/// let extra = FlexFloat::from(4.25);
/// let total = distance + extra;
/// assert_ff_almost_eq!(total, FlexFloat::from(124.75));
/// ```
impl<E1: BitArrayArith, F1: BitArrayArith, E2: BitArrayConversion, F2: BitArrayConversion>
    Add<FlexFloat<E2, F2>> for FlexFloat<E1, F1>
{
    type Output = Self;

    fn add(mut self, rhs: FlexFloat<E2, F2>) -> Self::Output {
        let rhs: FlexFloat<E1, F1> = rhs.convert_to();
        add(&mut self, rhs);
        self
    }
}

/// Subtraction operation for `FlexFloat`.
///
/// Implements the [`Sub`] trait for FlexFloat, performing floating-point subtraction
/// with automatic exponent growth and precision preservation.
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// let balance = FlexFloat::from(250.0);
/// let withdrawal = FlexFloat::from(19.95);
/// let remaining = balance - withdrawal;
/// assert_ff_almost_eq!(remaining, FlexFloat::from(230.05));
/// ```
impl<E1: BitArrayArith, F1: BitArrayArith, E2: BitArrayConversion, F2: BitArrayConversion>
    Sub<FlexFloat<E2, F2>> for FlexFloat<E1, F1>
{
    type Output = Self;

    fn sub(mut self, rhs: FlexFloat<E2, F2>) -> Self::Output {
        let rhs: FlexFloat<E1, F1> = rhs.convert_to();
        sub(&mut self, rhs);
        self
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
    fn test_add(mut rng: impl Rng, n_experiments: usize) {
        // Test random basic case
        let a = 1.5;
        let b = 2.25;
        let c = FlexFloat::from(a) + FlexFloat::from(b);
        assert_eq!(c.to_f64(), Some(a + b));

        // Test overflow case: exponent AND fraction must grow together.
        let a = f64::MAX;
        let b = f64::MAX / 2.0;
        let c = FlexFloat::from(a) + FlexFloat::from(b);
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

        // grown + normal: fraction must remain grown.
        let grown = FlexFloat::from(f64::MAX) + FlexFloat::from(f64::MAX / 2.0);
        let sum = grown + FlexFloat::from(1.0_f64);
        assert_eq!(
            sum.fraction.len(),
            115,
            "grown + normal fraction should stay 115 bits"
        );

        // Test wierd edge case
        let a = f64::MAX;
        let b = f64::MIN_POSITIVE;
        let c = FlexFloat::from(a) + FlexFloat::from(b);
        assert_eq!(c.to_f64(), Some(a + b));

        let a = 1.72e-169;
        let b = 1.59e265;
        let c = FlexFloat::from(a) + FlexFloat::from(b);
        assert_eq!(c.to_f64(), Some(a + b));

        let tiny = FlexFloat::<DefaultBitArray>::from(f64::from_bits(1));
        let zero = FlexFloat::<DefaultBitArray>::from(0.0);
        let result = (tiny.clone() + zero).to_f64().unwrap();
        assert_eq!(result.to_bits(), f64::from_bits(1).to_bits());

        let result = (tiny.clone() + tiny).to_f64().unwrap();
        assert_eq!(result.to_bits(), f64::from_bits(2).to_bits());

        let nan = FlexFloat::<DefaultBitArray>::nan();
        assert!((nan.clone() + FlexFloat::from(1.0)).is_nan());
        assert!((FlexFloat::from(1.0) + nan).is_nan());
        assert!(
            (FlexFloat::<DefaultBitArray>::pos_infinity()
                + FlexFloat::<DefaultBitArray>::neg_infinity())
            .is_nan()
        );
        assert!(
            (FlexFloat::from(1.0) + FlexFloat::<DefaultBitArray>::pos_infinity()).is_infinite()
        );

        let zero_sum = (FlexFloat::<DefaultBitArray>::zero()
            + FlexFloat::<DefaultBitArray>::zero_with_sign(true))
        .to_f64()
        .unwrap();
        assert_eq!(zero_sum.to_bits(), 0.0f64.to_bits());

        test_binary_flexfloat_op(
            &mut rng,
            n_experiments,
            |a, b| a + b,
            |a: f64, b: f64| a + b,
            "add",
            identity_2,
            None,
        );
    }

    #[rstest]
    fn test_sub(mut rng: impl Rng, n_experiments: usize) {
        let a = 1.5;
        let b = 2.25;
        let c = FlexFloat::from(a) - FlexFloat::from(b);
        assert_eq!(c.to_f64(), Some(a - b));

        // Test overflow case: exponent AND fraction must grow together.
        let a = FlexFloat::from(-f64::MAX);
        let b = FlexFloat::from(f64::MAX / 2.0);
        let c = a - b;
        assert!(!c.is_infinite(), "Result should not overflow");
        assert_eq!(
            c.exponent.len(),
            12,
            "Sub exponent should have grown to 12 bits"
        );
        assert_eq!(
            c.fraction.len(),
            115,
            "Sub fraction should have grown to 115 bits"
        );

        // Test wierd edge case
        let a = f64::MAX;
        let b = f64::MIN_POSITIVE;
        let c = FlexFloat::from(a) - FlexFloat::from(b);
        assert_eq!(c.to_f64(), Some(a - b));

        let a = -1.09e-23;
        let b = -1.73e29;
        let c = FlexFloat::from(a) - FlexFloat::from(b);
        assert_eq!(c.to_f64(), Some(a - b));

        let nan = FlexFloat::<DefaultBitArray>::nan();
        assert!((nan.clone() - FlexFloat::from(1.0)).is_nan());
        assert!((FlexFloat::from(1.0) - nan).is_nan());
        assert!(
            (FlexFloat::<DefaultBitArray>::pos_infinity()
                - FlexFloat::<DefaultBitArray>::pos_infinity())
            .is_nan()
        );
        assert!(
            (FlexFloat::from(1.0) - FlexFloat::<DefaultBitArray>::pos_infinity()).is_infinite()
        );

        let zero_diff = (FlexFloat::<DefaultBitArray>::zero()
            - FlexFloat::<DefaultBitArray>::zero())
        .to_f64()
        .unwrap();
        assert_eq!(zero_diff.to_bits(), 0.0f64.to_bits());

        test_binary_flexfloat_op(
            &mut rng,
            n_experiments,
            |a, b| a - b,
            |a: f64, b: f64| a - b,
            "sub",
            identity_2,
            None,
        );
    }

    #[test]
    fn test_add_round_to_nearest_alignment() {
        let one = FlexFloat::<DefaultBitArray>::from(1.0);
        let eps = FlexFloat::<DefaultBitArray>::from(1.5_f64 * (2.0_f64).powi(-53));
        let result = (one + eps).to_f64().unwrap();
        let expected = 1.0_f64 + 1.5_f64 * (2.0_f64).powi(-53);
        assert_eq!(
            result.to_bits(),
            expected.to_bits(),
            "result={result:?} expected={expected:?}"
        );
    }

    #[rstest]
    fn test_subnormal_add_randomized(mut rng: impl Rng, n_experiments: usize) {
        for _ in 0..(n_experiments / 50) {
            let tiny_a = f64::from_bits(rng.random_range(1..=2048));
            let tiny_b = f64::from_bits(rng.random_range(1..=2048));

            test_common_logic(
                &format!("subnormal_add({tiny_a:?}, {tiny_b:?})"),
                FlexFloat::<DefaultBitArray>::from(tiny_a)
                    + FlexFloat::<DefaultBitArray>::from(tiny_b),
                tiny_a + tiny_b,
                Some(1e-12),
            );
        }
    }
}
