//! Addition and subtraction kernels.

use core::ops::{Add, Sub};

use num_bigint::BigInt;

use crate::bitarray::traits::{BitArrayRounding, ShiftRoundingInfo, ShiftRoundingResult};
use crate::bitarray::{BitArrayArith, BitArrayConversion};
use crate::flexfloat::FlexFloat;
use crate::flexfloat::arithmetic::normalize::{build_finite_result, extract_mantissa_and_exponent};

pub(super) fn add<Exp: BitArrayArith, Frac: BitArrayArith>(
    lhs: &mut FlexFloat<Exp, Frac>,
    mut rhs: FlexFloat<Exp, Frac>,
) {
    if lhs.sign != rhs.sign {
        sub(lhs, -rhs);
        return;
    }

    // OBJECTIVE: Add two FlexFloat instances together.
    // https://www.sciencedirect.com/topics/computer-science/floating-point-addition
    // and: https://cse.hkust.edu.hk/~cktang/cs180/notes/lec21.pdf
    //
    // Steps:
    // 0. Handle special cases (NaN, Infinity).
    // 1. Extract exponent and fraction bits.
    // 2. Prepend leading 1 to form the mantissa.
    // 3. Compare exponents.
    // 4. Shift smaller mantissa if necessary.
    // 5. Add mantissas.
    // 6. Normalize mantissa and adjust exponent if necessary.
    // 7. Grow exponent if necessary. (no limit on size)
    // 8. Round result.
    // 9. Return new FlexFloat instance.

    // 0. Handle special cases (NaN, Infinity).
    match (lhs.is_nan(), rhs.is_nan()) {
        (true, _) => return,
        (_, true) => return *lhs = rhs,
        _ => {}
    }

    match (lhs.is_infinite(), rhs.is_infinite(), lhs.sign == rhs.sign) {
        (true, true, true) => return, // inf + inf = inf
        (true, true, false) => return *lhs = FlexFloat::nan(), // inf + -inf = NaN
        (true, false, _) => return,   // inf + x = inf
        (false, true, _) => return *lhs = rhs, // x + inf = inf
        _ => {}
    }

    // 1. Extract exponent and fraction bits.
    // 2. Prepend leading 1 to form the mantissa.
    // 3. Compare exponents.
    let (lhs_mantissa, mut exp_lhs) = extract_mantissa_and_exponent(lhs);
    let (rhs_mantissa, mut exp_rhs) = extract_mantissa_and_exponent(&rhs);

    // Compare to make sure self has the larger exponent.
    if exp_lhs < exp_rhs {
        core::mem::swap(lhs, &mut rhs);
        lhs.fraction = rhs_mantissa;
        rhs.fraction = lhs_mantissa;
        core::mem::swap(&mut exp_lhs, &mut exp_rhs);
    } else {
        lhs.fraction = lhs_mantissa;
        rhs.fraction = rhs_mantissa;
    }

    // 4. Shift smaller mantissa if necessary.
    let exp_diff = &exp_lhs - exp_rhs;
    assert!(
        exp_diff >= BigInt::ZERO,
        "Self exponent should be larger/equal"
    );
    let saturation: usize = rhs.fraction.len() + 64;
    let exp_diff_usize: usize = exp_diff.try_into().unwrap_or(saturation);
    let ShiftRoundingResult {
        value: mant_rhs,
        info: align_info,
    } = rhs.fraction.shift_right_rounded(exp_diff_usize);

    assert_eq!(lhs.fraction.len(), 53, "Mantissa length should be 53 bits");
    assert_eq!(mant_rhs.len(), 53, "Mantissa length should be 53 bits");

    // 5. Add mantissas.
    let mut mantissa_result = lhs.fraction.clone() + mant_rhs;
    // 6. Normalize mantissa and adjust exponent if necessary.
    let mut rounding_info = align_info;
    if mantissa_result.len() > 53 {
        let ShiftRoundingResult {
            value: shifted,
            info: norm_info,
        } = mantissa_result.shift_right_rounded(1);
        mantissa_result = shifted.truncate(53);
        exp_lhs += 1_u8;
        rounding_info = rounding_info.combine(norm_info);
    }

    let result_lsb = mantissa_result.get(0).unwrap_or(false);
    if rounding_info.should_round_up(result_lsb) {
        mantissa_result.add_one_in_place();
        if mantissa_result.len() > 53 {
            mantissa_result = mantissa_result.shift_fixed(1).truncate(53);
            exp_lhs += 1_u8;
        }
    }

    // 7. Grow exponent if necessary. (no limit on size)
    *lhs = build_finite_result(
        lhs.sign,
        exp_lhs - 1_u8,
        lhs.exponent.len(),
        mantissa_result,
    );
}

pub(super) fn sub<Exp: BitArrayArith, Frac: BitArrayArith>(
    lhs: &mut FlexFloat<Exp, Frac>,
    mut rhs: FlexFloat<Exp, Frac>,
) {
    use num_bigint::Sign;

    if lhs.sign != rhs.sign {
        // a - (-b) == a + b
        add(lhs, -rhs);
        return;
    }

    // 0. Handle special cases (NaN, Infinity).
    match (lhs.is_nan(), rhs.is_nan()) {
        (true, _) => return,
        (_, true) => return *lhs = rhs,
        _ => {}
    }

    match (lhs.is_infinite(), rhs.is_infinite(), lhs.sign == rhs.sign) {
        (true, true, true) => return *lhs = FlexFloat::nan(), // inf - inf = NaN
        (true, true, false) => return,                        // inf - -inf = inf
        (true, false, _) => return,                           // inf - x = inf
        (false, true, _) => return *lhs = -rhs,               // x - inf = -inf
        _ => {}
    }

    // If |lhs| < |rhs|, result is negative of (rhs - lhs)
    if lhs.abs() < rhs.abs() {
        let mut tmp = rhs;
        sub(&mut tmp, lhs.clone());
        *lhs = -tmp;
        return;
    }

    let (lhs_mantissa, mut exp_lhs) = extract_mantissa_and_exponent(lhs);
    let (rhs_mantissa, exp_rhs) = extract_mantissa_and_exponent(&rhs);

    lhs.fraction = lhs_mantissa;
    rhs.fraction = rhs_mantissa;

    // 4. Shift smaller mantissa if necessary.
    let exp_diff = exp_lhs.clone() - exp_rhs;
    let exp_diff_signed: isize = match exp_diff.sign() {
        Sign::Plus => exp_diff
            .clone()
            .try_into()
            .unwrap_or((rhs.fraction.len() + 64) as isize),
        Sign::NoSign => 0,
        Sign::Minus => {
            -((-exp_diff.clone())
                .try_into()
                .unwrap_or((rhs.fraction.len() + 64) as isize))
        }
    };

    let (mant_rhs, sub_align_info) = if exp_diff_signed >= 0 {
        let amount = exp_diff_signed as usize;
        let r = rhs.fraction.shift_right_rounded(amount);
        (r.value, r.info)
    } else {
        (
            rhs.fraction.shift_fixed(-exp_diff_signed),
            ShiftRoundingInfo::default(),
        )
    };

    assert_eq!(lhs.fraction.len(), 53, "Mantissa length should be 53 bits");
    assert_eq!(mant_rhs.len(), 53, "Mantissa length should be 53 bits");

    // 5. Subtract mantissas.
    //
    // We work in a 54-bit extended space to incorporate the guard bit from the
    // alignment shift before normalization, not after.  Deferring the guard-bit
    // correction to a post-normalization rounding step is incorrect: a large
    // left-shift during normalization amplifies the 0.5-ulp error into a
    // multi-ULP error.
    //
    // Extended 54-bit representation: we place the 53-bit mantissas in the
    // upper bits and use bit 0 as the guard column:
    //
    //   lhs_54  = append_zero_msb(lhs.fraction) << 1    ← lhs.fraction × 2
    //   rhs_54  = append_zero_msb(mant_rhs)     << 1
    //           + guard                                  ← guard column at bit 0
    //
    // diff_54 = lhs_54 - rhs_54 = 2 × (lhs.fraction - mant_rhs) − guard,
    // which is the exact difference at half-LSB resolution.
    //
    // After normalization to MSB at bit 53, we extract the 53-bit mantissa
    // by >> 1 (dropping the now-zero guard column).  The exponent update is
    // the same as in the original 53-bit algorithm.  The only residual
    // rounding information is round + sticky (< 0.5 ulp at guard scale).
    let lhs_54 = lhs
        .fraction
        .clone()
        .append_bool_in_place(false)
        .shift_fixed(-1);
    let rhs_54_shifted = mant_rhs.append_bool_in_place(false).shift_fixed(-1);
    let rhs_54 = if sub_align_info.guard {
        rhs_54_shifted + Frac::from_bits(&[true]) // + 1 at guard column (bit 0)
    } else {
        rhs_54_shifted
    };

    let mut mantissa_result = lhs_54 - rhs_54;

    // Normalise: bring MSB to bit 53 (the MSB of the 54-bit mantissa)
    let msb_pos = mantissa_result.iter_bits().rposition(|b| b);
    match msb_pos {
        Some(msb_pos) => {
            let shift = msb_pos as isize - 53;
            mantissa_result = mantissa_result.shift_fixed(shift);
            exp_lhs += BigInt::from(shift); // same as original algorithm
        }
        None => {
            *lhs = FlexFloat::zero_with_sign(lhs.sign);
            return;
        }
    }

    // Extract 53-bit mantissa (MSB at bit 52) by right-shifting out the guard column.
    let mut mantissa_result = mantissa_result.shift_fixed(1).truncate(53);
    {
        // Only round + sticky remain; guard was already incorporated above.
        let post_sub_info = ShiftRoundingInfo {
            guard: sub_align_info.round,
            round: sub_align_info.sticky,
            sticky: false,
        };
        let result_lsb = mantissa_result.get(0).unwrap_or(false);
        if post_sub_info.should_round_up(result_lsb) {
            let mantissa_len = mantissa_result.len();
            mantissa_result.add_one_in_place();
            if mantissa_result.len() > mantissa_len {
                mantissa_result = mantissa_result.shift_fixed(1).truncate(mantissa_len);
                exp_lhs += 1_u8;
            }
        }
    }

    // 7. Grow exponent if necessary. (no limit on size)
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

        // Test overflow case
        let a = f64::MAX;
        let b = f64::MAX / 2.0;
        let c = FlexFloat::from(a) + FlexFloat::from(b);
        assert!(!c.is_infinite(), "Result should not overflow");
        assert!(c.exponent.len() > 11, "Exponent should have grown");

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

        // Test overflow case
        let a = FlexFloat::from(-f64::MAX);
        let b = FlexFloat::from(f64::MAX / 2.0);
        let c = a - b;
        assert!(!c.is_infinite(), "Result should not overflow");
        assert!(c.exponent.len() > 11, "Exponent should have grown");

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
