//! Mantissa extraction and result-building helpers shared by all arithmetic kernels.
//!
//! ## Helpers
//!
//! | Function | Purpose |
//! |----------|---------|
//! | [`ensure_width`] | Pad a bit-array to a minimum width by appending high zeros |
//! | [`align_mantissa`] | Left-shift a mantissa so its implicit leading 1 lands at a common MSB position |
//! | [`extract_mantissa_and_exponent`] | Decompose a `FlexFloat` into a normalised mantissa and unbiased exponent |
//! | [`build_finite_result`] | Assemble a `FlexFloat` from sign, stored exponent, and normalised mantissa, handling normal/subnormal/grown cases |
//!
//! ## Variable-width fraction contract
//!
//! `build_finite_result` accepts a mantissa of any width.  For values in the normal
//! IEEE 754 range the mantissa is rounded to 52 fraction bits.  For values whose
//! exponent exceeds the IEEE range both the exponent and fraction fields grow together
//! according to `n = 1 + e + m` where `e = round(log₂(n) × 1.5 + 2)`.  This means
//! arithmetic on grown-format values preserves precision beyond 52 bits.

use num_bigint::{BigInt, BigUint};
use num_traits::One;

use crate::bitarray::BitArray;
use crate::bitarray::traits::{BitArrayRounding, ShiftRoundingResult};
use crate::flexfloat::{
    FlexFloat, RoundedResult, exponent_bits_for_total, grow_exponent, truncate_fraction,
};

/// Pad `bits` to `width` by appending zeros to the high end.
///
/// Used for **fraction** fields, where the value is `0.b_{n-1}…b_0` and trailing
/// high zeros do not change the magnitude.
#[inline]
pub(super) fn ensure_width<B: BitArray>(bits: B, width: usize) -> B {
    let len = bits.len();
    if len < width {
        bits.append_repeated(false, width - len)
    } else {
        bits
    }
}

/// Align a mantissa to `width` bits by prepending zeros at the low end.
///
/// Mantissas are fixed-point values with the implicit leading 1 at the MSB
/// (bit `len - 1`). When two mantissas have different widths the shorter one
/// must be left-shifted so its implicit 1 lands at bit `width - 1`, keeping
/// the binary point aligned with the longer operand.
///
/// For example, a 53-bit mantissa `1.m₁…m₅₂` widened to 116 bits becomes
/// `1.m₁…m₅₂ 0…0` (63 low zeros appended), preserving the scaled value
/// `mantissa / 2^(width-1)`.
#[inline]
pub(super) fn align_mantissa<B: BitArray>(bits: B, width: usize) -> B {
    let len = bits.len();
    if len < width {
        bits.shift_grow((width - len) as isize)
    } else {
        bits
    }
}

/// Extract a normalised mantissa and the unbiased exponent from a finite `FlexFloat`.
///
/// Returns `(mantissa, exponent)` where `mantissa` is `fraction_len + 1` bits wide
/// (the leading implicit 1 is appended at the MSB) and `exponent` satisfies
/// `value = 2^exponent * (mantissa / 2^(mantissa.len()-1))`.
///
/// Subnormals are handled by prepending an explicit zero leading bit and
/// returning the minimum representable exponent for the field width.
#[inline]
pub(super) fn extract_mantissa_and_exponent<Exp: BitArray, Frac: BitArray>(
    value: &FlexFloat<Exp, Frac>,
) -> (Frac, BigInt) {
    let fraction = value.fraction.clone();
    let is_subnormal = value.exponent.is_zeros();
    if is_subnormal {
        // For an 11-bit exponent, subnormal effective exponent = -1022 = -(2^10) + 2.
        // Generalised: -(2^(exp_bits-1)) + 2.
        let exp_bits = value.exponent.len();
        let subnormal_exp = -(BigInt::from(1u64) << (exp_bits - 1)) + 2;
        (fraction.append_bool_in_place(false), subnormal_exp)
    } else {
        (
            fraction.append_bool_in_place(true),
            value.exponent.to_bigint() + 1_u8,
        )
    }
}

/// Build a finite `FlexFloat<Exp, Frac>` from a sign, stored exponent, minimum
/// exponent-field width hint, and normalised mantissa.
///
/// The mantissa must be in normalised form: the MSB (implicit leading 1) sits
/// at bit `mantissa.len() - 1`.  Three cases are handled:
///
/// | Range                         | Exponent | Fraction |
/// |-------------------------------|----------|----------|
/// | Normal IEEE  `[-1023, 1022]`  | 11 bits  | 52 bits  |
/// | Subnormal / underflow         | 11 zeros | 52 bits  |
/// | Overflow `> 1022`             | grown    | grown    |
///
/// The grown format satisfies `n = 1 + e + m` with `e = round(log₂(n)·1.5 + 2)`.
#[inline]
pub(super) fn build_finite_result<Exp: BitArray, Frac: BitArray>(
    sign: bool,
    stored_exponent: BigInt,
    min_exp_len: usize,
    mantissa: Frac,
) -> FlexFloat<Exp, Frac> {
    let mantissa_bits = mantissa.len();
    if mantissa_bits == 0 {
        return FlexFloat::zero_with_sign(sign);
    }

    let min_normal_stored = BigInt::from(-1023_i32);
    let max_normal_stored = BigInt::from(1022_i32);
    let has_leading_one = mantissa.get(mantissa_bits - 1).unwrap_or(false);

    // --- Case 1: Normal IEEE range ---
    if stored_exponent >= min_normal_stored
        && stored_exponent <= max_normal_stored
        && has_leading_one
    {
        debug_assert!(
            mantissa_bits > 1,
            "normalised mantissa must be at least 2 bits"
        );

        let exponent_field: BigUint = (stored_exponent.clone() + 1024_i32)
            .try_into()
            .expect("normal exponent always fits in IEEE field");

        let frac_part = mantissa.truncate(mantissa_bits - 1); // strip implicit leading 1
        let RoundedResult {
            fraction: frac52,
            carry,
        } = truncate_fraction(frac_part, 52);

        if carry {
            // Rounding carried into the exponent.
            let new_exp = stored_exponent + 1_i32;
            if new_exp > max_normal_stored {
                // Spilled out of normal range — recurse into Case 3.
                return build_finite_result(sign, new_exp, min_exp_len, unit_mantissa(53));
            }
            let exponent_field: BigUint = (new_exp + 1024_i32)
                .try_into()
                .expect("incremented exponent fits in IEEE field");
            return FlexFloat {
                sign,
                exponent: Exp::from_biguint_fixed(&exponent_field, 11),
                fraction: Frac::zeros(52),
            };
        }

        return FlexFloat {
            sign,
            exponent: Exp::from_biguint_fixed(&exponent_field, 11),
            fraction: ensure_width(frac52, 52),
        };
    }

    // --- Case 2: Subnormal / underflow range ---
    if stored_exponent < min_normal_stored
        || (stored_exponent == min_normal_stored && !has_leading_one)
    {
        let Ok(shift) = usize::try_from(min_normal_stored.clone() - stored_exponent.clone()) else {
            return FlexFloat::zero_with_sign(sign);
        };

        let ShiftRoundingResult { value, info } = mantissa.shift_right_rounded(shift);
        let mut fraction = ensure_width(value.truncate(52), 52);
        let result_lsb = fraction.get(0).unwrap_or(false);
        if info.should_round_up(result_lsb) {
            fraction.add_one_in_place();
            if fraction.len() > 52 {
                // Rounding carried out of subnormal range into the smallest normal.
                return FlexFloat {
                    sign,
                    exponent: Exp::from_biguint_fixed(&BigUint::one(), 11),
                    fraction: Frac::zeros(52),
                };
            }
        }

        return if !fraction.is_zeros() {
            FlexFloat {
                sign,
                exponent: Exp::zeros(11),
                fraction,
            }
        } else {
            FlexFloat::zero_with_sign(sign)
        };
    }

    // --- Case 3: Overflow range — grow exponent AND fraction ---
    //
    // grow_exponent encodes `stored_exponent` into the smallest bit-array
    // that fits it as a signed integer; its width `e` determines the total
    // format size `n` through the inverse of `e = round(log₂(n)·1.5 + 2)`.
    let grown_exp: Exp = grow_exponent(stored_exponent, min_exp_len);
    let e = grown_exp.len();

    let target_frac_len = frac_bits_for_exp_width(e);

    let frac_part = mantissa.truncate(mantissa_bits - 1); // strip implicit leading 1
    let RoundedResult {
        fraction: grown_fraction,
        carry,
    } = truncate_fraction(frac_part, target_frac_len);

    if carry {
        // Rounding carried into the exponent — rebuild with stored_exponent + 1.
        return build_finite_result(
            sign,
            grown_exp.to_bigint() + 2_u8, // to_bigint() == stored_exp - 1, so + 2 gives stored_exp + 1
            min_exp_len,
            unit_mantissa(mantissa_bits),
        );
    }

    FlexFloat {
        sign,
        exponent: grown_exp,
        fraction: ensure_width(grown_fraction, target_frac_len),
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Return a mantissa whose only set bit is the implicit leading 1 at the MSB.
///
/// Equivalent to the integer value `1` in a `width`-bit fixed-point mantissa,
/// representing the normalised value `1.000…`.
fn unit_mantissa<B: BitArray>(width: usize) -> B {
    debug_assert!(width >= 1);
    let mut bits = vec![false; width];
    bits[width - 1] = true;
    B::from_bits(&bits)
}

/// Compute the fraction-field width for a grown-format float whose exponent
/// field is `e` bits wide.
///
/// Finds the largest total bit-count `n` such that `exponent_bits_for_total(n) == e`,
/// then returns `n - e - 1` (sign bit excluded).  Capped at [`MAX_PRACTICAL_N`] to
/// avoid astronomical allocations for extreme exponents, falling back to 52 bits
/// (standard IEEE double precision) in that case.
fn frac_bits_for_exp_width(e: usize) -> usize {
    // The transition from e to e+1 occurs around n_high ≈ 2^((e-1.5)/1.5).
    const MAX_PRACTICAL_N: usize = 4096;
    let n_high_f = 2f64.powf((e as f64 - 1.5) / 1.5);
    if n_high_f >= MAX_PRACTICAL_N as f64 {
        return 52;
    }

    let n_high = (n_high_f as usize + 4).min(MAX_PRACTICAL_N);
    let n_low = {
        let n_low_f = 2f64.powf((e as f64 - 2.5) / 1.5);
        (n_low_f as usize).saturating_sub(2).max(e + 2)
    };

    let best_n = (n_low..=n_high)
        .filter(|&n| exponent_bits_for_total(n) == e)
        .max()
        .unwrap_or(e + 2);

    best_n.saturating_sub(e + 1)
}
