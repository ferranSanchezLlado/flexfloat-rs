//! Mantissa extraction and result-building helpers shared by all arithmetic kernels.

use num_bigint::{BigInt, BigUint};
use num_traits::One;

use crate::bitarray::BitArray;
use crate::bitarray::traits::{BitArrayRounding, ShiftRoundingResult};
use crate::flexfloat::{FlexFloat, grow_exponent};

#[inline]
pub(super) fn ensure_width<B: BitArray>(bits: B, width: usize) -> B {
    let len = bits.len();
    if len < width {
        bits.append_repeated(false, width - len)
    } else {
        bits
    }
}

/// Extract a normalised 53-bit mantissa and the unbiased IEEE exponent from a
/// finite `FlexFloat`. Subnormals are handled by appending an explicit zero
/// leading bit and adjusting the returned exponent.
#[inline]
pub(super) fn extract_mantissa_and_exponent<Exp: BitArray, Frac: BitArray>(
    value: &FlexFloat<Exp, Frac>,
) -> (Frac, BigInt) {
    let fraction = if value.fraction.len() < 52 {
        value
            .fraction
            .clone()
            .append_repeated(false, 52 - value.fraction.len())
    } else {
        value.fraction.clone()
    };
    let is_subnormal = value.exponent.is_zeros();
    if is_subnormal {
        (fraction.append_bool_in_place(false), BigInt::from(-1022))
    } else {
        (
            fraction.append_bool_in_place(true),
            value.exponent.to_bigint() + 1_u8,
        )
    }
}

/// Build a finite `FlexFloat<B, B>` from a sign, a stored (biased-1024) exponent,
/// a minimum exponent-field length hint, and a 53-bit mantissa.
///
/// Handles the three cases:
/// * Normal IEEE range  → standard 11-bit exponent
/// * Subnormal range    → zero exponent field + right-shifted mantissa
/// * Overflow range     → grown exponent via `grow_exponent`
#[inline]
pub(super) fn build_finite_result<Exp: BitArray, Frac: BitArray>(
    sign: bool,
    stored_exponent: BigInt,
    min_exp_len: usize,
    mantissa53: Frac,
) -> FlexFloat<Exp, Frac> {
    let mantissa53 = ensure_width(mantissa53, 53);
    let min_normal_stored = BigInt::from(-1023);
    let max_normal_stored = BigInt::from(1022);
    let has_implicit_leading_one = mantissa53.get(52).unwrap_or(false);

    if stored_exponent >= min_normal_stored
        && stored_exponent <= max_normal_stored
        && has_implicit_leading_one
    {
        let exponent_field: BigUint = (stored_exponent.clone() + BigInt::from(1024))
            .try_into()
            .expect("normal exponent fits into IEEE field");
        return FlexFloat {
            sign,
            exponent: Exp::from_biguint_fixed(&exponent_field, 11),
            fraction: ensure_width(mantissa53.truncate(52), 52),
        };
    }

    if stored_exponent < min_normal_stored
        || (stored_exponent == min_normal_stored && !has_implicit_leading_one)
    {
        let Ok(shift) = usize::try_from(min_normal_stored.clone() - stored_exponent.clone()) else {
            return FlexFloat::zero_with_sign(sign);
        };
        let ShiftRoundingResult { value, info } = mantissa53.shift_right_rounded(shift);
        let mut fraction = ensure_width(value.truncate(52), 52);
        let result_lsb = fraction.get(0).unwrap_or(false);
        if info.should_round_up(result_lsb) {
            fraction.add_one_in_place();
            if fraction.len() > 52 {
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

    FlexFloat {
        sign,
        exponent: grow_exponent(stored_exponent, min_exp_len),
        fraction: ensure_width(mantissa53.truncate(52), 52),
    }
}
