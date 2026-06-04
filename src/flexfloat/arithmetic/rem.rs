//! Remainder, `rem_euclid`, and `div_euclid` operations.
//!
//! # Algorithm
//!
//! IEEE 754 remainder (`a % b`) is defined as `a - trunc(a/b) * b`.  We
//! compute this entirely in the `BitArray` / `BigInt` domain – no `to_f64`
//! conversion is used.
//!
//! ## Steps
//!
//! 1. Handle special cases (NaN, infinity, zero).
//! 2. Extract the 53-bit mantissa and unbiased exponent of both operands
//!    using `extract_mantissa_and_exponent`.
//! 3. Compute the exponent difference `d = exp_a - exp_b`.
//!    * If `d < 0`, `|a| < |b|` so `a % b == a` (no reduction needed).
//! 4. Left-shift `mant_a` by `d` bits (`shift_grow(d)`) so both mantissas
//!    are expressed at the same binary scale.
//! 5. Integer-divide the aligned mantissa by `mant_b`:
//!    `mant_rem = aligned_mant_a % mant_b`
//!    (computed as `aligned - (aligned / mant_b) * mant_b`).
//! 6. If `mant_rem` is zero the result is signed zero.
//! 7. Normalise: find the MSB position, shift to place it at index 52,
//!    adjust the result exponent accordingly.
//! 8. Apply IEEE 754 round-to-nearest-even via `shift_right_rounded`.
//! 9. Build the final `FlexFloat` with `build_finite_result`.

use core::cmp::max;
use core::ops::Rem;

use num_bigint::BigInt;
use num_traits::Zero;

use crate::bitarray::traits::{BitArrayRounding, ShiftRoundingInfo};
use crate::bitarray::{BitArrayArith, BitArrayConversion};
use crate::flexfloat::FlexFloat;
use crate::flexfloat::arithmetic::normalize::{
    build_finite_result, ensure_width, extract_mantissa_and_exponent,
};

/// Core routine: compute `self % rhs` (truncated remainder) in the
/// `BitArray` domain and return the raw result together with its sign.
///
/// Both operands **must** be finite and non-zero when this is called.
/// The sign of the result follows the sign of `self` (C / IEEE 754
/// truncated-remainder convention).
fn rem_finite<E1: BitArrayArith, F1: BitArrayArith>(
    lhs: &FlexFloat<E1, F1>,
    rhs: &FlexFloat<E1, F1>,
) -> FlexFloat<E1, F1> {
    let sign = lhs.sign; // remainder carries the sign of the dividend

    let (mant_a, exp_a) = extract_mantissa_and_exponent(lhs);
    let (mant_b, exp_b) = extract_mantissa_and_exponent(rhs);

    // d = exp_a - exp_b.  When d < 0 the dividend is already smaller in
    // magnitude than the divisor, so the remainder is just the dividend.
    let d = &exp_a - &exp_b;
    let d_isize = match i64::try_from(&d) {
        Ok(v) => v as isize,
        // Astronomically large exponent difference → result is lhs unchanged.
        Err(_) => {
            if d < BigInt::zero() {
                return lhs.clone();
            } else {
                // Huge positive d: the aligned mantissa mod mant_b will be 0
                // only if lhs is an exact multiple, which is vanishingly rare;
                // clamp to a large-but-computable shift.
                isize::MAX
            }
        }
    };

    if d_isize < 0 {
        // |lhs| < |rhs|  →  remainder is lhs (sign already correct).
        return lhs.clone();
    }

    // Align: multiply mant_a by 2^d so that both mantissas share the same
    // binary weight.  shift_grow(positive) inserts zeros at the low end.
    let aligned = mant_a.shift_grow(d_isize);

    // Integer modulo: aligned_mant_a mod mant_b
    // BitArrayArith provides Div but not Rem, so we compute it manually.
    let quotient = aligned.clone() / mant_b.clone();
    let product = quotient * mant_b.clone();
    // aligned >= product because quotient = floor(aligned / mant_b)
    let mant_rem = aligned - product;

    // Zero remainder → signed zero.
    if mant_rem.is_zeros() {
        return FlexFloat::zero_with_sign(sign);
    }

    // Normalise: place the MSB at index 52.
    let msb_pos = mant_rem
        .iter_bits()
        .rposition(|b| b)
        .expect("mant_rem is non-zero");

    let shift = msb_pos as isize - 52;
    // The result exponent: the mantissa was expressed at scale 2^exp_b, and
    // normalisation adjusts by the shift we just computed.
    let mut exp_res = exp_b + BigInt::from(shift);

    let round_info;
    let mant_norm = if shift > 0 {
        let r = mant_rem.shift_right_rounded(shift as usize);
        round_info = r.info;
        r.value
    } else if shift < 0 {
        round_info = ShiftRoundingInfo::default();
        mant_rem.shift_grow(-shift)
    } else {
        round_info = ShiftRoundingInfo::default();
        mant_rem
    };

    let mut mant_final = ensure_width(mant_norm, 53).truncate(53);

    // Apply round-to-nearest-even.
    let result_lsb = mant_final.get(0).unwrap_or(false);
    if round_info.should_round_up(result_lsb) {
        mant_final.add_one_in_place();
        if mant_final.len() > 53 {
            mant_final = mant_final.shift_fixed(1).truncate(53);
            exp_res += 1_u8;
        }
    }

    // Build the result (stored exponent = unbiased − 1, matching the
    // convention in mul_div.rs and normalize.rs).
    let exp_stored = exp_res - 1_u8;
    let max_exp_len = max(lhs.exponent.len(), rhs.exponent.len());
    build_finite_result(sign, exp_stored, max_exp_len, mant_final)
}

// ── Rem trait ──────────────────────────────────────────────────────────────

impl<E1: BitArrayArith, F1: BitArrayArith, E2: BitArrayConversion, F2: BitArrayConversion>
    Rem<FlexFloat<E2, F2>> for FlexFloat<E1, F1>
{
    type Output = Self;

    fn rem(self, rhs: FlexFloat<E2, F2>) -> Self::Output {
        let rhs: FlexFloat<E1, F1> = rhs.convert_to();

        // Special cases ────────────────────────────────────────────────────
        if self.is_nan() || rhs.is_nan() {
            return Self::nan();
        }
        // inf % x  or  x % 0  →  NaN
        if self.is_infinite() || rhs.is_zero() {
            return Self::nan();
        }
        // 0 % x  →  0 (sign of dividend)
        if self.is_zero() {
            return Self::zero_with_sign(self.sign);
        }
        // x % inf  →  x
        if rhs.is_infinite() {
            return self;
        }

        rem_finite(&self, &rhs)
    }
}

// ── Euclidean helpers ──────────────────────────────────────────────────────

impl<Exp: BitArrayArith, Frac: BitArrayArith> FlexFloat<Exp, Frac> {
    /// Computes the non-negative remainder `self - k * rhs` for the smallest
    /// non-negative integer `k`, matching `f64::rem_euclid`.
    pub fn rem_euclid<E2: BitArrayConversion, F2: BitArrayConversion>(
        self,
        rhs: FlexFloat<E2, F2>,
    ) -> Self {
        let rhs: FlexFloat<Exp, Frac> = rhs.convert_to();

        // Delegate to Rem for the truncated remainder, then adjust sign.
        let r = self.clone() % rhs.clone();

        if r.is_nan() || r.is_zero() || !r.sign {
            // NaN, zero, or already non-negative: return as-is.
            r
        } else {
            // r < 0: add |rhs| to bring into [0, |rhs|).
            let rhs_abs = rhs.abs();
            r + rhs_abs
        }
    }

    /// Computes the Euclidean quotient `floor(self / rhs)` (or `ceil` when
    /// `rhs` is negative), matching `f64::div_euclid`.
    pub fn div_euclid<E2: BitArrayConversion, F2: BitArrayConversion>(
        self,
        rhs: FlexFloat<E2, F2>,
    ) -> Self {
        let rhs: FlexFloat<Exp, Frac> = rhs.convert_to();

        // div_euclid(a, b) = trunc(a / b) adjusted by ±1 when the truncated
        // remainder is negative:
        //   if rem < 0 and b > 0  →  trunc_quotient − 1
        //   if rem < 0 and b < 0  →  trunc_quotient + 1
        //
        // We compute the truncated integer quotient directly from the
        // truncated remainder:  trunc_q = (a - r) / b  where r = a % b.
        // This avoids needing to truncate a floating-point division result.
        let r = self.clone() % rhs.clone();

        if r.is_nan() {
            return Self::nan();
        }

        // trunc_q = (self - r) / rhs — exact integer division after subtracting r.
        let numerator = self - r.clone();
        let trunc_q = numerator / rhs.clone();

        if r.sign && !r.is_zero() {
            // remainder is negative — adjust by ±1
            let one = Self::from_f64(1.0_f64);
            if !rhs.sign {
                // b > 0: floor(a/b) = trunc(a/b) − 1
                trunc_q - one
            } else {
                // b < 0: floor(a/b) = trunc(a/b) + 1
                trunc_q + one
            }
        } else {
            trunc_q
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use crate::bitarray::DefaultBitArray;
    use crate::flexfloat::FlexFloat;
    use crate::test_support::*;

    #[rstest]
    fn test_rem_and_euclid(mut rng: impl Rng, n_experiments: usize) {
        let result = (FlexFloat::<DefaultBitArray>::from(5.3) % FlexFloat::from(2.0))
            .to_f64()
            .unwrap();
        assert_eq!(result.to_bits(), (5.3f64 % 2.0).to_bits());

        let result = FlexFloat::<DefaultBitArray>::from(-5.0)
            .rem_euclid(FlexFloat::from(3.0))
            .to_f64()
            .unwrap();
        assert_eq!(result.to_bits(), (-5.0f64).rem_euclid(3.0).to_bits());

        let result = FlexFloat::<DefaultBitArray>::from(-5.0)
            .div_euclid(FlexFloat::from(3.0))
            .to_f64()
            .unwrap();
        assert_eq!(result.to_bits(), (-5.0f64).div_euclid(3.0).to_bits());

        test_binary_flexfloat_op(
            &mut rng,
            n_experiments / 50,
            |a, b| a % b,
            |a: f64, b: f64| a % b,
            "rem",
            identity_2,
            None,
        );
    }

    #[rstest]
    fn test_rem_euclid(mut rng: impl Rng, n_experiments: usize) {
        test_binary_flexfloat_op(
            &mut rng,
            n_experiments / 50,
            |a, b| a.rem_euclid(b),
            |a: f64, b: f64| a.rem_euclid(b),
            "rem_euclid",
            identity_2,
            None,
        );
    }

    #[rstest]
    fn test_div_euclid(mut rng: impl Rng, n_experiments: usize) {
        test_binary_flexfloat_op(
            &mut rng,
            n_experiments / 50,
            |a, b| a.div_euclid(b),
            |a: f64, b: f64| a.div_euclid(b),
            "div_euclid",
            identity_2,
            None,
        );
    }
}
