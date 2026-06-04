//! # Arithmetic Operations
//!
//! Implements fundamental arithmetic operations for FlexFloat with automatic
//! exponent growth and precision preservation.
//!
//! ## Overview
//!
//! This module provides arithmetic operations that maintain FlexFloat's key
//! characteristics: growing exponents when needed and consistent precision
//! through the fixed-width fraction field.
//!
//! ## Exponent Growth Strategy
//!
//! When arithmetic operations produce results that exceed the current exponent
//! range, the exponent field automatically grows to accommodate the new value.
//! This ensures that FlexFloat can represent arbitrarily large or small numbers
//! without overflow.
//!
//! ## Implemented Operations
//!
//! - **Negation**: Sign bit flipping with `Neg` trait
//! - **Absolute value**: Sign bit clearing
//! - **Addition/Subtraction**: With automatic exponent expansion
//! - **Multiplication/Division**: With precision preservation
//!
//! ## Examples
//!
//! ```rust
//! use flexfloat::prelude::*;
//!
//! let subtotal = FlexFloat::from(19.99);
//! let tax = FlexFloat::from(1.60);
//!
//! let total = subtotal.clone() + tax.clone();
//! assert_ff_almost_eq!(total, FlexFloat::from(21.59));
//!
//! let pretax = total.clone() - tax.clone();
//! assert_ff_almost_eq!(pretax, FlexFloat::from(19.99));
//!
//! let doubled = subtotal.clone() * FlexFloat::from(2.0);
//! assert_ff_almost_eq!(doubled, FlexFloat::from(39.98));
//!
//! let split = total / FlexFloat::from(4.0);
//! assert_ff_almost_eq!(split, FlexFloat::from(5.3975));
//! ```

use core::cmp::max;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};

use crate::bitarray::traits::{BitArrayRounding, ShiftRoundingInfo, ShiftRoundingResult};
use crate::bitarray::{BitArray, BitArrayConversion};
use crate::flexfloat::math::trunc;
use crate::flexfloat::{FlexFloat, grow_exponent};

fn extract_mantissa_and_exponent<B: BitArray>(value: &FlexFloat<B>) -> (B, BigInt) {
    let fraction = if value.fraction.len() < 52 {
        value
            .fraction
            .clone()
            .append_repeated(false, 52 - value.fraction.len())
    } else {
        value.fraction.clone()
    };
    let is_subnormal = value.exponent.to_biguint().is_zero();
    if is_subnormal {
        (
            fraction.append_bool_in_place(false),
            value.exponent.to_bigint() + 2_u8,
        )
    } else {
        (
            fraction.append_bool_in_place(true),
            value.exponent.to_bigint() + 1_u8,
        )
    }
}

fn build_finite_result<B: BitArray>(
    sign: bool,
    stored_exponent: BigInt,
    min_exp_len: usize,
    mantissa53: B,
) -> FlexFloat<B> {
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
            exponent: B::from_biguint_fixed(&exponent_field, 11),
            fraction: mantissa53.truncate(52),
        };
    }

    if stored_exponent < min_normal_stored
        || (stored_exponent == min_normal_stored && !has_implicit_leading_one)
    {
        let Ok(shift) = usize::try_from(min_normal_stored.clone() - stored_exponent.clone()) else {
            return FlexFloat::new_zero_with_sign(sign);
        };
        let ShiftRoundingResult { value, info } = mantissa53.shift_right_rounded(shift);
        let mut fraction = value.truncate(52);
        let result_lsb = fraction.get(0).unwrap_or(false);
        if info.should_round_up(result_lsb) {
            let bumped = fraction.clone() + B::from_bits(&[true]);
            if bumped.len() > 52 {
                return FlexFloat {
                    sign,
                    exponent: B::from_biguint_fixed(&BigUint::one(), 11),
                    fraction: B::zeros(52),
                };
            }
            fraction = bumped;
        }

        return if fraction.iter_bits().any(|b| b) {
            FlexFloat {
                sign,
                exponent: B::zeros(11),
                fraction,
            }
        } else {
            FlexFloat::new_zero_with_sign(sign)
        };
    }

    FlexFloat {
        sign,
        exponent: grow_exponent(stored_exponent, min_exp_len),
        fraction: mantissa53.truncate(52),
    }
}

impl<B: BitArray> FlexFloat<B> {
    /// Returns the absolute value of this FlexFloat.
    ///
    /// Creates a new FlexFloat with the same magnitude but positive sign,
    /// effectively clearing the sign bit while preserving exponent and fraction.
    ///
    /// # Returns
    ///
    /// A new FlexFloat representing the absolute value
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let negative = FlexFloat::from(-3.14159);
    /// let positive = negative.abs();
    ///
    /// assert!(negative.sign());
    /// assert!(!positive.sign());
    /// // Magnitude is preserved
    /// assert_eq!(negative.exponent().to_bits(), positive.exponent().to_bits());
    /// assert_eq!(negative.fraction().to_bits(), positive.fraction().to_bits());
    /// ```
    pub fn abs(&self) -> Self
    where
        B: Clone,
    {
        Self {
            sign: false,
            exponent: self.exponent.clone(),
            fraction: self.fraction.clone(),
        }
    }

    pub fn rem_euclid<B2: BitArrayConversion>(self, rhs: &FlexFloat<B2>) -> Self {
        let rhs_converted: FlexFloat<B> = rhs.convert_to();
        let result = self.clone() % rhs;
        if result.is_nan() || rhs_converted.is_nan() {
            return result;
        }
        if result.is_negative() {
            result + rhs_converted.abs()
        } else {
            result
        }
    }

    pub fn div_euclid<B2: BitArrayConversion>(self, rhs: &FlexFloat<B2>) -> Self {
        let rhs_converted: FlexFloat<B> = rhs.convert_to();
        let quotient = trunc(self.clone() / rhs);
        let remainder = self % rhs;
        if remainder.is_nan() || quotient.is_nan() {
            return quotient;
        }
        if remainder.is_negative() {
            if rhs_converted.is_positive() {
                quotient - FlexFloat::<B>::from_f64(1.0)
            } else {
                quotient + FlexFloat::<B>::from_f64(1.0)
            }
        } else {
            quotient
        }
    }
}

impl<B: BitArray, B2: BitArrayConversion> Rem<&FlexFloat<B2>> for FlexFloat<B> {
    type Output = Self;

    fn rem(self, rhs: &FlexFloat<B2>) -> Self::Output {
        if let (Some(lhs), Some(rhs_f64)) = (self.to_f64(), rhs.to_f64()) {
            return FlexFloat::<B>::from_f64(lhs % rhs_f64);
        }

        let rhs: FlexFloat<B> = rhs.convert_to();
        if self.is_nan() || rhs.is_nan() || self.is_infinite() || rhs.is_zero() {
            return Self::new_nan();
        }
        if rhs.is_infinite() {
            return self;
        }
        if self.is_zero() {
            return Self::new_zero_with_sign(self.sign ^ rhs.sign ^ rhs.sign);
        }

        self.clone() - trunc(self / &rhs) * rhs
    }
}

/// Negation operation for FlexFloat.
///
/// Implements the `Neg` trait to provide unary minus operation,
/// flipping the sign bit while preserving the magnitude.
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
///
/// let positive = FlexFloat::from(2.71828);
/// let negative = -positive.clone();
///
/// assert!(!positive.sign());
/// assert!(negative.sign());
/// ```
impl<B: BitArray> Neg for FlexFloat<B> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            sign: !self.sign,
            exponent: self.exponent,
            fraction: self.fraction,
        }
    }
}

fn add<B: BitArray>(lhs: &mut FlexFloat<B>, mut rhs: FlexFloat<B>) {
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
        (true, true, false) => return *lhs = FlexFloat::new_nan(), // inf + -inf = NaN
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
        let bumped = mantissa_result + B::from_bits(&[true]);
        if bumped.len() > 53 {
            mantissa_result = bumped.shift_fixed(1).truncate(53);
            exp_lhs += 1_u8;
        } else {
            mantissa_result = bumped;
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

fn sub<B: BitArray>(lhs: &mut FlexFloat<B>, mut rhs: FlexFloat<B>) {
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
        (true, true, true) => return *lhs = FlexFloat::new_nan(), // inf - inf = NaN
        (true, true, false) => return,                            // inf - -inf = inf
        (true, false, _) => return,                               // inf - x = inf
        (false, true, _) => return *lhs = -rhs,                   // x - inf = -inf
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
    let mut mantissa_result = lhs.fraction.clone() - mant_rhs;
    // Find leading zeros for normalization
    let msb_pos = mantissa_result.iter_bits().rposition(|b| b);
    match msb_pos {
        Some(msb_pos) => {
            let shift = msb_pos as isize - 52;
            mantissa_result = mantissa_result.shift_fixed(shift);
            exp_lhs += BigInt::from(shift);
        }
        None => {
            *lhs = FlexFloat::new_zero_with_sign(lhs.sign);
            return;
        }
    }

    {
        let result_lsb = mantissa_result.get(0).unwrap_or(false);
        if sub_align_info.should_round_up(result_lsb) {
            let bumped = mantissa_result.clone() + B::from_bits(&[true]);
            if bumped.len() > mantissa_result.len() {
                mantissa_result = bumped.shift_fixed(1).truncate(mantissa_result.len());
                exp_lhs += 1_u8;
            } else {
                mantissa_result = bumped;
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

/// Addition operation for FlexFloat.
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
impl<B: BitArray, B2: BitArrayConversion> Add<&FlexFloat<B2>> for FlexFloat<B> {
    type Output = Self;

    fn add(mut self, rhs: &FlexFloat<B2>) -> Self::Output {
        let rhs: FlexFloat<B> = rhs.convert_to();
        add(&mut self, rhs);
        self
    }
}

/// Subtraction operation for FlexFloat.
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
impl<B: BitArray, B2: BitArrayConversion> Sub<&FlexFloat<B2>> for FlexFloat<B> {
    type Output = Self;

    fn sub(mut self, rhs: &FlexFloat<B2>) -> Self::Output {
        let rhs: FlexFloat<B> = rhs.convert_to();
        sub(&mut self, rhs);
        self
    }
}
/// Multiplication operation for FlexFloat.
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
impl<B: BitArray, B2: BitArrayConversion> Mul<&FlexFloat<B2>> for FlexFloat<B> {
    type Output = Self;

    fn mul(self, rhs: &FlexFloat<B2>) -> Self::Output {
        let rhs: FlexFloat<B> = rhs.convert_to();
        // 0. Handle special cases (NaN, Infinity).
        if self.is_nan() || rhs.is_nan() {
            return Self::new_nan();
        }

        if (self.is_infinite() && rhs.is_zero()) || (self.is_zero() && rhs.is_infinite()) {
            return Self::new_nan();
        }

        let sign = self.sign ^ rhs.sign;
        if self.is_zero() || rhs.is_zero() {
            return Self::new_zero_with_sign(sign);
        }

        if self.is_infinite() || rhs.is_infinite() {
            return Self::new_infinity(sign);
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

        let mut mant_res_53 = mant_res.get_range(0..53).unwrap();

        let result_lsb = mant_res_53.get(0).unwrap_or(false);
        if mul_round_info.should_round_up(result_lsb) {
            let bumped = mant_res_53.clone() + B::from_bits(&[true]);
            if bumped.len() > 53 {
                mant_res_53 = bumped.shift_fixed(1).truncate(53);
                exp_res += 1_u8;
            } else {
                mant_res_53 = bumped;
            }
        }

        // 7. Grow exponent if necessary (no limit on size)
        let exp_res = exp_res - 1_u8;
        let max_exp_len = max(self.exponent.len(), rhs.exponent.len());
        build_finite_result(sign, exp_res, max_exp_len, mant_res_53)
    }
}
/// Division operation for FlexFloat.
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
impl<B: BitArray, B2: BitArrayConversion> Div<&FlexFloat<B2>> for FlexFloat<B> {
    type Output = Self;

    fn div(self, rhs: &FlexFloat<B2>) -> Self::Output {
        let rhs: FlexFloat<B> = rhs.convert_to();
        // 0. Handle special cases (NaN, Infinity).
        if self.is_nan() || rhs.is_nan() {
            return Self::new_nan();
        }

        if rhs.is_zero() {
            if self.is_zero() {
                return Self::new_nan(); // 0 / 0 = NaN
            }
            return Self::new_infinity(self.sign ^ rhs.sign); // x / 0 = inf
        }

        let sign = self.sign ^ rhs.sign;
        if self.is_zero() {
            return Self::new_zero_with_sign(sign);
        }

        if self.is_infinite() {
            if rhs.is_infinite() {
                return Self::new_nan(); // inf / inf = NaN
            }
            return Self::new_infinity(sign);
        }

        if rhs.is_infinite() {
            return Self::new_zero_with_sign(sign); // x / inf = 0
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
        if mant_res.iter_bits().all(|b| !b) {
            return Self::new_zero_with_sign(sign);
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

        let mut mant_res_final = mant_res.get_range(0..53).unwrap();

        let result_lsb = mant_res_final.get(0).unwrap_or(false);
        if div_round_info.should_round_up(result_lsb) {
            let bumped = mant_res_final.clone() + B::from_bits(&[true]);
            if bumped.len() > 53 {
                mant_res_final = bumped.shift_fixed(1).truncate(53);
                exp_res += 1_u8;
            } else {
                mant_res_final = bumped;
            }
        }

        // 7. Grow exponent if necessary (no limit on size)
        let exp_res = exp_res - 1_u8;
        let max_exp_len = max(self.exponent.len(), rhs.exponent.len());
        build_finite_result(sign, exp_res, max_exp_len, mant_res_final)
    }
}

/// Macro to implement all combinations of binary ops for FlexFloat between references and owned values
macro_rules! impl_flexfloat_binop_refs {
    ($trait:ident, $method:ident) => {
        impl<B: BitArray, B2: BitArrayConversion> $trait<FlexFloat<B2>> for FlexFloat<B> {
            type Output = Self;

            fn $method(self, rhs: FlexFloat<B2>) -> Self::Output {
                self.$method(&rhs)
            }
        }

        impl<B: BitArray, B2: BitArrayConversion> $trait<&FlexFloat<B2>> for &FlexFloat<B> {
            type Output = FlexFloat<B>;

            fn $method(self, rhs: &FlexFloat<B2>) -> Self::Output {
                self.clone().$method(rhs)
            }
        }

        impl<B: BitArray, B2: BitArrayConversion> $trait<FlexFloat<B2>> for &FlexFloat<B> {
            type Output = FlexFloat<B>;

            fn $method(self, rhs: FlexFloat<B2>) -> Self::Output {
                self.clone().$method(&rhs)
            }
        }
    };
}

impl_flexfloat_binop_refs!(Add, add);
impl_flexfloat_binop_refs!(Sub, sub);
impl_flexfloat_binop_refs!(Mul, mul);
impl_flexfloat_binop_refs!(Div, div);
impl_flexfloat_binop_refs!(Rem, rem);

// Macro to implement all combinations of binary ops assign for FlexFloat between references and owned values
macro_rules! impl_flexfloat_binop_assign_refs {
    ($trait:ident, $method:ident, $op:ident) => {
        impl<B: BitArray, B2: BitArrayConversion> $trait<FlexFloat<B2>> for FlexFloat<B> {
            fn $method(&mut self, rhs: FlexFloat<B2>) {
                *self = self.clone().$op(&rhs);
            }
        }

        impl<B: BitArray, B2: BitArrayConversion> $trait<&FlexFloat<B2>> for FlexFloat<B> {
            fn $method(&mut self, rhs: &FlexFloat<B2>) {
                *self = self.clone().$op(rhs);
            }
        }
    };
}

impl_flexfloat_binop_assign_refs!(AddAssign, add_assign, add);
impl_flexfloat_binop_assign_refs!(SubAssign, sub_assign, sub);
impl_flexfloat_binop_assign_refs!(MulAssign, mul_assign, mul);
impl_flexfloat_binop_assign_refs!(DivAssign, div_assign, div);
impl_flexfloat_binop_assign_refs!(RemAssign, rem_assign, rem);

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::DefaultBitArray;
    use crate::bitarray::BitArrayAccess;
    use crate::tests::*;

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
        assert!((FlexFloat::pos_infinity() + FlexFloat::neg_infinity()).is_nan());
        assert!((FlexFloat::from(1.0) + FlexFloat::pos_infinity()).is_infinite());

        let zero_sum = (FlexFloat::<DefaultBitArray>::zero() + FlexFloat::zero_with_sign(true))
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
        assert!((FlexFloat::pos_infinity() - FlexFloat::pos_infinity()).is_nan());
        assert!((FlexFloat::from(1.0) - FlexFloat::pos_infinity()).is_infinite());

        let zero_diff = (FlexFloat::<DefaultBitArray>::zero() - FlexFloat::zero())
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

        assert!((FlexFloat::<DefaultBitArray>::zero() * FlexFloat::pos_infinity()).is_nan());
        assert!((FlexFloat::pos_infinity() * FlexFloat::from(-2.0)).is_infinite());
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

        let pos_inf = (FlexFloat::<DefaultBitArray>::from(1.0) / FlexFloat::zero())
            .to_f64()
            .unwrap();
        assert_eq!(pos_inf.to_bits(), f64::INFINITY.to_bits());
        assert!((FlexFloat::<DefaultBitArray>::zero() / FlexFloat::zero()).is_nan());
        assert!((FlexFloat::pos_infinity() / FlexFloat::pos_infinity()).is_nan());

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

    #[rstest]
    fn test_rem_and_euclid(mut rng: impl Rng, n_experiments: usize) {
        let result = (FlexFloat::<DefaultBitArray>::from(5.3) % FlexFloat::from(2.0))
            .to_f64()
            .unwrap();
        assert_eq!(result.to_bits(), (5.3f64 % 2.0).to_bits());

        let result = FlexFloat::<DefaultBitArray>::from(-5.0)
            .rem_euclid(&FlexFloat::from(3.0))
            .to_f64()
            .unwrap();
        assert_eq!(result.to_bits(), (-5.0f64).rem_euclid(3.0).to_bits());

        let result = FlexFloat::<DefaultBitArray>::from(-5.0)
            .div_euclid(&FlexFloat::from(3.0))
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
}
