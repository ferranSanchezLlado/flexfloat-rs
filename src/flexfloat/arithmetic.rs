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
//! - **Addition/Subtraction**: With automatic exponent expansion (planned)
//! - **Multiplication/Division**: With precision preservation (planned)
//!
//! ## Examples
//!
//! ```rust
//! use flexfloat::FlexFloat;
//!
//! let x = FlexFloat::from(3.14);
//! let neg_x = -x.clone();
//! let abs_x = x.abs();
//!
//! assert_eq!(x.sign(), false);
//! assert_eq!(neg_x.sign(), true);
//! assert_eq!(abs_x.sign(), false);
//! ```

use std::cmp::{max, min};
use std::ops::{Add, Div, Mul, Neg, Sub};

use num_bigint::BigInt;

use crate::bitarray::BitArray;
use crate::flexfloat::{FlexFloat, grow_exponent};

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
}

/// Negation operation for FlexFloat.
///
/// Implements the `Neg` trait to provide unary minus operation,
/// flipping the sign bit while preserving the magnitude.
///
/// # Examples
///
/// ```rust
/// use flexfloat::FlexFloat;
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

impl<B: BitArray> Add for FlexFloat<B> {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        if self.sign != rhs.sign {
            return self - (-rhs);
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
        match (self.is_nan(), rhs.is_nan()) {
            (true, _) => return self,
            (_, true) => return rhs,
            _ => {}
        }

        match (self.is_infinity(), rhs.is_infinity(), self.sign == rhs.sign) {
            (true, true, true) => return self, // inf + inf = inf
            (true, true, false) => return FlexFloat::new_nan(), // inf + -inf = NaN
            (true, false, _) => return self,   // inf + x = inf
            (false, true, _) => return rhs,    // x + inf = inf
            _ => {}
        }

        // 1. Extract exponent and fraction bits.
        // 2. Prepend leading 1 to form the mantissa.
        // 3. Compare exponents.
        let mut exp_self = self.exponent.to_bigint() + 1_u8;
        let mut exp_rhs = rhs.exponent.to_bigint() + 1_u8;

        // Compare to make sure self has the larger exponent.
        if exp_self < exp_rhs {
            std::mem::swap(&mut self, &mut rhs);
            std::mem::swap(&mut exp_self, &mut exp_rhs);
        }

        let mant_self = self.fraction.append_bool_in_place(true);
        let mant_rhs = rhs.fraction.append_bool_in_place(true);

        // 4. Shift smaller mantissa if necessary.
        let exp_diff = &exp_self - exp_rhs;
        assert!(
            exp_diff >= BigInt::ZERO,
            "Self exponent should be larger/equal"
        );
        let mant_rhs =
            mant_rhs.shift_fixed(exp_diff.try_into().expect("Exponent difference too large"));

        assert_eq!(mant_self.len(), 53, "Mantissa length should be 53 bits");
        assert_eq!(mant_rhs.len(), 53, "Mantissa length should be 53 bits");

        // 5. Add mantissas.
        let mut mantissa_result = mant_self + mant_rhs;
        // 6. Normalize mantissa and adjust exponent if necessary.
        if mantissa_result.len() > 53 {
            // Need to normalize
            exp_self += 1_u8;
            mantissa_result = mantissa_result.shift_fixed(1);
        }

        // 7. Grow exponent if necessary. (no limit on size)
        let exp_self = exp_self - 1_u8;
        let exponent = grow_exponent(exp_self, self.exponent.len());
        let fraction = mantissa_result.truncate(52);
        FlexFloat {
            sign: self.sign,
            exponent,
            fraction,
        }
    }
}

impl<B: BitArray> Sub for FlexFloat<B> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.sign != rhs.sign {
            return self + (-rhs);
        }

        // 0. Handle special cases (NaN, Infinity).
        match (self.is_nan(), rhs.is_nan()) {
            (true, _) => return self,
            (_, true) => return rhs,
            _ => {}
        }

        match (self.is_infinity(), rhs.is_infinity(), self.sign == rhs.sign) {
            (true, true, true) => return FlexFloat::new_nan(), // inf - inf = Nan
            (true, true, false) => return self,                // inf - -inf = inf
            (true, false, _) => return self,                   // inf - x = inf
            (false, true, _) => return -rhs,                   // x - inf = -inf
            _ => {}
        }

        if self.abs() < rhs.abs() {
            return -(rhs - self);
        }

        let mut exp_self = self.exponent.to_bigint() + 1_u8;
        let exp_rhs = rhs.exponent.to_bigint() + 1_u8;

        let mant_self = self.fraction.append_bool_in_place(true);
        let mant_rhs = rhs.fraction.append_bool_in_place(true);

        // 4. Shift smaller mantissa if necessary.
        let exp_diff = exp_self.clone() - exp_rhs;
        let mant_rhs =
            mant_rhs.shift_fixed(exp_diff.try_into().expect("Exponent difference too large"));

        assert_eq!(mant_self.len(), 53, "Mantissa length should be 53 bits");
        assert_eq!(mant_rhs.len(), 53, "Mantissa length should be 53 bits");

        // 5. Subtract mantissas.
        let mut mantissa_result = mant_self - mant_rhs;

        // Find leading zeros for normalization
        let msb_pos = mantissa_result.iter_bits().rposition(|b| b);
        match msb_pos {
            Some(msb_pos) => {
                let shift = msb_pos as isize - 52;
                mantissa_result = mantissa_result.shift_fixed(shift);
                exp_self += BigInt::from(shift);
            }
            None => return Self::new_zero_with_sign(self.sign),
        }

        // 7. Grow exponent if necessary. (no limit on size)
        let exp_self = exp_self - 1_u8;
        let exponent = grow_exponent(exp_self, self.exponent.len());

        let fraction = mantissa_result.truncate(52);
        FlexFloat {
            sign: self.sign,
            exponent,
            fraction,
        }
    }
}

impl<B: BitArray> Mul for FlexFloat<B> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // 0. Handle special cases (NaN, Infinity).
        if self.is_nan() || rhs.is_nan() {
            return Self::new_nan();
        }

        if (self.is_infinity() && rhs.is_zero()) || (self.is_zero() && rhs.is_infinity()) {
            return Self::new_nan();
        }

        let sign = self.sign ^ rhs.sign;
        if self.is_zero() || rhs.is_zero() {
            return Self::new_zero_with_sign(sign);
        }

        if self.is_infinity() || rhs.is_infinity() {
            return Self::new_infinity(sign);
        }

        let exp_self = self.exponent.to_bigint() + 1_u8;
        let exp_rhs = rhs.exponent.to_bigint() + 1_u8;
        let mut exp_res = exp_self + exp_rhs;

        let mant_self = self.fraction.append_bool_in_place(true);
        let mant_rhs = rhs.fraction.append_bool_in_place(true);

        let mut mant_res = mant_self * mant_rhs;

        let msb_pos = mant_res.iter_bits().rposition(|b| b).unwrap();

        // Adjust exponent based on MSB position (normalize the result)
        exp_res += BigInt::from(msb_pos) - 104;

        // TODO: Rounding
        let lsb_pos = max(msb_pos - 52, 0);
        let msb_pos_end = min(msb_pos, mant_res.len());
        mant_res = mant_res.get_range(lsb_pos..msb_pos_end).unwrap();

        // 7. Grow exponent if necessary (no limit on size)
        let exp_res = exp_res - 1_u8;
        let max_exp_len = max(self.exponent.len(), rhs.exponent.len());

        let exponent = grow_exponent(exp_res, max_exp_len);
        let fraction = mant_res.truncate(52);

        Self {
            sign,
            exponent,
            fraction,
        }
    }
}

impl<B: BitArray> Div for FlexFloat<B> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
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

        if self.is_infinity() {
            if rhs.is_infinity() {
                return Self::new_nan(); // inf / inf = NaN
            }
            return Self::new_infinity(sign);
        }

        if rhs.is_infinity() {
            return Self::new_zero_with_sign(sign); // x / inf = 0
        }

        let exp_self = self.exponent.to_bigint() + 1_u8;
        let exp_rhs = rhs.exponent.to_bigint() + 1_u8;
        let mut exp_res = exp_self - exp_rhs;

        let mant_self = self.fraction.append_bool_in_place(true);
        let mant_rhs = rhs.fraction.append_bool_in_place(true);
        // We shift by 52 to produce 53 bits (1 implicit + 52 fraction) of precision.
        let mut mant_res = mant_self.shift_grow(52) / mant_rhs;

        // If result is zero, return signed zero.
        if mant_res.iter_bits().all(|b| !b) {
            return Self::new_zero_with_sign(sign);
        }

        // Find MSB position and normalize so the MSB sits at index 52.
        let msb_pos = mant_res.iter_bits().rposition(|b| b).unwrap();
        let shift = msb_pos as isize - 52;
        mant_res = mant_res.shift_fixed(shift);
        exp_res += BigInt::from(shift);

        let lsb_pos = max(msb_pos.saturating_sub(52), 0);
        let msb_pos_end = min(msb_pos + 1, mant_res.len());
        let mant_res_final = mant_res.get_range(lsb_pos..msb_pos_end).unwrap();

        // 7. Grow exponent if necessary (no limit on size)
        let exp_res = exp_res - 1_u8;
        let max_exp_len = max(self.exponent.len(), rhs.exponent.len());
        let exponent = grow_exponent(exp_res, max_exp_len);
        let fraction = mant_res_final.truncate(52);

        Self {
            sign,
            exponent,
            fraction,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use super::*;
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
        assert!(!c.is_infinity(), "Result should not overflow");
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

        for _ in 0..n_experiments {
            let a = random_f64(&mut rng);
            let b = random_f64(&mut rng);
            let expected = a + b;

            let fa = FlexFloat::from(a);
            let fb = FlexFloat::from(b);
            let fc = fa + fb;

            if expected.is_infinite() {
                // TODO: check fc is larger tha f64::MAX)
                continue;
            }

            if expected.is_nan() {
                assert!(fc.is_nan());
                continue;
            }

            if let Some(result) = fc.to_f64() {
                assert_almost_eq(
                    result,
                    expected,
                    format!("Failed on {a:.5e} + {b:.5e}").as_str(),
                );
            } else {
                // TODO: Check is valid
            }
        }
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
        assert!(!c.is_infinity(), "Result should not overflow");
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

        for _ in 0..n_experiments {
            let a = random_f64(&mut rng);
            let b = random_f64(&mut rng);
            let expected = a - b;

            let fa = FlexFloat::from(a);
            let fb = FlexFloat::from(b);
            let fc = fa - fb;

            if expected.is_infinite() {
                // TODO: check fc is larger tha f64::MAX)
                continue;
            }

            if expected.is_nan() {
                assert!(fc.is_nan());
                continue;
            }

            if let Some(result) = fc.to_f64() {
                assert_almost_eq(
                    result,
                    expected,
                    format!("Failed on {a:.5e} - {b:.5e}").as_str(),
                );
            } else {
                // TODO: Check is valid
            }
        }
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
        assert!(!c.is_infinity(), "Result should not overflow");
        assert!(c.exponent.len() > 11, "Exponent should have grown");

        // // TODO: Test wierd edge case
        // let a = -5.18059e300;
        // let b = 1.97397e-308;
        // let c = FlexFloat::from(a) * FlexFloat::from(b);
        // assert_eq!(c.to_f64(), Some(a * b));

        for _ in 0..n_experiments {
            let a = random_f64(&mut rng);
            let b = random_f64(&mut rng);
            let expected = a * b;

            let fa = FlexFloat::from(a);
            let fb = FlexFloat::from(b);
            let fc = fa * fb;

            // TODO: Find why large exponents, the precisions fails
            if a.abs().max(b.abs()) > 1e270 || a.abs().min(b.abs()) < 1e-270 {
                continue;
            }

            if expected.is_infinite() {
                // TODO: check fc is larger tha f64::MAX)
                continue;
            }

            if expected.is_nan() {
                assert!(fc.is_nan());
                continue;
            }

            if let Some(result) = fc.to_f64() {
                assert_almost_eq(
                    result,
                    expected,
                    format!("Failed on {a:.5e} * {b:.5e}").as_str(),
                );
            } else {
                // TODO: Check is valid
            }
        }
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
        assert!(!c.is_infinity(), "Result should not overflow");
        assert!(c.exponent.len() > 11, "Exponent should have grown");

        for _ in 0..n_experiments {
            let a = random_f64(&mut rng);
            let b = random_f64(&mut rng);
            let expected = a / b;

            let fa = FlexFloat::from(a);
            let fb = FlexFloat::from(b);
            let fc = fa / fb;

            // TODO: Find why large exponents, the precisions fails
            if a.abs().max(b.abs()) > 1e270 || a.abs().min(b.abs()) < 1e-270 {
                continue;
            }

            if expected.is_infinite() {
                continue;
            }

            if expected.is_nan() {
                assert!(fc.is_nan());
                continue;
            }

            if let Some(result) = fc.to_f64() {
                assert_almost_eq(
                    result,
                    expected,
                    format!("Failed on {a:.5e} / {b:.5e}").as_str(),
                );
            } else {
                // TODO: Check is valid
            }
        }
    }
}
