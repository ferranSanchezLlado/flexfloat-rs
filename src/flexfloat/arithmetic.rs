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
use std::ops::{Add, Mul, Neg, Sub};

use num_bigint::BigInt;

use crate::bitarray::BitArray;
use crate::flexfloat::FlexFloat;

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

impl<B: BitArray> FlexFloat<B> {
    /// Calculates the minimum exponent bit width needed to represent a given exponent value.
    ///
    /// This function implements the core logic for exponent growth, determining
    /// how many bits are needed to represent an exponent value in two's complement form.
    ///
    /// # Arguments
    ///
    /// * `exp` - The exponent value as a BigInt
    /// * `current_len` - Current exponent field width in bits
    ///
    /// # Returns
    ///
    /// Minimum number of bits needed to represent the exponent
    ///
    /// # Algorithm
    ///
    /// Uses two's complement representation where for n bits:
    /// - Minimum value: -2^(n-1)
    /// - Maximum value: 2^(n-1) - 1
    ///
    /// Grows the bit width until the exponent fits within the representable range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// // This is an internal function, but conceptually:
    /// // For exponent value 1000:
    /// // - 11 bits: range [-1024, 1023] ✓ (fits)
    /// // - 10 bits: range [-512, 511] ✗ (doesn't fit)
    /// ```
    fn grow_exponent_bits(exp: &BigInt, current_len: usize) -> usize {
        let mut exponent_length = current_len;
        loop {
            let half = 1 << (exponent_length - 1);
            let min_exponent = BigInt::from(-half);
            let max_exponent = BigInt::from(half - 1);

            if &min_exponent <= exp && exp <= &max_exponent {
                break;
            }
            exponent_length += 1;
        }

        exponent_length
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
        let exp_diff = exp_self.clone() - exp_rhs.clone();
        assert!(
            exp_diff >= BigInt::ZERO,
            "Self exponent should be larger/equal"
        );
        let mant_rhs = mant_rhs.shift(exp_diff.try_into().expect("Exponent difference too large"));

        assert_eq!(mant_self.len(), 53, "Mantissa length should be 53 bits");
        assert_eq!(mant_rhs.len(), 53, "Mantissa length should be 53 bits");

        // 5. Add mantissas.
        let mut mantissa_result = B::zeros(53);
        let mut carry: u8 = 0;
        for i in 0..53 {
            let self_bit = *mant_self.get(i).unwrap() as u8;
            let rhs_bit = *mant_rhs.get(i).unwrap() as u8;
            let sum = self_bit + rhs_bit + carry;

            *mantissa_result.get_mut(i).unwrap() = !sum.is_multiple_of(2);
            carry = sum / 2;
            assert!(carry <= 1);
        }

        // 6. Normalize mantissa and adjust exponent if necessary.
        if carry > 0 {
            mantissa_result = mantissa_result.shift(1);
            exp_self += 1_u8;
        }

        // 7. Grow exponent if necessary. (no limit on size)
        let exp_result_length = Self::grow_exponent_bits(&exp_self, self.exponent.len());

        let exponent_result = B::from_bigint(&(exp_self - 1_u8), exp_result_length)
            .expect("Exponent lenght should have grown");
        let fraction_result = B::from_bits(&mantissa_result.to_bits()[..52]);
        FlexFloat {
            sign: self.sign,
            exponent: exponent_result,
            fraction: fraction_result,
        }
    }
}

impl<B: BitArray> Sub for FlexFloat<B> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.sign != rhs.sign {
            return self - (-rhs);
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

        if self < rhs {
            return -(rhs - self);
        }

        let mut exp_self = self.exponent.to_bigint() + 1_u8;
        let exp_rhs = rhs.exponent.to_bigint() + 1_u8;

        let mant_self = self.fraction.append_bool_in_place(true);
        let mant_rhs = rhs.fraction.append_bool_in_place(true);

        // 4. Shift smaller mantissa if necessary.
        let exp_diff = exp_self.clone() - exp_rhs.clone();
        assert!(
            exp_diff >= BigInt::ZERO,
            "Self exponent should be larger/equal"
        );
        let mant_rhs = mant_rhs.shift(exp_diff.try_into().expect("Exponent difference too large"));

        assert_eq!(mant_self.len(), 53, "Mantissa length should be 53 bits");
        assert_eq!(mant_rhs.len(), 53, "Mantissa length should be 53 bits");

        // 5. Add mantissas.
        let mut mantissa_result = B::zeros(53);
        let mut borrow: i8 = 0;
        for i in 0..53 {
            let self_bit = *mant_self.get(i).unwrap() as i8;
            let rhs_bit = *mant_rhs.get(i).unwrap() as i8;
            let sum = self_bit - rhs_bit - borrow;

            *mantissa_result.get_mut(i).unwrap() = (sum + 2) % 2 != 0;
            borrow = (sum < 0) as i8;
            assert!(borrow <= 1);
        }

        assert!(borrow == 0, "Should not have borrow after subtraction");

        // Find leading zeros for normalization
        let msb_pos = mantissa_result.iter_bits().rposition(|b| b);
        if let Some(msb_pos) = msb_pos {
            let shift = msb_pos as isize - 52;
            mantissa_result = mantissa_result.shift(shift);
            exp_self -= BigInt::from(shift.unsigned_abs());
        }

        // 7. Grow exponent if necessary. (no limit on size)
        let exp_result_length = Self::grow_exponent_bits(&exp_self, self.exponent.len());

        let exponent_result = B::from_bigint(&(exp_self - 1_u8), exp_result_length)
            .expect("Exponent lenght should have grown");
        let fraction_result = B::from_bits(&mantissa_result.to_bits()[..52]);
        FlexFloat {
            sign: self.sign,
            exponent: exponent_result,
            fraction: fraction_result,
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

        let mant_self_int = self.fraction.append_bool_in_place(true).to_biguint();
        let mant_rhs_int = rhs.fraction.append_bool_in_place(true).to_biguint();

        let mant_res_int = mant_self_int * mant_rhs_int;

        // Grow mantissa if necessary (it can be up to 106 bits)
        let mut mant_res = B::from_biguint_fixed(&mant_res_int, 106);

        let msb_pos = mant_res.iter_bits().rposition(|b| b).unwrap();

        // Adjust exponent based on MSB position (normalize the result)
        exp_res += BigInt::from(msb_pos) - 104;

        let lsb_pos = max(msb_pos - 52, 0);
        let msb_pos_end = min(msb_pos + 1, mant_res.len());
        mant_res = mant_res.get_range(lsb_pos..msb_pos_end).unwrap();

        // 7. Grow exponent if necessary (no limit on size)
        let max_exp_len = max(self.exponent.len(), rhs.exponent.len());
        let exp_result_length = Self::grow_exponent_bits(&exp_res, max_exp_len);

        let exp_res = B::from_bigint(&(exp_res - 1_u8), exp_result_length)
            .expect("Exponent lenght should have grown");
        let fraction_result = B::from_bits(&mant_res.to_bits()[..52]);

        Self {
            sign,
            exponent: exp_res,
            fraction: fraction_result,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::tests::*;

    const EPSILON: f64 = 1e-10;

    #[track_caller]
    fn assert_almost_eq(a: f64, b: f64, message: &str) {
        assert!(
            (a - b).abs() <= EPSILON,
            "{}: {} and {} differ more than {}",
            message,
            a,
            b,
            EPSILON
        );
    }

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

        for _ in 0..n_experiments {
            let a: f64 = rng.random();
            let b: f64 = rng.random();
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

            let result = fc.to_f64().expect("Result should fit in f64");
            assert_almost_eq(
                result,
                expected,
                format!("Failed on {} + {}", a, b).as_str(),
            );
        }
    }

    #[rstest]
    fn test_sub(mut rng: impl Rng, n_experiments: usize) {
        let a = 1.5;
        let b = 2.25;
        let c = FlexFloat::from(a) - FlexFloat::from(b);
        assert_eq!(c.to_f64(), Some(a - b));

        // Test overflow case
        // let a = FlexFloat::from(-f64::MAX);
        // let b = FlexFloat::from(f64::MAX / 2.0);
        // let c = a - b;
        // assert!(!c.is_infinity(), "Result should not overflow");
        // assert!(c.exponent.len() > 11, "Exponent should have grown");

        // Test wierd edge case
        let a = f64::MAX;
        let b = f64::MIN_POSITIVE;
        let c = FlexFloat::from(a) - FlexFloat::from(b);
        assert_eq!(c.to_f64(), Some(a - b));

        for _ in 0..n_experiments {
            let a: f64 = rng.random();
            let b: f64 = rng.random();
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

            let result = fc.to_f64().expect("Result should fit in f64");
            assert_almost_eq(
                result,
                expected,
                format!("Failed on {} - {}", a, b).as_str(),
            );
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

        // TODO: Test wierd edge case

        for _ in 0..n_experiments {
            let a: f64 = rng.random();
            let b: f64 = rng.random();
            let expected = a * b;

            let fa = FlexFloat::from(a);
            let fb = FlexFloat::from(b);
            let fc = fa * fb;

            if expected.is_infinite() {
                // TODO: check fc is larger tha f64::MAX)
                continue;
            }

            if expected.is_nan() {
                assert!(fc.is_nan());
                continue;
            }

            let result = fc.to_f64().expect("Result should fit in f64");
            assert_almost_eq(
                result,
                expected,
                format!("Failed on {} * {}", a, b).as_str(),
            );
        }
    }
}
