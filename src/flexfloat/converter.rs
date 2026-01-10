//! # Conversion Utilities
//!
//! Provides conversion functions between FlexFloat and standard IEEE 754 double-precision floats.
//! Handles the mapping between IEEE 754 bit layout and FlexFloat's component structure.
//!
//! ## Overview
//!
//! This module implements bidirectional conversion between `f64` and `FlexFloat`,
//! preserving the exact bit patterns and special values defined by IEEE 754.
//!
//! ## IEEE 754 Layout
//!
//! Standard 64-bit IEEE 754 double-precision format:
//! ```text
//! [Sign: 1 bit][Exponent: 11 bits][Fraction: 52 bits]
//! Bit:  63      62           52   51            0
//! ```
//!
//! ## Conversion Features
//!
//! - **Bit-perfect accuracy**: Exact preservation of IEEE 754 bit patterns
//! - **Special value support**: Correct handling of ±0, ±∞, and NaN
//! - **Lossless conversion**: No precision loss for values within f64 range
//! - **Automatic trait implementation**: Seamless From/Into trait integration
//!
//! ## Examples
//!
//! ```rust
//! use flexfloat::FlexFloat;
//!
//! // Convert from f64
//! let original = 3.141592653589793;
//! let flex = FlexFloat::from(original);
//!
//! // Convert back to f64
//! let recovered: f64 = flex.into();
//! assert_eq!(original, recovered);
//!
//! // Special values
//! let inf_flex = FlexFloat::from(f64::INFINITY);
//! assert!(inf_flex.is_infinity());
//! ```

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Signed, Zero};

use crate::bitarray::{BitArray, DefaultBitArray};
use crate::flexfloat::{FlexFloat, grow_exponent};

impl<B: BitArray> FlexFloat<B> {
    /// Creates a FlexFloat from an IEEE 754 double-precision float.
    ///
    /// Converts the 64-bit IEEE 754 representation into FlexFloat components,
    /// extracting the sign, exponent, and fraction fields according to the
    /// standard bit layout.
    ///
    /// # Arguments
    ///
    /// * `value` - The f64 value to convert
    ///
    /// # Returns
    ///
    /// A new FlexFloat representing the same value
    ///
    /// # IEEE 754 Bit Extraction
    ///
    /// - Bit 63: Sign bit
    /// - Bits 62-52: 11-bit exponent field  
    /// - Bits 51-0: 52-bit fraction field
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let flex: FlexFloat<DefaultBitArray> = FlexFloat::from_f64(2.5);
    /// assert!(!flex.sign());
    /// assert_eq!(flex.exponent().len(), 11);
    /// assert_eq!(flex.fraction().len(), 52);
    /// ```
    pub fn from_f64(value: f64) -> Self {
        let bits = B::from_f64(value);

        Self {
            sign: *bits.get(63).unwrap(),
            exponent: bits.get_range(52..63).unwrap(),
            fraction: bits.get_range(0..52).unwrap(),
        }
    }

    /// Converts this FlexFloat to an IEEE 754 double-precision float.
    ///
    /// Reconstructs the 64-bit IEEE 754 representation from the FlexFloat
    /// components. Returns None if the exponent or fraction fields are too
    /// small to represent a valid IEEE 754 double.
    ///
    /// # Returns
    ///
    /// Some(f64) if conversion is possible, None if components are incompatible
    ///
    /// # Requirements
    ///
    /// - Exponent must have at least 11 bits
    /// - Fraction must have at least 52 bits
    /// - Only the first 11 exponent bits and 52 fraction bits are used
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::FlexFloat;
    ///
    /// let flex = FlexFloat::from(3.14159);
    /// let f64_val = flex.to_f64().unwrap();
    /// assert_eq!(f64_val, 3.14159);
    /// ```
    pub fn to_f64(&self) -> Option<f64> {
        if self.exponent.len() != 11 || self.fraction.len() != 52 {
            return None;
        }

        let mut bits: Vec<bool> = vec![false; 64];
        bits[63] = self.sign;
        bits[52..63].copy_from_slice(&self.exponent.to_bits()[0..11]);
        bits[0..52].copy_from_slice(&self.fraction.to_bits()[0..52]);
        Some(
            B::from_bits(bits.as_slice())
                .to_float()
                .expect("Underlaying BitArray cannot convert to f64"),
        )
    }

    pub fn from_int(value: BigInt) -> Self {
        if value.is_zero() {
            return Self::new_zero();
        }

        let sign = value.is_negative();
        let abs_value = value.abs().to_biguint().unwrap();

        // Find the position of the most significant bit
        let bit_length = abs_value.bits() as i64 - 1;

        // The format represents: value = (1 + fraction/2^fraction_bits) * 2^exponent
        // where exponent is stored as (actual_exponent - 1)
        let stored_exponent_value = BigInt::from(bit_length - 1);
        let exponent: B = grow_exponent(stored_exponent_value, 11);

        // Extract the fractional part (all bits except the MSB)
        let msb_mask = BigUint::one() << bit_length;
        let fraction_bits = &abs_value & (msb_mask - 1u8);

        let fraction = B::from_biguint(&fraction_bits);

        let missing_zeros = bit_length as usize - fraction.len();
        let fraction = fraction.append_repeated(false, missing_zeros);

        let shift = 52 - fraction.len() as isize;
        let fraction = if shift >= 0 {
            fraction.shift_grow(shift)
        } else {
            let shift_abs = shift.unsigned_abs();

            let lsb = *fraction.get(shift_abs).unwrap();
            let guard = *fraction.get(shift_abs - 1).unwrap();
            let rest = fraction.iter_bits().take(shift_abs - 1).any(|el| el);

            // guard && rest || lsb && guard && !rest
            let rounding = guard && (lsb || rest);

            fraction.shift_fixed(-shift).truncate(52) + B::from_bits(&[rounding])
        };

        debug_assert!(exponent.len() >= 11);
        debug_assert_eq!(fraction.len(), 52);

        Self {
            sign,
            exponent,
            fraction,
        }
    }

    pub fn to_int(&self) -> Option<BigInt> {
        if self.is_nan() || self.is_infinity() {
            return None;
        }

        if self.is_zero() {
            return Some(BigInt::ZERO);
        }

        let exponent = self.exponent.to_bigint() + 1;

        // Fast path for exponent < 0
        if exponent < BigInt::ZERO {
            return Some(BigInt::ZERO);
        }

        // Get the mantissa (1.fraction)
        let fraction = self.fraction.to_biguint();
        let fraction_bits = self.fraction.len();

        // Add the implicit leading 1
        let fraction = fraction + (BigUint::one() << fraction_bits);

        // Calculate the integer value
        // The value is fraction * 2^(exponent - fraction_bits)
        let shift_amount = isize::try_from(BigInt::from(fraction_bits) - exponent).unwrap();

        let int_value = if shift_amount > 0 {
            fraction >> shift_amount
        } else {
            fraction << -shift_amount
        };

        let sign = if self.sign { Sign::Minus } else { Sign::Plus };
        Some(BigInt::from_biguint(sign, int_value))
    }
}

impl From<f64> for FlexFloat<DefaultBitArray> {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl From<BigInt> for FlexFloat<DefaultBitArray> {
    fn from(value: BigInt) -> Self {
        Self::from_int(value)
    }
}

impl<B: BitArray> From<FlexFloat<B>> for f64 {
    #[track_caller]
    fn from(value: FlexFloat<B>) -> Self {
        value.to_f64().unwrap()
    }
}

impl<B: BitArray> From<FlexFloat<B>> for BigInt {
    #[track_caller]
    fn from(value: FlexFloat<B>) -> Self {
        value.to_int().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use num_traits::{FromPrimitive, ToPrimitive};
    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::tests::*;

    fn test_from_f64(mut rng: impl Rng, n_experiments: usize) {
        let special_values = [
            (0.0, false, vec![false; 11], vec![false; 52]),
            (-0.0, true, vec![false; 11], vec![false; 52]),
            (f64::INFINITY, false, vec![true; 11], vec![false; 52]),
            (f64::NEG_INFINITY, true, vec![true; 11], vec![false; 52]),
        ];

        for (value, sign, exponent, fraction) in special_values {
            let ff = FlexFloat::from(value);
            assert_eq!(ff.sign, sign);
            assert_eq!(ff.exponent.to_bits(), exponent);
            assert_eq!(ff.fraction.to_bits(), fraction);
        }

        let ff = FlexFloat::from(f64::NAN);
        assert!(ff.is_nan());

        for _ in 0..n_experiments {
            let value = random_f64(&mut rng);
            let ff = FlexFloat::from(value);
            assert_eq!(value.is_sign_negative(), ff.sign);
            assert_eq!(value.is_nan(), ff.is_nan());
        }
    }

    fn test_to_f64(mut rng: impl Rng, n_experiments: usize) {
        let special_values = [
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
            f64::NAN,
            f64::MAX,
            f64::MIN,
        ];

        for value in special_values {
            let ff = FlexFloat::from(value);
            let converted_value: f64 = ff.into();
            if value.is_nan() {
                assert!(converted_value.is_nan());
            } else {
                assert_eq!(value, converted_value);
            }
        }

        let ff = FlexFloat {
            sign: false,
            exponent: DefaultBitArray::ones(11),
            fraction: DefaultBitArray::from_bits(&[true; 52]),
        };
        let converted_value: f64 = ff.into();
        assert!(converted_value.is_nan());

        for _ in 0..n_experiments {
            let value = random_f64(&mut rng);
            let ff = FlexFloat::from(value);
            let converted_value: f64 = ff.into();
            if value.is_nan() {
                assert!(converted_value.is_nan());
            } else {
                assert_eq!(value, converted_value);
            }
        }
    }

    #[rstest]
    fn test_conversion(mut rng: impl Rng, n_experiments: usize) {
        test_from_f64(&mut rng, n_experiments);
        test_to_f64(&mut rng, n_experiments);
    }

    #[rstest]
    fn test_to_int(mut rng: impl Rng, n_experiments: usize) {
        let special_values = [
            FlexFloat::nan(),
            FlexFloat::pos_infinity(),
            FlexFloat::neg_infinity(),
        ];

        for ff in special_values {
            let value = ff.to_int();
            assert!(
                value.is_none(),
                "Conversion of {:?} to int should return None",
                ff
            );
        }

        let values = [0.1, 0.5, 0.9, 1.0, -0.1, -0.5, 123.45, 1234.0];
        for v in values {
            let ff = FlexFloat::from(v);
            let converted_value = ff.to_int().unwrap();

            let expected_value = v as i64;
            assert_eq!(
                BigInt::from(expected_value),
                converted_value,
                "to_int({v:#?})",
            );
        }

        for _ in 0..n_experiments {
            let value = random_f64(&mut rng);
            let ff = FlexFloat::from(value);

            let converted_value = ff.to_int();
            let expected_value = BigInt::from_f64(value);

            assert_eq!(converted_value, expected_value, "to_int({value:#?})");
        }
    }

    #[rstest]
    fn test_from_int(mut rng: impl Rng, n_experiments: usize) {
        let int_values = [0_i64, 1, -1, 1234, -9876, 10213213];

        for &int_val in &int_values {
            let value = int_val as f64 as i64;
            let ff = FlexFloat::from(BigInt::from(int_val));

            let converted_value = ff.to_int().unwrap().to_i64().unwrap();
            assert_eq!(converted_value, value, "from_int({})", int_val);
        }

        for _ in 0..n_experiments {
            let int_val: i64 = rng.random();

            let value = int_val as f64 as i64;
            let ff = FlexFloat::from(BigInt::from(int_val));
            let converted_value = ff.clone().to_int().unwrap().to_i64().unwrap();
            dbg!(FlexFloat::from(int_val as f64), ff);

            assert_eq!(converted_value, value, "from_int({})", int_val);
        }
    }
}
