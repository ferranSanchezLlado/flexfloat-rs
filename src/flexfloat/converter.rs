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

use crate::bitarray::{BitArray, DefaultBitArray};
use crate::flexfloat::FlexFloat;

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
        if self.exponent.len() < 11 || self.fraction.len() < 52 {
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
}

/// Automatic conversion from f64 to FlexFloat using the default bit array.
///
/// This implementation provides seamless conversion from standard IEEE 754
/// double-precision floats to FlexFloat using the default BitArray implementation.
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
///
/// let flex: FlexFloat<_> = 3.14159.into();
/// // or equivalently:
/// let flex = FlexFloat::from(3.14159);
/// ```
impl From<f64> for FlexFloat<DefaultBitArray> {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

/// Automatic conversion from FlexFloat to f64 using the default bit array.
///
/// This implementation provides seamless conversion from FlexFloat back to
/// standard IEEE 754 double-precision floats. Panics if the FlexFloat cannot
/// be represented as an f64 (e.g., insufficient exponent or fraction bits).
///
/// # Panics
///
/// Panics if `to_f64()` returns None, indicating the FlexFloat components
/// are incompatible with IEEE 754 format.
///
/// # Examples
///
/// ```rust
/// use flexfloat::FlexFloat;
///
/// let flex = FlexFloat::from(2.71828);
/// let f64_val: f64 = flex.into();
/// assert_eq!(f64_val, 2.71828);
/// ```
impl From<FlexFloat<DefaultBitArray>> for f64 {
    fn from(value: FlexFloat<DefaultBitArray>) -> Self {
        value.to_f64().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use rand::Rng;
    use rstest::rstest;

    use crate::tests::n_experiments;

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
            let value: f64 = rng.random();
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
            let value: f64 = rng.random();
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
}
