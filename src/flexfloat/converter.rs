//! # Conversion Utilities
//!
//! Provides conversion functions between FlexFloat and standard numeric types including
//! IEEE 754 double-precision floats and arbitrary-precision integers.
//!
//! ## Overview
//!
//! This module implements bidirectional conversion between `f64`, `BigInt`, and `FlexFloat`.
//! FlexFloat supports variable-width exponents (starting at 11 bits minimum for IEEE 754
//! compatibility) that can grow as needed, while maintaining exact bit patterns and
//! special values for standard IEEE 754 operations.
//!
//! ## IEEE 754 Layout
//!
//! Standard 64-bit IEEE 754 double-precision format:
//! ```text
//! Bit: 63      62-52           51-0
//!      [Sign] [Exponent(11)]  [Fraction(52)]
//! ```
//!
//! ## Conversion Features
//!
//! - **Bit-perfect accuracy**: Exact preservation of IEEE 754 bit patterns for f64 conversions
//! - **Variable exponent width**: Automatic exponent growth beyond 11 bits when needed
//! - **Special value support**: Correct handling of ±0, ±∞, and NaN
//! - **Lossless conversion**: No precision loss for values within f64 range
//! - **BigInt interoperability**: Full support for arbitrary-precision integer conversion
//! - **Automatic trait implementation**: Seamless From/Into trait integration
//!
//! ## Examples
//!
//! ```rust
//! use flexfloat::prelude::*;
//!
//! // Convert from f64
//! let original = 3.141592653589793;
//! let flex = FlexFloat::from(original);
//!
//! // Convert back to f64
//! let recovered: f64 = flex.try_into().unwrap();
//! assert_eq!(original, recovered);
//!
//! // Convert from BigInt and back
//! let int_value = num_bigint::BigInt::from(42);
//! let flex_from_int = FlexFloat::from(int_value.clone());
//! let recovered_int: num_bigint::BigInt = flex_from_int.try_into().unwrap();
//! assert_eq!(int_value, recovered_int);
//! ```

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Signed, Zero};

use crate::BoolBitArray;
use crate::bitarray::{BitArray, BitArrayConstruction, BitArrayConversion, DefaultBitArray};
use crate::flexfloat::error::{FlexFloatToF64Error, FlexFloatToIntError};
use crate::flexfloat::{FlexFloat, RoundedResult, grow_exponent, truncate_fraction};

impl<Frac: BitArray, Exp: BitArray> FlexFloat<Exp, Frac> {
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
    /// let flex: FlexFloat = FlexFloat::from_f64(2.5);
    /// assert!(!flex.sign());
    /// assert_eq!(flex.exponent().len(), 11);
    /// assert_eq!(flex.fraction().len(), 52);
    /// ```
    pub fn from_f64(value: f64) -> Self {
        let bits = Frac::from_f64(value);

        Self {
            sign: bits.get(63).unwrap(),
            exponent: bits.get_range(52..63).unwrap().convert_to(),
            fraction: bits.get_range(0..52).unwrap(),
        }
    }

    /// Creates a FlexFloat from a BigInt integer value.
    ///
    /// Converts a signed arbitrary-precision integer into FlexFloat representation
    /// by determining the appropriate exponent and fraction fields. The conversion
    /// follows IEEE 754 semantics for representing integers in floating-point format.
    ///
    /// The exponent field will automatically grow beyond 11 bits if needed to represent
    /// integers larger than what fits in IEEE 754 double-precision format.
    ///
    /// # Arguments
    ///
    /// * `value` - The BigInt value to convert
    ///
    /// # Returns
    ///
    /// A new FlexFloat representing the integer value with at least 11-bit exponent
    /// (grown as needed) and 52-bit fraction
    ///
    /// # Format Details
    ///
    /// The representation follows the formula:
    /// ```text
    /// value = (-1)^sign × (1 + fraction/2^52) × 2^(exponent+1)
    /// ```
    ///
    /// Where:
    /// - The exponent is stored as (bit_length - 2)
    /// - The fraction contains all bits except the most significant bit
    /// - Zero values are handled specially as positive zero
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    /// use num_bigint::BigInt;
    ///
    /// let neg_value = BigInt::from(-123);
    /// let neg_flex = FlexFloat::from(neg_value);
    /// ```
    pub fn from_int(value: BigInt) -> Self {
        if value.is_zero() {
            return Self::zero();
        }

        let sign = value.is_negative();
        let abs_value = value.abs().to_biguint().unwrap();

        // Find the position of the most significant bit
        let bit_length = abs_value.bits() as i64 - 1;

        // The format represents: value = (1 + fraction/2^fraction_bits) * 2^exponent
        // where exponent is stored as (actual_exponent - 1)
        let stored_exponent_value = BigInt::from(bit_length - 1);
        let exponent: Exp = grow_exponent(stored_exponent_value, 11);

        // Extract the fractional part (all bits except the MSB)
        let msb_mask = BigUint::one() << bit_length;
        let fraction_bits = &abs_value & (msb_mask - 1u8);

        let fraction = Frac::from_biguint(&fraction_bits);

        let missing_zeros = bit_length as usize - fraction.len();
        let fraction = fraction.append_repeated(false, missing_zeros);

        let RoundedResult { fraction, carry } = truncate_fraction(fraction, 52);
        debug_assert!(
            !carry,
            "from_int truncate_fraction should never carry (input is already 52 bits)"
        );

        debug_assert!(exponent.len() >= 11);
        debug_assert_eq!(fraction.len(), 52);

        Self {
            sign,
            exponent,
            fraction,
        }
    }

    /// The machine epsilon at this instance's mantissa width.
    ///
    /// Returns `2^(1 − mantissa_digits)` — the difference between 1 and the next
    /// representable value, computed for the fraction width of this specific instance.
    pub fn epsilon(&self) -> Self {
        // mantissa_digits = fraction_len + 1, so exponent = 1 - (fraction_len + 1) = -fraction_len
        let exp = -(self.fraction.len() as i32);
        Self::from_f64(2.0_f64.powi(exp))
    }
}

impl<Exp: BitArrayConversion, Frac: BitArrayConversion> FlexFloat<Exp, Frac> {
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
    /// - Exponent must be exactly 11 bits (IEEE 754 standard)
    /// - Fraction must be exactly 52 bits (IEEE 754 standard)
    /// - FlexFloat values with larger exponents cannot be converted to f64 without loss
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
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
            BoolBitArray { bits }
                .to_float()
                .expect("Underlaying BitArray cannot convert to f64"),
        )
    }

    /// Converts this FlexFloat to a BigInt integer value.
    ///
    /// Extracts the integer portion of the FlexFloat value by reconstructing
    /// the mantissa and applying the exponent. Fractional parts are truncated
    /// towards zero. Returns None for special values (NaN, ±∞).
    ///
    /// # Returns
    ///
    /// - `Some(BigInt)` - The integer value (truncated towards zero)
    /// - `None` - If the value is NaN or infinity
    ///
    /// # Conversion Behavior
    ///
    /// - **Fractional values**: Truncated towards zero (e.g., 3.7 → 3, -2.9 → -2)
    /// - **Zero**: Returns BigInt::ZERO
    /// - **Negative exponent**: Returns 0 (value too small to represent as integer)
    /// - **Special values**: Returns None
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    /// use num_bigint::BigInt;
    ///
    /// // Fractional value (truncated)
    /// let flex = FlexFloat::from(3.7_f64);
    /// assert_eq!(flex.to_int().unwrap(), BigInt::from(3));
    ///
    /// // Special values return None
    /// let inf = FlexFloat::pos_infinity();
    /// assert!(inf.to_int().is_none());
    /// ```
    pub fn to_int(&self) -> Option<BigInt> {
        if self.is_nan() || self.is_infinite() {
            return None;
        }

        if self.is_zero() {
            return Some(BigInt::ZERO);
        }

        let is_subnormal = self.exponent.is_zeros();
        let exponent = if is_subnormal {
            BigInt::from(-1022)
        } else {
            self.exponent.to_bigint() + 1
        };

        // Fast path for exponent < 0
        if exponent < BigInt::ZERO {
            return Some(BigInt::ZERO);
        }

        // Get the mantissa (1.fraction)
        let fraction = self.fraction.to_biguint();
        let fraction_bits = self.fraction.len();

        let fraction = if is_subnormal {
            fraction
        } else {
            fraction + (BigUint::one() << fraction_bits)
        };

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

/// Convert any `FlexFloat<Exp, Frac>` (including const-backend types like
/// `FlexFloat<StaticBitArray<11>, StaticBitArray<52>>`) to any same-backend target.
impl<Exp, Frac> FlexFloat<Exp, Frac>
where
    Exp: BitArrayConversion + 'static,
    Frac: BitArrayConversion + 'static,
{
    pub fn convert_to<E2: BitArrayConstruction + 'static, F2: BitArrayConstruction + 'static>(
        self,
    ) -> FlexFloat<E2, F2> {
        FlexFloat {
            sign: self.sign,
            exponent: self.exponent.convert_to(),
            fraction: self.fraction.convert_to(),
        }
    }
}

/// Byte-serialisation for `FlexFloat`.
///
/// The layout is: `[sign byte (0 or 1)] ++ exponent_bytes ++ fraction_bytes`.
/// The returned `usize` is the number of exponent bits, needed to reconstruct the value
/// via [`FlexFloat::from_le_bytes`] / [`FlexFloat::from_be_bytes`].
impl<Exp: BitArrayConversion, Frac: BitArrayConversion> FlexFloat<Exp, Frac> {
    /// Serialise to little-endian bytes.
    ///
    /// Returns `(bytes, exponent_bits)`. Pass both to [`from_le_bytes`](FlexFloat::from_le_bytes) to reconstruct.
    pub fn to_le_bytes(&self) -> (Vec<u8>, usize) {
        let mut bytes = vec![u8::from(self.sign)];
        let exp_bits = self.exponent.len();
        let exp_bytes = self.exponent.to_bytes();
        let frac_bytes = self.fraction.to_bytes();
        bytes.extend_from_slice(&exp_bytes);
        bytes.extend_from_slice(&frac_bytes);
        (bytes, exp_bits)
    }

    /// Serialise to big-endian bytes.
    ///
    /// Returns `(bytes, exponent_bits)`. Pass both to [`from_be_bytes`](FlexFloat::from_be_bytes) to reconstruct.
    pub fn to_be_bytes(&self) -> (Vec<u8>, usize) {
        let (mut le, exp_bits) = self.to_le_bytes();
        // reverse each field independently: sign stays, then flip exp slice and frac slice
        let exp_byte_count = exp_bits.div_ceil(8);
        let frac_byte_count = 52_usize.div_ceil(8); // fraction is always 52 bits
        let exp_start = 1;
        le[exp_start..exp_start + exp_byte_count].reverse();
        le[exp_start + exp_byte_count..exp_start + exp_byte_count + frac_byte_count].reverse();
        (le, exp_bits)
    }
}

impl<Frac: BitArrayConstruction, Exp: BitArrayConstruction> FlexFloat<Exp, Frac> {
    /// Deserialise from little-endian bytes produced by [`to_le_bytes`](FlexFloat::to_le_bytes).
    pub fn from_le_bytes(bytes: &[u8], exp_bits: usize) -> Self {
        let sign = bytes[0] != 0;
        let exp_byte_count = exp_bits.div_ceil(8);
        let frac_byte_count = 52_usize.div_ceil(8);
        let exponent = Exp::from_bytes(&bytes[1..1 + exp_byte_count], exp_bits);
        let fraction = Frac::from_bytes(
            &bytes[1 + exp_byte_count..1 + exp_byte_count + frac_byte_count],
            52,
        );
        Self {
            sign,
            exponent,
            fraction,
        }
    }

    /// Deserialise from big-endian bytes produced by [`to_be_bytes`](FlexFloat::to_be_bytes).
    pub fn from_be_bytes(bytes: &[u8], exp_bits: usize) -> Self {
        let exp_byte_count = exp_bits.div_ceil(8);
        let frac_byte_count = 52_usize.div_ceil(8);
        let mut le = bytes.to_vec();
        le[1..1 + exp_byte_count].reverse();
        le[1 + exp_byte_count..1 + exp_byte_count + frac_byte_count].reverse();
        Self::from_le_bytes(&le, exp_bits)
    }
}

/// Conversion from f64 to FlexFloat with default bit array.
impl From<f64> for FlexFloat<DefaultBitArray> {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

/// Conversion from f32 to FlexFloat with default bit array.
impl From<f32> for FlexFloat<DefaultBitArray> {
    fn from(value: f32) -> Self {
        Self::from_f64(value as f64)
    }
}

/// Conversion from i64 to FlexFloat with default bit array.
impl From<i64> for FlexFloat<DefaultBitArray> {
    fn from(value: i64) -> Self {
        Self::from_int(BigInt::from(value))
    }
}

/// Conversion from u64 to FlexFloat with default bit array.
impl From<u64> for FlexFloat<DefaultBitArray> {
    fn from(value: u64) -> Self {
        Self::from_int(BigInt::from(value))
    }
}

/// Conversion from i32 to FlexFloat with default bit array.
impl From<i32> for FlexFloat<DefaultBitArray> {
    fn from(value: i32) -> Self {
        Self::from_int(BigInt::from(value))
    }
}

/// Conversion from u32 to FlexFloat with default bit array.
impl From<u32> for FlexFloat<DefaultBitArray> {
    fn from(value: u32) -> Self {
        Self::from_int(BigInt::from(value))
    }
}

/// Conversion from BigInt to FlexFloat with default bit array.
impl From<BigInt> for FlexFloat<DefaultBitArray> {
    fn from(value: BigInt) -> Self {
        Self::from_int(value)
    }
}

/// Fallible conversion from `FlexFloat` to `f64`.
///
/// Returns `Err(FlexFloatToF64Error)` when the value cannot be represented in `f64`.
impl<Exp: BitArrayConversion, Frac: BitArrayConversion> TryFrom<FlexFloat<Exp, Frac>> for f64 {
    type Error = FlexFloatToF64Error;

    fn try_from(value: FlexFloat<Exp, Frac>) -> Result<Self, Self::Error> {
        value.to_f64().ok_or(FlexFloatToF64Error::ExponentOverflow)
    }
}

/// Fallible conversion from `FlexFloat` to `BigInt`.
///
/// Returns `Err(FlexFloatToIntError)` when the value is NaN, infinite, or has a fractional part.
impl<Exp: BitArrayConversion, Frac: BitArrayConversion> TryFrom<FlexFloat<Exp, Frac>> for BigInt {
    type Error = FlexFloatToIntError;

    fn try_from(value: FlexFloat<Exp, Frac>) -> Result<Self, Self::Error> {
        if value.is_nan() {
            return Err(FlexFloatToIntError::NotANumber);
        }
        if value.is_infinite() {
            return Err(FlexFloatToIntError::Infinite);
        }
        value.to_int().ok_or(FlexFloatToIntError::NotAnInteger)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::{FromPrimitive, ToPrimitive};
    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::bitarray::{BitArrayConstruction, BitArrayConversion};
    use crate::test_support::*;

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
            let converted_value: f64 = ff.try_into().unwrap_or(f64::NAN);
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
        let converted_value: f64 = ff.try_into().unwrap_or(f64::NAN);
        assert!(converted_value.is_nan());

        for _ in 0..n_experiments {
            let value = random_f64(&mut rng);
            let ff = FlexFloat::from(value);
            let converted_value: f64 = ff.try_into().unwrap_or(f64::NAN);
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
            FlexFloat::<DefaultBitArray>::nan(),
            FlexFloat::<DefaultBitArray>::pos_infinity(),
            FlexFloat::<DefaultBitArray>::neg_infinity(),
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
