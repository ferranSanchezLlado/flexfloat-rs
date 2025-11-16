//! # FlexFloat Module
//!
//! Core implementation of the FlexFloat arbitrary precision floating-point type.
//! Provides IEEE 754-compatible floating-point operations with growable exponents.
//!
//! ## Overview
//!
//! FlexFloat extends the IEEE 754 double-precision format by allowing the exponent
//! field to grow dynamically while maintaining a fixed 52-bit fraction. This enables
//! representation of numbers far beyond the range of standard double-precision floats.
//!
//! ## Structure
//!
//! FlexFloat consists of three components:
//! - **Sign bit**: 1 bit indicating positive (false) or negative (true)
//! - **Exponent**: Variable-width signed integer (starts at 11 bits, grows as needed)
//! - **Fraction/Mantissa**: Fixed 52-bit fractional part for precision consistency
//!
//! ## Key Features
//!
//! - **Growable exponents**: Automatically expand to accommodate larger values
//! - **Fixed precision**: Consistent 52-bit mantissa maintains precision
//! - **IEEE 754 compatibility**: Full support for special values (±0, ±∞, NaN)
//! - **Seamless conversion**: Easy interop with standard f64 values
//! - **Generic backing**: Configurable bit array implementation
//!
//! ## Special Values
//!
//! FlexFloat supports all IEEE 754 special values:
//! - **Zero**: ±0.0 with separate positive and negative representations
//! - **Infinity**: ±∞ for overflow conditions
//! - **NaN**: Not-a-Number for undefined operations
//!
//! ## Examples
//!
//! ```rust
//! use flexfloat::FlexFloat;
//!
//! // Basic construction
//! let x = FlexFloat::from(3.14159);
//! let zero = FlexFloat::zero();
//! let inf = FlexFloat::pos_infinity();
//!
//! // Special value checks
//! assert!(zero.is_zero());
//! assert!(inf.is_infinity());
//!
//! // Component access
//! println!("Sign: {}, Exponent bits: {}, Fraction bits: {}",
//!          x.sign(), x.exponent().len(), x.fraction().len());
//! ```

use std::fmt::Debug;

use crate::bitarray::{BitArray, DefaultBitArray};

pub mod arithmetic;
pub mod cmp;
pub mod converter;

/// A flexible precision floating-point number with growable exponents.
///
/// FlexFloat represents floating-point numbers using a configurable bit array
/// backend, allowing for arbitrary precision arithmetic while maintaining
/// compatibility with IEEE 754 standards.
///
/// # Type Parameters
///
/// * `B` - The BitArray implementation used for internal storage
///
/// # Components
///
/// - `sign`: Boolean indicating number sign (false = positive, true = negative)
/// - `exponent`: Variable-width signed exponent field
/// - `fraction`: Fixed-width mantissa/fraction field (typically 52 bits)
///
/// # Memory Efficiency
///
/// The exponent field grows only when needed, starting at 11 bits (IEEE 754 standard)
/// and expanding to accommodate larger values. The fraction remains fixed at 52 bits
/// to maintain precision consistency.

#[derive(Clone)]
pub struct FlexFloat<B> {
    sign: bool,
    exponent: B,
    fraction: B,
}

impl<B: BitArray> FlexFloat<B> {
    /// Creates a new FlexFloat with the specified components.
    ///
    /// # Arguments
    ///
    /// * `sign` - Sign bit (false for positive, true for negative)
    /// * `exponent` - Exponent bit array
    /// * `fraction` - Fraction/mantissa bit array
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::{FlexFloat, bitarray::{BitArray, BoolBitArray}};
    ///
    /// let exp = BoolBitArray::from_bits(&[true, false, true]);
    /// let frac = BoolBitArray::zeros(52);
    /// let num = FlexFloat::new(false, exp, frac);
    /// ```
    pub fn new(sign: bool, exponent: B, fraction: B) -> Self {
        Self {
            sign,
            exponent,
            fraction,
        }
    }

    /// Creates a new FlexFloat representing zero with the specified sign.
    ///
    /// # Arguments
    ///
    /// * `sign` - Sign bit for the zero value
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::FlexFloat;
    ///
    /// let pos_zero = FlexFloat::new_zero_with_sign(false);
    /// let neg_zero = FlexFloat::new_zero_with_sign(true);
    /// ```
    pub fn new_zero_with_sign(sign: bool) -> Self {
        Self {
            sign,
            exponent: B::zeros(11),
            fraction: B::zeros(52),
        }
    }

    /// Creates a new FlexFloat representing positive zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::FlexFloat;
    ///
    /// let zero = FlexFloat::new_zero();
    /// assert!(zero.is_zero() && !zero.sign());
    /// ```
    pub fn new_zero() -> Self {
        Self::new_zero_with_sign(false)
    }

    /// Creates a new FlexFloat representing NaN (Not-a-Number).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::FlexFloat;
    ///
    /// let nan = FlexFloat::new_nan();
    /// assert!(nan.is_nan());
    /// ```
    pub fn new_nan() -> Self {
        Self {
            sign: false, // Sign doesn't matter for NaN
            exponent: B::ones(11),
            fraction: B::ones(52),
        }
    }

    /// Creates a new FlexFloat representing infinity with the specified sign.
    ///
    /// # Arguments
    ///
    /// * `sign` - Sign bit for the infinity value
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::FlexFloat;
    ///
    /// let pos_inf = FlexFloat::new_infinity(false);
    /// let neg_inf = FlexFloat::new_infinity(true);
    /// ```
    pub fn new_infinity(sign: bool) -> Self {
        Self {
            sign,
            exponent: B::ones(11),
            fraction: B::zeros(52),
        }
    }

    /// Checks if the exponent represents a special value (all ones).
    ///
    /// Special exponents indicate infinity or NaN values in IEEE 754 format.
    fn is_special_exponent(&self) -> bool {
        self.exponent.iter_bits().all(|b| b)
    }

    /// Returns true if this FlexFloat represents NaN (Not-a-Number).
    ///
    /// A value is NaN if it has a special exponent (all ones) and a non-zero fraction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::FlexFloat;
    ///
    /// let nan = FlexFloat::nan();
    /// assert!(nan.is_nan());
    ///
    /// let normal = FlexFloat::from(3.14);
    /// assert!(!normal.is_nan());
    /// ```
    pub fn is_nan(&self) -> bool {
        self.is_special_exponent() && self.fraction.iter_bits().any(|b| b)
    }

    /// Returns true if this FlexFloat represents infinity.
    ///
    /// A value is infinity if it has a special exponent (all ones) and a zero fraction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::FlexFloat;
    ///
    /// let inf = FlexFloat::pos_infinity();
    /// assert!(inf.is_infinity());
    ///
    /// let finite = FlexFloat::from(1.0);
    /// assert!(!finite.is_infinity());
    /// ```
    pub fn is_infinity(&self) -> bool {
        self.is_special_exponent() && self.fraction.iter_bits().all(|b| !b)
    }

    /// Returns true if this FlexFloat represents zero.
    ///
    /// A value is zero if both the exponent and fraction are all zeros.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::FlexFloat;
    ///
    /// let zero = FlexFloat::zero();
    /// assert!(zero.is_zero());
    ///
    /// let non_zero = FlexFloat::from(0.1);
    /// assert!(!non_zero.is_zero());
    /// ```
    pub fn is_zero(&self) -> bool {
        self.exponent.iter_bits().all(|b| !b) && self.fraction.iter_bits().all(|b| !b)
    }

    /// Returns the sign bit of this FlexFloat.
    ///
    /// # Returns
    ///
    /// false for positive numbers, true for negative numbers
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::FlexFloat;
    ///
    /// let pos = FlexFloat::from(3.14);
    /// let neg = FlexFloat::from(-2.71);
    /// assert!(!pos.sign());
    /// assert!(neg.sign());
    /// ```
    pub fn sign(&self) -> bool {
        self.sign
    }

    /// Returns a reference to the exponent bit array.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::FlexFloat;
    ///
    /// let num = FlexFloat::from(8.0);
    /// println!("Exponent has {} bits", num.exponent().len());
    /// ```
    pub fn exponent(&self) -> &B {
        &self.exponent
    }

    /// Returns a reference to the fraction bit array.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::FlexFloat;
    ///
    /// let num = FlexFloat::from(3.14159);
    /// println!("Fraction has {} bits", num.fraction().len());
    /// ```
    pub fn fraction(&self) -> &B {
        &self.fraction
    }
}

impl FlexFloat<DefaultBitArray> {
    /// Creates a new `FlexFloat` instance representing zero.
    ///
    /// This is a convenience constructor that creates a `FlexFloat` with the value of zero
    /// using the default bit array representation.
    ///
    /// # Returns
    ///
    /// A new `FlexFloat<DefaultBitArray>` instance with the value zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let zero = FlexFloat::zero();
    /// assert_eq!(zero.to_f64(), 0.0);
    /// ```
    pub fn zero() -> Self {
        Self::new_zero()
    }

    /// Creates a new `FlexFloat` instance representing zero with the specified sign.
    ///
    /// This is a convenience constructor that creates a `FlexFloat` with the value of 0
    /// using the default bit array representation.
    ///
    /// # Arguments
    ///
    /// * `sign` - The sign bit (false for +0, true for -0)
    ///
    /// # Returns
    ///
    /// A new `FlexFloat<DefaultBitArray>` instance representing zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let pos_zero = FlexFloat::zero_with_sign(false);
    /// let neg_zero = FlexFloat::zero_with_sign(true);
    /// assert!(pos_zero.is_zero() && !pos_zero.sign());
    /// assert!(neg_zero.is_zero() && neg_zero.sign());
    /// ```
    pub fn zero_with_sign(sign: bool) -> Self {
        Self::new_zero_with_sign(sign)
    }

    /// Creates a new `FlexFloat` instance representing NaN (Not-a-Number).
    ///
    /// This is a convenience constructor that creates a `FlexFloat` with the value of NaN
    /// using the default bit array representation.
    ///
    /// # Returns
    ///
    /// A new `FlexFloat<DefaultBitArray>` instance representing NaN.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let nan = FlexFloat::nan();
    /// assert!(nan.is_nan());
    /// ```
    pub fn nan() -> Self {
        Self::new_nan()
    }

    /// Creates a new `FlexFloat` instance representing positive infinity.
    ///
    /// This is a convenience constructor that creates a `FlexFloat` with the value of +inf
    /// using the default bit array representation.
    ///
    /// # Returns
    ///
    /// A new `FlexFloat<DefaultBitArray>` instance representing positive infinity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let pos_inf = FlexFloat::pos_infinity();
    /// assert!(pos_inf.is_infinity());
    /// assert!(!pos_inf.sign());
    /// ```
    pub fn pos_infinity() -> Self {
        Self::new_infinity(false)
    }

    /// Creates a new `FlexFloat` instance representing negative infinity.
    ///
    /// This is a convenience constructor that creates a `FlexFloat` with the value of -inf
    /// using the default bit array representation.
    ///
    /// # Returns
    ///
    /// A new `FlexFloat<DefaultBitArray>` instance representing negative infinity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let neg_inf = FlexFloat::neg_infinity();
    /// assert!(neg_inf.is_infinity());
    /// assert!(neg_inf.sign());
    /// ```
    pub fn neg_infinity() -> Self {
        Self::new_infinity(true)
    }

    /// Creates a new `FlexFloat` instance representing infinity with the specified sign.
    ///
    /// This is a convenience constructor that creates a `FlexFloat` with the value of inf
    /// using the default bit array representation.
    ///
    /// # Arguments
    ///
    /// * `sign` - The sign bit (false for +inf, true for -inf)
    ///
    /// # Returns
    ///
    /// A new `FlexFloat<DefaultBitArray>` instance representing infinity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let pos_inf = FlexFloat::infinity(false);
    /// let neg_inf = FlexFloat::infinity(true);
    /// assert!(pos_inf.is_infinity() && !pos_inf.sign());
    /// assert!(neg_inf.is_infinity() && neg_inf.sign());
    /// ```
    pub fn infinity(sign: bool) -> Self {
        Self::new_infinity(sign)
    }
}

impl Debug for FlexFloat<DefaultBitArray> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        {
            f.debug_struct("FlexFloat")
                .field("sign", if self.sign { &'-' } else { &'+' })
                .field("exponent", &(self.exponent.to_bigint() + 1_u8))
                .field("fraction", &self.fraction.to_biguint())
                .finish()
        }
        // #[cfg(not(feature = "bigint"))]
        // {
        //     f.debug_struct("FlexFloat")
        //         .field("sign", &self.sign)
        //         .field("exponent", &self.exponent.to_bits_string())
        //         .field("fraction", &self.fraction.to_bits_string())
        //         .finish()
        // }
    }
}
