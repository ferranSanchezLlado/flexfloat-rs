//! Constructors for FlexFloat.
//!
//! Provides ergonomic constructors for creating FlexFloat values including
//! special values (zero, NaN, infinity) on the generic `FlexFloat<B: BitArray>`.

use crate::bitarray::BitArray;
use crate::flexfloat::FlexFloat;

impl<Exp: BitArray, Frac: BitArray> FlexFloat<Exp, Frac> {
    /// Creates a new FlexFloat with the specified components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let exp = DefaultBitArray::from_bits(&[true, false, true]);
    /// let frac = DefaultBitArray::zeros(52);
    /// let num = FlexFloat::new(false, exp, frac);
    /// ```
    pub const fn new(sign: bool, exponent: Exp, fraction: Frac) -> Self {
        Self {
            sign,
            exponent,
            fraction,
        }
    }

    /// Creates a FlexFloat representing zero with the specified sign.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let pos_zero = FlexFloat::zero_with_sign(false);
    /// let neg_zero = FlexFloat::zero_with_sign(true);
    /// assert!(pos_zero.is_zero() && !pos_zero.sign());
    /// assert!(neg_zero.is_zero() && neg_zero.sign());
    /// ```
    pub fn zero_with_sign(sign: bool) -> Self {
        Self {
            sign,
            exponent: Exp::zeros(11),
            fraction: Frac::zeros(52),
        }
    }

    /// Creates a FlexFloat representing positive zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let zero = FlexFloat::zero();
    /// assert!(zero.is_zero() && !zero.sign());
    /// ```
    pub fn zero() -> Self {
        Self::zero_with_sign(false)
    }

    /// Creates a FlexFloat representing NaN (Not-a-Number).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let nan = FlexFloat::nan();
    /// assert!(nan.is_nan());
    /// ```
    pub fn nan() -> Self {
        Self {
            sign: false,
            exponent: Exp::ones(11),
            fraction: Frac::ones(52),
        }
    }

    /// Creates a FlexFloat representing infinity with the specified sign.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let pos_inf = FlexFloat::infinity(false);
    /// let neg_inf = FlexFloat::infinity(true);
    /// assert!(pos_inf.is_infinite() && !pos_inf.sign());
    /// assert!(neg_inf.is_infinite() && neg_inf.sign());
    /// ```
    pub fn infinity(sign: bool) -> Self {
        Self {
            sign,
            exponent: Exp::ones(11),
            fraction: Frac::zeros(52),
        }
    }

    /// Creates a positive infinity value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let pos_inf = FlexFloat::pos_infinity();
    /// assert!(pos_inf.is_infinite() && !pos_inf.sign());
    /// ```
    pub fn pos_infinity() -> Self {
        Self::infinity(false)
    }

    /// Creates a negative infinity value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let neg_inf = FlexFloat::neg_infinity();
    /// assert!(neg_inf.is_infinite() && neg_inf.sign());
    /// ```
    pub fn neg_infinity() -> Self {
        Self::infinity(true)
    }
}

/// Returns positive zero by default — consistent with IEEE 754.
impl<Exp: BitArray, Frac: BitArray> Default for FlexFloat<Exp, Frac> {
    fn default() -> Self {
        Self::zero()
    }
}
