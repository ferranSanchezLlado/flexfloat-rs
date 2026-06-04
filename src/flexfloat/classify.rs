//! Classification predicates for FlexFloat.
//!
//! Provides methods for testing the category of a FlexFloat value:
//! NaN, infinity, zero, finite, normal, subnormal, and `FpCategory`.

use core::num::FpCategory;
use num_bigint::BigInt;

use crate::bitarray::BitArrayAccess;
use crate::flexfloat::FlexFloat;

impl<Exp: BitArrayAccess, Frac: BitArrayAccess> FlexFloat<Exp, Frac> {
    /// Checks if the exponent represents a special value (all ones).
    ///
    /// Special exponents indicate infinity or NaN values in IEEE 754 format.
    fn is_special_exponent(&self) -> bool {
        self.exponent.len() == 11 && self.exponent.is_ones()
    }

    /// Returns true if this FlexFloat represents NaN (Not-a-Number).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let nan = FlexFloat::nan();
    /// assert!(nan.is_nan());
    /// let normal = FlexFloat::from(3.14);
    /// assert!(!normal.is_nan());
    /// ```
    pub fn is_nan(&self) -> bool {
        self.is_special_exponent() && !self.fraction.is_zeros()
    }

    /// Returns true if this FlexFloat represents infinity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let inf = FlexFloat::pos_infinity();
    /// assert!(inf.is_infinite());
    /// let finite = FlexFloat::from(1.0);
    /// assert!(!finite.is_infinite());
    /// ```
    pub fn is_infinite(&self) -> bool {
        self.is_special_exponent() && self.fraction.is_zeros()
    }

    /// Returns true if this FlexFloat represents zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let zero = FlexFloat::zero();
    /// assert!(zero.is_zero());
    /// let non_zero = FlexFloat::from(0.1);
    /// assert!(!non_zero.is_zero());
    /// ```
    pub fn is_zero(&self) -> bool {
        self.exponent.is_zeros() && self.fraction.is_zeros()
    }

    /// Returns `true` if the value is finite (not NaN and not ±∞).
    pub fn is_finite(&self) -> bool {
        !self.is_nan() && !self.is_infinite()
    }

    /// Returns `true` if the value is a normal floating-point number.
    pub fn is_normal(&self) -> bool {
        self.is_finite() && !self.is_zero() && !self.is_subnormal()
    }

    /// Returns `true` if the value is subnormal (denormalized).
    ///
    /// Subnormals only exist at the standard IEEE 754 double-precision
    /// exponent width (11 bits). With extended exponents (>11 bits),
    /// values are always either normal or special.
    pub fn is_subnormal(&self) -> bool {
        self.exponent.len() == 11 && self.exponent.is_zeros() && !self.fraction.is_zeros()
    }

    /// Returns the IEEE 754 floating-point category for this value.
    pub fn classify(&self) -> FpCategory {
        if self.is_nan() {
            FpCategory::Nan
        } else if self.is_infinite() {
            FpCategory::Infinite
        } else if self.is_zero() {
            FpCategory::Zero
        } else if self.is_subnormal() {
            FpCategory::Subnormal
        } else {
            FpCategory::Normal
        }
    }
}

/// Grown-aware instance methods — answers questions about this *specific* instance's
/// exponent/fraction width rather than returning fixed compile-time constants.
impl<Exp: BitArrayAccess, Frac: BitArrayAccess> FlexFloat<Exp, Frac> {
    /// Number of bits in the exponent field of this instance.
    pub fn exponent_bits(&self) -> usize {
        self.exponent.len()
    }

    /// Number of significant binary digits (fraction bits + 1 implicit leading bit).
    pub fn mantissa_digits(&self) -> usize {
        self.fraction.len() + 1
    }

    /// Minimum unbiased exponent representable at this instance's exponent width.
    ///
    /// Equivalent to `−2^(exp_bits−1) + 2` (matching IEEE 754 `DBL_MIN_EXP` convention).
    pub fn min_exp(&self) -> BigInt {
        let exp_bits = self.exponent.len();
        -(BigInt::from(1u8) << (exp_bits - 1)) + 2
    }

    /// Maximum unbiased exponent representable at this instance's exponent width.
    ///
    /// Equivalent to `2^(exp_bits−1) − 1` (matching IEEE 754 `DBL_MAX_EXP` convention).
    pub fn max_exp(&self) -> BigInt {
        let exp_bits = self.exponent.len();
        (BigInt::from(1u8) << (exp_bits - 1)) - 1
    }
}
