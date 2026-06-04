//! Negation and absolute-value operations.

use core::ops::Neg;

use crate::flexfloat::FlexFloat;

impl<Exp: Clone, Frac: Clone> FlexFloat<Exp, Frac> {
    /// Returns the absolute value of this `FlexFloat`.
    ///
    /// Creates a new `FlexFloat` with the same magnitude but positive sign,
    /// effectively clearing the sign bit while preserving exponent and fraction.
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
    pub fn abs(&self) -> Self {
        Self {
            sign: false,
            exponent: self.exponent.clone(),
            fraction: self.fraction.clone(),
        }
    }
}

/// Negation operation for `FlexFloat`.
///
/// Implements the [`Neg`] trait to provide unary minus operation,
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
impl<Exp, Frac> Neg for FlexFloat<Exp, Frac> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            sign: !self.sign,
            exponent: self.exponent,
            fraction: self.fraction,
        }
    }
}

/// Negation of a reference — avoids a clone when the caller only needs the negated value.
impl<Exp: Clone, Frac: Clone> Neg for &FlexFloat<Exp, Frac> {
    type Output = FlexFloat<Exp, Frac>;

    fn neg(self) -> Self::Output {
        FlexFloat {
            sign: !self.sign,
            exponent: self.exponent.clone(),
            fraction: self.fraction.clone(),
        }
    }
}
