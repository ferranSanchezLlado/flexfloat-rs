//! Field accessors and sign predicates for FlexFloat.

use crate::flexfloat::FlexFloat;

impl<Exp, Frac> FlexFloat<Exp, Frac> {
    /// Returns the sign bit of this FlexFloat.
    ///
    /// `false` = positive, `true` = negative.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
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
    /// use flexfloat::prelude::*;
    ///
    /// let num = FlexFloat::from(8.0);
    /// println!("Exponent has {} bits", num.exponent().len());
    /// ```
    pub fn exponent(&self) -> &Exp {
        &self.exponent
    }

    /// Returns a reference to the fraction bit array.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let num = FlexFloat::from(3.14159);
    /// println!("Fraction has {} bits", num.fraction().len());
    /// ```
    pub fn fraction(&self) -> &Frac {
        &self.fraction
    }

    /// Returns `true` if the sign bit is `false` (positive).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let pos = FlexFloat::from(1.23);
    /// let neg = FlexFloat::from(-4.56);
    /// assert!(pos.is_positive());
    /// assert!(!neg.is_positive());
    /// ```
    pub fn is_positive(&self) -> bool {
        !self.sign
    }

    /// Returns `true` if the sign bit is `true` (negative).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let pos = FlexFloat::from(1.23);
    /// let neg = FlexFloat::from(-4.56);
    /// assert!(!pos.is_negative());
    /// assert!(neg.is_negative());
    /// ```
    pub fn is_negative(&self) -> bool {
        self.sign
    }

    /// `f64`-parity alias for [`is_positive`](Self::is_positive).
    pub fn is_sign_positive(&self) -> bool {
        self.is_positive()
    }

    /// `f64`-parity alias for [`is_negative`](Self::is_negative).
    pub fn is_sign_negative(&self) -> bool {
        self.is_negative()
    }
}
