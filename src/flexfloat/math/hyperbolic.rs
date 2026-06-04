//! Hyperbolic trigonometric operations for FlexFloat.
//!
//! This module provides hyperbolic trigonometric functions including:
//! - `sinh`: Hyperbolic sine function
//! - `cosh`: Hyperbolic cosine function
//! - `tanh`: Hyperbolic tangent function

use crate::{BitArray, FlexFloat, flexfloat::consts};

/// Returns the hyperbolic sine of the value.
///
/// This function computes `sinh(x) = (e^x - e^(-x)) / 2` for a FlexFloat value.
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `±∞` returns `±∞`
/// - `±0` returns `±0`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let x = FlexFloat::from(0.0);
/// let result = math::sinh(x);
/// assert_eq!(result, FlexFloat::from(0.0));
/// ```
pub fn sinh<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    (value.clone().exp() - (-value).exp()) / &consts::TWO
}

/// Returns the hyperbolic cosine of the value.
///
/// This function computes `cosh(x) = (e^x + e^(-x)) / 2` for a FlexFloat value.
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `±∞` returns `+∞`
/// - `0` returns `1`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let x = FlexFloat::from(0.0);
/// let result = math::cosh(x);
/// assert_eq!(result, FlexFloat::from(1.0));
/// ```
pub fn cosh<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    (value.clone().exp() + (-value).exp()) / &consts::TWO
}

/// Returns the hyperbolic tangent of the value.
///
/// This function computes `tanh(x) = sinh(x) / cosh(x)` for a FlexFloat value.
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `±∞` returns `±1`
/// - `±0` returns `±0`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let x = FlexFloat::from(0.0);
/// let result = math::tanh(x);
/// assert_eq!(result, FlexFloat::from(0.0));
/// ```
pub fn tanh<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    sinh(value.clone()) / cosh(value)
}

pub fn asinh<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    if value.is_nan() || value.is_infinite() || value.is_zero() {
        return value;
    }
    let magnitude = value.abs();
    let one = consts::ONE.convert_to::<B>();
    let result = if magnitude.exponent.len() > 11 {
        magnitude.ln() + consts::LN_2.convert_to::<B>()
    } else {
        let square = magnitude.clone() * magnitude.clone();
        (magnitude + (square + one).sqrt()).ln()
    };
    if value.sign { -result } else { result }
}

pub fn acosh<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    if value.is_nan() {
        return value;
    }
    let one = consts::ONE.convert_to::<B>();
    if value < one.clone() {
        return FlexFloat::new_nan();
    }
    if value == one {
        return FlexFloat::new_zero();
    }
    if value.is_infinite() {
        return value;
    }
    if value.exponent.len() > 11 {
        value.ln() + consts::LN_2.convert_to::<B>()
    } else {
        ((value.clone() - one.clone()).sqrt() * (value.clone() + one).sqrt() + value).ln()
    }
}

pub fn atanh<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    if value.is_nan() {
        return value;
    }
    let one = consts::ONE.convert_to::<B>();
    if value.abs() > one.clone() {
        return FlexFloat::new_nan();
    }
    if value == one {
        return FlexFloat::new_infinity(false);
    }
    if value == -one.clone() {
        return FlexFloat::new_infinity(true);
    }
    ((one.clone() + value.clone()).ln() - (one - value).ln()) / &consts::TWO
}

impl<B: BitArray> FlexFloat<B> {
    /// Returns the hyperbolic sine of the value.
    ///
    /// This method computes `sinh(self)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let x = FlexFloat::from(0.0);
    /// let result = x.sinh();
    /// assert_eq!(result, FlexFloat::from(0.0));
    /// ```
    pub fn sinh(self) -> Self {
        sinh(self)
    }

    /// Returns the hyperbolic cosine of the value.
    ///
    /// This method computes `cosh(self)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let x = FlexFloat::from(0.0);
    /// let result = x.cosh();
    /// assert_eq!(result, FlexFloat::from(1.0));
    /// ```
    pub fn cosh(self) -> Self {
        cosh(self)
    }

    /// Returns the hyperbolic tangent of the value.
    ///
    /// This method computes `tanh(self)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let x = FlexFloat::from(0.0);
    /// let result = x.tanh();
    /// assert_eq!(result, FlexFloat::from(0.0));
    /// ```
    pub fn tanh(self) -> Self {
        tanh(self)
    }

    pub fn asinh(self) -> Self {
        asinh(self)
    }

    pub fn acosh(self) -> Self {
        acosh(self)
    }

    pub fn atanh(self) -> Self {
        atanh(self)
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use crate::FlexFloat;
    use crate::flexfloat::consts;
    use crate::tests::*;

    /// Tests the sinh operation for FlexFloat.
    #[rstest]
    fn test_sinh(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.sinh(),
            |v: f64| v.sinh(),
            "sinh",
            identity_1,
            Some(1e-5),
        );
    }

    /// Tests the cosh operation for FlexFloat.
    #[rstest]
    fn test_cosh(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.cosh(),
            |v: f64| v.cosh(),
            "cosh",
            identity_1,
            Some(1e-5),
        );
    }

    /// Tests the tanh operation for FlexFloat.
    #[rstest]
    fn test_tanh(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.tanh(),
            |v: f64| v.tanh(),
            "tanh",
            identity_1,
            Some(1e-5),
        );
    }

    #[rstest]
    fn test_asinh(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.asinh(),
            |v: f64| v.asinh(),
            "asinh",
            |v| if v.abs() < 1e150 { Some(v) } else { None },
            Some(1e-5),
        );
    }

    #[rstest]
    fn test_acosh(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.acosh(),
            |v: f64| v.acosh(),
            "acosh",
            |v| {
                let v = v.abs() + 1.0;
                if v < 1e150 { Some(v) } else { None }
            },
            Some(1e-5),
        );
    }

    #[rstest]
    fn test_atanh(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.atanh(),
            |v: f64| v.atanh(),
            "atanh",
            |v| Some(v.tanh()),
            Some(1e-4),
        );
    }

    #[test]
    fn test_inverse_hyperbolic_large_values_stay_finite() {
        let huge = FlexFloat::from(f64::MAX) * FlexFloat::from(2.0);
        let asinh = huge.clone().asinh();
        let acosh = huge.clone().acosh();

        assert!(
            asinh.is_finite(),
            "asinh should stay finite for grown exponents"
        );
        assert!(
            acosh.is_finite(),
            "acosh should stay finite for grown exponents"
        );
        assert!(asinh > consts::ONE.convert_to::<crate::DefaultBitArray>());
        assert!(acosh > consts::ONE.convert_to::<crate::DefaultBitArray>());
    }
}
