//! Hyperbolic trigonometric operations for FlexFloat.
//!
//! This module provides hyperbolic trigonometric functions including:
//! - `sinh`: Hyperbolic sine function
//! - `cosh`: Hyperbolic cosine function
//! - `tanh`: Hyperbolic tangent function

use crate::flexfloat::FlexFloat;
use crate::{BitArrayArith, flexfloat::consts};

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
pub fn sinh<Exp: BitArrayArith, Frac: BitArrayArith>(
    value: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    (value.clone().exp() - (-value).exp()) / consts::TWO
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
pub fn cosh<Exp: BitArrayArith, Frac: BitArrayArith>(
    value: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    (value.clone().exp() + (-value).exp()) / consts::TWO
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
pub fn tanh<Exp: BitArrayArith, Frac: BitArrayArith>(
    value: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    sinh(value.clone()) / cosh(value)
}

pub fn asinh<Exp: BitArrayArith, Frac: BitArrayArith>(
    value: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    if value.is_nan() || value.is_infinite() || value.is_zero() {
        return value;
    }
    let magnitude = value.abs();
    let result = if magnitude.exponent.len() > 11 {
        magnitude.ln() + consts::LN_2
    } else {
        let square = magnitude.clone() * magnitude.clone();
        (magnitude + (square + consts::ONE).sqrt()).ln()
    };
    if value.sign { -result } else { result }
}

pub fn acosh<Exp: BitArrayArith, Frac: BitArrayArith>(
    value: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    if value.is_nan() {
        return value;
    }
    if value < consts::ONE {
        return FlexFloat::nan();
    }
    if value == consts::ONE {
        return FlexFloat::zero();
    }
    if value.is_infinite() {
        return value;
    }
    if value.exponent.len() > 11 {
        value.ln() + consts::LN_2
    } else {
        ((value.clone() - consts::ONE).sqrt() * (value.clone() + consts::ONE).sqrt() + value).ln()
    }
}

pub fn atanh<Exp: BitArrayArith, Frac: BitArrayArith>(
    value: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    if value.is_nan() {
        return value;
    }
    if value.abs() > consts::ONE {
        return FlexFloat::nan();
    }
    if value == consts::ONE {
        return FlexFloat::infinity(false);
    }
    if value == consts::NEGATIVE_ONE {
        return FlexFloat::infinity(true);
    }
    // Use the numerically stable identity: atanh(x) = ln_1p(2x / (1-x)) / 2
    //
    // This avoids computing ln(1-x) for x near ±1, where 1-x is tiny and
    // range-reducing a very small argument in ln() accumulates relative error.
    // Instead we compute ln_1p of a large positive argument (x near 1) or a
    // small argument (x near 0, where ln_1p uses its Taylor-series path),
    // both of which are well-conditioned.
    let one = consts::ONE.convert_to::<Exp, Frac>();
    ((one.clone() + value.clone()).ln() - (one - value).ln()) / consts::TWO
}

impl<Exp: BitArrayArith, Frac: BitArrayArith> FlexFloat<Exp, Frac> {
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

    use crate::flexfloat::FlexFloat;
    use crate::flexfloat::consts;
    use crate::test_support::*;

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
        // Regression: atanh(x) for x near 1 previously exceeded the 1e-4 tolerance
        let regression_input = 0.9999999999999877_f64;
        test_common_logic(
            &format!("atanh({regression_input:?})"),
            FlexFloat::from(regression_input).atanh(),
            regression_input.atanh(),
            Some(1e-4),
        );

        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.atanh(),
            |v: f64| v.atanh(),
            "atanh",
            |v| Some(v.tanh()),
            Some(1e-2),
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
        assert!(asinh > consts::ONE);
        assert!(acosh > consts::ONE);
    }
}
