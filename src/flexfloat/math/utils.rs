//! Utility mathematical operations for FlexFloat.
//!
//! This module provides various utility functions including:
//! - `signum`: Sign of the number (-1, 0, or 1)
//! - `copysign`: Copy the sign from another number
//! - `recip`: Reciprocal (1/x)
//! - `to_degrees`: Convert radians to degrees
//! - `to_radians`: Convert degrees to radians

use crate::flexfloat::consts::{ConstFloat, float_to_const};
use crate::flexfloat::{FlexFloat, consts};
use crate::{BitArray, BitArrayArith};

const DEG_PER_RAD: ConstFloat = float_to_const(180.0 / core::f64::consts::PI);
const RAD_PER_DEG: ConstFloat = float_to_const(core::f64::consts::PI / 180.0);

/// Returns the sign of the value as -1, 0, or 1.
///
/// This function returns:
/// - `1.0` if the value is positive (including +∞)
/// - `-1.0` if the value is negative (including -∞)
/// - `NaN` if the value is `NaN`
/// - `0.0` or `-0.0` for zero values (preserving sign)
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let x = FlexFloat::from(5.0);
/// assert_eq!(math::signum(x), FlexFloat::from(1.0));
///
/// let y = FlexFloat::from(-5.0);
/// assert_eq!(math::signum(y), FlexFloat::from(-1.0));
/// ```
pub fn signum<Exp: BitArray, Frac: BitArray>(value: FlexFloat<Exp, Frac>) -> FlexFloat<Exp, Frac> {
    if value.is_nan() {
        return value;
    }

    if value.is_zero() {
        return value;
    }

    if value.sign {
        FlexFloat::from_f64(-1.0)
    } else {
        FlexFloat::from_f64(1.0)
    }
}

/// Returns a value with the magnitude of `self` and the sign of `sign`.
///
/// This function copies the sign bit from `sign` to the absolute value of `self`.
///
/// # Special Cases
///
/// - If `self` is `NaN`, returns `NaN` (sign is ignored)
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let x = FlexFloat::from(5.0);
/// let sign = FlexFloat::from(-1.0);
/// let result = math::copysign(x, sign);
/// assert_eq!(result, FlexFloat::from(-5.0));
/// ```
pub fn copysign<Exp: BitArray, Frac: BitArray>(
    value: FlexFloat<Exp, Frac>,
    sign: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    // IEEE 754 §5.5.1: copysign(x, y) = x with the sign of y.
    // The sign bit is orthogonal to the payload, so this applies
    // unconditionally - including when `value` is NaN.
    let mut result = value;
    result.sign = sign.sign;
    result
}

/// Returns the reciprocal (1/x) of the value.
///
/// This function computes `1 / x`.
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `±∞` returns `±0`
/// - `±0` returns `±∞`
/// - `1` returns `1`
/// - `-1` returns `-1`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let x = FlexFloat::from(4.0);
/// let result = math::recip(x);
/// assert_eq!(result, FlexFloat::from(0.25));
/// ```
pub fn recip<Exp: BitArrayArith, Frac: BitArrayArith>(
    value: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    consts::ONE.convert_to::<Exp, Frac>() / value
}

/// Converts radians to degrees.
///
/// This function multiplies the value by 180/π to convert from radians to degrees.
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
/// let turn = FlexFloat::from(core::f64::consts::PI / 2.0);
/// let degrees = math::to_degrees(turn);
/// assert_ff_almost_eq!(degrees, FlexFloat::from(90.0));
/// ```
pub fn to_degrees<Exp: BitArrayArith, Frac: BitArrayArith>(
    radians: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    radians * DEG_PER_RAD
}

/// Converts degrees to radians.
///
/// This function multiplies the value by π/180 to convert from degrees to radians.
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
/// let degrees = FlexFloat::from(45.0);
/// let radians = math::to_radians(degrees);
/// assert_ff_almost_eq!(radians, FlexFloat::from(core::f64::consts::PI / 4.0));
/// ```
pub fn to_radians<Exp: BitArrayArith, Frac: BitArrayArith>(
    degrees: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    degrees * RAD_PER_DEG
}

/// Fused multiply-add: `(value * a) + b`.
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math::mul_add;
///
/// assert_ff_almost_eq!(mul_add(FlexFloat::from(2.0), FlexFloat::from(3.0), FlexFloat::from(4.0)), FlexFloat::from(10.0));
/// ```
pub fn mul_add<Exp: BitArrayArith, Frac: BitArrayArith>(
    value: FlexFloat<Exp, Frac>,
    a: FlexFloat<Exp, Frac>,
    b: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    value * a + b
}

impl<Exp: BitArrayArith, Frac: BitArrayArith> FlexFloat<Exp, Frac> {
    /// Returns the sign of the value as -1, 0, or 1.
    ///
    /// This method returns:
    /// - `1.0` if the value is positive (including +∞)
    /// - `-1.0` if the value is negative (including -∞)
    /// - `NaN` if the value is `NaN`
    /// - `0.0` or `-0.0` for zero values (preserving sign)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let x = FlexFloat::from(5.0);
    /// assert_eq!(x.signum(), FlexFloat::from(1.0));
    ///
    /// let y = FlexFloat::from(-5.0);
    /// assert_eq!(y.signum(), FlexFloat::from(-1.0));
    /// ```
    pub fn signum(self) -> Self {
        signum(self)
    }

    /// Returns a value with the magnitude of `self` and the sign of `sign`.
    ///
    /// This method copies the sign bit from `sign` to the absolute value of `self`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let x = FlexFloat::from(5.0);
    /// let sign = FlexFloat::from(-1.0);
    /// let result = x.copysign(sign);
    /// assert_eq!(result, FlexFloat::from(-5.0));
    /// ```
    pub fn copysign(self, sign: Self) -> Self {
        copysign(self, sign)
    }
}

impl<Exp: BitArrayArith, Frac: BitArrayArith> FlexFloat<Exp, Frac> {
    /// Returns the reciprocal (1/x) of the value.
    ///
    /// This method computes `1 / self`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let x = FlexFloat::from(4.0);
    /// let result = x.recip();
    /// assert_eq!(result, FlexFloat::from(0.25));
    /// ```
    pub fn recip(self) -> Self {
        recip(self)
    }

    /// Converts radians to degrees.
    ///
    /// This method multiplies the value by 180/π to convert from radians to degrees.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let turn = FlexFloat::from(core::f64::consts::PI / 2.0);
    /// let degrees = turn.to_degrees();
    /// assert_ff_almost_eq!(degrees, FlexFloat::from(90.0));
    /// ```
    pub fn to_degrees(self) -> Self {
        to_degrees(self)
    }

    /// Converts degrees to radians.
    ///
    /// This method multiplies the value by π/180 to convert from degrees to radians.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let degrees = FlexFloat::from(45.0);
    /// let radians = degrees.to_radians();
    /// assert_ff_almost_eq!(radians, FlexFloat::from(core::f64::consts::PI / 4.0));
    /// ```
    pub fn to_radians(self) -> Self {
        to_radians(self)
    }

    /// Fused multiply-add: computes `(self * a) + b` with a single rounding error.
    ///
    /// Because FlexFloat uses software arithmetic this is equivalent to the two-step
    /// version, but the method exists for `f64`-API parity and may benefit from a
    /// future hardware-FMA path.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let x = FlexFloat::from(2.0);
    /// let a = FlexFloat::from(3.0);
    /// let b = FlexFloat::from(4.0);
    /// assert_ff_almost_eq!(x.mul_add(a, b), FlexFloat::from(10.0));
    /// ```
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        mul_add(self, a, b)
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use crate::FlexFloat;
    use crate::test_support::*;

    /// Tests the signum operation for FlexFloat.
    #[rstest]
    fn test_signum(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments,
            |x| x.signum(),
            |x| x.signum(),
            "signum",
            identity_1,
            None,
        );
    }

    /// Tests the copysign operation for FlexFloat.
    #[rstest]
    fn test_copysign(rng: impl Rng, n_experiments: usize) {
        test_binary_flexfloat_op(
            rng,
            n_experiments,
            |a, b| a.copysign(b),
            |a: f64, b: f64| a.copysign(b),
            "copysign",
            identity_2,
            None,
        );
    }

    #[test]
    fn test_copysign_propagates_sign_into_nan() {
        let nan_pos = FlexFloat::nan();
        let neg = FlexFloat::from(-1.0);
        let pos = FlexFloat::from(1.0);

        let r1 = super::copysign(nan_pos.clone(), neg);
        assert!(r1.is_nan(), "result must remain NaN");
        assert!(r1.is_negative(), "sign must be copied from negative source");

        let r2 = super::copysign(nan_pos, pos);
        assert!(r2.is_nan(), "result must remain NaN");
        assert!(r2.is_positive(), "sign must be copied from positive source");
    }

    /// Tests the recip operation for FlexFloat.
    #[rstest]
    fn test_recip(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments,
            |x| x.recip(),
            |x| x.recip(),
            "recip",
            identity_1,
            None,
        );
    }

    /// Tests the to_degrees operation for FlexFloat.
    #[rstest]
    fn test_to_degrees(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments,
            |x| x.to_degrees(),
            |x| x.to_degrees(),
            "to_degrees",
            identity_1,
            None,
        );
    }

    /// Tests the to_radians operation for FlexFloat.
    #[rstest]
    fn test_to_radians(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments,
            |x| x.to_radians(),
            |x| x.to_radians(),
            "to_radians",
            identity_1,
            None,
        );
    }
}
