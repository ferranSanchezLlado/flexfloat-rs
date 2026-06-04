//! Rounding operations for FlexFloat.
//!
//! This module provides rounding functions including:
//! - `round`: Round to nearest integer (round half away from zero, matches `f64::round`)
//! - `floor`: Round down to nearest integer
//! - `ceil`: Round up to nearest integer
//! - `trunc`: Truncate fractional part (round toward zero)
//! - `fract`: Extract fractional part

use crate::flexfloat::consts;
use crate::{BitArray, FlexFloat};

/// Rounds the value to the nearest integer using round half away from zero.
///
/// This function rounds the floating-point value to the nearest integer.
/// When the fractional part is exactly 0.5, it rounds away from zero. This matches `f64`
/// and is the same behavior as the method
/// [`FlexFloat::round`].
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
/// let x = FlexFloat::from(2.7);
/// let rounded = math::round(x);
/// assert_eq!(rounded, FlexFloat::from(3.0));
///
/// let y = FlexFloat::from(-2.3);
/// let rounded_y = math::round(y);
/// assert_eq!(rounded_y, FlexFloat::from(-2.0));
/// ```
pub fn round<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    // Handle special cases: NaN, Infinity
    if value.is_nan() || value.is_infinite() {
        value
    } else if value >= FlexFloat::zero() {
        floor(value + &consts::HALF)
    } else {
        ceil(value - &consts::HALF)
    }
}

/// Returns the largest integer less than or equal to the value.
///
/// This function computes the floor function, which rounds down to the nearest
/// integer. For positive values, this rounds toward zero; for negative values,
/// this rounds away from zero. This is the same behavior as the method
/// [`FlexFloat::floor`].
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `±∞` returns `±∞`
/// - `±0` returns `±0`
/// - Integer values return themselves
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let x = FlexFloat::from(2.7);
/// let floored = math::floor(x);
/// assert_eq!(floored, FlexFloat::from(2.0));
///
/// let y = FlexFloat::from(-2.3);
/// let floored_y = math::floor(y);
/// assert_eq!(floored_y, FlexFloat::from(-3.0));
/// ```
pub fn floor<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    if value.is_nan() || value.is_infinite() || value.is_zero() {
        return value;
    }

    let trunc = value.clone().to_int().expect("Handled NaN/Inf above");
    let value_trunc: FlexFloat<B> = FlexFloat::from_int(trunc.clone());
    if value_trunc == value {
        return value;
    }

    if !value.sign {
        // If positive and not integral, floor = trunc (truncation toward zero is already the flooring)
        value_trunc
    } else {
        FlexFloat::from_int(trunc - 1)
    }
}

/// Returns the smallest integer greater than or equal to the value.
///
/// This function computes the ceiling function, which rounds up to the nearest
/// integer. For positive values, this rounds away from zero; for negative values,
/// this rounds toward zero. This is the same behavior as the method
/// [`FlexFloat::ceil`].
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `±∞` returns `±∞`
/// - `±0` returns `±0`
/// - Integer values return themselves
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let x = FlexFloat::from(2.3);
/// let ceiled = math::ceil(x);
/// assert_eq!(ceiled, FlexFloat::from(3.0));
///
/// let y = FlexFloat::from(-2.7);
/// let ceiled_y = math::ceil(y);
/// assert_eq!(ceiled_y, FlexFloat::from(-2.0));
/// ```
pub fn ceil<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    if value.is_nan() || value.is_infinite() || value.is_zero() {
        return value;
    }

    let trunc = value.clone().to_int().expect("Handled NaN/Inf above");
    let value_trunc: FlexFloat<B> = FlexFloat::from_int(trunc.clone());
    if value_trunc == value {
        return value;
    }

    if !value.sign {
        FlexFloat::from_int(trunc + 1)
    } else {
        // If negative and not integral, ceil = trunc (truncation toward zero is already the ceiling)
        value_trunc
    }
}

/// Truncates the fractional part, returning the integer part toward zero.
///
/// This function removes the fractional part of the number, rounding toward zero.
/// For positive numbers, this behaves like `floor`, and for negative numbers like `ceil`.
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `±∞` returns `±∞`
/// - `±0` returns `±0`
/// - Integer values return themselves
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let x = FlexFloat::from(2.7);
/// let truncated = math::trunc(x);
/// assert_eq!(truncated, FlexFloat::from(2.0));
///
/// let y = FlexFloat::from(-2.7);
/// let truncated_y = math::trunc(y);
/// assert_eq!(truncated_y, FlexFloat::from(-2.0));
/// ```
pub fn trunc<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    if value.is_nan() || value.is_infinite() || value.is_zero() {
        return value;
    }

    let trunc_int = value.clone().to_int().expect("Handled NaN/Inf above");
    FlexFloat::from_int(trunc_int)
}

/// Returns the fractional part of the value.
///
/// This function returns the fractional component of the floating-point number.
/// The result has the same sign as the input.
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `±∞` returns `NaN`
/// - `±0` returns `±0`
/// - Integer values return `±0`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let x = FlexFloat::from(2.75);
/// let fractional = math::fract(x);
/// assert_eq!(fractional, FlexFloat::from(0.75));
///
/// let y = FlexFloat::from(-2.75);
/// let fractional_y = math::fract(y);
/// assert_eq!(fractional_y, FlexFloat::from(-0.75));
/// ```
pub fn fract<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    if value.is_nan() {
        return value;
    }
    if value.is_infinite() {
        return FlexFloat::new_nan();
    }

    value.clone() - trunc(value)
}

impl<B: BitArray> FlexFloat<B> {
    /// Rounds the value to the nearest integer using round half away from zero.
    ///
    /// This method rounds the floating-point value to the nearest integer.
    /// When the fractional part is exactly 0.5, it rounds away from zero.
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
    ///
    /// let x = FlexFloat::from(2.7);
    /// let rounded = x.round();
    /// assert_eq!(rounded, FlexFloat::from(3.0));
    ///
    /// let y = FlexFloat::from(-2.3);
    /// let rounded_y = y.round();
    /// assert_eq!(rounded_y, FlexFloat::from(-2.0));
    /// ```
    pub fn round(self) -> Self {
        round(self)
    }

    /// Returns the smallest integer greater than or equal to the value.
    ///
    /// This method computes the ceiling function, which rounds up to the nearest
    /// integer. For positive values, this rounds away from zero; for negative values,
    /// this rounds toward zero.
    ///
    /// # Special Cases
    ///
    /// - `NaN` returns `NaN`
    /// - `±∞` returns `±∞`
    /// - `±0` returns `±0`
    /// - Integer values return themselves
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let x = FlexFloat::from(2.3);
    /// let ceiled = x.ceil();
    /// assert_eq!(ceiled, FlexFloat::from(3.0));
    ///
    /// let y = FlexFloat::from(-2.7);
    /// let ceiled_y = y.ceil();
    /// assert_eq!(ceiled_y, FlexFloat::from(-2.0));
    /// ```
    pub fn ceil(self) -> Self {
        ceil(self)
    }

    /// Returns the largest integer less than or equal to the value.
    ///
    /// This method computes the floor function, which rounds down to the nearest
    /// integer. For positive values, this rounds toward zero; for negative values,
    /// this rounds away from zero.
    ///
    /// # Special Cases
    ///
    /// - `NaN` returns `NaN`
    /// - `±∞` returns `±∞`
    /// - `±0` returns `±0`
    /// - Integer values return themselves
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let x = FlexFloat::from(2.7);
    /// let floored = x.floor();
    /// assert_eq!(floored, FlexFloat::from(2.0));
    ///
    /// let y = FlexFloat::from(-2.3);
    /// let floored_y = y.floor();
    /// assert_eq!(floored_y, FlexFloat::from(-3.0));
    /// ```
    pub fn floor(self) -> Self {
        floor(self)
    }

    /// Truncates the fractional part, returning the integer part toward zero.
    ///
    /// This method removes the fractional part of the number, rounding toward zero.
    ///
    /// # Special Cases
    ///
    /// - `NaN` returns `NaN`
    /// - `±∞` returns `±∞`
    /// - `±0` returns `±0`
    /// - Integer values return themselves
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let x = FlexFloat::from(2.7);
    /// let truncated = x.trunc();
    /// assert_eq!(truncated, FlexFloat::from(2.0));
    ///
    /// let y = FlexFloat::from(-2.7);
    /// let truncated_y = y.trunc();
    /// assert_eq!(truncated_y, FlexFloat::from(-2.0));
    /// ```
    pub fn trunc(self) -> Self {
        trunc(self)
    }

    /// Returns the fractional part of the value.
    ///
    /// This method returns the fractional component of the floating-point number.
    ///
    /// # Special Cases
    ///
    /// - `NaN` returns `NaN`
    /// - `±∞` returns `NaN`
    /// - `±0` returns `±0`
    /// - Integer values return `±0`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let x = FlexFloat::from(2.7);
    /// let fractional = x.fract();
    /// // fractional ≈ 0.7
    ///
    /// let y = FlexFloat::from(-2.7);
    /// let fractional_y = y.fract();
    /// // fractional_y ≈ -0.7
    /// ```
    pub fn fract(self) -> Self {
        fract(self)
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::tests::*;

    /// Tests the ceil operation for FlexFloat.
    #[rstest]
    fn test_ceil(mut rng: impl Rng, n_experiments: usize) {
        for _ in 0..n_experiments {
            let value = random_f64(&mut rng);
            let ff = FlexFloat::from(value);
            let rounded_ff = ff.ceil();
            let expected = value.ceil();
            let converted_result: f64 = rounded_ff.into();
            assert_almost_eq(converted_result, expected, &format!("ceil({value:#?})"));
        }
    }

    /// Tests the floor operation for FlexFloat.
    #[rstest]
    fn test_floor(mut rng: impl Rng, n_experiments: usize) {
        for _ in 0..n_experiments {
            let value = random_f64(&mut rng);
            let ff = FlexFloat::from(value);
            let rounded_ff = ff.floor();
            let expected = value.floor();
            let converted_result: f64 = rounded_ff.into();
            assert_almost_eq(converted_result, expected, &format!("floor({value:#?})"));
        }
    }

    /// Tests the round operation for FlexFloat.
    #[rstest]
    fn test_round(mut rng: impl Rng, n_experiments: usize) {
        for _ in 0..n_experiments {
            let value = random_f64(&mut rng);
            let ff = FlexFloat::from(value);
            let rounded_ff = ff.round();
            let expected = value.round();
            let converted_result: f64 = rounded_ff.into();
            assert_almost_eq(converted_result, expected, &format!("round({value:#?})"));
        }
    }

    /// Tests the trunc operation for FlexFloat.
    #[rstest]
    fn test_trunc(mut rng: impl Rng, n_experiments: usize) {
        for _ in 0..n_experiments {
            let value = random_f64(&mut rng);
            let ff = FlexFloat::from(value);
            let truncated_ff = ff.trunc();
            let expected = value.trunc();
            let converted_result: f64 = truncated_ff.into();
            assert_almost_eq(converted_result, expected, &format!("trunc({value:#?})"));
        }
    }

    /// Tests the fract operation for FlexFloat.
    #[rstest]
    fn test_fract(mut rng: impl Rng, n_experiments: usize) {
        for _ in 0..n_experiments {
            let value = random_f64(&mut rng);
            let ff = FlexFloat::from(value);
            let fract_ff = ff.fract();
            let expected = value.fract();

            if let Some(converted_result) = fract_ff.to_f64() {
                assert_almost_eq(converted_result, expected, &format!("fract({value:#?})"));
            }
        }
    }
}
