//! Power and root operations for FlexFloat.
//!
//! This module provides power and root functions including:
//! - `pow`: Raise a value to a power
//! - `sqrt`: Square root
//! - `cbrt`: Cube root

use crate::{
    BitArray, FlexFloat,
    flexfloat::consts,
    math::{exp, fract, ln},
    prelude::BitArrayConversion,
};

/// Returns the value raised to the power of the exponent.
///
/// This function computes `base^exponent` for FlexFloat values.
///
/// # Special Cases
///
/// - `NaN` base or exponent returns `NaN`
/// - `base^0` returns `1` for any finite base
/// - `1^exponent` returns `1` for any finite exponent
/// - `0^exponent` returns `0` if exponent > 0, `+∞` if exponent < 0, `NaN` if exponent = 0
/// - `base^1` returns `base` for any finite base
/// - Negative base with non-integer exponent returns `NaN`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let base = FlexFloat::from(2.0);
/// let exponent = FlexFloat::from(1.0);
/// let result = math::pow(base, &exponent);
/// assert_eq!(result, FlexFloat::from(2.0));
/// ```
pub fn pow<B: BitArray, B2: BitArrayConversion>(
    base: FlexFloat<B>,
    exponent: &FlexFloat<B2>,
) -> FlexFloat<B> {
    // Handle special cases first
    if base.is_nan() || exponent.is_nan() {
        return FlexFloat::new_nan();
    }

    // Handle exponent = 0: base^0 = 1 (for any finite base, including 0^0 = 1)
    if exponent.is_zero() {
        return FlexFloat::from_f64(1.0);
    }

    // Handle base = 0
    if base.is_zero() {
        let exp_positive = !exponent.sign;
        if exp_positive {
            // 0^positive = 0
            return FlexFloat::from_f64(0.0);
        } else {
            // 0^negative = +∞
            return FlexFloat::new_infinity(false);
        }
    }

    // Handle exponent = 1: base^1 = base
    if exponent == &consts::ONE {
        return base;
    }

    // Handle base = 1: 1^exponent = 1 (for finite exponent)
    if base == consts::ONE && !exponent.is_infinite() {
        return consts::ONE.convert_to();
    }

    // For positive bases, use the standard exp(ln(base) * exponent) formula
    if base.is_positive() {
        return exp(ln(base) * exponent.convert_to::<B>());
    }

    // For negative bases, we need special handling
    let exponent_converted = exponent.convert_to::<B>();

    // Check if exponent has a fractional part
    let exponent_fract = fract(exponent_converted.clone());
    let is_integer = exponent_fract.is_zero();

    if is_integer {
        // For integer exponents with negative base, compute with absolute value
        let abs_base = base.abs();
        let result = exp(ln(abs_base) * exponent_converted.clone());

        // Determine the sign based on whether exponent is odd
        // An integer is odd if dividing it by 2 leaves a fractional part
        let half_exponent = exponent_converted / &consts::TWO;
        let is_odd = !fract(half_exponent).is_zero();

        if is_odd { -result } else { result }
    } else {
        // Negative base with non-integer exponent returns NaN
        FlexFloat::new_nan()
    }
}

/// Returns the square root of the value.
///
/// This function computes `√x` for a FlexFloat value.
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `x < 0` returns `NaN`
/// - `0` returns `0`
/// - `+∞` returns `+∞`
/// - `1` returns `1`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let x = FlexFloat::from(1.0);
/// let result = math::sqrt(x);
/// assert_eq!(result, FlexFloat::from(1.0));
/// ```
pub fn sqrt<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    pow(value, &consts::HALF)
}

/// Returns the cube root of the value.
///
/// This function computes `∛x` for a FlexFloat value.
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `±0` returns `±0`
/// - `±∞` returns `±∞`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let x = FlexFloat::from(1.0);
/// let result = math::cbrt(x);
/// assert_eq!(result, FlexFloat::from(1.0));
/// ```
pub fn cbrt<B: BitArray>(value: FlexFloat<B>) -> FlexFloat<B> {
    // Cube root is defined for negative numbers: cbrt(-x) = -cbrt(x)
    if value.sign {
        -pow(value.abs(), &consts::THIRD)
    } else {
        pow(value, &consts::THIRD)
    }
    // pow(value, &consts::THIRD)
}

pub fn hypot<B: BitArray>(x: FlexFloat<B>, y: &FlexFloat<B>) -> FlexFloat<B> {
    sqrt(x.clone() * x + y.clone() * y.clone())
}

impl<B: BitArray> FlexFloat<B> {
    /// Returns the value raised to the power of the exponent.
    ///
    /// This method computes `self^exponent`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let base = FlexFloat::from(2.0);
    /// let exponent = FlexFloat::from(1.0);
    /// let result = base.powf(&exponent);
    /// assert_eq!(result, FlexFloat::from(2.0));
    /// ```
    pub fn powf<B2: BitArrayConversion>(self, exponent: &FlexFloat<B2>) -> Self {
        pow(self, exponent)
    }

    /// Returns the square root of the value.
    ///
    /// This method computes `sqrt(self)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let x = FlexFloat::from(1.0);
    /// let result = x.sqrt();
    /// assert_eq!(result, FlexFloat::from(1.0));
    /// ```
    pub fn sqrt(self) -> Self {
        sqrt(self)
    }

    /// Returns the cube root of the value.
    ///
    /// This method computes `cbrt(self)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let x = FlexFloat::from(1.0);
    /// let result = x.cbrt();
    /// assert_eq!(result, FlexFloat::from(1.0));
    /// ```
    pub fn cbrt(self) -> Self {
        cbrt(self)
    }

    pub fn powi(self, n: i32) -> Self {
        if n == 0 {
            return FlexFloat::from_f64(1.0);
        }
        if self.is_nan() {
            return self;
        }

        let mut exp = n.unsigned_abs();
        let mut base = self;
        let mut result = FlexFloat::from_f64(1.0);
        while exp > 0 {
            if exp & 1 == 1 {
                result *= base.clone();
            }
            exp >>= 1;
            if exp > 0 {
                base *= base.clone();
            }
        }
        if n < 0 {
            FlexFloat::from_f64(1.0) / result
        } else {
            result
        }
    }

    pub fn hypot(self, other: &Self) -> Self {
        hypot(self, other)
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use crate::{FlexFloat, tests::*};

    /// Tests the pow operation for FlexFloat.
    #[rstest]
    fn test_pow(mut rng: impl Rng, n_experiments: usize) {
        // Test integer bases with integer and fractional exponents
        for base in -10..=10 {
            for exp in -5..=5 {
                for fract in [true, false] {
                    let base = base as f64;
                    let exp = if fract && exp != 0 {
                        1.0 / (exp as f64)
                    } else {
                        exp as f64
                    };

                    let base_ff = FlexFloat::from(base);
                    let exp_ff = FlexFloat::from(exp);
                    let expected = base.powf(exp);
                    let result = base_ff.powf(&exp_ff);
                    dbg!(base, exp, result.to_f64(), expected);

                    test_common_logic(
                        format!("pow({}, {})", base, exp).as_str(),
                        result,
                        expected,
                        None,
                    );
                }
            }
        }

        test_binary_flexfloat_op(
            &mut rng,
            n_experiments / 50,
            |a, b| a.powf(&b),
            |a: f64, b: f64| a.powf(b),
            "pow",
            identity_2,
            None,
        );
    }

    /// Tests the sqrt operation for FlexFloat.
    #[rstest]
    fn test_sqrt(mut rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            &mut rng,
            n_experiments / 50,
            |x| x.sqrt(),
            |x: f64| x.sqrt(),
            "sqrt",
            |x| Some(x.abs()), // Only test positive values
            None,
        );
    }

    /// Tests the cbrt operation for FlexFloat.
    #[rstest]
    fn test_cbrt(mut rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            &mut rng,
            n_experiments / 50,
            |x| x.cbrt(),
            |x: f64| x.cbrt(),
            "cbrt",
            identity_1,
            None,
        );
    }

    #[rstest]
    fn test_powi(mut rng: impl Rng, n_experiments: usize) {
        for _ in 0..(n_experiments / 100) {
            let base = random_f64(&mut rng);
            let exp = rng.random_range(-20..=20);
            let result = FlexFloat::from(base).powi(exp);
            let expected = base.powi(exp);
            test_common_logic(
                &format!("powi({base:?}, {exp})"),
                result,
                expected,
                Some(1e-8),
            );
        }
    }

    #[rstest]
    fn test_hypot(mut rng: impl Rng, n_experiments: usize) {
        test_binary_flexfloat_op(
            &mut rng,
            n_experiments / 50,
            |a, b| a.hypot(&b),
            |a: f64, b: f64| a.hypot(b),
            "hypot",
            identity_2,
            Some(1e-5),
        );
    }
}
