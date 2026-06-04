//! Trigonometric operations for FlexFloat.
//!
//! This module provides trigonometric functions including:
//! - `sin`: Sine function
//! - `cos`: Cosine function
//! - `tan`: Tangent function
//! - `asin`: Arcsine (inverse sine) function
//! - `acos`: Arccosine (inverse cosine) function
//! - `atan`: Arctangent (inverse tangent) function
//! - `atan2`: Two-argument arctangent function

use num_bigint::{BigInt, BigUint};
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::flexfloat::consts;
use crate::{BitArrayArith, FlexFloat};

const TWO_OVER_PI_PRECISION_BITS: usize = 1200;
const MIN_DYNAMIC_SHIFT_BITS: usize = 120;
const DYNAMIC_GUARD_BITS: usize = 128;

fn two_over_pi_biguint() -> BigUint {
    let mut bytes = Vec::with_capacity(consts::TWO_OVER_PI_BITS.len() * 8);
    for word in consts::TWO_OVER_PI_BITS {
        bytes.extend_from_slice(&word.to_be_bytes());
    }
    BigUint::from_bytes_be(&bytes)
}

fn arctan_reciprocal_scaled(k: u64, n_bits: usize, guard: usize) -> BigInt {
    let total_bits = n_bits + guard;
    let k_big = BigUint::from(k);
    let k_sq = &k_big * &k_big;

    let mut power = BigUint::one() << total_bits;
    power /= &k_big;

    let mut result = BigInt::from(power.clone());
    let mut i = 1u64;

    loop {
        power /= &k_sq;
        if power.is_zero() {
            break;
        }

        let term = &power / BigUint::from(2 * i + 1);
        if term.is_zero() {
            break;
        }

        if i % 2 == 0 {
            result += BigInt::from(term);
        } else {
            result -= BigInt::from(term);
        }

        i += 1;
    }

    (result + BigInt::from(BigUint::one() << (guard - 1))) >> guard
}

fn compute_pi_scaled(n_bits: usize) -> BigUint {
    let at5 = arctan_reciprocal_scaled(5, n_bits, DYNAMIC_GUARD_BITS);
    let at239 = arctan_reciprocal_scaled(239, n_bits, DYNAMIC_GUARD_BITS);
    let pi_over_4 = BigInt::from(4u8) * at5 - at239;

    (BigInt::from(4u8) * pi_over_4)
        .to_biguint()
        .expect("pi is positive")
}

fn compute_two_over_pi(n_bits: usize) -> BigUint {
    let pi = compute_pi_scaled(n_bits + DYNAMIC_GUARD_BITS);
    let numerator = BigUint::one() << (2 * n_bits + DYNAMIC_GUARD_BITS + 1);
    numerator / pi
}

fn scaled_fraction_to_flexfloat<B: BitArrayArith>(scaled: BigInt, shift: usize) -> FlexFloat<B> {
    if scaled.is_zero() {
        return FlexFloat::new_zero();
    }

    let sign = scaled.is_negative();
    let numerator = FlexFloat::<B>::from_int(scaled.abs());
    let denominator = FlexFloat::<B>::from_int(BigInt::one() << shift);
    let value = numerator / denominator;

    if sign { -value } else { value }
}

fn reduce_pi_over_two<B: BitArrayArith>(x: FlexFloat<B>) -> (u8, FlexFloat<B>) {
    let negative = x.sign;
    let x = x.abs();

    if x <= consts::FRAC_PI_4.convert_to::<B>() {
        return (0, if negative { -x } else { x });
    }

    let is_subnormal = x.exponent.to_biguint().is_zero();
    let mut significand = x.fraction.to_biguint();
    let exponent = if is_subnormal {
        BigInt::from(-1022)
    } else {
        significand |= BigUint::one() << 52;
        x.exponent.to_bigint() + 1
    };

    let (two_over_pi, precision_bits) = if let Some(exponent_usize) = exponent.to_usize() {
        let required_precision = exponent_usize.saturating_add(MIN_DYNAMIC_SHIFT_BITS);
        if required_precision <= TWO_OVER_PI_PRECISION_BITS {
            (two_over_pi_biguint(), TWO_OVER_PI_PRECISION_BITS)
        } else {
            let n_bits = exponent_usize
                .saturating_add(52)
                .saturating_add(DYNAMIC_GUARD_BITS);
            (compute_two_over_pi(n_bits), n_bits)
        }
    } else if exponent.is_negative() {
        (two_over_pi_biguint(), TWO_OVER_PI_PRECISION_BITS)
    } else {
        // Exponents beyond usize range cannot be reduced with dynamically grown
        // Payne-Hanek precision without requesting impossible allocations.
        return (0, FlexFloat::new_zero());
    };

    let product = significand * two_over_pi;
    let shift = BigInt::from(precision_bits + 52) - exponent;
    let shift =
        usize::try_from(shift).expect("dynamic Payne-Hanek produced insufficient precision");

    let k = if shift == 0 {
        product.clone()
    } else {
        (&product + (BigUint::one() << (shift - 1))) >> shift
    };
    let quadrant = (&k & BigUint::from(3u8))
        .to_u8()
        .expect("quadrant must fit in u8");

    let scaled_remainder = BigInt::from(product) - BigInt::from(k.clone() << shift);
    let remainder_fraction = scaled_fraction_to_flexfloat::<B>(scaled_remainder, shift);
    let remainder = remainder_fraction * consts::FRAC_PI_2.convert_to::<B>();

    if negative {
        ((4 - quadrant) % 4, -remainder)
    } else {
        (quadrant, remainder)
    }
}

fn sin_kernel<B: BitArrayArith>(r: FlexFloat<B>) -> FlexFloat<B> {
    let r2 = &r * &r;
    let mut poly: FlexFloat<B> = consts::SIN_COEFFS[5].convert_to();
    for coeff in consts::SIN_COEFFS[..5].iter().rev() {
        poly = &poly * &r2 + coeff.convert_to::<B>();
    }

    r.clone() + r * r2 * poly
}

fn cos_kernel<B: BitArrayArith>(r: FlexFloat<B>) -> FlexFloat<B> {
    let r2 = &r * &r;
    let mut poly: FlexFloat<B> = consts::COS_COEFFS[5].convert_to();
    for coeff in consts::COS_COEFFS[..5].iter().rev() {
        poly = &poly * &r2 + coeff.convert_to::<B>();
    }

    consts::ONE.convert_to::<B>() + r2 * poly
}

/// Returns the sine of the value (in radians).
///
/// This function computes `sin(x)` for a FlexFloat value.
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `±∞` returns `NaN`
/// - `±0` returns `±0`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let angle = FlexFloat::from(core::f64::consts::PI / 6.0);
/// let opposite_ratio = math::sin(angle);
/// assert_ff_almost_eq!(opposite_ratio, FlexFloat::from(0.5));
/// ```
pub fn sin<B: BitArrayArith>(value: FlexFloat<B>) -> FlexFloat<B> {
    if value.is_nan() || value.is_infinite() {
        return FlexFloat::new_nan();
    }
    if value.is_zero() {
        return value;
    }

    let (quadrant, remainder) = reduce_pi_over_two(value);
    let sin_r = sin_kernel(remainder.clone());
    let cos_r = cos_kernel(remainder);

    match quadrant {
        0 => sin_r,
        1 => cos_r,
        2 => -sin_r,
        _ => -cos_r,
    }
}

/// Returns the cosine of the value (in radians).
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `±∞` returns `NaN`
/// - `0` returns `1`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let angle = FlexFloat::from(0.0);
/// let adjacent_ratio = math::cos(angle);
/// assert_ff_almost_eq!(adjacent_ratio, FlexFloat::from(1.0));
/// ```
pub fn cos<B: BitArrayArith>(value: FlexFloat<B>) -> FlexFloat<B> {
    if value.is_nan() || value.is_infinite() {
        return FlexFloat::new_nan();
    }
    if value.is_zero() {
        return consts::ONE.convert_to();
    }

    let (quadrant, remainder) = reduce_pi_over_two(value);
    let sin_r = sin_kernel(remainder.clone());
    let cos_r = cos_kernel(remainder);

    match quadrant {
        0 => cos_r,
        1 => -sin_r,
        2 => -cos_r,
        _ => sin_r,
    }
}

/// Returns the tangent of the value (in radians).
///
/// This function computes `tan(x)` for a FlexFloat value.
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `±∞` returns `NaN`
/// - `±0` returns `±0`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let angle = FlexFloat::from(core::f64::consts::PI / 4.0);
/// let slope = math::tan(angle);
/// assert_ff_almost_eq!(slope, FlexFloat::from(1.0));
/// ```
pub fn tan<B: BitArrayArith>(value: FlexFloat<B>) -> FlexFloat<B> {
    if value.is_nan() || value.is_infinite() {
        return FlexFloat::new_nan();
    }
    if value.is_zero() {
        return value;
    }

    sin(value.clone()) / cos(value)
}

/// Returns the arcsine of the value (result in radians).
///
/// This function computes `arcsin(x)` for a FlexFloat value.
/// The result is in the range [-π/2, π/2].
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `|x| > 1` returns `NaN`
/// - `±0` returns `±0`
/// - `1` returns `π/2`
/// - `-1` returns `-π/2`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let ratio = FlexFloat::from(0.5);
/// let angle = math::asin(ratio);
/// assert_ff_almost_eq!(angle, FlexFloat::from(core::f64::consts::PI / 6.0));
/// ```
pub fn asin<B: BitArrayArith>(_value: FlexFloat<B>) -> FlexFloat<B> {
    if _value.is_nan() {
        return FlexFloat::new_nan();
    }
    if _value.is_zero() {
        return _value;
    }

    let one = consts::ONE.convert_to::<B>();
    let abs_value = _value.abs();
    if abs_value > one {
        return FlexFloat::new_nan();
    }

    let pi_over_2 = consts::FRAC_PI_2.convert_to::<B>();
    if abs_value == one {
        return if _value.sign { -pi_over_2 } else { pi_over_2 };
    }

    let denominator = super::power::sqrt(one - (&_value * &_value));
    atan2(_value, denominator)
}

/// Returns the arccosine of the value (result in radians).
///
/// This function computes `arccos(x)` for a FlexFloat value.
/// The result is in the range [0, π].
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `|x| > 1` returns `NaN`
/// - `1` returns `0`
/// - `-1` returns `π`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let ratio = FlexFloat::from(0.5);
/// let angle = math::acos(ratio);
/// assert_ff_almost_eq!(angle, FlexFloat::from(core::f64::consts::PI / 3.0));
/// ```
pub fn acos<B: BitArrayArith>(_value: FlexFloat<B>) -> FlexFloat<B> {
    if _value.is_nan() {
        return FlexFloat::new_nan();
    }

    let one = consts::ONE.convert_to::<B>();
    let abs_value = _value.abs();
    if abs_value > one {
        return FlexFloat::new_nan();
    }

    consts::FRAC_PI_2.convert_to::<B>() - asin(_value)
}

/// Returns the arctangent of the value (result in radians).
///
/// This function computes `arctan(x)` for a FlexFloat value.
/// The result is in the range (-π/2, π/2).
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `±0` returns `±0`
/// - `+∞` returns `π/2`
/// - `-∞` returns `-π/2`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let rise_over_run = FlexFloat::from(1.0);
/// let angle = math::atan(rise_over_run);
/// assert_ff_almost_eq!(angle, FlexFloat::from(core::f64::consts::PI / 4.0));
/// ```
pub fn atan<B: BitArrayArith>(value: FlexFloat<B>) -> FlexFloat<B> {
    if value.is_nan() {
        return FlexFloat::new_nan();
    }
    if value.is_zero() {
        return value;
    }
    if value.is_infinite() {
        let pi_over_2 = consts::FRAC_PI_2.convert_to::<B>();
        return if value.sign { -pi_over_2 } else { pi_over_2 };
    }

    let negative = value.sign;
    let one = consts::ONE.convert_to::<B>();
    let half = consts::HALF.convert_to::<B>();
    let mut x = value.abs();

    let reciprocal = x > one;
    if reciprocal {
        x = one.clone() / x;
    }

    let shifted = x >= half;
    if shifted {
        x = (x.clone() - &one) / (x + &one);
    }

    let tolerance = consts::TOLERANCE.convert_to::<B>();
    let x_sq = &x * &x;
    let mut power = x.clone();
    let mut result = x;
    for n in 1..=30 {
        power *= &x_sq;
        power = -power;
        let term = power.clone() / FlexFloat::<B>::from_f64((2 * n + 1) as f64);
        result += &term;

        if term.abs() < tolerance {
            break;
        }
    }

    if shifted {
        result += consts::FRAC_PI_4.convert_to::<B>();
    }
    if reciprocal {
        result = consts::FRAC_PI_2.convert_to::<B>() - result;
    }

    if negative { -result } else { result }
}

/// Returns the arctangent of y/x (result in radians).
///
/// This function computes `arctan(y/x)` using the signs of both arguments
/// to determine the quadrant of the result. The result is in the range [-π, π].
///
/// # Special Cases
///
/// - If either argument is `NaN`, returns `NaN`
/// - `atan2(±0, +x)` returns `±0` for positive x
/// - `atan2(±0, -x)` returns `±π` for negative x
/// - `atan2(±y, ±0)` returns `±π/2` for positive y
/// - `atan2(±∞, x)` returns `±π/2` for finite x
/// - `atan2(±y, ±∞)` returns `±0` for positive infinity, `±π` for negative infinity
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let y = FlexFloat::from(1.0);
/// let x = FlexFloat::from(1.0);
/// let bearing = math::atan2(y, x);
/// assert_ff_almost_eq!(bearing, FlexFloat::from(core::f64::consts::PI / 4.0));
/// ```
pub fn atan2<B: BitArrayArith>(y: FlexFloat<B>, x: FlexFloat<B>) -> FlexFloat<B> {
    if y.is_nan() || x.is_nan() {
        return FlexFloat::new_nan();
    }

    let pi = consts::PI.convert_to::<B>();
    let pi_over_2 = consts::FRAC_PI_2.convert_to::<B>();
    let pi_over_4 = consts::FRAC_PI_4.convert_to::<B>();
    let three_pi_over_4 = pi.clone() - pi_over_4.clone();

    if y.is_infinite() {
        if x.is_infinite() {
            return match (y.sign, x.sign) {
                (false, false) => pi_over_4,
                (false, true) => three_pi_over_4,
                (true, false) => -pi_over_4,
                (true, true) => -three_pi_over_4,
            };
        }

        return if y.sign { -pi_over_2 } else { pi_over_2 };
    }

    if y.is_zero() {
        if x.sign {
            return if y.sign { -pi } else { pi };
        }

        return FlexFloat::new_zero_with_sign(y.sign);
    }

    if x.is_infinite() {
        if x.sign {
            return if y.sign { -pi } else { pi };
        }

        return FlexFloat::new_zero_with_sign(y.sign);
    }

    if x.is_zero() {
        return if y.sign { -pi_over_2 } else { pi_over_2 };
    }

    let abs_y = y.abs();
    let abs_x = x.abs();
    let angle = if abs_y > abs_x {
        pi_over_2.clone() - atan(abs_x / abs_y)
    } else {
        atan(abs_y / abs_x)
    };

    match (y.sign, x.sign) {
        (false, false) => angle,
        (true, false) => -angle,
        (false, true) => pi - angle,
        (true, true) => angle - pi,
    }
}

impl<B: BitArrayArith> FlexFloat<B> {
    /// Returns the sine of the value (in radians).
    ///
    /// This method computes `sin(self)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let angle = FlexFloat::from(core::f64::consts::PI / 6.0);
    /// let opposite_ratio = angle.sin();
    /// assert_ff_almost_eq!(opposite_ratio, FlexFloat::from(0.5));
    /// ```
    pub fn sin(self) -> Self {
        sin(self)
    }

    /// Returns the cosine of the value (in radians).
    ///
    /// This method computes `cos(self)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let angle = FlexFloat::from(0.0);
    /// let adjacent_ratio = angle.cos();
    /// assert_ff_almost_eq!(adjacent_ratio, FlexFloat::from(1.0));
    /// ```
    pub fn cos(self) -> Self {
        cos(self)
    }

    /// Returns the tangent of the value (in radians).
    ///
    /// This method computes `tan(self)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let angle = FlexFloat::from(core::f64::consts::PI / 4.0);
    /// let slope = angle.tan();
    /// assert_ff_almost_eq!(slope, FlexFloat::from(1.0));
    /// ```
    pub fn tan(self) -> Self {
        tan(self)
    }

    /// Returns the arcsine of the value (result in radians).
    ///
    /// This method computes `arcsin(self)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let ratio = FlexFloat::from(0.5);
    /// let angle = ratio.asin();
    /// assert_ff_almost_eq!(angle, FlexFloat::from(core::f64::consts::PI / 6.0));
    /// ```
    pub fn asin(self) -> Self {
        asin(self)
    }

    /// Returns the arccosine of the value (result in radians).
    ///
    /// This method computes `arccos(self)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let ratio = FlexFloat::from(0.5);
    /// let angle = ratio.acos();
    /// assert_ff_almost_eq!(angle, FlexFloat::from(core::f64::consts::PI / 3.0));
    /// ```
    pub fn acos(self) -> Self {
        acos(self)
    }

    /// Returns the arctangent of the value (result in radians).
    ///
    /// This method computes `arctan(self)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let rise_over_run = FlexFloat::from(1.0);
    /// let angle = rise_over_run.atan();
    /// assert_ff_almost_eq!(angle, FlexFloat::from(core::f64::consts::PI / 4.0));
    /// ```
    pub fn atan(self) -> Self {
        atan(self)
    }

    /// Returns the arctangent of self/x (result in radians).
    ///
    /// This method computes `arctan(self/x)` using the signs of both arguments
    /// to determine the quadrant of the result.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let y = FlexFloat::from(1.0);
    /// let x = FlexFloat::from(1.0);
    /// let bearing = y.atan2(x);
    /// assert_ff_almost_eq!(bearing, FlexFloat::from(core::f64::consts::PI / 4.0));
    /// ```
    pub fn atan2(self, x: Self) -> Self {
        atan2(self, x)
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;
    use rand::Rng;
    use rstest::rstest;

    use crate::FlexFloat;
    use crate::tests::*;

    /// Tests the sin operation for FlexFloat.
    #[rstest]
    fn test_sin(mut rng: impl Rng, n_experiments: usize) {
        let huge: FlexFloat<crate::DefaultBitArray> =
            FlexFloat::from_int(BigInt::from(1u8) << 2000);
        let huge_sin = huge.clone().sin();
        let huge_cos = huge.cos();
        let identity = (&huge_sin * &huge_sin) + (&huge_cos * &huge_cos);

        assert!(!huge_sin.is_nan());
        assert!(!huge_sin.is_infinite());
        assert!(!huge_cos.is_nan());
        assert!(!huge_cos.is_infinite());
        assert_almost_eq_with_tolerance(
            identity.to_f64().unwrap(),
            1.0,
            1e-12,
            "sin^2(x) + cos^2(x) for x=2^2000",
        );

        test_unary_flexfloat_op(
            &mut rng,
            n_experiments / 50,
            |ff| ff.sin(),
            |v: f64| v.sin(),
            "sin",
            identity_1,
            Some(1e-11),
        );
    }

    /// Tests the cos operation for FlexFloat.
    #[rstest]
    fn test_cos(rng: impl Rng, n_experiments: usize) {
        let huge: FlexFloat<crate::DefaultBitArray> =
            FlexFloat::from_int(BigInt::from(1u8) << 2000);
        let huge_sin = huge.clone().sin();
        let huge_cos = huge.cos();
        let identity = (&huge_sin * &huge_sin) + (&huge_cos * &huge_cos);

        assert_almost_eq_with_tolerance(
            identity.to_f64().unwrap(),
            1.0,
            1e-12,
            "sin^2(x) + cos^2(x) for x=2^2000",
        );

        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.cos(),
            |v: f64| v.cos(),
            "cos",
            identity_1,
            Some(1e-7),
        );
    }

    #[test]
    fn test_trig_with_grown_exponent_does_not_request_unbounded_precision() {
        use crate::bitarray::BitArrayConstruction;
        use crate::flexfloat::grow_exponent;

        let huge = FlexFloat::new(
            false,
            grow_exponent::<crate::DefaultBitArray>(BigInt::from(1u8) << 2000, 11),
            crate::DefaultBitArray::zeros(52),
        );

        let s = huge.clone().sin();
        let c = huge.cos();

        assert!(!s.is_nan(), "sin should stay finite, got {s:?}");
        assert!(!s.is_infinite(), "sin should stay finite, got {s:?}");
        assert!(!c.is_nan(), "cos should stay finite, got {c:?}");
        assert!(!c.is_infinite(), "cos should stay finite, got {c:?}");
    }

    /// Tests the tan operation for FlexFloat.
    #[rstest]
    fn test_tan(rng: impl Rng, n_experiments: usize) {
        let huge: FlexFloat<crate::DefaultBitArray> =
            FlexFloat::from_int(BigInt::from(1u8) << 2000);
        let huge_sin = huge.clone().sin();
        let huge_cos = huge.clone().cos();
        let huge_tan = huge.tan();
        let expected_tan = huge_sin / huge_cos;

        assert_almost_eq_with_tolerance(
            huge_tan.to_f64().unwrap(),
            expected_tan.to_f64().unwrap(),
            1e-12,
            "tan(x) consistency for x=2^2000",
        );

        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.tan(),
            |v: f64| v.tan(),
            "tan",
            identity_1,
            Some(1e-8),
        );
    }

    /// Tests the asin operation for FlexFloat.
    #[rstest]
    fn test_asin(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.asin(),
            |v: f64| v.asin(),
            "asin",
            identity_1,
            Some(1e-8),
        );
    }

    /// Tests the acos operation for FlexFloat.
    #[rstest]
    fn test_acos(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.acos(),
            |v: f64| v.acos(),
            "acos",
            identity_1,
            Some(1e-8),
        );
    }

    /// Tests the atan operation for FlexFloat.
    #[rstest]
    fn test_atan(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.atan(),
            |v: f64| v.atan(),
            "atan",
            identity_1,
            Some(1e-8),
        );
    }

    /// Tests the atan2 operation for FlexFloat.
    #[rstest]
    fn test_atan2(mut rng: impl Rng, n_experiments: usize) {
        test_binary_flexfloat_op(
            &mut rng,
            n_experiments / 200,
            |a, b| a.atan2(b),
            |a: f64, b: f64| a.atan2(b),
            "atan2",
            identity_2,
            Some(1e-8),
        );
    }
}
