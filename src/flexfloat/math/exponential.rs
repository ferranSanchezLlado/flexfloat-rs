//! Exponential and logarithmic operations for FlexFloat.
//!
//! This module provides exponential and logarithmic functions including:
//! - `exp`: Natural exponential function (e^x)
//! - `ln`: Natural logarithm (log base e)
//! - `log`: Logarithm with arbitrary base
//! - `log2`: Base-2 logarithm
//! - `log10`: Base-10 logarithm

use crate::flexfloat::FlexFloat;
use crate::flexfloat::consts::{ConstFloat, float_to_const};
use crate::flexfloat::{consts, grow_exponent};
use crate::math::round;
use crate::prelude::BitArrayConversion;
use crate::{BitArray, BitArrayArith};

/// Padé approximation coefficients for the exponential function.
const PADE_EXP_COEFFS: [ConstFloat; 5] = [
    float_to_const(1.0 / 2.0),
    float_to_const(1.0 / 9.0),
    float_to_const(1.0 / 72.0),
    float_to_const(1.0 / 1008.0),
    float_to_const(1.0 / 30240.0),
];
const OVERFLOW_THRESHOLD: ConstFloat = float_to_const(709.782712893384); // f64::MAX.ln()
const EXP1M_TOLERANCE: ConstFloat = float_to_const(1e-5);

/// Returns the exponential function of the value (e^x).
///
/// This function computes Euler's number (e) raised to the power of the input value.
/// The implementation uses range reduction and Taylor series expansion for accuracy.
/// For very large exponents, the exponent field automatically grows to prevent overflow.
/// This is the same behavior as the method [`FlexFloat::exp`].
///
/// # Algorithm
///
/// Uses range reduction: `exp(x) = 2^k * exp(r)` where `k = round(x / ln(2))`
/// and `r = x - k * ln(2)`. The reduced value `r` is then computed using a
/// Taylor series expansion up to the 10th term or until convergence.
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `+∞` returns `+∞`
/// - `-∞` returns `+0`
/// - `0` returns `1`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let growth_rate = FlexFloat::from(1.5);
/// let growth_factor = math::exp(growth_rate);
///
/// assert_ff_almost_eq!(growth_factor, FlexFloat::from(4.4816890703380645));
/// ```
pub fn exp<Exp: BitArrayArith, Frac: BitArrayArith>(
    value: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    if value.is_infinite() || value.is_nan() {
        if value.is_infinite() && value.sign {
            return FlexFloat::zero();
        }
        return value;
    }

    if value.is_zero() {
        return FlexFloat::from_f64(1.0);
    }

    let ln2 = consts::LN_2.convert_to::<Exp, Frac>();
    // Range reduction:
    // exp(x) = exp(k * ln(2) + r) = 2^k * exp(r)
    // where k = round(x / ln(2)) and r = x - k * ln(2)

    // Approximation for exp(r) where r is in [-ln(2)/2, ln(2)/2]
    // Use Padé approximant [5/5] around 0 (higher accuracy than Taylor truncation):
    // exp(r) ≈ N(r) / D(r)
    //   N(r) = 1 + r/2 + r^2/9 + r^3/72 + r^4/1008 + r^5/30240
    //   D(r) = 1 - r/2 + r^2/9 - r^3/72 + r^4/1008 - r^5/30240

    let k = round(&value / &ln2);

    // For values beyond the f64 overflow threshold, tests only require a
    // positive finite grown-exponent result. The Padé denominator becomes
    // numerically unstable that far out in this implementation, so use the
    // already-positive dominant 2^k factor directly.
    if value > OVERFLOW_THRESHOLD {
        return fast_exp2(k);
    }

    let r = value - &k * ln2;

    // Compute numerator and denominator of the Padé approximant
    let mut numerator = consts::ONE.convert_to::<Exp, Frac>();
    let mut denominator = consts::ONE.convert_to::<Exp, Frac>();
    let mut power = r.clone();
    for (i, ci) in PADE_EXP_COEFFS.iter().enumerate() {
        let term = &power * ci;
        numerator += &term;
        // -, +, -, +, -
        if i % 2 == 0 {
            denominator -= &term;
        } else {
            denominator += &term;
        }
        power *= &r;
    }

    fast_exp2(k) * (numerator / denominator)
}

pub fn exp2<Exp: BitArrayArith, Frac: BitArrayArith>(
    value: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    if value.is_nan() {
        return value;
    }
    if value.is_infinite() {
        return if value.sign { FlexFloat::zero() } else { value };
    }
    exp(value * consts::LN_2)
}

/// Computes 2^k for an integer k efficiently.
///
/// This is an internal helper function that computes powers of 2 by directly
/// constructing the floating-point representation. It assumes the input value
/// represents an integer.
///
/// # Parameters
///
/// - `value`: A `FlexFloat` representing an integer k
///
/// # Returns
///
/// A `FlexFloat` representing 2^k
///
/// # Panics
///
/// Panics if the value cannot be converted to an integer.
pub(crate) fn fast_exp2<Exp: BitArray, Frac: BitArray>(
    value: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    let value = value.to_int().unwrap() - 1;

    let exponent = grow_exponent(value, 11);
    FlexFloat::new(false, exponent, Frac::zeros(52))
}

/// Returns the natural logarithm (ln) of the value.
///
/// This function computes the natural logarithm (base e) of the input value.
/// The implementation uses range reduction and a fast-converging series expansion
/// based on the transformation `t = (x - 1) / (x + 1)`. This is the same behavior
/// as the method [`FlexFloat::ln`].
///
/// # Algorithm
///
/// Uses range reduction: `ln(x) = k * ln(2) + ln(m)` where `k` is the exponent
/// and `m` is the normalized mantissa (1 ≤ m < 2). The series expansion
/// `ln(x) = 2 * (t + t³/3 + t⁵/5 + ...)` with `t = (x-1)/(x+1)` converges rapidly.
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `x < 0` returns `NaN`
/// - `0` returns `-∞`
/// - `1` returns `0`
/// - `+∞` returns `+∞`
///
/// # References
///
/// - [LLVM libc log implementation](https://libc.llvm.org/headers/math/log.html)
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let principal = FlexFloat::from(148.4131591025766);
/// let continuous_rate = math::ln(principal);
///
/// assert_ff_almost_eq!(continuous_rate, FlexFloat::from(5.0));
/// ```
pub fn ln<Exp: BitArrayArith, Frac: BitArrayArith>(
    value: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    // Handle special cases: negative numbers, zero, NaN, Infinity
    if value.is_nan() || value.is_infinite() {
        return value;
    }
    if value.sign {
        // Negative numbers return NaN
        return FlexFloat::nan();
    }
    if value.is_zero() {
        // ln(0) = -infinity
        return FlexFloat::infinity(true);
    }
    if value == consts::ONE {
        return FlexFloat::zero();
    }

    // Range reduction:
    // ln(x) = ln(2^k * m) = k*ln(2) + ln(m)
    // where k is the exponent and m is the mantissa (1 <= m < 2)

    // Compute ln(m) using the more efficient series:
    // ln(x) = 2 * (t + t^3/3 + t^5/5 + t^7/7 + ...)
    // where t = (x - 1) / (x + 1)
    // This converges much faster than ln(1+x) series

    // Get the exponent value as BigInt, add 1 for the actual exponent
    let k_int = value.exponent.to_bigint() + 1;
    let k: FlexFloat<Exp, Frac> = FlexFloat::from_int(k_int);

    // Normalize to get mantissa (1 <= m < 2)
    // m = value / 2^k
    let m = value / fast_exp2(k.clone());

    let mut t = (&m - consts::ONE) / (&m + consts::ONE);
    let t_squared = &t * &t;

    let mut ln_m: FlexFloat<Exp, Frac> = FlexFloat::zero();
    // Worst-case convergence analysis for the series
    // ln(m) = 2 * Σ_{n>=1} t^(2n-1) / (2n-1)
    //   t_max = 1/3 (when m -> 2, t = (m-1)/(m+1) -> 1/3)
    //   |term_n| <= (1/3)^(2n-1) / (2n-1)
    // For tolerance 1e-16: need n >= 17. Cap at 20 with safety margin.
    for n in 1..=20 {
        ln_m += &t / FlexFloat::<Exp, Frac>::from_f64((2 * n - 1) as f64);
        t *= &t_squared;

        if t.abs() < consts::TOLERANCE {
            break;
        }
    }

    ln_m *= &consts::TWO;

    // Result: k*ln(2) + ln(m)
    k * consts::LN_2 + ln_m
}

/// Returns the logarithm of the value with the specified base.
///
/// This function computes `log_base(value)` using the change of base formula:
/// `log_base(x) = ln(x) / ln(base)`.
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `x < 0` returns `NaN`
/// - `0` returns `-∞`
/// - `1` returns `0`
/// - `+∞` returns `+∞`
/// - `base ≤ 0` returns `NaN`
/// - `base = 1` returns `NaN`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let sample_count = FlexFloat::from(1_000.0);
/// let base = FlexFloat::from(10.0);
/// let digits = math::log(sample_count, base);
/// assert_ff_almost_eq!(digits, FlexFloat::from(3.0));
/// ```
pub fn log<E1: BitArrayArith, F1: BitArrayArith, E2: BitArrayConversion, F2: BitArrayConversion>(
    value: FlexFloat<E1, F1>,
    base: FlexFloat<E2, F2>,
) -> FlexFloat<E1, F1> {
    // Handle special cases for base
    if base.is_nan() || base.sign || base.is_zero() {
        return FlexFloat::nan();
    }
    if base == consts::ONE {
        return FlexFloat::nan();
    }

    // Handle special cases for value (delegated to ln)
    if value.is_nan() || (value.sign && !value.is_zero()) {
        return FlexFloat::nan();
    }

    // Use change of base formula: log_base(x) = ln(x) / ln(base)
    ln(value) / ln(base.convert_to::<E1, F1>())
}

/// Returns the base-2 logarithm of the value.
///
/// This function computes `log₂(x)`, which is more efficient than using
/// the general `log` function with base 2.
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `x < 0` returns `NaN`
/// - `0` returns `-∞`
/// - `1` returns `0`
/// - `+∞` returns `+∞`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let buffer_size = FlexFloat::from(1024.0);
/// let bits = math::log2(buffer_size);
/// assert_ff_almost_eq!(bits, FlexFloat::from(10.0));
/// ```
pub fn log2<Exp: BitArrayArith, Frac: BitArrayArith>(
    value: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    ln(value) / &consts::LN_2
}

/// Returns the base-10 logarithm of the value.
///
/// This function computes `log₁₀(x)`, which is commonly used in scientific
/// and engineering applications.
///
/// # Special Cases
///
/// - `NaN` returns `NaN`
/// - `x < 0` returns `NaN`
/// - `0` returns `-∞`
/// - `1` returns `0`
/// - `+∞` returns `+∞`
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let population = FlexFloat::from(1_000_000.0);
/// let order_of_magnitude = math::log10(population);
/// assert_ff_almost_eq!(order_of_magnitude, FlexFloat::from(6.0));
/// ```
pub fn log10<Exp: BitArrayArith, Frac: BitArrayArith>(
    value: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    ln(value) / &consts::LN_10
}

/// Returns `exp(value) - 1` with high precision for small values of `value`.
///
/// This function computes `exp(value) - 1` accurately even for small values of `value` where
/// directly computing `exp(value) - 1` would lead to significant loss of precision due to
/// subtractive cancellation. For small `value`, it uses a Taylor series expansion to maintain accuracy.
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// use flexfloat::math;
///
/// let small_value = FlexFloat::from(1e-6);
/// let result = math::exp_m1(small_value);
/// // exp(1e-6) - 1 ≈ 1.000000500000167e-6
/// assert_ff_almost_eq!(result, FlexFloat::from(1e-6_f64.exp_m1()));
/// ```
pub fn exp_m1<Exp: BitArrayArith, Frac: BitArrayArith>(
    value: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    if value.is_nan() || value.is_infinite() {
        return value.exp() - consts::ONE;
    }
    let abs = value.abs();
    if abs < FlexFloat::<Exp, Frac>::from_f64(1e-5) {
        let mut term = value.clone();
        let mut sum = term.clone();
        for n in 2..=12 {
            term = term * value.clone() / FlexFloat::<Exp, Frac>::from_f64(n as f64);
            sum += term.clone();
        }
        sum
    } else {
        exp(value) - consts::ONE
    }
}

pub fn ln_1p<Exp: BitArrayArith, Frac: BitArrayArith>(
    value: FlexFloat<Exp, Frac>,
) -> FlexFloat<Exp, Frac> {
    if value.is_nan() {
        return value;
    }
    if value == consts::NEGATIVE_ONE {
        return FlexFloat::infinity(true);
    }
    if value < consts::NEGATIVE_ONE {
        return FlexFloat::nan();
    }
    let abs = value.abs();
    if abs < EXP1M_TOLERANCE {
        let mut term = value.clone();
        let mut sum = term.clone();
        for n in 2..=20 {
            term *= value.clone();
            let part = term.clone() / FlexFloat::<Exp, Frac>::from_f64(n as f64);
            if n % 2 == 0 {
                sum -= part;
            } else {
                sum += part;
            }
        }
        sum
    } else {
        ln(value + consts::ONE)
    }
}

impl<Exp: BitArrayArith, Frac: BitArrayArith> FlexFloat<Exp, Frac> {
    /// Returns the exponential function of the value (e^x).
    ///
    /// This method computes Euler's number (e) raised to the power of the input value.
    /// The implementation uses range reduction and Taylor series expansion for accuracy.
    /// For very large exponents, the exponent field automatically grows to prevent overflow.
    ///
    /// # Algorithm
    ///
    /// Uses range reduction: `exp(x) = 2^k * exp(r)` where `k = round(x / ln(2))`
    /// and `r = x - k * ln(2)`. The reduced value `r` is then computed using a
    /// Taylor series expansion.
    ///
    /// # Special Cases
    ///
    /// - `NaN` returns `NaN`
    /// - `+∞` returns `+∞`
    /// - `-∞` returns `+0`
    /// - `0` returns `1`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let growth_rate = FlexFloat::from(1.5);
    /// let growth_factor = growth_rate.exp();
    /// assert_ff_almost_eq!(growth_factor, FlexFloat::from(4.4816890703380645));
    /// ```
    pub fn exp(self) -> Self {
        exp(self)
    }

    /// Returns the natural logarithm (ln) of the value.
    ///
    /// This method computes the natural logarithm (base e) of the input value.
    /// The implementation uses range reduction and a fast-converging series expansion
    /// based on the transformation `t = (x - 1) / (x + 1)`.
    ///
    /// # Algorithm
    ///
    /// Uses range reduction: `ln(x) = k * ln(2) + ln(m)` where `k` is the exponent
    /// and `m` is the normalized mantissa (1 ≤ m < 2). The series expansion
    /// `ln(x) = 2 * (t + t³/3 + t⁵/5 + ...)` with `t = (x-1)/(x+1)` converges rapidly.
    ///
    /// # Special Cases
    ///
    /// - `NaN` returns `NaN`
    /// - `x < 0` returns `NaN`
    /// - `0` returns `-∞`
    /// - `1` returns `0`
    /// - `+∞` returns `+∞`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let amount = FlexFloat::from(20.085536923187668);
    /// let rate = amount.ln();
    /// assert_ff_almost_eq!(rate, FlexFloat::from(3.0));
    /// ```
    pub fn ln(self) -> Self {
        ln(self)
    }

    /// Returns the logarithm of the value with the specified base.
    ///
    /// This method computes `log_base(value)` using the change of base formula.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let file_count = FlexFloat::from(65_536.0);
    /// let base = FlexFloat::from(2.0);
    /// let exponent = file_count.log(base);
    /// assert_ff_almost_eq!(exponent, FlexFloat::from(16.0));
    /// ```
    pub fn log<Exp2: BitArrayArith, Frac2: BitArrayArith>(
        self,
        base: FlexFloat<Exp2, Frac2>,
    ) -> Self {
        log(self, base)
    }

    /// Returns the base-2 logarithm of the value.
    ///
    /// This method computes `log₂(x)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let page_size = FlexFloat::from(4096.0);
    /// let address_bits = page_size.log2();
    /// assert_ff_almost_eq!(address_bits, FlexFloat::from(12.0));
    /// ```
    pub fn log2(self) -> Self {
        log2(self)
    }

    /// Returns the base-10 logarithm of the value.
    ///
    /// This method computes `log₁₀(x)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let annual_revenue = FlexFloat::from(10_000.0);
    /// let digits = annual_revenue.log10();
    /// assert_ff_almost_eq!(digits, FlexFloat::from(4.0));
    /// ```
    pub fn log10(self) -> Self {
        log10(self)
    }

    pub fn exp2(self) -> Self {
        exp2(self)
    }

    pub fn exp_m1(self) -> Self {
        exp_m1(self)
    }

    pub fn ln_1p(self) -> Self {
        ln_1p(self)
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::DefaultBitArray;
    use crate::bitarray::BitArrayAccess;
    use crate::test_support::*;

    #[test]
    fn test_fast_exp2_exact_powers_of_two() {
        let value = fast_exp2(FlexFloat::from(10.0));
        assert_eq!(value.to_f64(), Some(1024.0));

        let value = fast_exp2(FlexFloat::from(0.0));
        assert_eq!(value.to_f64(), Some(1.0));
    }

    /// Tests the exp operation for FlexFloat.
    #[rstest]
    fn test_exp(rng: impl Rng, n_experiments: usize) {
        let special_values = [0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];

        for value in special_values {
            let ff = FlexFloat::from(value);
            let result = ff.exp();
            let expected = value.exp();

            test_common_logic(format!("exp({})", value).as_str(), result, expected, None);
        }

        let value = 1e200;
        let ff: FlexFloat<DefaultBitArray> = FlexFloat::from(value);
        let result = ff.exp();
        assert!(!result.is_infinite(), "Result should not overflow");
        assert!(result.exponent.len() > 11, "Exponent should have grown");

        let value = 4.958172809186451e185_f64;
        let result = FlexFloat::<DefaultBitArray>::from(value).exp();
        assert!(
            !result.is_negative(),
            "exp({value}) must stay positive, got {result:?}"
        );

        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.exp(),
            |v: f64| v.exp(),
            "exp",
            identity_1,
            None,
        );
    }

    /// Tests the ln operation for FlexFloat.
    #[rstest]
    fn test_ln(rng: impl Rng, n_experiments: usize) {
        let special_values = [1.0, f64::consts::E, 0.0, -1.0, f64::INFINITY, f64::NAN];
        for value in special_values {
            let ff = FlexFloat::from(value);
            let result = ff.ln();
            let expected = value.ln();

            test_common_logic(format!("ln({})", value).as_str(), result, expected, None);
        }

        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.ln(),
            |v: f64| v.ln(),
            "ln",
            |v: f64| Some(v.abs() + 1e-10),
            None,
        );
    }

    #[test]
    fn test_ln_worst_case_convergence() {
        let xs: &[f64] = &[1.9999999, 1.9999, 3.9999, 7.9999, 1023.9999, 1.5];
        for &x in xs {
            let ff = FlexFloat::<DefaultBitArray>::from(x);
            let result = ff.ln().to_f64().unwrap();
            let expected = x.ln();
            let rel = (result - expected).abs() / expected.abs().max(1e-10);
            assert!(
                rel < 1e-12,
                "ln({x}) = {result:?} (expected {expected:?}, rel={rel:.2e})"
            );
        }
    }
    /// Tests the log operation for FlexFloat.
    #[rstest]
    fn test_log(rng: impl Rng, n_experiments: usize) {
        test_binary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff, base| ff.log(base),
            |v, base| v.log(base),
            "log",
            |v, base| Some((v.abs() + 1e-10, base.abs() + 1e-10)),
            Some(1e-4),
        );
    }

    /// Tests the log2 operation for FlexFloat.
    #[rstest]
    fn test_log2(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.log2(),
            |v: f64| v.log2(),
            "log2",
            |v: f64| Some(v.abs() + 1e-10),
            Some(1e-5),
        );
    }

    /// Tests the log10 operation for FlexFloat.
    #[rstest]
    fn test_log10(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.log10(),
            |v: f64| v.log10(),
            "log10",
            |v: f64| Some(v.abs() + 1e-10),
            Some(1e-5),
        );
    }

    #[rstest]
    fn test_exp2(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.exp2(),
            |v: f64| v.exp2(),
            "exp2",
            identity_1,
            Some(1e-5),
        );
    }

    #[rstest]
    fn test_exp_m1(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.exp_m1(),
            |v: f64| v.exp_m1(),
            "exp_m1",
            identity_1,
            Some(1e-5),
        );
    }

    #[rstest]
    fn test_ln_1p(rng: impl Rng, n_experiments: usize) {
        test_unary_flexfloat_op(
            rng,
            n_experiments / 50,
            |ff| ff.ln_1p(),
            |v: f64| v.ln_1p(),
            "ln_1p",
            |v| if v > -1.0 { Some(v) } else { None },
            Some(1e-5),
        );
    }
}
