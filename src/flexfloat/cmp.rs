//! # Comparison Operations
//!
//! Implements comparison traits for FlexFloat, including PartialEq and PartialOrd.
//!
//! ## Overview
//!
//! This module provides equality and ordering comparisons for FlexFloat types,
//! with correct handling of special values (NaN, Infinity) according to IEEE 754 semantics.
//!
//! ## IEEE 754 Comparison Semantics
//!
//! - NaN is not equal to any value, including itself
//! - NaN comparisons with any value (including NaN) return `None` for ordering
//! - Positive and negative zero are considered equal
//! - Infinities compare as expected (+∞ > any finite, -∞ < any finite)
//!
//! ## Implemented Traits
//!
//! - **PartialEq**: Equality comparison, returns false for any NaN comparison
//! - **PartialOrd**: Partial ordering, returns None for any NaN comparison
//!
//! ## Examples
//!
//! ```rust
//! use flexfloat::prelude::*;
//!
//! let a = FlexFloat::from(1.5);
//! let b = FlexFloat::from(2.5);
//! assert!(a < b);
//! assert!(a != FlexFloat::nan());
//! ```

use core::cmp::Ordering;

use crate::bitarray::{BitArrayAccess, BitArrayConversion};
use crate::flexfloat::FlexFloat;

/// Equality comparison for FlexFloat.
///
/// Implements the [`PartialEq`] trait for `FlexFloat`, returning `false` if either operand is NaN.
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// let a = FlexFloat::from(1.5);
/// let b = FlexFloat::from(2.5);
/// assert_eq!(a, a);
/// assert_ne!(a, b);
/// let nan = FlexFloat::nan();
/// assert!(nan != nan);
/// ```
impl<B1, B2> PartialEq<FlexFloat<B1>> for FlexFloat<B2>
where
    B1: BitArrayAccess + BitArrayConversion,
    B2: BitArrayAccess + BitArrayConversion,
{
    fn eq(&self, other: &FlexFloat<B1>) -> bool {
        if self.is_nan() || other.is_nan() {
            return false;
        }
        if self.is_zero() && other.is_zero() {
            return true;
        }
        self.sign == other.sign
            && self.exponent.to_bigint() == other.exponent.to_bigint()
            && self.fraction.to_biguint() == other.fraction.to_biguint()
    }
}

/// Partial ordering for FlexFloat.
///
/// Implements the [`PartialOrd`] trait for `FlexFloat`, returning `None` if either operand is NaN.
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
/// let a = FlexFloat::from(1.5);
/// let b = FlexFloat::from(2.5);
/// assert!(a < b);
/// let nan = FlexFloat::nan();
/// assert!(nan.partial_cmp(&nan).is_none());
/// ```
impl<B1, B2> PartialOrd<FlexFloat<B1>> for FlexFloat<B2>
where
    B1: BitArrayConversion,
    B2: BitArrayConversion,
{
    fn partial_cmp(&self, other: &FlexFloat<B1>) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            return None;
        }
        if self.is_zero() && other.is_zero() {
            return Some(Ordering::Equal);
        }

        let sign = self.sign.cmp(&other.sign);
        let exponent = self.exponent.to_bigint().cmp(&other.exponent.to_bigint());
        let fraction = self.fraction.to_biguint().cmp(&other.fraction.to_biguint());

        let sign_ordering = if self.sign {
            Ordering::Less
        } else {
            Ordering::Greater
        };
        match (sign, exponent, fraction) {
            (Ordering::Equal, Ordering::Equal, Ordering::Equal) => Some(Ordering::Equal),
            (Ordering::Less, _, _) => Some(Ordering::Greater),
            (Ordering::Greater, _, _) => Some(Ordering::Less),

            (Ordering::Equal, Ordering::Greater, _) => Some(sign_ordering),
            (Ordering::Equal, Ordering::Equal, Ordering::Greater) => Some(sign_ordering),
            _ => Some(sign_ordering.reverse()),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::tests::*;

    /// Tests the `PartialEq` implementation for `FlexFloat`, including NaN and random values.
    #[rstest]
    fn test_partial_eq(mut rng: impl Rng, n_experiments: usize) {
        let a = FlexFloat::from(1.5);
        let b = FlexFloat::from(2.5);
        assert_eq!(a, a);
        assert_ne!(a, b);
        assert_eq!(FlexFloat::zero(), FlexFloat::zero_with_sign(true));

        let nan = FlexFloat::nan();
        assert!(nan != nan);
        assert_ne!(nan, FlexFloat::from(1.0));
        assert_eq!(FlexFloat::pos_infinity(), FlexFloat::pos_infinity());
        assert_eq!(FlexFloat::neg_infinity(), FlexFloat::neg_infinity());
        assert_ne!(FlexFloat::pos_infinity(), FlexFloat::neg_infinity());

        for _ in 0..n_experiments {
            let a = random_f64(&mut rng);
            let b = random_f64(&mut rng);
            let expected1 = a == b;
            let expected2 = a == a;

            let fa = FlexFloat::from(a);
            let fb = FlexFloat::from(b);
            let result1 = fa == fb;
            let result2 = fa == fa;

            assert_eq!(result1, expected1);
            assert_eq!(result2, expected2);
        }
    }

    /// Tests the `PartialOrd` implementation for `FlexFloat`, including NaN and random values.
    #[rstest]
    #[allow(clippy::bool_comparison)]
    fn test_partial_ord(mut rng: impl Rng, n_experiments: usize) {
        let a = FlexFloat::from(1.5);
        let b = FlexFloat::from(2.5);
        assert!(a < b,);
        assert!((a >= b) == false);
        assert_eq!(
            FlexFloat::zero().partial_cmp(&FlexFloat::zero_with_sign(true)),
            Some(Ordering::Equal)
        );

        let nan = FlexFloat::nan();
        assert!((nan > nan) == false);
        assert_eq!(nan.partial_cmp(&nan), None);
        assert_eq!(nan.partial_cmp(&FlexFloat::from(1.0)), None);
        assert_eq!(
            FlexFloat::pos_infinity().partial_cmp(&FlexFloat::from(1.0)),
            Some(Ordering::Greater)
        );
        assert_eq!(
            FlexFloat::neg_infinity().partial_cmp(&FlexFloat::from(1.0)),
            Some(Ordering::Less)
        );

        for _ in 0..n_experiments {
            let a = random_f64(&mut rng);
            let b = random_f64(&mut rng);
            let expected1 = a < b;
            let expected2 = a >= a;

            let fa = FlexFloat::from(a);
            let fb = FlexFloat::from(b);
            let result1 = fa < fb;
            let result2 = fa >= fa;

            assert_eq!(result1, expected1);
            assert_eq!(result2, expected2);
        }
    }
}
