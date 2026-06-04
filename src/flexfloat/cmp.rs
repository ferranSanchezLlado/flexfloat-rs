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
use core::num::FpCategory;

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
impl<E1, F1, E2, F2> PartialEq<FlexFloat<E1, F1>> for FlexFloat<E2, F2>
where
    E1: BitArrayAccess + BitArrayConversion,
    F1: BitArrayAccess + BitArrayConversion,
    E2: BitArrayAccess + BitArrayConversion,
    F2: BitArrayAccess + BitArrayConversion,
{
    fn eq(&self, other: &FlexFloat<E1, F1>) -> bool {
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
impl<E1, F1, E2, F2> PartialOrd<FlexFloat<E1, F1>> for FlexFloat<E2, F2>
where
    E1: BitArrayConversion,
    F1: BitArrayConversion,
    E2: BitArrayConversion,
    F2: BitArrayConversion,
{
    fn partial_cmp(&self, other: &FlexFloat<E1, F1>) -> Option<Ordering> {
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

fn rank_total_cmp<Exp: BitArrayAccess, Frac: BitArrayAccess>(value: &FlexFloat<Exp, Frac>) -> u8 {
    match (value.sign, value.classify()) {
        (true, FpCategory::Nan) => 0,
        (true, FpCategory::Infinite) => 1,
        (true, FpCategory::Normal | FpCategory::Subnormal) => 2,
        (true, FpCategory::Zero) => 3,
        (false, FpCategory::Zero) => 4,
        (false, FpCategory::Normal | FpCategory::Subnormal) => 5,
        (false, FpCategory::Infinite) => 6,
        (false, FpCategory::Nan) => 7,
    }
}

impl<Exp: BitArrayConversion + BitArrayAccess, Frac: BitArrayConversion + BitArrayAccess>
    FlexFloat<Exp, Frac>
{
    pub fn total_cmp(&self, other: &Self) -> Ordering {
        let rank_cmp = rank_total_cmp(self).cmp(&rank_total_cmp(other));
        if rank_cmp != Ordering::Equal {
            return rank_cmp;
        }

        if self.is_nan() {
            return self
                .exponent
                .to_bigint()
                .cmp(&other.exponent.to_bigint())
                .then_with(|| self.fraction.to_biguint().cmp(&other.fraction.to_biguint()));
        }

        if self.is_zero() {
            return Ordering::Equal;
        }

        match self.partial_cmp(other) {
            Some(Ordering::Equal) => self
                .exponent
                .to_bigint()
                .cmp(&other.exponent.to_bigint())
                .then_with(|| self.fraction.to_biguint().cmp(&other.fraction.to_biguint())),
            Some(ordering) => ordering,
            None => Ordering::Equal,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::bitarray::DefaultBitArray;
    use crate::test_support::*;

    /// Tests the `PartialEq` implementation for `FlexFloat`, including NaN and random values.
    #[rstest]
    fn test_partial_eq(mut rng: impl Rng, n_experiments: usize) {
        let a = FlexFloat::from(1.5);
        let b = FlexFloat::from(2.5);
        assert_eq!(a, a);
        assert_ne!(a, b);
        assert_eq!(
            FlexFloat::<DefaultBitArray>::zero(),
            FlexFloat::<DefaultBitArray>::zero_with_sign(true)
        );

        let nan = FlexFloat::<DefaultBitArray>::nan();
        assert!(nan != nan);
        assert_ne!(nan, FlexFloat::from(1.0));
        assert_eq!(
            FlexFloat::<DefaultBitArray>::pos_infinity(),
            FlexFloat::<DefaultBitArray>::pos_infinity()
        );
        assert_eq!(
            FlexFloat::<DefaultBitArray>::neg_infinity(),
            FlexFloat::<DefaultBitArray>::neg_infinity()
        );
        assert_ne!(
            FlexFloat::<DefaultBitArray>::pos_infinity(),
            FlexFloat::<DefaultBitArray>::neg_infinity()
        );

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
            FlexFloat::<DefaultBitArray>::zero()
                .partial_cmp(&FlexFloat::<DefaultBitArray>::zero_with_sign(true)),
            Some(Ordering::Equal)
        );

        let nan = FlexFloat::<DefaultBitArray>::nan();
        assert!((nan > nan) == false);
        assert_eq!(nan.partial_cmp(&nan), None);
        assert_eq!(nan.partial_cmp(&FlexFloat::from(1.0)), None);
        assert_eq!(
            FlexFloat::<DefaultBitArray>::pos_infinity().partial_cmp(&FlexFloat::from(1.0)),
            Some(Ordering::Greater)
        );
        assert_eq!(
            FlexFloat::<DefaultBitArray>::neg_infinity().partial_cmp(&FlexFloat::from(1.0)),
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

    #[rstest]
    fn test_total_cmp(mut rng: impl Rng, n_experiments: usize) {
        let a = FlexFloat::from(1.5);
        let b = FlexFloat::from(2.5);
        assert_eq!(a.total_cmp(&b), Ordering::Less);
        assert_eq!(a.total_cmp(&a), Ordering::Equal);

        let nan = FlexFloat::<DefaultBitArray>::nan();
        assert_eq!(nan.total_cmp(&nan), Ordering::Equal);
        assert_eq!(
            nan.total_cmp(&FlexFloat::from(1.0)),
            f64::NAN.total_cmp(&1.0)
        );
        assert_eq!(
            FlexFloat::<DefaultBitArray>::pos_infinity().total_cmp(&FlexFloat::from(1.0)),
            Ordering::Greater
        );
        assert_eq!(
            FlexFloat::<DefaultBitArray>::neg_infinity().total_cmp(&FlexFloat::from(1.0)),
            Ordering::Less
        );

        for _ in 0..n_experiments {
            let a = random_f64(&mut rng);
            let b = random_f64(&mut rng);
            let expected = a.total_cmp(&b);

            let fa = FlexFloat::from(a);
            let fb = FlexFloat::from(b);
            let result = fa.total_cmp(&fb);

            assert_eq!(result, expected);
        }
    }
}
