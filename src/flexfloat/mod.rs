//! # FlexFloat Core Module
//!
//! Core implementation of the FlexFloat arbitrary-precision floating-point type.
//! Provides IEEE 754-compatible floating-point operations with growable exponents.
//!
//! ## Overview
//!
//! FlexFloat extends the IEEE 754 double-precision format by allowing the exponent
//! field to grow dynamically while maintaining a fixed 52-bit fraction. This enables
//! representation of numbers far beyond the range of standard double-precision floats.
//!
//! ## Structure
//!
//! FlexFloat consists of three components:
//! - **Sign bit**: 1 bit indicating positive (false) or negative (true)
//! - **Exponent**: Variable-width signed integer (starts at 11 bits, grows as needed)
//! - **Fraction/Mantissa**: Fixed 52-bit fractional part for precision consistency
//!
//! ## Special Values
//!
//! FlexFloat supports all IEEE 754 special values:
//! - **Zero**: ±0.0 with separate positive and negative representations
//! - **Infinity**: ±∞ for overflow conditions
//! - **NaN**: Not-a-Number for undefined operations
//!
//! ## Examples
//!
//! ```rust
//! use flexfloat::prelude::*;
//!
//! let x = FlexFloat::from(3.14159);
//! let zero = FlexFloat::zero();
//! let inf = FlexFloat::pos_infinity();
//!
//! assert!(zero.is_zero());
//! assert!(inf.is_infinite());
//! ```

use core::fmt::Debug;

use crate::bitarray::BitArray;

pub mod accessors;
pub mod arithmetic;
pub mod classify;
pub mod cmp;
pub mod construct;
pub mod consts;
pub mod converter;
pub mod display;
pub mod error;
pub mod internal;
pub mod math;
pub mod order;

// Re-export the pub(crate) internals so existing paths like
// `crate::flexfloat::grow_exponent` keep working.
pub(crate) use internal::{RoundedResult, grow_exponent, truncate_fraction};

/// A flexible-precision floating-point number with growable exponents.
///
/// FlexFloat represents floating-point numbers using a configurable bit array
/// backend, allowing for arbitrary-precision arithmetic while maintaining
/// compatibility with IEEE 754 standards.
///
/// # Type Parameters
///
/// * `Exp` - The [`BitArray`] implementation used for the exponent field.
/// * `Frac` - The [`BitArray`] implementation used for the fraction field
///   (defaults to `Exp` so that `FlexFloat<B, B>` is a valid shorthand).
///
/// # Components
///
/// - `sign`: Boolean indicating number sign (false = positive, true = negative)
/// - `exponent`: Variable-width signed exponent field (minimum 11 bits)
/// - `fraction`: Fixed-width mantissa/fraction field (52 bits)
#[derive(Clone)]
pub struct FlexFloat<Exp, Frac = Exp> {
    pub(crate) sign: bool,
    pub(crate) exponent: Exp,
    pub(crate) fraction: Frac,
}

impl<Exp: BitArray, Frac: BitArray> Debug for FlexFloat<Exp, Frac> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FlexFloat")
            .field("sign", if self.sign { &'-' } else { &'+' })
            .field("exponent", &(self.exponent.to_bigint() + 1_u8))
            .field("fraction", &self.fraction.to_biguint())
            .field("f64_value", &self.to_f64())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;
    use core::num::FpCategory;

    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::bitarray::DefaultBitArray;
    use crate::bitarray::{BitArrayAccess, BitArrayConversion};
    use crate::flexfloat::internal::grow_exponent;
    use crate::test_support::*;

    #[rstest]
    fn test_grow_exponent(mut rng: impl Rng, n_experiments: usize) {
        use num_bigint::BigInt;

        let value = BigInt::from(127);
        let bit_array: DefaultBitArray = grow_exponent(value.clone(), 8);
        assert_eq!(bit_array.to_bigint(), value);

        let value = BigInt::from(0b11111);
        let bit_array: DefaultBitArray = grow_exponent(value.clone(), 0);
        assert_eq!(bit_array.to_bigint(), value);

        for _ in 0..n_experiments {
            let n_bits = rng.random_range(1..100);
            let value = random_bigint(&mut rng, n_bits);
            let min_n_bits = rng.random_range(1..100);

            let bit_array: DefaultBitArray = grow_exponent(value.clone(), min_n_bits);
            assert_eq!(bit_array.to_bigint(), value);
            assert!(
                bit_array.len() >= min_n_bits,
                "{} >= {min_n_bits}",
                bit_array.len()
            );
        }
    }

    #[rstest]
    fn test_classification_predicates(mut rng: impl Rng, n_experiments: usize) {
        let normal = FlexFloat::<DefaultBitArray>::from(1.5);
        assert!(normal.is_finite());
        assert!(normal.is_normal());
        assert!(!normal.is_subnormal());

        let zero = FlexFloat::<DefaultBitArray>::zero();
        assert!(zero.is_finite());
        assert!(!zero.is_normal());
        assert!(!zero.is_subnormal());

        let nan = FlexFloat::<DefaultBitArray>::nan();
        assert!(!nan.is_finite());
        assert!(!nan.is_normal());
        assert!(!nan.is_subnormal());

        let pos_infinity = FlexFloat::<DefaultBitArray>::pos_infinity();
        assert!(!pos_infinity.is_finite());
        assert!(!pos_infinity.is_normal());
        assert!(!pos_infinity.is_subnormal());

        let neg_infinity = FlexFloat::<DefaultBitArray>::neg_infinity();
        assert!(!neg_infinity.is_finite());
        assert!(!neg_infinity.is_normal());
        assert!(!neg_infinity.is_subnormal());

        let subnormal = FlexFloat::<DefaultBitArray>::from(f64::from_bits(1));
        assert!(subnormal.is_finite());
        assert!(!subnormal.is_normal());
        assert!(
            subnormal.is_subnormal(),
            "expected subnormal, got {subnormal:?}"
        );

        let huge =
            FlexFloat::<DefaultBitArray>::from(f64::MAX) * FlexFloat::<DefaultBitArray>::from(2.0);
        assert!(huge.exponent().len() > 11, "precondition: exponent grew");
        assert!(huge.is_finite());
        assert!(huge.is_normal());
        assert!(!huge.is_subnormal());

        for _ in 0..n_experiments {
            let value = random_f64(&mut rng);
            let ff = FlexFloat::<DefaultBitArray>::from(value);

            assert_eq!(ff.is_finite(), value.is_finite(), "is_finite({value:?})");
            assert_eq!(ff.is_normal(), value.is_normal(), "is_normal({value:?})");
            assert_eq!(
                ff.is_subnormal(),
                value.is_subnormal(),
                "is_subnormal({value:?})"
            );
        }
    }

    #[test]
    fn test_classify_min_max_clamp_and_adjacent_helpers() {
        let nan = FlexFloat::<DefaultBitArray>::nan();
        let neg_zero = FlexFloat::<DefaultBitArray>::zero_with_sign(true);
        let pos_zero = FlexFloat::<DefaultBitArray>::zero();
        let one = FlexFloat::<DefaultBitArray>::from(1.0);
        let two = FlexFloat::<DefaultBitArray>::from(2.0);
        let subnormal = FlexFloat::<DefaultBitArray>::from(f64::from_bits(1));

        assert_eq!(nan.classify(), FpCategory::Nan);
        assert_eq!(
            FlexFloat::<DefaultBitArray>::pos_infinity().classify(),
            FpCategory::Infinite
        );
        assert_eq!(pos_zero.classify(), FpCategory::Zero);
        assert_eq!(subnormal.classify(), FpCategory::Subnormal);
        assert_eq!(one.classify(), FpCategory::Normal);

        assert_eq!(one.clone().min(two.clone()).to_f64(), Some(1.0));
        assert_eq!(one.clone().max(two.clone()).to_f64(), Some(2.0));
        assert_eq!(nan.clone().min(one.clone()).to_f64(), Some(1.0));
        assert_eq!(nan.max(one.clone()).to_f64(), Some(1.0));
        assert_eq!(
            pos_zero
                .clone()
                .min(neg_zero.clone())
                .to_f64()
                .unwrap()
                .to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(
            pos_zero
                .clone()
                .max(neg_zero.clone())
                .to_f64()
                .unwrap()
                .to_bits(),
            0.0f64.to_bits()
        );

        assert_eq!(
            FlexFloat::<DefaultBitArray>::from(3.0)
                .clamp(one.clone(), two.clone())
                .to_f64(),
            Some(2.0)
        );
        assert_eq!(
            FlexFloat::<DefaultBitArray>::from(-1.0)
                .clamp(one.clone(), two.clone())
                .to_f64(),
            Some(1.0)
        );
        assert_eq!(
            FlexFloat::<DefaultBitArray>::from(1.5)
                .clamp(one.clone(), two.clone())
                .to_f64(),
            Some(1.5)
        );

        assert_eq!(
            pos_zero.clone().next_down().to_f64().unwrap().to_bits(),
            (-f64::from_bits(1)).to_bits()
        );
        assert_eq!(
            neg_zero.clone().next_up().to_f64().unwrap().to_bits(),
            f64::from_bits(1).to_bits()
        );
        assert_eq!(
            FlexFloat::<DefaultBitArray>::from(1.0)
                .next_up()
                .to_f64()
                .unwrap()
                .to_bits(),
            1.0f64.to_bits() + 1
        );
        assert_eq!(
            FlexFloat::<DefaultBitArray>::from(1.0)
                .next_down()
                .to_f64()
                .unwrap()
                .to_bits(),
            1.0f64.to_bits() - 1
        );

        assert_eq!(neg_zero.total_cmp(&pos_zero), Ordering::Less);
        assert_eq!(pos_zero.total_cmp(&neg_zero), Ordering::Greater);
        assert_eq!(
            FlexFloat::<DefaultBitArray>::from(1.0).total_cmp(&FlexFloat::from(2.0)),
            1.0f64.total_cmp(&2.0)
        );
    }
}
