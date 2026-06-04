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
//! ## Key Features
//!
//! - **Growable exponents**: Automatically expand to accommodate larger values
//! - **Fixed precision**: Consistent 52-bit mantissa maintains precision
//! - **IEEE 754 compatibility**: Full support for special values (±0, ±∞, NaN)
//! - **Seamless conversion**: Easy interop with standard f64 values
//! - **Generic backing**: Configurable bit array implementation
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
//! // Basic construction
//! let x = FlexFloat::from(3.14159);
//! let zero = FlexFloat::zero();
//! let inf = FlexFloat::pos_infinity();
//!
//! // Special value checks
//! assert!(zero.is_zero());
//! assert!(inf.is_infinite());
//!
//! // Component access
//! println!("Sign: {}, Exponent bits: {}, Fraction bits: {}",
//!          x.sign(), x.exponent().len(), x.fraction().len());
//! ```

use core::cmp::{Ordering, max};
use core::fmt::Debug;
use core::num::FpCategory;
use core::str::FromStr;

use num_bigint::BigInt;

use crate::bitarray::static_boolean_array::StaticBoolArray;
use crate::bitarray::traits::BitArrayRounding;
use crate::bitarray::{BitArray, BitArrayAccess, BitArrayConversion, DefaultBitArray};

pub mod arithmetic;
pub mod cmp;
pub mod consts;
pub mod converter;
pub mod display;
pub mod math;

/// A flexible-precision floating-point number with growable exponents.
///
/// FlexFloat represents floating-point numbers using a configurable bit array
/// backend, allowing for arbitrary-precision arithmetic while maintaining
/// compatibility with IEEE 754 standards.
///
/// # Type Parameters
///
/// * `B` - The BitArray implementation used for internal storage (e.g., BoolBitArray)
///
/// # Components
///
/// - `sign`: Boolean indicating number sign (false = positive, true = negative)
/// - `exponent`: Variable-width signed exponent field (minimum 11 bits)
/// - `fraction`: Fixed-width mantissa/fraction field (52 bits)
///
/// # Memory Efficiency
///
/// The exponent field grows only when needed, starting at 11 bits (IEEE 754 standard)
/// and expanding to accommodate larger values. The fraction remains fixed at 52 bits
/// to maintain precision consistency across all operations.

#[derive(Clone)]
pub struct FlexFloat<B> {
    pub(crate) sign: bool,
    pub(crate) exponent: B,
    pub(crate) fraction: B,
}

impl<B: BitArray> FlexFloat<B> {
    /// Creates a new FlexFloat with the specified components.
    ///
    /// # Arguments
    ///
    /// * `sign` - Sign bit (false for positive, true for negative)
    /// * `exponent` - Exponent bit array
    /// * `fraction` - Fraction/mantissa bit array
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let exp = DefaultBitArray::from_bits(&[true, false, true]);
    /// let frac = DefaultBitArray::zeros(52);
    /// let num = FlexFloat::new(false, exp, frac);
    /// ```
    pub const fn new(sign: bool, exponent: B, fraction: B) -> Self {
        Self {
            sign,
            exponent,
            fraction,
        }
    }

    /// Creates a new FlexFloat representing zero with the specified sign.
    ///
    /// # Arguments
    ///
    /// * `sign` - Sign bit for the zero value
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let pos_zero: FlexFloat<DefaultBitArray> = FlexFloat::new_zero_with_sign(false);
    /// let neg_zero: FlexFloat<DefaultBitArray> = FlexFloat::new_zero_with_sign(true);
    ///
    /// assert!(pos_zero.is_zero() && !pos_zero.sign());
    /// assert!(neg_zero.is_zero() && neg_zero.sign());
    /// ```
    pub fn new_zero_with_sign(sign: bool) -> Self {
        Self {
            sign,
            exponent: B::zeros(11),
            fraction: B::zeros(52),
        }
    }

    /// Creates a new FlexFloat representing positive zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let zero: FlexFloat<DefaultBitArray> = FlexFloat::new_zero();
    /// assert!(zero.is_zero() && !zero.sign());
    /// ```
    pub fn new_zero() -> Self {
        Self::new_zero_with_sign(false)
    }

    /// Creates a new FlexFloat representing NaN (Not-a-Number).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let nan: FlexFloat<DefaultBitArray> = FlexFloat::new_nan();
    /// assert!(nan.is_nan());
    /// ```
    pub fn new_nan() -> Self {
        Self {
            sign: false, // Sign doesn't matter for NaN
            exponent: B::ones(11),
            fraction: B::ones(52),
        }
    }

    /// Creates a new FlexFloat representing infinity with the specified sign.
    ///
    /// # Arguments
    ///
    /// * `sign` - Sign bit for the infinity value
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let pos_inf: FlexFloat<DefaultBitArray> = FlexFloat::new_infinity(false);
    /// let neg_inf: FlexFloat<DefaultBitArray> = FlexFloat::new_infinity(true);
    /// assert!(pos_inf.is_infinite() && !pos_inf.sign());
    /// assert!(neg_inf.is_infinite() && neg_inf.sign());
    /// ```
    pub fn new_infinity(sign: bool) -> Self {
        Self {
            sign,
            exponent: B::ones(11),
            fraction: B::zeros(52),
        }
    }
}

impl<B: BitArrayAccess> FlexFloat<B> {
    /// Checks if the exponent represents a special value (all ones).
    ///
    /// Special exponents indicate infinity or NaN values in IEEE 754 format.
    fn is_special_exponent(&self) -> bool {
        self.exponent.iter_bits().all(|b| b)
    }

    /// Returns true if this FlexFloat represents NaN (Not-a-Number).
    ///
    /// A value is NaN if it has a special exponent (all ones) and a non-zero fraction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let nan = FlexFloat::nan();
    /// assert!(nan.is_nan());
    ///
    /// let normal = FlexFloat::from(3.14);
    /// assert!(!normal.is_nan());
    /// ```
    pub fn is_nan(&self) -> bool {
        self.is_special_exponent() && self.fraction.iter_bits().any(|b| b)
    }

    /// Returns true if this FlexFloat represents infinity.
    ///
    /// A value is infinity if it has a special exponent (all ones) and a zero fraction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let inf = FlexFloat::pos_infinity();
    /// assert!(inf.is_infinite());
    ///
    /// let finite = FlexFloat::from(1.0);
    /// assert!(!finite.is_infinite());
    /// ```
    pub fn is_infinite(&self) -> bool {
        self.is_special_exponent() && self.fraction.iter_bits().all(|b| !b)
    }

    /// Returns true if this FlexFloat represents zero.
    ///
    /// A value is zero if both the exponent and fraction are all zeros.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let zero = FlexFloat::zero();
    /// assert!(zero.is_zero());
    ///
    /// let non_zero = FlexFloat::from(0.1);
    /// assert!(!non_zero.is_zero());
    /// ```
    pub fn is_zero(&self) -> bool {
        self.exponent.iter_bits().all(|b| !b) && self.fraction.iter_bits().all(|b| !b)
    }

    /// Returns `true` if the value is finite - i.e., not NaN and not ±∞.
    ///
    /// This mirrors `f64::is_finite`. Zero, subnormals, and normals all
    /// return `true`.
    pub fn is_finite(&self) -> bool {
        !self.is_nan() && !self.is_infinite()
    }

    /// Returns `true` if the value is a normal floating-point number -
    /// finite, non-zero, and not subnormal.
    ///
    /// Mirrors `f64::is_normal`. NaN, ±∞, ±0, and subnormals return `false`.
    pub fn is_normal(&self) -> bool {
        self.is_finite() && !self.is_zero() && !self.is_subnormal()
    }

    /// Returns `true` if the value is subnormal (denormalized).
    ///
    /// Subnormals only exist at the standard IEEE 754 double-precision
    /// exponent width (11 bits). With extended exponents (>11 bits),
    /// values are always either normal or special - the extended range
    /// eliminates the need for gradual underflow.
    ///
    /// A value is subnormal when:
    /// - exponent width == 11 bits, AND
    /// - exponent field is all zeros, AND
    /// - fraction field is non-zero.
    pub fn is_subnormal(&self) -> bool {
        self.exponent.len() == 11
            && self.exponent.iter_bits().all(|b| !b)
            && self.fraction.iter_bits().any(|b| b)
    }

    /// Returns the IEEE 754 floating-point category for this value.
    pub fn classify(&self) -> FpCategory {
        if self.is_nan() {
            FpCategory::Nan
        } else if self.is_infinite() {
            FpCategory::Infinite
        } else if self.is_zero() {
            FpCategory::Zero
        } else if self.is_subnormal() {
            FpCategory::Subnormal
        } else {
            FpCategory::Normal
        }
    }
}

impl<B> FlexFloat<B> {
    /// Returns the sign bit of this FlexFloat.
    ///
    /// # Returns
    ///
    /// false for positive numbers, true for negative numbers
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let pos = FlexFloat::from(3.14);
    /// let neg = FlexFloat::from(-2.71);
    /// assert!(!pos.sign());
    /// assert!(neg.sign());
    /// ```
    pub fn sign(&self) -> bool {
        self.sign
    }

    /// Returns a reference to the exponent bit array.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let num = FlexFloat::from(8.0);
    /// println!("Exponent has {} bits", num.exponent().len());
    /// ```
    pub fn exponent(&self) -> &B {
        &self.exponent
    }

    /// Returns a reference to the fraction bit array.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let num = FlexFloat::from(3.14159);
    /// println!("Fraction has {} bits", num.fraction().len());
    /// ```
    pub fn fraction(&self) -> &B {
        &self.fraction
    }

    /// Returns `true` if the number is positive (i.e., the sign bit is `false`).
    ///
    /// # Returns
    ///
    /// * `true` if the value is positive.
    /// * `false` if the value is negative.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let pos = FlexFloat::from(1.23);
    /// let neg = FlexFloat::from(-4.56);
    /// assert!(pos.is_positive());
    /// assert!(!neg.is_positive());
    /// ```
    pub fn is_positive(&self) -> bool {
        !self.sign
    }

    /// Returns `true` if the number is negative (i.e., the sign bit is `true`).
    ///
    /// # Returns
    ///
    /// * `true` if the value is negative.
    /// * `false` if the value is positive.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let pos = FlexFloat::from(1.23);
    /// let neg = FlexFloat::from(-4.56);
    /// assert!(!pos.is_negative());
    /// assert!(neg.is_negative());
    /// ```
    pub fn is_negative(&self) -> bool {
        self.sign
    }
}

impl<B: BitArray + BitArrayAccess> FlexFloat<B> {
    pub fn min(self, other: Self) -> Self {
        if self.is_nan() {
            return other;
        }
        if other.is_nan() {
            return self;
        }
        if self < other {
            self
        } else if other < self {
            other
        } else if self.is_zero() && other.is_zero() && self.sign {
            self
        } else {
            other
        }
    }

    pub fn max(self, other: Self) -> Self {
        if self.is_nan() {
            return other;
        }
        if other.is_nan() {
            return self;
        }
        if self > other {
            self
        } else if other > self {
            other
        } else if self.is_zero() && other.is_zero() && !self.sign {
            self
        } else {
            other
        }
    }

    pub fn clamp(self, min: Self, max: Self) -> Self {
        assert!(!min.is_nan(), "min must not be NaN");
        assert!(!max.is_nan(), "max must not be NaN");
        assert!(min <= max, "min must be less than or equal to max");

        if self.is_nan() {
            return self;
        }
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }

    pub fn next_up(self) -> Self {
        if self.is_nan() || self == Self::new_infinity(false) {
            return self;
        }
        if self == Self::new_infinity(true) {
            return Self::new_infinity_with_largest_finite(false, self.exponent.len());
        }
        crate::flexfloat::display::adjacent(&self, crate::flexfloat::display::Direction::Up)
    }

    pub fn next_down(self) -> Self {
        if self.is_nan() || self == Self::new_infinity(true) {
            return self;
        }
        if self == Self::new_infinity(false) {
            return Self::new_infinity_with_largest_finite(true, self.exponent.len());
        }
        crate::flexfloat::display::adjacent(&self, crate::flexfloat::display::Direction::Down)
    }

    fn new_infinity_with_largest_finite(sign: bool, exponent_bits: usize) -> Self {
        Self {
            sign,
            exponent: B::ones(exponent_bits.saturating_sub(1)),
            fraction: B::ones(52),
        }
    }
}

impl<B: BitArrayConversion + BitArrayAccess> FlexFloat<B> {
    pub fn total_cmp(&self, other: &Self) -> Ordering {
        if let (Some(lhs), Some(rhs)) = (self.to_f64(), other.to_f64()) {
            return lhs.total_cmp(&rhs);
        }

        fn rank<B: BitArrayAccess>(value: &FlexFloat<B>) -> u8 {
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

        let rank_cmp = rank(self).cmp(&rank(other));
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

impl FromStr for FlexFloat<DefaultBitArray> {
    type Err = core::num::ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let value = f64::from_str(s)?;
        Ok(Self::from(value))
    }
}

impl FlexFloat<StaticBoolArray> {
    pub fn to_default_bitarray(&self) -> FlexFloat<DefaultBitArray> {
        FlexFloat {
            sign: self.sign,
            exponent: (&self.exponent).into(),
            fraction: (&self.fraction).into(),
        }
    }
}

impl FlexFloat<DefaultBitArray> {
    /// Creates a new `FlexFloat` instance representing positive zero.
    ///
    /// This is a convenience constructor for the default bit array implementation.
    ///
    /// # Returns
    ///
    /// A new `FlexFloat<DefaultBitArray>` with the value +0.0
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let zero = FlexFloat::zero();
    /// assert_eq!(zero.to_f64(), Some(0.0));
    /// assert!(!zero.sign());
    /// ```
    pub fn zero() -> Self {
        Self::new_zero()
    }

    /// Creates a new `FlexFloat` instance representing signed zero.
    ///
    /// This is a convenience constructor for the default bit array implementation.
    /// IEEE 754 distinguishes between +0 and -0.
    ///
    /// # Arguments
    ///
    /// * `sign` - The sign bit (false for +0, true for -0)
    ///
    /// # Returns
    ///
    /// A new `FlexFloat<DefaultBitArray>` representing ±0.0
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let pos_zero = FlexFloat::zero_with_sign(false);
    /// let neg_zero = FlexFloat::zero_with_sign(true);
    /// assert!(pos_zero.is_zero() && !pos_zero.sign());
    /// assert!(neg_zero.is_zero() && neg_zero.sign());
    /// ```
    pub fn zero_with_sign(sign: bool) -> Self {
        Self::new_zero_with_sign(sign)
    }

    /// Creates a new `FlexFloat` instance representing NaN (Not-a-Number).
    ///
    /// This is a convenience constructor that creates a `FlexFloat` with the value of NaN
    /// using the default bit array representation.
    ///
    /// # Returns
    ///
    /// A new `FlexFloat<DefaultBitArray>` instance representing NaN.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let nan = FlexFloat::nan();
    /// assert!(nan.is_nan());
    /// ```
    pub fn nan() -> Self {
        Self::new_nan()
    }

    /// Creates a new `FlexFloat` instance representing positive infinity.
    ///
    /// This is a convenience constructor that creates a `FlexFloat` with the value of +inf
    /// using the default bit array representation.
    ///
    /// # Returns
    ///
    /// A new `FlexFloat<DefaultBitArray>` instance representing positive infinity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let pos_inf = FlexFloat::pos_infinity();
    /// assert!(pos_inf.is_infinite());
    /// assert!(!pos_inf.sign());
    /// ```
    pub fn pos_infinity() -> Self {
        Self::new_infinity(false)
    }

    /// Creates a new `FlexFloat` instance representing negative infinity.
    ///
    /// This is a convenience constructor that creates a `FlexFloat` with the value of -inf
    /// using the default bit array representation.
    ///
    /// # Returns
    ///
    /// A new `FlexFloat<DefaultBitArray>` instance representing negative infinity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let neg_inf = FlexFloat::neg_infinity();
    /// assert!(neg_inf.is_infinite());
    /// assert!(neg_inf.sign());
    /// ```
    pub fn neg_infinity() -> Self {
        Self::new_infinity(true)
    }

    /// Creates a new `FlexFloat` instance representing infinity with the specified sign.
    ///
    /// This is a convenience constructor that creates a `FlexFloat` with the value of inf
    /// using the default bit array representation.
    ///
    /// # Arguments
    ///
    /// * `sign` - The sign bit (false for +inf, true for -inf)
    ///
    /// # Returns
    ///
    /// A new `FlexFloat<DefaultBitArray>` instance representing infinity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let pos_inf = FlexFloat::infinity(false);
    /// let neg_inf = FlexFloat::infinity(true);
    /// assert!(pos_inf.is_infinite() && !pos_inf.sign());
    /// assert!(neg_inf.is_infinite() && neg_inf.sign());
    /// ```
    pub fn infinity(sign: bool) -> Self {
        Self::new_infinity(sign)
    }
}

impl<B: BitArray> Debug for FlexFloat<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        {
            let mut float = f.debug_struct("FlexFloat");

            float
                .field("sign", if self.sign { &'-' } else { &'+' })
                .field("exponent", &(self.exponent.to_bigint() + 1_u8))
                .field("fraction", &self.fraction.to_biguint());

            if let Some(value) = self.to_f64() {
                float.field("f64", &value);
            }

            float.finish()
        }
    }
}

pub(crate) fn grow_exponent<B: BitArray>(exponent: BigInt, min_bits: usize) -> B {
    let n_bits = exponent.bits();
    let ones = exponent
        .iter_u64_digits()
        .fold(0_u64, |count, el| count + u64::from(el.count_ones()));
    let needs_to_grow = usize::from(ones == n_bits);

    // +1 for the sign in signed vs unsigned
    let n_bits = max(n_bits as usize + needs_to_grow + 1, min_bits);
    B::from_bigint(&exponent, n_bits).unwrap()
}

/// Result of truncating a fraction to a target size with IEEE 754 rounding.
///
/// `carry` is `true` when round-to-nearest caused the fraction to overflow
/// its target size. Callers must add 1 to the exponent and shift the
/// fraction right by 1 in that case (this is already done internally;
/// `carry` is exposed so callers can update the exponent).
#[derive(Debug, Clone)]
pub(crate) struct RoundedResult<B: BitArray> {
    /// The rounded fraction, exactly `size` bits long.
    pub fraction: B,
    /// True if rounding overflowed the fraction (carry into exponent).
    pub carry: bool,
}

pub(crate) fn truncate_fraction<B: BitArray>(fraction: B, size: usize) -> RoundedResult<B> {
    let shift = size as isize - fraction.len() as isize;

    match shift.cmp(&0) {
        Ordering::Equal => RoundedResult {
            fraction,
            carry: false,
        },
        Ordering::Greater => RoundedResult {
            fraction: fraction.shift_grow(shift),
            carry: false,
        },
        Ordering::Less => {
            let shift_abs = shift.unsigned_abs();

            let lsb = fraction.get(shift_abs).unwrap();
            let guard = fraction.get(shift_abs - 1).unwrap();
            let rest = fraction.iter_bits().take(shift_abs - 1).any(|el| el);

            let rounding = guard && (lsb || rest);

            let mut truncated = fraction.shift_fixed(-shift).truncate(size);
            if rounding {
                truncated.add_one_in_place();
                if truncated.len() > size {
                    RoundedResult {
                        fraction: truncated.shift_fixed(1).truncate(size),
                        carry: true,
                    }
                } else {
                    RoundedResult {
                        fraction: truncated,
                        carry: false,
                    }
                }
            } else {
                RoundedResult {
                    fraction: truncated,
                    carry: false,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::bitarray::BitArrayConversion;
    use crate::tests::*;

    /// Tests the grow_exponent utility for correct bit array growth and value preservation.
    #[rstest]
    fn test_grow_exponent(mut rng: impl Rng, n_experiments: usize) {
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
            1.0f64.next_up().to_bits()
        );
        assert_eq!(
            FlexFloat::<DefaultBitArray>::from(1.0)
                .next_down()
                .to_f64()
                .unwrap()
                .to_bits(),
            1.0f64.next_down().to_bits()
        );

        assert_eq!(neg_zero.total_cmp(&pos_zero), Ordering::Less);
        assert_eq!(pos_zero.total_cmp(&neg_zero), Ordering::Greater);
        assert_eq!(
            FlexFloat::<DefaultBitArray>::from(1.0).total_cmp(&FlexFloat::from(2.0)),
            1.0f64.total_cmp(&2.0)
        );
    }
}
