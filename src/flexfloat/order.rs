//! Ordering utilities for FlexFloat: min, max, clamp, next_up, next_down,
//! and the low-level `adjacent` helper used by `next_up`/`next_down`.

use num_bigint::BigUint;
use num_traits::{One, Zero};

use crate::bitarray::BitArray;
use crate::flexfloat::FlexFloat;
use crate::prelude::{BitArrayAccess, BitArrayConstruction, BitArrayConversion};

/// Direction of adjacency step used by [`adjacent`].
#[derive(Clone, Copy)]
pub(crate) enum Direction {
    Up,
    Down,
}

/// Returns the floating-point value adjacent to `value` in the given direction.
///
/// Equivalent to `next_up` / `next_down` but without NaN / infinity guard —
/// callers must ensure `value` is finite and non-NaN.
pub(crate) fn adjacent<Exp: BitArray, Frac: BitArray>(
    value: &FlexFloat<Exp, Frac>,
    direction: Direction,
) -> FlexFloat<Exp, Frac> {
    debug_assert!(!value.is_nan());
    debug_assert!(!value.is_infinite());

    if value.is_zero() {
        return smallest_subnormal(
            matches!(direction, Direction::Down),
            value.exponent.len(),
            value.fraction.len(),
        );
    }

    let toward_more_negative = match direction {
        Direction::Up => value.sign,
        Direction::Down => !value.sign,
    };

    let exponent_bits = value.exponent.len();
    let fraction_bits = value.fraction.len();
    let exponent_max = (BigUint::one() << exponent_bits) - 1u8;

    let mut exponent = value.exponent.to_biguint();
    let mut fraction = value.fraction.to_biguint();

    if toward_more_negative {
        if exponent.is_zero() {
            if fraction.is_zero() {
                return smallest_subnormal(true, exponent_bits, fraction_bits);
            }
            fraction -= 1u8;
            return FlexFloat::new(
                value.sign,
                Exp::from_biguint_fixed(&exponent, exponent_bits),
                Frac::from_biguint_fixed(&fraction, fraction_bits),
            );
        }

        if fraction.is_zero() {
            exponent -= 1u8;
            fraction = (BigUint::one() << fraction_bits) - 1u8;
        } else {
            fraction -= 1u8;
        }
    } else if exponent.is_zero() {
        fraction += 1u8;
        if fraction >= (BigUint::one() << fraction_bits) {
            exponent = BigUint::one();
            fraction = BigUint::zero();
        }
    } else {
        fraction += 1u8;
        if fraction >= (BigUint::one() << fraction_bits) {
            fraction = BigUint::zero();
            exponent += 1u8;
            if exponent >= exponent_max {
                return FlexFloat::infinity(value.sign);
            }
        }
    }

    FlexFloat::new(
        value.sign,
        Exp::from_biguint_fixed(&exponent, exponent_bits),
        Frac::from_biguint_fixed(&fraction, fraction_bits),
    )
}

fn smallest_subnormal<Exp: BitArrayConstruction, Frac: BitArrayConstruction>(
    negative: bool,
    exponent_bits: usize,
    fraction_bits: usize,
) -> FlexFloat<Exp, Frac> {
    FlexFloat {
        sign: negative,
        exponent: Exp::zeros(exponent_bits),
        fraction: Frac::from_biguint_fixed(&BigUint::one(), fraction_bits),
    }
}

impl<Exp: BitArrayAccess + BitArrayConversion, Frac: BitArrayAccess + BitArrayConversion>
    FlexFloat<Exp, Frac>
{
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
}

impl<Frac: BitArray, Exp: BitArray> FlexFloat<Exp, Frac> {
    pub fn next_up(self) -> Self {
        if self.is_nan() || self == Self::infinity(false) {
            return self;
        }
        if self == Self::infinity(true) {
            return Self::new_infinity_with_largest_finite(false, self.exponent.len());
        }
        adjacent(&self, Direction::Up)
    }

    pub fn next_down(self) -> Self {
        if self.is_nan() || self == Self::infinity(true) {
            return self;
        }
        if self == Self::infinity(false) {
            return Self::new_infinity_with_largest_finite(true, self.exponent.len());
        }
        adjacent(&self, Direction::Down)
    }

    pub(crate) fn new_infinity_with_largest_finite(sign: bool, exponent_bits: usize) -> Self {
        Self {
            sign,
            exponent: Exp::ones(exponent_bits.saturating_sub(1)),
            fraction: Frac::ones(52),
        }
    }
}
