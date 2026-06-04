//! Arithmetic operations for bit-array implementations.
//!
//! This module defines [`BitArrayArith`], a supertrait of [`BitArray`] for
//! backends that additionally support bignum-style arithmetic.

use core::ops::{Add, Div, Mul, Sub};

use super::BitArray;

/// Extension of [`BitArray`] for implementations that support bignum arithmetic.
///
/// Not every bit-array backend needs to implement arithmetic; general consumers
/// of bit sequences (set membership, bloom filters, bitmaps, …) only depend on
/// [`BitArray`]. Arithmetic-capable consumers (such as `FlexFloat`) narrow the
/// bound to `BitArrayArith`.
///
/// ## Length contract
///
/// The `Add`, `Sub`, `Mul`, `Div` impls produce a result whose length is **at
/// least** the natural bit-length of the mathematical result, but is otherwise
/// unspecified.  Implementations **MAY** return a result longer than strictly
/// necessary.  Callers that need a fixed-width result **MUST** call
/// `truncate(n_bits)` afterwards.
///
/// * `Sub` panics on underflow (result would be negative).
/// * `Div` panics on division by zero.
pub trait BitArrayArith:
    BitArray + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
}

/// Blanket implementation: any type that satisfies all bounds is `BitArrayArith`.
impl<T> BitArrayArith for T where
    T: BitArray + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
}
