//! # Arithmetic Operations
//!
//! Implements fundamental arithmetic operations for FlexFloat with automatic
//! exponent growth and precision preservation.
//!
//! ## Overview
//!
//! This module provides arithmetic operations that maintain FlexFloat's key
//! characteristics: growing exponents when needed and consistent precision
//! through the fixed-width fraction field.
//!
//! ## Exponent Growth Strategy
//!
//! When arithmetic operations produce results that exceed the current exponent
//! range, the exponent field automatically grows to accommodate the new value.
//! This ensures that FlexFloat can represent arbitrarily large or small numbers
//! without overflow.
//!
//! ## Implemented Operations
//!
//! - **Negation**: Sign bit flipping with `Neg` trait
//! - **Absolute value**: Sign bit clearing
//! - **Addition/Subtraction**: With automatic exponent expansion
//! - **Multiplication/Division**: With precision preservation
//!
//! ## Examples
//!
//! ```rust
//! use flexfloat::prelude::*;
//!
//! let subtotal = FlexFloat::from(19.99);
//! let tax = FlexFloat::from(1.60);
//!
//! let total = subtotal.clone() + tax.clone();
//! assert_ff_almost_eq!(total, FlexFloat::from(21.59));
//!
//! let pretax = total.clone() - tax.clone();
//! assert_ff_almost_eq!(pretax, FlexFloat::from(19.99));
//!
//! let doubled = subtotal.clone() * FlexFloat::from(2.0);
//! assert_ff_almost_eq!(doubled, FlexFloat::from(39.98));
//!
//! let split = total / FlexFloat::from(4.0);
//! assert_ff_almost_eq!(split, FlexFloat::from(5.3975));
//! ```

use core::ops::{Add, Div, Mul, Rem, Sub};
use core::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};

use crate::bitarray::{BitArrayArith, BitArrayConversion};
use crate::flexfloat::FlexFloat;

pub(crate) mod add_sub;
pub(crate) mod mul_div;
pub(crate) mod neg_abs;
pub(crate) mod normalize;
pub(crate) mod rem;
pub(crate) mod sums;

/// Macro to implement all combinations of binary ops for FlexFloat between
/// references and owned values.
macro_rules! impl_flexfloat_binop_refs {
    ($trait:ident, $method:ident) => {
        impl<
            E1: BitArrayArith,
            F1: BitArrayArith,
            E2: BitArrayConversion + Clone,
            F2: BitArrayConversion + Clone,
        > $trait<&FlexFloat<E2, F2>> for FlexFloat<E1, F1>
        {
            type Output = Self;

            fn $method(self, rhs: &FlexFloat<E2, F2>) -> Self::Output {
                self.$method(rhs.clone())
            }
        }

        impl<
            E1: BitArrayArith,
            F1: BitArrayArith,
            E2: BitArrayConversion + Clone,
            F2: BitArrayConversion + Clone,
        > $trait<&FlexFloat<E2, F2>> for &FlexFloat<E1, F1>
        {
            type Output = FlexFloat<E1, F1>;

            fn $method(self, rhs: &FlexFloat<E2, F2>) -> Self::Output {
                self.clone().$method(rhs.clone())
            }
        }

        impl<E1: BitArrayArith, F1: BitArrayArith, E2: BitArrayConversion, F2: BitArrayConversion>
            $trait<FlexFloat<E2, F2>> for &FlexFloat<E1, F1>
        {
            type Output = FlexFloat<E1, F1>;

            fn $method(self, rhs: FlexFloat<E2, F2>) -> Self::Output {
                self.clone().$method(rhs)
            }
        }
    };
}

impl_flexfloat_binop_refs!(Add, add);
impl_flexfloat_binop_refs!(Sub, sub);
impl_flexfloat_binop_refs!(Mul, mul);
impl_flexfloat_binop_refs!(Div, div);
impl_flexfloat_binop_refs!(Rem, rem);

/// Macro to implement all combinations of binary assign ops for FlexFloat
/// between references and owned values.
macro_rules! impl_flexfloat_binop_assign_refs {
    ($trait:ident, $method:ident, $op:ident) => {
        impl<E1: BitArrayArith, F1: BitArrayArith, E2: BitArrayConversion, F2: BitArrayConversion>
            $trait<FlexFloat<E2, F2>> for FlexFloat<E1, F1>
        {
            fn $method(&mut self, rhs: FlexFloat<E2, F2>) {
                *self = self.clone().$op(rhs);
            }
        }

        impl<
            E1: BitArrayArith,
            F1: BitArrayArith,
            E2: BitArrayConversion + Clone,
            F2: BitArrayConversion + Clone,
        > $trait<&FlexFloat<E2, F2>> for FlexFloat<E1, F1>
        {
            fn $method(&mut self, rhs: &FlexFloat<E2, F2>) {
                *self = self.clone().$op(rhs.clone());
            }
        }
    };
}

impl_flexfloat_binop_assign_refs!(AddAssign, add_assign, add);
impl_flexfloat_binop_assign_refs!(SubAssign, sub_assign, sub);
impl_flexfloat_binop_assign_refs!(MulAssign, mul_assign, mul);
impl_flexfloat_binop_assign_refs!(DivAssign, div_assign, div);
impl_flexfloat_binop_assign_refs!(RemAssign, rem_assign, rem);
