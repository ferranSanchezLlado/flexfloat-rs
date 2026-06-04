//! `Sum` and `Product` iterator trait implementations.

use core::iter::{Product, Sum};

use crate::bitarray::BitArrayArith;
use crate::flexfloat::FlexFloat;

impl<Exp: BitArrayArith, Frac: BitArrayArith> Sum<FlexFloat<Exp, Frac>> for FlexFloat<Exp, Frac> {
    fn sum<I: Iterator<Item = FlexFloat<Exp, Frac>>>(iter: I) -> Self {
        iter.fold(FlexFloat::zero(), |acc, x| acc + x)
    }
}

impl<'a, Exp: BitArrayArith, Frac: BitArrayArith> Sum<&'a FlexFloat<Exp, Frac>>
    for FlexFloat<Exp, Frac>
{
    fn sum<I: Iterator<Item = &'a FlexFloat<Exp, Frac>>>(iter: I) -> Self {
        iter.fold(FlexFloat::zero(), |acc, x| acc + x)
    }
}

impl<Exp: BitArrayArith, Frac: BitArrayArith> Product<FlexFloat<Exp, Frac>>
    for FlexFloat<Exp, Frac>
{
    fn product<I: Iterator<Item = FlexFloat<Exp, Frac>>>(iter: I) -> Self {
        iter.fold(FlexFloat::from_f64(1.0), |acc, x| acc * x)
    }
}

impl<'a, Exp: BitArrayArith, Frac: BitArrayArith> Product<&'a FlexFloat<Exp, Frac>>
    for FlexFloat<Exp, Frac>
{
    fn product<I: Iterator<Item = &'a FlexFloat<Exp, Frac>>>(iter: I) -> Self {
        iter.fold(FlexFloat::from_f64(1.0), |acc, x| acc * x)
    }
}
