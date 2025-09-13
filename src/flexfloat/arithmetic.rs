use std::ops::Neg;

use crate::bitarray::BitArray;
use crate::flexfloat::FlexFloat;

impl<B: BitArray> FlexFloat<B> {
    pub fn abs(&self) -> Self
    where
        B: Clone,
    {
        Self {
            sign: false,
            exponent: self.exponent.clone(),
            fraction: self.fraction.clone(),
        }
    }
}

impl<B: BitArray> Neg for FlexFloat<B> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            sign: !self.sign,
            exponent: self.exponent,
            fraction: self.fraction,
        }
    }
}
