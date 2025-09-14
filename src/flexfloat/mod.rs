use std::fmt::Debug;

use crate::bitarray::{BitArray, DefaultBitArray};

pub mod arithmetic;
pub mod converter;

#[derive(Clone)]
pub struct FlexFloat<B> {
    sign: bool,
    exponent: B,
    fraction: B,
}

impl<B: BitArray> FlexFloat<B> {
    pub fn new(sign: bool, exponent: B, fraction: B) -> Self {
        Self {
            sign,
            exponent,
            fraction,
        }
    }

    pub fn zero_with_sign(sign: bool) -> Self {
        Self {
            sign: sign,
            exponent: B::zeros(11),
            fraction: B::zeros(52),
        }
    }

    pub fn zero(sign: bool) -> Self {
        Self::zero_with_sign(sign)
    }

    pub fn nan() -> Self {
        Self {
            sign: false, // Sign doesn't matter for NaN
            exponent: B::ones(11),
            fraction: B::ones(52),
        }
    }

    pub fn infinity(sign: bool) -> Self {
        Self {
            sign,
            exponent: B::ones(11),
            fraction: B::zeros(52),
        }
    }

    fn is_special_exponent(&self) -> bool {
        self.exponent.iter_bits().all(|&b| b)
    }

    pub fn is_nan(&self) -> bool {
        self.is_special_exponent() && self.fraction.iter_bits().any(|&b| b)
    }

    pub fn is_infinity(&self) -> bool {
        self.is_special_exponent() && self.fraction.iter_bits().all(|&b| b == false)
    }

    pub fn is_zero(&self) -> bool {
        self.exponent.iter_bits().all(|&b| b == false)
            && self.fraction.iter_bits().all(|&b| b == false)
    }

    pub fn sign(&self) -> bool {
        self.sign
    }

    pub fn exponent(&self) -> &B {
        &self.exponent
    }

    pub fn fraction(&self) -> &B {
        &self.fraction
    }
}

impl Debug for FlexFloat<DefaultBitArray> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(feature = "bigint")]
        {
            f.debug_struct("FlexFloat")
                .field("sign", if self.sign { &'-' } else { &'+' })
                .field("exponent", &(self.exponent.to_bigint() + 1_u8))
                .field("fraction", &self.fraction.to_biguint())
                .finish()
        }
        #[cfg(not(feature = "bigint"))]
        {
            f.debug_struct("FlexFloat")
                .field("sign", &self.sign)
                .field("exponent", &self.exponent.to_bits_string())
                .field("fraction", &self.fraction.to_bits_string())
                .finish()
        }
    }
}
