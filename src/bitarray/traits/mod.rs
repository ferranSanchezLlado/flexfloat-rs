//! Traits module for BitArray implementations.
//!
//! This module provides a set of composable traits that together define
//! the complete BitArray interface. Each trait focuses on a specific aspect:
//!
//! - [`BitArrayConstruction`] - Creating BitArrays from various sources
//! - [`BitArrayConversion`] - Converting BitArrays to other types
//! - [`BitArrayAccess`] - Reading and querying BitArray contents
//! - [`BitArrayManipulation`] - Modifying and transforming BitArrays
//! - [`BitArrayArith`] - Bignum arithmetic (optional, for arithmetic-capable backends)
//!
//! The main [`BitArray`] trait combines construction, conversion, access,
//! manipulation, and primitives without committing to arithmetic semantics.
//! [`BitArrayArith`] extends [`BitArray`] for backends that also implement
//! `Add + Sub + Mul + Div`.

use core::fmt::Debug;

mod access;
mod arith;
mod construction;
mod conversion;
mod manipulation;
pub(crate) mod rounding;

pub use access::{BitArrayAccess, BitArrayMutAccess};
pub use arith::BitArrayArith;
pub use construction::BitArrayConstruction;
pub use conversion::BitArrayConversion;
pub use manipulation::BitArrayManipulation;
pub(crate) use rounding::{BitArrayRounding, ShiftRoundingInfo, ShiftRoundingResult};

/// Core trait for bit array implementations with little-endian byte ordering.
///
/// This trait combines construction, conversion, access, manipulation, and
/// backend-primitive capabilities. It deliberately does **not** include bignum
/// arithmetic — use [`BitArrayArith`] for that.
///
/// ## Byte Order
///
/// All implementations use **little-endian (LE)** byte order for consistency
/// with common CPU architectures and IEEE 754 standards.
///
/// ## Derivable bounds
///
/// Implementors must also satisfy `Debug + Clone + Default + PartialEq + Eq`.
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
///
/// // Create from various sources
/// let from_bits = BoolBitArray::from_bits(&[true, false, true]);
/// let from_bytes = BoolBitArray::from_bytes(&[0xAB], 8);
/// let from_f64 = BoolBitArray::from_f64(3.14159);
///
/// // Access and manipulate bits
/// let bit_3 = from_bits[2]; // true
/// let range = from_bits.get_range(0..2).unwrap();
///
/// // Convert back to other types
/// let bytes = from_bits.to_bytes();
/// let bits = from_bits.to_bits();
/// ```
pub trait BitArray:
    BitArrayConstruction
    + BitArrayConversion
    + BitArrayMutAccess
    + BitArrayManipulation
    + crate::bitarray::backend::BitArrayPrimitives
    + Debug
    + Clone
    + Default
    + PartialEq
    + Eq
{
}

/// Blanket implementation of BitArray for any type that implements all required traits.
impl<T> BitArray for T where
    T: BitArrayConstruction
        + BitArrayConversion
        + BitArrayMutAccess
        + BitArrayManipulation
        + crate::bitarray::backend::BitArrayPrimitives
        + Debug
        + Clone
        + Default
        + PartialEq
        + Eq
{
}

#[cfg(test)]
pub(crate) mod tests {
    use core::ops::Range;

    use super::*;
    use crate::BoolBitArray;
    use crate::bitarray::backend::BitArrayPrimitives;

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct BitArrayTest {
        bits: Vec<bool>,
    }

    impl BitArrayConstruction for BitArrayTest {
        fn from_bytes(bytes: &[u8], n_bits: usize) -> Self
        where
            Self: Sized,
        {
            let normalized = BoolBitArray::from_bytes(bytes, n_bits);
            Self {
                bits: normalized.to_bits(),
            }
        }
    }

    impl BitArrayConversion for BitArrayTest {
        fn to_bytes(&self) -> Vec<u8> {
            let bool_array = BoolBitArray::from_bits(&self.bits);
            bool_array.to_bytes()
        }

        fn to_bits(&self) -> Vec<bool> {
            self.bits.clone()
        }
    }

    impl BitArrayAccess for BitArrayTest {
        fn iter_bits(&self) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator {
            self.bits.iter().cloned()
        }

        fn len(&self) -> usize {
            self.bits.len()
        }

        fn get(&self, index: usize) -> Option<bool> {
            self.bits.get(index).cloned()
        }

        fn get_range(&self, range: Range<usize>) -> Option<Self>
        where
            Self: Sized,
        {
            if range.end > self.len() || range.start > range.end {
                return None;
            }
            Some(Self {
                bits: self.bits[range].to_vec(),
            })
        }
    }

    impl BitArrayMutAccess for BitArrayTest {
        type BitMut<'a> = &'a mut bool;

        fn get_mut(&mut self, index: usize) -> Option<&mut bool> {
            self.bits.get_mut(index)
        }
    }

    impl BitArrayPrimitives for BitArrayTest {
        fn append_bool(&mut self, value: bool) {
            self.bits.push(value);
        }

        fn fill_range(&mut self, range: core::ops::Range<usize>, value: bool) {
            for i in range {
                if i < self.bits.len() {
                    self.bits[i] = value;
                }
            }
        }

        fn copy_within_bits(&mut self, src: core::ops::Range<usize>, dst_start: usize) {
            let src_bits: Vec<bool> = self.bits[src].to_vec();
            for (i, b) in src_bits.into_iter().enumerate() {
                let dst = dst_start + i;
                if dst < self.bits.len() {
                    self.bits[dst] = b;
                }
            }
        }

        fn extend_with(&mut self, count: usize, value: bool) {
            self.bits.extend(core::iter::repeat_n(value, count));
        }

        fn truncate_in_place(&mut self, n_bits: usize) {
            self.bits.truncate(n_bits);
        }
    }

    impl BitArrayManipulation for BitArrayTest {
        fn reset(self) -> Self {
            Self {
                bits: vec![false; self.bits.len()],
            }
        }
    }
}
