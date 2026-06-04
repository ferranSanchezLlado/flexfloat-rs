//! Conversion methods for BitArray implementations.
//!
//! This module defines the `BitArrayConversion` trait, which provides
//! methods for converting BitArrays to various numeric and string representations.

use core::{any::TypeId, mem::transmute_copy};
use std::mem::ManuallyDrop;

use num_bigint::{BigInt, BigUint};

use crate::bitarray::BitArrayConstruction;

use super::BitArrayAccess;

/// Trait for converting BitArray instances to other data types.
///
/// Provides methods for converting BitArrays to bytes, bits, numeric types,
/// and string representations.
pub trait BitArrayConversion: BitArrayAccess + Clone + 'static {
    /// Converts the BitArray to a byte vector in little-endian format.
    ///
    /// # Returns
    ///
    /// Byte vector representing the bit sequence
    fn to_bytes(&self) -> Vec<u8>;

    /// Converts the BitArray to a boolean vector.
    ///
    /// # Returns
    ///
    /// Vector of booleans representing each bit
    fn to_bits(&self) -> Vec<bool> {
        self.iter_bits().collect()
    }

    /// Converts the BitArray to an f64 if it represents exactly 64 bits.
    ///
    /// # Returns
    ///
    /// Some(f64) if the array has exactly 64 bits, None otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let bits = BoolBitArray::from_f64(3.14);
    /// let recovered = bits.to_float().unwrap();
    /// assert_eq!(recovered, 3.14);
    /// ```
    fn to_float(&self) -> Option<f64> {
        if self.len() != 64 {
            return None;
        }
        // UNSAFE: We just checked that to_bytes() returns exactly 8 bytes for 64 bits.
        let bytes: [u8; 8] = unsafe { self.to_bytes().try_into().unwrap_unchecked() };
        Some(f64::from_bits(u64::from_le_bytes(bytes)))
    }

    /// Converts the BitArray to a binary string representation.
    ///
    /// # Returns
    ///
    /// String of '0' and '1' characters representing the bits
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let bits = BoolBitArray::from_bits(&[true, false, true, true, false]);
    /// assert_eq!(bits.to_bits_string(), "01101");
    /// ```
    fn to_bits_string(&self) -> String {
        self.iter_bits()
            .rev()
            .map(|b| if b { '1' } else { '0' })
            .collect()
    }

    /// Converts the BitArray to a BigUint.
    ///
    /// Interprets the bit sequence as an unsigned binary number.
    ///
    /// # Returns
    ///
    /// BigUint representation of the bit sequence
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    /// use num_bigint::BigUint;
    ///
    /// let bits = BoolBitArray::from_bits(&[true, false, true]); // 0b101 = 5
    /// assert_eq!(bits.to_biguint(), BigUint::from(5u32));
    /// ```
    fn to_biguint(&self) -> BigUint {
        BigUint::from_bytes_le(&self.to_bytes())
    }

    /// Converts the BitArray to a signed BigInt using offset bias (half the range).
    ///
    /// Interprets the bit sequence as an unsigned value, then subtracts 2^(n_bits-1)
    /// to convert to a signed representation.
    ///
    /// # Returns
    ///
    /// BigInt representation in the range [-2^(n_bits-1), 2^(n_bits-1) - 1]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    /// use num_bigint::BigInt;
    ///
    /// let bits = BoolBitArray::from_bits(&[true, false, true]); // 0b101 = 5 in unsigned
    /// assert_eq!(bits.to_bigint(), BigInt::from(1)); // 5 - 2^(3-1) = 5 - 4 = 1
    /// ```
    fn to_bigint(&self) -> BigInt {
        let n_bits = self.len();
        let half = BigInt::from(1u8) << (n_bits - 1);
        let unsigned = BigInt::from(self.to_biguint());
        unsigned - half
    }

    /// Converts this BitArray to another BitArray implementation type.
    ///
    /// # Returns
    ///
    /// New BitArray of the target type with the same bit contents
    fn convert_to<B: BitArrayConstruction + 'static>(self) -> B
    where
        Self: Sized,
    {
        if TypeId::of::<Self>() == TypeId::of::<B>() {
            // SAFETY: We just checked that Self and B are the same type, so this transmute is safe.
            unsafe { transmute_copy(&ManuallyDrop::new(self)) }
        } else {
            B::from_bytes(&self.to_bytes(), self.len())
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::bitarray::traits::tests::BitArrayTest;
    use crate::bitarray::traits::*;
    use crate::test_support::*;

    #[rstest]
    fn test_default_to_bytes(mut rng: impl Rng, n_experiments: usize) {
        let bytes = vec![0b00001111, 0b00000010];
        let bit_array = BitArrayTest::from_bytes(&bytes, 12);
        assert_eq!(bit_array.to_bytes(), bytes);

        for _ in 0..n_experiments {
            let n = rng.random_range(1..=128);
            let bits: Vec<bool> = (0..n).map(|_| rng.random()).collect();
            let bit_array = BitArrayTest::from_bits(&bits);
            let bytes = bit_array.to_bytes();
            let bit_array2 = BitArrayTest::from_bytes(&bytes, n);
            assert_eq!(bit_array2.to_bits(), bits);
        }
    }

    #[rstest]
    fn test_default_to_bits_string() {
        let bits = vec![true, false, true, true, false];
        let bit_array = BitArrayTest::from_bits(&bits);
        let s = bit_array.to_bits_string();
        assert_eq!(s, "01101");
    }
}
