//! Conversion methods for BitArray implementations.
//!
//! This module defines the `BitArrayConversion` trait, which provides
//! methods for converting BitArrays to various numeric and string representations.

use num_bigint::{BigInt, BigUint};

use crate::bitarray::BitArrayConstruction;

use super::BitArrayAccess;

/// Trait for converting BitArray instances to other data types.
///
/// Provides methods for converting BitArrays to bytes, bits, numeric types,
/// and string representations.
pub trait BitArrayConversion: BitArrayAccess {
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
        let bytes: [u8; 8] = self.to_bytes().try_into().ok()?;
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
    /// let bits = BoolBitArray::from_bits(&[true, false, true]);
    /// assert_eq!(bits.to_bits_string(), "101");
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
    fn convert_to<B: BitArrayConstruction + Clone>(&self) -> B
    where
        Self: Sized,
    {
        B::from_bytes(&self.to_bytes(), self.len())
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::bitarray::traits::tests::BitArrayTest;
    use crate::bitarray::traits::*;
    use crate::tests::*;

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
