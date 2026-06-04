//! Construction methods for BitArray implementations.
//!
//! This module defines the `BitArrayConstruction` trait, which provides
//! factory methods for creating new BitArray instances from various sources.

use num_bigint::{BigInt, BigUint};

/// Trait for constructing BitArray instances from various data sources.
///
/// Provides factory methods for creating BitArrays from bytes, bits, numeric types,
/// and special values like all-zeros or all-ones patterns.
pub trait BitArrayConstruction: Sized {
    /// Creates a new BitArray from a byte array with specified bit length.
    ///
    /// Interprets bytes in little-endian order, where the least significant bit
    /// of the first byte becomes bit 0 of the array.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Source byte array in little-endian format
    /// * `n_bits` - Total number of bits to extract (may be less than `bytes.len() * 8`)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// // 0x7A = 0b01111010 in binary -> [0,1,0,1,1,1,1,0] in LE bit order
    /// let bits = BoolBitArray::from_bytes(&[0x7A], 8);
    /// assert_eq!(bits.len(), 8);
    /// assert_eq!(bits[0], false);  // LSB
    /// assert_eq!(bits[1], true);
    /// assert_eq!(bits[6], true);   // MSB (7th bit)
    /// ```
    fn from_bytes(bytes: &[u8], n_bits: usize) -> Self;

    /// Creates a new BitArray from a boolean slice.
    ///
    /// This is the most direct way to create a BitArray, with each boolean
    /// representing one bit in the sequence.
    ///
    /// # Arguments
    ///
    /// * `bits` - Slice of boolean values representing individual bits
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let bits = BoolBitArray::from_bits(&[true, false, false]);
    /// assert_eq!(bits.len(), 3);
    /// assert_eq!(bits[0], true);
    /// assert_eq!(bits[1], false);
    /// assert_eq!(bits[2], false);
    /// ```
    fn from_bits(bits: &[bool]) -> Self {
        let n_bits = bits.len();
        let n_bytes = n_bits.div_ceil(8);
        let mut bytes = vec![0u8; n_bytes];

        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                let byte_index = i / 8;
                let bit_index = i % 8;
                bytes[byte_index] |= 1 << bit_index;
            }
        }

        Self::from_bytes(&bytes, n_bits)
    }

    /// Creates a BitArray from an IEEE 754 double-precision float.
    ///
    /// Converts the 64-bit representation directly to a BitArray, preserving
    /// the exact bit pattern including sign, exponent, and mantissa fields.
    ///
    /// # Arguments
    ///
    /// * `value` - The f64 value to convert
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let bits = BoolBitArray::from_f64(1.0);
    /// assert_eq!(bits.len(), 64);
    /// assert_eq!(bits[63], false);  // Sign bit (positive)
    /// // Exponent for 1.0: biased exponent 1023 = 0b01111111111, stored at bits[52..63]
    /// assert_eq!(bits[52..63], [true, true, true, true, true, true, true, true, true, true, false]);
    /// // Mantissa for 1.0 is all zeros (bits[0..52])
    /// assert_eq!(bits[0..52], [false; 52]);
    /// ```
    fn from_f64(value: f64) -> Self {
        let bits = value.to_bits();
        Self::from_bytes(&bits.to_le_bytes(), 64)
    }

    /// Creates a BitArray from a BigUint.
    ///
    /// Converts the arbitrary-precision unsigned integer to its binary
    /// representation, automatically sizing the BitArray to fit.
    ///
    /// # Arguments
    ///
    /// * `value` - The BigUint to convert
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    /// use num_bigint::BigUint;
    ///
    /// let big_num = BigUint::from(255u32);
    /// let bits = BoolBitArray::from_biguint(&big_num);
    /// assert_eq!(bits.len(), 8);
    /// assert_eq!(bits[0..8], [true; 8]); // 255 is 0b11111111
    /// ```
    fn from_biguint(value: &BigUint) -> Self {
        let bytes = value.to_bytes_le();
        let n_bits = value.bits() as usize;
        Self::from_bytes(&bytes, n_bits)
    }

    /// Creates a BitArray from a BigUint with a fixed number of bits.
    ///
    /// Similar to [`from_biguint`](BitArrayConstruction::from_biguint) but pads or truncates
    /// to exactly the specified bit width.
    ///
    /// # Arguments
    ///
    /// * `value` - The BigUint to convert
    /// * `n_bits` - The exact number of bits in the resulting BitArray
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    /// use num_bigint::BigUint;
    ///
    /// let big_num = BigUint::from(15u32);  // 0b1111
    /// let bits = BoolBitArray::from_biguint_fixed(&big_num, 8);
    /// assert_eq!(bits.len(), 8);  // Padded to 8 bits
    /// assert_eq!(bits[0..4], [true; 4]); // Original bits
    /// assert_eq!(bits[4..8], [false; 4]); // Padding bits
    /// ```
    fn from_biguint_fixed(value: &BigUint, n_bits: usize) -> Self {
        let bytes = value.to_bytes_le();
        Self::from_bytes(&bytes, n_bits)
    }

    /// Creates a BitArray from a signed BigInt with specified bit width.
    ///
    /// Uses offset bias (half the range) for conversion, where the value is shifted
    /// by 2^(n_bits-1) to map signed integers to unsigned bit representation.
    /// Returns None if the value cannot fit in the specified bit width.
    ///
    /// # Arguments
    ///
    /// * `value` - The BigInt to convert
    /// * `n_bits` - Target bit width (must accommodate the value)
    ///
    /// # Returns
    ///
    /// `Some(BitArray)` if the value fits in the range [-2^(n_bits-1), 2^(n_bits-1) - 1], `None` otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    /// use num_bigint::BigInt;
    ///
    /// let num = BigInt::from(-5);
    /// let bits = BoolBitArray::from_bigint(&num, 8).unwrap();
    /// assert_eq!(bits.len(), 8);
    /// // Offset-bias encoding: stored value = -5 + 2^(8-1) = -5 + 128 = 123 = 0b01111011
    /// assert_eq!(bits[0..8], [true, true, false, true, true, true, true, false]);
    /// ```
    fn from_bigint(value: &BigInt, n_bits: usize) -> Option<Self> {
        let half = BigInt::from(1u8) << (n_bits - 1);
        let max_value = &half - 1u8;
        let min_value = -&half;

        if value < &min_value || value > &max_value {
            return None;
        }

        let unsigned_value = (value + half).to_biguint()?;
        let bytes = unsigned_value.to_bytes_le();
        Some(Self::from_bytes(&bytes, n_bits))
    }

    /// Creates a BitArray of all zeros with specified length.
    ///
    /// # Arguments
    ///
    /// * `n_bits` - Number of bits in the array
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let zeros = BoolBitArray::zeros(8);
    /// assert_eq!(zeros.len(), 8);
    /// assert!(zeros.iter_bits().all(|b| !b));
    /// ```
    fn zeros(n_bits: usize) -> Self {
        let n_bytes = n_bits.div_ceil(8);
        let bytes = vec![0u8; n_bytes];
        Self::from_bytes(&bytes, n_bits)
    }

    /// Creates a BitArray of all ones with specified length.
    ///
    /// # Arguments
    ///
    /// * `n_bits` - Number of bits in the array
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let ones = BoolBitArray::ones(8);
    /// assert_eq!(ones.len(), 8);
    /// assert!(ones.iter_bits().all(|b| b));
    /// ```
    fn ones(n_bits: usize) -> Self {
        let n_bytes = n_bits.div_ceil(8);
        let last_num_bits = n_bits % 8;
        let mut bytes = vec![0xffu8; n_bytes];

        if last_num_bits > 0 {
            bytes[n_bytes - 1] = (1 << last_num_bits) - 1;
        }

        Self::from_bytes(&bytes, n_bits)
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
    fn test_default_from_bits(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = BitArrayTest::from_bits(&bits);
        assert_eq!(bit_array.to_bits(), bits);

        for _ in 0..n_experiments {
            let n = rng.random_range(1..=128);
            let bits: Vec<bool> = (0..n).map(|_| rng.random()).collect();
            let bit_array = BitArrayTest::from_bits(&bits);
            assert_eq!(bit_array.to_bits(), bits);
        }
    }

    #[rstest]
    fn test_default_from_bytes(mut rng: impl Rng, n_experiments: usize) {
        let bytes = vec![0b10111101, 0b00110010];
        let expected_bits = vec![
            true, false, true, true, true, true, false, true, false, true,
        ];
        let bit_array = BitArrayTest::from_bytes(&bytes, 10);
        assert_eq!(bit_array.to_bits(), expected_bits);

        for _ in 0..n_experiments {
            let n: usize = rng.random_range(1..=128);
            let bytes: Vec<u8> = (0..n.div_ceil(8)).map(|_| rng.random()).collect();
            let bit_array = BitArrayTest::from_bytes(&bytes, n);
            let bits = bit_array.to_bits();
            assert_eq!(bits.len(), n);

            let expected_bits: Vec<bool> = (0..n)
                .map(|i| {
                    let byte = bytes[i / 8];
                    ((byte >> (i % 8)) & 1) == 1
                })
                .collect();
            assert_eq!(bits, expected_bits);
        }
    }


    #[rstest]
    fn test_default_zeros(mut rng: impl Rng, n_experiments: usize) {
        let bit_array = BitArrayTest::zeros(10);
        assert_eq!(bit_array.to_bits(), vec![false; 10]);

        for _ in 0..n_experiments {
            let n = rng.random_range(1..=128);
            let bit_array = BitArrayTest::zeros(n);
            let bits = bit_array.to_bits();
            assert_eq!(bits.len(), n);
            assert!(bits.iter().all(|&b| !b));
        }
    }

    #[rstest]
    fn test_default_ones(mut rng: impl Rng, n_experiments: usize) {
        let bit_array = BitArrayTest::ones(10);
        assert_eq!(bit_array.to_bits(), vec![true; 10]);

        for _ in 0..n_experiments {
            let n = rng.random_range(1..=128);
            let bit_array = BitArrayTest::ones(n);
            let bits = bit_array.to_bits();
            assert_eq!(bits.len(), n);
            assert!(bits.iter().all(|&b| b));
        }
    }
}
