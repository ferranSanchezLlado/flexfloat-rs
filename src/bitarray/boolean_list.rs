use std::ops::{Index, IndexMut, Range};

use crate::bitarray::BitArray;

#[derive(Debug, Clone)]
pub struct BoolBitArray {
    bits: Vec<bool>,
}

impl BitArray for BoolBitArray {
    fn from_bits(bits: &[bool]) -> Self
    where
        Self: Sized,
    {
        Self {
            bits: bits.to_vec(),
        }
    }

    // [7, 1], 9 -> [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    fn from_bytes(bytes: &[u8], n_bits: usize) -> Self
    where
        Self: Sized,
    {
        let mut bits = Vec::with_capacity(n_bits);

        for i in 0..n_bits {
            let byte_index = i / 8;
            let bit_index = i % 8;
            let bit = if byte_index < bytes.len() {
                (bytes[byte_index] >> bit_index) & 1
            } else {
                0
            };
            bits.push(bit == 1);
        }

        // Fill with zeros if n_bits is more than available bits in bytes
        while bits.len() < n_bits {
            bits.push(false);
        }
        Self { bits }
    }

    fn zeros(n_bits: usize) -> Self
    where
        Self: Sized,
    {
        Self {
            bits: vec![false; n_bits],
        }
    }

    fn ones(n_bits: usize) -> Self
    where
        Self: Sized,
    {
        Self {
            bits: vec![true; n_bits],
        }
    }

    fn to_bits(&self) -> Vec<bool> {
        self.bits.clone()
    }

    fn to_bytes(&self) -> Vec<u8> {
        let n_bits = self.bits.len();
        let n_bytes = n_bits.div_ceil(8);
        let mut bytes = vec![0u8; n_bytes];

        for (i, &bit) in self.bits.iter().enumerate() {
            if bit {
                let byte_index = i / 8;
                let bit_index = i % 8;
                bytes[byte_index] |= 1 << bit_index;
            }
        }

        bytes
    }

    fn iter_bits(&self) -> impl Iterator<Item = bool> {
        self.bits.iter().copied()
    }

    fn len(&self) -> usize {
        self.bits.len()
    }

    fn get(&self, index: usize) -> Option<&bool> {
        self.bits.get(index)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut bool> {
        self.bits.get_mut(index)
    }

    fn get_range(&self, range: Range<usize>) -> Option<Self>
    where
        Self: Sized,
    {
        if range.end > self.len() || range.start > range.end {
            return None;
        }
        Some(Self::from_bits(&self.bits[range]))
    }
}

impl Index<Range<usize>> for BoolBitArray {
    type Output = [bool];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        &self.bits[index]
    }
}

impl IndexMut<Range<usize>> for BoolBitArray {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        &mut self.bits[index]
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    #[cfg(feature = "bigint")]
    use num_bigint::{BigInt, BigUint};
    use rstest::rstest;

    use super::super::tests::*;
    use super::*;
    use crate::tests::*;

    fn test_from_bits(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = BoolBitArray::from_bits(&bits);
        assert_eq!(bit_array.bits, bits);

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let bits = random_bits(&mut rng, len);
            let bit_array = BoolBitArray::from_bits(&bits);
            assert_eq!(bit_array.bits, bits);
        }
    }

    fn test_from_bytes(mut rng: impl Rng, n_experiments: usize) {
        let bytes = vec![0b00001111, 0b10101010];
        let expected_bits = vec![
            true, true, true, true, false, false, false, false, false, true, false, true,
        ];
        let bit_array = BoolBitArray::from_bytes(&bytes, 12);
        assert_eq!(bit_array.bits, expected_bits);

        // n_experiments is divided by 50 to take into account the inner loop
        // that use len from 1 to 100 (so an average of 50 iterations)
        let n_experiments = n_experiments / 50;

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let bool_string = random_bits_string(&mut rng, len);
            let bytes = string_to_bytes(&bool_string);
            let expected_bits = string_to_bits(&bool_string);

            for len in 1..len {
                let bit_array = BoolBitArray::from_bytes(&bytes, len);
                assert_eq!(bit_array.bits, expected_bits[..len]);
            }
        }
    }

    #[rstest]
    fn test_zeros(mut rng: impl Rng, n_experiments: usize) {
        let bit_array = BoolBitArray::zeros(10);
        assert_eq!(bit_array.bits, vec![false; 10]);

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100_000);
            let bit_array = BoolBitArray::zeros(len);
            assert!(bit_array.bits.into_iter().all(|b| !b));
        }
    }

    #[rstest]
    fn test_ones(mut rng: impl Rng, n_experiments: usize) {
        let bit_array = BoolBitArray::ones(10);
        assert_eq!(bit_array.bits, vec![true; 10]);

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100_000);
            let bit_array = BoolBitArray::ones(len);
            assert!(bit_array.bits.into_iter().all(|b| b));
        }
    }

    fn f64_to_bits(value: f64) -> Vec<bool> {
        let bytes = value.to_le_bytes();
        let mut bits = Vec::with_capacity(64);
        for byte in &bytes {
            for i in 0..8 {
                bits.push((byte >> i) & 1 == 1);
            }
        }
        bits
    }

    fn test_from_float(mut rng: impl Rng, n_experiments: usize) {
        let float = f64::consts::PI;
        let bit_array = BoolBitArray::from_f64(float);
        assert_eq!(bit_array.bits.len(), 64);
        assert_eq!(f64_to_bits(float), bit_array.bits);

        for _ in 0..n_experiments {
            let float: f64 = rng.random();
            let bit_array = BoolBitArray::from_f64(float);
            assert_eq!(bit_array.bits.len(), 64);
            assert_eq!(f64_to_bits(float), bit_array.bits);
        }
    }

    fn test_to_float(mut rng: impl Rng, n_experiments: usize) {
        let float = f64::consts::E;
        let bit_array = BoolBitArray::from_f64(float);
        assert_eq!(bit_array.to_float().unwrap(), float);

        let bit_array = BoolBitArray::zeros(65);
        assert!(bit_array.to_float().is_none());

        let bit_array = BoolBitArray::ones(63);
        assert!(bit_array.to_float().is_none());

        for _ in 0..n_experiments {
            let float: f64 = rng.random();
            let bit_array = BoolBitArray::from_f64(float);
            assert_eq!(bit_array.to_float().unwrap(), float);
        }
    }

    fn test_to_bits(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = BoolBitArray::from_bits(&bits);
        assert_eq!(bit_array.to_bits(), bits);

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let bits = random_bits(&mut rng, len);
            let bit_array = BoolBitArray::from_bits(&bits);
            assert_eq!(bit_array.to_bits(), bits);
        }
    }

    fn test_to_bytes(mut rng: impl Rng, n_experiments: usize) {
        let bytes = vec![0b00001111, 0b00000010];
        let bit_array = BoolBitArray::from_bytes(&bytes, 12);
        assert_eq!(bit_array.to_bytes(), bytes);

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let bytes = random_bytes(&mut rng, len);
            let bit_array = BoolBitArray::from_bytes(&bytes, len * 8);
            assert_eq!(bit_array.to_bytes(), bytes);
        }
    }

    #[rstest]
    fn test_len(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = BoolBitArray::from_bits(&bits);
        assert_eq!(bit_array.len(), bits.len());

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100_000);
            let bits = vec![true; len];
            let bit_array = BoolBitArray::from_bits(&bits);
            assert_eq!(bit_array.len(), bits.len());
        }
    }

    #[rstest]
    fn test_get(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = BoolBitArray::from_bits(&bits);
        for (i, &bit) in bits.iter().enumerate() {
            assert_eq!(bit_array.get(i).unwrap(), &bit);
        }
        assert!(bit_array.get(bits.len()).is_none());

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let bits = random_bits(&mut rng, len);
            let bit_array = BoolBitArray::from_bits(&bits);

            let i = rng.random_range(0..len);
            assert_eq!(bit_array.get(i).unwrap(), &bits[i]);

            assert!(bit_array.get(bits.len()).is_none());
        }
    }

    #[rstest]
    fn test_get_mut(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let mut bit_array = BoolBitArray::from_bits(&bits);
        for (i, &bit) in bits.iter().enumerate() {
            assert_eq!(*bit_array.get_mut(i).unwrap(), bit);
        }
        assert!(bit_array.get_mut(bits.len()).is_none());

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let bits = random_bits(&mut rng, len);
            let mut bit_array = BoolBitArray::from_bits(&bits);
            for (i, &bit) in bits.iter().enumerate() {
                assert_eq!(*bit_array.get_mut(i).unwrap(), bit);
            }
            assert!(bit_array.get_mut(bits.len()).is_none());
        }
    }

    #[rstest]
    fn test_bitarray_bits(mut rng: impl Rng, n_experiments: usize) {
        test_from_bits(&mut rng, n_experiments);
        test_to_bits(&mut rng, n_experiments);
    }

    #[rstest]
    fn test_bitarray_bytes(mut rng: impl Rng, n_experiments: usize) {
        test_from_bytes(&mut rng, n_experiments);
        test_to_bytes(&mut rng, n_experiments);
    }

    #[rstest]
    fn test_bitarray_float(mut rng: impl Rng, n_experiments: usize) {
        test_from_float(&mut rng, n_experiments);
        test_to_float(&mut rng, n_experiments);
    }

    #[cfg(feature = "bigint")]
    fn test_from_biguint(mut rng: impl Rng, n_experiments: usize) {
        // Test with a known value
        let biguint = BigUint::from(0b11110000u8);
        let bit_array = BoolBitArray::from_biguint(&biguint);
        let expected_bits = vec![false, false, false, false, true, true, true, true];
        assert_eq!(bit_array.bits, expected_bits);

        // Test with zero
        let biguint = BigUint::from(0u8);
        let bit_array = BoolBitArray::from_biguint(&biguint);
        assert_eq!(bit_array.bits, vec![false; 8]);

        // Test with larger known values
        let biguint = BigUint::from(0x1234u16);
        let bit_array = BoolBitArray::from_biguint(&biguint);
        // 0x1234 = 0001001000110100 in binary (MSB first)
        // But stored as LSB first: [0,0,1,0,1,1,0,0,0,1,0,0,1,0,0,0]
        let expected_bits = vec![
            false, false, true, false, true, true, false, false, // 0x34 = 52
            false, true, false, false, true, false, false, false, // 0x12 = 18
        ];
        assert_eq!(bit_array.bits, expected_bits);

        for _ in 0..n_experiments {
            let n_bits = rng.random_range(1..100);
            let biguint = random_biguint(&mut rng, n_bits);
            let bit_array = BoolBitArray::from_biguint(&biguint);

            // Verify by checking the bytes directly
            let expected_bytes = biguint.to_bytes_le();
            let actual_bytes = bit_array.to_bytes();

            assert_eq!(actual_bytes, expected_bytes);
        }
    }

    #[cfg(feature = "bigint")]
    fn test_to_biguint(mut rng: impl Rng, n_experiments: usize) {
        let big_uint = BigUint::from(0b11110000u8);
        let bit_array = BoolBitArray::from_biguint(&big_uint);
        assert_eq!(bit_array.to_biguint(), big_uint);

        let big_uint = BigUint::from(0u8);
        let bit_array = BoolBitArray::from_biguint(&big_uint);
        assert_eq!(bit_array.to_biguint(), big_uint);

        let big_uint = BigUint::from(0x1234u16);
        let bit_array = BoolBitArray::from_biguint(&big_uint);
        assert_eq!(bit_array.to_biguint(), big_uint);

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let bytes = random_bytes(&mut rng, len);
            let big_uint = BigUint::from_bytes_le(&bytes);
            let bit_array = BoolBitArray::from_biguint(&big_uint);
            assert_eq!(bit_array.to_biguint(), big_uint);
        }
    }

    #[cfg(feature = "bigint")]
    fn test_from_bigint(mut rng: impl Rng, n_experiments: usize) {
        // Test with positive value
        let bigint = BigInt::from(7i8);
        let bit_array = BoolBitArray::from_bigint(&bigint, 4).unwrap();
        // 7 + 8 = 15 = 0b1111
        let expected_bits = vec![true, true, true, true]; // LSB first
        assert_eq!(bit_array.bits, expected_bits);

        // Test with negative value
        let bigint = BigInt::from(-1i8);
        let bit_array = BoolBitArray::from_bigint(&bigint, 4).unwrap();
        // -1 + 8 = 7 = 0b0111
        let expected_bits = vec![true, true, true, false];
        assert_eq!(bit_array.bits, expected_bits);

        // Test with zero
        let bigint = BigInt::from(0i8);
        let bit_array = BoolBitArray::from_bigint(&bigint, 4).unwrap();
        // 0 + 8 = 8 = 0b1000
        let expected_bits = vec![false, false, false, true];
        assert_eq!(bit_array.bits, expected_bits);

        // Test with minimum value
        let bigint = BigInt::from(-8i8); // Minimum for 4-bit signed
        let bit_array = BoolBitArray::from_bigint(&bigint, 4).unwrap();
        // -8 + 8 = 0 = 0b0000
        let expected_bits = vec![false, false, false, false];
        assert_eq!(bit_array.bits, expected_bits);

        // Test overflow cases
        let bigint = BigInt::from(8i8); // Too large for 4 bits (max is 7)
        assert!(BoolBitArray::from_bigint(&bigint, 4).is_none());

        let bigint = BigInt::from(-9i8); // Too small for 4 bits (min is -8)
        assert!(BoolBitArray::from_bigint(&bigint, 4).is_none());

        // Test with larger bit widths
        let bigint = BigInt::from(100i16);
        let bit_array = BoolBitArray::from_bigint(&bigint, 8).unwrap();
        // 100 + 128 = 228 = 0b11100100
        let expected_bits = vec![false, false, true, false, false, true, true, true];
        assert_eq!(bit_array.bits, expected_bits);

        for _ in 0..n_experiments {
            let n_bits = rng.random_range(2..32); // At least 2 bits for sign
            let bigint = random_bigint(&mut rng, n_bits);

            let bit_array = BoolBitArray::from_bigint(&bigint, n_bits)
                .expect("Should fit in the given bit width");

            // Check sign
            let sign_bit = !bit_array.bits[n_bits - 1];
            let expected_sign = bigint.sign() == num_bigint::Sign::Minus;
            assert_eq!(sign_bit, expected_sign);

            let biguint = (bigint.clone() + (BigInt::from(1u8) << (n_bits - 1)))
                .to_biguint()
                .unwrap();
            let mut expected_bytes = biguint.to_bytes_le();
            let actual_bytes = bit_array.to_bytes();

            // Biguint could miss leading zeros, specified on the n_bits
            if expected_bytes.len() != actual_bytes.len() {
                expected_bytes.resize(actual_bytes.len(), 0);
            }
            assert_eq!(actual_bytes, expected_bytes);
        }
    }

    #[cfg(feature = "bigint")]
    fn test_to_bigint(mut rng: impl Rng, n_experiments: usize) {
        let bigint = BigInt::from(7i8);
        let bit_array = BoolBitArray::from_bigint(&bigint, 4).unwrap();
        assert_eq!(bit_array.to_bigint(), bigint);

        let bigint = BigInt::from(-1i8);
        let bit_array = BoolBitArray::from_bigint(&bigint, 4).unwrap();
        assert_eq!(bit_array.to_bigint(), bigint);

        let bigint = BigInt::from(0i8);
        let bit_array = BoolBitArray::from_bigint(&bigint, 4).unwrap();
        assert_eq!(bit_array.to_bigint(), bigint);

        for _ in 0..n_experiments {
            let n_bits = rng.random_range(2..100); // At least 2 bits for sign
            let bigint = random_bigint(&mut rng, n_bits);
            let bit_array = BoolBitArray::from_bigint(&bigint, n_bits).unwrap();
            assert_eq!(bit_array.to_bigint(), bigint);
        }
    }

    #[cfg(feature = "bigint")]
    #[rstest]
    fn test_bitarray_biguint(mut rng: impl Rng, n_experiments: usize) {
        test_from_biguint(&mut rng, n_experiments);
        test_to_biguint(&mut rng, n_experiments);
    }

    #[cfg(feature = "bigint")]
    #[rstest]
    fn test_bitarray_bigint(mut rng: impl Rng, n_experiments: usize) {
        test_from_bigint(&mut rng, n_experiments);
        test_to_bigint(&mut rng, n_experiments);
    }
}
