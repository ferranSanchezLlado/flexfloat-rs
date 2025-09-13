use std::ops::{Index, IndexMut, Range};

use crate::bitarray::BitArray;

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
        let mut bits = vec![false; n_bits];

        for i in 0..n_bits {
            let byte_index = i / 8;
            let bit_index = i % 8;
            let bit = if byte_index < bytes.len() {
                (bytes[byte_index] >> bit_index) & 1
            } else {
                0
            };
            bits[i] = bit == 1;
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
        let bytes = (n_bits + 7) / 8;
        let mut byte_vec = vec![0u8; bytes];

        for (i, &bit) in self.bits.iter().enumerate() {
            if bit {
                let byte_index = i / 8;
                let bit_index = i % 8;
                byte_vec[byte_index] |= 1 << bit_index;
            }
        }

        byte_vec
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
    use rstest::rstest;

    use super::super::tests::*;
    use super::*;

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
            assert!(bit_array.bits.into_iter().all(|b| b == false));
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
        let float = 3.141592653589793;
        let bit_array = BoolBitArray::from_float(float);
        assert_eq!(bit_array.bits.len(), 64);
        assert_eq!(f64_to_bits(float), bit_array.bits);

        for _ in 0..n_experiments {
            let float: f64 = rng.random();
            let bit_array = BoolBitArray::from_float(float);
            assert_eq!(bit_array.bits.len(), 64);
            assert_eq!(f64_to_bits(float), bit_array.bits);
        }
    }

    fn test_to_float(mut rng: impl Rng, n_experiments: usize) {
        let float = 3.141592653589793;
        let bit_array = BoolBitArray::from_float(float);
        assert_eq!(bit_array.to_float().unwrap(), float);

        let bit_array = BoolBitArray::zeros(65);
        assert!(bit_array.to_float().is_none());

        let bit_array = BoolBitArray::ones(63);
        assert!(bit_array.to_float().is_none());

        for _ in 0..n_experiments {
            let float: f64 = rng.random();
            let bit_array = BoolBitArray::from_float(float);
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
}
