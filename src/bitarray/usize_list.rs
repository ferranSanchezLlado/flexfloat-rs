use core::cmp::max;
use core::ops::{Add, Div, Mul, Sub};
use core::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use num_traits::Euclid;

use crate::bitarray::bit_ref::BitRef;
use crate::prelude::{
    BitArrayAccess, BitArrayConstruction, BitArrayConversion, BitArrayManipulation,
    BitArrayMutAccess,
};

#[derive(Debug, Clone)]
pub struct UsizeBitArray {
    words: Vec<usize>,
    length: usize,
}

struct UsizeBitIterator<'a> {
    data: &'a [usize],
    length: usize,
    index: usize,
}

impl Iterator for UsizeBitIterator<'_> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            None
        } else {
            let (word_i, bit_i) = self.index.div_rem_euclid(&(usize::BITS as usize));

            self.index += 1;
            Some((self.data[word_i] & (1_usize << bit_i)) > 0)
        }
    }
}

impl DoubleEndedIterator for UsizeBitIterator<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            None
        } else {
            let (word_i, bit_i) = (self.length - 1).div_rem_euclid(&(usize::BITS as usize));

            self.length -= 1;
            Some((self.data[word_i] & (1_usize << bit_i)) > 0)
        }
    }
}

impl ExactSizeIterator for UsizeBitIterator<'_> {
    fn len(&self) -> usize {
        self.length - self.index
    }
}

impl BitArrayConstruction for UsizeBitArray {
    fn from_bytes(bytes: &[u8], n_bits: usize) -> Self {
        let bits_per_word = usize::BITS as usize;
        let n_words = n_bits.div_ceil(bits_per_word);
        let mut words = vec![0usize; n_words];

        for bit_index in 0..n_bits {
            let byte_index = bit_index / 8;
            let bit_in_byte = bit_index % 8;
            let bit_is_set =
                byte_index < bytes.len() && ((bytes[byte_index] >> bit_in_byte) & 1) == 1;

            if bit_is_set {
                let word_index = bit_index / bits_per_word;
                let bit_in_word = bit_index % bits_per_word;
                words[word_index] |= 1usize << bit_in_word;
            }
        }

        Self {
            words,
            length: n_bits,
        }
    }
}

impl BitArrayAccess for UsizeBitArray {
    fn iter_bits(&self) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator {
        UsizeBitIterator {
            data: &self.words,
            length: self.length,
            index: 0,
        }
    }

    fn get(&self, index: usize) -> Option<bool> {
        if index >= self.length {
            None
        } else {
            let (word_i, bit_i) = index.div_rem_euclid(&(usize::BITS as usize));
            Some((self.words[word_i] & (1_usize << bit_i)) > 0)
        }
    }
}

impl BitArrayMutAccess for UsizeBitArray {
    type BitMut<'a> = BitRef<'a>;

    fn get_mut(&mut self, index: usize) -> Option<BitRef<'_>> {
        if index >= self.length {
            None
        } else {
            let (word_i, bit_i) = index.div_rem_euclid(&(usize::BITS as usize));
            let bit_mask = 1_usize << bit_i;
            Some(BitRef::new(&mut self.words[word_i], bit_mask))
        }
    }

    fn get_range(&self, range: std::ops::Range<usize>) -> Option<Self>
    where
        Self: Sized,
    {
        if range.start > range.end || range.end > self.length {
            return None;
        }
        let mut bits = Vec::with_capacity(range.end - range.start);
        for i in range.clone() {
            bits.push(self.get(i).unwrap_or(false));
        }
        Some(UsizeBitArray::from_bits(&bits))
    }
}

impl BitArrayManipulation for UsizeBitArray {
    fn append_bool(&mut self, value: bool) {
        let bits_per_word = usize::BITS as usize;
        let bit_index = self.length;
        let word_index = bit_index / bits_per_word;
        let bit_in_word = bit_index % bits_per_word;
        if word_index >= self.words.len() {
            self.words.push(0);
        }
        if value {
            self.words[word_index] |= 1 << bit_in_word;
        }
        self.length += 1;
    }
}

impl BitArrayConversion for UsizeBitArray {
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; self.length.div_ceil(8)];
        for i in 0..self.length {
            if let Some(true) = self.get(i) {
                let byte_index = i / 8;
                let bit_in_byte = i % 8;
                bytes[byte_index] |= 1 << bit_in_byte;
            }
        }
        bytes
    }
}

impl Add for UsizeBitArray {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let a = self.to_biguint();
        let b = rhs.to_biguint();
        let sum = a + b;
        UsizeBitArray::from_biguint(&sum)
    }
}

impl AddAssign for UsizeBitArray {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl Sub for UsizeBitArray {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        let max_words = max(self.words.len(), rhs.words.len());

        let mut borrow = 0usize;
        for i in 0..max_words {
            let a_word = self.words.get(i).copied().unwrap_or(0);
            let b_word = rhs.words.get(i).copied().unwrap_or(0);

            let (sub1, borrow1) = a_word.overflowing_sub(b_word);
            let (diff, borrow2) = sub1.overflowing_sub(borrow);
            borrow = (borrow1 as usize) + (borrow2 as usize);

            if i < self.words.len() {
                self.words[i] = diff;
            } else {
                self.words.push(diff);
            }
        }

        if borrow > 0 {
            panic!("Underflow in UsizeBitArray subtraction");
        }

        self.length = max(self.length, rhs.length);
        self
    }
}

impl SubAssign for UsizeBitArray {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl Mul for UsizeBitArray {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let a = self.to_biguint();
        let b = rhs.to_biguint();
        let prod = a * b;
        UsizeBitArray::from_biguint(&prod)
    }
}

impl MulAssign for UsizeBitArray {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl Div for UsizeBitArray {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        use num_bigint::BigUint;
        use num_traits::Zero;

        let a = self.to_biguint();
        let b = rhs.to_biguint();
        assert!(!b.is_zero(), "division by zero in UsizeBitArray");
        let quot: BigUint = a / b;
        UsizeBitArray::from_biguint(&quot)
    }
}

impl DivAssign for UsizeBitArray {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;
    use core::f64;
    use core::iter::repeat_n;

    use num_bigint::{BigInt, BigUint};
    use num_traits::identities::Zero;
    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::tests::*;

    fn test_from_bits(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = UsizeBitArray::from_bits(&bits);
        assert_eq!(bit_array.to_bits(), bits);

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let bits = random_bits(&mut rng, len);
            let bit_array = UsizeBitArray::from_bits(&bits);
            assert_eq!(bit_array.to_bits(), bits);
        }
    }

    fn test_from_bytes(mut rng: impl Rng, n_experiments: usize) {
        let bytes = vec![0b10111101, 0b00110010];
        let expected_bits = vec![
            true, false, true, true, true, true, false, true, false, true,
        ];
        let bit_array = UsizeBitArray::from_bytes(&bytes, 10);
        assert_eq!(bit_array.to_bits(), expected_bits);

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let n_bits = rng.random_range(1..100);
            let bool_string = random_bits_string(&mut rng, len);

            let bytes = string_to_bytes(&bool_string);
            let mut expected_bits = string_to_bits(&bool_string);

            let bit_array = UsizeBitArray::from_bytes(&bytes, n_bits);

            let expected_bits = if n_bits <= len {
                Vec::from_iter(expected_bits.get(0..n_bits).unwrap().to_owned())
            } else {
                expected_bits.extend(repeat_n(false, n_bits - len));
                expected_bits
            };
            assert_eq!(expected_bits, bit_array.to_bits(), "{len} {n_bits}");
        }
    }

    #[rstest]
    fn test_zeros(mut rng: impl Rng, n_experiments: usize) {
        let bit_array = UsizeBitArray::zeros(10);
        assert_eq!(bit_array.to_bits(), vec![false; 10]);

        for _ in 0..(n_experiments / 10) {
            let len = rng.random_range(1..100_000);
            let bit_array = UsizeBitArray::zeros(len);
            assert_eq!(bit_array.len(), len);
            assert!(bit_array.to_bits().into_iter().all(|bit| !bit));
        }
    }

    #[rstest]
    fn test_ones(mut rng: impl Rng, n_experiments: usize) {
        let bit_array = UsizeBitArray::ones(10);
        assert_eq!(bit_array.to_bits(), vec![true; 10]);

        for _ in 0..(n_experiments / 10) {
            let len = rng.random_range(1..10_000);
            let bit_array = UsizeBitArray::ones(len);
            assert_eq!(bit_array.len(), len);
            assert!(bit_array.to_bits().into_iter().all(|b| b));
        }
    }

    fn test_from_float(mut rng: impl Rng, n_experiments: usize) {
        let float = f64::consts::PI;
        let bit_array = UsizeBitArray::from_f64(float);
        assert_eq!(bit_array.to_bits().len(), 64);
        assert_eq!(f64_to_bits(float), bit_array.to_bits());

        for _ in 0..n_experiments {
            let float = random_f64(&mut rng);
            let bit_array = UsizeBitArray::from_f64(float);
            assert_eq!(bit_array.to_bits().len(), 64);
            assert_eq!(f64_to_bits(float), bit_array.to_bits());
        }
    }

    fn test_to_float(mut rng: impl Rng, n_experiments: usize) {
        let float = f64::consts::E;
        let bit_array = UsizeBitArray::from_f64(float);
        assert_eq!(bit_array.to_float().unwrap(), float);

        let bit_array = UsizeBitArray::zeros(65);
        assert!(bit_array.to_float().is_none());

        let bit_array = UsizeBitArray::ones(63);
        assert!(bit_array.to_float().is_none());

        for _ in 0..n_experiments {
            let float = random_f64(&mut rng);
            let bit_array = UsizeBitArray::from_f64(float);
            assert_eq!(bit_array.to_float().unwrap(), float);
        }
    }

    fn test_to_bits(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = UsizeBitArray::from_bits(&bits);
        assert_eq!(bit_array.to_bits(), bits);

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let bits = random_bits(&mut rng, len);
            let bit_array = UsizeBitArray::from_bits(&bits);
            assert_eq!(bit_array.to_bits(), bits);
        }
    }

    fn test_to_bytes(mut rng: impl Rng, n_experiments: usize) {
        let bytes = vec![0b00001111, 0b00000010];
        let bit_array = UsizeBitArray::from_bytes(&bytes, 12);
        assert_eq!(bit_array.to_bytes(), bytes);

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let bytes = random_bytes(&mut rng, len);
            let bit_array = UsizeBitArray::from_bytes(&bytes, len * 8);
            assert_eq!(bit_array.to_bytes(), bytes);
        }
    }

    #[rstest]
    fn test_len(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = UsizeBitArray::from_bits(&bits);
        assert_eq!(bit_array.len(), bits.len());

        for _ in 0..(n_experiments / 10) {
            let n_bits: usize = rng.random_range(1..100_000);
            let bytes = vec![0u8; n_bits.div_ceil(8usize)];
            let bit_array = UsizeBitArray::from_bytes(&bytes, n_bits);
            assert_eq!(bit_array.len(), n_bits);
        }
    }

    #[rstest]
    fn test_get(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = UsizeBitArray::from_bits(&bits);
        for (i, &bit) in bits.iter().enumerate() {
            assert_eq!(bit_array.get(i).unwrap(), bit);
        }
        assert!(bit_array.get(bits.len()).is_none());

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let bits = random_bits(&mut rng, len);
            let bit_array = UsizeBitArray::from_bits(&bits);

            let i = rng.random_range(0..len);
            assert_eq!(bit_array.get(i).unwrap(), bits[i]);

            assert!(bit_array.get(bits.len()).is_none());
        }
    }

    #[rstest]
    fn test_get_mut(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let mut bit_array = UsizeBitArray::from_bits(&bits);
        for (i, &bit) in bits.iter().enumerate() {
            assert_eq!(*bit_array.get_mut(i).unwrap(), bit);
        }

        *bit_array.get_mut(1).unwrap() = true;
        assert_eq!(bit_array.get(1), Some(true));

        *bit_array.get_mut(1).unwrap() = false;
        assert_eq!(bit_array.get(1), Some(false));

        assert!(bit_array.get_mut(bits.len()).is_none());

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let bits = random_bits(&mut rng, len);
            let mut bit_array = UsizeBitArray::from_bits(&bits);

            for (i, &bit) in bits.iter().enumerate() {
                assert_eq!(*bit_array.get_mut(i).unwrap(), bit);
            }

            let i = rng.random_range(0..len);
            let updated = !bits[i];
            *bit_array.get_mut(i).unwrap() = updated;
            assert_eq!(bit_array.get(i), Some(updated));

            assert!(bit_array.get_mut(bits.len()).is_none());
        }
    }

    #[test]
    fn test_sub_word_boundary_correctness() {
        let mut a_bits = vec![false; 130];
        a_bits[0] = true;
        a_bits[1] = true;
        a_bits[2] = true;
        a_bits[129] = true;
        let mut b_bits = vec![false; 130];
        b_bits[0] = true;
        b_bits[1] = true;
        b_bits[129] = true;
        let a = UsizeBitArray::from_bits(&a_bits);
        let b = UsizeBitArray::from_bits(&b_bits);

        let diff = a - b;
        assert!(diff.len() >= 130, "length must be at least 130 bits");
        for i in 0..diff.len() {
            let expect = i == 2;
            assert_eq!(
                diff.get(i),
                Some(expect),
                "bit {i} mismatch in diff: got {:?}, expected {:?}",
                diff.get(i),
                expect
            );
        }
    }

    #[test]
    #[should_panic(expected = "division by zero in UsizeBitArray")]
    fn test_div_by_zero_panics() {
        let a = UsizeBitArray::from_bits(&[true, false, true]);
        let b = UsizeBitArray::from_bits(&[false, false, false]);
        let _ = a / b;
    }

    #[rstest]
    fn test_get_range(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = UsizeBitArray::from_bits(&bits);

        // Test valid ranges
        let range = bit_array.get_range(1..3).unwrap();
        assert_eq!(range.to_bits(), vec![false, true]);

        let range = bit_array.get_range(0..5).unwrap();
        assert_eq!(range.to_bits(), bits);

        // Test invalid ranges
        assert!(bit_array.get_range(3..10).is_none());
        assert!(bit_array.get_range(5..5).is_some());

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let bits = random_bits(&mut rng, len);
            let bit_array = UsizeBitArray::from_bits(&bits);

            let start = rng.random_range(0..len);
            let end = rng.random_range(start..=len);

            let range = bit_array.get_range(start..end).unwrap();
            assert_eq!(range.to_bits(), bits[start..end].to_vec());
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
    fn test_from_biguint(mut rng: impl Rng, n_experiments: usize) {
        // Test with a known value
        let biguint = BigUint::from(0b11110000u8);
        let bit_array = UsizeBitArray::from_biguint(&biguint);
        let expected_bits = vec![false, false, false, false, true, true, true, true];
        assert_eq!(bit_array.to_bits(), expected_bits);

        // Test with zero
        let biguint = BigUint::from(0u8);
        let bit_array = UsizeBitArray::from_biguint(&biguint);
        assert_eq!(bit_array.to_bits(), vec![]);

        // Test with larger known values
        let biguint = BigUint::from(0x1234u16);
        let bit_array = UsizeBitArray::from_biguint(&biguint);
        // 0x1234 = 0001001000110100 in binary (MSB first)
        // But stored as LSB first: [0,0,1,0,1,1,0,0,0,1,0,0,1,0,0,0]
        let expected_bits = vec![
            false, false, true, false, true, true, false, false, // 0x34 = 52
            false, true, false, false, true, // 0x12 = 18
        ];
        assert_eq!(bit_array.to_bits(), expected_bits);

        for _ in 0..n_experiments {
            let n_bits = rng.random_range(1..100);
            let biguint = random_biguint(&mut rng, n_bits);
            let bit_array = UsizeBitArray::from_biguint_fixed(&biguint, n_bits);
            assert_eq!(bit_array.len(), n_bits);
            let expected_bits: Vec<bool> = (0..n_bits).map(|i| biguint.bit(i as u64)).collect();
            assert_eq!(bit_array.to_bits(), expected_bits);
        }
    }

    fn test_to_biguint(mut rng: impl Rng, n_experiments: usize) {
        let big_uint = BigUint::from(0b11110000u8);
        let bit_array = UsizeBitArray::from_biguint(&big_uint);
        assert_eq!(bit_array.to_biguint(), big_uint);

        let big_uint = BigUint::from(0u8);
        let bit_array = UsizeBitArray::from_biguint(&big_uint);
        assert_eq!(bit_array.to_biguint(), big_uint);

        let big_uint = BigUint::from(0x1234u16);
        let bit_array = UsizeBitArray::from_biguint(&big_uint);
        assert_eq!(bit_array.to_biguint(), big_uint);

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let bytes = random_bytes(&mut rng, len);
            let big_uint = BigUint::from_bytes_le(&bytes);
            let bit_array = UsizeBitArray::from_biguint(&big_uint);
            assert_eq!(bit_array.to_biguint(), big_uint);
        }
    }

    fn test_from_bigint(mut rng: impl Rng, n_experiments: usize) {
        // Test with positive value
        let bigint = BigInt::from(7i8);
        let bit_array = UsizeBitArray::from_bigint(&bigint, 4).unwrap();
        // 7 + 8 = 15 = 0b1111
        let expected_bits = vec![true, true, true, true]; // LSB first
        assert_eq!(bit_array.to_bits(), expected_bits);

        // Test with negative value
        let bigint = BigInt::from(-1i8);
        let bit_array = UsizeBitArray::from_bigint(&bigint, 4).unwrap();
        // -1 + 8 = 7 = 0b0111
        let expected_bits = vec![true, true, true, false];
        assert_eq!(bit_array.to_bits(), expected_bits);

        // Test with zero
        let bigint = BigInt::from(0i8);
        let bit_array = UsizeBitArray::from_bigint(&bigint, 4).unwrap();
        // 0 + 8 = 8 = 0b1000
        let expected_bits = vec![false, false, false, true];
        assert_eq!(bit_array.to_bits(), expected_bits);

        // Test with minimum value
        let bigint = BigInt::from(-8i8); // Minimum for 4-bit signed
        let bit_array = UsizeBitArray::from_bigint(&bigint, 4).unwrap();
        // -8 + 8 = 0 = 0b0000
        let expected_bits = vec![false, false, false, false];
        assert_eq!(bit_array.to_bits(), expected_bits);

        // Test overflow cases
        let bigint = BigInt::from(8i8); // Too large for 4 bits (max is 7)
        assert!(UsizeBitArray::from_bigint(&bigint, 4).is_none());

        let bigint = BigInt::from(-9i8); // Too small for 4 bits (min is -8)
        assert!(UsizeBitArray::from_bigint(&bigint, 4).is_none());

        // Test with larger bit widths
        let bigint = BigInt::from(100i16);
        let bit_array = UsizeBitArray::from_bigint(&bigint, 8).unwrap();
        // 100 + 128 = 228 = 0b11100100
        let expected_bits = vec![false, false, true, false, false, true, true, true];
        assert_eq!(bit_array.to_bits(), expected_bits);

        for _ in 0..n_experiments {
            let n_bits = rng.random_range(2..32); // At least 2 bits for sign
            let bigint = random_bigint(&mut rng, n_bits);

            let bit_array = UsizeBitArray::from_bigint(&bigint, n_bits)
                .expect("Should fit in the given bit width");

            // Check sign
            let bits = bit_array.to_bits();
            let sign_bit = !bits[n_bits - 1];
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

    fn test_to_bigint(mut rng: impl Rng, n_experiments: usize) {
        let bigint = BigInt::from(7i8);
        let bit_array = UsizeBitArray::from_bigint(&bigint, 4).unwrap();
        assert_eq!(bit_array.to_bigint(), bigint);

        let bigint = BigInt::from(-1i8);
        let bit_array = UsizeBitArray::from_bigint(&bigint, 4).unwrap();
        assert_eq!(bit_array.to_bigint(), bigint);

        let bigint = BigInt::from(0i8);
        let bit_array = UsizeBitArray::from_bigint(&bigint, 4).unwrap();
        assert_eq!(bit_array.to_bigint(), bigint);

        for _ in 0..n_experiments {
            let n_bits = rng.random_range(2..100); // At least 2 bits for sign
            let bigint = random_bigint(&mut rng, n_bits);
            let bit_array = UsizeBitArray::from_bigint(&bigint, n_bits).unwrap();
            assert_eq!(bit_array.to_bigint(), bigint);
        }
    }

    #[rstest]
    fn test_bitarray_biguint(mut rng: impl Rng, n_experiments: usize) {
        test_from_biguint(&mut rng, n_experiments);
        test_to_biguint(&mut rng, n_experiments);
    }

    #[rstest]
    fn test_bitarray_bigint(mut rng: impl Rng, n_experiments: usize) {
        test_from_bigint(&mut rng, n_experiments);
        test_to_bigint(&mut rng, n_experiments);
    }

    #[rstest]
    fn test_append_bool(mut rng: impl Rng, n_experiments: usize) {
        // Test appending to empty bit array
        let empty_array = UsizeBitArray::zeros(0);
        let result_true = empty_array.clone().append_bool_in_place(true);
        assert_eq!(result_true.len(), 1);
        assert_eq!(result_true.to_bits(), vec![true]);

        // Test multiple appends
        let mut bit_array = UsizeBitArray::from_bits(&[true, false]);
        bit_array = bit_array.append_bool_in_place(true);
        bit_array = bit_array.append_bool_in_place(false);
        bit_array = bit_array.append_bool_in_place(true);
        bit_array = bit_array.append_bool_in_place(true);

        let expected = vec![true, false, true, false, true, true];
        assert_eq!(bit_array.to_bits(), expected);

        // Random tests
        for _ in 0..n_experiments {
            let len = rng.random_range(0..100);
            let mut original_bits = random_bits(&mut rng, len);
            let mut bit_array = UsizeBitArray::from_bits(&original_bits);
            let n_extra_bits = rng.random_range(1..20);

            for _ in 0..n_extra_bits {
                let append_value = rng.random_bool(0.5);
                let mut new_bit_array = bit_array.clone().append_bool_in_place(append_value);
                original_bits.push(append_value);
                core::mem::swap(&mut bit_array, &mut new_bit_array);
            }
            assert_eq!(bit_array.to_bits(), original_bits);
        }
    }

    #[rstest]
    fn test_shift_with_bool(mut rng: impl Rng, n_experiments: usize) {
        // Test shift by 0 (no change)
        let original_bits = vec![true, false, true, true, false];
        let bit_array = UsizeBitArray::from_bits(&original_bits);
        let result = bit_array.clone().shift_fixed(0);
        assert_eq!(result.to_bits(), original_bits);

        // Test positive shift with true fill
        let result = bit_array.clone().shift_fixed_with_fill(2, true);
        let expected = vec![true, true, false, true, true]; // Takes bits[2..], then adds true fill
        assert_eq!(result.to_bits(), expected);

        // Test negative shift with true fill
        let result = bit_array.clone().shift_fixed_with_fill(-2, true);
        let expected = vec![true, true, true, false, true]; // Adds true fill, then takes bits[..3]
        assert_eq!(result.to_bits(), expected);

        // Test with empty array
        let empty_array = UsizeBitArray::zeros(0);
        let result = empty_array.clone().shift_fixed_with_fill(5, true);
        assert_eq!(result.len(), 0);
        let result = empty_array.shift_fixed_with_fill(-5, false);
        assert_eq!(result.len(), 0);

        // Random tests with boolean fill values
        for _ in 0..n_experiments {
            let len = rng.random_range(1..20);
            let original_bits = random_bits(&mut rng, len);
            let bit_array = UsizeBitArray::from_bits(&original_bits);
            let shift_amount = rng.random_range(-10..10) as isize;
            let fill_value = rng.random_bool(0.5);

            let result = bit_array
                .clone()
                .shift_fixed_with_fill(shift_amount, fill_value);
            assert_eq!(result.len(), len);
            let shift_abs = usize::min(shift_amount.unsigned_abs(), len);

            match shift_amount.cmp(&0) {
                Ordering::Equal => {
                    assert_eq!(result.to_bits(), original_bits);
                }
                Ordering::Less => {
                    // Negative shift: fill at start
                    let bits = result.to_bits();
                    assert!(bits[..shift_abs].iter().all(|&b| b == fill_value));
                    assert_eq!(&bits[shift_abs..], &original_bits[..len - shift_abs]);
                }
                Ordering::Greater => {
                    // Positive shift: fill at end
                    let bits = result.to_bits();
                    assert!(bits[len - shift_abs..].iter().all(|&b| b == fill_value));
                    assert_eq!(&bits[..len - shift_abs], &original_bits[shift_abs..]);
                }
            }
        }
    }

    #[rstest]
    fn test_add(mut rng: impl Rng, n_experiments: usize) {
        // Test simple cases
        // 3 + 5 = 8
        let a = UsizeBitArray::from_bits(&[true, true, false]); // 3
        let b = UsizeBitArray::from_bits(&[true, false, true]); // 5
        let result = a + b;
        let expected_value = 8u32;
        assert_eq!(result.to_biguint(), BigUint::from(expected_value));

        // 0 + 0 = 0
        let a = UsizeBitArray::zeros(1);
        let b = UsizeBitArray::zeros(1);
        let result = a + b;
        assert_eq!(result.to_biguint(), BigUint::from(0u8));

        // 7 + 1 = 8 (with carry, requires extra bit)
        let a = UsizeBitArray::from_bits(&[true, true, true]); // 7
        let b = UsizeBitArray::from_bits(&[true]); // 1
        let result = a + b;
        assert_eq!(result.to_biguint(), BigUint::from(8u8));
        assert!(
            result.len() >= 3,
            "Result should have grown to accommodate carry"
        );

        // Random tests
        for _ in 0..n_experiments {
            let len_a = rng.random_range(1..20);
            let len_b = rng.random_range(1..20);
            let a_uint = random_biguint(&mut rng, len_a);
            let b_uint = random_biguint(&mut rng, len_b);

            let a = UsizeBitArray::from_biguint(&a_uint);
            let b = UsizeBitArray::from_biguint(&b_uint);
            let len_a_actual = a.len();
            let len_b_actual = b.len();
            let result = a + b;

            let expected = &a_uint + &b_uint;
            assert_eq!(result.to_biguint(), expected);

            // Verify array grew to accommodate the result
            let expected_bits = expected.bits() as usize;
            let max_len = core::cmp::max(len_a_actual, len_b_actual);
            assert!(result.len() >= core::cmp::max(max_len, expected_bits));
        }
    }

    #[rstest]
    fn test_sub(mut rng: impl Rng, n_experiments: usize) {
        // Test simple positive subtraction cases
        // 7 - 3 = 4
        let a = UsizeBitArray::from_bits(&[true, true, true, false]); // 7
        let b = UsizeBitArray::from_bits(&[true, true, false, false]); // 3
        let result = a - b;
        assert_eq!(result.to_biguint(), BigUint::from(4u8));

        // 5 - 5 = 0
        let a = UsizeBitArray::from_bits(&[true, false, true]); // 5
        let b = UsizeBitArray::from_bits(&[true, false, true]); // 5
        let result = a - b;
        assert_eq!(result.to_biguint(), BigUint::from(0u8));

        // Larger subtraction: 15 - 1 = 14
        let a = UsizeBitArray::from_bits(&[true, true, true, true]); // 15
        let b = UsizeBitArray::from_bits(&[true]); // 1
        let result = a - b;
        assert_eq!(result.to_biguint(), BigUint::from(14u8));

        // Test bigger lhs
        let a = UsizeBitArray::from_bits(&[true, true]);
        let b = UsizeBitArray::from_bits(&[true, false, false]);
        let result = a - b;
        let expected = UsizeBitArray::from_bits(&[false, true, false]);
        assert_eq!(result.to_bits(), expected.to_bits());

        for _ in 0..n_experiments {
            let len_a = rng.random_range(1..20);
            let len_b = rng.random_range(1..20);
            let mut a_uint = random_biguint(&mut rng, len_a);
            let mut b_uint = random_biguint(&mut rng, len_b);

            // Ensure a >= b to avoid underflow
            if a_uint < b_uint {
                core::mem::swap(&mut a_uint, &mut b_uint);
            }

            let a = UsizeBitArray::from_biguint(&a_uint);
            let b = UsizeBitArray::from_biguint(&b_uint);
            let result = a - b;

            let expected = &a_uint - &b_uint;
            assert_eq!(result.to_biguint(), expected);
        }
    }

    #[rstest]
    fn test_mul(mut rng: impl Rng, n_experiments: usize) {
        // Test simple multiplication cases
        // 3 * 5 = 15
        let a = UsizeBitArray::from_bits(&[true, true, false]); // 3
        let b = UsizeBitArray::from_bits(&[true, false, true]); // 5
        let result = a * b;
        assert_eq!(result.to_biguint(), BigUint::from(15u8));

        // 0 * 7 = 0
        let a = UsizeBitArray::zeros(1); // 0
        let b = UsizeBitArray::from_bits(&[true, true, true]); // 7
        let result = a * b;
        assert_eq!(result.to_biguint(), BigUint::from(0u8));

        // 4 * 2 = 8
        let a = UsizeBitArray::from_bits(&[false, false, true]); // 4
        let b = UsizeBitArray::from_bits(&[false, true]); // 2
        let result = a * b;
        assert_eq!(result.to_biguint(), BigUint::from(8u8));

        for _ in 0..n_experiments {
            let len_a = rng.random_range(1..100);
            let len_b = rng.random_range(1..100);
            let a_uint = random_biguint(&mut rng, len_a);
            let b_uint = random_biguint(&mut rng, len_b);

            let a = UsizeBitArray::from_biguint(&a_uint);
            let b = UsizeBitArray::from_biguint(&b_uint);
            let result = a * b;

            let expected = &a_uint * &b_uint;
            assert_eq!(result.to_biguint(), expected);
        }
    }

    #[rstest]
    fn test_div(mut rng: impl Rng, n_experiments: usize) {
        // Test simple division cases
        // 4 / 2 = 2
        let a = UsizeBitArray::from_bits(&[false, false, true]); // 4
        let b = UsizeBitArray::from_bits(&[false, true]); // 2
        let result = a / b;
        assert_eq!(result.to_biguint(), BigUint::from(2u8));

        // 15 / 3 = 5
        let a = UsizeBitArray::from_bits(&[true, true, true, true]); // 15
        let b = UsizeBitArray::from_bits(&[true, true, false]); // 3
        let result = a / b;
        assert_eq!(result.to_biguint(), BigUint::from(5u8));

        // 1 / 2 = 0
        let a = UsizeBitArray::from_bits(&[true]); // 1
        let b = UsizeBitArray::from_bits(&[false, true]); // 2
        let result = a / b;
        assert_eq!(result.to_biguint(), BigUint::from(0u8));

        for _ in 0..n_experiments {
            let len_a = rng.random_range(1..10);
            let len_b = rng.random_range(1..10);
            let a_uint = random_biguint(&mut rng, len_a);
            let mut b_uint = random_biguint(&mut rng, len_b);

            // Ensure b != 0 to avoid division by zero
            if b_uint.is_zero() {
                b_uint = BigUint::from(1u8);
            }

            let a = UsizeBitArray::from_biguint(&a_uint);
            let b = UsizeBitArray::from_biguint(&b_uint);
            let result = a / b;

            let expected = &a_uint / &b_uint;
            assert_eq!(result.to_biguint(), expected);
        }
    }

    #[rstest]
    fn test_shift_grow_with_fill(mut rng: impl Rng, n_experiments: usize) {
        // Shift right (positive shift)
        let bits = vec![true, false, true, false];
        let bit_array = UsizeBitArray::from_bits(&bits);
        let shifted = bit_array.clone().shift_grow_with_fill(2, true);
        let expected = vec![true, true, true, false, true, false];
        assert_eq!(shifted.to_bits(), expected);

        // Shift left (negative shift)
        let shifted = bit_array.clone().shift_grow_with_fill(-2, false);
        let expected = vec![true, false, true, false, false, false];
        assert_eq!(shifted.to_bits(), expected);

        // Shift by 0 (no change)
        let shifted = bit_array.clone().shift_grow_with_fill(0, true);
        assert_eq!(shifted.to_bits(), bits);

        // Randomized tests
        for _ in 0..n_experiments {
            let len = rng.random_range(1..20);
            let bits = random_bits(&mut rng, len);
            let bit_array = UsizeBitArray::from_bits(&bits);
            let shift = rng.random_range(-10..10) as isize;
            let fill = rng.random_bool(0.5);
            let shifted = bit_array.clone().shift_grow_with_fill(shift, fill);

            assert_eq!(shifted.len(), len + shift.unsigned_abs());
            if shift == 0 {
                assert_eq!(shifted.to_bits(), bits);
            } else if shift > 0 {
                assert!(
                    shifted.to_bits()[..shift as usize]
                        .iter()
                        .all(|&b| b == fill)
                );
                assert_eq!(&shifted.to_bits()[shift as usize..], &bits[..]);
            } else {
                assert!(shifted.to_bits()[len..].iter().all(|&b| b == fill));
                assert_eq!(&shifted.to_bits()[..len], &bits[..]);
            }
        }
    }

    #[rstest]
    fn test_reset(mut rng: impl Rng, n_experiments: usize) {
        // Test reset on a non-empty array
        let bits = vec![true, false, true, true, false];
        let bit_array = UsizeBitArray::from_bits(&bits);
        let reset_array = bit_array.clone().reset();
        assert_eq!(reset_array.to_bits(), vec![false; bits.len()]);

        // Test reset on an already zeroed array
        let zero_array = UsizeBitArray::zeros(10);
        let reset_zero = zero_array.clone().reset();
        assert_eq!(reset_zero.to_bits(), vec![false; 10]);

        // Randomized tests
        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let bits = random_bits(&mut rng, len);
            let bit_array = UsizeBitArray::from_bits(&bits);
            let reset_array = bit_array.reset();
            assert_eq!(reset_array.to_bits(), vec![false; len]);
        }
    }

    #[rstest]
    fn test_truncate(mut rng: impl Rng, n_experiments: usize) {
        // Test truncate to smaller size
        let bits = vec![true, false, true, true, false];
        let bit_array = UsizeBitArray::from_bits(&bits);
        let truncated = bit_array.clone().truncate(3);
        assert_eq!(truncated.to_bits(), vec![true, false, true]);

        // Test truncate to same size
        let truncated = bit_array.clone().truncate(5);
        assert_eq!(truncated.to_bits(), bits);

        // Test truncate to larger size (should keep same)
        let truncated = bit_array.clone().truncate(10);
        assert_eq!(truncated.to_bits(), bits);

        // Randomized tests
        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let bits = random_bits(&mut rng, len);
            let bit_array = UsizeBitArray::from_bits(&bits);
            let new_len = rng.random_range(0..len + 10);
            let truncated = bit_array.truncate(new_len);

            if new_len <= len {
                assert_eq!(truncated.to_bits(), bits[..new_len].to_vec());
            } else {
                assert_eq!(truncated.to_bits(), bits);
            }
        }
    }

    #[rstest]
    fn test_iter_bits(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = UsizeBitArray::from_bits(&bits);

        // Test forward iteration
        let collected: Vec<bool> = bit_array.iter_bits().collect();
        assert_eq!(collected, bits);

        // Test reverse iteration
        let mut reversed: Vec<bool> = bit_array.iter_bits().rev().collect();
        reversed.reverse();
        assert_eq!(reversed, bits);

        // Test length
        assert_eq!(bit_array.iter_bits().len(), bits.len());

        for _ in 0..n_experiments {
            let len = rng.random_range(0..100);
            let bits = random_bits(&mut rng, len);
            let bit_array = UsizeBitArray::from_bits(&bits);

            let collected: Vec<bool> = bit_array.iter_bits().collect();
            assert_eq!(collected, bits);
            assert_eq!(bit_array.iter_bits().len(), len);
        }
    }
}
