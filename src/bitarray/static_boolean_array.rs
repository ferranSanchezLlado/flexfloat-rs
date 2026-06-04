#![allow(dead_code)]
use core::ops::{Index, Range};

use crate::{DefaultBitArray, bitarray::traits::*};

#[derive(Debug)]
pub(crate) struct StaticBoolArray {
    bits: [bool; 56],
    size: usize,
}

impl StaticBoolArray {
    pub const fn new<const N: usize>(bits: [bool; N]) -> Self {
        if bits.len() > 56 {
            panic!("StaticBoolArray can hold up to 56 bits");
        }

        let mut fixed_bits = [false; 56];
        let mut i = 0;
        while i < N {
            fixed_bits[i] = bits[i];
            i += 1;
        }
        StaticBoolArray {
            bits: fixed_bits,
            size: N,
        }
    }

    pub const fn bits(&self) -> &[bool] {
        // SAFETY: The size is always <= 56, so it never exceeds the allocated array size.
        unsafe { core::slice::from_raw_parts(self.bits.as_ptr(), self.size) }
    }
}

impl BitArrayConversion for StaticBoolArray {
    fn to_bits(&self) -> Vec<bool> {
        Vec::from(self.bits())
    }

    fn to_bytes(&self) -> Vec<u8> {
        let n_bits = self.size;
        let n_bytes = n_bits.div_ceil(8);
        let mut bytes = vec![0u8; n_bytes];

        for (i, &bit) in self.bits().iter().enumerate() {
            if bit {
                let byte_index = i / 8;
                let bit_index = i % 8;
                bytes[byte_index] |= 1 << bit_index;
            }
        }
        bytes
    }
}

impl BitArrayAccess for StaticBoolArray {
    fn iter_bits(&self) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator {
        self.bits().iter().copied()
    }

    fn len(&self) -> usize {
        self.size
    }

    fn get(&self, index: usize) -> Option<bool> {
        self.bits().get(index).copied()
    }

    fn get_range(&self, range: Range<usize>) -> Option<Self>
    where
        Self: Sized,
    {
        if range.start > range.end || range.end > self.size {
            return None;
        }

        let mut bits = [false; 56];
        let mut i = 0;
        while i < range.end - range.start {
            bits[i] = self.bits[range.start + i];
            i += 1;
        }

        Some(Self {
            bits,
            size: range.end - range.start,
        })
    }
}

impl Index<usize> for StaticBoolArray {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        &self.bits()[index]
    }
}

impl Index<Range<usize>> for StaticBoolArray {
    type Output = [bool];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        &self.bits()[index]
    }
}

impl From<&StaticBoolArray> for DefaultBitArray {
    fn from(array: &StaticBoolArray) -> Self {
        DefaultBitArray::from_bits(array.bits())
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::tests::*;

    #[rstest]
    fn test_boolean_array(mut rng: impl Rng, n_experiments: usize) {
        let array = [true, false, true, true, false];
        let arr = StaticBoolArray::new(array);
        assert_eq!(arr.bits(), &array);

        for _ in 0..(n_experiments / 2) {
            let bits = random_bits(&mut rng, 11);
            let bits: [bool; 11] = bits.try_into().unwrap();
            let arr = StaticBoolArray::new(bits);
            assert_eq!(arr.bits(), &bits);

            let bits = random_bits(&mut rng, 52);
            let bits: [bool; 52] = bits.try_into().unwrap();
            let arr = StaticBoolArray::new(bits);
            assert_eq!(arr.bits(), &bits);
        }
    }

    const fn check_equal(a: &[bool], b: &[bool]) {
        let mut i = 0;
        while i < a.len() {
            if a[i] != b[i] {
                panic!("Found unequal values");
            }
            i += 1;
        }
    }

    #[test]
    const fn test_static_bool_array_const_time() {
        const ARRAY: [bool; 11] = [
            true, false, true, false, true, true, false, false, true, false, true,
        ];
        const STATIC_ARRAY: StaticBoolArray = StaticBoolArray::new(ARRAY);
        check_equal(STATIC_ARRAY.bits(), &ARRAY);

        const ARRAY2: [bool; 52] = [false; 52];
        const STATIC_ARRAY2: StaticBoolArray = StaticBoolArray::new(ARRAY2);
        check_equal(STATIC_ARRAY2.bits(), &ARRAY2);
    }
}
