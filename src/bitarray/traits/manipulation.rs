//! Manipulation methods for BitArray implementations.
//!
//! This module defines the `BitArrayManipulation` trait, which provides
//! methods for modifying and transforming BitArrays.

use core::cmp::Ordering;
use core::hint::unreachable_unchecked;
use core::iter::repeat_n;

use crate::bitarray::{BitArrayConstruction, BitArrayConversion};

/// Trait for manipulating BitArray contents.
///
/// Provides methods for appending, shifting, truncating, and otherwise
/// transforming BitArrays.
pub trait BitArrayManipulation: Sized + BitArrayConversion + BitArrayConstruction {
    fn append_bool(&mut self, value: bool);

    /// Appends a single bit to the end of the BitArray.
    ///
    /// # Arguments
    ///
    /// * `value` - The bit value to append
    ///
    /// # Returns
    ///
    /// New BitArray with the bit appended
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let bits = BoolBitArray::from_bits(&[true, false]);
    /// let extended = bits.append_bool_in_place(true);
    /// assert_eq!(extended.to_bits(), vec![true, false, true]);
    /// ```
    fn append_bool_in_place(self, value: bool) -> Self {
        let mut new_array = self;
        new_array.append_bool(value);
        new_array
    }

    /// Shifts the BitArray by the specified number of positions with fill value.
    ///
    /// - Positive shift: moves bits to higher indices (right shift), vacating lower indices
    /// - Negative shift: moves bits to lower indices (left shift), vacating higher indices
    /// - Vacated positions are filled with the specified fill value
    /// - Array length remains constant
    ///
    /// # Arguments
    ///
    /// * `shift` - Number of positions to shift (positive = right, negative = left)
    /// * `fill` - Value to use for newly created bit positions
    ///
    /// # Returns
    ///
    /// New BitArray with bits shifted
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let bits = BoolBitArray::from_bits(&[true, false, true]);
    /// let shifted = bits.shift_fixed_with_fill(1, false);
    /// assert_eq!(shifted.to_bits(), vec![false, true, false]);
    /// ```
    fn shift_fixed_with_fill(self, shift: isize, fill: bool) -> Self {
        if shift == 0 {
            return self;
        }

        let bits = self.to_bits();
        let shift_abs = usize::min(shift.unsigned_abs(), bits.len());
        let empty_bits = vec![fill; shift_abs];
        match shift.cmp(&0) {
            Ordering::Less => {
                let mut new_bits = empty_bits;
                new_bits.extend_from_slice(&bits[..bits.len() - shift_abs]);
                Self::from_bits(&new_bits)
            }
            Ordering::Greater => {
                let mut new_bits = bits[shift_abs..].to_vec();
                new_bits.extend_from_slice(&empty_bits);
                Self::from_bits(&new_bits)
            }
            // UNSAFE: This branch is unreachable because zero is handled at the top
            Ordering::Equal => unsafe { unreachable_unchecked() },
        }
    }

    /// Shifts the BitArray by the specified number of positions, filling with zeros.
    ///
    /// This is a convenience method equivalent to `shift_with_fill(shift, false)`.
    ///
    /// # Arguments
    ///
    /// * `shift` - Number of positions to shift (positive = right, negative = left)
    ///
    /// # Returns
    ///
    /// New BitArray with bits shifted and zero-filled
    fn shift_fixed(self, shift: isize) -> Self {
        self.shift_fixed_with_fill(shift, false)
    }

    /// Appends a repeated bit value to the end of the BitArray.
    ///
    /// # Arguments
    ///
    /// * `value` - The bit value to append repeatedly
    /// * `count` - Number of times to append the value
    ///
    /// # Returns
    ///
    /// New BitArray with the repeated bits appended
    fn append_repeated(self, value: bool, count: usize) -> Self {
        if count == 0 {
            return self;
        }
        let mut bits = self.to_bits();
        bits.extend(repeat_n(value, count));
        Self::from_bits(&bits)
    }

    /// Truncates the BitArray to the specified number of bits.
    ///
    /// # Arguments
    ///
    /// * `n_bits` - The number of bits to keep
    ///
    /// # Returns
    ///
    /// New BitArray with at most `n_bits` bits
    fn truncate(self, n_bits: usize) -> Self {
        let bits = self.to_bits();
        let truncated_bits = &bits[..usize::min(n_bits, bits.len())];
        Self::from_bits(truncated_bits)
    }

    /// Shifts the BitArray by the specified number of positions, growing the array if needed.
    ///
    /// - Positive shift: moves bits to higher indices (right shift), extends array size
    /// - Negative shift: moves bits to lower indices (left shift), may extend array size
    /// - Newly created positions are filled with the specified fill value
    /// - Array grows as needed to accommodate the shift
    ///
    /// # Arguments
    ///
    /// * `shift` - Number of positions to shift (positive = right, negative = left)
    /// * `fill` - Value to use for newly created bit positions
    ///
    /// # Returns
    ///
    /// New BitArray with bits shifted and grown as needed
    fn shift_grow_with_fill(self, shift: isize, fill: bool) -> Self {
        if shift == 0 {
            return self;
        }

        let bits = self.to_bits();
        match shift.cmp(&0) {
            Ordering::Less => {
                let shift_abs = shift.unsigned_abs();
                let mut new_bits = bits;
                new_bits.extend(repeat_n(fill, shift_abs));
                Self::from_bits(&new_bits)
            }
            Ordering::Greater => {
                let shift_abs = shift as usize;
                let mut new_bits = Vec::with_capacity(bits.len() + shift_abs);
                new_bits.extend(repeat_n(fill, shift_abs));
                new_bits.extend(bits);
                Self::from_bits(&new_bits)
            }
            // UNSAFE: This branch is unreachable because zero is handled at the top
            Ordering::Equal => unsafe { unreachable_unchecked() },
        }
    }

    /// Shifts the BitArray by the specified number of positions, growing the array if needed, filling with zeros.
    ///
    /// This is a convenience method equivalent to `shift_grow_with_fill(shift, false)`.
    ///
    /// # Arguments
    ///
    /// * `shift` - Number of positions to shift (positive = right, negative = left)
    ///
    /// # Returns
    ///
    /// New BitArray with bits shifted, grown, and zero-filled
    fn shift_grow(self, shift: isize) -> Self {
        self.shift_grow_with_fill(shift, false)
    }

    /// Sets all bits in the BitArray to false (zero).
    ///
    /// # Returns
    ///
    /// New BitArray with all bits cleared to false
    fn reset(self) -> Self {
        let n_bits = self.len();
        let n_bytes = n_bits.div_ceil(8);
        let bytes = vec![0u8; n_bytes];
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
    use crate::tests::*;

    #[rstest]
    fn test_default_append_bool(mut rng: impl Rng, n_experiments: usize) {
        let empty_array = BitArrayTest::zeros(0);
        let result_true = empty_array.clone().append_bool_in_place(true);
        assert_eq!(result_true.len(), 1);
        assert_eq!(result_true.to_bits(), vec![true]);

        let mut bit_array = BitArrayTest::from_bits(&[true, false]);
        bit_array = bit_array.append_bool_in_place(true);
        bit_array = bit_array.append_bool_in_place(false);
        let expected = vec![true, false, true, false];
        assert_eq!(bit_array.to_bits(), expected);

        for _ in 0..n_experiments {
            let n = rng.random_range(0..=64);
            let mut bit_array = BitArrayTest::zeros(0);
            let mut bits = Vec::new();
            for _ in 0..n {
                let b: bool = rng.random();
                bit_array = bit_array.append_bool_in_place(b);
                bits.push(b);
            }
            assert_eq!(bit_array.to_bits(), bits);
        }
    }

    #[rstest]
    fn test_default_shift_fixed_with_fill(mut rng: impl Rng, n_experiments: usize) {
        let original_bits = vec![true, false, true, true, false];
        let bit_array = BitArrayTest::from_bits(&original_bits);
        let result = bit_array.clone().shift_fixed(0);
        assert_eq!(result.to_bits(), original_bits);

        let result = bit_array.clone().shift_fixed_with_fill(2, true);
        let expected = vec![true, true, false, true, true];
        assert_eq!(result.to_bits(), expected);

        let result = bit_array.clone().shift_fixed_with_fill(-2, true);
        let expected = vec![true, true, true, false, true];
        assert_eq!(result.to_bits(), expected);

        let empty_array = BitArrayTest::zeros(0);
        let result = empty_array.clone().shift_fixed_with_fill(5, true);
        assert_eq!(result.len(), 0);
        let result = empty_array.shift_fixed_with_fill(-5, false);
        assert_eq!(result.len(), 0);

        for _ in 0..n_experiments {
            let n = rng.random_range(1..=64);
            let bits: Vec<bool> = (0..n).map(|_| rng.random()).collect();
            let bit_array = BitArrayTest::from_bits(&bits);
            let shift = rng.random_range(-(n as i64 - 1)..=n as i64 - 1) as isize;
            let fill: bool = rng.random();
            let result = bit_array.clone().shift_fixed_with_fill(shift, fill);
            let expected = if shift >= 0 {
                let shift = shift as usize;
                bits[shift..]
                    .iter()
                    .copied()
                    .chain(std::iter::repeat_n(fill, shift))
                    .collect::<Vec<_>>()
            } else {
                let shift = (-shift) as usize;
                std::iter::repeat_n(fill, shift)
                    .chain(bits[..n - shift].iter().copied())
                    .collect::<Vec<_>>()
            };
            assert_eq!(result.len(), n);
            assert_eq!(result.to_bits(), expected);
        }
    }

    #[rstest]
    fn test_default_truncate(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = BitArrayTest::from_bits(&bits);
        let truncated = bit_array.clone().truncate(3);
        assert_eq!(truncated.to_bits(), vec![true, false, true]);
        let truncated = bit_array.truncate(10);
        assert_eq!(truncated.to_bits(), bits);

        for _ in 0..n_experiments {
            let n = rng.random_range(1..=64);
            let bits: Vec<bool> = (0..n).map(|_| rng.random()).collect();
            let bit_array = BitArrayTest::from_bits(&bits);
            let trunc = rng.random_range(0..=n + 5);
            let truncated = bit_array.truncate(trunc);
            assert_eq!(truncated.to_bits(), bits[..usize::min(trunc, n)].to_vec());
        }
    }

    #[rstest]
    fn test_default_reset(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = BitArrayTest::from_bits(&bits);
        let reset = bit_array.reset();
        assert_eq!(reset.to_bits(), vec![false; bits.len()]);

        for _ in 0..n_experiments {
            let n = rng.random_range(1..=64);
            let bits: Vec<bool> = (0..n).map(|_| rng.random()).collect();
            let bit_array = BitArrayTest::from_bits(&bits);
            let reset = bit_array.reset();
            let reset_bits = reset.to_bits();
            assert_eq!(reset_bits.len(), n);
            assert!(reset_bits.iter().all(|&b| !b));
        }
    }
}
