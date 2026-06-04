//! Access methods for BitArray implementations.
//!
//! This module defines the [`BitArrayAccess`](crate::bitarray::traits::BitArrayAccess) trait, which provides
//! methods for reading and querying BitArray contents.

use core::ops::{DerefMut, Range};

/// Trait for accessing BitArray elements and metadata.
///
/// Provides methods for reading individual bits, ranges of bits,
/// and querying array properties like length.
pub trait BitArrayAccess {
    /// Returns an iterator over the bits in the array.
    ///
    /// # Returns
    ///
    /// Iterator yielding each bit as a boolean value
    fn iter_bits(&self) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator;

    /// Checks if all bits in the array are zero (false).
    ///
    /// # Returns
    ///
    /// true if all bits are false, false otherwise
    fn is_zeros(&self) -> bool {
        self.iter_bits().all(|b| !b)
    }

    /// Checks if all bits in the array are one (true).
    ///
    /// # Returns
    ///
    /// true if all bits are true, false otherwise
    fn is_ones(&self) -> bool {
        self.iter_bits().all(|b| b)
    }

    /// Returns the number of bits in the array.
    ///
    /// # Returns
    ///
    /// Total number of bits stored
    fn len(&self) -> usize {
        self.iter_bits().len()
    }

    /// Gets a reference to the bit at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - Zero-based bit index
    ///
    /// # Returns
    ///
    /// Some(&bool) if index is valid, None otherwise
    fn get(&self, index: usize) -> Option<bool>;

    /// Checks if the BitArray is empty (has zero bits).
    ///
    /// # Returns
    ///
    /// true if the array has no bits, false otherwise
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Extension trait for BitArrays that support extracting sub-ranges as new BitArrays.
///
/// This trait is separate from `BitArrayAccess` to allow implementations that do not support range extraction.
/// The `get_range` method returns a new BitArray containing the specified range of bits, or None if the range is invalid.
/// Implementors of `BitArrayAccess` can choose to implement `BitArrayRangeAccess` if they support this functionality.
pub trait BitArrayRangeAccess: BitArrayAccess {
    /// Extracts a sub-range of bits as a new BitArray.
    ///
    /// # Arguments
    ///
    /// * `range` - Range of bit indices to extract
    ///
    /// # Returns
    ///
    /// Some(BitArray) containing the specified range, None if range is invalid
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let bits = DefaultBitArray::from_bits(&[true, false, true, false]);
    /// let sub = bits.get_range(1..3).unwrap();
    /// assert_eq!(sub.to_bits(), vec![false, true]);
    /// ```
    fn get_range(&self, range: Range<usize>) -> Option<Self>
    where
        Self: Sized;
}

pub trait BitArrayMutAccess: BitArrayAccess {
    /// The mutable-handle type returned by [`Self::get_mut`].
    ///
    /// Each backend selects a representation appropriate for its storage:
    /// - `Vec<bool>`-backed backends can use `&'a mut bool` directly.
    /// - Packed (e.g. word-backed) backends use a proxy such as [`BitRef<'a>`](crate::bitarray::BitRef)
    ///   that flushes the value back to storage when dropped.
    ///
    /// Generic code over `T: BitArrayMutAccess` reads via `Deref` and writes
    /// via `DerefMut`; it never names the concrete type.
    type BitMut<'a>: DerefMut<Target = bool>
    where
        Self: 'a;

    /// Gets a mutable handle to the bit at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - Zero-based bit index
    ///
    /// # Returns
    ///
    /// `Some(handle)` if the index is valid, `None` otherwise. The returned
    /// handle dereferences to `bool` via `Deref`/`DerefMut`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::prelude::*;
    ///
    /// let mut bits = DefaultBitArray::from_bits(&[true, false, true]);
    /// if let Some(mut bit) = bits.get_mut(1) {
    ///     *bit = true; // Set the second bit to true
    /// }
    /// assert_eq!(bits.to_bits(), vec![true, true, true]);
    /// ```
    fn get_mut(&mut self, index: usize) -> Option<Self::BitMut<'_>>;
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
    fn test_default_len(mut rng: impl Rng, n_experiments: usize) {
        let empty = BitArrayTest::zeros(0);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());

        let bits = vec![true, false, true, true, false];
        let bit_array = BitArrayTest::from_bits(&bits);
        assert_eq!(bit_array.len(), bits.len());
        assert!(!bit_array.is_empty());

        for _ in 0..n_experiments {
            let n = rng.random_range(0..=128);
            let bits: Vec<bool> = (0..n).map(|_| rng.random()).collect();
            let bit_array = BitArrayTest::from_bits(&bits);
            assert_eq!(bit_array.len(), n);
            assert_eq!(bit_array.is_empty(), n == 0);
        }
    }

    #[rstest]
    fn test_default_get(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = BitArrayTest::from_bits(&bits);
        for (i, &bit) in bits.iter().enumerate() {
            assert_eq!(bit_array.get(i), Some(bit));
        }
        assert!(bit_array.get(bits.len()).is_none());

        for _ in 0..n_experiments {
            let n = rng.random_range(1..=128);
            let bits: Vec<bool> = (0..n).map(|_| rng.random()).collect();
            let bit_array = BitArrayTest::from_bits(&bits);
            for (i, &bit) in bits.iter().enumerate() {
                assert_eq!(bit_array.get(i), Some(bit));
            }
            assert!(bit_array.get(n).is_none());
        }
    }

    #[rstest]
    fn test_default_get_range(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = BitArrayTest::from_bits(&bits);
        let sub = bit_array.get_range(1..4).unwrap();
        assert_eq!(sub.to_bits(), vec![false, true, true]);
        #[allow(clippy::reversed_empty_ranges)]
        let result = bit_array.get_range(4..2).is_none();
        assert!(result);
        assert!(bit_array.get_range(0..6).is_none());

        for _ in 0..n_experiments {
            let n = rng.random_range(3..=128);
            let bits: Vec<bool> = (0..n).map(|_| rng.random()).collect();
            let bit_array = BitArrayTest::from_bits(&bits);
            let start = rng.random_range(0..n - 2);
            let end = rng.random_range(start + 1..=n);
            let sub = bit_array.get_range(start..end).unwrap();
            assert_eq!(sub.to_bits(), bits[start..end].to_vec());
        }
    }

    #[rstest]
    fn test_default_is_zeros(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = BitArrayTest::from_bits(&bits);
        assert!(!bit_array.is_zeros());

        let zeros = BitArrayTest::zeros(bits.len());
        assert!(zeros.is_zeros());

        let ones = BitArrayTest::ones(bits.len());
        assert!(!ones.is_zeros());

        for _ in 0..n_experiments {
            let n = rng.random_range(1..=128);
            let bits: Vec<bool> = (0..n).map(|_| rng.random()).collect();
            let bit_array = BitArrayTest::from_bits(&bits);
            assert_eq!(bit_array.is_zeros(), bits.iter().all(|&b| !b));
        }
    }

    #[rstest]
    fn test_default_is_ones(mut rng: impl Rng, n_experiments: usize) {
        let bits = vec![true, false, true, true, false];
        let bit_array = BitArrayTest::from_bits(&bits);
        assert!(!bit_array.is_ones());

        let zeros = BitArrayTest::zeros(bits.len());
        assert!(!zeros.is_ones());

        let ones = BitArrayTest::ones(bits.len());
        assert!(ones.is_ones());

        for _ in 0..n_experiments {
            let n = rng.random_range(1..=128);
            let bits: Vec<bool> = (0..n).map(|_| rng.random()).collect();
            let bit_array = BitArrayTest::from_bits(&bits);
            assert_eq!(bit_array.is_ones(), bits.iter().all(|&b| b));
        }
    }
}
