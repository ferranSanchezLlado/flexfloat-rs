//! # BitArray Module
//!
//! Provides flexible bit manipulation and storage abstractions for FlexFloat operations.
//! The module defines a common interface for bit arrays with various backing implementations.
//!
//! ## Overview
//!
//! The BitArray trait provides a unified interface for:
//! - **Bit storage**: Efficient storage of arbitrary-length bit sequences
//! - **Type conversion**: Seamless conversion between numeric types and bit representations
//! - **Bit manipulation**: Common operations like shifting, range extraction, and iteration
//! - **Endianness handling**: Little-endian byte order for consistency
//!
//! ## Key Features
//!
//! - **Multiple implementations**: Currently supports boolean vector backing with room for optimization
//! - **Type safety**: Strong typing prevents common bit manipulation errors
//! - **Conversion utilities**: Built-in support for f64, BigUint, BigInt, and byte arrays
//! - **Memory efficiency**: Compact representation with configurable bit lengths
//!
//! ## Usage Examples
//!
//! ```rust
//! use flexfloat::bitarray::{BitArray, BoolBitArray};
//!
//! // Create from various sources
//! let from_bits = BoolBitArray::from_bits(&[true, false, true]);
//! let from_bytes = BoolBitArray::from_bytes(&[0xAB], 8);
//! let from_f64 = BoolBitArray::from_f64(3.14159);
//!
//! // Access and manipulate bits
//! let bit_3 = from_bits[2]; // true
//! let range = from_bits.get_range(0..2).unwrap();
//!
//! // Convert back to other types
//! let bytes = from_bits.to_bytes();
//! let bits = from_bits.to_bits();
//! ```

use num_bigint::{BigInt, BigUint};
use std::cmp::Ordering;
use std::hint::unreachable_unchecked;
use std::iter::repeat_n;
use std::ops::{Index, IndexMut, Range};

pub mod boolean_list;

pub use boolean_list::BoolBitArray;

/// Default BitArray implementation for general use
pub type DefaultBitArray = BoolBitArray;

/// Core trait for bit array implementations with little-endian byte ordering.
///
/// Provides a unified interface for storing and manipulating sequences of bits
/// with support for various numeric type conversions and bit operations.
///
/// ## Byte Order
///
/// All implementations use **little-endian (LE)** byte order for consistency
/// with common CPU architectures and IEEE 754 standards.
///
/// ## Implementation Requirements
///
/// Implementors must provide efficient storage and access for bit sequences,
/// with particular attention to memory usage and performance for common operations
/// like indexing and range extraction.
pub trait BitArray {
    // === Construction Methods ===

    /// Creates a new BitArray from a byte array with specified bit length.
    ///
    /// Interprets bytes in little-endian order, where the least significant bit
    /// of the first byte becomes bit 0 of the array.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Source byte array in little-endian format
    /// * `n_bits` - Total number of bits to extract (may be less than bytes.len() * 8)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::bitarray::{BitArray, BoolBitArray};
    ///
    /// // 0xAB = 10101011 in binary -> [1,1,0,1,0,1,0,1] in LE bit order
    /// let bits = BoolBitArray::from_bytes(&[0xAB], 8);
    /// assert_eq!(bits[0], true);  // LSB
    /// assert_eq!(bits[7], true);  // MSB
    /// ```
    fn from_bytes(bytes: &[u8], n_bits: usize) -> Self
    where
        Self: Sized;

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
    /// use flexfloat::bitarray::{BitArray, BoolBitArray};
    ///
    /// let bits = BoolBitArray::from_bits(&[true, false, true]);
    /// assert_eq!(bits.len(), 3);
    /// assert_eq!(bits[0], true);
    /// ```
    fn from_bits(bits: &[bool]) -> Self
    where
        Self: Sized,
    {
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
    /// use flexfloat::bitarray::{BitArray, BoolBitArray};
    ///
    /// let bits = BoolBitArray::from_f64(1.0);
    /// assert_eq!(bits.len(), 64);
    /// // Bit 63 is sign, bits 62-52 are exponent, bits 51-0 are mantissa
    /// ```
    fn from_f64(value: f64) -> Self
    where
        Self: Sized,
    {
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
    /// use flexfloat::bitarray::{BitArray, BoolBitArray};
    /// use num_bigint::BigUint;
    ///
    /// let big_num = BigUint::from(255u32);
    /// let bits = BoolBitArray::from_biguint(&big_num);
    /// ```
    fn from_biguint(value: &BigUint) -> Self
    where
        Self: Sized,
    {
        let bytes = value.to_bytes_le();
        let n_bits = value.bits() as usize;
        Self::from_bytes(&bytes, n_bits)
    }

    fn from_biguint_fixed(value: &BigUint, n_bits: usize) -> Self
    where
        Self: Sized,
    {
        let bytes = value.to_bytes_le();
        Self::from_bytes(&bytes, n_bits)
    }

    /// Creates a BitArray from a signed BigInt with specified bit width.
    ///
    /// Uses two's complement representation for negative numbers.
    /// Returns None if the value cannot fit in the specified bit width.
    ///
    /// # Arguments
    ///
    /// * `value` - The BigInt to convert
    /// * `n_bits` - Target bit width (must accommodate the value)
    ///
    /// # Returns
    ///
    /// Some(BitArray) if the value fits, None otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use flexfloat::bitarray::{BitArray, BoolBitArray};
    /// use num_bigint::BigInt;
    ///
    /// let num = BigInt::from(-5);
    /// let bits = BoolBitArray::from_bigint(&num, 8).unwrap();
    /// ```
    fn from_bigint(value: &BigInt, n_bits: usize) -> Option<Self>
    where
        Self: Sized,
    {
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
    /// use flexfloat::bitarray::{BitArray, BoolBitArray};
    ///
    /// let zeros = BoolBitArray::zeros(8);
    /// assert!(zeros.iter_bits().all(|b| !b));
    /// ```
    fn zeros(n_bits: usize) -> Self
    where
        Self: Sized,
    {
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
    /// use flexfloat::bitarray::{BitArray, BoolBitArray};
    ///
    /// let ones = BoolBitArray::ones(8);
    /// assert!(ones.iter_bits().all(|b| b));
    /// ```
    fn ones(n_bits: usize) -> Self
    where
        Self: Sized,
    {
        let n_bytes = n_bits.div_ceil(8);
        let mut bytes = vec![0xffu8; n_bytes];

        let last_num_bits = n_bits % 8;
        if last_num_bits > 0 {
            bytes[n_bytes - 1] = (1 << last_num_bits) - 1;
        }

        Self::from_bytes(&bytes, n_bits)
    }

    // === Conversion Methods ===

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
    /// use flexfloat::bitarray::{BitArray, BoolBitArray};
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
    /// use flexfloat::bitarray::{BitArray, BoolBitArray};
    ///
    /// let bits = BoolBitArray::from_bits(&[true, false, true]);
    /// assert_eq!(bits.to_bits_string(), "101");
    /// ```
    fn to_bits_string(&self) -> String {
        self.iter_bits()
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

    /// Converts the BitArray to a signed BigInt using two's complement.
    ///
    /// # Returns
    ///
    /// BigInt representation of the bit sequence as a signed number
    fn to_bigint(&self) -> BigInt {
        let n_bits = self.len();
        let half = BigInt::from(1u8) << (n_bits - 1);
        let unsigned = BigInt::from(self.to_biguint());
        unsigned - half
    }

    // === Access Methods ===

    /// Returns an iterator over the bits in the array.
    ///
    /// # Returns
    ///
    /// Iterator yielding each bit as a boolean value
    fn iter_bits(&self) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator;

    /// Returns the number of bits in the array.
    ///
    /// # Returns
    ///
    /// Total number of bits stored
    fn len(&self) -> usize;

    /// Gets a reference to the bit at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - Zero-based bit index
    ///
    /// # Returns
    ///
    /// Some(&bool) if index is valid, None otherwise
    fn get(&self, index: usize) -> Option<&bool>;

    /// Gets a mutable reference to the bit at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - Zero-based bit index
    ///
    /// # Returns
    ///
    /// Some(&mut bool) if index is valid, None otherwise
    fn get_mut(&mut self, index: usize) -> Option<&mut bool>;

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
    /// use flexfloat::bitarray::{BitArray, BoolBitArray};
    ///
    /// let bits = BoolBitArray::from_bits(&[true, false, true, false]);
    /// let sub = bits.get_range(1..3).unwrap();
    /// assert_eq!(sub.to_bits(), vec![false, true]);
    /// ```
    fn get_range(&self, range: Range<usize>) -> Option<Self>
    where
        Self: Sized,
    {
        if range.end > self.len() || range.start > range.end {
            return None;
        }
        let bits: Vec<bool> = range.map(|i| *self.get(i).unwrap()).collect();
        Some(Self::from_bits(&bits))
    }

    /// Checks if the BitArray is empty (has zero bits).
    ///
    /// # Returns
    ///
    /// true if the array has no bits, false otherwise
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // === Manipulation Methods ===

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
    /// use flexfloat::bitarray::{BitArray, BoolBitArray};
    ///
    /// let bits = BoolBitArray::from_bits(&[true, false]);
    /// let extended = bits.append_bool_in_place(true);
    /// assert_eq!(extended.to_bits(), vec![true, false, true]);
    /// ```
    fn append_bool_in_place(self, value: bool) -> Self
    where
        Self: Sized,
    {
        let mut bits = self.to_bits();
        bits.push(value);
        Self::from_bits(&bits)
    }

    /// Shifts the BitArray by the specified number of positions with fill value.
    ///
    /// Positive shift moves bits to the right (towards higher indices),
    /// negative shift moves bits to the left (towards lower indices).
    /// Vacated positions are filled with the specified fill value.
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
    /// use flexfloat::bitarray::{BitArray, BoolBitArray};
    ///
    /// let bits = BoolBitArray::from_bits(&[true, false, true]);
    /// let shifted = bits.shift_with_fill(1, false);
    /// assert_eq!(shifted.to_bits(), vec![false, true, false]);
    /// ```
    fn shift_with_fill(self, shift: isize, fill: bool) -> Self
    where
        Self: Sized,
    {
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
    /// * `shift` - Number of positions to shift
    ///
    /// # Returns
    ///
    /// New BitArray with bits shifted and zero-filled
    fn shift(self, shift: isize) -> Self
    where
        Self: Sized,
    {
        self.shift_with_fill(shift, false)
    }

    fn append_repeated(self, value: bool, count: usize) -> Self
    where
        Self: Sized,
    {
        if count == 0 {
            return self;
        }
        let mut bits = self.to_bits();
        bits.extend(repeat_n(value, count));
        Self::from_bits(&bits)
    }
}

macro_rules! impl_index {
    ($t:ty) => {
        impl Index<usize> for $t {
            type Output = bool;

            fn index(&self, index: usize) -> &Self::Output {
                self.get(index).expect("Index out of bounds")
            }
        }

        impl IndexMut<usize> for $t {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                self.get_mut(index).expect("Index out of bounds")
            }
        }
    };
    () => {};
}

impl_index!(BoolBitArray);

#[cfg(test)]
mod tests {
    use num_bigint::{BigInt, BigUint};
    pub use rand::Rng;

    pub fn random_bits(rng: &mut impl Rng, len: usize) -> Vec<bool> {
        (0..len).map(|_| rng.random_bool(0.5)).collect()
    }

    pub fn random_bytes(rng: &mut impl Rng, len: usize) -> Vec<u8> {
        (0..len).map(|_| rng.random()).collect()
    }

    pub fn random_bits_string(rng: &mut impl Rng, len: usize) -> String {
        (0..len)
            .map(|_| if rng.random_bool(0.5) { '1' } else { '0' })
            .collect()
    }

    pub fn random_biguint(rng: &mut impl Rng, n_bits: usize) -> BigUint {
        let n_bytes = n_bits.div_ceil(8);
        let mut bytes = vec![0u8; n_bytes];
        rng.fill(&mut bytes[..]);
        let last_num_bits = n_bits % 8;
        if last_num_bits > 0 {
            bytes[n_bytes - 1] &= (1 << last_num_bits) - 1;
        }
        BigUint::from_bytes_le(&bytes)
    }

    pub fn random_bigint(rng: &mut impl Rng, n_bits: usize) -> BigInt {
        let uint = random_biguint(rng, n_bits - 1);
        if rng.random_bool(0.5) {
            BigInt::from(uint)
        } else {
            -BigInt::from(uint)
        }
    }

    pub fn string_to_bits(s: &str) -> Vec<bool> {
        s.chars().map(|c| c == '1').collect()
    }

    pub fn string_to_bytes(s: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        for bits in s.as_bytes().chunks(8) {
            let string = std::str::from_utf8(bits).unwrap();
            let mut byte = u8::from_str_radix(string, 2).unwrap();
            byte = byte.reverse_bits() >> (8 - bits.len());
            bytes.push(byte);
        }
        bytes
    }
}
