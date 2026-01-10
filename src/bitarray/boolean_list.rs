//! # Boolean List Implementation
//!
//! Provides a straightforward BitArray implementation using a `Vec<bool>` for storage.
//! This implementation prioritizes simplicity and debugging ease over memory efficiency.
//!
//! ## Overview
//!
//! `BoolBitArray` stores each bit as a separate boolean value in a vector, making
//! bit manipulation operations straightforward but using more memory than packed
//! bit representations.
//!
//! ## Characteristics
//!
//! - **Memory usage**: 1 byte per bit (8x overhead compared to packed storage)
//! - **Performance**: Fast individual bit access and modification
//! - **Debugging**: Easy to inspect and understand bit patterns
//! - **Simplicity**: Straightforward implementation of BitArray trait
//!
//! ## Use Cases
//!
//! Best suited for:
//! - Development and debugging where clarity is important
//! - Applications where memory usage is not critical
//! - Frequent individual bit access operations
//!
//! ## Examples
//!
//! ```rust
//! use flexfloat::bitarray::{BitArray, BoolBitArray};
//!
//! let mut bits = BoolBitArray::from_bits(&[true, false, true]);
//! bits[1] = true;  // Direct bit modification
//! assert_eq!(bits[1], true);
//!
//! // Range operations
//! let sub_range = bits.get_range(0..2).unwrap();
//! assert_eq!(sub_range.to_bits(), vec![true, true]);
//! ```

use std::cmp::max;
use std::ops::{Add, Div, Index, IndexMut, Mul, Range, Sub};

use crate::bitarray::BitArray;

/// A BitArray implementation using `Vec<bool>` for bit storage.
///
/// This implementation uses one boolean per bit, providing simple and direct
/// bit manipulation at the cost of memory efficiency. Each bit consumes a full
/// byte of memory.
///
/// # Memory Layout
///
/// Bits are stored in order from index 0 to len()-1, with no packing or compression.
/// This makes debugging and individual bit operations very fast.
///
/// # Thread Safety
///
/// BoolBitArray is not thread-safe by default. Wrap in appropriate synchronization
/// primitives when sharing across threads.

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

    // [7, 1], 10 -> [1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
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

    fn iter_bits(&self) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator {
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

    fn append_bool_in_place(mut self, value: bool) -> Self
    where
        Self: Sized,
    {
        self.bits.push(value);
        self
    }

    fn truncate(mut self, n_bits: usize) -> Self
    where
        Self: Sized,
    {
        self.bits.truncate(n_bits);
        self
    }

    fn shift_grow_with_fill(mut self, shift: isize, fill: bool) -> Self {
        if shift == 0 {
            return self;
        }

        if shift < 0 {
            return self.append_repeated(fill, shift.unsigned_abs());
        }

        let shift_abs = shift as usize;
        self.bits.extend(std::iter::repeat_n(false, shift_abs));

        // Move bits to the right
        for i in (0..self.len() - shift_abs).rev() {
            self.bits[i + shift_abs] = self.bits[i];
        }

        // Fill the beginning with the fill value
        for i in 0..shift_abs {
            self.bits[i] = fill;
        }
        self
    }

    fn clear(mut self) -> Self {
        self.bits.fill(false);
        self
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

impl Add for BoolBitArray {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        // Ensure self is the longer one
        if rhs.len() > self.len() {
            std::mem::swap(&mut self, &mut rhs);
        }

        // Add bit by bit with carry
        let mut carry = 0u8;
        for i in 0..self.len() {
            let a_bit = self.bits[i] as u8;
            let b_bit = rhs.bits.get(i).copied().unwrap_or(false) as u8;

            let sum = a_bit + b_bit + carry;
            self.bits[i] = (sum & 1) == 1;
            carry = (sum >> 1) & 1;
        }

        // Overflow carry -> Grow the array
        if carry > 0 {
            self.bits.push(true);
        }
        self
    }
}

impl Sub for BoolBitArray {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let max_len = max(self.bits.len(), rhs.bits.len());
        let mut bitarray = Self::zeros(max_len);

        // Subtract bit by bit with borrow
        let mut borrow = 0i8;
        for i in 0..max_len {
            let a_bit = self.bits.get(i).copied().unwrap_or(false) as i8;
            let b_bit = rhs.bits.get(i).copied().unwrap_or(false) as i8;

            let diff = a_bit - b_bit - borrow;
            bitarray.bits[i] = (diff & 1) == 1;
            borrow = (diff < 0) as i8;
        }

        // Underflow borrow -> Panic (no negative results supported)
        if borrow > 0 {
            panic!("Underflow in BoolBitArray subtraction");
        }

        bitarray
    }
}

impl Mul for BoolBitArray {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let n = self.len();
        let m = rhs.len();
        let mut result = Self::zeros(n + m);

        for i in 0..n {
            if self.bits[i] {
                // Add rhs shifted left by i positions
                let mut carry = false;
                for j in 0..m {
                    let pos = i + j;
                    let bit = rhs.bits[j];
                    let sum = (result.bits[pos] as u8) + (bit as u8) + (carry as u8);
                    result.bits[pos] = (sum & 1) == 1;
                    carry = sum > 1;
                }
                // Propagate remaining carry
                if carry {
                    let mut pos = i + m;
                    while pos < result.len() && carry {
                        let sum = (result.bits[pos] as u8) + 1;
                        result.bits[pos] = (sum & 1) == 1;
                        carry = sum > 1;
                        pos += 1;
                    }
                }
            }
        }

        result
    }
}

/// Helper function for div to check if a >= b using bit representation
/// Returns true if a >= b, false otherwise
fn is_gte(a: &[bool], b: &[bool]) -> bool {
    // Compare from most significant bit to least significant
    let max_len = std::cmp::max(a.len(), b.len());

    for i in (0..max_len).rev() {
        let a_bit = a.get(i).copied().unwrap_or(false);
        let b_bit = b.get(i).copied().unwrap_or(false);

        if a_bit && !b_bit {
            return true;
        }
        if !a_bit && b_bit {
            return false;
        }
    }

    // If all bits are equal, a >= b
    true
}

/// Helper function for div to perform in-place subtraction: a -= b
/// Assumes a >= b
fn subtract_in_place(a: &mut [bool], b: &[bool]) {
    let mut borrow = 0i8;
    for (i, a_bit_mut) in a.iter_mut().enumerate() {
        let a_bit = *a_bit_mut as i8;
        let b_bit = b.get(i).copied().unwrap_or(false) as i8;

        let diff = a_bit - b_bit - borrow;
        *a_bit_mut = (diff & 1) == 1;
        borrow = (diff < 0) as i8;
    }
}

impl BoolBitArray {
    /// Computes both quotient and remainder for division.
    ///
    /// Returns a tuple (quotient, remainder) where:
    /// - quotient = self / divisor
    /// - remainder = self % divisor
    ///
    /// # Panics
    /// Panics if divisor is zero.
    pub fn div_rem(mut self, divisor: Self) -> (Self, Self) {
        // Handle division by zero - undefined behavior, panic
        if divisor.bits.iter().all(|&b| !b) {
            panic!("Division by zero in BoolBitArray");
        }

        // Handle dividend being zero
        if self.bits.iter().all(|&b| !b) {
            self.bits.truncate(1);
            return (self.clone(), Self::zeros(1));
        }

        // Binary long division algorithm
        let dividend_bits = self.bits.len();
        let mut quotient = Self::zeros(dividend_bits);
        let mut remainder = Self::zeros(dividend_bits);

        // Process each bit of the dividend from most significant to least significant
        for i in (0..dividend_bits).rev() {
            // Shift remainder left by 1 and bring in the next dividend bit
            remainder.bits.insert(0, self.bits[i]);

            // Trim leading zeros in remainder (but keep at least 1 bit)
            while remainder.bits.len() > 1 && !remainder.bits.last().copied().unwrap() {
                remainder.bits.pop();
            }

            // Check if remainder >= divisor
            if is_gte(&remainder.bits, &divisor.bits) {
                // Perform in-place subtraction: remainder -= divisor
                subtract_in_place(&mut remainder.bits, &divisor.bits);
                quotient.bits[i] = true;
            }
        }

        (quotient, remainder)
    }
}

impl Div for BoolBitArray {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.div_rem(rhs).0
    }
}

#[cfg(test)]
mod tests {
    use core::f64;
    use std::cmp::Ordering;
    use std::iter::repeat_n;

    use num_bigint::{BigInt, BigUint};
    use num_traits::identities::Zero;
    use rand::Rng;
    use rstest::rstest;

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
        let bytes = vec![0b10111101, 0b00110010];
        let expected_bits = vec![
            true, false, true, true, true, true, false, true, false, true,
        ];
        let bit_array = BoolBitArray::from_bytes(&bytes, 10);
        assert_eq!(bit_array.bits, expected_bits);

        for _ in 0..n_experiments {
            let len = rng.random_range(1..100);
            let n_bits = rng.random_range(1..100);
            let bool_string = random_bits_string(&mut rng, len);

            let bytes = string_to_bytes(&bool_string);
            let mut expected_bits = string_to_bits(&bool_string);

            let bit_array = BoolBitArray::from_bytes(&bytes, n_bits);

            let expected_bits = if n_bits <= len {
                Vec::from_iter(expected_bits.get(0..n_bits).unwrap().to_owned())
            } else {
                expected_bits.extend(repeat_n(false, n_bits - len));
                expected_bits
            };
            assert_eq!(expected_bits, bit_array.bits, "{len} {n_bits}");
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
            let len = rng.random_range(1..10_000);
            let bit_array = BoolBitArray::ones(len);
            assert!(bit_array.bits.into_iter().all(|b| b));
        }
    }

    fn test_from_float(mut rng: impl Rng, n_experiments: usize) {
        let float = f64::consts::PI;
        let bit_array = BoolBitArray::from_f64(float);
        assert_eq!(bit_array.bits.len(), 64);
        assert_eq!(f64_to_bits(float), bit_array.bits);

        for _ in 0..n_experiments {
            let float = random_f64(&mut rng);
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
            let float = random_f64(&mut rng);
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

    fn rstrip_zeros(mut bytes: Vec<u8>) -> Vec<u8> {
        if let Some(pos) = bytes.iter().rposition(|&b| b != 0) {
            bytes.truncate(pos + 1);
        } else {
            bytes.clear();
        }
        bytes
    }

    fn test_from_biguint(mut rng: impl Rng, n_experiments: usize) {
        // Test with a known value
        let biguint = BigUint::from(0b11110000u8);
        let bit_array = BoolBitArray::from_biguint(&biguint);
        let expected_bits = vec![false, false, false, false, true, true, true, true];
        assert_eq!(bit_array.bits, expected_bits);

        // Test with zero
        let biguint = BigUint::from(0u8);
        let bit_array = BoolBitArray::from_biguint(&biguint);
        assert_eq!(bit_array.bits, vec![]);

        // Test with larger known values
        let biguint = BigUint::from(0x1234u16);
        let bit_array = BoolBitArray::from_biguint(&biguint);
        // 0x1234 = 0001001000110100 in binary (MSB first)
        // But stored as LSB first: [0,0,1,0,1,1,0,0,0,1,0,0,1,0,0,0]
        let expected_bits = vec![
            false, false, true, false, true, true, false, false, // 0x34 = 52
            false, true, false, false, true, // 0x12 = 18
        ];
        assert_eq!(bit_array.bits, expected_bits);

        for _ in 0..n_experiments {
            let n_bits = rng.random_range(1..100);
            let biguint = random_biguint(&mut rng, n_bits);
            let bit_array = BoolBitArray::from_biguint_fixed(&biguint, n_bits);

            // Verify by checking the bytes directly
            let expected_bytes = rstrip_zeros(biguint.to_bytes_le());
            let actual_bytes = rstrip_zeros(bit_array.to_bytes());

            assert_eq!(actual_bytes, expected_bytes);
        }
    }

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
        let empty_array = BoolBitArray::zeros(0);
        let result_true = empty_array.clone().append_bool_in_place(true);
        assert_eq!(result_true.len(), 1);
        assert_eq!(result_true.to_bits(), vec![true]);

        // Test multiple appends
        let mut bit_array = BoolBitArray::from_bits(&[true, false]);
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
            let mut bit_array = BoolBitArray::from_bits(&original_bits);
            let n_extra_bits = rng.random_range(1..20);

            for _ in 0..n_extra_bits {
                let append_value = rng.random_bool(0.5);
                let mut new_bit_array = bit_array.clone().append_bool_in_place(append_value);
                original_bits.push(append_value);
                std::mem::swap(&mut bit_array, &mut new_bit_array);
            }
            assert_eq!(bit_array.to_bits(), original_bits);
        }
    }

    #[rstest]
    fn test_shift_with_bool(mut rng: impl Rng, n_experiments: usize) {
        // Test shift by 0 (no change)
        let original_bits = vec![true, false, true, true, false];
        let bit_array = BoolBitArray::from_bits(&original_bits);
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
        let empty_array = BoolBitArray::zeros(0);
        let result = empty_array.clone().shift_fixed_with_fill(5, true);
        assert_eq!(result.len(), 0);
        let result = empty_array.shift_fixed_with_fill(-5, false);
        assert_eq!(result.len(), 0);

        // Random tests with boolean fill values
        for _ in 0..n_experiments {
            let len = rng.random_range(1..20);
            let original_bits = random_bits(&mut rng, len);
            let bit_array = BoolBitArray::from_bits(&original_bits);
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
        let a = BoolBitArray::from_bits(&[true, true, false]); // 3
        let b = BoolBitArray::from_bits(&[true, false, true]); // 5
        let result = a + b;
        let expected_value = 8u32;
        assert_eq!(result.to_biguint(), BigUint::from(expected_value));

        // 0 + 0 = 0
        let a = BoolBitArray::zeros(1);
        let b = BoolBitArray::zeros(1);
        let result = a + b;
        assert_eq!(result.to_biguint(), BigUint::from(0u8));

        // 7 + 1 = 8 (with carry, requires extra bit)
        let a = BoolBitArray::from_bits(&[true, true, true]); // 7
        let b = BoolBitArray::from_bits(&[true]); // 1
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

            let a = BoolBitArray::from_biguint(&a_uint);
            let b = BoolBitArray::from_biguint(&b_uint);
            let len_a_actual = a.len();
            let len_b_actual = b.len();
            let result = a + b;

            let expected = &a_uint + &b_uint;
            assert_eq!(result.to_biguint(), expected);

            // Verify array grew to accommodate the result
            let expected_bits = expected.bits() as usize;
            let max_len = std::cmp::max(len_a_actual, len_b_actual);
            assert!(result.len() >= std::cmp::max(max_len, expected_bits));
        }
    }

    #[rstest]
    fn test_sub(mut rng: impl Rng, n_experiments: usize) {
        // Test simple positive subtraction cases
        // 7 - 3 = 4
        let a = BoolBitArray::from_bits(&[true, true, true, false]); // 7
        let b = BoolBitArray::from_bits(&[true, true, false, false]); // 3
        let result = a - b;
        assert_eq!(result.to_biguint(), BigUint::from(4u8));

        // 5 - 5 = 0
        let a = BoolBitArray::from_bits(&[true, false, true]); // 5
        let b = BoolBitArray::from_bits(&[true, false, true]); // 5
        let result = a - b;
        assert_eq!(result.to_biguint(), BigUint::from(0u8));

        // Larger subtraction: 15 - 1 = 14
        let a = BoolBitArray::from_bits(&[true, true, true, true]); // 15
        let b = BoolBitArray::from_bits(&[true]); // 1
        let result = a - b;
        assert_eq!(result.to_biguint(), BigUint::from(14u8));

        // Test bigger lhs
        let a = BoolBitArray::from_bits(&[true, true]);
        let b = BoolBitArray::from_bits(&[true, false, false]);
        let result = a - b;
        let expected = BoolBitArray::from_bits(&[false, true, false]);
        assert_eq!(result.bits, expected.bits);

        for _ in 0..n_experiments {
            let len_a = rng.random_range(1..20);
            let len_b = rng.random_range(1..20);
            let mut a_uint = random_biguint(&mut rng, len_a);
            let mut b_uint = random_biguint(&mut rng, len_b);

            // Ensure a >= b to avoid underflow
            if a_uint < b_uint {
                std::mem::swap(&mut a_uint, &mut b_uint);
            }

            let a = BoolBitArray::from_biguint(&a_uint);
            let b = BoolBitArray::from_biguint(&b_uint);
            let result = a - b;

            let expected = &a_uint - &b_uint;
            assert_eq!(result.to_biguint(), expected);
        }
    }

    #[rstest]
    fn test_mul(mut rng: impl Rng, n_experiments: usize) {
        // Test simple multiplication cases
        // 3 * 5 = 15
        let a = BoolBitArray::from_bits(&[true, true, false]); // 3
        let b = BoolBitArray::from_bits(&[true, false, true]); // 5
        let result = a * b;
        assert_eq!(result.to_biguint(), BigUint::from(15u8));

        // 0 * 7 = 0
        let a = BoolBitArray::zeros(1); // 0
        let b = BoolBitArray::from_bits(&[true, true, true]); // 7
        let result = a * b;
        assert_eq!(result.to_biguint(), BigUint::from(0u8));

        // 4 * 2 = 8
        let a = BoolBitArray::from_bits(&[false, false, true]); // 4
        let b = BoolBitArray::from_bits(&[false, true]); // 2
        let result = a * b;
        assert_eq!(result.to_biguint(), BigUint::from(8u8));

        for _ in 0..n_experiments {
            let len_a = rng.random_range(1..100);
            let len_b = rng.random_range(1..100);
            let a_uint = random_biguint(&mut rng, len_a);
            let b_uint = random_biguint(&mut rng, len_b);

            let a = BoolBitArray::from_biguint(&a_uint);
            let b = BoolBitArray::from_biguint(&b_uint);
            let result = a * b;

            let expected = &a_uint * &b_uint;
            assert_eq!(result.to_biguint(), expected);
        }
    }

    #[rstest]
    fn test_div(mut rng: impl Rng, n_experiments: usize) {
        // Test simple division cases
        // 4 / 2 = 2
        let a = BoolBitArray::from_bits(&[false, false, true]); // 4
        let b = BoolBitArray::from_bits(&[false, true]); // 2
        let result = a / b;
        assert_eq!(result.to_biguint(), BigUint::from(2u8));

        // 15 / 3 = 5
        let a = BoolBitArray::from_bits(&[true, true, true, true]); // 15
        let b = BoolBitArray::from_bits(&[true, true, false]); // 3
        let result = a / b;
        assert_eq!(result.to_biguint(), BigUint::from(5u8));

        // 1 / 2 = 0
        let a = BoolBitArray::from_bits(&[true]); // 1
        let b = BoolBitArray::from_bits(&[false, true]); // 2
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

            let a = BoolBitArray::from_biguint(&a_uint);
            let b = BoolBitArray::from_biguint(&b_uint);
            let result = a / b;

            let expected = &a_uint / &b_uint;
            assert_eq!(result.to_biguint(), expected);
        }
    }
}
