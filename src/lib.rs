//! # FlexFloat
//!
//! A high-precision library for arbitrary precision floating-point arithmetic with growable exponents
//! and fixed-size fractions. FlexFloat extends IEEE 754 double-precision format to handle numbers
//! beyond the standard range while maintaining computational efficiency and precision consistency.
//!
//! ## Overview
//!
//! FlexFloat provides:
//! - **Growable exponents**: Automatically expand exponent bit width when needed
//! - **Fixed-size fractions**: Maintain consistent precision (52-bit mantissa by default)
//! - **IEEE 754 compatibility**: Full support for standard floating-point operations
//! - **Arbitrary precision**: Handle numbers far beyond standard double-precision range
//! - **Efficient operations**: Optimized for performance while maintaining precision
//!
//! ## Architecture
//!
//! The library is built around two main components:
//!
//! ### BitArray Module
//! Provides flexible bit manipulation and storage with support for various backing implementations:
//! - Boolean vector-based storage for simplicity and debugging
//! - Extensible to other bit storage formats
//! - Conversion utilities for common numeric types
//!
//! ### FlexFloat Module
//! Implements the core floating-point type with:
//! - Sign bit (1 bit)
//! - Variable-width exponent (starts at 11 bits, grows as needed)
//! - Fixed-width fraction (52 bits for IEEE 754 compatibility)
//!
//! ## Quick Start
//!
//! ```rust
//! use flexfloat::FlexFloat;
//!
//! // Create from standard f64
//! let x = FlexFloat::from(3.14159);
//! let y = FlexFloat::from(2.71828);
//!
//! // Perform arithmetic operations
//! let neg_x = -x;
//! let abs_y = y.abs();
//!
//! // Convert back to f64 (when in range)
//! let result: f64 = neg_x.into();
//! ```
//!
//! ## Special Values
//!
//! FlexFloat supports all IEEE 754 special values:
//!
//! ```rust
//! use flexfloat::FlexFloat;
//!
//! let zero = FlexFloat::zero();
//! let pos_inf = FlexFloat::pos_infinity();
//! let neg_inf = FlexFloat::neg_infinity();
//! let nan = FlexFloat::nan();
//!
//! assert!(zero.is_zero());
//! assert!(pos_inf.is_infinity());
//! assert!(nan.is_nan());
//! ```

pub mod bitarray;
pub mod flexfloat;

// Re-export the main types for convenience
pub use bitarray::{BitArray, BoolBitArray, DefaultBitArray};
pub use flexfloat::FlexFloat;

pub mod prelude {
    //! Prelude module for FlexFloat.
    //!
    //! This module re-exports commonly used types and traits from the FlexFloat crate,
    //! allowing for easier imports in user code.

    pub use crate::bitarray::*;
    pub use crate::flexfloat::FlexFloat;
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use num_bigint::{BigInt, BigUint};
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rstest::fixture;

    const EPSILON: f64 = 1e-10;
    static SEED: OnceLock<u64> = OnceLock::new();

    #[fixture]
    pub const fn n_experiments() -> usize {
        100_000
    }

    #[fixture]
    pub fn seed() -> u64 {
        *SEED.get_or_init(|| rand::rng().random())
    }

    #[fixture]
    pub fn rng(n_experiments: usize, seed: u64) -> impl Rng {
        println!("{} experiments with seed {}", n_experiments, seed);
        StdRng::seed_from_u64(seed)
    }

    #[track_caller]
    pub fn assert_almost_eq(a: f64, b: f64, message: &str) {
        let diff = (a - b).abs();
        assert!(
            diff <= EPSILON,
            "{}: {} and {} differ by {}, which is more than {}",
            message,
            a,
            b,
            diff,
            EPSILON
        );
    }

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
