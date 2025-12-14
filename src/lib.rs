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

    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rstest::fixture;

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
}
