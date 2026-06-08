//! # FlexFloat
//!
//! A high-precision library for arbitrary-precision floating-point arithmetic with growable
//! exponents and fixed-size fractions.  FlexFloat extends IEEE 754 double-precision format
//! to handle numbers far beyond the standard range while maintaining computational efficiency
//! and precision consistency.
//!
//! ## Quick start
//!
//! ```rust
//! use flexfloat::prelude::*;
//!
//! let radius = FlexFloat::from(3.0);
//! let circumference = radius * FlexFloat::from(core::f64::consts::TAU);
//!
//! assert_ff_almost_eq!(circumference, FlexFloat::from(18.84955592153876));
//! ```
//!
//! ## Type alias
//!
//! For ergonomic use, the crate root re-exports:
//!
//! ```rust
//! // Equivalent to crate::flexfloat::FlexFloat<crate::bitarray::DefaultBitArray>
//! use flexfloat::FlexFloat;
//! ```
//!
//! The full generic type `flexfloat::flexfloat::FlexFloat<Exp, Frac>` is available for
//! custom-backend users.
//!
//! ## Special values
//!
//! ```rust
//! use flexfloat::prelude::*;
//!
//! let zero = FlexFloat::zero();
//! let pos_inf = FlexFloat::pos_infinity();
//! let neg_inf = FlexFloat::neg_infinity();
//! let nan = FlexFloat::nan();
//!
//! assert!(zero.is_zero());
//! assert!(pos_inf.is_infinite());
//! assert!(nan.is_nan());
//! ```
//!
//! ## Arithmetic
//!
//! FlexFloat supports all standard arithmetic operators (`+`, `-`, `*`, `/`, `%`) plus
//! `rem_euclid`, `div_euclid`, `powi`, `powf`, and `mul_add`.  Exponents grow automatically
//! when a result would overflow the current width:
//!
//! ```rust
//! use flexfloat::prelude::*;
//!
//! let huge = FlexFloat::from(f64::MAX) * FlexFloat::from(f64::MAX);
//! assert!(!huge.is_infinite()); // exponent grew instead of overflowing
//! assert!(huge.exponent_bits() > 11);
//! ```
//!
//! ## Math functions
//!
//! All standard transcendental functions are available through the [`math`] module and as
//! methods:
//!
//! | Function     | Method                                              | Free fn          |
//! |--------------|-----------------------------------------------------|------------------|
//! | Exponential  | `.exp()`                                            | `math::exp`      |
//! | Natural log  | `.ln()`                                             | `math::ln`       |
//! | Square root  | `.sqrt()`                                           | `math::sqrt`     |
//! | Trigonometry | `.sin()`, `.cos()`, `.tan()`, …                     | `math::sin`, …   |
//! | Hyperbolic   | `.sinh()`, `.cosh()`, `.tanh()`, …                  | `math::sinh`, …  |
//! | Rounding     | `.round()`, `.floor()`, `.ceil()`, `.round_ties_even()` | `math::round`, … |
//! | Simultaneous | `.sin_cos()`                                        | `math::sin_cos`  |
//!
//! ## Conversions
//!
//! ### Into FlexFloat (lossless `From`)
//!
//! ```rust
//! use flexfloat::FlexFloat;
//! use num_bigint::BigInt;
//!
//! let _ = FlexFloat::from(1.5_f64);
//! let _ = FlexFloat::from(1.0_f32);
//! let _ = FlexFloat::from(42_i64);
//! let _ = FlexFloat::from(42_u64);
//! let _ = FlexFloat::from(42_i32);
//! let _ = FlexFloat::from(42_u32);
//! let _ = FlexFloat::from(BigInt::from(12345));
//! ```
//!
//! ### Out of FlexFloat (fallible `TryFrom`)
//!
//! ```rust
//! use flexfloat::FlexFloat;
//! use flexfloat::FlexFloatToF64Error;
//!
//! let x = FlexFloat::from(1.5_f64);
//! let f: Result<f64, _> = x.try_into();
//! assert_eq!(f, Ok(1.5_f64));
//! ```
//!
//! ## Const-context constants
//!
//! Math constants like `PI`, `E`, `TAU` etc. in [`flexfloat::consts`] are typed as
//! `FlexFloat<StaticBitArray<11>, StaticBitArray<52>>` — a zero-overhead const-generic
//! type — and can be used directly in arithmetic expressions without `.convert_to()`:
//!
//! ```rust
//! use flexfloat::prelude::*;
//!
//! let pi = FlexFloat::from(core::f64::consts::PI);
//! // consts::PI is FlexFloat<StaticBitArray<11>, StaticBitArray<52>> and works
//! // transparently as an RHS operand via the mixed-backend arithmetic impls.
//! ```
//!
//! ## Grown-aware instance methods
//!
//! Because FlexFloat has no meaningful compile-time `MIN`/`MAX`/`EPSILON` constants
//! (the range depends on the current exponent width), runtime methods are provided instead:
//!
//! ```rust
//! use flexfloat::prelude::*;
//!
//! let x = FlexFloat::from(1.0);
//! assert_eq!(x.exponent_bits(), 11);        // standard IEEE 754 width
//! assert_eq!(x.mantissa_digits(), 53);       // 52 fraction bits + 1 implicit
//! assert_eq!(x.epsilon(), FlexFloat::from(f64::EPSILON));
//! ```
//!
//! ## Module map
//!
//! | Module                  | Contents                                                                          |
//! |-------------------------|-----------------------------------------------------------------------------------|
//! | [`bitarray`]            | Pluggable bit-array backends (`BoolBitArray`, `UsizeBitArray`, `StaticBitArray<N>`) |
//! | [`flexfloat`]           | Core `FlexFloat<Exp, Frac>` struct and all operation impls                        |
//! | [`flexfloat::math`]     | Transcendental / rounding functions                                               |
//! | [`flexfloat::consts`]   | Compile-time math constants (`PI`, `E`, `TAU`, …)                                 |
//! | [`flexfloat::error`]    | `FlexFloatToF64Error`, `FlexFloatToIntError`, `ConversionError`                   |

pub mod bitarray;
pub mod flexfloat;

// Re-export the main types for convenience
pub use bitarray::{BitArray, BitArrayArith, BoolBitArray, DefaultBitArray};
/// Ergonomic alias for `FlexFloat` using the default (`BoolBitArray`) backend.
///
/// This is equivalent to `crate::flexfloat::FlexFloat<crate::bitarray::DefaultBitArray>`.
/// All methods defined on `FlexFloat<B: BitArray>` are available without turbofish.
pub type FlexFloat = crate::flexfloat::FlexFloat<crate::bitarray::DefaultBitArray>;
pub use flexfloat::error::{ConversionError, FlexFloatToF64Error, FlexFloatToIntError};
pub use flexfloat::math;

#[doc(hidden)]
pub mod __private {
    use crate::{flexfloat::FlexFloat, prelude::BitArrayArith};

    /// Default relative tolerance used by [`assert_ff_almost_eq`].
    pub const DEFAULT_ASSERT_FF_ALMOST_EQ_TOLERANCE: f64 = 1e-8;

    #[track_caller]
    pub fn assert_almost_eq(result: f64, expected: f64, tolerance: f64, message: &str) {
        let diff = (result - expected).abs() / result.abs().max(expected.abs()).max(1e-10);
        assert!(
            diff <= tolerance,
            "{message}: result={result:?} vs expected={expected:?} ({diff:.2e} > {tolerance:.2e})",
        );
    }

    #[track_caller]
    pub fn assert_ff_almost_eq<E1, F1, E2, F2>(
        result: &FlexFloat<E1, F1>,
        expected: &FlexFloat<E2, F2>,
        tolerance: f64,
        message: &str,
    ) where
        E1: BitArrayArith,
        F1: BitArrayArith,
        E2: BitArrayArith,
        F2: BitArrayArith,
    {
        let diff = (result.clone() - expected).abs()
            / result
                .abs()
                .max(expected.abs().convert_to())
                .max(FlexFloat::from_f64(1e-10));
        assert!(
            diff <= FlexFloat::from(tolerance),
            "{message}: result={result:?} vs expected={expected:?} (relative diff {diff:.2e} > {tolerance:.2e})",
        );
    }
}

/// Asserts that two `FlexFloat` values are approximately equal.
#[macro_export]
macro_rules! assert_ff_almost_eq {
    ($result:expr, $expected:expr $(,)?) => {
        $crate::__private::assert_ff_almost_eq(
            &$result,
            &$expected,
            $crate::__private::DEFAULT_ASSERT_FF_ALMOST_EQ_TOLERANCE,
            concat!("assert_ff_almost_eq! failed at ", file!(), ":", line!()),
        )
    };
    ($result:expr, $expected:expr, $tolerance:expr $(,)?) => {
        $crate::__private::assert_ff_almost_eq(
            &$result,
            &$expected,
            $tolerance,
            concat!("assert_ff_almost_eq! failed at ", file!(), ":", line!()),
        )
    };
}

/// Prelude module for FlexFloat.
///
/// This module re-exports commonly used types and traits from the FlexFloat crate,
/// allowing for easier imports in user code. For `1.x`, these re-exports are
/// part of the supported public API for the default backend and generic
/// `BitArray`-based integrations.
///
/// # Usage
///
/// Instead of importing individual items:
/// ```rust
/// use flexfloat::FlexFloat;
/// use flexfloat::bitarray::{BitArray, DefaultBitArray};
/// ```
///
/// You can import everything at once:
/// ```rust
/// use flexfloat::prelude::*;
/// ```
///
/// # Examples
///
/// ```rust
/// use flexfloat::prelude::*;
///
/// let x = FlexFloat::from(1.0);
/// let bits = DefaultBitArray::from_bits(&[true, false]);
/// assert_ff_almost_eq!(x, FlexFloat::from(1.0));
/// ```
pub mod prelude {
    pub use crate::FlexFloat;
    pub use crate::assert_ff_almost_eq;
    pub use crate::bitarray::traits::{
        BitArray, BitArrayAccess, BitArrayArith, BitArrayConstruction, BitArrayConversion,
        BitArrayManipulation, BitArrayMutAccess, BitArrayRangeAccess,
    };
    pub use crate::bitarray::{BoolBitArray, DefaultBitArray};
}

#[cfg(test)]
pub(crate) mod test_support;
