//! # Mathematical Functions for FlexFloat
//!
//! This module provides a comprehensive collection of mathematical functions for FlexFloat,
//! organized into submodules by category. It offers a static interface similar to Python's math module,
//! allowing functions to be called both as static functions and as methods.
//!
//! ## Modules
//!
//! - [`rounding`]: Round, floor, ceil, trunc, and fract operations
//! - [`exponential`]: Exponential and logarithmic functions (exp, ln, log, log2, log10)
//! - [`power`]: Power and root functions (pow, sqrt, cbrt)
//! - [`trigonometry`]: Trigonometric functions (sin, cos, tan, asin, acos, atan, atan2)
//! - [`hyperbolic`]: Hyperbolic functions (sinh, cosh, tanh)
//! - [`utils`]: Utility functions (signum, copysign, recip, to_degrees, to_radians)
//!
//! ## Usage
//!
//! Functions can be used in two ways:
//!
//! ### As static functions:
//! ```rust
//! use flexfloat::prelude::*;
//! use flexfloat::math;
//!
//! let x = FlexFloat::from(2.7);
//! let rounded = math::round(x.clone());
//! let exp_result = math::exp(x.clone());
//! let ln_result = math::ln(x);
//! ```
//!
//! ### As methods:
//! ```rust
//! use flexfloat::prelude::*;
//!
//! let x = FlexFloat::from(2.7);
//! let rounded = x.clone().round();
//! let exp_result = x.clone().exp();
//! let ln_result = x.ln();
//! ```
//!
//! ## Implementation Notes
//!
//! Mathematical functions automatically handle exponent growth when results exceed
//! the current exponent range, maintaining FlexFloat's arbitrary-range capability.

// Submodules for different categories of mathematical functions
pub mod exponential;
pub mod hyperbolic;
pub mod power;
pub mod rounding;
pub mod trigonometry;
pub mod utils;

// Re-export all functions for convenient access (Python math-like interface)
pub use exponential::{exp, exp_m1, exp2, ln, ln_1p, log, log2, log10};
pub use hyperbolic::{acosh, asinh, atanh, cosh, sinh, tanh};
pub use power::{cbrt, hypot, pow, sqrt};
pub use rounding::{ceil, floor, fract, round, round_ties_even, trunc};
pub use trigonometry::{acos, asin, atan, atan2, cos, sin, sin_cos, tan};
pub use utils::{copysign, mul_add, recip, signum, to_degrees, to_radians};
