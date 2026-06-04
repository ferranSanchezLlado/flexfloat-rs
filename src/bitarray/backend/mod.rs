//! Backend-author surface for `BitArray` implementors.
//!
//! This module is intended for authors writing new `BitArray` backends
//! (e.g., SIMD-backed, rope-backed, or arena-allocated bit arrays).
//! Most library users should never need to import anything from here.
//!
//! Implementing [`BitArrayPrimitives`] for your type lets the default
//! implementations of [`BitArrayManipulation`] work without falling back
//! to slow `Vec<bool>` round-trips.
//!
//! # Stability
//!
//! This module is `pub` so third-party crates can implement backends, but
//! it is `#[doc(hidden)]` to keep it off the headline API surface. Its
//! contents will be promoted to a stable extension point when `bitarray`
//! is split into its own crate at 1.0.

#[doc(hidden)]
pub mod primitives;

#[doc(hidden)]
pub use primitives::BitArrayPrimitives;
