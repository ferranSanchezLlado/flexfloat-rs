//! # BitArray Module
//!
//! A sequence-of-bits type with multiple backends, designed as a reusable
//! bit-array library that `flexfloat` happens to be the first consumer of.
//!
//! ## What this is
//!
//! A dynamically-sized, little-endian sequence of bits backed by a pluggable
//! storage implementation. The module defines a layered trait hierarchy so
//! consumers can depend on only the capability they actually need.
//!
//! ## Trait composition
//!
//! | Trait | Responsibility |
//! |---|---|
//! | [`BitArrayAccess`] | Read bits, query length, extract ranges |
//! | [`BitArrayMutAccess`] | Mutable bit access via GAT handle |
//! | [`BitArrayConstruction`] | Factory methods (`from_bits`, `from_bytes`, …) |
//! | [`BitArrayConversion`] | Serialisation (`to_bits`, `to_bytes`, `to_biguint`, …) |
//! | [`BitArrayManipulation`] | High-level in-place / consuming transforms |
//! | [`BitArray`] | Umbrella: all of the above + `Debug + Clone + Default + PartialEq + Eq` |
//! | [`BitArrayArith`] | Extends `BitArray` with `Add + Sub + Mul + Div` (bignum semantics) |
//!
//! Depend on the smallest set you need. `FlexFloat<B>` bounds most of its
//! API on `B: BitArray`; arithmetic operations additionally require
//! `B: BitArrayArith`.
//!
//! ## Backends
//!
//! - **[`BoolBitArray`]** — `Vec<bool>` storage, one byte per bit. Slow but
//!   easy to inspect; ideal for debugging and unit tests. This is also the
//!   [`DefaultBitArray`].
//! - **[`UsizeBitArray`]** — word-packed storage. Faster for large widths.
//! - **[`StaticBitArray<N>`]** — const-generic, compile-time bit array. Zero
//!   overhead for constants, but read-only and no arithmetic. Ideal for
//!   `const` contexts and `FlexFloat` constant definitions.
//!
//! ## Writing your own backend
//!
//! Implement [`backend::BitArrayPrimitives`] plus the access, construction, and
//! conversion traits; the high-level [`BitArrayManipulation`] default impls
//! will be provided automatically.
//!
//! ## Extension traits
//!
//! [`BitArrayArith`] is optional. A bit-array implementation that only needs to
//! be used as a set or bitmap does not need to provide arithmetic.
//!
//! ## Stability
//!
//! This module will be extracted into a standalone `flexfloat-bitarray` crate
//! at 1.0. Nothing inside `bitarray::*` depends on `flexfloat::*`, so the
//! split will be mechanical. The `backend` submodule contents are `pub` but
//! `#[doc(hidden)]`; they will become a stable extension point in the new crate.

#[doc(hidden)]
pub mod backend;
pub mod bit_ref;
pub mod boolean_list;
pub mod static_bit_array;
pub mod traits;
pub mod usize_list;

pub use bit_ref::BitRef;
pub use boolean_list::BoolBitArray;
pub use static_bit_array::StaticBitArray;
pub use traits::BitArray;
pub use traits::BitArrayArith;
pub use usize_list::UsizeBitArray;

// Re-export trait methods for convenience - users only need to import BitArray
pub use traits::{
    BitArrayAccess, BitArrayConstruction, BitArrayConversion, BitArrayManipulation,
    BitArrayMutAccess,
};

/// Default `BitArray` implementation for general use in `1.x`.
pub type DefaultBitArray = UsizeBitArray;
