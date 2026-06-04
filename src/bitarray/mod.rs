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
//! - **Endianness handling**: Little-endian byte order for consistency with IEEE 754 and common CPU architectures
//!
//! ## Key Features
//!
//! - **Multiple implementations**: Currently supports boolean vector backing with room for optimization
//! - **Type safety**: Strong typing prevents common bit manipulation errors
//! - **Conversion utilities**: Built-in support for f64, BigUint, BigInt, and byte arrays
//! - **Memory efficiency**: Compact representation with configurable bit lengths
//! - **Modular traits**: The BitArray trait is composed of smaller, focused traits
//!
//! ## Trait Organization
//!
//! The BitArray functionality is split into focused traits:
//! - **[`BitArrayConstruction`]** - Creating BitArrays from various sources
//! - **[`BitArrayConversion`]** - Converting BitArrays to other types
//! - **[`BitArrayAccess`]** - Reading and querying BitArray contents
//! - **[`BitArrayManipulation`]** - Modifying and transforming BitArrays
//!
//! ## Usage Examples
//!
//! ```rust
//! use flexfloat::prelude::*;
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

pub mod bit_ref;
pub mod boolean_list;
pub(crate) mod static_boolean_array;
pub mod traits;
pub mod usize_list;

pub use bit_ref::BitRef;
pub use boolean_list::BoolBitArray;
pub use traits::BitArray;
pub use usize_list::UsizeBitArray;

// Re-export trait methods for convenience - users only need to import BitArray
pub use traits::{
    BitArrayAccess, BitArrayConstruction, BitArrayConversion, BitArrayManipulation,
    BitArrayMutAccess,
};

/// Default `BitArray` implementation for general use in `1.x`.
pub type DefaultBitArray = BoolBitArray;
