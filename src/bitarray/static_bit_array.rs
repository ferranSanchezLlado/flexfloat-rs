//! Const-generic static bit array backend.
//!
//! `StaticBitArray<N>` is a zero-overhead bit array whose length is known at
//! compile time.  It implements [`BitArrayAccess`] and [`BitArrayConversion`]
//! so it can be used as the exponent or fraction field of a
//! `FlexFloat<StaticBitArray<11>, StaticBitArray<52>>` constant.
//!
//! It intentionally does **not** implement `BitArrayArith`, `BitArrayManipulation`,
//! or `Default` — const-context constants are read-only and arithmetic is
//! performed on the dynamic runtime backends after conversion.

use crate::bitarray::DefaultBitArray;
use crate::bitarray::traits::{BitArrayAccess, BitArrayConstruction, BitArrayConversion};

/// A const-generic, compile-time bit array with exactly `N` bits.
///
/// The canonical use is as the backend for compile-time `FlexFloat` constants:
///
/// ```rust
/// use flexfloat::bitarray::StaticBitArray;
/// use flexfloat::flexfloat::FlexFloat;
///
/// type ConstFloat = FlexFloat<StaticBitArray<11>, StaticBitArray<52>>;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticBitArray<const N: usize> {
    bits: [bool; N],
}

impl<const N: usize> StaticBitArray<N> {
    /// Construct a `StaticBitArray` from a fixed-size bool array.
    ///
    /// This is a `const fn` so it can be used in const contexts.
    pub const fn new(bits: [bool; N]) -> Self {
        Self { bits }
    }

    /// Return a reference to the underlying bit slice.
    pub const fn bits(&self) -> &[bool; N] {
        &self.bits
    }
}

// ─── BitArrayAccess ──────────────────────────────────────────────────────────

impl<const N: usize> BitArrayAccess for StaticBitArray<N> {
    fn iter_bits(&self) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator {
        self.bits.iter().copied()
    }

    fn len(&self) -> usize {
        N
    }

    fn get(&self, index: usize) -> Option<bool> {
        self.bits.get(index).copied()
    }
}

// ─── BitArrayConversion ──────────────────────────────────────────────────────

impl<const N: usize> BitArrayConversion for StaticBitArray<N> {
    fn to_bits(&self) -> Vec<bool> {
        self.bits.to_vec()
    }

    fn to_bytes(&self) -> Vec<u8> {
        let n_bytes = N.div_ceil(8);
        let mut bytes = vec![0u8; n_bytes];
        for (i, &bit) in self.bits.iter().enumerate() {
            if bit {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }
        bytes
    }

    fn convert_to<B: BitArrayConstruction + 'static>(self) -> B
    where
        Self: Sized,
    {
        B::from_bits(&self.bits)
    }
}

// ─── Conversion to DefaultBitArray ───────────────────────────────────────────

impl<const N: usize> From<&StaticBitArray<N>> for DefaultBitArray {
    fn from(a: &StaticBitArray<N>) -> Self {
        DefaultBitArray::from_bits(&a.bits)
    }
}

impl<const N: usize> From<StaticBitArray<N>> for DefaultBitArray {
    fn from(a: StaticBitArray<N>) -> Self {
        DefaultBitArray::from_bits(&a.bits)
    }
}
