//! Low-level primitives for bit-array backend authors.
//!
//! Implementing these methods lets a backend opt out of the slow
//! `to_bits()` / `from_bits()` round-trips that the default impls of
//! [`BitArrayManipulation`] fall back on.
//!
//! Most consumers should never call these methods directly; use the
//! high-level operations on [`BitArrayManipulation`] instead.

use core::ops::Range;

use crate::bitarray::traits::BitArrayAccess;

/// Low-level primitive operations for `BitArray` backend authors.
///
/// Implementing this trait allows a backend to provide efficient in-place
/// bit manipulation without going through the slow `to_bits()` / `from_bits()`
/// round-trips used by the default implementations of [`BitArrayManipulation`].
///
/// This is the "narrow waist" of the backend API: the high-level manipulation
/// methods delegate to these primitives.
///
/// # For backend authors
///
/// Import this trait explicitly when writing a new backend:
/// ```ignore
/// use flexfloat::bitarray::backend::BitArrayPrimitives;
/// ```
pub trait BitArrayPrimitives: BitArrayAccess {
    /// Appends a single bit to the end of the array.
    fn append_bool(&mut self, value: bool);

    /// Sets all bits in `range` to `value`.
    fn fill_range(&mut self, range: Range<usize>, value: bool);

    /// Copies the bits in `src` to a destination starting at `dst_start`.
    ///
    /// Behavior is unspecified if `src` and the destination range overlap.
    fn copy_within_bits(&mut self, src: Range<usize>, dst_start: usize);

    /// Appends `count` copies of `value` to the end of the array.
    fn extend_with(&mut self, count: usize, value: bool);

    /// Truncates the array to `n_bits`.
    ///
    /// If `n_bits >= self.len()` this is a no-op.
    fn truncate_in_place(&mut self, n_bits: usize);

    /// Hints to the backend that `n_bits` additional bits will be appended.
    ///
    /// Backends may use this to pre-allocate capacity. The default
    /// implementation does nothing.
    fn reserve(&mut self, _n_bits: usize) {}

    /// Returns `true` if any bit below `bit_index` is set.
    ///
    /// Equivalent to `self.iter_bits().take(bit_index).any(|b| b)` but
    /// backends can implement this more efficiently.
    fn any_set_below(&self, bit_index: usize) -> bool {
        let take = bit_index.min(self.len());
        self.iter_bits().take(take).any(|b| b)
    }
}
