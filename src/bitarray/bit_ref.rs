//! A proxy reference to a single bit in packed bit storage.
//!
//! Used by bit-array backends that pack multiple bits into a single integer
//! word (e.g., `UsizeBitArray`). Such backends cannot soundly return
//! `&mut bool` because no `bool` actually exists in storage.
//!
//! `BitRef` solves this by:
//! 1. Reading the bit value into a local `bool` field (`value`).
//! 2. Allowing reads/writes through `Deref`/`DerefMut`.
//! 3. Writing the (possibly modified) value back into the underlying word
//!    when dropped.

use core::ops::{Deref, DerefMut};

/// A proxy reference to a single bit packed inside a `usize` word.
///
/// Reads from `*bit_ref` return the current value; writes via
/// `*bit_ref = ...` update an internal field that is flushed back to
/// the underlying word on `Drop`.
pub struct BitRef<'a> {
    value: bool,
    word: &'a mut usize,
    bit_mask: usize,
}

impl<'a> BitRef<'a> {
    /// Construct a `BitRef` for the bit selected by `bit_mask` in `word`.
    pub(crate) fn new(word: &'a mut usize, bit_mask: usize) -> Self {
        let value = (*word & bit_mask) != 0;
        Self {
            value,
            word,
            bit_mask,
        }
    }
}

impl<'a> Deref for BitRef<'a> {
    type Target = bool;

    fn deref(&self) -> &bool {
        &self.value
    }
}

impl<'a> DerefMut for BitRef<'a> {
    fn deref_mut(&mut self) -> &mut bool {
        &mut self.value
    }
}

impl<'a> Drop for BitRef<'a> {
    fn drop(&mut self) {
        if self.value {
            *self.word |= self.bit_mask;
        } else {
            *self.word &= !self.bit_mask;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_returns_current_bit() {
        let mut word: usize = 0b1010;
        let r = BitRef::new(&mut word, 1 << 1);
        assert!(*r);
    }

    #[test]
    fn read_returns_unset_bit() {
        let mut word: usize = 0b1010;
        let r = BitRef::new(&mut word, 1 << 0);
        assert!(!*r);
    }

    #[test]
    fn write_propagates_on_drop_set() {
        let mut word: usize = 0b0000;
        {
            let mut r = BitRef::new(&mut word, 1 << 2);
            *r = true;
        }
        assert_eq!(word, 0b0100);
    }

    #[test]
    fn write_propagates_on_drop_clear() {
        let mut word: usize = 0b1111;
        {
            let mut r = BitRef::new(&mut word, 1 << 1);
            *r = false;
        }
        assert_eq!(word, 0b1101);
    }

    #[test]
    fn no_write_on_drop_leaves_word_unchanged() {
        let mut word: usize = 0b1010;
        {
            let _r = BitRef::new(&mut word, 1 << 1);
        }
        assert_eq!(word, 0b1010);
    }
}
