//! Rounding-aware bit operations.
//!
//! Provides [`BitArrayRounding`], an internal extension of
//! [`BitArrayManipulation`] that tracks bits discarded during a
//! right-shift so that callers can apply IEEE 754 round-to-nearest-even.
//!
//! The shift convention used here: `shift_right_rounded(amount)` is
//! the bit-shift conceptually equivalent to dividing by 2^amount,
//! returning both the quotient and the discarded low bits in the form
//! of guard / round / sticky.

use crate::bitarray::backend::BitArrayPrimitives;
use crate::bitarray::traits::BitArrayMutAccess;
use crate::bitarray::traits::access::BitArrayAccess;
use crate::bitarray::traits::manipulation::BitArrayManipulation;

/// Information about bits discarded during a right shift, in IEEE 754
/// guard/round/sticky form.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ShiftRoundingInfo {
    /// The first bit shifted out (most significant discarded bit).
    pub guard: bool,
    /// The second bit shifted out.
    pub round: bool,
    /// OR of all remaining shifted-out bits below `round`.
    pub sticky: bool,
}

impl ShiftRoundingInfo {
    /// Decide whether to round up given the LSB of the (already shifted) result.
    ///
    /// Implements round-to-nearest, ties-to-even:
    ///   - guard=1 AND (round=1 OR sticky=1)         -> round up (clearly nearer)
    ///   - guard=1 AND round=0 AND sticky=0 AND lsb=1 -> tie, round to even (up)
    ///   - guard=1 AND round=0 AND sticky=0 AND lsb=0 -> tie, round to even (down)
    ///   - guard=0                                    -> round down
    pub fn should_round_up(&self, result_lsb: bool) -> bool {
        self.guard && (self.round || self.sticky || result_lsb)
    }

    /// Combine two consecutive rounding observations.
    ///
    /// Used when an alignment shift is followed by a normalization shift:
    /// the second shift's discarded bits subsume the first's into its sticky.
    pub fn combine(self, later: ShiftRoundingInfo) -> ShiftRoundingInfo {
        let prior_any = self.guard || self.round || self.sticky;
        ShiftRoundingInfo {
            guard: later.guard,
            round: later.round,
            sticky: later.sticky || prior_any,
        }
    }
}

/// Result of a rounding-aware right shift: the shifted value plus the
/// rounding info from the discarded bits.
#[derive(Debug, Clone)]
pub(crate) struct ShiftRoundingResult<B> {
    /// The shifted value (low bits replaced by zeros).
    pub value: B,
    /// Rounding information from the discarded bits.
    pub info: ShiftRoundingInfo,
}

/// Internal trait providing rounding-aware bit operations on top of
/// the regular bit-array traits.
pub(crate) trait BitArrayRounding:
    BitArrayManipulation + BitArrayAccess + BitArrayMutAccess + BitArrayPrimitives
{
    /// Adds 1 to the value in-place, growing by one bit on overflow.
    fn add_one_in_place(&mut self) {
        for i in 0..self.len() {
            let mut bit = self
                .get_mut(i)
                .expect("bit index must remain valid while incrementing");
            if *bit {
                *bit = false;
            } else {
                *bit = true;
                return;
            }
        }

        self.append_bool(true);
    }

    /// Right-shift by `amount` positions, returning the shifted value and
    /// guard/round/sticky info for the bits that were dropped.
    fn shift_right_rounded(self, amount: usize) -> ShiftRoundingResult<Self>
    where
        Self: Sized;
}

impl<T> BitArrayRounding for T
where
    T: BitArrayManipulation + BitArrayAccess + BitArrayMutAccess + BitArrayPrimitives + Sized,
{
    fn shift_right_rounded(self, amount: usize) -> ShiftRoundingResult<Self> {
        if amount == 0 {
            return ShiftRoundingResult {
                value: self,
                info: ShiftRoundingInfo::default(),
            };
        }

        let len = self.len();
        let guard = if amount >= 1 && amount - 1 < len {
            self.get(amount - 1).unwrap_or(false)
        } else {
            false
        };
        let round = if amount >= 2 && amount - 2 < len {
            self.get(amount - 2).unwrap_or(false)
        } else {
            false
        };
        let sticky = if amount >= 3 {
            self.any_set_below((amount - 2).min(len))
        } else {
            false
        };

        let value = match isize::try_from(amount) {
            Ok(amount_isize) => self.shift_fixed(amount_isize),
            Err(_) => self.reset(),
        };

        ShiftRoundingResult {
            value,
            info: ShiftRoundingInfo {
                guard,
                round,
                sticky,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitarray::BoolBitArray;
    use crate::bitarray::traits::construction::BitArrayConstruction;
    use crate::bitarray::traits::conversion::BitArrayConversion;

    #[test]
    fn shift_right_zero_is_noop() {
        let b = BoolBitArray::from_bits(&[true, false, true, true]);
        let r = b.shift_right_rounded(0);
        assert_eq!(r.value.to_bits(), vec![true, false, true, true]);
        assert!(!r.info.guard);
        assert!(!r.info.round);
        assert!(!r.info.sticky);
    }

    #[test]
    fn shift_right_by_one_captures_guard() {
        let b = BoolBitArray::from_bits(&[true, false, true, true]);
        let r = b.shift_right_rounded(1);
        assert_eq!(r.value.to_bits(), vec![false, true, true, false]);
        assert!(r.info.guard);
        assert!(!r.info.round);
        assert!(!r.info.sticky);
        assert!(!r.info.should_round_up(false));
        assert!(r.info.should_round_up(true));
    }

    #[test]
    fn shift_right_captures_round_and_sticky() {
        let b = BoolBitArray::from_bits(&[true, true, true, true, false, false, false, false]);
        let r = b.shift_right_rounded(4);
        assert_eq!(
            r.value.to_bits(),
            vec![false, false, false, false, false, false, false, false]
        );
        assert!(r.info.guard);
        assert!(r.info.round);
        assert!(r.info.sticky);
        assert!(r.info.should_round_up(false));
        assert!(r.info.should_round_up(true));
    }

    #[test]
    fn ties_to_even() {
        let b = BoolBitArray::from_bits(&[true, false, false, true, true]);
        let r = b.shift_right_rounded(1);
        assert_eq!(r.value.to_bits(), vec![false, false, true, true, false]);
        assert!(r.info.guard);
        assert!(!r.info.round);
        assert!(!r.info.sticky);
        assert!(!r.info.should_round_up(false));
        assert!(r.info.should_round_up(true));
    }

    #[test]
    fn combine_propagates_prior_into_sticky() {
        let first = ShiftRoundingInfo {
            guard: true,
            round: false,
            sticky: false,
        };
        let second = ShiftRoundingInfo {
            guard: false,
            round: false,
            sticky: false,
        };
        let c = first.combine(second);
        assert!(!c.guard);
        assert!(!c.round);
        assert!(c.sticky);
    }
}
