//! Internal helpers for FlexFloat arithmetic: exponent growth and fraction rounding.
//!
//! These are `pub(crate)` — not part of the public API.

use core::cmp::{Ordering, max};

use num_bigint::BigInt;

use crate::bitarray::BitArray;
use crate::bitarray::traits::BitArrayRounding;
use crate::prelude::BitArrayConstruction;

/// Grow a `BigInt` exponent into a `BitArray` of sufficient width.
pub(crate) fn grow_exponent<B: BitArrayConstruction>(exponent: BigInt, min_bits: usize) -> B {
    let n_bits = exponent.bits();
    let ones = exponent
        .iter_u64_digits()
        .fold(0_u64, |count, el| count + u64::from(el.count_ones()));
    let needs_to_grow = usize::from(ones == n_bits);

    // +1 for the sign in signed vs unsigned
    let n_bits = max(n_bits as usize + needs_to_grow + 1, min_bits);
    B::from_bigint(&exponent, n_bits).unwrap()
}

/// Result of truncating a fraction to a target size with IEEE 754 rounding.
///
/// `carry` is `true` when round-to-nearest caused the fraction to overflow
/// its target size. Callers must add 1 to the exponent and shift the
/// fraction right by 1 in that case.
#[derive(Debug, Clone)]
pub(crate) struct RoundedResult<B> {
    /// The rounded fraction, exactly `size` bits long.
    pub fraction: B,
    /// True if rounding overflowed the fraction (carry into exponent).
    pub carry: bool,
}

pub(crate) fn truncate_fraction<B: BitArray>(fraction: B, size: usize) -> RoundedResult<B> {
    let shift = size as isize - fraction.len() as isize;

    match shift.cmp(&0) {
        Ordering::Equal => RoundedResult {
            fraction,
            carry: false,
        },
        Ordering::Greater => RoundedResult {
            fraction: fraction.shift_grow(shift),
            carry: false,
        },
        Ordering::Less => {
            let shift_abs = shift.unsigned_abs();

            let lsb = fraction.get(shift_abs).unwrap();
            let guard = fraction.get(shift_abs - 1).unwrap();
            let rest = fraction.iter_bits().take(shift_abs - 1).any(|el| el);

            let rounding = guard && (lsb || rest);

            let mut truncated = fraction.shift_fixed(-shift).truncate(size);
            if rounding {
                truncated.add_one_in_place();
                if truncated.len() > size {
                    RoundedResult {
                        fraction: truncated.shift_fixed(1).truncate(size),
                        carry: true,
                    }
                } else {
                    RoundedResult {
                        fraction: truncated,
                        carry: false,
                    }
                }
            } else {
                RoundedResult {
                    fraction: truncated,
                    carry: false,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitarray::{BitArrayConversion, DefaultBitArray};

    #[test]
    fn test_grow_exponent_basic() {
        use num_bigint::BigInt;
        let value = BigInt::from(127);
        let bit_array: DefaultBitArray = grow_exponent(value.clone(), 8);
        assert_eq!(bit_array.to_bigint(), value);
    }
}
