#![allow(private_interfaces, dead_code)]
use core::f64::consts;

use crate::bitarray::static_bit_array::StaticBitArray;

pub(crate) type ConstFloat = crate::flexfloat::FlexFloat<StaticBitArray<11>, StaticBitArray<52>>;

pub(crate) const fn float_to_const(value: f64) -> ConstFloat {
    let bits = value.to_bits();
    let sign = (bits >> 63) & 1 == 1;
    let exponent = ((bits >> 52) & 0x7FF) as u16;
    let fraction = bits & 0xFFFFFFFFFFFFF;

    let mut exponent_bits = [false; 11];
    let mut i = 0;
    while i < 11 {
        exponent_bits[i] = (exponent & (1 << i)) != 0;
        i += 1;
    }

    let mut fraction_bits = [false; 52];
    let mut i = 0;
    while i < 52 {
        fraction_bits[i] = (fraction & (1 << i)) != 0;
        i += 1;
    }

    crate::flexfloat::FlexFloat {
        sign,
        exponent: StaticBitArray::new(exponent_bits),
        fraction: StaticBitArray::new(fraction_bits),
    }
}

/// Archimedes' constant (π) = 3.141592653589793
pub const PI: ConstFloat = float_to_const(consts::PI);
/// Full circle constant (τ) = 6.283185307179586
pub const TAU: ConstFloat = float_to_const(consts::TAU);
/// Euler's number (e) = 2.718281828459045
pub const E: ConstFloat = float_to_const(consts::E);
/// π/2 = 1.5707963267948966
pub const FRAC_PI_2: ConstFloat = float_to_const(consts::FRAC_PI_2);
/// π/3 = 1.0471975511965979
pub const FRAC_PI_3: ConstFloat = float_to_const(consts::FRAC_PI_3);
/// π/4 = 0.7853981633974483
pub const FRAC_PI_4: ConstFloat = float_to_const(consts::FRAC_PI_4);
/// π/6 = 0.5235987755982989
pub const FRAC_PI_6: ConstFloat = float_to_const(consts::FRAC_PI_6);
/// π/8 = 0.39269908169872414
pub const FRAC_PI_8: ConstFloat = float_to_const(consts::FRAC_PI_8);
/// 1/π = 0.3183098861837907
pub const FRAC_1_PI: ConstFloat = float_to_const(consts::FRAC_1_PI);
/// 2/π = 0.6366197723675814
pub const FRAC_2_PI: ConstFloat = float_to_const(consts::FRAC_2_PI);
/// √2 = 1.4142135623730951
pub const SQRT_2: ConstFloat = float_to_const(consts::SQRT_2);
/// 1/√2 = 0.7071067811865476
pub const FRAC_1_SQRT_2: ConstFloat = float_to_const(consts::FRAC_1_SQRT_2);
/// log₂(10) = 3.321928094887362
pub const LOG2_10: ConstFloat = float_to_const(consts::LOG2_10);
/// log₂(e) = 1.4426950408889634
pub const LOG2_E: ConstFloat = float_to_const(consts::LOG2_E);
/// log₁₀(2) = 0.3010299956639812
pub const LOG10_2: ConstFloat = float_to_const(consts::LOG10_2);
/// log₁₀(e) = 0.4342944819032518
pub const LOG10_E: ConstFloat = float_to_const(consts::LOG10_E);
/// ln(2) = 0.6931471805599453
pub const LN_2: ConstFloat = float_to_const(consts::LN_2);
/// ln(10) = 2.302585092994046
pub const LN_10: ConstFloat = float_to_const(consts::LN_10);
/// 2/sqrt(pi) = 1.1283791670955126
pub const FRAC_2_SQRT_PI: ConstFloat = float_to_const(consts::FRAC_2_SQRT_PI);

pub(crate) const ZERO: ConstFloat = float_to_const(0.0);
pub(crate) const HALF: ConstFloat = float_to_const(0.5);
pub(crate) const THIRD: ConstFloat = float_to_const(1.0 / 3.0);
pub(crate) const ONE: ConstFloat = float_to_const(1.0);
pub(crate) const NEGATIVE_ONE: ConstFloat = float_to_const(-1.0);
pub(crate) const TWO: ConstFloat = float_to_const(2.0);
pub(crate) const TOLERANCE: ConstFloat = float_to_const(1e-16);
pub(crate) const MAX_F64: ConstFloat = float_to_const(f64::MAX);
pub(crate) const MIN_POSITIVE_F64: ConstFloat = float_to_const(f64::MIN_POSITIVE);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(PI.to_f64().unwrap(), consts::PI);
        assert_eq!(TAU.to_f64().unwrap(), consts::TAU);
        assert_eq!(E.to_f64().unwrap(), consts::E);
        assert_eq!(FRAC_PI_2.to_f64().unwrap(), consts::FRAC_PI_2);
        assert_eq!(FRAC_PI_3.to_f64().unwrap(), consts::FRAC_PI_3);
        assert_eq!(FRAC_PI_4.to_f64().unwrap(), consts::FRAC_PI_4);
        assert_eq!(FRAC_PI_6.to_f64().unwrap(), consts::FRAC_PI_6);
        assert_eq!(FRAC_PI_8.to_f64().unwrap(), consts::FRAC_PI_8);
        assert_eq!(FRAC_1_PI.to_f64().unwrap(), consts::FRAC_1_PI);
        assert_eq!(FRAC_2_PI.to_f64().unwrap(), consts::FRAC_2_PI);
        assert_eq!(SQRT_2.to_f64().unwrap(), consts::SQRT_2);
        assert_eq!(FRAC_1_SQRT_2.to_f64().unwrap(), consts::FRAC_1_SQRT_2);
        assert_eq!(LOG2_10.to_f64().unwrap(), consts::LOG2_10);
        assert_eq!(LOG2_E.to_f64().unwrap(), consts::LOG2_E);
        assert_eq!(LOG10_2.to_f64().unwrap(), consts::LOG10_2);
        assert_eq!(LOG10_E.to_f64().unwrap(), consts::LOG10_E);
        assert_eq!(LN_2.to_f64().unwrap(), consts::LN_2);
        assert_eq!(LN_10.to_f64().unwrap(), consts::LN_10);
        assert_eq!(FRAC_2_SQRT_PI.to_f64().unwrap(), consts::FRAC_2_SQRT_PI);
    }

    #[test]
    fn test_private_constants() {
        assert_eq!(ZERO.to_f64().unwrap(), 0.0);
        assert_eq!(HALF.to_f64().unwrap(), 0.5);
        assert_eq!(ONE.to_f64().unwrap(), 1.0);
        assert_eq!(NEGATIVE_ONE.to_f64().unwrap(), -1.0);
        assert_eq!(TWO.to_f64().unwrap(), 2.0);
        assert_eq!(THIRD.to_f64().unwrap(), 1.0 / 3.0);
        assert_eq!(TOLERANCE.to_f64().unwrap(), 1e-16);
        assert_eq!(MAX_F64.to_f64().unwrap(), f64::MAX);
        assert_eq!(MIN_POSITIVE_F64.to_f64().unwrap(), f64::MIN_POSITIVE);
    }
}
