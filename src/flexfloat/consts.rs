#![allow(private_interfaces, dead_code)]
use core::f64::consts;

use crate::bitarray::static_boolean_array::StaticBoolArray;

type FlexFloat = crate::FlexFloat<StaticBoolArray>;
// https://geometrian.com/util/float/

const fn float_to_flexfloat_static(value: f64) -> FlexFloat {
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

    crate::FlexFloat {
        sign,
        exponent: StaticBoolArray::new(exponent_bits),
        fraction: StaticBoolArray::new(fraction_bits),
    }
}

/// Archimedes' constant (π) = 3.141592653589793
pub const PI: FlexFloat = float_to_flexfloat_static(consts::PI);
/// Full circle constant (τ) = 6.283185307179586
pub const TAU: FlexFloat = float_to_flexfloat_static(consts::TAU);
/// Euler's number (e) = 2.718281828459045
pub const E: FlexFloat = float_to_flexfloat_static(consts::E);
/// π/2 = 1.5707963267948966
pub const FRAC_PI_2: FlexFloat = float_to_flexfloat_static(consts::FRAC_PI_2);
/// π/3 = 1.0471975511965979
pub const FRAC_PI_3: FlexFloat = float_to_flexfloat_static(consts::FRAC_PI_3);
/// π/4 = 0.7853981633974483
pub const FRAC_PI_4: FlexFloat = float_to_flexfloat_static(consts::FRAC_PI_4);
/// π/6 = 0.5235987755982989
pub const FRAC_PI_6: FlexFloat = float_to_flexfloat_static(consts::FRAC_PI_6);
/// π/8 = 0.39269908169872414
pub const FRAC_PI_8: FlexFloat = float_to_flexfloat_static(consts::FRAC_PI_8);
/// 1/π = 0.3183098861837907
pub const FRAC_1_PI: FlexFloat = float_to_flexfloat_static(consts::FRAC_1_PI);
/// 2/π = 0.6366197723675814
pub const FRAC_2_PI: FlexFloat = float_to_flexfloat_static(consts::FRAC_2_PI);
/// √2 = 1.4142135623730951
pub const SQRT_2: FlexFloat = float_to_flexfloat_static(consts::SQRT_2);
/// 1/√2 = 0.7071067811865476
pub const FRAC_1_SQRT_2: FlexFloat = float_to_flexfloat_static(consts::FRAC_1_SQRT_2);
/// log₂(10) = 3.321928094887362
pub const LOG2_10: FlexFloat = float_to_flexfloat_static(consts::LOG2_10);
/// log₂(e) = 1.4426950408889634
pub const LOG2_E: FlexFloat = float_to_flexfloat_static(consts::LOG2_E);
/// log₁₀(2) = 0.3010299956639812
pub const LOG10_2: FlexFloat = float_to_flexfloat_static(consts::LOG10_2);
/// log₁₀(e) = 0.4342944819032518
pub const LOG10_E: FlexFloat = float_to_flexfloat_static(consts::LOG10_E);
/// ln(2) = 0.6931471805599453
pub const LN_2: FlexFloat = float_to_flexfloat_static(consts::LN_2);
/// ln(10) = 2.302585092994046
pub const LN_10: FlexFloat = float_to_flexfloat_static(consts::LN_10);
/// 2/sqrt(pi) = 1.1283791670955126
pub const FRAC_2_SQRT_PI: FlexFloat = float_to_flexfloat_static(consts::FRAC_2_SQRT_PI);

// Private constantes
// https://geometrian.com/util/float/
pub(crate) const ZERO: FlexFloat = float_to_flexfloat_static(0.0);
pub(crate) const HALF: FlexFloat = float_to_flexfloat_static(0.5);
pub(crate) const THIRD: FlexFloat = float_to_flexfloat_static(1.0 / 3.0);
pub(crate) const ONE: FlexFloat = float_to_flexfloat_static(1.0);
pub(crate) const TWO: FlexFloat = float_to_flexfloat_static(2.0);
pub(crate) const TOLERANCE: FlexFloat = float_to_flexfloat_static(1e-16);

/// Padé approximation coefficients for exponential function
/// - 1 / 2
/// - 1 / 9
/// - 1 / 72
/// - 1 / 1008
/// - 1 / 30240
pub(crate) const PADE_EXP_COEFFS: [FlexFloat; 5] = [
    float_to_flexfloat_static(1.0 / 2.0),
    float_to_flexfloat_static(1.0 / 9.0),
    float_to_flexfloat_static(1.0 / 72.0),
    float_to_flexfloat_static(1.0 / 1008.0),
    float_to_flexfloat_static(1.0 / 30240.0),
];

/// Minimax kernel coefficients for sine on [-pi/4, pi/4].
pub(crate) const SIN_COEFFS: [FlexFloat; 6] = [
    float_to_flexfloat_static(-1.666_666_666_666_663_2e-1),
    float_to_flexfloat_static(8.333_333_333_322_49e-3),
    float_to_flexfloat_static(-1.984_126_982_985_795e-4),
    float_to_flexfloat_static(2.755_731_370_707_006_6e-6),
    float_to_flexfloat_static(-2.505_076_025_340_686_3e-8),
    float_to_flexfloat_static(1.589_690_995_211_55e-10),
];

/// Minimax kernel coefficients for cosine on [-pi/4, pi/4].
pub(crate) const COS_COEFFS: [FlexFloat; 6] = [
    float_to_flexfloat_static(-5.0e-1),
    float_to_flexfloat_static(4.166_666_666_666_66e-2),
    float_to_flexfloat_static(-1.388_888_888_887_411e-3),
    float_to_flexfloat_static(2.480_158_728_947_673e-5),
    float_to_flexfloat_static(-2.755_731_435_139_066_5e-7),
    float_to_flexfloat_static(2.087_572_321_298_175e-9),
];

pub(crate) const MIN_F64: FlexFloat = float_to_flexfloat_static(f64::MIN);
pub(crate) const MAX_F64: FlexFloat = float_to_flexfloat_static(f64::MAX);
pub(crate) const MIN_POSITIVE_F64: FlexFloat = float_to_flexfloat_static(f64::MIN_POSITIVE);
pub(crate) const DEG_PER_RAD: FlexFloat = float_to_flexfloat_static(180.0 / consts::PI);
pub(crate) const RAD_PER_DEG: FlexFloat = float_to_flexfloat_static(consts::PI / 180.0);
pub(crate) const FRAC_3_PI_2: FlexFloat = float_to_flexfloat_static(3.0 * consts::PI / 2.0);
pub(crate) const INV_TWO_PI: FlexFloat = float_to_flexfloat_static(1.0 / consts::TAU);

// Extended precision constants for Cody-Waite range reduction
// 2π split into multiple parts for better precision
// 2π = TWO_PI_HIGH + TWO_PI_MID + TWO_PI_LOW
// Using three parts allows better precision for larger input values
pub(crate) const TWO_PI_HIGH: FlexFloat = float_to_flexfloat_static(6.28125); // Exactly representable
pub(crate) const TWO_PI_MID: FlexFloat = float_to_flexfloat_static(0.0019353071795862676);
pub(crate) const TWO_PI_LOW: FlexFloat = float_to_flexfloat_static(-6.077094383272197e-11); // Correction term

// Extended Cody-Waite split of π/2 for accurate range reduction
// π/2 = HIGH + MID + LOW + VERYLOW + ULTRA + PART6 + PART7 + PART8 + PART9 + PART10
// This 10-part split provides extremely high precision (~114 bits of representation accuracy)
// for range reduction, even with very large k values
pub(crate) const FRAC_PI_2_HIGH: FlexFloat = float_to_flexfloat_static(1.5703125);
pub(crate) const FRAC_PI_2_MID: FlexFloat = float_to_flexfloat_static(0.0004838267948965669);
pub(crate) const FRAC_PI_2_LOW: FlexFloat = float_to_flexfloat_static(-1.5192735958180493e-11);
pub(crate) const FRAC_PI_2_VERYLOW: FlexFloat =
    float_to_flexfloat_static(5.233_132_169_163_975e-17);
pub(crate) const FRAC_PI_2_ULTRA: FlexFloat = float_to_flexfloat_static(9.999_974_420_985_847e-28);
pub(crate) const FRAC_PI_2_PART6: FlexFloat =
    float_to_flexfloat_static(1.519_273_595_818_049_2e-38);
pub(crate) const FRAC_PI_2_PART7: FlexFloat = float_to_flexfloat_static(-3.124_470_895_125_277e-46);
pub(crate) const FRAC_PI_2_PART8: FlexFloat =
    float_to_flexfloat_static(-3.846_091_796_890_400_6e-63);
pub(crate) const FRAC_PI_2_PART9: FlexFloat =
    float_to_flexfloat_static(-1.488_944_785_170_662_4e-80);
pub(crate) const FRAC_PI_2_PART10: FlexFloat = float_to_flexfloat_static(6.643_528_478_135_194e-98);

// floor((2/pi) * 2^1200), stored as big-endian u64 words.
pub(crate) const TWO_OVER_PI_BITS: [u64; 19] = [
    0x0000a2f9836e4e44,
    0x1529fc2757d1f534,
    0xddc0db6295993c43,
    0x9041fe5163abdebb,
    0xc561b7246e3a424d,
    0xd2e006492eea09d1,
    0x921cfe1deb1cb129,
    0xa73ee88235f52ebb,
    0x4484e99c7026b45f,
    0x7e413991d6398353,
    0x39f49c845f8bbdf9,
    0x283b1ff897ffde05,
    0x980fef2f118b5a0a,
    0x6d1f6d367ecf27cb,
    0x09b74f463f669e5f,
    0xea2d7527bac7ebe5,
    0xf17b3d0739f78a52,
    0x92ea6bfb5fb11f8d,
    0x5d0856033046fc7b,
];

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
        assert_eq!(TWO.to_f64().unwrap(), 2.0);
        assert_eq!(THIRD.to_f64().unwrap(), 1.0 / 3.0);
        assert_eq!(TOLERANCE.to_f64().unwrap(), 1e-16);
        assert_eq!(MIN_F64.to_f64().unwrap(), f64::MIN);
        assert_eq!(MAX_F64.to_f64().unwrap(), f64::MAX);
        assert_eq!(DEG_PER_RAD.to_f64().unwrap(), 180.0 / consts::PI);
        assert_eq!(RAD_PER_DEG.to_f64().unwrap(), consts::PI / 180.0);
        assert_eq!(FRAC_3_PI_2.to_f64().unwrap(), 3.0 * consts::PI / 2.0);
        assert_eq!(INV_TWO_PI.to_f64().unwrap(), 1.0 / consts::TAU);
    }
}
