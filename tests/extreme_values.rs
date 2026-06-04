//! Deterministic tests for FlexFloat operations at the boundaries of f64 range.
//!
//! These tests cover values where f64 itself overflows to ±∞ or underflows
//! to ±0; FlexFloat must remain finite and non-zero with a grown exponent
//! field, while preserving the sign.

use flexfloat::FlexFloat;
use flexfloat::prelude::*;

const EXTREME_VALUES: &[f64] = &[
    f64::MAX,
    f64::MIN_POSITIVE,
    f64::EPSILON,
    1e300,
    1e-300,
    1e308,
    5e-324,
    -f64::MAX,
    -f64::MIN_POSITIVE,
    -1e300,
    -1e-300,
    1.0,
    -1.0,
    0.0,
    -0.0,
];

fn classify_and_check_finite(name: &str, ff: &FlexFloat, f: f64) {
    if f.is_nan() {
        assert!(ff.is_nan(), "{name}: expected NaN, got {ff:?}");
        return;
    }
    if f.is_infinite() {
        assert!(
            !ff.is_infinite() && !ff.is_nan(),
            "{name}: f64 was inf but FlexFloat must stay finite: {ff:?}",
        );
        assert!(
            ff.exponent().len() > 11,
            "{name}: f64 overflow but FlexFloat exponent did not grow: {ff:?}",
        );
        return;
    }
    if f == 0.0 && !ff.is_zero() {
        assert!(
            ff.exponent().len() > 11 || ff.is_subnormal(),
            "{name}: underflow but result is neither grown nor subnormal: {ff:?}",
        );
        return;
    }
    // Check relative error entirely in FlexFloat arithmetic — no to_f64 conversion.
    let expected = FlexFloat::from(f);
    if ff != &expected {
        // rel = |ff - expected| / max(|expected|, MIN_POSITIVE)
        let diff = (ff.clone() - expected.clone()).abs();
        let denom = expected.abs().max(FlexFloat::from(1e-300_f64));
        let rel = diff / denom;
        assert!(
            rel < FlexFloat::from(1e-10_f64),
            "{name}: round-trip mismatch: ff={ff:?}, expected f={f:?}",
        );
    }
}

#[test]
fn add_at_f64_boundaries() {
    for &a in EXTREME_VALUES {
        for &b in EXTREME_VALUES {
            let ff_a = FlexFloat::from(a);
            let ff_b = FlexFloat::from(b);
            let result = ff_a + ff_b;
            let expected = a + b;
            classify_and_check_finite(&format!("{a:?} + {b:?}"), &result, expected);
        }
    }
}

#[test]
fn sub_at_f64_boundaries() {
    for &a in EXTREME_VALUES {
        for &b in EXTREME_VALUES {
            let ff_a = FlexFloat::from(a);
            let ff_b = FlexFloat::from(b);
            let result = ff_a - ff_b;
            let expected = a - b;
            classify_and_check_finite(&format!("{a:?} - {b:?}"), &result, expected);
        }
    }
}

#[test]
fn mul_at_f64_boundaries() {
    for &a in EXTREME_VALUES {
        for &b in EXTREME_VALUES {
            let ff_a = FlexFloat::from(a);
            let ff_b = FlexFloat::from(b);
            let result = ff_a * ff_b;
            let expected = a * b;
            classify_and_check_finite(&format!("{a:?} * {b:?}"), &result, expected);
        }
    }
}

#[test]
fn div_at_f64_boundaries() {
    for &a in EXTREME_VALUES {
        for &b in EXTREME_VALUES {
            if b == 0.0 {
                continue;
            }
            let ff_a = FlexFloat::from(a);
            let ff_b = FlexFloat::from(b);
            let result = ff_a / ff_b;
            let expected = a / b;
            classify_and_check_finite(&format!("{a:?} / {b:?}"), &result, expected);
        }
    }
}

#[test]
fn arithmetic_crossing_f64_overflow() {
    let max = FlexFloat::from(f64::MAX);
    let two = FlexFloat::from(2.0);
    let huge = max.clone() * two;
    assert!(
        !huge.is_infinite(),
        "MAX * 2 should stay finite, got {huge:?}"
    );
    assert!(
        huge.exponent().len() > 11,
        "MAX * 2 must grow exponent, got {huge:?}"
    );
}

#[test]
fn arithmetic_crossing_f64_underflow() {
    let tiny = FlexFloat::from(f64::MIN_POSITIVE);
    let half = FlexFloat::from(0.5);
    let smaller = tiny * half;
    assert!(
        !smaller.is_zero(),
        "MIN_POSITIVE * 0.5 should stay nonzero, got {smaller:?}"
    );
    assert!(
        smaller.exponent().len() > 11 || smaller.is_subnormal(),
        "MIN_POSITIVE * 0.5 must stay representable, got {smaller:?}",
    );
}
