//! Conversions between FlexFloat and primitive types.
//!
//! Run with:
//!   cargo run --example conversions

use flexfloat::{FlexFloat, FlexFloatToF64Error};
use num_bigint::BigInt;

fn main() {
    // --- From primitives (lossless) ---
    println!("Constructing FlexFloat from primitives:");
    let from_f64 = FlexFloat::from(1.5_f64);
    let from_f32 = FlexFloat::from(1.5_f32);
    let from_i64 = FlexFloat::from(-42_i64);
    let from_u64 = FlexFloat::from(42_u64);
    let from_i32 = FlexFloat::from(-7_i32);
    let from_u32 = FlexFloat::from(7_u32);
    let from_big = FlexFloat::from(BigInt::from(123_456_789_i64));

    println!("  f64 1.5  → {from_f64}");
    println!("  f32 1.5  → {from_f32}");
    println!("  i64 -42  → {from_i64}");
    println!("  u64 42   → {from_u64}");
    println!("  i32 -7   → {from_i32}");
    println!("  u32 7    → {from_u32}");
    println!("  BigInt 123456789 → {from_big}");

    // --- TryFrom: FlexFloat → f64 (succeeds when in range) ---
    println!("\nConverting back to f64:");
    let y = FlexFloat::from(1.5_f64);
    let result: Result<f64, _> = y.try_into();
    println!("  3.14 → f64: {:?}", result);

    // --- TryFrom: FlexFloat → f64 (fails when exponent grew) ---
    let huge = FlexFloat::from(f64::MAX) * FlexFloat::from(f64::MAX);
    let overflow: Result<f64, FlexFloatToF64Error> = huge.try_into();
    println!("  MAX*MAX → f64: {:?}", overflow); // Err(ExponentOverflow)

    // --- TryFrom: FlexFloat → BigInt ---
    println!("\nConverting to BigInt:");
    let int_val = FlexFloat::from(42_i64);
    let big: Result<BigInt, _> = int_val.try_into();
    println!("  42 → BigInt: {:?}", big);

    let frac_val = FlexFloat::from(3.7_f64);
    let not_int: Result<BigInt, _> = frac_val.try_into();
    println!("  3.7 → BigInt: {:?}", not_int); // Ok(3) — truncated toward zero

    // --- Byte serialisation round-trip ---
    println!("\nByte serialisation round-trip:");
    let original = FlexFloat::from(1.23_f64);
    let (le_bytes, exp_bits) = original.to_le_bytes();
    let restored = FlexFloat::from_le_bytes(&le_bytes, exp_bits);
    println!("  original  = {original}");
    println!("  restored  = {restored}");
    println!("  equal     = {}", original == restored);
}
