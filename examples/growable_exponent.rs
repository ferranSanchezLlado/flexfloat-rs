//! Demonstrates the growable-exponent feature of FlexFloat.
//!
//! With `f64`, multiplying two `f64::MAX` values produces `+inf`.
//! FlexFloat detects the impending overflow and widens the exponent field
//! automatically, keeping the result finite.
//!
//! Run with:
//!   cargo run --example growable_exponent

use flexfloat::prelude::*;

fn main() {
    // --- Standard f64 overflows to infinity ---
    let f64_overflow = f64::MAX * f64::MAX;
    println!("f64::MAX * f64::MAX = {f64_overflow}");

    // --- FlexFloat stays finite ---
    let max = FlexFloat::from(f64::MAX);
    let huge = max.clone() * max.clone();

    println!("\nFlexFloat::from(f64::MAX) * FlexFloat::from(f64::MAX):");
    println!("  is_infinite   = {}", huge.is_infinite()); // false
    println!("  is_finite     = {}", huge.is_finite()); // true
    println!("  result        = {huge:e}");

    // --- Chain multiplications; exponent keeps growing ---
    println!("\nRepeated squaring of f64::MAX:");
    let mut val = FlexFloat::from(f64::MAX);
    for i in 1..=5 {
        val = val.clone() * val.clone();
        println!("  step {i}: {val:e}");
    }

    // --- A grown value can be divided back down ---
    let max2 = FlexFloat::from(f64::MAX) * FlexFloat::from(f64::MAX);
    let divided = max2 / FlexFloat::from(f64::MAX);
    println!("\n(MAX * MAX) / MAX:");
    println!("  result ≈ f64::MAX: {:e}", divided);
}
