//! Basic FlexFloat usage: construction, arithmetic, and display.
//!
//! Run with:
//!   cargo run --example basics

use flexfloat::prelude::*;

fn main() {
    // --- Construction ---
    let a = FlexFloat::from(3.0_f64);
    let b = FlexFloat::from(2.0_f64);

    println!("a = {a}");
    println!("b = {b}");

    // --- Basic arithmetic ---
    println!("a + b = {}", a.clone() + b.clone());
    println!("a - b = {}", a.clone() - b.clone());
    println!("a * b = {}", a.clone() * b.clone());
    println!("a / b = {}", a.clone() / b.clone());
    println!("a % b = {}", a.clone() % b.clone());

    // --- Negation and absolute value ---
    let neg = -a.clone();
    println!("-a = {neg}");
    println!("|{neg}| = {}", neg.abs());

    // --- Comparison ---
    println!("a > b: {}", a > b);
    println!("a == a: {}", a == a.clone());

    // --- Special values ---
    let zero = FlexFloat::zero();
    let pos_inf = FlexFloat::pos_infinity();
    let neg_inf = FlexFloat::neg_infinity();
    let nan = FlexFloat::nan();

    println!("\nSpecial values:");
    println!("  zero     = {zero},  is_zero = {}", zero.is_zero());
    println!(
        "  +inf     = {pos_inf},  is_infinite = {}",
        pos_inf.is_infinite()
    );
    println!("  -inf     = {neg_inf}");
    println!("  NaN      = {nan},  is_nan = {}", nan.is_nan());

    // --- Display formats ---
    let pi_approx = FlexFloat::from(std::f64::consts::PI);
    println!("\nPI in different formats:");
    println!("  fixed      : {pi_approx}");
    println!("  lower exp  : {pi_approx:e}");
    println!("  upper exp  : {pi_approx:E}");
    println!("  precision  : {pi_approx:.4}");
}
