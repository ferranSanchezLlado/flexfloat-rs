//! Math functions available on FlexFloat: exponential, power, trig, hyperbolic.
//!
//! Run with:
//!   cargo run --example math_functions

use flexfloat::math;
use flexfloat::prelude::*;

fn main() {
    let x = FlexFloat::from(2.0_f64);
    let angle = FlexFloat::from(std::f64::consts::FRAC_PI_4); // π/4 = 45°

    // --- Rounding ---
    let y = FlexFloat::from(2.7_f64);
    println!("Rounding 2.7:");
    println!("  floor  = {}", math::floor(y.clone()));
    println!("  ceil   = {}", math::ceil(y.clone()));
    println!("  round  = {}", math::round(y.clone()));
    println!("  trunc  = {}", math::trunc(y.clone()));
    println!("  fract  = {}", math::fract(y.clone()));

    // --- Exponential & logarithm ---
    println!("\nExponential / logarithm (x = 2):");
    println!("  exp(x)  = {}", math::exp(x.clone()));
    println!("  exp2(x) = {}", math::exp2(x.clone()));
    println!("  ln(x)   = {}", math::ln(x.clone()));
    println!("  log2(x) = {}", math::log2(x.clone()));
    println!("  log10(x)= {}", math::log10(x.clone()));

    // --- Power / root ---
    println!("\nPower / root (x = 2):");
    println!("  sqrt(x) = {}", math::sqrt(x.clone()));
    println!("  cbrt(x) = {}", math::cbrt(x.clone()));
    println!("  x^10    = {}", x.clone().powi(10));
    println!("  x^2.5   = {}", x.clone().powf(FlexFloat::from(2.5_f64)));

    // --- Trigonometry (angle = π/4) ---
    println!("\nTrigonometry (angle = π/4):");
    println!("  sin  = {}", math::sin(angle.clone()));
    println!("  cos  = {}", math::cos(angle.clone()));
    println!("  tan  = {}", math::tan(angle.clone()));
    let (s, c) = angle.clone().sin_cos();
    println!("  sin_cos = ({s}, {c})");
    println!("  atan(1) = {}", math::atan(FlexFloat::from(1.0_f64)));

    // --- Hyperbolic ---
    println!("\nHyperbolic (x = 2):");
    println!("  sinh = {}", math::sinh(x.clone()));
    println!("  cosh = {}", math::cosh(x.clone()));
    println!("  tanh = {}", math::tanh(x.clone()));

    // --- Utility ---
    println!("\nUtility (x = 2):");
    println!("  recip      = {}", math::recip(x.clone()));
    println!("  to_degrees = {}", math::to_degrees(angle.clone()));
    println!("  signum(-5) = {}", math::signum(FlexFloat::from(-5.0_f64)));
    println!(
        "  mul_add(2, 3, 4) = {}",
        math::mul_add(
            FlexFloat::from(2.0_f64),
            FlexFloat::from(3.0_f64),
            FlexFloat::from(4.0_f64)
        )
    );
}
