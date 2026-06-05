//! Computes π using the Gauss–Legendre AGM iteration with FlexFloat.
//!
//! The algorithm converges quadratically — each step roughly doubles the
//! correct digits. This example shows how FlexFloat naturally handles the
//! growing intermediate values without any overflow to ±∞.
//!
//! Run with:
//!   cargo run --example agm_pi

use flexfloat::prelude::*;

/// Estimate π using `iters` steps of the Gauss–Legendre AGM.
fn agm_pi(iters: usize) -> FlexFloat {
    let one = FlexFloat::from(1.0_f64);
    let two = FlexFloat::from(2.0_f64);
    let four = FlexFloat::from(4.0_f64);

    // a₀ = 1,  b₀ = 1/√2,  t₀ = 1/4,  p₀ = 1
    let mut a = one.clone();
    let mut b = one.clone() / &two.clone().sqrt();
    let mut t = one.clone() / &four;
    let mut p = one.clone();

    for i in 0..iters {
        let a_next = (a.clone() + &b) / &two;
        let b_next = (a.clone() * &b).sqrt();
        let diff = a.clone() - &a_next;
        let t_next = t - &(p.clone() * &(diff.clone() * &diff));
        let p_next = two.clone() * &p;

        a = a_next;
        b = b_next;
        t = t_next;
        p = p_next;

        // π ≈ (a + b)² / (4t) at each step
        let sum = a.clone() + &b;
        let pi_approx = (sum.clone() * &sum) / &(four.clone() * &t);
        println!("  iter {}: π ≈ {:.*}", i + 1, 15, pi_approx);
    }

    let sum = a + &b;
    (sum.clone() * &sum) / &(four * &t)
}

fn main() {
    println!("Gauss–Legendre AGM: computing π\n");
    let pi = agm_pi(6);
    println!("\nFinal estimate : {pi:.15}");
    println!("Reference (f64): {:.15}", std::f64::consts::PI);
}
