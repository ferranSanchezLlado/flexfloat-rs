# FlexFloat

A high-precision Rust library for arbitrary-precision floating-point arithmetic with growable exponents and fixed-size fractions. FlexFloat extends IEEE 754 double-precision format to handle numbers beyond the standard range while maintaining computational efficiency and precision consistency.

[![Crates.io](https://img.shields.io/crates/v/flexfloat.svg)](https://crates.io/crates/flexfloat)
[![Documentation](https://docs.rs/flexfloat/badge.svg)](https://docs.rs/flexfloat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

FlexFloat provides a flexible floating-point arithmetic system that automatically adapts to the scale of your computations:

- **🔄 Growable Exponents**: Automatically expand exponent bit width when values exceed the current range
- **📏 Fixed Precision**: Maintain consistent 52-bit mantissa (fraction) for precision consistency
- **🔗 IEEE 754 Compatible**: Full support for standard floating-point operations and special values (±0, ±∞, NaN)
- **⚡ Performance Focused**: Efficient operations while maintaining arbitrary-precision capabilities
- **🛠️ Backend-generic API**: The public API is generic over the `BitArray` trait, with `BoolBitArray` as the default backend

## Quick Start

Add FlexFloat to your `Cargo.toml`:

```toml
[dependencies]
flexfloat = "0.1.1"
```

### Basic Usage

```rust
use flexfloat::FlexFloat;

// Create FlexFloat from standard types
let x = FlexFloat::from(3.14159);
let y = FlexFloat::from(2.71828);

// Perform arithmetic operations
let sum = x + y;  // Arithmetic operations
let neg_x = -x;   // Negation
let abs_y = y.abs();  // Absolute value

// Work with special values
let zero = FlexFloat::zero();
let infinity = FlexFloat::pos_infinity();
let nan = FlexFloat::nan();

// Convert back to f64 when in range
let result: f64 = sum.into();
```

### Advanced Examples

```rust
use flexfloat::{FlexFloat, bitarray::{BitArray, BoolBitArray, UsizeBitArray}};

// Inspect internal representation
let num = FlexFloat::from(123.456);
println!("Sign: {}", num.sign());
println!("Exponent bits: {}", num.exponent().len());  
println!("Fraction bits: {}", num.fraction().len());

// Work with bit arrays directly
let custom_bits = BoolBitArray::from_bits(&[true, false, true]);
let from_bytes = BoolBitArray::from_bytes(&[0xFF, 0x00], 16);
let packed_bits = UsizeBitArray::from_bits(&[true, false, true, true]);

// Handle special cases
if num.is_nan() {
    println!("Not a number");
} else if num.is_infinity() {
    println!("Infinite value");
} else if num.is_zero() {
    println!("Zero value");
}
```

## Architecture

FlexFloat is built around two main components:

### BitArray Module

Provides flexible bit manipulation with the backends currently shipped in this crate:

- **BoolBitArray**: Vector of booleans for simplicity and debugging
- **UsizeBitArray**: Packed `usize`-backed storage used for backend consistency coverage
- **Rich conversion utilities**: Support for bytes, BigUint/BigInt, f64, and more

### FlexFloat Module  

Core floating-point implementation featuring:

- **Variable exponent**: Starts at 11 bits, grows automatically
- **Fixed mantissa**: 52 bits for IEEE 754 compatibility
- **Special values**: Full support for ±0, ±∞, and NaN
- **Generic backend**: Works with any BitArray implementation

## Comparison

| Feature     | f64     | BigDecimal | FlexFloat       |
| ----------- | ------- | ---------- | ----------------|
| Range       | Limited | Unlimited  | Unlimited       |
| Precision   | 52 bits | Arbitrary  | 52 bits (fixed) |
| Performance | Fastest | Slower     | Balanced        |
| Memory      | 8 bytes | Variable   | Variable        |
| IEEE 754    | Full    | Partial    | Full            |

## Configuration

### Backend Selection

```rust
use flexfloat::FlexFloat;
use flexfloat::bitarray::BoolBitArray;

// Use specific bit array implementation
type CustomFloat = FlexFloat<BoolBitArray>;
let num = CustomFloat::from(42.0);
```

## Technical Details

### Exponent Growth Algorithm

FlexFloat uses an adaptive exponent sizing algorithm:

1. **Start**: 11-bit exponent (IEEE 754 standard)
2. **Monitor**: Check if values fit in current range
3. **Expand**: Grow exponent field when overflow/underflow is detected
4. **Signed representation**: Uses offset bias (half the range) for efficient signed exponent storage

### Memory Layout

```text
FlexFloat<B> {
    sign: bool           // 1 bit
    exponent: B          // Variable width (≥11 bits)
    fraction: B          // Fixed 52 bits
}
```

### IEEE 754 Compatibility

- **Bit-perfect conversion** with standard f64
- **Special values** handled correctly

## Current Status

Current release highlights:

- ✅ Core FlexFloat structure
- ✅ BitArray trait and BoolBitArray implementation
- ✅ IEEE 754 conversion (to/from f64)
- ✅ Special values (±0, ±∞, NaN)
- ✅ Basic operations (negation, absolute value)
- ✅ Arithmetic operations (addition, subtraction, multiplication, division)
- ✅ Comparison operations (PartialEq, PartialOrd, classify, total_cmp)
- ✅ Mathematical functions (round, floor, ceil, exp, exp2, exp_m1, ln, ln_1p, sqrt, cbrt, hypot, trig, hyperbolic)
- ✅ f64-style helpers (`min`, `max`, `clamp`, `%`, `rem_euclid`, `div_euclid`, `powi`, `next_up`, `next_down`, `FromStr`)
- ✅ Additional packed backend coverage via `UsizeBitArray`
- ⏳ Further backend optimization work (SIMD, specialized storage)
- ⏳ Serialization support

## Contributing

Contributions are welcome! Areas where help is needed:

- **Optimized bit arrays**: Packed bit storage, SIMD operations
- **Performance optimization**: Benchmarking and profiling
- **Documentation**: Examples, tutorials, API documentation
- **Testing**: Edge cases, property-based testing
- **Serialization**: serde support for FlexFloat types

Development workflow details live in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- IEEE 754-2019 Standard for reference
- `num-bigint` crate for arbitrary precision integer support
- Rust community for excellent floating-point resources

---

**Note**: FlexFloat is designed for applications requiring extended-range floating-point arithmetic while maintaining precision consistency through a fixed 52-bit mantissa. For applications needing arbitrary precision with variable mantissa sizes, consider `BigDecimal` or similar libraries.

## Git Hook: Local Pre-Commit Checks

This repository includes a local Git hook that runs the same checks as the project's CI workflow (format check, clippy, tests, builds, and docs). The hook lives in `.githooks/pre-commit` and is not enabled by default.

To enable the hook locally run:

```bash
bash scripts/install-hooks.sh
```

Note: these checks may be slow because they run the full local quality suite.

## Release Workflow

This repository includes a manual GitHub Actions workflow at `.github/workflows/release.yml` for crates.io releases.

Before running it, add a `CARGO_REGISTRY_TOKEN` repository secret with a crates.io API token that has publish access.

The workflow only succeeds when dispatched from `main`.

`CHANGELOG.md` is managed manually as part of release preparation. The workflow verifies that an entry for the requested version already exists before it tags and publishes.

When you trigger the `Release` workflow, it will:

- bump `Cargo.toml`, the README dependency snippet, and the changelog heading for the requested version
- run formatting, clippy, tests, and `cargo publish --dry-run`
- build docs with rustdoc warnings denied
- create a release commit and `v<version>` tag
- push the commit and tag to `main`
- publish the crate to crates.io
