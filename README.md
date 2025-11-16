# FlexFloat

A high-precision Rust library for arbitrary precision floating-point arithmetic with growable exponents and fixed-size fractions. FlexFloat extends IEEE 754 double-precision format to handle numbers beyond the standard range while maintaining computational efficiency and precision consistency.

[![Crates.io](https://img.shields.io/crates/v/flexfloat.svg)](https://crates.io/crates/flexfloat)
[![Documentation](https://docs.rs/flexfloat/badge.svg)](https://docs.rs/flexfloat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

FlexFloat provides a flexible floating-point arithmetic system that automatically adapts to the scale of your computations:

- **🔄 Growable Exponents**: Automatically expand exponent bit width when needed
- **📏 Fixed Precision**: Maintain consistent 52-bit mantissa for precision consistency  
- **🔗 IEEE 754 Compatible**: Full support for standard floating-point operations and special values
- **⚡ Performance Focused**: Efficient operations while maintaining arbitrary precision capabilities
- **🛠️ Extensible**: Configurable bit array backends for different performance needs

## 🚀 Quick Start

Add FlexFloat to your `Cargo.toml`:

```toml
[dependencies]
flexfloat = "0.1.0"
```

### Basic Usage

```rust
use flexfloat::FlexFloat;

// Create FlexFloat from standard types
let x = FlexFloat::from(3.14159);
let y = FlexFloat::from(2.71828);

// Perform arithmetic operations
let sum = x + y;  // Planned
let neg_x = -x;
let abs_y = y.abs();

// Work with special values
let zero = FlexFloat::zero();
let infinity = FlexFloat::pos_infinity();
let nan = FlexFloat::nan();

// Convert back to f64 when in range
let result: f64 = sum.into();
```

### Advanced Examples

```rust
use flexfloat::{FlexFloat, bitarray::{BitArray, BoolBitArray}};

// Inspect internal representation
let num = FlexFloat::from(123.456);
println!("Sign: {}", num.sign());
println!("Exponent bits: {}", num.exponent().len());  
println!("Fraction bits: {}", num.fraction().len());

// Work with bit arrays directly
let custom_bits = BoolBitArray::from_bits(&[true, false, true]);
let from_bytes = BoolBitArray::from_bytes(&[0xFF, 0x00], 16);

// Handle special cases
if num.is_nan() {
    println!("Not a number");
} else if num.is_infinity() {
    println!("Infinite value");
} else if num.is_zero() {
    println!("Zero value");
}
```

## 🏗️ Architecture

FlexFloat is built around two main components:

### BitArray Module

Provides flexible bit manipulation with multiple implementations:

- **BoolBitArray**: Vector of booleans for simplicity and debugging
- **Extensible design**: Easy to add optimized implementations (packed bits, SIMD, etc.)
- **Rich conversion utilities**: Support for bytes, BigInt, f64, and more

### FlexFloat Module  

Core floating-point implementation featuring:

- **Variable exponent**: Starts at 11 bits, grows automatically
- **Fixed mantissa**: 52 bits for IEEE 754 compatibility
- **Special values**: Full support for ±0, ±∞, and NaN
- **Generic backend**: Works with any BitArray implementation

## 📊 Comparison

| Feature     | f64     | BigDecimal | FlexFloat       |
| ----------- | ------- | ---------- | ----------------|
| Range       | Limited | Unlimited  | Unlimited       |
| Precision   | 52 bits | Arbitrary  | 52 bits (fixed) |
| Performance | Fastest | Slower     | Balanced        |
| Memory      | 8 bytes | Variable   | Variable        |
| IEEE 754    | Full    | Partial    | Full            |

## 🎛️ Configuration

### Custom Bit Array Backends

```rust
use flexfloat::FlexFloat;
use flexfloat::bitarray::BoolBitArray;

// Use specific bit array implementation
type CustomFloat = FlexFloat<BoolBitArray>;
let num = CustomFloat::from(42.0);
```

## 🔬 Technical Details

### Exponent Growth Algorithm

FlexFloat uses an adaptive exponent sizing algorithm:

1. **Start**: 11-bit exponent (IEEE 754 standard)
2. **Monitor**: Check if values fit in current range
3. **Expand**: Grow exponent field when overflow detected  
4. **Two's complement**: Use signed representation for efficiency

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
- **Rounding modes** preserved (planned)
- **Exception handling** IEEE compliant (planned)

## 🚧 Current Status

**This library is currently in development.** Implemented features:

- ✅ Core FlexFloat structure
- ✅ BitArray trait and BoolBitArray implementation  
- ✅ IEEE 754 conversion (to/from f64)
- ✅ Special values (±0, ±∞, NaN)
- ✅ Basic operations (negation, absolute value)
- ⏳ Arithmetic operations (addition, subtraction, multiplication, division)
- ⏳ Comparison operations
- ⏳ Mathematical functions (sqrt, exp, log, trig)
- ⏳ Optimized bit array implementations
- ⏳ Serialization support

## 🤝 Contributing

Contributions are welcome! Areas where help is needed:

- **Arithmetic implementations**: Addition, subtraction, multiplication, division
- **Optimized bit arrays**: Packed bit storage, SIMD operations  
- **Mathematical functions**: Elementary functions, special functions
- **Performance optimization**: Benchmarking and profiling
- **Documentation**: Examples, tutorials, API documentation
- **Testing**: Edge cases, property-based testing

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- IEEE 754-2019 Standard for reference
- `num-bigint` crate for arbitrary precision integer support
- Rust community for excellent floating-point resources

---

**Note**: FlexFloat is designed for applications requiring extended range floating-point arithmetic while maintaining precision consistency. For applications needing arbitrary precision (variable mantissa), consider `BigDecimal` or similar libraries.