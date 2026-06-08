# FlexFloat

A high-precision Rust library for arbitrary-precision floating-point arithmetic with growable exponents and fractions. FlexFloat extends IEEE 754 double-precision format to handle numbers far beyond the standard range while maintaining computational efficiency and precision consistency.

[![Crates.io](https://img.shields.io/crates/v/flexfloat.svg)](https://crates.io/crates/flexfloat)
[![Documentation](https://docs.rs/flexfloat/badge.svg)](https://docs.rs/flexfloat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

FlexFloat automatically adapts to the scale of your computations:

- **Growable exponents and fractions** — both fields expand together automatically when values exceed the current range
- **52-bit mantissa by default** — IEEE 754-compatible precision; grows alongside the exponent on overflow
- **Full IEEE 754 special values** — ±0, ±∞, NaN
- **Backend-generic** — the `FlexFloat<Exp, Frac>` struct is generic over `BitArray` implementations; `FlexFloat` (no params) is a type alias for `FlexFloat<BoolBitArray>`

## Quick Start

```toml
[dependencies]
flexfloat = "1.1.0"
```

```rust
use flexfloat::prelude::*;

// Works like f64 — use From to construct, then standard operators
let a = FlexFloat::from(f64::MAX);
let b = FlexFloat::from(f64::MAX);
let huge = a * b;

// Unlike f64, this never overflows to inf — the range grows automatically
assert!(!huge.is_infinite());

// Convert back to f64 when you need to; returns Err if the value no longer fits
let result: Result<f64, _> = huge.try_into();
assert!(result.is_err());
```

See the [`examples/`](examples/) directory for more (`cargo run --example <name>`).

## Conversions

### Into `FlexFloat` — lossless `From`

```rust
use flexfloat::FlexFloat;
use num_bigint::BigInt;

let _ = FlexFloat::from(1.5_f64);
let _ = FlexFloat::from(1.0_f32);
let _ = FlexFloat::from(42_i64);
let _ = FlexFloat::from(42_u64);
let _ = FlexFloat::from(42_i32);
let _ = FlexFloat::from(42_u32);
let _ = FlexFloat::from(BigInt::from(12345));
```

### Out of `FlexFloat` — fallible `TryFrom`

Conversions to `f64` and `BigInt` are fallible and return typed errors:

```rust
use flexfloat::{FlexFloat, FlexFloatToF64Error};

let x = FlexFloat::from(f64::MAX) * FlexFloat::from(f64::MAX); // exponent grew
let result: Result<f64, FlexFloatToF64Error> = x.try_into();
assert!(result.is_err()); // value is outside f64 range

let y = FlexFloat::from(1.5_f64);
let f: f64 = y.try_into().unwrap();
assert_eq!(f, 1.5);
```

> **Note**: `MIN`, `MAX`, and `EPSILON` associated constants are intentionally absent —
> they are meaningless for an unlimited-range type.  Use the grown-aware instance methods
> `exponent_bits()`, `mantissa_digits()`, `min_exp()`, `max_exp()`, and `epsilon()` instead.

## Math functions

| Category      | Functions                                                          |
|---------------|--------------------------------------------------------------------|
| Rounding      | `round`, `floor`, `ceil`, `trunc`, `fract`, `round_ties_even`      |
| Exponential   | `exp`, `exp2`, `exp_m1`, `ln`, `ln_1p`, `log`, `log2`, `log10`    |
| Power/root    | `pow`, `sqrt`, `cbrt`, `hypot`, `powi`, `powf`                     |
| Trigonometry  | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sin_cos`    |
| Hyperbolic    | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`                  |
| Utility       | `signum`, `copysign`, `recip`, `abs`, `mul_add`, `to_degrees`, `to_radians` |
| Comparison    | `min`, `max`, `clamp`, `total_cmp`, `next_up`, `next_down`         |

All functions are available as free functions in `flexfloat::math` and as methods on `FlexFloat`.

## Const-context constants

```rust
use flexfloat::flexfloat::consts;
// PI, TAU, E, FRAC_PI_2, SQRT_2, LN_2, etc. are all available as
// FlexFloat<StaticBitArray<11>, StaticBitArray<52>> — zero-overhead const types
// that work directly in arithmetic expressions.
```

## Grown-aware instance methods

```rust
use flexfloat::prelude::*;

let x = FlexFloat::from(1.0_f64);
assert_eq!(x.exponent_bits(), 11);       // standard IEEE 754 exponent width
assert_eq!(x.mantissa_digits(), 53);      // 52 fraction bits + 1 implicit
assert_eq!(x.epsilon(), FlexFloat::from(f64::EPSILON));

// After an overflow-growing operation:
let huge = FlexFloat::from(f64::MAX) * FlexFloat::from(f64::MAX);
assert!(huge.exponent_bits() > 11);
println!("max_exp = {}", huge.max_exp());
```

## Byte serialisation

```rust
use flexfloat::FlexFloat;

let x = FlexFloat::from(3.14_f64);
let (le_bytes, exp_bits, frac_bits) = x.to_le_bytes();
let restored = FlexFloat::from_le_bytes(&le_bytes, exp_bits, frac_bits);
assert_eq!(x, restored);
```

## Iterator support — `Sum` and `Product`

```rust
use flexfloat::FlexFloat;

let values: Vec<FlexFloat> = vec![FlexFloat::from(1.0), FlexFloat::from(2.0), FlexFloat::from(3.0)];
let sum: FlexFloat = values.iter().sum();
assert_eq!(sum, FlexFloat::from(6.0));

let product: FlexFloat = values.iter().product();
assert_eq!(product, FlexFloat::from(6.0));
```

## Architecture

```
src/
├── lib.rs                  # crate doc, FlexFloat alias, re-exports, prelude
├── bitarray/               # pluggable bit-array backends
│   ├── boolean_list.rs     # BoolBitArray (default)
│   ├── usize_list.rs       # UsizeBitArray (word-packed)
│   └── static_bit_array.rs # StaticBitArray<N> (const-context)
└── flexfloat/
    ├── mod.rs              # FlexFloat<Exp, Frac = Exp> struct
    ├── construct.rs        # zero/nan/pos_infinity/…/Default
    ├── classify.rs         # is_nan/classify/exponent_bits/…
    ├── accessors.rs        # sign/exponent/fraction/is_sign_*
    ├── order.rs            # min/max/clamp/next_up/adjacent
    ├── cmp.rs              # PartialEq/PartialOrd/total_cmp
    ├── converter.rs        # From<f64/f32/i*/u*>/TryFrom/bytes
    ├── error.rs            # FlexFloatToF64Error/FlexFloatToIntError
    ├── consts.rs           # PI/E/TAU/… as StaticBitArray constants
    ├── arithmetic/         # Add/Sub/Mul/Div/Rem/Neg/Sum/Product
    └── math/               # transcendental functions
```

## Comparison

| Feature     | `f64`   | `BigDecimal` | `FlexFloat`                   |
|-------------|---------|--------------|-------------------------------|
| Range       | Limited | Unlimited    | Unlimited                     |
| Precision   | 52 bits | Arbitrary    | 52 bits (grows with exponent) |
| Performance | Fastest | Slower       | Balanced                      |
| Memory      | 8 bytes | Variable     | Variable                      |
| IEEE 754    | Full    | Partial      | Full                          |

## Contributing

Development workflow details live in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT — see the [LICENSE](LICENSE) file for details.

## Git Hook: Local Pre-Commit Checks

```bash
bash scripts/install-hooks.sh
```

## Release Workflow

See `.github/workflows/release.yml`. Requires a `CARGO_REGISTRY_TOKEN` secret.
`CHANGELOG.md` must have an entry for the target version before triggering the workflow.
