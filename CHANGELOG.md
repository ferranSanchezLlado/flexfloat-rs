# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

## [1.0.0] - TBD

### Architecture restructure (1.0 plan)

#### Breaking changes

- **`FlexFloat<B>`** struct changed to **`FlexFloat<Exp, Frac = Exp>`** — two backend type
  parameters with a default.  The crate-root type alias
  `pub type FlexFloat = FlexFloat<DefaultBitArray>` preserves ergonomic use without turbofish.
- **`new_*` constructors removed** — use `zero()`, `nan()`, `pos_infinity()`, `neg_infinity()`,
  `infinity()`, `zero_with_sign()` directly.
- **`From<FlexFloat> for f64` / `BigInt` replaced by `TryFrom`** — returns
  `FlexFloatToF64Error` / `FlexFloatToIntError` instead of panicking.
- **`static_boolean_array::StaticBoolArray` removed** — replaced by
  `static_bit_array::StaticBitArray<const N: usize>`.
- **`bitarray::BitArrayPrimitives` no longer re-exported** from `bitarray/mod.rs` — use
  `bitarray::backend::BitArrayPrimitives` directly.

#### New features

- **`FlexFloat<Exp, Frac = Exp>`** dual-backend struct — exponent and fraction fields can
  have different `BitArray` backends.
- **`StaticBitArray<const N: usize>`** — const-generic zero-overhead backend for compile-time
  constants (`PI`, `E`, `TAU`, etc. are now typed as
  `FlexFloat<StaticBitArray<11>, StaticBitArray<52>>`).
- **Mixed-backend arithmetic** — `FlexFloat<B, B>` arithmetic accepts any
  `FlexFloat<Exp2, Frac2>` as the RHS (including const-backend constants) without
  `.convert_to()`.
- **New `From` impls** — `From<f32>`, `From<i64>`, `From<u64>`, `From<i32>`, `From<u32>`.
- **`Default for FlexFloat<B, B>`** — returns positive zero.
- **`Sum` / `Product` for iterators**.
- **`Neg for &FlexFloat<B, B>`** — reference negation without clone at the call site.
- **`mul_add(self, a, b)`** — fused multiply-add (software implementation).
- **`sin_cos(self)`** — computes sine and cosine in a single range-reduction pass.
- **`round_ties_even(self)`** — IEEE 754 round-to-nearest-even (banker's rounding).
- **`is_sign_positive()` / `is_sign_negative()`** — `f64`-parity aliases.
- **`to_le_bytes()` / `to_be_bytes()` / `from_le_bytes()` / `from_be_bytes()`** — byte
  serialisation/deserialisation.
- **Grown-aware instance methods** — `exponent_bits()`, `mantissa_digits()`, `min_exp()`,
  `max_exp()`, `epsilon()`.
- **`flexfloat::error`** module with `FlexFloatToF64Error`, `FlexFloatToIntError`, and
  `ConversionError`.
- **Module split** — `mod.rs` split into `construct.rs`, `classify.rs`, `accessors.rs`,
  `order.rs`, `parse.rs`, `internal.rs`; `arithmetic.rs` split into `arithmetic/` subdirectory
  with `normalize.rs`, `add_sub.rs`, `mul_div.rs`, `rem.rs`, `neg_abs.rs`, `sums.rs`.

## [1.0.0] - TBD

- First stable release of `flexfloat`.
- Commits to the documented `1.x` public surface: `FlexFloat`, `flexfloat::math`, `flexfloat::prelude`, the `BitArray` trait family, and the shipped `BoolBitArray`/`UsizeBitArray` backends.
- Documents the stable behavioral contract for extended-range arithmetic beyond `f64`'s exponent limits.
- Completes release packaging metadata, repository policy files, MSRV declaration, and release validation workflow hardening.

## [0.1.1] - 2026-05-18

- Current published crate release.
- Includes the core `FlexFloat` type, `BitArray` abstractions, arithmetic, comparisons, parsing, and the current math module surface.
- Serves as the baseline for the `1.0.0` stabilization review.
