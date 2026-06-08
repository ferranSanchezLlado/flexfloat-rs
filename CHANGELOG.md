# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Breaking changes

- **`to_le_bytes` / `to_be_bytes`** now return `(Vec<u8>, usize, usize)` — the third element is
  `frac_bits`, required by the updated deserialisation functions.
- **`from_le_bytes` / `from_be_bytes`** now take an additional `frac_bits: usize` parameter so
  grown-format values (with a fraction field wider than 52 bits) round-trip correctly.

### New features / behaviour

- **Variable-width fraction on overflow** — `build_finite_result` in `normalize.rs` now grows the
  fraction field alongside the exponent field when a result exceeds the IEEE 754 double-precision
  exponent range.  The grown format satisfies `n = 1 + e + m` where
  `e = round(log₂(n) × 1.5 + 2)`.  Previously the fraction was always truncated to 52 bits even
  for grown-exponent values.
- **`exponent_bits_for_total`** moved from `normalize.rs` into `internal.rs` and is now
  `pub(crate)`, making it available to all arithmetic kernels.  It is also re-exported from
  `flexfloat::mod` so existing internal call sites are unaffected.

### Fixes

- **`extract_mantissa_and_exponent`** — subnormal effective exponent is now computed dynamically
  as `-(2^(exp_bits-1)) + 2` instead of being hardcoded to `-1022`.  Values using a grown
  exponent field (wider than 11 bits) previously decoded subnormals with a wrong exponent.
- **`to_f64` / subnormal conversion** — `to_f64` now calls `self.min_exp()` instead of the
  hardcoded constant `-1022` when decoding subnormal values, consistent with the fix above.
- **Multiplication / division precision** — both kernels now call `align_mantissa` to widen
  operands to a common `mant_len` before multiplying or dividing.  The result is normalised
  relative to `mant_len` rather than the hardcoded index `52`, so grown-format operands
  preserve their full precision instead of being silently truncated.
- **`from_int` overflow guard** — the `missing_zeros` calculation in `from_int` no longer
  underflows when `fraction.len() > bit_length`.

### Internal

- **`add_sub.rs` cleanup** — import formatting fixed; `use num_bigint::Sign` removed after
  `exp_diff` conversion was simplified to a direct `BigInt::try_into().unwrap_or(saturation)`
  call; spurious `mut` dropped from the `rhs` parameter of `sub`; `debug_assert!` in `add`
  relocated to immediately after the operand-swap block; unnecessary braced scope around the
  post-subtraction rounding block removed.
- **`normalize.rs`** — `align_mantissa` extracted as a named helper (previously inlined at each
  call site); `build_finite_result` parameter renamed from `mantissa53` to `mantissa` to reflect
  the variable-width contract; new private helpers `unit_mantissa` and `frac_bits_for_exp_width`
  added.
- **`fraction_bits_for_total`** removed from `internal.rs` and its re-export from `mod.rs`
  (was unused).

## [1.0.1] - TBD

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
