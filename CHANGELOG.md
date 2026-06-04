# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

- Refactored `bitarray` into a layered API: `BitArray` now excludes arithmetic, `BitArrayArith` covers `Add`/`Sub`/`Mul`/`Div`, and backend-author primitives moved under `bitarray::backend`.
- Moved `get_range` onto `BitArrayAccess`, tightened `BitArray` implementor requirements to `Default + PartialEq + Eq`, and updated shipped backends to the new backend-primitives surface.
- Switched internal rounding paths to in-place increment support, which removes the old `+ B::from_bits(&[true])` round-up pattern in `flexfloat` arithmetic.
- Prepared release-readiness metadata for the first stable release.
- Added an explicit MSRV policy and CI validation.
- Added missing repository policy and packaging files.

## [1.0.0] - TBD

- First stable release of `flexfloat`.
- Commits to the documented `1.x` public surface: `FlexFloat`, `flexfloat::math`, `flexfloat::prelude`, the `BitArray` trait family, and the shipped `BoolBitArray`/`UsizeBitArray` backends.
- Documents the stable behavioral contract for extended-range arithmetic beyond `f64`'s exponent limits.
- Completes release packaging metadata, repository policy files, MSRV declaration, and release validation workflow hardening.

## [0.1.1] - 2026-05-18

- Current published crate release.
- Includes the core `FlexFloat` type, `BitArray` abstractions, arithmetic, comparisons, parsing, and the current math module surface.
- Serves as the baseline for the `1.0.0` stabilization review.
