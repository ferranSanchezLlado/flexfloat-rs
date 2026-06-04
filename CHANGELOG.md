# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

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
