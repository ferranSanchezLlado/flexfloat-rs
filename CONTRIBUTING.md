# Contributing

## Local Setup

1. Install Rust `1.85` or newer.
2. Clone the repository.
3. Run `cargo test` to confirm the workspace builds locally.

## Required Checks

Run these commands before opening a change:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

## Optional Git Hook

Install the local pre-commit hook with:

```bash
bash scripts/install-hooks.sh
```

## Contribution Expectations

- Keep changes as small as possible while still solving the problem.
- Add or update tests when behavior changes.
- Keep README and rustdoc examples accurate when public behavior changes.
- Do not broaden the public API accidentally; stable surface changes should be deliberate.

## Public API Policy

For `1.x`, changes to these surfaces should be treated as stability-sensitive:

- `FlexFloat`
- `flexfloat::math`
- `flexfloat::prelude`
- `BitArray` and its companion traits
- `BoolBitArray`, `UsizeBitArray`, and `DefaultBitArray`

Hidden implementation helpers and internal modules are not considered stable extension points.

## Release Notes

User-facing changes should be reflected in `CHANGELOG.md` as part of release preparation.

## Release Preparation

Before running the manual release workflow:

1. Finalize the matching `CHANGELOG.md` entry.
2. Ensure `Cargo.toml` and the README dependency snippet are ready for the target version.
3. Run the full local validation suite.
4. Confirm the workflow is being dispatched from `main`.
