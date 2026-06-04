#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  printf 'Usage: %s <version>\n' "$0" >&2
  exit 1
fi

version="$1"

if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$ ]]; then
  printf 'Invalid version: %s\n' "$version" >&2
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

sed -Ei '0,/^version\s*=\s*"[^"]+"\s*$/s//version = "'"$version"'"/' Cargo.toml
sed -Ei '0,/flexfloat\s*=\s*"[^"]+"\s*/s//flexfloat = "'"$version"'"/' README.md

if grep -q '^## \[Unreleased\]' CHANGELOG.md; then
  sed -Ei '0,/^## \[Unreleased\]\s*$/s//## [Unreleased]\n\n## ['"$version"'] - TBD/' CHANGELOG.md
fi

printf 'Bumped crate version to %s in Cargo.toml, README.md, and CHANGELOG.md\n' "$version"
