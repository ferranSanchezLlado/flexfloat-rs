#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

echo "Installing git hooks: setting core.hooksPath to .githooks"
git config core.hooksPath .githooks

if [ -f .githooks/pre-commit ]; then
  chmod +x .githooks/pre-commit || true
  echo "Installed .githooks/pre-commit (made executable)."
else
  echo "Warning: .githooks/pre-commit not found. Please ensure the file exists." >&2
fi

echo "Done. To enable hooks for this repo run:"
echo "  git config core.hooksPath .githooks"
