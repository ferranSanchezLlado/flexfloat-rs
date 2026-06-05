#!/usr/bin/env bash
# Convert Criterion's estimates.json files into the bencher line format that
# github-action-benchmark (tool: cargo) understands:
#
#   test <name> ... bench: <mean_ns> ns/iter (+/- <std_dev_ns>)
#
# Usage: bash scripts/criterion-to-bencher.sh [criterion-dir]
#   criterion-dir defaults to target/criterion

set -euo pipefail

criterion_dir="${1:-target/criterion}"

if [ ! -d "$criterion_dir" ]; then
  echo "Criterion output directory not found: $criterion_dir" >&2
  exit 1
fi

# Walk every new/estimates.json and emit one bencher line per file.
# Path pattern (two depths):
#   <group>/<bench>/new/estimates.json          — no param  (e.g. flexfloat/bool/add)
#   <group>/<bench>/<param>/new/estimates.json  — with param (e.g. bitarray/bool/from_bits/64)
find "$criterion_dir" -path "*/new/estimates.json" | sort | while read -r file; do
  rel="${file#"$criterion_dir/"}"   # strip leading dir

  # Remove trailing /new/estimates.json
  path="${rel%/new/estimates.json}"

  # Build a slash-joined name from all path segments
  name="$path"

  # Extract mean and std_dev using python3 (available on all GitHub runners)
  read -r mean_int std_int < <(python3 -c "
import json, sys
with open('$file') as f:
    d = json.load(f)
mean = d['mean']['point_estimate']
std  = d['std_dev']['point_estimate']
print(round(mean), round(std))
")

  echo "test ${name} ... bench:        ${mean_int} ns/iter (+/- ${std_int})"
done
