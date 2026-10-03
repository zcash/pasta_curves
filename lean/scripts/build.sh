#!/usr/bin/env bash
# Build the Lean package, failing on any warning except those in Aeneas' library. The package
# requires that library but does not maintain it, and `lake build --wfail` would fail on its
# warnings too, since Lake has no per-package setting for them. A warning names its file first,
# relative to its package, so those of Aeneas' modules begin `Aeneas/`.
#
# Mathlib's prebuilt oleans are fetched first if they are missing for the pinned revision and
# toolchain (`lake exe cache get` is a no-op when they are present). Run from anywhere; LAKE
# selects lake (default: `lake` from PATH). Arguments are build targets (default: the package's
# default target).
set -euo pipefail
cd "$(dirname "$0")/.."

LAKE=${LAKE:-lake}
log=$(mktemp)
trap 'rm -f "$log"' EXIT

# Without the cache, the build would compile Mathlib from source; offline, a present cache still
# serves.
if ! "$LAKE" exe cache get; then
  echo "note: could not fetch Mathlib's cache; building with whatever is present" >&2
fi
"$LAKE" build "$@" 2>&1 | tee "$log"
if grep -E '^warning: ' "$log" | grep -v -E '^warning: Aeneas/'; then
  echo "error: the warnings above are outside Aeneas' library" >&2
  exit 1
fi
