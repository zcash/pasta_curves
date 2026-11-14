#!/usr/bin/env bash
# Build the Lean package, failing on any warning, as `lake build --wfail` would. That includes
# warnings in Aeneas' library, which Lake normally fetches prebuilt from the pinned release but
# compiles when it cannot.
#
# Mathlib's prebuilt oleans are fetched first if they are missing for the pinned revision and
# toolchain (`lake exe cache get` is a no-op when they are present). Run from anywhere; LAKE
# selects lake (default: `lake` from PATH). Arguments are build targets (default: the package's
# default target). With `--check-log LOG`, it checks the build log LOG instead of building, which
# is how `test_build.sh` tests the check.
set -euo pipefail

# check_log LOG: fail if LOG has a warning. Lake may colour its output and overwrite progress
# lines, so ANSI escape sequences are removed and carriage returns read as line breaks before the
# matching.
check_log() {
  local esc
  esc=$(printf '\033')
  if tr '\r' '\n' < "$1" | sed "s/${esc}\[[0-9;?]*[A-Za-z]//g" | grep -E '^warning: '; then
    echo "error: the build printed the warnings above" >&2
    exit 1
  fi
}

if [ "${1:-}" = "--check-log" ]; then
  check_log "${2:?usage: build.sh --check-log LOG}"
  exit 0
fi

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
check_log "$log"
