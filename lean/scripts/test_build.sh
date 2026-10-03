#!/usr/bin/env bash
# Check that `build.sh` rejects a warning outside Aeneas' library: a throwaway module whose
# theorem uses `sorry` builds, with a warning, and the script must fail on it and say why.
#
# Run from anywhere; LAKE selects lake (default: `lake` from PATH).
set -euo pipefail
cd "$(dirname "$0")/.."

probe=PastaCurves/BuildWarningProbe
cleanup() {
  rm -f "$probe.lean"
  find .lake/build -path "*/$probe.*" -delete 2>/dev/null || true
}
trap cleanup EXIT

printf 'theorem buildWarningProbe : False := sorry\n' > "$probe.lean"
if out=$(scripts/build.sh "${probe//\//.}" 2>&1); then
  echo "$out"
  echo "error: build.sh accepted a module that warns" >&2
  exit 1
fi
grep -q "^warning: $probe.lean:.*sorry" <<< "$out" ||
  { echo "$out"; echo "error: build.sh failed, but not on the probe's warning" >&2; exit 1; }
grep -q "outside Aeneas' library" <<< "$out" ||
  { echo "$out"; echo "error: build.sh failed without its own message" >&2; exit 1; }
echo "build.sh rejects a warning outside Aeneas' library."
