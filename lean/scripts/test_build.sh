#!/usr/bin/env bash
# Check that `build.sh` accepts a warning inside Aeneas' library and rejects one outside it. First
# its check runs on synthetic logs: a warning of Aeneas', and one outside Aeneas plain, coloured,
# and after a carriage return. Then a throwaway module whose theorem uses `sorry` builds, with a
# warning, and the script must fail on it and say why.
#
# Run from anywhere; LAKE selects lake (default: `lake` from PATH).
set -euo pipefail
cd "$(dirname "$0")/.."

probe=PastaCurves/BuildWarningProbe
logs=$(mktemp -d)
cleanup() {
  rm -f "$probe.lean"
  rm -rf "$logs"
  find .lake/build -path "*/$probe.*" -delete 2>/dev/null || true
}
trap cleanup EXIT

esc=$(printf '\033')
printf 'warning: Aeneas/Std/WP.lean:1:1: this tactic does nothing\n' > "$logs/aeneas"
printf 'warning: PastaCurves/X.lean:1:1: unused variable\n' > "$logs/plain"
printf '%s[33mwarning:%s[0m PastaCurves/X.lean:1:1: unused variable\n' "$esc" "$esc" \
  > "$logs/coloured"
printf 'Building PastaCurves.X\rwarning: PastaCurves/X.lean:1:1: unused variable\n' \
  > "$logs/overwritten"
scripts/build.sh --check-log "$logs/aeneas" > /dev/null ||
  { echo "error: build.sh rejected a warning inside Aeneas' library" >&2; exit 1; }
for kind in plain coloured overwritten; do
  if scripts/build.sh --check-log "$logs/$kind" > /dev/null 2>&1; then
    echo "error: build.sh accepted a $kind warning outside Aeneas' library" >&2
    exit 1
  fi
done

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
echo "build.sh accepts Aeneas' warnings and rejects any other."
