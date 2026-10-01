#!/usr/bin/env bash
# Regenerate the Aeneas translation of the portable inversion blocks, `lean/PastaCurves/Portable/`
# `Types.lean` and `Funs.lean`, from the driver in `src/inversion.rs` and the blocks in
# `src/inversion/portable.rs`; with `--check`, fail instead if the committed files differ from a
# fresh translation.
#
# It runs the Charon and Aeneas binaries that `lean/scripts/fetch_aeneas.sh` puts in
# `lean/work/aeneas/`, or those that CHARON and AENEAS name. Either way, they must be built at the
# Aeneas revision that `lean/lakefile.toml` pins, because the translation must match the Lean
# library that it imports. If `lean/work/aeneas/` holds another release, as after the pin moves,
# the script fails rather than fetching the pinned one: run `fetch_aeneas.sh` again. Run from
# anywhere.
set -euo pipefail
cd "$(dirname "$0")/../.."

fetched=$PWD/lean/work/aeneas
if [ -z "${CHARON:-}" ] || [ -z "${AENEAS:-}" ]; then
  pinned=$(lean/scripts/fetch_aeneas.sh --tag)
  if [ "$(cat "$fetched/release" 2>/dev/null)" != "$pinned" ]; then
    echo "error: $fetched does not hold Aeneas release $pinned;" \
      "lean/scripts/fetch_aeneas.sh fetches it" >&2
    exit 1
  fi
fi
CHARON=${CHARON:-$fetched/charon}
AENEAS=${AENEAS:-$fetched/aeneas}
for tool in "$CHARON" "$AENEAS"; do
  if ! command -v "$tool" > /dev/null; then
    echo "error: $tool is missing; lean/scripts/fetch_aeneas.sh fetches Charon and Aeneas" >&2
    exit 1
  fi
done
check=false
if [ "${1:-}" = "--check" ]; then
  check=true
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# The translation covers the driver and the portable blocks, as the crate compiles them without
# the assembly backend; the entry point's debug assertion is not part of it.
RUSTFLAGS="--cfg pasta_curves_noasm" CARGO_TARGET_DIR="$work/target" \
  "$CHARON" cargo --preset=aeneas \
  --start-from crate::inversion::invert_with --start-from crate::inversion::portable \
  --dest-file "$work/pasta_curves.llbc" > "$work/charon.log" 2>&1 ||
  { cat "$work/charon.log"; exit 1; }
"$AENEAS" -backend lean -dest "$work/lean" -subdir PastaCurves/Portable -split-files \
  "$work/pasta_curves.llbc" > "$work/aeneas.log" 2>&1 ||
  { cat "$work/aeneas.log"; exit 1; }

for file in Types.lean Funs.lean; do
  fresh="$work/lean/PastaCurves/Portable/$file"
  committed="lean/PastaCurves/Portable/$file"
  if $check; then
    if ! cmp -s "$fresh" "$committed"; then
      diff -u "$committed" "$fresh" || true
      echo "error: $committed differs from a fresh translation" >&2
      exit 1
    fi
  else
    cp "$fresh" "$committed"
  fi
done
if $check; then
  echo "The translation of the portable blocks is current."
fi
