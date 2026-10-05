#!/usr/bin/env bash
# Regenerate the Aeneas translations of the crate's Rust that are listed at the end of this
# script. Each is a `Types.lean` and a `Funs.lean` under `lean/PastaCurves/`. With `--check`, fail
# instead if the committed files differ from a fresh translation.
#
# It runs the Charon and Aeneas binaries that `lean/scripts/fetch_aeneas.sh` puts in
# `lean/work/aeneas/`, or those that CHARON and AENEAS name. Either way, they must be built at the
# Aeneas revision that `lean/lakefile.toml` pins, because the translations must match the Lean
# library that they import. If `lean/work/aeneas/` holds another release, as after the pin moves,
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

# translate DIR CFG FEATURES START...: translate into `lean/PastaCurves/DIR/` the items reachable
# from the START patterns. The crate is compiled with the `--cfg` flags CFG and the cargo FEATURES,
# either of which may be empty. With `--check`, compare with the committed files instead.
translate() {
  local dir=$1 cfg=$2 features=$3
  shift 3
  local starts=() cargo_args=()
  local pattern
  for pattern in "$@"; do
    starts+=(--start-from "$pattern")
  done
  if [ -n "$features" ]; then
    cargo_args=(-- --features "$features")
  fi
  local out="$work/$dir"
  mkdir -p "$out"
  RUSTFLAGS="$cfg" CARGO_TARGET_DIR="$out/target" \
    "$CHARON" cargo --preset=aeneas "${starts[@]}" \
    --dest-file "$out/pasta_curves.llbc" "${cargo_args[@]}" > "$out/charon.log" 2>&1 ||
    { cat "$out/charon.log"; exit 1; }
  "$AENEAS" -backend lean -all-computable -dest "$out/lean" -subdir "PastaCurves/$dir" \
    -split-files "$out/pasta_curves.llbc" > "$out/aeneas.log" 2>&1 ||
    { cat "$out/aeneas.log"; exit 1; }

  local file fresh committed
  for file in Types.lean Funs.lean; do
    fresh="$out/lean/PastaCurves/$dir/$file"
    committed="lean/PastaCurves/$dir/$file"
    if $check; then
      if ! cmp -s "$fresh" "$committed"; then
        diff -u "$committed" "$fresh" || true
        echo "error: $committed differs from a fresh translation" >&2
        exit 1
      fi
    else
      mkdir -p "lean/PastaCurves/$dir"
      cp "$fresh" "$committed"
    fi
  done
  if $check; then
    echo "The translation in lean/PastaCurves/$dir/ is current."
  fi
}

# The portable inversion blocks and the driver that they run under, as the crate compiles them
# without the assembly backend; the entry point's debug assertion is not part of it.
translate Portable "--cfg pasta_curves_noasm" "" \
  crate::inversion::invert_with crate::inversion::portable

# The generic compositions that run the entry points over a backend's Montgomery blocks, as the
# crate compiles them with the assembly backend. They do not reach any backend's blocks, only the
# trait that declares them, so the translation is the same for every backend.
translate Glue "" asm \
  crate::asm::entry::add_with crate::asm::entry::sub_with crate::asm::entry::mul_with \
  crate::asm::entry::square_with crate::asm::entry::sqr_n_mul_with \
  crate::asm::entry::from_mont_with
