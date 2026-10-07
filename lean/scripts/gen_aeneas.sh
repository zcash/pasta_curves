#!/usr/bin/env bash
# Regenerate the Aeneas translations of the crate's Rust that are listed at the end of this
# script. Each is a `Types.lean` and a `Funs.lean` under `lean/PastaCurves/`. With `--check`, fail
# instead if the committed files differ from a fresh translation.
#
# It runs the Charon and Aeneas binaries that `lean/scripts/fetch_aeneas.sh` puts in
# `lean/work/aeneas/`, or those that CHARON and AENEAS name. Either way, they must be built at the
# Aeneas revision that `lean/lakefile.toml` pins, because the translations must match the Lean
# library that they import. If `lean/work/aeneas/` holds another release, as after the pin moves,
# the script fails rather than fetching the pinned one: run `fetch_aeneas.sh` again. It reports
# each step on stdout; with VERBOSE=1 it also streams Charon's and Aeneas' own output, which it
# otherwise shows only when they fail. Run from anywhere.
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

# log MESSAGE...: report progress on stdout; errors go to stderr.
log() {
  printf 'gen_aeneas: %s\n' "$*"
}

# run LOG COMMAND...: run COMMAND with its output in the file LOG, and show LOG if it fails. With
# VERBOSE=1, also stream the output to stdout as it runs.
run() {
  local log=$1
  shift
  if [ "${VERBOSE:-}" = 1 ]; then
    "$@" 2>&1 | tee "$log" || exit 1
  else
    "$@" > "$log" 2>&1 || { cat "$log"; exit 1; }
  fi
}

if $check; then
  log "checking that the committed translations are current"
else
  log "regenerating the translations"
fi
log "charon: $CHARON"
log "aeneas: $AENEAS"
if [ -f "$fetched/release" ] && [ "$CHARON" = "$fetched/charon" ]; then
  log "release: $(cat "$fetched/release")"
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
start=$SECONDS
translations=0

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
  local charon=("$CHARON" cargo --preset=aeneas "${starts[@]}"
    --dest-file "$out/pasta_curves.llbc" ${cargo_args[@]+"${cargo_args[@]}"})
  local aeneas=("$AENEAS" -backend lean -all-computable -dest "$out/lean"
    -subdir "PastaCurves/$dir" -split-files "$out/pasta_curves.llbc")
  local step=$SECONDS

  log "$dir: from $*"
  log "$dir: compiled with --cfg flags '${cfg:-none}' and features '${features:-none}'"
  log "$dir: charon extracts the crate's MIR to LLBC"
  run "$out/charon.log" env RUSTFLAGS="$cfg" CARGO_TARGET_DIR="$out/target" "${charon[@]}"
  log "$dir: charon done in $((SECONDS - step))s"
  step=$SECONDS
  log "$dir: aeneas translates the LLBC to Lean"
  run "$out/aeneas.log" "${aeneas[@]}"
  log "$dir: aeneas done in $((SECONDS - step))s"

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
      log "$dir: $committed is current"
    elif cmp -s "$fresh" "$committed"; then
      log "$dir: $committed is unchanged"
    else
      mkdir -p "lean/PastaCurves/$dir"
      cp "$fresh" "$committed"
      log "$dir: $committed is updated"
    fi
  done
  translations=$((translations + 1))
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

log "done: $translations translations in $((SECONDS - start))s"
