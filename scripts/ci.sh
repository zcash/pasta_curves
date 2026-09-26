#!/usr/bin/env bash
# Run the checks CI runs, in one go: the crate's checks (.github/workflows/ci.yml and
# lints-stable.yml), the assembly backend's (asm.yml), the formalization's (lean.yml), and the
# workflow audit (zizmor.yml). Intended to be run interactively from a checkout: a check whose
# tool is not installed is skipped with a note on how to install it, and the first failing
# check stops the run. The checks that need another host (the 32-bit tests, the book, code
# coverage) are not mirrored.
#
# Usage, from anywhere in the checkout: scripts/ci.sh
# LAKE selects the lake that builds the formalization (default: `lake` from PATH, which
# should be the elan-managed one; see AGENTS.md).
set -euo pipefail
cd "$(dirname "$0")/.."

LAKE=${LAKE:-lake}
skipped=""

step() { printf '\n==> %s\n' "$1"; }
skip() {
  skipped="$skipped  $1"$'\n'
  printf 'skipped: %s\n%s\n' "$1" "$2"
}

# ---- The crate (ci.yml) ----
before=$(git status --porcelain)

for features in --all-features --no-default-features; do
  step "cargo build $features"
  cargo build $features
  step "cargo test --release $features"
  cargo test --release $features
done

step "with the assembly disabled, the tests pass and the backend has none to run"
for profile in "" --release; do
  out=$(RUSTFLAGS="--cfg pasta_curves_noasm" cargo test $profile --all-features 2>&1) ||
    { echo "$out"; exit 1; }
  if printf '%s\n' "$out" | grep -E '^test asm::'; then
    echo "backend tests ran with the assembly disabled"
    exit 1
  fi
done

step "the build and the tests changed no tracked file"
test "$before" = "$(git status --porcelain)"

step "bitrot: the benches build"
cargo build --benches --all-features

step "no-std targets"
for target in thumbv6m-none-eabi wasm32-unknown-unknown wasm32-wasip1; do
  if rustup target list --installed | grep -qx "$target"; then
    cargo build --target "$target" --no-default-features
    cargo build --target "$target" --no-default-features --features zeroize
  else
    skip "the no-std build for $target" "  install the target: rustup target add $target"
  fi
done

# ---- The assembly backend (asm.yml) ----
# On an AArch64 host the backend's tests run, and the run must report exactly as many passed
# as `src/asm` declares, so that a test that compiles out cannot pass silently.
if [ "$(uname -m)" = "arm64" ] || [ "$(uname -m)" = "aarch64" ]; then
  expected=$(grep -rh '^\s*#\[test\]' src/asm | wc -l | tr -d ' ')
  echo "tests in the backend: $expected"
  test "$expected" -gt 0
  for profile in "" --release; do
    step "cargo test $profile asm::"
    out=$(cargo test $profile asm:: 2>&1) || { echo "$out"; exit 1; }
    echo "$out"
    echo "$out" | grep -q "^test result: ok. $expected passed" ||
      { echo "expected exactly $expected tests of the backend to pass"; exit 1; }
  done

  step "no_std: build against core alone, with no std to fall back on"
  if rustup run nightly rustc --version >/dev/null 2>&1 &&
     rustup component list --toolchain nightly --installed | grep -q '^rust-src'; then
    cargo +nightly build --release --no-default-features \
      -Z build-std=core,compiler_builtins --target aarch64-apple-darwin
  else
    skip "the no_std build" \
      "  it needs a nightly toolchain with the library sources:
    rustup toolchain install nightly --profile minimal --component rust-src"
  fi
else
  echo "host: $(uname -sm); the backend's own tests run on AArch64 hosts only"
fi

# ---- Lints and documentation (lints-stable.yml, ci.yml) ----
step "cargo clippy"
cargo clippy --all-features --all-targets -- -D warnings

step "cargo doc"
cargo doc --all-features --document-private-items

step "cargo fmt"
cargo fmt -- --check

# ---- The workflow audit (zizmor.yml) ----
step "zizmor"
if command -v zizmor >/dev/null; then
  if [ -n "${GH_TOKEN:-}" ]; then
    zizmor --min-severity informational .github/workflows
  else
    zizmor --min-severity informational --offline .github/workflows
    echo "(the online audits, which CI runs with its token, need GH_TOKEN set)"
  fi
else
  skip "zizmor" \
    "  install it with one of: cargo install zizmor; brew install zizmor; pipx install zizmor
  (https://docs.zizmor.sh/)"
fi

# ---- The formalization (lean.yml) ----
step "lake build --wfail"
(cd lean && "$LAKE" build --wfail)

step "the transcription and the proof skeletons are current"
lean/scripts/check.sh

step "nanoda re-check"
lean4export=lean/work/lean4export/.lake/build/bin/lean4export
nanoda=lean/work/nanoda_lib/target/release/nanoda_bin
if [ -x "$lean4export" ] && [ -x "$nanoda" ]; then
  (cd lean && LAKE="$LAKE" scripts/check_nanoda.sh \
    work/lean4export/.lake/build/bin/lean4export work/nanoda_lib/target/release/nanoda_bin)
else
  tag=$(cut -d: -f2 lean/lean-toolchain)
  revision=$(grep -o -- '--revision=[0-9a-f]*' .github/workflows/lean.yml | cut -d= -f2)
  skip "the nanoda re-check" \
    "  build the two checkers under lean/work/ as CI does (see .github/workflows/lean.yml):
    cd lean
    git clone --depth 1 --branch $tag https://github.com/leanprover/lean4export work/lean4export
    (cd work/lean4export && $LAKE build)
    git clone --depth 1 --revision=${revision:-<see lean.yml>} \\
      https://github.com/ammkrn/nanoda_lib work/nanoda_lib
    cargo build --release --manifest-path work/nanoda_lib/Cargo.toml"
fi

printf '\n==> every check that ran passed'
if [ -n "$skipped" ]; then
  printf '; skipped:\n%s' "$skipped"
else
  printf '\n'
fi
