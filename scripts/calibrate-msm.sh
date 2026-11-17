#!/usr/bin/env bash
# Calibrate the GLV MSM's machine-dependent constants (src/glv/msm/calibration.rs) on this
# machine. It records the host, builds once, runs the timing grid
# (glv::msm::calibration::grid) at several thread counts, and writes one CSV to
# CALIBRATION_DIR. It then runs the fitter on every grid in that directory, which prints the
# constants it recommends and how each machine fares with the current and the recommended ones.
# To weigh several machines, copy their grids into one CALIBRATION_DIR.
#
# Grids are never committed or pushed: they describe the host (CPU, OS, kernel, libc). So
# CALIBRATION_DIR must lie outside the checkout, and the script refuses one inside it. Nothing
# under src/ is changed either: updating the constants is a reviewed commit of the constants,
# with the fitter's report (losses, machines and builds) in its body.
#
# Usage, from anywhere in the checkout, on a machine left otherwise idle: scripts/calibrate-msm.sh
#
# Environment, all optional:
#   CALIBRATION_DIR       where grids are written and read (default:
#                         ${XDG_DATA_HOME:-~/.local/share}/pasta_curves/calibration)
#   CALIBRATION_GLV       the GLV feature: "glv" (default) or "glv-eisenstein", which adds the
#                         Eisenstein recoding (the MSM does not use it yet)
#   CALIBRATION_THREADS   thread counts to measure (default: 1 2 4 8 and the core count)
#   CALIBRATION_BUILD     a label for the build, part of the file name (default: "asm", or
#                         "native" when RUSTFLAGS has -C target-cpu=native)
#   CALIBRATION_HOST      a label for the machine (default: the CPU model)
#   CALIBRATION_REPS, CALIBRATION_SIZES, CALIBRATION_SEED
#                         passed to the timing grid (see its module documentation)
#   RUSTFLAGS             passed to Cargo and recorded, e.g. "-C target-cpu=native"
#   PYTHON                the interpreter that runs the fitter (default: python3)
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
# The features the calibrated code runs with: the GLV MSM (glv, or glv-eisenstein, which implies
# it), parallel evaluation (multicore) and the assembly field arithmetic (asm). None is on by
# default, and the timing grid only exists with all of them. On a target without an assembly
# backend the grid records `backend: portable`.
GLV=${CALIBRATION_GLV:-glv}
case "$GLV" in
  glv | glv-eisenstein) ;;
  *)
    echo "error: CALIBRATION_GLV must be glv or glv-eisenstein, not $GLV" >&2
    exit 1
    ;;
esac
FEATURES="$GLV multicore asm"

cpu_model() {
  if [[ "$(uname -s)" == Darwin ]]; then
    sysctl -n machdep.cpu.brand_string
  else
    sed -n 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo | head -n 1
  fi
}

core_count() {
  if [[ "$(uname -s)" == Darwin ]]; then
    sysctl -n hw.ncpu
  else
    nproc
  fi
}

libc_version() {
  if [[ "$(uname -s)" == Darwin ]]; then
    echo "libSystem (macOS $(sw_vers -productVersion))"
  else
    ldd --version 2>&1 | head -n 1
  fi
}

slug() { tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/^-//; s/-$//'; }

cores=$(core_count)
threads=${CALIBRATION_THREADS:-$(printf '%s\n' 1 2 4 8 "$cores" | awk -v max="$cores" '$1 <= max' |
  sort -n -u | tr '\n' ' ')}
case "${RUSTFLAGS:-}" in
  *target-cpu=native*) default_build=native ;;
  *) default_build=asm ;;
esac
build=${CALIBRATION_BUILD:-$default_build}
if [[ $GLV == glv-eisenstein ]]; then
  build="$build-eisenstein"
fi
host=${CALIBRATION_HOST:-$(cpu_model | slug)}
dir=${CALIBRATION_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/pasta_curves/calibration}
mkdir -p "$dir"
dir=$(cd "$dir" && pwd -P)
checkout=$(pwd -P)
case "$dir/" in
  "$checkout"/*)
    rmdir "$dir" 2> /dev/null || true
    echo "error: CALIBRATION_DIR ($dir) is inside the checkout; grids must stay outside it" >&2
    exit 1
    ;;
esac
out="$dir/${host}-${build}-$(date -u +%Y-%m-%d).csv"
if [[ -e "$out" ]]; then
  echo "error: $out exists; remove it or set CALIBRATION_BUILD" >&2
  exit 1
fi

{
  echo "# host: $host"
  echo "# cpu: $(cpu_model)"
  echo "# cores: $cores"
  echo "# os: $(uname -srm)"
  echo "# libc: $(libc_version)"
  echo "# rustc: $(rustc -V)"
  echo "# features: $FEATURES"
  echo "# rustflags: ${RUSTFLAGS:-}"
  echo "# commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (modified)')"
  echo "# date: $(date -u +%Y-%m-%dT%H:%MZ)"
  echo "threads,terms,method,param,median_ns,reps"
} > "$out"
grep '^#' "$out"

echo "==> building"
cargo test --release --features "$FEATURES" --no-run -q

for t in $threads; do
  echo "==> timing grid, $t thread(s)"
  before=$(wc -l < "$out")
  RAYON_NUM_THREADS=$t CALIBRATION_GRID="$out" \
    cargo test --release --features "$FEATURES" -q glv::msm::calibration::grid \
    -- --ignored --test-threads=1
  if [[ $(wc -l < "$out") -le $before ]]; then
    echo "error: the timing grid wrote nothing for $t thread(s); is it compiled in?" >&2
    exit 1
  fi
done
echo "==> wrote $out"

echo "==> fitting"
"$PYTHON" scripts/msm_calibration_fit.py --this "$out" "$dir"/*.csv
