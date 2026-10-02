#!/usr/bin/env bash
# Re-check the built library with nanoda, an independent reimplementation of the Lean
# kernel in Rust (https://github.com/ammkrn/nanoda_lib). The point is kernel-level
# defence in depth: a proof of False exploiting a kernel soundness bug would also have
# to get past a second, independently implemented checker (cf. the postmortem of
# lean4#14576, which recommends exactly this).
#
# Usage, from the `lean/` directory: scripts/check_nanoda.sh <lean4export-binary> <nanoda-binary>
# The library must already be built (`lake build`). Set LAKE to choose the lake that
# provides the export environment (default: `lake` from PATH).
#
# scripts/nanoda-config.json is the machine-checked axiom census: nanoda's strict mode
# fails on any axiom declared outside its `permitted_axioms` (the three standard axioms,
# and the ones Lean core declares unconditionally). scripts/check_export_axioms.py closes
# the remaining gap: the axioms that must be permitted because core declares them, but
# must remain unused (`sorryAx` in particular), are checked to be cited by nothing, so a
# `sorry` anywhere in the development fails here, and no `native_decide` is admitted.
set -euo pipefail

LEAN4EXPORT=${1:?usage: check_nanoda.sh <lean4export-binary> <nanoda-binary>}
NANODA=${2:?usage: check_nanoda.sh <lean4export-binary> <nanoda-binary>}
LAKE=${LAKE:-lake}

# The export root: the package's root module, which imports every other module (checked
# below), so a module cannot silently drop out of the re-check.
ROOTS=(PastaAsm)

python3 scripts/check_export_coverage.py "${ROOTS[@]}"

mkdir -p work
"$LAKE" env "$LEAN4EXPORT" "${ROOTS[@]}" > work/export.ndjson
python3 scripts/check_export_axioms.py scripts/nanoda-config.json
"$NANODA" scripts/nanoda-config.json
