#!/usr/bin/env bash
# Check that the Lean transcription of the crate's inline AArch64 Pasta Montgomery blocks
# is current: regenerating the Lean files from the crate's `asm!` blocks reproduces the
# committed files exactly, and the generated parts of the proofs in Spec.lean are the ones
# the generator produces.
#
# Run from the repository root; exits non-zero on violation.
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 lean/scripts/gen.py
git diff --exit-code -- \
  lean/PastaAsm/AArch64/Transcription.lean lean/PastaAsm/AArch64/Vectors.lean
python3 lean/scripts/gen.py --check-spec lean/PastaAsm/AArch64/Spec.lean
echo "Lean transcription: current."
