#!/usr/bin/env bash
# Check that the generated Lean files are current: regenerating the transcriptions from the
# crate's `asm!` blocks reproduces the committed files exactly, the generated parts of the proofs
# in the `Spec/` files are the ones the generator produces, and the hull certificate's data module
# is what its JSON generates. The certificate JSON is also verified independently of Lean, in
# exact arithmetic.
#
# Run from the repository root; exits non-zero on violation. PYTHON selects the interpreter
# (default: `python3` from PATH); no run leaves bytecode caches behind.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" lean/scripts/gen.py --check
"$PYTHON" lean/scripts/gen.py --check-specs
"$PYTHON" lean/scripts/test_gen.py
"$PYTHON" lean/scripts/test_vectors.py
"$PYTHON" lean/scripts/test_asm_source.py
echo "Lean transcription: current."
"$PYTHON" lean/scripts/gen_hull.py --check
"$PYTHON" lean/scripts/verify_hull_certificate.py
