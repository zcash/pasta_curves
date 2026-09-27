#!/usr/bin/env bash
# Check that the generated Lean files are current: regenerating the transcriptions from the
# crate's `asm!` blocks reproduces the committed files exactly, the generated parts of the proofs
# in the `Spec/` files are the ones the generator produces, and the hull certificate's data module
# is what its JSON generates. The certificate JSON is also verified independently of Lean, in
# exact arithmetic.
#
# Run from the repository root; exits non-zero on violation.
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 lean/scripts/gen.py --check
python3 lean/scripts/gen.py --check-specs
PYTHONDONTWRITEBYTECODE=1 python3 lean/scripts/test_gen.py
PYTHONDONTWRITEBYTECODE=1 python3 lean/scripts/test_vectors.py
PYTHONDONTWRITEBYTECODE=1 python3 lean/scripts/test_asm_source.py
echo "Lean transcription: current."
python3 lean/scripts/gen_hull.py --check
python3 lean/scripts/verify_hull_certificate.py
