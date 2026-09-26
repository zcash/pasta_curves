/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.AArch64.Compositions
import PastaAsm.VectorCheck

/-!
# The AArch64 blocks on the reference vectors

Each vector in `Vectors.lean` is the output of the real assembly (Semolina's `mul_mont_pasta`,
`sqr_mont_pasta`, and `from_mont_pasta`) on its operands, and the theorem below has the kernel
evaluate the transcription on the same operands. The multiplication and squaring vectors
exercise the inline blocks, which transcribe those routines; the conversion vectors exercise
`fromMont`, the multiplication block with `1` as its right operand, as the crate composes it.
-/

namespace PastaAsm.AArch64

/-- The AArch64 routines that the vectors exercise. -/
def vectorBackend : VectorBackend := ⟨mulMont, sqrMont, fromMont⟩

/-- The AArch64 blocks reproduce every reference vector. -/
theorem vectors_reproduced : vectorBackend.failures = ([], [], []) :=
  vectorBackend.failures_eq_nil
    (by intro k hk; unfold pieces at hk; interval_cases k <;> decide +kernel)
    (by decide +kernel) (by decide +kernel)

-- The evaluation names the vectors that fail, should any.
/-- info: ([], [], []) -/
#guard_msgs in
#eval vectorBackend.failures

end PastaAsm.AArch64
