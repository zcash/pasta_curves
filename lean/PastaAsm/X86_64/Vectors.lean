/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.X86_64.Compositions
import PastaAsm.VectorCheck

/-!
# The x86-64 blocks on the reference vectors

These are cross-backend checks: the vectors in `Vectors.lean` were produced by the real AArch64
assembly, and the theorem below has the kernel evaluate the x86-64 transcription on the same
operands; they are not captures from x86-64 hardware. The multiplication vectors exercise the
standalone `mulMont` block. The squaring vectors exercise `sqrMont`, the Rust composition
`squareHi (squareLo value)`. The conversion vectors exercise x86-64's standalone `fromMont`
assembly block, rather than AArch64's multiplication by one composition.
-/

namespace PastaAsm.X86_64

/-- The x86-64 routines that the vectors exercise. -/
def vectorBackend : VectorBackend := ⟨mulMont, sqrMont, fromMont⟩

/-- The x86-64 blocks reproduce every reference vector. -/
theorem vectors_reproduced : vectorBackend.failures = ([], [], []) :=
  vectorBackend.failures_eq_nil
    (by intro k hk; unfold pieces at hk; interval_cases k <;> decide +kernel)
    (by decide +kernel) (by decide +kernel)

-- The evaluation names the vectors that fail, should any.
/-- info: ([], [], []) -/
#guard_msgs in
#eval vectorBackend.failures

end PastaAsm.X86_64
