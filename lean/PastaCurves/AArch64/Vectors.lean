import PastaCurves.AArch64.Compositions
import PastaCurves.VectorCheck
import PastaCurves.Inversion.VectorCheck

/-!
# The AArch64 blocks on the reference vectors

Each vector in `Vectors.lean` is the output of the real assembly (Semolina's `mul_mont_pasta`,
`sqr_mont_pasta`, and `from_mont_pasta`) on its operands, and the theorem below has the kernel
evaluate the transcription on the same operands. The multiplication and squaring vectors
exercise the inline blocks, which transcribe those routines; the conversion vectors exercise
`fromMont`, the multiplication block with `1` as its right operand, as the crate composes it.
-/

namespace PastaCurves.AArch64

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

/-! ## The inversion's blocks

The six blocks are checked against the known answers of `Inversion/Vectors.lean` through
`Inversion/VectorCheck.lean`, and the packed divstep round step by step against the crate's trace
of the first batch on the first vector. -/

-- One theorem per block, so that each kernel evaluation is a declaration of its own: an
-- independent checker such as nanoda frees what it builds for a declaration once it is
-- checked.

/-- The AArch64 `divstep59` block reproduces every known answer. -/
theorem divstep59_vectors_reproduced : invertBlocks.divstep59Failures = [] := by decide +kernel

/-- The AArch64 `sign_mag` block reproduces every known answer. -/
theorem signMag_vectors_reproduced : invertBlocks.signMagFailures = [] := by decide +kernel

/-- The AArch64 `fg_row` block reproduces every known answer. -/
theorem fgRow_vectors_reproduced : invertBlocks.fgRowFailures = [] := by decide +kernel

/-- The AArch64 `de_row` block reproduces every known answer. -/
theorem deRow_vectors_reproduced : invertBlocks.deRowFailures = [] := by decide +kernel

/-- The AArch64 `amontred` block reproduces every known answer. -/
theorem amontred_vectors_reproduced : invertBlocks.amontredFailures = [] := by decide +kernel

/-- The AArch64 `cond_sub` block reproduces every known answer. -/
theorem condSub_vectors_reproduced : invertBlocks.condSubFailures = [] := by decide +kernel

-- The evaluation names the known answers that fail, should any.
/-- info: ([], [], [], [], [], []) -/
#guard_msgs in
#eval invertBlocks.failures

/-- The state as `divstepRound` carries it: with the flags of the parity test of `g`, which the
block sets before a batch's first step and each step sets for the next. -/
def traceState (s : Nat × Nat × Nat) : DivstepState :=
  ⟨s.1, s.2.1, s.2.2, tstFlags (andw s.2.2 1)⟩

/-- The AArch64 round follows the crate's trace step by step. -/
theorem divstep_trace_reproduced : Inversion.divstepTraceFailures traceState divstepRound = [] := by
  decide +kernel

/-- info: [] -/
#guard_msgs in
#eval Inversion.divstepTraceFailures traceState divstepRound

end PastaCurves.AArch64
