import PastaCurves.X86_64.Compositions
import PastaCurves.VectorCheck
import PastaCurves.Inversion.VectorCheck

/-!
# The x86-64 blocks on the reference vectors

These are cross-backend checks: the vectors in `Vectors.lean` were produced by the real AArch64
assembly, and the theorem below has the kernel evaluate the x86-64 transcription on the same
operands; they are not captures from x86-64 hardware. The multiplication vectors exercise the
standalone `mulMont` block. The squaring vectors exercise `sqrMont`, the Rust composition
`squareHi (squareLo value)`. The conversion vectors exercise x86-64's standalone `fromMont`
assembly block, rather than AArch64's multiplication by one composition. The inversion's blocks
are checked against the known answers of `Inversion/Vectors.lean`, which the integer model of the
algorithm computes, as AArch64's are.
-/

namespace PastaCurves.X86_64

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

/-! ## The inversion's blocks

The six blocks are checked against the known answers of `Inversion/Vectors.lean` through
`Inversion/VectorCheck.lean`, and the packed divstep round step by step against the crate's trace
of the first batch on the first vector. -/

-- One theorem per block, so that each kernel evaluation is a declaration of its own: an
-- independent checker such as nanoda frees what it builds for a declaration once it is
-- checked.

/-- The x86-64 `divstep59` block reproduces every known answer. -/
theorem divstep59_vectors_reproduced : invertBlocks.divstep59Failures = [] := by decide +kernel

/-- The x86-64 `sign_mag` block reproduces every known answer. -/
theorem signMag_vectors_reproduced : invertBlocks.signMagFailures = [] := by decide +kernel

/-- The x86-64 `fg_row` block reproduces every known answer. -/
theorem fgRow_vectors_reproduced : invertBlocks.fgRowFailures = [] := by decide +kernel

/-- The x86-64 `de_row` block reproduces every known answer. -/
theorem deRow_vectors_reproduced : invertBlocks.deRowFailures = [] := by decide +kernel

/-- The x86-64 `amontred` block reproduces every known answer. -/
theorem amontred_vectors_reproduced : invertBlocks.amontredFailures = [] := by decide +kernel

/-- The x86-64 `cond_sub` block reproduces every known answer. -/
theorem condSub_vectors_reproduced : invertBlocks.condSubFailures = [] := by decide +kernel

-- The evaluation names the known answers that fail, should any.
/-- info: ([], [], [], [], [], []) -/
#guard_msgs in
#eval invertBlocks.failures

/-- The state as `divstepRound` carries it: the packed state with the next step's decision applied,
as the block decides a batch's first step, from the sign of `two_delta` and the parity of `g`. A
round finishes the step and decides the next, so it maps one such state to the next. -/
def traceState (s : Nat × Nat × Nat) : DivstepState :=
  let k := cmovz (andw s.2.2 1) (cmovs (andw s.1 s.1) 18446744073709551614 0) 0
  ⟨eorw s.1 k, cmovb (bt k 63) s.2.1 s.2.2, s.2.2, eorw (cmovz (andw s.2.2 1) s.2.1 0) k, 0⟩

/-- The x86-64 round follows the crate's trace step by step. -/
theorem divstep_trace_reproduced : Inversion.divstepTraceFailures traceState divstepRound = [] := by
  decide +kernel

/-- info: [] -/
#guard_msgs in
#eval Inversion.divstepTraceFailures traceState divstepRound

end PastaCurves.X86_64
