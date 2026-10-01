import PastaCurves.AArch64.Compositions
import PastaCurves.VectorCheck
import PastaCurves.Inversion.Vectors

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

/-! ## The divstep block

The inversion has no shared vectors yet. The crate's tests check `divstep59` against the integer
divstep recurrence on a few inputs, and trace the first batch of the first one step by step
against the packed recurrence of `Inversion/Packed.lean`; the same values are checked against the
transcription here. -/

/-- The indices of the vectors that the block's transcription does not reproduce. -/
def divstep59Failures : List Nat :=
  (List.range Inversion.divstep59Vectors.length).filter fun i =>
    match Inversion.divstep59Vectors[i]? with
    | some (two_delta, f0, g0, res) => divstep59Block two_delta f0 g0 != res
    | none => false

theorem divstep59_vectors_reproduced : divstep59Failures = [] := by decide +kernel

/-- info: [] -/
#guard_msgs in
#eval divstep59Failures

/-- The packed state `(two_delta, f, g)` before each of the first batch's twenty steps on the first
vector, and after the last, as in the crate's `divstep_trace`. -/
def divstepTrace : List (Nat × Nat × Nat) := [
  (0x0000000000000001, 0xfffffe0000000001, 0xc0000000000d9046),
  (0x0000000000000003, 0xfffffe0000000001, 0xe00000000006c823),
  (0xffffffffffffffff, 0xe00000000006c823, 0xf000010000036411),
  (0x0000000000000001, 0xe00000000006c823, 0xe80000800005161a),
  (0x0000000000000003, 0xe00000000006c823, 0xf400004000028b0d),
  (0xffffffffffffffff, 0xf400004000028b0d, 0x0a00001ffffde175),
  (0x0000000000000001, 0xf400004000028b0d, 0xff00003000003641),
  (0x0000000000000001, 0xff00003000003641, 0x057ffff7fffed59a),
  (0x0000000000000003, 0xff00003000003641, 0x02bffffbffff6acd),
  (0xffffffffffffffff, 0x02bffffbffff6acd, 0x01dfffe5ffff9a46),
  (0x0000000000000001, 0x02bffffbffff6acd, 0x00effff2ffffcd23),
  (0x0000000000000001, 0x00effff2ffffcd23, 0xff17fffb8000312b),
  (0x0000000000000001, 0xff17fffb8000312b, 0xff14000440003204),
  (0x0000000000000003, 0xff17fffb8000312b, 0xff8a000220001902),
  (0x0000000000000005, 0xff17fffb8000312b, 0xffc5000110000c81),
  (0xfffffffffffffffd, 0xffc5000110000c81, 0x00568002c7ffedab),
  (0xffffffffffffffff, 0xffc5000110000c81, 0x000dc001ebfffd16),
  (0x0000000000000001, 0xffc5000110000c81, 0x0006e000f5fffe8b),
  (0x0000000000000001, 0x0006e000f5fffe8b, 0x0020effff2fff905),
  (0x0000000000000001, 0x0020effff2fff905, 0x000d07ff7e7ffd3d),
  (0x0000000000000001, 0x000d07ff7e7ffd3d, 0xfff60bffc5c0021c)]

/-- The state as `divstepRound` carries it: with the flags of the parity test of `g`, which the
block sets before a batch's first step and each step sets for the next. -/
def traceState (s : Nat × Nat × Nat) : DivstepState :=
  ⟨s.1, s.2.1, s.2.2, tstFlags (andw s.2.2 1)⟩

/-- The steps, numbered from one, after which the transcribed round's state differs from the
trace. -/
def divstepTraceFailures : List Nat :=
  (List.range (divstepTrace.length - 1)).filterMap fun i =>
    match divstepTrace[i]?, divstepTrace[i + 1]? with
    | some a, some b => if divstepRound (traceState a) != traceState b then some (i + 1) else none
    | _, _ => none

theorem divstep_trace_reproduced : divstepTraceFailures = [] := by decide +kernel

/-- info: [] -/
#guard_msgs in
#eval divstepTraceFailures

/-! ## The sign-magnitude block -/

/-- The indices of the vectors that the block's transcription does not reproduce. -/
def signMagFailures : List Nat :=
  (List.range Inversion.signMagVectors.length).filter fun i =>
    match Inversion.signMagVectors[i]? with
    | some (u, v, q, r, res) => signMagBlock u v q r != res
    | none => false

theorem signMag_vectors_reproduced : signMagFailures = [] := by decide +kernel

/-- info: [] -/
#guard_msgs in
#eval signMagFailures

/-! ## The `f`, `g` row block -/

/-- The indices of the vectors that the block's transcription does not reproduce. -/
def fgRowFailures : List Nat :=
  (List.range Inversion.fgRowVectors.length).filter fun i =>
    match Inversion.fgRowVectors[i]? with
    | some (f, g, m0, m1, s0, s1, res) => fgRowBlock f g m0 m1 s0 s1 != res
    | none => false

theorem fgRow_vectors_reproduced : fgRowFailures = [] := by decide +kernel

/-- info: [] -/
#guard_msgs in
#eval fgRowFailures

/-! ## The `d`, `e` row block -/

/-- The indices of the vectors that the block's transcription does not reproduce. -/
def deRowFailures : List Nat :=
  (List.range Inversion.deRowVectors.length).filter fun i =>
    match Inversion.deRowVectors[i]? with
    | some (d, e, m0, m1, s0, s1, res) => deRowBlock d e m0 m1 s0 s1 != res
    | none => false

theorem deRow_vectors_reproduced : deRowFailures = [] := by decide +kernel

/-- info: [] -/
#guard_msgs in
#eval deRowFailures

/-! ## The almost-Montgomery reduction block -/

/-- The indices of the vectors that the block's transcription does not reproduce. -/
def amontredFailures : List Nat :=
  (List.range Inversion.amontredVectors.length).filter fun i =>
    match Inversion.amontredVectors[i]? with
    | some (t, F, res) => amontredBlock t F.modulus F.inv != res
    | none => false

theorem amontred_vectors_reproduced : amontredFailures = [] := by decide +kernel

/-- info: [] -/
#guard_msgs in
#eval amontredFailures

/-! ## The conditional subtraction block -/

/-- The indices of the vectors that the block's transcription does not reproduce. -/
def condSubFailures : List Nat :=
  (List.range Inversion.condSubVectors.length).filter fun i =>
    match Inversion.condSubVectors[i]? with
    | some (x, F, res) => condSubBlock x F.modulus != res
    | none => false

theorem condSub_vectors_reproduced : condSubFailures = [] := by decide +kernel

/-- info: [] -/
#guard_msgs in
#eval condSubFailures

end PastaCurves.AArch64
