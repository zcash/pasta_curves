import PastaCurves.AArch64.Compositions
import PastaCurves.VectorCheck

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

/-- `(d, f0, g0)` and the expected `(d', m00, m01, m10, m11)`, as in the crate's
`divstep59_known_answers`. -/
def divstep59Vectors : List (Nat × Nat × Nat × Divstep59Result) := [
  (0x0000000000000001, 0x992d30ed00000001, 0x2a5f8c1b7e3d9046,
    ⟨0x0000000000000001, 0xffffffffd94098a0, 0x000000001c166090, 0xffffffffc77ed7e2,
      0xfffffffff41aad25⟩),
  (0x0000000000000001, 0xffffffffffffffff, 0x8000000000000001,
    ⟨0x0000000000000073, 0xfc00000000000000, 0x0400000000000000, 0xffffffffffffffff,
      0xffffffffffffffff⟩),
  (0xfffffffffffffffb, 0x1234567890abcdef, 0xfedcba0987654321,
    ⟨0x000000000000000d, 0xfffffffee6d31a00, 0x000000014cbb7a00, 0xfffffffff5435e89,
      0x00000000056bfef9⟩),
  (0x0000000000000011, 0x0000000000000001, 0x0000000000000000,
    ⟨0x0000000000000087, 0x0800000000000000, 0x0000000000000000, 0x0000000000000000,
      0x0000000000000001⟩),
  (0xfffffffffffffb65, 0xdeadbeefcafef00d, 0x0123456789abcdef,
    ⟨0xfffffffffffffbdb, 0x0800000000000000, 0x0000000000000000, 0x025c39e1b6d34515,
      0x0000000000000001⟩),
  (0x0000000000000007, 0xc8f1e2d3b4a59687, 0x1e2d3c4b5a697887,
    ⟨0x0000000000000007, 0xffffffffb89e1e30, 0xfffffffe50f081d0, 0xffffffffff43c3c5,
      0xffffffffdede823b⟩)]

/-- The indices of the vectors that the block's transcription does not reproduce. -/
def divstep59Failures : List Nat :=
  (List.range divstep59Vectors.length).filter fun i =>
    match divstep59Vectors[i]? with
    | some (d, f0, g0, r) => divstep59Block d f0 g0 != r
    | none => false

theorem divstep59_vectors_reproduced : divstep59Failures = [] := by decide +kernel

/-- info: [] -/
#guard_msgs in
#eval divstep59Failures

/-- The packed state `(d, f, g)` before each of the first batch's twenty steps on the first
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

end PastaCurves.AArch64
