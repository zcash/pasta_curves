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

/-- `(two_delta, f0, g0)` and the expected `(two_delta_new, u, v, q, r)`, as in the crate's
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

/-- Matrix entries as words and the expected magnitudes and masks, as in the crate's
`sign_mag_known_answers`. -/
def signMagVectors : List (Nat × Nat × Nat × Nat × SignMag) := [
  (0xffce000000000000, 0x004a000000000000, 0xffffffffffffffe5, 0xffffffffffffffff,
    ⟨0x0032000000000000, 0x004a000000000000, 0x000000000000001b, 0x0000000000000001,
      0xffffffffffffffff, 0x0000000000000000, 0xffffffffffffffff, 0xffffffffffffffff⟩),
  (0x0000000000000000, 0x0000008000000000, 0xfffffffffff00000, 0x00000058b6db6db7,
    ⟨0x0000000000000000, 0x0000008000000000, 0x0000000000100000, 0x00000058b6db6db7,
      0x0000000000000000, 0x0000000000000000, 0xffffffffffffffff, 0x0000000000000000⟩),
  (0x0800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000001,
    ⟨0x0800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000001,
      0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000⟩),
  (0xffffffffc5706190, 0x000000004e1e1b64, 0xfffffffffe7f182c, 0xffffffffdf08984b,
    ⟨0x000000003a8f9e70, 0x000000004e1e1b64, 0x000000000180e7d4, 0x0000000020f767b5,
      0xffffffffffffffff, 0x0000000000000000, 0xffffffffffffffff, 0xffffffffffffffff⟩),
  (0xffffffffb32e679c, 0xffffffffb4ff6206, 0xffffffffe38509c2, 0xffffffffc9886ce5,
    ⟨0x000000004cd19864, 0x000000004b009dfa, 0x000000001c7af63e, 0x000000003677931b,
      0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff⟩),
  (0x8000000000000001, 0x7fffffffffffffff, 0x0000000000000000, 0xffffffffffffffff,
    ⟨0x7fffffffffffffff, 0x7fffffffffffffff, 0x0000000000000000, 0x0000000000000001,
      0xffffffffffffffff, 0x0000000000000000, 0x0000000000000000, 0xffffffffffffffff⟩)]

/-- The indices of the vectors that the block's transcription does not reproduce. -/
def signMagFailures : List Nat :=
  (List.range signMagVectors.length).filter fun i =>
    match signMagVectors[i]? with
    | some (u, v, q, r, res) => signMagBlock u v q r != res
    | none => false

theorem signMag_vectors_reproduced : signMagFailures = [] := by decide +kernel

/-- info: [] -/
#guard_msgs in
#eval signMagFailures

/-! ## The `f`, `g` row block -/

/-- `f`, `g`, a matrix row's magnitudes and masks, and the expected row of the update, as in the
crate's `fg_row_known_answers`. -/
def fgRowVectors : List (Signed5 × Signed5 × Nat × Nat × Nat × Nat × Signed5) := [
  (⟨0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000,
    0x0000000000000000⟩, ⟨0xd83bd700ffffffe5, 0x628ddd6b04e1ba16, 0xfffffffffffffffc,
    0x3fffffffffffffff, 0x0000000000000000⟩, 0x0032000000000000, 0x004a000000000000,
    0xffffffffffffffff, 0x0000000000000000, ⟨0x66d2cf12ffffffff, 0xddb96703f6b306e4,
    0xffffffffffffffff, 0x00bfffffffffffff, 0x0000000000000000⟩),
  (⟨0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000,
    0x0000000000000000⟩, ⟨0xd83bd700ffffffe5, 0x628ddd6b04e1ba16, 0xfffffffffffffffc,
    0x3fffffffffffffff, 0x0000000000000000⟩, 0x000000000000001b, 0x0000000000000001,
    0xffffffffffffffff, 0xffffffffffffffff, ⟨0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0xffffffffffffff20, 0xffffffffffffffff⟩),
  (⟨0x66d2cf12ffffffff, 0xddb96703f6b306e4, 0xffffffffffffffff, 0x00bfffffffffffff,
    0x0000000000000000⟩, ⟨0xffffffffff900000, 0xffffffffffffffff, 0xffffffffffffffff,
    0xffffffffffffffff, 0xffffffffffffffff⟩, 0x0000000000000000, 0x0000008000000000,
    0x0000000000000000, 0x0000000000000000, ⟨0xfffffffffffffff9, 0xffffffffffffffff,
    0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff⟩),
  (⟨0x66d2cf12ffffffff, 0xddb96703f6b306e4, 0xffffffffffffffff, 0x00bfffffffffffff,
    0x0000000000000000⟩, ⟨0xffffffffff900000, 0xffffffffffffffff, 0xffffffffffffffff,
    0xffffffffffffffff, 0xffffffffffffffff⟩, 0x0000000000100000, 0x00000058b6db6db7,
    0xffffffffffffffff, 0x0000000000000000, ⟨0xf81299f237325a5d, 0x0000000000448d31,
    0x0000000000000000, 0xfffffffffffe8000, 0xffffffffffffffff⟩),
  (⟨0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000,
    0x0000000000000000⟩, ⟨0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000⟩, 0x0800000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, ⟨0x992d30ed00000001, 0x224698fc094cf91b,
    0x0000000000000000, 0x4000000000000000, 0x0000000000000000⟩),
  (⟨0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000,
    0x0000000000000000⟩, ⟨0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000⟩, 0x0000000000000000, 0x0000000000000001,
    0x0000000000000000, 0x0000000000000000, ⟨0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000⟩),
  (⟨0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000,
    0x0000000000000000⟩, ⟨0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000⟩, 0x0800000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, ⟨0x992d30ed00000001, 0x224698fc094cf91b,
    0x0000000000000000, 0x4000000000000000, 0x0000000000000000⟩),
  (⟨0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000,
    0x0000000000000000⟩, ⟨0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000⟩, 0x0000000000000000, 0x0000000000000001,
    0x0000000000000000, 0x0000000000000000, ⟨0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000⟩),
  (⟨0xdb241905fa248697, 0xfffffffffffffff9, 0x872517d55c1331ff, 0xfffffffffffffff4,
    0xffffffffffffffff⟩, ⟨0x599ff9f880e4b2a4, 0xfffffffffffffffe, 0xeb5679967b925eff,
    0xfffffffffffffffc, 0xffffffffffffffff⟩, 0x000000003a8f9e70, 0x000000004e1e1b64,
    0xffffffffffffffff, 0x0000000000000000, ⟨0x0000001cdd290a4d, 0x3d11b618fe078000,
    0x00000035e519a92f, 0x0000000000000000, 0x0000000000000000⟩),
  (⟨0xdb241905fa248697, 0xfffffffffffffff9, 0x872517d55c1331ff, 0xfffffffffffffff4,
    0xffffffffffffffff⟩, ⟨0x599ff9f880e4b2a4, 0xfffffffffffffffe, 0xeb5679967b925eff,
    0xfffffffffffffffc, 0xffffffffffffffff⟩, 0x000000000180e7d4, 0x0000000020f767b5,
    0xffffffffffffffff, 0xffffffffffffffff, ⟨0x00000007f42196ae, 0xc132e71048cda000,
    0x0000000ed9e178b1, 0x0000000000000000, 0x0000000000000000⟩),
  (⟨0xe71681d6d322b145, 0x000000000000577d, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000⟩, ⟨0x74b8fd762c36d47e, 0x00000000000014ae, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000⟩, 0x000000004cd19864, 0x000000004b009dfa,
    0xffffffffffffffff, 0xffffffffffffffff, ⟨0xfffbf5fa922aef71, 0xffffffffffffffff,
    0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff⟩),
  (⟨0xe71681d6d322b145, 0x000000000000577d, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000⟩, ⟨0x74b8fd762c36d47e, 0x00000000000014ae, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000⟩, 0x000000001c7af63e, 0x000000003677931b,
    0xffffffffffffffff, 0xffffffffffffffff, ⟨0xfffe3bb7def33478, 0xffffffffffffffff,
    0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff⟩)]

/-- The indices of the vectors that the block's transcription does not reproduce. -/
def fgRowFailures : List Nat :=
  (List.range fgRowVectors.length).filter fun i =>
    match fgRowVectors[i]? with
    | some (f, g, m0, m1, s0, s1, res) => fgRowBlock f g m0 m1 s0 s1 != res
    | none => false

theorem fgRow_vectors_reproduced : fgRowFailures = [] := by decide +kernel

/-- info: [] -/
#guard_msgs in
#eval fgRowFailures

/-! ## The `d`, `e` row block -/

/-- `d`, `e`, a matrix row's magnitudes and masks, and the expected row combination, as in the
crate's `de_row_known_answers`. -/
def deRowVectors : List (Limbs × Limbs × Nat × Nat × Nat × Nat × Signed5) := [
  (⟨0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000⟩,
    ⟨0x9a5f583ce5084635, 0x4f417e233776c195, 0x74634b1a733f7785, 0x1c51de5ea66f0f25⟩,
    0x0032000000000000, 0x004a000000000000, 0xffffffffffffffff, 0x0000000000000000,
    ⟨0x4b52000000000000, 0xf53e9f8f819a3464, 0x8c88e8ee762e0853, 0x60d3a4b3b5a55058,
    0x00082faa475c1c1a⟩),
  (⟨0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000⟩,
    ⟨0x9a5f583ce5084635, 0x4f417e233776c195, 0x74634b1a733f7785, 0x1c51de5ea66f0f25⟩,
    0x000000000000001b, 0x0000000000000001, 0xffffffffffffffff, 0xffffffffffffffff,
    ⟨0x65a0a7c31af7b9cb, 0xb0be81dcc8893e6a, 0x8b9cb4e58cc0887a, 0xe3ae21a15990f0da,
    0xffffffffffffffff⟩),
  (⟨0xc6c552cd2258cd61, 0xb4f028949f9dad38, 0x3834c1a749676b4a, 0x25cdda675f548eb8⟩,
    ⟨0x993d30ed00000001, 0xb4002bcf181cf91b, 0x000224698fc094cf, 0x4000000000000000⟩,
    0x0000000000000000, 0x0000008000000000, 0x0000000000000000, 0x0000000000000000,
    ⟨0x0000008000000000, 0x0e7c8dcc9e987680, 0xe04a67da0015e78c, 0x00000000011234c7,
    0x0000002000000000⟩),
  (⟨0xc6c552cd2258cd61, 0xb4f028949f9dad38, 0x3834c1a749676b4a, 0x25cdda675f548eb8⟩,
    ⟨0x993d30ed00000001, 0xb4002bcf181cf91b, 0x000224698fc094cf, 0x4000000000000000⟩,
    0x0000000000100000, 0x00000058b6db6db7, 0xffffffffffffffff, 0x0000000000000000,
    ⟨0xc47fbd36e0cb6db7, 0x34c05fd325d0d03d, 0x6d486e24e511ab96, 0x198a0ab7153a88b6,
    0x000000162db47e90⟩),
  (⟨0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000⟩,
    ⟨0x9a5f583ce5084635, 0x4f417e233776c195, 0x74634b1a733f7785, 0x1c51de5ea66f0f25⟩,
    0x0800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    ⟨0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000⟩),
  (⟨0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000⟩,
    ⟨0x9a5f583ce5084635, 0x4f417e233776c195, 0x74634b1a733f7785, 0x1c51de5ea66f0f25⟩,
    0x0000000000000000, 0x0000000000000001, 0x0000000000000000, 0x0000000000000000,
    ⟨0x9a5f583ce5084635, 0x4f417e233776c195, 0x74634b1a733f7785, 0x1c51de5ea66f0f25,
    0x0000000000000000⟩),
  (⟨0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000⟩,
    ⟨0x993130ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000⟩,
    0x0800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    ⟨0x0800000000000000, 0xdcc9698768000000, 0x011234c7e04a67c8, 0x0000000000000000,
    0x0200000000000000⟩),
  (⟨0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000⟩,
    ⟨0x993130ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000⟩,
    0x0000000000000000, 0x0000000000000001, 0x0000000000000000, 0x0000000000000000,
    ⟨0x993130ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000,
    0x0000000000000000⟩),
  (⟨0x48cb952368f0cbda, 0x63a7041d70d5fc08, 0xc5673a829079038d, 0x27df43973ee24a3f⟩,
    ⟨0x6dcd9761281f03dc, 0x04f780e32ffca047, 0x664f5e8fa4f2dbe5, 0x209079863e8aa1ad⟩,
    0x000000003a8f9e70, 0x000000004e1e1b64, 0xffffffffffffffff, 0x0000000000000000,
    ⟨0x1737410aa35dfa90, 0xb8e66c17d71f7ca3, 0xd8211d46aeb7b8ab, 0xd6e9e255bb861dfb,
    0x0000000000d0e5bc⟩),
  (⟨0x48cb952368f0cbda, 0x63a7041d70d5fc08, 0xc5673a829079038d, 0x27df43973ee24a3f⟩,
    ⟨0x6dcd9761281f03dc, 0x04f780e32ffca047, 0x664f5e8fa4f2dbe5, 0x209079863e8aa1ad⟩,
    0x000000000180e7d4, 0x0000000020f767b5, 0xffffffffffffffff, 0xffffffffffffffff,
    ⟨0xc0b26c8bf7e63aec, 0xbfc45c1c4733ea1c, 0x267edd41b4b9a6a1, 0xf2dd9e86b1ca270f,
    0xfffffffffb928537⟩),
  (⟨0xe6271d6294fbc898, 0x3e2f8c089c7ceef4, 0x767a844ff1a42a71, 0x460ec71de713dfb1⟩,
    ⟨0xd641284a6ae5037b, 0xc3b6ebf1abb53dea, 0xee9e6be06bcf421d, 0x09541103d50b15e9⟩,
    0x000000004cd19864, 0x000000004b009dfa, 0xffffffffffffffff, 0xffffffffffffffff,
    ⟨0x9ee787ac8aab8f82, 0x7c279d554451297d, 0x4a202205c178e33b, 0xa20ab2aa3e30dfa2,
    0xffffffffe83e9a60⟩),
  (⟨0xe6271d6294fbc898, 0x3e2f8c089c7ceef4, 0x767a844ff1a42a71, 0x460ec71de713dfb1⟩,
    ⟨0xd641284a6ae5037b, 0xc3b6ebf1abb53dea, 0xee9e6be06bcf421d, 0x09541103d50b15e9⟩,
    0x000000001c7af63e, 0x000000003677931b, 0xffffffffffffffff, 0xffffffffffffffff,
    ⟨0x72dda29c687f5c37, 0x3d5ddd77e2361e16, 0x27f0f1b1b7be7085, 0xb23361418edf7439,
    0xfffffffff638a4c3⟩),
  (⟨0x493586b8db6db6ca, 0x860d0369a70b9cb1, 0x6db6db6db6db6db4, 0x36db6db6db6db6db⟩,
    ⟨0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000⟩,
    0x0800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    ⟨0x5000000000000000, 0x8a49ac35c6db6db6, 0xa430681b4d385ce5, 0xdb6db6db6db6db6d,
    0x01b6db6db6db6db6⟩),
  (⟨0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000⟩,
    ⟨0xb4becc383c4c0001, 0x22460fe1a55cd3e7, 0x0000000000000000, 0x3fff000000000000⟩,
    0x0800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    ⟨0x0800000000000000, 0xdcc9698768000000, 0x011234c7e04a67c8, 0x0000000000000000,
    0x0200000000000000⟩),
  (⟨0x6ec45e40fffffe25, 0xb0ff453accc4c098, 0x0d0acc8786d45ce5, 0x1257ca108c691d71⟩,
    ⟨0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000⟩,
    0x0800000000000000, 0x0000000000000000, 0xffffffffffffffff, 0xffffffffffffffff,
    ⟨0xd800000000000000, 0x3c89dd0df800000e, 0xd27805d62999d9fb, 0x7797a99bc3c95d18,
    0xff6d41af7b9cb714⟩),
  (⟨0x2a68d2ac000001dc, 0x714753c13c883883, 0xf2f53378792ba31a, 0x2da835ef7396e28e⟩,
    ⟨0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000⟩,
    0x0800000000000000, 0x0000000000000000, 0xffffffffffffffff, 0xffffffffffffffff,
    ⟨0x2000000000000000, 0xe6acb96a9ffffff1, 0x2c75c561f61bbe3b, 0x886856643c36a2e7,
    0xfe92be50846348eb⟩),
  (⟨0x48434e89d6a56baa, 0x2089cebea655c8a7, 0x8391299eb98be0ad, 0x20b8eb4c2df91a89⟩,
    ⟨0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000⟩,
    0x0800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    ⟨0x5000000000000000, 0x3a421a744eb52b5d, 0x69044e75f532ae45, 0x4c1c894cf5cc5f05,
    0x0105c75a616fc8d4⟩),
  (⟨0x2a076bc0db6db6ca, 0x860d0369a22a37c6, 0x6db6db6db6db6db4, 0x36db6db6db6db6db⟩,
    ⟨0x8c46eb2100000001, 0x224698fc0994a8dd, 0x0000000000000000, 0x4000000000000000⟩,
    0x0800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    ⟨0x5000000000000000, 0x31503b5e06db6db6, 0xa430681b4d1151be, 0xdb6db6db6db6db6d,
    0x01b6db6db6db6db6⟩),
  (⟨0x8c46eb2100000001, 0x224698fc0994a8dd, 0x0000000000000000, 0x4000000000000000⟩,
    ⟨0xe8d0ba05537c0001, 0x22460fe1a5a4828a, 0x0000000000000000, 0x3fff000000000000⟩,
    0x0800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    ⟨0x0800000000000000, 0xec62375908000000, 0x011234c7e04ca546, 0x0000000000000000,
    0x0200000000000000⟩),
  (⟨0x61b3735c000001dc, 0x6e4e03c0fcf03909, 0xf5c46200999eb20c, 0x2da835ef99fb552f⟩,
    ⟨0x8c46eb2100000001, 0x224698fc0994a8dd, 0x0000000000000000, 0x4000000000000000⟩,
    0x0800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    ⟨0xe000000000000000, 0x4b0d9b9ae000000e, 0x6372701e07e781c8, 0x7fae231004ccf590,
    0x016d41af7ccfdaa9⟩),
  (⟨0x2a9377c4fffffe25, 0xb3f8953b0ca46fd4, 0x0a3b9dff66614df3, 0x1257ca106604aad0⟩,
    ⟨0x8c46eb2100000001, 0x224698fc0994a8dd, 0x0000000000000000, 0x4000000000000000⟩,
    0x0800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    ⟨0x2800000000000000, 0xa1549bbe27fffff1, 0x9d9fc4a9d865237e, 0x8051dceffb330a6f,
    0x0092be5083302556⟩),
  (⟨0xca612b52ba19e957, 0x8954aaeb1967c172, 0xac4ce37d081b0c2b, 0x0a79ccdfa3a30d0e⟩,
    ⟨0x8c46eb2100000001, 0x224698fc0994a8dd, 0x0000000000000000, 0x4000000000000000⟩,
    0x0800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    ⟨0xb800000000000000, 0x9653095a95d0cf4a, 0x5c4aa55758cb3e0b, 0x7562671be840d861,
    0x0053ce66fd1d1868⟩)]

/-- The indices of the vectors that the block's transcription does not reproduce. -/
def deRowFailures : List Nat :=
  (List.range deRowVectors.length).filter fun i =>
    match deRowVectors[i]? with
    | some (d, e, m0, m1, s0, s1, res) => deRowBlock d e m0 m1 s0 s1 != res
    | none => false

theorem deRow_vectors_reproduced : deRowFailures = [] := by decide +kernel

/-- info: [] -/
#guard_msgs in
#eval deRowFailures

/-! ## The almost-Montgomery reduction block -/

/-- A row combination, the field, and the expected reduction, as in the crate's
`amontred_known_answers`. -/
def amontredVectors : List (Signed5 × PastaField × Limbs) := [
  (⟨0x4b52000000000000, 0xf53e9f8f819a3464, 0x8c88e8ee762e0853, 0x60d3a4b3b5a55058,
    0x00082faa475c1c1a⟩, pallasBase, ⟨0xadb482ad66b03465, 0xa4b9d87ba80679cc, 0x60d3a4b3b5a55058,
    0x2d33afaa475c1c1a⟩),
  (⟨0x65a0a7c31af7b9cb, 0xb0be81dcc8893e6a, 0x8b9cb4e58cc0887a, 0xe3ae21a15990f0da,
    0xffffffffffffffff⟩, pallasBase, ⟨0x7806e5a0c4c00001, 0xadeb423bad68faee, 0x23ae21a15990f0da,
    0x400eda4af942118d⟩),
  (⟨0x0000008000000000, 0x0e7c8dcc9e987680, 0xe04a67da0015e78c, 0x00000000011234c7,
    0x0000002000000000⟩, pallasBase, ⟨0x012d30ed08000001, 0x029100c4e61662a3, 0x00000000011234c8,
    0x4000000000000000⟩),
  (⟨0xc47fbd36e0cb6db7, 0x34c05fd325d0d03d, 0x6d486e24e511ab96, 0x198a0ab7153a88b6,
    0x000000162db47e90⟩, pallasBase, ⟨0xcd5f403284675db7, 0x722dece6c4bee381, 0x598a0ab7153a88b6,
    0x092489633581a322⟩),
  (⟨0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000⟩, pallasBase, ⟨0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000,
    0x4000000000000000⟩),
  (⟨0x9a5f583ce5084635, 0x4f417e233776c195, 0x74634b1a733f7785, 0x1c51de5ea66f0f25,
    0x0000000000000000⟩, pallasBase, ⟨0xba537c393b400001, 0x96a1efbc6530f748, 0xdc51de5ea66f0f25,
    0x3ff125b506bdee72⟩),
  (⟨0x0800000000000000, 0xdcc9698768000000, 0x011234c7e04a67c8, 0x0000000000000000,
    0x0200000000000000⟩, pallasBase, ⟨0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000,
    0x4000000000000000⟩),
  (⟨0x993130ed00000001, 0x224698fc094cf91b, 0x0000000000000000, 0x4000000000000000,
    0x0000000000000000⟩, pallasBase, ⟨0xb4becc383c4c0001, 0x22460fe1a55cd3e7, 0x0000000000000000,
    0x3fff000000000000⟩),
  (⟨0x1737410aa35dfa90, 0xb8e66c17d71f7ca3, 0xd8211d46aeb7b8ab, 0xd6e9e255bb861dfb,
    0x0000000000d0e5bc⟩, pallasBase, ⟨0x6e3d45a481a7dcc4, 0xf643efa7a0c6948d, 0xd6e9e255bb861dfb,
    0x38452d9157f96718⟩),
  (⟨0xc0b26c8bf7e63aec, 0xbfc45c1c4733ea1c, 0x267edd41b4b9a6a1, 0xf2dd9e86b1ca270f,
    0xfffffffffb928537⟩, pallasBase, ⟨0xdaef3d96915ccd46, 0x3178b9732fd6b26a, 0xf2dd9e86b1ca270f,
    0x147e97fbfd98f67c⟩),
  (⟨0x9ee787ac8aab8f82, 0x7c279d554451297d, 0x4a202205c178e33b, 0xa20ab2aa3e30dfa2,
    0xffffffffe83e9a60⟩, pallasBase, ⟨0x55fb4a4f35158971, 0x67231724fb0356b4, 0x220ab2aa3e30dfa2,
    0x362baceb4593b680⟩),
  (⟨0x72dda29c687f5c37, 0x3d5ddd77e2361e16, 0x27f0f1b1b7be7085, 0xb23361418edf7439,
    0xfffffffff638a4c3⟩, pallasBase, ⟨0xb4bb2db5fe7889f4, 0x30a4e02f8b124676, 0xf23361418edf7439,
    0x104003139c18cdb5⟩),
  (⟨0x5000000000000000, 0x8a49ac35c6db6db6, 0xa430681b4d385ce5, 0xdb6db6db6db6db6d,
    0x01b6db6db6db6db6⟩, pallasBase, ⟨0x8398bdd8b6db6db7, 0xbbc0f148939d4828, 0xdb6db6db6db6db6d,
    0x2db6db6db6db6db6⟩),
  (⟨0x0800000000000000, 0xdcc9698768000000, 0x011234c7e04a67c8, 0x0000000000000000,
    0x0200000000000000⟩, pallasBase, ⟨0x992d30ed00000001, 0x224698fc094cf91b, 0x0000000000000000,
    0x4000000000000000⟩),
  (⟨0xd800000000000000, 0x3c89dd0df800000e, 0xd27805d62999d9fb, 0x7797a99bc3c95d18,
    0xff6d41af7b9cb714⟩, pallasBase, ⟨0x8c78ecb30000000f, 0xd7d30dbd8b0de0e7, 0x7797a99bc3c95d18,
    0x096d41af7b9cb714⟩),
  (⟨0x2000000000000000, 0xe6acb96a9ffffff1, 0x2c75c561f61bbe3b, 0x886856643c36a2e7,
    0xfe92be50846348eb⟩, pallasBase, ⟨0x0cb44439fffffff2, 0x4a738b3e7e3f1834, 0x886856643c36a2e7,
    0x3692be50846348eb⟩),
  (⟨0x5000000000000000, 0x3a421a744eb52b5d, 0x69044e75f532ae45, 0x4c1c894cf5cc5f05,
    0x0105c75a616fc8d4⟩, pallasBase, ⟨0x33912c173eb52b5e, 0x8094d7a33b979988, 0x4c1c894cf5cc5f05,
    0x2d05c75a616fc8d4⟩),
  (⟨0x5000000000000000, 0x31503b5e06db6db6, 0xa430681b4d1151be, 0xdb6db6db6db6db6d,
    0x01b6db6db6db6db6⟩, vestaBase, ⟨0x81c0fd04b6db6db7, 0xbbc0f14893a785d6, 0xdb6db6db6db6db6d,
    0x2db6db6db6db6db6⟩),
  (⟨0x0800000000000000, 0xec62375908000000, 0x011234c7e04ca546, 0x0000000000000000,
    0x0200000000000000⟩, vestaBase, ⟨0x8c46eb2100000001, 0x224698fc0994a8dd, 0x0000000000000000,
    0x4000000000000000⟩),
  (⟨0xe000000000000000, 0x4b0d9b9ae000000e, 0x6372701e07e781c8, 0x7fae231004ccf590,
    0x016d41af7ccfdaa9⟩, vestaBase, ⟨0xfc9678ff0000000f, 0x67bb433d891a16e3, 0x7fae231004ccf590,
    0x096d41af7ccfdaa9⟩),
  (⟨0x2800000000000000, 0xa1549bbe27fffff1, 0x9d9fc4a9d865237e, 0x8051dceffb330a6f,
    0x0092be5083302556⟩, vestaBase, ⟨0x8fb07221fffffff2, 0xba8b55be807a91f9, 0x8051dceffb330a6f,
    0x3692be5083302556⟩),
  (⟨0xb800000000000000, 0x9653095a95d0cf4a, 0x5c4aa55758cb3e0b, 0x7562671be840d861,
    0x0053ce66fd1d1868⟩, vestaBase, ⟨0xe5c6fb7bddd0cf4b, 0x65ee805e3b7d0d89, 0x7562671be840d861,
    0x1253ce66fd1d1868⟩)]

/-- The indices of the vectors that the block's transcription does not reproduce. -/
def amontredFailures : List Nat :=
  (List.range amontredVectors.length).filter fun i =>
    match amontredVectors[i]? with
    | some (t, F, res) => amontredBlock t F.modulus F.inv != res
    | none => false

theorem amontred_vectors_reproduced : amontredFailures = [] := by decide +kernel

/-- info: [] -/
#guard_msgs in
#eval amontredFailures

end PastaCurves.AArch64
