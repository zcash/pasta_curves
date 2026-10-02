import PastaCurves.Compositions

/-!
# The inversion's schedule of block calls

`invert` in `PastaCurves/Compositions.lean` mirrors the Rust driver as a pure function of its
blocks, so it says what the inversion computes but not which blocks it runs, in which order. For
constant time the order matters: a driver that called a block only for some inputs, or a
different number of times, would leak through the sequence of blocks even if each block were
constant-time.

`invertM` is the same driver over blocks with an effect, in a monad `m`. Over the identity monad
it is `invert` (`invertM_id`), so it is a faithful copy of the mirrored driver, not a second
model. Over blocks that also append their names to a log (`InvertBlocks.logged`), its result is
still `invert`, and its log is `invertSchedule` whatever the input and whatever the blocks compute
(`invertM_logged`): nine rounds of eight calls, then the last round's five. The driver has no branch on the data, so the sequence of calls is a constant.
-/

namespace PastaCurves

/-- The six blocks of the inversion, by name. -/
inductive BlockName
  | divstep59
  | signMag
  | fgRow
  | uvRow
  | amontred
  | condSub
  deriving DecidableEq, Repr

/-- The inversion's six blocks with an effect in the monad `m`: the record `InvertBlocks` with
each result in `m`. -/
structure InvertBlocksM (m : Type → Type) where
  divstep59 : Nat → Nat → Nat → m Divstep59Result
  signMag : Nat → Nat → Nat → Nat → m SignMag
  fgRow : Signed5 → Signed5 → Nat → Nat → Nat → Nat → m Signed5
  uvRow : Limbs → Limbs → Nat → Nat → Nat → Nat → m Signed5
  amontred : Signed5 → Limbs → Nat → m Limbs
  condSub : Limbs → Limbs → m Limbs

variable {m : Type → Type} [Monad m]

/-- `invertRound` over effectful blocks, calling them in the order of `src/inversion.rs`. -/
def invertRoundM (B : InvertBlocksM m) (modulus : Limbs) (inv : Nat) (st : InvertState) :
    m InvertState := do
  let dm ← B.divstep59 st.d st.f.l0 st.g.l0
  let sm ← B.signMag dm.m00 dm.m01 dm.m10 dm.m11
  let f ← B.fgRow st.f st.g sm.m00 sm.m01 sm.s00 sm.s01
  let g ← B.fgRow st.f st.g sm.m10 sm.m11 sm.s10 sm.s11
  let tu ← B.uvRow st.u st.v sm.m00 sm.m01 sm.s00 sm.s01
  let tv ← B.uvRow st.u st.v sm.m10 sm.m11 sm.s10 sm.s11
  let u ← B.amontred tu modulus inv
  let v ← B.amontred tv modulus inv
  pure ⟨dm.d, f, g, u, v⟩

/-- `n` iterations of a Kleisli arrow, the first applied first, as `Nat.iterate` iterates. -/
def iterateM {α : Type} (f : α → m α) : Nat → α → m α
  | 0, a => pure a
  | n + 1, a => f a >>= iterateM f n

/-- `invert` over effectful blocks: nine rounds, then the last round, which calls `divstep59`,
`signMag`, `uvRow`, `amontred`, and `condSub`. -/
def invertM (B : InvertBlocksM m) (x modulus : Limbs) (inv : Nat) (v0 : Limbs) : m Limbs := do
  let st ← iterateM (invertRoundM B modulus inv) 9
    ⟨1, ⟨modulus.l0, modulus.l1, modulus.l2, modulus.l3, 0⟩, ⟨x.l0, x.l1, x.l2, x.l3, 0⟩,
      ⟨0, 0, 0, 0⟩, v0⟩
  let dm ← B.divstep59 st.d st.f.l0 st.g.l0
  let sign := signWord st.f.l0 st.g.l0 dm.m00 dm.m01
  let sm ← B.signMag dm.m00 dm.m01 dm.m10 dm.m11
  let t ← B.uvRow st.u st.v sm.m00 sm.m01 (sm.s00 ^^^ sign) (sm.s01 ^^^ sign)
  let r ← B.amontred t modulus inv
  B.condSub r modulus

/-- Pure blocks as blocks in the identity monad. -/
def InvertBlocks.toId (B : InvertBlocks) : InvertBlocksM Id where
  divstep59 d f0 g0 := pure (B.divstep59 d f0 g0)
  signMag m00 m01 m10 m11 := pure (B.signMag m00 m01 m10 m11)
  fgRow f g m0 m1 s0 s1 := pure (B.fgRow f g m0 m1 s0 s1)
  uvRow u v m0 m1 s0 s1 := pure (B.uvRow u v m0 m1 s0 s1)
  amontred t modulus inv := pure (B.amontred t modulus inv)
  condSub value modulus := pure (B.condSub value modulus)

/-- In the identity monad, iterating a Kleisli arrow is iterating the function. -/
theorem iterateM_id {α : Type} (f : α → α) (n : Nat) (a : α) :
    iterateM (m := Id) (fun a => pure (f a)) n a = f^[n] a := by
  induction n generalizing a with
  | zero => rfl
  | succ n ih => exact ih (f a)

/-- Over the identity monad, `invertM` is the mirrored driver `invert`. -/
theorem invertM_id (B : InvertBlocks) (x modulus : Limbs) (inv : Nat) (v0 : Limbs) :
    invertM B.toId x modulus inv v0 = invert B x modulus inv v0 := by
  have hround : invertRoundM B.toId modulus inv = fun st => pure (invertRound B modulus inv st) :=
    rfl
  simp only [invertM, hround, iterateM_id]
  rfl

/-- Pure blocks that also append their name to a log, in the state monad over the log. -/
def InvertBlocks.logged (B : InvertBlocks) : InvertBlocksM (StateM (List BlockName)) where
  divstep59 d f0 g0 := do
    modify (· ++ [.divstep59]); pure (B.divstep59 d f0 g0)
  signMag m00 m01 m10 m11 := do
    modify (· ++ [.signMag]); pure (B.signMag m00 m01 m10 m11)
  fgRow f g m0 m1 s0 s1 := do
    modify (· ++ [.fgRow]); pure (B.fgRow f g m0 m1 s0 s1)
  uvRow u v m0 m1 s0 s1 := do
    modify (· ++ [.uvRow]); pure (B.uvRow u v m0 m1 s0 s1)
  amontred t modulus inv := do
    modify (· ++ [.amontred]); pure (B.amontred t modulus inv)
  condSub value modulus := do
    modify (· ++ [.condSub]); pure (B.condSub value modulus)

/-- The calls of one round, in order. -/
def roundSchedule : List BlockName :=
  [.divstep59, .signMag, .fgRow, .fgRow, .uvRow, .uvRow, .amontred, .amontred]

/-- The calls of the inversion, in order: nine rounds, then the last round's five. -/
def invertSchedule : List BlockName :=
  (List.replicate 9 roundSchedule).flatten ++ [.divstep59, .signMag, .uvRow, .amontred, .condSub]

/-- A round of logged blocks appends `roundSchedule` to the log, and computes `invertRound`. -/
theorem invertRoundM_logged (B : InvertBlocks) (modulus : Limbs) (inv : Nat) (st : InvertState)
    (log : List BlockName) :
    (invertRoundM B.logged modulus inv st).run log =
      (invertRound B modulus inv st, log ++ roundSchedule) := by
  simp [invertRoundM, InvertBlocks.logged, invertRound, roundSchedule, modify, modifyGet,
    MonadStateOf.modifyGet, StateT.modifyGet, StateT.run, bind, StateT.bind, pure, StateT.pure]

/-- `n` logged rounds append `n` copies of `roundSchedule`, and compute `n` rounds. -/
theorem iterateM_logged (B : InvertBlocks) (modulus : Limbs) (inv : Nat) (n : Nat)
    (st : InvertState) (log : List BlockName) :
    (iterateM (invertRoundM B.logged modulus inv) n st).run log =
      ((invertRound B modulus inv)^[n] st, log ++ (List.replicate n roundSchedule).flatten) := by
  induction n generalizing st log with
  | zero => simp [iterateM, StateT.run, pure, StateT.pure]
  | succ n ih =>
    have h := invertRoundM_logged B modulus inv st log
    simp only [iterateM, StateT.run_bind, Function.iterate_succ, Function.comp_apply]
    simp only [StateT.run] at h ih ⊢
    rw [h]
    simp only [ih, List.replicate_succ, List.flatten_cons]
    rw [← List.append_assoc]
    rfl

/-- The logged inversion computes `invert`, and logs `invertSchedule` after the log it starts
from, whatever the input `x` and whatever the blocks compute. -/
theorem invertM_logged (B : InvertBlocks) (x modulus : Limbs) (inv : Nat) (v0 : Limbs)
    (log : List BlockName) :
    (invertM B.logged x modulus inv v0).run log =
      (invert B x modulus inv v0, log ++ invertSchedule) := by
  simp only [invertM, StateT.run_bind, iterateM_logged]
  simp [InvertBlocks.logged, invert, invertSchedule, StateT.run, modify, modifyGet,
    MonadStateOf.modifyGet, StateT.modifyGet, bind, StateT.bind, pure, StateT.pure]

/-- The schedule of block calls does not depend on the input: two runs of the logged inversion,
on any inputs, log the same calls. -/
theorem invertM_logged_log (B : InvertBlocks) (x x' modulus modulus' : Limbs) (inv inv' : Nat)
    (v0 v0' : Limbs) :
    ((invertM B.logged x modulus inv v0).run []).2 =
      ((invertM B.logged x' modulus' inv' v0').run []).2 := by
  rw [invertM_logged, invertM_logged]

end PastaCurves
