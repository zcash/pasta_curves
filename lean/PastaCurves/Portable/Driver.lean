import PastaCurves.Portable.Words
import PastaCurves.Inversion.Composition

/-!
# The translated driver computes the composition

Aeneas translates the crate's generic `invert_with` (`src/inversion.rs`) along with the portable
blocks, over any record of the trait's six blocks. Over a record whose blocks succeed where the
portable ones do (`Succeeds`), the translation returns the composition `invert` of
`Compositions.lean` over the record's blocks read as functions on the model's values
(`pureBlocks`). So when those functions satisfy `InvertBlocks.Spec` at a Pasta field, the
translation returns the canonical Montgomery inverse of a canonical input, or `0` for `0`, by
`invert_entry_spec`.

Every portable block but `fg_row` and `de_row` succeeds whatever words it is given. Those two
succeed whenever the magnitudes `m0` and `m1` of their row's entries sum to at most `2^63`, which
keeps their 128-bit columns from overflowing. Along the composition they sum to at most `2^59`
(`invertRound_rows_le`).
-/

namespace PastaCurves.Portable

open Aeneas Aeneas.Std pasta_curves

/-- The result of `divstep59` from Aeneas' array of five words. -/
def divstep59OfArray (a : Std.Array Std.U64 5#usize) : Divstep59Result :=
  ⟨a[0].val, a[1].val, a[2].val, a[3].val, a[4].val⟩

/-- The result of `sign_mag` from Aeneas' array of eight words. -/
def signMagOfArray (a : Std.Array Std.U64 8#usize) : SignMag :=
  ⟨a[0].val, a[1].val, a[2].val, a[3].val, a[4].val, a[5].val, a[6].val, a[7].val⟩

/-- A translated record's blocks as functions on the model's values: each block on the words of
its inputs, read back where it succeeds. -/
def pureBlocks {B : Type} (I : inversion.InvertBlocks B) : InvertBlocks where
  divstep59 two_delta f0 g0 := divstep59OfArray
    ((Option.ofResult (I.divstep59 (word two_delta) (word f0) (word g0))).getD
      (Std.Array.repeat 5#usize 0#u64))
  signMag u v q r := signMagOfArray
    ((Option.ofResult (I.sign_mag (word u) (word v) (word q) (word r))).getD
      (Std.Array.repeat 8#usize 0#u64))
  fgRow f g m0 m1 s0 s1 := signed5OfArray
    ((Option.ofResult (I.fg_row (signed5Array f) (signed5Array g) (word m0) (word m1) (word s0)
      (word s1))).getD (Std.Array.repeat 5#usize 0#u64))
  deRow d e m0 m1 s0 s1 := signed5OfArray
    ((Option.ofResult (I.de_row (limbsArray d) (limbsArray e) (word m0) (word m1) (word s0)
      (word s1))).getD (Std.Array.repeat 5#usize 0#u64))
  amontred t modulus inv := limbsOfArray
    ((Option.ofResult (I.amontred (signed5Array t) (limbsArray modulus) (word inv))).getD
      (Std.Array.repeat 4#usize 0#u64))
  condSub value modulus := limbsOfArray
    ((Option.ofResult (I.cond_sub (limbsArray value) (limbsArray modulus))).getD
      (Std.Array.repeat 4#usize 0#u64))

/-- Where a translated record's blocks succeed. Every block but `fg_row` and `de_row` succeeds
whatever words it is given. Those two succeed when the magnitudes `m0` and `m1` of their row's
entries sum to at most `2^63`. -/
structure Succeeds {B : Type} (I : inversion.InvertBlocks B) : Prop where
  /-- `divstep59` succeeds on all inputs. -/
  divstep59 : ∀ two_delta f0 g0, ∃ a, I.divstep59 two_delta f0 g0 = .ok a
  /-- `sign_mag` succeeds on all inputs. -/
  signMag : ∀ u v q r, ∃ a, I.sign_mag u v q r = .ok a
  /-- `fg_row` succeeds when the magnitudes of the row sum to at most `2^63`. -/
  fgRow : ∀ f g m0 m1 s0 s1, m0.val + m1.val ≤ 2^63 → ∃ a, I.fg_row f g m0 m1 s0 s1 = .ok a
  /-- `de_row` succeeds when the magnitudes of the row sum to at most `2^63`. -/
  deRow : ∀ d e m0 m1 s0 s1, m0.val + m1.val ≤ 2^63 → ∃ a, I.de_row d e m0 m1 s0 s1 = .ok a
  /-- `amontred` succeeds on all inputs. -/
  amontred : ∀ t modulus inv, ∃ a, I.amontred t modulus inv = .ok a
  /-- `cond_sub` succeeds on all inputs. -/
  condSub : ∀ value modulus, ∃ a, I.cond_sub value modulus = .ok a

section Blocks

variable {B : Type} {I : inversion.InvertBlocks B} (hS : Succeeds I)
include hS

/-- `divstep59` returns the words of its function on the model's values. -/
theorem Succeeds.divstep59_spec (two_delta f0 g0 : Std.U64) :
    I.divstep59 two_delta f0 g0 ⦃ a =>
      (pureBlocks I).divstep59 two_delta.val f0.val g0.val = divstep59OfArray a ⦄ := by
  obtain ⟨a, ha⟩ := hS.divstep59 two_delta f0 g0
  simp [ha, pureBlocks, Option.ofResult, word_val_self]

/-- `sign_mag` returns the words of its function on the model's values. -/
theorem Succeeds.sign_mag_spec (u v q r : Std.U64) :
    I.sign_mag u v q r ⦃ a =>
      (pureBlocks I).signMag u.val v.val q.val r.val = signMagOfArray a ⦄ := by
  obtain ⟨a, ha⟩ := hS.signMag u v q r
  simp [ha, pureBlocks, Option.ofResult, word_val_self]

/-- `fg_row`, on a row whose magnitudes sum to at most `2^63`, returns the words of its function
on the model's values. -/
theorem Succeeds.fg_row_spec (f g : Std.Array Std.U64 5#usize) (m0 m1 s0 s1 : Std.U64)
    (h : m0.val + m1.val ≤ 2^63) :
    I.fg_row f g m0 m1 s0 s1 ⦃ a => (pureBlocks I).fgRow (signed5OfArray f) (signed5OfArray g)
      m0.val m1.val s0.val s1.val = signed5OfArray a ⦄ := by
  obtain ⟨a, ha⟩ := hS.fgRow f g m0 m1 s0 s1 h
  simp [ha, pureBlocks, Option.ofResult, word_val_self, signed5Array_signed5OfArray]

/-- `de_row`, on a row whose magnitudes sum to at most `2^63`, returns the words of its function
on the model's values. -/
theorem Succeeds.de_row_spec (d e : Std.Array Std.U64 4#usize) (m0 m1 s0 s1 : Std.U64)
    (h : m0.val + m1.val ≤ 2^63) :
    I.de_row d e m0 m1 s0 s1 ⦃ a => (pureBlocks I).deRow (limbsOfArray d) (limbsOfArray e)
      m0.val m1.val s0.val s1.val = signed5OfArray a ⦄ := by
  obtain ⟨a, ha⟩ := hS.deRow d e m0 m1 s0 s1 h
  simp [ha, pureBlocks, Option.ofResult, word_val_self, limbsArray_limbsOfArray]

/-- `amontred` returns the words of its function on the model's values. -/
theorem Succeeds.amontred_spec (t : Std.Array Std.U64 5#usize)
    (modulus : Std.Array Std.U64 4#usize) (inv : Std.U64) :
    I.amontred t modulus inv ⦃ a => (pureBlocks I).amontred (signed5OfArray t)
      (limbsOfArray modulus) inv.val = limbsOfArray a ⦄ := by
  obtain ⟨a, ha⟩ := hS.amontred t modulus inv
  simp [ha, pureBlocks, Option.ofResult, word_val_self, signed5Array_signed5OfArray,
    limbsArray_limbsOfArray]

/-- `cond_sub` returns the words of its function on the model's values. -/
theorem Succeeds.cond_sub_spec (value modulus : Std.Array Std.U64 4#usize) :
    I.cond_sub value modulus ⦃ a => (pureBlocks I).condSub (limbsOfArray value)
      (limbsOfArray modulus) = limbsOfArray a ⦄ := by
  obtain ⟨a, ha⟩ := hS.condSub value modulus
  simp [ha, pureBlocks, Option.ofResult, limbsArray_limbsOfArray]

end Blocks

/-- The range iterator over `i32`, in the form of Aeneas' lemmas for the unsigned ones: below the
end it yields the start and advances it by one, and at or past the end it yields nothing. -/
@[step]
theorem next_I32_spec (range : core.ops.range.Range Std.I32) :
    core.iter.range.IteratorRange.next core.iter.range.StepI32 range
    ⦃ (opt : Option Std.I32) (range' : core.ops.range.Range Std.I32) =>
      (if range.start.val < range.end.val then
         opt = some range.start ∧ range'.start.val = range.start.val + 1
       else
         opt = none ∧ range'.start = range.start) ∧
      range'.end = range.end ⦄ := by
  simp only [core.iter.range.IteratorRange.next, core.iter.range.StepI32,
    core.iter.range.IScalarStep, core.iter.range.IScalarStep.forward_checked]
  simp [core.cmp.impls.PartialOrdI32.lt, core.clone.impls.CloneI32.clone]
  split_ifs
  · simp
  · agrind
  · simp

/-- The word state of a round, read from the translation's words. -/
def stateOf (two_delta : Std.U64) (f g : Std.Array Std.U64 5#usize)
    (d e : Std.Array Std.U64 4#usize) : InvertState :=
  ⟨two_delta.val, signed5OfArray f, signed5OfArray g, limbsOfArray d, limbsOfArray e⟩

/-- The state that `invert` starts its rounds from, at a field `F` and the input `x`. -/
def startState (F : PastaField) (x : Limbs) : InvertState :=
  ⟨1, ⟨F.modulus.l0, F.modulus.l1, F.modulus.l2, F.modulus.l3, 0⟩,
    ⟨x.l0, x.l1, x.l2, x.l3, 0⟩, ⟨0, 0, 0, 0⟩, Inversion.startE F⟩

section Driver

variable {B : Type} {I : inversion.InvertBlocks B}

attribute [local step] Succeeds.divstep59_spec Succeeds.sign_mag_spec Succeeds.fg_row_spec
  Succeeds.de_row_spec Succeeds.amontred_spec Succeeds.cond_sub_spec
attribute [local grind =] divstep59OfArray signMagOfArray signed5OfArray

/-- The loop of `invert_with`, from any point of its nine rounds: words whose state is the
composition's after `iter.start` rounds end with its state after nine. -/
theorem invert_with_loop_spec (hS : Succeeds I) (F : PastaField) (hB : (pureBlocks I).Spec F)
    (x : Limbs) (hx : x.Bounded) (iter : core.ops.range.Range Std.I32) (two_delta : Std.U64)
    (f g : Std.Array Std.U64 5#usize) (d e : Std.Array Std.U64 4#usize)
    (hend : iter.«end».val = 9) (h0 : 0 ≤ iter.start.val) (h9 : iter.start.val ≤ 9)
    (hst : stateOf two_delta f g d e =
      (invertRound (pureBlocks I) F.modulus F.inv)^[iter.start.val.toNat] (startState F x)) :
    inversion.invert_with_loop I iter (limbsArray F.modulus) (word F.inv) two_delta f g d e
    ⦃ (two_delta' : Std.U64) (f' g' : Std.Array Std.U64 5#usize)
      (d' e' : Std.Array Std.U64 4#usize) =>
      stateOf two_delta' f' g' d' e' =
        (invertRound (pureBlocks I) F.modulus F.inv)^[9] (startState F x) ⦄ := by
  unfold inversion.invert_with_loop
  apply loop.spec_decr_nat (fun (iter, _) => (iter.«end».val - iter.start.val).toNat)
    (fun (iter, two_delta, f, g, d, e) =>
      iter.«end».val = 9 ∧ 0 ≤ iter.start.val ∧ iter.start.val ≤ 9 ∧
        stateOf two_delta f g d e =
          (invertRound (pureBlocks I) F.modulus F.inv)^[iter.start.val.toNat] (startState F x))
  · clear hend h0 h9 hst
    rintro ⟨iter, two_delta, f, g, d, e⟩ ⟨hend, h0, h9, hst⟩
    -- The round's state carries the model's, so the rows meet their bound.
    have hcarry :
        Carries (stateOf two_delta f g d e) (Inversion.rounds F x iter.start.val.toNat) := by
      rw [hst]
      exact invert_carries _ F hB x hx _ (by agrind)
    have hrows := invertRound_rows_le _ F hB x hx _ (by agrind) _ hcarry
    dsimp only [stateOf, signed5OfArray] at hrows
    unfold inversion.invert_with_loop.body
    step*
    · agrind
    · grind
    · grind
    · grind
    · grind
    · split_conjs
      · agrind
      · agrind
      · agrind
      · have hsucc : iter1.start.val.toNat = iter.start.val.toNat + 1 := by agrind
        have hm : limbsOfArray (limbsArray F.modulus) = F.modulus :=
          limbsOfArray_limbsArray _ F.bounded
        have hinv : (word F.inv).val = F.inv := word_val _ F.inv_lt
        rw [hsucc, Function.iterate_succ_apply', ← hst]
        grind [invertRound, stateOf]
      · agrind
  · exact ⟨hend, h0, h9, hst⟩

/-- `invert_with` over a translated record whose blocks succeed where the portable ones do, and
whose functions on the model's values satisfy `InvertBlocks.Spec` at `F`, returns the
composition `invert` over those functions. -/
theorem invert_with_spec (hS : Succeeds I) (F : PastaField) (hB : (pureBlocks I).Spec F)
    (x : Limbs) (hx : x.Bounded) :
    inversion.invert_with I (limbsArray x) (limbsArray F.modulus) (word F.inv)
      (limbsArray (Inversion.startE F))
    ⦃ (res : Std.Array Std.U64 4#usize) =>
      limbsOfArray res = invert (pureBlocks I) x F.modulus F.inv (Inversion.startE F) ⦄ := by
  unfold inversion.invert_with
  step*
  step with invert_with_loop_spec hS F hB x hx as ⟨two_delta, f, g, d, e, hloop⟩
  · -- The words that the loop starts from are `invert`'s starting state.
    obtain ⟨hm0, hm1, hm2, hm3⟩ := F.bounded
    obtain ⟨hx0, hx1, hx2, hx3⟩ := hx
    obtain ⟨he0, he1, he2, he3⟩ : (Inversion.startE F).Bounded := Limbs.ofNat_bounded _
    simp [startState, stateOf, signed5OfArray, limbsOfArray, limbsArray, word_val, *]
  -- The last round's state carries the model's, so its row meets the bound.
  have hcarry : Carries (stateOf two_delta f g d e) (Inversion.rounds F x 9) := by
    rw [hloop]
    exact invert_carries _ F hB x hx 9 (by agrind)
  have hrows := invertRound_rows_le _ F hB x hx 9 (by agrind) _ hcarry
  dsimp only [stateOf, signed5OfArray] at hrows
  step*
  · grind
  · -- The last round, with the sign of the new `f` in the masks of its row, is `invert`'s.
    have hm : limbsOfArray (limbsArray F.modulus) = F.modulus :=
      limbsOfArray_limbsArray _ F.bounded
    have hinv : (word F.inv).val = F.inv := word_val _ F.inv_lt
    have hsign : sign.val = signWord i8.val i9.val u.val v.val := by
      subst i13
      rw [sign_post, sign_mask_val i14_post]
      simp [signWord, addw, mulLo, regMod, U64.size, U64.numBits, i12_post, i10_post, i11_post]
    simp only [startState] at hloop
    simp only [invert]
    rw [← hloop]
    grind [stateOf]

/-- `invert_with` over such a record, at a Pasta field, on a canonical input: the result is
canonical, and it is `0` for `x = 0` and the Montgomery inverse otherwise. -/
theorem invert_with_entry_spec (hS : Succeeds I) (F : PastaField) (hB : (pureBlocks I).Spec F)
    (x : Limbs) (hx : x.Bounded) (h : isCanonical x F.modulus = true) :
    inversion.invert_with I (limbsArray x) (limbsArray F.modulus) (word F.inv)
      (limbsArray (Inversion.startE F))
    ⦃ (res : Std.Array Std.U64 4#usize) =>
      (limbsOfArray res).toNat < F.modulus.toNat ∧
        (x.toNat = 0 → limbsOfArray res = Limbs.ofNat 0) ∧
        (x.toNat ≠ 0 →
          x.toNat * (limbsOfArray res).toNat ≡ R^2 [MOD F.modulus.toNat]) ⦄ := by
  apply WP.spec_mono (invert_with_spec hS F hB x hx)
  intro res hres
  obtain ⟨-, hlt, hzero, hinverse⟩ := invert_entry_spec _ F hB x hx h _ rfl
  rw [hres]
  exact ⟨hlt, hzero, hinverse⟩

end Driver

end PastaCurves.Portable
