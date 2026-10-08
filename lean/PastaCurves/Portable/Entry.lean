import PastaCurves.Portable.Divstep59
import PastaCurves.Portable.SignMag
import PastaCurves.Portable.Amontred
import PastaCurves.Portable.Driver

/-!
# The portable blocks at the crate's fields

The translated portable blocks succeed where the driver needs them to (`blocks_succeeds`). As
functions on the model's values, they satisfy `InvertBlocks.Spec` at any Pasta field
(`blocks_spec`), by the blocks' theorems. So Aeneas' translation of `invert_with` over them returns
the Montgomery inverse at either of the crate's fields (`invert_entry_spec`), as the AArch64 blocks
do.

Every block but `fg_row` and `de_row` succeeds whatever words it is given. Those two succeed when
the magnitudes `m0` and `m1` of their row's entries sum to at most `2^63`. The contracts' step
lemmas assume more than that, so the proofs of success (`divstep59_ok` and the others) name the
callee's own success lemma wherever `step*` would apply a contract.
-/

set_option exponentiation.threshold 400

namespace PastaCurves.Portable

open Aeneas Aeneas.Std

open pasta_curves.inversion.portable in
/-- One packed divstep succeeds on any words. -/
theorem divstep_ok (two_delta f g : Std.U64) :
    divstep two_delta f g ⦃ _ => True ⦄ := by
  unfold divstep
  step*
  · agrind [Nat.and_one_is_mod]
  · agrind [mask_and]
  · agrind [mask_and]
  · agrind [mask_and]

open pasta_curves.inversion.portable in
/-- A batch's loop succeeds on any words. -/
theorem batch_loop_ok (iter : core.ops.range.Range Std.U32) (two_delta f g : Std.U64) :
    batch_loop iter two_delta f g ⦃ _ => True ⦄ := by
  unfold batch_loop
  apply loop.spec_decr_nat (fun (it, _) => it.«end».val - it.start.val) (fun _ => True)
  · rintro ⟨it, two_delta, f, g⟩ -
    unfold batch_loop.body
    step
    split
    · simp
    · step with divstep_ok
      agrind
  · trivial

open pasta_curves.inversion.portable in
/-- The decoder succeeds on any word, for a batch of at most 20 steps. -/
theorem unpack_ok (k : Std.U32) (hk : k.val ≤ 20) (w : Std.U64) :
    unpack k w ⦃ _ => True ⦄ := by
  unfold unpack
  step*

open pasta_curves.inversion.portable in
/-- A batch of at most 20 steps succeeds on any words. -/
theorem batch_ok (k : Std.U32) (hk : k.val ≤ 20) (two_delta f g : Std.U64) :
    batch k two_delta f g ⦃ _ => True ⦄ := by
  unfold batch
  step*
  step with batch_loop_ok
  step with unpack_ok k hk
  step with unpack_ok k hk

open pasta_curves.inversion.portable in
/-- The next low word succeeds on any words, for a shift below `64`. -/
theorem next_low_ok (k : Std.U32) (hk : k.val < 64) (a b f g : Std.U64) :
    next_low k a b f g ⦃ _ => True ⦄ := by
  unfold next_low
  step*

open pasta_curves.inversion.portable in
/-- The matrix product succeeds on any words. -/
theorem mat_mul_ok (m n : Std.Array Std.U64 4#usize) : mat_mul m n ⦃ _ => True ⦄ := by
  unfold mat_mul
  step*

open pasta_curves.inversion.portable in
/-- `divstep59` succeeds on any words. -/
theorem divstep59_ok (two_delta f0 g0 : Std.U64) :
    Backend.Insts.Pasta_curvesInversionInvertBlocks.divstep59 two_delta f0 g0 ⦃ _ => True ⦄ := by
  unfold Backend.Insts.Pasta_curvesInversionInvertBlocks.divstep59
  -- `step*` would reach for the contracts' step lemmas, so each call names its success lemma.
  iterate 2
    step with batch_ok 20#u32 (by decide)
    iterate 5 step
    iterate 2 step with next_low_ok 20#u32 (by decide)
  step with batch_ok 19#u32 (by decide)
  iterate 5 step
  iterate 2 step with mat_mul_ok
  step*

open pasta_curves.inversion.portable in
/-- `sign_mag` succeeds on any words. -/
theorem sign_mag_ok (u v q r : Std.U64) :
    Backend.Insts.Pasta_curvesInversionInvertBlocks.sign_mag u v q r ⦃ _ => True ⦄ := by
  unfold Backend.Insts.Pasta_curvesInversionInvertBlocks.sign_mag
  step*

open pasta_curves.inversion.portable in
/-- `negate` succeeds on any words and mask. -/
theorem negate_ok (x : Std.Array Std.U64 5#usize) (s : Std.U64) : negate x s ⦃ _ => True ⦄ := by
  unfold negate
  step*
  rw [carry_post, UScalar.val_and]
  exact Nat.and_le_right

open pasta_curves.inversion.portable in
/-- `row` succeeds on any words and masks when the magnitudes sum to at most `2^63`. -/
theorem row_ok (x y : Std.Array Std.U64 5#usize) (m0 m1 s0 s1 : Std.U64)
    (hm : m0.val + m1.val ≤ 2^63) :
    row x y m0 m1 s0 s1 ⦃ _ => True ⦄ := by
  unfold row
  step with negate_ok x s0
  step with negate_ok y s1
  step*

open pasta_curves.inversion.portable in
/-- `fg_row` succeeds on any words and masks when the magnitudes sum to at most `2^63`. -/
theorem fg_row_ok (f g : Std.Array Std.U64 5#usize) (m0 m1 s0 s1 : Std.U64)
    (hm : m0.val + m1.val ≤ 2^63) :
    Backend.Insts.Pasta_curvesInversionInvertBlocks.fg_row f g m0 m1 s0 s1 ⦃ _ => True ⦄ := by
  unfold Backend.Insts.Pasta_curvesInversionInvertBlocks.fg_row
  step with row_ok f g m0 m1 s0 s1 hm
  step*

open pasta_curves.inversion.portable in
/-- `de_row` succeeds on any words and masks when the magnitudes sum to at most `2^63`. -/
theorem de_row_ok (d e : Std.Array Std.U64 4#usize) (m0 m1 s0 s1 : Std.U64)
    (hm : m0.val + m1.val ≤ 2^63) :
    Backend.Insts.Pasta_curvesInversionInvertBlocks.de_row d e m0 m1 s0 s1 ⦃ _ => True ⦄ := by
  unfold Backend.Insts.Pasta_curvesInversionInvertBlocks.de_row
  -- The eight limb reads, then the row.
  iterate 8 step
  step with row_ok _ _ m0 m1 s0 s1 hm

open pasta_curves.inversion.portable in
/-- `amontred` succeeds on any words. -/
theorem amontred_ok (t : Std.Array Std.U64 5#usize) (modulus : Std.Array Std.U64 4#usize)
    (inv : Std.U64) :
    Backend.Insts.Pasta_curvesInversionInvertBlocks.amontred t modulus inv ⦃ _ => True ⦄ := by
  unfold Backend.Insts.Pasta_curvesInversionInvertBlocks.amontred
  step*

open pasta_curves.inversion.portable in
/-- `cond_sub` succeeds on any words: its contract holds at any bounded modulus, and every array
of words is the array of its readback. -/
theorem cond_sub_ok (value modulus : Std.Array Std.U64 4#usize) :
    Backend.Insts.Pasta_curvesInversionInvertBlocks.cond_sub value modulus ⦃ _ => True ⦄ := by
  rw [← limbsArray_limbsOfArray value, ← limbsArray_limbsOfArray modulus]
  exact WP.spec_mono (cond_sub_spec _ _ (limbsOfArray_bounded value)
    (limbsOfArray_bounded modulus)) fun _ _ => trivial

/-- A call that satisfies a triple succeeds. -/
theorem ok_of_spec {α : Type} {m : Result α} {P : α → Prop} (h : m ⦃ a => P a ⦄) :
    ∃ a, m = .ok a :=
  (WP.spec_imp_exists h).imp fun _ h => h.1

/-- The result of a call that satisfies a triple, read back with any default, satisfies the
triple's postcondition. -/
theorem getD_of_spec {α : Type} {m : Result α} {P : α → Prop} (h : m ⦃ a => P a ⦄) (d : α) :
    P ((Option.ofResult m).getD d) := by
  obtain ⟨a, rfl, ha⟩ := WP.spec_imp_exists h
  simpa [Option.ofResult] using ha

/-- The portable blocks succeed where the driver needs them to. Every block but `fg_row` and
`de_row` succeeds whatever words it is given. Those two succeed when the magnitudes `m0` and `m1` of
their row's entries sum to at most `2^63`. -/
theorem blocks_succeeds : Succeeds blocks where
  divstep59 two_delta f0 g0 := ok_of_spec (divstep59_ok two_delta f0 g0)
  signMag u v q r := ok_of_spec (sign_mag_ok u v q r)
  fgRow f g m0 m1 s0 s1 hm := ok_of_spec (fg_row_ok f g m0 m1 s0 s1 hm)
  deRow d e m0 m1 s0 s1 hm := ok_of_spec (de_row_ok d e m0 m1 s0 s1 hm)
  amontred t modulus inv := ok_of_spec (amontred_ok t modulus inv)
  condSub value modulus := ok_of_spec (cond_sub_ok value modulus)

/-- A natural number that is some integer's residue modulo `2^64` is below `2^64`. -/
theorem lt_of_eq_emod {n : ℕ} {z : ℤ} (h : (n : ℤ) = z % 2^64) : n < 2^64 := by
  have := Int.emod_lt_of_pos z (by norm_num : (0 : ℤ) < 2^64)
  agrind

/-- The portable blocks, as functions on the model's values, satisfy `InvertBlocks.Spec` at any
Pasta field, by the blocks' theorems. -/
theorem blocks_spec (F : PastaField) : (pureBlocks blocks).Spec F where
  divstep59 two_delta f0 g0 s hf hd hD ed ef eg := by
    rw [← word_val (lt_of_eq_emod ed)] at ed
    rw [← word_val (lt_of_eq_emod ef)] at ef
    rw [← word_val (lt_of_eq_emod eg)] at eg
    exact getD_of_spec (divstep59_spec _ _ _ s hf hd hD ed ef eg) _
  signMag zu zv zq zr u v q r hzu hzv hzq hzr eu ev eq er := by
    rw [← word_val (lt_of_eq_emod eu)] at eu
    rw [← word_val (lt_of_eq_emod ev)] at ev
    rw [← word_val (lt_of_eq_emod eq)] at eq
    rw [← word_val (lt_of_eq_emod er)] at er
    have h := getD_of_spec (sign_mag_spec _ _ _ _ zu zv zq zr hzu hzv hzq hzr eu ev eq er)
      (Std.Array.repeat 8#usize 0#u64)
    obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7⟩ := h
    simp only [pureBlocks, signMagOfArray]
    rw [h0, h1, h2, h3, h4, h5, h6, h7]
  fgRow a b f g m0 m1 s0 s1 hf hg hfv hgv hab ha hb := by
    have := abs_nonneg a
    have := abs_nonneg b
    obtain ⟨hm0, hs0⟩ := Inversion.SignMagRep.lt m0 s0 a ha (by agrind)
    obtain ⟨hm1, hs1⟩ := Inversion.SignMagRep.lt m1 s1 b hb (by agrind)
    rw [← word_val hm0, ← word_val hs0] at ha
    rw [← word_val hm1, ← word_val hs1] at hb
    exact getD_of_spec (fg_row_spec a b f g _ _ _ _ hf hg hfv hgv hab ha hb) _
  deRow a b d e m0 m1 s0 s1 hd he hab ha hb := by
    have := abs_nonneg a
    have := abs_nonneg b
    obtain ⟨hm0, hs0⟩ := Inversion.SignMagRep.lt m0 s0 a ha (by agrind)
    obtain ⟨hm1, hs1⟩ := Inversion.SignMagRep.lt m1 s1 b hb (by agrind)
    rw [← word_val hm0, ← word_val hs0] at ha
    rw [← word_val hm1, ← word_val hs1] at hb
    exact getD_of_spec (de_row_spec a b d e _ _ _ _ hd he hab ha hb) _
  amontred t ht htv := getD_of_spec (amontred_spec F t ht htv) _
  condSub value hv := getD_of_spec (cond_sub_spec value F.modulus hv F.bounded) _

/-- The crate's `invert` at a Pasta field, on the portable blocks: Aeneas' translation of
`invert_with` over them. The input is canonical, as the entry point asserts, and the starting `e` is
`2^562 mod p`, as its contract requires. The result is canonical. For `x = 0` it is `0`; otherwise
it is the Montgomery inverse, with `x * result ≡ R^2 (mod p)`. -/
theorem invert_entry_spec (F : PastaField) (x : Limbs) (hx : x.Bounded)
    (h : isCanonical x F.modulus = true) :
    pasta_curves.inversion.invert_with blocks (limbsArray x) (limbsArray F.modulus) (word F.inv)
      (limbsArray (Inversion.startE F))
    ⦃ (res : Std.Array Std.U64 4#usize) =>
      (limbsOfArray res).toNat < F.modulus.toNat ∧
        (x.toNat = 0 → limbsOfArray res = Limbs.ofNat 0) ∧
        (x.toNat ≠ 0 →
          x.toNat * (limbsOfArray res).toNat ≡ R^2 [MOD F.modulus.toNat]) ⦄ :=
  invert_with_entry_spec blocks_succeeds F (blocks_spec F) x hx h

end PastaCurves.Portable
