import PastaCurves.Portable.Words
import PastaCurves.Inversion.Composition

/-!
# The translated portable blocks meet their contracts

Each theorem here states that a block of Aeneas' translation of `src/inversion/portable.rs`
(`Funs.lean`) succeeds and returns what the block's contract in `InvertBlocks.Spec`
(`Inversion/Composition.lean`) states, on the model's values written as the translation's
words (`Words.lean`). The statements are Aeneas' Hoare triples, `f ⦃ res => P res ⦄`, and the proofs
step through the translation with Aeneas' `step*`, then close the arithmetic by hand.

The blocks' helpers have step lemmas of their own, which `step*` applies at each call: `sbb`, one
limb of a borrow chain, proved on natural numbers; `select`, the masked select, proved on 64-bit
vectors; and `mask_of_bit`, the 64-bit mask of a bit.
-/

namespace PastaCurves.Portable

open Aeneas Aeneas.Std

open pasta_curves.inversion.portable in
/-- One limb of a borrow chain: with a borrow in of zero or one, `sbb` returns the difference
word and the borrow out, which with the subtrahend and the borrow in make up the minuend; the
borrow out is zero or one. -/
@[step]
theorem sbb_spec (a b borrow : Std.U64) (h : borrow.val ≤ 1) :
    sbb a b borrow ⦃ (difference : Std.U64) (borrow' : Std.U64) =>
      difference.val + b.val + borrow.val = a.val + 2^64 * borrow'.val ∧ borrow'.val ≤ 1 ⦄ := by
  unfold sbb
  step*
  split_ifs at * <;> scalar_tac

open pasta_curves.inversion.portable in
/-- The masked select: for a mask of all zeros or all ones, `select` returns `b` under the zero
mask and `a` under the all-ones mask. -/
@[step]
theorem select_spec (mask a b : Std.U64) (h : mask.val = 0 ∨ mask.val = 2^64 - 1) :
    select mask a b ⦃ (res : Std.U64) => res = if mask.val = 0 then b else a ⦄ := by
  unfold select
  step*
  subst i1_post
  rcases h with h | h
  · have hm : mask.bv = 0#64 := by apply BitVec.eq_of_toNat_eq; simpa using h
    rw [if_pos h, UScalar.eq_equiv_bv_eq, UScalar.bv_or, i_post1, i2_post1, UScalar.bv_not]
    simp [hm]
  · have hm : mask.bv = BitVec.allOnes 64 := by apply BitVec.eq_of_toNat_eq; simpa using h
    rw [if_neg (by omega), UScalar.eq_equiv_bv_eq, UScalar.bv_or, i_post1, i2_post1,
      UScalar.bv_not]
    simp [hm]

/-- The 64-bit mask of a bit: `0 - b` is zero when `b` is zero, and all ones when it is one. -/
theorem wrapping_sub_zero_bit_spec (b : Std.U64) (h : b.val ≤ 1) :
    lift (core.num.U64.wrapping_sub 0#u64 b) ⦃ (mask : Std.U64) =>
      (b.val = 0 ∧ mask.val = 0) ∨ (b.val = 1 ∧ mask.val = 2^64 - 1) ⦄ := by
  simp only [lift, WP.spec_ok, core.num.U64.wrapping_sub_val_eq]
  scalar_tac

open pasta_curves.inversion.portable in
/-- `mask_of_bit` is all ones for the bit one, and zero for zero. -/
@[step]
theorem mask_of_bit_spec (bit : Std.U64) (h : bit.val ≤ 1) :
    mask_of_bit bit ⦃ (mask : Std.U64) =>
      (bit.val = 0 ∧ mask.val = 0) ∨ (bit.val = 1 ∧ mask.val = 2^64 - 1) ⦄ := by
  unfold mask_of_bit
  exact wrapping_sub_zero_bit_spec bit h

open pasta_curves.inversion.portable in
/-- `cond_sub` subtracts `m` from `v` exactly when `v` is not below it: the translation meets the
`condSub` field of `InvertBlocks.Spec`, at any bounded `m`, not only a field's modulus. -/
theorem cond_sub_spec (v m : Limbs) (hv : v.Bounded) (hm : m.Bounded) :
    Backend.Insts.Pasta_curvesInversionInvertBlocks.cond_sub (limbsArray v) (limbsArray m)
      ⦃ res => (limbsOfArray res).Bounded ∧
        ((v.toNat < m.toNat ∧ (limbsOfArray res).toNat = v.toNat) ∨
          (m.toNat ≤ v.toNat ∧ (limbsOfArray res).toNat + m.toNat = v.toNat)) ⦄ := by
  unfold Backend.Insts.Pasta_curvesInversionInvertBlocks.cond_sub
  step*
  -- The words read from the inputs are the limbs, which are bounded.
  obtain ⟨hv0, hv1, hv2, hv3⟩ := hv
  obtain ⟨hm0, hm1, hm2, hm3⟩ := hm
  simp only [limbsArray, Std.Array.from_val, List.getElem_cons_zero, List.getElem_cons_succ]
    at i_post i1_post i3_post i4_post i6_post i7_post i9_post i10_post
  subst i_post i1_post i3_post i4_post i6_post i7_post i9_post i10_post
  rw [word_val _ hv0, word_val _ hm0] at i2_post
  rw [word_val _ hv1, word_val _ hm1] at i5_post
  rw [word_val _ hv2, word_val _ hm2] at i8_post
  rw [word_val _ hv3, word_val _ hm3] at i11_post
  -- The differences, read back from the array they were written to.
  subst difference1_post difference2_post difference3_post difference4_post
  simp at i12_post i14_post i16_post i18_post
  subst i12_post i14_post i16_post i18_post
  have res_eq : limbsOfArray res = ⟨i13.val, i15.val, i17.val, i19.val⟩ := by
    simp [limbsOfArray, res_post, out3_post, out2_post, out1_post]
  rw [res_eq]
  have d0 : i12.val < 2^64 := i12.hBounds
  have d1 : i14.val < 2^64 := i14.hBounds
  have d2 : i16.val < 2^64 := i16.hBounds
  have d3 : i18.val < 2^64 := i18.hBounds
  refine ⟨⟨i13.hBounds, i15.hBounds, i17.hBounds, i19.hBounds⟩, ?_⟩
  simp only [Limbs.toNat]
  -- Each output word is the difference's when the subtraction does not borrow, the input's
  -- otherwise.
  rcases keep_post with ⟨hb, hk⟩ | ⟨hb, hk⟩
  · simp only [hk, if_true] at i13_post i15_post i17_post i19_post
    subst i13_post i15_post i17_post i19_post
    rw [hb] at i11_post
    omega
  · have hk' : keep.val ≠ 0 := by omega
    simp only [hk', if_false] at i13_post i15_post i17_post i19_post
    rw [i13_post, i15_post, i17_post, i19_post, word_val _ hv0, word_val _ hv1, word_val _ hv2,
      word_val _ hv3]
    rw [hb] at i11_post
    omega

end PastaCurves.Portable
