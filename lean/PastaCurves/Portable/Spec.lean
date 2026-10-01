import PastaCurves.Portable.Words
import PastaCurves.Inversion.Composition

/-!
# The translated portable blocks meet their contracts

Each theorem here states that a block of Aeneas' translation of `src/inversion/portable.rs`
(`Funs.lean`) succeeds and returns what the block's contract in `InvertBlocks.Spec`
(`Inversion/Composition.lean`) states, on the model's values written as the translation's
words (`Words.lean`). The statements are Aeneas' Hoare triples, `f ⦃ res => P res ⦄`, and the proofs
step through the translation with Aeneas' `step*`, then close the arithmetic by hand.

The masked selects are proved on 64-bit vectors, by `bv_decide`, and the borrow chains on natural
numbers, one limb at a time.
-/

namespace PastaCurves.Portable

open Aeneas Aeneas.Std

/-- The value of a Boolean as a word: `1` for true, `0` for false. -/
@[step_pure core.convert.num.FromU64Bool.from b]
theorem fromU64Bool_val (b : Bool) :
    (core.convert.num.FromU64Bool.from b).val = if b then 1 else 0 := by
  cases b <;> rfl

-- Aeneas' `wrapping_sub_bv_eq` states `(core.num.U64.wrapping_sub x y).bv = x.bv - y.bv`.
-- `step_pure` makes it a `step` lemma for `lift (core.num.U64.wrapping_sub x y)`, so that `step*`
-- records each wrapping subtraction's result as that difference of bit vectors, the form that
-- `select_post` takes. `local` confines the lemma to this file.
attribute [local step_pure core.num.U64.wrapping_sub x y] core.num.U64.wrapping_sub_bv_eq

/-- The masked select on 64-bit words: with `b` zero or one, `(0 - b) & a | !(0 - b) & d` is `a`
when `b` is one and `d` when it is zero. -/
theorem select_bv (a d b : BitVec 64) (hb : b = 0#64 ∨ b = 1#64) :
    ((0#64 - b) &&& a) ||| (~~~(0#64 - b) &&& d) = if b = 1#64 then a else d := by
  rcases hb with rfl | rfl <;> bv_decide

/-- The masked select on one word, from the post-conditions of its operations: with `b` zero or
one, the output is `a` when `b` is one and `d` when it is zero. -/
theorem select_post (o x y keep nk a d b : Std.U64) (hb : b.val = 0 ∨ b.val = 1)
    (hkeep : keep.bv = (0#u64 : Std.U64).bv - b.bv) (hnk : nk = ~~~keep)
    (hx : x.bv = keep.bv &&& a.bv) (hy : y.bv = nk.bv &&& d.bv) (ho : o.bv = x.bv ||| y.bv) :
    o.val = if b.val = 1 then a.val else d.val := by
  have hbv : b.bv = 0#64 ∨ b.bv = 1#64 := by
    rcases hb with hb | hb
    · left; apply BitVec.eq_of_toNat_eq; simpa using hb
    · right; apply BitVec.eq_of_toNat_eq; simpa using hb
  have hcond : (b.val = 1) ↔ (b.bv = 1#64) := by
    constructor
    · intro h; apply BitVec.eq_of_toNat_eq; simpa using h
    · intro h; have := congrArg BitVec.toNat h; simpa using this
  have hobv : o.bv = if b.bv = 1#64 then a.bv else d.bv := by
    subst hnk
    rw [ho, hx, hy]
    simp only [UScalar.bv_not, hkeep, show (0#u64 : Std.U64).bv = 0#64 from rfl]
    exact select_bv a.bv d.bv b.bv hbv
  have hval : o.val = o.bv.toNat := rfl
  split_ifs with h
  · rw [hval, hobv, if_pos (hcond.1 h)]; rfl
  · rw [hval, hobv, if_neg (fun h' => h (hcond.2 h'))]; rfl

/-- One limb of a borrow chain, from the post-conditions of its two subtractions and of the
borrow's conversion: the difference word, the subtrahend, and the borrow in make the minuend and
the borrow out, which is zero or one. -/
theorem sbb_step (a b cin : Nat) (word word1 bout : Std.U64) (u1 u2 : Bool) (hb : b < 2^64)
    (hcin : cin ≤ 1)
    (h1 : if a < b then word.val + b = a + U64.size ∧ u1 = true else word.val = a - b ∧ u1 = false)
    (h2 : if word.val < cin then word1.val + cin = word.val + U64.size ∧ u2 = true
      else word1.val = word.val - cin ∧ u2 = false)
    (h3 : bout.val = if (u1 || u2) = true then 1 else 0) :
    word1.val + b + cin = a + 2^64 * bout.val ∧ bout.val ≤ 1 := by
  have hw := word.hBounds
  have hw1 := word1.hBounds
  simp only [U64.size] at h1 h2
  cases u1 <;> cases u2 <;> split_ifs at h1 h2 <;> simp_all <;> scalar_tac

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
    at i_post i1_post i2_post i3_post i4_post i5_post i6_post i7_post
  subst i_post i1_post i2_post i3_post i4_post i5_post i6_post i7_post
  rw [word_val _ hv0, word_val _ hm0] at word_post
  rw [word_val _ hv1, word_val _ hm1] at word2_post
  rw [word_val _ hv2, word_val _ hm2] at word4_post
  rw [word_val _ hv3, word_val _ hm3] at word6_post
  -- The differences, read back from the array they were written to.
  subst difference1_post difference2_post difference3_post difference4_post
  simp at i10_post i15_post i20_post i25_post
  -- The borrow chain, limb by limb.
  obtain ⟨c0, b0⟩ := sbb_step v.l0 m.l0 0 word word1 borrow underflow1 underflow2 hm0 (by omega)
    word_post (by simpa using word1_post) (by simpa using borrow_post)
  obtain ⟨c1, b1⟩ := sbb_step v.l1 m.l1 borrow.val word2 word3 borrow1 underflow11 underflow21 hm1
    b0 word2_post word3_post (by simpa using borrow1_post)
  obtain ⟨c2, b2⟩ := sbb_step v.l2 m.l2 borrow1.val word4 word5 borrow2 underflow12 underflow22 hm2
    b1 word4_post word5_post (by simpa using borrow2_post)
  obtain ⟨c3, b3⟩ := sbb_step v.l3 m.l3 borrow2.val word6 word7 borrow3 underflow13 underflow23 hm3
    b2 word6_post word7_post (by simpa using borrow3_post)
  have hb3 : borrow3.val = 0 ∨ borrow3.val = 1 := by omega
  -- Each output word is the input's when the subtraction borrows, the difference's otherwise.
  have o0 := select_post i12 i8 i11 keep i9 _ i10 borrow3 hb3 keep_post i9_post i8_post1 i11_post1
    i12_post1
  have o1 := select_post i17 i13 i16 keep i14 _ i15 borrow3 hb3 keep_post i14_post i13_post1
    i16_post1 i17_post1
  have o2 := select_post i22 i18 i21 keep i19 _ i20 borrow3 hb3 keep_post i19_post i18_post1
    i21_post1 i22_post1
  have o3 := select_post i27 i23 i26 keep i24 _ i25 borrow3 hb3 keep_post i24_post i23_post1
    i26_post1 i27_post1
  rw [word_val _ hv0, i10_post] at o0
  rw [word_val _ hv1, i15_post] at o1
  rw [word_val _ hv2, i20_post] at o2
  rw [word_val _ hv3, i25_post] at o3
  have res_eq : limbsOfArray res = ⟨i12.val, i17.val, i22.val, i27.val⟩ := by
    simp [limbsOfArray, res_post, out3_post, out2_post, out1_post]
  rw [res_eq]
  have d0 : word1.val < 2^64 := word1.hBounds
  have d1 : word3.val < 2^64 := word3.hBounds
  have d2 : word5.val < 2^64 := word5.hBounds
  have d3 : word7.val < 2^64 := word7.hBounds
  simp only [Limbs.Bounded, Limbs.toNat]
  clear * - c0 c1 c2 c3 b0 b1 b2 hb3 o0 o1 o2 o3 hv0 hv1 hv2 hv3 hm0 hm1 hm2 hm3 d0 d1 d2 d3
  rcases hb3 with h | h
  · rw [h] at c3 o0 o1 o2 o3
    simp only [show ¬((0 : Nat) = 1) by decide, if_false] at o0 o1 o2 o3
    rw [o0, o1, o2, o3]
    omega
  · rw [h] at c3 o0 o1 o2 o3
    simp only [if_true] at o0 o1 o2 o3
    rw [o0, o1, o2, o3]
    omega

end PastaCurves.Portable
