import PastaCurves.Portable.Words
import PastaCurves.Inversion.Composition

/-!
# The translated portable blocks meet their contracts

Each theorem here states that a block of Aeneas' translation of `src/inversion/portable.rs`
(`Funs.lean`) succeeds and returns what the block's contract in `InvertBlocks.Spec`
(`Inversion/Composition.lean`) states, on the model's values written as the translation's
words (`Words.lean`). The statements are Aeneas' Hoare triples, `f ⦃ res => P res ⦄`, and the proofs
step through the translation with Aeneas' `step*`, then close the arithmetic by hand.

The blocks' helpers have step lemmas of their own, which `step*` applies at each call:

- `sbb`, one limb of a borrow chain, proved on natural numbers;
- `adc`, one limb of a carry chain;
- `row_column`, one column of a row, whose bound on the carry passes from each column to the next;
- `mac`, one column of a multiplication;
- `select`, the masked select, proved on 64-bit vectors;
- `mask_of_bit`, the 64-bit mask of a bit;
- `sign_mask`, the mask of a word's sign; and
- `halve`, the arithmetic shift right by one.
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
/-- One limb of a carry chain: with a carry in of zero or one, `adc` returns the sum word and the
carry out, which make up the sum of the operands and the carry in; the carry out is zero or one. -/
@[step]
theorem adc_spec (a b carry : Std.U64) (h : carry.val ≤ 1) :
    adc a b carry ⦃ (sum : Std.U64) (carry' : Std.U64) =>
      sum.val + 2^64 * carry'.val = a.val + b.val + carry.val ∧ carry'.val ≤ 1 ⦄ := by
  unfold adc
  step*
  split_ifs at * <;> scalar_tac

open pasta_curves.inversion.portable in
/-- One column of a row: with `carry ≤ m0 + m1 ≤ 2^63`, `row_column` returns the column's low word
and the carry into the next column, which make up `x m0 + y m1 + carry`; the carry out is again
at most `m0 + m1`. -/
@[step]
theorem row_column_spec (carry : Std.U128) (x m0 y m1 : Std.U64) (hm : m0.val + m1.val ≤ 2^63)
    (hc : carry.val ≤ m0.val + m1.val) :
    row_column carry x m0 y m1 ⦃ (word : Std.U64) (carry' : Std.U128) =>
      word.val + 2^64 * carry'.val = x.val * m0.val + y.val * m1.val + carry.val ∧
        carry'.val ≤ m0.val + m1.val ⦄ := by
  unfold row_column
  have hxm : x.val * m0.val ≤ (2^64 - 1) * m0.val := Nat.mul_le_mul_right _ (by scalar_tac)
  have hym : y.val * m1.val ≤ (2^64 - 1) * m1.val := Nat.mul_le_mul_right _ (by scalar_tac)
  step*
  -- Neither addition wraps: the column is at most `2^64 (m0 + m1) ≤ 2^127`.
  have hcol : column.val = carry.val + x.val * m0.val + y.val * m1.val := by
    rw [column_post, core.num.U128.wrapping_add_val_eq, i3_post, core.num.U128.wrapping_add_val_eq,
      i2_post, i6_post, i_post, i1_post, i4_post, i5_post]
    scalar_tac
  rw [i7_post, UScalar.cast_val_eq, UScalarTy.U64_numBits_eq, i8_post, Nat.shiftRight_eq_div_pow]
  refine ⟨?_, Nat.div_le_of_le_mul (by agrind)⟩
  rw [Nat.mod_add_div, hcol]
  agrind

open pasta_curves.inversion.portable in
/-- One column of a multiplication: `mac` returns the low and high words of `a + b c + carry`,
which is below `2^128` for any words. -/
@[step]
theorem mac_spec (a b c carry : Std.U64) :
    mac a b c carry ⦃ (lo : Std.U64) (hi : Std.U64) =>
      lo.val + 2^64 * hi.val = a.val + b.val * c.val + carry.val ⦄ := by
  unfold mac
  have hbc : b.val * c.val ≤ (2^64 - 1) * (2^64 - 1) :=
    Nat.mul_le_mul (by scalar_tac) (by scalar_tac)
  step*
  have hsum : sum.val / 2^64 < 2^64 := Nat.div_lt_of_lt_mul (by scalar_tac)
  rw [i6_post, i8_post, UScalar.cast_val_eq, UScalar.cast_val_eq, UScalarTy.U64_numBits_eq, i7_post,
    Nat.shiftRight_eq_div_pow, Nat.mod_eq_of_lt hsum, Nat.mod_add_div, sum_post, i4_post, i3_post,
    i_post, i1_post, i2_post, i5_post]

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
    rw [if_neg (by agrind), UScalar.eq_equiv_bv_eq, UScalar.bv_or, i_post1, i2_post1,
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
/-- The mask of a word's sign: zero for a word below `2^63`, and all ones otherwise. -/
@[step]
theorem sign_mask_spec (x : Std.U64) :
    sign_mask x ⦃ (res : Std.U64) => res.val = if x.val < 2^63 then 0 else 2^64 - 1 ⦄ := by
  unfold sign_mask
  step*
  exact sign_mask_val (by rw [i1_post, i_post])

open pasta_curves.inversion.portable in
/-- `halve` is the arithmetic shift right by one, `asr` of `Semantics.lean`. -/
@[step]
theorem halve_spec (x : Std.U64) :
    halve x ⦃ (res : Std.U64) => res.val = asr x.val 1 ⦄ := by
  unfold halve
  step*
  rw [IScalar.hcast_val_eq, i1_post, i_post, UScalar.hcast_val_eq, Int.shiftRight_eq_div_pow]
  agrind [asr, Int.bmod]

open pasta_curves.inversion.portable in
/-- `cond_sub` subtracts `m` from `v` exactly when `v` is not below it: the translation satisfies
the `condSub` field of `InvertBlocks.Spec`, at any bounded `m`, not only a field's modulus. -/
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
  rw [word_val hv0, word_val hm0] at i2_post
  rw [word_val hv1, word_val hm1] at i5_post
  rw [word_val hv2, word_val hm2] at i8_post
  rw [word_val hv3, word_val hm3] at i11_post
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
    grind
  · have hk' : keep.val ≠ 0 := by agrind
    simp only [hk', if_false] at i13_post i15_post i17_post i19_post
    rw [i13_post, i15_post, i17_post, i19_post, word_val hv0, word_val hv1, word_val hv2,
      word_val hv3]
    rw [hb] at i11_post
    agrind

end PastaCurves.Portable
