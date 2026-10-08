import PastaCurves.Spec
import PastaCurves.X86_64.Spec.Arithmetic
import PastaCurves.X86_64.Transcription
import Mathlib.Tactic.ClearExcept

/-!
# Correctness of x86-64 modular addition

The proof follows the three carry chains in the transcribed block: addition of the operands,
in-place subtraction of the modulus, and conditional addition of the modulus after a borrow.
The final public theorem states the canonical contract checked by the Rust entry point.
-/

namespace PastaCurves.X86_64

-- BEGIN addMod_spec statement
/-- Modular addition by the inline block, for operands whose sum fits in four limbs (so the carry
out of the addition, which the block drops, is `0`): the result is the sum when that is below `p`,
and the sum minus `p` otherwise, since the subtraction of `p` borrows exactly when the sum is below
`p`. -/
theorem addMod_spec (lhs rhs modulus : Limbs) (hlhs : lhs.Bounded) (hrhs : rhs.Bounded)
    (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hsum : lhs.toNat + rhs.toNat < 2^256) :
    ∀ res, res = addMod lhs rhs modulus →
      res.Bounded ∧
        ((lhs.toNat + rhs.toNat < modulus.toNat ∧ res.toNat = lhs.toNat + rhs.toNat) ∨
          (modulus.toNat ≤ lhs.toNat + rhs.toNat ∧
            res.toNat + modulus.toNat = lhs.toNat + rhs.toNat)) := by
  intro res hres
-- END addMod_spec statement
  -- generated skeleton for `addMod`: do not edit between the annotations
  unfold addMod at hres
  lift_lets -merge at hres
  -- r0: input r0
  word_step r0 := lhs.l0 using hlhs.1
  -- r1: input r1
  word_step r1 := lhs.l1 using hlhs.2.1
  -- r2: input r2
  word_step r2 := lhs.l2 using hlhs.2.2.1
  -- r3: input r3
  word_step r3 := lhs.l3 using hlhs.2.2.2
  -- b0: input rhs[0]
  word_step b0 := rhs.l0 using hrhs.1
  -- b1: input rhs[1]
  word_step b1 := rhs.l1 using hrhs.2.1
  -- b2: input rhs[2]
  word_step b2 := rhs.l2 using hrhs.2.2.1
  -- b3: input rhs[3]
  word_step b3 := rhs.l3 using hrhs.2.2.2
  -- p0: input modulus[0]
  word_step p0 := modulus.l0 using hm.1
  -- p1: input modulus[1]
  word_step p1 := modulus.l1 using hm.2.1
  -- p3: input modulus[3]
  word_step p3 := modulus.l3 using hm.2.2.2
  -- r0_1: add {r0}, {b0}
  word_step s, r0_1 := (addc r0 b0 0).1 using addc_value_lt r0 b0 0, cf := (addc r0 b0 0).2
  have l_r0_1 : r0_1 + 2^64 * cf = r0 + b0 + 0 := by
    rw [e_r0_1, e_cf]; exact addc_lin r0 b0 0
  have b_cf : cf ≤ 1 := by rw [e_cf]; exact addc_carry_le_one r0 b0 0 b_r0 b_b0 (by decide)
  clear e_r0_1 e_cf
  -- r1_1: adc {r1}, {b1}
  word_step s_1, r1_1 := (addc r1 b1 cf).1 using addc_value_lt r1 b1 cf, cf_1 := (addc r1 b1 cf).2
  have l_r1_1 : r1_1 + 2^64 * cf_1 = r1 + b1 + cf := by
    rw [e_r1_1, e_cf_1]; exact addc_lin r1 b1 cf
  have b_cf_1 : cf_1 ≤ 1 := by rw [e_cf_1]; exact addc_carry_le_one r1 b1 cf b_r1 b_b1 b_cf
  clear e_r1_1 e_cf_1
  -- r2_1: adc {r2}, {b2}
  word_step s_2, r2_1 := (addc r2 b2 cf_1).1 using addc_value_lt r2 b2 cf_1,
      cf_2 := (addc r2 b2 cf_1).2
  have l_r2_1 : r2_1 + 2^64 * cf_2 = r2 + b2 + cf_1 := by
    rw [e_r2_1, e_cf_2]; exact addc_lin r2 b2 cf_1
  have b_cf_2 : cf_2 ≤ 1 := by rw [e_cf_2]; exact addc_carry_le_one r2 b2 cf_1 b_r2 b_b2 b_cf_1
  clear e_r2_1 e_cf_2
  -- r3_1: adc {r3}, {b3}
  word_step s_3, r3_1 := (addc r3 b3 cf_2).1 using addc_value_lt r3 b3 cf_2,
      cf_3 := (addc r3 b3 cf_2).2
  have l_r3_1 : r3_1 + 2^64 * cf_3 = r3 + b3 + cf_2 := by
    rw [e_r3_1, e_cf_3]; exact addc_lin r3 b3 cf_2
  have b_cf_3 : cf_3 ≤ 1 := by rw [e_cf_3]; exact addc_carry_le_one r3 b3 cf_2 b_r3 b_b3 b_cf_2
  clear e_r3_1 e_cf_3
  -- r0_2: sub {r0}, {p0}
  word_step d, r0_2 := (sbb r0_1 p0 0).1 using sbb_value_lt r0_1 p0 0, cf_4 := (sbb r0_1 p0 0).2
  have l_r0_2 : r0_2 + p0 + 0 = r0_1 + 2^64 * cf_4 := by
    rw [e_r0_2, e_cf_4]; exact sbb_lin r0_1 p0 0 b_r0_1 b_p0 (by decide)
  have b_cf_4 : cf_4 ≤ 1 := by rw [e_cf_4]; exact sbb_borrow_le_one r0_1 p0 0
  clear e_r0_2 e_cf_4
  -- r1_2: sbb {r1}, {p1}
  word_step d_1, r1_2 := (sbb r1_1 p1 cf_4).1 using sbb_value_lt r1_1 p1 cf_4,
      cf_5 := (sbb r1_1 p1 cf_4).2
  have l_r1_2 : r1_2 + p1 + cf_4 = r1_1 + 2^64 * cf_5 := by
    rw [e_r1_2, e_cf_5]; exact sbb_lin r1_1 p1 cf_4 b_r1_1 b_p1 b_cf_4
  have b_cf_5 : cf_5 ≤ 1 := by rw [e_cf_5]; exact sbb_borrow_le_one r1_1 p1 cf_4
  clear e_r1_2 e_cf_5
  -- r2_2: sbb {r2}, 0
  word_step d_2, r2_2 := (sbb r2_1 0 cf_5).1 using sbb_value_lt r2_1 0 cf_5,
      cf_6 := (sbb r2_1 0 cf_5).2
  have l_r2_2 : r2_2 + 0 + cf_5 = r2_1 + 2^64 * cf_6 := by
    rw [e_r2_2, e_cf_6]; exact sbb_lin r2_1 0 cf_5 b_r2_1 (by decide) b_cf_5
  have b_cf_6 : cf_6 ≤ 1 := by rw [e_cf_6]; exact sbb_borrow_le_one r2_1 0 cf_5
  clear e_r2_2 e_cf_6
  -- r3_2: sbb {r3}, {p3}
  word_step d_3, r3_2 := (sbb r3_1 p3 cf_6).1 using sbb_value_lt r3_1 p3 cf_6,
      cf_7 := (sbb r3_1 p3 cf_6).2
  have l_r3_2 : r3_2 + p3 + cf_6 = r3_1 + 2^64 * cf_7 := by
    rw [e_r3_2, e_cf_7]; exact sbb_lin r3_1 p3 cf_6 b_r3_1 b_p3 b_cf_6
  have b_cf_7 : cf_7 ≤ 1 := by rw [e_cf_7]; exact sbb_borrow_le_one r3_1 p3 cf_6
  clear e_r3_2 e_cf_7
  -- z: mov {z}, 0
  word_step z := 0 using (by decide)
  -- p0_1: cmovnc {p0}, {z}
  word_step p0_1 := (if cf_7 = 0 then z else p0) using ite_lt b_z b_p0
  -- p1_1: cmovnc {p1}, {z}
  word_step p1_1 := (if cf_7 = 0 then z else p1) using ite_lt b_z b_p1
  -- p3_1: cmovnc {p3}, {z}
  word_step p3_1 := (if cf_7 = 0 then z else p3) using ite_lt b_z b_p3
  -- r0_3: add {r0}, {p0}
  word_step s_4, r0_3 := (addc r0_2 p0_1 0).1 using addc_value_lt r0_2 p0_1 0,
      cf_8 := (addc r0_2 p0_1 0).2
  have l_r0_3 : r0_3 + 2^64 * cf_8 = r0_2 + p0_1 + 0 := by
    rw [e_r0_3, e_cf_8]; exact addc_lin r0_2 p0_1 0
  have b_cf_8 : cf_8 ≤ 1 := by rw [e_cf_8]; exact addc_carry_le_one r0_2 p0_1 0 b_r0_2 b_p0_1 (by decide)
  clear e_r0_3 e_cf_8
  -- r1_3: adc {r1}, {p1}
  word_step s_5, r1_3 := (addc r1_2 p1_1 cf_8).1 using addc_value_lt r1_2 p1_1 cf_8,
      cf_9 := (addc r1_2 p1_1 cf_8).2
  have l_r1_3 : r1_3 + 2^64 * cf_9 = r1_2 + p1_1 + cf_8 := by
    rw [e_r1_3, e_cf_9]; exact addc_lin r1_2 p1_1 cf_8
  have b_cf_9 : cf_9 ≤ 1 := by rw [e_cf_9]; exact addc_carry_le_one r1_2 p1_1 cf_8 b_r1_2 b_p1_1 b_cf_8
  clear e_r1_3 e_cf_9
  -- r2_3: adc {r2}, 0
  word_step s_6, r2_3 := (addc r2_2 0 cf_9).1 using addc_value_lt r2_2 0 cf_9,
      cf_10 := (addc r2_2 0 cf_9).2
  have l_r2_3 : r2_3 + 2^64 * cf_10 = r2_2 + 0 + cf_9 := by
    rw [e_r2_3, e_cf_10]; exact addc_lin r2_2 0 cf_9
  have b_cf_10 : cf_10 ≤ 1 := by rw [e_cf_10]; exact addc_carry_le_one r2_2 0 cf_9 b_r2_2 (by decide) b_cf_9
  clear e_r2_3 e_cf_10
  -- r3_3: adc {r3}, {p3}
  word_step s_7, r3_3 := (addc r3_2 p3_1 cf_10).1 using addc_value_lt r3_2 p3_1 cf_10,
      cf_11 := (addc r3_2 p3_1 cf_10).2
  have l_r3_3 : r3_3 + 2^64 * cf_11 = r3_2 + p3_1 + cf_10 := by
    rw [e_r3_3, e_cf_11]; exact addc_lin r3_2 p3_1 cf_10
  have b_cf_11 : cf_11 ≤ 1 := by rw [e_cf_11]; exact addc_carry_le_one r3_2 p3_1 cf_10 b_r3_2 b_p3_1 b_cf_10
  clear e_r3_3 e_cf_11
  subst hres
  -- BEGIN conclusion
  have hL : lhs.toNat = r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3 := by
    rw [e_r0, e_r1, e_r2, e_r3]; rfl
  have hR : rhs.toNat = b0 + 2^64 * b1 + 2^128 * b2 + 2^192 * b3 := by
    rw [e_b0, e_b1, e_b2, e_b3]; rfl
  have hP : modulus.toNat = p0 + 2^64 * p1 + 2^192 * p3 := by
    rw [e_p0, e_p1, e_p3]
    simp only [Limbs.toNat, hshape.1, mul_zero, add_zero]
  have hS : r0_1 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 + 2^256 * cf_3
      = lhs.toNat + rhs.toNat := by
    rw [hL, hR]
    clear * - l_r0_1 l_r1_1 l_r2_1 l_r3_1
    omega
  have hcf_3 : cf_3 = 0 := by
    clear * - hS hsum
    omega
  have hD : r0_2 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + modulus.toNat
      = r0_1 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 + 2^256 * cf_7 := by
    rw [hP]
    clear * - l_r0_2 l_r1_2 l_r2_2 l_r3_2
    omega
  have hA : r0_3 + 2^64 * r1_3 + 2^128 * r2_3 + 2^192 * r3_3 + 2^256 * cf_11
      = r0_2 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 +
        (p0_1 + 2^64 * p1_1 + 2^192 * p3_1) := by
    clear * - l_r0_3 l_r1_3 l_r2_3 l_r3_3
    omega
  have hSD : r0_2 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + modulus.toNat
      = lhs.toNat + rhs.toNat + 2^256 * cf_7 := by
    clear * - hS hcf_3 hD
    omega
  have hD_lt : r0_2 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 < 2^256 := by
    clear * - b_r0_2 b_r1_2 b_r2_2 b_r3_2
    omega
  refine ⟨⟨b_r0_3, b_r1_3, b_r2_3, b_r3_3⟩, ?_⟩
  show (lhs.toNat + rhs.toNat < modulus.toNat ∧
      r0_3 + 2^64 * r1_3 + 2^128 * r2_3 + 2^192 * r3_3 = lhs.toNat + rhs.toNat) ∨
    (modulus.toNat ≤ lhs.toNat + rhs.toNat ∧
      r0_3 + 2^64 * r1_3 + 2^128 * r2_3 + 2^192 * r3_3 + modulus.toNat =
        lhs.toNat + rhs.toNat)
  obtain hborrow | hborrow : cf_7 = 0 ∨ cf_7 = 1 := by
    clear * - b_cf_7
    omega
  · rw [if_pos hborrow] at e_p0_1 e_p1_1 e_p3_1
    have hselected : p0_1 + 2^64 * p1_1 + 2^192 * p3_1 = 0 := by
      clear * - e_z e_p0_1 e_p1_1 e_p3_1
      omega
    have hOut : r0_3 + 2^64 * r1_3 + 2^128 * r2_3 + 2^192 * r3_3 + 2^256 * cf_11
        = r0_2 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 := by
      clear * - hA hselected
      omega
    have hfinalCarry : cf_11 = 0 := by
      clear * - hOut hD_lt
      omega
    right
    clear * - hSD hborrow hOut hfinalCarry
    omega
  · rw [if_neg (by omega)] at e_p0_1 e_p1_1 e_p3_1
    have hselected : p0_1 + 2^64 * p1_1 + 2^192 * p3_1 = modulus.toNat := by
      clear * - e_p0_1 e_p1_1 e_p3_1 hP
      omega
    have hOut : r0_3 + 2^64 * r1_3 + 2^128 * r2_3 + 2^192 * r3_3 + 2^256 * cf_11
        = r0_2 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + modulus.toNat := by
      clear * - hA hselected
      omega
    have hfinalCarry : cf_11 = 1 := by
      clear * - hSD hborrow hOut hsum b_r0_3 b_r1_3 b_r2_3 b_r3_3 b_cf_11
      omega
    left
    clear * - hSD hborrow hD_lt hOut hfinalCarry
    omega
  -- END conclusion

-- BEGIN addMod corollary
/-- Addition of canonical operands produces a canonical residue congruent to their sum. -/
theorem addMod_spec_of_lt (lhs rhs modulus : Limbs) (hlhs : lhs.Bounded) (hrhs : rhs.Bounded)
    (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hlhs_lt : lhs.toNat < modulus.toNat) (hrhs_lt : rhs.toNat < modulus.toNat) :
    ∀ res, res = addMod lhs rhs modulus →
      res.Bounded ∧ res.toNat < modulus.toNat ∧
        res.toNat ≡ lhs.toNat + rhs.toNat [MOD modulus.toNat] := by
  intro res hres
  have hP := Limbs.toNat_lt_of_shape modulus hm hshape
  obtain ⟨hb, hcases⟩ := addMod_spec lhs rhs modulus hlhs hrhs hm hshape (by omega) res hres
  refine ⟨hb, ?_⟩
  rcases hcases with ⟨hlt, heq⟩ | ⟨hge, heq⟩
  · exact ⟨by omega, modEq_of_add_mul _ _ 0 0 _ (by omega)⟩
  · exact ⟨by omega, modEq_of_add_mul _ _ 1 0 _ (by omega)⟩
-- END addMod corollary

end PastaCurves.X86_64
