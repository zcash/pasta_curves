import PastaCurves.Spec
import PastaCurves.X86_64.Spec.Arithmetic
import PastaCurves.X86_64.Transcription
import Mathlib.Tactic.ClearExcept

/-!
# Correctness of x86-64 modular subtraction

The transcribed block first subtracts its operands in place. On x86-64, CF is set on borrow, so
the conditional moves retain the modulus limbs exactly in the underflow case. The second carry
chain adds those selected limbs and deliberately discards its final carry.
-/

namespace PastaCurves.X86_64

-- BEGIN subMod_spec statement
/-- The subtraction block returns either the exact difference or the difference plus one modulus. -/
theorem subMod_spec (lhs rhs modulus : Limbs) (hlhs : lhs.Bounded) (hrhs : rhs.Bounded)
    (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hrhs_lt : rhs.toNat < modulus.toNat) :
    ∀ r, r = subMod lhs rhs modulus →
      r.Bounded ∧
        ((rhs.toNat ≤ lhs.toNat ∧ r.toNat + rhs.toNat = lhs.toNat) ∨
          (lhs.toNat < rhs.toNat ∧ r.toNat + rhs.toNat = lhs.toNat + modulus.toNat)) := by
  intro r hr
-- END subMod_spec statement
  -- generated skeleton for `subMod`: do not edit between the annotations
  unfold subMod at hr
  lift_lets -merge at hr
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
  -- r0_1: sub {r0}, {b0}
  word_step d, r0_1 := (sbb r0 b0 0).1 using sbb_value_lt r0 b0 0, cf := (sbb r0 b0 0).2
  have l_r0_1 : r0_1 + b0 + 0 = r0 + 2^64 * cf := by
    rw [e_r0_1, e_cf]; exact sbb_lin r0 b0 0 b_r0 b_b0 (by decide)
  have b_cf : cf ≤ 1 := by rw [e_cf]; exact sbb_borrow_le_one r0 b0 0
  clear e_r0_1 e_cf
  -- r1_1: sbb {r1}, {b1}
  word_step d_1, r1_1 := (sbb r1 b1 cf).1 using sbb_value_lt r1 b1 cf, cf_1 := (sbb r1 b1 cf).2
  have l_r1_1 : r1_1 + b1 + cf = r1 + 2^64 * cf_1 := by
    rw [e_r1_1, e_cf_1]; exact sbb_lin r1 b1 cf b_r1 b_b1 b_cf
  have b_cf_1 : cf_1 ≤ 1 := by rw [e_cf_1]; exact sbb_borrow_le_one r1 b1 cf
  clear e_r1_1 e_cf_1
  -- r2_1: sbb {r2}, {b2}
  word_step d_2, r2_1 := (sbb r2 b2 cf_1).1 using sbb_value_lt r2 b2 cf_1,
      cf_2 := (sbb r2 b2 cf_1).2
  have l_r2_1 : r2_1 + b2 + cf_1 = r2 + 2^64 * cf_2 := by
    rw [e_r2_1, e_cf_2]; exact sbb_lin r2 b2 cf_1 b_r2 b_b2 b_cf_1
  have b_cf_2 : cf_2 ≤ 1 := by rw [e_cf_2]; exact sbb_borrow_le_one r2 b2 cf_1
  clear e_r2_1 e_cf_2
  -- r3_1: sbb {r3}, {b3}
  word_step d_3, r3_1 := (sbb r3 b3 cf_2).1 using sbb_value_lt r3 b3 cf_2,
      cf_3 := (sbb r3 b3 cf_2).2
  have l_r3_1 : r3_1 + b3 + cf_2 = r3 + 2^64 * cf_3 := by
    rw [e_r3_1, e_cf_3]; exact sbb_lin r3 b3 cf_2 b_r3 b_b3 b_cf_2
  have b_cf_3 : cf_3 ≤ 1 := by rw [e_cf_3]; exact sbb_borrow_le_one r3 b3 cf_2
  clear e_r3_1 e_cf_3
  -- z: mov {z}, 0
  word_step z := 0 using (by decide)
  -- p0_1: cmovnc {p0}, {z}
  word_step p0_1 := (if cf_3 = 0 then z else p0) using ite_lt b_z b_p0
  -- p1_1: cmovnc {p1}, {z}
  word_step p1_1 := (if cf_3 = 0 then z else p1) using ite_lt b_z b_p1
  -- p3_1: cmovnc {p3}, {z}
  word_step p3_1 := (if cf_3 = 0 then z else p3) using ite_lt b_z b_p3
  -- r0_2: add {r0}, {p0}
  word_step s, r0_2 := (addc r0_1 p0_1 0).1 using addc_value_lt r0_1 p0_1 0,
      cf_4 := (addc r0_1 p0_1 0).2
  have l_r0_2 : r0_2 + 2^64 * cf_4 = r0_1 + p0_1 + 0 := by
    rw [e_r0_2, e_cf_4]; exact addc_lin r0_1 p0_1 0
  have b_cf_4 : cf_4 ≤ 1 := by rw [e_cf_4]; exact addc_carry_le_one r0_1 p0_1 0 b_r0_1 b_p0_1 (by decide)
  clear e_r0_2 e_cf_4
  -- r1_2: adc {r1}, {p1}
  word_step s_1, r1_2 := (addc r1_1 p1_1 cf_4).1 using addc_value_lt r1_1 p1_1 cf_4,
      cf_5 := (addc r1_1 p1_1 cf_4).2
  have l_r1_2 : r1_2 + 2^64 * cf_5 = r1_1 + p1_1 + cf_4 := by
    rw [e_r1_2, e_cf_5]; exact addc_lin r1_1 p1_1 cf_4
  have b_cf_5 : cf_5 ≤ 1 := by rw [e_cf_5]; exact addc_carry_le_one r1_1 p1_1 cf_4 b_r1_1 b_p1_1 b_cf_4
  clear e_r1_2 e_cf_5
  -- r2_2: adc {r2}, 0
  word_step s_2, r2_2 := (addc r2_1 0 cf_5).1 using addc_value_lt r2_1 0 cf_5,
      cf_6 := (addc r2_1 0 cf_5).2
  have l_r2_2 : r2_2 + 2^64 * cf_6 = r2_1 + 0 + cf_5 := by
    rw [e_r2_2, e_cf_6]; exact addc_lin r2_1 0 cf_5
  have b_cf_6 : cf_6 ≤ 1 := by rw [e_cf_6]; exact addc_carry_le_one r2_1 0 cf_5 b_r2_1 (by decide) b_cf_5
  clear e_r2_2 e_cf_6
  -- r3_2: adc {r3}, {p3}
  word_step s_3, r3_2 := (addc r3_1 p3_1 cf_6).1 using addc_value_lt r3_1 p3_1 cf_6,
      cf_7 := (addc r3_1 p3_1 cf_6).2
  have l_r3_2 : r3_2 + 2^64 * cf_7 = r3_1 + p3_1 + cf_6 := by
    rw [e_r3_2, e_cf_7]; exact addc_lin r3_1 p3_1 cf_6
  have b_cf_7 : cf_7 ≤ 1 := by rw [e_cf_7]; exact addc_carry_le_one r3_1 p3_1 cf_6 b_r3_1 b_p3_1 b_cf_6
  clear e_r3_2 e_cf_7
  subst hr
  -- BEGIN conclusion
  have hL : lhs.toNat = r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3 := by
    rw [e_r0, e_r1, e_r2, e_r3]; rfl
  have hR : rhs.toNat = b0 + 2^64 * b1 + 2^128 * b2 + 2^192 * b3 := by
    rw [e_b0, e_b1, e_b2, e_b3]; rfl
  have hP : modulus.toNat = p0 + 2^64 * p1 + 2^192 * p3 := by
    rw [e_p0, e_p1, e_p3]; simp only [Limbs.toNat, hshape.1, mul_zero, add_zero]
  -- The subtraction, wrapped modulo `2^256`: CF is set exactly when `lhs < rhs` (a borrow).
  have hD : r0_1 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 + rhs.toNat
      = lhs.toNat + 2^256 * cf_3 := by
    rw [hL, hR]
    clear * - l_r0_1 l_r1_1 l_r2_1 l_r3_1
    omega
  have hD_lt : r0_1 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 < 2^256 := by
    clear * - b_r0_1 b_r1_1 b_r2_1 b_r3_1
    omega
  -- The add-back of the selected limbs, with the carry that the block drops.
  have hA : r0_2 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + 2^256 * cf_7
      = r0_1 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 +
        (p0_1 + 2^64 * p1_1 + 2^192 * p3_1) := by
    clear * - l_r0_2 l_r1_2 l_r2_2 l_r3_2
    omega
  refine ⟨⟨b_r0_2, b_r1_2, b_r2_2, b_r3_2⟩, ?_⟩
  show (rhs.toNat ≤ lhs.toNat ∧
      r0_2 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + rhs.toNat = lhs.toNat) ∨
    (lhs.toNat < rhs.toNat ∧
      r0_2 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + rhs.toNat =
        lhs.toNat + modulus.toNat)
  obtain hborrow | hborrow : cf_3 = 0 ∨ cf_3 = 1 := by clear * - b_cf_3; omega
  · -- No borrow: the difference is exact, and the selected limbs are `0`.
    rw [if_pos hborrow] at e_p0_1 e_p1_1 e_p3_1
    have hselected : p0_1 + 2^64 * p1_1 + 2^192 * p3_1 = 0 := by
      clear * - e_z e_p0_1 e_p1_1 e_p3_1
      omega
    have hOut : r0_2 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + 2^256 * cf_7
        = r0_1 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 := by
      clear * - hA hselected
      omega
    have hfinalCarry : cf_7 = 0 := by
      clear * - hOut hD_lt
      omega
    left
    clear * - hD hborrow hOut hfinalCarry
    omega
  · -- Borrow: `lhs < rhs`, the selected limbs are `p`, and the dropped carry undoes the wrap.
    rw [if_neg (by omega)] at e_p0_1 e_p1_1 e_p3_1
    have hselected : p0_1 + 2^64 * p1_1 + 2^192 * p3_1 = modulus.toNat := by
      clear * - e_p0_1 e_p1_1 e_p3_1 hP
      omega
    have hOut : r0_2 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + 2^256 * cf_7
        = r0_1 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 + modulus.toNat := by
      clear * - hA hselected
      omega
    have hadd_gt : 2^256 <
        r0_1 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 + modulus.toNat := by
      clear * - hD hborrow hrhs_lt
      omega
    have hOut_lt : r0_2 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 < 2^256 := by
      clear * - b_r0_2 b_r1_2 b_r2_2 b_r3_2
      omega
    have hfinalCarry : cf_7 = 1 := by
      clear * - hOut hadd_gt hOut_lt b_cf_7
      omega
    right
    clear * - hD hD_lt hborrow hOut hfinalCarry
    omega
  -- END conclusion

-- BEGIN subMod corollary
/-- Subtraction of canonical operands produces a canonical modular difference. -/
theorem subMod_spec_of_lt (lhs rhs modulus : Limbs) (hlhs : lhs.Bounded) (hrhs : rhs.Bounded)
    (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hlhs_lt : lhs.toNat < modulus.toNat) (hrhs_lt : rhs.toNat < modulus.toNat) :
    ∀ r, r = subMod lhs rhs modulus →
      r.Bounded ∧ r.toNat < modulus.toNat ∧
        r.toNat + rhs.toNat ≡ lhs.toNat [MOD modulus.toNat] := by
  intro r hr
  obtain ⟨hb, hcases⟩ := subMod_spec lhs rhs modulus hlhs hrhs hm hshape (by omega) r hr
  refine ⟨hb, ?_⟩
  rcases hcases with ⟨hge, heq⟩ | ⟨hlt, heq⟩
  · exact ⟨by omega, modEq_of_add_mul _ _ 0 0 _ (by omega)⟩
  · exact ⟨by omega, modEq_of_add_mul _ _ 0 1 _ (by omega)⟩
-- END subMod corollary

end PastaCurves.X86_64
