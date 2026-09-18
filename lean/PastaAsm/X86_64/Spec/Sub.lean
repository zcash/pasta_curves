/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.Spec
import PastaAsm.X86_64.Spec.Arithmetic
import PastaAsm.X86_64.Transcription
import Mathlib.Tactic.ClearExcept

/-!
# Correctness of x86-64 modular subtraction

The transcribed block first subtracts its operands in place. On x86-64, CF is set on borrow, so
the conditional moves retain the modulus limbs exactly in the underflow case. The second carry
chain adds those selected limbs and deliberately discards its final carry.
-/

namespace PastaAsm.X86_64

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
  extract_lets -merge +onlyGivenNames r0 at hr
  have e_r0 : r0 = lhs.l0 := rfl
  clear_value r0
  have b_r0 : r0 < 2^64 := by rw [e_r0]; exact hlhs.1
  -- r1: input r1
  extract_lets -merge +onlyGivenNames r1 at hr
  have e_r1 : r1 = lhs.l1 := rfl
  clear_value r1
  have b_r1 : r1 < 2^64 := by rw [e_r1]; exact hlhs.2.1
  -- r2: input r2
  extract_lets -merge +onlyGivenNames r2 at hr
  have e_r2 : r2 = lhs.l2 := rfl
  clear_value r2
  have b_r2 : r2 < 2^64 := by rw [e_r2]; exact hlhs.2.2.1
  -- r3: input r3
  extract_lets -merge +onlyGivenNames r3 at hr
  have e_r3 : r3 = lhs.l3 := rfl
  clear_value r3
  have b_r3 : r3 < 2^64 := by rw [e_r3]; exact hlhs.2.2.2
  -- b0: input rhs[0]
  extract_lets -merge +onlyGivenNames b0 at hr
  have e_b0 : b0 = rhs.l0 := rfl
  clear_value b0
  have b_b0 : b0 < 2^64 := by rw [e_b0]; exact hrhs.1
  -- b1: input rhs[1]
  extract_lets -merge +onlyGivenNames b1 at hr
  have e_b1 : b1 = rhs.l1 := rfl
  clear_value b1
  have b_b1 : b1 < 2^64 := by rw [e_b1]; exact hrhs.2.1
  -- b2: input rhs[2]
  extract_lets -merge +onlyGivenNames b2 at hr
  have e_b2 : b2 = rhs.l2 := rfl
  clear_value b2
  have b_b2 : b2 < 2^64 := by rw [e_b2]; exact hrhs.2.2.1
  -- b3: input rhs[3]
  extract_lets -merge +onlyGivenNames b3 at hr
  have e_b3 : b3 = rhs.l3 := rfl
  clear_value b3
  have b_b3 : b3 < 2^64 := by rw [e_b3]; exact hrhs.2.2.2
  -- p0: input modulus[0]
  extract_lets -merge +onlyGivenNames p0 at hr
  have e_p0 : p0 = modulus.l0 := rfl
  clear_value p0
  have b_p0 : p0 < 2^64 := by rw [e_p0]; exact hm.1
  -- p1: input modulus[1]
  extract_lets -merge +onlyGivenNames p1 at hr
  have e_p1 : p1 = modulus.l1 := rfl
  clear_value p1
  have b_p1 : p1 < 2^64 := by rw [e_p1]; exact hm.2.1
  -- p3: input modulus[3]
  extract_lets -merge +onlyGivenNames p3 at hr
  have e_p3 : p3 = modulus.l3 := rfl
  clear_value p3
  have b_p3 : p3 < 2^64 := by rw [e_p3]; exact hm.2.2.2
  -- r0_1: sub {r0}, {b0}
  extract_lets -merge +onlyGivenNames d r0_1 cf at hr
  have e_r0_1 : r0_1 = (sbb r0 b0 0).1 := rfl
  have e_cf : cf = (sbb r0 b0 0).2 := rfl
  clear_value d r0_1 cf
  have l_r0_1 : r0_1 + b0 + 0 = r0 + 2^64 * cf := by
    rw [e_r0_1, e_cf]; exact sbb_lin r0 b0 0 b_r0 b_b0 (by decide)
  have b_r0_1 : r0_1 < 2^64 := by rw [e_r0_1]; exact sbb_value_lt r0 b0 0
  have b_cf : cf ≤ 1 := by rw [e_cf]; exact sbb_borrow_le_one r0 b0 0
  clear e_r0_1 e_cf
  -- r1_1: sbb {r1}, {b1}
  extract_lets -merge +onlyGivenNames d_1 r1_1 cf_1 at hr
  have e_r1_1 : r1_1 = (sbb r1 b1 cf).1 := rfl
  have e_cf_1 : cf_1 = (sbb r1 b1 cf).2 := rfl
  clear_value d_1 r1_1 cf_1
  have l_r1_1 : r1_1 + b1 + cf = r1 + 2^64 * cf_1 := by
    rw [e_r1_1, e_cf_1]; exact sbb_lin r1 b1 cf b_r1 b_b1 b_cf
  have b_r1_1 : r1_1 < 2^64 := by rw [e_r1_1]; exact sbb_value_lt r1 b1 cf
  have b_cf_1 : cf_1 ≤ 1 := by rw [e_cf_1]; exact sbb_borrow_le_one r1 b1 cf
  clear e_r1_1 e_cf_1
  -- r2_1: sbb {r2}, {b2}
  extract_lets -merge +onlyGivenNames d_2 r2_1 cf_2 at hr
  have e_r2_1 : r2_1 = (sbb r2 b2 cf_1).1 := rfl
  have e_cf_2 : cf_2 = (sbb r2 b2 cf_1).2 := rfl
  clear_value d_2 r2_1 cf_2
  have l_r2_1 : r2_1 + b2 + cf_1 = r2 + 2^64 * cf_2 := by
    rw [e_r2_1, e_cf_2]; exact sbb_lin r2 b2 cf_1 b_r2 b_b2 b_cf_1
  have b_r2_1 : r2_1 < 2^64 := by rw [e_r2_1]; exact sbb_value_lt r2 b2 cf_1
  have b_cf_2 : cf_2 ≤ 1 := by rw [e_cf_2]; exact sbb_borrow_le_one r2 b2 cf_1
  clear e_r2_1 e_cf_2
  -- r3_1: sbb {r3}, {b3}
  extract_lets -merge +onlyGivenNames d_3 r3_1 cf_3 at hr
  have e_r3_1 : r3_1 = (sbb r3 b3 cf_2).1 := rfl
  have e_cf_3 : cf_3 = (sbb r3 b3 cf_2).2 := rfl
  clear_value d_3 r3_1 cf_3
  have l_r3_1 : r3_1 + b3 + cf_2 = r3 + 2^64 * cf_3 := by
    rw [e_r3_1, e_cf_3]; exact sbb_lin r3 b3 cf_2 b_r3 b_b3 b_cf_2
  have b_r3_1 : r3_1 < 2^64 := by rw [e_r3_1]; exact sbb_value_lt r3 b3 cf_2
  have b_cf_3 : cf_3 ≤ 1 := by rw [e_cf_3]; exact sbb_borrow_le_one r3 b3 cf_2
  clear e_r3_1 e_cf_3
  -- z: mov {z}, 0
  extract_lets -merge +onlyGivenNames z at hr
  have e_z : z = 0 := rfl
  clear_value z
  have b_z : z < 2^64 := by rw [e_z]; decide
  -- p0_1: cmovnc {p0}, {z}
  extract_lets -merge +onlyGivenNames p0_1 at hr
  have e_p0_1 : p0_1 = (if cf_3 = 0 then z else p0) := rfl
  clear_value p0_1
  have b_p0_1 : p0_1 < 2^64 := by
    rw [e_p0_1]; split <;> first | exact b_z | exact b_p0
  -- p1_1: cmovnc {p1}, {z}
  extract_lets -merge +onlyGivenNames p1_1 at hr
  have e_p1_1 : p1_1 = (if cf_3 = 0 then z else p1) := rfl
  clear_value p1_1
  have b_p1_1 : p1_1 < 2^64 := by
    rw [e_p1_1]; split <;> first | exact b_z | exact b_p1
  -- p3_1: cmovnc {p3}, {z}
  extract_lets -merge +onlyGivenNames p3_1 at hr
  have e_p3_1 : p3_1 = (if cf_3 = 0 then z else p3) := rfl
  clear_value p3_1
  have b_p3_1 : p3_1 < 2^64 := by
    rw [e_p3_1]; split <;> first | exact b_z | exact b_p3
  -- r0_2: add {r0}, {p0}
  extract_lets -merge +onlyGivenNames s r0_2 cf_4 at hr
  have e_r0_2 : r0_2 = (addc r0_1 p0_1 0).1 := rfl
  have e_cf_4 : cf_4 = (addc r0_1 p0_1 0).2 := rfl
  clear_value s r0_2 cf_4
  have l_r0_2 : r0_2 + 2^64 * cf_4 = r0_1 + p0_1 + 0 := by
    rw [e_r0_2, e_cf_4]; exact addc_lin r0_1 p0_1 0
  have b_r0_2 : r0_2 < 2^64 := by rw [e_r0_2]; exact addc_value_lt r0_1 p0_1 0
  have b_cf_4 : cf_4 ≤ 1 := by rw [e_cf_4]; exact addc_carry_le_one r0_1 p0_1 0 b_r0_1 b_p0_1 (by decide)
  clear e_r0_2 e_cf_4
  -- r1_2: adc {r1}, {p1}
  extract_lets -merge +onlyGivenNames s_1 r1_2 cf_5 at hr
  have e_r1_2 : r1_2 = (addc r1_1 p1_1 cf_4).1 := rfl
  have e_cf_5 : cf_5 = (addc r1_1 p1_1 cf_4).2 := rfl
  clear_value s_1 r1_2 cf_5
  have l_r1_2 : r1_2 + 2^64 * cf_5 = r1_1 + p1_1 + cf_4 := by
    rw [e_r1_2, e_cf_5]; exact addc_lin r1_1 p1_1 cf_4
  have b_r1_2 : r1_2 < 2^64 := by rw [e_r1_2]; exact addc_value_lt r1_1 p1_1 cf_4
  have b_cf_5 : cf_5 ≤ 1 := by rw [e_cf_5]; exact addc_carry_le_one r1_1 p1_1 cf_4 b_r1_1 b_p1_1 b_cf_4
  clear e_r1_2 e_cf_5
  -- r2_2: adc {r2}, 0
  extract_lets -merge +onlyGivenNames s_2 r2_2 cf_6 at hr
  have e_r2_2 : r2_2 = (addc r2_1 0 cf_5).1 := rfl
  have e_cf_6 : cf_6 = (addc r2_1 0 cf_5).2 := rfl
  clear_value s_2 r2_2 cf_6
  have l_r2_2 : r2_2 + 2^64 * cf_6 = r2_1 + 0 + cf_5 := by
    rw [e_r2_2, e_cf_6]; exact addc_lin r2_1 0 cf_5
  have b_r2_2 : r2_2 < 2^64 := by rw [e_r2_2]; exact addc_value_lt r2_1 0 cf_5
  have b_cf_6 : cf_6 ≤ 1 := by rw [e_cf_6]; exact addc_carry_le_one r2_1 0 cf_5 b_r2_1 (by decide) b_cf_5
  clear e_r2_2 e_cf_6
  -- r3_2: adc {r3}, {p3}
  extract_lets -merge +onlyGivenNames s_3 r3_2 cf_7 at hr
  have e_r3_2 : r3_2 = (addc r3_1 p3_1 cf_6).1 := rfl
  have e_cf_7 : cf_7 = (addc r3_1 p3_1 cf_6).2 := rfl
  clear_value s_3 r3_2 cf_7
  have l_r3_2 : r3_2 + 2^64 * cf_7 = r3_1 + p3_1 + cf_6 := by
    rw [e_r3_2, e_cf_7]; exact addc_lin r3_1 p3_1 cf_6
  have b_r3_2 : r3_2 < 2^64 := by rw [e_r3_2]; exact addc_value_lt r3_1 p3_1 cf_6
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

end PastaAsm.X86_64
