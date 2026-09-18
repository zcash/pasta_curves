/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.Spec
import PastaAsm.X86_64.Compositions
import PastaAsm.X86_64.Spec.Arithmetic
import PastaAsm.X86_64.Transcription

/-!
# Correctness of the x86-64 squaring blocks and their compositions

The `squareLo` proof follows every instruction in the generated unreduced-square skeleton. Its
annotated arithmetic splits each `mulx` into its low and high halves, telescopes the four carry
chains which accumulate the six cross products, follows the seven-word doubling chain, and
finally follows the carry chain which adds the four diagonal squares.
-/

namespace PastaAsm.X86_64

-- BEGIN squareLo arithmetic helpers
set_option exponentiation.threshold 512 in
private theorem cross_terms
    {a0 a1 a2 a3 z1 t1 z2 t2 z3 z4 z2a c0 z3a c1 z4a c2
      l12 h12 z3b c3 z4b c4 z5a c5 l13 h13 z4c c6 z5b c7 z6a c8
      l23 h23 z5c c9 z6b c10 z7a c11 : Nat}
    (d01 : z1 + 2^64 * t1 = a0 * a1)
    (d02 : z2 + 2^64 * t2 = a0 * a2)
    (d03 : z3 + 2^64 * z4 = a0 * a3)
    (l2a : z2a + 2^64 * c0 = z2 + t1)
    (l3a : z3a + 2^64 * c1 = z3 + t2 + c0)
    (l4a : z4a + 2^64 * c2 = z4 + c1) (hc2 : c2 = 0)
    (d12 : l12 + 2^64 * h12 = a1 * a2)
    (l3b : z3b + 2^64 * c3 = z3a + l12)
    (l4b : z4b + 2^64 * c4 = z4a + h12 + c3)
    (l5a : z5a + 2^64 * c5 = c4) (hc5 : c5 = 0)
    (d13 : l13 + 2^64 * h13 = a1 * a3)
    (l4c : z4c + 2^64 * c6 = z4b + l13)
    (l5b : z5b + 2^64 * c7 = z5a + h13 + c6)
    (l6a : z6a + 2^64 * c8 = c7) (hc8 : c8 = 0)
    (d23 : l23 + 2^64 * h23 = a2 * a3)
    (l5c : z5c + 2^64 * c9 = z5b + l23)
    (l6b : z6b + 2^64 * c10 = z6a + h23 + c9)
    (l7a : z7a + 2^64 * c11 = c10) (hc11 : c11 = 0) :
    2^64 * z1 + 2^128 * z2a + 2^192 * z3b + 2^256 * z4c +
        2^320 * z5c + 2^384 * z6b + 2^448 * z7a =
      2^64 * (a0 * a1) + 2^128 * (a0 * a2) +
        2^192 * (a0 * a3 + a1 * a2) + 2^256 * (a1 * a3) +
        2^320 * (a2 * a3) := by
  omega

set_option exponentiation.threshold 512 in
private theorem double_terms
    {z1 z2 z3 z4 z5 z6 z7 d1 c1 d2 c2 d3 c3 d4 c4 d5 c5 d6 c6 d7 c7 : Nat}
    (l1 : d1 + 2^64 * c1 = z1 + z1)
    (l2 : d2 + 2^64 * c2 = z2 + z2 + c1)
    (l3 : d3 + 2^64 * c3 = z3 + z3 + c2)
    (l4 : d4 + 2^64 * c4 = z4 + z4 + c3)
    (l5 : d5 + 2^64 * c5 = z5 + z5 + c4)
    (l6 : d6 + 2^64 * c6 = z6 + z6 + c5)
    (l7 : d7 + 2^64 * c7 = z7 + z7 + c6) :
    2^64 * d1 + 2^128 * d2 + 2^192 * d3 + 2^256 * d4 +
        2^320 * d5 + 2^384 * d6 + 2^448 * d7 + 2^512 * c7 =
      2 * (2^64 * z1 + 2^128 * z2 + 2^192 * z3 + 2^256 * z4 +
        2^320 * z5 + 2^384 * z6 + 2^448 * z7) := by
  omega

set_option exponentiation.threshold 512 in
private theorem diagonal_terms
    {a0 a1 a2 a3 z0 h00 l11 h11 l22 h22 l33 h33
      d1 d2 d3 d4 d5 d6 d7 o1 c1 o2 c2 o3 c3 o4 c4 o5 c5 o6 c6 o7 c7 : Nat}
    (d00 : z0 + 2^64 * h00 = a0 * a0)
    (d11 : l11 + 2^64 * h11 = a1 * a1)
    (d22 : l22 + 2^64 * h22 = a2 * a2)
    (d33 : l33 + 2^64 * h33 = a3 * a3)
    (l1 : o1 + 2^64 * c1 = d1 + h00)
    (l2 : o2 + 2^64 * c2 = d2 + l11 + c1)
    (l3 : o3 + 2^64 * c3 = d3 + h11 + c2)
    (l4 : o4 + 2^64 * c4 = d4 + l22 + c3)
    (l5 : o5 + 2^64 * c5 = d5 + h22 + c4)
    (l6 : o6 + 2^64 * c6 = d6 + l33 + c5)
    (l7 : o7 + 2^64 * c7 = d7 + h33 + c6) :
    z0 + 2^64 * o1 + 2^128 * o2 + 2^192 * o3 + 2^256 * o4 +
        2^320 * o5 + 2^384 * o6 + 2^448 * o7 + 2^512 * c7 =
      (2^64 * d1 + 2^128 * d2 + 2^192 * d3 + 2^256 * d4 +
        2^320 * d5 + 2^384 * d6 + 2^448 * d7) +
      (a0 * a0 + 2^128 * (a1 * a1) + 2^256 * (a2 * a2) + 2^384 * (a3 * a3)) := by
  omega

set_option exponentiation.threshold 512 in
private theorem top_carry_zero {body carry total : Nat}
    (h : body + 2^512 * carry = total) (hlt : total < 2^512) : carry = 0 := by
  omega

private theorem square_parts {output doubled diagonal cross square : Nat}
    (ho : output = doubled + diagonal) (hd : doubled = 2 * cross)
    (hs : square = 2 * cross + diagonal) : output = square := by
  omega

private theorem drop_zero_carry {body carry total limit : Nat}
    (h : body + limit * carry = total) (hc : carry = 0) : body = total := by
  simp only [hc, mul_zero, add_zero] at h
  exact h

private theorem add_diagonal {body carry cross diagonal square : Nat}
    (hd : body + 2^512 * carry = 2 * cross)
    (hs : square = 2 * cross + diagonal) : body + diagonal + 2^512 * carry = square := by
  omega
-- END squareLo arithmetic helpers

-- BEGIN squareLo_spec_traced statement
set_option exponentiation.threshold 512 in
private theorem squareLo_spec_traced (value : Limbs) (hv : value.Bounded) :
    ∀ r, r = squareLo value →
      r.Bounded ∧ r.toNat = value.toNat * value.toNat := by
  intro r hr
-- END squareLo_spec_traced statement
  -- generated skeleton for `squareLo`: do not edit between the annotations
  unfold squareLo at hr
  lift_lets -merge at hr
  -- a0: input a0
  extract_lets -merge +onlyGivenNames a0 at hr
  have e_a0 : a0 = value.l0 := rfl
  clear_value a0
  have b_a0 : a0 < 2^64 := by rw [e_a0]; exact hv.1
  -- a1: input a1
  extract_lets -merge +onlyGivenNames a1 at hr
  have e_a1 : a1 = value.l1 := rfl
  clear_value a1
  have b_a1 : a1 < 2^64 := by rw [e_a1]; exact hv.2.1
  -- a2: input a2
  extract_lets -merge +onlyGivenNames a2 at hr
  have e_a2 : a2 = value.l2 := rfl
  clear_value a2
  have b_a2 : a2 < 2^64 := by rw [e_a2]; exact hv.2.2.1
  -- a3: input a3
  extract_lets -merge +onlyGivenNames a3 at hr
  have e_a3 : a3 = value.l3 := rfl
  clear_value a3
  have b_a3 : a3 < 2^64 := by rw [e_a3]; exact hv.2.2.2
  -- z5: xor {z5:e}, {z5:e}
  extract_lets -merge +onlyGivenNames z5 cf ofl at hr
  have e_z5 : z5 = 0 := rfl
  have e_cf : cf = 0 := rfl
  have e_ofl : ofl = 0 := rfl
  clear_value z5 cf ofl
  have b_z5 : z5 < 2^64 := by rw [e_z5]; decide
  have b_cf : cf ≤ 1 := by rw [e_cf]; decide
  have b_ofl : ofl ≤ 1 := by rw [e_ofl]; decide
  -- z6: xor {z6:e}, {z6:e}
  extract_lets -merge +onlyGivenNames z6 cf_1 ofl_1 at hr
  have e_z6 : z6 = 0 := rfl
  have e_cf_1 : cf_1 = 0 := rfl
  have e_ofl_1 : ofl_1 = 0 := rfl
  clear_value z6 cf_1 ofl_1
  have b_z6 : z6 < 2^64 := by rw [e_z6]; decide
  have b_cf_1 : cf_1 ≤ 1 := by rw [e_cf_1]; decide
  have b_ofl_1 : ofl_1 ≤ 1 := by rw [e_ofl_1]; decide
  -- z7: xor {z7:e}, {z7:e}
  extract_lets -merge +onlyGivenNames z7 cf_2 ofl_2 at hr
  have e_z7 : z7 = 0 := rfl
  have e_cf_2 : cf_2 = 0 := rfl
  have e_ofl_2 : ofl_2 = 0 := rfl
  clear_value z7 cf_2 ofl_2
  have b_z7 : z7 < 2^64 := by rw [e_z7]; decide
  have b_cf_2 : cf_2 ≤ 1 := by rw [e_cf_2]; decide
  have b_ofl_2 : ofl_2 ≤ 1 := by rw [e_ofl_2]; decide
  -- rdx: mov rdx, {a0}
  extract_lets -merge +onlyGivenNames rdx at hr
  have e_rdx : rdx = a0 := rfl
  clear_value rdx
  have b_rdx : rdx < 2^64 := by rw [e_rdx]; exact b_a0
  -- m: mulx {t1}, {z1}, {a1}
  extract_lets -merge +onlyGivenNames m t1 z1 at hr
  have e_t1 : t1 = (mulx rdx a1).1 := rfl
  have e_z1 : z1 = (mulx rdx a1).2 := rfl
  clear_value m t1 z1
  have b_t1 : t1 < 2^64 := by rw [e_t1]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx b_a1)
  have b_z1 : z1 < 2^64 := by rw [e_z1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t1 : z1 + 2^64 * t1 = rdx * a1 := by
    rw [e_z1, e_t1]; exact Nat.mod_add_div _ _
  -- m_1: mulx {t2}, {z2}, {a2}
  extract_lets -merge +onlyGivenNames m_1 t2 z2 at hr
  have e_t2 : t2 = (mulx rdx a2).1 := rfl
  have e_z2 : z2 = (mulx rdx a2).2 := rfl
  clear_value m_1 t2 z2
  have b_t2 : t2 < 2^64 := by rw [e_t2]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx b_a2)
  have b_z2 : z2 < 2^64 := by rw [e_z2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t2 : z2 + 2^64 * t2 = rdx * a2 := by
    rw [e_z2, e_t2]; exact Nat.mod_add_div _ _
  -- m_2: mulx {z4}, {z3}, {a3}
  extract_lets -merge +onlyGivenNames m_2 z4 z3 at hr
  have e_z4 : z4 = (mulx rdx a3).1 := rfl
  have e_z3 : z3 = (mulx rdx a3).2 := rfl
  clear_value m_2 z4 z3
  have b_z4 : z4 < 2^64 := by rw [e_z4]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx b_a3)
  have b_z3 : z3 < 2^64 := by rw [e_z3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_z4 : z3 + 2^64 * z4 = rdx * a3 := by
    rw [e_z3, e_z4]; exact Nat.mod_add_div _ _
  -- z2_1: add {z2}, {t1}
  extract_lets -merge +onlyGivenNames s z2_1 cf_3 at hr
  have e_z2_1 : z2_1 = (addc z2 t1 0).1 := rfl
  have e_cf_3 : cf_3 = (addc z2 t1 0).2 := rfl
  clear_value s z2_1 cf_3
  have l_z2_1 : z2_1 + 2^64 * cf_3 = z2 + t1 + 0 := by
    rw [e_z2_1, e_cf_3]; exact addc_lin z2 t1 0
  have b_z2_1 : z2_1 < 2^64 := by rw [e_z2_1]; exact addc_value_lt z2 t1 0
  have b_cf_3 : cf_3 ≤ 1 := by rw [e_cf_3]; exact addc_carry_le_one z2 t1 0 b_z2 b_t1 (by decide)
  clear e_z2_1 e_cf_3
  -- z3_1: adc {z3}, {t2}
  extract_lets -merge +onlyGivenNames s_1 z3_1 cf_4 at hr
  have e_z3_1 : z3_1 = (addc z3 t2 cf_3).1 := rfl
  have e_cf_4 : cf_4 = (addc z3 t2 cf_3).2 := rfl
  clear_value s_1 z3_1 cf_4
  have l_z3_1 : z3_1 + 2^64 * cf_4 = z3 + t2 + cf_3 := by
    rw [e_z3_1, e_cf_4]; exact addc_lin z3 t2 cf_3
  have b_z3_1 : z3_1 < 2^64 := by rw [e_z3_1]; exact addc_value_lt z3 t2 cf_3
  have b_cf_4 : cf_4 ≤ 1 := by rw [e_cf_4]; exact addc_carry_le_one z3 t2 cf_3 b_z3 b_t2 b_cf_3
  clear e_z3_1 e_cf_4
  -- z4_1: adc {z4}, 0
  extract_lets -merge +onlyGivenNames s_2 z4_1 cf_5 at hr
  have e_z4_1 : z4_1 = (addc z4 0 cf_4).1 := rfl
  have e_cf_5 : cf_5 = (addc z4 0 cf_4).2 := rfl
  clear_value s_2 z4_1 cf_5
  have l_z4_1 : z4_1 + 2^64 * cf_5 = z4 + 0 + cf_4 := by
    rw [e_z4_1, e_cf_5]; exact addc_lin z4 0 cf_4
  have b_z4_1 : z4_1 < 2^64 := by rw [e_z4_1]; exact addc_value_lt z4 0 cf_4
  have b_cf_5 : cf_5 ≤ 1 := by rw [e_cf_5]; exact addc_carry_le_one z4 0 cf_4 b_z4 (by decide) b_cf_4
  clear e_z4_1 e_cf_5
  -- rdx_1: mov rdx, {a1}
  extract_lets -merge +onlyGivenNames rdx_1 at hr
  have e_rdx_1 : rdx_1 = a1 := rfl
  clear_value rdx_1
  have b_rdx_1 : rdx_1 < 2^64 := by rw [e_rdx_1]; exact b_a1
  -- m_3: mulx {t2}, {t1}, {a2}
  extract_lets -merge +onlyGivenNames m_3 t2_1 t1_1 at hr
  have e_t2_1 : t2_1 = (mulx rdx_1 a2).1 := rfl
  have e_t1_1 : t1_1 = (mulx rdx_1 a2).2 := rfl
  clear_value m_3 t2_1 t1_1
  have b_t2_1 : t2_1 < 2^64 := by rw [e_t2_1]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_1 b_a2)
  have b_t1_1 : t1_1 < 2^64 := by rw [e_t1_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t2_1 : t1_1 + 2^64 * t2_1 = rdx_1 * a2 := by
    rw [e_t1_1, e_t2_1]; exact Nat.mod_add_div _ _
  -- z3_2: add {z3}, {t1}
  extract_lets -merge +onlyGivenNames s_3 z3_2 cf_6 at hr
  have e_z3_2 : z3_2 = (addc z3_1 t1_1 0).1 := rfl
  have e_cf_6 : cf_6 = (addc z3_1 t1_1 0).2 := rfl
  clear_value s_3 z3_2 cf_6
  have l_z3_2 : z3_2 + 2^64 * cf_6 = z3_1 + t1_1 + 0 := by
    rw [e_z3_2, e_cf_6]; exact addc_lin z3_1 t1_1 0
  have b_z3_2 : z3_2 < 2^64 := by rw [e_z3_2]; exact addc_value_lt z3_1 t1_1 0
  have b_cf_6 : cf_6 ≤ 1 := by rw [e_cf_6]; exact addc_carry_le_one z3_1 t1_1 0 b_z3_1 b_t1_1 (by decide)
  clear e_z3_2 e_cf_6
  -- z4_2: adc {z4}, {t2}
  extract_lets -merge +onlyGivenNames s_4 z4_2 cf_7 at hr
  have e_z4_2 : z4_2 = (addc z4_1 t2_1 cf_6).1 := rfl
  have e_cf_7 : cf_7 = (addc z4_1 t2_1 cf_6).2 := rfl
  clear_value s_4 z4_2 cf_7
  have l_z4_2 : z4_2 + 2^64 * cf_7 = z4_1 + t2_1 + cf_6 := by
    rw [e_z4_2, e_cf_7]; exact addc_lin z4_1 t2_1 cf_6
  have b_z4_2 : z4_2 < 2^64 := by rw [e_z4_2]; exact addc_value_lt z4_1 t2_1 cf_6
  have b_cf_7 : cf_7 ≤ 1 := by rw [e_cf_7]; exact addc_carry_le_one z4_1 t2_1 cf_6 b_z4_1 b_t2_1 b_cf_6
  clear e_z4_2 e_cf_7
  -- z5_1: adc {z5}, 0
  extract_lets -merge +onlyGivenNames s_5 z5_1 cf_8 at hr
  have e_z5_1 : z5_1 = (addc z5 0 cf_7).1 := rfl
  have e_cf_8 : cf_8 = (addc z5 0 cf_7).2 := rfl
  clear_value s_5 z5_1 cf_8
  have l_z5_1 : z5_1 + 2^64 * cf_8 = z5 + 0 + cf_7 := by
    rw [e_z5_1, e_cf_8]; exact addc_lin z5 0 cf_7
  have b_z5_1 : z5_1 < 2^64 := by rw [e_z5_1]; exact addc_value_lt z5 0 cf_7
  have b_cf_8 : cf_8 ≤ 1 := by rw [e_cf_8]; exact addc_carry_le_one z5 0 cf_7 b_z5 (by decide) b_cf_7
  clear e_z5_1 e_cf_8
  -- m_4: mulx {t2}, {t1}, {a3}
  extract_lets -merge +onlyGivenNames m_4 t2_2 t1_2 at hr
  have e_t2_2 : t2_2 = (mulx rdx_1 a3).1 := rfl
  have e_t1_2 : t1_2 = (mulx rdx_1 a3).2 := rfl
  clear_value m_4 t2_2 t1_2
  have b_t2_2 : t2_2 < 2^64 := by rw [e_t2_2]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_1 b_a3)
  have b_t1_2 : t1_2 < 2^64 := by rw [e_t1_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t2_2 : t1_2 + 2^64 * t2_2 = rdx_1 * a3 := by
    rw [e_t1_2, e_t2_2]; exact Nat.mod_add_div _ _
  -- z4_3: add {z4}, {t1}
  extract_lets -merge +onlyGivenNames s_6 z4_3 cf_9 at hr
  have e_z4_3 : z4_3 = (addc z4_2 t1_2 0).1 := rfl
  have e_cf_9 : cf_9 = (addc z4_2 t1_2 0).2 := rfl
  clear_value s_6 z4_3 cf_9
  have l_z4_3 : z4_3 + 2^64 * cf_9 = z4_2 + t1_2 + 0 := by
    rw [e_z4_3, e_cf_9]; exact addc_lin z4_2 t1_2 0
  have b_z4_3 : z4_3 < 2^64 := by rw [e_z4_3]; exact addc_value_lt z4_2 t1_2 0
  have b_cf_9 : cf_9 ≤ 1 := by rw [e_cf_9]; exact addc_carry_le_one z4_2 t1_2 0 b_z4_2 b_t1_2 (by decide)
  clear e_z4_3 e_cf_9
  -- z5_2: adc {z5}, {t2}
  extract_lets -merge +onlyGivenNames s_7 z5_2 cf_10 at hr
  have e_z5_2 : z5_2 = (addc z5_1 t2_2 cf_9).1 := rfl
  have e_cf_10 : cf_10 = (addc z5_1 t2_2 cf_9).2 := rfl
  clear_value s_7 z5_2 cf_10
  have l_z5_2 : z5_2 + 2^64 * cf_10 = z5_1 + t2_2 + cf_9 := by
    rw [e_z5_2, e_cf_10]; exact addc_lin z5_1 t2_2 cf_9
  have b_z5_2 : z5_2 < 2^64 := by rw [e_z5_2]; exact addc_value_lt z5_1 t2_2 cf_9
  have b_cf_10 : cf_10 ≤ 1 := by rw [e_cf_10]; exact addc_carry_le_one z5_1 t2_2 cf_9 b_z5_1 b_t2_2 b_cf_9
  clear e_z5_2 e_cf_10
  -- z6_1: adc {z6}, 0
  extract_lets -merge +onlyGivenNames s_8 z6_1 cf_11 at hr
  have e_z6_1 : z6_1 = (addc z6 0 cf_10).1 := rfl
  have e_cf_11 : cf_11 = (addc z6 0 cf_10).2 := rfl
  clear_value s_8 z6_1 cf_11
  have l_z6_1 : z6_1 + 2^64 * cf_11 = z6 + 0 + cf_10 := by
    rw [e_z6_1, e_cf_11]; exact addc_lin z6 0 cf_10
  have b_z6_1 : z6_1 < 2^64 := by rw [e_z6_1]; exact addc_value_lt z6 0 cf_10
  have b_cf_11 : cf_11 ≤ 1 := by rw [e_cf_11]; exact addc_carry_le_one z6 0 cf_10 b_z6 (by decide) b_cf_10
  clear e_z6_1 e_cf_11
  -- rdx_2: mov rdx, {a2}
  extract_lets -merge +onlyGivenNames rdx_2 at hr
  have e_rdx_2 : rdx_2 = a2 := rfl
  clear_value rdx_2
  have b_rdx_2 : rdx_2 < 2^64 := by rw [e_rdx_2]; exact b_a2
  -- m_5: mulx {t2}, {t1}, {a3}
  extract_lets -merge +onlyGivenNames m_5 t2_3 t1_3 at hr
  have e_t2_3 : t2_3 = (mulx rdx_2 a3).1 := rfl
  have e_t1_3 : t1_3 = (mulx rdx_2 a3).2 := rfl
  clear_value m_5 t2_3 t1_3
  have b_t2_3 : t2_3 < 2^64 := by rw [e_t2_3]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_2 b_a3)
  have b_t1_3 : t1_3 < 2^64 := by rw [e_t1_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t2_3 : t1_3 + 2^64 * t2_3 = rdx_2 * a3 := by
    rw [e_t1_3, e_t2_3]; exact Nat.mod_add_div _ _
  -- z5_3: add {z5}, {t1}
  extract_lets -merge +onlyGivenNames s_9 z5_3 cf_12 at hr
  have e_z5_3 : z5_3 = (addc z5_2 t1_3 0).1 := rfl
  have e_cf_12 : cf_12 = (addc z5_2 t1_3 0).2 := rfl
  clear_value s_9 z5_3 cf_12
  have l_z5_3 : z5_3 + 2^64 * cf_12 = z5_2 + t1_3 + 0 := by
    rw [e_z5_3, e_cf_12]; exact addc_lin z5_2 t1_3 0
  have b_z5_3 : z5_3 < 2^64 := by rw [e_z5_3]; exact addc_value_lt z5_2 t1_3 0
  have b_cf_12 : cf_12 ≤ 1 := by rw [e_cf_12]; exact addc_carry_le_one z5_2 t1_3 0 b_z5_2 b_t1_3 (by decide)
  clear e_z5_3 e_cf_12
  -- z6_2: adc {z6}, {t2}
  extract_lets -merge +onlyGivenNames s_10 z6_2 cf_13 at hr
  have e_z6_2 : z6_2 = (addc z6_1 t2_3 cf_12).1 := rfl
  have e_cf_13 : cf_13 = (addc z6_1 t2_3 cf_12).2 := rfl
  clear_value s_10 z6_2 cf_13
  have l_z6_2 : z6_2 + 2^64 * cf_13 = z6_1 + t2_3 + cf_12 := by
    rw [e_z6_2, e_cf_13]; exact addc_lin z6_1 t2_3 cf_12
  have b_z6_2 : z6_2 < 2^64 := by rw [e_z6_2]; exact addc_value_lt z6_1 t2_3 cf_12
  have b_cf_13 : cf_13 ≤ 1 := by rw [e_cf_13]; exact addc_carry_le_one z6_1 t2_3 cf_12 b_z6_1 b_t2_3 b_cf_12
  clear e_z6_2 e_cf_13
  -- z7_1: adc {z7}, 0
  extract_lets -merge +onlyGivenNames s_11 z7_1 cf_14 at hr
  have e_z7_1 : z7_1 = (addc z7 0 cf_13).1 := rfl
  have e_cf_14 : cf_14 = (addc z7 0 cf_13).2 := rfl
  clear_value s_11 z7_1 cf_14
  have l_z7_1 : z7_1 + 2^64 * cf_14 = z7 + 0 + cf_13 := by
    rw [e_z7_1, e_cf_14]; exact addc_lin z7 0 cf_13
  have b_z7_1 : z7_1 < 2^64 := by rw [e_z7_1]; exact addc_value_lt z7 0 cf_13
  have b_cf_14 : cf_14 ≤ 1 := by rw [e_cf_14]; exact addc_carry_le_one z7 0 cf_13 b_z7 (by decide) b_cf_13
  clear e_z7_1 e_cf_14
  -- BEGIN squareLo cross arithmetic
  have t_z4 : z4 ≤ 2^64 - 2 := by
    have h := Nat.mul_le_mul (Nat.le_sub_one_of_lt b_rdx) (Nat.le_sub_one_of_lt b_a3)
    norm_num at h
    clear * - h d_z4
    omega
  have z_cf_5 : cf_5 = 0 := by
    clear * - l_z4_1 t_z4 b_cf_4
    omega
  have z_cf_8 : cf_8 = 0 := by
    clear * - l_z5_1 e_z5 b_cf_7
    omega
  have z_cf_11 : cf_11 = 0 := by
    clear * - l_z6_1 e_z6 b_cf_10
    omega
  have z_cf_14 : cf_14 = 0 := by
    clear * - l_z7_1 e_z7 b_cf_13
    omega
  have hcross :
      2^64 * z1 + 2^128 * z2_1 + 2^192 * z3_2 + 2^256 * z4_3 +
          2^320 * z5_3 + 2^384 * z6_2 + 2^448 * z7_1 =
        2^64 * (a0 * a1) + 2^128 * (a0 * a2) +
          2^192 * (a0 * a3 + a1 * a2) + 2^256 * (a1 * a3) +
          2^320 * (a2 * a3) :=
    cross_terms (by simpa only [e_rdx] using d_t1) (by simpa only [e_rdx] using d_t2)
      (by simpa only [e_rdx] using d_z4) (by simpa only [add_zero] using l_z2_1)
      l_z3_1 (by simpa only [zero_add] using l_z4_1) z_cf_5
      (by simpa only [e_rdx_1] using d_t2_1) (by simpa only [add_zero] using l_z3_2)
      l_z4_2 (by simpa only [e_z5, zero_add] using l_z5_1) z_cf_8
      (by simpa only [e_rdx_1] using d_t2_2) (by simpa only [add_zero] using l_z4_3)
      l_z5_2 (by simpa only [e_z6, zero_add] using l_z6_1) z_cf_11
      (by simpa only [e_rdx_2] using d_t2_3) (by simpa only [add_zero] using l_z5_3)
      l_z6_2 (by simpa only [e_z7, zero_add] using l_z7_1) z_cf_14
  -- END squareLo cross arithmetic
  -- z1_1: add {z1}, {z1}
  extract_lets -merge +onlyGivenNames s_12 z1_1 cf_15 at hr
  have e_z1_1 : z1_1 = (addc z1 z1 0).1 := rfl
  have e_cf_15 : cf_15 = (addc z1 z1 0).2 := rfl
  clear_value s_12 z1_1 cf_15
  have l_z1_1 : z1_1 + 2^64 * cf_15 = z1 + z1 + 0 := by
    rw [e_z1_1, e_cf_15]; exact addc_lin z1 z1 0
  have b_z1_1 : z1_1 < 2^64 := by rw [e_z1_1]; exact addc_value_lt z1 z1 0
  have b_cf_15 : cf_15 ≤ 1 := by rw [e_cf_15]; exact addc_carry_le_one z1 z1 0 b_z1 b_z1 (by decide)
  clear e_z1_1 e_cf_15
  -- z2_2: adc {z2}, {z2}
  extract_lets -merge +onlyGivenNames s_13 z2_2 cf_16 at hr
  have e_z2_2 : z2_2 = (addc z2_1 z2_1 cf_15).1 := rfl
  have e_cf_16 : cf_16 = (addc z2_1 z2_1 cf_15).2 := rfl
  clear_value s_13 z2_2 cf_16
  have l_z2_2 : z2_2 + 2^64 * cf_16 = z2_1 + z2_1 + cf_15 := by
    rw [e_z2_2, e_cf_16]; exact addc_lin z2_1 z2_1 cf_15
  have b_z2_2 : z2_2 < 2^64 := by rw [e_z2_2]; exact addc_value_lt z2_1 z2_1 cf_15
  have b_cf_16 : cf_16 ≤ 1 := by rw [e_cf_16]; exact addc_carry_le_one z2_1 z2_1 cf_15 b_z2_1 b_z2_1 b_cf_15
  clear e_z2_2 e_cf_16
  -- z3_3: adc {z3}, {z3}
  extract_lets -merge +onlyGivenNames s_14 z3_3 cf_17 at hr
  have e_z3_3 : z3_3 = (addc z3_2 z3_2 cf_16).1 := rfl
  have e_cf_17 : cf_17 = (addc z3_2 z3_2 cf_16).2 := rfl
  clear_value s_14 z3_3 cf_17
  have l_z3_3 : z3_3 + 2^64 * cf_17 = z3_2 + z3_2 + cf_16 := by
    rw [e_z3_3, e_cf_17]; exact addc_lin z3_2 z3_2 cf_16
  have b_z3_3 : z3_3 < 2^64 := by rw [e_z3_3]; exact addc_value_lt z3_2 z3_2 cf_16
  have b_cf_17 : cf_17 ≤ 1 := by rw [e_cf_17]; exact addc_carry_le_one z3_2 z3_2 cf_16 b_z3_2 b_z3_2 b_cf_16
  clear e_z3_3 e_cf_17
  -- z4_4: adc {z4}, {z4}
  extract_lets -merge +onlyGivenNames s_15 z4_4 cf_18 at hr
  have e_z4_4 : z4_4 = (addc z4_3 z4_3 cf_17).1 := rfl
  have e_cf_18 : cf_18 = (addc z4_3 z4_3 cf_17).2 := rfl
  clear_value s_15 z4_4 cf_18
  have l_z4_4 : z4_4 + 2^64 * cf_18 = z4_3 + z4_3 + cf_17 := by
    rw [e_z4_4, e_cf_18]; exact addc_lin z4_3 z4_3 cf_17
  have b_z4_4 : z4_4 < 2^64 := by rw [e_z4_4]; exact addc_value_lt z4_3 z4_3 cf_17
  have b_cf_18 : cf_18 ≤ 1 := by rw [e_cf_18]; exact addc_carry_le_one z4_3 z4_3 cf_17 b_z4_3 b_z4_3 b_cf_17
  clear e_z4_4 e_cf_18
  -- z5_4: adc {z5}, {z5}
  extract_lets -merge +onlyGivenNames s_16 z5_4 cf_19 at hr
  have e_z5_4 : z5_4 = (addc z5_3 z5_3 cf_18).1 := rfl
  have e_cf_19 : cf_19 = (addc z5_3 z5_3 cf_18).2 := rfl
  clear_value s_16 z5_4 cf_19
  have l_z5_4 : z5_4 + 2^64 * cf_19 = z5_3 + z5_3 + cf_18 := by
    rw [e_z5_4, e_cf_19]; exact addc_lin z5_3 z5_3 cf_18
  have b_z5_4 : z5_4 < 2^64 := by rw [e_z5_4]; exact addc_value_lt z5_3 z5_3 cf_18
  have b_cf_19 : cf_19 ≤ 1 := by rw [e_cf_19]; exact addc_carry_le_one z5_3 z5_3 cf_18 b_z5_3 b_z5_3 b_cf_18
  clear e_z5_4 e_cf_19
  -- z6_3: adc {z6}, {z6}
  extract_lets -merge +onlyGivenNames s_17 z6_3 cf_20 at hr
  have e_z6_3 : z6_3 = (addc z6_2 z6_2 cf_19).1 := rfl
  have e_cf_20 : cf_20 = (addc z6_2 z6_2 cf_19).2 := rfl
  clear_value s_17 z6_3 cf_20
  have l_z6_3 : z6_3 + 2^64 * cf_20 = z6_2 + z6_2 + cf_19 := by
    rw [e_z6_3, e_cf_20]; exact addc_lin z6_2 z6_2 cf_19
  have b_z6_3 : z6_3 < 2^64 := by rw [e_z6_3]; exact addc_value_lt z6_2 z6_2 cf_19
  have b_cf_20 : cf_20 ≤ 1 := by rw [e_cf_20]; exact addc_carry_le_one z6_2 z6_2 cf_19 b_z6_2 b_z6_2 b_cf_19
  clear e_z6_3 e_cf_20
  -- z7_2: adc {z7}, {z7}
  extract_lets -merge +onlyGivenNames s_18 z7_2 cf_21 at hr
  have e_z7_2 : z7_2 = (addc z7_1 z7_1 cf_20).1 := rfl
  have e_cf_21 : cf_21 = (addc z7_1 z7_1 cf_20).2 := rfl
  clear_value s_18 z7_2 cf_21
  have l_z7_2 : z7_2 + 2^64 * cf_21 = z7_1 + z7_1 + cf_20 := by
    rw [e_z7_2, e_cf_21]; exact addc_lin z7_1 z7_1 cf_20
  have b_z7_2 : z7_2 < 2^64 := by rw [e_z7_2]; exact addc_value_lt z7_1 z7_1 cf_20
  have b_cf_21 : cf_21 ≤ 1 := by rw [e_cf_21]; exact addc_carry_le_one z7_1 z7_1 cf_20 b_z7_1 b_z7_1 b_cf_20
  clear e_z7_2 e_cf_21
  -- BEGIN squareLo doubling arithmetic
  have hdouble :
      2^64 * z1_1 + 2^128 * z2_2 + 2^192 * z3_3 + 2^256 * z4_4 +
          2^320 * z5_4 + 2^384 * z6_3 + 2^448 * z7_2 + 2^512 * cf_21 =
        2 * (2^64 * z1 + 2^128 * z2_1 + 2^192 * z3_2 + 2^256 * z4_3 +
          2^320 * z5_3 + 2^384 * z6_2 + 2^448 * z7_1) :=
    double_terms (by simpa only [add_zero] using l_z1_1) l_z2_2 l_z3_3 l_z4_4
      l_z5_4 l_z6_3 l_z7_2
  have hvalue : value.toNat < 2^256 := Limbs.toNat_lt value hv
  have hsquare_lt : value.toNat * value.toNat < 2^512 := by
    have h := Nat.mul_lt_mul'' hvalue hvalue
    norm_num at h ⊢
    exact h
  have hsquare_parts : value.toNat * value.toNat =
      2 * (2^64 * (a0 * a1) + 2^128 * (a0 * a2) +
        2^192 * (a0 * a3 + a1 * a2) + 2^256 * (a1 * a3) + 2^320 * (a2 * a3)) +
      (a0 * a0 + 2^128 * (a1 * a1) + 2^256 * (a2 * a2) + 2^384 * (a3 * a3)) := by
    simp only [Limbs.toNat]
    rw [← e_a0, ← e_a1, ← e_a2, ← e_a3]
    ring
  have hdouble_math :
      (2^64 * z1_1 + 2^128 * z2_2 + 2^192 * z3_3 + 2^256 * z4_4 +
          2^320 * z5_4 + 2^384 * z6_3 + 2^448 * z7_2) + 2^512 * cf_21 =
        2 * (2^64 * (a0 * a1) + 2^128 * (a0 * a2) +
          2^192 * (a0 * a3 + a1 * a2) + 2^256 * (a1 * a3) + 2^320 * (a2 * a3)) := by
    rw [hdouble, hcross]
  have hpre :
      (2^64 * z1_1 + 2^128 * z2_2 + 2^192 * z3_3 + 2^256 * z4_4 +
          2^320 * z5_4 + 2^384 * z6_3 + 2^448 * z7_2) +
          (a0 * a0 + 2^128 * (a1 * a1) + 2^256 * (a2 * a2) + 2^384 * (a3 * a3)) +
          2^512 * cf_21 = value.toNat * value.toNat :=
    add_diagonal hdouble_math hsquare_parts
  have z_cf_21 : cf_21 = 0 :=
    top_carry_zero hpre hsquare_lt
  -- END squareLo doubling arithmetic
  -- rdx_3: mov rdx, {a0}
  extract_lets -merge +onlyGivenNames rdx_3 at hr
  have e_rdx_3 : rdx_3 = a0 := rfl
  clear_value rdx_3
  have b_rdx_3 : rdx_3 < 2^64 := by rw [e_rdx_3]; exact b_a0
  -- m_6: mulx {t2}, {z0}, rdx
  extract_lets -merge +onlyGivenNames m_6 t2_4 z0 at hr
  have e_t2_4 : t2_4 = (mulx rdx_3 rdx_3).1 := rfl
  have e_z0 : z0 = (mulx rdx_3 rdx_3).2 := rfl
  clear_value m_6 t2_4 z0
  have b_t2_4 : t2_4 < 2^64 := by rw [e_t2_4]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_3 b_rdx_3)
  have b_z0 : z0 < 2^64 := by rw [e_z0]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t2_4 : z0 + 2^64 * t2_4 = rdx_3 * rdx_3 := by
    rw [e_z0, e_t2_4]; exact Nat.mod_add_div _ _
  -- z1_2: add {z1}, {t2}
  extract_lets -merge +onlyGivenNames s_19 z1_2 cf_22 at hr
  have e_z1_2 : z1_2 = (addc z1_1 t2_4 0).1 := rfl
  have e_cf_22 : cf_22 = (addc z1_1 t2_4 0).2 := rfl
  clear_value s_19 z1_2 cf_22
  have l_z1_2 : z1_2 + 2^64 * cf_22 = z1_1 + t2_4 + 0 := by
    rw [e_z1_2, e_cf_22]; exact addc_lin z1_1 t2_4 0
  have b_z1_2 : z1_2 < 2^64 := by rw [e_z1_2]; exact addc_value_lt z1_1 t2_4 0
  have b_cf_22 : cf_22 ≤ 1 := by rw [e_cf_22]; exact addc_carry_le_one z1_1 t2_4 0 b_z1_1 b_t2_4 (by decide)
  clear e_z1_2 e_cf_22
  -- rdx_4: mov rdx, {a1}
  extract_lets -merge +onlyGivenNames rdx_4 at hr
  have e_rdx_4 : rdx_4 = a1 := rfl
  clear_value rdx_4
  have b_rdx_4 : rdx_4 < 2^64 := by rw [e_rdx_4]; exact b_a1
  -- m_7: mulx {t2}, {t1}, rdx
  extract_lets -merge +onlyGivenNames m_7 t2_5 t1_4 at hr
  have e_t2_5 : t2_5 = (mulx rdx_4 rdx_4).1 := rfl
  have e_t1_4 : t1_4 = (mulx rdx_4 rdx_4).2 := rfl
  clear_value m_7 t2_5 t1_4
  have b_t2_5 : t2_5 < 2^64 := by rw [e_t2_5]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_4 b_rdx_4)
  have b_t1_4 : t1_4 < 2^64 := by rw [e_t1_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t2_5 : t1_4 + 2^64 * t2_5 = rdx_4 * rdx_4 := by
    rw [e_t1_4, e_t2_5]; exact Nat.mod_add_div _ _
  -- z2_3: adc {z2}, {t1}
  extract_lets -merge +onlyGivenNames s_20 z2_3 cf_23 at hr
  have e_z2_3 : z2_3 = (addc z2_2 t1_4 cf_22).1 := rfl
  have e_cf_23 : cf_23 = (addc z2_2 t1_4 cf_22).2 := rfl
  clear_value s_20 z2_3 cf_23
  have l_z2_3 : z2_3 + 2^64 * cf_23 = z2_2 + t1_4 + cf_22 := by
    rw [e_z2_3, e_cf_23]; exact addc_lin z2_2 t1_4 cf_22
  have b_z2_3 : z2_3 < 2^64 := by rw [e_z2_3]; exact addc_value_lt z2_2 t1_4 cf_22
  have b_cf_23 : cf_23 ≤ 1 := by rw [e_cf_23]; exact addc_carry_le_one z2_2 t1_4 cf_22 b_z2_2 b_t1_4 b_cf_22
  clear e_z2_3 e_cf_23
  -- z3_4: adc {z3}, {t2}
  extract_lets -merge +onlyGivenNames s_21 z3_4 cf_24 at hr
  have e_z3_4 : z3_4 = (addc z3_3 t2_5 cf_23).1 := rfl
  have e_cf_24 : cf_24 = (addc z3_3 t2_5 cf_23).2 := rfl
  clear_value s_21 z3_4 cf_24
  have l_z3_4 : z3_4 + 2^64 * cf_24 = z3_3 + t2_5 + cf_23 := by
    rw [e_z3_4, e_cf_24]; exact addc_lin z3_3 t2_5 cf_23
  have b_z3_4 : z3_4 < 2^64 := by rw [e_z3_4]; exact addc_value_lt z3_3 t2_5 cf_23
  have b_cf_24 : cf_24 ≤ 1 := by rw [e_cf_24]; exact addc_carry_le_one z3_3 t2_5 cf_23 b_z3_3 b_t2_5 b_cf_23
  clear e_z3_4 e_cf_24
  -- rdx_5: mov rdx, {a2}
  extract_lets -merge +onlyGivenNames rdx_5 at hr
  have e_rdx_5 : rdx_5 = a2 := rfl
  clear_value rdx_5
  have b_rdx_5 : rdx_5 < 2^64 := by rw [e_rdx_5]; exact b_a2
  -- m_8: mulx {t2}, {t1}, rdx
  extract_lets -merge +onlyGivenNames m_8 t2_6 t1_5 at hr
  have e_t2_6 : t2_6 = (mulx rdx_5 rdx_5).1 := rfl
  have e_t1_5 : t1_5 = (mulx rdx_5 rdx_5).2 := rfl
  clear_value m_8 t2_6 t1_5
  have b_t2_6 : t2_6 < 2^64 := by rw [e_t2_6]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_5 b_rdx_5)
  have b_t1_5 : t1_5 < 2^64 := by rw [e_t1_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t2_6 : t1_5 + 2^64 * t2_6 = rdx_5 * rdx_5 := by
    rw [e_t1_5, e_t2_6]; exact Nat.mod_add_div _ _
  -- z4_5: adc {z4}, {t1}
  extract_lets -merge +onlyGivenNames s_22 z4_5 cf_25 at hr
  have e_z4_5 : z4_5 = (addc z4_4 t1_5 cf_24).1 := rfl
  have e_cf_25 : cf_25 = (addc z4_4 t1_5 cf_24).2 := rfl
  clear_value s_22 z4_5 cf_25
  have l_z4_5 : z4_5 + 2^64 * cf_25 = z4_4 + t1_5 + cf_24 := by
    rw [e_z4_5, e_cf_25]; exact addc_lin z4_4 t1_5 cf_24
  have b_z4_5 : z4_5 < 2^64 := by rw [e_z4_5]; exact addc_value_lt z4_4 t1_5 cf_24
  have b_cf_25 : cf_25 ≤ 1 := by rw [e_cf_25]; exact addc_carry_le_one z4_4 t1_5 cf_24 b_z4_4 b_t1_5 b_cf_24
  clear e_z4_5 e_cf_25
  -- z5_5: adc {z5}, {t2}
  extract_lets -merge +onlyGivenNames s_23 z5_5 cf_26 at hr
  have e_z5_5 : z5_5 = (addc z5_4 t2_6 cf_25).1 := rfl
  have e_cf_26 : cf_26 = (addc z5_4 t2_6 cf_25).2 := rfl
  clear_value s_23 z5_5 cf_26
  have l_z5_5 : z5_5 + 2^64 * cf_26 = z5_4 + t2_6 + cf_25 := by
    rw [e_z5_5, e_cf_26]; exact addc_lin z5_4 t2_6 cf_25
  have b_z5_5 : z5_5 < 2^64 := by rw [e_z5_5]; exact addc_value_lt z5_4 t2_6 cf_25
  have b_cf_26 : cf_26 ≤ 1 := by rw [e_cf_26]; exact addc_carry_le_one z5_4 t2_6 cf_25 b_z5_4 b_t2_6 b_cf_25
  clear e_z5_5 e_cf_26
  -- rdx_6: mov rdx, {a3}
  extract_lets -merge +onlyGivenNames rdx_6 at hr
  have e_rdx_6 : rdx_6 = a3 := rfl
  clear_value rdx_6
  have b_rdx_6 : rdx_6 < 2^64 := by rw [e_rdx_6]; exact b_a3
  -- m_9: mulx {t2}, {t1}, rdx
  extract_lets -merge +onlyGivenNames m_9 t2_7 t1_6 at hr
  have e_t2_7 : t2_7 = (mulx rdx_6 rdx_6).1 := rfl
  have e_t1_6 : t1_6 = (mulx rdx_6 rdx_6).2 := rfl
  clear_value m_9 t2_7 t1_6
  have b_t2_7 : t2_7 < 2^64 := by rw [e_t2_7]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_6 b_rdx_6)
  have b_t1_6 : t1_6 < 2^64 := by rw [e_t1_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t2_7 : t1_6 + 2^64 * t2_7 = rdx_6 * rdx_6 := by
    rw [e_t1_6, e_t2_7]; exact Nat.mod_add_div _ _
  -- z6_4: adc {z6}, {t1}
  extract_lets -merge +onlyGivenNames s_24 z6_4 cf_27 at hr
  have e_z6_4 : z6_4 = (addc z6_3 t1_6 cf_26).1 := rfl
  have e_cf_27 : cf_27 = (addc z6_3 t1_6 cf_26).2 := rfl
  clear_value s_24 z6_4 cf_27
  have l_z6_4 : z6_4 + 2^64 * cf_27 = z6_3 + t1_6 + cf_26 := by
    rw [e_z6_4, e_cf_27]; exact addc_lin z6_3 t1_6 cf_26
  have b_z6_4 : z6_4 < 2^64 := by rw [e_z6_4]; exact addc_value_lt z6_3 t1_6 cf_26
  have b_cf_27 : cf_27 ≤ 1 := by rw [e_cf_27]; exact addc_carry_le_one z6_3 t1_6 cf_26 b_z6_3 b_t1_6 b_cf_26
  clear e_z6_4 e_cf_27
  -- z7_3: adc {z7}, {t2}
  extract_lets -merge +onlyGivenNames s_25 z7_3 cf_28 at hr
  have e_z7_3 : z7_3 = (addc z7_2 t2_7 cf_27).1 := rfl
  have e_cf_28 : cf_28 = (addc z7_2 t2_7 cf_27).2 := rfl
  clear_value s_25 z7_3 cf_28
  have l_z7_3 : z7_3 + 2^64 * cf_28 = z7_2 + t2_7 + cf_27 := by
    rw [e_z7_3, e_cf_28]; exact addc_lin z7_2 t2_7 cf_27
  have b_z7_3 : z7_3 < 2^64 := by rw [e_z7_3]; exact addc_value_lt z7_2 t2_7 cf_27
  have b_cf_28 : cf_28 ≤ 1 := by rw [e_cf_28]; exact addc_carry_le_one z7_2 t2_7 cf_27 b_z7_2 b_t2_7 b_cf_27
  clear e_z7_3 e_cf_28
  subst hr
  -- BEGIN squareLo diagonal arithmetic and conclusion
  have hdiagonal :
      z0 + 2^64 * z1_2 + 2^128 * z2_3 + 2^192 * z3_4 + 2^256 * z4_5 +
          2^320 * z5_5 + 2^384 * z6_4 + 2^448 * z7_3 + 2^512 * cf_28 =
        (2^64 * z1_1 + 2^128 * z2_2 + 2^192 * z3_3 + 2^256 * z4_4 +
          2^320 * z5_4 + 2^384 * z6_3 + 2^448 * z7_2) +
        (a0 * a0 + 2^128 * (a1 * a1) + 2^256 * (a2 * a2) + 2^384 * (a3 * a3)) :=
    diagonal_terms (by simpa only [e_rdx_3, e_a0] using d_t2_4)
      (by simpa only [e_rdx_4, e_a1] using d_t2_5)
      (by simpa only [e_rdx_5, e_a2] using d_t2_6)
      (by simpa only [e_rdx_6, e_a3] using d_t2_7)
      (by simpa only [add_zero] using l_z1_2)
      l_z2_3 l_z3_4 l_z4_5 l_z5_5 l_z6_4 l_z7_3
  have hdouble_body :
      2^64 * z1_1 + 2^128 * z2_2 + 2^192 * z3_3 + 2^256 * z4_4 +
          2^320 * z5_4 + 2^384 * z6_3 + 2^448 * z7_2 =
        2 * (2^64 * (a0 * a1) + 2^128 * (a0 * a2) +
          2^192 * (a0 * a3 + a1 * a2) + 2^256 * (a1 * a3) + 2^320 * (a2 * a3)) := by
    simpa only [z_cf_21, mul_zero, add_zero] using hdouble_math
  have hfull :
      z0 + 2^64 * z1_2 + 2^128 * z2_3 + 2^192 * z3_4 + 2^256 * z4_5 +
          2^320 * z5_5 + 2^384 * z6_4 + 2^448 * z7_3 + 2^512 * cf_28 =
        value.toNat * value.toNat :=
    square_parts hdiagonal hdouble_body hsquare_parts
  have z_cf_28 : cf_28 = 0 :=
    top_carry_zero hfull hsquare_lt
  refine ⟨⟨b_z0, b_z1_2, b_z2_3, b_z3_4, b_z4_5, b_z5_5, b_z6_4, b_z7_3⟩, ?_⟩
  show z0 + 2^64 * z1_2 + 2^128 * z2_3 + 2^192 * z3_4 + 2^256 * z4_5 +
      2^320 * z5_5 + 2^384 * z6_4 + 2^448 * z7_3 = value.toNat * value.toNat
  exact drop_zero_carry hfull z_cf_28
  -- END squareLo diagonal arithmetic and conclusion

-- BEGIN squareLo_spec corollary
/-- The x86-64 unreduced squaring block returns the bounded 512-bit square. -/
theorem squareLo_spec (value : Limbs) (hv : value.Bounded) :
    (squareLo value).Bounded ∧ (squareLo value).toNat = value.toNat * value.toNat :=
  squareLo_spec_traced value hv _ rfl
-- END squareLo_spec corollary

end PastaAsm.X86_64
