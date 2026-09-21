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

The `squareHi` proof follows the separate reduction block through four Montgomery cancellation
steps, addition of the product's high half, and conditional subtraction. The composition theorems
combine these two blocks for Montgomery squaring and repeated squaring, then use multiplication
for `sqr_n_mul`.
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

-- BEGIN squareHi arithmetic helpers
/-- One x86-64 Montgomery cancellation step. The low product cancels `t0`; the two carry
chains add the remaining halves of `q * p`, and shifting drops the cancelled low limb. -/
private theorem squareHi_step
    {t0 t1 t2 t3 q p0 p1 p0lo p0hi p1lo p1hi qlo qhi nc
      u1 c1 u2 c2 u3 c3 top c4 v0 c5 v1 c6 v2 c7 v3 c8 : Nat}
    (hc : t0 + p0lo = 2^64 * nc)
    (dp0 : p0lo + 2^64 * p0hi = q * p0)
    (dp1 : p1lo + 2^64 * p1hi = q * p1)
    (dq : qlo + 2^64 * qhi = q * 2^62)
    (lu1 : u1 + 2^64 * c1 = t1 + p1lo + nc)
    (lu2 : u2 + 2^64 * c2 = t2 + 0 + c1)
    (lu3 : u3 + 2^64 * c3 = t3 + qlo + c2)
    (ltop : top + 2^64 * c4 = 0 + 0 + c3)
    (lv0 : v0 + 2^64 * c5 = u1 + p0hi + 0)
    (lv1 : v1 + 2^64 * c6 = u2 + p1hi + c5)
    (lv2 : v2 + 2^64 * c7 = u3 + 0 + c6)
    (lv3 : v3 + 2^64 * c8 = top + qhi + c7)
    (hk : c4 = 0 ∧ c8 = 0) :
    2^64 * (v0 + 2^64 * v1 + 2^128 * v2 + 2^192 * v3) =
      (t0 + 2^64 * t1 + 2^128 * t2 + 2^192 * t3) +
        (q * p0 + 2^64 * (q * p1) + q * 2^254) := by
  omega

/-- Finish the x86-64 conditional subtraction. CF is a borrow flag, so `cmovnc` selects the
subtracted limbs exactly when the candidate is at least the modulus. -/
private theorem squareHi_conclude
    {a0 a1 a2 a3 ka d0 d1 d2 d3 b0 b1 b2 b3 r0 r1 r2 r3 p3 productNat Q : Nat}
    {modulus : Limbs}
    (hmain : 2^256 * (a0 + 2^64 * a1 + 2^128 * a2 + 2^192 * a3 + 2^256 * ka) =
      productNat + Q * modulus.toNat)
    (hA : a0 + 2^64 * a1 + 2^128 * a2 + 2^192 * a3 + 2^256 * ka <
      2 * modulus.toNat)
    (hk : ka = 0)
    (ld0 : d0 + modulus.l0 + 0 = a0 + 2^64 * b0)
    (ld1 : d1 + modulus.l1 + b0 = a1 + 2^64 * b1)
    (ld2 : d2 + 0 + b1 = a2 + 2^64 * b2)
    (ld3 : d3 + p3 + b2 = a3 + 2^64 * b3)
    (hp3 : p3 = 4611686018427387904)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hb3 : b3 ≤ 1)
    (bd0 : d0 < 2^64) (bd1 : d1 < 2^64) (bd2 : d2 < 2^64) (bd3 : d3 < 2^64)
    (br0 : r0 < 2^64) (br1 : r1 < 2^64) (br2 : r2 < 2^64) (br3 : r3 < 2^64)
    (er0 : r0 = if b3 = 0 then d0 else a0)
    (er1 : r1 = if b3 = 0 then d1 else a1)
    (er2 : r2 = if b3 = 0 then d2 else a2)
    (er3 : r3 = if b3 = 0 then d3 else a3) :
    (⟨r0, r1, r2, r3⟩ : Limbs).Bounded ∧
      (⟨r0, r1, r2, r3⟩ : Limbs).toNat < modulus.toNat ∧
      2^256 * (⟨r0, r1, r2, r3⟩ : Limbs).toNat ≡ productNat [MOD modulus.toNat] := by
  have hP : modulus.toNat = modulus.l0 + 2^64 * modulus.l1 + 2^192 * p3 := by
    simp only [Limbs.toNat, hshape.1, hshape.2, hp3]
    ring
  have hD : d0 + 2^64 * d1 + 2^128 * d2 + 2^192 * d3 + modulus.toNat =
      a0 + 2^64 * a1 + 2^128 * a2 + 2^192 * a3 + 2^256 * b3 := by
    clear * - ld0 ld1 ld2 ld3 hP
    omega
  refine ⟨⟨br0, br1, br2, br3⟩, ?_⟩
  show r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3 < modulus.toNat ∧
    2^256 * (r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3) ≡
      productNat [MOD modulus.toNat]
  obtain hb | hb : b3 = 0 ∨ b3 = 1 := by omega
  · rw [if_pos hb] at er0 er1 er2 er3
    constructor
    · clear * - hA hk hD hb er0 er1 er2 er3
      omega
    · exact modEq_of_add_mul _ _ (2^256) Q _ (by
        clear * - hmain hk hD hb er0 er1 er2 er3
        omega)
  · rw [if_neg (by omega)] at er0 er1 er2 er3
    constructor
    · clear * - hD hb bd0 bd1 bd2 bd3 er0 er1 er2 er3
      omega
    · exact modEq_of_add_mul _ _ 0 Q _ (by
        clear * - hmain hk er0 er1 er2 er3
        omega)
-- END squareHi arithmetic helpers

-- BEGIN squareHi_spec statement
set_option exponentiation.threshold 512 in
/-- The high squaring block Montgomery-reduces a bounded eight-limb product. The product bound is
exactly the one supplied by a canonical four-limb square. -/
theorem squareHi_spec (product : WideLimbs) (modulus : Limbs) (inv : Nat)
    (hproduct : product.Bounded) (hm : modulus.Bounded)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinv_lt : inv < 2^64) (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hproduct_lt : product.toNat < 2^256 * modulus.toNat) :
    ∀ r, r = squareHi product modulus inv →
      r.Bounded ∧ r.toNat < modulus.toNat ∧
        2^256 * r.toNat ≡ product.toNat [MOD modulus.toNat] := by
  intro r hr
-- END squareHi_spec statement
  -- generated skeleton for `squareHi`: do not edit between the annotations
  unfold squareHi at hr
  lift_lets -merge at hr
  -- inv': scalar input
  extract_lets -merge +onlyGivenNames inv' at hr
  have e_inv' : inv' = inv := rfl
  clear_value inv'
  have b_inv' : inv' < 2^64 := by rw [e_inv']; exact hinv_lt
  -- p3: operand p3 = const PASTA_HIGH_LIMB
  extract_lets -merge +onlyGivenNames p3 at hr
  have e_p3 : p3 = 4611686018427387904 := rfl
  clear_value p3
  have b_p3 : p3 < 2^64 := by rw [e_p3]; decide
  -- z0: input product[0]
  extract_lets -merge +onlyGivenNames z0 at hr
  have e_z0 : z0 = product.l0 := rfl
  clear_value z0
  have b_z0 : z0 < 2^64 := by rw [e_z0]; exact hproduct.1
  -- z1: input product[1]
  extract_lets -merge +onlyGivenNames z1 at hr
  have e_z1 : z1 = product.l1 := rfl
  clear_value z1
  have b_z1 : z1 < 2^64 := by rw [e_z1]; exact hproduct.2.1
  -- z2: input product[2]
  extract_lets -merge +onlyGivenNames z2 at hr
  have e_z2 : z2 = product.l2 := rfl
  clear_value z2
  have b_z2 : z2 < 2^64 := by rw [e_z2]; exact hproduct.2.2.1
  -- z3: input product[3]
  extract_lets -merge +onlyGivenNames z3 at hr
  have e_z3 : z3 = product.l3 := rfl
  clear_value z3
  have b_z3 : z3 < 2^64 := by rw [e_z3]; exact hproduct.2.2.2.1
  -- z4: input product[4]
  extract_lets -merge +onlyGivenNames z4 at hr
  have e_z4 : z4 = product.l4 := rfl
  clear_value z4
  have b_z4 : z4 < 2^64 := by rw [e_z4]; exact hproduct.2.2.2.2.1
  -- z5: input product[5]
  extract_lets -merge +onlyGivenNames z5 at hr
  have e_z5 : z5 = product.l5 := rfl
  clear_value z5
  have b_z5 : z5 < 2^64 := by rw [e_z5]; exact hproduct.2.2.2.2.2.1
  -- z6: input product[6]
  extract_lets -merge +onlyGivenNames z6 at hr
  have e_z6 : z6 = product.l6 := rfl
  clear_value z6
  have b_z6 : z6 < 2^64 := by rw [e_z6]; exact hproduct.2.2.2.2.2.2.1
  -- z7: input product[7]
  extract_lets -merge +onlyGivenNames z7 at hr
  have e_z7 : z7 = product.l7 := rfl
  clear_value z7
  have b_z7 : z7 < 2^64 := by rw [e_z7]; exact hproduct.2.2.2.2.2.2.2
  -- rdx: mov rdx, {z0}
  extract_lets -merge +onlyGivenNames rdx at hr
  have e_rdx : rdx = z0 := rfl
  clear_value rdx
  have b_rdx : rdx < 2^64 := by rw [e_rdx]; exact b_z0
  -- rdx_1: imul rdx, {inv}
  extract_lets -merge +onlyGivenNames rdx_1 at hr
  have e_rdx_1 : rdx_1 = rdx * inv' % 2^64 := rfl
  clear_value rdx_1
  have b_rdx_1 : rdx_1 < 2^64 := by rw [e_rdx_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- m: mulx {t2}, {t1}, qword ptr [{p} + 8]
  extract_lets -merge +onlyGivenNames m t2 t1 at hr
  have e_t2 : t2 = (mulx rdx_1 modulus.l1).1 := rfl
  have e_t1 : t1 = (mulx rdx_1 modulus.l1).2 := rfl
  clear_value m t2 t1
  have b_t2 : t2 < 2^64 := by rw [e_t2]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_1 hm.2.1)
  have b_t1 : t1 < 2^64 := by rw [e_t1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t2 : t1 + 2^64 * t2 = rdx_1 * modulus.l1 := by
    rw [e_t1, e_t2]; exact Nat.mod_add_div _ _
  -- a: mov {a}, rdx
  extract_lets -merge +onlyGivenNames a at hr
  have e_a : a = rdx_1 := rfl
  clear_value a
  have b_a : a < 2^64 := by rw [e_a]; exact b_rdx_1
  -- a_1: shl {a}, 62
  extract_lets -merge +onlyGivenNames a_1 at hr
  have e_a_1 : a_1 = a * 2^62 % 2^64 := rfl
  clear_value a_1
  have b_a_1 : a_1 < 2^64 := by rw [e_a_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_a_1 : a_1 + 2^64 * (a / 2^2) = a * 2^62 := by
    rw [e_a_1]; exact lsl62_lsr2_split _
  -- n: neg {z0}
  extract_lets -merge +onlyGivenNames n z0_1 cf at hr
  have e_z0_1 : z0_1 = (neg z0).1 := rfl
  have e_cf : cf = (neg z0).2 := rfl
  clear_value n z0_1 cf
  have b_z0_1 : z0_1 < 2^64 := by rw [e_z0_1]; exact sbb_value_lt 0 z0 0
  have b_cf : cf ≤ 1 := by
    rw [e_cf]; simp only [neg]; split <;> omega
  -- z1_1: adc {z1}, {t1}
  extract_lets -merge +onlyGivenNames s z1_1 cf_1 at hr
  have e_z1_1 : z1_1 = (addc z1 t1 cf).1 := rfl
  have e_cf_1 : cf_1 = (addc z1 t1 cf).2 := rfl
  clear_value s z1_1 cf_1
  have l_z1_1 : z1_1 + 2^64 * cf_1 = z1 + t1 + cf := by
    rw [e_z1_1, e_cf_1]; exact addc_lin z1 t1 cf
  have b_z1_1 : z1_1 < 2^64 := by rw [e_z1_1]; exact addc_value_lt z1 t1 cf
  have b_cf_1 : cf_1 ≤ 1 := by rw [e_cf_1]; exact addc_carry_le_one z1 t1 cf b_z1 b_t1 b_cf
  clear e_z1_1 e_cf_1
  -- z2_1: adc {z2}, 0
  extract_lets -merge +onlyGivenNames s_1 z2_1 cf_2 at hr
  have e_z2_1 : z2_1 = (addc z2 0 cf_1).1 := rfl
  have e_cf_2 : cf_2 = (addc z2 0 cf_1).2 := rfl
  clear_value s_1 z2_1 cf_2
  have l_z2_1 : z2_1 + 2^64 * cf_2 = z2 + 0 + cf_1 := by
    rw [e_z2_1, e_cf_2]; exact addc_lin z2 0 cf_1
  have b_z2_1 : z2_1 < 2^64 := by rw [e_z2_1]; exact addc_value_lt z2 0 cf_1
  have b_cf_2 : cf_2 ≤ 1 := by rw [e_cf_2]; exact addc_carry_le_one z2 0 cf_1 b_z2 (by decide) b_cf_1
  clear e_z2_1 e_cf_2
  -- z3_1: adc {z3}, {a}
  extract_lets -merge +onlyGivenNames s_2 z3_1 cf_3 at hr
  have e_z3_1 : z3_1 = (addc z3 a_1 cf_2).1 := rfl
  have e_cf_3 : cf_3 = (addc z3 a_1 cf_2).2 := rfl
  clear_value s_2 z3_1 cf_3
  have l_z3_1 : z3_1 + 2^64 * cf_3 = z3 + a_1 + cf_2 := by
    rw [e_z3_1, e_cf_3]; exact addc_lin z3 a_1 cf_2
  have b_z3_1 : z3_1 < 2^64 := by rw [e_z3_1]; exact addc_value_lt z3 a_1 cf_2
  have b_cf_3 : cf_3 ≤ 1 := by rw [e_cf_3]; exact addc_carry_le_one z3 a_1 cf_2 b_z3 b_a_1 b_cf_2
  clear e_z3_1 e_cf_3
  -- a_2: mov {a}, 0
  extract_lets -merge +onlyGivenNames a_2 at hr
  have e_a_2 : a_2 = 0 := rfl
  clear_value a_2
  have b_a_2 : a_2 < 2^64 := by rw [e_a_2]; decide
  -- a_3: adc {a}, 0
  extract_lets -merge +onlyGivenNames s_3 a_3 cf_4 at hr
  have e_a_3 : a_3 = (addc a_2 0 cf_3).1 := rfl
  have e_cf_4 : cf_4 = (addc a_2 0 cf_3).2 := rfl
  clear_value s_3 a_3 cf_4
  have l_a_3 : a_3 + 2^64 * cf_4 = a_2 + 0 + cf_3 := by
    rw [e_a_3, e_cf_4]; exact addc_lin a_2 0 cf_3
  have b_a_3 : a_3 < 2^64 := by rw [e_a_3]; exact addc_value_lt a_2 0 cf_3
  have b_cf_4 : cf_4 ≤ 1 := by rw [e_cf_4]; exact addc_carry_le_one a_2 0 cf_3 b_a_2 (by decide) b_cf_3
  clear e_a_3 e_cf_4
  -- m_1: mulx {t1}, {z0}, qword ptr [{p}]
  extract_lets -merge +onlyGivenNames m_1 t1_1 z0_2 at hr
  have e_t1_1 : t1_1 = (mulx rdx_1 modulus.l0).1 := rfl
  have e_z0_2 : z0_2 = (mulx rdx_1 modulus.l0).2 := rfl
  clear_value m_1 t1_1 z0_2
  have b_t1_1 : t1_1 < 2^64 := by rw [e_t1_1]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_1 hm.1)
  have b_z0_2 : z0_2 < 2^64 := by rw [e_z0_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t1_1 : z0_2 + 2^64 * t1_1 = rdx_1 * modulus.l0 := by
    rw [e_z0_2, e_t1_1]; exact Nat.mod_add_div _ _
  -- z0_3: mov {z0}, rdx
  extract_lets -merge +onlyGivenNames z0_3 at hr
  have e_z0_3 : z0_3 = rdx_1 := rfl
  clear_value z0_3
  have b_z0_3 : z0_3 < 2^64 := by rw [e_z0_3]; exact b_rdx_1
  -- z0_4: shr {z0}, 2
  extract_lets -merge +onlyGivenNames z0_4 at hr
  have e_z0_4 : z0_4 = z0_3 / 2^2 := rfl
  clear_value z0_4
  have b_z0_4 : z0_4 < 2^62 := by
    rw [e_z0_4]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_z0_3 (by norm_num))
  -- z1_2: add {z1}, {t1}
  extract_lets -merge +onlyGivenNames s_4 z1_2 cf_5 at hr
  have e_z1_2 : z1_2 = (addc z1_1 t1_1 0).1 := rfl
  have e_cf_5 : cf_5 = (addc z1_1 t1_1 0).2 := rfl
  clear_value s_4 z1_2 cf_5
  have l_z1_2 : z1_2 + 2^64 * cf_5 = z1_1 + t1_1 + 0 := by
    rw [e_z1_2, e_cf_5]; exact addc_lin z1_1 t1_1 0
  have b_z1_2 : z1_2 < 2^64 := by rw [e_z1_2]; exact addc_value_lt z1_1 t1_1 0
  have b_cf_5 : cf_5 ≤ 1 := by rw [e_cf_5]; exact addc_carry_le_one z1_1 t1_1 0 b_z1_1 b_t1_1 (by decide)
  clear e_z1_2 e_cf_5
  -- z2_2: adc {z2}, {t2}
  extract_lets -merge +onlyGivenNames s_5 z2_2 cf_6 at hr
  have e_z2_2 : z2_2 = (addc z2_1 t2 cf_5).1 := rfl
  have e_cf_6 : cf_6 = (addc z2_1 t2 cf_5).2 := rfl
  clear_value s_5 z2_2 cf_6
  have l_z2_2 : z2_2 + 2^64 * cf_6 = z2_1 + t2 + cf_5 := by
    rw [e_z2_2, e_cf_6]; exact addc_lin z2_1 t2 cf_5
  have b_z2_2 : z2_2 < 2^64 := by rw [e_z2_2]; exact addc_value_lt z2_1 t2 cf_5
  have b_cf_6 : cf_6 ≤ 1 := by rw [e_cf_6]; exact addc_carry_le_one z2_1 t2 cf_5 b_z2_1 b_t2 b_cf_5
  clear e_z2_2 e_cf_6
  -- z3_2: adc {z3}, 0
  extract_lets -merge +onlyGivenNames s_6 z3_2 cf_7 at hr
  have e_z3_2 : z3_2 = (addc z3_1 0 cf_6).1 := rfl
  have e_cf_7 : cf_7 = (addc z3_1 0 cf_6).2 := rfl
  clear_value s_6 z3_2 cf_7
  have l_z3_2 : z3_2 + 2^64 * cf_7 = z3_1 + 0 + cf_6 := by
    rw [e_z3_2, e_cf_7]; exact addc_lin z3_1 0 cf_6
  have b_z3_2 : z3_2 < 2^64 := by rw [e_z3_2]; exact addc_value_lt z3_1 0 cf_6
  have b_cf_7 : cf_7 ≤ 1 := by rw [e_cf_7]; exact addc_carry_le_one z3_1 0 cf_6 b_z3_1 (by decide) b_cf_6
  clear e_z3_2 e_cf_7
  -- a_4: adc {a}, {z0}
  extract_lets -merge +onlyGivenNames s_7 a_4 cf_8 at hr
  have e_a_4 : a_4 = (addc a_3 z0_4 cf_7).1 := rfl
  have e_cf_8 : cf_8 = (addc a_3 z0_4 cf_7).2 := rfl
  clear_value s_7 a_4 cf_8
  have l_a_4 : a_4 + 2^64 * cf_8 = a_3 + z0_4 + cf_7 := by
    rw [e_a_4, e_cf_8]; exact addc_lin a_3 z0_4 cf_7
  have b_a_4 : a_4 < 2^64 := by rw [e_a_4]; exact addc_value_lt a_3 z0_4 cf_7
  have b_cf_8 : cf_8 ≤ 1 := by rw [e_cf_8]; exact addc_carry_le_one a_3 z0_4 cf_7 b_a_3 (lt_of_lt_of_le b_z0_4 (by norm_num)) b_cf_7
  clear e_a_4 e_cf_8
  -- BEGIN squareHi step 0
  have hc_0 : z0 + z0_2 = 2^64 * cf := by
    have hq : rdx_1 = mulLo inv' z0 := by
      simp only [mulLo, e_rdx_1, e_rdx, Nat.mul_comm]
    have hc := neg_carry_cancel z0 inv' modulus.l0 rdx_1 b_z0 b_inv' hm.1
      (by simpa only [e_inv'] using hinv) hq
    rw [e_z0_2]
    simp only [mulx]
    rw [e_cf]
    simpa only [neg] using hc
  have hk_0 : cf_4 = 0 ∧ cf_8 = 0 := by
    clear * - l_a_3 e_a_2 b_cf_3 l_a_4 b_z0_4 b_cf_7
    omega
  have I_0 : 2^64 * (z1_2 + 2^64 * z2_2 + 2^128 * z3_2 + 2^192 * a_4) =
      (z0 + 2^64 * z1 + 2^128 * z2 + 2^192 * z3) +
        (rdx_1 * modulus.l0 + 2^64 * (rdx_1 * modulus.l1) + rdx_1 * 2^254) :=
    squareHi_step hc_0 d_t1_1 d_t2 (by rw [e_z0_4, e_z0_3]; simpa only [e_a] using sh_a_1)
      l_z1_1 l_z2_1 l_z3_1 (by simpa only [e_a_2] using l_a_3)
      l_z1_2 l_z2_2 l_z3_2 l_a_4 hk_0
  -- END squareHi step 0
  -- rdx_2: mov rdx, {z1}
  extract_lets -merge +onlyGivenNames rdx_2 at hr
  have e_rdx_2 : rdx_2 = z1_2 := rfl
  clear_value rdx_2
  have b_rdx_2 : rdx_2 < 2^64 := by rw [e_rdx_2]; exact b_z1_2
  -- rdx_3: imul rdx, {inv}
  extract_lets -merge +onlyGivenNames rdx_3 at hr
  have e_rdx_3 : rdx_3 = rdx_2 * inv' % 2^64 := rfl
  clear_value rdx_3
  have b_rdx_3 : rdx_3 < 2^64 := by rw [e_rdx_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- m_2: mulx {t2}, {t1}, qword ptr [{p} + 8]
  extract_lets -merge +onlyGivenNames m_2 t2_1 t1_2 at hr
  have e_t2_1 : t2_1 = (mulx rdx_3 modulus.l1).1 := rfl
  have e_t1_2 : t1_2 = (mulx rdx_3 modulus.l1).2 := rfl
  clear_value m_2 t2_1 t1_2
  have b_t2_1 : t2_1 < 2^64 := by rw [e_t2_1]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_3 hm.2.1)
  have b_t1_2 : t1_2 < 2^64 := by rw [e_t1_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t2_1 : t1_2 + 2^64 * t2_1 = rdx_3 * modulus.l1 := by
    rw [e_t1_2, e_t2_1]; exact Nat.mod_add_div _ _
  -- z0_5: mov {z0}, rdx
  extract_lets -merge +onlyGivenNames z0_5 at hr
  have e_z0_5 : z0_5 = rdx_3 := rfl
  clear_value z0_5
  have b_z0_5 : z0_5 < 2^64 := by rw [e_z0_5]; exact b_rdx_3
  -- z0_6: shl {z0}, 62
  extract_lets -merge +onlyGivenNames z0_6 at hr
  have e_z0_6 : z0_6 = z0_5 * 2^62 % 2^64 := rfl
  clear_value z0_6
  have b_z0_6 : z0_6 < 2^64 := by rw [e_z0_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_z0_6 : z0_6 + 2^64 * (z0_5 / 2^2) = z0_5 * 2^62 := by
    rw [e_z0_6]; exact lsl62_lsr2_split _
  -- n_1: neg {z1}
  extract_lets -merge +onlyGivenNames n_1 z1_3 cf_9 at hr
  have e_z1_3 : z1_3 = (neg z1_2).1 := rfl
  have e_cf_9 : cf_9 = (neg z1_2).2 := rfl
  clear_value n_1 z1_3 cf_9
  have b_z1_3 : z1_3 < 2^64 := by rw [e_z1_3]; exact sbb_value_lt 0 z1_2 0
  have b_cf_9 : cf_9 ≤ 1 := by
    rw [e_cf_9]; simp only [neg]; split <;> omega
  -- z2_3: adc {z2}, {t1}
  extract_lets -merge +onlyGivenNames s_8 z2_3 cf_10 at hr
  have e_z2_3 : z2_3 = (addc z2_2 t1_2 cf_9).1 := rfl
  have e_cf_10 : cf_10 = (addc z2_2 t1_2 cf_9).2 := rfl
  clear_value s_8 z2_3 cf_10
  have l_z2_3 : z2_3 + 2^64 * cf_10 = z2_2 + t1_2 + cf_9 := by
    rw [e_z2_3, e_cf_10]; exact addc_lin z2_2 t1_2 cf_9
  have b_z2_3 : z2_3 < 2^64 := by rw [e_z2_3]; exact addc_value_lt z2_2 t1_2 cf_9
  have b_cf_10 : cf_10 ≤ 1 := by rw [e_cf_10]; exact addc_carry_le_one z2_2 t1_2 cf_9 b_z2_2 b_t1_2 b_cf_9
  clear e_z2_3 e_cf_10
  -- z3_3: adc {z3}, 0
  extract_lets -merge +onlyGivenNames s_9 z3_3 cf_11 at hr
  have e_z3_3 : z3_3 = (addc z3_2 0 cf_10).1 := rfl
  have e_cf_11 : cf_11 = (addc z3_2 0 cf_10).2 := rfl
  clear_value s_9 z3_3 cf_11
  have l_z3_3 : z3_3 + 2^64 * cf_11 = z3_2 + 0 + cf_10 := by
    rw [e_z3_3, e_cf_11]; exact addc_lin z3_2 0 cf_10
  have b_z3_3 : z3_3 < 2^64 := by rw [e_z3_3]; exact addc_value_lt z3_2 0 cf_10
  have b_cf_11 : cf_11 ≤ 1 := by rw [e_cf_11]; exact addc_carry_le_one z3_2 0 cf_10 b_z3_2 (by decide) b_cf_10
  clear e_z3_3 e_cf_11
  -- a_5: adc {a}, {z0}
  extract_lets -merge +onlyGivenNames s_10 a_5 cf_12 at hr
  have e_a_5 : a_5 = (addc a_4 z0_6 cf_11).1 := rfl
  have e_cf_12 : cf_12 = (addc a_4 z0_6 cf_11).2 := rfl
  clear_value s_10 a_5 cf_12
  have l_a_5 : a_5 + 2^64 * cf_12 = a_4 + z0_6 + cf_11 := by
    rw [e_a_5, e_cf_12]; exact addc_lin a_4 z0_6 cf_11
  have b_a_5 : a_5 < 2^64 := by rw [e_a_5]; exact addc_value_lt a_4 z0_6 cf_11
  have b_cf_12 : cf_12 ≤ 1 := by rw [e_cf_12]; exact addc_carry_le_one a_4 z0_6 cf_11 b_a_4 b_z0_6 b_cf_11
  clear e_a_5 e_cf_12
  -- z0_7: mov {z0}, 0
  extract_lets -merge +onlyGivenNames z0_7 at hr
  have e_z0_7 : z0_7 = 0 := rfl
  clear_value z0_7
  have b_z0_7 : z0_7 < 2^64 := by rw [e_z0_7]; decide
  -- z0_8: adc {z0}, 0
  extract_lets -merge +onlyGivenNames s_11 z0_8 cf_13 at hr
  have e_z0_8 : z0_8 = (addc z0_7 0 cf_12).1 := rfl
  have e_cf_13 : cf_13 = (addc z0_7 0 cf_12).2 := rfl
  clear_value s_11 z0_8 cf_13
  have l_z0_8 : z0_8 + 2^64 * cf_13 = z0_7 + 0 + cf_12 := by
    rw [e_z0_8, e_cf_13]; exact addc_lin z0_7 0 cf_12
  have b_z0_8 : z0_8 < 2^64 := by rw [e_z0_8]; exact addc_value_lt z0_7 0 cf_12
  have b_cf_13 : cf_13 ≤ 1 := by rw [e_cf_13]; exact addc_carry_le_one z0_7 0 cf_12 b_z0_7 (by decide) b_cf_12
  clear e_z0_8 e_cf_13
  -- m_3: mulx {t1}, {z1}, qword ptr [{p}]
  extract_lets -merge +onlyGivenNames m_3 t1_3 z1_4 at hr
  have e_t1_3 : t1_3 = (mulx rdx_3 modulus.l0).1 := rfl
  have e_z1_4 : z1_4 = (mulx rdx_3 modulus.l0).2 := rfl
  clear_value m_3 t1_3 z1_4
  have b_t1_3 : t1_3 < 2^64 := by rw [e_t1_3]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_3 hm.1)
  have b_z1_4 : z1_4 < 2^64 := by rw [e_z1_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t1_3 : z1_4 + 2^64 * t1_3 = rdx_3 * modulus.l0 := by
    rw [e_z1_4, e_t1_3]; exact Nat.mod_add_div _ _
  -- z1_5: mov {z1}, rdx
  extract_lets -merge +onlyGivenNames z1_5 at hr
  have e_z1_5 : z1_5 = rdx_3 := rfl
  clear_value z1_5
  have b_z1_5 : z1_5 < 2^64 := by rw [e_z1_5]; exact b_rdx_3
  -- z1_6: shr {z1}, 2
  extract_lets -merge +onlyGivenNames z1_6 at hr
  have e_z1_6 : z1_6 = z1_5 / 2^2 := rfl
  clear_value z1_6
  have b_z1_6 : z1_6 < 2^62 := by
    rw [e_z1_6]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_z1_5 (by norm_num))
  -- z2_4: add {z2}, {t1}
  extract_lets -merge +onlyGivenNames s_12 z2_4 cf_14 at hr
  have e_z2_4 : z2_4 = (addc z2_3 t1_3 0).1 := rfl
  have e_cf_14 : cf_14 = (addc z2_3 t1_3 0).2 := rfl
  clear_value s_12 z2_4 cf_14
  have l_z2_4 : z2_4 + 2^64 * cf_14 = z2_3 + t1_3 + 0 := by
    rw [e_z2_4, e_cf_14]; exact addc_lin z2_3 t1_3 0
  have b_z2_4 : z2_4 < 2^64 := by rw [e_z2_4]; exact addc_value_lt z2_3 t1_3 0
  have b_cf_14 : cf_14 ≤ 1 := by rw [e_cf_14]; exact addc_carry_le_one z2_3 t1_3 0 b_z2_3 b_t1_3 (by decide)
  clear e_z2_4 e_cf_14
  -- z3_4: adc {z3}, {t2}
  extract_lets -merge +onlyGivenNames s_13 z3_4 cf_15 at hr
  have e_z3_4 : z3_4 = (addc z3_3 t2_1 cf_14).1 := rfl
  have e_cf_15 : cf_15 = (addc z3_3 t2_1 cf_14).2 := rfl
  clear_value s_13 z3_4 cf_15
  have l_z3_4 : z3_4 + 2^64 * cf_15 = z3_3 + t2_1 + cf_14 := by
    rw [e_z3_4, e_cf_15]; exact addc_lin z3_3 t2_1 cf_14
  have b_z3_4 : z3_4 < 2^64 := by rw [e_z3_4]; exact addc_value_lt z3_3 t2_1 cf_14
  have b_cf_15 : cf_15 ≤ 1 := by rw [e_cf_15]; exact addc_carry_le_one z3_3 t2_1 cf_14 b_z3_3 b_t2_1 b_cf_14
  clear e_z3_4 e_cf_15
  -- a_6: adc {a}, 0
  extract_lets -merge +onlyGivenNames s_14 a_6 cf_16 at hr
  have e_a_6 : a_6 = (addc a_5 0 cf_15).1 := rfl
  have e_cf_16 : cf_16 = (addc a_5 0 cf_15).2 := rfl
  clear_value s_14 a_6 cf_16
  have l_a_6 : a_6 + 2^64 * cf_16 = a_5 + 0 + cf_15 := by
    rw [e_a_6, e_cf_16]; exact addc_lin a_5 0 cf_15
  have b_a_6 : a_6 < 2^64 := by rw [e_a_6]; exact addc_value_lt a_5 0 cf_15
  have b_cf_16 : cf_16 ≤ 1 := by rw [e_cf_16]; exact addc_carry_le_one a_5 0 cf_15 b_a_5 (by decide) b_cf_15
  clear e_a_6 e_cf_16
  -- z0_9: adc {z0}, {z1}
  extract_lets -merge +onlyGivenNames s_15 z0_9 cf_17 at hr
  have e_z0_9 : z0_9 = (addc z0_8 z1_6 cf_16).1 := rfl
  have e_cf_17 : cf_17 = (addc z0_8 z1_6 cf_16).2 := rfl
  clear_value s_15 z0_9 cf_17
  have l_z0_9 : z0_9 + 2^64 * cf_17 = z0_8 + z1_6 + cf_16 := by
    rw [e_z0_9, e_cf_17]; exact addc_lin z0_8 z1_6 cf_16
  have b_z0_9 : z0_9 < 2^64 := by rw [e_z0_9]; exact addc_value_lt z0_8 z1_6 cf_16
  have b_cf_17 : cf_17 ≤ 1 := by rw [e_cf_17]; exact addc_carry_le_one z0_8 z1_6 cf_16 b_z0_8 (lt_of_lt_of_le b_z1_6 (by norm_num)) b_cf_16
  clear e_z0_9 e_cf_17
  -- BEGIN squareHi step 1
  have hc_1 : z1_2 + z1_4 = 2^64 * cf_9 := by
    have hq : rdx_3 = mulLo inv' z1_2 := by
      simp only [mulLo, e_rdx_3, e_rdx_2, Nat.mul_comm]
    have hc := neg_carry_cancel z1_2 inv' modulus.l0 rdx_3 b_z1_2 b_inv' hm.1
      (by simpa only [e_inv'] using hinv) hq
    rw [e_z1_4]
    simp only [mulx]
    rw [e_cf_9]
    simpa only [neg] using hc
  have hk_1 : cf_13 = 0 ∧ cf_17 = 0 := by
    clear * - l_z0_8 e_z0_7 b_cf_12 l_z0_9 b_z1_6 b_cf_16
    omega
  have I_1 : 2^64 * (z2_4 + 2^64 * z3_4 + 2^128 * a_6 + 2^192 * z0_9) =
      (z1_2 + 2^64 * z2_2 + 2^128 * z3_2 + 2^192 * a_4) +
        (rdx_3 * modulus.l0 + 2^64 * (rdx_3 * modulus.l1) + rdx_3 * 2^254) :=
    squareHi_step hc_1 d_t1_3 d_t2_1 (by rw [e_z1_6, e_z1_5]; simpa only [e_z0_5] using sh_z0_6)
      l_z2_3 l_z3_3 l_a_5 (by simpa only [e_z0_7] using l_z0_8)
      l_z2_4 l_z3_4 l_a_6 l_z0_9 hk_1
  -- END squareHi step 1
  -- rdx_4: mov rdx, {z2}
  extract_lets -merge +onlyGivenNames rdx_4 at hr
  have e_rdx_4 : rdx_4 = z2_4 := rfl
  clear_value rdx_4
  have b_rdx_4 : rdx_4 < 2^64 := by rw [e_rdx_4]; exact b_z2_4
  -- rdx_5: imul rdx, {inv}
  extract_lets -merge +onlyGivenNames rdx_5 at hr
  have e_rdx_5 : rdx_5 = rdx_4 * inv' % 2^64 := rfl
  clear_value rdx_5
  have b_rdx_5 : rdx_5 < 2^64 := by rw [e_rdx_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- m_4: mulx {t2}, {t1}, qword ptr [{p} + 8]
  extract_lets -merge +onlyGivenNames m_4 t2_2 t1_4 at hr
  have e_t2_2 : t2_2 = (mulx rdx_5 modulus.l1).1 := rfl
  have e_t1_4 : t1_4 = (mulx rdx_5 modulus.l1).2 := rfl
  clear_value m_4 t2_2 t1_4
  have b_t2_2 : t2_2 < 2^64 := by rw [e_t2_2]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_5 hm.2.1)
  have b_t1_4 : t1_4 < 2^64 := by rw [e_t1_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t2_2 : t1_4 + 2^64 * t2_2 = rdx_5 * modulus.l1 := by
    rw [e_t1_4, e_t2_2]; exact Nat.mod_add_div _ _
  -- z1_7: mov {z1}, rdx
  extract_lets -merge +onlyGivenNames z1_7 at hr
  have e_z1_7 : z1_7 = rdx_5 := rfl
  clear_value z1_7
  have b_z1_7 : z1_7 < 2^64 := by rw [e_z1_7]; exact b_rdx_5
  -- z1_8: shl {z1}, 62
  extract_lets -merge +onlyGivenNames z1_8 at hr
  have e_z1_8 : z1_8 = z1_7 * 2^62 % 2^64 := rfl
  clear_value z1_8
  have b_z1_8 : z1_8 < 2^64 := by rw [e_z1_8]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_z1_8 : z1_8 + 2^64 * (z1_7 / 2^2) = z1_7 * 2^62 := by
    rw [e_z1_8]; exact lsl62_lsr2_split _
  -- n_2: neg {z2}
  extract_lets -merge +onlyGivenNames n_2 z2_5 cf_18 at hr
  have e_z2_5 : z2_5 = (neg z2_4).1 := rfl
  have e_cf_18 : cf_18 = (neg z2_4).2 := rfl
  clear_value n_2 z2_5 cf_18
  have b_z2_5 : z2_5 < 2^64 := by rw [e_z2_5]; exact sbb_value_lt 0 z2_4 0
  have b_cf_18 : cf_18 ≤ 1 := by
    rw [e_cf_18]; simp only [neg]; split <;> omega
  -- z3_5: adc {z3}, {t1}
  extract_lets -merge +onlyGivenNames s_16 z3_5 cf_19 at hr
  have e_z3_5 : z3_5 = (addc z3_4 t1_4 cf_18).1 := rfl
  have e_cf_19 : cf_19 = (addc z3_4 t1_4 cf_18).2 := rfl
  clear_value s_16 z3_5 cf_19
  have l_z3_5 : z3_5 + 2^64 * cf_19 = z3_4 + t1_4 + cf_18 := by
    rw [e_z3_5, e_cf_19]; exact addc_lin z3_4 t1_4 cf_18
  have b_z3_5 : z3_5 < 2^64 := by rw [e_z3_5]; exact addc_value_lt z3_4 t1_4 cf_18
  have b_cf_19 : cf_19 ≤ 1 := by rw [e_cf_19]; exact addc_carry_le_one z3_4 t1_4 cf_18 b_z3_4 b_t1_4 b_cf_18
  clear e_z3_5 e_cf_19
  -- a_7: adc {a}, 0
  extract_lets -merge +onlyGivenNames s_17 a_7 cf_20 at hr
  have e_a_7 : a_7 = (addc a_6 0 cf_19).1 := rfl
  have e_cf_20 : cf_20 = (addc a_6 0 cf_19).2 := rfl
  clear_value s_17 a_7 cf_20
  have l_a_7 : a_7 + 2^64 * cf_20 = a_6 + 0 + cf_19 := by
    rw [e_a_7, e_cf_20]; exact addc_lin a_6 0 cf_19
  have b_a_7 : a_7 < 2^64 := by rw [e_a_7]; exact addc_value_lt a_6 0 cf_19
  have b_cf_20 : cf_20 ≤ 1 := by rw [e_cf_20]; exact addc_carry_le_one a_6 0 cf_19 b_a_6 (by decide) b_cf_19
  clear e_a_7 e_cf_20
  -- z0_10: adc {z0}, {z1}
  extract_lets -merge +onlyGivenNames s_18 z0_10 cf_21 at hr
  have e_z0_10 : z0_10 = (addc z0_9 z1_8 cf_20).1 := rfl
  have e_cf_21 : cf_21 = (addc z0_9 z1_8 cf_20).2 := rfl
  clear_value s_18 z0_10 cf_21
  have l_z0_10 : z0_10 + 2^64 * cf_21 = z0_9 + z1_8 + cf_20 := by
    rw [e_z0_10, e_cf_21]; exact addc_lin z0_9 z1_8 cf_20
  have b_z0_10 : z0_10 < 2^64 := by rw [e_z0_10]; exact addc_value_lt z0_9 z1_8 cf_20
  have b_cf_21 : cf_21 ≤ 1 := by rw [e_cf_21]; exact addc_carry_le_one z0_9 z1_8 cf_20 b_z0_9 b_z1_8 b_cf_20
  clear e_z0_10 e_cf_21
  -- z1_9: mov {z1}, 0
  extract_lets -merge +onlyGivenNames z1_9 at hr
  have e_z1_9 : z1_9 = 0 := rfl
  clear_value z1_9
  have b_z1_9 : z1_9 < 2^64 := by rw [e_z1_9]; decide
  -- z1_10: adc {z1}, 0
  extract_lets -merge +onlyGivenNames s_19 z1_10 cf_22 at hr
  have e_z1_10 : z1_10 = (addc z1_9 0 cf_21).1 := rfl
  have e_cf_22 : cf_22 = (addc z1_9 0 cf_21).2 := rfl
  clear_value s_19 z1_10 cf_22
  have l_z1_10 : z1_10 + 2^64 * cf_22 = z1_9 + 0 + cf_21 := by
    rw [e_z1_10, e_cf_22]; exact addc_lin z1_9 0 cf_21
  have b_z1_10 : z1_10 < 2^64 := by rw [e_z1_10]; exact addc_value_lt z1_9 0 cf_21
  have b_cf_22 : cf_22 ≤ 1 := by rw [e_cf_22]; exact addc_carry_le_one z1_9 0 cf_21 b_z1_9 (by decide) b_cf_21
  clear e_z1_10 e_cf_22
  -- m_5: mulx {t1}, {z2}, qword ptr [{p}]
  extract_lets -merge +onlyGivenNames m_5 t1_5 z2_6 at hr
  have e_t1_5 : t1_5 = (mulx rdx_5 modulus.l0).1 := rfl
  have e_z2_6 : z2_6 = (mulx rdx_5 modulus.l0).2 := rfl
  clear_value m_5 t1_5 z2_6
  have b_t1_5 : t1_5 < 2^64 := by rw [e_t1_5]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_5 hm.1)
  have b_z2_6 : z2_6 < 2^64 := by rw [e_z2_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t1_5 : z2_6 + 2^64 * t1_5 = rdx_5 * modulus.l0 := by
    rw [e_z2_6, e_t1_5]; exact Nat.mod_add_div _ _
  -- z2_7: mov {z2}, rdx
  extract_lets -merge +onlyGivenNames z2_7 at hr
  have e_z2_7 : z2_7 = rdx_5 := rfl
  clear_value z2_7
  have b_z2_7 : z2_7 < 2^64 := by rw [e_z2_7]; exact b_rdx_5
  -- z2_8: shr {z2}, 2
  extract_lets -merge +onlyGivenNames z2_8 at hr
  have e_z2_8 : z2_8 = z2_7 / 2^2 := rfl
  clear_value z2_8
  have b_z2_8 : z2_8 < 2^62 := by
    rw [e_z2_8]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_z2_7 (by norm_num))
  -- z3_6: add {z3}, {t1}
  extract_lets -merge +onlyGivenNames s_20 z3_6 cf_23 at hr
  have e_z3_6 : z3_6 = (addc z3_5 t1_5 0).1 := rfl
  have e_cf_23 : cf_23 = (addc z3_5 t1_5 0).2 := rfl
  clear_value s_20 z3_6 cf_23
  have l_z3_6 : z3_6 + 2^64 * cf_23 = z3_5 + t1_5 + 0 := by
    rw [e_z3_6, e_cf_23]; exact addc_lin z3_5 t1_5 0
  have b_z3_6 : z3_6 < 2^64 := by rw [e_z3_6]; exact addc_value_lt z3_5 t1_5 0
  have b_cf_23 : cf_23 ≤ 1 := by rw [e_cf_23]; exact addc_carry_le_one z3_5 t1_5 0 b_z3_5 b_t1_5 (by decide)
  clear e_z3_6 e_cf_23
  -- a_8: adc {a}, {t2}
  extract_lets -merge +onlyGivenNames s_21 a_8 cf_24 at hr
  have e_a_8 : a_8 = (addc a_7 t2_2 cf_23).1 := rfl
  have e_cf_24 : cf_24 = (addc a_7 t2_2 cf_23).2 := rfl
  clear_value s_21 a_8 cf_24
  have l_a_8 : a_8 + 2^64 * cf_24 = a_7 + t2_2 + cf_23 := by
    rw [e_a_8, e_cf_24]; exact addc_lin a_7 t2_2 cf_23
  have b_a_8 : a_8 < 2^64 := by rw [e_a_8]; exact addc_value_lt a_7 t2_2 cf_23
  have b_cf_24 : cf_24 ≤ 1 := by rw [e_cf_24]; exact addc_carry_le_one a_7 t2_2 cf_23 b_a_7 b_t2_2 b_cf_23
  clear e_a_8 e_cf_24
  -- z0_11: adc {z0}, 0
  extract_lets -merge +onlyGivenNames s_22 z0_11 cf_25 at hr
  have e_z0_11 : z0_11 = (addc z0_10 0 cf_24).1 := rfl
  have e_cf_25 : cf_25 = (addc z0_10 0 cf_24).2 := rfl
  clear_value s_22 z0_11 cf_25
  have l_z0_11 : z0_11 + 2^64 * cf_25 = z0_10 + 0 + cf_24 := by
    rw [e_z0_11, e_cf_25]; exact addc_lin z0_10 0 cf_24
  have b_z0_11 : z0_11 < 2^64 := by rw [e_z0_11]; exact addc_value_lt z0_10 0 cf_24
  have b_cf_25 : cf_25 ≤ 1 := by rw [e_cf_25]; exact addc_carry_le_one z0_10 0 cf_24 b_z0_10 (by decide) b_cf_24
  clear e_z0_11 e_cf_25
  -- z1_11: adc {z1}, {z2}
  extract_lets -merge +onlyGivenNames s_23 z1_11 cf_26 at hr
  have e_z1_11 : z1_11 = (addc z1_10 z2_8 cf_25).1 := rfl
  have e_cf_26 : cf_26 = (addc z1_10 z2_8 cf_25).2 := rfl
  clear_value s_23 z1_11 cf_26
  have l_z1_11 : z1_11 + 2^64 * cf_26 = z1_10 + z2_8 + cf_25 := by
    rw [e_z1_11, e_cf_26]; exact addc_lin z1_10 z2_8 cf_25
  have b_z1_11 : z1_11 < 2^64 := by rw [e_z1_11]; exact addc_value_lt z1_10 z2_8 cf_25
  have b_cf_26 : cf_26 ≤ 1 := by rw [e_cf_26]; exact addc_carry_le_one z1_10 z2_8 cf_25 b_z1_10 (lt_of_lt_of_le b_z2_8 (by norm_num)) b_cf_25
  clear e_z1_11 e_cf_26
  -- BEGIN squareHi step 2
  have hc_2 : z2_4 + z2_6 = 2^64 * cf_18 := by
    have hq : rdx_5 = mulLo inv' z2_4 := by
      simp only [mulLo, e_rdx_5, e_rdx_4, Nat.mul_comm]
    have hc := neg_carry_cancel z2_4 inv' modulus.l0 rdx_5 b_z2_4 b_inv' hm.1
      (by simpa only [e_inv'] using hinv) hq
    rw [e_z2_6]
    simp only [mulx]
    rw [e_cf_18]
    simpa only [neg] using hc
  have hk_2 : cf_22 = 0 ∧ cf_26 = 0 := by
    clear * - l_z1_10 e_z1_9 b_cf_21 l_z1_11 b_z2_8 b_cf_25
    omega
  have I_2 : 2^64 * (z3_6 + 2^64 * a_8 + 2^128 * z0_11 + 2^192 * z1_11) =
      (z2_4 + 2^64 * z3_4 + 2^128 * a_6 + 2^192 * z0_9) +
        (rdx_5 * modulus.l0 + 2^64 * (rdx_5 * modulus.l1) + rdx_5 * 2^254) :=
    squareHi_step hc_2 d_t1_5 d_t2_2 (by rw [e_z2_8, e_z2_7]; simpa only [e_z1_7] using sh_z1_8)
      l_z3_5 l_a_7 l_z0_10 (by simpa only [e_z1_9] using l_z1_10)
      l_z3_6 l_a_8 l_z0_11 l_z1_11 hk_2
  -- END squareHi step 2
  -- rdx_6: mov rdx, {z3}
  extract_lets -merge +onlyGivenNames rdx_6 at hr
  have e_rdx_6 : rdx_6 = z3_6 := rfl
  clear_value rdx_6
  have b_rdx_6 : rdx_6 < 2^64 := by rw [e_rdx_6]; exact b_z3_6
  -- rdx_7: imul rdx, {inv}
  extract_lets -merge +onlyGivenNames rdx_7 at hr
  have e_rdx_7 : rdx_7 = rdx_6 * inv' % 2^64 := rfl
  clear_value rdx_7
  have b_rdx_7 : rdx_7 < 2^64 := by rw [e_rdx_7]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- m_6: mulx {t2}, {t1}, qword ptr [{p} + 8]
  extract_lets -merge +onlyGivenNames m_6 t2_3 t1_6 at hr
  have e_t2_3 : t2_3 = (mulx rdx_7 modulus.l1).1 := rfl
  have e_t1_6 : t1_6 = (mulx rdx_7 modulus.l1).2 := rfl
  clear_value m_6 t2_3 t1_6
  have b_t2_3 : t2_3 < 2^64 := by rw [e_t2_3]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_7 hm.2.1)
  have b_t1_6 : t1_6 < 2^64 := by rw [e_t1_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t2_3 : t1_6 + 2^64 * t2_3 = rdx_7 * modulus.l1 := by
    rw [e_t1_6, e_t2_3]; exact Nat.mod_add_div _ _
  -- z2_9: mov {z2}, rdx
  extract_lets -merge +onlyGivenNames z2_9 at hr
  have e_z2_9 : z2_9 = rdx_7 := rfl
  clear_value z2_9
  have b_z2_9 : z2_9 < 2^64 := by rw [e_z2_9]; exact b_rdx_7
  -- z2_10: shl {z2}, 62
  extract_lets -merge +onlyGivenNames z2_10 at hr
  have e_z2_10 : z2_10 = z2_9 * 2^62 % 2^64 := rfl
  clear_value z2_10
  have b_z2_10 : z2_10 < 2^64 := by rw [e_z2_10]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_z2_10 : z2_10 + 2^64 * (z2_9 / 2^2) = z2_9 * 2^62 := by
    rw [e_z2_10]; exact lsl62_lsr2_split _
  -- n_3: neg {z3}
  extract_lets -merge +onlyGivenNames n_3 z3_7 cf_27 at hr
  have e_z3_7 : z3_7 = (neg z3_6).1 := rfl
  have e_cf_27 : cf_27 = (neg z3_6).2 := rfl
  clear_value n_3 z3_7 cf_27
  have b_z3_7 : z3_7 < 2^64 := by rw [e_z3_7]; exact sbb_value_lt 0 z3_6 0
  have b_cf_27 : cf_27 ≤ 1 := by
    rw [e_cf_27]; simp only [neg]; split <;> omega
  -- a_9: adc {a}, {t1}
  extract_lets -merge +onlyGivenNames s_24 a_9 cf_28 at hr
  have e_a_9 : a_9 = (addc a_8 t1_6 cf_27).1 := rfl
  have e_cf_28 : cf_28 = (addc a_8 t1_6 cf_27).2 := rfl
  clear_value s_24 a_9 cf_28
  have l_a_9 : a_9 + 2^64 * cf_28 = a_8 + t1_6 + cf_27 := by
    rw [e_a_9, e_cf_28]; exact addc_lin a_8 t1_6 cf_27
  have b_a_9 : a_9 < 2^64 := by rw [e_a_9]; exact addc_value_lt a_8 t1_6 cf_27
  have b_cf_28 : cf_28 ≤ 1 := by rw [e_cf_28]; exact addc_carry_le_one a_8 t1_6 cf_27 b_a_8 b_t1_6 b_cf_27
  clear e_a_9 e_cf_28
  -- z0_12: adc {z0}, 0
  extract_lets -merge +onlyGivenNames s_25 z0_12 cf_29 at hr
  have e_z0_12 : z0_12 = (addc z0_11 0 cf_28).1 := rfl
  have e_cf_29 : cf_29 = (addc z0_11 0 cf_28).2 := rfl
  clear_value s_25 z0_12 cf_29
  have l_z0_12 : z0_12 + 2^64 * cf_29 = z0_11 + 0 + cf_28 := by
    rw [e_z0_12, e_cf_29]; exact addc_lin z0_11 0 cf_28
  have b_z0_12 : z0_12 < 2^64 := by rw [e_z0_12]; exact addc_value_lt z0_11 0 cf_28
  have b_cf_29 : cf_29 ≤ 1 := by rw [e_cf_29]; exact addc_carry_le_one z0_11 0 cf_28 b_z0_11 (by decide) b_cf_28
  clear e_z0_12 e_cf_29
  -- z1_12: adc {z1}, {z2}
  extract_lets -merge +onlyGivenNames s_26 z1_12 cf_30 at hr
  have e_z1_12 : z1_12 = (addc z1_11 z2_10 cf_29).1 := rfl
  have e_cf_30 : cf_30 = (addc z1_11 z2_10 cf_29).2 := rfl
  clear_value s_26 z1_12 cf_30
  have l_z1_12 : z1_12 + 2^64 * cf_30 = z1_11 + z2_10 + cf_29 := by
    rw [e_z1_12, e_cf_30]; exact addc_lin z1_11 z2_10 cf_29
  have b_z1_12 : z1_12 < 2^64 := by rw [e_z1_12]; exact addc_value_lt z1_11 z2_10 cf_29
  have b_cf_30 : cf_30 ≤ 1 := by rw [e_cf_30]; exact addc_carry_le_one z1_11 z2_10 cf_29 b_z1_11 b_z2_10 b_cf_29
  clear e_z1_12 e_cf_30
  -- z2_11: mov {z2}, 0
  extract_lets -merge +onlyGivenNames z2_11 at hr
  have e_z2_11 : z2_11 = 0 := rfl
  clear_value z2_11
  have b_z2_11 : z2_11 < 2^64 := by rw [e_z2_11]; decide
  -- z2_12: adc {z2}, 0
  extract_lets -merge +onlyGivenNames s_27 z2_12 cf_31 at hr
  have e_z2_12 : z2_12 = (addc z2_11 0 cf_30).1 := rfl
  have e_cf_31 : cf_31 = (addc z2_11 0 cf_30).2 := rfl
  clear_value s_27 z2_12 cf_31
  have l_z2_12 : z2_12 + 2^64 * cf_31 = z2_11 + 0 + cf_30 := by
    rw [e_z2_12, e_cf_31]; exact addc_lin z2_11 0 cf_30
  have b_z2_12 : z2_12 < 2^64 := by rw [e_z2_12]; exact addc_value_lt z2_11 0 cf_30
  have b_cf_31 : cf_31 ≤ 1 := by rw [e_cf_31]; exact addc_carry_le_one z2_11 0 cf_30 b_z2_11 (by decide) b_cf_30
  clear e_z2_12 e_cf_31
  -- m_7: mulx {t1}, {z3}, qword ptr [{p}]
  extract_lets -merge +onlyGivenNames m_7 t1_7 z3_8 at hr
  have e_t1_7 : t1_7 = (mulx rdx_7 modulus.l0).1 := rfl
  have e_z3_8 : z3_8 = (mulx rdx_7 modulus.l0).2 := rfl
  clear_value m_7 t1_7 z3_8
  have b_t1_7 : t1_7 < 2^64 := by rw [e_t1_7]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_7 hm.1)
  have b_z3_8 : z3_8 < 2^64 := by rw [e_z3_8]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t1_7 : z3_8 + 2^64 * t1_7 = rdx_7 * modulus.l0 := by
    rw [e_z3_8, e_t1_7]; exact Nat.mod_add_div _ _
  -- z3_9: mov {z3}, rdx
  extract_lets -merge +onlyGivenNames z3_9 at hr
  have e_z3_9 : z3_9 = rdx_7 := rfl
  clear_value z3_9
  have b_z3_9 : z3_9 < 2^64 := by rw [e_z3_9]; exact b_rdx_7
  -- z3_10: shr {z3}, 2
  extract_lets -merge +onlyGivenNames z3_10 at hr
  have e_z3_10 : z3_10 = z3_9 / 2^2 := rfl
  clear_value z3_10
  have b_z3_10 : z3_10 < 2^62 := by
    rw [e_z3_10]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_z3_9 (by norm_num))
  -- a_10: add {a}, {t1}
  extract_lets -merge +onlyGivenNames s_28 a_10 cf_32 at hr
  have e_a_10 : a_10 = (addc a_9 t1_7 0).1 := rfl
  have e_cf_32 : cf_32 = (addc a_9 t1_7 0).2 := rfl
  clear_value s_28 a_10 cf_32
  have l_a_10 : a_10 + 2^64 * cf_32 = a_9 + t1_7 + 0 := by
    rw [e_a_10, e_cf_32]; exact addc_lin a_9 t1_7 0
  have b_a_10 : a_10 < 2^64 := by rw [e_a_10]; exact addc_value_lt a_9 t1_7 0
  have b_cf_32 : cf_32 ≤ 1 := by rw [e_cf_32]; exact addc_carry_le_one a_9 t1_7 0 b_a_9 b_t1_7 (by decide)
  clear e_a_10 e_cf_32
  -- z0_13: adc {z0}, {t2}
  extract_lets -merge +onlyGivenNames s_29 z0_13 cf_33 at hr
  have e_z0_13 : z0_13 = (addc z0_12 t2_3 cf_32).1 := rfl
  have e_cf_33 : cf_33 = (addc z0_12 t2_3 cf_32).2 := rfl
  clear_value s_29 z0_13 cf_33
  have l_z0_13 : z0_13 + 2^64 * cf_33 = z0_12 + t2_3 + cf_32 := by
    rw [e_z0_13, e_cf_33]; exact addc_lin z0_12 t2_3 cf_32
  have b_z0_13 : z0_13 < 2^64 := by rw [e_z0_13]; exact addc_value_lt z0_12 t2_3 cf_32
  have b_cf_33 : cf_33 ≤ 1 := by rw [e_cf_33]; exact addc_carry_le_one z0_12 t2_3 cf_32 b_z0_12 b_t2_3 b_cf_32
  clear e_z0_13 e_cf_33
  -- z1_13: adc {z1}, 0
  extract_lets -merge +onlyGivenNames s_30 z1_13 cf_34 at hr
  have e_z1_13 : z1_13 = (addc z1_12 0 cf_33).1 := rfl
  have e_cf_34 : cf_34 = (addc z1_12 0 cf_33).2 := rfl
  clear_value s_30 z1_13 cf_34
  have l_z1_13 : z1_13 + 2^64 * cf_34 = z1_12 + 0 + cf_33 := by
    rw [e_z1_13, e_cf_34]; exact addc_lin z1_12 0 cf_33
  have b_z1_13 : z1_13 < 2^64 := by rw [e_z1_13]; exact addc_value_lt z1_12 0 cf_33
  have b_cf_34 : cf_34 ≤ 1 := by rw [e_cf_34]; exact addc_carry_le_one z1_12 0 cf_33 b_z1_12 (by decide) b_cf_33
  clear e_z1_13 e_cf_34
  -- z2_13: adc {z2}, {z3}
  extract_lets -merge +onlyGivenNames s_31 z2_13 cf_35 at hr
  have e_z2_13 : z2_13 = (addc z2_12 z3_10 cf_34).1 := rfl
  have e_cf_35 : cf_35 = (addc z2_12 z3_10 cf_34).2 := rfl
  clear_value s_31 z2_13 cf_35
  have l_z2_13 : z2_13 + 2^64 * cf_35 = z2_12 + z3_10 + cf_34 := by
    rw [e_z2_13, e_cf_35]; exact addc_lin z2_12 z3_10 cf_34
  have b_z2_13 : z2_13 < 2^64 := by rw [e_z2_13]; exact addc_value_lt z2_12 z3_10 cf_34
  have b_cf_35 : cf_35 ≤ 1 := by rw [e_cf_35]; exact addc_carry_le_one z2_12 z3_10 cf_34 b_z2_12 (lt_of_lt_of_le b_z3_10 (by norm_num)) b_cf_34
  clear e_z2_13 e_cf_35
  -- BEGIN squareHi step 3 and reduction
  have hc_3 : z3_6 + z3_8 = 2^64 * cf_27 := by
    have hq : rdx_7 = mulLo inv' z3_6 := by
      simp only [mulLo, e_rdx_7, e_rdx_6, Nat.mul_comm]
    have hc := neg_carry_cancel z3_6 inv' modulus.l0 rdx_7 b_z3_6 b_inv' hm.1
      (by simpa only [e_inv'] using hinv) hq
    rw [e_z3_8]
    simp only [mulx]
    rw [e_cf_27]
    simpa only [neg] using hc
  have hk_3 : cf_31 = 0 ∧ cf_35 = 0 := by
    clear * - l_z2_12 e_z2_11 b_cf_30 l_z2_13 b_z3_10 b_cf_34
    omega
  have I_3 : 2^64 * (a_10 + 2^64 * z0_13 + 2^128 * z1_13 + 2^192 * z2_13) =
      (z3_6 + 2^64 * a_8 + 2^128 * z0_11 + 2^192 * z1_11) +
        (rdx_7 * modulus.l0 + 2^64 * (rdx_7 * modulus.l1) + rdx_7 * 2^254) :=
    squareHi_step hc_3 d_t1_7 d_t2_3 (by rw [e_z3_10, e_z3_9]; simpa only [e_z2_9] using sh_z2_10)
      l_a_9 l_z0_12 l_z1_12 (by simpa only [e_z2_11] using l_z2_12)
      l_a_10 l_z0_13 l_z1_13 l_z2_13 hk_3
  have hQ : rdx_1 + 2^64 * rdx_3 + 2^128 * rdx_5 + 2^192 * rdx_7 < 2^256 := by
    clear * - b_rdx_1 b_rdx_3 b_rdx_5 b_rdx_7
    omega
  have hQp : (rdx_1 + 2^64 * rdx_3 + 2^128 * rdx_5 + 2^192 * rdx_7) * modulus.toNat =
      (rdx_1 * modulus.l0 + 2^64 * (rdx_1 * modulus.l1) + rdx_1 * 2^254) +
      2^64 * (rdx_3 * modulus.l0 + 2^64 * (rdx_3 * modulus.l1) + rdx_3 * 2^254) +
      2^128 * (rdx_5 * modulus.l0 + 2^64 * (rdx_5 * modulus.l1) + rdx_5 * 2^254) +
      2^192 * (rdx_7 * modulus.l0 + 2^64 * (rdx_7 * modulus.l1) + rdx_7 * 2^254) := by
    simp only [Limbs.toNat, hshape.1, hshape.2]
    ring
  have hR : 2^256 * (a_10 + 2^64 * z0_13 + 2^128 * z1_13 + 2^192 * z2_13) =
      (z0 + 2^64 * z1 + 2^128 * z2 + 2^192 * z3) +
      (rdx_1 + 2^64 * rdx_3 + 2^128 * rdx_5 + 2^192 * rdx_7) * modulus.toNat := by
    clear * - I_0 I_1 I_2 I_3 hQp
    omega
  -- END squareHi step 3 and reduction
  -- a_11: add {a}, {z4}
  extract_lets -merge +onlyGivenNames s_32 a_11 cf_36 at hr
  have e_a_11 : a_11 = (addc a_10 z4 0).1 := rfl
  have e_cf_36 : cf_36 = (addc a_10 z4 0).2 := rfl
  clear_value s_32 a_11 cf_36
  have l_a_11 : a_11 + 2^64 * cf_36 = a_10 + z4 + 0 := by
    rw [e_a_11, e_cf_36]; exact addc_lin a_10 z4 0
  have b_a_11 : a_11 < 2^64 := by rw [e_a_11]; exact addc_value_lt a_10 z4 0
  have b_cf_36 : cf_36 ≤ 1 := by rw [e_cf_36]; exact addc_carry_le_one a_10 z4 0 b_a_10 b_z4 (by decide)
  clear e_a_11 e_cf_36
  -- z0_14: adc {z0}, {z5}
  extract_lets -merge +onlyGivenNames s_33 z0_14 cf_37 at hr
  have e_z0_14 : z0_14 = (addc z0_13 z5 cf_36).1 := rfl
  have e_cf_37 : cf_37 = (addc z0_13 z5 cf_36).2 := rfl
  clear_value s_33 z0_14 cf_37
  have l_z0_14 : z0_14 + 2^64 * cf_37 = z0_13 + z5 + cf_36 := by
    rw [e_z0_14, e_cf_37]; exact addc_lin z0_13 z5 cf_36
  have b_z0_14 : z0_14 < 2^64 := by rw [e_z0_14]; exact addc_value_lt z0_13 z5 cf_36
  have b_cf_37 : cf_37 ≤ 1 := by rw [e_cf_37]; exact addc_carry_le_one z0_13 z5 cf_36 b_z0_13 b_z5 b_cf_36
  clear e_z0_14 e_cf_37
  -- z1_14: adc {z1}, {z6}
  extract_lets -merge +onlyGivenNames s_34 z1_14 cf_38 at hr
  have e_z1_14 : z1_14 = (addc z1_13 z6 cf_37).1 := rfl
  have e_cf_38 : cf_38 = (addc z1_13 z6 cf_37).2 := rfl
  clear_value s_34 z1_14 cf_38
  have l_z1_14 : z1_14 + 2^64 * cf_38 = z1_13 + z6 + cf_37 := by
    rw [e_z1_14, e_cf_38]; exact addc_lin z1_13 z6 cf_37
  have b_z1_14 : z1_14 < 2^64 := by rw [e_z1_14]; exact addc_value_lt z1_13 z6 cf_37
  have b_cf_38 : cf_38 ≤ 1 := by rw [e_cf_38]; exact addc_carry_le_one z1_13 z6 cf_37 b_z1_13 b_z6 b_cf_37
  clear e_z1_14 e_cf_38
  -- z2_14: adc {z2}, {z7}
  extract_lets -merge +onlyGivenNames s_35 z2_14 cf_39 at hr
  have e_z2_14 : z2_14 = (addc z2_13 z7 cf_38).1 := rfl
  have e_cf_39 : cf_39 = (addc z2_13 z7 cf_38).2 := rfl
  clear_value s_35 z2_14 cf_39
  have l_z2_14 : z2_14 + 2^64 * cf_39 = z2_13 + z7 + cf_38 := by
    rw [e_z2_14, e_cf_39]; exact addc_lin z2_13 z7 cf_38
  have b_z2_14 : z2_14 < 2^64 := by rw [e_z2_14]; exact addc_value_lt z2_13 z7 cf_38
  have b_cf_39 : cf_39 ≤ 1 := by rw [e_cf_39]; exact addc_carry_le_one z2_13 z7 cf_38 b_z2_13 b_z7 b_cf_38
  clear e_z2_14 e_cf_39
  -- BEGIN squareHi candidate
  have hC : a_11 + 2^64 * z0_14 + 2^128 * z1_14 + 2^192 * z2_14 + 2^256 * cf_39 =
      (a_10 + 2^64 * z0_13 + 2^128 * z1_13 + 2^192 * z2_13) +
      (z4 + 2^64 * z5 + 2^128 * z6 + 2^192 * z7) := by
    clear * - l_a_11 l_z0_14 l_z1_14 l_z2_14
    omega
  have hT : product.toNat =
      (z0 + 2^64 * z1 + 2^128 * z2 + 2^192 * z3) +
      2^256 * (z4 + 2^64 * z5 + 2^128 * z6 + 2^192 * z7) := by
    simp only [WideLimbs.toNat, e_z0, e_z1, e_z2, e_z3, e_z4, e_z5, e_z6, e_z7]
    ring
  have hmain : 2^256 *
      (a_11 + 2^64 * z0_14 + 2^128 * z1_14 + 2^192 * z2_14 + 2^256 * cf_39) =
      product.toNat +
      (rdx_1 + 2^64 * rdx_3 + 2^128 * rdx_5 + 2^192 * rdx_7) * modulus.toNat := by
    clear * - hC hR hT
    omega
  have hP := Limbs.toNat_lt_of_shape modulus hm hshape
  have hQPle :
      (rdx_1 + 2^64 * rdx_3 + 2^128 * rdx_5 + 2^192 * rdx_7) * modulus.toNat +
        modulus.toNat ≤ 2^256 * modulus.toNat := by
    rw [← Nat.succ_mul]
    exact Nat.mul_le_mul_right _ hQ
  have hA : a_11 + 2^64 * z0_14 + 2^128 * z1_14 + 2^192 * z2_14 + 2^256 * cf_39 <
      2 * modulus.toNat := by
    clear * - hmain hproduct_lt hQPle
    omega
  have hk : cf_39 = 0 := by
    clear * - hA hP
    omega
  -- END squareHi candidate
  -- rdx_8: movabs rdx, {p3}
  extract_lets -merge +onlyGivenNames rdx_8 at hr
  have e_rdx_8 : rdx_8 = p3 := rfl
  clear_value rdx_8
  have b_rdx_8 : rdx_8 < 2^64 := by rw [e_rdx_8]; exact b_p3
  -- t1_8: mov {t1}, {a}
  extract_lets -merge +onlyGivenNames t1_8 at hr
  have e_t1_8 : t1_8 = a_11 := rfl
  clear_value t1_8
  have b_t1_8 : t1_8 < 2^64 := by rw [e_t1_8]; exact b_a_11
  -- t2_4: mov {t2}, {z0}
  extract_lets -merge +onlyGivenNames t2_4 at hr
  have e_t2_4 : t2_4 = z0_14 := rfl
  clear_value t2_4
  have b_t2_4 : t2_4 < 2^64 := by rw [e_t2_4]; exact b_z0_14
  -- z3_11: mov {z3}, {z1}
  extract_lets -merge +onlyGivenNames z3_11 at hr
  have e_z3_11 : z3_11 = z1_14 := rfl
  clear_value z3_11
  have b_z3_11 : z3_11 < 2^64 := by rw [e_z3_11]; exact b_z1_14
  -- z4_1: mov {z4}, {z2}
  extract_lets -merge +onlyGivenNames z4_1 at hr
  have e_z4_1 : z4_1 = z2_14 := rfl
  clear_value z4_1
  have b_z4_1 : z4_1 < 2^64 := by rw [e_z4_1]; exact b_z2_14
  -- t1_9: sub {t1}, qword ptr [{p}]
  extract_lets -merge +onlyGivenNames d t1_9 cf_40 at hr
  have e_t1_9 : t1_9 = (sbb t1_8 modulus.l0 0).1 := rfl
  have e_cf_40 : cf_40 = (sbb t1_8 modulus.l0 0).2 := rfl
  clear_value d t1_9 cf_40
  have l_t1_9 : t1_9 + modulus.l0 + 0 = t1_8 + 2^64 * cf_40 := by
    rw [e_t1_9, e_cf_40]; exact sbb_lin t1_8 modulus.l0 0 b_t1_8 hm.1 (by decide)
  have b_t1_9 : t1_9 < 2^64 := by rw [e_t1_9]; exact sbb_value_lt t1_8 modulus.l0 0
  have b_cf_40 : cf_40 ≤ 1 := by rw [e_cf_40]; exact sbb_borrow_le_one t1_8 modulus.l0 0
  clear e_t1_9 e_cf_40
  -- t2_5: sbb {t2}, qword ptr [{p} + 8]
  extract_lets -merge +onlyGivenNames d_1 t2_5 cf_41 at hr
  have e_t2_5 : t2_5 = (sbb t2_4 modulus.l1 cf_40).1 := rfl
  have e_cf_41 : cf_41 = (sbb t2_4 modulus.l1 cf_40).2 := rfl
  clear_value d_1 t2_5 cf_41
  have l_t2_5 : t2_5 + modulus.l1 + cf_40 = t2_4 + 2^64 * cf_41 := by
    rw [e_t2_5, e_cf_41]; exact sbb_lin t2_4 modulus.l1 cf_40 b_t2_4 hm.2.1 b_cf_40
  have b_t2_5 : t2_5 < 2^64 := by rw [e_t2_5]; exact sbb_value_lt t2_4 modulus.l1 cf_40
  have b_cf_41 : cf_41 ≤ 1 := by rw [e_cf_41]; exact sbb_borrow_le_one t2_4 modulus.l1 cf_40
  clear e_t2_5 e_cf_41
  -- z3_12: sbb {z3}, 0
  extract_lets -merge +onlyGivenNames d_2 z3_12 cf_42 at hr
  have e_z3_12 : z3_12 = (sbb z3_11 0 cf_41).1 := rfl
  have e_cf_42 : cf_42 = (sbb z3_11 0 cf_41).2 := rfl
  clear_value d_2 z3_12 cf_42
  have l_z3_12 : z3_12 + 0 + cf_41 = z3_11 + 2^64 * cf_42 := by
    rw [e_z3_12, e_cf_42]; exact sbb_lin z3_11 0 cf_41 b_z3_11 (by decide) b_cf_41
  have b_z3_12 : z3_12 < 2^64 := by rw [e_z3_12]; exact sbb_value_lt z3_11 0 cf_41
  have b_cf_42 : cf_42 ≤ 1 := by rw [e_cf_42]; exact sbb_borrow_le_one z3_11 0 cf_41
  clear e_z3_12 e_cf_42
  -- z4_2: sbb {z4}, rdx
  extract_lets -merge +onlyGivenNames d_3 z4_2 cf_43 at hr
  have e_z4_2 : z4_2 = (sbb z4_1 rdx_8 cf_42).1 := rfl
  have e_cf_43 : cf_43 = (sbb z4_1 rdx_8 cf_42).2 := rfl
  clear_value d_3 z4_2 cf_43
  have l_z4_2 : z4_2 + rdx_8 + cf_42 = z4_1 + 2^64 * cf_43 := by
    rw [e_z4_2, e_cf_43]; exact sbb_lin z4_1 rdx_8 cf_42 b_z4_1 b_rdx_8 b_cf_42
  have b_z4_2 : z4_2 < 2^64 := by rw [e_z4_2]; exact sbb_value_lt z4_1 rdx_8 cf_42
  have b_cf_43 : cf_43 ≤ 1 := by rw [e_cf_43]; exact sbb_borrow_le_one z4_1 rdx_8 cf_42
  clear e_z4_2 e_cf_43
  -- a_12: cmovnc {a}, {t1}
  extract_lets -merge +onlyGivenNames a_12 at hr
  have e_a_12 : a_12 = (if cf_43 = 0 then t1_9 else a_11) := rfl
  clear_value a_12
  have b_a_12 : a_12 < 2^64 := by
    rw [e_a_12]; split <;> first | exact b_t1_9 | exact b_a_11
  -- z0_15: cmovnc {z0}, {t2}
  extract_lets -merge +onlyGivenNames z0_15 at hr
  have e_z0_15 : z0_15 = (if cf_43 = 0 then t2_5 else z0_14) := rfl
  clear_value z0_15
  have b_z0_15 : z0_15 < 2^64 := by
    rw [e_z0_15]; split <;> first | exact b_t2_5 | exact b_z0_14
  -- z1_15: cmovnc {z1}, {z3}
  extract_lets -merge +onlyGivenNames z1_15 at hr
  have e_z1_15 : z1_15 = (if cf_43 = 0 then z3_12 else z1_14) := rfl
  clear_value z1_15
  have b_z1_15 : z1_15 < 2^64 := by
    rw [e_z1_15]; split <;> first | exact b_z3_12 | exact b_z1_14
  -- z2_15: cmovnc {z2}, {z4}
  extract_lets -merge +onlyGivenNames z2_15 at hr
  have e_z2_15 : z2_15 = (if cf_43 = 0 then z4_2 else z2_14) := rfl
  clear_value z2_15
  have b_z2_15 : z2_15 < 2^64 := by
    rw [e_z2_15]; split <;> first | exact b_z4_2 | exact b_z2_14
  subst hr
  -- BEGIN squareHi conclusion
  rw [e_t1_8] at l_t1_9
  rw [e_t2_4] at l_t2_5
  rw [e_z3_11] at l_z3_12
  rw [e_z4_1, e_rdx_8] at l_z4_2
  exact squareHi_conclude hmain hA hk l_t1_9 l_t2_5 l_z3_12 l_z4_2 e_p3 hshape b_cf_43
    b_t1_9 b_t2_5 b_z3_12 b_z4_2 b_a_12 b_z0_15 b_z1_15 b_z2_15
    e_a_12 e_z0_15 e_z1_15 e_z2_15
  -- END squareHi conclusion

-- BEGIN sqrMont_spec statement
/-- Montgomery squaring by the two inline blocks: for a canonical `value`, the result is below `p`
and `2^256 * result ≡ value * value (mod p)`. `squareLo` forms the eight-limb square exactly;
`squareHi` reduces its low half by four Montgomery cancellation steps, adds the high half, and
reduces once conditionally. The candidate is below `2 * p < 2^256`, so the carry dropped by the
high-half addition is `0`. -/
theorem sqrMont_spec (value modulus : Limbs) (inv : Nat) (hv : value.Bounded)
    (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinv_lt : inv < 2^64) (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hlt : value.toNat < modulus.toNat) :
    ∀ r, r = sqrMont value modulus inv →
      r.Bounded ∧ r.toNat < modulus.toNat ∧
        2^256 * r.toNat ≡ value.toNat * value.toNat [MOD modulus.toNat] := by
  intro r hr
  obtain ⟨hproduct, eproduct⟩ := squareLo_spec value hv
  have hvalue := Limbs.toNat_lt value hv
  have hproduct_lt : (squareLo value).toNat < 2^256 * modulus.toNat := by
    rw [eproduct]
    exact Nat.mul_lt_mul'' hvalue hlt
  obtain ⟨hb, hl, hc⟩ := squareHi_spec (squareLo value) modulus inv hproduct hm hshape
    hinv_lt hinv hproduct_lt r (by simpa only [sqrMont] using hr)
  exact ⟨hb, hl, by simpa only [eproduct] using hc⟩
-- END sqrMont_spec statement
-- BEGIN sqrMont_spec corollaries
/-- The squaring pair applied `count` times to a canonical `value`: the output remains canonical,
and the Montgomery weight records one factor of `2^-256` per squaring. -/
theorem sqrN_spec (value modulus : Limbs) (inv : Nat) (hv : value.Bounded)
    (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinv_lt : inv < 2^64) (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hlt : value.toNat < modulus.toNat) (count : Nat) :
    ∀ r, r = sqrN value modulus inv count →
      r.Bounded ∧ r.toNat < modulus.toNat ∧
        2^(256 * (2^count - 1)) * r.toNat ≡ value.toNat^(2^count) [MOD modulus.toNat] := by
  induction count with
  | zero =>
    intro r hr
    subst hr
    simp only [pow_zero, Nat.sub_self, mul_zero, one_mul, pow_one]
    exact ⟨hv, hlt, Nat.ModEq.refl _⟩
  | succ n ih =>
    intro r hr
    obtain ⟨hb, hl, hc⟩ := ih _ rfl
    obtain ⟨hb', hl', hc'⟩ := sqrMont_spec (sqrN value modulus inv n) modulus inv hb hm hshape
      hinv_lt hinv hl r (hr.trans rfl)
    refine ⟨hb', hl', ?_⟩
    have hpos := Nat.two_pow_pos n
    have e1 : 2^(256 * (2^(n+1) - 1)) =
        2^(256 * (2^n - 1)) * 2^(256 * (2^n - 1)) * 2^256 := by
      rw [← pow_add, ← pow_add, pow_succ]
      congr 1
      generalize 2^n = m at hpos ⊢
      omega
    have e2 : value.toNat^(2^(n+1)) = value.toNat^(2^n) * value.toNat^(2^n) := by
      rw [pow_succ, pow_mul, sq]
    rw [e1, e2]
    calc 2^(256 * (2^n - 1)) * 2^(256 * (2^n - 1)) * 2^256 * r.toNat
        = 2^(256 * (2^n - 1)) * 2^(256 * (2^n - 1)) * (2^256 * r.toNat) := by ring
      _ ≡ 2^(256 * (2^n - 1)) * 2^(256 * (2^n - 1)) *
            ((sqrN value modulus inv n).toNat * (sqrN value modulus inv n).toNat)
            [MOD modulus.toNat] := Nat.ModEq.mul_left _ hc'
      _ = (2^(256 * (2^n - 1)) * (sqrN value modulus inv n).toNat) *
            (2^(256 * (2^n - 1)) * (sqrN value modulus inv n).toNat) := by ring
      _ ≡ value.toNat^(2^n) * value.toNat^(2^n) [MOD modulus.toNat] := Nat.ModEq.mul hc hc
-- END sqrMont_spec corollaries

end PastaAsm.X86_64
