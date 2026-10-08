/-
Copyright Amazon.com, Inc. or its affiliates (the block, adapted from s2n-bignum).
Copyright (c) 2026 the pasta_curves contributors (the transcription and the proofs).
-/
import PastaCurves.AArch64.Spec.SignMag
import PastaCurves.Inversion.Round

/-!
# Correctness of the inversion's `d`, `e` row block

See the parent module's documentation for details. The block forms `a d + b e` as a five-word signed
value from the sign-magnitude forms of `a` and `b`: each word of `d` is complemented by the mask,
the products are accumulated, and the two's-complement corrections (`|a|` in the low word, `-|a|` in
the top word for a negative `a`, likewise for `b`) turn the products of complements into the signed
products. The row bound `|a| + |b| ≤ 2^63` keeps every column's carry within the next word.
-/

set_option exponentiation.threshold 400

namespace PastaCurves.AArch64

open Inversion (SignMagRep row_side)

-- BEGIN deRowBlock_spec statement
/-- The row `a d + b e` of `updateDE` before its `amontred`, as the exact integer in five words.
The row bound `|a| + |b| ≤ 2^63` is what keeps every column's carry within the next word; the
rows of a 59-step matrix are at most `2^59`. `row_side` supplies the sign handling. -/
theorem deRowBlock_spec (a b : ℤ) (d e : Limbs) (m0 m1 s0 s1 : Nat)
    (hd : d.Bounded) (he : e.Bounded) (hab : |a| + |b| ≤ 2^63)
    (hrep0 : SignMagRep m0 s0 a) (hrep1 : SignMagRep m1 s1 b) :
    ∀ res, res = deRowBlock d e m0 m1 s0 s1 →
      res.Bounded ∧ res.toInt = a * d.toNat + b * e.toNat := by
  intro res hres
  have ha0 := abs_nonneg a
  have hb0 := abs_nonneg b
  have hm0z : (m0 : ℤ) = |a| := hrep0.natCast_eq_abs _ _ _
  have hm1z : (m1 : ℤ) = |b| := hrep1.natCast_eq_abs _ _ _
  have hm0 : m0 < 2^64 := by omega
  have hm1 : m1 < 2^64 := by omega
  have hs0 : s0 < 2^64 := (hrep0.lt _ _ _ (by omega)).2
  have hs1 : s1 < 2^64 := (hrep1.lt _ _ _ (by omega)).2
-- END deRowBlock_spec statement
  -- generated skeleton for `deRowBlock`: do not edit between the annotations
  unfold deRowBlock at hres
  lift_lets -merge at hres
  -- d0: argument
  word_step d0 := d.l0 using hd.1
  -- d1: argument
  word_step d1 := d.l1 using hd.2.1
  -- d2: argument
  word_step d2 := d.l2 using hd.2.2.1
  -- d3: argument
  word_step d3 := d.l3 using hd.2.2.2
  -- e0: argument
  word_step e0 := e.l0 using he.1
  -- e1: argument
  word_step e1 := e.l1 using he.2.1
  -- e2: argument
  word_step e2 := e.l2 using he.2.2.1
  -- e3: argument
  word_step e3 := e.l3 using he.2.2.2
  -- m0': argument
  word_step m0' := m0 using hm0
  -- m1': argument
  word_step m1' := m1 using hm1
  -- s0': argument
  word_step s0' := s0 using hs0
  -- s1': argument
  word_step s1' := s1 using hs1
  -- lo: and lo,m0,s0
  word_step lo := andw m0' s0' using andw_lt m0' s0' b_m0' b_s0'
  -- w: and w,m1,s1
  word_step w := andw m1' s1' using andw_lt m1' s1' b_m1' b_s1'
  -- t0: add t0,lo,w
  word_step t0 := addw lo w using addw_lt lo w
  -- w_1: eor w,d0,s0
  word_step w_1 := eorw d0 s0' using eorw_lt d0 s0' b_d0 b_s0'
  -- lo_1: mul lo,w,m0
  word_step lo_1 := w_1 * m0' % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_2: umulh w,w,m0
  word_step w_2 := w_1 * m0' / 2^64 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_w_1 b_m0')
  have d_w_2 : lo_1 + 2^64 * w_2 = w_1 * m0' := by
    rw [e_lo_1, e_w_2]; exact Nat.mod_add_div _ _
  clear e_lo_1 e_w_2
  -- t0_1: adds t0,t0,lo
  word_step s, t0_1 := (t0 + lo_1 + 0) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _),
      c := (t0 + lo_1 + 0) / 2^64
  have l_t0_1 : t0_1 + 2^64 * c = t0 + lo_1 + 0 := by
    rw [e_t0_1, e_c]; exact Nat.mod_add_div _ _
  have b_c : c ≤ 1 := by
    rw [e_c]; exact addc_carry_le_one t0 lo_1 0 b_t0 b_lo_1 (by decide)
  clear e_t0_1 e_c
  -- t1: adc t1,xzr,w
  word_step t1 := (0 + w_2 + c) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t1, b_k_t1, l_t1⟩ :
      ∃ k, k ≤ 1 ∧ t1 + 2^64 * k = 0 + w_2 + c :=
    ⟨(0 + w_2 + c) / 2^64, addc_carry_le_one 0 w_2 c (by decide) b_w_2 b_c,
      by rw [e_t1]; exact Nat.mod_add_div _ _⟩
  clear e_t1
  -- w_3: eor w,e0,s1
  word_step w_3 := eorw e0 s1' using eorw_lt e0 s1' b_e0 b_s1'
  -- lo_2: mul lo,w,m1
  word_step lo_2 := w_3 * m1' % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_4: umulh w,w,m1
  word_step w_4 := w_3 * m1' / 2^64 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_w_3 b_m1')
  have d_w_4 : lo_2 + 2^64 * w_4 = w_3 * m1' := by
    rw [e_lo_2, e_w_4]; exact Nat.mod_add_div _ _
  clear e_lo_2 e_w_4
  -- t0_2: adds t0,t0,lo
  word_step s_1, t0_2 := (t0_1 + lo_2 + 0) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _),
      c_1 := (t0_1 + lo_2 + 0) / 2^64
  have l_t0_2 : t0_2 + 2^64 * c_1 = t0_1 + lo_2 + 0 := by
    rw [e_t0_2, e_c_1]; exact Nat.mod_add_div _ _
  have b_c_1 : c_1 ≤ 1 := by
    rw [e_c_1]; exact addc_carry_le_one t0_1 lo_2 0 b_t0_1 b_lo_2 (by decide)
  clear e_t0_2 e_c_1
  -- t1_1: adc t1,t1,w
  word_step t1_1 := (t1 + w_4 + c_1) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t1_1, b_k_t1_1, l_t1_1⟩ :
      ∃ k, k ≤ 1 ∧ t1_1 + 2^64 * k = t1 + w_4 + c_1 :=
    ⟨(t1 + w_4 + c_1) / 2^64, addc_carry_le_one t1 w_4 c_1 b_t1 b_w_4 b_c_1,
      by rw [e_t1_1]; exact Nat.mod_add_div _ _⟩
  clear e_t1_1
  -- w_5: eor w,d1,s0
  word_step w_5 := eorw d1 s0' using eorw_lt d1 s0' b_d1 b_s0'
  -- lo_3: mul lo,w,m0
  word_step lo_3 := w_5 * m0' % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_6: umulh w,w,m0
  word_step w_6 := w_5 * m0' / 2^64 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_w_5 b_m0')
  have d_w_6 : lo_3 + 2^64 * w_6 = w_5 * m0' := by
    rw [e_lo_3, e_w_6]; exact Nat.mod_add_div _ _
  clear e_lo_3 e_w_6
  -- t1_2: adds t1,t1,lo
  word_step s_2, t1_2 := (t1_1 + lo_3 + 0) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _),
      c_2 := (t1_1 + lo_3 + 0) / 2^64
  have l_t1_2 : t1_2 + 2^64 * c_2 = t1_1 + lo_3 + 0 := by
    rw [e_t1_2, e_c_2]; exact Nat.mod_add_div _ _
  have b_c_2 : c_2 ≤ 1 := by
    rw [e_c_2]; exact addc_carry_le_one t1_1 lo_3 0 b_t1_1 b_lo_3 (by decide)
  clear e_t1_2 e_c_2
  -- t2: adc t2,xzr,w
  word_step t2 := (0 + w_6 + c_2) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t2, b_k_t2, l_t2⟩ :
      ∃ k, k ≤ 1 ∧ t2 + 2^64 * k = 0 + w_6 + c_2 :=
    ⟨(0 + w_6 + c_2) / 2^64, addc_carry_le_one 0 w_6 c_2 (by decide) b_w_6 b_c_2,
      by rw [e_t2]; exact Nat.mod_add_div _ _⟩
  clear e_t2
  -- w_7: eor w,e1,s1
  word_step w_7 := eorw e1 s1' using eorw_lt e1 s1' b_e1 b_s1'
  -- lo_4: mul lo,w,m1
  word_step lo_4 := w_7 * m1' % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_8: umulh w,w,m1
  word_step w_8 := w_7 * m1' / 2^64 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_w_7 b_m1')
  have d_w_8 : lo_4 + 2^64 * w_8 = w_7 * m1' := by
    rw [e_lo_4, e_w_8]; exact Nat.mod_add_div _ _
  clear e_lo_4 e_w_8
  -- t1_3: adds t1,t1,lo
  word_step s_3, t1_3 := (t1_2 + lo_4 + 0) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _),
      c_3 := (t1_2 + lo_4 + 0) / 2^64
  have l_t1_3 : t1_3 + 2^64 * c_3 = t1_2 + lo_4 + 0 := by
    rw [e_t1_3, e_c_3]; exact Nat.mod_add_div _ _
  have b_c_3 : c_3 ≤ 1 := by
    rw [e_c_3]; exact addc_carry_le_one t1_2 lo_4 0 b_t1_2 b_lo_4 (by decide)
  clear e_t1_3 e_c_3
  -- t2_1: adc t2,t2,w
  word_step t2_1 := (t2 + w_8 + c_3) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t2_1, b_k_t2_1, l_t2_1⟩ :
      ∃ k, k ≤ 1 ∧ t2_1 + 2^64 * k = t2 + w_8 + c_3 :=
    ⟨(t2 + w_8 + c_3) / 2^64, addc_carry_le_one t2 w_8 c_3 b_t2 b_w_8 b_c_3,
      by rw [e_t2_1]; exact Nat.mod_add_div _ _⟩
  clear e_t2_1
  -- w_9: eor w,d2,s0
  word_step w_9 := eorw d2 s0' using eorw_lt d2 s0' b_d2 b_s0'
  -- lo_5: mul lo,w,m0
  word_step lo_5 := w_9 * m0' % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_10: umulh w,w,m0
  word_step w_10 := w_9 * m0' / 2^64 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_w_9 b_m0')
  have d_w_10 : lo_5 + 2^64 * w_10 = w_9 * m0' := by
    rw [e_lo_5, e_w_10]; exact Nat.mod_add_div _ _
  clear e_lo_5 e_w_10
  -- t2_2: adds t2,t2,lo
  word_step s_4, t2_2 := (t2_1 + lo_5 + 0) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _),
      c_4 := (t2_1 + lo_5 + 0) / 2^64
  have l_t2_2 : t2_2 + 2^64 * c_4 = t2_1 + lo_5 + 0 := by
    rw [e_t2_2, e_c_4]; exact Nat.mod_add_div _ _
  have b_c_4 : c_4 ≤ 1 := by
    rw [e_c_4]; exact addc_carry_le_one t2_1 lo_5 0 b_t2_1 b_lo_5 (by decide)
  clear e_t2_2 e_c_4
  -- t3: adc t3,xzr,w
  word_step t3 := (0 + w_10 + c_4) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t3, b_k_t3, l_t3⟩ :
      ∃ k, k ≤ 1 ∧ t3 + 2^64 * k = 0 + w_10 + c_4 :=
    ⟨(0 + w_10 + c_4) / 2^64, addc_carry_le_one 0 w_10 c_4 (by decide) b_w_10 b_c_4,
      by rw [e_t3]; exact Nat.mod_add_div _ _⟩
  clear e_t3
  -- w_11: eor w,e2,s1
  word_step w_11 := eorw e2 s1' using eorw_lt e2 s1' b_e2 b_s1'
  -- lo_6: mul lo,w,m1
  word_step lo_6 := w_11 * m1' % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_12: umulh w,w,m1
  word_step w_12 := w_11 * m1' / 2^64 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_w_11 b_m1')
  have d_w_12 : lo_6 + 2^64 * w_12 = w_11 * m1' := by
    rw [e_lo_6, e_w_12]; exact Nat.mod_add_div _ _
  clear e_lo_6 e_w_12
  -- t2_3: adds t2,t2,lo
  word_step s_5, t2_3 := (t2_2 + lo_6 + 0) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _),
      c_5 := (t2_2 + lo_6 + 0) / 2^64
  have l_t2_3 : t2_3 + 2^64 * c_5 = t2_2 + lo_6 + 0 := by
    rw [e_t2_3, e_c_5]; exact Nat.mod_add_div _ _
  have b_c_5 : c_5 ≤ 1 := by
    rw [e_c_5]; exact addc_carry_le_one t2_2 lo_6 0 b_t2_2 b_lo_6 (by decide)
  clear e_t2_3 e_c_5
  -- t3_1: adc t3,t3,w
  word_step t3_1 := (t3 + w_12 + c_5) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t3_1, b_k_t3_1, l_t3_1⟩ :
      ∃ k, k ≤ 1 ∧ t3_1 + 2^64 * k = t3 + w_12 + c_5 :=
    ⟨(t3 + w_12 + c_5) / 2^64, addc_carry_le_one t3 w_12 c_5 b_t3 b_w_12 b_c_5,
      by rw [e_t3_1]; exact Nat.mod_add_div _ _⟩
  clear e_t3_1
  -- w_13: eor w,d3,s0
  word_step w_13 := eorw d3 s0' using eorw_lt d3 s0' b_d3 b_s0'
  -- t4: and t4,s0,m0
  word_step t4 := andw s0' m0' using andw_lt s0' m0' b_s0' b_m0'
  -- t4_1: neg t4,t4
  word_step t4_1 := negw t4 using negw_lt t4
  -- lo_7: mul lo,w,m0
  word_step lo_7 := w_13 * m0' % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_14: umulh w,w,m0
  word_step w_14 := w_13 * m0' / 2^64 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_w_13 b_m0')
  have d_w_14 : lo_7 + 2^64 * w_14 = w_13 * m0' := by
    rw [e_lo_7, e_w_14]; exact Nat.mod_add_div _ _
  clear e_lo_7 e_w_14
  -- t3_2: adds t3,t3,lo
  word_step s_6, t3_2 := (t3_1 + lo_7 + 0) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _),
      c_6 := (t3_1 + lo_7 + 0) / 2^64
  have l_t3_2 : t3_2 + 2^64 * c_6 = t3_1 + lo_7 + 0 := by
    rw [e_t3_2, e_c_6]; exact Nat.mod_add_div _ _
  have b_c_6 : c_6 ≤ 1 := by
    rw [e_c_6]; exact addc_carry_le_one t3_1 lo_7 0 b_t3_1 b_lo_7 (by decide)
  clear e_t3_2 e_c_6
  -- t4_2: adc t4,t4,w
  word_step t4_2 := (t4_1 + w_14 + c_6) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t4_2, b_k_t4_2, l_t4_2⟩ :
      ∃ k, k ≤ 1 ∧ t4_2 + 2^64 * k = t4_1 + w_14 + c_6 :=
    ⟨(t4_1 + w_14 + c_6) / 2^64, addc_carry_le_one t4_1 w_14 c_6 b_t4_1 b_w_14 b_c_6,
      by rw [e_t4_2]; exact Nat.mod_add_div _ _⟩
  clear e_t4_2
  -- w_15: eor w,e3,s1
  word_step w_15 := eorw e3 s1' using eorw_lt e3 s1' b_e3 b_s1'
  -- lo_8: and lo,s1,m1
  word_step lo_8 := andw s1' m1' using andw_lt s1' m1' b_s1' b_m1'
  -- t4_3: sub t4,t4,lo
  word_step t4_3 := subw t4_2 lo_8 using subw_lt t4_2 lo_8
  -- lo_9: mul lo,w,m1
  word_step lo_9 := w_15 * m1' % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_16: umulh w,w,m1
  word_step w_16 := w_15 * m1' / 2^64 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_w_15 b_m1')
  have d_w_16 : lo_9 + 2^64 * w_16 = w_15 * m1' := by
    rw [e_lo_9, e_w_16]; exact Nat.mod_add_div _ _
  clear e_lo_9 e_w_16
  -- t3_3: adds t3,t3,lo
  word_step s_7, t3_3 := (t3_2 + lo_9 + 0) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _),
      c_7 := (t3_2 + lo_9 + 0) / 2^64
  have l_t3_3 : t3_3 + 2^64 * c_7 = t3_2 + lo_9 + 0 := by
    rw [e_t3_3, e_c_7]; exact Nat.mod_add_div _ _
  have b_c_7 : c_7 ≤ 1 := by
    rw [e_c_7]; exact addc_carry_le_one t3_2 lo_9 0 b_t3_2 b_lo_9 (by decide)
  clear e_t3_3 e_c_7
  -- t4_4: adc t4,t4,w
  word_step t4_4 := (t4_3 + w_16 + c_7) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t4_4, b_k_t4_4, l_t4_4⟩ :
      ∃ k, k ≤ 1 ∧ t4_4 + 2^64 * k = t4_3 + w_16 + c_7 :=
    ⟨(t4_3 + w_16 + c_7) / 2^64, addc_carry_le_one t4_3 w_16 c_7 b_t4_3 b_w_16 b_c_7,
      by rw [e_t4_4]; exact Nat.mod_add_div _ _⟩
  clear e_t4_4
  subst hres
  -- BEGIN conclusion
  -- The two sides on the integers, in the block's words.
  have hA := row_side a (by omega) d hd m0' s0' (by rw [e_m0', e_s0']; exact hrep0)
  have hB := row_side b (by omega) e he m1' s1' (by rw [e_m1', e_s1']; exact hrep1)
  rw [← e_d0, ← e_d1, ← e_d2, ← e_d3, ← e_w_1, ← e_w_5, ← e_w_9, ← e_w_13, ← e_lo, ← e_t4] at hA
  rw [← e_e0, ← e_e1, ← e_e2, ← e_e3, ← e_w_3, ← e_w_7, ← e_w_11, ← e_w_15, ← e_w, ← e_lo_8] at hB
  -- The row bound on the magnitudes: the low corrections add without a carry, and each high
  -- half is at most its magnitude, so no column's `adc` overflows.
  have hrow : m0' + m1' ≤ 2^63 := by rw [e_m0', e_m1']; omega
  have hlo_le : lo ≤ m0' := by rw [e_lo]; exact Nat.and_le_left
  have hw_le : w ≤ m1' := by rw [e_w]; exact Nat.and_le_left
  have ht0 : t0 = lo + w := by
    rw [e_t0]; show (lo + w) % 2^64 = lo + w; exact Nat.mod_eq_of_lt (by omega)
  have hh_1 : w_1 * m0' ≤ (2^64 - 1) * m0' := Nat.mul_le_mul_right _ (by omega)
  have hh_3 : w_3 * m1' ≤ (2^64 - 1) * m1' := Nat.mul_le_mul_right _ (by omega)
  have hh_5 : w_5 * m0' ≤ (2^64 - 1) * m0' := Nat.mul_le_mul_right _ (by omega)
  have hh_7 : w_7 * m1' ≤ (2^64 - 1) * m1' := Nat.mul_le_mul_right _ (by omega)
  have hh_9 : w_9 * m0' ≤ (2^64 - 1) * m0' := Nat.mul_le_mul_right _ (by omega)
  have hh_11 : w_11 * m1' ≤ (2^64 - 1) * m1' := Nat.mul_le_mul_right _ (by omega)
  have hh_13 : w_13 * m0' ≤ (2^64 - 1) * m0' := Nat.mul_le_mul_right _ (by omega)
  have hh_15 : w_15 * m1' ≤ (2^64 - 1) * m1' := Nat.mul_le_mul_right _ (by omega)
  have hk_t1 : k_t1 = 0 := by
    clear * - l_t1 b_k_t1 d_w_2 hh_1 hrow b_c; omega
  have hk_t1_1 : k_t1_1 = 0 := by
    clear * - l_t1_1 b_k_t1_1 l_t1 hk_t1 d_w_2 hh_1 d_w_4 hh_3 hrow b_c b_c_1; omega
  have hk_t2 : k_t2 = 0 := by
    clear * - l_t2 b_k_t2 d_w_6 hh_5 hrow b_c_2; omega
  have hk_t2_1 : k_t2_1 = 0 := by
    clear * - l_t2_1 b_k_t2_1 l_t2 hk_t2 d_w_6 hh_5 d_w_8 hh_7 hrow b_c_2 b_c_3; omega
  have hk_t3 : k_t3 = 0 := by
    clear * - l_t3 b_k_t3 d_w_10 hh_9 hrow b_c_4; omega
  have hk_t3_1 : k_t3_1 = 0 := by
    clear * - l_t3_1 b_k_t3_1 l_t3 hk_t3 d_w_10 hh_9 d_w_12 hh_11 hrow b_c_4 b_c_5; omega
  -- The top word's negation and subtraction, modulo `2^64`.
  obtain ⟨j_t4_1, b_j_t4_1, l_t4_1⟩ : ∃ j, j ≤ 1 ∧ t4_1 + 2^64 * j = 2^64 - t4 := by
    have h : t4_1 = (2^64 - t4) % 2^64 := e_t4_1
    clear * - h b_t4
    exact ⟨(2^64 - t4) / 2^64, by omega, by omega⟩
  obtain ⟨j_t4_3, b_j_t4_3, l_t4_3⟩ : ∃ j, j ≤ 1 ∧ t4_3 + 2^64 * j = t4_2 + 2^64 - lo_8 := by
    have h : t4_3 = (t4_2 + 2^64 - lo_8) % 2^64 := e_t4_3
    clear * - h b_t4_2 b_lo_8
    exact ⟨(t4_2 + 2^64 - lo_8) / 2^64, by omega, by omega⟩
  -- The five words modulo `2^320`: the corrections, the eight products, and `2^321` from the
  -- two negations in the top word.
  have hS : t0_2 + 2^64 * t1_3 + 2^128 * t2_3 + 2^192 * t3_3 + 2^256 * t4_4
        + 2^320 * (j_t4_1 + j_t4_3 + k_t4_2 + k_t4_4) + 2^256 * (t4 + lo_8)
      = lo + w
        + (w_1 * m0' + 2^64 * (w_5 * m0') + 2^128 * (w_9 * m0') + 2^192 * (w_13 * m0'))
        + (w_3 * m1' + 2^64 * (w_7 * m1') + 2^128 * (w_11 * m1') + 2^192 * (w_15 * m1'))
        + 2^321 := by
    clear * - ht0 l_t0_1 l_t1 l_t0_2 l_t1_1 l_t1_2 l_t2 l_t1_3 l_t2_1 l_t2_2 l_t3 l_t2_3 l_t3_1
      l_t3_2 l_t4_2 l_t3_3 l_t4_4 l_t4_1 l_t4_3 d_w_2 d_w_4 d_w_6 d_w_8 d_w_10 d_w_12 d_w_14
      d_w_16 hk_t1 hk_t1_1 hk_t2 hk_t2_1 hk_t3 hk_t3_1 b_t4 b_lo_8
    omega
  -- On the integers: the words are `a d + b e` up to a multiple of `2^320`, which the bounds
  -- of both sides fix.
  have hSz : (t0_2 : ℤ) + 2^64 * t1_3 + 2^128 * t2_3 + 2^192 * t3_3 + 2^256 * t4_4
        + 2^320 * (j_t4_1 + j_t4_3 + k_t4_2 + k_t4_4) + 2^256 * (t4 + lo_8)
      = lo + w
        + (w_1 * m0' + 2^64 * (w_5 * m0') + 2^128 * (w_9 * m0') + 2^192 * (w_13 * m0'))
        + (w_3 * m1' + 2^64 * (w_7 * m1') + 2^128 * (w_11 * m1') + 2^192 * (w_15 * m1'))
        + 2^321 := by exact_mod_cast hS
  have hval : (t0_2 : ℤ) + 2^64 * t1_3 + 2^128 * t2_3 + 2^192 * t3_3 + 2^256 * t4_4
      = a * d.toNat + b * e.toNat
        + 2^320 * (2 - (j_t4_1 + j_t4_3 + k_t4_2 + k_t4_4 : ℕ)) := by
    push_cast at hSz hA hB ⊢
    linear_combination hSz + hA + hB
  have hd' : |(d.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (by positivity)]; exact_mod_cast Limbs.toNat_lt d hd
  have he' : |(e.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (by positivity)]; exact_mod_cast Limbs.toNat_lt e he
  have hbound : |a * d.toNat + b * e.toNat| < 2^319 := by
    have h := Inversion.row_abs_lt a b _ _ _ _ hab hd' he' (by positivity)
    rwa [show (2 : ℤ)^63 * 2^256 = 2^319 by norm_num] at h
  rw [abs_lt] at hbound
  refine ⟨⟨b_t0_2, b_t1_3, b_t2_3, b_t3_3, b_t4_4⟩, ?_⟩
  unfold Signed5.toInt
  dsimp only
  clear * - hval hbound b_t0_2 b_t1_3 b_t2_3 b_t3_3 b_t4_4 b_j_t4_1 b_j_t4_3 b_k_t4_2 b_k_t4_4
  split_ifs <;> omega
  -- END conclusion

end PastaCurves.AArch64
