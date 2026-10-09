/-
Copyright Amazon.com, Inc. or its affiliates (the block, adapted from s2n-bignum).
Copyright (c) 2026 the pasta_curves contributors (the transcription and the proofs).
-/
import PastaCurves.AArch64.Spec.DeRow

/-!
# Correctness of the inversion's `f`, `g` row block

See the parent module's documentation for details. The block is the accumulation of the `d`, `e` row
block on five-word signed inputs, followed by the shift right by `59`. The sign word of each input
enters only the top correction: `(f4 ^ s0) & m0`, negated, is `-|a|` when the sign word of `f` and
the mask of `a` differ, and `0` otherwise (`row_side5`). The shift is four `extr`s and an `asr`,
which together divide the 320-bit two's-complement value by `2^59`, rounding down.
-/

set_option exponentiation.threshold 400

namespace PastaCurves.AArch64

open Inversion (SignMagRep row_side5 shift59_spec)

-- BEGIN fgRowBlock_spec statement
/-- The row `(a f + b g) / 2^59` of `updateFG`, rounded down, in five words. The inputs are bounded
five-word values below `2^256` in magnitude, as the rounds maintain; the matrix entries come in
their sign-magnitude forms; and the row bound `|a| + |b| ≤ 2^63` keeps every column's carry within
the next word and the unshifted value within `2^319`. -/
theorem fgRowBlock_spec (a b : ℤ) (f g : Signed5) (m0 m1 s0 s1 : Nat)
    (hf : f.Bounded) (hg : g.Bounded) (hfv : |f.toInt| < 2^256) (hgv : |g.toInt| < 2^256)
    (hab : |a| + |b| ≤ 2^63)
    (hrep0 : SignMagRep m0 s0 a) (hrep1 : SignMagRep m1 s1 b) :
    ∀ res, res = fgRowBlock f g m0 m1 s0 s1 →
      res.Bounded ∧ res.toInt = (a * f.toInt + b * g.toInt) / 2^59 := by
  intro res hres
  have ha0 := abs_nonneg a
  have hb0 := abs_nonneg b
  have hm0z : (m0 : ℤ) = |a| := hrep0.natCast_eq_abs _ _ _
  have hm1z : (m1 : ℤ) = |b| := hrep1.natCast_eq_abs _ _ _
  have hm0 : m0 < 2^64 := by omega
  have hm1 : m1 < 2^64 := by omega
  have hs0 : s0 < 2^64 := (hrep0.lt _ _ _ (by omega)).2
  have hs1 : s1 < 2^64 := (hrep1.lt _ _ _ (by omega)).2
-- END fgRowBlock_spec statement
  -- generated skeleton for `fgRowBlock`: do not edit between the annotations
  unfold fgRowBlock at hres
  lift_lets -merge at hres
  -- f0: argument
  word_step f0 := f.l0 using hf.1
  -- f1: argument
  word_step f1 := f.l1 using hf.2.1
  -- f2: argument
  word_step f2 := f.l2 using hf.2.2.1
  -- f3: argument
  word_step f3 := f.l3 using hf.2.2.2.1
  -- f4: argument
  word_step f4 := f.l4 using hf.2.2.2.2
  -- g0: argument
  word_step g0 := g.l0 using hg.1
  -- g1: argument
  word_step g1 := g.l1 using hg.2.1
  -- g2: argument
  word_step g2 := g.l2 using hg.2.2.1
  -- g3: argument
  word_step g3 := g.l3 using hg.2.2.2.1
  -- g4: argument
  word_step g4 := g.l4 using hg.2.2.2.2
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
  -- w_1: eor w,f0,s0
  word_step w_1 := eorw f0 s0' using eorw_lt f0 s0' b_f0 b_s0'
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
  -- w_3: eor w,g0,s1
  word_step w_3 := eorw g0 s1' using eorw_lt g0 s1' b_g0 b_s1'
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
  -- w_5: eor w,f1,s0
  word_step w_5 := eorw f1 s0' using eorw_lt f1 s0' b_f1 b_s0'
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
  -- w_7: eor w,g1,s1
  word_step w_7 := eorw g1 s1' using eorw_lt g1 s1' b_g1 b_s1'
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
  -- t0_3: extr t0,t1,t0,#59
  word_step t0_3 := extr t1_3 t0_2 59 using extr_lt t1_3 t0_2 59
  -- w_9: eor w,f2,s0
  word_step w_9 := eorw f2 s0' using eorw_lt f2 s0' b_f2 b_s0'
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
  -- w_11: eor w,g2,s1
  word_step w_11 := eorw g2 s1' using eorw_lt g2 s1' b_g2 b_s1'
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
  -- t1_4: extr t1,t2,t1,#59
  word_step t1_4 := extr t2_3 t1_3 59 using extr_lt t2_3 t1_3 59
  -- w_13: eor w,f3,s0
  word_step w_13 := eorw f3 s0' using eorw_lt f3 s0' b_f3 b_s0'
  -- t4: eor t4,f4,s0
  word_step t4 := eorw f4 s0' using eorw_lt f4 s0' b_f4 b_s0'
  -- t4_1: and t4,t4,m0
  word_step t4_1 := andw t4 m0' using andw_lt t4 m0' b_t4 b_m0'
  -- t4_2: neg t4,t4
  word_step t4_2 := negw t4_1 using negw_lt t4_1
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
  -- t4_3: adc t4,t4,w
  word_step t4_3 := (t4_2 + w_14 + c_6) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t4_3, b_k_t4_3, l_t4_3⟩ :
      ∃ k, k ≤ 1 ∧ t4_3 + 2^64 * k = t4_2 + w_14 + c_6 :=
    ⟨(t4_2 + w_14 + c_6) / 2^64, addc_carry_le_one t4_2 w_14 c_6 b_t4_2 b_w_14 b_c_6,
      by rw [e_t4_3]; exact Nat.mod_add_div _ _⟩
  clear e_t4_3
  -- w_15: eor w,g3,s1
  word_step w_15 := eorw g3 s1' using eorw_lt g3 s1' b_g3 b_s1'
  -- lo_8: eor lo,g4,s1
  word_step lo_8 := eorw g4 s1' using eorw_lt g4 s1' b_g4 b_s1'
  -- lo_9: and lo,lo,m1
  word_step lo_9 := andw lo_8 m1' using andw_lt lo_8 m1' b_lo_8 b_m1'
  -- t4_4: sub t4,t4,lo
  word_step t4_4 := subw t4_3 lo_9 using subw_lt t4_3 lo_9
  -- lo_10: mul lo,w,m1
  word_step lo_10 := w_15 * m1' % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_16: umulh w,w,m1
  word_step w_16 := w_15 * m1' / 2^64 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_w_15 b_m1')
  have d_w_16 : lo_10 + 2^64 * w_16 = w_15 * m1' := by
    rw [e_lo_10, e_w_16]; exact Nat.mod_add_div _ _
  clear e_lo_10 e_w_16
  -- t3_3: adds t3,t3,lo
  word_step s_7, t3_3 := (t3_2 + lo_10 + 0) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _),
      c_7 := (t3_2 + lo_10 + 0) / 2^64
  have l_t3_3 : t3_3 + 2^64 * c_7 = t3_2 + lo_10 + 0 := by
    rw [e_t3_3, e_c_7]; exact Nat.mod_add_div _ _
  have b_c_7 : c_7 ≤ 1 := by
    rw [e_c_7]; exact addc_carry_le_one t3_2 lo_10 0 b_t3_2 b_lo_10 (by decide)
  clear e_t3_3 e_c_7
  -- t4_5: adc t4,t4,w
  word_step t4_5 := (t4_4 + w_16 + c_7) % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t4_5, b_k_t4_5, l_t4_5⟩ :
      ∃ k, k ≤ 1 ∧ t4_5 + 2^64 * k = t4_4 + w_16 + c_7 :=
    ⟨(t4_4 + w_16 + c_7) / 2^64, addc_carry_le_one t4_4 w_16 c_7 b_t4_4 b_w_16 b_c_7,
      by rw [e_t4_5]; exact Nat.mod_add_div _ _⟩
  clear e_t4_5
  -- t2_4: extr t2,t3,t2,#59
  word_step t2_4 := extr t3_3 t2_3 59 using extr_lt t3_3 t2_3 59
  -- t3_4: extr t3,t4,t3,#59
  word_step t3_4 := extr t4_5 t3_3 59 using extr_lt t4_5 t3_3 59
  -- t4_6: asr t4,t4,#59
  word_step t4_6 := asr t4_5 59 using asr_lt t4_5 59 b_t4_5
  subst hres
  -- BEGIN conclusion
  -- The two sides on the integers, in the block's words.
  have hf4 := Signed5.l4_of_abs_lt f hf hfv
  have hg4 := Signed5.l4_of_abs_lt g hg hgv
  have hA := row_side5 a (by omega) f hf hf4 m0' s0' (by rw [e_m0', e_s0']; exact hrep0)
  have hB := row_side5 b (by omega) g hg hg4 m1' s1' (by rw [e_m1', e_s1']; exact hrep1)
  rw [← e_f0, ← e_f1, ← e_f2, ← e_f3, ← e_f4, ← e_w_1, ← e_w_5, ← e_w_9, ← e_w_13, ← e_lo, ← e_t4,
    ← e_t4_1] at hA
  rw [← e_g0, ← e_g1, ← e_g2, ← e_g3, ← e_g4, ← e_w_3, ← e_w_7, ← e_w_11, ← e_w_15, ← e_w, ← e_lo_8,
    ← e_lo_9] at hB
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
  have hk_d1 : k_t1 = 0 := by
    clear * - l_t1 b_k_t1 d_w_2 hh_1 hrow b_c; omega
  have hk_d1_1 : k_t1_1 = 0 := by
    clear * - l_t1_1 b_k_t1_1 l_t1 hk_d1 d_w_2 hh_1 d_w_4 hh_3 hrow b_c b_c_1; omega
  have hk_d2 : k_t2 = 0 := by
    clear * - l_t2 b_k_t2 d_w_6 hh_5 hrow b_c_2; omega
  have hk_d2_1 : k_t2_1 = 0 := by
    clear * - l_t2_1 b_k_t2_1 l_t2 hk_d2 d_w_6 hh_5 d_w_8 hh_7 hrow b_c_2 b_c_3; omega
  have hk_d3 : k_t3 = 0 := by
    clear * - l_t3 b_k_t3 d_w_10 hh_9 hrow b_c_4; omega
  have hk_d3_1 : k_t3_1 = 0 := by
    clear * - l_t3_1 b_k_t3_1 l_t3 hk_d3 d_w_10 hh_9 d_w_12 hh_11 hrow b_c_4 b_c_5; omega
  -- The top word's negation and subtraction, modulo `2^64`.
  obtain ⟨j_d4_2, b_j_d4_2, l_t4_2⟩ : ∃ j, j ≤ 1 ∧ t4_2 + 2^64 * j = 2^64 - t4_1 := by
    have h : t4_2 = (2^64 - t4_1) % 2^64 := e_t4_2
    clear * - h b_t4_1
    exact ⟨(2^64 - t4_1) / 2^64, by omega, by omega⟩
  obtain ⟨j_d4_4, b_j_d4_4, l_t4_4⟩ : ∃ j, j ≤ 1 ∧ t4_4 + 2^64 * j = t4_3 + 2^64 - lo_9 := by
    have h : t4_4 = (t4_3 + 2^64 - lo_9) % 2^64 := e_t4_4
    clear * - h b_t4_3 b_lo_9
    exact ⟨(t4_3 + 2^64 - lo_9) / 2^64, by omega, by omega⟩
  -- The five words before the shift, modulo `2^320`: the corrections, the eight products, and
  -- `2^321` from the two negations in the top word.
  have hS : t0_2 + 2^64 * t1_3 + 2^128 * t2_3 + 2^192 * t3_3 + 2^256 * t4_5
        + 2^320 * (j_d4_2 + j_d4_4 + k_t4_3 + k_t4_5) + 2^256 * (t4_1 + lo_9)
      = lo + w
        + (w_1 * m0' + 2^64 * (w_5 * m0') + 2^128 * (w_9 * m0') + 2^192 * (w_13 * m0'))
        + (w_3 * m1' + 2^64 * (w_7 * m1') + 2^128 * (w_11 * m1') + 2^192 * (w_15 * m1'))
        + 2^321 := by
    clear * - ht0 l_t0_1 l_t1 l_t0_2 l_t1_1 l_t1_2 l_t2 l_t1_3 l_t2_1 l_t2_2 l_t3 l_t2_3 l_t3_1
      l_t3_2 l_t4_3 l_t3_3 l_t4_5 l_t4_2 l_t4_4 d_w_2 d_w_4 d_w_6 d_w_8 d_w_10 d_w_12 d_w_14
      d_w_16 hk_d1 hk_d1_1 hk_d2 hk_d2_1 hk_d3 hk_d3_1 b_t4_1 b_lo_9
    omega
  -- On the integers: the unshifted words are `a f + b g` up to a multiple of `2^320`, which
  -- the bounds of both sides fix.
  have hSz : (t0_2 : ℤ) + 2^64 * t1_3 + 2^128 * t2_3 + 2^192 * t3_3 + 2^256 * t4_5
        + 2^320 * (j_d4_2 + j_d4_4 + k_t4_3 + k_t4_5) + 2^256 * (t4_1 + lo_9)
      = lo + w
        + (w_1 * m0' + 2^64 * (w_5 * m0') + 2^128 * (w_9 * m0') + 2^192 * (w_13 * m0'))
        + (w_3 * m1' + 2^64 * (w_7 * m1') + 2^128 * (w_11 * m1') + 2^192 * (w_15 * m1'))
        + 2^321 := by exact_mod_cast hS
  have hval : (t0_2 : ℤ) + 2^64 * t1_3 + 2^128 * t2_3 + 2^192 * t3_3 + 2^256 * t4_5
      = a * f.toInt + b * g.toInt
        + 2^320 * (2 - (j_d4_2 + j_d4_4 + k_t4_3 + k_t4_5 : ℕ)) := by
    push_cast at hSz hA hB ⊢
    linear_combination hSz + hA + hB
  have hbound : |a * f.toInt + b * g.toInt| < 2^319 := by
    have h := Inversion.row_abs_lt a b _ _ _ _ hab hfv hgv (by positivity)
    rwa [show (2 : ℤ)^63 * 2^256 = 2^319 by norm_num] at h
  rw [abs_lt] at hbound
  have hT : (Signed5.mk t0_2 t1_3 t2_3 t3_3 t4_5).toInt = a * f.toInt + b * g.toInt := by
    rw [show (Signed5.mk t0_2 t1_3 t2_3 t3_3 t4_5).toInt = (t0_2 : ℤ) + 2^64 * t1_3 + 2^128 * t2_3
      + 2^192 * t3_3 + 2^256 * (if t4_5 < 2^63 then (t4_5 : ℤ) else (t4_5 : ℤ) - 2^64) from rfl]
    clear * - hval hbound b_t0_2 b_t1_3 b_t2_3 b_t3_3 b_t4_5 b_j_d4_2 b_j_d4_4 b_k_t4_3 b_k_t4_5
    split_ifs <;> omega
  -- The shift by `59`.
  obtain ⟨hB5, hS5⟩ := shift59_spec t0_2 t1_3 t2_3 t3_3 t4_5 b_t0_2 b_t1_3 b_t2_3 b_t3_3 b_t4_5
  rw [e_t0_3, e_t1_4, e_t2_4, e_t3_4, e_t4_6, hS5, hT]
  exact ⟨hB5, rfl⟩
  -- END conclusion

end PastaCurves.AArch64
