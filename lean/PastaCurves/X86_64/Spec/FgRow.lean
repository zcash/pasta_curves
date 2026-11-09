/-
Copyright Amazon.com, Inc. or its affiliates (the block, adapted from s2n-bignum).
Copyright (c) 2026 the pasta_curves contributors (the transcription and the proofs).
-/
import PastaCurves.X86_64.Spec.DeRow

/-!
# Correctness of the inversion's `f`, `g` row block

The statement is the AArch64 block's. The block is the accumulation of the `d`, `e` row block on
five-word signed inputs, followed by the shift right by `59`. The top column multiplies the
complemented sign word of each input by its magnitude with `imul`; both the sign word and the mask
are `0` or all ones, so the complemented word is too (`eorw_mask`), and `mask_mul` turns the product
into `row_side5`'s top correction. The shift is four `shrd`s and a `sar`, which are the `extr`s and
the `asr` of `shift59_spec`.
-/

set_option exponentiation.threshold 400

namespace PastaCurves.X86_64

open Inversion (SignMagRep row_side5 shift59_spec)

-- BEGIN fgRowBlock_spec lemmas
/-- Xoring two masks gives a mask. The top column of `fg_row` complements a sign word, which is a
mask, under the sign mask of a matrix entry. -/
theorem eorw_mask (x s : ℕ) (hx : x = 0 ∨ x = 2^64 - 1) (hs : s = 0 ∨ s = 2^64 - 1) :
    eorw x s = 0 ∨ eorw x s = 2^64 - 1 := by
  rcases hx with rfl | rfl <;> rcases hs with rfl | rfl <;> simp [eorw]

-- END fgRowBlock_spec lemmas

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
  -- m0': scalar input
  word_step m0' := m0 using hm0
  -- m1': scalar input
  word_step m1' := m1 using hm1
  -- s0': scalar input
  word_step s0' := s0 using hs0
  -- s1': scalar input
  word_step s1' := s1 using hs1
  -- t0: mov {t0}, {m0}
  word_step t0 := m0' using b_m0'
  -- t0_1: and {t0}, {s0}
  word_step t0_1 := andw t0 s0' using andw_lt t0 s0' b_t0 b_s0'
  -- rax: mov rax, {m1}
  word_step rax := m1' using b_m1'
  -- rax_1: and rax, {s1}
  word_step rax_1 := andw rax s1' using andw_lt rax s1' b_rax b_s1'
  -- t0_2: add {t0}, rax
  word_step s, t0_2 := (addc t0_1 rax_1 0).1 using addc_value_lt t0_1 rax_1 0,
      cf := (addc t0_1 rax_1 0).2
  have l_t0_2 : t0_2 + 2^64 * cf = t0_1 + rax_1 + 0 := by
    rw [e_t0_2, e_cf]; exact addc_lin t0_1 rax_1 0
  have b_cf : cf ≤ 1 := by rw [e_cf]; exact addc_carry_le_one t0_1 rax_1 0 b_t0_1 b_rax_1 (by decide)
  clear e_t0_2 e_cf
  -- t1: xor {t1:e}, {t1:e}
  word_step t1 := 0 using (by decide), cf_1 := 0, ofl := 0
  have b_cf_1 : cf_1 ≤ 1 := by rw [e_cf_1]; decide
  have b_ofl : ofl ≤ 1 := by rw [e_ofl]; decide
  -- t2: xor {t2:e}, {t2:e}
  word_step t2 := 0 using (by decide), cf_2 := 0, ofl_1 := 0
  have b_cf_2 : cf_2 ≤ 1 := by rw [e_cf_2]; decide
  have b_ofl_1 : ofl_1 ≤ 1 := by rw [e_ofl_1]; decide
  -- t3: xor {t3:e}, {t3:e}
  word_step t3 := 0 using (by decide), cf_3 := 0, ofl_2 := 0
  have b_cf_3 : cf_3 ≤ 1 := by rw [e_cf_3]; decide
  have b_ofl_2 : ofl_2 ≤ 1 := by rw [e_ofl_2]; decide
  -- t4: xor {t4:e}, {t4:e}
  word_step t4 := 0 using (by decide), cf_4 := 0, ofl_3 := 0
  have b_cf_4 : cf_4 ≤ 1 := by rw [e_cf_4]; decide
  have b_ofl_3 : ofl_3 ≤ 1 := by rw [e_ofl_3]; decide
  -- rax_2: mov rax, qword ptr [{x}]
  word_step rax_2 := f.l0 using hf.1
  -- rax_3: xor rax, {s0}
  word_step rax_3 := eorw rax_2 s0' using eorw_lt rax_2 s0' b_rax_2 b_s0'
  -- m: mul {m0}
  word_step m, rax_4 := (mulq rax_3 m0').1 using Nat.mod_lt _ (Nat.two_pow_pos _),
      rdx := (mulq rax_3 m0').2 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rax_3 b_m0')
  have d_rdx : rax_4 + 2^64 * rdx = rax_3 * m0' := by
    rw [e_rax_4, e_rdx]; exact Nat.mod_add_div _ _
  -- t0_3: add {t0}, rax
  word_step s_1, t0_3 := (addc t0_2 rax_4 0).1 using addc_value_lt t0_2 rax_4 0,
      cf_5 := (addc t0_2 rax_4 0).2
  have l_t0_3 : t0_3 + 2^64 * cf_5 = t0_2 + rax_4 + 0 := by
    rw [e_t0_3, e_cf_5]; exact addc_lin t0_2 rax_4 0
  have b_cf_5 : cf_5 ≤ 1 := by rw [e_cf_5]; exact addc_carry_le_one t0_2 rax_4 0 b_t0_2 b_rax_4 (by decide)
  clear e_t0_3 e_cf_5
  -- t1_1: adc {t1}, rdx
  word_step s_2, t1_1 := (addc t1 rdx cf_5).1 using addc_value_lt t1 rdx cf_5,
      cf_6 := (addc t1 rdx cf_5).2
  have l_t1_1 : t1_1 + 2^64 * cf_6 = t1 + rdx + cf_5 := by
    rw [e_t1_1, e_cf_6]; exact addc_lin t1 rdx cf_5
  have b_cf_6 : cf_6 ≤ 1 := by rw [e_cf_6]; exact addc_carry_le_one t1 rdx cf_5 b_t1 b_rdx b_cf_5
  clear e_t1_1 e_cf_6
  -- rax_5: mov rax, qword ptr [{y}]
  word_step rax_5 := g.l0 using hg.1
  -- rax_6: xor rax, {s1}
  word_step rax_6 := eorw rax_5 s1' using eorw_lt rax_5 s1' b_rax_5 b_s1'
  -- m_1: mul {m1}
  word_step m_1, rax_7 := (mulq rax_6 m1').1 using Nat.mod_lt _ (Nat.two_pow_pos _),
      rdx_1 := (mulq rax_6 m1').2 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rax_6 b_m1')
  have d_rdx_1 : rax_7 + 2^64 * rdx_1 = rax_6 * m1' := by
    rw [e_rax_7, e_rdx_1]; exact Nat.mod_add_div _ _
  -- t0_4: add {t0}, rax
  word_step s_3, t0_4 := (addc t0_3 rax_7 0).1 using addc_value_lt t0_3 rax_7 0,
      cf_7 := (addc t0_3 rax_7 0).2
  have l_t0_4 : t0_4 + 2^64 * cf_7 = t0_3 + rax_7 + 0 := by
    rw [e_t0_4, e_cf_7]; exact addc_lin t0_3 rax_7 0
  have b_cf_7 : cf_7 ≤ 1 := by rw [e_cf_7]; exact addc_carry_le_one t0_3 rax_7 0 b_t0_3 b_rax_7 (by decide)
  clear e_t0_4 e_cf_7
  -- t1_2: adc {t1}, rdx
  word_step s_4, t1_2 := (addc t1_1 rdx_1 cf_7).1 using addc_value_lt t1_1 rdx_1 cf_7,
      cf_8 := (addc t1_1 rdx_1 cf_7).2
  have l_t1_2 : t1_2 + 2^64 * cf_8 = t1_1 + rdx_1 + cf_7 := by
    rw [e_t1_2, e_cf_8]; exact addc_lin t1_1 rdx_1 cf_7
  have b_cf_8 : cf_8 ≤ 1 := by rw [e_cf_8]; exact addc_carry_le_one t1_1 rdx_1 cf_7 b_t1_1 b_rdx_1 b_cf_7
  clear e_t1_2 e_cf_8
  -- rax_8: mov rax, qword ptr [{x} + 8]
  word_step rax_8 := f.l1 using hf.2.1
  -- rax_9: xor rax, {s0}
  word_step rax_9 := eorw rax_8 s0' using eorw_lt rax_8 s0' b_rax_8 b_s0'
  -- m_2: mul {m0}
  word_step m_2, rax_10 := (mulq rax_9 m0').1 using Nat.mod_lt _ (Nat.two_pow_pos _),
      rdx_2 := (mulq rax_9 m0').2 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rax_9 b_m0')
  have d_rdx_2 : rax_10 + 2^64 * rdx_2 = rax_9 * m0' := by
    rw [e_rax_10, e_rdx_2]; exact Nat.mod_add_div _ _
  -- t1_3: add {t1}, rax
  word_step s_5, t1_3 := (addc t1_2 rax_10 0).1 using addc_value_lt t1_2 rax_10 0,
      cf_9 := (addc t1_2 rax_10 0).2
  have l_t1_3 : t1_3 + 2^64 * cf_9 = t1_2 + rax_10 + 0 := by
    rw [e_t1_3, e_cf_9]; exact addc_lin t1_2 rax_10 0
  have b_cf_9 : cf_9 ≤ 1 := by rw [e_cf_9]; exact addc_carry_le_one t1_2 rax_10 0 b_t1_2 b_rax_10 (by decide)
  clear e_t1_3 e_cf_9
  -- t2_1: adc {t2}, rdx
  word_step s_6, t2_1 := (addc t2 rdx_2 cf_9).1 using addc_value_lt t2 rdx_2 cf_9,
      cf_10 := (addc t2 rdx_2 cf_9).2
  have l_t2_1 : t2_1 + 2^64 * cf_10 = t2 + rdx_2 + cf_9 := by
    rw [e_t2_1, e_cf_10]; exact addc_lin t2 rdx_2 cf_9
  have b_cf_10 : cf_10 ≤ 1 := by rw [e_cf_10]; exact addc_carry_le_one t2 rdx_2 cf_9 b_t2 b_rdx_2 b_cf_9
  clear e_t2_1 e_cf_10
  -- rax_11: mov rax, qword ptr [{y} + 8]
  word_step rax_11 := g.l1 using hg.2.1
  -- rax_12: xor rax, {s1}
  word_step rax_12 := eorw rax_11 s1' using eorw_lt rax_11 s1' b_rax_11 b_s1'
  -- m_3: mul {m1}
  word_step m_3, rax_13 := (mulq rax_12 m1').1 using Nat.mod_lt _ (Nat.two_pow_pos _),
      rdx_3 := (mulq rax_12 m1').2 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rax_12 b_m1')
  have d_rdx_3 : rax_13 + 2^64 * rdx_3 = rax_12 * m1' := by
    rw [e_rax_13, e_rdx_3]; exact Nat.mod_add_div _ _
  -- t1_4: add {t1}, rax
  word_step s_7, t1_4 := (addc t1_3 rax_13 0).1 using addc_value_lt t1_3 rax_13 0,
      cf_11 := (addc t1_3 rax_13 0).2
  have l_t1_4 : t1_4 + 2^64 * cf_11 = t1_3 + rax_13 + 0 := by
    rw [e_t1_4, e_cf_11]; exact addc_lin t1_3 rax_13 0
  have b_cf_11 : cf_11 ≤ 1 := by rw [e_cf_11]; exact addc_carry_le_one t1_3 rax_13 0 b_t1_3 b_rax_13 (by decide)
  clear e_t1_4 e_cf_11
  -- t2_2: adc {t2}, rdx
  word_step s_8, t2_2 := (addc t2_1 rdx_3 cf_11).1 using addc_value_lt t2_1 rdx_3 cf_11,
      cf_12 := (addc t2_1 rdx_3 cf_11).2
  have l_t2_2 : t2_2 + 2^64 * cf_12 = t2_1 + rdx_3 + cf_11 := by
    rw [e_t2_2, e_cf_12]; exact addc_lin t2_1 rdx_3 cf_11
  have b_cf_12 : cf_12 ≤ 1 := by rw [e_cf_12]; exact addc_carry_le_one t2_1 rdx_3 cf_11 b_t2_1 b_rdx_3 b_cf_11
  clear e_t2_2 e_cf_12
  -- rax_14: mov rax, qword ptr [{x} + 16]
  word_step rax_14 := f.l2 using hf.2.2.1
  -- rax_15: xor rax, {s0}
  word_step rax_15 := eorw rax_14 s0' using eorw_lt rax_14 s0' b_rax_14 b_s0'
  -- m_4: mul {m0}
  word_step m_4, rax_16 := (mulq rax_15 m0').1 using Nat.mod_lt _ (Nat.two_pow_pos _),
      rdx_4 := (mulq rax_15 m0').2 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rax_15 b_m0')
  have d_rdx_4 : rax_16 + 2^64 * rdx_4 = rax_15 * m0' := by
    rw [e_rax_16, e_rdx_4]; exact Nat.mod_add_div _ _
  -- t2_3: add {t2}, rax
  word_step s_9, t2_3 := (addc t2_2 rax_16 0).1 using addc_value_lt t2_2 rax_16 0,
      cf_13 := (addc t2_2 rax_16 0).2
  have l_t2_3 : t2_3 + 2^64 * cf_13 = t2_2 + rax_16 + 0 := by
    rw [e_t2_3, e_cf_13]; exact addc_lin t2_2 rax_16 0
  have b_cf_13 : cf_13 ≤ 1 := by rw [e_cf_13]; exact addc_carry_le_one t2_2 rax_16 0 b_t2_2 b_rax_16 (by decide)
  clear e_t2_3 e_cf_13
  -- t3_1: adc {t3}, rdx
  word_step s_10, t3_1 := (addc t3 rdx_4 cf_13).1 using addc_value_lt t3 rdx_4 cf_13,
      cf_14 := (addc t3 rdx_4 cf_13).2
  have l_t3_1 : t3_1 + 2^64 * cf_14 = t3 + rdx_4 + cf_13 := by
    rw [e_t3_1, e_cf_14]; exact addc_lin t3 rdx_4 cf_13
  have b_cf_14 : cf_14 ≤ 1 := by rw [e_cf_14]; exact addc_carry_le_one t3 rdx_4 cf_13 b_t3 b_rdx_4 b_cf_13
  clear e_t3_1 e_cf_14
  -- rax_17: mov rax, qword ptr [{y} + 16]
  word_step rax_17 := g.l2 using hg.2.2.1
  -- rax_18: xor rax, {s1}
  word_step rax_18 := eorw rax_17 s1' using eorw_lt rax_17 s1' b_rax_17 b_s1'
  -- m_5: mul {m1}
  word_step m_5, rax_19 := (mulq rax_18 m1').1 using Nat.mod_lt _ (Nat.two_pow_pos _),
      rdx_5 := (mulq rax_18 m1').2 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rax_18 b_m1')
  have d_rdx_5 : rax_19 + 2^64 * rdx_5 = rax_18 * m1' := by
    rw [e_rax_19, e_rdx_5]; exact Nat.mod_add_div _ _
  -- t2_4: add {t2}, rax
  word_step s_11, t2_4 := (addc t2_3 rax_19 0).1 using addc_value_lt t2_3 rax_19 0,
      cf_15 := (addc t2_3 rax_19 0).2
  have l_t2_4 : t2_4 + 2^64 * cf_15 = t2_3 + rax_19 + 0 := by
    rw [e_t2_4, e_cf_15]; exact addc_lin t2_3 rax_19 0
  have b_cf_15 : cf_15 ≤ 1 := by rw [e_cf_15]; exact addc_carry_le_one t2_3 rax_19 0 b_t2_3 b_rax_19 (by decide)
  clear e_t2_4 e_cf_15
  -- t3_2: adc {t3}, rdx
  word_step s_12, t3_2 := (addc t3_1 rdx_5 cf_15).1 using addc_value_lt t3_1 rdx_5 cf_15,
      cf_16 := (addc t3_1 rdx_5 cf_15).2
  have l_t3_2 : t3_2 + 2^64 * cf_16 = t3_1 + rdx_5 + cf_15 := by
    rw [e_t3_2, e_cf_16]; exact addc_lin t3_1 rdx_5 cf_15
  have b_cf_16 : cf_16 ≤ 1 := by rw [e_cf_16]; exact addc_carry_le_one t3_1 rdx_5 cf_15 b_t3_1 b_rdx_5 b_cf_15
  clear e_t3_2 e_cf_16
  -- rax_20: mov rax, qword ptr [{x} + 24]
  word_step rax_20 := f.l3 using hf.2.2.2.1
  -- rax_21: xor rax, {s0}
  word_step rax_21 := eorw rax_20 s0' using eorw_lt rax_20 s0' b_rax_20 b_s0'
  -- m_6: mul {m0}
  word_step m_6, rax_22 := (mulq rax_21 m0').1 using Nat.mod_lt _ (Nat.two_pow_pos _),
      rdx_6 := (mulq rax_21 m0').2 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rax_21 b_m0')
  have d_rdx_6 : rax_22 + 2^64 * rdx_6 = rax_21 * m0' := by
    rw [e_rax_22, e_rdx_6]; exact Nat.mod_add_div _ _
  -- t3_3: add {t3}, rax
  word_step s_13, t3_3 := (addc t3_2 rax_22 0).1 using addc_value_lt t3_2 rax_22 0,
      cf_17 := (addc t3_2 rax_22 0).2
  have l_t3_3 : t3_3 + 2^64 * cf_17 = t3_2 + rax_22 + 0 := by
    rw [e_t3_3, e_cf_17]; exact addc_lin t3_2 rax_22 0
  have b_cf_17 : cf_17 ≤ 1 := by rw [e_cf_17]; exact addc_carry_le_one t3_2 rax_22 0 b_t3_2 b_rax_22 (by decide)
  clear e_t3_3 e_cf_17
  -- t4_1: adc {t4}, rdx
  word_step s_14, t4_1 := (addc t4 rdx_6 cf_17).1 using addc_value_lt t4 rdx_6 cf_17,
      cf_18 := (addc t4 rdx_6 cf_17).2
  have l_t4_1 : t4_1 + 2^64 * cf_18 = t4 + rdx_6 + cf_17 := by
    rw [e_t4_1, e_cf_18]; exact addc_lin t4 rdx_6 cf_17
  have b_cf_18 : cf_18 ≤ 1 := by rw [e_cf_18]; exact addc_carry_le_one t4 rdx_6 cf_17 b_t4 b_rdx_6 b_cf_17
  clear e_t4_1 e_cf_18
  -- rax_23: mov rax, qword ptr [{y} + 24]
  word_step rax_23 := g.l3 using hg.2.2.2.1
  -- rax_24: xor rax, {s1}
  word_step rax_24 := eorw rax_23 s1' using eorw_lt rax_23 s1' b_rax_23 b_s1'
  -- m_7: mul {m1}
  word_step m_7, rax_25 := (mulq rax_24 m1').1 using Nat.mod_lt _ (Nat.two_pow_pos _),
      rdx_7 := (mulq rax_24 m1').2 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rax_24 b_m1')
  have d_rdx_7 : rax_25 + 2^64 * rdx_7 = rax_24 * m1' := by
    rw [e_rax_25, e_rdx_7]; exact Nat.mod_add_div _ _
  -- t3_4: add {t3}, rax
  word_step s_15, t3_4 := (addc t3_3 rax_25 0).1 using addc_value_lt t3_3 rax_25 0,
      cf_19 := (addc t3_3 rax_25 0).2
  have l_t3_4 : t3_4 + 2^64 * cf_19 = t3_3 + rax_25 + 0 := by
    rw [e_t3_4, e_cf_19]; exact addc_lin t3_3 rax_25 0
  have b_cf_19 : cf_19 ≤ 1 := by rw [e_cf_19]; exact addc_carry_le_one t3_3 rax_25 0 b_t3_3 b_rax_25 (by decide)
  clear e_t3_4 e_cf_19
  -- t4_2: adc {t4}, rdx
  word_step s_16, t4_2 := (addc t4_1 rdx_7 cf_19).1 using addc_value_lt t4_1 rdx_7 cf_19,
      cf_20 := (addc t4_1 rdx_7 cf_19).2
  have l_t4_2 : t4_2 + 2^64 * cf_20 = t4_1 + rdx_7 + cf_19 := by
    rw [e_t4_2, e_cf_20]; exact addc_lin t4_1 rdx_7 cf_19
  have b_cf_20 : cf_20 ≤ 1 := by rw [e_cf_20]; exact addc_carry_le_one t4_1 rdx_7 cf_19 b_t4_1 b_rdx_7 b_cf_19
  clear e_t4_2 e_cf_20
  -- rax_26: mov rax, qword ptr [{x} + 32]
  word_step rax_26 := f.l4 using hf.2.2.2.2
  -- rax_27: xor rax, {s0}
  word_step rax_27 := eorw rax_26 s0' using eorw_lt rax_26 s0' b_rax_26 b_s0'
  -- rax_28: imul rax, {m0}
  word_step rax_28 := rax_27 * m0' % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t4_3: add {t4}, rax
  word_step s_17, t4_3 := (addc t4_2 rax_28 0).1 using addc_value_lt t4_2 rax_28 0,
      cf_21 := (addc t4_2 rax_28 0).2
  have l_t4_3 : t4_3 + 2^64 * cf_21 = t4_2 + rax_28 + 0 := by
    rw [e_t4_3, e_cf_21]; exact addc_lin t4_2 rax_28 0
  have b_cf_21 : cf_21 ≤ 1 := by rw [e_cf_21]; exact addc_carry_le_one t4_2 rax_28 0 b_t4_2 b_rax_28 (by decide)
  clear e_t4_3 e_cf_21
  -- rax_29: mov rax, qword ptr [{y} + 32]
  word_step rax_29 := g.l4 using hg.2.2.2.2
  -- rax_30: xor rax, {s1}
  word_step rax_30 := eorw rax_29 s1' using eorw_lt rax_29 s1' b_rax_29 b_s1'
  -- rax_31: imul rax, {m1}
  word_step rax_31 := rax_30 * m1' % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t4_4: add {t4}, rax
  word_step s_18, t4_4 := (addc t4_3 rax_31 0).1 using addc_value_lt t4_3 rax_31 0,
      cf_22 := (addc t4_3 rax_31 0).2
  have l_t4_4 : t4_4 + 2^64 * cf_22 = t4_3 + rax_31 + 0 := by
    rw [e_t4_4, e_cf_22]; exact addc_lin t4_3 rax_31 0
  have b_cf_22 : cf_22 ≤ 1 := by rw [e_cf_22]; exact addc_carry_le_one t4_3 rax_31 0 b_t4_3 b_rax_31 (by decide)
  clear e_t4_4 e_cf_22
  -- t0_5: shrd {t0}, {t1}, 59
  word_step t0_5 := extr t1_4 t0_4 59 using extr_lt t1_4 t0_4 59
  -- t1_5: shrd {t1}, {t2}, 59
  word_step t1_5 := extr t2_4 t1_4 59 using extr_lt t2_4 t1_4 59
  -- t2_5: shrd {t2}, {t3}, 59
  word_step t2_5 := extr t3_4 t2_4 59 using extr_lt t3_4 t2_4 59
  -- t3_5: shrd {t3}, {t4}, 59
  word_step t3_5 := extr t4_4 t3_4 59 using extr_lt t4_4 t3_4 59
  -- t4_5: sar {t4}, 59
  word_step t4_5 := asr t4_4 59 using asr_lt t4_4 59 b_t4_4
  subst hres
  -- BEGIN conclusion
  -- The two sides on the integers, in the block's words.
  have hf4 := Signed5.l4_of_abs_lt f hf hfv
  have hg4 := Signed5.l4_of_abs_lt g hg hgv
  have hrep0' : SignMagRep m0' s0' a := by rw [e_m0', e_s0']; exact hrep0
  have hrep1' : SignMagRep m1' s1' b := by rw [e_m1', e_s1']; exact hrep1
  have hA := row_side5 a (by omega) f hf hf4 m0' s0' hrep0'
  have hB := row_side5 b (by omega) g hg hg4 m1' s1' hrep1'
  rw [← e_rax_2, ← e_rax_8, ← e_rax_14, ← e_rax_20, ← e_rax_26, ← e_rax_3, ← e_rax_9,
    ← e_rax_15, ← e_rax_21, ← e_rax_27] at hA
  rw [← e_rax_5, ← e_rax_11, ← e_rax_17, ← e_rax_23, ← e_rax_29, ← e_rax_6, ← e_rax_12,
    ← e_rax_18, ← e_rax_24, ← e_rax_30] at hB
  -- The top column: each complemented sign word times its magnitude is the negated top
  -- correction.
  have hx0 : rax_27 = 0 ∨ rax_27 = 2^64 - 1 := by
    rw [e_rax_27, e_rax_26]; exact eorw_mask _ _ hf4 (SignMagRep.mask _ _ _ hrep0')
  have hx1 : rax_30 = 0 ∨ rax_30 = 2^64 - 1 := by
    rw [e_rax_30, e_rax_29]; exact eorw_mask _ _ hg4 (SignMagRep.mask _ _ _ hrep1')
  obtain ⟨j0, b_j0, hj0⟩ := mask_mul rax_27 m0' hx0 b_m0'
  obtain ⟨j1, b_j1, hj1⟩ := mask_mul rax_30 m1' hx1 b_m1'
  have hj0' : rax_28 + andw rax_27 m0' = 2^64 * j0 := by rw [e_rax_28]; exact hj0
  have hj1' : rax_31 + andw rax_30 m1' = 2^64 * j1 := by rw [e_rax_31]; exact hj1
  -- The row bound on the magnitudes: the low corrections add without a carry, and each column's
  -- products and carry stay below `2^128`, so no column's final `adc` carries out.
  have hrow : m0' + m1' ≤ 2^63 := by rw [e_m0', e_m1']; omega
  have ht01 : t0_1 = andw m0' s0' := by rw [e_t0_1, e_t0]
  have hr1 : rax_1 = andw m1' s1' := by rw [e_rax_1, e_rax]
  have hlo_le : t0_1 ≤ m0' := by rw [ht01]; exact Nat.and_le_left
  have hw_le : rax_1 ≤ m1' := by rw [hr1]; exact Nat.and_le_left
  have hcf : cf = 0 := by clear * - l_t0_2 hlo_le hw_le hrow b_cf; omega
  have hh_3 : rax_3 * m0' ≤ (2^64 - 1) * m0' := Nat.mul_le_mul_right _ (by omega)
  have hh_6 : rax_6 * m1' ≤ (2^64 - 1) * m1' := Nat.mul_le_mul_right _ (by omega)
  have hh_9 : rax_9 * m0' ≤ (2^64 - 1) * m0' := Nat.mul_le_mul_right _ (by omega)
  have hh_12 : rax_12 * m1' ≤ (2^64 - 1) * m1' := Nat.mul_le_mul_right _ (by omega)
  have hh_15 : rax_15 * m0' ≤ (2^64 - 1) * m0' := Nat.mul_le_mul_right _ (by omega)
  have hh_18 : rax_18 * m1' ≤ (2^64 - 1) * m1' := Nat.mul_le_mul_right _ (by omega)
  have hc0 : cf_6 = 0 ∧ cf_8 = 0 := by
    clear * - l_t0_2 hcf hlo_le hw_le hrow l_t0_3 l_t1_1 e_t1 l_t0_4 l_t1_2 d_rdx d_rdx_1 hh_3
      hh_6
    omega
  have hc1 : cf_10 = 0 ∧ cf_12 = 0 := by
    clear * - b_t1_2 hrow l_t1_3 l_t2_1 e_t2 l_t1_4 l_t2_2 d_rdx_2 d_rdx_3 hh_9 hh_12
    omega
  have hc2 : cf_14 = 0 ∧ cf_16 = 0 := by
    clear * - b_t2_2 hrow l_t2_3 l_t3_1 e_t3 l_t2_4 l_t3_2 d_rdx_4 d_rdx_5 hh_15 hh_18
    omega
  -- The five words before the shift, modulo `2^320`: the corrections, the eight products, and
  -- the top column.
  have hS : t0_4 + 2^64 * t1_4 + 2^128 * t2_4 + 2^192 * t3_4 + 2^256 * t4_4
        + 2^320 * (cf_18 + cf_20 + cf_21 + cf_22) + 2^256 * (andw rax_27 m0' + andw rax_30 m1')
      = andw m0' s0' + andw m1' s1'
        + (rax_3 * m0' + 2^64 * (rax_9 * m0') + 2^128 * (rax_15 * m0') + 2^192 * (rax_21 * m0'))
        + (rax_6 * m1' + 2^64 * (rax_12 * m1') + 2^128 * (rax_18 * m1') + 2^192 * (rax_24 * m1'))
        + 2^320 * (j0 + j1) := by
    rw [← ht01, ← hr1]
    clear * - l_t0_2 hcf l_t0_3 l_t1_1 e_t1 l_t0_4 l_t1_2 hc0 l_t1_3 l_t2_1 e_t2 l_t1_4 l_t2_2
      hc1 l_t2_3 l_t3_1 e_t3 l_t2_4 l_t3_2 hc2 l_t3_3 l_t4_1 e_t4 l_t3_4 l_t4_2 l_t4_3 l_t4_4
      hj0' hj1' d_rdx d_rdx_1 d_rdx_2 d_rdx_3 d_rdx_4 d_rdx_5 d_rdx_6 d_rdx_7
    omega
  -- On the integers: the unshifted words are `a f + b g` up to a multiple of `2^320`, which
  -- the bounds of both sides fix.
  have hSz : (t0_4 : ℤ) + 2^64 * t1_4 + 2^128 * t2_4 + 2^192 * t3_4 + 2^256 * t4_4
        + 2^320 * (cf_18 + cf_20 + cf_21 + cf_22) + 2^256 * (andw rax_27 m0' + andw rax_30 m1')
      = andw m0' s0' + andw m1' s1'
        + (rax_3 * m0' + 2^64 * (rax_9 * m0') + 2^128 * (rax_15 * m0') + 2^192 * (rax_21 * m0'))
        + (rax_6 * m1' + 2^64 * (rax_12 * m1') + 2^128 * (rax_18 * m1') + 2^192 * (rax_24 * m1'))
        + 2^320 * (j0 + j1) := by exact_mod_cast hS
  have hval : (t0_4 : ℤ) + 2^64 * t1_4 + 2^128 * t2_4 + 2^192 * t3_4 + 2^256 * t4_4
      = a * f.toInt + b * g.toInt
        + 2^320 * ((j0 + j1 : ℕ) - (cf_18 + cf_20 + cf_21 + cf_22 : ℕ)) := by
    push_cast at hSz hA hB ⊢
    linear_combination hSz + hA + hB
  have hbound : |a * f.toInt + b * g.toInt| < 2^319 := by
    have h := Inversion.row_abs_lt a b _ _ _ _ hab hfv hgv (by positivity)
    rwa [show (2 : ℤ)^63 * 2^256 = 2^319 by norm_num] at h
  rw [abs_lt] at hbound
  have hT : (Signed5.mk t0_4 t1_4 t2_4 t3_4 t4_4).toInt = a * f.toInt + b * g.toInt := by
    rw [show (Signed5.mk t0_4 t1_4 t2_4 t3_4 t4_4).toInt = (t0_4 : ℤ) + 2^64 * t1_4 + 2^128 * t2_4
      + 2^192 * t3_4 + 2^256 * (if t4_4 < 2^63 then (t4_4 : ℤ) else (t4_4 : ℤ) - 2^64) from rfl]
    clear * - hval hbound b_t0_4 b_t1_4 b_t2_4 b_t3_4 b_t4_4 b_j0 b_j1 b_cf_18 b_cf_20 b_cf_21
      b_cf_22
    split_ifs <;> omega
  -- The shift by `59`.
  obtain ⟨hB5, hS5⟩ := shift59_spec t0_4 t1_4 t2_4 t3_4 t4_4 b_t0_4 b_t1_4 b_t2_4 b_t3_4 b_t4_4
  rw [e_t0_5, e_t1_5, e_t2_5, e_t3_5, e_t4_5, hS5, hT]
  exact ⟨hB5, rfl⟩
  -- END conclusion

end PastaCurves.X86_64
