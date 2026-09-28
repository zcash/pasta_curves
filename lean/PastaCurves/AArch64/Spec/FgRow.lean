/-
Copyright Amazon.com, Inc. or its affiliates (the block, adapted from s2n-bignum).
Copyright (c) 2026 the pasta_curves contributors (the transcription and the proofs).
-/
import PastaCurves.AArch64.Spec.UvRow

/-!
# Correctness of the inversion's `f`, `g` row block

See the parent module's documentation for details. The block is the accumulation of the `u`, `v`
row block on five-word signed inputs, followed by the shift right by `59`. The sign word of each
input enters only the top correction: `(f4 ^ s0) & m0`, negated, is `-|a|` exactly when `a f` is
negative, which is the top word of the two's-complement product. The shift is four `extr`s and an
`asr`, which together divide the 320-bit two's-complement value by `2^59`, rounding down.
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
    ∀ r, r = fgRowBlock f g m0 m1 s0 s1 →
      r.Bounded ∧ r.toInt = (a * f.toInt + b * g.toInt) / 2^59 := by
  intro r hr
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
  unfold fgRowBlock at hr
  lift_lets -merge at hr
  -- f0: argument
  extract_lets -merge +onlyGivenNames f0 at hr
  have e_f0 : f0 = f.l0 := rfl
  clear_value f0
  have b_f0 : f0 < 2^64 := by rw [e_f0]; exact hf.1
  -- f1: argument
  extract_lets -merge +onlyGivenNames f1 at hr
  have e_f1 : f1 = f.l1 := rfl
  clear_value f1
  have b_f1 : f1 < 2^64 := by rw [e_f1]; exact hf.2.1
  -- f2: argument
  extract_lets -merge +onlyGivenNames f2 at hr
  have e_f2 : f2 = f.l2 := rfl
  clear_value f2
  have b_f2 : f2 < 2^64 := by rw [e_f2]; exact hf.2.2.1
  -- f3: argument
  extract_lets -merge +onlyGivenNames f3 at hr
  have e_f3 : f3 = f.l3 := rfl
  clear_value f3
  have b_f3 : f3 < 2^64 := by rw [e_f3]; exact hf.2.2.2.1
  -- f4: argument
  extract_lets -merge +onlyGivenNames f4 at hr
  have e_f4 : f4 = f.l4 := rfl
  clear_value f4
  have b_f4 : f4 < 2^64 := by rw [e_f4]; exact hf.2.2.2.2
  -- g0: argument
  extract_lets -merge +onlyGivenNames g0 at hr
  have e_g0 : g0 = g.l0 := rfl
  clear_value g0
  have b_g0 : g0 < 2^64 := by rw [e_g0]; exact hg.1
  -- g1: argument
  extract_lets -merge +onlyGivenNames g1 at hr
  have e_g1 : g1 = g.l1 := rfl
  clear_value g1
  have b_g1 : g1 < 2^64 := by rw [e_g1]; exact hg.2.1
  -- g2: argument
  extract_lets -merge +onlyGivenNames g2 at hr
  have e_g2 : g2 = g.l2 := rfl
  clear_value g2
  have b_g2 : g2 < 2^64 := by rw [e_g2]; exact hg.2.2.1
  -- g3: argument
  extract_lets -merge +onlyGivenNames g3 at hr
  have e_g3 : g3 = g.l3 := rfl
  clear_value g3
  have b_g3 : g3 < 2^64 := by rw [e_g3]; exact hg.2.2.2.1
  -- g4: argument
  extract_lets -merge +onlyGivenNames g4 at hr
  have e_g4 : g4 = g.l4 := rfl
  clear_value g4
  have b_g4 : g4 < 2^64 := by rw [e_g4]; exact hg.2.2.2.2
  -- m0': argument
  extract_lets -merge +onlyGivenNames m0' at hr
  have e_m0' : m0' = m0 := rfl
  clear_value m0'
  have b_m0' : m0' < 2^64 := by rw [e_m0']; exact hm0
  -- m1': argument
  extract_lets -merge +onlyGivenNames m1' at hr
  have e_m1' : m1' = m1 := rfl
  clear_value m1'
  have b_m1' : m1' < 2^64 := by rw [e_m1']; exact hm1
  -- s0': argument
  extract_lets -merge +onlyGivenNames s0' at hr
  have e_s0' : s0' = s0 := rfl
  clear_value s0'
  have b_s0' : s0' < 2^64 := by rw [e_s0']; exact hs0
  -- s1': argument
  extract_lets -merge +onlyGivenNames s1' at hr
  have e_s1' : s1' = s1 := rfl
  clear_value s1'
  have b_s1' : s1' < 2^64 := by rw [e_s1']; exact hs1
  -- lo: and lo,m0,s0
  extract_lets -merge +onlyGivenNames lo at hr
  have e_lo : lo = andw m0' s0' := rfl
  clear_value lo
  have b_lo : lo < 2^64 := by rw [e_lo]; exact andw_lt m0' s0' b_m0' b_s0'
  -- w: and w,m1,s1
  extract_lets -merge +onlyGivenNames w at hr
  have e_w : w = andw m1' s1' := rfl
  clear_value w
  have b_w : w < 2^64 := by rw [e_w]; exact andw_lt m1' s1' b_m1' b_s1'
  -- d0: add d0,lo,w
  extract_lets -merge +onlyGivenNames d0 at hr
  have e_d0 : d0 = addw lo w := rfl
  clear_value d0
  have b_d0 : d0 < 2^64 := by rw [e_d0]; exact addw_lt lo w
  -- w_1: eor w,f0,s0
  extract_lets -merge +onlyGivenNames w_1 at hr
  have e_w_1 : w_1 = eorw f0 s0' := rfl
  clear_value w_1
  have b_w_1 : w_1 < 2^64 := by rw [e_w_1]; exact eorw_lt f0 s0' b_f0 b_s0'
  -- lo_1: mul lo,w,m0
  extract_lets -merge +onlyGivenNames lo_1 at hr
  have e_lo_1 : lo_1 = w_1 * m0' % 2^64 := rfl
  clear_value lo_1
  have b_lo_1 : lo_1 < 2^64 := by rw [e_lo_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_2: umulh w,w,m0
  extract_lets -merge +onlyGivenNames w_2 at hr
  have e_w_2 : w_2 = w_1 * m0' / 2^64 := rfl
  clear_value w_2
  have p_w_2 : w_1 * m0' < 2^64 * 2^64 := Nat.mul_lt_mul'' b_w_1 b_m0'
  have b_w_2 : w_2 < 2^64 := by rw [e_w_2]; exact Nat.div_lt_of_lt_mul p_w_2
  have d_w_2 : lo_1 + 2^64 * w_2 = w_1 * m0' := by
    rw [e_lo_1, e_w_2]; exact Nat.mod_add_div _ _
  clear e_lo_1 e_w_2
  -- d0_1: adds d0,d0,lo
  extract_lets -merge +onlyGivenNames s d0_1 c at hr
  have e_d0_1 : d0_1 = (d0 + lo_1 + 0) % 2^64 := rfl
  have e_c : c = (d0 + lo_1 + 0) / 2^64 := rfl
  clear_value s d0_1 c
  have l_d0_1 : d0_1 + 2^64 * c = d0 + lo_1 + 0 := by
    rw [e_d0_1, e_c]; exact Nat.mod_add_div _ _
  have b_d0_1 : d0_1 < 2^64 := by rw [e_d0_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c : c ≤ 1 := by
    rw [e_c]; exact addc_carry_le_one d0 lo_1 0 b_d0 b_lo_1 (by decide)
  clear e_d0_1 e_c
  -- d1: adc d1,xzr,w
  extract_lets -merge +onlyGivenNames d1 at hr
  have e_d1 : d1 = (0 + w_2 + c) % 2^64 := rfl
  clear_value d1
  have b_d1 : d1 < 2^64 := by rw [e_d1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_d1, b_k_d1, l_d1⟩ :
      ∃ k, k ≤ 1 ∧ d1 + 2^64 * k = 0 + w_2 + c :=
    ⟨(0 + w_2 + c) / 2^64, addc_carry_le_one 0 w_2 c (by decide) b_w_2 b_c,
      by rw [e_d1]; exact Nat.mod_add_div _ _⟩
  clear e_d1
  -- w_3: eor w,g0,s1
  extract_lets -merge +onlyGivenNames w_3 at hr
  have e_w_3 : w_3 = eorw g0 s1' := rfl
  clear_value w_3
  have b_w_3 : w_3 < 2^64 := by rw [e_w_3]; exact eorw_lt g0 s1' b_g0 b_s1'
  -- lo_2: mul lo,w,m1
  extract_lets -merge +onlyGivenNames lo_2 at hr
  have e_lo_2 : lo_2 = w_3 * m1' % 2^64 := rfl
  clear_value lo_2
  have b_lo_2 : lo_2 < 2^64 := by rw [e_lo_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_4: umulh w,w,m1
  extract_lets -merge +onlyGivenNames w_4 at hr
  have e_w_4 : w_4 = w_3 * m1' / 2^64 := rfl
  clear_value w_4
  have p_w_4 : w_3 * m1' < 2^64 * 2^64 := Nat.mul_lt_mul'' b_w_3 b_m1'
  have b_w_4 : w_4 < 2^64 := by rw [e_w_4]; exact Nat.div_lt_of_lt_mul p_w_4
  have d_w_4 : lo_2 + 2^64 * w_4 = w_3 * m1' := by
    rw [e_lo_2, e_w_4]; exact Nat.mod_add_div _ _
  clear e_lo_2 e_w_4
  -- d0_2: adds d0,d0,lo
  extract_lets -merge +onlyGivenNames s_1 d0_2 c_1 at hr
  have e_d0_2 : d0_2 = (d0_1 + lo_2 + 0) % 2^64 := rfl
  have e_c_1 : c_1 = (d0_1 + lo_2 + 0) / 2^64 := rfl
  clear_value s_1 d0_2 c_1
  have l_d0_2 : d0_2 + 2^64 * c_1 = d0_1 + lo_2 + 0 := by
    rw [e_d0_2, e_c_1]; exact Nat.mod_add_div _ _
  have b_d0_2 : d0_2 < 2^64 := by rw [e_d0_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_1 : c_1 ≤ 1 := by
    rw [e_c_1]; exact addc_carry_le_one d0_1 lo_2 0 b_d0_1 b_lo_2 (by decide)
  clear e_d0_2 e_c_1
  -- d1_1: adc d1,d1,w
  extract_lets -merge +onlyGivenNames d1_1 at hr
  have e_d1_1 : d1_1 = (d1 + w_4 + c_1) % 2^64 := rfl
  clear_value d1_1
  have b_d1_1 : d1_1 < 2^64 := by rw [e_d1_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_d1_1, b_k_d1_1, l_d1_1⟩ :
      ∃ k, k ≤ 1 ∧ d1_1 + 2^64 * k = d1 + w_4 + c_1 :=
    ⟨(d1 + w_4 + c_1) / 2^64, addc_carry_le_one d1 w_4 c_1 b_d1 b_w_4 b_c_1,
      by rw [e_d1_1]; exact Nat.mod_add_div _ _⟩
  clear e_d1_1
  -- w_5: eor w,f1,s0
  extract_lets -merge +onlyGivenNames w_5 at hr
  have e_w_5 : w_5 = eorw f1 s0' := rfl
  clear_value w_5
  have b_w_5 : w_5 < 2^64 := by rw [e_w_5]; exact eorw_lt f1 s0' b_f1 b_s0'
  -- lo_3: mul lo,w,m0
  extract_lets -merge +onlyGivenNames lo_3 at hr
  have e_lo_3 : lo_3 = w_5 * m0' % 2^64 := rfl
  clear_value lo_3
  have b_lo_3 : lo_3 < 2^64 := by rw [e_lo_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_6: umulh w,w,m0
  extract_lets -merge +onlyGivenNames w_6 at hr
  have e_w_6 : w_6 = w_5 * m0' / 2^64 := rfl
  clear_value w_6
  have p_w_6 : w_5 * m0' < 2^64 * 2^64 := Nat.mul_lt_mul'' b_w_5 b_m0'
  have b_w_6 : w_6 < 2^64 := by rw [e_w_6]; exact Nat.div_lt_of_lt_mul p_w_6
  have d_w_6 : lo_3 + 2^64 * w_6 = w_5 * m0' := by
    rw [e_lo_3, e_w_6]; exact Nat.mod_add_div _ _
  clear e_lo_3 e_w_6
  -- d1_2: adds d1,d1,lo
  extract_lets -merge +onlyGivenNames s_2 d1_2 c_2 at hr
  have e_d1_2 : d1_2 = (d1_1 + lo_3 + 0) % 2^64 := rfl
  have e_c_2 : c_2 = (d1_1 + lo_3 + 0) / 2^64 := rfl
  clear_value s_2 d1_2 c_2
  have l_d1_2 : d1_2 + 2^64 * c_2 = d1_1 + lo_3 + 0 := by
    rw [e_d1_2, e_c_2]; exact Nat.mod_add_div _ _
  have b_d1_2 : d1_2 < 2^64 := by rw [e_d1_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_2 : c_2 ≤ 1 := by
    rw [e_c_2]; exact addc_carry_le_one d1_1 lo_3 0 b_d1_1 b_lo_3 (by decide)
  clear e_d1_2 e_c_2
  -- d2: adc d2,xzr,w
  extract_lets -merge +onlyGivenNames d2 at hr
  have e_d2 : d2 = (0 + w_6 + c_2) % 2^64 := rfl
  clear_value d2
  have b_d2 : d2 < 2^64 := by rw [e_d2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_d2, b_k_d2, l_d2⟩ :
      ∃ k, k ≤ 1 ∧ d2 + 2^64 * k = 0 + w_6 + c_2 :=
    ⟨(0 + w_6 + c_2) / 2^64, addc_carry_le_one 0 w_6 c_2 (by decide) b_w_6 b_c_2,
      by rw [e_d2]; exact Nat.mod_add_div _ _⟩
  clear e_d2
  -- w_7: eor w,g1,s1
  extract_lets -merge +onlyGivenNames w_7 at hr
  have e_w_7 : w_7 = eorw g1 s1' := rfl
  clear_value w_7
  have b_w_7 : w_7 < 2^64 := by rw [e_w_7]; exact eorw_lt g1 s1' b_g1 b_s1'
  -- lo_4: mul lo,w,m1
  extract_lets -merge +onlyGivenNames lo_4 at hr
  have e_lo_4 : lo_4 = w_7 * m1' % 2^64 := rfl
  clear_value lo_4
  have b_lo_4 : lo_4 < 2^64 := by rw [e_lo_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_8: umulh w,w,m1
  extract_lets -merge +onlyGivenNames w_8 at hr
  have e_w_8 : w_8 = w_7 * m1' / 2^64 := rfl
  clear_value w_8
  have p_w_8 : w_7 * m1' < 2^64 * 2^64 := Nat.mul_lt_mul'' b_w_7 b_m1'
  have b_w_8 : w_8 < 2^64 := by rw [e_w_8]; exact Nat.div_lt_of_lt_mul p_w_8
  have d_w_8 : lo_4 + 2^64 * w_8 = w_7 * m1' := by
    rw [e_lo_4, e_w_8]; exact Nat.mod_add_div _ _
  clear e_lo_4 e_w_8
  -- d1_3: adds d1,d1,lo
  extract_lets -merge +onlyGivenNames s_3 d1_3 c_3 at hr
  have e_d1_3 : d1_3 = (d1_2 + lo_4 + 0) % 2^64 := rfl
  have e_c_3 : c_3 = (d1_2 + lo_4 + 0) / 2^64 := rfl
  clear_value s_3 d1_3 c_3
  have l_d1_3 : d1_3 + 2^64 * c_3 = d1_2 + lo_4 + 0 := by
    rw [e_d1_3, e_c_3]; exact Nat.mod_add_div _ _
  have b_d1_3 : d1_3 < 2^64 := by rw [e_d1_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_3 : c_3 ≤ 1 := by
    rw [e_c_3]; exact addc_carry_le_one d1_2 lo_4 0 b_d1_2 b_lo_4 (by decide)
  clear e_d1_3 e_c_3
  -- d2_1: adc d2,d2,w
  extract_lets -merge +onlyGivenNames d2_1 at hr
  have e_d2_1 : d2_1 = (d2 + w_8 + c_3) % 2^64 := rfl
  clear_value d2_1
  have b_d2_1 : d2_1 < 2^64 := by rw [e_d2_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_d2_1, b_k_d2_1, l_d2_1⟩ :
      ∃ k, k ≤ 1 ∧ d2_1 + 2^64 * k = d2 + w_8 + c_3 :=
    ⟨(d2 + w_8 + c_3) / 2^64, addc_carry_le_one d2 w_8 c_3 b_d2 b_w_8 b_c_3,
      by rw [e_d2_1]; exact Nat.mod_add_div _ _⟩
  clear e_d2_1
  -- d0_3: extr d0,d1,d0,#59
  extract_lets -merge +onlyGivenNames d0_3 at hr
  have e_d0_3 : d0_3 = extr d1_3 d0_2 59 := rfl
  clear_value d0_3
  have b_d0_3 : d0_3 < 2^64 := by rw [e_d0_3]; exact extr_lt d1_3 d0_2 59
  -- w_9: eor w,f2,s0
  extract_lets -merge +onlyGivenNames w_9 at hr
  have e_w_9 : w_9 = eorw f2 s0' := rfl
  clear_value w_9
  have b_w_9 : w_9 < 2^64 := by rw [e_w_9]; exact eorw_lt f2 s0' b_f2 b_s0'
  -- lo_5: mul lo,w,m0
  extract_lets -merge +onlyGivenNames lo_5 at hr
  have e_lo_5 : lo_5 = w_9 * m0' % 2^64 := rfl
  clear_value lo_5
  have b_lo_5 : lo_5 < 2^64 := by rw [e_lo_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_10: umulh w,w,m0
  extract_lets -merge +onlyGivenNames w_10 at hr
  have e_w_10 : w_10 = w_9 * m0' / 2^64 := rfl
  clear_value w_10
  have p_w_10 : w_9 * m0' < 2^64 * 2^64 := Nat.mul_lt_mul'' b_w_9 b_m0'
  have b_w_10 : w_10 < 2^64 := by rw [e_w_10]; exact Nat.div_lt_of_lt_mul p_w_10
  have d_w_10 : lo_5 + 2^64 * w_10 = w_9 * m0' := by
    rw [e_lo_5, e_w_10]; exact Nat.mod_add_div _ _
  clear e_lo_5 e_w_10
  -- d2_2: adds d2,d2,lo
  extract_lets -merge +onlyGivenNames s_4 d2_2 c_4 at hr
  have e_d2_2 : d2_2 = (d2_1 + lo_5 + 0) % 2^64 := rfl
  have e_c_4 : c_4 = (d2_1 + lo_5 + 0) / 2^64 := rfl
  clear_value s_4 d2_2 c_4
  have l_d2_2 : d2_2 + 2^64 * c_4 = d2_1 + lo_5 + 0 := by
    rw [e_d2_2, e_c_4]; exact Nat.mod_add_div _ _
  have b_d2_2 : d2_2 < 2^64 := by rw [e_d2_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_4 : c_4 ≤ 1 := by
    rw [e_c_4]; exact addc_carry_le_one d2_1 lo_5 0 b_d2_1 b_lo_5 (by decide)
  clear e_d2_2 e_c_4
  -- d3: adc d3,xzr,w
  extract_lets -merge +onlyGivenNames d3 at hr
  have e_d3 : d3 = (0 + w_10 + c_4) % 2^64 := rfl
  clear_value d3
  have b_d3 : d3 < 2^64 := by rw [e_d3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_d3, b_k_d3, l_d3⟩ :
      ∃ k, k ≤ 1 ∧ d3 + 2^64 * k = 0 + w_10 + c_4 :=
    ⟨(0 + w_10 + c_4) / 2^64, addc_carry_le_one 0 w_10 c_4 (by decide) b_w_10 b_c_4,
      by rw [e_d3]; exact Nat.mod_add_div _ _⟩
  clear e_d3
  -- w_11: eor w,g2,s1
  extract_lets -merge +onlyGivenNames w_11 at hr
  have e_w_11 : w_11 = eorw g2 s1' := rfl
  clear_value w_11
  have b_w_11 : w_11 < 2^64 := by rw [e_w_11]; exact eorw_lt g2 s1' b_g2 b_s1'
  -- lo_6: mul lo,w,m1
  extract_lets -merge +onlyGivenNames lo_6 at hr
  have e_lo_6 : lo_6 = w_11 * m1' % 2^64 := rfl
  clear_value lo_6
  have b_lo_6 : lo_6 < 2^64 := by rw [e_lo_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_12: umulh w,w,m1
  extract_lets -merge +onlyGivenNames w_12 at hr
  have e_w_12 : w_12 = w_11 * m1' / 2^64 := rfl
  clear_value w_12
  have p_w_12 : w_11 * m1' < 2^64 * 2^64 := Nat.mul_lt_mul'' b_w_11 b_m1'
  have b_w_12 : w_12 < 2^64 := by rw [e_w_12]; exact Nat.div_lt_of_lt_mul p_w_12
  have d_w_12 : lo_6 + 2^64 * w_12 = w_11 * m1' := by
    rw [e_lo_6, e_w_12]; exact Nat.mod_add_div _ _
  clear e_lo_6 e_w_12
  -- d2_3: adds d2,d2,lo
  extract_lets -merge +onlyGivenNames s_5 d2_3 c_5 at hr
  have e_d2_3 : d2_3 = (d2_2 + lo_6 + 0) % 2^64 := rfl
  have e_c_5 : c_5 = (d2_2 + lo_6 + 0) / 2^64 := rfl
  clear_value s_5 d2_3 c_5
  have l_d2_3 : d2_3 + 2^64 * c_5 = d2_2 + lo_6 + 0 := by
    rw [e_d2_3, e_c_5]; exact Nat.mod_add_div _ _
  have b_d2_3 : d2_3 < 2^64 := by rw [e_d2_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_5 : c_5 ≤ 1 := by
    rw [e_c_5]; exact addc_carry_le_one d2_2 lo_6 0 b_d2_2 b_lo_6 (by decide)
  clear e_d2_3 e_c_5
  -- d3_1: adc d3,d3,w
  extract_lets -merge +onlyGivenNames d3_1 at hr
  have e_d3_1 : d3_1 = (d3 + w_12 + c_5) % 2^64 := rfl
  clear_value d3_1
  have b_d3_1 : d3_1 < 2^64 := by rw [e_d3_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_d3_1, b_k_d3_1, l_d3_1⟩ :
      ∃ k, k ≤ 1 ∧ d3_1 + 2^64 * k = d3 + w_12 + c_5 :=
    ⟨(d3 + w_12 + c_5) / 2^64, addc_carry_le_one d3 w_12 c_5 b_d3 b_w_12 b_c_5,
      by rw [e_d3_1]; exact Nat.mod_add_div _ _⟩
  clear e_d3_1
  -- d1_4: extr d1,d2,d1,#59
  extract_lets -merge +onlyGivenNames d1_4 at hr
  have e_d1_4 : d1_4 = extr d2_3 d1_3 59 := rfl
  clear_value d1_4
  have b_d1_4 : d1_4 < 2^64 := by rw [e_d1_4]; exact extr_lt d2_3 d1_3 59
  -- w_13: eor w,f3,s0
  extract_lets -merge +onlyGivenNames w_13 at hr
  have e_w_13 : w_13 = eorw f3 s0' := rfl
  clear_value w_13
  have b_w_13 : w_13 < 2^64 := by rw [e_w_13]; exact eorw_lt f3 s0' b_f3 b_s0'
  -- d4: eor d4,f4,s0
  extract_lets -merge +onlyGivenNames d4 at hr
  have e_d4 : d4 = eorw f4 s0' := rfl
  clear_value d4
  have b_d4 : d4 < 2^64 := by rw [e_d4]; exact eorw_lt f4 s0' b_f4 b_s0'
  -- d4_1: and d4,d4,m0
  extract_lets -merge +onlyGivenNames d4_1 at hr
  have e_d4_1 : d4_1 = andw d4 m0' := rfl
  clear_value d4_1
  have b_d4_1 : d4_1 < 2^64 := by rw [e_d4_1]; exact andw_lt d4 m0' b_d4 b_m0'
  -- d4_2: neg d4,d4
  extract_lets -merge +onlyGivenNames d4_2 at hr
  have e_d4_2 : d4_2 = negw d4_1 := rfl
  clear_value d4_2
  have b_d4_2 : d4_2 < 2^64 := by rw [e_d4_2]; exact negw_lt d4_1
  -- lo_7: mul lo,w,m0
  extract_lets -merge +onlyGivenNames lo_7 at hr
  have e_lo_7 : lo_7 = w_13 * m0' % 2^64 := rfl
  clear_value lo_7
  have b_lo_7 : lo_7 < 2^64 := by rw [e_lo_7]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_14: umulh w,w,m0
  extract_lets -merge +onlyGivenNames w_14 at hr
  have e_w_14 : w_14 = w_13 * m0' / 2^64 := rfl
  clear_value w_14
  have p_w_14 : w_13 * m0' < 2^64 * 2^64 := Nat.mul_lt_mul'' b_w_13 b_m0'
  have b_w_14 : w_14 < 2^64 := by rw [e_w_14]; exact Nat.div_lt_of_lt_mul p_w_14
  have d_w_14 : lo_7 + 2^64 * w_14 = w_13 * m0' := by
    rw [e_lo_7, e_w_14]; exact Nat.mod_add_div _ _
  clear e_lo_7 e_w_14
  -- d3_2: adds d3,d3,lo
  extract_lets -merge +onlyGivenNames s_6 d3_2 c_6 at hr
  have e_d3_2 : d3_2 = (d3_1 + lo_7 + 0) % 2^64 := rfl
  have e_c_6 : c_6 = (d3_1 + lo_7 + 0) / 2^64 := rfl
  clear_value s_6 d3_2 c_6
  have l_d3_2 : d3_2 + 2^64 * c_6 = d3_1 + lo_7 + 0 := by
    rw [e_d3_2, e_c_6]; exact Nat.mod_add_div _ _
  have b_d3_2 : d3_2 < 2^64 := by rw [e_d3_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_6 : c_6 ≤ 1 := by
    rw [e_c_6]; exact addc_carry_le_one d3_1 lo_7 0 b_d3_1 b_lo_7 (by decide)
  clear e_d3_2 e_c_6
  -- d4_3: adc d4,d4,w
  extract_lets -merge +onlyGivenNames d4_3 at hr
  have e_d4_3 : d4_3 = (d4_2 + w_14 + c_6) % 2^64 := rfl
  clear_value d4_3
  have b_d4_3 : d4_3 < 2^64 := by rw [e_d4_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_d4_3, b_k_d4_3, l_d4_3⟩ :
      ∃ k, k ≤ 1 ∧ d4_3 + 2^64 * k = d4_2 + w_14 + c_6 :=
    ⟨(d4_2 + w_14 + c_6) / 2^64, addc_carry_le_one d4_2 w_14 c_6 b_d4_2 b_w_14 b_c_6,
      by rw [e_d4_3]; exact Nat.mod_add_div _ _⟩
  clear e_d4_3
  -- w_15: eor w,g3,s1
  extract_lets -merge +onlyGivenNames w_15 at hr
  have e_w_15 : w_15 = eorw g3 s1' := rfl
  clear_value w_15
  have b_w_15 : w_15 < 2^64 := by rw [e_w_15]; exact eorw_lt g3 s1' b_g3 b_s1'
  -- lo_8: eor lo,g4,s1
  extract_lets -merge +onlyGivenNames lo_8 at hr
  have e_lo_8 : lo_8 = eorw g4 s1' := rfl
  clear_value lo_8
  have b_lo_8 : lo_8 < 2^64 := by rw [e_lo_8]; exact eorw_lt g4 s1' b_g4 b_s1'
  -- lo_9: and lo,lo,m1
  extract_lets -merge +onlyGivenNames lo_9 at hr
  have e_lo_9 : lo_9 = andw lo_8 m1' := rfl
  clear_value lo_9
  have b_lo_9 : lo_9 < 2^64 := by rw [e_lo_9]; exact andw_lt lo_8 m1' b_lo_8 b_m1'
  -- d4_4: sub d4,d4,lo
  extract_lets -merge +onlyGivenNames d4_4 at hr
  have e_d4_4 : d4_4 = subw d4_3 lo_9 := rfl
  clear_value d4_4
  have b_d4_4 : d4_4 < 2^64 := by rw [e_d4_4]; exact subw_lt d4_3 lo_9
  -- lo_10: mul lo,w,m1
  extract_lets -merge +onlyGivenNames lo_10 at hr
  have e_lo_10 : lo_10 = w_15 * m1' % 2^64 := rfl
  clear_value lo_10
  have b_lo_10 : lo_10 < 2^64 := by rw [e_lo_10]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_16: umulh w,w,m1
  extract_lets -merge +onlyGivenNames w_16 at hr
  have e_w_16 : w_16 = w_15 * m1' / 2^64 := rfl
  clear_value w_16
  have p_w_16 : w_15 * m1' < 2^64 * 2^64 := Nat.mul_lt_mul'' b_w_15 b_m1'
  have b_w_16 : w_16 < 2^64 := by rw [e_w_16]; exact Nat.div_lt_of_lt_mul p_w_16
  have d_w_16 : lo_10 + 2^64 * w_16 = w_15 * m1' := by
    rw [e_lo_10, e_w_16]; exact Nat.mod_add_div _ _
  clear e_lo_10 e_w_16
  -- d3_3: adds d3,d3,lo
  extract_lets -merge +onlyGivenNames s_7 d3_3 c_7 at hr
  have e_d3_3 : d3_3 = (d3_2 + lo_10 + 0) % 2^64 := rfl
  have e_c_7 : c_7 = (d3_2 + lo_10 + 0) / 2^64 := rfl
  clear_value s_7 d3_3 c_7
  have l_d3_3 : d3_3 + 2^64 * c_7 = d3_2 + lo_10 + 0 := by
    rw [e_d3_3, e_c_7]; exact Nat.mod_add_div _ _
  have b_d3_3 : d3_3 < 2^64 := by rw [e_d3_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_7 : c_7 ≤ 1 := by
    rw [e_c_7]; exact addc_carry_le_one d3_2 lo_10 0 b_d3_2 b_lo_10 (by decide)
  clear e_d3_3 e_c_7
  -- d4_5: adc d4,d4,w
  extract_lets -merge +onlyGivenNames d4_5 at hr
  have e_d4_5 : d4_5 = (d4_4 + w_16 + c_7) % 2^64 := rfl
  clear_value d4_5
  have b_d4_5 : d4_5 < 2^64 := by rw [e_d4_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_d4_5, b_k_d4_5, l_d4_5⟩ :
      ∃ k, k ≤ 1 ∧ d4_5 + 2^64 * k = d4_4 + w_16 + c_7 :=
    ⟨(d4_4 + w_16 + c_7) / 2^64, addc_carry_le_one d4_4 w_16 c_7 b_d4_4 b_w_16 b_c_7,
      by rw [e_d4_5]; exact Nat.mod_add_div _ _⟩
  clear e_d4_5
  -- d2_4: extr d2,d3,d2,#59
  extract_lets -merge +onlyGivenNames d2_4 at hr
  have e_d2_4 : d2_4 = extr d3_3 d2_3 59 := rfl
  clear_value d2_4
  have b_d2_4 : d2_4 < 2^64 := by rw [e_d2_4]; exact extr_lt d3_3 d2_3 59
  -- d3_4: extr d3,d4,d3,#59
  extract_lets -merge +onlyGivenNames d3_4 at hr
  have e_d3_4 : d3_4 = extr d4_5 d3_3 59 := rfl
  clear_value d3_4
  have b_d3_4 : d3_4 < 2^64 := by rw [e_d3_4]; exact extr_lt d4_5 d3_3 59
  -- d4_6: asr d4,d4,#59
  extract_lets -merge +onlyGivenNames d4_6 at hr
  have e_d4_6 : d4_6 = asr d4_5 59 := rfl
  clear_value d4_6
  have b_d4_6 : d4_6 < 2^64 := by rw [e_d4_6]; exact asr_lt d4_5 59 b_d4_5
  subst hr
  -- BEGIN conclusion
  -- The two sides on the integers, in the block's words.
  have hf4 := Signed5.l4_of_abs_lt f hf hfv
  have hg4 := Signed5.l4_of_abs_lt g hg hgv
  have hA := row_side5 a (by omega) f hf hf4 m0' s0' (by rw [e_m0', e_s0']; exact hrep0)
  have hB := row_side5 b (by omega) g hg hg4 m1' s1' (by rw [e_m1', e_s1']; exact hrep1)
  rw [← e_f0, ← e_f1, ← e_f2, ← e_f3, ← e_f4, ← e_w_1, ← e_w_5, ← e_w_9, ← e_w_13, ← e_lo, ← e_d4,
    ← e_d4_1] at hA
  rw [← e_g0, ← e_g1, ← e_g2, ← e_g3, ← e_g4, ← e_w_3, ← e_w_7, ← e_w_11, ← e_w_15, ← e_w, ← e_lo_8,
    ← e_lo_9] at hB
  -- The row bound on the magnitudes: the low corrections add without a carry, and each high
  -- half is at most its magnitude, so no column's `adc` overflows.
  have hrow : m0' + m1' ≤ 2^63 := by rw [e_m0', e_m1']; omega
  have hlo_le : lo ≤ m0' := by rw [e_lo]; exact Nat.and_le_left
  have hw_le : w ≤ m1' := by rw [e_w]; exact Nat.and_le_left
  have hd0 : d0 = lo + w := by
    rw [e_d0]; show (lo + w) % 2^64 = lo + w; exact Nat.mod_eq_of_lt (by omega)
  have hh_1 : w_1 * m0' ≤ (2^64 - 1) * m0' := Nat.mul_le_mul_right _ (by omega)
  have hh_3 : w_3 * m1' ≤ (2^64 - 1) * m1' := Nat.mul_le_mul_right _ (by omega)
  have hh_5 : w_5 * m0' ≤ (2^64 - 1) * m0' := Nat.mul_le_mul_right _ (by omega)
  have hh_7 : w_7 * m1' ≤ (2^64 - 1) * m1' := Nat.mul_le_mul_right _ (by omega)
  have hh_9 : w_9 * m0' ≤ (2^64 - 1) * m0' := Nat.mul_le_mul_right _ (by omega)
  have hh_11 : w_11 * m1' ≤ (2^64 - 1) * m1' := Nat.mul_le_mul_right _ (by omega)
  have hh_13 : w_13 * m0' ≤ (2^64 - 1) * m0' := Nat.mul_le_mul_right _ (by omega)
  have hh_15 : w_15 * m1' ≤ (2^64 - 1) * m1' := Nat.mul_le_mul_right _ (by omega)
  have hk_d1 : k_d1 = 0 := by
    clear * - l_d1 b_k_d1 d_w_2 hh_1 hrow b_c; omega
  have hk_d1_1 : k_d1_1 = 0 := by
    clear * - l_d1_1 b_k_d1_1 l_d1 hk_d1 d_w_2 hh_1 d_w_4 hh_3 hrow b_c b_c_1; omega
  have hk_d2 : k_d2 = 0 := by
    clear * - l_d2 b_k_d2 d_w_6 hh_5 hrow b_c_2; omega
  have hk_d2_1 : k_d2_1 = 0 := by
    clear * - l_d2_1 b_k_d2_1 l_d2 hk_d2 d_w_6 hh_5 d_w_8 hh_7 hrow b_c_2 b_c_3; omega
  have hk_d3 : k_d3 = 0 := by
    clear * - l_d3 b_k_d3 d_w_10 hh_9 hrow b_c_4; omega
  have hk_d3_1 : k_d3_1 = 0 := by
    clear * - l_d3_1 b_k_d3_1 l_d3 hk_d3 d_w_10 hh_9 d_w_12 hh_11 hrow b_c_4 b_c_5; omega
  -- The top word's negation and subtraction, modulo `2^64`.
  obtain ⟨j_d4_2, b_j_d4_2, l_d4_2⟩ : ∃ j, j ≤ 1 ∧ d4_2 + 2^64 * j = 2^64 - d4_1 := by
    have h : d4_2 = (2^64 - d4_1) % 2^64 := e_d4_2
    clear * - h b_d4_1
    exact ⟨(2^64 - d4_1) / 2^64, by omega, by omega⟩
  obtain ⟨j_d4_4, b_j_d4_4, l_d4_4⟩ : ∃ j, j ≤ 1 ∧ d4_4 + 2^64 * j = d4_3 + 2^64 - lo_9 := by
    have h : d4_4 = (d4_3 + 2^64 - lo_9) % 2^64 := e_d4_4
    clear * - h b_d4_3 b_lo_9
    exact ⟨(d4_3 + 2^64 - lo_9) / 2^64, by omega, by omega⟩
  -- The five words before the shift, modulo `2^320`: the corrections, the eight products, and
  -- `2^321` from the two negations in the top word.
  have hS : d0_2 + 2^64 * d1_3 + 2^128 * d2_3 + 2^192 * d3_3 + 2^256 * d4_5
        + 2^320 * (j_d4_2 + j_d4_4 + k_d4_3 + k_d4_5) + 2^256 * (d4_1 + lo_9)
      = lo + w
        + (w_1 * m0' + 2^64 * (w_5 * m0') + 2^128 * (w_9 * m0') + 2^192 * (w_13 * m0'))
        + (w_3 * m1' + 2^64 * (w_7 * m1') + 2^128 * (w_11 * m1') + 2^192 * (w_15 * m1'))
        + 2^321 := by
    clear * - hd0 l_d0_1 l_d1 l_d0_2 l_d1_1 l_d1_2 l_d2 l_d1_3 l_d2_1 l_d2_2 l_d3 l_d2_3 l_d3_1
      l_d3_2 l_d4_3 l_d3_3 l_d4_5 l_d4_2 l_d4_4 d_w_2 d_w_4 d_w_6 d_w_8 d_w_10 d_w_12 d_w_14
      d_w_16 hk_d1 hk_d1_1 hk_d2 hk_d2_1 hk_d3 hk_d3_1 b_d4_1 b_lo_9
    omega
  -- On the integers: the unshifted words are `a f + b g` up to a multiple of `2^320`, which
  -- the bounds of both sides fix.
  have hSz : (d0_2 : ℤ) + 2^64 * d1_3 + 2^128 * d2_3 + 2^192 * d3_3 + 2^256 * d4_5
        + 2^320 * (j_d4_2 + j_d4_4 + k_d4_3 + k_d4_5) + 2^256 * (d4_1 + lo_9)
      = lo + w
        + (w_1 * m0' + 2^64 * (w_5 * m0') + 2^128 * (w_9 * m0') + 2^192 * (w_13 * m0'))
        + (w_3 * m1' + 2^64 * (w_7 * m1') + 2^128 * (w_11 * m1') + 2^192 * (w_15 * m1'))
        + 2^321 := by exact_mod_cast hS
  have hval : (d0_2 : ℤ) + 2^64 * d1_3 + 2^128 * d2_3 + 2^192 * d3_3 + 2^256 * d4_5
      = a * f.toInt + b * g.toInt
        + 2^320 * (2 - (j_d4_2 + j_d4_4 + k_d4_3 + k_d4_5 : ℕ)) := by
    push_cast at hSz hA hB ⊢
    linear_combination hSz + hA + hB
  have hbound : |a * f.toInt + b * g.toInt| < 2^319 := by
    have h := Inversion.row_abs_lt a b _ _ _ _ hab hfv hgv (by positivity)
    rwa [show (2 : ℤ)^63 * 2^256 = 2^319 by norm_num] at h
  rw [abs_lt] at hbound
  have hT : (Signed5.mk d0_2 d1_3 d2_3 d3_3 d4_5).toInt = a * f.toInt + b * g.toInt := by
    rw [show (Signed5.mk d0_2 d1_3 d2_3 d3_3 d4_5).toInt = (d0_2 : ℤ) + 2^64 * d1_3 + 2^128 * d2_3
      + 2^192 * d3_3 + 2^256 * (if d4_5 < 2^63 then (d4_5 : ℤ) else (d4_5 : ℤ) - 2^64) from rfl]
    clear * - hval hbound b_d0_2 b_d1_3 b_d2_3 b_d3_3 b_d4_5 b_j_d4_2 b_j_d4_4 b_k_d4_3 b_k_d4_5
    split_ifs <;> omega
  -- The shift by `59`.
  obtain ⟨hB5, hS5⟩ := shift59_spec d0_2 d1_3 d2_3 d3_3 d4_5 b_d0_2 b_d1_3 b_d2_3 b_d3_3 b_d4_5
  rw [e_d0_3, e_d1_4, e_d2_4, e_d3_4, e_d4_6, hS5, hT]
  exact ⟨hB5, rfl⟩
  -- END conclusion

end PastaCurves.AArch64
