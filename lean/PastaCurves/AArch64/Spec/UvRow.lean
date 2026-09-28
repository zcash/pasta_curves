/-
Copyright Amazon.com, Inc. or its affiliates (the block, adapted from s2n-bignum).
Copyright (c) 2026 the pasta_curves contributors (the transcription and the proofs).
-/
import PastaCurves.AArch64.Spec.SignMag
import PastaCurves.Inversion.Round

/-!
# Correctness of the inversion's `u`, `v` row block

See the parent module's documentation for details. The block forms `a u + b v` as a five-word signed
value from the sign-magnitude forms of `a` and `b`: each word of `u` is complemented by the mask,
the products are accumulated, and the two's-complement corrections (`|a|` in the low word, `-|a|` in
the top word for a negative `a`, likewise for `b`) turn the products of complements into the signed
products. The row bound `|a| + |b| ≤ 2^63` keeps every column's carry within the next word.
-/

set_option exponentiation.threshold 400

namespace PastaCurves.AArch64

open Inversion (SignMagRep)

-- BEGIN uvRowBlock_spec lemmas
/-- `eor` with the all-ones word complements a word. -/
theorem eorw_ones (x : ℕ) (hx : x < 2^64) : eorw x (2^64 - 1) = 2^64 - 1 - x := by
  unfold eorw
  rw [show 2^64 - 1 - x = 2^64 - (x + 1) by omega]
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_xor, Nat.testBit_two_pow_sub_one, Nat.testBit_two_pow_sub_succ hx]
  by_cases hi : i < 64
  · simp [hi]
  · have hxi : x < 2^i := lt_of_lt_of_le hx (Nat.pow_le_pow_right (by decide) (by omega))
    simp [hi, Nat.testBit_lt_two_pow hxi]

/-- **Why the block's corrections give the signed product:** the block multiplies by the
magnitude `|z|` and makes the product signed by complementing the words of `x` when `z` is
negative. Since `|z| (2^256 - 1 - x) = 2^256 |z| - |z| - |z| x`, adding `|z|` in the low word and
subtracting `2^256 |z|` in the top word leaves `-|z| x = z x`. When `z` is nonnegative the mask
is zero, the words are unchanged, and both corrections vanish. This is that identity for one
side of a row, so the block proof only accounts for the carries. -/
theorem row_side (z : ℤ) (hz : |z| < 2^64) (x : Limbs) (hx : x.Bounded) (m s : ℕ)
    (hrep : SignMagRep m s z) :
    ((eorw x.l0 s + 2^64 * eorw x.l1 s + 2^128 * eorw x.l2 s + 2^192 * eorw x.l3 s : ℕ) : ℤ)
        * m + (andw m s : ℤ) - 2^256 * (andw s m : ℤ) = z * x.toNat := by
  obtain ⟨h0, h1, h2, h3⟩ := hx
  obtain ⟨hm64, -⟩ := hrep.lt _ _ _ hz
  unfold Limbs.toNat
  rcases hrep with ⟨hs, hm⟩ | ⟨hs, hm⟩
  · subst hs
    simp only [eorw, andw, Nat.xor_zero, Nat.and_zero, Nat.zero_and]
    push_cast
    rw [← hm]
    ring
  · subst hs
    have a1 : andw m (2^64 - 1) = m := by
      unfold andw; rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hm64]
    have a2 : andw (2^64 - 1) m = m := by rw [andw, Nat.and_comm]; exact a1
    have hsum : ((eorw x.l0 (2^64 - 1) + 2^64 * eorw x.l1 (2^64 - 1)
        + 2^128 * eorw x.l2 (2^64 - 1) + 2^192 * eorw x.l3 (2^64 - 1) : ℕ) : ℤ)
        = 2^256 - 1 - (x.l0 + 2^64 * x.l1 + 2^128 * x.l2 + 2^192 * x.l3) := by
      rw [eorw_ones _ h0, eorw_ones _ h1, eorw_ones _ h2, eorw_ones _ h3]; omega
    rw [hsum, a1, a2]
    push_cast
    rw [hm]
    ring

-- END uvRowBlock_spec lemmas

-- BEGIN uvRowBlock_spec statement
/-- The row `a u + b v` of `updateUV` before its `amontred`, as the exact integer in five words.
The row bound `|a| + |b| ≤ 2^63` is what keeps every column's carry within the next word; the
rows of a 59-step matrix are at most `2^59`. `row_side` supplies the sign handling. -/
theorem uvRowBlock_spec (a b : ℤ) (u v : Limbs) (m0 m1 s0 s1 : Nat)
    (hu : u.Bounded) (hv : v.Bounded) (hab : |a| + |b| ≤ 2^63)
    (hrep0 : SignMagRep m0 s0 a) (hrep1 : SignMagRep m1 s1 b) :
    ∀ r, r = uvRowBlock u v m0 m1 s0 s1 →
      r.Bounded ∧ r.toInt = a * u.toNat + b * v.toNat := by
  intro r hr
  have ha0 := abs_nonneg a
  have hb0 := abs_nonneg b
  have hm0z : (m0 : ℤ) = |a| := hrep0.natCast_eq_abs _ _ _
  have hm1z : (m1 : ℤ) = |b| := hrep1.natCast_eq_abs _ _ _
  have hm0 : m0 < 2^64 := by omega
  have hm1 : m1 < 2^64 := by omega
  have hs0 : s0 < 2^64 := (hrep0.lt _ _ _ (by omega)).2
  have hs1 : s1 < 2^64 := (hrep1.lt _ _ _ (by omega)).2
-- END uvRowBlock_spec statement
  -- generated skeleton for `uvRowBlock`: do not edit between the annotations
  unfold uvRowBlock at hr
  lift_lets -merge at hr
  -- u0: argument
  extract_lets -merge +onlyGivenNames u0 at hr
  have e_u0 : u0 = u.l0 := rfl
  clear_value u0
  have b_u0 : u0 < 2^64 := by rw [e_u0]; exact hu.1
  -- u1: argument
  extract_lets -merge +onlyGivenNames u1 at hr
  have e_u1 : u1 = u.l1 := rfl
  clear_value u1
  have b_u1 : u1 < 2^64 := by rw [e_u1]; exact hu.2.1
  -- u2: argument
  extract_lets -merge +onlyGivenNames u2 at hr
  have e_u2 : u2 = u.l2 := rfl
  clear_value u2
  have b_u2 : u2 < 2^64 := by rw [e_u2]; exact hu.2.2.1
  -- u3: argument
  extract_lets -merge +onlyGivenNames u3 at hr
  have e_u3 : u3 = u.l3 := rfl
  clear_value u3
  have b_u3 : u3 < 2^64 := by rw [e_u3]; exact hu.2.2.2
  -- v0: argument
  extract_lets -merge +onlyGivenNames v0 at hr
  have e_v0 : v0 = v.l0 := rfl
  clear_value v0
  have b_v0 : v0 < 2^64 := by rw [e_v0]; exact hv.1
  -- v1: argument
  extract_lets -merge +onlyGivenNames v1 at hr
  have e_v1 : v1 = v.l1 := rfl
  clear_value v1
  have b_v1 : v1 < 2^64 := by rw [e_v1]; exact hv.2.1
  -- v2: argument
  extract_lets -merge +onlyGivenNames v2 at hr
  have e_v2 : v2 = v.l2 := rfl
  clear_value v2
  have b_v2 : v2 < 2^64 := by rw [e_v2]; exact hv.2.2.1
  -- v3: argument
  extract_lets -merge +onlyGivenNames v3 at hr
  have e_v3 : v3 = v.l3 := rfl
  clear_value v3
  have b_v3 : v3 < 2^64 := by rw [e_v3]; exact hv.2.2.2
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
  -- t0: add t0,lo,w
  extract_lets -merge +onlyGivenNames t0 at hr
  have e_t0 : t0 = addw lo w := rfl
  clear_value t0
  have b_t0 : t0 < 2^64 := by rw [e_t0]; exact addw_lt lo w
  -- w_1: eor w,u0,s0
  extract_lets -merge +onlyGivenNames w_1 at hr
  have e_w_1 : w_1 = eorw u0 s0' := rfl
  clear_value w_1
  have b_w_1 : w_1 < 2^64 := by rw [e_w_1]; exact eorw_lt u0 s0' b_u0 b_s0'
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
  -- t0_1: adds t0,t0,lo
  extract_lets -merge +onlyGivenNames s t0_1 c at hr
  have e_t0_1 : t0_1 = (t0 + lo_1 + 0) % 2^64 := rfl
  have e_c : c = (t0 + lo_1 + 0) / 2^64 := rfl
  clear_value s t0_1 c
  have l_t0_1 : t0_1 + 2^64 * c = t0 + lo_1 + 0 := by
    rw [e_t0_1, e_c]; exact Nat.mod_add_div _ _
  have b_t0_1 : t0_1 < 2^64 := by rw [e_t0_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c : c ≤ 1 := by
    rw [e_c]; exact addc_carry_le_one t0 lo_1 0 b_t0 b_lo_1 (by decide)
  clear e_t0_1 e_c
  -- t1: adc t1,xzr,w
  extract_lets -merge +onlyGivenNames t1 at hr
  have e_t1 : t1 = (0 + w_2 + c) % 2^64 := rfl
  clear_value t1
  have b_t1 : t1 < 2^64 := by rw [e_t1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t1, b_k_t1, l_t1⟩ :
      ∃ k, k ≤ 1 ∧ t1 + 2^64 * k = 0 + w_2 + c :=
    ⟨(0 + w_2 + c) / 2^64, addc_carry_le_one 0 w_2 c (by decide) b_w_2 b_c,
      by rw [e_t1]; exact Nat.mod_add_div _ _⟩
  clear e_t1
  -- w_3: eor w,v0,s1
  extract_lets -merge +onlyGivenNames w_3 at hr
  have e_w_3 : w_3 = eorw v0 s1' := rfl
  clear_value w_3
  have b_w_3 : w_3 < 2^64 := by rw [e_w_3]; exact eorw_lt v0 s1' b_v0 b_s1'
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
  -- t0_2: adds t0,t0,lo
  extract_lets -merge +onlyGivenNames s_1 t0_2 c_1 at hr
  have e_t0_2 : t0_2 = (t0_1 + lo_2 + 0) % 2^64 := rfl
  have e_c_1 : c_1 = (t0_1 + lo_2 + 0) / 2^64 := rfl
  clear_value s_1 t0_2 c_1
  have l_t0_2 : t0_2 + 2^64 * c_1 = t0_1 + lo_2 + 0 := by
    rw [e_t0_2, e_c_1]; exact Nat.mod_add_div _ _
  have b_t0_2 : t0_2 < 2^64 := by rw [e_t0_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_1 : c_1 ≤ 1 := by
    rw [e_c_1]; exact addc_carry_le_one t0_1 lo_2 0 b_t0_1 b_lo_2 (by decide)
  clear e_t0_2 e_c_1
  -- t1_1: adc t1,t1,w
  extract_lets -merge +onlyGivenNames t1_1 at hr
  have e_t1_1 : t1_1 = (t1 + w_4 + c_1) % 2^64 := rfl
  clear_value t1_1
  have b_t1_1 : t1_1 < 2^64 := by rw [e_t1_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t1_1, b_k_t1_1, l_t1_1⟩ :
      ∃ k, k ≤ 1 ∧ t1_1 + 2^64 * k = t1 + w_4 + c_1 :=
    ⟨(t1 + w_4 + c_1) / 2^64, addc_carry_le_one t1 w_4 c_1 b_t1 b_w_4 b_c_1,
      by rw [e_t1_1]; exact Nat.mod_add_div _ _⟩
  clear e_t1_1
  -- w_5: eor w,u1,s0
  extract_lets -merge +onlyGivenNames w_5 at hr
  have e_w_5 : w_5 = eorw u1 s0' := rfl
  clear_value w_5
  have b_w_5 : w_5 < 2^64 := by rw [e_w_5]; exact eorw_lt u1 s0' b_u1 b_s0'
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
  -- t1_2: adds t1,t1,lo
  extract_lets -merge +onlyGivenNames s_2 t1_2 c_2 at hr
  have e_t1_2 : t1_2 = (t1_1 + lo_3 + 0) % 2^64 := rfl
  have e_c_2 : c_2 = (t1_1 + lo_3 + 0) / 2^64 := rfl
  clear_value s_2 t1_2 c_2
  have l_t1_2 : t1_2 + 2^64 * c_2 = t1_1 + lo_3 + 0 := by
    rw [e_t1_2, e_c_2]; exact Nat.mod_add_div _ _
  have b_t1_2 : t1_2 < 2^64 := by rw [e_t1_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_2 : c_2 ≤ 1 := by
    rw [e_c_2]; exact addc_carry_le_one t1_1 lo_3 0 b_t1_1 b_lo_3 (by decide)
  clear e_t1_2 e_c_2
  -- t2: adc t2,xzr,w
  extract_lets -merge +onlyGivenNames t2 at hr
  have e_t2 : t2 = (0 + w_6 + c_2) % 2^64 := rfl
  clear_value t2
  have b_t2 : t2 < 2^64 := by rw [e_t2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t2, b_k_t2, l_t2⟩ :
      ∃ k, k ≤ 1 ∧ t2 + 2^64 * k = 0 + w_6 + c_2 :=
    ⟨(0 + w_6 + c_2) / 2^64, addc_carry_le_one 0 w_6 c_2 (by decide) b_w_6 b_c_2,
      by rw [e_t2]; exact Nat.mod_add_div _ _⟩
  clear e_t2
  -- w_7: eor w,v1,s1
  extract_lets -merge +onlyGivenNames w_7 at hr
  have e_w_7 : w_7 = eorw v1 s1' := rfl
  clear_value w_7
  have b_w_7 : w_7 < 2^64 := by rw [e_w_7]; exact eorw_lt v1 s1' b_v1 b_s1'
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
  -- t1_3: adds t1,t1,lo
  extract_lets -merge +onlyGivenNames s_3 t1_3 c_3 at hr
  have e_t1_3 : t1_3 = (t1_2 + lo_4 + 0) % 2^64 := rfl
  have e_c_3 : c_3 = (t1_2 + lo_4 + 0) / 2^64 := rfl
  clear_value s_3 t1_3 c_3
  have l_t1_3 : t1_3 + 2^64 * c_3 = t1_2 + lo_4 + 0 := by
    rw [e_t1_3, e_c_3]; exact Nat.mod_add_div _ _
  have b_t1_3 : t1_3 < 2^64 := by rw [e_t1_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_3 : c_3 ≤ 1 := by
    rw [e_c_3]; exact addc_carry_le_one t1_2 lo_4 0 b_t1_2 b_lo_4 (by decide)
  clear e_t1_3 e_c_3
  -- t2_1: adc t2,t2,w
  extract_lets -merge +onlyGivenNames t2_1 at hr
  have e_t2_1 : t2_1 = (t2 + w_8 + c_3) % 2^64 := rfl
  clear_value t2_1
  have b_t2_1 : t2_1 < 2^64 := by rw [e_t2_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t2_1, b_k_t2_1, l_t2_1⟩ :
      ∃ k, k ≤ 1 ∧ t2_1 + 2^64 * k = t2 + w_8 + c_3 :=
    ⟨(t2 + w_8 + c_3) / 2^64, addc_carry_le_one t2 w_8 c_3 b_t2 b_w_8 b_c_3,
      by rw [e_t2_1]; exact Nat.mod_add_div _ _⟩
  clear e_t2_1
  -- w_9: eor w,u2,s0
  extract_lets -merge +onlyGivenNames w_9 at hr
  have e_w_9 : w_9 = eorw u2 s0' := rfl
  clear_value w_9
  have b_w_9 : w_9 < 2^64 := by rw [e_w_9]; exact eorw_lt u2 s0' b_u2 b_s0'
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
  -- t2_2: adds t2,t2,lo
  extract_lets -merge +onlyGivenNames s_4 t2_2 c_4 at hr
  have e_t2_2 : t2_2 = (t2_1 + lo_5 + 0) % 2^64 := rfl
  have e_c_4 : c_4 = (t2_1 + lo_5 + 0) / 2^64 := rfl
  clear_value s_4 t2_2 c_4
  have l_t2_2 : t2_2 + 2^64 * c_4 = t2_1 + lo_5 + 0 := by
    rw [e_t2_2, e_c_4]; exact Nat.mod_add_div _ _
  have b_t2_2 : t2_2 < 2^64 := by rw [e_t2_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_4 : c_4 ≤ 1 := by
    rw [e_c_4]; exact addc_carry_le_one t2_1 lo_5 0 b_t2_1 b_lo_5 (by decide)
  clear e_t2_2 e_c_4
  -- t3: adc t3,xzr,w
  extract_lets -merge +onlyGivenNames t3 at hr
  have e_t3 : t3 = (0 + w_10 + c_4) % 2^64 := rfl
  clear_value t3
  have b_t3 : t3 < 2^64 := by rw [e_t3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t3, b_k_t3, l_t3⟩ :
      ∃ k, k ≤ 1 ∧ t3 + 2^64 * k = 0 + w_10 + c_4 :=
    ⟨(0 + w_10 + c_4) / 2^64, addc_carry_le_one 0 w_10 c_4 (by decide) b_w_10 b_c_4,
      by rw [e_t3]; exact Nat.mod_add_div _ _⟩
  clear e_t3
  -- w_11: eor w,v2,s1
  extract_lets -merge +onlyGivenNames w_11 at hr
  have e_w_11 : w_11 = eorw v2 s1' := rfl
  clear_value w_11
  have b_w_11 : w_11 < 2^64 := by rw [e_w_11]; exact eorw_lt v2 s1' b_v2 b_s1'
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
  -- t2_3: adds t2,t2,lo
  extract_lets -merge +onlyGivenNames s_5 t2_3 c_5 at hr
  have e_t2_3 : t2_3 = (t2_2 + lo_6 + 0) % 2^64 := rfl
  have e_c_5 : c_5 = (t2_2 + lo_6 + 0) / 2^64 := rfl
  clear_value s_5 t2_3 c_5
  have l_t2_3 : t2_3 + 2^64 * c_5 = t2_2 + lo_6 + 0 := by
    rw [e_t2_3, e_c_5]; exact Nat.mod_add_div _ _
  have b_t2_3 : t2_3 < 2^64 := by rw [e_t2_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_5 : c_5 ≤ 1 := by
    rw [e_c_5]; exact addc_carry_le_one t2_2 lo_6 0 b_t2_2 b_lo_6 (by decide)
  clear e_t2_3 e_c_5
  -- t3_1: adc t3,t3,w
  extract_lets -merge +onlyGivenNames t3_1 at hr
  have e_t3_1 : t3_1 = (t3 + w_12 + c_5) % 2^64 := rfl
  clear_value t3_1
  have b_t3_1 : t3_1 < 2^64 := by rw [e_t3_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t3_1, b_k_t3_1, l_t3_1⟩ :
      ∃ k, k ≤ 1 ∧ t3_1 + 2^64 * k = t3 + w_12 + c_5 :=
    ⟨(t3 + w_12 + c_5) / 2^64, addc_carry_le_one t3 w_12 c_5 b_t3 b_w_12 b_c_5,
      by rw [e_t3_1]; exact Nat.mod_add_div _ _⟩
  clear e_t3_1
  -- w_13: eor w,u3,s0
  extract_lets -merge +onlyGivenNames w_13 at hr
  have e_w_13 : w_13 = eorw u3 s0' := rfl
  clear_value w_13
  have b_w_13 : w_13 < 2^64 := by rw [e_w_13]; exact eorw_lt u3 s0' b_u3 b_s0'
  -- t4: and t4,s0,m0
  extract_lets -merge +onlyGivenNames t4 at hr
  have e_t4 : t4 = andw s0' m0' := rfl
  clear_value t4
  have b_t4 : t4 < 2^64 := by rw [e_t4]; exact andw_lt s0' m0' b_s0' b_m0'
  -- t4_1: neg t4,t4
  extract_lets -merge +onlyGivenNames t4_1 at hr
  have e_t4_1 : t4_1 = negw t4 := rfl
  clear_value t4_1
  have b_t4_1 : t4_1 < 2^64 := by rw [e_t4_1]; exact negw_lt t4
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
  -- t3_2: adds t3,t3,lo
  extract_lets -merge +onlyGivenNames s_6 t3_2 c_6 at hr
  have e_t3_2 : t3_2 = (t3_1 + lo_7 + 0) % 2^64 := rfl
  have e_c_6 : c_6 = (t3_1 + lo_7 + 0) / 2^64 := rfl
  clear_value s_6 t3_2 c_6
  have l_t3_2 : t3_2 + 2^64 * c_6 = t3_1 + lo_7 + 0 := by
    rw [e_t3_2, e_c_6]; exact Nat.mod_add_div _ _
  have b_t3_2 : t3_2 < 2^64 := by rw [e_t3_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_6 : c_6 ≤ 1 := by
    rw [e_c_6]; exact addc_carry_le_one t3_1 lo_7 0 b_t3_1 b_lo_7 (by decide)
  clear e_t3_2 e_c_6
  -- t4_2: adc t4,t4,w
  extract_lets -merge +onlyGivenNames t4_2 at hr
  have e_t4_2 : t4_2 = (t4_1 + w_14 + c_6) % 2^64 := rfl
  clear_value t4_2
  have b_t4_2 : t4_2 < 2^64 := by rw [e_t4_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t4_2, b_k_t4_2, l_t4_2⟩ :
      ∃ k, k ≤ 1 ∧ t4_2 + 2^64 * k = t4_1 + w_14 + c_6 :=
    ⟨(t4_1 + w_14 + c_6) / 2^64, addc_carry_le_one t4_1 w_14 c_6 b_t4_1 b_w_14 b_c_6,
      by rw [e_t4_2]; exact Nat.mod_add_div _ _⟩
  clear e_t4_2
  -- w_15: eor w,v3,s1
  extract_lets -merge +onlyGivenNames w_15 at hr
  have e_w_15 : w_15 = eorw v3 s1' := rfl
  clear_value w_15
  have b_w_15 : w_15 < 2^64 := by rw [e_w_15]; exact eorw_lt v3 s1' b_v3 b_s1'
  -- lo_8: and lo,s1,m1
  extract_lets -merge +onlyGivenNames lo_8 at hr
  have e_lo_8 : lo_8 = andw s1' m1' := rfl
  clear_value lo_8
  have b_lo_8 : lo_8 < 2^64 := by rw [e_lo_8]; exact andw_lt s1' m1' b_s1' b_m1'
  -- t4_3: sub t4,t4,lo
  extract_lets -merge +onlyGivenNames t4_3 at hr
  have e_t4_3 : t4_3 = subw t4_2 lo_8 := rfl
  clear_value t4_3
  have b_t4_3 : t4_3 < 2^64 := by rw [e_t4_3]; exact subw_lt t4_2 lo_8
  -- lo_9: mul lo,w,m1
  extract_lets -merge +onlyGivenNames lo_9 at hr
  have e_lo_9 : lo_9 = w_15 * m1' % 2^64 := rfl
  clear_value lo_9
  have b_lo_9 : lo_9 < 2^64 := by rw [e_lo_9]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_16: umulh w,w,m1
  extract_lets -merge +onlyGivenNames w_16 at hr
  have e_w_16 : w_16 = w_15 * m1' / 2^64 := rfl
  clear_value w_16
  have p_w_16 : w_15 * m1' < 2^64 * 2^64 := Nat.mul_lt_mul'' b_w_15 b_m1'
  have b_w_16 : w_16 < 2^64 := by rw [e_w_16]; exact Nat.div_lt_of_lt_mul p_w_16
  have d_w_16 : lo_9 + 2^64 * w_16 = w_15 * m1' := by
    rw [e_lo_9, e_w_16]; exact Nat.mod_add_div _ _
  clear e_lo_9 e_w_16
  -- t3_3: adds t3,t3,lo
  extract_lets -merge +onlyGivenNames s_7 t3_3 c_7 at hr
  have e_t3_3 : t3_3 = (t3_2 + lo_9 + 0) % 2^64 := rfl
  have e_c_7 : c_7 = (t3_2 + lo_9 + 0) / 2^64 := rfl
  clear_value s_7 t3_3 c_7
  have l_t3_3 : t3_3 + 2^64 * c_7 = t3_2 + lo_9 + 0 := by
    rw [e_t3_3, e_c_7]; exact Nat.mod_add_div _ _
  have b_t3_3 : t3_3 < 2^64 := by rw [e_t3_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_7 : c_7 ≤ 1 := by
    rw [e_c_7]; exact addc_carry_le_one t3_2 lo_9 0 b_t3_2 b_lo_9 (by decide)
  clear e_t3_3 e_c_7
  -- t4_4: adc t4,t4,w
  extract_lets -merge +onlyGivenNames t4_4 at hr
  have e_t4_4 : t4_4 = (t4_3 + w_16 + c_7) % 2^64 := rfl
  clear_value t4_4
  have b_t4_4 : t4_4 < 2^64 := by rw [e_t4_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t4_4, b_k_t4_4, l_t4_4⟩ :
      ∃ k, k ≤ 1 ∧ t4_4 + 2^64 * k = t4_3 + w_16 + c_7 :=
    ⟨(t4_3 + w_16 + c_7) / 2^64, addc_carry_le_one t4_3 w_16 c_7 b_t4_3 b_w_16 b_c_7,
      by rw [e_t4_4]; exact Nat.mod_add_div _ _⟩
  clear e_t4_4
  subst hr
  -- BEGIN conclusion
  -- The two sides on the integers, in the block's words.
  have hA := row_side a (by omega) u hu m0' s0' (by rw [e_m0', e_s0']; exact hrep0)
  have hB := row_side b (by omega) v hv m1' s1' (by rw [e_m1', e_s1']; exact hrep1)
  rw [← e_u0, ← e_u1, ← e_u2, ← e_u3, ← e_w_1, ← e_w_5, ← e_w_9, ← e_w_13, ← e_lo, ← e_t4] at hA
  rw [← e_v0, ← e_v1, ← e_v2, ← e_v3, ← e_w_3, ← e_w_7, ← e_w_11, ← e_w_15, ← e_w, ← e_lo_8] at hB
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
  -- On the integers: the words are `a u + b v` up to a multiple of `2^320`, which the bounds
  -- of both sides fix.
  have hSz : (t0_2 : ℤ) + 2^64 * t1_3 + 2^128 * t2_3 + 2^192 * t3_3 + 2^256 * t4_4
        + 2^320 * (j_t4_1 + j_t4_3 + k_t4_2 + k_t4_4) + 2^256 * (t4 + lo_8)
      = lo + w
        + (w_1 * m0' + 2^64 * (w_5 * m0') + 2^128 * (w_9 * m0') + 2^192 * (w_13 * m0'))
        + (w_3 * m1' + 2^64 * (w_7 * m1') + 2^128 * (w_11 * m1') + 2^192 * (w_15 * m1'))
        + 2^321 := by exact_mod_cast hS
  have hval : (t0_2 : ℤ) + 2^64 * t1_3 + 2^128 * t2_3 + 2^192 * t3_3 + 2^256 * t4_4
      = a * u.toNat + b * v.toNat
        + 2^320 * (2 - (j_t4_1 + j_t4_3 + k_t4_2 + k_t4_4 : ℕ)) := by
    push_cast at hSz hA hB ⊢
    linear_combination hSz + hA + hB
  have hu' : |(u.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (by positivity)]; exact_mod_cast Limbs.toNat_lt u hu
  have hv' : |(v.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (by positivity)]; exact_mod_cast Limbs.toNat_lt v hv
  have hbound : |a * u.toNat + b * v.toNat| < 2^319 := by
    have h := Inversion.row_abs_lt a b _ _ _ _ hab hu' hv' (by positivity)
    rwa [show (2 : ℤ)^63 * 2^256 = 2^319 by norm_num] at h
  rw [abs_lt] at hbound
  refine ⟨⟨b_t0_2, b_t1_3, b_t2_3, b_t3_3, b_t4_4⟩, ?_⟩
  unfold Signed5.toInt
  dsimp only
  clear * - hval hbound b_t0_2 b_t1_3 b_t2_3 b_t3_3 b_t4_4 b_j_t4_1 b_j_t4_3 b_k_t4_2 b_k_t4_4
  split_ifs <;> omega
  -- END conclusion

end PastaCurves.AArch64
