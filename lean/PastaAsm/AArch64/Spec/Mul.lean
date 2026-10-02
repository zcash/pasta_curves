/-
Copyright Supranational LLC (the routines, transcribed from Semolina v0.1.4).
Copyright (c) 2026 the pasta-asm contributors (the transcription and the proofs).
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.Spec
import PastaAsm.AArch64.Transcription
import PastaAsm.AArch64.Compositions
import Mathlib.Tactic.NormNum

/-!
# Correctness of the transcribed Pasta multiplication block

See the parent module's documentation for details.
-/

namespace PastaAsm.AArch64

-- BEGIN mulMontRound_spec statement
/-- One round of the multiplication, on any accumulator `acc` whose quotient and reduction terms
are those the previous round left (`hq`, `ht1`, `ht3`): it returns bounded registers with the
same relations, and `2^64 * s'.toNat = s.toNat + q * p + 2^64 * (lhs * b)`, that is, the
previous accumulator reduced by its quotient, shifted down one limb, plus the round's product.
`hH1` keeps the five-limb accumulator below `2^320` while the reduction's low terms are added;
`hH2` keeps the shifted accumulator plus the product below `2^320`. -/
theorem mulMontRound_spec (lhs modulus : Limbs) (inv b : Nat) (acc : MulMontAcc)
    (hlhs : lhs.Bounded) (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinv_lt : inv < 2^64) (hinv : (inv * modulus.l0 + 1) % 2^64 = 0) (hb : b < 2^64)
    (hacc : acc.Bounded) (hq : acc.q = inv * acc.r0 % 2^64) (ht1 : acc.t1 = modulus.l1 * acc.q % 2^64)
    (ht3 : acc.t3 = acc.q * 2^62 % 2^64)
    (hH1 : acc.toNat + 2^128 + 3 * 2^254 < 2^320)
    (hH2 : acc.toNat / 2^64 + modulus.toNat + lhs.toNat * b < 2^320) :
    ∀ s', s' = mulMontRound lhs modulus inv b acc →
      s'.Bounded ∧ s'.q = inv * s'.r0 % 2^64 ∧ s'.t1 = modulus.l1 * s'.q % 2^64 ∧
        s'.t3 = s'.q * 2^62 % 2^64 ∧
        2^64 * s'.toNat = acc.toNat + acc.q * modulus.toNat + 2^64 * (lhs.toNat * b) := by
  intro s' hr
-- END mulMontRound_spec statement
  -- generated skeleton for `mulMontRound`: do not edit between the annotations
  unfold mulMontRound at hr
  lift_lets -merge at hr
  -- a0: argument
  extract_lets -merge +onlyGivenNames a0 at hr
  have e_a0 : a0 = lhs.l0 := rfl
  clear_value a0
  have b_a0 : a0 < 2^64 := by rw [e_a0]; exact hlhs.1
  -- a1: argument
  extract_lets -merge +onlyGivenNames a1 at hr
  have e_a1 : a1 = lhs.l1 := rfl
  clear_value a1
  have b_a1 : a1 < 2^64 := by rw [e_a1]; exact hlhs.2.1
  -- a2: argument
  extract_lets -merge +onlyGivenNames a2 at hr
  have e_a2 : a2 = lhs.l2 := rfl
  clear_value a2
  have b_a2 : a2 < 2^64 := by rw [e_a2]; exact hlhs.2.2.1
  -- a3: argument
  extract_lets -merge +onlyGivenNames a3 at hr
  have e_a3 : a3 = lhs.l3 := rfl
  clear_value a3
  have b_a3 : a3 < 2^64 := by rw [e_a3]; exact hlhs.2.2.2
  -- p0: argument
  extract_lets -merge +onlyGivenNames p0 at hr
  have e_p0 : p0 = modulus.l0 := rfl
  clear_value p0
  have b_p0 : p0 < 2^64 := by rw [e_p0]; exact hm.1
  -- p1: argument
  extract_lets -merge +onlyGivenNames p1 at hr
  have e_p1 : p1 = modulus.l1 := rfl
  clear_value p1
  have b_p1 : p1 < 2^64 := by rw [e_p1]; exact hm.2.1
  -- inv': argument
  extract_lets -merge +onlyGivenNames inv' at hr
  have e_inv' : inv' = inv := rfl
  clear_value inv'
  have b_inv' : inv' < 2^64 := by rw [e_inv']; exact hinv_lt
  -- b1: argument
  extract_lets -merge +onlyGivenNames b1 at hr
  have e_b1 : b1 = b := rfl
  clear_value b1
  have b_b1 : b1 < 2^64 := by rw [e_b1]; exact hb
  -- r0: argument
  extract_lets -merge +onlyGivenNames r0 at hr
  have e_r0 : r0 = acc.r0 := rfl
  clear_value r0
  have b_r0 : r0 < 2^64 := by rw [e_r0]; exact hacc.1
  -- r1: argument
  extract_lets -merge +onlyGivenNames r1 at hr
  have e_r1 : r1 = acc.r1 := rfl
  clear_value r1
  have b_r1 : r1 < 2^64 := by rw [e_r1]; exact hacc.2.1
  -- r2: argument
  extract_lets -merge +onlyGivenNames r2 at hr
  have e_r2 : r2 = acc.r2 := rfl
  clear_value r2
  have b_r2 : r2 < 2^64 := by rw [e_r2]; exact hacc.2.2.1
  -- r3: argument
  extract_lets -merge +onlyGivenNames r3 at hr
  have e_r3 : r3 = acc.r3 := rfl
  clear_value r3
  have b_r3 : r3 < 2^64 := by rw [e_r3]; exact hacc.2.2.2.1
  -- r4: argument
  extract_lets -merge +onlyGivenNames r4 at hr
  have e_r4 : r4 = acc.r4 := rfl
  clear_value r4
  have b_r4 : r4 < 2^64 := by rw [e_r4]; exact hacc.2.2.2.2.1
  -- q: argument
  extract_lets -merge +onlyGivenNames q at hr
  have e_q : q = acc.q := rfl
  clear_value q
  have b_q : q < 2^64 := by rw [e_q]; exact hacc.2.2.2.2.2.1
  -- t1: argument
  extract_lets -merge +onlyGivenNames t1 at hr
  have e_t1 : t1 = acc.t1 := rfl
  clear_value t1
  have b_t1 : t1 < 2^64 := by rw [e_t1]; exact hacc.2.2.2.2.2.2.1
  -- t3: argument
  extract_lets -merge +onlyGivenNames t3 at hr
  have e_t3 : t3 = acc.t3 := rfl
  clear_value t3
  have b_t3 : t3 < 2^64 := by rw [e_t3]; exact hacc.2.2.2.2.2.2.2
  -- c: subs xzr,r0,#1
  extract_lets -merge +onlyGivenNames c at hr
  have e_c : c = (r0 + 2^64 - 1 - (1 - 1)) / 2^64 := rfl
  clear_value c
  have b_c : c ≤ 1 := by rw [e_c]; exact subc_carry_le_one r0 1 1 b_r0
  have l_c : (c = 1 ∧ 1 + 1 ≤ r0 + 1) ∨ (c = 0 ∧ r0 + 1 < 1 + 1) :=
    subc_carry_cases r0 1 1 _ e_c b_r0 (by decide) (by decide)
  clear e_c
  -- t0: umulh t0,p0,q
  extract_lets -merge +onlyGivenNames t0 at hr
  have e_t0 : t0 = p0 * q / 2^64 := rfl
  clear_value t0
  have p_t0 : p0 * q < 2^64 * 2^64 := Nat.mul_lt_mul'' b_p0 b_q
  have b_t0 : t0 < 2^64 := by rw [e_t0]; exact Nat.div_lt_of_lt_mul p_t0
  obtain ⟨lo_t0, b_lo_t0, d_t0⟩ :
      ∃ lo, lo < 2^64 ∧ lo + 2^64 * t0 = p0 * q :=
    ⟨p0 * q % 2^64, Nat.mod_lt _ (Nat.two_pow_pos _),
      by rw [e_t0]; exact Nat.mod_add_div _ _⟩
  clear e_t0
  -- BEGIN round entry
  have hq' : q = inv' * r0 % 2^64 := by rw [e_q, e_inv', e_r0]; exact hq
  have ht1' : t1 = p1 * q % 2^64 := by rw [e_t1, e_p1, e_q]; exact ht1
  have ht3' : t3 = q * 2^62 % 2^64 := by rw [e_t3, e_q]; exact ht3
  have hP : modulus.toNat = p0 + 2^64 * p1 + 2^192 * 2^62 := by
    rw [e_p0, e_p1]; simp only [Limbs.toNat, hshape.1, hshape.2, Nat.mul_zero, Nat.add_zero]
  have hPq : q * modulus.toNat = p0 * q + 2^64 * (p1 * q) + 2^254 * q := by
    rw [hP]; ring
  have hA : acc.toNat = r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3 + 2^256 * r4 := by
    simp only [MulMontAcc.toNat, e_r0, e_r1, e_r2, e_r3, e_r4]
  rw [hA] at hH1 hH2
  -- Cancellation: the low limb of `r0 + p0 * q` is zero, so `r0 + lo_t0` is `0` or
  -- `2^64`, and `subs xzr, r0, #1` set the carry exactly when it is `2^64`.
  have hc : r0 + lo_t0 = 2^64 * c := by
    have h := cancel_low r0 inv' p0 (by rw [e_inv', e_p0]; exact hinv)
    rw [← hq', ← d_t0, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt b_lo_t0] at h
    clear * - h b_r0 b_lo_t0 l_c
    omega
  -- END round entry
  -- r1_1: adcs r1,r1,t1
  extract_lets -merge +onlyGivenNames s r1_1 c_1 at hr
  have e_r1_1 : r1_1 = (r1 + t1 + c) % 2^64 := rfl
  have e_c_1 : c_1 = (r1 + t1 + c) / 2^64 := rfl
  clear_value s r1_1 c_1
  have l_r1_1 : r1_1 + 2^64 * c_1 = r1 + t1 + c := by
    rw [e_r1_1, e_c_1]; exact Nat.mod_add_div _ _
  have b_r1_1 : r1_1 < 2^64 := by rw [e_r1_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_1 : c_1 ≤ 1 := by
    rw [e_c_1]; exact addc_carry_le_one r1 t1 c b_r1 b_t1 b_c
  clear e_r1_1 e_c_1
  -- t1_1: umulh t1,p1,q
  extract_lets -merge +onlyGivenNames t1_1 at hr
  have e_t1_1 : t1_1 = p1 * q / 2^64 := rfl
  clear_value t1_1
  have p_t1_1 : p1 * q < 2^64 * 2^64 := Nat.mul_lt_mul'' b_p1 b_q
  have b_t1_1 : t1_1 < 2^64 := by rw [e_t1_1]; exact Nat.div_lt_of_lt_mul p_t1_1
  obtain ⟨lo_t1_1, b_lo_t1_1, d_t1_1⟩ :
      ∃ lo, lo < 2^64 ∧ lo + 2^64 * t1_1 = p1 * q :=
    ⟨p1 * q % 2^64, Nat.mod_lt _ (Nat.two_pow_pos _),
      by rw [e_t1_1]; exact Nat.mod_add_div _ _⟩
  clear e_t1_1
  -- r2_1: adcs r2,r2,xzr
  extract_lets -merge +onlyGivenNames s_1 r2_1 c_2 at hr
  have e_r2_1 : r2_1 = (r2 + 0 + c_1) % 2^64 := rfl
  have e_c_2 : c_2 = (r2 + 0 + c_1) / 2^64 := rfl
  clear_value s_1 r2_1 c_2
  have l_r2_1 : r2_1 + 2^64 * c_2 = r2 + 0 + c_1 := by
    rw [e_r2_1, e_c_2]; exact Nat.mod_add_div _ _
  have b_r2_1 : r2_1 < 2^64 := by rw [e_r2_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_2 : c_2 ≤ 1 := by
    rw [e_c_2]; exact addc_carry_le_one r2 0 c_1 b_r2 (by decide) b_c_1
  clear e_r2_1 e_c_2
  -- r3_1: adcs r3,r3,t3
  extract_lets -merge +onlyGivenNames s_2 r3_1 c_3 at hr
  have e_r3_1 : r3_1 = (r3 + t3 + c_2) % 2^64 := rfl
  have e_c_3 : c_3 = (r3 + t3 + c_2) / 2^64 := rfl
  clear_value s_2 r3_1 c_3
  have l_r3_1 : r3_1 + 2^64 * c_3 = r3 + t3 + c_2 := by
    rw [e_r3_1, e_c_3]; exact Nat.mod_add_div _ _
  have b_r3_1 : r3_1 < 2^64 := by rw [e_r3_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_3 : c_3 ≤ 1 := by
    rw [e_c_3]; exact addc_carry_le_one r3 t3 c_2 b_r3 b_t3 b_c_2
  clear e_r3_1 e_c_3
  -- t3_1: lsr t3,q,#2
  extract_lets -merge +onlyGivenNames t3_1 at hr
  have e_t3_1 : t3_1 = q / 2^2 := rfl
  clear_value t3_1
  have b_t3_1 : t3_1 < 2^62 := by
    rw [e_t3_1]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_q (by norm_num))
  -- r4_1: adc r4,r4,xzr
  extract_lets -merge +onlyGivenNames r4_1 at hr
  have e_r4_1 : r4_1 = (r4 + 0 + c_3) % 2^64 := rfl
  clear_value r4_1
  have b_r4_1 : r4_1 < 2^64 := by rw [e_r4_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_r4_1, b_k_r4_1, l_r4_1⟩ :
      ∃ k, k ≤ 1 ∧ r4_1 + 2^64 * k = r4 + 0 + c_3 :=
    ⟨(r4 + 0 + c_3) / 2^64, addc_carry_le_one r4 0 c_3 b_r4 (by decide) b_c_3,
      by rw [e_r4_1]; exact Nat.mod_add_div _ _⟩
  clear e_r4_1
  -- BEGIN round reduction
  have hbt3 : t3 ≤ 3 * 2^62 := by clear * - ht3'; omega
  have hsumr : (r0 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 + 2^256 * r4_1) + 2^320 * k_r4_1
      = (r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3 + 2^256 * r4) + 2^64 * t1 + 2^64 * c + 2^192 * t3 := by
    clear * - l_r1_1 l_r2_1 l_r3_1 l_r4_1
    omega
  have hkr : k_r4_1 = 0 := by
    clear * - hsumr hH1 b_t1 b_c hbt3 b_k_r4_1 b_r0 b_r1_1 b_r2_1 b_r3_1 b_r4_1
    omega
  -- END round reduction
  -- r0_1: adds r0,r1,t0
  extract_lets -merge +onlyGivenNames s_3 r0_1 c_4 at hr
  have e_r0_1 : r0_1 = (r1_1 + t0 + 0) % 2^64 := rfl
  have e_c_4 : c_4 = (r1_1 + t0 + 0) / 2^64 := rfl
  clear_value s_3 r0_1 c_4
  have l_r0_1 : r0_1 + 2^64 * c_4 = r1_1 + t0 + 0 := by
    rw [e_r0_1, e_c_4]; exact Nat.mod_add_div _ _
  have b_r0_1 : r0_1 < 2^64 := by rw [e_r0_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_4 : c_4 ≤ 1 := by
    rw [e_c_4]; exact addc_carry_le_one r1_1 t0 0 b_r1_1 b_t0 (by decide)
  clear e_r0_1 e_c_4
  -- t0_1: mul t0,a0,b1
  extract_lets -merge +onlyGivenNames t0_1 at hr
  have e_t0_1 : t0_1 = a0 * b1 % 2^64 := rfl
  clear_value t0_1
  have b_t0_1 : t0_1 < 2^64 := by rw [e_t0_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- r1_2: adcs r1,r2,t1
  extract_lets -merge +onlyGivenNames s_4 r1_2 c_5 at hr
  have e_r1_2 : r1_2 = (r2_1 + t1_1 + c_4) % 2^64 := rfl
  have e_c_5 : c_5 = (r2_1 + t1_1 + c_4) / 2^64 := rfl
  clear_value s_4 r1_2 c_5
  have l_r1_2 : r1_2 + 2^64 * c_5 = r2_1 + t1_1 + c_4 := by
    rw [e_r1_2, e_c_5]; exact Nat.mod_add_div _ _
  have b_r1_2 : r1_2 < 2^64 := by rw [e_r1_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_5 : c_5 ≤ 1 := by
    rw [e_c_5]; exact addc_carry_le_one r2_1 t1_1 c_4 b_r2_1 b_t1_1 b_c_4
  clear e_r1_2 e_c_5
  -- t1_2: mul t1,a1,b1
  extract_lets -merge +onlyGivenNames t1_2 at hr
  have e_t1_2 : t1_2 = a1 * b1 % 2^64 := rfl
  clear_value t1_2
  have b_t1_2 : t1_2 < 2^64 := by rw [e_t1_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- r2_2: adcs r2,r3,xzr
  extract_lets -merge +onlyGivenNames s_5 r2_2 c_6 at hr
  have e_r2_2 : r2_2 = (r3_1 + 0 + c_5) % 2^64 := rfl
  have e_c_6 : c_6 = (r3_1 + 0 + c_5) / 2^64 := rfl
  clear_value s_5 r2_2 c_6
  have l_r2_2 : r2_2 + 2^64 * c_6 = r3_1 + 0 + c_5 := by
    rw [e_r2_2, e_c_6]; exact Nat.mod_add_div _ _
  have b_r2_2 : r2_2 < 2^64 := by rw [e_r2_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_6 : c_6 ≤ 1 := by
    rw [e_c_6]; exact addc_carry_le_one r3_1 0 c_5 b_r3_1 (by decide) b_c_5
  clear e_r2_2 e_c_6
  -- t2: mul t2,a2,b1
  extract_lets -merge +onlyGivenNames t2 at hr
  have e_t2 : t2 = a2 * b1 % 2^64 := rfl
  clear_value t2
  have b_t2 : t2 < 2^64 := by rw [e_t2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- r3_2: adcs r3,r4,t3
  extract_lets -merge +onlyGivenNames s_6 r3_2 c_7 at hr
  have e_r3_2 : r3_2 = (r4_1 + t3_1 + c_6) % 2^64 := rfl
  have e_c_7 : c_7 = (r4_1 + t3_1 + c_6) / 2^64 := rfl
  clear_value s_6 r3_2 c_7
  have l_r3_2 : r3_2 + 2^64 * c_7 = r4_1 + t3_1 + c_6 := by
    rw [e_r3_2, e_c_7]; exact Nat.mod_add_div _ _
  have b_r3_2 : r3_2 < 2^64 := by rw [e_r3_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_7 : c_7 ≤ 1 := by
    rw [e_c_7]; exact addc_carry_le_one r4_1 t3_1 c_6 b_r4_1 (lt_of_lt_of_le b_t3_1 (by norm_num)) b_c_6
  clear e_r3_2 e_c_7
  -- t3_2: mul t3,a3,b1
  extract_lets -merge +onlyGivenNames t3_2 at hr
  have e_t3_2 : t3_2 = a3 * b1 % 2^64 := rfl
  clear_value t3_2
  have b_t3_2 : t3_2 < 2^64 := by rw [e_t3_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- r4_2: adc r4,xzr,xzr
  extract_lets -merge +onlyGivenNames r4_2 at hr
  have e_r4_2 : r4_2 = (0 + 0 + c_7) % 2^64 := rfl
  clear_value r4_2
  have b_r4_2 : r4_2 < 2^64 := by rw [e_r4_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_r4_2, b_k_r4_2, l_r4_2⟩ :
      ∃ k, k ≤ 1 ∧ r4_2 + 2^64 * k = 0 + 0 + c_7 :=
    ⟨(0 + 0 + c_7) / 2^64, addc_carry_le_one 0 0 c_7 (by decide) (by decide) b_c_7,
      by rw [e_r4_2]; exact Nat.mod_add_div _ _⟩
  clear e_r4_2
  -- BEGIN round shift
  have hsums : (r0_1 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + 2^256 * r4_2) + 2^320 * k_r4_2
      = (r1_1 + 2^64 * r2_1 + 2^128 * r3_1 + 2^192 * r4_1) + t0 + 2^64 * t1_1
        + 2^192 * t3_1 := by
    clear * - l_r0_1 l_r1_2 l_r2_2 l_r3_2 l_r4_2
    omega
  have hks : k_r4_2 = 0 := by
    clear * - hsums b_r1_1 b_r2_1 b_r3_1 b_r4_1 b_t0 b_t1_1 b_t3_1 b_k_r4_2 b_r0_1 b_r1_2
        b_r2_2 b_r3_2 b_r4_2
    omega
  -- The reduction's low terms are the ones the previous round left in `t1` and `t3`.
  have hlot1 : t1 = lo_t1_1 := by clear * - ht1' d_t1_1 b_lo_t1_1; omega
  have hsh : t3 + 2^64 * t3_1 = q * 2^62 := by clear * - ht3' e_t3_1; omega
  have I : 2^64 * (r0_1 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + 2^256 * r4_2) = (r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3 + 2^256 * r4) + q * modulus.toNat := by
    clear * - hsums hks hsumr hkr hc d_t0 d_t1_1 hlot1 hsh hPq
    omega
  -- END round shift
  -- r0_2: adds r0,r0,t0
  extract_lets -merge +onlyGivenNames s_7 r0_2 c_8 at hr
  have e_r0_2 : r0_2 = (r0_1 + t0_1 + 0) % 2^64 := rfl
  have e_c_8 : c_8 = (r0_1 + t0_1 + 0) / 2^64 := rfl
  clear_value s_7 r0_2 c_8
  have l_r0_2 : r0_2 + 2^64 * c_8 = r0_1 + t0_1 + 0 := by
    rw [e_r0_2, e_c_8]; exact Nat.mod_add_div _ _
  have b_r0_2 : r0_2 < 2^64 := by rw [e_r0_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_8 : c_8 ≤ 1 := by
    rw [e_c_8]; exact addc_carry_le_one r0_1 t0_1 0 b_r0_1 b_t0_1 (by decide)
  clear e_r0_2 e_c_8
  -- t0_2: umulh t0,a0,b1
  extract_lets -merge +onlyGivenNames t0_2 at hr
  have e_t0_2 : t0_2 = a0 * b1 / 2^64 := rfl
  clear_value t0_2
  have p_t0_2 : a0 * b1 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a0 b_b1
  have b_t0_2 : t0_2 < 2^64 := by rw [e_t0_2]; exact Nat.div_lt_of_lt_mul p_t0_2
  have d_t0_2 : t0_1 + 2^64 * t0_2 = a0 * b1 := by
    rw [e_t0_1, e_t0_2]; exact Nat.mod_add_div _ _
  clear e_t0_1 e_t0_2
  -- r1_3: adcs r1,r1,t1
  extract_lets -merge +onlyGivenNames s_8 r1_3 c_9 at hr
  have e_r1_3 : r1_3 = (r1_2 + t1_2 + c_8) % 2^64 := rfl
  have e_c_9 : c_9 = (r1_2 + t1_2 + c_8) / 2^64 := rfl
  clear_value s_8 r1_3 c_9
  have l_r1_3 : r1_3 + 2^64 * c_9 = r1_2 + t1_2 + c_8 := by
    rw [e_r1_3, e_c_9]; exact Nat.mod_add_div _ _
  have b_r1_3 : r1_3 < 2^64 := by rw [e_r1_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_9 : c_9 ≤ 1 := by
    rw [e_c_9]; exact addc_carry_le_one r1_2 t1_2 c_8 b_r1_2 b_t1_2 b_c_8
  clear e_r1_3 e_c_9
  -- t1_3: umulh t1,a1,b1
  extract_lets -merge +onlyGivenNames t1_3 at hr
  have e_t1_3 : t1_3 = a1 * b1 / 2^64 := rfl
  clear_value t1_3
  have p_t1_3 : a1 * b1 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a1 b_b1
  have b_t1_3 : t1_3 < 2^64 := by rw [e_t1_3]; exact Nat.div_lt_of_lt_mul p_t1_3
  have d_t1_3 : t1_2 + 2^64 * t1_3 = a1 * b1 := by
    rw [e_t1_2, e_t1_3]; exact Nat.mod_add_div _ _
  clear e_t1_2 e_t1_3
  -- r2_3: adcs r2,r2,t2
  extract_lets -merge +onlyGivenNames s_9 r2_3 c_10 at hr
  have e_r2_3 : r2_3 = (r2_2 + t2 + c_9) % 2^64 := rfl
  have e_c_10 : c_10 = (r2_2 + t2 + c_9) / 2^64 := rfl
  clear_value s_9 r2_3 c_10
  have l_r2_3 : r2_3 + 2^64 * c_10 = r2_2 + t2 + c_9 := by
    rw [e_r2_3, e_c_10]; exact Nat.mod_add_div _ _
  have b_r2_3 : r2_3 < 2^64 := by rw [e_r2_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_10 : c_10 ≤ 1 := by
    rw [e_c_10]; exact addc_carry_le_one r2_2 t2 c_9 b_r2_2 b_t2 b_c_9
  clear e_r2_3 e_c_10
  -- q_1: mul q,inv,r0
  extract_lets -merge +onlyGivenNames q_1 at hr
  have e_q_1 : q_1 = inv' * r0_2 % 2^64 := rfl
  clear_value q_1
  have b_q_1 : q_1 < 2^64 := by rw [e_q_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t2_1: umulh t2,a2,b1
  extract_lets -merge +onlyGivenNames t2_1 at hr
  have e_t2_1 : t2_1 = a2 * b1 / 2^64 := rfl
  clear_value t2_1
  have p_t2_1 : a2 * b1 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a2 b_b1
  have b_t2_1 : t2_1 < 2^64 := by rw [e_t2_1]; exact Nat.div_lt_of_lt_mul p_t2_1
  have d_t2_1 : t2 + 2^64 * t2_1 = a2 * b1 := by
    rw [e_t2, e_t2_1]; exact Nat.mod_add_div _ _
  clear e_t2 e_t2_1
  -- r3_3: adcs r3,r3,t3
  extract_lets -merge +onlyGivenNames s_10 r3_3 c_11 at hr
  have e_r3_3 : r3_3 = (r3_2 + t3_2 + c_10) % 2^64 := rfl
  have e_c_11 : c_11 = (r3_2 + t3_2 + c_10) / 2^64 := rfl
  clear_value s_10 r3_3 c_11
  have l_r3_3 : r3_3 + 2^64 * c_11 = r3_2 + t3_2 + c_10 := by
    rw [e_r3_3, e_c_11]; exact Nat.mod_add_div _ _
  have b_r3_3 : r3_3 < 2^64 := by rw [e_r3_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_11 : c_11 ≤ 1 := by
    rw [e_c_11]; exact addc_carry_le_one r3_2 t3_2 c_10 b_r3_2 b_t3_2 b_c_10
  clear e_r3_3 e_c_11
  -- t3_3: umulh t3,a3,b1
  extract_lets -merge +onlyGivenNames t3_3 at hr
  have e_t3_3 : t3_3 = a3 * b1 / 2^64 := rfl
  clear_value t3_3
  have p_t3_3 : a3 * b1 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a3 b_b1
  have b_t3_3 : t3_3 < 2^64 := by rw [e_t3_3]; exact Nat.div_lt_of_lt_mul p_t3_3
  have d_t3_3 : t3_2 + 2^64 * t3_3 = a3 * b1 := by
    rw [e_t3_2, e_t3_3]; exact Nat.mod_add_div _ _
  clear e_t3_2 e_t3_3
  -- r4_3: adc r4,r4,xzr
  extract_lets -merge +onlyGivenNames r4_3 at hr
  have e_r4_3 : r4_3 = (r4_2 + 0 + c_11) % 2^64 := rfl
  clear_value r4_3
  have b_r4_3 : r4_3 < 2^64 := by rw [e_r4_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_r4_3, b_k_r4_3, l_r4_3⟩ :
      ∃ k, k ≤ 1 ∧ r4_3 + 2^64 * k = r4_2 + 0 + c_11 :=
    ⟨(r4_2 + 0 + c_11) / 2^64, addc_carry_le_one r4_2 0 c_11 b_r4_2 (by decide) b_c_11,
      by rw [e_r4_3]; exact Nat.mod_add_div _ _⟩
  clear e_r4_3
  -- r1_4: adds r1,r1,t0
  extract_lets -merge +onlyGivenNames s_11 r1_4 c_12 at hr
  have e_r1_4 : r1_4 = (r1_3 + t0_2 + 0) % 2^64 := rfl
  have e_c_12 : c_12 = (r1_3 + t0_2 + 0) / 2^64 := rfl
  clear_value s_11 r1_4 c_12
  have l_r1_4 : r1_4 + 2^64 * c_12 = r1_3 + t0_2 + 0 := by
    rw [e_r1_4, e_c_12]; exact Nat.mod_add_div _ _
  have b_r1_4 : r1_4 < 2^64 := by rw [e_r1_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_12 : c_12 ≤ 1 := by
    rw [e_c_12]; exact addc_carry_le_one r1_3 t0_2 0 b_r1_3 b_t0_2 (by decide)
  clear e_r1_4 e_c_12
  -- r2_4: adcs r2,r2,t1
  extract_lets -merge +onlyGivenNames s_12 r2_4 c_13 at hr
  have e_r2_4 : r2_4 = (r2_3 + t1_3 + c_12) % 2^64 := rfl
  have e_c_13 : c_13 = (r2_3 + t1_3 + c_12) / 2^64 := rfl
  clear_value s_12 r2_4 c_13
  have l_r2_4 : r2_4 + 2^64 * c_13 = r2_3 + t1_3 + c_12 := by
    rw [e_r2_4, e_c_13]; exact Nat.mod_add_div _ _
  have b_r2_4 : r2_4 < 2^64 := by rw [e_r2_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_13 : c_13 ≤ 1 := by
    rw [e_c_13]; exact addc_carry_le_one r2_3 t1_3 c_12 b_r2_3 b_t1_3 b_c_12
  clear e_r2_4 e_c_13
  -- t1_4: mul t1,p1,q
  extract_lets -merge +onlyGivenNames t1_4 at hr
  have e_t1_4 : t1_4 = p1 * q_1 % 2^64 := rfl
  clear_value t1_4
  have b_t1_4 : t1_4 < 2^64 := by rw [e_t1_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- r3_4: adcs r3,r3,t2
  extract_lets -merge +onlyGivenNames s_13 r3_4 c_14 at hr
  have e_r3_4 : r3_4 = (r3_3 + t2_1 + c_13) % 2^64 := rfl
  have e_c_14 : c_14 = (r3_3 + t2_1 + c_13) / 2^64 := rfl
  clear_value s_13 r3_4 c_14
  have l_r3_4 : r3_4 + 2^64 * c_14 = r3_3 + t2_1 + c_13 := by
    rw [e_r3_4, e_c_14]; exact Nat.mod_add_div _ _
  have b_r3_4 : r3_4 < 2^64 := by rw [e_r3_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_14 : c_14 ≤ 1 := by
    rw [e_c_14]; exact addc_carry_le_one r3_3 t2_1 c_13 b_r3_3 b_t2_1 b_c_13
  clear e_r3_4 e_c_14
  -- r4_4: adc r4,r4,t3
  extract_lets -merge +onlyGivenNames r4_4 at hr
  have e_r4_4 : r4_4 = (r4_3 + t3_3 + c_14) % 2^64 := rfl
  clear_value r4_4
  have b_r4_4 : r4_4 < 2^64 := by rw [e_r4_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_r4_4, b_k_r4_4, l_r4_4⟩ :
      ∃ k, k ≤ 1 ∧ r4_4 + 2^64 * k = r4_3 + t3_3 + c_14 :=
    ⟨(r4_3 + t3_3 + c_14) / 2^64, addc_carry_le_one r4_3 t3_3 c_14 b_r4_3 b_t3_3 b_c_14,
      by rw [e_r4_4]; exact Nat.mod_add_div _ _⟩
  clear e_r4_4
  -- BEGIN round fold
  have hL : lhs.toNat * b
      = a0 * b1 + 2^64 * (a1 * b1) + 2^128 * (a2 * b1) + 2^192 * (a3 * b1) := by
    rw [← e_b1, e_a0, e_a1, e_a2, e_a3]; simp only [Limbs.toNat]; ring
  have hsumf : (r0_2 + 2^64 * r1_4 + 2^128 * r2_4 + 2^192 * r3_4 + 2^256 * r4_4) + 2^320 * (k_r4_3 + k_r4_4)
      = (r0_1 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + 2^256 * r4_2)
        + (t0_1 + 2^64 * t1_2 + 2^128 * t2 + 2^192 * t3_2)
        + 2^64 * (t0_2 + 2^64 * t1_3 + 2^128 * t2_1 + 2^192 * t3_3) := by
    clear * - l_r0_2 l_r1_3 l_r2_3 l_r3_3 l_r4_3 l_r1_4 l_r2_4 l_r3_4 l_r4_4
    omega
  have hprod : (t0_1 + 2^64 * t1_2 + 2^128 * t2 + 2^192 * t3_2)
      + 2^64 * (t0_2 + 2^64 * t1_3 + 2^128 * t2_1 + 2^192 * t3_3) = lhs.toNat * b := by
    clear * - hL d_t0_2 d_t1_3 d_t2_1 d_t3_3
    omega
  have hqP : q * modulus.toNat + modulus.toNat ≤ 2^64 * modulus.toNat := by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ b_q
  have hkf : k_r4_3 = 0 ∧ k_r4_4 = 0 := by
    clear * - hsumf hprod I hqP hH2 b_k_r4_3 b_k_r4_4 b_r0_2 b_r1_4 b_r2_4 b_r3_4 b_r4_4
    omega
  have F : 2^64 * (r0_2 + 2^64 * r1_4 + 2^128 * r2_4 + 2^192 * r3_4 + 2^256 * r4_4)
      = (r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3 + 2^256 * r4) + q * modulus.toNat + 2^64 * (lhs.toNat * b) := by
    clear * - hsumf hprod hkf I
    omega
  -- END round fold
  -- t3_4: lsl t3,q,#62
  extract_lets -merge +onlyGivenNames t3_4 at hr
  have e_t3_4 : t3_4 = q_1 * 2^62 % 2^64 := rfl
  clear_value t3_4
  have b_t3_4 : t3_4 < 2^64 := by rw [e_t3_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_t3_4 : t3_4 + 2^64 * (q_1 / 2^2) = q_1 * 2^62 := by
    rw [e_t3_4]; exact lsl62_lsr2_split _
  subst hr
  -- BEGIN round conclusion
  refine ⟨⟨b_r0_2, b_r1_4, b_r2_4, b_r3_4, b_r4_4, b_q_1, b_t1_4, b_t3_4⟩, ?_, ?_, ?_, ?_⟩
  · show q_1 = inv * r0_2 % 2^64
    rw [e_q_1, e_inv']
  · show t1_4 = modulus.l1 * q_1 % 2^64
    rw [e_t1_4, e_p1]
  · exact e_t3_4
  · show 2^64 * (r0_2 + 2^64 * r1_4 + 2^128 * r2_4 + 2^192 * r3_4 + 2^256 * r4_4)
      = acc.toNat + acc.q * modulus.toNat + 2^64 * (lhs.toNat * b)
    rw [hA, ← e_q]; exact F
  -- END round conclusion

-- BEGIN mulMont_spec statement
/-- Montgomery multiplication by the inline block: the result is below `p` and
`2^256 * result ≡ lhs * rhs (mod p)`, under two arithmetic conditions that each operand contract
implies (`mulMont_spec_of_lhs_lt` and `mulMont_spec_of_rhs_lt`). `hsafe` keeps the five-limb
accumulator below `2^320` in rounds 1 to 3, where it holds the previous round's result (below
`lhs + p`) plus `lhs * rhs_i`, and then the low limbs of the reduction; without it the `adc` that
closes each fold can drop a carry. `hfinal` keeps the final candidate below `2 * p`: the block
keeps four limbs of it, and the carry out of the fourth limb, which it drops, is `0`, so one
conditional subtraction reduces it. -/
theorem mulMont_spec (lhs rhs modulus : Limbs) (inv : Nat) (hlhs : lhs.Bounded)
    (hrhs : rhs.Bounded) (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinv_lt : inv < 2^64) (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hsafe : lhs.toNat * (rhs.l1 + 1) + modulus.toNat + 3 * 2^254 + 2^128 ≤ 2^320 ∧
      lhs.toNat * (rhs.l2 + 1) + modulus.toNat + 3 * 2^254 + 2^128 ≤ 2^320 ∧
      lhs.toNat * (rhs.l3 + 1) + modulus.toNat + 3 * 2^254 + 2^128 ≤ 2^320)
    (hfinal : lhs.toNat * rhs.toNat < 2^256 * modulus.toNat) :
    ∀ r, r = mulMont lhs rhs modulus inv →
      r.Bounded ∧ r.toNat < modulus.toNat ∧
        2^256 * r.toNat ≡ lhs.toNat * rhs.toNat [MOD modulus.toNat] := by
  intro r hr
-- END mulMont_spec statement
  -- generated skeleton for `mulMont`: do not edit between the annotations
  unfold mulMont at hr
  lift_lets -merge at hr
  -- a0: argument
  extract_lets -merge +onlyGivenNames a0 at hr
  have e_a0 : a0 = lhs.l0 := rfl
  clear_value a0
  have b_a0 : a0 < 2^64 := by rw [e_a0]; exact hlhs.1
  -- a1: argument
  extract_lets -merge +onlyGivenNames a1 at hr
  have e_a1 : a1 = lhs.l1 := rfl
  clear_value a1
  have b_a1 : a1 < 2^64 := by rw [e_a1]; exact hlhs.2.1
  -- a2: argument
  extract_lets -merge +onlyGivenNames a2 at hr
  have e_a2 : a2 = lhs.l2 := rfl
  clear_value a2
  have b_a2 : a2 < 2^64 := by rw [e_a2]; exact hlhs.2.2.1
  -- a3: argument
  extract_lets -merge +onlyGivenNames a3 at hr
  have e_a3 : a3 = lhs.l3 := rfl
  clear_value a3
  have b_a3 : a3 < 2^64 := by rw [e_a3]; exact hlhs.2.2.2
  -- b0: argument
  extract_lets -merge +onlyGivenNames b0 at hr
  have e_b0 : b0 = rhs.l0 := rfl
  clear_value b0
  have b_b0 : b0 < 2^64 := by rw [e_b0]; exact hrhs.1
  -- b1: argument
  extract_lets -merge +onlyGivenNames b1 at hr
  have e_b1 : b1 = rhs.l1 := rfl
  clear_value b1
  have b_b1 : b1 < 2^64 := by rw [e_b1]; exact hrhs.2.1
  -- b2: argument
  extract_lets -merge +onlyGivenNames b2 at hr
  have e_b2 : b2 = rhs.l2 := rfl
  clear_value b2
  have b_b2 : b2 < 2^64 := by rw [e_b2]; exact hrhs.2.2.1
  -- b3: argument
  extract_lets -merge +onlyGivenNames b3 at hr
  have e_b3 : b3 = rhs.l3 := rfl
  clear_value b3
  have b_b3 : b3 < 2^64 := by rw [e_b3]; exact hrhs.2.2.2
  -- p0: argument
  extract_lets -merge +onlyGivenNames p0 at hr
  have e_p0 : p0 = modulus.l0 := rfl
  clear_value p0
  have b_p0 : p0 < 2^64 := by rw [e_p0]; exact hm.1
  -- p1: argument
  extract_lets -merge +onlyGivenNames p1 at hr
  have e_p1 : p1 = modulus.l1 := rfl
  clear_value p1
  have b_p1 : p1 < 2^64 := by rw [e_p1]; exact hm.2.1
  -- inv': argument
  extract_lets -merge +onlyGivenNames inv' at hr
  have e_inv' : inv' = inv := rfl
  clear_value inv'
  have b_inv' : inv' < 2^64 := by rw [e_inv']; exact hinv_lt
  -- r0: mul r0,a0,b0
  extract_lets -merge +onlyGivenNames r0 at hr
  have e_r0 : r0 = a0 * b0 % 2^64 := rfl
  clear_value r0
  have b_r0 : r0 < 2^64 := by rw [e_r0]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- r1: mul r1,a1,b0
  extract_lets -merge +onlyGivenNames r1 at hr
  have e_r1 : r1 = a1 * b0 % 2^64 := rfl
  clear_value r1
  have b_r1 : r1 < 2^64 := by rw [e_r1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- r2: mul r2,a2,b0
  extract_lets -merge +onlyGivenNames r2 at hr
  have e_r2 : r2 = a2 * b0 % 2^64 := rfl
  clear_value r2
  have b_r2 : r2 < 2^64 := by rw [e_r2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- r3: mul r3,a3,b0
  extract_lets -merge +onlyGivenNames r3 at hr
  have e_r3 : r3 = a3 * b0 % 2^64 := rfl
  clear_value r3
  have b_r3 : r3 < 2^64 := by rw [e_r3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t0: umulh t0,a0,b0
  extract_lets -merge +onlyGivenNames t0 at hr
  have e_t0 : t0 = a0 * b0 / 2^64 := rfl
  clear_value t0
  have p_t0 : a0 * b0 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a0 b_b0
  have b_t0 : t0 < 2^64 := by rw [e_t0]; exact Nat.div_lt_of_lt_mul p_t0
  have d_t0 : r0 + 2^64 * t0 = a0 * b0 := by
    rw [e_r0, e_t0]; exact Nat.mod_add_div _ _
  clear e_r0 e_t0
  -- t1: umulh t1,a1,b0
  extract_lets -merge +onlyGivenNames t1 at hr
  have e_t1 : t1 = a1 * b0 / 2^64 := rfl
  clear_value t1
  have p_t1 : a1 * b0 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a1 b_b0
  have b_t1 : t1 < 2^64 := by rw [e_t1]; exact Nat.div_lt_of_lt_mul p_t1
  have d_t1 : r1 + 2^64 * t1 = a1 * b0 := by
    rw [e_r1, e_t1]; exact Nat.mod_add_div _ _
  clear e_r1 e_t1
  -- q: mul q,inv,r0
  extract_lets -merge +onlyGivenNames q at hr
  have e_q : q = inv' * r0 % 2^64 := rfl
  clear_value q
  have b_q : q < 2^64 := by rw [e_q]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t2: umulh t2,a2,b0
  extract_lets -merge +onlyGivenNames t2 at hr
  have e_t2 : t2 = a2 * b0 / 2^64 := rfl
  clear_value t2
  have p_t2 : a2 * b0 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a2 b_b0
  have b_t2 : t2 < 2^64 := by rw [e_t2]; exact Nat.div_lt_of_lt_mul p_t2
  have d_t2 : r2 + 2^64 * t2 = a2 * b0 := by
    rw [e_r2, e_t2]; exact Nat.mod_add_div _ _
  clear e_r2 e_t2
  -- t3: umulh t3,a3,b0
  extract_lets -merge +onlyGivenNames t3 at hr
  have e_t3 : t3 = a3 * b0 / 2^64 := rfl
  clear_value t3
  have p_t3 : a3 * b0 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a3 b_b0
  have b_t3 : t3 < 2^64 := by rw [e_t3]; exact Nat.div_lt_of_lt_mul p_t3
  have d_t3 : r3 + 2^64 * t3 = a3 * b0 := by
    rw [e_r3, e_t3]; exact Nat.mod_add_div _ _
  clear e_r3 e_t3
  -- r1_1: adds r1,r1,t0
  extract_lets -merge +onlyGivenNames s r1_1 c at hr
  have e_r1_1 : r1_1 = (r1 + t0 + 0) % 2^64 := rfl
  have e_c : c = (r1 + t0 + 0) / 2^64 := rfl
  clear_value s r1_1 c
  have l_r1_1 : r1_1 + 2^64 * c = r1 + t0 + 0 := by
    rw [e_r1_1, e_c]; exact Nat.mod_add_div _ _
  have b_r1_1 : r1_1 < 2^64 := by rw [e_r1_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c : c ≤ 1 := by
    rw [e_c]; exact addc_carry_le_one r1 t0 0 b_r1 b_t0 (by decide)
  clear e_r1_1 e_c
  -- r2_1: adcs r2,r2,t1
  extract_lets -merge +onlyGivenNames s_1 r2_1 c_1 at hr
  have e_r2_1 : r2_1 = (r2 + t1 + c) % 2^64 := rfl
  have e_c_1 : c_1 = (r2 + t1 + c) / 2^64 := rfl
  clear_value s_1 r2_1 c_1
  have l_r2_1 : r2_1 + 2^64 * c_1 = r2 + t1 + c := by
    rw [e_r2_1, e_c_1]; exact Nat.mod_add_div _ _
  have b_r2_1 : r2_1 < 2^64 := by rw [e_r2_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_1 : c_1 ≤ 1 := by
    rw [e_c_1]; exact addc_carry_le_one r2 t1 c b_r2 b_t1 b_c
  clear e_r2_1 e_c_1
  -- t1_1: mul t1,p1,q
  extract_lets -merge +onlyGivenNames t1_1 at hr
  have e_t1_1 : t1_1 = p1 * q % 2^64 := rfl
  clear_value t1_1
  have b_t1_1 : t1_1 < 2^64 := by rw [e_t1_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- r3_1: adcs r3,r3,t2
  extract_lets -merge +onlyGivenNames s_2 r3_1 c_2 at hr
  have e_r3_1 : r3_1 = (r3 + t2 + c_1) % 2^64 := rfl
  have e_c_2 : c_2 = (r3 + t2 + c_1) / 2^64 := rfl
  clear_value s_2 r3_1 c_2
  have l_r3_1 : r3_1 + 2^64 * c_2 = r3 + t2 + c_1 := by
    rw [e_r3_1, e_c_2]; exact Nat.mod_add_div _ _
  have b_r3_1 : r3_1 < 2^64 := by rw [e_r3_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_2 : c_2 ≤ 1 := by
    rw [e_c_2]; exact addc_carry_le_one r3 t2 c_1 b_r3 b_t2 b_c_1
  clear e_r3_1 e_c_2
  -- r4: adc r4,xzr,t3
  extract_lets -merge +onlyGivenNames r4 at hr
  have e_r4 : r4 = (0 + t3 + c_2) % 2^64 := rfl
  clear_value r4
  have b_r4 : r4 < 2^64 := by rw [e_r4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_r4, b_k_r4, l_r4⟩ :
      ∃ k, k ≤ 1 ∧ r4 + 2^64 * k = 0 + t3 + c_2 :=
    ⟨(0 + t3 + c_2) / 2^64, addc_carry_le_one 0 t3 c_2 (by decide) b_t3 b_c_2,
      by rw [e_r4]; exact Nat.mod_add_div _ _⟩
  clear e_r4
  -- t3_1: lsl t3,q,#62
  extract_lets -merge +onlyGivenNames t3_1 at hr
  have e_t3_1 : t3_1 = q * 2^62 % 2^64 := rfl
  clear_value t3_1
  have b_t3_1 : t3_1 < 2^64 := by rw [e_t3_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- BEGIN round 0
  have hP : modulus.toNat = p0 + 2^64 * p1 + 2^192 * 2^62 := by
    rw [e_p0, e_p1]; simp only [Limbs.toNat, hshape.1, hshape.2, Nat.mul_zero, Nat.add_zero]
  have hP_lt : modulus.toNat < 2^255 := by clear * - hP b_p0 b_p1; omega
  have hL256 : lhs.toNat < 2^256 := by
    clear * - e_a0 e_a1 e_a2 e_a3 b_a0 b_a1 b_a2 b_a3
    simp only [Limbs.toNat]; omega
  have hL_0 : lhs.toNat * b0
      = a0 * b0 + 2^64 * (a1 * b0) + 2^128 * (a2 * b0) + 2^192 * (a3 * b0) := by
    rw [e_a0, e_a1, e_a2, e_a3]; simp only [Limbs.toNat]; ring
  -- The one `adc` of round 0's fold cannot wrap: the high half of a product of two limbs is
  -- at most `2^64 - 2`.
  have hbt3 : a3 * b0 ≤ (2^64 - 1) * (2^64 - 1) :=
    Nat.mul_le_mul (Nat.le_sub_one_of_lt b_a3) (Nat.le_sub_one_of_lt b_b0)
  norm_num at hbt3
  have hk_0 : k_r4 = 0 := by
    clear * - l_r4 d_t3 hbt3 b_c_2
    omega
  have F_0 : r0 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 + 2^256 * r4 = lhs.toNat * b0 := by
    clear * - l_r1_1 l_r2_1 l_r3_1 l_r4 d_t0 d_t1 d_t2 d_t3 hL_0 hk_0
    omega
  have hF0b : lhs.toNat * b0 ≤ (2^256 - 1) * (2^64 - 1) :=
    Nat.mul_le_mul (Nat.le_sub_one_of_lt hL256) (Nat.le_sub_one_of_lt b_b0)
  norm_num at hF0b
  have hLx_0 : lhs.toNat * b0 ≤ lhs.toNat * 18446744073709551615 :=
    Nat.mul_le_mul_left _ (Nat.le_sub_one_of_lt b_b0)
  -- END round 0
  have sh_t3_1 : t3_1 + 2^64 * (q / 2^2) = q * 2^62 := by
    rw [e_t3_1]; exact lsl62_lsr2_split _
  -- round1: round 1
  extract_lets -merge +onlyGivenNames round1 at hr
  have e_round1 : round1 = mulMontRound lhs modulus inv' b1 ⟨r0, r1_1, r2_1, r3_1, r4, q, t1_1, t3_1⟩ := rfl
  clear_value round1
  -- r0_1: round 1 output
  extract_lets -merge +onlyGivenNames r0_1 at hr
  have e_r0_1 : r0_1 = round1.r0 := rfl
  clear_value r0_1
  -- r1_2: round 1 output
  extract_lets -merge +onlyGivenNames r1_2 at hr
  have e_r1_2 : r1_2 = round1.r1 := rfl
  clear_value r1_2
  -- r2_2: round 1 output
  extract_lets -merge +onlyGivenNames r2_2 at hr
  have e_r2_2 : r2_2 = round1.r2 := rfl
  clear_value r2_2
  -- r3_2: round 1 output
  extract_lets -merge +onlyGivenNames r3_2 at hr
  have e_r3_2 : r3_2 = round1.r3 := rfl
  clear_value r3_2
  -- r4_1: round 1 output
  extract_lets -merge +onlyGivenNames r4_1 at hr
  have e_r4_1 : r4_1 = round1.r4 := rfl
  clear_value r4_1
  -- q_1: round 1 output
  extract_lets -merge +onlyGivenNames q_1 at hr
  have e_q_1 : q_1 = round1.q := rfl
  clear_value q_1
  -- t1_2: round 1 output
  extract_lets -merge +onlyGivenNames t1_2 at hr
  have e_t1_2 : t1_2 = round1.t1 := rfl
  clear_value t1_2
  -- t3_2: round 1 output
  extract_lets -merge +onlyGivenNames t3_2 at hr
  have e_t3_2 : t3_2 = round1.t3 := rfl
  clear_value t3_2
  -- BEGIN round 1
  have hround1 : round1 = ⟨r0_1, r1_2, r2_2, r3_2, r4_1, q_1, t1_2, t3_2⟩ := by
    rw [e_r0_1, e_r1_2, e_r2_2, e_r3_2, e_r4_1, e_q_1, e_t1_2, e_t3_2]
  subst hround1
  have hs_1 : lhs.toNat * b1 + lhs.toNat + modulus.toNat + 3 * 2^254 + 2^128 ≤ 2^320 := by
    rw [e_b1, ← Nat.mul_add_one]; exact hsafe.1
  have hLx_1 : lhs.toNat * b1 ≤ lhs.toNat * 18446744073709551615 :=
    Nat.mul_le_mul_left _ (Nat.le_sub_one_of_lt b_b1)
  have hqP_1 : q * modulus.toNat + modulus.toNat ≤ 2^64 * modulus.toNat := by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ b_q
  have H1_1 : (r0 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 + 2^256 * r4) + 2^128 + 3 * 2^254 < 2^320 := by
    clear * - F_0 hF0b hL256
    omega
  have H2_1 : (r0 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 + 2^256 * r4) / 2^64 + modulus.toNat + lhs.toNat * b1 < 2^320 := by
    clear * - F_0 hF0b hs_1 hLx_1 hP_lt
    omega
  obtain ⟨⟨b_r0_1, b_r1_2, b_r2_2, b_r3_2, b_r4_1, b_q_1, b_t1_2, b_t3_2⟩, hq_1, ht1_1, ht3_1, I_1⟩ :=
    mulMontRound_spec lhs modulus inv' b1 ⟨r0, r1_1, r2_1, r3_1, r4, q, t1_1, t3_1⟩ hlhs hm hshape b_inv'
      (by rw [e_inv']; exact hinv) b_b1 ⟨b_r0, b_r1_1, b_r2_1, b_r3_1, b_r4, b_q, b_t1_1, b_t3_1⟩
      e_q (by show t1_1 = modulus.l1 * q % 2^64; rw [e_t1_1, e_p1])
      e_t3_1 H1_1 H2_1 _ e_round1
  simp only [MulMontAcc.toNat] at hq_1 ht1_1 ht3_1 I_1 b_r0_1 b_r1_2 b_r2_2 b_r3_2 b_r4_1 b_q_1 b_t1_2 b_t3_2
  have hF_1 : (r0_1 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + 2^256 * r4_1) < lhs.toNat + modulus.toNat + lhs.toNat * b1 := by
    clear * - I_1 hqP_1 hP F_0 hF0b hLx_0
    omega
  -- END round 1
  -- round2: round 2
  extract_lets -merge +onlyGivenNames round2 at hr
  have e_round2 : round2 = mulMontRound lhs modulus inv' b2 ⟨r0_1, r1_2, r2_2, r3_2, r4_1, q_1, t1_2, t3_2⟩ := rfl
  clear_value round2
  -- r0_2: round 2 output
  extract_lets -merge +onlyGivenNames r0_2 at hr
  have e_r0_2 : r0_2 = round2.r0 := rfl
  clear_value r0_2
  -- r1_3: round 2 output
  extract_lets -merge +onlyGivenNames r1_3 at hr
  have e_r1_3 : r1_3 = round2.r1 := rfl
  clear_value r1_3
  -- r2_3: round 2 output
  extract_lets -merge +onlyGivenNames r2_3 at hr
  have e_r2_3 : r2_3 = round2.r2 := rfl
  clear_value r2_3
  -- r3_3: round 2 output
  extract_lets -merge +onlyGivenNames r3_3 at hr
  have e_r3_3 : r3_3 = round2.r3 := rfl
  clear_value r3_3
  -- r4_2: round 2 output
  extract_lets -merge +onlyGivenNames r4_2 at hr
  have e_r4_2 : r4_2 = round2.r4 := rfl
  clear_value r4_2
  -- q_2: round 2 output
  extract_lets -merge +onlyGivenNames q_2 at hr
  have e_q_2 : q_2 = round2.q := rfl
  clear_value q_2
  -- t1_3: round 2 output
  extract_lets -merge +onlyGivenNames t1_3 at hr
  have e_t1_3 : t1_3 = round2.t1 := rfl
  clear_value t1_3
  -- t3_3: round 2 output
  extract_lets -merge +onlyGivenNames t3_3 at hr
  have e_t3_3 : t3_3 = round2.t3 := rfl
  clear_value t3_3
  -- BEGIN round 2
  have hround2 : round2 = ⟨r0_2, r1_3, r2_3, r3_3, r4_2, q_2, t1_3, t3_3⟩ := by
    rw [e_r0_2, e_r1_3, e_r2_3, e_r3_3, e_r4_2, e_q_2, e_t1_3, e_t3_3]
  subst hround2
  have hs_2 : lhs.toNat * b2 + lhs.toNat + modulus.toNat + 3 * 2^254 + 2^128 ≤ 2^320 := by
    rw [e_b2, ← Nat.mul_add_one]; exact hsafe.2.1
  have hLx_2 : lhs.toNat * b2 ≤ lhs.toNat * 18446744073709551615 :=
    Nat.mul_le_mul_left _ (Nat.le_sub_one_of_lt b_b2)
  have hqP_2 : q_1 * modulus.toNat + modulus.toNat ≤ 2^64 * modulus.toNat := by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ b_q_1
  have H1_2 : (r0_1 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + 2^256 * r4_1) + 2^128 + 3 * 2^254 < 2^320 := by
    clear * - hF_1 hs_1
    omega
  have H2_2 : (r0_1 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + 2^256 * r4_1) / 2^64 + modulus.toNat + lhs.toNat * b2 < 2^320 := by
    clear * - hF_1 hs_2 hLx_2 hLx_1 hP_lt
    omega
  obtain ⟨⟨b_r0_2, b_r1_3, b_r2_3, b_r3_3, b_r4_2, b_q_2, b_t1_3, b_t3_3⟩, hq_2, ht1_2, ht3_2, I_2⟩ :=
    mulMontRound_spec lhs modulus inv' b2 ⟨r0_1, r1_2, r2_2, r3_2, r4_1, q_1, t1_2, t3_2⟩ hlhs hm hshape b_inv'
      (by rw [e_inv']; exact hinv) b_b2 ⟨b_r0_1, b_r1_2, b_r2_2, b_r3_2, b_r4_1, b_q_1, b_t1_2, b_t3_2⟩
      hq_1 ht1_1
      ht3_1 H1_2 H2_2 _ e_round2
  simp only [MulMontAcc.toNat] at hq_2 ht1_2 ht3_2 I_2 b_r0_2 b_r1_3 b_r2_3 b_r3_3 b_r4_2 b_q_2 b_t1_3 b_t3_3
  have hF_2 : (r0_2 + 2^64 * r1_3 + 2^128 * r2_3 + 2^192 * r3_3 + 2^256 * r4_2) < lhs.toNat + modulus.toNat + lhs.toNat * b2 := by
    clear * - I_2 hqP_2 hP hF_1 hLx_1
    omega
  -- END round 2
  -- round3: round 3
  extract_lets -merge +onlyGivenNames round3 at hr
  have e_round3 : round3 = mulMontRound lhs modulus inv' b3 ⟨r0_2, r1_3, r2_3, r3_3, r4_2, q_2, t1_3, t3_3⟩ := rfl
  clear_value round3
  -- r0_3: round 3 output
  extract_lets -merge +onlyGivenNames r0_3 at hr
  have e_r0_3 : r0_3 = round3.r0 := rfl
  clear_value r0_3
  -- r1_4: round 3 output
  extract_lets -merge +onlyGivenNames r1_4 at hr
  have e_r1_4 : r1_4 = round3.r1 := rfl
  clear_value r1_4
  -- r2_4: round 3 output
  extract_lets -merge +onlyGivenNames r2_4 at hr
  have e_r2_4 : r2_4 = round3.r2 := rfl
  clear_value r2_4
  -- r3_4: round 3 output
  extract_lets -merge +onlyGivenNames r3_4 at hr
  have e_r3_4 : r3_4 = round3.r3 := rfl
  clear_value r3_4
  -- r4_3: round 3 output
  extract_lets -merge +onlyGivenNames r4_3 at hr
  have e_r4_3 : r4_3 = round3.r4 := rfl
  clear_value r4_3
  -- q_3: round 3 output
  extract_lets -merge +onlyGivenNames q_3 at hr
  have e_q_3 : q_3 = round3.q := rfl
  clear_value q_3
  -- t1_4: round 3 output
  extract_lets -merge +onlyGivenNames t1_4 at hr
  have e_t1_4 : t1_4 = round3.t1 := rfl
  clear_value t1_4
  -- t3_4: round 3 output
  extract_lets -merge +onlyGivenNames t3_4 at hr
  have e_t3_4 : t3_4 = round3.t3 := rfl
  clear_value t3_4
  -- BEGIN round 3
  have hround3 : round3 = ⟨r0_3, r1_4, r2_4, r3_4, r4_3, q_3, t1_4, t3_4⟩ := by
    rw [e_r0_3, e_r1_4, e_r2_4, e_r3_4, e_r4_3, e_q_3, e_t1_4, e_t3_4]
  subst hround3
  have hs_3 : lhs.toNat * b3 + lhs.toNat + modulus.toNat + 3 * 2^254 + 2^128 ≤ 2^320 := by
    rw [e_b3, ← Nat.mul_add_one]; exact hsafe.2.2
  have hLx_3 : lhs.toNat * b3 ≤ lhs.toNat * 18446744073709551615 :=
    Nat.mul_le_mul_left _ (Nat.le_sub_one_of_lt b_b3)
  have hqP_3 : q_2 * modulus.toNat + modulus.toNat ≤ 2^64 * modulus.toNat := by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ b_q_2
  have H1_3 : (r0_2 + 2^64 * r1_3 + 2^128 * r2_3 + 2^192 * r3_3 + 2^256 * r4_2) + 2^128 + 3 * 2^254 < 2^320 := by
    clear * - hF_2 hs_2
    omega
  have H2_3 : (r0_2 + 2^64 * r1_3 + 2^128 * r2_3 + 2^192 * r3_3 + 2^256 * r4_2) / 2^64 + modulus.toNat + lhs.toNat * b3 < 2^320 := by
    clear * - hF_2 hs_3 hLx_3 hLx_2 hP_lt
    omega
  obtain ⟨⟨b_r0_3, b_r1_4, b_r2_4, b_r3_4, b_r4_3, b_q_3, b_t1_4, b_t3_4⟩, hq_3, ht1_3, ht3_3, I_3⟩ :=
    mulMontRound_spec lhs modulus inv' b3 ⟨r0_2, r1_3, r2_3, r3_3, r4_2, q_2, t1_3, t3_3⟩ hlhs hm hshape b_inv'
      (by rw [e_inv']; exact hinv) b_b3 ⟨b_r0_2, b_r1_3, b_r2_3, b_r3_3, b_r4_2, b_q_2, b_t1_3, b_t3_3⟩
      hq_2 ht1_2
      ht3_2 H1_3 H2_3 _ e_round3
  simp only [MulMontAcc.toNat] at hq_3 ht1_3 ht3_3 I_3 b_r0_3 b_r1_4 b_r2_4 b_r3_4 b_r4_3 b_q_3 b_t1_4 b_t3_4
  have hF_3 : (r0_3 + 2^64 * r1_4 + 2^128 * r2_4 + 2^192 * r3_4 + 2^256 * r4_3) < lhs.toNat + modulus.toNat + lhs.toNat * b3 := by
    clear * - I_3 hqP_3 hP hF_2 hLx_2
    omega
  -- END round 3
  -- c_3: subs xzr,r0,#1
  extract_lets -merge +onlyGivenNames c_3 at hr
  have e_c_3 : c_3 = (r0_3 + 2^64 - 1 - (1 - 1)) / 2^64 := rfl
  clear_value c_3
  have b_c_3 : c_3 ≤ 1 := by rw [e_c_3]; exact subc_carry_le_one r0_3 1 1 b_r0_3
  have l_c_3 : (c_3 = 1 ∧ 1 + 1 ≤ r0_3 + 1) ∨ (c_3 = 0 ∧ r0_3 + 1 < 1 + 1) :=
    subc_carry_cases r0_3 1 1 _ e_c_3 b_r0_3 (by decide) (by decide)
  clear e_c_3
  -- t0_1: umulh t0,p0,q
  extract_lets -merge +onlyGivenNames t0_1 at hr
  have e_t0_1 : t0_1 = p0 * q_3 / 2^64 := rfl
  clear_value t0_1
  have p_t0_1 : p0 * q_3 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_p0 b_q_3
  have b_t0_1 : t0_1 < 2^64 := by rw [e_t0_1]; exact Nat.div_lt_of_lt_mul p_t0_1
  obtain ⟨lo_t0_1, b_lo_t0_1, d_t0_1⟩ :
      ∃ lo, lo < 2^64 ∧ lo + 2^64 * t0_1 = p0 * q_3 :=
    ⟨p0 * q_3 % 2^64, Nat.mod_lt _ (Nat.two_pow_pos _),
      by rw [e_t0_1]; exact Nat.mod_add_div _ _⟩
  clear e_t0_1
  -- r1_5: adcs r1,r1,t1
  extract_lets -merge +onlyGivenNames s_3 r1_5 c_4 at hr
  have e_r1_5 : r1_5 = (r1_4 + t1_4 + c_3) % 2^64 := rfl
  have e_c_4 : c_4 = (r1_4 + t1_4 + c_3) / 2^64 := rfl
  clear_value s_3 r1_5 c_4
  have l_r1_5 : r1_5 + 2^64 * c_4 = r1_4 + t1_4 + c_3 := by
    rw [e_r1_5, e_c_4]; exact Nat.mod_add_div _ _
  have b_r1_5 : r1_5 < 2^64 := by rw [e_r1_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_4 : c_4 ≤ 1 := by
    rw [e_c_4]; exact addc_carry_le_one r1_4 t1_4 c_3 b_r1_4 b_t1_4 b_c_3
  clear e_r1_5 e_c_4
  -- t1_5: umulh t1,p1,q
  extract_lets -merge +onlyGivenNames t1_5 at hr
  have e_t1_5 : t1_5 = p1 * q_3 / 2^64 := rfl
  clear_value t1_5
  have p_t1_5 : p1 * q_3 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_p1 b_q_3
  have b_t1_5 : t1_5 < 2^64 := by rw [e_t1_5]; exact Nat.div_lt_of_lt_mul p_t1_5
  obtain ⟨lo_t1_5, b_lo_t1_5, d_t1_5⟩ :
      ∃ lo, lo < 2^64 ∧ lo + 2^64 * t1_5 = p1 * q_3 :=
    ⟨p1 * q_3 % 2^64, Nat.mod_lt _ (Nat.two_pow_pos _),
      by rw [e_t1_5]; exact Nat.mod_add_div _ _⟩
  clear e_t1_5
  -- r2_5: adcs r2,r2,xzr
  extract_lets -merge +onlyGivenNames s_4 r2_5 c_5 at hr
  have e_r2_5 : r2_5 = (r2_4 + 0 + c_4) % 2^64 := rfl
  have e_c_5 : c_5 = (r2_4 + 0 + c_4) / 2^64 := rfl
  clear_value s_4 r2_5 c_5
  have l_r2_5 : r2_5 + 2^64 * c_5 = r2_4 + 0 + c_4 := by
    rw [e_r2_5, e_c_5]; exact Nat.mod_add_div _ _
  have b_r2_5 : r2_5 < 2^64 := by rw [e_r2_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_5 : c_5 ≤ 1 := by
    rw [e_c_5]; exact addc_carry_le_one r2_4 0 c_4 b_r2_4 (by decide) b_c_4
  clear e_r2_5 e_c_5
  -- r3_5: adcs r3,r3,t3
  extract_lets -merge +onlyGivenNames s_5 r3_5 c_6 at hr
  have e_r3_5 : r3_5 = (r3_4 + t3_4 + c_5) % 2^64 := rfl
  have e_c_6 : c_6 = (r3_4 + t3_4 + c_5) / 2^64 := rfl
  clear_value s_5 r3_5 c_6
  have l_r3_5 : r3_5 + 2^64 * c_6 = r3_4 + t3_4 + c_5 := by
    rw [e_r3_5, e_c_6]; exact Nat.mod_add_div _ _
  have b_r3_5 : r3_5 < 2^64 := by rw [e_r3_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_6 : c_6 ≤ 1 := by
    rw [e_c_6]; exact addc_carry_le_one r3_4 t3_4 c_5 b_r3_4 b_t3_4 b_c_5
  clear e_r3_5 e_c_6
  -- t3_5: lsr t3,q,#2
  extract_lets -merge +onlyGivenNames t3_5 at hr
  have e_t3_5 : t3_5 = q_3 / 2^2 := rfl
  clear_value t3_5
  have b_t3_5 : t3_5 < 2^62 := by
    rw [e_t3_5]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_q_3 (by norm_num))
  -- r4_4: adc r4,r4,xzr
  extract_lets -merge +onlyGivenNames r4_4 at hr
  have e_r4_4 : r4_4 = (r4_3 + 0 + c_6) % 2^64 := rfl
  clear_value r4_4
  have b_r4_4 : r4_4 < 2^64 := by rw [e_r4_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_r4_4, b_k_r4_4, l_r4_4⟩ :
      ∃ k, k ≤ 1 ∧ r4_4 + 2^64 * k = r4_3 + 0 + c_6 :=
    ⟨(r4_3 + 0 + c_6) / 2^64, addc_carry_le_one r4_3 0 c_6 b_r4_3 (by decide) b_c_6,
      by rw [e_r4_4]; exact Nat.mod_add_div _ _⟩
  clear e_r4_4
  -- BEGIN final reduction
  have hc_3 : r0_3 + lo_t0_1 = 2^64 * c_3 := by
    have h := cancel_low r0_3 inv' p0 (by rw [e_inv', e_p0]; exact hinv)
    rw [← hq_3, ← d_t0_1, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt b_lo_t0_1] at h
    clear * - h b_r0_3 b_lo_t0_1 l_c_3
    omega
  have hPq_3 : q_3 * modulus.toNat = p0 * q_3 + 2^64 * (p1 * q_3) + 2^254 * q_3 := by
    rw [hP]; ring
  have hbt3_4 : t3_4 ≤ 3 * 2^62 := by clear * - ht3_3; omega
  have hsumr : (r0_3 + 2^64 * r1_5 + 2^128 * r2_5 + 2^192 * r3_5 + 2^256 * r4_4) + 2^320 * k_r4_4
      = (r0_3 + 2^64 * r1_4 + 2^128 * r2_4 + 2^192 * r3_4 + 2^256 * r4_3) + 2^64 * t1_4 + 2^64 * c_3 + 2^192 * t3_4 := by
    clear * - l_r1_5 l_r2_5 l_r3_5 l_r4_4
    omega
  have hkr : k_r4_4 = 0 := by
    clear * - hsumr hF_3 hs_3 b_t1_4 b_c_3 hbt3_4 b_k_r4_4 b_r0_3 b_r1_5 b_r2_5 b_r3_5 b_r4_4
    omega
  -- END final reduction
  -- r0_4: adds r0,r1,t0
  extract_lets -merge +onlyGivenNames s_6 r0_4 c_7 at hr
  have e_r0_4 : r0_4 = (r1_5 + t0_1 + 0) % 2^64 := rfl
  have e_c_7 : c_7 = (r1_5 + t0_1 + 0) / 2^64 := rfl
  clear_value s_6 r0_4 c_7
  have l_r0_4 : r0_4 + 2^64 * c_7 = r1_5 + t0_1 + 0 := by
    rw [e_r0_4, e_c_7]; exact Nat.mod_add_div _ _
  have b_r0_4 : r0_4 < 2^64 := by rw [e_r0_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_7 : c_7 ≤ 1 := by
    rw [e_c_7]; exact addc_carry_le_one r1_5 t0_1 0 b_r1_5 b_t0_1 (by decide)
  clear e_r0_4 e_c_7
  -- r1_6: adcs r1,r2,t1
  extract_lets -merge +onlyGivenNames s_7 r1_6 c_8 at hr
  have e_r1_6 : r1_6 = (r2_5 + t1_5 + c_7) % 2^64 := rfl
  have e_c_8 : c_8 = (r2_5 + t1_5 + c_7) / 2^64 := rfl
  clear_value s_7 r1_6 c_8
  have l_r1_6 : r1_6 + 2^64 * c_8 = r2_5 + t1_5 + c_7 := by
    rw [e_r1_6, e_c_8]; exact Nat.mod_add_div _ _
  have b_r1_6 : r1_6 < 2^64 := by rw [e_r1_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_8 : c_8 ≤ 1 := by
    rw [e_c_8]; exact addc_carry_le_one r2_5 t1_5 c_7 b_r2_5 b_t1_5 b_c_7
  clear e_r1_6 e_c_8
  -- r2_6: adcs r2,r3,xzr
  extract_lets -merge +onlyGivenNames s_8 r2_6 c_9 at hr
  have e_r2_6 : r2_6 = (r3_5 + 0 + c_8) % 2^64 := rfl
  have e_c_9 : c_9 = (r3_5 + 0 + c_8) / 2^64 := rfl
  clear_value s_8 r2_6 c_9
  have l_r2_6 : r2_6 + 2^64 * c_9 = r3_5 + 0 + c_8 := by
    rw [e_r2_6, e_c_9]; exact Nat.mod_add_div _ _
  have b_r2_6 : r2_6 < 2^64 := by rw [e_r2_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_9 : c_9 ≤ 1 := by
    rw [e_c_9]; exact addc_carry_le_one r3_5 0 c_8 b_r3_5 (by decide) b_c_8
  clear e_r2_6 e_c_9
  -- r3_6: adcs r3,r4,t3
  extract_lets -merge +onlyGivenNames s_9 r3_6 at hr
  have e_r3_6 : r3_6 = (r4_4 + t3_5 + c_9) % 2^64 := rfl
  clear_value s_9 r3_6
  have b_r3_6 : r3_6 < 2^64 := by rw [e_r3_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_r3_6, b_k_r3_6, l_r3_6⟩ :
      ∃ k, k ≤ 1 ∧ r3_6 + 2^64 * k = r4_4 + t3_5 + c_9 :=
    ⟨(r4_4 + t3_5 + c_9) / 2^64, addc_carry_le_one r4_4 t3_5 c_9 b_r4_4 (lt_of_lt_of_le b_t3_5 (by norm_num)) b_c_9,
      by rw [e_r3_6]; exact Nat.mod_add_div _ _⟩
  clear e_r3_6
  -- BEGIN final shift
  have hsums : (r0_4 + 2^64 * r1_6 + 2^128 * r2_6 + 2^192 * r3_6 + 2^256 * k_r3_6)
      = (r1_5 + 2^64 * r2_5 + 2^128 * r3_5 + 2^192 * r4_4) + t0_1 + 2^64 * t1_5
        + 2^192 * t3_5 := by
    clear * - l_r0_4 l_r1_6 l_r2_6 l_r3_6
    omega
  have hlot1 : t1_4 = lo_t1_5 := by clear * - ht1_3 e_p1 d_t1_5 b_lo_t1_5; subst e_p1; omega
  have hsh : t3_4 + 2^64 * t3_5 = q_3 * 2^62 := by clear * - ht3_3 e_t3_5; omega
  have I_4 : 2^64 * (r0_4 + 2^64 * r1_6 + 2^128 * r2_6 + 2^192 * r3_6 + 2^256 * k_r3_6) = (r0_3 + 2^64 * r1_4 + 2^128 * r2_4 + 2^192 * r3_4 + 2^256 * r4_3) + q_3 * modulus.toNat := by
    clear * - hsums hsumr hkr hc_3 d_t0_1 d_t1_5 hlot1 hsh hPq_3
    omega
  have hLR : lhs.toNat * rhs.toNat
      = lhs.toNat * b0 + 2^64 * (lhs.toNat * b1) + 2^128 * (lhs.toNat * b2)
        + 2^192 * (lhs.toNat * b3) := by
    rw [e_b0, e_b1, e_b2, e_b3]; simp only [Limbs.toNat]; ring
  have hQ : q + 2^64 * q_1 + 2^128 * q_2 + 2^192 * q_3 < 2^256 := by
    clear * - b_q b_q_1 b_q_2 b_q_3; omega
  have hQP : (q + 2^64 * q_1 + 2^128 * q_2 + 2^192 * q_3) * modulus.toNat
      = q * modulus.toNat + 2^64 * (q_1 * modulus.toNat) + 2^128 * (q_2 * modulus.toNat)
        + 2^192 * (q_3 * modulus.toNat) := by
    ring
  -- The four rounds and the final reduction compose to `2^256 * acc = lhs * rhs + Q * p`.
  have hmain : 2^256 * (r0_4 + 2^64 * r1_6 + 2^128 * r2_6 + 2^192 * r3_6 + 2^256 * k_r3_6)
      = lhs.toNat * rhs.toNat
        + (q + 2^64 * q_1 + 2^128 * q_2 + 2^192 * q_3) * modulus.toNat := by
    clear * - F_0 I_1 I_2 I_3 I_4 hLR hQP
    omega
  -- END final shift
  -- q_4: mov q,#0x4000000000000000
  extract_lets -merge +onlyGivenNames q_4 at hr
  have e_q_4 : q_4 = 4611686018427387904 := rfl
  clear_value q_4
  have b_q_4 : q_4 < 2^64 := by rw [e_q_4]; decide
  -- t0_2: subs t0,r0,p0
  extract_lets -merge +onlyGivenNames s_10 t0_2 c_10 at hr
  have e_t0_2 : t0_2 = (r0_4 + 2^64 - p0 - (1 - 1)) % 2^64 := rfl
  have e_c_10 : c_10 = (r0_4 + 2^64 - p0 - (1 - 1)) / 2^64 := rfl
  clear_value s_10 t0_2 c_10
  have l_t0_2 : t0_2 + 2^64 * c_10 + p0 + 1 = r0_4 + 2^64 + 1 := by
    rw [e_t0_2, e_c_10]; exact subc_lin r0_4 p0 1 b_p0 (by decide)
  have b_t0_2 : t0_2 < 2^64 := by rw [e_t0_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_10 : c_10 ≤ 1 := by
    rw [e_c_10]; exact subc_carry_le_one r0_4 p0 1 b_r0_4
  clear e_t0_2 e_c_10
  -- t1_6: sbcs t1,r1,p1
  extract_lets -merge +onlyGivenNames s_11 t1_6 c_11 at hr
  have e_t1_6 : t1_6 = (r1_6 + 2^64 - p1 - (1 - c_10)) % 2^64 := rfl
  have e_c_11 : c_11 = (r1_6 + 2^64 - p1 - (1 - c_10)) / 2^64 := rfl
  clear_value s_11 t1_6 c_11
  have l_t1_6 : t1_6 + 2^64 * c_11 + p1 + 1 = r1_6 + 2^64 + c_10 := by
    rw [e_t1_6, e_c_11]; exact subc_lin r1_6 p1 c_10 b_p1 b_c_10
  have b_t1_6 : t1_6 < 2^64 := by rw [e_t1_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_11 : c_11 ≤ 1 := by
    rw [e_c_11]; exact subc_carry_le_one r1_6 p1 c_10 b_r1_6
  clear e_t1_6 e_c_11
  -- t2_1: sbcs t2,r2,xzr
  extract_lets -merge +onlyGivenNames s_12 t2_1 c_12 at hr
  have e_t2_1 : t2_1 = (r2_6 + 2^64 - 0 - (1 - c_11)) % 2^64 := rfl
  have e_c_12 : c_12 = (r2_6 + 2^64 - 0 - (1 - c_11)) / 2^64 := rfl
  clear_value s_12 t2_1 c_12
  have l_t2_1 : t2_1 + 2^64 * c_12 + 0 + 1 = r2_6 + 2^64 + c_11 := by
    rw [e_t2_1, e_c_12]; exact subc_lin r2_6 0 c_11 (by decide) b_c_11
  have b_t2_1 : t2_1 < 2^64 := by rw [e_t2_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_12 : c_12 ≤ 1 := by
    rw [e_c_12]; exact subc_carry_le_one r2_6 0 c_11 b_r2_6
  clear e_t2_1 e_c_12
  -- t3_6: sbcs t3,r3,q
  extract_lets -merge +onlyGivenNames s_13 t3_6 c_13 at hr
  have e_t3_6 : t3_6 = (r3_6 + 2^64 - q_4 - (1 - c_12)) % 2^64 := rfl
  have e_c_13 : c_13 = (r3_6 + 2^64 - q_4 - (1 - c_12)) / 2^64 := rfl
  clear_value s_13 t3_6 c_13
  have l_t3_6 : t3_6 + 2^64 * c_13 + q_4 + 1 = r3_6 + 2^64 + c_12 := by
    rw [e_t3_6, e_c_13]; exact subc_lin r3_6 q_4 c_12 b_q_4 b_c_12
  have b_t3_6 : t3_6 < 2^64 := by rw [e_t3_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_13 : c_13 ≤ 1 := by
    rw [e_c_13]; exact subc_carry_le_one r3_6 q_4 c_12 b_r3_6
  clear e_t3_6 e_c_13
  -- r0_5: csel r0,r0,t0,lo
  extract_lets -merge +onlyGivenNames r0_5 at hr
  have e_r0_5 : r0_5 = (if c_13 = 0 then r0_4 else t0_2) := rfl
  clear_value r0_5
  have b_r0_5 : r0_5 < 2^64 := by
    rw [e_r0_5]; split <;> first | exact b_r0_4 | exact b_t0_2
  -- r1_7: csel r1,r1,t1,lo
  extract_lets -merge +onlyGivenNames r1_7 at hr
  have e_r1_7 : r1_7 = (if c_13 = 0 then r1_6 else t1_6) := rfl
  clear_value r1_7
  have b_r1_7 : r1_7 < 2^64 := by
    rw [e_r1_7]; split <;> first | exact b_r1_6 | exact b_t1_6
  -- r2_7: csel r2,r2,t2,lo
  extract_lets -merge +onlyGivenNames r2_7 at hr
  have e_r2_7 : r2_7 = (if c_13 = 0 then r2_6 else t2_1) := rfl
  clear_value r2_7
  have b_r2_7 : r2_7 < 2^64 := by
    rw [e_r2_7]; split <;> first | exact b_r2_6 | exact b_t2_1
  -- r3_7: csel r3,r3,t3,lo
  extract_lets -merge +onlyGivenNames r3_7 at hr
  have e_r3_7 : r3_7 = (if c_13 = 0 then r3_6 else t3_6) := rfl
  clear_value r3_7
  have b_r3_7 : r3_7 < 2^64 := by
    rw [e_r3_7]; split <;> first | exact b_r3_6 | exact b_t3_6
  subst hr
  -- BEGIN conclusion
  have hQPle : (q + 2^64 * q_1 + 2^128 * q_2 + 2^192 * q_3) * modulus.toNat + modulus.toNat
      ≤ 2^256 * modulus.toNat := by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hQ
  -- `hfinal` puts the accumulator below `2 * p`, hence below `2^256`: the carry out of its
  -- fourth limb, which the block drops, is `0`. The four-limb subtraction's carry is set
  -- (`c_13 = 1`) exactly when the accumulator is at least `p`; then the result is the
  -- difference, otherwise the accumulator.
  have hA : r0_4 + 2^64 * r1_6 + 2^128 * r2_6 + 2^192 * r3_6 + 2^256 * k_r3_6
      < 2 * modulus.toNat := by
    clear * - hmain hfinal hQPle; omega
  have hk : k_r3_6 = 0 := by clear * - hA hP_lt; omega
  have hD : t0_2 + 2^64 * t1_6 + 2^128 * t2_1 + 2^192 * t3_6
        + (p0 + 2^64 * p1 + 2^192 * q_4) + 2^256 * c_13
      = r0_4 + 2^64 * r1_6 + 2^128 * r2_6 + 2^192 * r3_6 + 2^256 := by
    clear * - l_t0_2 l_t1_6 l_t2_1 l_t3_6; omega
  refine ⟨⟨b_r0_5, b_r1_7, b_r2_7, b_r3_7⟩, ?_⟩
  show r0_5 + 2^64 * r1_7 + 2^128 * r2_7 + 2^192 * r3_7 < modulus.toNat ∧
    2^256 * (r0_5 + 2^64 * r1_7 + 2^128 * r2_7 + 2^192 * r3_7)
      ≡ lhs.toNat * rhs.toNat [MOD modulus.toNat]
  obtain hc | hc : c_13 = 0 ∨ c_13 = 1 := by clear * - b_c_13; omega
  · rw [if_pos hc] at e_r0_5 e_r1_7 e_r2_7 e_r3_7
    refine ⟨?_, modEq_of_add_mul _ _ 0 (q + 2^64 * q_1 + 2^128 * q_2 + 2^192 * q_3) _ (by
      clear * - hmain hk e_r0_5 e_r1_7 e_r2_7 e_r3_7
      omega)⟩
    · clear * - hD hc hk hP e_q_4 b_t0_2 b_t1_6 b_t2_1 b_t3_6 e_r0_5 e_r1_7 e_r2_7 e_r3_7
      omega
  · rw [if_neg (by clear * - hc; omega)] at e_r0_5 e_r1_7 e_r2_7 e_r3_7
    refine ⟨?_, modEq_of_add_mul _ _ (2^256) (q + 2^64 * q_1 + 2^128 * q_2 + 2^192 * q_3) _ (by
      clear * - hmain hD hc hk hP e_q_4 e_r0_5 e_r1_7 e_r2_7 e_r3_7
      omega)⟩
    · clear * - hD hc hk hA hP e_q_4 e_r0_5 e_r1_7 e_r2_7 e_r3_7
      omega
  -- END conclusion

-- BEGIN mulMont_spec corollaries
/-- The contract the crate's callers use: a canonical left operand and any four-limb right
operand. -/
theorem mulMont_spec_of_lhs_lt (lhs rhs modulus : Limbs) (inv : Nat) (hlhs : lhs.Bounded)
    (hrhs : rhs.Bounded) (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinv_lt : inv < 2^64) (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hlt : lhs.toNat < modulus.toNat) :
    ∀ r, r = mulMont lhs rhs modulus inv →
      r.Bounded ∧ r.toNat < modulus.toNat ∧
        2^256 * r.toNat ≡ lhs.toNat * rhs.toNat [MOD modulus.toNat] := by
  have hP := Limbs.toNat_lt_of_shape modulus hm hshape
  have hR := Limbs.toNat_lt rhs hrhs
  -- `lhs * (rhs_i + 1) ≤ (p - 1) * 2^64`, and `lhs * rhs < p * 2^256`.
  have s1 : lhs.toNat * (rhs.l1 + 1) ≤ (modulus.toNat - 1) * 2^64 :=
    Nat.mul_le_mul (Nat.le_sub_one_of_lt hlt) hrhs.2.1
  have s2 : lhs.toNat * (rhs.l2 + 1) ≤ (modulus.toNat - 1) * 2^64 :=
    Nat.mul_le_mul (Nat.le_sub_one_of_lt hlt) hrhs.2.2.1
  have s3 : lhs.toNat * (rhs.l3 + 1) ≤ (modulus.toNat - 1) * 2^64 :=
    Nat.mul_le_mul (Nat.le_sub_one_of_lt hlt) hrhs.2.2.2
  have f : (lhs.toNat + 1) * (rhs.toNat + 1) ≤ modulus.toNat * 2^256 := Nat.mul_le_mul hlt hR
  rw [Nat.add_one_mul, Nat.mul_add_one] at f
  exact mulMont_spec lhs rhs modulus inv hlhs hrhs hm hshape hinv_lt hinv
    ⟨by omega, by omega, by omega⟩ (by omega)

/-- The other safe contract: any four-limb left operand, a canonical right operand whose limbs 1
to 3 are at most `2^64 - 3`. -/
theorem mulMont_spec_of_rhs_lt (lhs rhs modulus : Limbs) (inv : Nat) (hlhs : lhs.Bounded)
    (hrhs : rhs.Bounded) (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinv_lt : inv < 2^64) (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hlt : rhs.toNat < modulus.toNat)
    (hlimbs : rhs.l1 + 3 ≤ 2^64 ∧ rhs.l2 + 3 ≤ 2^64 ∧ rhs.l3 + 3 ≤ 2^64) :
    ∀ r, r = mulMont lhs rhs modulus inv →
      r.Bounded ∧ r.toNat < modulus.toNat ∧
        2^256 * r.toNat ≡ lhs.toNat * rhs.toNat [MOD modulus.toNat] := by
  have hP := Limbs.toNat_lt_of_shape modulus hm hshape
  have hL := Limbs.toNat_lt lhs hlhs
  -- `lhs * (rhs_i + 1) ≤ (2^256 - 1) * (2^64 - 2)`, and `lhs * rhs < 2^256 * p`.
  have c1 : rhs.l1 + 1 ≤ 2^64 - 2 := by omega
  have c2 : rhs.l2 + 1 ≤ 2^64 - 2 := by omega
  have c3 : rhs.l3 + 1 ≤ 2^64 - 2 := by omega
  have s1 : lhs.toNat * (rhs.l1 + 1) ≤ (2^256 - 1) * (2^64 - 2) :=
    Nat.mul_le_mul (Nat.le_sub_one_of_lt hL) c1
  have s2 : lhs.toNat * (rhs.l2 + 1) ≤ (2^256 - 1) * (2^64 - 2) :=
    Nat.mul_le_mul (Nat.le_sub_one_of_lt hL) c2
  have s3 : lhs.toNat * (rhs.l3 + 1) ≤ (2^256 - 1) * (2^64 - 2) :=
    Nat.mul_le_mul (Nat.le_sub_one_of_lt hL) c3
  norm_num at s1 s2 s3
  have f : (lhs.toNat + 1) * (rhs.toNat + 1) ≤ 2^256 * modulus.toNat := Nat.mul_le_mul hL hlt
  rw [Nat.add_one_mul, Nat.mul_add_one] at f
  exact mulMont_spec lhs rhs modulus inv hlhs hrhs hm hshape hinv_lt hinv
    ⟨by omega, by omega, by omega⟩ (by omega)

/-- The crate's conversion out of Montgomery form, `fromMont`, is the multiplication block with
`1` as its right operand, a canonical operand whose limbs 1 to 3 are zero: for every four-limb
`value`, the result is below `p` and `2^256 * result ≡ value (mod p)`. -/
theorem fromMont_spec (value modulus : Limbs) (inv : Nat) (hv : value.Bounded)
    (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinv_lt : inv < 2^64) (hinv : (inv * modulus.l0 + 1) % 2^64 = 0) :
    ∀ r, r = fromMont value modulus inv →
      r.Bounded ∧ r.toNat < modulus.toNat ∧
        2^256 * r.toNat ≡ value.toNat [MOD modulus.toNat] := by
  intro r hr
  have h1 : (⟨1, 0, 0, 0⟩ : Limbs).toNat = 1 := by decide
  have hlt : (⟨1, 0, 0, 0⟩ : Limbs).toNat < modulus.toNat := by
    rw [h1]; simp only [Limbs.toNat, hshape.1, hshape.2]; omega
  have h := mulMont_spec_of_rhs_lt value ⟨1, 0, 0, 0⟩ modulus inv hv
    (by unfold Limbs.Bounded; decide) hm hshape hinv_lt hinv hlt
    ⟨by decide, by decide, by decide⟩ r (hr.trans rfl)
  rw [h1, Nat.mul_one] at h
  exact h
-- END mulMont_spec corollaries

end PastaAsm.AArch64
