/-
Copyright Amazon.com, Inc. or its affiliates (the block, adapted from s2n-bignum).
Copyright (c) 2026 the pasta_curves contributors (the transcription and the proofs).
-/
import PastaCurves.Spec
import PastaCurves.X86_64.Spec.Arithmetic
import PastaCurves.X86_64.Transcription
import PastaCurves.Inversion.Round
import Mathlib.Tactic.ClearExcept

/-!
# Correctness of the inversion's almost-Montgomery reduction block

The block is proved equal to the shared word-level function `amontred` of `Inversion/Round.lean`,
with the AArch64 block's statement. As there, Lemma 10 (`amontredZ_spec`) supplies the bound that
lets the five-word arithmetic modulo `2^320` stand for the integers. The x86-64 block adds `w p`
as `from_mont` does: `neg` sets the cancellation's carry, one chain adds the high halves and
`w 2^254`, and a second chain adds the low half of `w p1`.
-/

-- The conclusion's arithmetic reaches `2^320`, which the default threshold leaves unevaluated.
set_option exponentiation.threshold 400

namespace PastaCurves.X86_64

-- BEGIN amontredBlock_spec statement
/-- The almost-Montgomery reduction by the inline block is the shared `amontred`, for a bounded
five-word input below `2^315` in magnitude and the modulus and `inv` of a Pasta field. The words
compute modulo `2^320`; the input's sign word and the dropped top carries are accounted for by
`amontredZ_spec`'s bound, which places the true result in `[0, 2^256)`. -/
theorem amontredBlock_spec (F : PastaField) (t : Signed5) (modulus : Limbs) (inv : Nat)
    (hmod : modulus = F.modulus) (hinv' : inv = F.inv)
    (ht : t.Bounded) (htv : |t.toInt| < 2^315) :
    ∀ res, res = amontredBlock t modulus inv → res = Inversion.amontred t F.modulus F.inv := by
  intro res hres
  have hm : modulus.Bounded := hmod ▸ F.bounded
  have hinv_lt : inv < 2^64 := hinv' ▸ F.inv_lt
  have hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62 := hmod ▸ F.shape
  have hinv : (inv * modulus.l0 + 1) % 2^64 = 0 := by rw [hmod, hinv']; exact F.inv_spec
-- END amontredBlock_spec statement
  -- generated skeleton for `amontredBlock`: do not edit between the annotations
  unfold amontredBlock at hres
  lift_lets -merge at hres
  -- t0: input t[0]
  word_step t0 := t.l0 using ht.1
  -- t1: input t[1]
  word_step t1 := t.l1 using ht.2.1
  -- t2: input t[2]
  word_step t2 := t.l2 using ht.2.2.1
  -- t3: input t[3]
  word_step t3 := t.l3 using ht.2.2.2.1
  -- t4: input t[4]
  word_step t4 := t.l4 using ht.2.2.2.2
  -- inv': scalar input
  word_step inv' := inv using hinv_lt
  -- w0: mov {w0}, qword ptr [{p}]
  word_step w0 := modulus.l0 using hm.1
  -- w0_1: shl {w0}, 61
  word_step w0_1 := w0 * 2^61 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w2: mov {w2}, qword ptr [{p} + 8]
  word_step w2 := modulus.l1 using hm.2.1
  -- w1: mov {w1}, qword ptr [{p}]
  word_step w1 := modulus.l0 using hm.1
  -- w1_1: shrd {w1}, {w2}, 3
  word_step w1_1 := extr w2 w1 3 using extr_lt w2 w1 3
  -- w2_1: shr {w2}, 3
  word_step w2_1 := w2 / 2^3
  have b_w2_1 : w2_1 < 2^61 := by
    rw [e_w2_1]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_w2 (by norm_num))
  -- w3: movabs {w3}, 0x800000000000000
  word_step w3 := 576460752303423488 using (by decide)
  -- t0_1: add {t0}, {w0}
  word_step s, t0_1 := (addc t0 w0_1 0).1 using addc_value_lt t0 w0_1 0, cf := (addc t0 w0_1 0).2
  have l_t0_1 : t0_1 + 2^64 * cf = t0 + w0_1 + 0 := by
    rw [e_t0_1, e_cf]; exact addc_lin t0 w0_1 0
  have b_cf : cf ≤ 1 := by rw [e_cf]; exact addc_carry_le_one t0 w0_1 0 b_t0 b_w0_1 (by decide)
  clear e_t0_1 e_cf
  -- t1_1: adc {t1}, {w1}
  word_step s_1, t1_1 := (addc t1 w1_1 cf).1 using addc_value_lt t1 w1_1 cf,
      cf_1 := (addc t1 w1_1 cf).2
  have l_t1_1 : t1_1 + 2^64 * cf_1 = t1 + w1_1 + cf := by
    rw [e_t1_1, e_cf_1]; exact addc_lin t1 w1_1 cf
  have b_cf_1 : cf_1 ≤ 1 := by rw [e_cf_1]; exact addc_carry_le_one t1 w1_1 cf b_t1 b_w1_1 b_cf
  clear e_t1_1 e_cf_1
  -- t2_1: adc {t2}, {w2}
  word_step s_2, t2_1 := (addc t2 w2_1 cf_1).1 using addc_value_lt t2 w2_1 cf_1,
      cf_2 := (addc t2 w2_1 cf_1).2
  have l_t2_1 : t2_1 + 2^64 * cf_2 = t2 + w2_1 + cf_1 := by
    rw [e_t2_1, e_cf_2]; exact addc_lin t2 w2_1 cf_1
  have b_cf_2 : cf_2 ≤ 1 := by rw [e_cf_2]; exact addc_carry_le_one t2 w2_1 cf_1 b_t2 (lt_of_lt_of_le b_w2_1 (by norm_num)) b_cf_1
  clear e_t2_1 e_cf_2
  -- t3_1: adc {t3}, 0
  word_step s_3, t3_1 := (addc t3 0 cf_2).1 using addc_value_lt t3 0 cf_2,
      cf_3 := (addc t3 0 cf_2).2
  have l_t3_1 : t3_1 + 2^64 * cf_3 = t3 + 0 + cf_2 := by
    rw [e_t3_1, e_cf_3]; exact addc_lin t3 0 cf_2
  have b_cf_3 : cf_3 ≤ 1 := by rw [e_cf_3]; exact addc_carry_le_one t3 0 cf_2 b_t3 (by decide) b_cf_2
  clear e_t3_1 e_cf_3
  -- t4_1: adc {t4}, {w3}
  word_step s_4, t4_1 := (addc t4 w3 cf_3).1 using addc_value_lt t4 w3 cf_3,
      cf_4 := (addc t4 w3 cf_3).2
  have l_t4_1 : t4_1 + 2^64 * cf_4 = t4 + w3 + cf_3 := by
    rw [e_t4_1, e_cf_4]; exact addc_lin t4 w3 cf_3
  have b_cf_4 : cf_4 ≤ 1 := by rw [e_cf_4]; exact addc_carry_le_one t4 w3 cf_3 b_t4 b_w3 b_cf_3
  clear e_t4_1 e_cf_4
  -- rdx: mov rdx, {t0}
  word_step rdx := t0_1 using b_t0_1
  -- rdx_1: imul rdx, {inv}
  word_step rdx_1 := rdx * inv' % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- m: mulx {w0}, {w1}, qword ptr [{p}]
  word_step m,
      w0_2 := (mulx rdx_1 modulus.l0).1 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_1 hm.1),
      w1_2 := (mulx rdx_1 modulus.l0).2 using Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_w0_2 : w1_2 + 2^64 * w0_2 = rdx_1 * modulus.l0 := by
    rw [e_w1_2, e_w0_2]; exact Nat.mod_add_div _ _
  -- m_1: mulx {w2}, {w3}, qword ptr [{p} + 8]
  word_step m_1,
      w2_2 := (mulx rdx_1 modulus.l1).1 using Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_1 hm.2.1),
      w3_1 := (mulx rdx_1 modulus.l1).2 using Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_w2_2 : w3_1 + 2^64 * w2_2 = rdx_1 * modulus.l1 := by
    rw [e_w3_1, e_w2_2]; exact Nat.mod_add_div _ _
  -- w1_3: mov {w1}, rdx
  word_step w1_3 := rdx_1 using b_rdx_1
  -- w1_4: shl {w1}, 62
  word_step w1_4 := w1_3 * 2^62 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_w1_4 : w1_4 + 2^64 * (w1_3 / 2^2) = w1_3 * 2^62 := by
    rw [e_w1_4]; exact lsl62_lsr2_split _
  -- w4: mov {w4}, rdx
  word_step w4 := rdx_1 using b_rdx_1
  -- w4_1: shr {w4}, 2
  word_step w4_1 := w4 / 2^2
  have b_w4_1 : w4_1 < 2^62 := by
    rw [e_w4_1]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_w4 (by norm_num))
  -- n: neg {t0}
  word_step n, t0_2 := (neg t0_1).1 using neg_value_lt t0_1, cf_5 := (neg t0_1).2
  have b_cf_5 : cf_5 ≤ 1 := by
    rw [e_cf_5]; simp only [neg]; split <;> omega
  -- t1_2: adc {t1}, {w0}
  word_step s_5, t1_2 := (addc t1_1 w0_2 cf_5).1 using addc_value_lt t1_1 w0_2 cf_5,
      cf_6 := (addc t1_1 w0_2 cf_5).2
  have l_t1_2 : t1_2 + 2^64 * cf_6 = t1_1 + w0_2 + cf_5 := by
    rw [e_t1_2, e_cf_6]; exact addc_lin t1_1 w0_2 cf_5
  have b_cf_6 : cf_6 ≤ 1 := by rw [e_cf_6]; exact addc_carry_le_one t1_1 w0_2 cf_5 b_t1_1 b_w0_2 b_cf_5
  clear e_t1_2 e_cf_6
  -- t2_2: adc {t2}, {w2}
  word_step s_6, t2_2 := (addc t2_1 w2_2 cf_6).1 using addc_value_lt t2_1 w2_2 cf_6,
      cf_7 := (addc t2_1 w2_2 cf_6).2
  have l_t2_2 : t2_2 + 2^64 * cf_7 = t2_1 + w2_2 + cf_6 := by
    rw [e_t2_2, e_cf_7]; exact addc_lin t2_1 w2_2 cf_6
  have b_cf_7 : cf_7 ≤ 1 := by rw [e_cf_7]; exact addc_carry_le_one t2_1 w2_2 cf_6 b_t2_1 b_w2_2 b_cf_6
  clear e_t2_2 e_cf_7
  -- t3_2: adc {t3}, {w1}
  word_step s_7, t3_2 := (addc t3_1 w1_4 cf_7).1 using addc_value_lt t3_1 w1_4 cf_7,
      cf_8 := (addc t3_1 w1_4 cf_7).2
  have l_t3_2 : t3_2 + 2^64 * cf_8 = t3_1 + w1_4 + cf_7 := by
    rw [e_t3_2, e_cf_8]; exact addc_lin t3_1 w1_4 cf_7
  have b_cf_8 : cf_8 ≤ 1 := by rw [e_cf_8]; exact addc_carry_le_one t3_1 w1_4 cf_7 b_t3_1 b_w1_4 b_cf_7
  clear e_t3_2 e_cf_8
  -- t4_2: adc {t4}, {w4}
  word_step s_8, t4_2 := (addc t4_1 w4_1 cf_8).1 using addc_value_lt t4_1 w4_1 cf_8,
      cf_9 := (addc t4_1 w4_1 cf_8).2
  have l_t4_2 : t4_2 + 2^64 * cf_9 = t4_1 + w4_1 + cf_8 := by
    rw [e_t4_2, e_cf_9]; exact addc_lin t4_1 w4_1 cf_8
  have b_cf_9 : cf_9 ≤ 1 := by rw [e_cf_9]; exact addc_carry_le_one t4_1 w4_1 cf_8 b_t4_1 (lt_of_lt_of_le b_w4_1 (by norm_num)) b_cf_8
  clear e_t4_2 e_cf_9
  -- t1_3: add {t1}, {w3}
  word_step s_9, t1_3 := (addc t1_2 w3_1 0).1 using addc_value_lt t1_2 w3_1 0,
      cf_10 := (addc t1_2 w3_1 0).2
  have l_t1_3 : t1_3 + 2^64 * cf_10 = t1_2 + w3_1 + 0 := by
    rw [e_t1_3, e_cf_10]; exact addc_lin t1_2 w3_1 0
  have b_cf_10 : cf_10 ≤ 1 := by rw [e_cf_10]; exact addc_carry_le_one t1_2 w3_1 0 b_t1_2 b_w3_1 (by decide)
  clear e_t1_3 e_cf_10
  -- t2_3: adc {t2}, 0
  word_step s_10, t2_3 := (addc t2_2 0 cf_10).1 using addc_value_lt t2_2 0 cf_10,
      cf_11 := (addc t2_2 0 cf_10).2
  have l_t2_3 : t2_3 + 2^64 * cf_11 = t2_2 + 0 + cf_10 := by
    rw [e_t2_3, e_cf_11]; exact addc_lin t2_2 0 cf_10
  have b_cf_11 : cf_11 ≤ 1 := by rw [e_cf_11]; exact addc_carry_le_one t2_2 0 cf_10 b_t2_2 (by decide) b_cf_10
  clear e_t2_3 e_cf_11
  -- t3_3: adc {t3}, 0
  word_step s_11, t3_3 := (addc t3_2 0 cf_11).1 using addc_value_lt t3_2 0 cf_11,
      cf_12 := (addc t3_2 0 cf_11).2
  have l_t3_3 : t3_3 + 2^64 * cf_12 = t3_2 + 0 + cf_11 := by
    rw [e_t3_3, e_cf_12]; exact addc_lin t3_2 0 cf_11
  have b_cf_12 : cf_12 ≤ 1 := by rw [e_cf_12]; exact addc_carry_le_one t3_2 0 cf_11 b_t3_2 (by decide) b_cf_11
  clear e_t3_3 e_cf_12
  -- t4_3: adc {t4}, 0
  word_step s_12, t4_3 := (addc t4_2 0 cf_12).1 using addc_value_lt t4_2 0 cf_12,
      cf_13 := (addc t4_2 0 cf_12).2
  have l_t4_3 : t4_3 + 2^64 * cf_13 = t4_2 + 0 + cf_12 := by
    rw [e_t4_3, e_cf_13]; exact addc_lin t4_2 0 cf_12
  have b_cf_13 : cf_13 ≤ 1 := by rw [e_cf_13]; exact addc_carry_le_one t4_2 0 cf_12 b_t4_2 (by decide) b_cf_12
  clear e_t4_3 e_cf_13
  subst hres
  -- BEGIN conclusion
  subst hmod hinv'
  have hP : F.modulus.toNat = w0 + 2^64 * w2 + 2^254 := by
    rw [e_w0, e_w2]; simp only [Limbs.toNat, hshape.1, hshape.2]; norm_num
  have hP61 : 2^61 * F.modulus.toNat = 2^61 * w0 + 2^125 * w2 + 2^315 := by rw [hP]; ring
  -- The input's words and its value: the sign word subtracts `2^320` when set.
  obtain ⟨σ, hσ, hT⟩ : ∃ σ : ℤ, (σ = 0 ∨ σ = 1) ∧
      t.toInt = ((t0 + 2^64 * t1 + 2^128 * t2 + 2^192 * t3 + 2^256 * t4 : ℕ) : ℤ) - 2^320 * σ := by
    unfold Signed5.toInt
    rw [← e_t0, ← e_t1, ← e_t2, ← e_t3, ← e_t4]
    split_ifs
    · exact ⟨0, Or.inl rfl, by push_cast; ring⟩
    · exact ⟨1, Or.inr rfl, by push_cast; ring⟩
  -- Adding `2^61 p`: the words added are `2^61 p0 + 2^125 p1` split at the word boundaries, and
  -- `2^59` at word 4 is `2^315`.
  have hw1 : w1_1 = (w0 / 2^3 + w2 * 2^61) % 2^64 := by
    rw [e_w1_1, e_w1, ← e_w0]; rfl
  have hS : t0_1 + 2^64 * t1_1 + 2^128 * t2_1 + 2^192 * t3_1 + 2^256 * t4_1 + 2^320 * cf_4
      = (t0 + 2^64 * t1 + 2^128 * t2 + 2^192 * t3 + 2^256 * t4) + 2^61 * F.modulus.toNat := by
    rw [hP61]
    clear * - l_t0_1 l_t1_1 l_t2_1 l_t3_1 l_t4_1 e_w0_1 hw1 e_w2_1 e_w3 b_w0 b_w2
    omega
  -- The cancellation: the low word of `t0_1 + p0 w` is zero, so `t0_1` plus the low half of
  -- `w p0` is `0` or `2^64`, and `neg` set the carry exactly when it is `2^64`.
  have hq : rdx_1 = mulLo inv' t0_1 := by
    simp only [mulLo, e_rdx_1, e_rdx, Nat.mul_comm]
  have hc : t0_1 + w1_2 = 2^64 * cf_5 := by
    have h := neg_carry_cancel t0_1 inv' w0 rdx_1 b_t0_1 b_inv' b_w0
      (by rw [e_inv', e_w0]; exact hinv) hq
    have hlo : mulLo rdx_1 w0 = w1_2 := by rw [e_w1_2, e_w0]; rfl
    have hcf : cf_5 = if t0_1 = 0 then 0 else 1 := by rw [e_cf_5]; rfl
    rwa [hlo, ← hcf] at h
  -- The two chains add `w p` and drop the cancelled low word.
  have hq4 : w1_4 + 2^64 * w4_1 = rdx_1 * 2^62 := by
    rw [e_w4_1, e_w4, ← e_w1_3]; exact sh_w1_4
  have hqp : rdx_1 * F.modulus.toNat
      = (w1_2 + 2^64 * w0_2) + 2^64 * (w3_1 + 2^64 * w2_2) + 2^192 * (w1_4 + 2^64 * w4_1) := by
    rw [hP, d_w0_2, d_w2_2, hq4, e_w0, e_w2]; ring
  have hR : 2^64 * (t1_3 + 2^64 * t2_3 + 2^128 * t3_3 + 2^192 * t4_3) + 2^320 * (cf_9 + cf_13)
      = (t0_1 + 2^64 * t1_1 + 2^128 * t2_1 + 2^192 * t3_1 + 2^256 * t4_1)
        + rdx_1 * F.modulus.toNat := by
    rw [hqp]
    clear * - hc l_t1_2 l_t2_2 l_t3_2 l_t4_2 l_t1_3 l_t2_3 l_t3_3 l_t4_3
    omega
  -- On the integers: the block's multiplier is the model's, and the block's four words are the
  -- model's result, which `amontredZ_spec` places in `[0, 2^256)`.
  set T : ℕ := t0 + 2^64 * t1 + 2^128 * t2 + 2^192 * t3 + 2^256 * t4 with hTdef
  set V : ℕ := t1_3 + 2^64 * t2_3 + 2^128 * t3_3 + 2^192 * t4_3 with hVdef
  set A : ℕ := t1_1 + 2^64 * t2_1 + 2^128 * t3_1 + 2^192 * t4_1 + 2^256 * cf_4 with hAdef
  set p : ℕ := F.modulus.toNat with hp
  set K : ℕ := cf_4 + cf_9 + cf_13 with hKdef
  have hS' : T + 2^61 * p = t0_1 + 2^64 * A := by clear * - hS hAdef; omega
  have hR1 : t0_1 + 2^64 * A + rdx_1 * p = 2^64 * V + 2^320 * K := by
    clear * - hR hAdef hVdef hKdef; omega
  have hV : V < 2^256 := by clear * - hVdef b_t1_3 b_t2_3 b_t3_3 b_t4_3; omega
  have hS'z : ((T : ℕ) : ℤ) + 2^61 * (p : ℤ) = t0_1 + 2^64 * A := by exact_mod_cast hS'
  have hR1z : (t0_1 : ℤ) + 2^64 * A + rdx_1 * p = 2^64 * V + 2^320 * K := by exact_mod_cast hR1
  have hs : t.toInt + 2^61 * (p : ℤ) = (t0_1 : ℤ) + 2^64 * ((A : ℤ) - 2^256 * σ) := by
    rw [hT]; linear_combination hS'z
  have hw : ((t.toInt + 2^61 * (p : ℤ)) * (F.inv : ℤ)) % 2^64 = (rdx_1 : ℤ) := by
    rw [hs, show ((t0_1 : ℤ) + 2^64 * ((A : ℤ) - 2^256 * σ)) * F.inv
        = t0_1 * F.inv + 2^64 * (((A : ℤ) - 2^256 * σ) * F.inv) by ring,
      Int.add_mul_emod_self_left, e_rdx_1, e_rdx, e_inv']
    norm_cast
  have hsum : t.toInt + 2^61 * (p : ℤ) + (rdx_1 : ℤ) * p
      = 2^64 * ((V : ℤ) + 2^256 * ((K : ℤ) - σ)) := by
    rw [hs]; linear_combination hR1z
  have hval : Inversion.amontredZ t.toInt p F.inv = (V : ℤ) + 2^256 * ((K : ℤ) - σ) := by
    show (t.toInt + 2^61 * (p : ℤ) + (t.toInt + 2^61 * (p : ℤ)) * F.inv % 2^64 * p) / 2^64 = _
    rw [hw, hsum, Int.mul_ediv_cancel_left _ (by norm_num)]
  obtain ⟨h0, -, -, h256, -⟩ := Inversion.amontredZ_spec F t.toInt htv
  rw [← hp, hval] at h0 h256
  have hD : (K : ℤ) - σ = 0 := by clear * - h0 h256 hV hσ; omega
  have hres : Inversion.amontred t F.modulus F.inv = Limbs.ofNat V := by
    unfold Inversion.amontred
    rw [← hp, hval, hD, mul_zero, add_zero, Int.toNat_natCast]
  rw [hres]
  simp only [Limbs.ofNat, hVdef, Limbs.mk.injEq]
  clear * - b_t1_3 b_t2_3 b_t3_3 b_t4_3
  omega
  -- END conclusion

end PastaCurves.X86_64
