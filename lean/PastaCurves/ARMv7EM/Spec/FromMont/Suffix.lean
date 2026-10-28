/-
Copyright (c) 2026 the pasta_curves contributors.
-/
import PastaCurves.ARMv7EM.Spec.FromMont.Round

/-!
# The checked suffix of one ARMv7E-M Montgomery conversion round

The eighteen-instruction suffix multiplies the sparse modulus by the quotient from the round prefix
and propagates the resulting carry through the accumulator.
-/

namespace PastaCurves.ARMv7EM

-- BEGIN fromMontRoundSuffix_spec statement
/-- The generated suffix is the pure sparse multiply-accumulate chain and its projected round. -/
theorem fromMontRoundSuffix_spec (modulus : Limbs) (st : FromMontState) (q carry : Nat)
    (hm : modulus.Bounded) (hst : st.Bounded) (hq : q < wordMod) (hcarry : carry < wordMod)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hqeq : q = fromMontQuotient st) (hcarryeq : carry = (adc st.w0 q 0).2) :
    ∀ res, res = fromMontRoundSuffix modulus st q carry →
      res = fromMontSparseChain (fieldWordsOfLimbs modulus) st q carry ∧
        res = FromMontState.ofMulAcc
          (shiftRound (macSparseModulus st.toMulAcc (fieldWordsOfLimbs modulus) q)) := by
  intro res hres
-- END fromMontRoundSuffix_spec statement
  -- generated skeleton for `fromMontRoundSuffix`: do not edit between the annotations
  unfold fromMontRoundSuffix at hres
  lift_lets -merge at hres
  -- r1_4: input r1[4]
  word_step r1_4 := modulus.l0 / 2^32
  have b_r1_4 : r1_4 < 2^32 := by
    rw [e_r1_4]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq hm.1 (by norm_num))
  -- r1_8: input r1[8]
  word_step r1_8 := modulus.l1 % 2^32
  have b_r1_8 : r1_8 < 2^32 := by
    rw [e_r1_8]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- r1_12: input r1[12]
  word_step r1_12 := modulus.l1 / 2^32
  have b_r1_12 : r1_12 < 2^32 := by
    rw [e_r1_12]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq hm.2.1 (by norm_num))
  -- w1: accumulator argument
  word_step w1 := st.w1
  have b_w1 : w1 < 2^32 := by rw [e_w1]; exact hst.2.1
  -- w2: accumulator argument
  word_step w2 := st.w2
  have b_w2 : w2 < 2^32 := by rw [e_w2]; exact hst.2.2.1
  -- w3: accumulator argument
  word_step w3 := st.w3
  have b_w3 : w3 < 2^32 := by rw [e_w3]; exact hst.2.2.2.1
  -- w4: accumulator argument
  word_step w4 := st.w4
  have b_w4 : w4 < 2^32 := by rw [e_w4]; exact hst.2.2.2.2.1
  -- w5: accumulator argument
  word_step w5 := st.w5
  have b_w5 : w5 < 2^32 := by rw [e_w5]; exact hst.2.2.2.2.2.1
  -- w6: accumulator argument
  word_step w6 := st.w6
  have b_w6 : w6 < 2^32 := by rw [e_w6]; exact hst.2.2.2.2.2.2.1
  -- w7: accumulator argument
  word_step w7 := st.w7
  have b_w7 : w7 < 2^32 := by rw [e_w7]; exact hst.2.2.2.2.2.2.2.1
  -- w8: accumulator argument
  word_step w8 := st.w8
  have b_w8 : w8 < 2^32 := by rw [e_w8]; exact hst.2.2.2.2.2.2.2.2
  -- w0: Montgomery quotient
  word_step w0 := q
  have b_w0 : w0 < 2^32 := by rw [e_w0]; exact hq
  -- r12: low-word cancellation carry
  word_step r12 := carry
  have b_r12 : r12 < 2^32 := by rw [e_r12]; exact hcarry
  -- tmp: ldr.w r0, [r1, #4]
  word_step tmp := r1_4
  have b_tmp : tmp < 2^32 := by rw [e_tmp]; exact b_r1_4
  -- w1_1: umaal r3, r12, r0, r2
  word_step m, w1_1 := (tmp * w0 + w1 + r12) % 2^32, r12_1 := (tmp * w0 + w1 + r12) / 2^32
  have b_w1_1 : w1_1 < 2^32 := by rw [e_w1_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have m_r12_1 : tmp * w0 ≤ (2^32 - 1) * (2^32 - 1) := Nat.mul_le_mul (Nat.le_sub_one_of_lt b_tmp) (Nat.le_sub_one_of_lt b_w0)
  have t_r12_1 : (tmp * w0 + w1 + r12) < 2^64 := by omega
  have b_r12_1 : r12_1 < 2^32 := by
    rw [e_r12_1]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq t_r12_1 (by norm_num))
  have d_r12_1 : w1_1 + 2^32 * r12_1 = (tmp * w0 + w1 + r12) := by
    rw [e_w1_1, e_r12_1]; exact Nat.mod_add_div _ _
  clear e_w1_1 e_r12_1
  -- tmp_1: ldr.w r0, [r1, #8]
  word_step tmp_1 := r1_8
  have b_tmp_1 : tmp_1 < 2^32 := by rw [e_tmp_1]; exact b_r1_8
  -- w2_1: umaal r4, r12, r0, r2
  word_step m_1, w2_1 := (tmp_1 * w0 + w2 + r12_1) % 2^32, r12_2 := (tmp_1 * w0 + w2 + r12_1) / 2^32
  have b_w2_1 : w2_1 < 2^32 := by rw [e_w2_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have m_r12_2 : tmp_1 * w0 ≤ (2^32 - 1) * (2^32 - 1) := Nat.mul_le_mul (Nat.le_sub_one_of_lt b_tmp_1) (Nat.le_sub_one_of_lt b_w0)
  have t_r12_2 : (tmp_1 * w0 + w2 + r12_1) < 2^64 := by omega
  have b_r12_2 : r12_2 < 2^32 := by
    rw [e_r12_2]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq t_r12_2 (by norm_num))
  have d_r12_2 : w2_1 + 2^32 * r12_2 = (tmp_1 * w0 + w2 + r12_1) := by
    rw [e_w2_1, e_r12_2]; exact Nat.mod_add_div _ _
  clear e_w2_1 e_r12_2
  -- tmp_2: ldr.w r0, [r1, #12]
  word_step tmp_2 := r1_12
  have b_tmp_2 : tmp_2 < 2^32 := by rw [e_tmp_2]; exact b_r1_12
  -- w3_1: umaal r5, r12, r0, r2
  word_step m_2, w3_1 := (tmp_2 * w0 + w3 + r12_2) % 2^32, r12_3 := (tmp_2 * w0 + w3 + r12_2) / 2^32
  have b_w3_1 : w3_1 < 2^32 := by rw [e_w3_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have m_r12_3 : tmp_2 * w0 ≤ (2^32 - 1) * (2^32 - 1) := Nat.mul_le_mul (Nat.le_sub_one_of_lt b_tmp_2) (Nat.le_sub_one_of_lt b_w0)
  have t_r12_3 : (tmp_2 * w0 + w3 + r12_2) < 2^64 := by omega
  have b_r12_3 : r12_3 < 2^32 := by
    rw [e_r12_3]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq t_r12_3 (by norm_num))
  have d_r12_3 : w3_1 + 2^32 * r12_3 = (tmp_2 * w0 + w3 + r12_2) := by
    rw [e_w3_1, e_r12_3]; exact Nat.mod_add_div _ _
  clear e_w3_1 e_r12_3
  -- tmp_3: mov.w r0, #0
  word_step tmp_3 := 0
  have b_tmp_3 : tmp_3 < 2^32 := by rw [e_tmp_3]; exact (by decide)
  -- w4_1: adds.w r8, r8, r12
  word_step s, w4_1 := (w4 + r12_3 + 0) % 2^32, c := (w4 + r12_3 + 0) / 2^32
  have b_w4_1 : w4_1 < 2^32 := by rw [e_w4_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w4_1 : w4_1 + 2^32 * c = w4 + r12_3 + 0 := by
    rw [e_w4_1, e_c]; exact Nat.mod_add_div _ _
  have b_c : c ≤ 1 := by
    rw [e_c]; exact addc32_carry_le_one w4 r12_3 0 b_w4 b_r12_3 (by decide)
  clear e_w4_1 e_c
  -- r12_4: adc.w r12, r0, r0
  word_step r12_4 := (tmp_3 + tmp_3 + c) % 2^32
  have b_r12_4 : r12_4 < 2^32 := by rw [e_r12_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_r12_4, b_k_r12_4, l_r12_4⟩ :
      ∃ k, k ≤ 1 ∧ r12_4 + 2^32 * k = tmp_3 + tmp_3 + c :=
    ⟨(tmp_3 + tmp_3 + c) / 2^32, addc32_carry_le_one tmp_3 tmp_3 c b_tmp_3 b_tmp_3 b_c,
      by rw [e_r12_4]; exact Nat.mod_add_div _ _⟩
  clear e_r12_4
  -- w5_1: adds.w r9, r9, r12
  word_step s_1, w5_1 := (w5 + r12_4 + 0) % 2^32, c_1 := (w5 + r12_4 + 0) / 2^32
  have b_w5_1 : w5_1 < 2^32 := by rw [e_w5_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w5_1 : w5_1 + 2^32 * c_1 = w5 + r12_4 + 0 := by
    rw [e_w5_1, e_c_1]; exact Nat.mod_add_div _ _
  have b_c_1 : c_1 ≤ 1 := by
    rw [e_c_1]; exact addc32_carry_le_one w5 r12_4 0 b_w5 b_r12_4 (by decide)
  clear e_w5_1 e_c_1
  -- r12_5: adc.w r12, r0, r0
  word_step r12_5 := (tmp_3 + tmp_3 + c_1) % 2^32
  have b_r12_5 : r12_5 < 2^32 := by rw [e_r12_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_r12_5, b_k_r12_5, l_r12_5⟩ :
      ∃ k, k ≤ 1 ∧ r12_5 + 2^32 * k = tmp_3 + tmp_3 + c_1 :=
    ⟨(tmp_3 + tmp_3 + c_1) / 2^32, addc32_carry_le_one tmp_3 tmp_3 c_1 b_tmp_3 b_tmp_3 b_c_1,
      by rw [e_r12_5]; exact Nat.mod_add_div _ _⟩
  clear e_r12_5
  -- w6_1: adds.w r10, r10, r12
  word_step s_2, w6_1 := (w6 + r12_5 + 0) % 2^32, c_2 := (w6 + r12_5 + 0) / 2^32
  have b_w6_1 : w6_1 < 2^32 := by rw [e_w6_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w6_1 : w6_1 + 2^32 * c_2 = w6 + r12_5 + 0 := by
    rw [e_w6_1, e_c_2]; exact Nat.mod_add_div _ _
  have b_c_2 : c_2 ≤ 1 := by
    rw [e_c_2]; exact addc32_carry_le_one w6 r12_5 0 b_w6 b_r12_5 (by decide)
  clear e_w6_1 e_c_2
  -- r12_6: adc.w r12, r0, r0
  word_step r12_6 := (tmp_3 + tmp_3 + c_2) % 2^32
  have b_r12_6 : r12_6 < 2^32 := by rw [e_r12_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_r12_6, b_k_r12_6, l_r12_6⟩ :
      ∃ k, k ≤ 1 ∧ r12_6 + 2^32 * k = tmp_3 + tmp_3 + c_2 :=
    ⟨(tmp_3 + tmp_3 + c_2) / 2^32, addc32_carry_le_one tmp_3 tmp_3 c_2 b_tmp_3 b_tmp_3 b_c_2,
      by rw [e_r12_6]; exact Nat.mod_add_div _ _⟩
  clear e_r12_6
  -- tmp_4: mov.w r0, #0x40000000
  word_step tmp_4 := 0x40000000
  have b_tmp_4 : tmp_4 < 2^32 := by rw [e_tmp_4]; exact (by decide)
  -- w7_1: umaal r11, r12, r0, r2
  word_step m_3, w7_1 := (tmp_4 * w0 + w7 + r12_6) % 2^32, r12_7 := (tmp_4 * w0 + w7 + r12_6) / 2^32
  have b_w7_1 : w7_1 < 2^32 := by rw [e_w7_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have m_r12_7 : tmp_4 * w0 ≤ (2^32 - 1) * (2^32 - 1) := Nat.mul_le_mul (Nat.le_sub_one_of_lt b_tmp_4) (Nat.le_sub_one_of_lt b_w0)
  have t_r12_7 : (tmp_4 * w0 + w7 + r12_6) < 2^64 := by omega
  have b_r12_7 : r12_7 < 2^32 := by
    rw [e_r12_7]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq t_r12_7 (by norm_num))
  have d_r12_7 : w7_1 + 2^32 * r12_7 = (tmp_4 * w0 + w7 + r12_6) := by
    rw [e_w7_1, e_r12_7]; exact Nat.mod_add_div _ _
  clear e_w7_1 e_r12_7
  -- tmp_5: mov.w r0, #0
  word_step tmp_5 := 0
  have b_tmp_5 : tmp_5 < 2^32 := by rw [e_tmp_5]; exact (by decide)
  -- w8_1: adds.w lr, lr, r12
  word_step s_3, w8_1 := (w8 + r12_7 + 0) % 2^32, c_3 := (w8 + r12_7 + 0) / 2^32
  have b_w8_1 : w8_1 < 2^32 := by rw [e_w8_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w8_1 : w8_1 + 2^32 * c_3 = w8 + r12_7 + 0 := by
    rw [e_w8_1, e_c_3]; exact Nat.mod_add_div _ _
  have b_c_3 : c_3 ≤ 1 := by
    rw [e_c_3]; exact addc32_carry_le_one w8 r12_7 0 b_w8 b_r12_7 (by decide)
  clear e_w8_1 e_c_3
  -- tmp_6: adc.w r0, r0, r0
  word_step tmp_6 := (tmp_5 + tmp_5 + c_3) % 2^32
  have b_tmp_6 : tmp_6 < 2^32 := by rw [e_tmp_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_tmp_6, b_k_tmp_6, l_tmp_6⟩ :
      ∃ k, k ≤ 1 ∧ tmp_6 + 2^32 * k = tmp_5 + tmp_5 + c_3 :=
    ⟨(tmp_5 + tmp_5 + c_3) / 2^32, addc32_carry_le_one tmp_5 tmp_5 c_3 b_tmp_5 b_tmp_5 b_c_3,
      by rw [e_tmp_6]; exact Nat.mod_add_div _ _⟩
  clear e_tmp_6
  subst hres
  -- BEGIN fromMontRoundSuffix_spec conclusion
  let p := fieldWordsOfLimbs modulus
  have hp1 : p.w1 = r1_4 := by
    change modulus.l0 / wordMod % wordMod = r1_4
    rw [show wordMod = 2^32 by rfl, ← e_r1_4, Nat.mod_eq_of_lt b_r1_4]
  have hp2 : p.w2 = r1_8 := by simp only [p, fieldWordsOfLimbs, e_r1_8]
  have hp3 : p.w3 = r1_12 := by
    change modulus.l1 / wordMod % wordMod = r1_12
    rw [show wordMod = 2^32 by rfl, ← e_r1_12, Nat.mod_eq_of_lt b_r1_12]
  have hp7 : p.w7 = tmp_4 := by
    norm_num [p, fieldWordsOfLimbs, hshape.2, e_tmp_4, wordMod]
  have h1 : macWord st.w1 p.w1 q carry = (w1_1, r12_1) :=
    macWord_eq_pair st.w1 p.w1 q carry w1_1 r12_1 hst.2.1
      (by simpa only [hp1] using b_r1_4) hq hcarry b_w1_1 (by
        simpa only [wordMod, hp1, e_tmp, e_w0, e_w1, e_r12] using d_r12_1)
  have h2 : macWord st.w2 p.w2 q r12_1 = (w2_1, r12_2) :=
    macWord_eq_pair st.w2 p.w2 q r12_1 w2_1 r12_2 hst.2.2.1
      (by simpa only [hp2] using b_r1_8) hq b_r12_1 b_w2_1 (by
        simpa only [wordMod, hp2, e_tmp_1, e_w0, e_w2] using d_r12_2)
  have h3 : macWord st.w3 p.w3 q r12_2 = (w3_1, r12_3) :=
    macWord_eq_pair st.w3 p.w3 q r12_2 w3_1 r12_3 hst.2.2.2.1
      (by simpa only [hp3] using b_r1_12) hq b_r12_2 b_w3_1 (by
        simpa only [wordMod, hp3, e_tmp_2, e_w0, e_w3] using d_r12_3)
  have hc4 : r12_4 = c := by
    clear * - e_tmp_3 l_r12_4 b_c b_r12_4 b_k_r12_4
    omega
  have h4 : addWord st.w4 r12_3 = (w4_1, r12_4) :=
    addWord_eq_pair st.w4 r12_3 w4_1 r12_4 hst.2.2.2.2.1 b_r12_3 b_w4_1 (by
      simpa only [wordMod, hc4, e_w4, Nat.add_zero] using l_w4_1)
  have hc5 : r12_5 = c_1 := by
    clear * - e_tmp_3 l_r12_5 b_c_1 b_r12_5 b_k_r12_5
    omega
  have h5 : addWord st.w5 r12_4 = (w5_1, r12_5) :=
    addWord_eq_pair st.w5 r12_4 w5_1 r12_5 hst.2.2.2.2.2.1 b_r12_4 b_w5_1 (by
      simpa only [wordMod, hc5, e_w5, Nat.add_zero] using l_w5_1)
  have hc6 : r12_6 = c_2 := by
    clear * - e_tmp_3 l_r12_6 b_c_2 b_r12_6 b_k_r12_6
    omega
  have h6 : addWord st.w6 r12_5 = (w6_1, r12_6) :=
    addWord_eq_pair st.w6 r12_5 w6_1 r12_6 hst.2.2.2.2.2.2.1 b_r12_5 b_w6_1 (by
      simpa only [wordMod, hc6, e_w6, Nat.add_zero] using l_w6_1)
  have h7 : macWord st.w7 p.w7 q r12_6 = (w7_1, r12_7) :=
    macWord_eq_pair st.w7 p.w7 q r12_6 w7_1 r12_7 hst.2.2.2.2.2.2.2.1
      (by simpa only [hp7] using b_tmp_4) hq b_r12_6 b_w7_1 (by
        simpa only [wordMod, hp7, e_w0, e_w7] using d_r12_7)
  have hc8 : tmp_6 = c_3 := by
    clear * - e_tmp_5 l_tmp_6 b_c_3 b_tmp_6 b_k_tmp_6
    omega
  have h8 : addWord st.w8 r12_7 = (w8_1, tmp_6) :=
    addWord_eq_pair st.w8 r12_7 w8_1 tmp_6 hst.2.2.2.2.2.2.2.2 b_r12_7 b_w8_1 (by
      simpa only [wordMod, hc8, e_w8, Nat.add_zero] using l_w8_1)
  have h9 : addWord 0 tmp_6 = (tmp_6, 0) :=
    addWord_eq_pair 0 tmp_6 tmp_6 0 (by norm_num [wordMod]) b_tmp_6 b_tmp_6 (by omega)
  have hchain :
      (⟨w1_1, w2_1, w3_1, w4_1, w5_1, w6_1, w7_1, w8_1, tmp_6⟩ : FromMontState) =
        fromMontSparseChain p st q carry :=
    fromMontSparseChain_of_eqs p st q carry
      w1_1 w2_1 w3_1 w4_1 w5_1 w6_1 w7_1 w8_1 tmp_6
      r12_1 r12_2 r12_3 r12_4 r12_5 r12_6 r12_7 h1 h2 h3 h4 h5 h6 h7 h8 h9
  have hlow : adc st.w0 q 0 = (0, carry) := by
    apply Prod.ext
    · have hcancel := (fromMontQuotient_spec st hst).2
      simp only [adc, addc32, hqeq, Nat.add_zero, hcancel, Nat.zero_mod]
    · exact hcarryeq.symm
  clear * - hchain hlow
  constructor
  · simpa only [p] using hchain
  · rw [hchain, fromMontSparseChain_eq p st q carry hlow]
  -- END fromMontRoundSuffix_spec conclusion

end PastaCurves.ARMv7EM
