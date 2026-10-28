/-
Copyright (c) 2026 the pasta_curves contributors.
-/
import PastaCurves.ARMv7EM.Spec.FromMont.FinalReduction
import PastaCurves.ARMv7EM.Spec.FromMont.Round

/-!
# Masked add-back for ARMv7E-M Montgomery conversion

The final seventeen instructions of the final reduction add the sparse modulus under the borrow
mask produced by the subtraction helper. This proof identifies the instruction chain with the
shared eight-word addition model and preserves the generated output bounds.
-/

namespace PastaCurves.ARMv7EM

open ListArithmetic

-- BEGIN fromMontFinalAddBackBlock_spec statement
/-- The generated add-back helper is exactly an eight-word addition of the masked sparse modulus. -/
theorem fromMontFinalAddBackBlock_spec (modulus : Limbs) (sub : FromMontFinalSubtraction)
    (hm : modulus.Bounded) (hsub : sub.Bounded) :
    ∀ res, res = fromMontFinalAddBackBlock modulus sub →
      res.Bounded ∧ res.w8 = sub.w8 ∧
        ∃ carry, carry ≤ 1 ∧
          adcChain 0 [sub.w0, sub.w1, sub.w2, sub.w3, sub.w4, sub.w5, sub.w6, sub.w7]
            (maskedModulus
              [1, modulus.l0 / wordMod, modulus.l1 % wordMod, modulus.l1 / wordMod,
                0, 0, 0, 0x40000000]
              sub.mask) =
            ([res.w0, res.w1, res.w2, res.w3, res.w4, res.w5, res.w6, res.w7], carry) := by
  intro res hres
-- END fromMontFinalAddBackBlock_spec statement
  -- generated skeleton for `fromMontFinalAddBackBlock`: do not edit between the annotations
  unfold fromMontFinalAddBackBlock at hres
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
  -- w0: subtraction helper output
  word_step w0 := sub.w0
  have b_w0 : w0 < 2^32 := by rw [e_w0]; exact hsub.1
  -- w1: subtraction helper output
  word_step w1 := sub.w1
  have b_w1 : w1 < 2^32 := by rw [e_w1]; exact hsub.2.1
  -- w2: subtraction helper output
  word_step w2 := sub.w2
  have b_w2 : w2 < 2^32 := by rw [e_w2]; exact hsub.2.2.1
  -- w3: subtraction helper output
  word_step w3 := sub.w3
  have b_w3 : w3 < 2^32 := by rw [e_w3]; exact hsub.2.2.2.1
  -- w4: subtraction helper output
  word_step w4 := sub.w4
  have b_w4 : w4 < 2^32 := by rw [e_w4]; exact hsub.2.2.2.2.1
  -- w5: subtraction helper output
  word_step w5 := sub.w5
  have b_w5 : w5 < 2^32 := by rw [e_w5]; exact hsub.2.2.2.2.2.1
  -- w6: subtraction helper output
  word_step w6 := sub.w6
  have b_w6 : w6 < 2^32 := by rw [e_w6]; exact hsub.2.2.2.2.2.2.1
  -- w7: subtraction helper output
  word_step w7 := sub.w7
  have b_w7 : w7 < 2^32 := by rw [e_w7]; exact hsub.2.2.2.2.2.2.2.1
  -- w8: subtraction helper output
  word_step w8 := sub.w8
  have b_w8 : w8 < 2^32 := by rw [e_w8]; exact hsub.2.2.2.2.2.2.2.2.1
  -- r12: subtraction borrow mask
  word_step r12 := sub.mask
  have b_r12 : r12 < 2^32 := by rw [e_r12]; exact hsub.2.2.2.2.2.2.2.2.2.2
  -- r11: and.w r11, r12, #1
  word_step r11 := andw r12 1
  have b_r11 : r11 < 2^32 := by
    rw [e_r11]; exact Nat.and_lt_two_pow r12 (by decide)
  -- w0_1: adds.w lr, lr, r11
  word_step s, w0_1 := (w0 + r11 + 0) % 2^32, c := (w0 + r11 + 0) / 2^32
  have b_w0_1 : w0_1 < 2^32 := by rw [e_w0_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w0_1 : w0_1 + 2^32 * c = w0 + r11 + 0 := by
    rw [e_w0_1, e_c]; exact Nat.mod_add_div _ _
  have b_c : c ≤ 1 := by
    rw [e_c]; exact addc32_carry_le_one w0 r11 0 b_w0 b_r11 (by decide)
  clear e_w0_1 e_c
  -- r11_1: ldr.w r11, [r1, #4]
  word_step r11_1 := r1_4
  have b_r11_1 : r11_1 < 2^32 := by rw [e_r11_1]; exact b_r1_4
  -- r11_2: and.w r11, r11, r12
  word_step r11_2 := andw r11_1 r12
  have b_r11_2 : r11_2 < 2^32 := by
    rw [e_r11_2]; exact Nat.and_lt_two_pow r11_1 b_r12
  -- w1_1: adcs.w r0, r0, r11
  word_step adcs_pair, w1_1 := (w1 + r11_2 + c) % 2^32, c_1 := (w1 + r11_2 + c) / 2^32
  have b_w1_1 : w1_1 < 2^32 := by rw [e_w1_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w1_1 : w1_1 + 2^32 * c_1 = w1 + r11_2 + c := by
    rw [e_w1_1, e_c_1]; exact Nat.mod_add_div _ _
  have b_c_1 : c_1 ≤ 1 := by
    rw [e_c_1]; exact addc32_carry_le_one w1 r11_2 c b_w1 b_r11_2 b_c
  clear e_w1_1 e_c_1
  -- r11_3: ldr.w r11, [r1, #8]
  word_step r11_3 := r1_8
  have b_r11_3 : r11_3 < 2^32 := by rw [e_r11_3]; exact b_r1_8
  -- r11_4: and.w r11, r11, r12
  word_step r11_4 := andw r11_3 r12
  have b_r11_4 : r11_4 < 2^32 := by
    rw [e_r11_4]; exact Nat.and_lt_two_pow r11_3 b_r12
  -- w2_1: adcs.w r2, r2, r11
  word_step adcs_pair_1, w2_1 := (w2 + r11_4 + c_1) % 2^32, c_2 := (w2 + r11_4 + c_1) / 2^32
  have b_w2_1 : w2_1 < 2^32 := by rw [e_w2_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w2_1 : w2_1 + 2^32 * c_2 = w2 + r11_4 + c_1 := by
    rw [e_w2_1, e_c_2]; exact Nat.mod_add_div _ _
  have b_c_2 : c_2 ≤ 1 := by
    rw [e_c_2]; exact addc32_carry_le_one w2 r11_4 c_1 b_w2 b_r11_4 b_c_1
  clear e_w2_1 e_c_2
  -- r11_5: ldr.w r11, [r1, #12]
  word_step r11_5 := r1_12
  have b_r11_5 : r11_5 < 2^32 := by rw [e_r11_5]; exact b_r1_12
  -- r11_6: and.w r11, r11, r12
  word_step r11_6 := andw r11_5 r12
  have b_r11_6 : r11_6 < 2^32 := by
    rw [e_r11_6]; exact Nat.and_lt_two_pow r11_5 b_r12
  -- w3_1: adcs.w r3, r3, r11
  word_step adcs_pair_2, w3_1 := (w3 + r11_6 + c_2) % 2^32, c_3 := (w3 + r11_6 + c_2) / 2^32
  have b_w3_1 : w3_1 < 2^32 := by rw [e_w3_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w3_1 : w3_1 + 2^32 * c_3 = w3 + r11_6 + c_2 := by
    rw [e_w3_1, e_c_3]; exact Nat.mod_add_div _ _
  have b_c_3 : c_3 ≤ 1 := by
    rw [e_c_3]; exact addc32_carry_le_one w3 r11_6 c_2 b_w3 b_r11_6 b_c_2
  clear e_w3_1 e_c_3
  -- r11_7: mov.w r11, #0
  word_step r11_7 := 0
  have b_r11_7 : r11_7 < 2^32 := by rw [e_r11_7]; exact (by decide)
  -- w4_1: adcs.w r4, r4, r11
  word_step adcs_pair_3, w4_1 := (w4 + r11_7 + c_3) % 2^32, c_4 := (w4 + r11_7 + c_3) / 2^32
  have b_w4_1 : w4_1 < 2^32 := by rw [e_w4_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w4_1 : w4_1 + 2^32 * c_4 = w4 + r11_7 + c_3 := by
    rw [e_w4_1, e_c_4]; exact Nat.mod_add_div _ _
  have b_c_4 : c_4 ≤ 1 := by
    rw [e_c_4]; exact addc32_carry_le_one w4 r11_7 c_3 b_w4 b_r11_7 b_c_3
  clear e_w4_1 e_c_4
  -- w5_1: adcs.w r5, r5, r11
  word_step adcs_pair_4, w5_1 := (w5 + r11_7 + c_4) % 2^32, c_5 := (w5 + r11_7 + c_4) / 2^32
  have b_w5_1 : w5_1 < 2^32 := by rw [e_w5_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w5_1 : w5_1 + 2^32 * c_5 = w5 + r11_7 + c_4 := by
    rw [e_w5_1, e_c_5]; exact Nat.mod_add_div _ _
  have b_c_5 : c_5 ≤ 1 := by
    rw [e_c_5]; exact addc32_carry_le_one w5 r11_7 c_4 b_w5 b_r11_7 b_c_4
  clear e_w5_1 e_c_5
  -- w6_1: adcs.w r8, r8, r11
  word_step adcs_pair_5, w6_1 := (w6 + r11_7 + c_5) % 2^32, c_6 := (w6 + r11_7 + c_5) / 2^32
  have b_w6_1 : w6_1 < 2^32 := by rw [e_w6_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w6_1 : w6_1 + 2^32 * c_6 = w6 + r11_7 + c_5 := by
    rw [e_w6_1, e_c_6]; exact Nat.mod_add_div _ _
  have b_c_6 : c_6 ≤ 1 := by
    rw [e_c_6]; exact addc32_carry_le_one w6 r11_7 c_5 b_w6 b_r11_7 b_c_5
  clear e_w6_1 e_c_6
  -- r11_8: and.w r11, r12, #0x40000000
  word_step r11_8 := andw r12 0x40000000
  have b_r11_8 : r11_8 < 2^32 := by
    rw [e_r11_8]; exact Nat.and_lt_two_pow r12 (by decide)
  -- w7_1: adc.w r9, r9, r11
  word_step w7_1 := (w7 + r11_8 + c_6) % 2^32
  have b_w7_1 : w7_1 < 2^32 := by rw [e_w7_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_w7_1, b_k_w7_1, l_w7_1⟩ :
      ∃ k, k ≤ 1 ∧ w7_1 + 2^32 * k = w7 + r11_8 + c_6 :=
    ⟨(w7 + r11_8 + c_6) / 2^32, addc32_carry_le_one w7 r11_8 c_6 b_w7 b_r11_8 b_c_6,
      by rw [e_w7_1]; exact Nat.mod_add_div _ _⟩
  clear e_w7_1
  subst hres
  -- BEGIN fromMontFinalAddBackBlock_spec conclusion
  have h0 : adc sub.w0 (andw 1 sub.mask) 0 = (w0_1, c) :=
    adc_eq_pair sub.w0 (andw 1 sub.mask) 0 w0_1 c hsub.1
      (by simpa only [wordMod, e_r11, e_r12, andw, Nat.and_comm] using b_r11)
      (by norm_num [wordMod]) (by decide) b_w0_1 (by
        simpa only [wordMod, e_w0, e_r11, e_r12, andw, Nat.and_comm, Nat.add_zero]
          using l_w0_1)
  have h1 : adc sub.w1 (andw (modulus.l0 / wordMod) sub.mask) c = (w1_1, c_1) :=
    adc_eq_pair sub.w1 (andw (modulus.l0 / wordMod) sub.mask) c w1_1 c_1 hsub.2.1
      (by simpa only [wordMod, e_r11_2, e_r11_1, e_r1_4, e_r12] using b_r11_2)
      (by norm_num [wordMod]; omega) b_c b_w1_1 (by
        simpa only [wordMod, e_w1, e_r11_2, e_r11_1, e_r1_4, e_r12] using l_w1_1)
  have h2 : adc sub.w2 (andw (modulus.l1 % wordMod) sub.mask) c_1 = (w2_1, c_2) :=
    adc_eq_pair sub.w2 (andw (modulus.l1 % wordMod) sub.mask) c_1 w2_1 c_2 hsub.2.2.1
      (by simpa only [wordMod, e_r11_4, e_r11_3, e_r1_8, e_r12] using b_r11_4)
      (by norm_num [wordMod]; omega) b_c_1 b_w2_1 (by
        simpa only [wordMod, e_w2, e_r11_4, e_r11_3, e_r1_8, e_r12] using l_w2_1)
  have h3 : adc sub.w3 (andw (modulus.l1 / wordMod) sub.mask) c_2 = (w3_1, c_3) :=
    adc_eq_pair sub.w3 (andw (modulus.l1 / wordMod) sub.mask) c_2 w3_1 c_3 hsub.2.2.2.1
      (by simpa only [wordMod, e_r11_6, e_r11_5, e_r1_12, e_r12] using b_r11_6)
      (by norm_num [wordMod]; omega) b_c_2 b_w3_1 (by
        simpa only [wordMod, e_w3, e_r11_6, e_r11_5, e_r1_12, e_r12] using l_w3_1)
  have h4 : adc sub.w4 (andw 0 sub.mask) c_3 = (w4_1, c_4) :=
    adc_eq_pair sub.w4 (andw 0 sub.mask) c_3 w4_1 c_4 hsub.2.2.2.2.1
      (by norm_num [andw, wordMod])
      (by norm_num [wordMod]; omega) b_c_3 b_w4_1 (by
        simpa only [wordMod, e_w4, e_r11_7, andw, Nat.zero_and] using l_w4_1)
  have h5 : adc sub.w5 (andw 0 sub.mask) c_4 = (w5_1, c_5) :=
    adc_eq_pair sub.w5 (andw 0 sub.mask) c_4 w5_1 c_5 hsub.2.2.2.2.2.1
      (by norm_num [andw, wordMod])
      (by norm_num [wordMod]; omega) b_c_4 b_w5_1 (by
        simpa only [wordMod, e_w5, e_r11_7, andw, Nat.zero_and] using l_w5_1)
  have h6 : adc sub.w6 (andw 0 sub.mask) c_5 = (w6_1, c_6) :=
    adc_eq_pair sub.w6 (andw 0 sub.mask) c_5 w6_1 c_6 hsub.2.2.2.2.2.2.1
      (by norm_num [andw, wordMod])
      (by norm_num [wordMod]; omega) b_c_5 b_w6_1 (by
        simpa only [wordMod, e_w6, e_r11_7, andw, Nat.zero_and] using l_w6_1)
  have h7 : adc sub.w7 (andw 0x40000000 sub.mask) c_6 = (w7_1, k_w7_1) :=
    adc_eq_pair sub.w7 (andw 0x40000000 sub.mask) c_6 w7_1 k_w7_1
      hsub.2.2.2.2.2.2.2.1
      (by simpa only [wordMod, e_r11_8, e_r12, andw, Nat.and_comm] using b_r11_8)
      (by norm_num [wordMod]; omega) b_c_6 b_w7_1 (by
        simpa only [wordMod, e_w7, e_r11_8, e_r12, andw, Nat.and_comm] using l_w7_1)
  have hchain :
      adcChain 0 [sub.w0, sub.w1, sub.w2, sub.w3, sub.w4, sub.w5, sub.w6, sub.w7]
          (maskedModulus
            [1, modulus.l0 / wordMod, modulus.l1 % wordMod, modulus.l1 / wordMod,
              0, 0, 0, 0x40000000]
            sub.mask) =
        ([w0_1, w1_1, w2_1, w3_1, w4_1, w5_1, w6_1, w7_1], k_w7_1) := by
    simp only [maskedModulus, List.map_cons, List.map_nil, adcChain, h0, h1, h2, h3, h4,
      h5, h6, h7]
  refine ⟨⟨b_w0_1, b_w1_1, b_w2_1, b_w3_1, b_w4_1, b_w5_1, b_w6_1, b_w7_1,
    b_w8⟩, e_w8, k_w7_1, b_k_w7_1, ?_⟩
  simpa only using hchain
  -- END fromMontFinalAddBackBlock_spec conclusion

end PastaCurves.ARMv7EM
