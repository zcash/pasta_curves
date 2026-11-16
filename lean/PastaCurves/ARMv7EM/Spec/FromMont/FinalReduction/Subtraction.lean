/-
Copyright (c) 2026 the pasta_curves contributors.
-/
import PastaCurves.ARMv7EM.Spec.Invert.CondSub.Arithmetic
import PastaCurves.ARMv7EM.Spec.Mul.Reduction

/-!
# Checked subtraction and mask for ARMv7E-M Montgomery conversion

The first fourteen instructions of the final reduction subtract the sparse modulus words and turn
the final no-borrow carry into the mask consumed by the add-back helper.
-/

namespace PastaCurves.ARMv7EM

open ListArithmetic

-- BEGIN fromMontFinalSubtractionBlock_spec statement
/-- The generated subtraction helper is exactly an eight-word no-borrow chain and its mask. -/
theorem fromMontFinalSubtractionBlock_spec (modulus : Limbs) (st : FromMontState)
    (hm : modulus.Bounded) (hst : st.Bounded) :
    ∀ res, res = fromMontFinalSubtractionBlock modulus st →
      res.Bounded ∧
        subcChain 1 [st.w0, st.w1, st.w2, st.w3, st.w4, st.w5, st.w6, st.w7]
          [1, modulus.l0 / wordMod, modulus.l1 % wordMod, modulus.l1 / wordMod,
            0, 0, 0, 0x40000000] =
          ([res.w0, res.w1, res.w2, res.w3, res.w4, res.w5, res.w6, res.w7], res.carry) ∧
        res.w8 = st.w8 ∧ res.mask = (subc32 st.w8 st.w8 res.carry).1 := by
  intro res hres
-- END fromMontFinalSubtractionBlock_spec statement
  -- generated skeleton for `fromMontFinalSubtractionBlock`: do not edit between the annotations
  unfold fromMontFinalSubtractionBlock at hres
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
  -- w0: post-round accumulator argument
  word_step w0 := st.w0
  have b_w0 : w0 < 2^32 := by rw [e_w0]; exact hst.1
  -- w1: post-round accumulator argument
  word_step w1 := st.w1
  have b_w1 : w1 < 2^32 := by rw [e_w1]; exact hst.2.1
  -- w2: post-round accumulator argument
  word_step w2 := st.w2
  have b_w2 : w2 < 2^32 := by rw [e_w2]; exact hst.2.2.1
  -- w3: post-round accumulator argument
  word_step w3 := st.w3
  have b_w3 : w3 < 2^32 := by rw [e_w3]; exact hst.2.2.2.1
  -- w4: post-round accumulator argument
  word_step w4 := st.w4
  have b_w4 : w4 < 2^32 := by rw [e_w4]; exact hst.2.2.2.2.1
  -- w5: post-round accumulator argument
  word_step w5 := st.w5
  have b_w5 : w5 < 2^32 := by rw [e_w5]; exact hst.2.2.2.2.2.1
  -- w6: post-round accumulator argument
  word_step w6 := st.w6
  have b_w6 : w6 < 2^32 := by rw [e_w6]; exact hst.2.2.2.2.2.2.1
  -- w7: post-round accumulator argument
  word_step w7 := st.w7
  have b_w7 : w7 < 2^32 := by rw [e_w7]; exact hst.2.2.2.2.2.2.2.1
  -- w8: post-round accumulator argument
  word_step w8 := st.w8
  have b_w8 : w8 < 2^32 := by rw [e_w8]; exact hst.2.2.2.2.2.2.2.2
  -- w0_1: subs.w lr, lr, #1
  word_step d, w0_1 := (w0 + 2^32 - 1 - (1 - 1)) % 2^32, c := (w0 + 2^32 - 1 - (1 - 1)) / 2^32
  have b_w0_1 : w0_1 < 2^32 := by rw [e_w0_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w0_1 : w0_1 + 2^32 * c + 1 + 1 = w0 + 2^32 + 1 := by
    rw [e_w0_1, e_c]; exact subc32_lin w0 1 1 (by decide) (by decide)
  have b_c : c ≤ 1 := by
    rw [e_c]; exact subc32_carry_le_one w0 1 1 b_w0
  -- BEGIN subtraction word 0
  have hstep0 : subc32 st.w0 1 1 = (w0_1, c) := by
    simp only [subc32, wordMod, e_w0, e_w0_1, e_c]
  -- END subtraction word 0
  clear e_w0_1 e_c
  -- r11: ldr.w r11, [r1, #4]
  word_step r11 := r1_4
  have b_r11 : r11 < 2^32 := by rw [e_r11]; exact b_r1_4
  -- w1_1: sbcs.w r0, r0, r11
  word_step d_1, w1_1 := (w1 + 2^32 - r11 - (1 - c)) % 2^32,
      c_1 := (w1 + 2^32 - r11 - (1 - c)) / 2^32
  have b_w1_1 : w1_1 < 2^32 := by rw [e_w1_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w1_1 : w1_1 + 2^32 * c_1 + r11 + 1 = w1 + 2^32 + c := by
    rw [e_w1_1, e_c_1]; exact subc32_lin w1 r11 c b_r11 b_c
  have b_c_1 : c_1 ≤ 1 := by
    rw [e_c_1]; exact subc32_carry_le_one w1 r11 c b_w1
  -- BEGIN subtraction word 1
  have hstep1 : subc32 st.w1 (modulus.l0 / wordMod) c = (w1_1, c_1) := by
    simp only [subc32, wordMod, e_w1, e_r11, e_r1_4, e_w1_1, e_c_1]
  -- END subtraction word 1
  clear e_w1_1 e_c_1
  -- r11_1: ldr.w r11, [r1, #8]
  word_step r11_1 := r1_8
  have b_r11_1 : r11_1 < 2^32 := by rw [e_r11_1]; exact b_r1_8
  -- w2_1: sbcs.w r2, r2, r11
  word_step d_2, w2_1 := (w2 + 2^32 - r11_1 - (1 - c_1)) % 2^32,
      c_2 := (w2 + 2^32 - r11_1 - (1 - c_1)) / 2^32
  have b_w2_1 : w2_1 < 2^32 := by rw [e_w2_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w2_1 : w2_1 + 2^32 * c_2 + r11_1 + 1 = w2 + 2^32 + c_1 := by
    rw [e_w2_1, e_c_2]; exact subc32_lin w2 r11_1 c_1 b_r11_1 b_c_1
  have b_c_2 : c_2 ≤ 1 := by
    rw [e_c_2]; exact subc32_carry_le_one w2 r11_1 c_1 b_w2
  -- BEGIN subtraction word 2
  have hstep2 : subc32 st.w2 (modulus.l1 % wordMod) c_1 = (w2_1, c_2) := by
    simp only [subc32, wordMod, e_w2, e_r11_1, e_r1_8, e_w2_1, e_c_2]
  -- END subtraction word 2
  clear e_w2_1 e_c_2
  -- r11_2: ldr.w r11, [r1, #12]
  word_step r11_2 := r1_12
  have b_r11_2 : r11_2 < 2^32 := by rw [e_r11_2]; exact b_r1_12
  -- w3_1: sbcs.w r3, r3, r11
  word_step d_3, w3_1 := (w3 + 2^32 - r11_2 - (1 - c_2)) % 2^32,
      c_3 := (w3 + 2^32 - r11_2 - (1 - c_2)) / 2^32
  have b_w3_1 : w3_1 < 2^32 := by rw [e_w3_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w3_1 : w3_1 + 2^32 * c_3 + r11_2 + 1 = w3 + 2^32 + c_2 := by
    rw [e_w3_1, e_c_3]; exact subc32_lin w3 r11_2 c_2 b_r11_2 b_c_2
  have b_c_3 : c_3 ≤ 1 := by
    rw [e_c_3]; exact subc32_carry_le_one w3 r11_2 c_2 b_w3
  -- BEGIN subtraction word 3
  have hstep3 : subc32 st.w3 (modulus.l1 / wordMod) c_2 = (w3_1, c_3) := by
    simp only [subc32, wordMod, e_w3, e_r11_2, e_r1_12, e_w3_1, e_c_3]
  -- END subtraction word 3
  clear e_w3_1 e_c_3
  -- r11_3: mov.w r11, #0
  word_step r11_3 := 0
  have b_r11_3 : r11_3 < 2^32 := by rw [e_r11_3]; exact (by decide)
  -- w4_1: sbcs.w r4, r4, r11
  word_step d_4, w4_1 := (w4 + 2^32 - r11_3 - (1 - c_3)) % 2^32,
      c_4 := (w4 + 2^32 - r11_3 - (1 - c_3)) / 2^32
  have b_w4_1 : w4_1 < 2^32 := by rw [e_w4_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w4_1 : w4_1 + 2^32 * c_4 + r11_3 + 1 = w4 + 2^32 + c_3 := by
    rw [e_w4_1, e_c_4]; exact subc32_lin w4 r11_3 c_3 b_r11_3 b_c_3
  have b_c_4 : c_4 ≤ 1 := by
    rw [e_c_4]; exact subc32_carry_le_one w4 r11_3 c_3 b_w4
  -- BEGIN subtraction word 4
  have hstep4 : subc32 st.w4 0 c_3 = (w4_1, c_4) := by
    simp only [subc32, wordMod, e_w4, e_r11_3, e_w4_1, e_c_4]
  -- END subtraction word 4
  clear e_w4_1 e_c_4
  -- w5_1: sbcs.w r5, r5, r11
  word_step d_5, w5_1 := (w5 + 2^32 - r11_3 - (1 - c_4)) % 2^32,
      c_5 := (w5 + 2^32 - r11_3 - (1 - c_4)) / 2^32
  have b_w5_1 : w5_1 < 2^32 := by rw [e_w5_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w5_1 : w5_1 + 2^32 * c_5 + r11_3 + 1 = w5 + 2^32 + c_4 := by
    rw [e_w5_1, e_c_5]; exact subc32_lin w5 r11_3 c_4 b_r11_3 b_c_4
  have b_c_5 : c_5 ≤ 1 := by
    rw [e_c_5]; exact subc32_carry_le_one w5 r11_3 c_4 b_w5
  -- BEGIN subtraction word 5
  have hstep5 : subc32 st.w5 0 c_4 = (w5_1, c_5) := by
    simp only [subc32, wordMod, e_w5, e_r11_3, e_w5_1, e_c_5]
  -- END subtraction word 5
  clear e_w5_1 e_c_5
  -- w6_1: sbcs.w r8, r8, r11
  word_step d_6, w6_1 := (w6 + 2^32 - r11_3 - (1 - c_5)) % 2^32,
      c_6 := (w6 + 2^32 - r11_3 - (1 - c_5)) / 2^32
  have b_w6_1 : w6_1 < 2^32 := by rw [e_w6_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w6_1 : w6_1 + 2^32 * c_6 + r11_3 + 1 = w6 + 2^32 + c_5 := by
    rw [e_w6_1, e_c_6]; exact subc32_lin w6 r11_3 c_5 b_r11_3 b_c_5
  have b_c_6 : c_6 ≤ 1 := by
    rw [e_c_6]; exact subc32_carry_le_one w6 r11_3 c_5 b_w6
  -- BEGIN subtraction word 6
  have hstep6 : subc32 st.w6 0 c_5 = (w6_1, c_6) := by
    simp only [subc32, wordMod, e_w6, e_r11_3, e_w6_1, e_c_6]
  -- END subtraction word 6
  clear e_w6_1 e_c_6
  -- r11_4: mov.w r11, #0x40000000
  word_step r11_4 := 0x40000000
  have b_r11_4 : r11_4 < 2^32 := by rw [e_r11_4]; exact (by decide)
  -- w7_1: sbcs.w r9, r9, r11
  word_step d_7, w7_1 := (w7 + 2^32 - r11_4 - (1 - c_6)) % 2^32,
      c_7 := (w7 + 2^32 - r11_4 - (1 - c_6)) / 2^32
  have b_w7_1 : w7_1 < 2^32 := by rw [e_w7_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_w7_1 : w7_1 + 2^32 * c_7 + r11_4 + 1 = w7 + 2^32 + c_6 := by
    rw [e_w7_1, e_c_7]; exact subc32_lin w7 r11_4 c_6 b_r11_4 b_c_6
  have b_c_7 : c_7 ≤ 1 := by
    rw [e_c_7]; exact subc32_carry_le_one w7 r11_4 c_6 b_w7
  -- BEGIN subtraction word 7
  have hstep7 : subc32 st.w7 0x40000000 c_6 = (w7_1, c_7) := by
    simp only [subc32, wordMod, e_w7, e_r11_4, e_w7_1, e_c_7]
  -- END subtraction word 7
  clear e_w7_1 e_c_7
  -- r12: sbc.w r12, r10, r10
  word_step r12 := (w8 + 2^32 - w8 - (1 - c_7)) % 2^32
  have b_r12 : r12 < 2^32 := by rw [e_r12]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  subst hres
  -- BEGIN fromMontFinalSubtractionBlock_spec conclusion
  have b_c_7_word : c_7 < 2^32 := by
    clear * - b_c_7
    omega
  refine ⟨⟨b_w0_1, b_w1_1, b_w2_1, b_w3_1, b_w4_1, b_w5_1, b_w6_1,
    b_w7_1, b_w8, b_c_7_word, b_r12⟩, ?_, ?_, ?_⟩
  · simp only [subcChain, hstep0, hstep1, hstep2, hstep3, hstep4, hstep5, hstep6,
      hstep7]
  · exact e_w8
  · simp only [subc32, wordMod, e_w8, e_r12]
  -- END fromMontFinalSubtractionBlock_spec conclusion

end PastaCurves.ARMv7EM
