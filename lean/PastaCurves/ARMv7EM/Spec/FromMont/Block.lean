/-
Copyright (c) 2026 the pasta_curves contributors.
-/
import PastaCurves.ARMv7EM.Spec.FromMont.Correctness
import PastaCurves.ARMv7EM.Spec.FromMont.FinalReduction.Block

/-!
# Correctness of the generated ARMv7E-M Montgomery conversion block

The checked outer caller loads eight words, applies the checked conversion round eight times,
applies the checked final reduction once, and joins the canonical output words into four limbs.
-/

namespace PastaCurves.ARMv7EM

open ListArithmetic

set_option maxHeartbeats 1000000 in
-- BEGIN fromMontBlock_spec statement
/-- The generated conversion block returns the canonical value whose multiplication by the
Montgomery radix is congruent to the input. -/
theorem fromMontBlock_spec (value modulus : Limbs) (hv : value.Bounded)
    (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hlow : modulus.l0 % wordMod = 1) :
    ∀ res, res = fromMontBlock value modulus →
      res.Bounded ∧ res.toNat < modulus.toNat ∧
        wordMod^8 * res.toNat ≡ value.toNat [MOD modulus.toNat] := by
  intro res hres
-- END fromMontBlock_spec statement
  -- generated skeleton for `fromMontBlock`: do not edit between the annotations
  unfold fromMontBlock at hres
  lift_lets -merge at hres
  -- r0_0: input r0[0]
  word_step r0_0 := value.l0 % 2^32
  have b_r0_0 : r0_0 < 2^32 := by
    rw [e_r0_0]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- r0_4: input r0[4]
  word_step r0_4 := value.l0 / 2^32
  have b_r0_4 : r0_4 < 2^32 := by
    rw [e_r0_4]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq hv.1 (by norm_num))
  -- r0_8: input r0[8]
  word_step r0_8 := value.l1 % 2^32
  have b_r0_8 : r0_8 < 2^32 := by
    rw [e_r0_8]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- r0_12: input r0[12]
  word_step r0_12 := value.l1 / 2^32
  have b_r0_12 : r0_12 < 2^32 := by
    rw [e_r0_12]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq hv.2.1 (by norm_num))
  -- r0_16: input r0[16]
  word_step r0_16 := value.l2 % 2^32
  have b_r0_16 : r0_16 < 2^32 := by
    rw [e_r0_16]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- r0_20: input r0[20]
  word_step r0_20 := value.l2 / 2^32
  have b_r0_20 : r0_20 < 2^32 := by
    rw [e_r0_20]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq hv.2.2.1 (by norm_num))
  -- r0_24: input r0[24]
  word_step r0_24 := value.l3 % 2^32
  have b_r0_24 : r0_24 < 2^32 := by
    rw [e_r0_24]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- r0_28: input r0[28]
  word_step r0_28 := value.l3 / 2^32
  have b_r0_28 : r0_28 < 2^32 := by
    rw [e_r0_28]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq hv.2.2.2 (by norm_num))
  -- r2: ldr.w r2, [r0, #0]
  word_step r2 := r0_0
  have b_r2 : r2 < 2^32 := by rw [e_r2]; exact b_r0_0
  -- r3: ldr.w r3, [r0, #4]
  word_step r3 := r0_4
  have b_r3 : r3 < 2^32 := by rw [e_r3]; exact b_r0_4
  -- r4: ldr.w r4, [r0, #8]
  word_step r4 := r0_8
  have b_r4 : r4 < 2^32 := by rw [e_r4]; exact b_r0_8
  -- r5: ldr.w r5, [r0, #12]
  word_step r5 := r0_12
  have b_r5 : r5 < 2^32 := by rw [e_r5]; exact b_r0_12
  -- r8: ldr.w r8, [r0, #16]
  word_step r8 := r0_16
  have b_r8 : r8 < 2^32 := by rw [e_r8]; exact b_r0_16
  -- r9: ldr.w r9, [r0, #20]
  word_step r9 := r0_20
  have b_r9 : r9 < 2^32 := by rw [e_r9]; exact b_r0_20
  -- r10: ldr.w r10, [r0, #24]
  word_step r10 := r0_24
  have b_r10 : r10 < 2^32 := by rw [e_r10]; exact b_r0_24
  -- r11: ldr.w r11, [r0, #28]
  word_step r11 := r0_28
  have b_r11 : r11 < 2^32 := by rw [e_r11]; exact b_r0_28
  -- r14: mov.w lr, #0
  word_step r14 := 0
  have b_r14 : r14 < 2^32 := by rw [e_r14]; exact (by decide)
  -- BEGIN fromMontBlock initial state
  let s0 : FromMontState := ⟨r2, r3, r4, r5, r8, r9, r10, r11, r14⟩
  have bs0 : s0.Bounded :=
    ⟨b_r2, b_r3, b_r4, b_r5, b_r8, b_r9, b_r10, b_r11, b_r14⟩
  have vs0 : s0.toMulAcc.value = value.toNat := by
    simpa only [s0, FromMontState.toMulAcc, MulAcc.value, toWords,
      e_r2, e_r3, e_r4, e_r5, e_r8, e_r9, e_r10, e_r11, e_r14,
      e_r0_0, e_r0_4, e_r0_8, e_r0_12, e_r0_16, e_r0_20, e_r0_24, e_r0_28,
      ListArithmetic.wordsValue, Nat.mul_zero, Nat.add_zero] using wordsValue_toWords value hv
  -- END fromMontBlock initial state
  -- round1: from_mont_round!, invocation 1
  word_step round1 := fromMontRound modulus ⟨r2, r3, r4, r5, r8, r9, r10, r11, r14⟩
  -- BEGIN fromMontBlock round 1 call
  have hr1 := fromMontRound_value_spec modulus s0 round1 hm bs0 hshape hlow (by
    simpa only [s0] using e_round1)
  have b_round1 : round1.Bounded := hr1.1
  -- END fromMontBlock round 1 call
  -- r3_1: round 1 output
  word_step r3_1 := round1.w0
  -- r4_1: round 1 output
  word_step r4_1 := round1.w1
  -- r5_1: round 1 output
  word_step r5_1 := round1.w2
  -- r8_1: round 1 output
  word_step r8_1 := round1.w3
  -- r9_1: round 1 output
  word_step r9_1 := round1.w4
  -- r10_1: round 1 output
  word_step r10_1 := round1.w5
  -- r11_1: round 1 output
  word_step r11_1 := round1.w6
  -- r14_1: round 1 output
  word_step r14_1 := round1.w7
  -- r0: round 1 output
  word_step r0 := round1.w8
  -- BEGIN fromMontBlock round 1 state
  let s1 : FromMontState := round1
  have es1 : s1 = round1 := rfl
  have bs1 : s1.Bounded := b_round1
  let quotient0 := fromMontQuotient s0
  have bq0 : quotient0 < wordMod := hr1.2.1
  have step0 : wordMod * s1.toMulAcc.value = s0.toMulAcc.value + modulus.toNat * quotient0 :=
    hr1.2.2
  -- END fromMontBlock round 1 state
  -- round2: from_mont_round!, invocation 2
  word_step round2 := fromMontRound modulus ⟨r3_1, r4_1, r5_1, r8_1, r9_1, r10_1, r11_1, r14_1, r0⟩
  -- BEGIN fromMontBlock round 2 call
  have eround2Input : (⟨r3_1, r4_1, r5_1, r8_1, r9_1, r10_1, r11_1, r14_1, r0⟩ : FromMontState) = s1 := by
    dsimp only [s1]
    rw [e_r3_1, e_r4_1, e_r5_1, e_r8_1, e_r9_1, e_r10_1, e_r11_1, e_r14_1, e_r0]
  have hr2 := fromMontRound_value_spec modulus s1 round2 hm bs1 hshape hlow (by
    rw [e_round2, eround2Input])
  have b_round2 : round2.Bounded := hr2.1
  -- END fromMontBlock round 2 call
  -- r4_2: round 2 output
  word_step r4_2 := round2.w0
  -- r5_2: round 2 output
  word_step r5_2 := round2.w1
  -- r8_2: round 2 output
  word_step r8_2 := round2.w2
  -- r9_2: round 2 output
  word_step r9_2 := round2.w3
  -- r10_2: round 2 output
  word_step r10_2 := round2.w4
  -- r11_2: round 2 output
  word_step r11_2 := round2.w5
  -- r14_2: round 2 output
  word_step r14_2 := round2.w6
  -- r0_1: round 2 output
  word_step r0_1 := round2.w7
  -- r2_1: round 2 output
  word_step r2_1 := round2.w8
  -- BEGIN fromMontBlock round 2 state
  let s2 : FromMontState := round2
  have es2 : s2 = round2 := rfl
  have bs2 : s2.Bounded := b_round2
  let quotient1 := fromMontQuotient s1
  have bq1 : quotient1 < wordMod := hr2.2.1
  have step1 : wordMod * s2.toMulAcc.value = s1.toMulAcc.value + modulus.toNat * quotient1 :=
    hr2.2.2
  -- END fromMontBlock round 2 state
  -- round3: from_mont_round!, invocation 3
  word_step round3 := fromMontRound modulus ⟨r4_2, r5_2, r8_2, r9_2, r10_2, r11_2, r14_2, r0_1, r2_1⟩
  -- BEGIN fromMontBlock round 3 call
  have eround3Input : (⟨r4_2, r5_2, r8_2, r9_2, r10_2, r11_2, r14_2, r0_1, r2_1⟩ : FromMontState) = s2 := by
    dsimp only [s2]
    rw [e_r4_2, e_r5_2, e_r8_2, e_r9_2, e_r10_2, e_r11_2, e_r14_2, e_r0_1, e_r2_1]
  have hr3 := fromMontRound_value_spec modulus s2 round3 hm bs2 hshape hlow (by
    rw [e_round3, eround3Input])
  have b_round3 : round3.Bounded := hr3.1
  -- END fromMontBlock round 3 call
  -- r5_3: round 3 output
  word_step r5_3 := round3.w0
  -- r8_3: round 3 output
  word_step r8_3 := round3.w1
  -- r9_3: round 3 output
  word_step r9_3 := round3.w2
  -- r10_3: round 3 output
  word_step r10_3 := round3.w3
  -- r11_3: round 3 output
  word_step r11_3 := round3.w4
  -- r14_3: round 3 output
  word_step r14_3 := round3.w5
  -- r0_2: round 3 output
  word_step r0_2 := round3.w6
  -- r2_2: round 3 output
  word_step r2_2 := round3.w7
  -- r3_2: round 3 output
  word_step r3_2 := round3.w8
  -- BEGIN fromMontBlock round 3 state
  let s3 : FromMontState := round3
  have es3 : s3 = round3 := rfl
  have bs3 : s3.Bounded := b_round3
  let quotient2 := fromMontQuotient s2
  have bq2 : quotient2 < wordMod := hr3.2.1
  have step2 : wordMod * s3.toMulAcc.value = s2.toMulAcc.value + modulus.toNat * quotient2 :=
    hr3.2.2
  -- END fromMontBlock round 3 state
  -- round4: from_mont_round!, invocation 4
  word_step round4 := fromMontRound modulus ⟨r5_3, r8_3, r9_3, r10_3, r11_3, r14_3, r0_2, r2_2, r3_2⟩
  -- BEGIN fromMontBlock round 4 call
  have eround4Input : (⟨r5_3, r8_3, r9_3, r10_3, r11_3, r14_3, r0_2, r2_2, r3_2⟩ : FromMontState) = s3 := by
    dsimp only [s3]
    rw [e_r5_3, e_r8_3, e_r9_3, e_r10_3, e_r11_3, e_r14_3, e_r0_2, e_r2_2, e_r3_2]
  have hr4 := fromMontRound_value_spec modulus s3 round4 hm bs3 hshape hlow (by
    rw [e_round4, eround4Input])
  have b_round4 : round4.Bounded := hr4.1
  -- END fromMontBlock round 4 call
  -- r8_4: round 4 output
  word_step r8_4 := round4.w0
  -- r9_4: round 4 output
  word_step r9_4 := round4.w1
  -- r10_4: round 4 output
  word_step r10_4 := round4.w2
  -- r11_4: round 4 output
  word_step r11_4 := round4.w3
  -- r14_4: round 4 output
  word_step r14_4 := round4.w4
  -- r0_3: round 4 output
  word_step r0_3 := round4.w5
  -- r2_3: round 4 output
  word_step r2_3 := round4.w6
  -- r3_3: round 4 output
  word_step r3_3 := round4.w7
  -- r4_3: round 4 output
  word_step r4_3 := round4.w8
  -- BEGIN fromMontBlock round 4 state
  let s4 : FromMontState := round4
  have es4 : s4 = round4 := rfl
  have bs4 : s4.Bounded := b_round4
  let quotient3 := fromMontQuotient s3
  have bq3 : quotient3 < wordMod := hr4.2.1
  have step3 : wordMod * s4.toMulAcc.value = s3.toMulAcc.value + modulus.toNat * quotient3 :=
    hr4.2.2
  -- END fromMontBlock round 4 state
  -- round5: from_mont_round!, invocation 5
  word_step round5 := fromMontRound modulus ⟨r8_4, r9_4, r10_4, r11_4, r14_4, r0_3, r2_3, r3_3, r4_3⟩
  -- BEGIN fromMontBlock round 5 call
  have eround5Input : (⟨r8_4, r9_4, r10_4, r11_4, r14_4, r0_3, r2_3, r3_3, r4_3⟩ : FromMontState) = s4 := by
    dsimp only [s4]
    rw [e_r8_4, e_r9_4, e_r10_4, e_r11_4, e_r14_4, e_r0_3, e_r2_3, e_r3_3, e_r4_3]
  have hr5 := fromMontRound_value_spec modulus s4 round5 hm bs4 hshape hlow (by
    rw [e_round5, eround5Input])
  have b_round5 : round5.Bounded := hr5.1
  -- END fromMontBlock round 5 call
  -- r9_5: round 5 output
  word_step r9_5 := round5.w0
  -- r10_5: round 5 output
  word_step r10_5 := round5.w1
  -- r11_5: round 5 output
  word_step r11_5 := round5.w2
  -- r14_5: round 5 output
  word_step r14_5 := round5.w3
  -- r0_4: round 5 output
  word_step r0_4 := round5.w4
  -- r2_4: round 5 output
  word_step r2_4 := round5.w5
  -- r3_4: round 5 output
  word_step r3_4 := round5.w6
  -- r4_4: round 5 output
  word_step r4_4 := round5.w7
  -- r5_4: round 5 output
  word_step r5_4 := round5.w8
  -- BEGIN fromMontBlock round 5 state
  let s5 : FromMontState := round5
  have es5 : s5 = round5 := rfl
  have bs5 : s5.Bounded := b_round5
  let quotient4 := fromMontQuotient s4
  have bq4 : quotient4 < wordMod := hr5.2.1
  have step4 : wordMod * s5.toMulAcc.value = s4.toMulAcc.value + modulus.toNat * quotient4 :=
    hr5.2.2
  -- END fromMontBlock round 5 state
  -- round6: from_mont_round!, invocation 6
  word_step round6 := fromMontRound modulus ⟨r9_5, r10_5, r11_5, r14_5, r0_4, r2_4, r3_4, r4_4, r5_4⟩
  -- BEGIN fromMontBlock round 6 call
  have eround6Input : (⟨r9_5, r10_5, r11_5, r14_5, r0_4, r2_4, r3_4, r4_4, r5_4⟩ : FromMontState) = s5 := by
    dsimp only [s5]
    rw [e_r9_5, e_r10_5, e_r11_5, e_r14_5, e_r0_4, e_r2_4, e_r3_4, e_r4_4, e_r5_4]
  have hr6 := fromMontRound_value_spec modulus s5 round6 hm bs5 hshape hlow (by
    rw [e_round6, eround6Input])
  have b_round6 : round6.Bounded := hr6.1
  -- END fromMontBlock round 6 call
  -- r10_6: round 6 output
  word_step r10_6 := round6.w0
  -- r11_6: round 6 output
  word_step r11_6 := round6.w1
  -- r14_6: round 6 output
  word_step r14_6 := round6.w2
  -- r0_5: round 6 output
  word_step r0_5 := round6.w3
  -- r2_5: round 6 output
  word_step r2_5 := round6.w4
  -- r3_5: round 6 output
  word_step r3_5 := round6.w5
  -- r4_5: round 6 output
  word_step r4_5 := round6.w6
  -- r5_5: round 6 output
  word_step r5_5 := round6.w7
  -- r8_5: round 6 output
  word_step r8_5 := round6.w8
  -- BEGIN fromMontBlock round 6 state
  let s6 : FromMontState := round6
  have es6 : s6 = round6 := rfl
  have bs6 : s6.Bounded := b_round6
  let quotient5 := fromMontQuotient s5
  have bq5 : quotient5 < wordMod := hr6.2.1
  have step5 : wordMod * s6.toMulAcc.value = s5.toMulAcc.value + modulus.toNat * quotient5 :=
    hr6.2.2
  -- END fromMontBlock round 6 state
  -- round7: from_mont_round!, invocation 7
  word_step round7 := fromMontRound modulus ⟨r10_6, r11_6, r14_6, r0_5, r2_5, r3_5, r4_5, r5_5, r8_5⟩
  -- BEGIN fromMontBlock round 7 call
  have eround7Input : (⟨r10_6, r11_6, r14_6, r0_5, r2_5, r3_5, r4_5, r5_5, r8_5⟩ : FromMontState) = s6 := by
    dsimp only [s6]
    rw [e_r10_6, e_r11_6, e_r14_6, e_r0_5, e_r2_5, e_r3_5, e_r4_5, e_r5_5, e_r8_5]
  have hr7 := fromMontRound_value_spec modulus s6 round7 hm bs6 hshape hlow (by
    rw [e_round7, eround7Input])
  have b_round7 : round7.Bounded := hr7.1
  -- END fromMontBlock round 7 call
  -- r11_7: round 7 output
  word_step r11_7 := round7.w0
  -- r14_7: round 7 output
  word_step r14_7 := round7.w1
  -- r0_6: round 7 output
  word_step r0_6 := round7.w2
  -- r2_6: round 7 output
  word_step r2_6 := round7.w3
  -- r3_6: round 7 output
  word_step r3_6 := round7.w4
  -- r4_6: round 7 output
  word_step r4_6 := round7.w5
  -- r5_6: round 7 output
  word_step r5_6 := round7.w6
  -- r8_6: round 7 output
  word_step r8_6 := round7.w7
  -- r9_6: round 7 output
  word_step r9_6 := round7.w8
  -- BEGIN fromMontBlock round 7 state
  let s7 : FromMontState := round7
  have es7 : s7 = round7 := rfl
  have bs7 : s7.Bounded := b_round7
  let quotient6 := fromMontQuotient s6
  have bq6 : quotient6 < wordMod := hr7.2.1
  have step6 : wordMod * s7.toMulAcc.value = s6.toMulAcc.value + modulus.toNat * quotient6 :=
    hr7.2.2
  -- END fromMontBlock round 7 state
  -- round8: from_mont_round!, invocation 8
  word_step round8 := fromMontRound modulus ⟨r11_7, r14_7, r0_6, r2_6, r3_6, r4_6, r5_6, r8_6, r9_6⟩
  -- BEGIN fromMontBlock round 8 call
  have eround8Input : (⟨r11_7, r14_7, r0_6, r2_6, r3_6, r4_6, r5_6, r8_6, r9_6⟩ : FromMontState) = s7 := by
    dsimp only [s7]
    rw [e_r11_7, e_r14_7, e_r0_6, e_r2_6, e_r3_6, e_r4_6, e_r5_6, e_r8_6, e_r9_6]
  have hr8 := fromMontRound_value_spec modulus s7 round8 hm bs7 hshape hlow (by
    rw [e_round8, eround8Input])
  have b_round8 : round8.Bounded := hr8.1
  -- END fromMontBlock round 8 call
  -- r14_8: round 8 output
  word_step r14_8 := round8.w0
  -- r0_7: round 8 output
  word_step r0_7 := round8.w1
  -- r2_7: round 8 output
  word_step r2_7 := round8.w2
  -- r3_7: round 8 output
  word_step r3_7 := round8.w3
  -- r4_7: round 8 output
  word_step r4_7 := round8.w4
  -- r5_7: round 8 output
  word_step r5_7 := round8.w5
  -- r8_7: round 8 output
  word_step r8_7 := round8.w6
  -- r9_7: round 8 output
  word_step r9_7 := round8.w7
  -- r10_7: round 8 output
  word_step r10_7 := round8.w8
  -- BEGIN fromMontBlock round 8 state
  let s8 : FromMontState := round8
  have es8 : s8 = round8 := rfl
  have bs8 : s8.Bounded := b_round8
  let quotient7 := fromMontQuotient s7
  have bq7 : quotient7 < wordMod := hr8.2.1
  have step7 : wordMod * s8.toMulAcc.value = s7.toMulAcc.value + modulus.toNat * quotient7 :=
    hr8.2.2
  have hrounds := fromMont_eight_explicit_spec value modulus s0 s1 s2 s3 s4 s5 s6 s7 s8
    quotient0 quotient1 quotient2 quotient3 quotient4 quotient5 quotient6 quotient7
    hv hm hshape vs0 step0 step1 step2 step3 step4 step5 step6 step7
    bq0 bq1 bq2 bq3 bq4 bq5 bq6 bq7
  -- END fromMontBlock round 8 state
  -- finalReduction: checked final reduction, PCs 177 through 207
  word_step finalReduction := fromMontFinalReductionBlock modulus ⟨r14_8, r0_7, r2_7, r3_7, r4_7, r5_7, r8_7, r9_7, r10_7⟩
  -- BEGIN fromMontBlock final reduction call
  have efinalInput :
      (⟨r14_8, r0_7, r2_7, r3_7, r4_7, r5_7, r8_7, r9_7, r10_7⟩ : FromMontState) = s8 := by
    dsimp only [s8]
    rw [e_r14_8, e_r0_7, e_r2_7, e_r3_7, e_r4_7, e_r5_7, e_r8_7, e_r9_7, e_r10_7]
  have hfinal := fromMontFinalReductionBlock_sparse_spec modulus s8 hm bs8 hshape hlow
    hrounds.1 hrounds.2.1 finalReduction (by rw [e_finalReduction, efinalInput])
  dsimp only at hfinal
  have b_finalReduction : finalReduction.Bounded := hfinal.2.1
  -- END fromMontBlock final reduction call
  -- r14_9: final reduction output
  word_step r14_9 := finalReduction.w0
  -- BEGIN fromMontBlock final word 0 bound
  have b_r14_9 : r14_9 < 2^32 := by rw [e_r14_9]; exact b_finalReduction.1
  -- END fromMontBlock final word 0 bound
  -- r0_8: final reduction output
  word_step r0_8 := finalReduction.w1
  -- BEGIN fromMontBlock final word 1 bound
  have b_r0_8 : r0_8 < 2^32 := by rw [e_r0_8]; exact b_finalReduction.2.1
  -- END fromMontBlock final word 1 bound
  -- r2_8: final reduction output
  word_step r2_8 := finalReduction.w2
  -- BEGIN fromMontBlock final word 2 bound
  have b_r2_8 : r2_8 < 2^32 := by rw [e_r2_8]; exact b_finalReduction.2.2.1
  -- END fromMontBlock final word 2 bound
  -- r3_8: final reduction output
  word_step r3_8 := finalReduction.w3
  -- BEGIN fromMontBlock final word 3 bound
  have b_r3_8 : r3_8 < 2^32 := by rw [e_r3_8]; exact b_finalReduction.2.2.2.1
  -- END fromMontBlock final word 3 bound
  -- r4_8: final reduction output
  word_step r4_8 := finalReduction.w4
  -- BEGIN fromMontBlock final word 4 bound
  have b_r4_8 : r4_8 < 2^32 := by rw [e_r4_8]; exact b_finalReduction.2.2.2.2.1
  -- END fromMontBlock final word 4 bound
  -- r5_8: final reduction output
  word_step r5_8 := finalReduction.w5
  -- BEGIN fromMontBlock final word 5 bound
  have b_r5_8 : r5_8 < 2^32 := by rw [e_r5_8]; exact b_finalReduction.2.2.2.2.2.1
  -- END fromMontBlock final word 5 bound
  -- r8_8: final reduction output
  word_step r8_8 := finalReduction.w6
  -- BEGIN fromMontBlock final word 6 bound
  have b_r8_8 : r8_8 < 2^32 := by rw [e_r8_8]; exact b_finalReduction.2.2.2.2.2.2.1
  -- END fromMontBlock final word 6 bound
  -- r9_8: final reduction output
  word_step r9_8 := finalReduction.w7
  -- BEGIN fromMontBlock final word 7 bound
  have b_r9_8 : r9_8 < 2^32 := by rw [e_r9_8]; exact b_finalReduction.2.2.2.2.2.2.2.1
  -- END fromMontBlock final word 7 bound
  -- result0: join returned words
  word_step result0 := r14_9 + 2^32 * r0_8
  have b_result0 : result0 < 2^64 := by
    rw [e_result0]; clear * - b_r14_9 b_r0_8; omega
  -- result1: join returned words
  word_step result1 := r2_8 + 2^32 * r3_8
  have b_result1 : result1 < 2^64 := by
    rw [e_result1]; clear * - b_r2_8 b_r3_8; omega
  -- result2: join returned words
  word_step result2 := r4_8 + 2^32 * r5_8
  have b_result2 : result2 < 2^64 := by
    rw [e_result2]; clear * - b_r4_8 b_r5_8; omega
  -- result3: join returned words
  word_step result3 := r8_8 + 2^32 * r9_8
  have b_result3 : result3 < 2^64 := by
    rw [e_result3]; clear * - b_r8_8 b_r9_8; omega
  subst hres
  -- BEGIN fromMontBlock_spec conclusion
  have houtput := fromMont_output_spec finalReduction b_finalReduction
  dsimp only at houtput
  have hcandidateValue :
      wordsValue [s8.w0, s8.w1, s8.w2, s8.w3, s8.w4, s8.w5, s8.w6, s8.w7] =
        s8.toMulAcc.value := by
    simp only [FromMontState.toMulAcc, MulAcc.value, wordsValue, hrounds.1]
    ring
  have hfinalCongruence :
      wordsValue [finalReduction.w0, finalReduction.w1, finalReduction.w2,
        finalReduction.w3, finalReduction.w4, finalReduction.w5,
        finalReduction.w6, finalReduction.w7] ≡ s8.toMulAcc.value [MOD modulus.toNat] := by
    rw [← hcandidateValue]
    exact hfinal.2.2.2.1
  have hresult :
      (⟨result0, result1, result2, result3⟩ : Limbs) =
        ⟨finalReduction.w0 + wordMod * finalReduction.w1,
          finalReduction.w2 + wordMod * finalReduction.w3,
          finalReduction.w4 + wordMod * finalReduction.w5,
          finalReduction.w6 + wordMod * finalReduction.w7⟩ := by
    rw [e_result0, e_result1, e_result2, e_result3,
      e_r14_9, e_r0_8, e_r2_8, e_r3_8, e_r4_8, e_r5_8, e_r8_8, e_r9_8]
  rw [hresult]
  refine ⟨houtput.1, ?_, ?_⟩
  · rw [houtput.2]
    exact hfinal.2.2.1
  · have houtputCongruence :
        (⟨finalReduction.w0 + wordMod * finalReduction.w1,
          finalReduction.w2 + wordMod * finalReduction.w3,
          finalReduction.w4 + wordMod * finalReduction.w5,
          finalReduction.w6 + wordMod * finalReduction.w7⟩ : Limbs).toNat ≡
            s8.toMulAcc.value [MOD modulus.toNat] := by
      rw [houtput.2]
      exact hfinalCongruence
    exact (Nat.ModEq.mul_left (wordMod^8) houtputCongruence).trans hrounds.2.2
  -- END fromMontBlock_spec conclusion

end PastaCurves.ARMv7EM
