/-
Copyright (c) 2026 the pasta_curves contributors.
-/
import PastaCurves.ARMv7EM.Spec.FromMont.Prefix
import PastaCurves.ARMv7EM.Spec.FromMont.Suffix

/-!
# The checked caller for one ARMv7E-M Montgomery conversion round

The generated caller composes the checked quotient prefix and sparse multiply-accumulate suffix.
Its result is exactly the projected pure sparse round, with the corresponding word bound and value
equation.
-/

namespace PastaCurves.ARMv7EM

-- BEGIN fromMontRound_spec statement
/-- One generated conversion-round caller is exactly one pure sparse Montgomery round. -/
theorem fromMontRound_spec (modulus : Limbs) (st : FromMontState)
    (hm : modulus.Bounded) (hst : st.Bounded)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hlow : modulus.l0 % wordMod = 1) :
    ∀ res, res = fromMontRound modulus st →
      let q := fromMontQuotient st
      let out := shiftRound (macSparseModulus st.toMulAcc (fieldWordsOfLimbs modulus) q)
      res = FromMontState.ofMulAcc out ∧ res.Bounded ∧ out.value < wordMod^9 ∧
        wordMod * out.value = st.toMulAcc.value + modulus.toNat * q := by
  intro res hres
-- END fromMontRound_spec statement
  -- generated skeleton for `fromMontRound`: do not edit between the annotations
  unfold fromMontRound at hres
  lift_lets -merge at hres
  -- quotientResult: mechanically factored PCs 0 through 2
  word_step quotientResult := fromMontRoundPrefix st.w0
  -- q: quotient prefix output
  word_step q := quotientResult.quotient
  -- carry: quotient prefix output
  word_step carry := quotientResult.carry
  -- suffix: mechanically factored PCs 3 through 20
  word_step suffix := fromMontRoundSuffix modulus st q carry
  -- w0: sparse suffix output
  word_step w0 := suffix.w0
  -- w1: sparse suffix output
  word_step w1 := suffix.w1
  -- w2: sparse suffix output
  word_step w2 := suffix.w2
  -- w3: sparse suffix output
  word_step w3 := suffix.w3
  -- w4: sparse suffix output
  word_step w4 := suffix.w4
  -- w5: sparse suffix output
  word_step w5 := suffix.w5
  -- w6: sparse suffix output
  word_step w6 := suffix.w6
  -- w7: sparse suffix output
  word_step w7 := suffix.w7
  -- w8: sparse suffix output
  word_step w8 := suffix.w8
  subst hres
  -- BEGIN fromMontRound_spec conclusion
  have hprefix := fromMontRoundPrefix_spec st hst quotientResult e_quotientResult
  have hqeq : q = fromMontQuotient st := by rw [e_q, hprefix.1]
  have hcarryeq : carry = (adc st.w0 q 0).2 := by
    rw [e_carry, hprefix.2, hqeq]
  have hq : q < wordMod := by
    rw [hqeq]
    exact (fromMontQuotient_spec st hst).1
  have hcarry : carry < wordMod := by
    rw [hcarryeq, hqeq, adc_fromMontQuotient_carry st hst]
    split <;> norm_num [wordMod]
  have hsuffix := fromMontRoundSuffix_spec modulus st q carry hm hst hq hcarry hshape
    hqeq hcarryeq suffix e_suffix
  have hpure := fromMontSparseRound_spec modulus st hm hshape hlow hst
  let out := shiftRound
    (macSparseModulus st.toMulAcc (fieldWordsOfLimbs modulus) (fromMontQuotient st))
  have hwords : (⟨w0, w1, w2, w3, w4, w5, w6, w7, w8⟩ : FromMontState) = suffix := by
    cases suffix
    simp_all
  have hresult : (⟨w0, w1, w2, w3, w4, w5, w6, w7, w8⟩ : FromMontState) =
      FromMontState.ofMulAcc out := by
    rw [hwords]
    simpa only [out, hqeq] using hsuffix.2
  dsimp only
  refine ⟨hresult, ?_, hpure.2.2.2.1, ?_⟩
  · rw [hresult]
    exact FromMontState.ofMulAcc_bounded out hpure.2.1
  · simpa only [fieldWordsOfLimbs_value modulus hm] using hpure.2.2.2.2
  -- END fromMontRound_spec conclusion

end PastaCurves.ARMv7EM
