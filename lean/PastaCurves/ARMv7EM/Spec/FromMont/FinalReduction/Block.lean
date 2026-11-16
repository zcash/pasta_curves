/-
Copyright (c) 2026 the pasta_curves contributors.
-/
import PastaCurves.ARMv7EM.Spec.FromMont.FinalReduction.Subtraction
import PastaCurves.ARMv7EM.Spec.FromMont.FinalReduction.AddBack

/-!
# Checked final reduction caller for ARMv7E-M Montgomery conversion

The generated caller composes the checked subtraction-mask and masked-add-back helpers. Its first
eight result words are exactly the pure final-reduction list, while its ninth word is unchanged.
-/

namespace PastaCurves.ARMv7EM

open ListArithmetic

-- BEGIN fromMontFinalReductionBlock_spec statement
/-- The generated final-reduction caller is exactly the pure subtraction and masked add-back. -/
theorem fromMontFinalReductionBlock_spec (modulus : Limbs) (st : FromMontState)
    (hm : modulus.Bounded) (hst : st.Bounded) :
    ∀ res, res = fromMontFinalReductionBlock modulus st →
      let candidate := [st.w0, st.w1, st.w2, st.w3, st.w4, st.w5, st.w6, st.w7]
      let modulusWords :=
        pastaSparseWords (modulus.l0 / wordMod) (modulus.l1 % wordMod)
          (modulus.l1 / wordMod)
      [res.w0, res.w1, res.w2, res.w3, res.w4, res.w5, res.w6, res.w7] =
          (fromMontFinalReduction candidate modulusWords st.w8).1 ∧
        res.w8 = st.w8 ∧ res.Bounded := by
  intro res hres
-- END fromMontFinalReductionBlock_spec statement
  -- generated skeleton for `fromMontFinalReductionBlock`: do not edit between the annotations
  unfold fromMontFinalReductionBlock at hres
  lift_lets -merge at hres
  -- subtraction: mechanically factored PCs 0 through 13
  word_step subtraction := fromMontFinalSubtractionBlock modulus st
  -- addback: mechanically factored PCs 14 through 30
  word_step addback := fromMontFinalAddBackBlock modulus subtraction
  subst hres
  -- BEGIN fromMontFinalReductionBlock_spec conclusion
  have hsub := fromMontFinalSubtractionBlock_spec modulus st hm hst subtraction e_subtraction
  have hadd := fromMontFinalAddBackBlock_spec modulus subtraction hm hsub.1 res e_addback
  obtain ⟨carry, _hcarry, haddChain⟩ := hadd.2.2
  have hsubChain :
      subcChain 1 [st.w0, st.w1, st.w2, st.w3, st.w4, st.w5, st.w6, st.w7]
        (pastaSparseWords (modulus.l0 / wordMod) (modulus.l1 % wordMod)
          (modulus.l1 / wordMod)) =
        ([subtraction.w0, subtraction.w1, subtraction.w2, subtraction.w3,
          subtraction.w4, subtraction.w5, subtraction.w6, subtraction.w7],
          subtraction.carry) := by
    simpa only [pastaSparseWords, show 2^30 = 0x40000000 by norm_num] using hsub.2.1
  have hpure :
      (fromMontFinalReduction
        [st.w0, st.w1, st.w2, st.w3, st.w4, st.w5, st.w6, st.w7]
        (pastaSparseWords (modulus.l0 / wordMod) (modulus.l1 % wordMod)
          (modulus.l1 / wordMod)) st.w8).1 =
        [res.w0, res.w1, res.w2, res.w3, res.w4, res.w5, res.w6, res.w7] := by
    unfold fromMontFinalReduction
    rw [hsubChain]
    dsimp only
    rw [← hsub.2.2.2]
    change
      (adcChain 0
        [subtraction.w0, subtraction.w1, subtraction.w2, subtraction.w3,
          subtraction.w4, subtraction.w5, subtraction.w6, subtraction.w7]
        (maskedModulus
          [1, modulus.l0 / wordMod, modulus.l1 % wordMod, modulus.l1 / wordMod,
            0, 0, 0, 0x40000000]
          subtraction.mask)).1 =
        [res.w0, res.w1, res.w2, res.w3, res.w4, res.w5, res.w6, res.w7]
    exact congrArg Prod.fst haddChain
  dsimp only
  exact ⟨hpure.symm, hadd.2.1.trans hsub.2.2.1, hadd.1⟩
  -- END fromMontFinalReductionBlock_spec conclusion

/-- For the sparse Pasta field shape, the checked caller returns the canonical representative of
its eight-word candidate and preserves the zero ninth word. -/
theorem fromMontFinalReductionBlock_sparse_spec (modulus : Limbs) (st : FromMontState)
    (hm : modulus.Bounded) (hst : st.Bounded)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hlow : modulus.l0 % wordMod = 1) (hhigh : st.w8 = 0)
    (hcandidateLt :
      wordsValue [st.w0, st.w1, st.w2, st.w3, st.w4, st.w5, st.w6, st.w7] <
        2 * modulus.toNat) :
    ∀ res, res = fromMontFinalReductionBlock modulus st →
      let candidate := [st.w0, st.w1, st.w2, st.w3, st.w4, st.w5, st.w6, st.w7]
      let output := [res.w0, res.w1, res.w2, res.w3, res.w4, res.w5, res.w6, res.w7]
      output.toArray = selectReduced candidate.toArray (toWords modulus).toArray 0 ∧
        res.Bounded ∧ wordsValue output < modulus.toNat ∧
        wordsValue output ≡ wordsValue candidate [MOD modulus.toNat] ∧ res.w8 = 0 := by
  intro res hres
  let candidate := [st.w0, st.w1, st.w2, st.w3, st.w4, st.w5, st.w6, st.w7]
  let modulusWords :=
    pastaSparseWords (modulus.l0 / wordMod) (modulus.l1 % wordMod)
      (modulus.l1 / wordMod)
  let output := [res.w0, res.w1, res.w2, res.w3, res.w4, res.w5, res.w6, res.w7]
  have hcandidateLength : candidate.length = 8 := by simp [candidate]
  have hcandidate : WordsBounded candidate := by
    intro x hx
    simp only [candidate, List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact hst.1
    · exact hst.2.1
    · exact hst.2.2.1
    · exact hst.2.2.2.1
    · exact hst.2.2.2.2.1
    · exact hst.2.2.2.2.2.1
    · exact hst.2.2.2.2.2.2.1
    · exact hst.2.2.2.2.2.2.2.1
  have ha : modulus.l0 / wordMod < wordMod := by
    exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq hm.1 (by norm_num [wordMod]))
  have hb : modulus.l1 % wordMod < wordMod := Nat.mod_lt _ (by norm_num [wordMod])
  have hc : modulus.l1 / wordMod < wordMod := by
    exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq hm.2.1 (by norm_num [wordMod]))
  have hmodulusWords : modulusWords = toWords modulus := by
    simp [modulusWords, pastaSparseWords, toWords, hshape.1, hshape.2, hlow, wordMod]
  have hmodulusValue : wordsValue modulusWords = modulus.toNat := by
    rw [hmodulusWords]
    exact wordsValue_toWords modulus hm
  have hblock := fromMontFinalReductionBlock_spec modulus st hm hst res hres
  dsimp only at hblock
  have hpure := fromMontFinalReduction_sparse_spec candidate
    (modulus.l0 / wordMod) (modulus.l1 % wordMod) (modulus.l1 / wordMod) st.w8
    hcandidateLength hcandidate ha hb hc hhigh (by
      rw [hmodulusValue]
      exact hcandidateLt)
  dsimp only at hpure
  have hpure' :
      (fromMontFinalReduction candidate modulusWords st.w8).1.toArray =
          selectReduced candidate.toArray modulusWords.toArray 0 ∧
        WordsBounded (fromMontFinalReduction candidate modulusWords st.w8).1 ∧
        wordsValue (fromMontFinalReduction candidate modulusWords st.w8).1 <
            wordsValue modulusWords ∧
          wordsValue (fromMontFinalReduction candidate modulusWords st.w8).1 ≡
            wordsValue candidate [MOD wordsValue modulusWords] := by
    simpa only [modulusWords] using hpure
  dsimp only
  refine ⟨?_, hblock.2.2, ?_, ?_, hblock.2.1.trans hhigh⟩
  · rw [hblock.1, ← hmodulusWords]
    exact hpure'.1
  · rw [hblock.1, ← hmodulusValue]
    exact hpure'.2.2.1
  · rw [hblock.1, ← hmodulusValue]
    exact hpure'.2.2.2

end PastaCurves.ARMv7EM
