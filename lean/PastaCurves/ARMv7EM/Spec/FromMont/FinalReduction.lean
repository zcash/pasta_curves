import PastaCurves.ARMv7EM.Spec.Invert.CondSub.Arithmetic
import PastaCurves.ARMv7EM.Spec.Mul.Reduction

/-!
# Final reduction for the ARMv7E-M Montgomery conversion

The generated `fromMontBlock` epilogue subtracts the eight sparse modulus words, turns the
resulting no-borrow carry into a mask with `sbc high, high`, and adds the masked modulus back.
This module isolates that arithmetic bridge from the generated instruction trace.
-/

namespace PastaCurves.ARMv7EM

open ListArithmetic

/-- Apply the borrow mask to every modulus word before the add-back chain. -/
def maskedModulus (modulus : List Nat) (mask : Nat) : List Nat :=
  modulus.map fun word => andw word mask

/-- The arithmetic model of the final subtraction and masked add-back in `fromMontBlock`.
The returned carry is the carry discarded by the final `adc`. -/
def fromMontFinalReduction (candidate modulus : List Nat) (high : Nat) : List Nat × Nat :=
  let difference := subcChain 1 candidate modulus
  let mask := (subc32 high high difference.2).1
  adcChain 0 difference.1 (maskedModulus modulus mask)

/-- The eight-word shape of a Pasta modulus used by the generated sparse epilogue. -/
def pastaSparseWords (a b c : Nat) : List Nat :=
  [1, a, b, c, 0, 0, 0, 2^30]

/-- With a zero ninth word, `sbc high, high` is all ones exactly when subtraction borrowed. -/
theorem fromMontBorrowMask (high finalCarry : Nat) (hhigh : high = 0)
    (hcarry : finalCarry ≤ 1) :
    (subc32 high high finalCarry).1 =
      if finalCarry = 0 then wordMod - 1 else 0 := by
  subst high
  exact subc32_zero_mask finalCarry hcarry

private theorem maskedModulus_max (modulus : List Nat) (hmodulus : WordsBounded modulus) :
    maskedModulus modulus (wordMod - 1) = modulus := by
  induction modulus with
  | nil => rfl
  | cons word modulus ih =>
      have hword : word < wordMod := hmodulus word (by simp)
      have htail : WordsBounded modulus := by
        intro x hx
        exact hmodulus x (by simp [hx])
      simp only [maskedModulus, List.map_cons, List.cons.injEq]
      constructor
      · unfold andw
        rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hword]
      · exact ih htail

private theorem maskedModulus_zero (modulus : List Nat) :
    maskedModulus modulus 0 = List.replicate modulus.length 0 := by
  induction modulus with
  | nil => rfl
  | cons word modulus ih =>
      rw [show (word :: modulus).length = modulus.length + 1 by simp,
        List.replicate_succ, ← ih]
      simp [maskedModulus, andw]

private theorem wordsValue_injective (xs ys : List Nat) (hlen : xs.length = ys.length)
    (hxs : WordsBounded xs) (hys : WordsBounded ys)
    (hvalue : wordsValue xs = wordsValue ys) : xs = ys := by
  induction xs generalizing ys with
  | nil => simpa using hlen
  | cons x xs ih =>
      cases ys with
      | nil => simp at hlen
      | cons y ys =>
          have hx : x < wordMod := hxs x (by simp)
          have hy : y < wordMod := hys y (by simp)
          have hxy : x = y := by
            have hmod := congrArg (fun value => value % wordMod) hvalue
            simpa [wordsValue, Nat.add_mod, Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt hy] using hmod
          subst y
          have htailValue : wordsValue xs = wordsValue ys := by
            simp only [wordsValue] at hvalue
            have hmul : wordMod * wordsValue xs = wordMod * wordsValue ys :=
              Nat.add_left_cancel hvalue
            norm_num [wordMod] at hmul ⊢
            omega
          have htailLength : xs.length = ys.length := by simpa using hlen
          have hxsTail : WordsBounded xs := by
            intro z hz
            exact hxs z (by simp [hz])
          have hysTail : WordsBounded ys := by
            intro z hz
            exact hys z (by simp [hz])
          rw [ih ys htailLength hxsTail hysTail htailValue]

private theorem selectReduced_size (candidate modulus : Array Nat) :
    (selectReduced candidate modulus 0).size = 8 := by
  unfold selectReduced
  dsimp only
  rw [show List.range words = [0, 1, 2, 3, 4, 5, 6, 7] by decide]
  simp [words]

/-- For bounded eight-word inputs and a zero ninth word, the generated subtraction-mask-addition
schedule returns exactly the same eight words as the shared conditional reduction. -/
theorem fromMontFinalReduction_eq_selectReduced (candidate modulus : List Nat) (high : Nat)
    (hcandidateLength : candidate.length = 8) (hmodulusLength : modulus.length = 8)
    (hcandidate : WordsBounded candidate) (hmodulus : WordsBounded modulus)
    (hhigh : high = 0) :
    (fromMontFinalReduction candidate modulus high).1.toArray =
      selectReduced candidate.toArray modulus.toArray 0 := by
  let difference := subcChain 1 candidate modulus
  let restored := fromMontFinalReduction candidate modulus high
  have hchain : subcChain 1 candidate modulus = (difference.1, difference.2) := rfl
  have hsub := subcChain_eight_spec candidate modulus difference.1 difference.2
    hcandidateLength hmodulusLength hcandidate hmodulus hchain
  have hdifferenceLength : difference.1.length = 8 := hsub.2.1
  have hdifferenceLt : wordsValue difference.1 < 2^256 := by
    have hlt := wordsValue_lt difference.1 hsub.1
    rw [hdifferenceLength] at hlt
    norm_num [wordMod] at hlt ⊢
    exact hlt
  have hcandidateLt : wordsValue candidate < 2^256 := by
    have hlt := wordsValue_lt candidate hcandidate
    rw [hcandidateLength] at hlt
    norm_num [wordMod] at hlt ⊢
    exact hlt
  have hrestored : WordsBounded restored.1 ∧ restored.1.length = 8 ∧
      wordsValue restored.1 =
        if wordsValue candidate < wordsValue modulus then
          wordsValue candidate
        else
          wordsValue candidate - wordsValue modulus := by
    obtain hcarryZero | hcarryOne : difference.2 = 0 ∨ difference.2 = 1 := by
      omega
    · have hmask : (subc32 high high difference.2).1 = wordMod - 1 := by
        rw [fromMontBorrowMask high difference.2 hhigh hsub.2.2.1, if_pos hcarryZero]
      have haddend :
          maskedModulus modulus (subc32 high high difference.2).1 = modulus := by
        rw [hmask]
        exact maskedModulus_max modulus hmodulus
      have hadd := adcChain_spec difference.1 modulus 0
        (by omega) hsub.1 hmodulus (by omega)
      dsimp only at hadd
      have hrestoredDef : restored = adcChain 0 difference.1
          (maskedModulus modulus (subc32 high high difference.2).1) := by
        simp only [restored, fromMontFinalReduction, hchain]
      rw [haddend] at hrestoredDef
      rw [← hrestoredDef] at hadd
      change WordsBounded restored.1 ∧ restored.1.length = difference.1.length ∧
        restored.2 ≤ 1 ∧
          wordsValue restored.1 + wordMod ^ difference.1.length * restored.2 =
            wordsValue difference.1 + wordsValue modulus + 0 at hadd
      have hsubValue : wordsValue difference.1 + wordsValue modulus =
          wordsValue candidate + 2^256 := by
        simpa [hcarryZero] using hsub.2.2.2
      have hrestoredValue : wordsValue restored.1 + 2^256 * restored.2 =
          wordsValue difference.1 + wordsValue modulus := by
        simpa [hdifferenceLength, wordMod] using hadd.2.2.2
      have hrestoredLt : wordsValue restored.1 < 2^256 := by
        have hlt := wordsValue_lt restored.1 hadd.1
        rw [hadd.2.1, hdifferenceLength] at hlt
        norm_num [wordMod] at hlt ⊢
        exact hlt
      have hcandidateBelow : wordsValue candidate < wordsValue modulus := by omega
      have hvalue : wordsValue restored.1 = wordsValue candidate := by omega
      exact ⟨hadd.1, by omega, by simp [hcandidateBelow, hvalue]⟩
    · have hmask : (subc32 high high difference.2).1 = 0 := by
        rw [fromMontBorrowMask high difference.2 hhigh hsub.2.2.1, if_neg]
        omega
      have haddend :
          maskedModulus modulus (subc32 high high difference.2).1 =
            List.replicate modulus.length 0 := by
        rw [hmask]
        exact maskedModulus_zero modulus
      have hzeros : WordsBounded (List.replicate modulus.length 0) := by
        simp [WordsBounded, wordMod]
      have hadd := adcChain_spec difference.1 (List.replicate modulus.length 0) 0
        (by simp [hdifferenceLength, hmodulusLength]) hsub.1 hzeros (by omega)
      dsimp only at hadd
      have hrestoredDef : restored = adcChain 0 difference.1
          (maskedModulus modulus (subc32 high high difference.2).1) := by
        simp only [restored, fromMontFinalReduction, hchain]
      rw [haddend] at hrestoredDef
      rw [← hrestoredDef] at hadd
      change WordsBounded restored.1 ∧ restored.1.length = difference.1.length ∧
        restored.2 ≤ 1 ∧
          wordsValue restored.1 + wordMod ^ difference.1.length * restored.2 =
            wordsValue difference.1 +
              wordsValue (List.replicate modulus.length 0) + 0 at hadd
      have hsubValue : wordsValue difference.1 + wordsValue modulus =
          wordsValue candidate := by
        simpa [hcarryOne] using hsub.2.2.2
      have hrestoredValue : wordsValue restored.1 + 2^256 * restored.2 =
          wordsValue difference.1 := by
        simpa [hdifferenceLength, hmodulusLength, wordsValue, wordMod] using hadd.2.2.2
      have hrestoredLt : wordsValue restored.1 < 2^256 := by
        have hlt := wordsValue_lt restored.1 hadd.1
        rw [hadd.2.1, hdifferenceLength] at hlt
        norm_num [wordMod] at hlt ⊢
        exact hlt
      have hcandidateNotBelow : ¬ wordsValue candidate < wordsValue modulus := by omega
      have hvalue : wordsValue restored.1 =
          wordsValue candidate - wordsValue modulus := by omega
      exact ⟨hadd.1, by omega, by simp [hcandidateNotBelow, hvalue]⟩
  have hselected := selectReduced_zero_value candidate.toArray modulus.toArray
    (by simpa using hcandidateLength) (by simpa using hmodulusLength)
    (by simpa [ArrayWordsBounded] using hcandidate)
    (by simpa [ArrayWordsBounded] using hmodulus)
  have hselectedLength : (selectReduced candidate.toArray modulus.toArray 0).toList.length = 8 := by
    simpa using selectReduced_size candidate.toArray modulus.toArray
  have hselectedBounded :
      WordsBounded (selectReduced candidate.toArray modulus.toArray 0).toList := by
    simpa [ArrayWordsBounded] using hselected.1
  have hselectedValue :
      wordsValue (selectReduced candidate.toArray modulus.toArray 0).toList =
        if wordsValue candidate < wordsValue modulus then
          wordsValue candidate
        else
          wordsValue candidate - wordsValue modulus := by
    simpa [arrayValue] using hselected.2
  rw [← Array.toList_inj]
  exact wordsValue_injective restored.1
    (selectReduced candidate.toArray modulus.toArray 0).toList
    (by omega) hrestored.1 hselectedBounded (by rw [hrestored.2.2, hselectedValue])

/-- For a candidate below twice a sparse Pasta modulus, the exact final epilogue output is bounded,
canonical, and congruent to the candidate. -/
theorem fromMontFinalReduction_sparse_spec (candidate : List Nat) (a b c high : Nat)
    (hcandidateLength : candidate.length = 8) (hcandidate : WordsBounded candidate)
    (ha : a < wordMod) (hb : b < wordMod) (hc : c < wordMod) (hhigh : high = 0)
    (hcandidateLt : wordsValue candidate < 2 * wordsValue (pastaSparseWords a b c)) :
    let output := (fromMontFinalReduction candidate (pastaSparseWords a b c) high).1
    output.toArray = selectReduced candidate.toArray (pastaSparseWords a b c).toArray 0 ∧
      WordsBounded output ∧
      wordsValue output < wordsValue (pastaSparseWords a b c) ∧
      wordsValue output ≡ wordsValue candidate [MOD wordsValue (pastaSparseWords a b c)] := by
  dsimp only
  have hmodulusLength : (pastaSparseWords a b c).length = 8 := by
    simp [pastaSparseWords]
  have hmodulus : WordsBounded (pastaSparseWords a b c) := by
    simp only [pastaSparseWords, WordsBounded, List.mem_cons, forall_eq_or_imp]
    norm_num [wordMod] at ha hb hc ⊢
    exact ⟨ha, hb, hc⟩
  have heq := fromMontFinalReduction_eq_selectReduced candidate
    (pastaSparseWords a b c) high hcandidateLength hmodulusLength hcandidate hmodulus hhigh
  have hselected := selectReduced_zero_value candidate.toArray
    (pastaSparseWords a b c).toArray (by simpa using hcandidateLength)
    (by simpa using hmodulusLength)
    (by simpa [ArrayWordsBounded] using hcandidate)
    (by simpa [ArrayWordsBounded] using hmodulus)
  have hsingle := single_reduction (wordsValue candidate)
    (wordsValue (pastaSparseWords a b c)) hcandidateLt
  have houtputBounded :
      WordsBounded (fromMontFinalReduction candidate (pastaSparseWords a b c) high).1 := by
    have hbounded := hselected.1
    rw [← heq] at hbounded
    simpa [ArrayWordsBounded] using hbounded
  have houtputValue :
      wordsValue (fromMontFinalReduction candidate (pastaSparseWords a b c) high).1 =
        if wordsValue candidate < wordsValue (pastaSparseWords a b c) then
          wordsValue candidate
        else
          wordsValue candidate - wordsValue (pastaSparseWords a b c) := by
    have hvalue := hselected.2
    rw [← heq] at hvalue
    simpa [arrayValue] using hvalue
  rw [← houtputValue] at hsingle
  exact ⟨heq, houtputBounded, hsingle.1, hsingle.2⟩

end PastaCurves.ARMv7EM
