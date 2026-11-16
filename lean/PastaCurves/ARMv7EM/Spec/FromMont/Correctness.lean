import PastaCurves.ARMv7EM.Spec.FromMont.RoundComposition
import PastaCurves.ARMv7EM.Spec.Mul.EightRounds
import PastaCurves.ARMv7EM.Spec.Mul.Reduction

/-!
# Output helpers for ARMv7E-M Montgomery conversion

These lemmas telescope the eight checked conversion rounds and connect bounded arrays of eight
32-bit words to the four-limb result type. The generated caller proof uses them to keep its
instruction skeleton compact.
-/

namespace PastaCurves.ARMv7EM

private theorem arrayValue_fieldWords_toArray (x : FieldWords) :
    arrayValue x.toArray = x.value := by
  simp [arrayValue, FieldWords.toArray, FieldWords.value]

private theorem foldl_setIfInBounds_size (indices : List Nat) (initial : Array Nat)
    (value : Array Nat → Nat → Nat) :
    (indices.foldl (fun result i => result.setIfInBounds i (value result i)) initial).size =
      initial.size := by
  induction indices generalizing initial with
  | nil => rfl
  | cons i indices ih =>
      rw [List.foldl_cons, ih]
      simp

private theorem selectReduced_size (candidate modulus : Array Nat) (highNonzero : Nat) :
    (selectReduced candidate modulus highNonzero).size = 8 := by
  unfold selectReduced
  dsimp only
  rw [foldl_setIfInBounds_size]
  simp [words]

private theorem fromWords_toNat (xs : List Nat) (hlen : xs.length = 8) :
    (ListArithmetic.fromWords xs).toNat = ListArithmetic.wordsValue xs := by
  rcases xs with _ | ⟨w0, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨w1, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨w2, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨w3, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨w4, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨w5, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨w6, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨w7, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨w8, xs⟩ <;> simp at hlen
  exact ListArithmetic.fromWords_value w0 w1 w2 w3 w4 w5 w6 w7

private theorem limbsOfWordsArray_spec (xs : Array Nat) (hsize : xs.size = 8)
    (hbounded : ArrayWordsBounded xs) :
    (limbsOfWordsArray xs).Bounded ∧ (limbsOfWordsArray xs).toNat = arrayValue xs := by
  have hlength : xs.toList.length = 8 := by simpa using hsize
  rw [limbsOfWordsArray_toArray xs.toList hlength]
  exact ⟨fromWords_bounded xs.toList hlength hbounded,
    fromWords_toNat xs.toList hlength⟩

/-- Projecting an accumulator whose tenth word is zero into the generated conversion state
preserves its complete value. -/
theorem fromMont_ofMulAcc_value (acc : MulAcc) (htop : acc.w9 = 0) :
    (FromMontState.ofMulAcc acc).toMulAcc.value = acc.value := by
  simp only [FromMontState.ofMulAcc, FromMontState.toMulAcc, MulAcc.value,
    ListArithmetic.wordsValue, htop]

/-- The compiled round theorem, rewritten as an equation between the concrete input and output
states carried by the generated outer caller. -/
theorem fromMontRound_value_spec (modulus : Limbs) (st out : FromMontState)
    (hm : modulus.Bounded) (hst : st.Bounded)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hlow : modulus.l0 % wordMod = 1) (hout : out = fromMontRound modulus st) :
    out.Bounded ∧ fromMontQuotient st < wordMod ∧
      wordMod * out.toMulAcc.value =
        st.toMulAcc.value + modulus.toNat * fromMontQuotient st := by
  obtain ⟨houtEq, houtBounded, _hpureLt, hpureValue⟩ :=
    fromMontRound_spec modulus st hm hst hshape hlow out hout
  let pureOut := shiftRound
    (macSparseModulus st.toMulAcc (fieldWordsOfLimbs modulus) (fromMontQuotient st))
  have hpureTop : pureOut.w9 = 0 := rfl
  refine ⟨houtBounded, (fromMontQuotient_spec st hst).1, ?_⟩
  rw [houtEq, fromMont_ofMulAcc_value pureOut hpureTop]
  exact hpureValue

set_option maxHeartbeats 1000000 in
/-- Eight checked conversion rounds telescope to the Montgomery invariant and put the concrete
candidate below twice the sparse Pasta modulus, with a zero ninth word. -/
theorem fromMont_eight_rounds_spec (value modulus : Limbs)
    (states : Nat → FromMontState) (quotient : Nat → Nat)
    (hv : value.Bounded) (hm : modulus.Bounded)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinitialValue : (states 0).toMulAcc.value = value.toNat)
    (hsteps : ∀ i < 8,
      wordMod * (states (i + 1)).toMulAcc.value =
        (states i).toMulAcc.value + modulus.toNat * quotient i)
    (hquotient : ∀ i < 8, quotient i < wordMod) :
    let candidate := states 8
    candidate.w8 = 0 ∧
      ListArithmetic.wordsValue
          [candidate.w0, candidate.w1, candidate.w2, candidate.w3,
            candidate.w4, candidate.w5, candidate.w6, candidate.w7] <
        2 * modulus.toNat ∧
      wordMod^8 * candidate.toMulAcc.value ≡ value.toNat [MOD modulus.toNat] := by
  dsimp only
  have hstep : ∀ i < 8,
      wordMod * (states (i + 1)).toMulAcc.value =
        (states i).toMulAcc.value + 0 * 0 + modulus.toNat * quotient i := by
    intro i hi
    simpa only [Nat.mul_zero, Nat.add_zero] using hsteps i hi
  have hmain := cios_eight_round_invariant 0 modulus.toNat (fun _ => 0) quotient
    (fun i => (states i).toMulAcc.value) hstep
  simp only [hinitialValue, Nat.zero_mul, Nat.add_zero] at hmain
  have hq : prefixValue quotient 8 < wordMod^8 := prefixValue_eight_lt quotient hquotient
  have hvalue := Limbs.toNat_lt value hv
  have hmodulusPos : 0 < modulus.toNat := by
    simp only [Limbs.toNat, hshape.2]
    omega
  have hvalueProduct : value.toNat < wordMod^8 * modulus.toNat := by
    have hmodulusOne : 1 ≤ modulus.toNat := hmodulusPos
    have hradixLe : wordMod^8 ≤ wordMod^8 * modulus.toNat := by
      simpa only [Nat.mul_one] using Nat.mul_le_mul_left (wordMod^8) hmodulusOne
    exact lt_of_lt_of_le (by simpa [wordMod] using hvalue) hradixLe
  have hcandidate : (states 8).toMulAcc.value < 2 * modulus.toNat :=
    cios_candidate_lt_two_mul (wordMod^8) (states 8).toMulAcc.value value.toNat
      (prefixValue quotient 8) modulus.toNat (by norm_num [wordMod]) hmodulusPos hmain
      hvalueProduct hq
  have hmodulusUpper := Limbs.toNat_lt_of_shape modulus hm hshape
  have hcandidateRadix : (states 8).toMulAcc.value < wordMod^8 := by
    norm_num [wordMod] at hmodulusUpper ⊢
    omega
  have hhigh : (states 8).w8 = 0 := by
    apply MulAcc.w8_eq_zero_of_value_lt (states 8).toMulAcc hcandidateRadix
  have hcandidateValue :
      ListArithmetic.wordsValue
          [(states 8).w0, (states 8).w1, (states 8).w2, (states 8).w3,
            (states 8).w4, (states 8).w5, (states 8).w6, (states 8).w7] =
        (states 8).toMulAcc.value := by
    simp only [FromMontState.toMulAcc, MulAcc.value, ListArithmetic.wordsValue, hhigh]
    ring
  refine ⟨hhigh, ?_, ?_⟩
  · rw [hcandidateValue]
    exact hcandidate
  · apply modEq_of_add_mul _ _ 0 (prefixValue quotient 8) modulus.toNat
    simpa [Nat.mul_comm] using hmain

/-- Eight explicit one-step equations telescope without constructing indexed state functions. -/
theorem fromMont_telescope_eight (a0 a1 a2 a3 a4 a5 a6 a7 a8 p q0 q1 q2 q3 q4 q5 q6 q7 : Nat)
    (h0 : wordMod * a1 = a0 + p * q0)
    (h1 : wordMod * a2 = a1 + p * q1)
    (h2 : wordMod * a3 = a2 + p * q2)
    (h3 : wordMod * a4 = a3 + p * q3)
    (h4 : wordMod * a5 = a4 + p * q4)
    (h5 : wordMod * a6 = a5 + p * q5)
    (h6 : wordMod * a7 = a6 + p * q6)
    (h7 : wordMod * a8 = a7 + p * q7) :
    wordMod^8 * a8 = a0 + p *
      (q0 + q1 * wordMod + q2 * wordMod^2 + q3 * wordMod^3 +
        q4 * wordMod^4 + q5 * wordMod^5 + q6 * wordMod^6 + q7 * wordMod^7) := by
  let acc : Nat → Nat
    | 0 => a0
    | 1 => a1
    | 2 => a2
    | 3 => a3
    | 4 => a4
    | 5 => a5
    | 6 => a6
    | 7 => a7
    | _ => a8
  let q : Nat → Nat
    | 0 => q0
    | 1 => q1
    | 2 => q2
    | 3 => q3
    | 4 => q4
    | 5 => q5
    | 6 => q6
    | _ => q7
  have hstep : ∀ i < 8, wordMod * acc (i + 1) = acc i + 0 * 0 + p * q i := by
    intro i hi
    interval_cases i <;> simp only [acc, q, Nat.mul_zero, Nat.add_zero] <;> assumption
  have hmain := cios_eight_round_invariant 0 p (fun _ => 0) q acc hstep
  simpa only [acc, q, prefixValue, Nat.zero_mul, Nat.add_zero, pow_zero, pow_one,
    Nat.mul_one, Nat.zero_add] using hmain

set_option maxHeartbeats 1000000 in
/-- The explicit eight checked equations used by the generated caller imply the final candidate
facts, without making the caller elaborate a dependent finite dispatch. -/
theorem fromMont_eight_explicit_spec (value modulus : Limbs)
    (s0 s1 s2 s3 s4 s5 s6 s7 s8 : FromMontState)
    (q0 q1 q2 q3 q4 q5 q6 q7 : Nat)
    (hv : value.Bounded) (hm : modulus.Bounded)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinitial : s0.toMulAcc.value = value.toNat)
    (h0 : wordMod * s1.toMulAcc.value = s0.toMulAcc.value + modulus.toNat * q0)
    (h1 : wordMod * s2.toMulAcc.value = s1.toMulAcc.value + modulus.toNat * q1)
    (h2 : wordMod * s3.toMulAcc.value = s2.toMulAcc.value + modulus.toNat * q2)
    (h3 : wordMod * s4.toMulAcc.value = s3.toMulAcc.value + modulus.toNat * q3)
    (h4 : wordMod * s5.toMulAcc.value = s4.toMulAcc.value + modulus.toNat * q4)
    (h5 : wordMod * s6.toMulAcc.value = s5.toMulAcc.value + modulus.toNat * q5)
    (h6 : wordMod * s7.toMulAcc.value = s6.toMulAcc.value + modulus.toNat * q6)
    (h7 : wordMod * s8.toMulAcc.value = s7.toMulAcc.value + modulus.toNat * q7)
    (bq0 : q0 < wordMod) (bq1 : q1 < wordMod) (bq2 : q2 < wordMod)
    (bq3 : q3 < wordMod) (bq4 : q4 < wordMod) (bq5 : q5 < wordMod)
    (bq6 : q6 < wordMod) (bq7 : q7 < wordMod) :
    s8.w8 = 0 ∧
      ListArithmetic.wordsValue
          [s8.w0, s8.w1, s8.w2, s8.w3, s8.w4, s8.w5, s8.w6, s8.w7] <
        2 * modulus.toNat ∧
      wordMod^8 * s8.toMulAcc.value ≡ value.toNat [MOD modulus.toNat] := by
  let quotient := q0 + q1 * wordMod + q2 * wordMod^2 + q3 * wordMod^3 +
    q4 * wordMod^4 + q5 * wordMod^5 + q6 * wordMod^6 + q7 * wordMod^7
  have hmain : wordMod^8 * s8.toMulAcc.value = value.toNat + modulus.toNat * quotient := by
    rw [← hinitial]
    exact fromMont_telescope_eight s0.toMulAcc.value s1.toMulAcc.value s2.toMulAcc.value
      s3.toMulAcc.value s4.toMulAcc.value s5.toMulAcc.value s6.toMulAcc.value
      s7.toMulAcc.value s8.toMulAcc.value modulus.toNat q0 q1 q2 q3 q4 q5 q6 q7
      h0 h1 h2 h3 h4 h5 h6 h7
  have hquotient : quotient < wordMod^8 := by
    dsimp only [quotient]
    norm_num [wordMod] at bq0 bq1 bq2 bq3 bq4 bq5 bq6 bq7 ⊢
    omega
  have hvalue := Limbs.toNat_lt value hv
  have hmodulusPos : 0 < modulus.toNat := by
    simp only [Limbs.toNat, hshape.2]
    omega
  have hvalueProduct : value.toNat < wordMod^8 * modulus.toNat := by
    have hmodulusOne : 1 ≤ modulus.toNat := hmodulusPos
    have hradixLe : wordMod^8 ≤ wordMod^8 * modulus.toNat := by
      simpa only [Nat.mul_one] using Nat.mul_le_mul_left (wordMod^8) hmodulusOne
    exact lt_of_lt_of_le (by simpa [wordMod] using hvalue) hradixLe
  have hcandidate : s8.toMulAcc.value < 2 * modulus.toNat :=
    cios_candidate_lt_two_mul (wordMod^8) s8.toMulAcc.value value.toNat quotient
      modulus.toNat (by norm_num [wordMod]) hmodulusPos hmain hvalueProduct hquotient
  have hmodulusUpper := Limbs.toNat_lt_of_shape modulus hm hshape
  have hcandidateRadix : s8.toMulAcc.value < wordMod^8 := by
    norm_num [wordMod] at hmodulusUpper ⊢
    omega
  have hhigh : s8.w8 = 0 := by
    apply MulAcc.w8_eq_zero_of_value_lt s8.toMulAcc hcandidateRadix
  have hcandidateValue :
      ListArithmetic.wordsValue [s8.w0, s8.w1, s8.w2, s8.w3, s8.w4, s8.w5, s8.w6, s8.w7] =
        s8.toMulAcc.value := by
    simp only [FromMontState.toMulAcc, MulAcc.value, ListArithmetic.wordsValue, hhigh]
    ring
  refine ⟨hhigh, ?_, ?_⟩
  · rw [hcandidateValue]
    exact hcandidate
  · apply modEq_of_add_mul _ _ 0 quotient modulus.toNat
    simpa [Nat.mul_comm] using hmain

/-- Joining the eight bounded output words returned by final reduction produces the four bounded
limbs expected by the outer block, with the same integer value. -/
theorem fromMont_output_spec (st : FromMontState) (hst : st.Bounded) :
    let output := [st.w0, st.w1, st.w2, st.w3, st.w4, st.w5, st.w6, st.w7]
    let res : Limbs :=
      ⟨st.w0 + wordMod * st.w1, st.w2 + wordMod * st.w3,
        st.w4 + wordMod * st.w5, st.w6 + wordMod * st.w7⟩
    res.Bounded ∧ res.toNat = ListArithmetic.wordsValue output := by
  dsimp only
  constructor
  · rcases hst with ⟨h0, h1, h2, h3, h4, h5, h6, h7, _⟩
    constructor
    · norm_num [wordMod] at h0 h1 ⊢
      omega
    constructor
    · norm_num [wordMod] at h2 h3 ⊢
      omega
    constructor
    · norm_num [wordMod] at h4 h5 ⊢
      omega
    · norm_num [wordMod] at h6 h7 ⊢
      omega
  · simp only [Limbs.toNat, ListArithmetic.wordsValue]
    ring

end PastaCurves.ARMv7EM
