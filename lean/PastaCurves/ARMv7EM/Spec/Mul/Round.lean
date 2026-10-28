import PastaCurves.ARMv7EM.Spec.Mul.Correctness

/-!
# Correctness of one ARMv7E-M multiplication round

This file proves the arithmetic effect of dropping the cancelled low word from the rolling
accumulator, then combines the two multiply-accumulate helper specifications into one equation
for `mulRound`.
-/

namespace PastaCurves.ARMv7EM

/-- Dropping a zero low word divides the accumulator's value exactly by the word radix. The
shifted accumulator remains bounded, has a zero top word, and therefore fits in nine words. -/
theorem shiftRound_spec (acc : MulAcc) (hacc : acc.Bounded) (hlow : acc.w0 = 0) :
    let out := shiftRound acc
    out.Bounded ∧ out.w9 = 0 ∧ out.value < wordMod^9 ∧
      wordMod * out.value = acc.value ∧ out.value = acc.value / wordMod := by
  have houtBounded : (shiftRound acc).Bounded := by
    rcases hacc with ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9⟩
    exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, by simp [shiftRound, wordMod]⟩
  have htop : (shiftRound acc).w9 = 0 := rfl
  have hvalue : wordMod * (shiftRound acc).value = acc.value := by
    simp only [shiftRound, MulAcc.value, ListArithmetic.wordsValue]
    rw [hlow]
    ring
  have hlt : (shiftRound acc).value < wordMod^9 :=
    MulAcc.value_lt_nine (shiftRound acc) houtBounded htop
  refine ⟨houtBounded, htop, hlt, hvalue, ?_⟩
  rw [← hvalue]
  simp [wordMod]

/-- Adding the Montgomery quotient times a modulus whose low word is one cancels the low word. -/
theorem macSparseModulus_low_eq_zero (acc : MulAcc) (modulus : FieldWords) (inv : Nat)
    (hshape : modulus.w0 = 1) (hinv : (inv * modulus.w0 + 1) % wordMod = 0) :
    (macSparseModulus acc modulus (acc.w0 * inv % wordMod)).w0 = 0 := by
  have hcancel := cancel_low_word acc.w0 inv modulus.w0 hinv
  rw [hshape] at hcancel
  simpa [macSparseModulus, adc, addc32, wordMod, Nat.mul_comm] using hcancel

/-- One rolling CIOS round has the expected pure value equation. The strict total bound is the
exact numerical premise needed to rule out the carry discarded by `macSparseModulus`; the product
helper's discarded carry is ruled out by the zero input top word and the remaining word bounds. -/
theorem mulRound_spec (lhs modulus : FieldWords) (inv b : Nat) (acc : MulAcc)
    (hacc : acc.Bounded) (haccTop : acc.w9 = 0)
    (hlhs : lhs.Bounded) (hb : b < wordMod)
    (hmodulus : modulus.Bounded)
    (hshape : modulus.w0 = 1 ∧ modulus.w4 = 0 ∧ modulus.w5 = 0 ∧
      modulus.w6 = 0 ∧ modulus.w7 = 2^30)
    (hinv : (inv * modulus.w0 + 1) % wordMod = 0)
    (htotal : acc.value + lhs.value * b + modulus.value *
      ((macProduct acc lhs b).w0 * inv % wordMod) < wordMod^10) :
    let q := (macProduct acc lhs b).w0 * inv % wordMod
    let out := mulRound lhs modulus inv b acc
    out.Bounded ∧ out.w9 = 0 ∧ out.value < wordMod^9 ∧
      wordMod * out.value = acc.value + lhs.value * b + modulus.value * q ∧
      out.value = (acc.value + lhs.value * b + modulus.value * q) / wordMod := by
  let product := macProduct acc lhs b
  let q := product.w0 * inv % wordMod
  let reduced := macSparseModulus product modulus q
  obtain ⟨hproductBounded, hproductValue⟩ :=
    macProduct_spec acc lhs b hacc haccTop hlhs hb
  have hq : q < wordMod := by
    exact Nat.mod_lt _ (by norm_num [wordMod])
  have hsparseTotal : product.value + modulus.value * q < wordMod^10 := by
    rw [hproductValue]
    exact htotal
  obtain ⟨_, _, hreducedBounded, hreducedValue, _⟩ :=
    macSparseModulus_spec product modulus q hproductBounded hmodulus hq hshape hsparseTotal
  have hreducedLow : reduced.w0 = 0 := by
    dsimp only [reduced, q, product]
    exact macSparseModulus_low_eq_zero (macProduct acc lhs b) modulus inv hshape.1 hinv
  obtain ⟨houtBounded, houtTop, houtLt, houtValue, houtDiv⟩ :=
    shiftRound_spec reduced hreducedBounded hreducedLow
  change (shiftRound reduced).Bounded ∧ (shiftRound reduced).w9 = 0 ∧
    (shiftRound reduced).value < wordMod^9 ∧
    wordMod * (shiftRound reduced).value =
      acc.value + lhs.value * b + modulus.value * q ∧
    (shiftRound reduced).value =
      (acc.value + lhs.value * b + modulus.value * q) / wordMod
  refine ⟨houtBounded, houtTop, houtLt, ?_, ?_⟩
  · rw [houtValue, hreducedValue, hproductValue]
  · rw [houtDiv, hreducedValue, hproductValue]

end PastaCurves.ARMv7EM
