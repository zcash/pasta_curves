import PastaCurves.Fields
import PastaCurves.Glue.Spec
import PastaCurves.ARMv7EM.Compositions.Mul.Value
import PastaCurves.ARMv7EM.Compositions.Mul.Overflow
import PastaCurves.ARMv7EM.Spec.Mul.Reduction
import PastaCurves.ARMv7EM.Spec.Mul.EightRounds

/-!
# ARMv7E-M generated multiplication at the crate's fields

The generated Rust wrapper is proved against the same `mulContract` that its caller asserts.  The
32-bit sparse-modulus implementation additionally relies on the low 32-bit word of the modulus
being one.  `PastaField.shape` does not record that fact, so the generic theorem states it as a
field premise; the Pallas and Vesta corollaries discharge it from their documented constants.
-/

namespace PastaCurves.ARMv7EM.Compositions.Mul.Entry

open _root_.Aeneas _root_.Aeneas.Std
open PastaCurves
open PastaCurves.Glue
open PastaCurves.ARMv7EM
open PastaCurves.ARMv7EM.Compositions.Shared

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
  exact ⟨fromWords_bounded xs.toList hlength hbounded, fromWords_toNat xs.toList hlength⟩

/-- The pure ARMv7E-M Montgomery composition at a field whose low 32-bit modulus word is one.
The operand premise is exactly the generated Rust caller's `mulContract`. -/
theorem mulMont_spec (F : PastaField) (hlow : F.modulus.l0 % wordMod = 1)
    (lhs rhs : Limbs) (hlhs : lhs.Bounded) (hrhs : rhs.Bounded)
    (hcontract : mulContract lhs rhs F.modulus = true) :
    let res := ARMv7EM.mulMont lhs rhs F.modulus F.inv
    res.Bounded ∧ res.toNat < F.modulus.toNat ∧
      R * res.toNat ≡ lhs.toNat * rhs.toNat [MOD F.modulus.toNat] := by
  obtain ⟨hacc, htop, hhigh, hquotient, hmain, hcandidate, hmul⟩ :=
    mulMont_eight_rounds_spec lhs rhs F.modulus F.inv hlhs hrhs F.bounded F.shape hlow
      F.inv_spec hcontract
  let lhsWords := fieldWordsOfLimbs lhs
  let rhsWords := fieldWordsOfLimbs rhs
  let modulusWords := fieldWordsOfLimbs F.modulus
  let wordInv := F.inv % wordMod
  let acc := mulRounds lhsWords rhsWords modulusWords wordInv 8
  let quotient := prefixValue (roundQuotient lhsWords rhsWords modulusWords wordInv) 8
  let candidate : Array Nat :=
    FieldWords.toArray ⟨acc.w0, acc.w1, acc.w2, acc.w3, acc.w4, acc.w5, acc.w6, acc.w7⟩
  let output := selectReduced candidate modulusWords.toArray
    ((acc.w8 ||| ((wordMod - acc.w8) % wordMod)) / 2^31)
  have hcandidateSize : candidate.size = 8 := by simp [candidate, FieldWords.toArray]
  have hmodulusSize : modulusWords.toArray.size = 8 := by simp [FieldWords.toArray]
  have hcandidateBounded : ArrayWordsBounded candidate := by
    have hacc' : acc.Bounded := hacc
    have hwords : acc.w0 < wordMod ∧ acc.w1 < wordMod ∧ acc.w2 < wordMod ∧
        acc.w3 < wordMod ∧ acc.w4 < wordMod ∧ acc.w5 < wordMod ∧
        acc.w6 < wordMod ∧ acc.w7 < wordMod :=
      ⟨hacc'.1, hacc'.2.1, hacc'.2.2.1, hacc'.2.2.2.1, hacc'.2.2.2.2.1,
        hacc'.2.2.2.2.2.1, hacc'.2.2.2.2.2.2.1, hacc'.2.2.2.2.2.2.2.1⟩
    simpa [candidate, ArrayWordsBounded, FieldWords.toArray, ListArithmetic.WordsBounded]
      using hwords
  have hmodulusBounded : ArrayWordsBounded modulusWords.toArray := by
    have hm : modulusWords.Bounded := fieldWordsOfLimbs_bounded F.modulus F.bounded
    simpa [ArrayWordsBounded, FieldWords.toArray, FieldWords.Bounded,
      ListArithmetic.WordsBounded] using hm
  have hcandidateValue : arrayValue candidate + wordMod^8 * acc.w8 = acc.value := by
    have htop' : acc.w9 = 0 := htop
    change ListArithmetic.wordsValue
        [acc.w0, acc.w1, acc.w2, acc.w3, acc.w4, acc.w5, acc.w6, acc.w7] +
      wordMod^8 * acc.w8 =
        ListArithmetic.wordsValue
          [acc.w0, acc.w1, acc.w2, acc.w3, acc.w4, acc.w5, acc.w6, acc.w7, acc.w8, acc.w9]
    rw [htop']
    simp only [ListArithmetic.wordsValue, Nat.mul_zero, Nat.add_zero]
    ring
  have hmodulusValue : arrayValue modulusWords.toArray = F.modulus.toNat := by
    rw [arrayValue_fieldWords_toArray, fieldWordsOfLimbs_value F.modulus F.bounded]
  have hcandidateLt :
      arrayValue candidate + wordMod^8 * acc.w8 <
        2 * arrayValue modulusWords.toArray := by
    rw [hcandidateValue, hmodulusValue]
    exact hcandidate
  have hmodulusRange :
      2^254 ≤ arrayValue modulusWords.toArray ∧ arrayValue modulusWords.toArray < 2^255 := by
    rw [hmodulusValue]
    constructor
    · simp only [Limbs.toNat, F.shape.1, F.shape.2]
      omega
    · exact Limbs.toNat_lt_of_shape F.modulus F.bounded F.shape
  obtain ⟨houtputBounded, _, houtputLt, houtputCongruent⟩ :=
    selectReduced_actual_high_canonical candidate modulusWords.toArray acc.w8
      hcandidateSize hmodulusSize hcandidateBounded hmodulusBounded hacc.2.2.2.2.2.2.2.2.1
      hmodulusRange hcandidateLt
  have houtputSize : output.size = 8 := by
    exact selectReduced_size candidate modulusWords.toArray
      ((acc.w8 ||| ((wordMod - acc.w8) % wordMod)) / 2^31)
  have houtputSpec := limbsOfWordsArray_spec output houtputSize houtputBounded
  have hres : ARMv7EM.mulMont lhs rhs F.modulus F.inv = limbsOfWordsArray output := by
    rw [hmul]
    rfl
  have hroundCongruence :
      wordMod^8 * acc.value ≡ lhs.toNat * rhs.toNat [MOD F.modulus.toNat] := by
    apply modEq_of_add_mul _ _ 0 quotient F.modulus.toNat
    simpa [Nat.mul_comm] using hmain
  have houtputCongruence :
      (limbsOfWordsArray output).toNat ≡ acc.value [MOD F.modulus.toNat] := by
    rw [houtputSpec.2, ← hcandidateValue, ← hmodulusValue]
    simpa only [output] using houtputCongruent
  dsimp only
  rw [hres]
  refine ⟨houtputSpec.1, ?_, ?_⟩
  · rw [houtputSpec.2, ← hmodulusValue]
    exact houtputLt
  · have hradix : wordMod^8 = R := by norm_num [wordMod, R]
    rw [← hradix]
    exact (Nat.ModEq.mul_left (wordMod^8) houtputCongruence).trans hroundCongruence

/-- The generated ARMv7E-M `mul` wrapper at a `PastaField` whose low 32-bit modulus word is one.
Its sole operand condition is the actual Rust `mulContract`: either the left operand is canonical,
or the right operand is canonical and its upper three 64-bit limbs have the asserted headroom. -/
theorem mul_entry_spec (F : PastaField) (hlow : F.modulus.l0 % wordMod = 1)
    (lhs rhs : Std.Array Std.U64 4#usize)
    (hcontract : mulContract (limbsOfArray lhs) (limbsOfArray rhs) F.modulus = true) :
    pasta_curves.asm.armv7em.mul lhs rhs (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res R ((limbsOfArray lhs).toNat * (limbsOfArray rhs).toNat) ⦄ := by
  have hmodulus : limbsOfArray (limbsArray F.modulus) = F.modulus :=
    limbsOfArray_limbsArray F.bounded
  have hinv : (word F.inv).val = F.inv := word_val F.inv_lt
  apply WP.spec_mono (Value.mul_ok_value lhs rhs (limbsArray F.modulus) (word F.inv) (by
    simpa [hmodulus] using hcontract) ?_)
  · intro res hres
    unfold Canonical
    rw [hres, hmodulus, hinv]
    exact (mulMont_spec F hlow (limbsOfArray lhs) (limbsOfArray rhs)
      (limbsOfArray_bounded lhs) (limbsOfArray_bounded rhs) hcontract).2
  · intro lhsWords rhsWords modulusWords _ _ hmodulusWords
    apply Overflow.mulPrerequisites
    · have hwords : Loops.fieldWords modulusWords = fieldWordsOfLimbs F.modulus := by
        have : Loops.natArray modulusWords = wordsOfLimbsArray F.modulus := by
          simpa [Loops.natArray, Conversions.wordValues, hmodulus] using hmodulusWords
        rw [Loops.fieldWords, this]
        simp [wordsOfLimbsArray, fieldWordsOfLimbs, Array.getD]
      rw [hwords]
      exact fieldWordsOfLimbs_sparse_shape F.modulus F.shape hlow
    · have hwords : Loops.fieldWords modulusWords = fieldWordsOfLimbs F.modulus := by
        have : Loops.natArray modulusWords = wordsOfLimbsArray F.modulus := by
          simpa [Loops.natArray, Conversions.wordValues, hmodulus] using hmodulusWords
        rw [Loops.fieldWords, this]
        simp [wordsOfLimbsArray, fieldWordsOfLimbs, Array.getD]
      rw [hwords, UScalar.cast_val_eq]
      simpa [fieldWordsOfLimbs, wordMod, hinv] using
        inverse_relation_word F.inv F.modulus.l0 F.inv_spec

end PastaCurves.ARMv7EM.Compositions.Mul.Entry
