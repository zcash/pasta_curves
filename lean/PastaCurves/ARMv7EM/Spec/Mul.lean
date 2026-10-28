import PastaCurves.ARMv7EM.Spec.Mul.Invariant
import PastaCurves.ARMv7EM.Spec.Mul.Correctness
import PastaCurves.ARMv7EM.Spec.Mul.Round
import PastaCurves.ARMv7EM.Spec.Mul.EightRounds
import PastaCurves.ARMv7EM.Spec.Mul.Reduction

/-!
# ARMv7E-M Montgomery multiplication

The nested modules prove the arithmetic invariant, one-round and eight-round properties, and final
reduction of the pure ARMv7E-M multiplication composition. The operation-level theorems below state
the same generic modulus and inverse contracts as the other architecture specifications.
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

/-- Montgomery multiplication by the pure ARMv7E-M composition under the exact Rust operand
contract. The modulus assumptions describe the sparse Pasta shape but do not select a field. -/
theorem mulMont_spec (lhs rhs modulus : Limbs) (inv : Nat)
    (hlhs : lhs.Bounded) (hrhs : rhs.Bounded) (hm : modulus.Bounded)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hlow : modulus.l0 % wordMod = 1) (_hinv_lt : inv < 2^64)
    (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hcontract : mulContract lhs rhs modulus = true) :
    ∀ res, res = mulMont lhs rhs modulus inv →
      res.Bounded ∧ res.toNat < modulus.toNat ∧
        2^256 * res.toNat ≡ lhs.toNat * rhs.toNat [MOD modulus.toNat] := by
  intro res hres
  obtain ⟨hacc, htop, hhigh, hquotient, hmain, hcandidate, hmul⟩ :=
    mulMont_eight_rounds_spec lhs rhs modulus inv hlhs hrhs hm hshape hlow hinv hcontract
  let lhsWords := fieldWordsOfLimbs lhs
  let rhsWords := fieldWordsOfLimbs rhs
  let modulusWords := fieldWordsOfLimbs modulus
  let wordInv := inv % wordMod
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
    have hm' : modulusWords.Bounded := fieldWordsOfLimbs_bounded modulus hm
    simpa [ArrayWordsBounded, FieldWords.toArray, FieldWords.Bounded,
      ListArithmetic.WordsBounded] using hm'
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
  have hmodulusValue : arrayValue modulusWords.toArray = modulus.toNat := by
    rw [arrayValue_fieldWords_toArray, fieldWordsOfLimbs_value modulus hm]
  have hcandidateLt :
      arrayValue candidate + wordMod^8 * acc.w8 < 2 * arrayValue modulusWords.toArray := by
    rw [hcandidateValue, hmodulusValue]
    exact hcandidate
  have hmodulusRange :
      2^254 ≤ arrayValue modulusWords.toArray ∧ arrayValue modulusWords.toArray < 2^255 := by
    rw [hmodulusValue]
    constructor
    · simp only [Limbs.toNat, hshape.1, hshape.2]
      omega
    · exact Limbs.toNat_lt_of_shape modulus hm hshape
  obtain ⟨houtputBounded, _, houtputLt, houtputCongruent⟩ :=
    selectReduced_actual_high_canonical candidate modulusWords.toArray acc.w8
      hcandidateSize hmodulusSize hcandidateBounded hmodulusBounded hacc.2.2.2.2.2.2.2.2.1
      hmodulusRange hcandidateLt
  have houtputSize : output.size = 8 := by
    exact selectReduced_size candidate modulusWords.toArray
      ((acc.w8 ||| ((wordMod - acc.w8) % wordMod)) / 2^31)
  have houtputSpec := limbsOfWordsArray_spec output houtputSize houtputBounded
  have hresult : mulMont lhs rhs modulus inv = limbsOfWordsArray output := by
    rw [hmul]
    rfl
  have hroundCongruence :
      wordMod^8 * acc.value ≡ lhs.toNat * rhs.toNat [MOD modulus.toNat] := by
    apply modEq_of_add_mul _ _ 0 quotient modulus.toNat
    simpa [Nat.mul_comm] using hmain
  have houtputCongruence :
      (limbsOfWordsArray output).toNat ≡ acc.value [MOD modulus.toNat] := by
    rw [houtputSpec.2, ← hcandidateValue, ← hmodulusValue]
    simpa only [output] using houtputCongruent
  subst res
  rw [hresult]
  refine ⟨houtputSpec.1, ?_, ?_⟩
  · rw [houtputSpec.2, ← hmodulusValue]
    exact houtputLt
  · have hradix : wordMod^8 = 2^256 := by norm_num [wordMod]
    rw [← hradix]
    exact (Nat.ModEq.mul_left (wordMod^8) houtputCongruence).trans hroundCongruence

/-- The contract the crate's callers use: a canonical left operand and any four-limb right
operand. -/
theorem mulMont_spec_of_lhs_lt (lhs rhs modulus : Limbs) (inv : Nat)
    (hlhs : lhs.Bounded) (hrhs : rhs.Bounded) (hm : modulus.Bounded)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hlow : modulus.l0 % wordMod = 1) (hinv_lt : inv < 2^64)
    (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hlt : lhs.toNat < modulus.toNat) :
    ∀ res, res = mulMont lhs rhs modulus inv →
      res.Bounded ∧ res.toNat < modulus.toNat ∧
        2^256 * res.toNat ≡ lhs.toNat * rhs.toNat [MOD modulus.toNat] := by
  have hcontract : mulContract lhs rhs modulus = true :=
    (mulContract_iff lhs rhs modulus hlhs hrhs hm).2 (Or.inl hlt)
  exact mulMont_spec lhs rhs modulus inv hlhs hrhs hm hshape hlow hinv_lt hinv hcontract

/-- The other safe contract: any four-limb left operand, a canonical right operand whose limbs 1
to 3 have the headroom asserted by the Rust caller. -/
theorem mulMont_spec_of_rhs_lt (lhs rhs modulus : Limbs) (inv : Nat)
    (hlhs : lhs.Bounded) (hrhs : rhs.Bounded) (hm : modulus.Bounded)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hlow : modulus.l0 % wordMod = 1) (hinv_lt : inv < 2^64)
    (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hlt : rhs.toNat < modulus.toNat)
    (hlimbs : rhs.l1 + 3 ≤ 2^64 ∧ rhs.l2 + 3 ≤ 2^64 ∧ rhs.l3 + 3 ≤ 2^64) :
    ∀ res, res = mulMont lhs rhs modulus inv →
      res.Bounded ∧ res.toNat < modulus.toNat ∧
        2^256 * res.toNat ≡ lhs.toNat * rhs.toNat [MOD modulus.toNat] := by
  have hcontract : mulContract lhs rhs modulus = true :=
    (mulContract_iff lhs rhs modulus hlhs hrhs hm).2 (Or.inr ⟨hlt, hlimbs⟩)
  exact mulMont_spec lhs rhs modulus inv hlhs hrhs hm hshape hlow hinv_lt hinv hcontract

end PastaCurves.ARMv7EM
