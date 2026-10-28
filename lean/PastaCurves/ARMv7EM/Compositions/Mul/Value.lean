import PastaCurves.ARMv7EM.Compositions.Mul.Wrapper

/-!
# The returned value of translated ARMv7E-M multiplication

This file identifies the value returned by the generated multiplication wrapper with the
ARMv7E-M Montgomery multiplication composition.
-/

namespace PastaCurves.ARMv7EM.Compositions.Mul.Value

open _root_.Aeneas _root_.Aeneas.Std Result
open PastaCurves
open PastaCurves.ARMv7EM
open PastaCurves.ARMv7EM.Compositions.Shared
open PastaCurves.ARMv7EM.Compositions.Shared.Loops

private theorem fieldWords_eq (words : Std.Array Std.U32 8#usize) (value : Limbs)
    (hwords : Conversions.wordValues words = wordsOfLimbsArray value) :
    fieldWords words = fieldWordsOfLimbs value := by
  have hwords' : natArray words = wordsOfLimbsArray value := by
    simpa [natArray, Conversions.wordValues] using hwords
  rw [fieldWords, hwords']
  simp [wordsOfLimbsArray, fieldWordsOfLimbs, Array.getD]

private theorem initial_eq :
    mulAcc (Std.Array.repeat 10#usize 0#u32) =
      (⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩ : MulAcc) := by
  simp [mulAcc, natArray, Array.getD]

private def pureRounds (lhs rhs modulus : FieldWords) (inv : Nat) : MulAcc :=
  let acc : MulAcc := ⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩
  let acc := mulRound lhs modulus inv rhs.w0 acc
  let acc := mulRound lhs modulus inv rhs.w1 acc
  let acc := mulRound lhs modulus inv rhs.w2 acc
  let acc := mulRound lhs modulus inv rhs.w3 acc
  let acc := mulRound lhs modulus inv rhs.w4 acc
  let acc := mulRound lhs modulus inv rhs.w5 acc
  let acc := mulRound lhs modulus inv rhs.w6 acc
  mulRound lhs modulus inv rhs.w7 acc

private theorem mulRounds_eq (lhs rhs modulus : Std.Array Std.U32 8#usize) (inv : Std.U32) :
    Loop.mulRounds lhs rhs modulus inv
        (mulAcc (Std.Array.repeat 10#usize 0#u32)) 8 =
      pureRounds (fieldWords lhs) (fieldWords rhs) (fieldWords modulus) inv.val := by
  rw [initial_eq]
  simp only [Loop.mulRounds]
  rfl

private theorem mulMont_eq_pureRounds (lhs rhs modulus : Limbs) (inv : Nat) :
    mulMont lhs rhs modulus inv =
      let acc := pureRounds (fieldWordsOfLimbs lhs) (fieldWordsOfLimbs rhs)
        (fieldWordsOfLimbs modulus) (inv % wordMod)
      let candidate : FieldWords :=
        ⟨acc.w0, acc.w1, acc.w2, acc.w3, acc.w4, acc.w5, acc.w6, acc.w7⟩
      let highNonzero := (acc.w8 ||| ((wordMod - acc.w8) % wordMod)) / 2^31
      limbsOfWordsArray (selectReduced candidate.toArray
        (fieldWordsOfLimbs modulus).toArray highNonzero) := by
  unfold mulMont pureRounds
  rfl

private theorem wordValues_eq_wordsValue (words : Std.Array Std.U32 8#usize) :
    Conversions.wordValues words = Reduction.wordsValue words := rfl

private theorem modulusWords_eq (words : Std.Array Std.U32 8#usize) (value : Limbs)
    (hwords : Conversions.wordValues words = wordsOfLimbsArray value) :
    Reduction.wordsValue words = (fieldWordsOfLimbs value).toArray := by
  rw [show Reduction.wordsValue words = Conversions.wordValues words by rfl, hwords]
  simp [wordsOfLimbsArray, fieldWordsOfLimbs, FieldWords.toArray]

private theorem candidateWords_eq (acc : Std.Array Std.U32 10#usize)
    (candidate : Std.Array Std.U32 8#usize) (hcandidate : candidate.val = acc.val.take 8) :
    Reduction.wordsValue candidate =
      (⟨(mulAcc acc).w0, (mulAcc acc).w1, (mulAcc acc).w2, (mulAcc acc).w3,
        (mulAcc acc).w4, (mulAcc acc).w5, (mulAcc acc).w6, (mulAcc acc).w7⟩ :
        FieldWords).toArray := by
  apply Array.ext_getElem?
  intro i
  by_cases hi : i < 8
  · have hiCases : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4 ∨ i = 5 ∨ i = 6 ∨ i = 7 := by
      omega
    rcases hiCases with h0 | h1 | h2 | h3 | h4 | h5 | h6 | h7
    all_goals subst i
    all_goals simp [Reduction.wordsValue, hcandidate, mulAcc, natArray,
      FieldWords.toArray, Array.getD]
  · rw [Array.getElem?_eq_none (by simp [Reduction.wordsValue]; omega)]
    rw [Array.getElem?_eq_none (by simp [FieldWords.toArray]; omega)]

private theorem negVal_eq (x : Std.U32) :
    (⟨-x.bv⟩ : Std.U32).val = (wordMod - x.val) % wordMod := by
  rw [UScalar.val, BitVec.toNat_neg]
  simp only [UScalarTy.U32_numBits_eq, wordMod]
  rfl

private theorem highNonzero_eq (high result : Std.U32)
    (hresult : result.val = (high ||| ⟨-high.bv⟩ : Std.U32).val >>> 31) :
    result.val = (high.val ||| ((wordMod - high.val) % wordMod)) / 2^31 := by
  rw [hresult, Nat.shiftRight_eq_div_pow, UScalar.val_or, negVal_eq]

private theorem mulAcc_word8 (acc : Std.Array Std.U32 10#usize) :
    (acc.val[8]!).val = (mulAcc acc).w8 := by
  simp [mulAcc, natArray, Array.getD]

private theorem mulAcc_word9 (acc : Std.Array Std.U32 10#usize) :
    (acc.val[9]!).val = (mulAcc acc).w9 := by
  simp [mulAcc, natArray, Array.getD]

/-- Under the asserted multiplication contract and the translated helpers' explicit arithmetic
prerequisites, the generated ARMv7E-M wrapper returns exactly the pure Montgomery composition. -/
theorem mul_ok_value (lhs rhs modulus : Std.Array Std.U64 4#usize) (inv : Std.U64)
    (_hcontract : mulContract (limbsOfArray lhs) (limbsOfArray rhs)
      (limbsOfArray modulus) = true)
    (hprereq : ∀ (lhsWords rhsWords modulusWords : Std.Array Std.U32 8#usize),
      Conversions.wordValues lhsWords = wordsOfLimbsArray (limbsOfArray lhs) →
      Conversions.wordValues rhsWords = wordsOfLimbsArray (limbsOfArray rhs) →
      Conversions.wordValues modulusWords = wordsOfLimbsArray (limbsOfArray modulus) →
      Wrapper.MulPrerequisites lhsWords rhsWords modulusWords (UScalar.cast .U32 inv)) :
    pasta_curves.asm.armv7em.mul lhs rhs modulus inv ⦃ result =>
      limbsOfArray result = ARMv7EM.mulMont (limbsOfArray lhs) (limbsOfArray rhs)
        (limbsOfArray modulus) inv.val ⦄ := by
  unfold pasta_curves.asm.armv7em.mul
  step with Conversions.toWords_eq lhs as ⟨lhsWords, hlhsWords, _⟩
  step with Conversions.toWords_eq rhs as ⟨rhsWords, hrhsWords, _⟩
  step with Conversions.toWords_eq modulus as ⟨modulusWords, hmodulusWords, _⟩
  let invWord : Std.U32 := UScalar.cast .U32 inv
  have hpre := hprereq lhsWords rhsWords modulusWords hlhsWords hrhsWords hmodulusWords
  rcases hpre with ⟨hshape, hinv, hproduct, hsparse, hfinal⟩
  let initial := Std.Array.repeat 10#usize 0#u32
  simp only [Std.lift, Std.bind_ok]
  step with Loop.mulLoop_eq lhsWords rhsWords modulusWords invWord initial hshape hinv
    hproduct hsparse as ⟨acc, hacc⟩
  simp only [Std.Array.to_slice_mut]
  step with core.slice.index.SliceIndexRangeToUsizeSlice.index.step_spec
    { «end» := pasta_curves.asm.armv7em.WORDS } acc.to_slice
    (by norm_num [pasta_curves.asm.armv7em.WORDS])
  have hcopyLength : (Std.Array.repeat 8#usize 0#u32).to_slice.length = result.length := by
    rw [Std.Array.length_to_slice, result_post1]
    simp only [pasta_curves.asm.armv7em.WORDS.eq_def]
  step with core.slice.Slice.copy_from_slice.step_spec core.marker.CopyU32
    (Std.Array.repeat 8#usize 0#u32).to_slice result hcopyLength as ⟨s2, hs2⟩
  simp only [hs2]
  step
  step with Std.Array.index_usize_spec acc i (by scalar_tac) as ⟨left, hleft⟩
  have hleftZero : left = 0#u32 := by
    apply UScalar.eq_of_val_eq
    rw [hleft]
    have hword9 := mulAcc_word9 acc
    rw [hacc] at hword9
    simpa [i_post, pasta_curves.asm.armv7em.WORDS.eq_def] using hword9.trans hfinal
  simp only [hleftZero, massert_True, Std.bind_ok,
    pasta_curves.asm.armv7em.WORDS.eq_def]
  step with Std.Array.index_usize_spec acc 8#usize (by norm_num) as ⟨high, hhigh⟩
  rw [Primitives.wrappingNeg_eq]
  simp only [bind_ok]
  step with U32.ShiftRight_IScalar_spec (high ||| ⟨-high.bv⟩) 31#i32
    (by norm_num) (by norm_num) as ⟨highNonzero, hhighNonzero, _⟩
  have hhighNonzeroBit : highNonzero.val ≤ 1 := by
    rw [hhighNonzero, Nat.shiftRight_eq_div_pow]
    have horBound : (high ||| ⟨-high.bv⟩ : Std.U32).val < 2^32 :=
      (high ||| ⟨-high.bv⟩ : Std.U32).hBounds
    norm_num at horBound ⊢
    omega
  let candidate := (Std.Array.repeat 8#usize 0#u32).from_slice result
  have hresultVal : result.val = acc.val.take 8 := by
    rw [result_post]
    simp [pasta_curves.asm.armv7em.WORDS]
  have hcandidateVal : candidate.val = acc.val.take 8 := by
    simp only [candidate]
    rw [Std.Array.from_slice_val]
    exact hresultVal
    rw [hresultVal, List.length_take, acc.property]
    norm_num
  step with Reduction.selectReduced_eq candidate modulusWords highNonzero hhighNonzeroBit
    as ⟨reduced, hreduced⟩
  step with Conversions.fromWords_eq reduced as ⟨output, houtput, _, _⟩
  rw [houtput, wordValues_eq_wordsValue, hreduced]
  have hlhsField := fieldWords_eq lhsWords (limbsOfArray lhs) hlhsWords
  have hrhsField := fieldWords_eq rhsWords (limbsOfArray rhs) hrhsWords
  have hmodulusField := fieldWords_eq modulusWords (limbsOfArray modulus) hmodulusWords
  have hmodulusArray := modulusWords_eq modulusWords (limbsOfArray modulus) hmodulusWords
  have hcandidateArray := candidateWords_eq acc candidate hcandidateVal
  have hhighVal : high.val = (mulAcc acc).w8 := by
    rw [hhigh]
    rw [List.Inhabited_getElem_eq_getElem! acc.val 8 (by simp)]
    exact mulAcc_word8 acc
  have hinvWordVal : invWord.val = inv.val % 2^32 := by
    exact UScalar.cast_val_eq .U32 inv
  have hrounds : mulAcc acc =
      pureRounds (fieldWordsOfLimbs (limbsOfArray lhs))
        (fieldWordsOfLimbs (limbsOfArray rhs))
        (fieldWordsOfLimbs (limbsOfArray modulus)) (inv.val % wordMod) := by
    calc
      mulAcc acc = Loop.mulRounds lhsWords rhsWords modulusWords invWord
          (mulAcc (Std.Array.repeat 10#usize 0#u32)) 8 := hacc
      _ = pureRounds (fieldWords lhsWords) (fieldWords rhsWords)
          (fieldWords modulusWords) invWord.val := mulRounds_eq _ _ _ _
      _ = _ := by rw [hlhsField, hrhsField, hmodulusField, hinvWordVal, wordMod]
  have hhighNonzeroVal := highNonzero_eq high highNonzero hhighNonzero
  rw [hcandidateArray, hmodulusArray, hhighNonzeroVal, hhighVal, hrounds]
  rw [← mulMont_eq_pureRounds]

end PastaCurves.ARMv7EM.Compositions.Mul.Value
