import PastaCurves.ARMv7EM.Compositions.Shared.Conversions
import PastaCurves.ARMv7EM.Compositions.Mul.Loop
import PastaCurves.ARMv7EM.Compositions.Shared.Reduction

/-!
# The translated ARMv7E-M multiplication wrapper

This file proves successful execution and bounded output for the generated
`pasta_curves.asm.armv7em.mul` wrapper under its helper overflow and final-accumulator
prerequisites. `Compositions.Mul.Value` identifies its returned value with the proved Montgomery
multiplication model, and `Compositions.Mul.Entry` discharges the prerequisites at a Pasta field.
-/

namespace PastaCurves.ARMv7EM.Compositions.Mul.Wrapper

open _root_.Aeneas _root_.Aeneas.Std Result
open PastaCurves
open PastaCurves.ARMv7EM
open PastaCurves.ARMv7EM.Compositions.Shared
open PastaCurves.ARMv7EM.Compositions.Shared.Loops

/-- The arithmetic prerequisites needed by the translated multiplication loop and its final
accumulator assertion.  These are stated on the actual arrays returned by the generated conversion
helpers. -/
def MulPrerequisites (lhs rhs modulus : Std.Array Std.U32 8#usize) (inv : Std.U32) : Prop :=
  let initial := Std.Array.repeat 10#usize 0#u32
  let final := Loop.mulRounds lhs rhs modulus inv (mulAcc initial) 8
  ((fieldWords modulus).w0 = 1 ∧
    (fieldWords modulus).w4 = 0 ∧
    (fieldWords modulus).w5 = 0 ∧
    (fieldWords modulus).w6 = 0 ∧
    (fieldWords modulus).w7 = 2^30) ∧
  ((inv.val * (fieldWords modulus).w0 + 1) % wordMod = 0) ∧
  (∀ (current : Std.Array Std.U32 10#usize) (i : Std.Usize),
    i.val < 8 →
    mulAcc current = Loop.mulRounds lhs rhs modulus inv (mulAcc initial) i.val →
    ∀ multiplier : Std.U32,
      multiplier.val = (natArray rhs).getD i.val 0 →
      Rounds.MacProductNoOverflow current lhs multiplier) ∧
  (∀ (current : Std.Array Std.U32 10#usize) (i : Std.Usize),
    i.val < 8 →
    mulAcc current = Loop.mulRounds lhs rhs modulus inv (mulAcc initial) i.val →
    ∀ (afterProduct : Std.Array Std.U32 10#usize) (q : Std.U32),
      mulAcc afterProduct =
          macProduct (mulAcc current) (fieldWords lhs) ((natArray rhs).getD i.val 0) →
      q.val =
          ((macProduct (mulAcc current) (fieldWords lhs)
            ((natArray rhs).getD i.val 0)).w0 * inv.val) % wordMod →
      Rounds.MacSparseNoOverflow afterProduct modulus q) ∧
  final.w9 = 0

private theorem mulAcc_word9 (acc : Std.Array Std.U32 10#usize) :
    (acc.val[9]!).val = (mulAcc acc).w9 := by
  simp [mulAcc, natArray, Array.getD]

/-- Under the caller's exact operand assertion and the explicit unchecked arithmetic prerequisites,
the generated ARMv7E-M `mul` wrapper terminates successfully and returns bounded limbs.

The caller assertion is retained verbatim, although the current proof uses the separately stated
helper-overflow and final-high-word obligations rather than deriving them from that assertion. -/
theorem mul_ok_bounded (lhs rhs modulus : Std.Array Std.U64 4#usize) (inv : Std.U64)
    (_hcontract : mulContract (limbsOfArray lhs) (limbsOfArray rhs)
      (limbsOfArray modulus) = true)
    (hprereq : ∀ (lhsWords rhsWords modulusWords : Std.Array Std.U32 8#usize),
      Conversions.wordValues lhsWords = wordsOfLimbsArray (limbsOfArray lhs) →
      Conversions.wordValues rhsWords = wordsOfLimbsArray (limbsOfArray rhs) →
      Conversions.wordValues modulusWords = wordsOfLimbsArray (limbsOfArray modulus) →
      MulPrerequisites lhsWords rhsWords modulusWords (UScalar.cast .U32 inv)) :
    pasta_curves.asm.armv7em.mul lhs rhs modulus inv ⦃ result =>
      (limbsOfArray result).Bounded ⦄ := by
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
  step with Conversions.fromWords_eq reduced as ⟨output, _, _, houtputBounded⟩
  exact houtputBounded
