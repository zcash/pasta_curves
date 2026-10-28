import PastaCurves.ARMv7EM.Compositions.Mul.Wrapper
import PastaCurves.ARMv7EM.Spec.Mul.EightRounds

/-!
# Overflow bounds for translated ARMv7E-M multiplication helpers

The generated arrays already ensure that every represented word is a 32-bit value.  This file
connects those arrays to the pure rolling-accumulator specifications and reduces each helper's
checked top-word condition to the numeric invariant needed by the multiplication proof.
-/

namespace PastaCurves.ARMv7EM.Compositions.Mul.Overflow

open _root_.Aeneas _root_.Aeneas.Std
open PastaCurves.ARMv7EM
open PastaCurves.ARMv7EM.Compositions.Shared.Loops
open PastaCurves.ARMv7EM.Compositions.Mul.Rounds

private theorem u32_lt_wordMod (x : Std.U32) : x.val < wordMod := by
  simpa [wordMod, U32.numBits] using x.hBounds

private theorem natArray_getD_lt_wordMod {n : Std.Usize} (a : Std.Array Std.U32 n) (i : Nat) :
    (natArray a).getD i 0 < wordMod := by
  rw [natArray_getD]
  exact u32_lt_wordMod _

private theorem mulAcc_bounded (acc : Std.Array Std.U32 10#usize) : (mulAcc acc).Bounded := by
  exact ⟨natArray_getD_lt_wordMod acc 0, natArray_getD_lt_wordMod acc 1,
    natArray_getD_lt_wordMod acc 2, natArray_getD_lt_wordMod acc 3,
    natArray_getD_lt_wordMod acc 4, natArray_getD_lt_wordMod acc 5,
    natArray_getD_lt_wordMod acc 6, natArray_getD_lt_wordMod acc 7,
    natArray_getD_lt_wordMod acc 8, natArray_getD_lt_wordMod acc 9⟩

private theorem fieldWords_bounded (words : Std.Array Std.U32 8#usize) :
    (fieldWords words).Bounded := by
  exact ⟨natArray_getD_lt_wordMod words 0, natArray_getD_lt_wordMod words 1,
    natArray_getD_lt_wordMod words 2, natArray_getD_lt_wordMod words 3,
    natArray_getD_lt_wordMod words 4, natArray_getD_lt_wordMod words 5,
    natArray_getD_lt_wordMod words 6, natArray_getD_lt_wordMod words 7⟩

/-- A translated product helper cannot overflow when the rolling accumulator's top word is zero. -/
theorem macProductNoOverflow (acc : Std.Array Std.U32 10#usize)
    (multiplicand : Std.Array Std.U32 8#usize) (multiplier : Std.U32)
    (htop : (mulAcc acc).w9 = 0) :
    MacProductNoOverflow acc multiplicand multiplier := by
  let a := mulAcc acc
  let lhs := fieldWords multiplicand
  let r0 := macWord a.w0 lhs.w0 multiplier.val 0
  let r1 := macWord a.w1 lhs.w1 multiplier.val r0.2
  let r2 := macWord a.w2 lhs.w2 multiplier.val r1.2
  let r3 := macWord a.w3 lhs.w3 multiplier.val r2.2
  let r4 := macWord a.w4 lhs.w4 multiplier.val r3.2
  let r5 := macWord a.w5 lhs.w5 multiplier.val r4.2
  let r6 := macWord a.w6 lhs.w6 multiplier.val r5.2
  let r7 := macWord a.w7 lhs.w7 multiplier.val r6.2
  let r8 := addWord a.w8 r7.2
  have ha := mulAcc_bounded acc
  have hlhs := fieldWords_bounded multiplicand
  have hb : multiplier.val < wordMod := u32_lt_wordMod multiplier
  obtain ⟨_, _, b01⟩ := macWord_spec a.w0 lhs.w0 multiplier.val 0
    ha.1 hlhs.1 hb (by norm_num [wordMod]) r0 rfl
  obtain ⟨_, _, b11⟩ := macWord_spec a.w1 lhs.w1 multiplier.val r0.2
    ha.2.1 hlhs.2.1 hb b01 r1 rfl
  obtain ⟨_, _, b21⟩ := macWord_spec a.w2 lhs.w2 multiplier.val r1.2
    ha.2.2.1 hlhs.2.2.1 hb b11 r2 rfl
  obtain ⟨_, _, b31⟩ := macWord_spec a.w3 lhs.w3 multiplier.val r2.2
    ha.2.2.2.1 hlhs.2.2.2.1 hb b21 r3 rfl
  obtain ⟨_, _, b41⟩ := macWord_spec a.w4 lhs.w4 multiplier.val r3.2
    ha.2.2.2.2.1 hlhs.2.2.2.2.1 hb b31 r4 rfl
  obtain ⟨_, _, b51⟩ := macWord_spec a.w5 lhs.w5 multiplier.val r4.2
    ha.2.2.2.2.2.1 hlhs.2.2.2.2.2.1 hb b41 r5 rfl
  obtain ⟨_, _, b61⟩ := macWord_spec a.w6 lhs.w6 multiplier.val r5.2
    ha.2.2.2.2.2.2.1 hlhs.2.2.2.2.2.2.1 hb b51 r6 rfl
  obtain ⟨_, _, b71⟩ := macWord_spec a.w7 lhs.w7 multiplier.val r6.2
    ha.2.2.2.2.2.2.2.1 hlhs.2.2.2.2.2.2.2 hb b61 r7 rfl
  obtain ⟨_, _, b81⟩ := addWord_spec a.w8 r7.2
    ha.2.2.2.2.2.2.2.2.1 b71 r8 rfl
  have hover : a.w9 + r8.2 < wordMod := by
    rw [show a.w9 = 0 by exact htop]
    norm_num [wordMod] at b81 ⊢
    omega
  simpa [MacProductNoOverflow, macProductWords, a, lhs, r0, r1, r2, r3, r4, r5, r6,
    r7, r8, mulAcc, fieldWords] using hover

/-- The exact ten-word total bound from the pure sparse-modulus specification proves the checked
condition for the corresponding translated arrays. -/
theorem macSparseNoOverflow (acc : Std.Array Std.U32 10#usize)
    (modulus : Std.Array Std.U32 8#usize) (q : Std.U32)
    (hshape :
      (fieldWords modulus).w0 = 1 ∧
      (fieldWords modulus).w4 = 0 ∧
      (fieldWords modulus).w5 = 0 ∧
      (fieldWords modulus).w6 = 0 ∧
      (fieldWords modulus).w7 = 2^30)
    (htotal : (mulAcc acc).value + (fieldWords modulus).value * q.val < wordMod^10) :
    MacSparseNoOverflow acc modulus q := by
  obtain ⟨_, htop, _, _, _⟩ := macSparseModulus_spec
    (mulAcc acc) (fieldWords modulus) q.val (mulAcc_bounded acc)
    (fieldWords_bounded modulus) (u32_lt_wordMod q) hshape htotal
  simpa [MacSparseNoOverflow, macSparseWords, addWordSuffix, mulAcc, fieldWords] using htop

private theorem rhs_word (rhs : Std.Array Std.U32 8#usize) (i : Nat) (hi : i < 8) :
    (fieldWords rhs).word i = (natArray rhs).getD i 0 := by
  interval_cases i <;> simp [FieldWords.word, fieldWords]

private theorem initial_mulAcc :
    mulAcc (Std.Array.repeat 10#usize 0#u32) = zeroMulAcc := by
  simp [mulAcc, natArray, zeroMulAcc, Array.getD]

private theorem mulRounds_eq (lhs rhs modulus : Std.Array Std.U32 8#usize) (inv : Std.U32) :
    ∀ i ≤ 8,
      Loop.mulRounds lhs rhs modulus inv
          (mulAcc (Std.Array.repeat 10#usize 0#u32)) i =
        ARMv7EM.mulRounds (fieldWords lhs) (fieldWords rhs) (fieldWords modulus) inv.val i := by
  intro i hi
  induction i with
  | zero => simp [Loop.mulRounds, ARMv7EM.mulRounds, initial_mulAcc]
  | succ i ih =>
      rw [Loop.mulRounds, ARMv7EM.mulRounds, ih (by omega), rhs_word rhs i (by omega)]

/-- The ordinary round invariant supplies every helper-overflow premise and the final top-word
assertion currently packaged by the translated multiplication wrapper. -/
theorem mulPrerequisites (lhs rhs modulus : Std.Array Std.U32 8#usize) (inv : Std.U32)
    (hshape :
      (fieldWords modulus).w0 = 1 ∧
      (fieldWords modulus).w4 = 0 ∧
      (fieldWords modulus).w5 = 0 ∧
      (fieldWords modulus).w6 = 0 ∧
      (fieldWords modulus).w7 = 2^30)
    (hinv : (inv.val * (fieldWords modulus).w0 + 1) % wordMod = 0) :
    Wrapper.MulPrerequisites lhs rhs modulus inv := by
  let initial := Std.Array.repeat 10#usize 0#u32
  let pure := ARMv7EM.mulRounds (fieldWords lhs) (fieldWords rhs) (fieldWords modulus) inv.val
  have hstate : ∀ i ≤ 8,
      (pure i).Bounded ∧ (pure i).w9 = 0 ∧ (pure i).value < wordMod^9 :=
    ARMv7EM.mulRounds_state (fieldWords lhs) (fieldWords rhs) (fieldWords modulus) inv.val
      (fieldWords_bounded lhs) (fieldWords_bounded rhs) (fieldWords_bounded modulus) hshape hinv
  have hrounds : ∀ i ≤ 8, Loop.mulRounds lhs rhs modulus inv (mulAcc initial) i = pure i := by
    intro i hi
    exact mulRounds_eq lhs rhs modulus inv i hi
  refine ⟨hshape, hinv, ?_, ?_, ?_⟩
  · intro current i hi hcurrent multiplier hmultiplier
    apply macProductNoOverflow
    rw [hcurrent, hrounds i.val (by omega)]
    exact (hstate i.val (by omega)).2.1
  · intro current i hi hcurrent afterProduct q hafterProduct hq
    apply macSparseNoOverflow afterProduct modulus q hshape
    rw [hafterProduct, hq]
    have hcurrentPure : mulAcc current = pure i.val := hcurrent.trans (hrounds i.val (by omega))
    rw [hcurrentPure, ← rhs_word rhs i.val hi]
    obtain ⟨hacc, htop, _⟩ := hstate i.val (by omega)
    have hproduct := macProduct_spec (pure i.val) (fieldWords lhs)
      ((fieldWords rhs).word i.val) hacc htop (fieldWords_bounded lhs)
      ((fieldWords rhs).word_lt (fieldWords_bounded rhs) i.val hi)
    rw [hproduct.2]
    exact mulRound_total_lt (pure i.val) (fieldWords lhs) (fieldWords modulus)
      ((fieldWords rhs).word i.val)
      ((macProduct (pure i.val) (fieldWords lhs) ((fieldWords rhs).word i.val)).w0 * inv.val %
        wordMod)
      hacc htop (fieldWords_bounded lhs) (fieldWords_bounded modulus)
      ((fieldWords rhs).word_lt (fieldWords_bounded rhs) i.val hi)
      (Nat.mod_lt _ (by norm_num [wordMod]))
  · rw [hrounds 8 (by norm_num)]
    exact (hstate 8 (by norm_num)).2.1

end PastaCurves.ARMv7EM.Compositions.Mul.Overflow
