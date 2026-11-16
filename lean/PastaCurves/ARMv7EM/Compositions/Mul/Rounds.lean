import PastaCurves.ARMv7EM.Compositions.Shared.Loops
import PastaCurves.ARMv7EM.Spec.Mul.Invariant

/-!
# One translated ARMv7E-M multiplication round

This file relates one iteration of the generated `mul_loop` body to `ARMv7EM.mulRound`.
The hypotheses expose the two overflow assertions made by the Rust helpers; cancellation of the
low word is instead derived from the modulus shape and Montgomery inverse relation.
-/

namespace PastaCurves.ARMv7EM.Compositions.Mul.Rounds

open _root_.Aeneas _root_.Aeneas.Std Result ControlFlow
open PastaCurves.ARMv7EM
open PastaCurves.ARMv7EM.Compositions.Shared.Loops

/-- The exact top-word bound checked by `mac_product`. -/
def MacProductNoOverflow (acc : Std.Array Std.U32 10#usize)
    (multiplicand : Std.Array Std.U32 8#usize) (multiplier : Std.U32) : Prop :=
  let loopResult :=
    macProductWords (natArray acc) (natArray multiplicand) multiplier.val 0 0 8
  let r8 := addWord (loopResult.1.getD 8 0) loopResult.2
  loopResult.1.getD 9 0 + r8.2 < wordMod

/-- The exact top-word bound checked by `mac_sparse_modulus`. -/
def MacSparseNoOverflow (acc : Std.Array Std.U32 10#usize)
    (modulus : Std.Array Std.U32 8#usize) (q : Std.U32) : Prop :=
  let r0 := adc ((natArray acc).getD 0 0) q.val 0
  let lowResult :=
    macSparseWords ((natArray acc).setIfInBounds 0 r0.1) (natArray modulus) q.val r0.2 1 3
  let carryResult := addWordSuffix lowResult.1 lowResult.2 4 3
  let r7 :=
    macWord (carryResult.1.getD 7 0) ((natArray modulus).getD 7 0) q.val carryResult.2
  let r8 := addWord (carryResult.1.getD 8 0) r7.2
  carryResult.1.getD 9 0 + r8.2 < wordMod

/-- Adding the Montgomery quotient times a modulus whose low word is one cancels the low word. -/
theorem macSparseModulus_low_eq_zero (acc : MulAcc) (modulus : FieldWords) (inv : Nat)
    (hshape : modulus.w0 = 1) (hinv : (inv * modulus.w0 + 1) % wordMod = 0) :
    (macSparseModulus acc modulus (acc.w0 * inv % wordMod)).w0 = 0 := by
  have hcancel := cancel_low_word acc.w0 inv modulus.w0 hinv
  rw [hshape] at hcancel
  simpa [macSparseModulus, adc, addc32, wordMod, Nat.mul_comm] using hcancel

/-- One continuing invocation of the generated multiplication-loop body implements `mulRound`.
The premises are the modulus assertions made by `mac_sparse_modulus`, the Montgomery inverse
relation that makes `shift_round`'s low-word assertion true, and the exact overflow bounds for the
concrete helper calls made during this round. -/
theorem mulLoopBody_eq (lhs rhs modulus : Std.Array Std.U32 8#usize) (inv : Std.U32)
    (acc : Std.Array Std.U32 10#usize) (i : Std.Usize) (hi : i.val < 8)
    (hshape :
      (fieldWords modulus).w0 = 1 ∧
      (fieldWords modulus).w4 = 0 ∧
      (fieldWords modulus).w5 = 0 ∧
      (fieldWords modulus).w6 = 0 ∧
      (fieldWords modulus).w7 = 2^30)
    (hinv : (inv.val * (fieldWords modulus).w0 + 1) % wordMod = 0)
    (hproduct : ∀ multiplier : Std.U32,
      multiplier.val = (natArray rhs).getD i.val 0 →
      MacProductNoOverflow acc lhs multiplier)
    (hsparse : ∀ (afterProduct : Std.Array Std.U32 10#usize) (q : Std.U32),
      mulAcc afterProduct =
          macProduct (mulAcc acc) (fieldWords lhs) ((natArray rhs).getD i.val 0) →
      q.val =
          ((macProduct (mulAcc acc) (fieldWords lhs) ((natArray rhs).getD i.val 0)).w0 *
            inv.val) % wordMod →
      MacSparseNoOverflow afterProduct modulus q) :
    pasta_curves.asm.armv7em.mul_loop.body lhs rhs modulus inv acc i ⦃ flow =>
      match flow with
      | cont (result, next) =>
          mulAcc result =
              mulRound (fieldWords lhs) (fieldWords modulus) inv.val
                ((natArray rhs).getD i.val 0) (mulAcc acc) ∧
            next.val = i.val + 1
      | done _ => False ⦄ := by
  unfold pasta_curves.asm.armv7em.mul_loop.body
  have hlt : i < pasta_curves.asm.armv7em.WORDS := by
    simpa [pasta_curves.asm.armv7em.WORDS] using hi
  simp only [hlt, if_true]
  step
  have hi1 : i1.val = (natArray rhs).getD i.val 0 := by
    rw [i1_post]
    simp [natArray, Array.getD, hi]
  step with macProduct_eq acc lhs i1 (hproduct i1 hi1) as ⟨acc1, hacc1⟩
  rw [hi1] at hacc1
  step
  have hi2 : i2.val = (mulAcc acc1).w0 := by
    rw [i2_post]
    simp [mulAcc, natArray, Array.getD]
  step
  have hq : q.val =
      ((macProduct (mulAcc acc) (fieldWords lhs) ((natArray rhs).getD i.val 0)).w0 *
        inv.val) % wordMod := by
    rw [q_post, core.num.U32.wrapping_mul_val_eq, hi2, hacc1]
    simp [U32.size, U32.numBits, wordMod]
  step with macSparseModulus_eq acc1 modulus q hshape
    (hsparse acc1 q hacc1 hq) as ⟨acc2, hacc2⟩
  have hlow : (mulAcc acc2).w0 = 0 := by
    rw [hacc2, hacc1, hq]
    exact macSparseModulus_low_eq_zero
      (macProduct (mulAcc acc) (fieldWords lhs) ((natArray rhs).getD i.val 0))
      (fieldWords modulus) inv.val hshape.1 hinv
  step with shiftRound_eq acc2 hlow as ⟨acc3, hacc3⟩
  step
  constructor
  · rw [hacc3, hacc2, hq, hacc1]
    rfl
  · scalar_tac

end PastaCurves.ARMv7EM.Compositions.Mul.Rounds
