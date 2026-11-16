import PastaCurves.ARMv7EM.Spec.Mul.Invariant
import PastaCurves.ARMv7EM.Compositions.Shared.Loops

/-!
# Low-word cancellation for the translated ARMv7E-M multiplication

These lemmas connect the natural-number rolling-accumulator model to the wrapping
32-bit multiplication used by the Aeneas translation. They establish the assertion
required by `shift_round`: adding the sparse modulus multiple cancels the product's
low word.
-/

namespace PastaCurves.ARMv7EM.Compositions.Mul.Cancellation

open _root_.Aeneas _root_.Aeneas.Std
open PastaCurves.ARMv7EM
open PastaCurves.ARMv7EM.Compositions.Shared.Loops

/-- For a sparse modulus whose low word is one, the low word of `macSparseModulus`
is addition modulo the word radix. -/
theorem macSparseModulus_low (acc : MulAcc) (modulus : FieldWords) (q : Nat)
    (hword0 : modulus.w0 = 1) :
    (macSparseModulus acc modulus q).w0 =
      (acc.w0 + modulus.w0 * q % wordMod) % wordMod := by
  simp [macSparseModulus, adc, addc32, hword0]

/-- The Montgomery factor computed from the product's low word cancels that word
when the inverse relation holds for the sparse modulus. -/
theorem macSparseModulus_macProduct_low_eq_zero
    (acc : MulAcc) (lhs modulus : FieldWords) (b inv : Nat)
    (hword0 : modulus.w0 = 1)
    (hinv : (inv * modulus.w0 + 1) % wordMod = 0) :
    let product := macProduct acc lhs b
    let q := product.w0 * inv % wordMod
    (macSparseModulus product modulus q).w0 = 0 := by
  dsimp only
  rw [macSparseModulus_low _ _ _ hword0]
  simpa only [Nat.mul_comm] using
    cancel_low_word (macProduct acc lhs b).w0 inv modulus.w0 hinv

/-- Aeneas' wrapping `U32` multiplication computes the Montgomery factor used by
the pure cancellation theorem. The hypothesis identifies the translated low word
with the low word of the pure product composition. -/
theorem macSparseModulus_macProduct_wrapping_low_eq_zero
    (acc : MulAcc) (lhs modulus : FieldWords) (b : Nat) (low inv : Std.U32)
    (hlow : low.val = (macProduct acc lhs b).w0)
    (hword0 : modulus.w0 = 1)
    (hinv : (inv.val * modulus.w0 + 1) % wordMod = 0) :
    let q := core.num.U32.wrapping_mul low inv
    (macSparseModulus (macProduct acc lhs b) modulus q.val).w0 = 0 := by
  dsimp only
  rw [core.num.U32.wrapping_mul_val_eq, UScalar.size, UScalarTy.U32_numBits_eq]
  rw [hlow]
  exact macSparseModulus_macProduct_low_eq_zero acc lhs modulus b inv.val hword0 hinv

end PastaCurves.ARMv7EM.Compositions.Mul.Cancellation
