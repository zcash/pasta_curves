import PastaCurves.ARMv7EM.Compositions.Mul.Rounds

/-!
# The translated ARMv7E-M multiplication loop

This file relates all eight iterations of the generated `mul_loop` to the rolling CIOS
multiplication rounds. The overflow premises are stated for each concrete translated accumulator
that can occur at the corresponding loop index.
-/

namespace PastaCurves.ARMv7EM.Compositions.Mul.Loop

open _root_.Aeneas _root_.Aeneas.Std Result ControlFlow
open PastaCurves.ARMv7EM
open PastaCurves.ARMv7EM.Compositions.Shared.Loops
open PastaCurves.ARMv7EM.Compositions.Mul.Rounds

/-- The rolling accumulator after the first `i` words of the right operand have been processed. -/
def mulRounds (lhs rhs modulus : Std.Array Std.U32 8#usize) (inv : Std.U32)
    (initial : MulAcc) : Nat → MulAcc
  | 0 => initial
  | i + 1 =>
      mulRound (fieldWords lhs) (fieldWords modulus) inv.val
        ((natArray rhs).getD i 0) (mulRounds lhs rhs modulus inv initial i)

/-- The generated multiplication loop, started at zero, performs all eight rolling CIOS rounds.

The overflow hypotheses are the exact assertions made by the two helpers in each round. They are
required for the actual translated accumulator at index `i`, identified by the same invariant used
to prove the loop. -/
theorem mulLoop_eq (lhs rhs modulus : Std.Array Std.U32 8#usize) (inv : Std.U32)
    (acc : Std.Array Std.U32 10#usize)
    (hshape :
      (fieldWords modulus).w0 = 1 ∧
      (fieldWords modulus).w4 = 0 ∧
      (fieldWords modulus).w5 = 0 ∧
      (fieldWords modulus).w6 = 0 ∧
      (fieldWords modulus).w7 = 2^30)
    (hinv : (inv.val * (fieldWords modulus).w0 + 1) % wordMod = 0)
    (hproduct : ∀ (current : Std.Array Std.U32 10#usize) (i : Std.Usize),
      i.val < 8 →
      mulAcc current = mulRounds lhs rhs modulus inv (mulAcc acc) i.val →
      ∀ multiplier : Std.U32,
        multiplier.val = (natArray rhs).getD i.val 0 →
        MacProductNoOverflow current lhs multiplier)
    (hsparse : ∀ (current : Std.Array Std.U32 10#usize) (i : Std.Usize),
      i.val < 8 →
      mulAcc current = mulRounds lhs rhs modulus inv (mulAcc acc) i.val →
      ∀ (afterProduct : Std.Array Std.U32 10#usize) (q : Std.U32),
        mulAcc afterProduct =
            macProduct (mulAcc current) (fieldWords lhs) ((natArray rhs).getD i.val 0) →
        q.val =
            ((macProduct (mulAcc current) (fieldWords lhs)
              ((natArray rhs).getD i.val 0)).w0 * inv.val) % wordMod →
        MacSparseNoOverflow afterProduct modulus q) :
    pasta_curves.asm.armv7em.mul_loop lhs rhs modulus inv acc 0#usize ⦃ result =>
      mulAcc result = mulRounds lhs rhs modulus inv (mulAcc acc) 8 ⦄ := by
  unfold pasta_curves.asm.armv7em.mul_loop
  apply loop.spec_decr_nat (fun (_, i) => 8 - i.val)
    (fun (current, i) =>
      i.val ≤ 8 ∧
        mulAcc current = mulRounds lhs rhs modulus inv (mulAcc acc) i.val)
  · rintro ⟨current, i⟩ ⟨hi, hcurrent⟩
    by_cases hlt : i.val < 8
    · apply WP.spec_mono
      · exact Rounds.mulLoopBody_eq lhs rhs modulus inv current i hlt hshape hinv
          (hproduct current i hlt hcurrent) (hsparse current i hlt hcurrent)
      · intro flow hflow
        cases flow with
        | done result => simp at hflow
        | cont state =>
          obtain ⟨result, next⟩ := state
          obtain ⟨hresult, hnext⟩ := hflow
          refine ⟨⟨?_, ?_⟩, ?_⟩
          · omega
          · rw [hnext, mulRounds, ← hcurrent]
            exact hresult
          · change 8 - next.val < 8 - i.val
            omega
    · unfold pasta_curves.asm.armv7em.mul_loop.body
      have hnot : ¬i < pasta_curves.asm.armv7em.WORDS := by
        simpa [pasta_curves.asm.armv7em.WORDS] using hlt
      simp only [hnot, if_false, WP.spec_ok]
      have hi8 : i.val = 8 := by omega
      simpa [hi8] using hcurrent
  · exact ⟨by norm_num, rfl⟩

end PastaCurves.ARMv7EM.Compositions.Mul.Loop
