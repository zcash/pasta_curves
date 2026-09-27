import PastaCurves.Inversion.Divstep

/-!
# The termination bound, as a proposition

Theorem 5 of the proof: for `b`-bit inputs, `iterations b` half-delta divsteps reach `g = 0`.
The number is Theorem 1 of Bernstein, Chen, Harrison, Huang, Maxwell, Wang, Wuille, and Yang,
"Accelerating and verifying constant-time modular inversion" (EUROCRYPT 2026), and for
`b = 256` it is `590`, ten rounds of `59`. The proposition is stated here so that the
correctness theorem can take it as a hypothesis; its proof, by a hull certificate checked by
computation, is separate.
-/

namespace PastaCurves.Inversion

/-- The number of half-delta divsteps that suffice for `b`-bit inputs: `⌈(9437 b + 1) / 4096⌉`. -/
def iterations (b : ℕ) : ℕ := (9437 * b + 1 + 4095) / 4096

example : iterations 256 = 590 := by decide

/-- Theorem 5 as a proposition about `divsteps`: from `d = 1`, odd `f`, and `0 ≤ g ≤ f < 2^b`,
`g` is zero after `iterations b` steps. -/
def TerminationBound (b : ℕ) : Prop :=
  ∀ f g : ℤ, f % 2 = 1 → 0 ≤ g → g ≤ f → f < 2^b → (divsteps (iterations b) ⟨1, f, g⟩).g = 0

end PastaCurves.Inversion
