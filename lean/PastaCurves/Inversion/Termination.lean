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

/-- Theorem 5 as a proposition about `divsteps`: from `two_delta = 1`, odd `f`, and
`0 ≤ g ≤ f < 2^b`, `g` is zero after `iterations b` steps. -/
def TerminationBound (b : ℕ) : Prop :=
  ∀ f g : ℤ, f % 2 = 1 → 0 ≤ g → g ≤ f → f < 2^b → (divsteps (iterations b) ⟨1, f, g⟩).g = 0

/-- Theorem 5 as the book states it, for every `n` from the bound on: once `g` is zero it stays
zero (Lemma 4), so the bound at `iterations b` gives `g_n = 0` for every `n ≥ iterations b`. -/
theorem TerminationBound.ge {b : ℕ} (h : TerminationBound b) (f g : ℤ) (hf : f % 2 = 1)
    (hg0 : 0 ≤ g) (hgf : g ≤ f) (hfb : f < 2^b) (n : ℕ) (hn : iterations b ≤ n) :
    (divsteps n ⟨1, f, g⟩).g = 0 := by
  obtain ⟨k, rfl⟩ : ∃ k, n = iterations b + k := ⟨n - iterations b, by omega⟩
  rw [divsteps_add]
  obtain ⟨-, hz, -⟩ := divsteps_of_g_zero k _ (h f g hf hg0 hgf hfb)
  exact hz

end PastaCurves.Inversion
