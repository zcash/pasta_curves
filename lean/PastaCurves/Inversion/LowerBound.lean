import PastaCurves.Inversion.HullCert
import PastaCurves.Fields

/-!
# Lower bounds on the number of divsteps, by witness

The inversion runs a fixed number of half-delta divsteps, so it is correct for a modulus `p`
exactly when that number reaches `g = 0` from `(1, p, x)` for every `0 ≤ x ≤ p`. Call such a
number sufficient for `p` (`Suffices`). Sufficiency is upward closed, by Lemma 4, so the worst
case for `p` is the least sufficient number.

An upper bound on it needs a proof over every input: the hull certificate's `terminationBound`,
which gives `588` for 255-bit moduli such as the Pasta ones. A lower bound needs only one input
that is still running, checked by computation. This module proves:

- `pallas_suffices_iff`: a number of divsteps is sufficient for Pallas's `p` only if it is at
  least `554`, and every number from `588` on is;
- `vesta_suffices_iff`: the same for Vesta's `q`, with `555`;
- `iterations_256_tight`: for 256-bit inputs, `589` divsteps do not suffice, so the `590` of
  Theorem 1 is exact. The witness is the pair of Section 4.4 of Bernstein, Chen, Harrison,
  Huang, Maxwell, Wang, Wuille, and Yang, "Accelerating and verifying constant-time modular
  inversion" (EUROCRYPT 2026).

So a schedule of nine rounds of `59`, `531` divsteps, is wrong for both fields, and the worst
case lies between `554` and `588` for Pallas and between `555` and `588` for Vesta. The Pasta
witnesses are from `sage/divstep_lower_bounds.sage`; the Rust tests invert them.
-/

namespace PastaCurves.Inversion

/-- `n` divsteps suffice for the modulus `p`: from `(1, p, g)`, `g` is zero after `n` steps for
every `0 ≤ g ≤ p`. This is the condition under which a fixed schedule of `n` divsteps inverts
correctly modulo `p`. -/
def Suffices (n : ℕ) (p : ℤ) : Prop :=
  ∀ g : ℤ, 0 ≤ g → g ≤ p → (divsteps n ⟨1, p, g⟩).g = 0

/-- Once `g` is zero it stays zero, so more steps never hurt. -/
theorem divsteps_g_zero_mono {n m : ℕ} {s : State} (h : n ≤ m) (hg : (divsteps n s).g = 0) :
    (divsteps m s).g = 0 := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [divsteps_add]
  exact (divsteps_of_g_zero k _ hg).2.1

/-- Sufficiency is upward closed. -/
theorem Suffices.mono {n m : ℕ} {p : ℤ} (h : n ≤ m) (hn : Suffices n p) : Suffices m p :=
  fun g hg0 hgp => divsteps_g_zero_mono h (hn g hg0 hgp)

/-- An input still running after `n` steps refutes every number up to `n`. -/
theorem not_suffices_of_witness {n m : ℕ} {p g : ℤ} (hm : m ≤ n) (hg0 : 0 ≤ g)
    (hgp : g ≤ p) (hw : (divsteps n ⟨1, p, g⟩).g ≠ 0) : ¬ Suffices m p :=
  fun hs => hw (hs.mono hm g hg0 hgp)

/-- The hull certificate's bound for 255-bit moduli: `588` divsteps suffice for every odd `p`
below `2^255`. -/
theorem suffices_588 {p : ℤ} (hodd : p % 2 = 1) (hlt : p < 2^255) : Suffices 588 p := by
  intro g hg0 hgp
  have h := Hull.terminationBound 255 p g hodd hg0 hgp hlt
  rwa [show iterations 255 = 588 by decide] at h

/-! ## Pallas -/

/-- Pallas's `p`, as an integer. -/
def pallasP : ℤ := 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001

/-- It is the modulus of `pallasBase`. -/
theorem pallasP_eq : pallasP = (pallasBase.modulus.toNat : ℤ) := by decide +kernel

/-- An input that needs `554` divsteps modulo Pallas's `p`. -/
def pallasSlow : ℤ := 0x35afa69efdea975e84c918864d11250dc472e1964098c6d0316654b8b42304c4

/-- `pallasSlow` is still running after `553` divsteps, and done after `554`. -/
theorem pallasSlow_steps :
    (divsteps 553 ⟨1, pallasP, pallasSlow⟩).g ≠ 0 ∧
      (divsteps 554 ⟨1, pallasP, pallasSlow⟩).g = 0 := by
  decide +kernel

/-- The worst case for Pallas is between `554` and `588`: no number below `554` suffices, and
`588` does. -/
theorem pallas_suffices_iff :
    (∀ n, Suffices n pallasP → 554 ≤ n) ∧ Suffices 588 pallasP := by
  refine ⟨fun n hn => ?_, suffices_588 (by decide) (by decide)⟩
  by_contra hlt
  exact not_suffices_of_witness (by omega) (by decide) (by decide) pallasSlow_steps.1 hn

/-! ## Vesta -/

/-- Vesta's `q`, as an integer. -/
def vestaQ : ℤ := 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001

/-- It is the modulus of `vestaBase`. -/
theorem vestaQ_eq : vestaQ = (vestaBase.modulus.toNat : ℤ) := by decide +kernel

/-- An input that needs `555` divsteps modulo Vesta's `q`. -/
def vestaSlow : ℤ := 0x2dbe09392054def656be338eab1fb4c3ae437e766ab0e45bbd1db488b42304c4

/-- `vestaSlow` is still running after `554` divsteps, and done after `555`. -/
theorem vestaSlow_steps :
    (divsteps 554 ⟨1, vestaQ, vestaSlow⟩).g ≠ 0 ∧
      (divsteps 555 ⟨1, vestaQ, vestaSlow⟩).g = 0 := by
  decide +kernel

/-- The worst case for Vesta is between `555` and `588`. -/
theorem vesta_suffices_iff :
    (∀ n, Suffices n vestaQ → 555 ≤ n) ∧ Suffices 588 vestaQ := by
  refine ⟨fun n hn => ?_, suffices_588 (by decide) (by decide)⟩
  by_contra hlt
  exact not_suffices_of_witness (by omega) (by decide) (by decide) vestaSlow_steps.1 hn

/-! ## 256-bit inputs -/

/-- The modulus of the paper's 590-step witness pair. -/
def witnessF : ℤ := 0xeec9f80577a885d22f8d37c1946187e26805ea27b26c5ae10aa38a02e2ea3157

/-- The input of the paper's 590-step witness pair. -/
def witnessG : ℤ := 0xeb40350da50b11d23183ae8e88ffced0ad11263b6d62cde5e5dc1e934ef8229c

/-- The pair is still running after `589` divsteps, and done after `590`. -/
theorem witness_steps :
    (divsteps 589 ⟨1, witnessF, witnessG⟩).g ≠ 0 ∧
      (divsteps 590 ⟨1, witnessF, witnessG⟩).g = 0 := by
  decide +kernel

/-- Theorem 1's count for 256-bit inputs is exact: the hull bound `iterations 256 = 590` holds,
and the same statement with `589` fails. -/
theorem iterations_256_tight :
    TerminationBound 256 ∧
      ¬ ∀ f g : ℤ, f % 2 = 1 → 0 ≤ g → g ≤ f → f < 2^256 →
          (divsteps 589 ⟨1, f, g⟩).g = 0 :=
  ⟨Hull.terminationBound_256, fun h =>
    witness_steps.1 (h witnessF witnessG (by decide) (by decide) (by decide) (by decide))⟩

end PastaCurves.Inversion
