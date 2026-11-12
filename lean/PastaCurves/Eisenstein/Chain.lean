import PastaCurves.Eisenstein.Classes

/-!
# The seven-addition chain that builds the table

`Table::window_proj` in `src/glv_eisenstein.rs` builds the eight representatives `r_j P` from `P`
with seven point additions, moving between multiples with the endomorphism `φ(P) = ω P`, which costs
one base-field multiplication, and with negation, which is free. `Table::batch` runs the same chain
in affine coordinates. On multiples of `P`, every point is `x P` for an Eisenstein integer `x`, and
`φ`, `+`, `-` act on `x` as `rotate`, `+`, `-` do; so the chain is correct for every `P` exactly when
it computes `REPS` on the scalars, which is what `window_eq_reps` checks.

The same identity is checked in `sage/glv_eisenstein.sage` by step `[4a]`, and in the crate by the
`window_matches_representatives` test.
-/

namespace PastaCurves.Eisenstein

open E

/-- The chain of `Table::window_proj`, on the scalar `p` of the base point, line for line. Its seven
additions and subtractions are the only group operations; every `rotate` is `φ`. -/
def window (p : E ℤ) : List (E ℤ) :=
  let phi_p := rotate p
  let d1 := p - phi_p                -- add 1
  let b := d1 - rotate d1            -- add 2
  let phi_b := rotate b
  let m3 := rotate phi_b
  let r3 := -phi_b
  let t3a := m3 + phi_p              -- add 3
  let t3b := phi_p - m3              -- add 4
  let t4a := phi_p + r3              -- add 5
  let t4b := phi_p - r3              -- add 6
  let t19 := t4b + phi_p             -- add 7
  [p, d1, rotate t4a, -(rotate t3b), -m3, -t3a, rotate (rotate t4b), rotate (rotate t19)]

/-- From the base point's scalar `1`, the chain reaches exactly `REPS`, in `REPS`' order, so the
table entry `j` holds `r_j P`. -/
theorem window_eq_reps : window 1 = reps := by decide

/-- For any base point's scalar `p`, entry `j` of the chain is `r_j p`: the chain is linear, so it
builds the same table for every point, not only for the generator. -/
theorem window_eq_mul (p : E ℤ) : window p = reps.map (· * p) := by
  simp only [window, reps, List.map_cons, List.map_nil, List.cons.injEq, and_true]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ext <;> simp [rotate] <;> ring

end PastaCurves.Eisenstein
