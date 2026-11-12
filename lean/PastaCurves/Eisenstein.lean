import PastaCurves.Eisenstein.Ring
import PastaCurves.Eisenstein.Classes
import PastaCurves.Eisenstein.Chain
import PastaCurves.Eisenstein.Ladder

/-!
# The Eisenstein recoding of `glv_eisenstein`

The facts about `ℤ[ω]` that `src/glv_eisenstein.rs` relies on, proved over the crate's own
constants:

- `Eisenstein/Ring.lean`: `ℤ[ω]`, its norm, its unit group `μ₆`, and the inertness of 2.
- `Eisenstein/Classes.lean`: the 48 odd classes mod 8, the free action of `μ₆` on them, and the
  eight representatives `REPS` whose 48 unit multiples are exactly those classes.
- `Eisenstein/Chain.lean`: the seven-addition chain of `Table::window_proj` reaches `REPS`.
- `Eisenstein/Ladder.lean`: the batch-affine ladder of `Table::batch_mul_affine` has no
  exceptional case on Pallas or Vesta but the identity, given the Babai bound on `decompose`.

These are statements about the integers and the residue classes the Rust code computes with, not
about its instruction streams: the code's transcription of `REPS`, the look-up table, the chain, and
the curve constants into these modules is checked by inspection, and `sage/glv_eisenstein.sage`
re-derives the same facts independently.
-/
