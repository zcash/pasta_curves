import PastaCurves.Compositions
import PastaCurves.Inversion.Vectors

/-!
# Checking a backend's inversion blocks against their known answers

The checks that every backend's `Vectors.lean` runs on its transcribed inversion blocks, stated
once over the record `InvertBlocks` of the blocks. `InvertBlocks.failures` lists, block by block,
the indices of the known answers of `Inversion/Vectors.lean` that the record does not reproduce.
A backend proves each block's list empty by kernel evaluation, one theorem per block, and
evaluates `failures` to name any that fail.

`divstepTraceFailures` checks a backend's packed divstep round step by step against
`divstepTrace`. The round carries the words of its own instruction schedule, so the backend
supplies `encode`, which lays out a packed state `(two_delta, f, g)` as the round carries it
between two steps.
-/

namespace PastaCurves

namespace InvertBlocks

/-- The indices of the known answers of `divstep59` that the record does not reproduce. -/
def divstep59Failures (B : InvertBlocks) : List Nat :=
  (List.range Inversion.divstep59Vectors.length).filter fun i =>
    match Inversion.divstep59Vectors[i]? with
    | some (two_delta, f0, g0, res) => B.divstep59 two_delta f0 g0 != res
    | none => false

/-- The indices of the known answers of `sign_mag` that the record does not reproduce. -/
def signMagFailures (B : InvertBlocks) : List Nat :=
  (List.range Inversion.signMagVectors.length).filter fun i =>
    match Inversion.signMagVectors[i]? with
    | some (u, v, q, r, res) => B.signMag u v q r != res
    | none => false

/-- The indices of the known answers of `fg_row` that the record does not reproduce. -/
def fgRowFailures (B : InvertBlocks) : List Nat :=
  (List.range Inversion.fgRowVectors.length).filter fun i =>
    match Inversion.fgRowVectors[i]? with
    | some (f, g, m0, m1, s0, s1, res) => B.fgRow f g m0 m1 s0 s1 != res
    | none => false

/-- The indices of the known answers of `de_row` that the record does not reproduce. -/
def deRowFailures (B : InvertBlocks) : List Nat :=
  (List.range Inversion.deRowVectors.length).filter fun i =>
    match Inversion.deRowVectors[i]? with
    | some (d, e, m0, m1, s0, s1, res) => B.deRow d e m0 m1 s0 s1 != res
    | none => false

/-- The indices of the known answers of `amontred` that the record does not reproduce. -/
def amontredFailures (B : InvertBlocks) : List Nat :=
  (List.range Inversion.amontredVectors.length).filter fun i =>
    match Inversion.amontredVectors[i]? with
    | some (t, F, res) => B.amontred t F.modulus F.inv != res
    | none => false

/-- The indices of the known answers of `cond_sub` that the record does not reproduce. -/
def condSubFailures (B : InvertBlocks) : List Nat :=
  (List.range Inversion.condSubVectors.length).filter fun i =>
    match Inversion.condSubVectors[i]? with
    | some (x, F, res) => B.condSub x F.modulus != res
    | none => false

/-- The failures of the six blocks, in the order `divstep59`, `sign_mag`, `fg_row`, `de_row`,
`amontred`, `cond_sub`, for evaluation. -/
def failures (B : InvertBlocks) :
    List Nat × List Nat × List Nat × List Nat × List Nat × List Nat :=
  (B.divstep59Failures, B.signMagFailures, B.fgRowFailures, B.deRowFailures,
    B.amontredFailures, B.condSubFailures)

end InvertBlocks

namespace Inversion

/-- The steps, numbered from one, after which a packed divstep round differs from
`divstepTrace`: where `round (encode a)` is not `encode b` for consecutive states `a`, `b`. -/
def divstepTraceFailures {W : Type} [DecidableEq W] (encode : Nat × Nat × Nat → W)
    (round : W → W) : List Nat :=
  (List.range (divstepTrace.length - 1)).filterMap fun i =>
    match divstepTrace[i]?, divstepTrace[i + 1]? with
    | some a, some b => if round (encode a) ≠ encode b then some (i + 1) else none
    | _, _ => none

end Inversion

end PastaCurves
