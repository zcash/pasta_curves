import PastaCurves.Glue.Funs
import PastaCurves.Words

/-!
# A backend's blocks as the translation's record

Aeneas' translation of the entry points' generic compositions (`Glue/Funs.lean`) runs them over a
record of a backend's Montgomery blocks, the translation of the trait `MontgomeryBlocks`.
`blocksOf` makes that record from a backend's models of its blocks on limbs. Each method reads its
operands' limbs from the translation's arrays, runs the block's model, and writes the result back
as an array.
-/

namespace PastaCurves.Glue

open Aeneas Aeneas.Std

/-- The record of a backend's Montgomery blocks, from their models on limbs. -/
def blocksOf (add sub : Limbs → Limbs → Limbs → Limbs)
    (mul : Limbs → Limbs → Limbs → Nat → Limbs) (square fromMont : Limbs → Limbs → Nat → Limbs) :
    pasta_curves.montgomery.MontgomeryBlocks Unit where
  add lhs rhs modulus :=
    .ok (limbsArray (add (limbsOfArray lhs) (limbsOfArray rhs) (limbsOfArray modulus)))
  sub lhs rhs modulus :=
    .ok (limbsArray (sub (limbsOfArray lhs) (limbsOfArray rhs) (limbsOfArray modulus)))
  mul lhs rhs modulus inv :=
    .ok (limbsArray (mul (limbsOfArray lhs) (limbsOfArray rhs) (limbsOfArray modulus) inv.val))
  square value modulus inv :=
    .ok (limbsArray (square (limbsOfArray value) (limbsOfArray modulus) inv.val))
  from_mont value modulus inv :=
    .ok (limbsArray (fromMont (limbsOfArray value) (limbsOfArray modulus) inv.val))

end PastaCurves.Glue
