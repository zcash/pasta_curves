import PastaCurves.AArch64.Compositions
import PastaCurves.Glue.Funs
import PastaCurves.Words

/-!
# The AArch64 backend of the translated glue

Aeneas' translation of the entry points' generic compositions (`Glue/Funs.lean`) runs them over a
record of a backend's Montgomery blocks, the translation of the trait `MontgomeryBlocks`. The
record here is the AArch64 backend's. Each block reads its operands' limbs from the translation's
arrays, runs the block's transcription, and writes the result back as an array. As in the Rust
impl, `from_mont` is the multiplication by one.
-/

namespace PastaCurves.AArch64

open Aeneas Aeneas.Std

/-- The AArch64 backend as the translation's record of `MontgomeryBlocks`. -/
def montgomeryBlocks : pasta_curves.asm.entry.MontgomeryBlocks Unit where
  add lhs rhs modulus :=
    .ok (limbsArray (addMod (limbsOfArray lhs) (limbsOfArray rhs) (limbsOfArray modulus)))
  sub lhs rhs modulus :=
    .ok (limbsArray (subMod (limbsOfArray lhs) (limbsOfArray rhs) (limbsOfArray modulus)))
  mul lhs rhs modulus inv :=
    .ok (limbsArray (mulMont (limbsOfArray lhs) (limbsOfArray rhs) (limbsOfArray modulus) inv.val))
  square value modulus inv :=
    .ok (limbsArray (sqrMont (limbsOfArray value) (limbsOfArray modulus) inv.val))
  from_mont value modulus inv :=
    .ok (limbsArray (fromMont (limbsOfArray value) (limbsOfArray modulus) inv.val))

end PastaCurves.AArch64
