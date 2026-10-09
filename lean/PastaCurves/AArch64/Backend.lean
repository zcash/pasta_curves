import PastaCurves.AArch64.Compositions
import PastaCurves.Glue.Blocks

/-!
# The AArch64 backend of the translated glue

The AArch64 backend's record of Montgomery blocks, which the translated compositions of
`Glue/Funs.lean` run over. As in the Rust impl, `from_mont` is the multiplication by one.
-/

namespace PastaCurves.AArch64

/-- The AArch64 backend as the translation's record of `MontgomeryBlocks`. -/
def montgomeryBlocks : pasta_curves.montgomery.MontgomeryBlocks Unit :=
  Glue.blocksOf addMod subMod mulMont sqrMont fromMont

end PastaCurves.AArch64
