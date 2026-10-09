import PastaCurves.X86_64.Compositions
import PastaCurves.Glue.Blocks

/-!
# The x86-64 backend of the translated glue

The x86-64 backend's record of Montgomery blocks, which the translated compositions of
`Glue/Funs.lean` run over. As in the Rust impl, `square` is the two squaring blocks, and
`from_mont` is the backend's own block.
-/

namespace PastaCurves.X86_64

/-- The x86-64 backend as the translation's record of `MontgomeryBlocks`. -/
def montgomeryBlocks : pasta_curves.montgomery.MontgomeryBlocks Unit :=
  Glue.blocksOf addMod subMod mulMont sqrMont fromMont

end PastaCurves.X86_64
