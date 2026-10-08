import PastaCurves.Portable.Funs
import PastaCurves.Words

/-!
# The translated portable blocks as a record

Aeneas' translation of the portable blocks (`Funs.lean`) includes the record of the six blocks,
the trait's instance for the portable backend. The proofs and the tests name it `blocks`. The
model's values are written as the translation's words by the shared definitions of
`PastaCurves/Words.lean`.
-/

namespace PastaCurves.Portable

/-- The translated record of the portable blocks. -/
abbrev blocks := pasta_curves.inversion.portable.Backend.Insts.Pasta_curvesInversionInvertBlocks

end PastaCurves.Portable
