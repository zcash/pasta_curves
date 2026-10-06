import PastaCurves.Fields
import PastaCurves.Glue.Spec
import PastaCurves.Spec
import PastaCurves.X86_64.Backend
import PastaCurves.X86_64.Spec

/-!
# The x86-64 backend at the crate's fields

`PastaCurves.X86_64.Spec` proves the blocks for any modulus of the assumed shape, under arithmetic
conditions on the operands. `montgomeryBlocks_spec` instantiates them at either of the crate's
fields (a `PastaField`, whose facts discharge the hypotheses on the modulus). It shows that the
backend's record of Montgomery blocks (`Backend.lean`) meets the contracts that `Glue.BlocksSpec`
states, under the conditions that the entry points assert in a debug build. The theorems of
`Glue/Spec.lean` carry those contracts through Aeneas' translation of the entry points'
compositions, so they hold for the crate's `add`, `sub`, `mul`, `square`, `sqr_n_mul`, and
`from_mont` on this backend.
-/

namespace PastaCurves.X86_64

/-- The x86-64 record meets the blocks' contracts at a Pasta field, by the blocks' theorems,
including those of its own conversion out of Montgomery form. -/
theorem montgomeryBlocks_spec (F : PastaField) : Glue.BlocksSpec montgomeryBlocks F :=
  Glue.blocksOf_spec F
    { add := fun lhs rhs hl hr hlt hrt =>
        addMod_spec_of_lt lhs rhs F.modulus hl hr F.bounded F.shape hlt hrt
      sub := fun lhs rhs hl hr hlt hrt =>
        subMod_spec_of_lt lhs rhs F.modulus hl hr F.bounded F.shape hlt hrt
      mul_of_lhs_lt := fun lhs rhs hl hr hlt =>
        mulMont_spec_of_lhs_lt lhs rhs F.modulus F.inv hl hr F.bounded F.shape F.inv_lt
          F.inv_spec hlt
      mul_of_rhs_lt := fun lhs rhs hl hr hlt hlimbs =>
        mulMont_spec_of_rhs_lt lhs rhs F.modulus F.inv hl hr F.bounded F.shape F.inv_lt
          F.inv_spec hlt hlimbs
      square := fun value hv hlt =>
        sqrMont_spec value F.modulus F.inv hv F.bounded F.shape F.inv_lt F.inv_spec hlt
      fromMont := fun value hv =>
        fromMont_spec value F.modulus F.inv hv F.bounded F.shape F.inv_lt F.inv_spec }

end PastaCurves.X86_64
