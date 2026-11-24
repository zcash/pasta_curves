import PastaCurves.Entry
import PastaCurves.AArch64.Backend
import PastaCurves.AArch64.Spec

/-!
# The AArch64 backend at the crate's fields

`PastaCurves.AArch64.Spec` proves the blocks for any modulus of the assumed shape, under arithmetic
conditions on the operands. `montgomeryBlocks_spec` instantiates them at either of the crate's
fields (a `PastaField`, whose facts discharge the hypotheses on the modulus): the backend's record
of Montgomery blocks (`Backend.lean`) meets the contracts that `Glue.BlocksSpec` states, under the
conditions that the entry points assert in a debug build. With `invertBlocks_spec`, the contracts of
the inversion blocks, it gives `entrySpecs`: the shared `EntrySpecs` of `PastaCurves/Entry.lean`
for this backend, which states what the crate's `add`, `sub`, `mul`, `square`, `sqr_n_mul`,
`from_mont`, and `invert` compute on it.
-/

namespace PastaCurves.AArch64

/-- The AArch64 record meets the blocks' contracts at a Pasta field, by the blocks' theorems,
including those of `from_mont`, the multiplication by one. -/
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

/-- The crate's entry points on the AArch64 backend, at every Pasta field: `EntrySpecs` of
its records, from the contracts of its Montgomery blocks and of its inversion blocks. -/
theorem entrySpecs : EntrySpecs montgomeryBlocks invertBlocks :=
  .of montgomeryBlocks_spec invertBlocks_spec

end PastaCurves.AArch64
