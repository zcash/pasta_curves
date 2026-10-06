import PastaCurves.Fields
import PastaCurves.Glue.Spec
import PastaCurves.Spec
import PastaCurves.AArch64.Backend
import PastaCurves.AArch64.Spec

/-!
# The AArch64 backend at the crate's fields

`PastaCurves.AArch64.Spec` proves the blocks for any modulus of the assumed shape, under arithmetic
conditions on the operands. The theorems here instantiate them at either of the crate's fields (a
`PastaField`, whose facts discharge the hypotheses on the modulus).

`montgomeryBlocks_spec` shows that the backend's record of Montgomery blocks (`Backend.lean`) meets
the contracts that `Glue.BlocksSpec` states, under the conditions that the entry points assert in a
debug build. The theorems of `Glue/Spec.lean` carry those contracts through Aeneas' translation of
the entry points' compositions, so they hold for the crate's `add`, `sub`, `mul`, `square`,
`sqr_n_mul`, and `from_mont` on this backend. `invert_entry_spec` is the crate's `invert` on the
AArch64 inversion blocks. The results are stated against the Montgomery radix `R = 2^256` of
`Fields.lean`.
-/

namespace PastaCurves.AArch64

/-- The crate's `invert` at a Pasta field, on the AArch64 blocks: the shared `invert_entry_spec`
at `invertBlocks`. The input is canonical, as the entry point asserts, and `e0` is `2^562 mod p`,
as its contract requires. The result is canonical. For `x = 0` it is `0`; otherwise it is the
Montgomery inverse, with `x * result ≡ R^2 (mod p)`. -/
theorem invert_entry_spec (F : PastaField) (x : Limbs) (hx : x.Bounded)
    (h : isCanonical x F.modulus = true) (e0 : Limbs) (he0 : e0 = Inversion.startE F) :
    (invert invertBlocks x F.modulus F.inv e0).Bounded ∧
      (invert invertBlocks x F.modulus F.inv e0).toNat < F.modulus.toNat ∧
      (x.toNat = 0 → invert invertBlocks x F.modulus F.inv e0 = Limbs.ofNat 0) ∧
      (x.toNat ≠ 0 →
        x.toNat * (invert invertBlocks x F.modulus F.inv e0).toNat ≡ R^2 [MOD F.modulus.toNat]) :=
  PastaCurves.invert_entry_spec invertBlocks F (invertBlocks_spec F) x hx h e0 he0

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

end PastaCurves.AArch64
