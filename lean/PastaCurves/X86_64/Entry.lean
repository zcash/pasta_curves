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

section EntryPoints

open Aeneas Aeneas.Std pasta_curves Glue

/-- The crate's `add` on the x86-64 backend, at a Pasta field: for canonical operands, as it
asserts, the canonical sum. -/
theorem add_entry_spec (F : PastaField) (lhs rhs : Std.Array Std.U64 4#usize)
    (hl : isCanonical (limbsOfArray lhs) F.modulus = true)
    (hr : isCanonical (limbsOfArray rhs) F.modulus = true) :
    montgomery.add_with montgomeryBlocks lhs rhs (limbsArray F.modulus) ⦃ res =>
      Canonical F res 1 ((limbsOfArray lhs).toNat + (limbsOfArray rhs).toNat) ⦄ :=
  add_with_spec montgomeryBlocks F (montgomeryBlocks_spec F) lhs rhs hl hr

/-- The crate's `sub` on the x86-64 backend, at a Pasta field: for canonical operands, as it
asserts, the canonical difference. -/
theorem sub_entry_spec (F : PastaField) (lhs rhs : Std.Array Std.U64 4#usize)
    (hl : isCanonical (limbsOfArray lhs) F.modulus = true)
    (hr : isCanonical (limbsOfArray rhs) F.modulus = true) :
    montgomery.sub_with montgomeryBlocks lhs rhs (limbsArray F.modulus) ⦃ res =>
      (limbsOfArray res).toNat < F.modulus.toNat ∧
        (limbsOfArray res).toNat + (limbsOfArray rhs).toNat ≡ (limbsOfArray lhs).toNat
          [MOD F.modulus.toNat] ⦄ :=
  sub_with_spec montgomeryBlocks F (montgomeryBlocks_spec F) lhs rhs hl hr

/-- The crate's `mul` on the x86-64 backend, at a Pasta field: under the contract that it asserts,
a canonical `res` with `R * res ≡ lhs * rhs (mod p)`. -/
theorem mul_entry_spec (F : PastaField) (lhs rhs : Std.Array Std.U64 4#usize)
    (h : mulContract (limbsOfArray lhs) (limbsOfArray rhs) F.modulus = true) :
    montgomery.mul_with montgomeryBlocks lhs rhs (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res R ((limbsOfArray lhs).toNat * (limbsOfArray rhs).toNat) ⦄ :=
  mul_with_spec montgomeryBlocks F (montgomeryBlocks_spec F) lhs rhs h

/-- The crate's `square` on the x86-64 backend, at a Pasta field: for a canonical operand, as it
asserts, a canonical `res` with `R * res ≡ value² (mod p)`. -/
theorem square_entry_spec (F : PastaField) (value : Std.Array Std.U64 4#usize)
    (h : isCanonical (limbsOfArray value) F.modulus = true) :
    montgomery.square_with montgomeryBlocks value (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res R ((limbsOfArray value).toNat * (limbsOfArray value).toNat) ⦄ :=
  square_with_spec montgomeryBlocks F (montgomeryBlocks_spec F) value h

/-- The crate's `sqr_n_mul` on the x86-64 backend, at a Pasta field: for a canonical `value`, as
it asserts, a canonical `res` with `R^(2^count) * res ≡ value^(2^count) * rhs (mod p)`. -/
theorem sqr_n_mul_entry_spec (F : PastaField) (value : Std.Array Std.U64 4#usize)
    (count : Std.Usize) (rhs : Std.Array Std.U64 4#usize)
    (h : isCanonical (limbsOfArray value) F.modulus = true) :
    montgomery.sqr_n_mul_with montgomeryBlocks value count rhs (limbsArray F.modulus)
      (word F.inv) ⦃ res =>
      Canonical F res (R^(2^count.val))
        ((limbsOfArray value).toNat^(2^count.val) * (limbsOfArray rhs).toNat) ⦄ :=
  sqr_n_mul_with_spec montgomeryBlocks F (montgomeryBlocks_spec F) value count rhs h

/-- The crate's `from_mont` on the x86-64 backend, at a Pasta field: for any operand, a canonical
`res` with `R * res ≡ value (mod p)`. -/
theorem from_mont_entry_spec (F : PastaField) (value : Std.Array Std.U64 4#usize) :
    montgomery.from_mont_with montgomeryBlocks value (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res R (limbsOfArray value).toNat ⦄ :=
  from_mont_with_spec montgomeryBlocks F (montgomeryBlocks_spec F) value

end EntryPoints

end PastaCurves.X86_64
