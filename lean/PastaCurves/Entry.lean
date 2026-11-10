import PastaCurves.Fields
import PastaCurves.Glue.Spec
import PastaCurves.Spec
import PastaCurves.Inversion.Composition

/-!
# The crate's entry points on a backend

`EntrySpecs M I` states what the crate's entry points compute at either of the crate's fields (a
`PastaField`) when they run over a backend's record `M` of Montgomery blocks and its record `I` of
inversion blocks, each under the conditions that the entry point asserts in a debug build:

* `add`, `sub`, `mul`, `square`, `sqr_n_mul`, and `from_mont` are Aeneas' translation of the generic
  compositions in `src/asm/entry.rs`, run over `M`;
* `invert` is the composition of `Compositions.lean`, which mirrors `src/inversion.rs`, run over
  `I`.

`EntrySpecs.of` proves all seven from the two records' contracts at every field: `Glue.BlocksSpec`
through the theorems of `Glue/Spec.lean`, and `InvertBlocks.Spec` through `invert_entry_spec` of
`Inversion/Composition.lean`. Each architecture's `Entry.lean` proves its records' contracts and
instantiates the bundle. The results are stated against the Montgomery radix `R = 2^256` of
`Fields.lean`.
-/

namespace PastaCurves

open Aeneas Aeneas.Std pasta_curves Glue

/-- The crate's entry points over the records `M` and `I`, at every Pasta field. -/
structure EntrySpecs (M : asm.entry.MontgomeryBlocks Unit) (I : InvertBlocks) : Prop where
  /-- The crate's `add`: for canonical operands, as it asserts, the canonical sum. -/
  add : ∀ (F : PastaField) (lhs rhs : Std.Array Std.U64 4#usize),
    isCanonical (limbsOfArray lhs) F.modulus = true →
    isCanonical (limbsOfArray rhs) F.modulus = true →
    asm.entry.add_with M lhs rhs (limbsArray F.modulus) ⦃ res =>
      Canonical F res 1 ((limbsOfArray lhs).toNat + (limbsOfArray rhs).toNat) ⦄
  /-- The crate's `sub`: for canonical operands, as it asserts, the canonical difference. -/
  sub : ∀ (F : PastaField) (lhs rhs : Std.Array Std.U64 4#usize),
    isCanonical (limbsOfArray lhs) F.modulus = true →
    isCanonical (limbsOfArray rhs) F.modulus = true →
    asm.entry.sub_with M lhs rhs (limbsArray F.modulus) ⦃ res =>
      (limbsOfArray res).toNat < F.modulus.toNat ∧
        (limbsOfArray res).toNat + (limbsOfArray rhs).toNat ≡ (limbsOfArray lhs).toNat
          [MOD F.modulus.toNat] ⦄
  /-- The crate's `mul`: under the contract that it asserts, a canonical `res` with
  `R * res ≡ lhs * rhs (mod p)`. -/
  mul : ∀ (F : PastaField) (lhs rhs : Std.Array Std.U64 4#usize),
    mulContract (limbsOfArray lhs) (limbsOfArray rhs) F.modulus = true →
    asm.entry.mul_with M lhs rhs (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res R ((limbsOfArray lhs).toNat * (limbsOfArray rhs).toNat) ⦄
  /-- The crate's `square`: for a canonical operand, as it asserts, a canonical `res` with
  `R * res ≡ value² (mod p)`. -/
  square : ∀ (F : PastaField) (value : Std.Array Std.U64 4#usize),
    isCanonical (limbsOfArray value) F.modulus = true →
    asm.entry.square_with M value (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res R ((limbsOfArray value).toNat * (limbsOfArray value).toNat) ⦄
  /-- The crate's `sqr_n_mul`: for a canonical `value`, as it asserts, a canonical `res` with
  `R^(2^count) * res ≡ value^(2^count) * rhs (mod p)`. -/
  sqr_n_mul : ∀ (F : PastaField) (value : Std.Array Std.U64 4#usize) (count : Std.Usize)
    (rhs : Std.Array Std.U64 4#usize),
    isCanonical (limbsOfArray value) F.modulus = true →
    asm.entry.sqr_n_mul_with M value count rhs (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res (R^(2^count.val))
        ((limbsOfArray value).toNat^(2^count.val) * (limbsOfArray rhs).toNat) ⦄
  /-- The crate's `from_mont`: for any operand, a canonical `res` with
  `R * res ≡ value (mod p)`. -/
  from_mont : ∀ (F : PastaField) (value : Std.Array Std.U64 4#usize),
    asm.entry.from_mont_with M value (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res R (limbsOfArray value).toNat ⦄
  /-- The crate's `invert`: the input is canonical, as the entry point asserts, and `e0` is
  `2^562 mod p`, as its contract requires. The result is canonical. For `x = 0` it is `0`;
  otherwise it is the Montgomery inverse, with `x * result ≡ R^2 (mod p)`. -/
  invert : ∀ (F : PastaField) (x : Limbs), x.Bounded →
    isCanonical x F.modulus = true → ∀ (e0 : Limbs), e0 = Inversion.startE F →
    (invert I x F.modulus F.inv e0).Bounded ∧
      (invert I x F.modulus F.inv e0).toNat < F.modulus.toNat ∧
      (x.toNat = 0 → invert I x F.modulus F.inv e0 = Limbs.ofNat 0) ∧
      (x.toNat ≠ 0 →
        x.toNat * (invert I x F.modulus F.inv e0).toNat ≡ R^2 [MOD F.modulus.toNat])

/-- A backend whose records meet their contracts at every Pasta field runs every entry point
correctly. -/
theorem EntrySpecs.of {M : asm.entry.MontgomeryBlocks Unit} {I : InvertBlocks}
    (hM : ∀ F : PastaField, BlocksSpec M F) (hI : ∀ F : PastaField, I.Spec F) :
    EntrySpecs M I where
  add F lhs rhs hl hr := add_with_spec M F (hM F) lhs rhs hl hr
  sub F lhs rhs hl hr := sub_with_spec M F (hM F) lhs rhs hl hr
  mul F lhs rhs h := mul_with_spec M F (hM F) lhs rhs h
  square F value h := square_with_spec M F (hM F) value h
  sqr_n_mul F value count rhs h := sqr_n_mul_with_spec M F (hM F) value count rhs h
  from_mont F value := from_mont_with_spec M F (hM F) value
  invert F x hx h e0 he0 := invert_entry_spec I F (hI F) x hx h e0 he0

end PastaCurves
