import PastaCurves.ARMv7EM.Compositions.Mul.Entry
import PastaCurves.ARMv7EM.Spec.FromMont.Block

/-!
# ARMv7E-M generated Montgomery conversion

The Aeneas external calls the generated instruction transcription directly. The representation
lemma below connects its word arrays to the checked block theorem and the translated backend method.
-/

namespace PastaCurves.ARMv7EM.Compositions.Mul.Conversion

open _root_.Aeneas _root_.Aeneas.Std Result
open PastaCurves
open PastaCurves.Glue
open PastaCurves.ARMv7EM

/-- The generated ARMv7E-M `from_mont` method returns the checked instruction model's result on
arbitrary four-word input and modulus arrays. -/
theorem generated_from_mont_eq_block (value modulus : Std.Array Std.U64 4#usize)
    (inv : Std.U64) :
    pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesAsmEntryMontgomeryBlocks.from_mont
        value modulus inv =
      .ok (limbsArray (ARMv7EM.fromMontBlock (limbsOfArray value) (limbsOfArray modulus))) := by
  rfl

/-- At a Pasta field, the checked ARMv7E-M conversion block produces a canonical result for every
four-word input. -/
theorem fromMontBlock_canonical (F : PastaField) (hlow : F.modulus.l0 % wordMod = 1)
    (value : Std.Array Std.U64 4#usize) :
    LimbsCanonical F (ARMv7EM.fromMontBlock (limbsOfArray value) F.modulus)
      R (limbsOfArray value).toNat := by
  let result := ARMv7EM.fromMontBlock (limbsOfArray value) F.modulus
  have hblock : result.Bounded ∧ result.toNat < F.modulus.toNat ∧
      wordMod^8 * result.toNat ≡ (limbsOfArray value).toNat [MOD F.modulus.toNat] :=
    ARMv7EM.fromMontBlock_spec (limbsOfArray value) F.modulus
      (limbsOfArray_bounded value) F.bounded F.shape hlow result rfl
  change result.Bounded ∧ result.toNat < F.modulus.toNat ∧
    R * result.toNat ≡ (limbsOfArray value).toNat [MOD F.modulus.toNat]
  refine ⟨hblock.1, hblock.2.1, ?_⟩
  have hradix : R = wordMod^8 := by norm_num [R, wordMod]
  rw [hradix]
  exact hblock.2.2

/-- The translated ARMv7E-M `from_mont` method at a Pasta field returns, for every four-word input,
a canonical result whose multiplication by the Montgomery radix is congruent to the input. -/
theorem from_mont_entry_spec (F : PastaField) (hlow : F.modulus.l0 % wordMod = 1)
    (value : Std.Array Std.U64 4#usize) :
    pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesAsmEntryMontgomeryBlocks.from_mont
      value (limbsArray F.modulus) (word F.inv) ⦃ res =>
        Canonical F res R (limbsOfArray value).toNat ⦄ := by
  rw [generated_from_mont_eq_block, limbsOfArray_limbsArray F.bounded]
  exact Glue.ok_limbsArray_spec (fromMontBlock_canonical F hlow value)

end PastaCurves.ARMv7EM.Compositions.Mul.Conversion
