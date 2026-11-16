import PastaCurves.ARMv7EM.Compositions.Mul.Entry
import PastaCurves.ARMv7EM.Compositions.Mul.Conversion

/-!
# ARMv7E-M generated squaring and Montgomery conversion

The generated ARMv7E-M backend dispatches `square` to `mul value value` and `from_mont` to
the dedicated conversion block. Squaring follows from `mul_entry_spec`; conversion follows from the
checked `ARMv7EM.fromMontBlock_spec` through the representation bridge in `Conversion.lean`.
Squaring requires a canonical operand, while conversion accepts every four-limb value.

Aeneas translates these backend methods, but not the generic `square_with` and `from_mont_with`
wrappers in `src/asm/entry.rs`.  Thus the statements below cover the actual translated backend
dispatch; the square hypothesis is exactly the assertion made by its untranslated wrapper.
-/

namespace PastaCurves.ARMv7EM.Compositions.Mul.SquareConversion

open _root_.Aeneas _root_.Aeneas.Std
open PastaCurves
open PastaCurves.Glue
open PastaCurves.ARMv7EM
open PastaCurves.ARMv7EM.Compositions.Mul.Entry

/-- The generated ARMv7E-M `square` dispatch at a field whose low 32-bit modulus word is one.
The operand condition is exactly the canonicity condition asserted by Rust's `square_with`. -/
theorem square_entry_spec (F : PastaField) (hlow : F.modulus.l0 % wordMod = 1)
    (value : Std.Array Std.U64 4#usize)
    (hcanonical : isCanonical (limbsOfArray value) F.modulus = true) :
    pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesAsmEntryMontgomeryBlocks.square
      value (limbsArray F.modulus) (word F.inv) ⦃ res =>
        Canonical F res R ((limbsOfArray value).toNat * (limbsOfArray value).toNat) ⦄ := by
  unfold pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesAsmEntryMontgomeryBlocks.square
  apply mul_entry_spec F hlow value value
  simp [mulContract, hcanonical]

/-- The generated ARMv7E-M `from_mont` dispatch for every four-limb `U64` input. -/
theorem from_mont_entry_spec (F : PastaField) (hlow : F.modulus.l0 % wordMod = 1)
    (value : Std.Array Std.U64 4#usize) :
    pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesAsmEntryMontgomeryBlocks.from_mont
      value (limbsArray F.modulus) (word F.inv) ⦃ res =>
        Canonical F res R (limbsOfArray value).toNat ⦄ :=
  Conversion.from_mont_entry_spec F hlow value

end PastaCurves.ARMv7EM.Compositions.Mul.SquareConversion
