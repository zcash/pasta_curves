import PastaCurves.ARMv7EM.Compositions.Funs
import PastaCurves.Glue.Types
import PastaCurves.Portable.Driver

/-!
# The ARMv7E-M backend of the translated glue

These records use the methods that Aeneas generated from the ARMv7E-M implementation. Thus the
translated generic Montgomery and inversion compositions run the same Rust methods as the backend,
rather than separately written pure models. The addition and subtraction methods panic in the
translation; only multiplication, squaring, and Montgomery conversion meet the arithmetic contracts.
-/

namespace PastaCurves.ARMv7EM

/-- The generated ARMv7E-M methods as the translation's record of `MontgomeryBlocks`. -/
def montgomeryBlocks : pasta_curves.asm.entry.MontgomeryBlocks Unit where
  add := pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesAsmEntryMontgomeryBlocks.add
  sub := pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesAsmEntryMontgomeryBlocks.sub
  mul := pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesAsmEntryMontgomeryBlocks.mul
  square := pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesAsmEntryMontgomeryBlocks.square
  from_mont := pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesAsmEntryMontgomeryBlocks.from_mont

/-- The generated ARMv7E-M methods as the translation's record of inversion blocks. The three
methods delegated by the Rust implementation remain delegated in the generated definitions. -/
def inversionBlocks : pasta_curves.inversion.InvertBlocks Unit where
  divstep59 := pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.divstep59
  sign_mag := pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.sign_mag
  fg_row := pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.fg_row
  de_row := pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.de_row
  amontred := pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.amontred
  cond_sub := pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.cond_sub

/-- The generated ARMv7E-M inversion methods read back as total functions on the shared word-level
model. This projects the generated method record; it does not replace ARM-specific methods with the
portable backend. -/
def generatedInvertBlocks : InvertBlocks :=
  Portable.pureBlocks inversionBlocks

end PastaCurves.ARMv7EM
