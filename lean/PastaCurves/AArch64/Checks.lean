import PastaCurves.ArithmeticChecks
import PastaCurves.AArch64.Vectors

/-!
# Kernel-checked arithmetic checks for the AArch64 transcription

The kernel evaluates the shared checks of `PastaCurves.ArithmeticChecks` on the AArch64 routines,
for both Pasta fields. Unlike x86-64, AArch64 squares and reduces in one block, so there is no
unreduced square to check.
-/

namespace PastaCurves.AArch64

/-- The AArch64 routines that the checks exercise. -/
def arithmeticBackend : ArithmeticBackend := ⟨vectorBackend, addMod, subMod⟩

example : arithmeticBackend.checks pallasBase = true := by decide +kernel
example : arithmeticBackend.checks vestaBase = true := by decide +kernel
example : arithmeticBackend.fromMontAllOnes pallasBase = true := by decide +kernel
example : arithmeticBackend.fromMontAllOnes vestaBase = true := by decide +kernel

end PastaCurves.AArch64
