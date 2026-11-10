import PastaCurves.ArithmeticChecks
import PastaCurves.X86_64.Vectors

/-!
# Kernel-checked arithmetic checks for the x86-64 transcription

The kernel evaluates the shared checks of `PastaCurves.ArithmeticChecks` on the x86-64 routines,
for both Pasta fields. x86-64 squares in two blocks, so the unreduced eight-word square of
`squareLo` is also checked on the same inputs.
-/

namespace PastaCurves.X86_64

/-- The x86-64 routines that the checks exercise. -/
def arithmeticBackend : ArithmeticBackend := ⟨vectorBackend, addMod, subMod⟩

example : arithmeticBackend.checks pallasBase = true := by decide +kernel
example : arithmeticBackend.checks vestaBase = true := by decide +kernel
example : arithmeticBackend.fromMontAllOnes pallasBase = true := by decide +kernel
example : arithmeticBackend.fromMontAllOnes vestaBase = true := by decide +kernel

/-- Whether `squareLo` is the exact square on every input of `checkCases F`. -/
private def squareLoChecks (F : PastaField) : Bool :=
  (checkCases F).all fun a => (squareLo a).toNat == a.toNat * a.toNat

example : squareLoChecks pallasBase = true := by decide +kernel
example : squareLoChecks vestaBase = true := by decide +kernel

end PastaCurves.X86_64
