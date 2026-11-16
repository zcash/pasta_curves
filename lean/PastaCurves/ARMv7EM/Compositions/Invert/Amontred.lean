import PastaCurves.ARMv7EM.Compositions.Funs
import PastaCurves.Portable.Amontred

/-!
# ARMv7E-M almost-Montgomery reduction helper correspondence

The ARMv7E-M translation calls a hand-written external model of the portable
`shifted_modulus` helper. The theorems below identify that model with the generated portable
translation and transfer its arithmetic specification.
-/

namespace PastaCurves.ARMv7EM.Compositions.Invert.Amontred

open _root_.Aeneas _root_.Aeneas.Std
open PastaCurves

/-- The hand-written external model of `shifted_modulus` is definitionally the same computation as
the generated portable translation. -/
theorem shifted_modulus_eq (modulus : Std.Array Std.U64 4#usize) :
    inversion.portable.shifted_modulus modulus =
      pasta_curves.inversion.portable.shifted_modulus modulus := by
  rfl

/-- The hand-written external `shifted_modulus` returns the five words of the modulus shifted left
by 61 bits. -/
@[step]
theorem shifted_modulus_spec (modulus : Std.Array Std.U64 4#usize) :
    inversion.portable.shifted_modulus modulus ⦃ res =>
      (signed5OfArray res).toNat = 2^61 * (limbsOfArray modulus).toNat ⦄ := by
  rw [shifted_modulus_eq]
  exact Portable.shifted_modulus_spec modulus

end PastaCurves.ARMv7EM.Compositions.Invert.Amontred
