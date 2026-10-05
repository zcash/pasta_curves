import PastaCurves.Inversion.Model
import PastaCurves.Inversion.HullCert

/-!
# The correctness theorem, unconditionally

`montInv_spec` (`Model.lean`) proves Theorem 12 of `book/src/design/inversion.md` with the
termination bound for `256` bits as a hypothesis, and `terminationBound_256` (`HullCert.lean`)
proves that bound from the hull certificate. Neither module imports the other, so the two meet
here: `montInv_correct` is Theorem 12 with no hypothesis beyond the input being canonical.
-/

namespace PastaCurves.Inversion

/-- Theorem 12, unconditionally: for a canonical input `x`, the model returns the canonical
Montgomery residue `z` with `x z ≡ R^2 (mod p)`, and zero for zero. -/
theorem montInv_correct (F : PastaField) (x : Limbs) (hx : x.Bounded)
    (hxlt : x.toNat < F.modulus.toNat) :
    (montInvModel F x).Bounded ∧ (montInvModel F x).toNat < F.modulus.toNat ∧
      (x.toNat = 0 → montInvModel F x = Limbs.ofNat 0) ∧
      (x.toNat ≠ 0 → x.toNat * (montInvModel F x).toNat ≡ R^2 [MOD F.modulus.toNat]) :=
  montInv_spec F Hull.terminationBound_256 x hx hxlt

end PastaCurves.Inversion
