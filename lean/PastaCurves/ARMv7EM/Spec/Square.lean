import PastaCurves.ARMv7EM.Spec.Mul

/-!
# ARMv7E-M Montgomery squaring

The pure ARMv7E-M backend models squaring as Montgomery multiplication with identical operands. It
is not a separate assembly-block proof.
-/

namespace PastaCurves.ARMv7EM

/-- Squaring by the pure ARMv7E-M composition, derived from its multiplication contract. -/
theorem sqrMont_spec (value modulus : Limbs) (inv : Nat) (hv : value.Bounded)
    (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hlow : modulus.l0 % wordMod = 1) (hinv_lt : inv < 2^64)
    (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hlt : value.toNat < modulus.toNat) :
    ∀ res, res = sqrMont value modulus inv →
      res.Bounded ∧ res.toNat < modulus.toNat ∧
        2^256 * res.toNat ≡ value.toNat * value.toNat [MOD modulus.toNat] := by
  intro res hres
  apply mulMont_spec_of_lhs_lt value value modulus inv hv hv hm hshape hlow hinv_lt hinv hlt res
  simpa [sqrMont] using hres

end PastaCurves.ARMv7EM
