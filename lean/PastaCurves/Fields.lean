import PastaCurves.Semantics
import PastaCurves.Primality

/-!
# The crate's two fields

The crate's routines take the modulus limbs and `inv` from the caller, for either Pasta base
field. This module states those constants once, with the facts about them that the theorems in
`AArch64/Spec.lean` assume: the limbs are `[p0, p1, 0, 2^62]` and `inv * p0 ≡ -1 (mod 2^64)`. Each
fact is closed by `decide`, and two examples check that the limbs encode the primes as pasta_curves
states them. The modulus is prime, by the Pratt certificates of `Primality.lean`. The vectors in
`AArch64/Vectors.lean` exercise both fields' constants against the hardware outputs.
-/

namespace PastaCurves

/-- The Montgomery radix, one more than the largest four-limb value: the crate holds a residue
`x` as `x * R mod p`, and the entry-point theorems state their results against it. -/
abbrev R : Nat := 2^256

/-- A Pasta base field as the crate takes it: the modulus limbs and `inv`, with the facts about
them that the proofs assume. -/
structure PastaField where
  /-- The modulus, as four little-endian limbs. -/
  modulus : Limbs
  /-- `-p^-1 mod 2^64`, the Montgomery constant. -/
  inv : Nat
  /-- Every limb of the modulus is below `2^64`. -/
  bounded : modulus.Bounded
  /-- The shape the blocks hard-code: limb 2 is `0` and limb 3 is `2^62`. -/
  shape : modulus.l2 = 0 ∧ modulus.l3 = 2^62
  /-- `inv` is a 64-bit value. -/
  inv_lt : inv < 2^64
  /-- `inv * p0 ≡ -1 (mod 2^64)`, which is what makes the Montgomery cancellation work. -/
  inv_spec : (inv * modulus.l0 + 1) % 2^64 = 0
  /-- The modulus is prime. -/
  prime : Nat.Prime modulus.toNat

/-- The Pallas base field, `pasta_curves::Fp`: its `MODULUS` limbs and `INV` in `src/fields/fp.rs`,
which `FieldTypes.lean` checks against these. -/
def pallasBase : PastaField where
  modulus := ⟨0x992d30ed00000001, 0x224698fc094cf91b, 0, 0x4000000000000000⟩
  inv := 0x992d30ecffffffff
  bounded := by unfold Limbs.Bounded; decide
  shape := by decide
  inv_lt := by decide
  inv_spec := by decide
  prime := Pratt.pallasBase_prime

-- The limbs encode `p` as pasta_curves states it.
example : pallasBase.modulus.toNat =
    0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001 := by decide

/-- The Vesta base field, `pasta_curves::Fq`: its `MODULUS` limbs and `INV` in `src/fields/fq.rs`,
which `FieldTypes.lean` checks against these. -/
def vestaBase : PastaField where
  modulus := ⟨0x8c46eb2100000001, 0x224698fc0994a8dd, 0, 0x4000000000000000⟩
  inv := 0x8c46eb20ffffffff
  bounded := by unfold Limbs.Bounded; decide
  shape := by decide
  inv_lt := by decide
  inv_spec := by decide
  prime := Pratt.vestaBase_prime

-- The limbs encode `q` as pasta_curves states it.
example : vestaBase.modulus.toNat =
    0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001 := by decide

end PastaCurves
