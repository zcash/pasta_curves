import PastaCurves.VectorCheck

/-!
# Kernel-checked arithmetic checks, shared by the backends

These closed checks compare a backend's routines with integer arithmetic for a Pasta field, on a
few boundary inputs. They are regression checks, not substitutes for universally quantified
correctness theorems, and are not hardware-generated vectors. The reference vectors exercise
multiplication, squaring, and conversion; these checks also exercise addition and subtraction.

A backend supplies its five routines as an `ArithmeticBackend`, and its `Checks.lean` has the
kernel evaluate `ArithmeticBackend.checks` and `ArithmeticBackend.fromMontAllOnes` at both fields.
-/

namespace PastaCurves

/-- The routines that the checks exercise: the three of `VectorBackend`, with modular addition and
subtraction, each taking the modulus limbs last. -/
structure ArithmeticBackend extends VectorBackend where
  addMod : Limbs → Limbs → Limbs → Limbs
  subMod : Limbs → Limbs → Limbs → Limbs

/-- The inputs of the checks: the first words, the limb boundaries, and the largest canonical
element of `F`. -/
def checkCases (F : PastaField) : List Limbs :=
  [Limbs.ofNat 0, Limbs.ofNat 1, Limbs.ofNat 2,
   Limbs.ofNat (2^64 - 1), Limbs.ofNat (2^128 - 1),
   Limbs.ofNat (F.modulus.toNat - 1)]

namespace ArithmeticBackend

/-- Whether every routine agrees with integer arithmetic modulo `F` on every input, and every pair
of inputs, of `checkCases F`, and whether the Montgomery results are reduced. -/
def checks (B : ArithmeticBackend) (F : PastaField) : Bool :=
  (checkCases F).all fun a =>
    ((B.sqrMont a F.modulus F.inv).toNat * 2^256 % F.modulus.toNat ==
      a.toNat * a.toNat % F.modulus.toNat) &&
    ((B.sqrMont a F.modulus F.inv).toNat < F.modulus.toNat) &&
    ((B.fromMont a F.modulus F.inv).toNat * 2^256 % F.modulus.toNat ==
      a.toNat % F.modulus.toNat) &&
    ((B.fromMont a F.modulus F.inv).toNat < F.modulus.toNat) &&
    (checkCases F).all fun b =>
      ((B.addMod a b F.modulus).toNat == (a.toNat + b.toNat) % F.modulus.toNat) &&
      ((B.subMod a b F.modulus).toNat ==
        (a.toNat + F.modulus.toNat - b.toNat) % F.modulus.toNat) &&
      ((B.mulMont a b F.modulus F.inv).toNat * 2^256 % F.modulus.toNat ==
        a.toNat * b.toNat % F.modulus.toNat) &&
      ((B.mulMont a b F.modulus F.inv).toNat < F.modulus.toNat)

/-- Conversion accepts arbitrary words, not just canonical field elements: on the all-ones word it
returns a reduced result with the right residue. -/
def fromMontAllOnes (B : ArithmeticBackend) (F : PastaField) : Bool :=
  let value := Limbs.ofNat (2^256 - 1)
  let r := B.fromMont value F.modulus F.inv
  r.toNat < F.modulus.toNat &&
    r.toNat * 2^256 % F.modulus.toNat == value.toNat % F.modulus.toNat

end ArithmeticBackend

end PastaCurves
