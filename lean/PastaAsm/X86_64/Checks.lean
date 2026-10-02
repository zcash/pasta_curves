/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.Fields
import PastaAsm.X86_64.Compositions

/-!
# Kernel-checked arithmetic checks for the x86-64 transcription

These closed examples compare the transcribed programs with integer arithmetic for both
Pasta fields. They are regression checks, not substitutes for universally quantified
correctness theorems, and are not claimed to be hardware-generated vectors.
-/

namespace PastaAsm.X86_64

private def cases (F : PastaField) : List Limbs :=
  [Limbs.ofNat 0, Limbs.ofNat 1, Limbs.ofNat 2,
   Limbs.ofNat (2^64 - 1), Limbs.ofNat (2^128 - 1),
   Limbs.ofNat (F.modulus.toNat - 1)]

private def arithmeticChecks (F : PastaField) : Bool :=
  (cases F).all fun a =>
    ((sqrMont a F.modulus F.inv).toNat * 2^256 % F.modulus.toNat ==
      a.toNat * a.toNat % F.modulus.toNat) &&
    ((sqrMont a F.modulus F.inv).toNat < F.modulus.toNat) &&
    ((fromMont a F.modulus F.inv).toNat * 2^256 % F.modulus.toNat ==
      a.toNat % F.modulus.toNat) &&
    ((fromMont a F.modulus F.inv).toNat < F.modulus.toNat) &&
    ((squareLo a).toNat == a.toNat * a.toNat) &&
    (cases F).all fun b =>
      ((addMod a b F.modulus).toNat == (a.toNat + b.toNat) % F.modulus.toNat) &&
      ((subMod a b F.modulus).toNat == (a.toNat + F.modulus.toNat - b.toNat) % F.modulus.toNat) &&
      ((mulMont a b F.modulus F.inv).toNat * 2^256 % F.modulus.toNat ==
        a.toNat * b.toNat % F.modulus.toNat) &&
      ((mulMont a b F.modulus F.inv).toNat < F.modulus.toNat)

example : arithmeticChecks pallasBase = true := by decide +kernel
example : arithmeticChecks vestaBase = true := by decide +kernel

-- Conversion accepts arbitrary words, not just canonical field elements.
example :
    let value := Limbs.ofNat (2^256 - 1)
    let r := fromMont value pallasBase.modulus pallasBase.inv
    r.toNat < pallasBase.modulus.toNat ∧
      r.toNat * 2^256 % pallasBase.modulus.toNat = value.toNat % pallasBase.modulus.toNat := by
  decide +kernel

example :
    let value := Limbs.ofNat (2^256 - 1)
    let r := fromMont value vestaBase.modulus vestaBase.inv
    r.toNat < vestaBase.modulus.toNat ∧
      r.toNat * 2^256 % vestaBase.modulus.toNat = value.toNat % vestaBase.modulus.toNat := by
  decide +kernel

end PastaAsm.X86_64
