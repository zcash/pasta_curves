/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/

/-!
# Generic semantics for the Pasta arithmetic routines

A register value is a natural number below `2^64`. The bound is maintained by construction:
every instruction reduces its result modulo `2^64`, and the carry flag is the quotient of the
same sum by `2^64`, so it is `0` or `1` whenever the inputs are in range. Working in `Nat`
rather than a fixed-width type keeps the proofs in `omega`'s fragment (`%` and `/` by
literals) and lets the reference vectors be checked by the kernel with `decide`.
-/

namespace PastaAsm

/-- The register width, as the modulus of every register write. -/
abbrev regMod : Nat := 2^64

/-- `mul`: the low 64 bits of the product. -/
def mulLo (a b : Nat) : Nat := a * b % regMod

/-- `umulh`: the high 64 bits of the product. -/
def umulh (a b : Nat) : Nat := a * b / regMod

/-- `adds`, `adcs`, and `adc`: the low 64 bits of `a + b + c` and the carry-out. `adds` passes
carry-in `0`; `adc` discards the carry-out. -/
def addc (a b c : Nat) : Nat × Nat := ((a + b + c) % regMod, (a + b + c) / regMod)

/-- `lsl` by an immediate. -/
def lsl (a k : Nat) : Nat := a * 2^k % regMod

/-- `lsr` by an immediate. -/
def lsr (a k : Nat) : Nat := a / 2^k

/-- Four little-endian 64-bit limbs, the shape of every operand of the routines. -/
structure Limbs where
  /-- Limb of weight `2^0`. -/
  l0 : Nat
  /-- Limb of weight `2^64`. -/
  l1 : Nat
  /-- Limb of weight `2^128`. -/
  l2 : Nat
  /-- Limb of weight `2^192`. -/
  l3 : Nat
  deriving DecidableEq, Repr

namespace Limbs

/-- The integer that a limb vector represents. -/
def toNat (x : Limbs) : Nat := x.l0 + 2^64 * x.l1 + 2^128 * x.l2 + 2^192 * x.l3

/-- The limbs of a natural number; bits at and above `2^256` are dropped. -/
def ofNat (n : Nat) : Limbs :=
  ⟨n % 2^64, n / 2^64 % 2^64, n / 2^128 % 2^64, n / 2^192 % 2^64⟩

/-- Every limb is below `2^64`. -/
def Bounded (x : Limbs) : Prop := x.l0 < 2^64 ∧ x.l1 < 2^64 ∧ x.l2 < 2^64 ∧ x.l3 < 2^64

end Limbs

end PastaAsm
