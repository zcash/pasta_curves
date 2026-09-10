/-
Copyright (c) 2026 the pasta-aarch64-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/

/-!
# AArch64 instruction semantics for the Pasta Montgomery routines

The subset of AArch64 that the crate's routines and inline blocks use: `mul`, `umulh`,
`adds`/`adcs`/`adc`, `subs`/`sbcs`, `lsl`/`lsr` by an immediate, `csel` on the `lo` condition,
and `mov`. Loads and stores are not modelled as memory operations: the generated programs read
their operand limbs where the assembly loads them and return the limbs the assembly stores.

A register value is a natural number below `2^64`. The bound is maintained by construction:
every instruction reduces its result modulo `2^64`, and the carry flag is the quotient of the
same sum by `2^64`, so it is `0` or `1` whenever the inputs are in range. Working in `Nat`
rather than a fixed-width type keeps the proofs in `omega`'s fragment (`%` and `/` by
literals) and lets the reference vectors be checked by the kernel with `decide`.

AArch64's carry convention for subtraction is the one modelled here: after `subs`/`sbcs` the
carry is set exactly when no borrow occurred, so `sbcs` subtracts `1 - carry` and `subs`
behaves as `sbcs` with the carry set. The `lo` condition of `csel` is "carry clear".
-/

namespace PastaAArch64Asm

/-- The register width, as the modulus of every register write. -/
abbrev regMod : Nat := 2^64

/-- `mul`: the low 64 bits of the product. -/
def mulLo (a b : Nat) : Nat := a * b % regMod

/-- `umulh`: the high 64 bits of the product. -/
def umulh (a b : Nat) : Nat := a * b / regMod

/-- `adds`, `adcs`, and `adc`: the low 64 bits of `a + b + c` and the carry-out. `adds` passes
carry-in `0`; `adc` discards the carry-out. -/
def addc (a b c : Nat) : Nat × Nat := ((a + b + c) % regMod, (a + b + c) / regMod)

/-- `subs` and `sbcs`: the low 64 bits of `a - b - (1 - c)` and the carry-out, where a carry of
`1` means that no borrow occurred. `subs` passes carry-in `1`. The difference is formed as
`a + 2^64 - b - (1 - c)`, which is nonnegative for in-range operands, so the quotient by `2^64`
is `1` exactly when `a ≥ b + (1 - c)`. -/
def subc (a b c : Nat) : Nat × Nat :=
  ((a + regMod - b - (1 - c)) % regMod, (a + regMod - b - (1 - c)) / regMod)

/-- `lsl` by an immediate. -/
def lsl (a k : Nat) : Nat := a * 2^k % regMod

/-- `lsr` by an immediate. -/
def lsr (a k : Nat) : Nat := a / 2^k

/-- `csel d, x, y, lo`: `x` when the carry is clear, else `y`. -/
def cselLo (c x y : Nat) : Nat := if c = 0 then x else y

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

end PastaAArch64Asm
