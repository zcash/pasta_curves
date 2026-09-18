/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.Semantics

/-!
# AArch64 instruction semantics for the Pasta Montgomery routines

The subset of AArch64 that the crate's routines and inline blocks use: `mul`, `umulh`,
`adds`/`adcs`/`adc`, `subs`/`sbcs`, `lsl`/`lsr` by an immediate, `csel` on the `lo` and `cs`
conditions, and `mov`. Loads and stores are not modelled as memory operations: the generated
programs read their operand limbs where the assembly loads them and return the limbs the assembly
stores.

AArch64's carry convention for subtraction is the one modelled here: after `subs`/`sbcs` the
carry is set exactly when no borrow occurred, so `sbcs` subtracts `1 - carry` and `subs`
behaves as `sbcs` with the carry set. The `lo` condition of `csel` (also written `cc`) is
"carry clear", and `cs` is "carry set".
-/

namespace PastaAsm.AArch64

/-- `subs` and `sbcs`: the low 64 bits of `a - b - (1 - c)` and the carry-out, where a carry of
`1` means that no borrow occurred. `subs` passes carry-in `1`. The difference is formed as
`a + 2^64 - b - (1 - c)`, which is nonnegative for in-range operands, so the quotient by `2^64`
is `1` exactly when `a ≥ b + (1 - c)`. -/
def subc (a b c : Nat) : Nat × Nat :=
  ((a + regMod - b - (1 - c)) % regMod, (a + regMod - b - (1 - c)) / regMod)

/-- `csel d, x, y, lo`: `x` when the carry is clear, else `y`. -/
def cselLo (c x y : Nat) : Nat := if c = 0 then x else y

/-- `csel d, x, y, cs`: `x` when the carry is set, else `y`. -/
def cselCs (c x y : Nat) : Nat := if c = 0 then y else x

end PastaAsm.AArch64
