import PastaCurves.Semantics

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

namespace PastaCurves.AArch64

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

-- Boundary cases distinguish AArch64's carry convention from x86-64's borrow convention. Each
-- mirrors the example of the same operation in `PastaCurves.X86_64.Semantics`, named in its
-- comment: where x86-64's CF is `1` on borrow, AArch64's carry is `0`.
-- The instructions these examples test correspond as follows (x86-64, then AArch64):
-- - `sub`, `sbb` and `subs`, `sbcs`: the same difference, with complementary flags (`CF = 1 - C`).
-- - `mulx` and `umulh` with `mul`: the high and low words of the product.
-- - `cmovnc d, s` and `csel d, s, d, cs` (equivalently `csel d, d, s, lo`): both take `s`
--   exactly when the subtraction did not borrow.
-- - `neg r` and `subs xzr, r, #1`: the same flag value, set exactly when `r` is nonzero.
-- No borrow (`sbb 0 0 0 = (0, 0)`, `neg 0 = (0, 0)`): `subs` sets the carry.
example : subc 0 0 1 = (0, 1) := by decide +kernel
-- Borrow in and out (`sbb 0 0 1 = (2^64 - 1, 1)` and
-- `sbb (2^64 - 1) (2^64 - 1) 1 = (2^64 - 1, 1)`): `sbcs` with the carry clear clears it.
example : subc 0 0 0 = (2^64 - 1, 0) := by decide +kernel
example : subc (2^64 - 1) (2^64 - 1) 0 = (2^64 - 1, 0) := by decide +kernel
-- `0 - 1`, the subtraction that `neg 1 = (2^64 - 1, 1)` performs: it borrows, so the carry is
-- clear.
example : subc 0 1 1 = (2^64 - 1, 0) := by decide +kernel
-- The two words of a product (`mulx (2^64 - 1) 2 = (1, 2^64 - 2)`), by `umulh` and `mul`.
example : umulh (2^64 - 1) 2 = 1 ∧ mulLo (2^64 - 1) 2 = 2^64 - 2 := by decide +kernel
-- Selection on the carry (`cmovnc 0 3 5 = 5 ∧ cmovnc 1 3 5 = 3`), on either condition.
example : cselLo 0 3 5 = 3 ∧ cselLo 1 3 5 = 5 := by decide +kernel
example : cselCs 0 3 5 = 5 ∧ cselCs 1 3 5 = 3 := by decide +kernel
-- `subs xzr, r, #1`, which the multiplication and squaring blocks use to set the carry of the
-- low-limb cancellation: the carry is set exactly when `r` is nonzero.
example : (subc 0 1 1).2 = 0 ∧ (subc 1 1 1).2 = 1 ∧ (subc (2^64 - 1) 1 1).2 = 1 := by
  decide +kernel

end PastaCurves.AArch64
