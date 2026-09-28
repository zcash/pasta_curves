
/-!
# Generic semantics for the Pasta arithmetic routines

A register value is a natural number below `2^64`. The bound is maintained by construction:
every instruction reduces its result modulo `2^64`, and the carry flag is the quotient of the
same sum by `2^64`, so it is `0` or `1` whenever the inputs are in range. Working in `Nat`
rather than a fixed-width type keeps the proofs in `omega`'s fragment (`%` and `/` by
literals) and lets the reference vectors be checked by the kernel with `decide`.
-/

namespace PastaCurves

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

/-- Five words, the top one carrying the sign: the value is
`l0 + 2^64 l1 + 2^128 l2 + 2^192 l3 + 2^256 · (signed l4)`. The shape of the inversion's
signed intermediate values (`f`, `g`, and the row combinations of `u` and `v`). -/
structure Signed5 where
  /-- Word of weight `2^0`. -/
  l0 : Nat
  /-- Word of weight `2^64`. -/
  l1 : Nat
  /-- Word of weight `2^128`. -/
  l2 : Nat
  /-- Word of weight `2^192`. -/
  l3 : Nat
  /-- Word of weight `2^256`, read as a two's-complement word. -/
  l4 : Nat
  deriving DecidableEq, Repr

namespace Signed5

/-- The integer that the five words represent. -/
def toInt (x : Signed5) : Int :=
  x.l0 + 2^64 * x.l1 + 2^128 * x.l2 + 2^192 * x.l3
    + 2^256 * (if x.l4 < 2^63 then (x.l4 : Int) else (x.l4 : Int) - 2^64)

/-- Every word is below `2^64`. -/
def Bounded (x : Signed5) : Prop :=
  x.l0 < 2^64 ∧ x.l1 < 2^64 ∧ x.l2 < 2^64 ∧ x.l3 < 2^64 ∧ x.l4 < 2^64

end Signed5

end PastaCurves
