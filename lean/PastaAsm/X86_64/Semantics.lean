/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.Semantics

/-!
# x86-64 instruction semantics for the Pasta Montgomery routines

The subset of x86-64 that the crate's routines and inline blocks use. The definitions here
model the values and flags consumed by the blocks, not the entire architectural state.
`addc` supplies the result and unsigned carry of `add`, `adc`, `adcx`, and `adox`: the latter
uses OF as an unsigned carry chain, independently of CF. A transcription must keep these two
chains separate. `adcx` preserves OF, `adox` preserves CF, and `mulx` and moves preserve both.

Unlike AArch64's subtraction carry, x86-64 CF is set when subtraction borrows. `sub` passes
zero to `sbb`; `neg` sets CF exactly when its operand is nonzero. `cmovnc` replaces its
old destination only when CF is clear, without changing flags.

The low product of two-operand `imul` is `mulLo`, and the word results of immediate `shl`
and `shr` are `lsl` and `lsr`. Their flag writes must not be mistaken for preservation:
a generator may omit a flag result only when it checks that the result is overwritten before
being read. In particular, OF is undefined after the multi-bit shifts used by these routines.
The zeroing idiom `xor r, r` writes zero and clears both CF and OF, including when written to
a 32-bit subregister (which zero-extends to the full register).

Loads are modeled as reads of the corresponding input limb, not as memory operations. This
leaves pointer validity, register allocation, and the inline-assembly operand bindings in the
same external trust boundary as the AArch64 transcription.
-/

namespace PastaAsm.X86_64

/-- `sub` and `sbb`: the low word of `a - b - borrow` and CF, which is `1` on borrow.
For bounded operands and `borrow ≤ 1`, adding `2^64` keeps the natural subtraction from
truncating; its quotient is `1` exactly when the original subtraction did not borrow. -/
def sbb (a b borrow : Nat) : Nat × Nat :=
  ((a + regMod - b - borrow) % regMod, 1 - (a + regMod - b - borrow) / regMod)

/-- `neg`: the low word of `-a` and CF. -/
def neg (a : Nat) : Nat × Nat := ((sbb 0 a 0).1, if a = 0 then 0 else 1)

/-- `cmovnc dst, src`: keep `dst` on borrow, replace it with `src` when CF is clear. -/
def cmovnc (cf dst src : Nat) : Nat := if cf = 0 then src else dst

/-- `mulx hi, lo, src` in Intel syntax, with implicit multiplicand RDX.
The pair is in destination order, high word first; neither CF nor OF changes. -/
def mulx (rdx src : Nat) : Nat × Nat := (umulh rdx src, mulLo rdx src)

/-- Eight little-endian words, the output of the squaring block before Montgomery reduction. -/
structure WideLimbs where
  l0 : Nat
  l1 : Nat
  l2 : Nat
  l3 : Nat
  l4 : Nat
  l5 : Nat
  l6 : Nat
  l7 : Nat
  deriving DecidableEq, Repr

/-- The integer represented by the eight words. -/
def WideLimbs.toNat (x : WideLimbs) : Nat :=
  x.l0 + 2^64 * x.l1 + 2^128 * x.l2 + 2^192 * x.l3 +
    2^256 * x.l4 + 2^320 * x.l5 + 2^384 * x.l6 + 2^448 * x.l7

/-- Every word of the unreduced product is below `2^64`. -/
def WideLimbs.Bounded (x : WideLimbs) : Prop :=
  x.l0 < 2^64 ∧ x.l1 < 2^64 ∧ x.l2 < 2^64 ∧ x.l3 < 2^64 ∧
    x.l4 < 2^64 ∧ x.l5 < 2^64 ∧ x.l6 < 2^64 ∧ x.l7 < 2^64

-- Boundary cases distinguish x86-64's borrow convention from AArch64's carry convention.
example : sbb 0 0 0 = (0, 0) := by decide +kernel
example : sbb 0 0 1 = (2^64 - 1, 1) := by decide +kernel
example : sbb (2^64 - 1) (2^64 - 1) 1 = (2^64 - 1, 1) := by decide +kernel
example : neg 0 = (0, 0) := by decide +kernel
example : neg 1 = (2^64 - 1, 1) := by decide +kernel
example : mulx (2^64 - 1) 2 = (1, 2^64 - 2) := by decide +kernel
example : cmovnc 0 3 5 = 5 ∧ cmovnc 1 3 5 = 3 := by decide +kernel

end PastaAsm.X86_64
