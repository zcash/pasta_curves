/-
Copyright (c) 2026 the pasta-aarch64-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAArch64Asm.Transcription

/-!
# The crate's Rust around the blocks

`src/asm/mod.rs` composes the crate's `sqr_n_mul` and `from_mont` from the `square` and `mul`
blocks, and in a debug build checks the operand contracts of `mul` and `square` before entering
the blocks. These definitions mirror that Rust: the two compositions, the limb comparison
`is_canonical`, and the condition that `mul` asserts.
-/

namespace PastaAArch64Asm

/-- `from_mont`: the multiplication block with `1` as its right operand, `value * 2^-256 mod p`.
-/
def fromMont (value modulus : Limbs) (inv : Nat) : Limbs :=
  mulMont value ⟨1, 0, 0, 0⟩ modulus inv

/-- The squaring block applied `count` times. -/
def sqrN (value modulus : Limbs) (inv : Nat) : Nat → Limbs
  | 0 => value
  | count + 1 => sqrMont (sqrN value modulus inv count) modulus inv

/-- `sqr_n_mul`: the squaring block `count` times, then the multiplication block by `rhs`. -/
def sqrNMul (value : Limbs) (count : Nat) (rhs modulus : Limbs) (inv : Nat) : Limbs :=
  mulMont (sqrN value modulus inv count) rhs modulus inv

/-- The crate's `is_canonical`: whether `value < modulus`, comparing the limbs from the most
significant down, as `src/asm/mod.rs` does. -/
def isCanonical (value modulus : Limbs) : Bool :=
  if value.l3 ≠ modulus.l3 then decide (value.l3 < modulus.l3)
  else if value.l2 ≠ modulus.l2 then decide (value.l2 < modulus.l2)
  else if value.l1 ≠ modulus.l1 then decide (value.l1 < modulus.l1)
  else if value.l0 ≠ modulus.l0 then decide (value.l0 < modulus.l0)
  else false

/-- The condition that the crate's `mul` asserts in a debug build: a canonical `lhs`, or a
canonical `rhs` whose limbs 1 to 3 are at most `2^64 - 3`. -/
def mulContract (lhs rhs modulus : Limbs) : Bool :=
  isCanonical lhs modulus ||
    (isCanonical rhs modulus &&
      decide (rhs.l1 ≤ 2^64 - 3) && decide (rhs.l2 ≤ 2^64 - 3) && decide (rhs.l3 ≤ 2^64 - 3))

end PastaAArch64Asm
