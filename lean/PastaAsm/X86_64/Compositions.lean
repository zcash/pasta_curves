/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.X86_64.Transcription
import PastaAsm.Compositions

/-!
# The x86-64 Rust around the blocks

`src/asm/x86_64.rs` implements `square` as `square_hi(square_lo(*value), modulus, inv)`.
Its repeated-squaring loop uses the same pair, then calls the multiplication block once.
Unlike AArch64, conversion out of Montgomery form has its own transcribed assembly block.

The backend has additional debug assertions beyond the public wrappers in `src/asm/mod.rs`:
`mul` checks a canonical right operand, and `sqr_n_mul` checks both its input and its right
operand. The contract definitions below record these checks, not correctness claims about
inputs outside them. Limb boundedness is a separate hypothesis, modeling Rust's `u64` type.
-/

namespace PastaAsm.X86_64

/-- `square`: the full eight-word product followed by Montgomery reduction. -/
def sqrMont (value modulus : Limbs) (inv : Nat) : Limbs :=
  squareHi (squareLo value) modulus inv

/-- The two squaring blocks applied `count` times. -/
def sqrN (value modulus : Limbs) (inv : Nat) : Nat → Limbs
  | 0 => value
  | count + 1 => sqrMont (sqrN value modulus inv count) modulus inv

/-- `sqr_n_mul`: the squaring pair `count` times, then the multiplication block by `rhs`. -/
def sqrNMul (value : Limbs) (count : Nat) (rhs modulus : Limbs) (inv : Nat) : Limbs :=
  mulMont (sqrN value modulus inv count) rhs modulus inv

/-- Both the public multiplication wrapper's check and the x86-64 backend's check. -/
def mulEntryContract (lhs rhs modulus : Limbs) : Bool :=
  mulContract lhs rhs modulus && isCanonical rhs modulus

/-- The x86-64 repeated-squaring entry point checks both operands, even when `count = 0`. -/
def sqrNMulEntryContract (value rhs modulus : Limbs) : Bool :=
  isCanonical value modulus && isCanonical rhs modulus

end PastaAsm.X86_64
