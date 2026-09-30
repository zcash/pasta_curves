import PastaAsm.Compositions
import PastaAsm.AArch64.Transcription

/-!
# The crate's Rust around the blocks

`src/asm/mod.rs` composes the crate's `sqr_n_mul` and `from_mont` from the `square` and `mul`
blocks. These definitions mirror that Rust: the two compositions.
-/

namespace PastaAsm.AArch64

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

end PastaAsm.AArch64
