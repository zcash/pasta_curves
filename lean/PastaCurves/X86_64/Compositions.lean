import PastaCurves.X86_64.Transcription
import PastaCurves.Compositions

/-!
# The x86-64 Rust around the blocks

The x86-64 backend in `src/asm/x86_64.rs` implements `square` as
`square_hi(square_lo(*value), modulus, inv)`. The repeated-squaring loop in `src/asm/entry.rs`
runs it `count` times, then the multiplication block once. Unlike AArch64, conversion out of
Montgomery form has its own transcribed assembly block.
-/

namespace PastaCurves.X86_64

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

end PastaCurves.X86_64
