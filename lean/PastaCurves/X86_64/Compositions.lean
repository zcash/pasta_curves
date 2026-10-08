import PastaCurves.X86_64.Transcription
import PastaCurves.Compositions

/-!
# The x86-64 Rust around the blocks

The x86-64 backend in `src/asm/x86_64.rs` implements `square` as
`square_hi(square_lo(*value), modulus, inv)`, which `sqrMont` mirrors. Unlike AArch64, conversion
out of Montgomery form has its own transcribed assembly block.
-/

namespace PastaCurves.X86_64

/-- `square`: the full eight-word product followed by Montgomery reduction. -/
def sqrMont (value modulus : Limbs) (inv : Nat) : Limbs :=
  squareHi (squareLo value) modulus inv

end PastaCurves.X86_64
