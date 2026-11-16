import PastaCurves.X86_64.Transcription
import PastaCurves.Compositions

/-!
# The x86-64 Rust around the blocks

The x86-64 backend in `src/asm/x86_64.rs` implements `square` as
`square_hi(square_lo(*value), modulus, inv)`, which `sqrMont` mirrors. Unlike AArch64, conversion
out of Montgomery form has its own transcribed assembly block. `src/inversion.rs` composes `invert`
from a backend's six blocks, which `invert` of `PastaCurves/Compositions.lean` mirrors over a record
of the blocks, and `invertBlocks` supplies the x86-64 blocks.
-/

namespace PastaCurves.X86_64

/-- `square`: the full eight-word product followed by Montgomery reduction. -/
def sqrMont (value modulus : Limbs) (inv : Nat) : Limbs :=
  squareHi (squareLo value) modulus inv

/-! ## The inversion -/

/-- The x86-64 transcriptions of the inversion's six blocks, as `src/inversion.rs` composes
them. -/
def invertBlocks : InvertBlocks :=
  ⟨divstep59Block, signMagBlock, fgRowBlock, deRowBlock, amontredBlock, condSubBlock⟩

end PastaCurves.X86_64
