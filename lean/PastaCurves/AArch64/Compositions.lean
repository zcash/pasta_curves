import PastaCurves.Compositions
import PastaCurves.AArch64.Transcription

/-!
# The AArch64 Rust around the blocks

The AArch64 backend in `src/asm/aarch64.rs` implements `from_mont` as the `mul` block with one,
which `fromMont` mirrors. `src/inversion.rs` composes `invert` from a backend's six blocks, which
`invert` of `PastaCurves/Compositions.lean` mirrors over a record of the blocks, and
`invertBlocks` supplies the AArch64 blocks. The entry points' compositions of the Montgomery blocks
are Aeneas' translation in `Glue/Funs.lean`, run over the record in `AArch64/Backend.lean`.
-/

namespace PastaCurves.AArch64

/-- `from_mont`: the multiplication block with `1` as its right operand, `value * 2^-256 mod p`.
-/
def fromMont (value modulus : Limbs) (inv : Nat) : Limbs :=
  mulMont value ⟨1, 0, 0, 0⟩ modulus inv

/-! ## The inversion -/

/-- The AArch64 transcriptions of the inversion's six blocks, as `src/inversion.rs` composes
them. -/
def invertBlocks : InvertBlocks :=
  ⟨divstep59Block, signMagBlock, fgRowBlock, deRowBlock, amontredBlock, condSubBlock⟩

end PastaCurves.AArch64
