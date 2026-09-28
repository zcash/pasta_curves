import PastaCurves.Compositions
import PastaCurves.AArch64.Transcription

/-!
# The crate's Rust around the blocks

`src/asm/entry.rs` composes the crate's `sqr_n_mul` and `from_mont` from the `square` and `mul`
blocks, and `src/asm/inversion.rs` composes `invert` from a backend's six blocks, which `invert` of
`PastaCurves/Compositions.lean` mirrors over a record of the blocks. These definitions mirror the
first two and supply the AArch64 blocks for the third.
-/

namespace PastaCurves.AArch64

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

/-! ## The inversion -/

/-- The AArch64 transcriptions of the inversion's six blocks, as `src/asm/inversion.rs` composes
them. -/
def invertBlocks : InvertBlocks :=
  ⟨divstep59Block, signMagBlock, fgRowBlock, deRowBlock, amontredBlock, condSubBlock⟩

end PastaCurves.AArch64
