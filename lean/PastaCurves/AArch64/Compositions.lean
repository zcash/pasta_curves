import PastaCurves.Compositions
import PastaCurves.AArch64.Transcription

/-!
# The crate's Rust around the blocks

`src/asm/entry.rs` composes the crate's `sqr_n_mul` and `from_mont` from the `square` and `mul`
blocks, and `src/asm/aarch64.rs` composes `invert` from the inversion's six blocks. These definitions
mirror that Rust.
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

/-- The state that the inversion's rounds carry: `d` as a word, the five-word `f` and `g`, and
the four-limb `u` and `v`. -/
structure InvertState where
  d : Nat
  f : Signed5
  g : Signed5
  u : Limbs
  v : Limbs

/-- One round of `invert`: `divstep59` on the low words, the sign-magnitude form of its matrix,
`fg_row` for each row of the update of `f` and `g`, and `uv_row` for each row of the combination
of `u` and `v`, each reduced by `amontred`. -/
def invertRound (modulus : Limbs) (inv : Nat) (st : InvertState) : InvertState :=
  let dm := divstep59Block st.d st.f.l0 st.g.l0
  let sm := signMagBlock dm.m00 dm.m01 dm.m10 dm.m11
  let f := fgRowBlock st.f st.g sm.m00 sm.m01 sm.s00 sm.s01
  let g := fgRowBlock st.f st.g sm.m10 sm.m11 sm.s10 sm.s11
  let tu := uvRowBlock st.u st.v sm.m00 sm.m01 sm.s00 sm.s01
  let tv := uvRowBlock st.u st.v sm.m10 sm.m11 sm.s10 sm.s11
  ⟨dm.d, f, g, amontredBlock tu modulus inv, amontredBlock tv modulus inv⟩

/-- The sign word of the last round, as `src/asm/aarch64.rs` computes it: the low word of
`f0 * m00 + g0 * m01`, shifted arithmetically by 63, so all ones when the new `f` is negative and
zero otherwise. -/
def signWordBlock (f0 g0 m00 m01 : Nat) : Nat := asr (addw (mulLo f0 m00) (mulLo g0 m01)) 63

/-- `invert`: nine rounds from `(d, f, g, u, v) = (1, p, x, 0, v0)`, then the last round, which
computes only `u`, with the sign of the new `f` xored into the masks of its row, and reduces
strictly. -/
def invert (x modulus : Limbs) (inv : Nat) (v0 : Limbs) : Limbs :=
  let st := (invertRound modulus inv)^[9]
    ⟨1, ⟨modulus.l0, modulus.l1, modulus.l2, modulus.l3, 0⟩, ⟨x.l0, x.l1, x.l2, x.l3, 0⟩,
      ⟨0, 0, 0, 0⟩, v0⟩
  let dm := divstep59Block st.d st.f.l0 st.g.l0
  let sign := signWordBlock st.f.l0 st.g.l0 dm.m00 dm.m01
  let sm := signMagBlock dm.m00 dm.m01 dm.m10 dm.m11
  let t := uvRowBlock st.u st.v sm.m00 sm.m01 (eorw sm.s00 sign) (eorw sm.s01 sign)
  condSubBlock (amontredBlock t modulus inv) modulus

end PastaCurves.AArch64
