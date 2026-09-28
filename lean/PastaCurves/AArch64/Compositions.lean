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

/-- The state that the inversion's rounds carry: `two_delta` as a word, the five-word `f` and `g`,
and the four-limb `d` and `e`. -/
structure InvertState where
  two_delta : Nat
  f : Signed5
  g : Signed5
  d : Limbs
  e : Limbs

/-- One round of `invert`: `divstep59` on the low words, the sign-magnitude form of its matrix,
`fg_row` for each row of the update of `f` and `g`, and `de_row` for each row of the combination
of `d` and `e`, each reduced by `amontred`. -/
def invertRound (modulus : Limbs) (inv : Nat) (st : InvertState) : InvertState :=
  let dm := divstep59Block st.two_delta st.f.l0 st.g.l0
  let sm := signMagBlock dm.u dm.v dm.q dm.r
  let f := fgRowBlock st.f st.g sm.u sm.v sm.su sm.sv
  let g := fgRowBlock st.f st.g sm.q sm.r sm.sq sm.sr
  let td := deRowBlock st.d st.e sm.u sm.v sm.su sm.sv
  let te := deRowBlock st.d st.e sm.q sm.r sm.sq sm.sr
  ⟨dm.two_delta, f, g, amontredBlock td modulus inv, amontredBlock te modulus inv⟩

/-- The sign word of the last round, as `src/asm/aarch64.rs` computes it: the low word of
`f0 * u + g0 * v`, shifted arithmetically by 63, so all ones when the new `f` is negative and
zero otherwise. -/
def signWordBlock (f0 g0 u v : Nat) : Nat := asr (addw (mulLo f0 u) (mulLo g0 v)) 63

/-- `invert`: nine rounds from `(two_delta, f, g, d, e) = (1, p, x, 0, e0)`, then the last round,
which computes only `d`, with the sign of the new `f` xored into the masks of its row, and reduces
strictly. -/
def invert (x modulus : Limbs) (inv : Nat) (e0 : Limbs) : Limbs :=
  let st := (invertRound modulus inv)^[9]
    ⟨1, ⟨modulus.l0, modulus.l1, modulus.l2, modulus.l3, 0⟩, ⟨x.l0, x.l1, x.l2, x.l3, 0⟩,
      ⟨0, 0, 0, 0⟩, e0⟩
  let dm := divstep59Block st.two_delta st.f.l0 st.g.l0
  let sign := signWordBlock st.f.l0 st.g.l0 dm.u dm.v
  let sm := signMagBlock dm.u dm.v dm.q dm.r
  let t := deRowBlock st.d st.e sm.u sm.v (eorw sm.su sign) (eorw sm.sv sign)
  condSubBlock (amontredBlock t modulus inv) modulus

end PastaCurves.AArch64
