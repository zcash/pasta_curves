import PastaCurves.Semantics
import Mathlib.Tactic.Ring

/-!
# The crate's Rust around the blocks

`src/asm/entry.rs`, in a debug build, checks the operand contracts of `mul` and `square` before
entering the blocks, and `src/asm/inversion.rs` composes `invert` from a backend's six blocks. These
definitions mirror that Rust: the canonicity check `is_canonical`, a borrow chain, the condition
`mul_contract` that `mul` asserts, and `invert` over a record of the blocks.
-/

namespace PastaCurves

/-- The crate's `is_canonical`: whether `value < modulus`, as the borrow out of the four-limb
subtraction `value - modulus`, limb by limb from the least significant, as `src/asm/entry.rs`
computes it. A limb borrows when it is below the other limb plus the borrow in. -/
def isCanonical (value modulus : Limbs) : Bool :=
  let borrow0 := if value.l0 < modulus.l0 then 1 else 0
  let borrow1 := if value.l1 < modulus.l1 + borrow0 then 1 else 0
  let borrow2 := if value.l2 < modulus.l2 + borrow1 then 1 else 0
  decide (value.l3 < modulus.l3 + borrow2)

/-- `isCanonical` decides `value < modulus` on four-limb values. -/
theorem isCanonical_iff (value modulus : Limbs) (hv : value.Bounded) (hm : modulus.Bounded) :
    isCanonical value modulus = true ↔ value.toNat < modulus.toNat := by
  obtain ⟨hv0, hv1, hv2, hv3⟩ := hv
  obtain ⟨hm0, hm1, hm2, hm3⟩ := hm
  unfold isCanonical Limbs.toNat
  simp only [decide_eq_true_iff]
  split_ifs <;> omega

/-- The crate's `mul_contract`, the condition that `mul` asserts in a debug build: a canonical
`lhs`, or a canonical `rhs` whose limbs 1 to 3 are at most `2^64 - 3`. -/
def mulContract (lhs rhs modulus : Limbs) : Bool :=
  isCanonical lhs modulus ||
    (isCanonical rhs modulus &&
      decide (rhs.l1 ≤ 2^64 - 3) && decide (rhs.l2 ≤ 2^64 - 3) && decide (rhs.l3 ≤ 2^64 - 3))

/-- `mulContract` decides the disjunction of the two proved operand contracts of the
multiplication block. -/
theorem mulContract_iff (lhs rhs modulus : Limbs) (hlhs : lhs.Bounded) (hrhs : rhs.Bounded)
    (hm : modulus.Bounded) :
    mulContract lhs rhs modulus = true ↔
      lhs.toNat < modulus.toNat ∨
        (rhs.toNat < modulus.toNat ∧
          rhs.l1 + 3 ≤ 2^64 ∧ rhs.l2 + 3 ≤ 2^64 ∧ rhs.l3 + 3 ≤ 2^64) := by
  unfold mulContract
  simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_iff,
    isCanonical_iff lhs modulus hlhs hm, isCanonical_iff rhs modulus hrhs hm]
  omega

/-! ## The inversion -/

/-- The inversion's six blocks, as one backend transcribes them: `divstep59` on `d` and the low
words of `f` and `g`; `sign_mag` on the four matrix words; `fg_row` and `uv_row` on a row's
magnitudes and masks; `amontred` and `cond_sub` with the modulus limbs (and `inv`) last. -/
structure InvertBlocks where
  divstep59 : Nat → Nat → Nat → Divstep59Result
  signMag : Nat → Nat → Nat → Nat → SignMag
  fgRow : Signed5 → Signed5 → Nat → Nat → Nat → Nat → Signed5
  uvRow : Limbs → Limbs → Nat → Nat → Nat → Nat → Signed5
  amontred : Signed5 → Limbs → Nat → Limbs
  condSub : Limbs → Limbs → Limbs

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
def invertRound (B : InvertBlocks) (modulus : Limbs) (inv : Nat) (st : InvertState) :
    InvertState :=
  let dm := B.divstep59 st.d st.f.l0 st.g.l0
  let sm := B.signMag dm.m00 dm.m01 dm.m10 dm.m11
  let f := B.fgRow st.f st.g sm.m00 sm.m01 sm.s00 sm.s01
  let g := B.fgRow st.f st.g sm.m10 sm.m11 sm.s10 sm.s11
  let tu := B.uvRow st.u st.v sm.m00 sm.m01 sm.s00 sm.s01
  let tv := B.uvRow st.u st.v sm.m10 sm.m11 sm.s10 sm.s11
  ⟨dm.d, f, g, B.amontred tu modulus inv, B.amontred tv modulus inv⟩

/-- The sign word of the last round, as `src/asm/inversion.rs` computes it in Rust: the low word of
`f0 * m00 + g0 * m01`, shifted arithmetically by 63, so all ones when the new `f` is negative and
zero otherwise. -/
def signWord (f0 g0 m00 m01 : Nat) : Nat :=
  if (mulLo f0 m00 + mulLo g0 m01) % regMod < 2^63 then 0 else regMod - 1

/-- `invert`: nine rounds from `(d, f, g, u, v) = (1, p, x, 0, v0)`, then the last round, which
computes only `u`, with the sign of the new `f` xored into the masks of its row, and reduces
strictly. -/
def invert (B : InvertBlocks) (x modulus : Limbs) (inv : Nat) (v0 : Limbs) : Limbs :=
  let st := (invertRound B modulus inv)^[9]
    ⟨1, ⟨modulus.l0, modulus.l1, modulus.l2, modulus.l3, 0⟩, ⟨x.l0, x.l1, x.l2, x.l3, 0⟩,
      ⟨0, 0, 0, 0⟩, v0⟩
  let dm := B.divstep59 st.d st.f.l0 st.g.l0
  let sign := signWord st.f.l0 st.g.l0 dm.m00 dm.m01
  let sm := B.signMag dm.m00 dm.m01 dm.m10 dm.m11
  let t := B.uvRow st.u st.v sm.m00 sm.m01 (sm.s00 ^^^ sign) (sm.s01 ^^^ sign)
  B.condSub (B.amontred t modulus inv) modulus

end PastaCurves
