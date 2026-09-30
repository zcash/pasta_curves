import PastaAsm.Semantics
import Mathlib.Tactic.Ring

/-!
# The crate's Rust around the blocks

`src/asm/entry.rs`, in a debug build, checks the operand contracts of `mul` and `square` before
entering the blocks. These definitions mirror that Rust: the canonicity check `is_canonical`,
a borrow chain, and the condition `mul_contract` that `mul` asserts.
-/

namespace PastaAsm

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

end PastaAsm
