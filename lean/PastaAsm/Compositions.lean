/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.Semantics
import Mathlib.Tactic.Ring

/-!
# The crate's Rust around the blocks

`src/asm/mod.rs`, in a debug build, checks the operand contracts of `mul` and `square` before entering
the blocks. These definitions mirror that Rust: the limb comparison `is_canonical`, and the
condition that `mul` asserts.
-/

namespace PastaAsm

/-- The crate's `is_canonical`: whether `value < modulus`, comparing the limbs from the most
significant down, as `src/asm/mod.rs` does. -/
def isCanonical (value modulus : Limbs) : Bool :=
  if value.l3 ≠ modulus.l3 then decide (value.l3 < modulus.l3)
  else if value.l2 ≠ modulus.l2 then decide (value.l2 < modulus.l2)
  else if value.l1 ≠ modulus.l1 then decide (value.l1 < modulus.l1)
  else if value.l0 ≠ modulus.l0 then decide (value.l0 < modulus.l0)
  else false

/-- `isCanonical` decides `value < modulus` on four-limb values. -/
theorem isCanonical_iff (value modulus : Limbs) (hv : value.Bounded) (hm : modulus.Bounded) :
    isCanonical value modulus = true ↔ value.toNat < modulus.toNat := by
  obtain ⟨hv0, hv1, hv2, hv3⟩ := hv
  obtain ⟨hm0, hm1, hm2, hm3⟩ := hm
  unfold isCanonical Limbs.toNat
  split_ifs <;> simp only [decide_eq_true_iff, false_iff, not_lt] <;> omega

/-- The condition that the crate's `mul` asserts in a debug build: a canonical `lhs`, or a
canonical `rhs` whose limbs 1 to 3 are at most `2^64 - 3`. -/
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
