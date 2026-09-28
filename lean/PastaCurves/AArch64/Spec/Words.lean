import PastaCurves.Spec
import PastaCurves.AArch64.Semantics

/-!
# Bounds of the AArch64 word operations

The generated skeletons of the inversion's block proofs bound each instruction's result below
`2^64` by one of these lemmas, or by those of the shared operations in `PastaCurves/Spec.lean`,
and keep the result's defining equation for the annotations.
-/

namespace PastaCurves.AArch64

/-- The skeleton's bound for a `madd` result, by the reduction modulo `2^64`. -/
theorem madd_lt (a b c : Nat) : madd a b c < 2^64 := Nat.mod_lt _ (by decide)

/-- The bound for an `msub` result, by the reduction. -/
theorem msub_lt (a b c : Nat) : msub a b c < 2^64 := Nat.mod_lt _ (by decide)

/-- The bound for an `mneg` result, by the reduction. -/
theorem mneg_lt (a b : Nat) : mneg a b < 2^64 := Nat.mod_lt _ (by decide)

/-- The bound for an `sbfx` result: the field, sign-extended, stays below `2^64`. -/
theorem sbfx_lt (a lsb w : Nat) (ha : a < 2^64) : sbfx a lsb w < 2^64 := by
  have hfield : a / 2^lsb % 2^w < 2^64 :=
    lt_of_le_of_lt (le_trans (Nat.mod_le _ _) (Nat.div_le_self _ _)) ha
  have hw : a / 2^lsb % 2^w < 2^w := Nat.mod_lt _ (Nat.two_pow_pos _)
  unfold sbfx
  by_cases hs : a / 2^lsb % 2^w < 2^(w - 1)
  · rw [if_pos hs]; exact hfield
  · rw [if_neg hs]
    rcases Nat.lt_or_ge 64 w with hk | hk
    · have hle : (2 : Nat)^64 ≤ 2^w := Nat.pow_le_pow_right (by decide) hk.le
      rw [Nat.sub_eq_zero_of_le hle, Nat.add_zero]
      exact hfield
    · have hle : (2 : Nat)^w ≤ 2^64 := Nat.pow_le_pow_right (by decide) hk
      calc a / 2^lsb % 2^w + (2^64 - 2^w) < 2^w + (2^64 - 2^w) := Nat.add_lt_add_right hw _
        _ = 2^64 := Nat.add_sub_of_le hle

/-- The bound for a `csel` on `ne`: one of two words. -/
theorem cselNe_lt (fl : Flags) (x y : Nat) (hx : x < 2^64) (hy : y < 2^64) :
    cselNe fl x y < 2^64 := by
  unfold cselNe; split <;> assumption

/-- The bound for a `csel` on `ge`: one of two words. -/
theorem cselGe_lt (fl : Flags) (x y : Nat) (hx : x < 2^64) (hy : y < 2^64) :
    cselGe fl x y < 2^64 := by
  unfold cselGe; split <;> assumption

/-- The bound for a `cneg` on `ge`: a word or its negation. -/
theorem cnegGe_lt (fl : Flags) (a : Nat) (ha : a < 2^64) : cnegGe fl a < 2^64 := by
  unfold cnegGe; split
  · exact negw_lt a
  · exact ha

/-- The bound for a `cneg` on `mi`: a word or its negation. -/
theorem cnegMi_lt (fl : Flags) (a : Nat) (ha : a < 2^64) : cnegMi fl a < 2^64 := by
  unfold cnegMi; split
  · exact negw_lt a
  · exact ha

/-- The bound for a `csetm` on `mi`: all ones or zero. -/
theorem csetmMi_lt (fl : Flags) : csetmMi fl < 2^64 := by
  unfold csetmMi; split <;> decide

end PastaCurves.AArch64
