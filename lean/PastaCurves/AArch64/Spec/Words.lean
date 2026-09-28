import PastaCurves.Spec
import PastaCurves.AArch64.Semantics

/-!
# Bounds of the AArch64 word operations

The generated skeletons of the inversion's block proofs bound each instruction's result below
`2^64` by one of these lemmas, and keep the result's defining equation for the annotations.
-/

namespace PastaCurves.AArch64

theorem addw_lt (a b : Nat) : addw a b < 2^64 := Nat.mod_lt _ (by decide)

theorem subw_lt (a b : Nat) : subw a b < 2^64 := Nat.mod_lt _ (by decide)

theorem negw_lt (a : Nat) : negw a < 2^64 := Nat.mod_lt _ (by decide)

theorem madd_lt (a b c : Nat) : madd a b c < 2^64 := Nat.mod_lt _ (by decide)

theorem msub_lt (a b c : Nat) : msub a b c < 2^64 := Nat.mod_lt _ (by decide)

theorem mneg_lt (a b : Nat) : mneg a b < 2^64 := Nat.mod_lt _ (by decide)

theorem extr_lt (hi lo k : Nat) : extr hi lo k < 2^64 := Nat.mod_lt _ (by decide)

theorem andw_lt (a b : Nat) (_ha : a < 2^64) (hb : b < 2^64) : andw a b < 2^64 :=
  Nat.and_lt_two_pow a hb

theorem orrw_lt (a b : Nat) (ha : a < 2^64) (hb : b < 2^64) : orrw a b < 2^64 :=
  Nat.or_lt_two_pow ha hb

theorem eorw_lt (a b : Nat) (ha : a < 2^64) (hb : b < 2^64) : eorw a b < 2^64 :=
  Nat.xor_lt_two_pow ha hb

/-- `a / 2^k < 2^(64 - k)` for `a < 2^64`, the fact behind the two branches of `asr`. -/
theorem div_two_pow_lt (a k : Nat) (ha : a < 2^64) : a / 2^k < 2^(64 - k) := by
  rcases Nat.lt_or_ge 64 k with hk | hk
  · have h : a < 2^k := lt_of_lt_of_le ha (Nat.pow_le_pow_right (by decide) hk.le)
    rw [Nat.div_eq_of_lt h]
    exact Nat.two_pow_pos _
  · have h : (2 : Nat)^64 = 2^k * 2^(64 - k) := by rw [← pow_add]; congr 1; omega
    exact Nat.div_lt_of_lt_mul (h ▸ ha)

theorem asr_lt (a k : Nat) (ha : a < 2^64) : asr a k < 2^64 := by
  have h := div_two_pow_lt a k ha
  have hle : (2 : Nat)^(64 - k) ≤ 2^64 := Nat.pow_le_pow_right (by decide) (by omega)
  unfold asr
  by_cases hs : a < 2^63
  · rw [if_pos hs]; exact lt_of_lt_of_le h hle
  · rw [if_neg hs]
    calc a / 2^k + (2^64 - 2^(64 - k)) < 2^(64 - k) + (2^64 - 2^(64 - k)) :=
          Nat.add_lt_add_right h _
      _ = 2^64 := Nat.add_sub_of_le hle

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

theorem cselNe_lt (fl : Flags) (x y : Nat) (hx : x < 2^64) (hy : y < 2^64) :
    cselNe fl x y < 2^64 := by
  unfold cselNe; split <;> assumption

theorem cselGe_lt (fl : Flags) (x y : Nat) (hx : x < 2^64) (hy : y < 2^64) :
    cselGe fl x y < 2^64 := by
  unfold cselGe; split <;> assumption

theorem cnegGe_lt (fl : Flags) (a : Nat) (ha : a < 2^64) : cnegGe fl a < 2^64 := by
  unfold cnegGe; split
  · exact negw_lt a
  · exact ha

theorem cnegMi_lt (fl : Flags) (a : Nat) (ha : a < 2^64) : cnegMi fl a < 2^64 := by
  unfold cnegMi; split
  · exact negw_lt a
  · exact ha

theorem csetmMi_lt (fl : Flags) : csetmMi fl < 2^64 := by
  unfold csetmMi; split <;> decide

end PastaCurves.AArch64
