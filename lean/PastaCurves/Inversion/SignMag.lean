import PastaCurves.Spec

/-!
# The sign-magnitude form of a matrix entry

The row blocks take each entry of the transition matrix as a magnitude and a sign mask, all
ones when the entry is negative and zero otherwise. `SignMagRep` is that representation as a
relation, so that a zero entry may carry either mask: the last round of the inversion xors the
masks with the sign word, and a zero entry then carries the all-ones mask with magnitude zero.
-/

namespace PastaCurves.Inversion

/-- The sign mask of an integer as the row blocks take it: all ones when it is negative, else
zero. -/
def signMask (z : ℤ) : ℕ := if z < 0 then 2^64 - 1 else 0

/-- The sign-magnitude representation of an integer that the row blocks take: the magnitude `m`
with the mask `s` clear for `a = m`, or set for `a = -m`. Zero has both forms, which is what the
last round's masking by the sign word needs. -/
def SignMagRep (m s : ℕ) (a : ℤ) : Prop := (s = 0 ∧ (m : ℤ) = a) ∨ (s = 2^64 - 1 ∧ (m : ℤ) = -a)

theorem SignMagRep.of_natAbs (a : ℤ) : SignMagRep a.natAbs (signMask a) a := by
  unfold SignMagRep signMask
  rcases lt_or_ge a 0 with h | h
  · right; rw [if_pos h, Int.natCast_natAbs, abs_of_neg h]; exact ⟨rfl, rfl⟩
  · left; rw [if_neg (not_lt.mpr h), Int.natCast_natAbs, abs_of_nonneg h]; exact ⟨rfl, rfl⟩

/-- Flipping the mask by the all-ones word negates the represented integer. -/
theorem SignMagRep.xor_ones (m s : ℕ) (a : ℤ) (h : SignMagRep m s a) :
    SignMagRep m (s ^^^ (2^64 - 1)) (-a) := by
  unfold SignMagRep at h ⊢
  rcases h with ⟨hs, hm⟩ | ⟨hs, hm⟩
  · right; rw [hs, Nat.zero_xor]; exact ⟨rfl, by rw [hm, neg_neg]⟩
  · left; rw [hs, Nat.xor_self]; exact ⟨rfl, hm⟩

theorem SignMagRep.xor_zero (m s : ℕ) (a : ℤ) (h : SignMagRep m s a) :
    SignMagRep m (s ^^^ 0) a := by
  rwa [Nat.xor_zero]

theorem SignMagRep.lt (m s : ℕ) (a : ℤ) (h : SignMagRep m s a) (ha : |a| < 2^64) :
    m < 2^64 ∧ s < 2^64 := by
  rw [abs_lt] at ha
  rcases h with ⟨hs, hm⟩ | ⟨hs, hm⟩ <;> constructor <;> omega

/-- The magnitude of a represented integer is its absolute value. -/
theorem SignMagRep.natCast_eq_abs (m s : ℕ) (a : ℤ) (h : SignMagRep m s a) : (m : ℤ) = |a| := by
  rcases h with ⟨-, hm⟩ | ⟨-, hm⟩
  · rw [hm, abs_of_nonneg (by omega)]
  · rw [hm, abs_of_nonpos (by omega)]

end PastaCurves.Inversion
