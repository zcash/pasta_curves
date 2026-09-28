import Mathlib.Algebra.Order.Ring.Abs
import Mathlib.Tactic.Linarith
import PastaCurves.Spec

/-!
# The sign-magnitude form of a matrix entry, and the row arithmetic on words

The row blocks take each entry of the transition matrix as a magnitude and a sign mask, all
ones when the entry is negative and zero otherwise. `SignMagRep` is that representation as a
relation, so that a zero entry may carry either mask: the last round of the inversion xors the
masks with the sign word, and a zero entry then carries the all-ones mask with magnitude zero.

`row_side` and `row_side5` are the identities behind a row block: how complementing the words of `d`
(or of `f`) by the mask and correcting by the magnitude in the low and top words gives the signed
product. `shift59_spec` is the shift of a five-word value right by 59. They are stated over the
shared word operations, so any backend's row blocks reduce to them.
-/

set_option exponentiation.threshold 400

namespace PastaCurves

/-- A bounded five-word value below `2^256` in magnitude has a sign word of zero or all ones. -/
theorem Signed5.l4_of_abs_lt (x : Signed5) (hx : x.Bounded) (hv : |x.toInt| < 2^256) :
    x.l4 = 0 ∨ x.l4 = 2^64 - 1 := by
  obtain ⟨h0, h1, h2, h3, h4⟩ := hx
  rw [abs_lt] at hv
  unfold Signed5.toInt at hv
  split_ifs at hv <;> omega

end PastaCurves

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

/-! ## One side of a row -/

/-- `eor` with the all-ones word complements a word. -/
theorem eorw_ones (x : ℕ) (hx : x < 2^64) : eorw x (2^64 - 1) = 2^64 - 1 - x := by
  unfold eorw
  rw [show 2^64 - 1 - x = 2^64 - (x + 1) by omega]
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_xor, Nat.testBit_two_pow_sub_one, Nat.testBit_two_pow_sub_succ hx]
  by_cases hi : i < 64
  · simp [hi]
  · have hxi : x < 2^i := lt_of_lt_of_le hx (Nat.pow_le_pow_right (by decide) (by omega))
    simp [hi, Nat.testBit_lt_two_pow hxi]

/-- **Why the block's corrections give the signed product:** the block multiplies by the magnitude
`|z|` and makes the product signed by complementing the words of `x` when `z` is negative. Since
`|z| (2^256 - 1 - x) = 2^256 |z| - |z| - |z| x`, adding `|z|` in the low word and subtracting
`2^256 |z|` in the top word leaves `-|z| x = z x`. When `z` is nonnegative the mask is zero, the
words are unchanged, and both corrections vanish. This is that identity for one side of a row, so
the block proof only accounts for the carries. -/
theorem row_side (z : ℤ) (hz : |z| < 2^64) (x : Limbs) (hx : x.Bounded) (m s : ℕ)
    (hrep : SignMagRep m s z) :
    ((eorw x.l0 s + 2^64 * eorw x.l1 s + 2^128 * eorw x.l2 s + 2^192 * eorw x.l3 s : ℕ) : ℤ)
        * m + (andw m s : ℤ) - 2^256 * (andw s m : ℤ) = z * x.toNat := by
  obtain ⟨h0, h1, h2, h3⟩ := hx
  obtain ⟨hm64, -⟩ := hrep.lt _ _ _ hz
  unfold Limbs.toNat
  rcases hrep with ⟨hs, hm⟩ | ⟨hs, hm⟩
  · subst hs
    simp only [eorw, andw, Nat.xor_zero, Nat.and_zero, Nat.zero_and]
    push_cast
    rw [← hm]
    ring
  · subst hs
    have a1 : andw m (2^64 - 1) = m := by
      unfold andw; rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hm64]
    have a2 : andw (2^64 - 1) m = m := by rw [andw, Nat.and_comm]; exact a1
    have hsum : ((eorw x.l0 (2^64 - 1) + 2^64 * eorw x.l1 (2^64 - 1)
        + 2^128 * eorw x.l2 (2^64 - 1) + 2^192 * eorw x.l3 (2^64 - 1) : ℕ) : ℤ)
        = 2^256 - 1 - (x.l0 + 2^64 * x.l1 + 2^128 * x.l2 + 2^192 * x.l3) := by
      rw [eorw_ones _ h0, eorw_ones _ h1, eorw_ones _ h2, eorw_ones _ h3]; omega
    rw [hsum, a1, a2]
    push_cast
    rw [hm]
    ring

/-- `row_side` for a five-word signed `x`: the sign word of `x`, complemented by the mask of `z`,
selects the top correction, which is `|z|` exactly when `z x` is negative. -/
theorem row_side5 (z : ℤ) (hz : |z| < 2^64) (x : Signed5) (hx : x.Bounded)
    (hx4 : x.l4 = 0 ∨ x.l4 = 2^64 - 1) (m s : ℕ) (hrep : SignMagRep m s z) :
    ((eorw x.l0 s + 2^64 * eorw x.l1 s + 2^128 * eorw x.l2 s + 2^192 * eorw x.l3 s : ℕ) : ℤ)
        * m + (andw m s : ℤ) - 2^256 * (andw (eorw x.l4 s) m : ℤ) = z * x.toInt := by
  obtain ⟨h0, h1, h2, h3, h4⟩ := hx
  obtain ⟨hm64, -⟩ := hrep.lt _ _ _ hz
  have a1 : andw m (2^64 - 1) = m := by
    unfold andw; rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hm64]
  have a2 : andw (2^64 - 1) m = m := by rw [andw, Nat.and_comm]; exact a1
  have a3 : andw 0 m = 0 := Nat.zero_and m
  have hones : (2 : ℕ)^64 - 1 < 2^64 := by omega
  unfold Signed5.toInt
  rcases hrep with ⟨hs, hm⟩ | ⟨hs, hm⟩
  · subst hs
    have a0 : andw m 0 = 0 := Nat.and_zero m
    simp only [eorw, Nat.xor_zero]
    rw [a0]
    rcases hx4 with h4' | h4'
    · rw [h4', a3, if_pos (by omega)]
      push_cast
      rw [hm]
      ring
    · rw [h4', a2, if_neg (by omega)]
      push_cast
      rw [hm]
      ring
  · subst hs
    have hsum : ((eorw x.l0 (2^64 - 1) + 2^64 * eorw x.l1 (2^64 - 1)
        + 2^128 * eorw x.l2 (2^64 - 1) + 2^192 * eorw x.l3 (2^64 - 1) : ℕ) : ℤ)
        = 2^256 - 1 - (x.l0 + 2^64 * x.l1 + 2^128 * x.l2 + 2^192 * x.l3) := by
      rw [eorw_ones _ h0, eorw_ones _ h1, eorw_ones _ h2, eorw_ones _ h3]; omega
    rw [hsum, a1]
    rcases hx4 with h4' | h4'
    · rw [h4', eorw_ones 0 (by norm_num), Nat.sub_zero, a2, if_pos (by omega)]
      push_cast
      rw [hm]
      ring
    · rw [h4', eorw_ones _ hones, Nat.sub_self, a3, if_neg (by omega)]
      push_cast
      rw [hm]
      ring

/-! ## The shift by 59 -/

/-- The shift of a five-word signed value right by `59`, as the row block does it: four `extr`s
and an `asr`, which is the floor of the value over `2^59` in five words. -/
theorem shift59_spec (d0 d1 d2 d3 d4 : ℕ) (h0 : d0 < 2^64) (h1 : d1 < 2^64) (h2 : d2 < 2^64)
    (h3 : d3 < 2^64) (h4 : d4 < 2^64) :
    (Signed5.mk (extr d1 d0 59) (extr d2 d1 59) (extr d3 d2 59) (extr d4 d3 59)
        (asr d4 59)).Bounded ∧
      (Signed5.mk (extr d1 d0 59) (extr d2 d1 59) (extr d3 d2 59) (extr d4 d3 59)
        (asr d4 59)).toInt = (Signed5.mk d0 d1 d2 d3 d4).toInt / 2^59 := by
  have e0 : extr d1 d0 59 = d0 / 2^59 + 2^5 * (d1 % 2^59) := by
    unfold extr; norm_num; omega
  have e1 : extr d2 d1 59 = d1 / 2^59 + 2^5 * (d2 % 2^59) := by
    unfold extr; norm_num; omega
  have e2 : extr d3 d2 59 = d2 / 2^59 + 2^5 * (d3 % 2^59) := by
    unfold extr; norm_num; omega
  have e3 : extr d4 d3 59 = d3 / 2^59 + 2^5 * (d4 % 2^59) := by
    unfold extr; norm_num; omega
  have e4 : asr d4 59 = if d4 < 2^63 then d4 / 2^59 else d4 / 2^59 + (2^64 - 2^5) := by
    unfold asr; norm_num
  simp only [Signed5.Bounded, Signed5.toInt]
  rw [e0, e1, e2, e3, e4]
  refine ⟨⟨by omega, by omega, by omega, by omega, by split_ifs <;> omega⟩, ?_⟩
  symm
  rw [Int.ediv_eq_iff_of_pos (by norm_num)]
  split_ifs <;> push_cast <;> constructor <;> omega

end PastaCurves.Inversion
