/-
Copyright Amazon.com, Inc. or its affiliates (the block, adapted from s2n-bignum).
Copyright (c) 2026 the pasta_curves contributors (the transcription and the proofs).
-/
import PastaCurves.AArch64.Spec.Words
import PastaCurves.AArch64.Transcription

/-!
# Correctness of the inversion's sign-magnitude block

See the parent module's documentation for details. The block takes the four entries of a
transition matrix as two's-complement words and returns each entry's magnitude and its sign
mask, the form in which the row blocks take the matrix.
-/

namespace PastaCurves.AArch64

-- BEGIN signMagBlock_spec lemmas
/-- The sign mask of an integer as the row blocks take it: all ones when it is negative, else
zero. -/
def signMask (z : ℤ) : ℕ := if z < 0 then 2^64 - 1 else 0

/-- The `mi` condition after `cmp m, xzr` reads bit 63 of `m`, which is the sign of `z` because
`|z| < 2^63`. `csetm` turns that bit into the mask, and `cneg` negates `m` modulo `2^64`, which
for a negative `z` is `2^64 - m = -z`. -/
theorem signMag_word (z : ℤ) (hz : |z| < 2^63) (m : ℕ) (hm : (m : ℤ) = z % 2^64) :
    cnegMi (cmpFlags m 0) m = z.natAbs ∧ csetmMi (cmpFlags m 0) = signMask z := by
  have hm64 : m < 2^64 := by omega
  have hfl : (cmpFlags m 0).n = m / 2^63 := by
    simp only [cmpFlags, regMod, Nat.sub_zero]
    rw [Nat.add_mod_right, Nat.mod_eq_of_lt hm64]
  rw [abs_lt] at hz
  unfold cnegMi csetmMi signMask negw
  simp only [regMod]
  rw [hfl]
  rcases lt_or_ge z 0 with hneg | hnn
  · have h1 : m / 2^63 = 1 := by omega
    rw [if_pos h1, if_pos h1, if_pos hneg]
    refine ⟨?_, rfl⟩
    omega
  · have h0 : ¬ m / 2^63 = 1 := by omega
    rw [if_neg h0, if_neg h0, if_neg (not_lt.mpr hnn)]
    refine ⟨?_, rfl⟩
    omega
-- END signMagBlock_spec lemmas

-- BEGIN signMagBlock_spec statement
/-- `divstep59` returns the transition matrix as two's-complement words, and the row blocks take
each entry as a magnitude and a sign mask; this block converts between the two forms. The
hypothesis `|z| < 2^63` on each entry is what makes bit 63 its sign; the entries of a 59-step
matrix satisfy it, since their row sums are at most `2^59` (`M_rowSum_le`). -/
theorem signMagBlock_spec (a b c d : ℤ) (m00 m01 m10 m11 : Nat)
    (ha : |a| < 2^63) (hb : |b| < 2^63) (hc : |c| < 2^63) (hd : |d| < 2^63)
    (em00 : (m00 : ℤ) = a % 2^64) (em01 : (m01 : ℤ) = b % 2^64)
    (em10 : (m10 : ℤ) = c % 2^64) (em11 : (m11 : ℤ) = d % 2^64) :
    ∀ r, r = signMagBlock m00 m01 m10 m11 →
      r = ⟨a.natAbs, b.natAbs, c.natAbs, d.natAbs,
        signMask a, signMask b, signMask c, signMask d⟩ := by
  intro r hr
  have hm00 : m00 < 2^64 := by omega
  have hm01 : m01 < 2^64 := by omega
  have hm10 : m10 < 2^64 := by omega
  have hm11 : m11 < 2^64 := by omega
-- END signMagBlock_spec statement
  -- generated skeleton for `signMagBlock`: do not edit between the annotations
  unfold signMagBlock at hr
  lift_lets -merge at hr
  -- m00': argument
  extract_lets -merge +onlyGivenNames m00' at hr
  have e_m00' : m00' = m00 := rfl
  clear_value m00'
  have b_m00' : m00' < 2^64 := by rw [e_m00']; exact hm00
  -- m01': argument
  extract_lets -merge +onlyGivenNames m01' at hr
  have e_m01' : m01' = m01 := rfl
  clear_value m01'
  have b_m01' : m01' < 2^64 := by rw [e_m01']; exact hm01
  -- m10': argument
  extract_lets -merge +onlyGivenNames m10' at hr
  have e_m10' : m10' = m10 := rfl
  clear_value m10'
  have b_m10' : m10' < 2^64 := by rw [e_m10']; exact hm10
  -- m11': argument
  extract_lets -merge +onlyGivenNames m11' at hr
  have e_m11' : m11' = m11 := rfl
  clear_value m11'
  have b_m11' : m11' < 2^64 := by rw [e_m11']; exact hm11
  -- fl: cmp m00,xzr
  extract_lets -merge +onlyGivenNames fl at hr
  have e_fl : fl = cmpFlags m00' 0 := rfl
  clear_value fl
  -- s00: csetm s00,mi
  extract_lets -merge +onlyGivenNames s00 at hr
  have e_s00 : s00 = csetmMi fl := rfl
  clear_value s00
  have b_s00 : s00 < 2^64 := by rw [e_s00]; exact csetmMi_lt fl
  -- m00_1: cneg m00,m00,mi
  extract_lets -merge +onlyGivenNames m00_1 at hr
  have e_m00_1 : m00_1 = cnegMi fl m00' := rfl
  clear_value m00_1
  have b_m00_1 : m00_1 < 2^64 := by rw [e_m00_1]; exact cnegMi_lt fl m00' b_m00'
  -- fl_1: cmp m01,xzr
  extract_lets -merge +onlyGivenNames fl_1 at hr
  have e_fl_1 : fl_1 = cmpFlags m01' 0 := rfl
  clear_value fl_1
  -- s01: csetm s01,mi
  extract_lets -merge +onlyGivenNames s01 at hr
  have e_s01 : s01 = csetmMi fl_1 := rfl
  clear_value s01
  have b_s01 : s01 < 2^64 := by rw [e_s01]; exact csetmMi_lt fl_1
  -- m01_1: cneg m01,m01,mi
  extract_lets -merge +onlyGivenNames m01_1 at hr
  have e_m01_1 : m01_1 = cnegMi fl_1 m01' := rfl
  clear_value m01_1
  have b_m01_1 : m01_1 < 2^64 := by rw [e_m01_1]; exact cnegMi_lt fl_1 m01' b_m01'
  -- fl_2: cmp m10,xzr
  extract_lets -merge +onlyGivenNames fl_2 at hr
  have e_fl_2 : fl_2 = cmpFlags m10' 0 := rfl
  clear_value fl_2
  -- s10: csetm s10,mi
  extract_lets -merge +onlyGivenNames s10 at hr
  have e_s10 : s10 = csetmMi fl_2 := rfl
  clear_value s10
  have b_s10 : s10 < 2^64 := by rw [e_s10]; exact csetmMi_lt fl_2
  -- m10_1: cneg m10,m10,mi
  extract_lets -merge +onlyGivenNames m10_1 at hr
  have e_m10_1 : m10_1 = cnegMi fl_2 m10' := rfl
  clear_value m10_1
  have b_m10_1 : m10_1 < 2^64 := by rw [e_m10_1]; exact cnegMi_lt fl_2 m10' b_m10'
  -- fl_3: cmp m11,xzr
  extract_lets -merge +onlyGivenNames fl_3 at hr
  have e_fl_3 : fl_3 = cmpFlags m11' 0 := rfl
  clear_value fl_3
  -- s11: csetm s11,mi
  extract_lets -merge +onlyGivenNames s11 at hr
  have e_s11 : s11 = csetmMi fl_3 := rfl
  clear_value s11
  have b_s11 : s11 < 2^64 := by rw [e_s11]; exact csetmMi_lt fl_3
  -- m11_1: cneg m11,m11,mi
  extract_lets -merge +onlyGivenNames m11_1 at hr
  have e_m11_1 : m11_1 = cnegMi fl_3 m11' := rfl
  clear_value m11_1
  have b_m11_1 : m11_1 < 2^64 := by rw [e_m11_1]; exact cnegMi_lt fl_3 m11' b_m11'
  subst hr
  -- BEGIN conclusion
  obtain ⟨hA, hA'⟩ := signMag_word a ha m00 em00
  obtain ⟨hB, hB'⟩ := signMag_word b hb m01 em01
  obtain ⟨hC, hC'⟩ := signMag_word c hc m10 em10
  obtain ⟨hD, hD'⟩ := signMag_word d hd m11 em11
  rw [e_m00_1, e_s00, e_fl, e_m00', e_m01_1, e_s01, e_fl_1, e_m01', e_m10_1, e_s10, e_fl_2,
    e_m10', e_m11_1, e_s11, e_fl_3, e_m11', hA, hA', hB, hB', hC, hC', hD, hD']
  -- END conclusion

end PastaCurves.AArch64
