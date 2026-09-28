/-
Copyright Amazon.com, Inc. or its affiliates (the block, adapted from s2n-bignum).
Copyright (c) 2026 the pasta_curves contributors (the transcription and the proofs).
-/
import PastaCurves.AArch64.Spec.Words
import PastaCurves.AArch64.Transcription
import PastaCurves.Inversion.SignMag

/-!
# Correctness of the inversion's sign-magnitude block

See the parent module's documentation for details. The block takes the four entries of a transition
matrix as two's-complement words and returns each entry's magnitude and its sign mask, the form in
which the row blocks take the matrix (`Inversion/SignMag.lean`).
-/

namespace PastaCurves.AArch64

open Inversion (signMask SignMagRep)

-- BEGIN signMagBlock_spec lemmas
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
  word_step m00' := m00 using hm00
  -- m01': argument
  word_step m01' := m01 using hm01
  -- m10': argument
  word_step m10' := m10 using hm10
  -- m11': argument
  word_step m11' := m11 using hm11
  -- fl: cmp m00,xzr
  word_step fl := cmpFlags m00' 0
  -- s00: csetm s00,mi
  word_step s00 := csetmMi fl using csetmMi_lt fl
  -- m00_1: cneg m00,m00,mi
  word_step m00_1 := cnegMi fl m00' using cnegMi_lt fl m00' b_m00'
  -- fl_1: cmp m01,xzr
  word_step fl_1 := cmpFlags m01' 0
  -- s01: csetm s01,mi
  word_step s01 := csetmMi fl_1 using csetmMi_lt fl_1
  -- m01_1: cneg m01,m01,mi
  word_step m01_1 := cnegMi fl_1 m01' using cnegMi_lt fl_1 m01' b_m01'
  -- fl_2: cmp m10,xzr
  word_step fl_2 := cmpFlags m10' 0
  -- s10: csetm s10,mi
  word_step s10 := csetmMi fl_2 using csetmMi_lt fl_2
  -- m10_1: cneg m10,m10,mi
  word_step m10_1 := cnegMi fl_2 m10' using cnegMi_lt fl_2 m10' b_m10'
  -- fl_3: cmp m11,xzr
  word_step fl_3 := cmpFlags m11' 0
  -- s11: csetm s11,mi
  word_step s11 := csetmMi fl_3 using csetmMi_lt fl_3
  -- m11_1: cneg m11,m11,mi
  word_step m11_1 := cnegMi fl_3 m11' using cnegMi_lt fl_3 m11' b_m11'
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
