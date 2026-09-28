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
theorem signMagBlock_spec (zu zv zq zr : ℤ) (u v q r : Nat)
    (hzu : |zu| < 2^63) (hzv : |zv| < 2^63) (hzq : |zq| < 2^63) (hzr : |zr| < 2^63)
    (eu : (u : ℤ) = zu % 2^64) (ev : (v : ℤ) = zv % 2^64)
    (eq : (q : ℤ) = zq % 2^64) (er : (r : ℤ) = zr % 2^64) :
    ∀ res, res = signMagBlock u v q r →
      res = ⟨zu.natAbs, zv.natAbs, zq.natAbs, zr.natAbs,
        signMask zu, signMask zv, signMask zq, signMask zr⟩ := by
  intro res hres
  have hu : u < 2^64 := by omega
  have hv : v < 2^64 := by omega
  have hq : q < 2^64 := by omega
  have hr : r < 2^64 := by omega
-- END signMagBlock_spec statement
  -- generated skeleton for `signMagBlock`: do not edit between the annotations
  unfold signMagBlock at hres
  lift_lets -merge at hres
  -- u': argument
  word_step u' := u using hu
  -- v': argument
  word_step v' := v using hv
  -- q': argument
  word_step q' := q using hq
  -- r': argument
  word_step r' := r using hr
  -- fl: cmp u,xzr
  word_step fl := cmpFlags u' 0
  -- su: csetm su,mi
  word_step su := csetmMi fl using csetmMi_lt fl
  -- u_1: cneg u,u,mi
  word_step u_1 := cnegMi fl u' using cnegMi_lt fl u' b_u'
  -- fl_1: cmp v,xzr
  word_step fl_1 := cmpFlags v' 0
  -- sv: csetm sv,mi
  word_step sv := csetmMi fl_1 using csetmMi_lt fl_1
  -- v_1: cneg v,v,mi
  word_step v_1 := cnegMi fl_1 v' using cnegMi_lt fl_1 v' b_v'
  -- fl_2: cmp q,xzr
  word_step fl_2 := cmpFlags q' 0
  -- sq: csetm sq,mi
  word_step sq := csetmMi fl_2 using csetmMi_lt fl_2
  -- q_1: cneg q,q,mi
  word_step q_1 := cnegMi fl_2 q' using cnegMi_lt fl_2 q' b_q'
  -- fl_3: cmp r,xzr
  word_step fl_3 := cmpFlags r' 0
  -- sr: csetm sr,mi
  word_step sr := csetmMi fl_3 using csetmMi_lt fl_3
  -- r_1: cneg r,r,mi
  word_step r_1 := cnegMi fl_3 r' using cnegMi_lt fl_3 r' b_r'
  subst hres
  -- BEGIN conclusion
  obtain ⟨hA, hA'⟩ := signMag_word zu hzu u eu
  obtain ⟨hB, hB'⟩ := signMag_word zv hzv v ev
  obtain ⟨hC, hC'⟩ := signMag_word zq hzq q eq
  obtain ⟨hD, hD'⟩ := signMag_word zr hzr r er
  rw [e_u_1, e_su, e_fl, e_u', e_v_1, e_sv, e_fl_1, e_v', e_q_1, e_sq, e_fl_2,
    e_q', e_r_1, e_sr, e_fl_3, e_r', hA, hA', hB, hB', hC, hC', hD, hD']
  -- END conclusion

end PastaCurves.AArch64
