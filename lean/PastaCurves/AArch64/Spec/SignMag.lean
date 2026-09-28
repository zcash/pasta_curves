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
  extract_lets -merge +onlyGivenNames u' at hres
  have e_u' : u' = u := rfl
  clear_value u'
  have b_u' : u' < 2^64 := by rw [e_u']; exact hu
  -- v': argument
  extract_lets -merge +onlyGivenNames v' at hres
  have e_v' : v' = v := rfl
  clear_value v'
  have b_v' : v' < 2^64 := by rw [e_v']; exact hv
  -- q': argument
  extract_lets -merge +onlyGivenNames q' at hres
  have e_q' : q' = q := rfl
  clear_value q'
  have b_q' : q' < 2^64 := by rw [e_q']; exact hq
  -- r': argument
  extract_lets -merge +onlyGivenNames r' at hres
  have e_r' : r' = r := rfl
  clear_value r'
  have b_r' : r' < 2^64 := by rw [e_r']; exact hr
  -- fl: cmp u,xzr
  extract_lets -merge +onlyGivenNames fl at hres
  have e_fl : fl = cmpFlags u' 0 := rfl
  clear_value fl
  -- su: csetm su,mi
  extract_lets -merge +onlyGivenNames su at hres
  have e_su : su = csetmMi fl := rfl
  clear_value su
  have b_su : su < 2^64 := by rw [e_su]; exact csetmMi_lt fl
  -- u_1: cneg u,u,mi
  extract_lets -merge +onlyGivenNames u_1 at hres
  have e_u_1 : u_1 = cnegMi fl u' := rfl
  clear_value u_1
  have b_u_1 : u_1 < 2^64 := by rw [e_u_1]; exact cnegMi_lt fl u' b_u'
  -- fl_1: cmp v,xzr
  extract_lets -merge +onlyGivenNames fl_1 at hres
  have e_fl_1 : fl_1 = cmpFlags v' 0 := rfl
  clear_value fl_1
  -- sv: csetm sv,mi
  extract_lets -merge +onlyGivenNames sv at hres
  have e_sv : sv = csetmMi fl_1 := rfl
  clear_value sv
  have b_sv : sv < 2^64 := by rw [e_sv]; exact csetmMi_lt fl_1
  -- v_1: cneg v,v,mi
  extract_lets -merge +onlyGivenNames v_1 at hres
  have e_v_1 : v_1 = cnegMi fl_1 v' := rfl
  clear_value v_1
  have b_v_1 : v_1 < 2^64 := by rw [e_v_1]; exact cnegMi_lt fl_1 v' b_v'
  -- fl_2: cmp q,xzr
  extract_lets -merge +onlyGivenNames fl_2 at hres
  have e_fl_2 : fl_2 = cmpFlags q' 0 := rfl
  clear_value fl_2
  -- sq: csetm sq,mi
  extract_lets -merge +onlyGivenNames sq at hres
  have e_sq : sq = csetmMi fl_2 := rfl
  clear_value sq
  have b_sq : sq < 2^64 := by rw [e_sq]; exact csetmMi_lt fl_2
  -- q_1: cneg q,q,mi
  extract_lets -merge +onlyGivenNames q_1 at hres
  have e_q_1 : q_1 = cnegMi fl_2 q' := rfl
  clear_value q_1
  have b_q_1 : q_1 < 2^64 := by rw [e_q_1]; exact cnegMi_lt fl_2 q' b_q'
  -- fl_3: cmp r,xzr
  extract_lets -merge +onlyGivenNames fl_3 at hres
  have e_fl_3 : fl_3 = cmpFlags r' 0 := rfl
  clear_value fl_3
  -- sr: csetm sr,mi
  extract_lets -merge +onlyGivenNames sr at hres
  have e_sr : sr = csetmMi fl_3 := rfl
  clear_value sr
  have b_sr : sr < 2^64 := by rw [e_sr]; exact csetmMi_lt fl_3
  -- r_1: cneg r,r,mi
  extract_lets -merge +onlyGivenNames r_1 at hres
  have e_r_1 : r_1 = cnegMi fl_3 r' := rfl
  clear_value r_1
  have b_r_1 : r_1 < 2^64 := by rw [e_r_1]; exact cnegMi_lt fl_3 r' b_r'
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
