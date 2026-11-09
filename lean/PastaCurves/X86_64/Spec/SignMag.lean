/-
Copyright Amazon.com, Inc. or its affiliates (the block, adapted from s2n-bignum).
Copyright (c) 2026 the pasta_curves contributors (the transcription and the proofs).
-/
import PastaCurves.Spec
import PastaCurves.X86_64.Spec.Arithmetic
import PastaCurves.X86_64.Transcription
import PastaCurves.Inversion.SignMag
import Mathlib.Tactic.ClearExcept

/-!
# Correctness of the inversion's sign-magnitude block

The block takes the four entries of a transition matrix as two's-complement words and returns each
entry's magnitude and its sign mask, the form in which the row blocks take the matrix
(`Inversion/SignMag.lean`). The statement is the AArch64 block's. Each entry is split as
s2n-bignum's x86-64 code splits it: the mask is the entry shifted right arithmetically by 63, and
the magnitude is the entry xored with the mask, minus the mask.
-/

namespace PastaCurves.X86_64

open Inversion (signMask SignMagRep eorw_ones)

-- BEGIN signMagBlock_spec lemmas
/-- One entry's split. The arithmetic shift by 63 of the word `m` of `z` is `0` or all ones as
bit 63 is clear or set, which is the sign of `z` because `|z| < 2^63`: the mask. Xoring `m` with
it complements `m` for a negative `z`, and subtracting the mask, `-1`, then adds one, so the
result is `2^64 - m = -z`; for a nonnegative `z` both leave `m = z`. `hy` is the subtraction's
linear equation, as the skeleton records it. -/
theorem signMag_word (z : ℤ) (hz : |z| < 2^63) (m s x y c : ℕ) (hm : (m : ℤ) = z % 2^64)
    (hs : s = asr m 63) (hx : x = eorw m s) (hy : y + s + 0 = x + 2^64 * c)
    (hy64 : y < 2^64) : y = z.natAbs ∧ s = signMask z := by
  have hm64 : m < 2^64 := by omega
  rw [abs_lt] at hz
  unfold signMask
  rcases lt_or_ge z 0 with hneg | hnn
  · have hbig : ¬ m < 2^63 := by omega
    have hs1 : s = 2^64 - 1 := by
      rw [hs]; unfold asr; rw [if_neg hbig]; simp only [regMod]; omega
    have hx1 : x = 2^64 - 1 - m := by rw [hx, hs1]; exact eorw_ones m hm64
    rw [if_pos hneg]
    refine ⟨?_, hs1⟩
    omega
  · have hsmall : m < 2^63 := by omega
    have hs0 : s = 0 := by rw [hs]; unfold asr; rw [if_pos hsmall]; omega
    have hx0 : x = m := by rw [hx, hs0]; unfold eorw; exact Nat.xor_zero m
    rw [if_neg (not_lt.mpr hnn)]
    refine ⟨?_, hs0⟩
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
  -- u': scalar input
  word_step u' := u using hu
  -- v': scalar input
  word_step v' := v using hv
  -- q': scalar input
  word_step q' := q using hq
  -- r': scalar input
  word_step r' := r using hr
  -- su: mov {su}, {u}
  word_step su := u' using b_u'
  -- su_1: sar {su}, 63
  word_step su_1 := asr su 63 using asr_lt su 63 b_su
  -- u_1: xor {u}, {su}
  word_step u_1 := eorw u' su_1 using eorw_lt u' su_1 b_u' b_su_1
  -- u_2: sub {u}, {su}
  word_step d, u_2 := (sbb u_1 su_1 0).1 using sbb_value_lt u_1 su_1 0, cf := (sbb u_1 su_1 0).2
  have l_u_2 : u_2 + su_1 + 0 = u_1 + 2^64 * cf := by
    rw [e_u_2, e_cf]; exact sbb_lin u_1 su_1 0 b_u_1 b_su_1 (by decide)
  have b_cf : cf ≤ 1 := by rw [e_cf]; exact sbb_borrow_le_one u_1 su_1 0
  clear e_u_2 e_cf
  -- sv: mov {sv}, {v}
  word_step sv := v' using b_v'
  -- sv_1: sar {sv}, 63
  word_step sv_1 := asr sv 63 using asr_lt sv 63 b_sv
  -- v_1: xor {v}, {sv}
  word_step v_1 := eorw v' sv_1 using eorw_lt v' sv_1 b_v' b_sv_1
  -- v_2: sub {v}, {sv}
  word_step d_1, v_2 := (sbb v_1 sv_1 0).1 using sbb_value_lt v_1 sv_1 0, cf_1 := (sbb v_1 sv_1 0).2
  have l_v_2 : v_2 + sv_1 + 0 = v_1 + 2^64 * cf_1 := by
    rw [e_v_2, e_cf_1]; exact sbb_lin v_1 sv_1 0 b_v_1 b_sv_1 (by decide)
  have b_cf_1 : cf_1 ≤ 1 := by rw [e_cf_1]; exact sbb_borrow_le_one v_1 sv_1 0
  clear e_v_2 e_cf_1
  -- sq: mov {sq}, {q}
  word_step sq := q' using b_q'
  -- sq_1: sar {sq}, 63
  word_step sq_1 := asr sq 63 using asr_lt sq 63 b_sq
  -- q_1: xor {q}, {sq}
  word_step q_1 := eorw q' sq_1 using eorw_lt q' sq_1 b_q' b_sq_1
  -- q_2: sub {q}, {sq}
  word_step d_2, q_2 := (sbb q_1 sq_1 0).1 using sbb_value_lt q_1 sq_1 0, cf_2 := (sbb q_1 sq_1 0).2
  have l_q_2 : q_2 + sq_1 + 0 = q_1 + 2^64 * cf_2 := by
    rw [e_q_2, e_cf_2]; exact sbb_lin q_1 sq_1 0 b_q_1 b_sq_1 (by decide)
  have b_cf_2 : cf_2 ≤ 1 := by rw [e_cf_2]; exact sbb_borrow_le_one q_1 sq_1 0
  clear e_q_2 e_cf_2
  -- sr: mov {sr}, {r}
  word_step sr := r' using b_r'
  -- sr_1: sar {sr}, 63
  word_step sr_1 := asr sr 63 using asr_lt sr 63 b_sr
  -- r_1: xor {r}, {sr}
  word_step r_1 := eorw r' sr_1 using eorw_lt r' sr_1 b_r' b_sr_1
  -- r_2: sub {r}, {sr}
  word_step d_3, r_2 := (sbb r_1 sr_1 0).1 using sbb_value_lt r_1 sr_1 0, cf_3 := (sbb r_1 sr_1 0).2
  have l_r_2 : r_2 + sr_1 + 0 = r_1 + 2^64 * cf_3 := by
    rw [e_r_2, e_cf_3]; exact sbb_lin r_1 sr_1 0 b_r_1 b_sr_1 (by decide)
  have b_cf_3 : cf_3 ≤ 1 := by rw [e_cf_3]; exact sbb_borrow_le_one r_1 sr_1 0
  clear e_r_2 e_cf_3
  subst hres
  -- BEGIN conclusion
  obtain ⟨hA, hA'⟩ := signMag_word zu hzu u su_1 u_1 u_2 cf eu
    (by rw [e_su_1, e_su, e_u']) (by rw [e_u_1, e_u']) l_u_2 b_u_2
  obtain ⟨hB, hB'⟩ := signMag_word zv hzv v sv_1 v_1 v_2 cf_1 ev
    (by rw [e_sv_1, e_sv, e_v']) (by rw [e_v_1, e_v']) l_v_2 b_v_2
  obtain ⟨hC, hC'⟩ := signMag_word zq hzq q sq_1 q_1 q_2 cf_2 eq
    (by rw [e_sq_1, e_sq, e_q']) (by rw [e_q_1, e_q']) l_q_2 b_q_2
  obtain ⟨hD, hD'⟩ := signMag_word zr hzr r sr_1 r_1 r_2 cf_3 er
    (by rw [e_sr_1, e_sr, e_r']) (by rw [e_r_1, e_r']) l_r_2 b_r_2
  rw [hA, hA', hB, hB', hC, hC', hD, hD']
  -- END conclusion

end PastaCurves.X86_64
