/-
Copyright Amazon.com, Inc. or its affiliates (the block, adapted from s2n-bignum).
Copyright (c) 2026 the pasta_curves contributors (the transcription and the proofs).
-/
import PastaCurves.AArch64.Spec.Divstep
import PastaCurves.Inversion.Divstep59
import PastaCurves.Inversion.PackedWords

/-!
# Correctness of the inversion's 59-step block

See the parent module's documentation for details. The block packs the low words of `f` and `g` with
the identity matrix (`f mod 2^20 - 2^41`, `g mod 2^20 - 2^62`), runs 20 packed divsteps, decodes the
matrix from the upper bits, forms the next low words, and repeats for 20 and 19 steps, multiplying
the three matrices together. Its proof follows the shared `divstep59` batch by batch. The assembly
departs from it in two ways, and the proof covers both:

* the decoder reads the negated matrix (`sbfx` gives `-u` and `asr` gives `-v`, from `w + 2^20`
  and `w + 2^20 + 2^41`, whose offsets round the low disturbance away), so the second batch runs
  on negated low words, which the recurrence's sign symmetry (`divsteps_neg`, `M_neg`) absorbs;
* the products of two negated matrices restore the sign, and `mneg` and `msub` do so for the
  third.

Each batch is `batch_words`: the step theorem iterated, with its no-wrap hypothesis from Lemma 6′.
-/

set_option exponentiation.threshold 400

namespace PastaCurves.AArch64

open Inversion (State divstep divsteps M packedStart mul_word addw_word pack_f_word pack_g_word
  asr20_word asr20_word44 decode_v20 decode_v19)

-- BEGIN divstep59Block_spec lemmas
/-! ## Words -/

theorem madd_word (x y z : ℕ) (X Y Z : ℤ) (hx : (x : ℤ) = X % 2^64) (hy : (y : ℤ) = Y % 2^64)
    (hz : (z : ℤ) = Z % 2^64) : ((madd x y z : ℕ) : ℤ) = (X * Y + Z) % 2^64 := by
  have h : ((madd x y z : ℕ) : ℤ) = ((x : ℤ) * y + z) % 2^64 := by
    unfold madd regMod; push_cast; norm_num
  rw [h, hx, hy, hz]
  exact ((Int.mod_modEq X _).mul (Int.mod_modEq Y _)).add (Int.mod_modEq Z _)

theorem mneg_word (x y : ℕ) (X Y : ℤ) (hx : (x : ℤ) = X % 2^64) (hy : (y : ℤ) = Y % 2^64) :
    ((mneg x y : ℕ) : ℤ) = (-(X * Y)) % 2^64 := by
  have h : ((mneg x y : ℕ) : ℤ) = (-((x : ℤ) * y)) % 2^64 := by
    unfold mneg regMod
    have h1 : x * y % 2^64 < 2^64 := Nat.mod_lt _ (by norm_num)
    omega
  rw [h, hx, hy]
  exact ((Int.mod_modEq X _).mul (Int.mod_modEq Y _)).neg

theorem msub_word (x y z : ℕ) (X Y Z : ℤ) (hx : (x : ℤ) = X % 2^64) (hy : (y : ℤ) = Y % 2^64)
    (hz : (z : ℤ) = Z % 2^64) : ((msub x y z : ℕ) : ℤ) = (Z - X * Y) % 2^64 := by
  have h : ((msub x y z : ℕ) : ℤ) = ((z : ℤ) - (x : ℤ) * y) % 2^64 := by
    unfold msub regMod
    have h1 : x * y % 2^64 < 2^64 := Nat.mod_lt _ (by norm_num)
    omega
  rw [h, hx, hy, hz]
  exact (Int.mod_modEq Z _).sub ((Int.mod_modEq X _).mul (Int.mod_modEq Y _))

/-- `tst pg, #1` on the packed `g` word reads the parity of the packed `g`. -/
theorem tst_one_z (w : ℕ) (W : ℤ) (hw : (w : ℤ) = W % 2^64) :
    (tstFlags (andw w 1)).z = if W % 2 = 0 then 1 else 0 := by
  have h : andw w 1 = w % 2 := by
    unfold andw; rw [show (1 : ℕ) = 2^1 - 1 by norm_num, Nat.and_two_pow_sub_one_eq_mod]
  show (if andw w 1 = 0 then 1 else 0) = _
  rw [h]
  by_cases hp : W % 2 = 0
  · rw [if_pos hp, if_pos (by omega)]
  · rw [if_neg hp, if_neg (by omega)]

/-! ## The decoder -/

/-- The decoder of a batch's packed word, for `k = 20`: `sbfx` at bit 21 of `w + 2^20` gives `-u`,
and `asr` by 42 of `w + 2^20 + 2^41` gives `-v`, as words. The offsets round the `|φ| < 2^20`
disturbance away, as Lemma 7 does with the opposite sign. The `asr` half is the shared `decode_v20`.
-/
theorem decode20 (w : ℕ) (W φ u v : ℤ) (hw : (w : ℤ) = W % 2^64)
    (hW : W = φ - 2^21 * u - 2^42 * v) (hφ : |φ| < 2^20)
    (hu : -(2 : ℤ)^20 < u ∧ u ≤ 2^20) (hv : -(2 : ℤ)^20 < v ∧ v ≤ 2^20) :
    ((sbfx (addw w 0x100000) 21 21 : ℕ) : ℤ) = (-u) % 2^64 ∧
      ((asr (addw w (addw 0x100000 (lsl 0x100000 21))) 42 : ℕ) : ℤ) = (-v) % 2^64 := by
  have hc : addw 0x100000 (lsl 0x100000 21) = 2^20 + 2^41 := by decide
  rw [hc]
  refine ⟨?_, decode_v20 w W φ u v hw hW hφ hu hv⟩
  rw [abs_lt] at hφ
  have hw64 : w < 2^64 := by omega
  have ha1 : addw w 0x100000 = (w + 2^20) % 2^64 := rfl
  unfold sbfx
  rw [ha1]
  norm_num
  split_ifs <;> omega

/-- The decoder for `k = 19`: `sbfx` at bit 22 and `asr` by 43. -/
theorem decode19 (w : ℕ) (W φ u v : ℤ) (hw : (w : ℤ) = W % 2^64)
    (hW : W = φ - 2^22 * u - 2^43 * v) (hφ : |φ| < 2^20)
    (hu : -(2 : ℤ)^19 < u ∧ u ≤ 2^19) (hv : -(2 : ℤ)^19 < v ∧ v ≤ 2^19) :
    ((sbfx (addw w 0x100000) 22 21 : ℕ) : ℤ) = (-u) % 2^64 ∧
      ((asr (addw w (addw 0x100000 (lsl 0x100000 21))) 43 : ℕ) : ℤ) = (-v) % 2^64 := by
  have hc : addw 0x100000 (lsl 0x100000 21) = 2^20 + 2^41 := by decide
  rw [hc]
  refine ⟨?_, decode_v19 w W φ u v hw hW hφ hu hv⟩
  rw [abs_lt] at hφ
  have hw64 : w < 2^64 := by omega
  have ha1 : addw w 0x100000 = (w + 2^20) % 2^64 := rfl
  unfold sbfx
  rw [ha1]
  norm_num
  split_ifs <;> omega

/-! ## Iterating the step -/

/-- `n` rounds on words carrying the packed state `P` carry `divsteps n P`, given the bounds of
every step: the size of `d`, and Lemma 6′'s no-wrap bound. -/
theorem rounds_words (n : ℕ) (P : State) (hf : P.f % 2 = 1) (hd : P.d % 2 = 1)
    (hD : |P.d| + 2 * n < 2^62) (hg0 : |P.g| < 2^63)
    (hg : ∀ j, j < n → |(divsteps (j + 1) P).g| < 2^62)
    (st : DivstepState) (hst : st.Bounded)
    (ed : (st.d : ℤ) = P.d % 2^64) (ef : (st.f : ℤ) = P.f % 2^64) (eg : (st.g : ℤ) = P.g % 2^64)
    (hz : st.fl.z = if P.g % 2 = 0 then 1 else 0) :
    (divstepRound^[n] st).Bounded ∧
      ((divstepRound^[n] st).d : ℤ) = (divsteps n P).d % 2^64 ∧
      ((divstepRound^[n] st).f : ℤ) = (divsteps n P).f % 2^64 ∧
      ((divstepRound^[n] st).g : ℤ) = (divsteps n P).g % 2^64 ∧
      (divstepRound^[n] st).fl.z = if (divsteps n P).g % 2 = 0 then 1 else 0 := by
  induction n with
  | zero => exact ⟨hst, ed, ef, eg, hz⟩
  | succ n ih =>
    obtain ⟨ihb, ihd, ihf, ihg, ihz⟩ := ih (by omega) (fun j hj => hg j (by omega))
    rw [Function.iterate_succ_apply', Inversion.divsteps_succ']
    have hsd : |(divsteps n P).d| < 2^62 := by
      have := Inversion.divsteps_d_abs_le n P; push_cast at this; omega
    have hsg : |(divsteps n P).g| < 2^63 := by
      rcases n with _ | n
      · exact hg0
      · exact lt_trans (hg n (by omega)) (by norm_num)
    exact divstepRound_spec _ ihb (divsteps n P) (Inversion.divsteps_f_odd n P hf)
      (by rw [Inversion.divsteps_d_emod_two]; exact hd) hsd hsg
      (by rw [← Inversion.divsteps_succ']; exact hg n (by omega)) ihd ihf ihg ihz _ rfl

/-- A batch of `n + 1` steps: `n` rounds and the last step. -/
theorem batch_words (n : ℕ) (P : State) (hf : P.f % 2 = 1) (hd : P.d % 2 = 1)
    (hD : |P.d| + 2 * (n + 1) < 2^62) (hg0 : |P.g| < 2^63)
    (hg : ∀ j, j < n + 1 → |(divsteps (j + 1) P).g| < 2^62)
    (st : DivstepState) (hst : st.Bounded)
    (ed : (st.d : ℤ) = P.d % 2^64) (ef : (st.f : ℤ) = P.f % 2^64) (eg : (st.g : ℤ) = P.g % 2^64)
    (hz : st.fl.z = if P.g % 2 = 0 then 1 else 0) :
    (divstepLast (divstepRound^[n] st)).Bounded ∧
      ((divstepLast (divstepRound^[n] st)).d : ℤ) = (divsteps (n + 1) P).d % 2^64 ∧
      ((divstepLast (divstepRound^[n] st)).f : ℤ) = (divsteps (n + 1) P).f % 2^64 ∧
      ((divstepLast (divstepRound^[n] st)).g : ℤ) = (divsteps (n + 1) P).g % 2^64 := by
  obtain ⟨ihb, ihd, ihf, ihg, ihz⟩ := rounds_words n P hf hd (by omega) hg0
    (fun j hj => hg j (by omega)) st hst ed ef eg hz
  rw [Inversion.divsteps_succ']
  have hsd : |(divsteps n P).d| < 2^62 := by
    have := Inversion.divsteps_d_abs_le n P; push_cast at this; omega
  have hsg : |(divsteps n P).g| < 2^63 := by
    rcases n with _ | m
    · exact hg0
    · exact lt_trans (hg m (by omega)) (by norm_num)
  exact divstepLast_spec _ ihb (divsteps n P) (Inversion.divsteps_f_odd _ P hf)
    (by rw [Inversion.divsteps_d_emod_two]; exact hd) hsd hsg
    (by rw [← Inversion.divsteps_succ']; exact hg n (by omega)) ihd ihf ihg ihz _ rfl

-- END divstep59Block_spec lemmas

-- BEGIN divstep59Block_spec statement
-- The generated skeleton extracts some 375 bindings; the budget is the sum of those small steps,
-- not any one of them.
set_option maxHeartbeats 400000 in
/-- The 59-step block on the words of a state `s` with odd `f` and `d`, and `|d| < 2^61`, which the
ten rounds of the inversion keep: its `d` word carries the `d` after 59 steps, and its matrix words
the 59-step matrix `M 59 s`, in two's complement. -/
theorem divstep59Block_spec (d f0 g0 : ℕ) (s : State) (hsf : s.f % 2 = 1) (hsd : s.d % 2 = 1)
    (hsD : |s.d| < 2^61) (ed : (d : ℤ) = s.d % 2^64) (ef0 : (f0 : ℤ) = s.f % 2^64)
    (eg0 : (g0 : ℤ) = s.g % 2^64) :
    ∀ r, r = divstep59Block d f0 g0 →
      (r.d : ℤ) = (divsteps 59 s).d % 2^64 ∧
        (r.m00 : ℤ) = (M 59 s).a % 2^64 ∧ (r.m01 : ℤ) = (M 59 s).b % 2^64 ∧
        (r.m10 : ℤ) = (M 59 s).c % 2^64 ∧ (r.m11 : ℤ) = (M 59 s).d % 2^64 := by
  intro r hr
  have hd : d < 2^64 := by omega
  have hf0 : f0 < 2^64 := by omega
  have hg0 : g0 < 2^64 := by omega
-- END divstep59Block_spec statement
  -- generated skeleton for `divstep59Block`: do not edit between the annotations
  unfold divstep59Block at hr
  lift_lets -merge at hr
  -- d': argument
  extract_lets -merge +onlyGivenNames d' at hr
  have e_d' : d' = d := rfl
  clear_value d'
  have b_d' : d' < 2^64 := by rw [e_d']; exact hd
  -- f: argument
  extract_lets -merge +onlyGivenNames f at hr
  have e_f : f = f0 := rfl
  clear_value f
  have b_f : f < 2^64 := by rw [e_f]; exact hf0
  -- g: argument
  extract_lets -merge +onlyGivenNames g at hr
  have e_g : g = g0 := rfl
  clear_value g
  have b_g : g < 2^64 := by rw [e_g]; exact hg0
  -- pf: and pf,f,#0xfffff
  extract_lets -merge +onlyGivenNames pf at hr
  have e_pf : pf = andw f 0xfffff := rfl
  clear_value pf
  have b_pf : pf < 2^64 := by rw [e_pf]; exact andw_lt f 0xfffff b_f (by decide)
  -- pf_1: orr pf,pf,#0xfffffe0000000000
  extract_lets -merge +onlyGivenNames pf_1 at hr
  have e_pf_1 : pf_1 = orrw pf 0xfffffe0000000000 := rfl
  clear_value pf_1
  have b_pf_1 : pf_1 < 2^64 := by rw [e_pf_1]; exact orrw_lt pf 0xfffffe0000000000 b_pf (by decide)
  -- pg: and pg,g,#0xfffff
  extract_lets -merge +onlyGivenNames pg at hr
  have e_pg : pg = andw g 0xfffff := rfl
  clear_value pg
  have b_pg : pg < 2^64 := by rw [e_pg]; exact andw_lt g 0xfffff b_g (by decide)
  -- pg_1: orr pg,pg,#0xc000000000000000
  extract_lets -merge +onlyGivenNames pg_1 at hr
  have e_pg_1 : pg_1 = orrw pg 0xc000000000000000 := rfl
  clear_value pg_1
  have b_pg_1 : pg_1 < 2^64 := by rw [e_pg_1]; exact orrw_lt pg 0xc000000000000000 b_pg (by decide)
  -- fl: tst pg,#1
  extract_lets -merge +onlyGivenNames fl at hr
  have e_fl : fl = tstFlags (andw pg_1 1) := rfl
  clear_value fl
  -- step1: divstep!(), invocation 1
  extract_lets -merge +onlyGivenNames step1 at hr
  have e_step1 : step1 = divstepRound ⟨d', pf_1, pg_1, fl⟩ := rfl
  clear_value step1
  -- d_1: divstep!(), invocation 1 output
  extract_lets -merge +onlyGivenNames d_1 at hr
  have e_d_1 : d_1 = step1.d := rfl
  clear_value d_1
  -- pf_2: divstep!(), invocation 1 output
  extract_lets -merge +onlyGivenNames pf_2 at hr
  have e_pf_2 : pf_2 = step1.f := rfl
  clear_value pf_2
  -- pg_2: divstep!(), invocation 1 output
  extract_lets -merge +onlyGivenNames pg_2 at hr
  have e_pg_2 : pg_2 = step1.g := rfl
  clear_value pg_2
  -- fl_1: divstep!(), invocation 1 output
  extract_lets -merge +onlyGivenNames fl_1 at hr
  have e_fl_1 : fl_1 = step1.fl := rfl
  clear_value fl_1
  -- step2: divstep!(), invocation 2
  extract_lets -merge +onlyGivenNames step2 at hr
  have e_step2 : step2 = divstepRound ⟨d_1, pf_2, pg_2, fl_1⟩ := rfl
  clear_value step2
  -- d_2: divstep!(), invocation 2 output
  extract_lets -merge +onlyGivenNames d_2 at hr
  have e_d_2 : d_2 = step2.d := rfl
  clear_value d_2
  -- pf_3: divstep!(), invocation 2 output
  extract_lets -merge +onlyGivenNames pf_3 at hr
  have e_pf_3 : pf_3 = step2.f := rfl
  clear_value pf_3
  -- pg_3: divstep!(), invocation 2 output
  extract_lets -merge +onlyGivenNames pg_3 at hr
  have e_pg_3 : pg_3 = step2.g := rfl
  clear_value pg_3
  -- fl_2: divstep!(), invocation 2 output
  extract_lets -merge +onlyGivenNames fl_2 at hr
  have e_fl_2 : fl_2 = step2.fl := rfl
  clear_value fl_2
  -- step3: divstep!(), invocation 3
  extract_lets -merge +onlyGivenNames step3 at hr
  have e_step3 : step3 = divstepRound ⟨d_2, pf_3, pg_3, fl_2⟩ := rfl
  clear_value step3
  -- d_3: divstep!(), invocation 3 output
  extract_lets -merge +onlyGivenNames d_3 at hr
  have e_d_3 : d_3 = step3.d := rfl
  clear_value d_3
  -- pf_4: divstep!(), invocation 3 output
  extract_lets -merge +onlyGivenNames pf_4 at hr
  have e_pf_4 : pf_4 = step3.f := rfl
  clear_value pf_4
  -- pg_4: divstep!(), invocation 3 output
  extract_lets -merge +onlyGivenNames pg_4 at hr
  have e_pg_4 : pg_4 = step3.g := rfl
  clear_value pg_4
  -- fl_3: divstep!(), invocation 3 output
  extract_lets -merge +onlyGivenNames fl_3 at hr
  have e_fl_3 : fl_3 = step3.fl := rfl
  clear_value fl_3
  -- step4: divstep!(), invocation 4
  extract_lets -merge +onlyGivenNames step4 at hr
  have e_step4 : step4 = divstepRound ⟨d_3, pf_4, pg_4, fl_3⟩ := rfl
  clear_value step4
  -- d_4: divstep!(), invocation 4 output
  extract_lets -merge +onlyGivenNames d_4 at hr
  have e_d_4 : d_4 = step4.d := rfl
  clear_value d_4
  -- pf_5: divstep!(), invocation 4 output
  extract_lets -merge +onlyGivenNames pf_5 at hr
  have e_pf_5 : pf_5 = step4.f := rfl
  clear_value pf_5
  -- pg_5: divstep!(), invocation 4 output
  extract_lets -merge +onlyGivenNames pg_5 at hr
  have e_pg_5 : pg_5 = step4.g := rfl
  clear_value pg_5
  -- fl_4: divstep!(), invocation 4 output
  extract_lets -merge +onlyGivenNames fl_4 at hr
  have e_fl_4 : fl_4 = step4.fl := rfl
  clear_value fl_4
  -- step5: divstep!(), invocation 5
  extract_lets -merge +onlyGivenNames step5 at hr
  have e_step5 : step5 = divstepRound ⟨d_4, pf_5, pg_5, fl_4⟩ := rfl
  clear_value step5
  -- d_5: divstep!(), invocation 5 output
  extract_lets -merge +onlyGivenNames d_5 at hr
  have e_d_5 : d_5 = step5.d := rfl
  clear_value d_5
  -- pf_6: divstep!(), invocation 5 output
  extract_lets -merge +onlyGivenNames pf_6 at hr
  have e_pf_6 : pf_6 = step5.f := rfl
  clear_value pf_6
  -- pg_6: divstep!(), invocation 5 output
  extract_lets -merge +onlyGivenNames pg_6 at hr
  have e_pg_6 : pg_6 = step5.g := rfl
  clear_value pg_6
  -- fl_5: divstep!(), invocation 5 output
  extract_lets -merge +onlyGivenNames fl_5 at hr
  have e_fl_5 : fl_5 = step5.fl := rfl
  clear_value fl_5
  -- step6: divstep!(), invocation 6
  extract_lets -merge +onlyGivenNames step6 at hr
  have e_step6 : step6 = divstepRound ⟨d_5, pf_6, pg_6, fl_5⟩ := rfl
  clear_value step6
  -- d_6: divstep!(), invocation 6 output
  extract_lets -merge +onlyGivenNames d_6 at hr
  have e_d_6 : d_6 = step6.d := rfl
  clear_value d_6
  -- pf_7: divstep!(), invocation 6 output
  extract_lets -merge +onlyGivenNames pf_7 at hr
  have e_pf_7 : pf_7 = step6.f := rfl
  clear_value pf_7
  -- pg_7: divstep!(), invocation 6 output
  extract_lets -merge +onlyGivenNames pg_7 at hr
  have e_pg_7 : pg_7 = step6.g := rfl
  clear_value pg_7
  -- fl_6: divstep!(), invocation 6 output
  extract_lets -merge +onlyGivenNames fl_6 at hr
  have e_fl_6 : fl_6 = step6.fl := rfl
  clear_value fl_6
  -- step7: divstep!(), invocation 7
  extract_lets -merge +onlyGivenNames step7 at hr
  have e_step7 : step7 = divstepRound ⟨d_6, pf_7, pg_7, fl_6⟩ := rfl
  clear_value step7
  -- d_7: divstep!(), invocation 7 output
  extract_lets -merge +onlyGivenNames d_7 at hr
  have e_d_7 : d_7 = step7.d := rfl
  clear_value d_7
  -- pf_8: divstep!(), invocation 7 output
  extract_lets -merge +onlyGivenNames pf_8 at hr
  have e_pf_8 : pf_8 = step7.f := rfl
  clear_value pf_8
  -- pg_8: divstep!(), invocation 7 output
  extract_lets -merge +onlyGivenNames pg_8 at hr
  have e_pg_8 : pg_8 = step7.g := rfl
  clear_value pg_8
  -- fl_7: divstep!(), invocation 7 output
  extract_lets -merge +onlyGivenNames fl_7 at hr
  have e_fl_7 : fl_7 = step7.fl := rfl
  clear_value fl_7
  -- step8: divstep!(), invocation 8
  extract_lets -merge +onlyGivenNames step8 at hr
  have e_step8 : step8 = divstepRound ⟨d_7, pf_8, pg_8, fl_7⟩ := rfl
  clear_value step8
  -- d_8: divstep!(), invocation 8 output
  extract_lets -merge +onlyGivenNames d_8 at hr
  have e_d_8 : d_8 = step8.d := rfl
  clear_value d_8
  -- pf_9: divstep!(), invocation 8 output
  extract_lets -merge +onlyGivenNames pf_9 at hr
  have e_pf_9 : pf_9 = step8.f := rfl
  clear_value pf_9
  -- pg_9: divstep!(), invocation 8 output
  extract_lets -merge +onlyGivenNames pg_9 at hr
  have e_pg_9 : pg_9 = step8.g := rfl
  clear_value pg_9
  -- fl_8: divstep!(), invocation 8 output
  extract_lets -merge +onlyGivenNames fl_8 at hr
  have e_fl_8 : fl_8 = step8.fl := rfl
  clear_value fl_8
  -- step9: divstep!(), invocation 9
  extract_lets -merge +onlyGivenNames step9 at hr
  have e_step9 : step9 = divstepRound ⟨d_8, pf_9, pg_9, fl_8⟩ := rfl
  clear_value step9
  -- d_9: divstep!(), invocation 9 output
  extract_lets -merge +onlyGivenNames d_9 at hr
  have e_d_9 : d_9 = step9.d := rfl
  clear_value d_9
  -- pf_10: divstep!(), invocation 9 output
  extract_lets -merge +onlyGivenNames pf_10 at hr
  have e_pf_10 : pf_10 = step9.f := rfl
  clear_value pf_10
  -- pg_10: divstep!(), invocation 9 output
  extract_lets -merge +onlyGivenNames pg_10 at hr
  have e_pg_10 : pg_10 = step9.g := rfl
  clear_value pg_10
  -- fl_9: divstep!(), invocation 9 output
  extract_lets -merge +onlyGivenNames fl_9 at hr
  have e_fl_9 : fl_9 = step9.fl := rfl
  clear_value fl_9
  -- step10: divstep!(), invocation 10
  extract_lets -merge +onlyGivenNames step10 at hr
  have e_step10 : step10 = divstepRound ⟨d_9, pf_10, pg_10, fl_9⟩ := rfl
  clear_value step10
  -- d_10: divstep!(), invocation 10 output
  extract_lets -merge +onlyGivenNames d_10 at hr
  have e_d_10 : d_10 = step10.d := rfl
  clear_value d_10
  -- pf_11: divstep!(), invocation 10 output
  extract_lets -merge +onlyGivenNames pf_11 at hr
  have e_pf_11 : pf_11 = step10.f := rfl
  clear_value pf_11
  -- pg_11: divstep!(), invocation 10 output
  extract_lets -merge +onlyGivenNames pg_11 at hr
  have e_pg_11 : pg_11 = step10.g := rfl
  clear_value pg_11
  -- fl_10: divstep!(), invocation 10 output
  extract_lets -merge +onlyGivenNames fl_10 at hr
  have e_fl_10 : fl_10 = step10.fl := rfl
  clear_value fl_10
  -- step11: divstep!(), invocation 11
  extract_lets -merge +onlyGivenNames step11 at hr
  have e_step11 : step11 = divstepRound ⟨d_10, pf_11, pg_11, fl_10⟩ := rfl
  clear_value step11
  -- d_11: divstep!(), invocation 11 output
  extract_lets -merge +onlyGivenNames d_11 at hr
  have e_d_11 : d_11 = step11.d := rfl
  clear_value d_11
  -- pf_12: divstep!(), invocation 11 output
  extract_lets -merge +onlyGivenNames pf_12 at hr
  have e_pf_12 : pf_12 = step11.f := rfl
  clear_value pf_12
  -- pg_12: divstep!(), invocation 11 output
  extract_lets -merge +onlyGivenNames pg_12 at hr
  have e_pg_12 : pg_12 = step11.g := rfl
  clear_value pg_12
  -- fl_11: divstep!(), invocation 11 output
  extract_lets -merge +onlyGivenNames fl_11 at hr
  have e_fl_11 : fl_11 = step11.fl := rfl
  clear_value fl_11
  -- step12: divstep!(), invocation 12
  extract_lets -merge +onlyGivenNames step12 at hr
  have e_step12 : step12 = divstepRound ⟨d_11, pf_12, pg_12, fl_11⟩ := rfl
  clear_value step12
  -- d_12: divstep!(), invocation 12 output
  extract_lets -merge +onlyGivenNames d_12 at hr
  have e_d_12 : d_12 = step12.d := rfl
  clear_value d_12
  -- pf_13: divstep!(), invocation 12 output
  extract_lets -merge +onlyGivenNames pf_13 at hr
  have e_pf_13 : pf_13 = step12.f := rfl
  clear_value pf_13
  -- pg_13: divstep!(), invocation 12 output
  extract_lets -merge +onlyGivenNames pg_13 at hr
  have e_pg_13 : pg_13 = step12.g := rfl
  clear_value pg_13
  -- fl_12: divstep!(), invocation 12 output
  extract_lets -merge +onlyGivenNames fl_12 at hr
  have e_fl_12 : fl_12 = step12.fl := rfl
  clear_value fl_12
  -- step13: divstep!(), invocation 13
  extract_lets -merge +onlyGivenNames step13 at hr
  have e_step13 : step13 = divstepRound ⟨d_12, pf_13, pg_13, fl_12⟩ := rfl
  clear_value step13
  -- d_13: divstep!(), invocation 13 output
  extract_lets -merge +onlyGivenNames d_13 at hr
  have e_d_13 : d_13 = step13.d := rfl
  clear_value d_13
  -- pf_14: divstep!(), invocation 13 output
  extract_lets -merge +onlyGivenNames pf_14 at hr
  have e_pf_14 : pf_14 = step13.f := rfl
  clear_value pf_14
  -- pg_14: divstep!(), invocation 13 output
  extract_lets -merge +onlyGivenNames pg_14 at hr
  have e_pg_14 : pg_14 = step13.g := rfl
  clear_value pg_14
  -- fl_13: divstep!(), invocation 13 output
  extract_lets -merge +onlyGivenNames fl_13 at hr
  have e_fl_13 : fl_13 = step13.fl := rfl
  clear_value fl_13
  -- step14: divstep!(), invocation 14
  extract_lets -merge +onlyGivenNames step14 at hr
  have e_step14 : step14 = divstepRound ⟨d_13, pf_14, pg_14, fl_13⟩ := rfl
  clear_value step14
  -- d_14: divstep!(), invocation 14 output
  extract_lets -merge +onlyGivenNames d_14 at hr
  have e_d_14 : d_14 = step14.d := rfl
  clear_value d_14
  -- pf_15: divstep!(), invocation 14 output
  extract_lets -merge +onlyGivenNames pf_15 at hr
  have e_pf_15 : pf_15 = step14.f := rfl
  clear_value pf_15
  -- pg_15: divstep!(), invocation 14 output
  extract_lets -merge +onlyGivenNames pg_15 at hr
  have e_pg_15 : pg_15 = step14.g := rfl
  clear_value pg_15
  -- fl_14: divstep!(), invocation 14 output
  extract_lets -merge +onlyGivenNames fl_14 at hr
  have e_fl_14 : fl_14 = step14.fl := rfl
  clear_value fl_14
  -- step15: divstep!(), invocation 15
  extract_lets -merge +onlyGivenNames step15 at hr
  have e_step15 : step15 = divstepRound ⟨d_14, pf_15, pg_15, fl_14⟩ := rfl
  clear_value step15
  -- d_15: divstep!(), invocation 15 output
  extract_lets -merge +onlyGivenNames d_15 at hr
  have e_d_15 : d_15 = step15.d := rfl
  clear_value d_15
  -- pf_16: divstep!(), invocation 15 output
  extract_lets -merge +onlyGivenNames pf_16 at hr
  have e_pf_16 : pf_16 = step15.f := rfl
  clear_value pf_16
  -- pg_16: divstep!(), invocation 15 output
  extract_lets -merge +onlyGivenNames pg_16 at hr
  have e_pg_16 : pg_16 = step15.g := rfl
  clear_value pg_16
  -- fl_15: divstep!(), invocation 15 output
  extract_lets -merge +onlyGivenNames fl_15 at hr
  have e_fl_15 : fl_15 = step15.fl := rfl
  clear_value fl_15
  -- step16: divstep!(), invocation 16
  extract_lets -merge +onlyGivenNames step16 at hr
  have e_step16 : step16 = divstepRound ⟨d_15, pf_16, pg_16, fl_15⟩ := rfl
  clear_value step16
  -- d_16: divstep!(), invocation 16 output
  extract_lets -merge +onlyGivenNames d_16 at hr
  have e_d_16 : d_16 = step16.d := rfl
  clear_value d_16
  -- pf_17: divstep!(), invocation 16 output
  extract_lets -merge +onlyGivenNames pf_17 at hr
  have e_pf_17 : pf_17 = step16.f := rfl
  clear_value pf_17
  -- pg_17: divstep!(), invocation 16 output
  extract_lets -merge +onlyGivenNames pg_17 at hr
  have e_pg_17 : pg_17 = step16.g := rfl
  clear_value pg_17
  -- fl_16: divstep!(), invocation 16 output
  extract_lets -merge +onlyGivenNames fl_16 at hr
  have e_fl_16 : fl_16 = step16.fl := rfl
  clear_value fl_16
  -- step17: divstep!(), invocation 17
  extract_lets -merge +onlyGivenNames step17 at hr
  have e_step17 : step17 = divstepRound ⟨d_16, pf_17, pg_17, fl_16⟩ := rfl
  clear_value step17
  -- d_17: divstep!(), invocation 17 output
  extract_lets -merge +onlyGivenNames d_17 at hr
  have e_d_17 : d_17 = step17.d := rfl
  clear_value d_17
  -- pf_18: divstep!(), invocation 17 output
  extract_lets -merge +onlyGivenNames pf_18 at hr
  have e_pf_18 : pf_18 = step17.f := rfl
  clear_value pf_18
  -- pg_18: divstep!(), invocation 17 output
  extract_lets -merge +onlyGivenNames pg_18 at hr
  have e_pg_18 : pg_18 = step17.g := rfl
  clear_value pg_18
  -- fl_17: divstep!(), invocation 17 output
  extract_lets -merge +onlyGivenNames fl_17 at hr
  have e_fl_17 : fl_17 = step17.fl := rfl
  clear_value fl_17
  -- step18: divstep!(), invocation 18
  extract_lets -merge +onlyGivenNames step18 at hr
  have e_step18 : step18 = divstepRound ⟨d_17, pf_18, pg_18, fl_17⟩ := rfl
  clear_value step18
  -- d_18: divstep!(), invocation 18 output
  extract_lets -merge +onlyGivenNames d_18 at hr
  have e_d_18 : d_18 = step18.d := rfl
  clear_value d_18
  -- pf_19: divstep!(), invocation 18 output
  extract_lets -merge +onlyGivenNames pf_19 at hr
  have e_pf_19 : pf_19 = step18.f := rfl
  clear_value pf_19
  -- pg_19: divstep!(), invocation 18 output
  extract_lets -merge +onlyGivenNames pg_19 at hr
  have e_pg_19 : pg_19 = step18.g := rfl
  clear_value pg_19
  -- fl_18: divstep!(), invocation 18 output
  extract_lets -merge +onlyGivenNames fl_18 at hr
  have e_fl_18 : fl_18 = step18.fl := rfl
  clear_value fl_18
  -- step19: divstep!(), invocation 19
  extract_lets -merge +onlyGivenNames step19 at hr
  have e_step19 : step19 = divstepRound ⟨d_18, pf_19, pg_19, fl_18⟩ := rfl
  clear_value step19
  -- d_19: divstep!(), invocation 19 output
  extract_lets -merge +onlyGivenNames d_19 at hr
  have e_d_19 : d_19 = step19.d := rfl
  clear_value d_19
  -- pf_20: divstep!(), invocation 19 output
  extract_lets -merge +onlyGivenNames pf_20 at hr
  have e_pf_20 : pf_20 = step19.f := rfl
  clear_value pf_20
  -- pg_20: divstep!(), invocation 19 output
  extract_lets -merge +onlyGivenNames pg_20 at hr
  have e_pg_20 : pg_20 = step19.g := rfl
  clear_value pg_20
  -- fl_19: divstep!(), invocation 19 output
  extract_lets -merge +onlyGivenNames fl_19 at hr
  have e_fl_19 : fl_19 = step19.fl := rfl
  clear_value fl_19
  -- step20: divstep!(last), invocation 20
  extract_lets -merge +onlyGivenNames step20 at hr
  have e_step20 : step20 = divstepLast ⟨d_19, pf_20, pg_20, fl_19⟩ := rfl
  clear_value step20
  -- d_20: divstep!(last), invocation 20 output
  extract_lets -merge +onlyGivenNames d_20 at hr
  have e_d_20 : d_20 = step20.d := rfl
  clear_value d_20
  -- pf_21: divstep!(last), invocation 20 output
  extract_lets -merge +onlyGivenNames pf_21 at hr
  have e_pf_21 : pf_21 = step20.f := rfl
  clear_value pf_21
  -- pg_21: divstep!(last), invocation 20 output
  extract_lets -merge +onlyGivenNames pg_21 at hr
  have e_pg_21 : pg_21 = step20.g := rfl
  clear_value pg_21
  -- a00: add a00,pf,#0x100,lsl #12
  extract_lets -merge +onlyGivenNames a00 at hr
  have e_a00 : a00 = addw pf_21 0x100000 := rfl
  clear_value a00
  have b_a00 : a00 < 2^64 := by rw [e_a00]; exact addw_lt pf_21 0x100000
  -- a00_1: sbfx a00,a00,#21,#21
  extract_lets -merge +onlyGivenNames a00_1 at hr
  have e_a00_1 : a00_1 = sbfx a00 21 21 := rfl
  clear_value a00_1
  have b_a00_1 : a00_1 < 2^64 := by rw [e_a00_1]; exact sbfx_lt a00 21 21 b_a00
  -- a11: mov a11,#0x100000
  extract_lets -merge +onlyGivenNames a11 at hr
  have e_a11 : a11 = 0x100000 := rfl
  clear_value a11
  have b_a11 : a11 < 2^64 := by rw [e_a11]; decide
  -- a11_1: add a11,a11,a11,lsl #21
  extract_lets -merge +onlyGivenNames a11_1 at hr
  have e_a11_1 : a11_1 = addw a11 (lsl a11 21) := rfl
  clear_value a11_1
  have b_a11_1 : a11_1 < 2^64 := by rw [e_a11_1]; exact addw_lt a11 (lsl a11 21)
  -- a01: add a01,pf,a11
  extract_lets -merge +onlyGivenNames a01 at hr
  have e_a01 : a01 = addw pf_21 a11_1 := rfl
  clear_value a01
  have b_a01 : a01 < 2^64 := by rw [e_a01]; exact addw_lt pf_21 a11_1
  -- a01_1: asr a01,a01,#42
  extract_lets -merge +onlyGivenNames a01_1 at hr
  have e_a01_1 : a01_1 = asr a01 42 := rfl
  clear_value a01_1
  have b_a01_1 : a01_1 < 2^64 := by rw [e_a01_1]; exact asr_lt a01 42 b_a01
  -- a10: add a10,pg,#0x100,lsl #12
  extract_lets -merge +onlyGivenNames a10 at hr
  have e_a10 : a10 = addw pg_21 0x100000 := rfl
  clear_value a10
  have b_a10 : a10 < 2^64 := by rw [e_a10]; exact addw_lt pg_21 0x100000
  -- a10_1: sbfx a10,a10,#21,#21
  extract_lets -merge +onlyGivenNames a10_1 at hr
  have e_a10_1 : a10_1 = sbfx a10 21 21 := rfl
  clear_value a10_1
  have b_a10_1 : a10_1 < 2^64 := by rw [e_a10_1]; exact sbfx_lt a10 21 21 b_a10
  -- a11_2: add a11,pg,a11
  extract_lets -merge +onlyGivenNames a11_2 at hr
  have e_a11_2 : a11_2 = addw pg_21 a11_1 := rfl
  clear_value a11_2
  have b_a11_2 : a11_2 < 2^64 := by rw [e_a11_2]; exact addw_lt pg_21 a11_1
  -- a11_3: asr a11,a11,#42
  extract_lets -merge +onlyGivenNames a11_3 at hr
  have e_a11_3 : a11_3 = asr a11_2 42 := rfl
  clear_value a11_3
  have b_a11_3 : a11_3 < 2^64 := by rw [e_a11_3]; exact asr_lt a11_2 42 b_a11_2
  -- t: mul t,a00,f
  extract_lets -merge +onlyGivenNames t at hr
  have e_t : t = a00_1 * f % 2^64 := rfl
  clear_value t
  have b_t : t < 2^64 := by rw [e_t]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t2: mul t2,a01,g
  extract_lets -merge +onlyGivenNames t2 at hr
  have e_t2 : t2 = a01_1 * g % 2^64 := rfl
  clear_value t2
  have b_t2 : t2 < 2^64 := by rw [e_t2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- f_1: mul f,a10,f
  extract_lets -merge +onlyGivenNames f_1 at hr
  have e_f_1 : f_1 = a10_1 * f % 2^64 := rfl
  clear_value f_1
  have b_f_1 : f_1 < 2^64 := by rw [e_f_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- g_1: mul g,a11,g
  extract_lets -merge +onlyGivenNames g_1 at hr
  have e_g_1 : g_1 = a11_3 * g % 2^64 := rfl
  clear_value g_1
  have b_g_1 : g_1 < 2^64 := by rw [e_g_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- pf_22: add pf,t,t2
  extract_lets -merge +onlyGivenNames pf_22 at hr
  have e_pf_22 : pf_22 = addw t t2 := rfl
  clear_value pf_22
  have b_pf_22 : pf_22 < 2^64 := by rw [e_pf_22]; exact addw_lt t t2
  -- pg_22: add pg,f,g
  extract_lets -merge +onlyGivenNames pg_22 at hr
  have e_pg_22 : pg_22 = addw f_1 g_1 := rfl
  clear_value pg_22
  have b_pg_22 : pg_22 < 2^64 := by rw [e_pg_22]; exact addw_lt f_1 g_1
  -- f_2: asr f,pf,#20
  extract_lets -merge +onlyGivenNames f_2 at hr
  have e_f_2 : f_2 = asr pf_22 20 := rfl
  clear_value f_2
  have b_f_2 : f_2 < 2^64 := by rw [e_f_2]; exact asr_lt pf_22 20 b_pf_22
  -- g_2: asr g,pg,#20
  extract_lets -merge +onlyGivenNames g_2 at hr
  have e_g_2 : g_2 = asr pg_22 20 := rfl
  clear_value g_2
  have b_g_2 : g_2 < 2^64 := by rw [e_g_2]; exact asr_lt pg_22 20 b_pg_22
  -- pf_23: and pf,f,#0xfffff
  extract_lets -merge +onlyGivenNames pf_23 at hr
  have e_pf_23 : pf_23 = andw f_2 0xfffff := rfl
  clear_value pf_23
  have b_pf_23 : pf_23 < 2^64 := by rw [e_pf_23]; exact andw_lt f_2 0xfffff b_f_2 (by decide)
  -- pf_24: orr pf,pf,#0xfffffe0000000000
  extract_lets -merge +onlyGivenNames pf_24 at hr
  have e_pf_24 : pf_24 = orrw pf_23 0xfffffe0000000000 := rfl
  clear_value pf_24
  have b_pf_24 : pf_24 < 2^64 := by rw [e_pf_24]; exact orrw_lt pf_23 0xfffffe0000000000 b_pf_23 (by decide)
  -- pg_23: and pg,g,#0xfffff
  extract_lets -merge +onlyGivenNames pg_23 at hr
  have e_pg_23 : pg_23 = andw g_2 0xfffff := rfl
  clear_value pg_23
  have b_pg_23 : pg_23 < 2^64 := by rw [e_pg_23]; exact andw_lt g_2 0xfffff b_g_2 (by decide)
  -- pg_24: orr pg,pg,#0xc000000000000000
  extract_lets -merge +onlyGivenNames pg_24 at hr
  have e_pg_24 : pg_24 = orrw pg_23 0xc000000000000000 := rfl
  clear_value pg_24
  have b_pg_24 : pg_24 < 2^64 := by rw [e_pg_24]; exact orrw_lt pg_23 0xc000000000000000 b_pg_23 (by decide)
  -- fl_20: tst pg,#1
  extract_lets -merge +onlyGivenNames fl_20 at hr
  have e_fl_20 : fl_20 = tstFlags (andw pg_24 1) := rfl
  clear_value fl_20
  -- step21: divstep!(), invocation 21
  extract_lets -merge +onlyGivenNames step21 at hr
  have e_step21 : step21 = divstepRound ⟨d_20, pf_24, pg_24, fl_20⟩ := rfl
  clear_value step21
  -- d_21: divstep!(), invocation 21 output
  extract_lets -merge +onlyGivenNames d_21 at hr
  have e_d_21 : d_21 = step21.d := rfl
  clear_value d_21
  -- pf_25: divstep!(), invocation 21 output
  extract_lets -merge +onlyGivenNames pf_25 at hr
  have e_pf_25 : pf_25 = step21.f := rfl
  clear_value pf_25
  -- pg_25: divstep!(), invocation 21 output
  extract_lets -merge +onlyGivenNames pg_25 at hr
  have e_pg_25 : pg_25 = step21.g := rfl
  clear_value pg_25
  -- fl_21: divstep!(), invocation 21 output
  extract_lets -merge +onlyGivenNames fl_21 at hr
  have e_fl_21 : fl_21 = step21.fl := rfl
  clear_value fl_21
  -- step22: divstep!(), invocation 22
  extract_lets -merge +onlyGivenNames step22 at hr
  have e_step22 : step22 = divstepRound ⟨d_21, pf_25, pg_25, fl_21⟩ := rfl
  clear_value step22
  -- d_22: divstep!(), invocation 22 output
  extract_lets -merge +onlyGivenNames d_22 at hr
  have e_d_22 : d_22 = step22.d := rfl
  clear_value d_22
  -- pf_26: divstep!(), invocation 22 output
  extract_lets -merge +onlyGivenNames pf_26 at hr
  have e_pf_26 : pf_26 = step22.f := rfl
  clear_value pf_26
  -- pg_26: divstep!(), invocation 22 output
  extract_lets -merge +onlyGivenNames pg_26 at hr
  have e_pg_26 : pg_26 = step22.g := rfl
  clear_value pg_26
  -- fl_22: divstep!(), invocation 22 output
  extract_lets -merge +onlyGivenNames fl_22 at hr
  have e_fl_22 : fl_22 = step22.fl := rfl
  clear_value fl_22
  -- step23: divstep!(), invocation 23
  extract_lets -merge +onlyGivenNames step23 at hr
  have e_step23 : step23 = divstepRound ⟨d_22, pf_26, pg_26, fl_22⟩ := rfl
  clear_value step23
  -- d_23: divstep!(), invocation 23 output
  extract_lets -merge +onlyGivenNames d_23 at hr
  have e_d_23 : d_23 = step23.d := rfl
  clear_value d_23
  -- pf_27: divstep!(), invocation 23 output
  extract_lets -merge +onlyGivenNames pf_27 at hr
  have e_pf_27 : pf_27 = step23.f := rfl
  clear_value pf_27
  -- pg_27: divstep!(), invocation 23 output
  extract_lets -merge +onlyGivenNames pg_27 at hr
  have e_pg_27 : pg_27 = step23.g := rfl
  clear_value pg_27
  -- fl_23: divstep!(), invocation 23 output
  extract_lets -merge +onlyGivenNames fl_23 at hr
  have e_fl_23 : fl_23 = step23.fl := rfl
  clear_value fl_23
  -- step24: divstep!(), invocation 24
  extract_lets -merge +onlyGivenNames step24 at hr
  have e_step24 : step24 = divstepRound ⟨d_23, pf_27, pg_27, fl_23⟩ := rfl
  clear_value step24
  -- d_24: divstep!(), invocation 24 output
  extract_lets -merge +onlyGivenNames d_24 at hr
  have e_d_24 : d_24 = step24.d := rfl
  clear_value d_24
  -- pf_28: divstep!(), invocation 24 output
  extract_lets -merge +onlyGivenNames pf_28 at hr
  have e_pf_28 : pf_28 = step24.f := rfl
  clear_value pf_28
  -- pg_28: divstep!(), invocation 24 output
  extract_lets -merge +onlyGivenNames pg_28 at hr
  have e_pg_28 : pg_28 = step24.g := rfl
  clear_value pg_28
  -- fl_24: divstep!(), invocation 24 output
  extract_lets -merge +onlyGivenNames fl_24 at hr
  have e_fl_24 : fl_24 = step24.fl := rfl
  clear_value fl_24
  -- step25: divstep!(), invocation 25
  extract_lets -merge +onlyGivenNames step25 at hr
  have e_step25 : step25 = divstepRound ⟨d_24, pf_28, pg_28, fl_24⟩ := rfl
  clear_value step25
  -- d_25: divstep!(), invocation 25 output
  extract_lets -merge +onlyGivenNames d_25 at hr
  have e_d_25 : d_25 = step25.d := rfl
  clear_value d_25
  -- pf_29: divstep!(), invocation 25 output
  extract_lets -merge +onlyGivenNames pf_29 at hr
  have e_pf_29 : pf_29 = step25.f := rfl
  clear_value pf_29
  -- pg_29: divstep!(), invocation 25 output
  extract_lets -merge +onlyGivenNames pg_29 at hr
  have e_pg_29 : pg_29 = step25.g := rfl
  clear_value pg_29
  -- fl_25: divstep!(), invocation 25 output
  extract_lets -merge +onlyGivenNames fl_25 at hr
  have e_fl_25 : fl_25 = step25.fl := rfl
  clear_value fl_25
  -- step26: divstep!(), invocation 26
  extract_lets -merge +onlyGivenNames step26 at hr
  have e_step26 : step26 = divstepRound ⟨d_25, pf_29, pg_29, fl_25⟩ := rfl
  clear_value step26
  -- d_26: divstep!(), invocation 26 output
  extract_lets -merge +onlyGivenNames d_26 at hr
  have e_d_26 : d_26 = step26.d := rfl
  clear_value d_26
  -- pf_30: divstep!(), invocation 26 output
  extract_lets -merge +onlyGivenNames pf_30 at hr
  have e_pf_30 : pf_30 = step26.f := rfl
  clear_value pf_30
  -- pg_30: divstep!(), invocation 26 output
  extract_lets -merge +onlyGivenNames pg_30 at hr
  have e_pg_30 : pg_30 = step26.g := rfl
  clear_value pg_30
  -- fl_26: divstep!(), invocation 26 output
  extract_lets -merge +onlyGivenNames fl_26 at hr
  have e_fl_26 : fl_26 = step26.fl := rfl
  clear_value fl_26
  -- step27: divstep!(), invocation 27
  extract_lets -merge +onlyGivenNames step27 at hr
  have e_step27 : step27 = divstepRound ⟨d_26, pf_30, pg_30, fl_26⟩ := rfl
  clear_value step27
  -- d_27: divstep!(), invocation 27 output
  extract_lets -merge +onlyGivenNames d_27 at hr
  have e_d_27 : d_27 = step27.d := rfl
  clear_value d_27
  -- pf_31: divstep!(), invocation 27 output
  extract_lets -merge +onlyGivenNames pf_31 at hr
  have e_pf_31 : pf_31 = step27.f := rfl
  clear_value pf_31
  -- pg_31: divstep!(), invocation 27 output
  extract_lets -merge +onlyGivenNames pg_31 at hr
  have e_pg_31 : pg_31 = step27.g := rfl
  clear_value pg_31
  -- fl_27: divstep!(), invocation 27 output
  extract_lets -merge +onlyGivenNames fl_27 at hr
  have e_fl_27 : fl_27 = step27.fl := rfl
  clear_value fl_27
  -- step28: divstep!(), invocation 28
  extract_lets -merge +onlyGivenNames step28 at hr
  have e_step28 : step28 = divstepRound ⟨d_27, pf_31, pg_31, fl_27⟩ := rfl
  clear_value step28
  -- d_28: divstep!(), invocation 28 output
  extract_lets -merge +onlyGivenNames d_28 at hr
  have e_d_28 : d_28 = step28.d := rfl
  clear_value d_28
  -- pf_32: divstep!(), invocation 28 output
  extract_lets -merge +onlyGivenNames pf_32 at hr
  have e_pf_32 : pf_32 = step28.f := rfl
  clear_value pf_32
  -- pg_32: divstep!(), invocation 28 output
  extract_lets -merge +onlyGivenNames pg_32 at hr
  have e_pg_32 : pg_32 = step28.g := rfl
  clear_value pg_32
  -- fl_28: divstep!(), invocation 28 output
  extract_lets -merge +onlyGivenNames fl_28 at hr
  have e_fl_28 : fl_28 = step28.fl := rfl
  clear_value fl_28
  -- step29: divstep!(), invocation 29
  extract_lets -merge +onlyGivenNames step29 at hr
  have e_step29 : step29 = divstepRound ⟨d_28, pf_32, pg_32, fl_28⟩ := rfl
  clear_value step29
  -- d_29: divstep!(), invocation 29 output
  extract_lets -merge +onlyGivenNames d_29 at hr
  have e_d_29 : d_29 = step29.d := rfl
  clear_value d_29
  -- pf_33: divstep!(), invocation 29 output
  extract_lets -merge +onlyGivenNames pf_33 at hr
  have e_pf_33 : pf_33 = step29.f := rfl
  clear_value pf_33
  -- pg_33: divstep!(), invocation 29 output
  extract_lets -merge +onlyGivenNames pg_33 at hr
  have e_pg_33 : pg_33 = step29.g := rfl
  clear_value pg_33
  -- fl_29: divstep!(), invocation 29 output
  extract_lets -merge +onlyGivenNames fl_29 at hr
  have e_fl_29 : fl_29 = step29.fl := rfl
  clear_value fl_29
  -- step30: divstep!(), invocation 30
  extract_lets -merge +onlyGivenNames step30 at hr
  have e_step30 : step30 = divstepRound ⟨d_29, pf_33, pg_33, fl_29⟩ := rfl
  clear_value step30
  -- d_30: divstep!(), invocation 30 output
  extract_lets -merge +onlyGivenNames d_30 at hr
  have e_d_30 : d_30 = step30.d := rfl
  clear_value d_30
  -- pf_34: divstep!(), invocation 30 output
  extract_lets -merge +onlyGivenNames pf_34 at hr
  have e_pf_34 : pf_34 = step30.f := rfl
  clear_value pf_34
  -- pg_34: divstep!(), invocation 30 output
  extract_lets -merge +onlyGivenNames pg_34 at hr
  have e_pg_34 : pg_34 = step30.g := rfl
  clear_value pg_34
  -- fl_30: divstep!(), invocation 30 output
  extract_lets -merge +onlyGivenNames fl_30 at hr
  have e_fl_30 : fl_30 = step30.fl := rfl
  clear_value fl_30
  -- step31: divstep!(), invocation 31
  extract_lets -merge +onlyGivenNames step31 at hr
  have e_step31 : step31 = divstepRound ⟨d_30, pf_34, pg_34, fl_30⟩ := rfl
  clear_value step31
  -- d_31: divstep!(), invocation 31 output
  extract_lets -merge +onlyGivenNames d_31 at hr
  have e_d_31 : d_31 = step31.d := rfl
  clear_value d_31
  -- pf_35: divstep!(), invocation 31 output
  extract_lets -merge +onlyGivenNames pf_35 at hr
  have e_pf_35 : pf_35 = step31.f := rfl
  clear_value pf_35
  -- pg_35: divstep!(), invocation 31 output
  extract_lets -merge +onlyGivenNames pg_35 at hr
  have e_pg_35 : pg_35 = step31.g := rfl
  clear_value pg_35
  -- fl_31: divstep!(), invocation 31 output
  extract_lets -merge +onlyGivenNames fl_31 at hr
  have e_fl_31 : fl_31 = step31.fl := rfl
  clear_value fl_31
  -- step32: divstep!(), invocation 32
  extract_lets -merge +onlyGivenNames step32 at hr
  have e_step32 : step32 = divstepRound ⟨d_31, pf_35, pg_35, fl_31⟩ := rfl
  clear_value step32
  -- d_32: divstep!(), invocation 32 output
  extract_lets -merge +onlyGivenNames d_32 at hr
  have e_d_32 : d_32 = step32.d := rfl
  clear_value d_32
  -- pf_36: divstep!(), invocation 32 output
  extract_lets -merge +onlyGivenNames pf_36 at hr
  have e_pf_36 : pf_36 = step32.f := rfl
  clear_value pf_36
  -- pg_36: divstep!(), invocation 32 output
  extract_lets -merge +onlyGivenNames pg_36 at hr
  have e_pg_36 : pg_36 = step32.g := rfl
  clear_value pg_36
  -- fl_32: divstep!(), invocation 32 output
  extract_lets -merge +onlyGivenNames fl_32 at hr
  have e_fl_32 : fl_32 = step32.fl := rfl
  clear_value fl_32
  -- step33: divstep!(), invocation 33
  extract_lets -merge +onlyGivenNames step33 at hr
  have e_step33 : step33 = divstepRound ⟨d_32, pf_36, pg_36, fl_32⟩ := rfl
  clear_value step33
  -- d_33: divstep!(), invocation 33 output
  extract_lets -merge +onlyGivenNames d_33 at hr
  have e_d_33 : d_33 = step33.d := rfl
  clear_value d_33
  -- pf_37: divstep!(), invocation 33 output
  extract_lets -merge +onlyGivenNames pf_37 at hr
  have e_pf_37 : pf_37 = step33.f := rfl
  clear_value pf_37
  -- pg_37: divstep!(), invocation 33 output
  extract_lets -merge +onlyGivenNames pg_37 at hr
  have e_pg_37 : pg_37 = step33.g := rfl
  clear_value pg_37
  -- fl_33: divstep!(), invocation 33 output
  extract_lets -merge +onlyGivenNames fl_33 at hr
  have e_fl_33 : fl_33 = step33.fl := rfl
  clear_value fl_33
  -- step34: divstep!(), invocation 34
  extract_lets -merge +onlyGivenNames step34 at hr
  have e_step34 : step34 = divstepRound ⟨d_33, pf_37, pg_37, fl_33⟩ := rfl
  clear_value step34
  -- d_34: divstep!(), invocation 34 output
  extract_lets -merge +onlyGivenNames d_34 at hr
  have e_d_34 : d_34 = step34.d := rfl
  clear_value d_34
  -- pf_38: divstep!(), invocation 34 output
  extract_lets -merge +onlyGivenNames pf_38 at hr
  have e_pf_38 : pf_38 = step34.f := rfl
  clear_value pf_38
  -- pg_38: divstep!(), invocation 34 output
  extract_lets -merge +onlyGivenNames pg_38 at hr
  have e_pg_38 : pg_38 = step34.g := rfl
  clear_value pg_38
  -- fl_34: divstep!(), invocation 34 output
  extract_lets -merge +onlyGivenNames fl_34 at hr
  have e_fl_34 : fl_34 = step34.fl := rfl
  clear_value fl_34
  -- step35: divstep!(), invocation 35
  extract_lets -merge +onlyGivenNames step35 at hr
  have e_step35 : step35 = divstepRound ⟨d_34, pf_38, pg_38, fl_34⟩ := rfl
  clear_value step35
  -- d_35: divstep!(), invocation 35 output
  extract_lets -merge +onlyGivenNames d_35 at hr
  have e_d_35 : d_35 = step35.d := rfl
  clear_value d_35
  -- pf_39: divstep!(), invocation 35 output
  extract_lets -merge +onlyGivenNames pf_39 at hr
  have e_pf_39 : pf_39 = step35.f := rfl
  clear_value pf_39
  -- pg_39: divstep!(), invocation 35 output
  extract_lets -merge +onlyGivenNames pg_39 at hr
  have e_pg_39 : pg_39 = step35.g := rfl
  clear_value pg_39
  -- fl_35: divstep!(), invocation 35 output
  extract_lets -merge +onlyGivenNames fl_35 at hr
  have e_fl_35 : fl_35 = step35.fl := rfl
  clear_value fl_35
  -- step36: divstep!(), invocation 36
  extract_lets -merge +onlyGivenNames step36 at hr
  have e_step36 : step36 = divstepRound ⟨d_35, pf_39, pg_39, fl_35⟩ := rfl
  clear_value step36
  -- d_36: divstep!(), invocation 36 output
  extract_lets -merge +onlyGivenNames d_36 at hr
  have e_d_36 : d_36 = step36.d := rfl
  clear_value d_36
  -- pf_40: divstep!(), invocation 36 output
  extract_lets -merge +onlyGivenNames pf_40 at hr
  have e_pf_40 : pf_40 = step36.f := rfl
  clear_value pf_40
  -- pg_40: divstep!(), invocation 36 output
  extract_lets -merge +onlyGivenNames pg_40 at hr
  have e_pg_40 : pg_40 = step36.g := rfl
  clear_value pg_40
  -- fl_36: divstep!(), invocation 36 output
  extract_lets -merge +onlyGivenNames fl_36 at hr
  have e_fl_36 : fl_36 = step36.fl := rfl
  clear_value fl_36
  -- step37: divstep!(), invocation 37
  extract_lets -merge +onlyGivenNames step37 at hr
  have e_step37 : step37 = divstepRound ⟨d_36, pf_40, pg_40, fl_36⟩ := rfl
  clear_value step37
  -- d_37: divstep!(), invocation 37 output
  extract_lets -merge +onlyGivenNames d_37 at hr
  have e_d_37 : d_37 = step37.d := rfl
  clear_value d_37
  -- pf_41: divstep!(), invocation 37 output
  extract_lets -merge +onlyGivenNames pf_41 at hr
  have e_pf_41 : pf_41 = step37.f := rfl
  clear_value pf_41
  -- pg_41: divstep!(), invocation 37 output
  extract_lets -merge +onlyGivenNames pg_41 at hr
  have e_pg_41 : pg_41 = step37.g := rfl
  clear_value pg_41
  -- fl_37: divstep!(), invocation 37 output
  extract_lets -merge +onlyGivenNames fl_37 at hr
  have e_fl_37 : fl_37 = step37.fl := rfl
  clear_value fl_37
  -- step38: divstep!(), invocation 38
  extract_lets -merge +onlyGivenNames step38 at hr
  have e_step38 : step38 = divstepRound ⟨d_37, pf_41, pg_41, fl_37⟩ := rfl
  clear_value step38
  -- d_38: divstep!(), invocation 38 output
  extract_lets -merge +onlyGivenNames d_38 at hr
  have e_d_38 : d_38 = step38.d := rfl
  clear_value d_38
  -- pf_42: divstep!(), invocation 38 output
  extract_lets -merge +onlyGivenNames pf_42 at hr
  have e_pf_42 : pf_42 = step38.f := rfl
  clear_value pf_42
  -- pg_42: divstep!(), invocation 38 output
  extract_lets -merge +onlyGivenNames pg_42 at hr
  have e_pg_42 : pg_42 = step38.g := rfl
  clear_value pg_42
  -- fl_38: divstep!(), invocation 38 output
  extract_lets -merge +onlyGivenNames fl_38 at hr
  have e_fl_38 : fl_38 = step38.fl := rfl
  clear_value fl_38
  -- step39: divstep!(), invocation 39
  extract_lets -merge +onlyGivenNames step39 at hr
  have e_step39 : step39 = divstepRound ⟨d_38, pf_42, pg_42, fl_38⟩ := rfl
  clear_value step39
  -- d_39: divstep!(), invocation 39 output
  extract_lets -merge +onlyGivenNames d_39 at hr
  have e_d_39 : d_39 = step39.d := rfl
  clear_value d_39
  -- pf_43: divstep!(), invocation 39 output
  extract_lets -merge +onlyGivenNames pf_43 at hr
  have e_pf_43 : pf_43 = step39.f := rfl
  clear_value pf_43
  -- pg_43: divstep!(), invocation 39 output
  extract_lets -merge +onlyGivenNames pg_43 at hr
  have e_pg_43 : pg_43 = step39.g := rfl
  clear_value pg_43
  -- fl_39: divstep!(), invocation 39 output
  extract_lets -merge +onlyGivenNames fl_39 at hr
  have e_fl_39 : fl_39 = step39.fl := rfl
  clear_value fl_39
  -- step40: divstep!(last), invocation 40
  extract_lets -merge +onlyGivenNames step40 at hr
  have e_step40 : step40 = divstepLast ⟨d_39, pf_43, pg_43, fl_39⟩ := rfl
  clear_value step40
  -- d_40: divstep!(last), invocation 40 output
  extract_lets -merge +onlyGivenNames d_40 at hr
  have e_d_40 : d_40 = step40.d := rfl
  clear_value d_40
  -- pf_44: divstep!(last), invocation 40 output
  extract_lets -merge +onlyGivenNames pf_44 at hr
  have e_pf_44 : pf_44 = step40.f := rfl
  clear_value pf_44
  -- pg_44: divstep!(last), invocation 40 output
  extract_lets -merge +onlyGivenNames pg_44 at hr
  have e_pg_44 : pg_44 = step40.g := rfl
  clear_value pg_44
  -- b00: add b00,pf,#0x100,lsl #12
  extract_lets -merge +onlyGivenNames b00 at hr
  have e_b00 : b00 = addw pf_44 0x100000 := rfl
  clear_value b00
  have b_b00 : b00 < 2^64 := by rw [e_b00]; exact addw_lt pf_44 0x100000
  -- b00_1: sbfx b00,b00,#21,#21
  extract_lets -merge +onlyGivenNames b00_1 at hr
  have e_b00_1 : b00_1 = sbfx b00 21 21 := rfl
  clear_value b00_1
  have b_b00_1 : b00_1 < 2^64 := by rw [e_b00_1]; exact sbfx_lt b00 21 21 b_b00
  -- b11: mov b11,#0x100000
  extract_lets -merge +onlyGivenNames b11 at hr
  have e_b11 : b11 = 0x100000 := rfl
  clear_value b11
  have b_b11 : b11 < 2^64 := by rw [e_b11]; decide
  -- b11_1: add b11,b11,b11,lsl #21
  extract_lets -merge +onlyGivenNames b11_1 at hr
  have e_b11_1 : b11_1 = addw b11 (lsl b11 21) := rfl
  clear_value b11_1
  have b_b11_1 : b11_1 < 2^64 := by rw [e_b11_1]; exact addw_lt b11 (lsl b11 21)
  -- b01: add b01,pf,b11
  extract_lets -merge +onlyGivenNames b01 at hr
  have e_b01 : b01 = addw pf_44 b11_1 := rfl
  clear_value b01
  have b_b01 : b01 < 2^64 := by rw [e_b01]; exact addw_lt pf_44 b11_1
  -- b01_1: asr b01,b01,#42
  extract_lets -merge +onlyGivenNames b01_1 at hr
  have e_b01_1 : b01_1 = asr b01 42 := rfl
  clear_value b01_1
  have b_b01_1 : b01_1 < 2^64 := by rw [e_b01_1]; exact asr_lt b01 42 b_b01
  -- b10: add b10,pg,#0x100,lsl #12
  extract_lets -merge +onlyGivenNames b10 at hr
  have e_b10 : b10 = addw pg_44 0x100000 := rfl
  clear_value b10
  have b_b10 : b10 < 2^64 := by rw [e_b10]; exact addw_lt pg_44 0x100000
  -- b10_1: sbfx b10,b10,#21,#21
  extract_lets -merge +onlyGivenNames b10_1 at hr
  have e_b10_1 : b10_1 = sbfx b10 21 21 := rfl
  clear_value b10_1
  have b_b10_1 : b10_1 < 2^64 := by rw [e_b10_1]; exact sbfx_lt b10 21 21 b_b10
  -- b11_2: add b11,pg,b11
  extract_lets -merge +onlyGivenNames b11_2 at hr
  have e_b11_2 : b11_2 = addw pg_44 b11_1 := rfl
  clear_value b11_2
  have b_b11_2 : b11_2 < 2^64 := by rw [e_b11_2]; exact addw_lt pg_44 b11_1
  -- b11_3: asr b11,b11,#42
  extract_lets -merge +onlyGivenNames b11_3 at hr
  have e_b11_3 : b11_3 = asr b11_2 42 := rfl
  clear_value b11_3
  have b_b11_3 : b11_3 < 2^64 := by rw [e_b11_3]; exact asr_lt b11_2 42 b_b11_2
  -- t_1: mul t,b00,f
  extract_lets -merge +onlyGivenNames t_1 at hr
  have e_t_1 : t_1 = b00_1 * f_2 % 2^64 := rfl
  clear_value t_1
  have b_t_1 : t_1 < 2^64 := by rw [e_t_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t2_1: mul t2,b01,g
  extract_lets -merge +onlyGivenNames t2_1 at hr
  have e_t2_1 : t2_1 = b01_1 * g_2 % 2^64 := rfl
  clear_value t2_1
  have b_t2_1 : t2_1 < 2^64 := by rw [e_t2_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- f_3: mul f,b10,f
  extract_lets -merge +onlyGivenNames f_3 at hr
  have e_f_3 : f_3 = b10_1 * f_2 % 2^64 := rfl
  clear_value f_3
  have b_f_3 : f_3 < 2^64 := by rw [e_f_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- g_3: mul g,b11,g
  extract_lets -merge +onlyGivenNames g_3 at hr
  have e_g_3 : g_3 = b11_3 * g_2 % 2^64 := rfl
  clear_value g_3
  have b_g_3 : g_3 < 2^64 := by rw [e_g_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- pf_45: add pf,t,t2
  extract_lets -merge +onlyGivenNames pf_45 at hr
  have e_pf_45 : pf_45 = addw t_1 t2_1 := rfl
  clear_value pf_45
  have b_pf_45 : pf_45 < 2^64 := by rw [e_pf_45]; exact addw_lt t_1 t2_1
  -- pg_45: add pg,f,g
  extract_lets -merge +onlyGivenNames pg_45 at hr
  have e_pg_45 : pg_45 = addw f_3 g_3 := rfl
  clear_value pg_45
  have b_pg_45 : pg_45 < 2^64 := by rw [e_pg_45]; exact addw_lt f_3 g_3
  -- f_4: asr f,pf,#20
  extract_lets -merge +onlyGivenNames f_4 at hr
  have e_f_4 : f_4 = asr pf_45 20 := rfl
  clear_value f_4
  have b_f_4 : f_4 < 2^64 := by rw [e_f_4]; exact asr_lt pf_45 20 b_pf_45
  -- g_4: asr g,pg,#20
  extract_lets -merge +onlyGivenNames g_4 at hr
  have e_g_4 : g_4 = asr pg_45 20 := rfl
  clear_value g_4
  have b_g_4 : g_4 < 2^64 := by rw [e_g_4]; exact asr_lt pg_45 20 b_pg_45
  -- pf_46: and pf,f,#0xfffff
  extract_lets -merge +onlyGivenNames pf_46 at hr
  have e_pf_46 : pf_46 = andw f_4 0xfffff := rfl
  clear_value pf_46
  have b_pf_46 : pf_46 < 2^64 := by rw [e_pf_46]; exact andw_lt f_4 0xfffff b_f_4 (by decide)
  -- pf_47: orr pf,pf,#0xfffffe0000000000
  extract_lets -merge +onlyGivenNames pf_47 at hr
  have e_pf_47 : pf_47 = orrw pf_46 0xfffffe0000000000 := rfl
  clear_value pf_47
  have b_pf_47 : pf_47 < 2^64 := by rw [e_pf_47]; exact orrw_lt pf_46 0xfffffe0000000000 b_pf_46 (by decide)
  -- pg_46: and pg,g,#0xfffff
  extract_lets -merge +onlyGivenNames pg_46 at hr
  have e_pg_46 : pg_46 = andw g_4 0xfffff := rfl
  clear_value pg_46
  have b_pg_46 : pg_46 < 2^64 := by rw [e_pg_46]; exact andw_lt g_4 0xfffff b_g_4 (by decide)
  -- pg_47: orr pg,pg,#0xc000000000000000
  extract_lets -merge +onlyGivenNames pg_47 at hr
  have e_pg_47 : pg_47 = orrw pg_46 0xc000000000000000 := rfl
  clear_value pg_47
  have b_pg_47 : pg_47 < 2^64 := by rw [e_pg_47]; exact orrw_lt pg_46 0xc000000000000000 b_pg_46 (by decide)
  -- fl_40: tst pg,#1
  extract_lets -merge +onlyGivenNames fl_40 at hr
  have e_fl_40 : fl_40 = tstFlags (andw pg_47 1) := rfl
  clear_value fl_40
  -- step41: divstep!(), invocation 41
  extract_lets -merge +onlyGivenNames step41 at hr
  have e_step41 : step41 = divstepRound ⟨d_40, pf_47, pg_47, fl_40⟩ := rfl
  clear_value step41
  -- d_41: divstep!(), invocation 41 output
  extract_lets -merge +onlyGivenNames d_41 at hr
  have e_d_41 : d_41 = step41.d := rfl
  clear_value d_41
  -- pf_48: divstep!(), invocation 41 output
  extract_lets -merge +onlyGivenNames pf_48 at hr
  have e_pf_48 : pf_48 = step41.f := rfl
  clear_value pf_48
  -- pg_48: divstep!(), invocation 41 output
  extract_lets -merge +onlyGivenNames pg_48 at hr
  have e_pg_48 : pg_48 = step41.g := rfl
  clear_value pg_48
  -- fl_41: divstep!(), invocation 41 output
  extract_lets -merge +onlyGivenNames fl_41 at hr
  have e_fl_41 : fl_41 = step41.fl := rfl
  clear_value fl_41
  -- step42: divstep!(), invocation 42
  extract_lets -merge +onlyGivenNames step42 at hr
  have e_step42 : step42 = divstepRound ⟨d_41, pf_48, pg_48, fl_41⟩ := rfl
  clear_value step42
  -- d_42: divstep!(), invocation 42 output
  extract_lets -merge +onlyGivenNames d_42 at hr
  have e_d_42 : d_42 = step42.d := rfl
  clear_value d_42
  -- pf_49: divstep!(), invocation 42 output
  extract_lets -merge +onlyGivenNames pf_49 at hr
  have e_pf_49 : pf_49 = step42.f := rfl
  clear_value pf_49
  -- pg_49: divstep!(), invocation 42 output
  extract_lets -merge +onlyGivenNames pg_49 at hr
  have e_pg_49 : pg_49 = step42.g := rfl
  clear_value pg_49
  -- fl_42: divstep!(), invocation 42 output
  extract_lets -merge +onlyGivenNames fl_42 at hr
  have e_fl_42 : fl_42 = step42.fl := rfl
  clear_value fl_42
  -- step43: divstep!(), invocation 43
  extract_lets -merge +onlyGivenNames step43 at hr
  have e_step43 : step43 = divstepRound ⟨d_42, pf_49, pg_49, fl_42⟩ := rfl
  clear_value step43
  -- d_43: divstep!(), invocation 43 output
  extract_lets -merge +onlyGivenNames d_43 at hr
  have e_d_43 : d_43 = step43.d := rfl
  clear_value d_43
  -- pf_50: divstep!(), invocation 43 output
  extract_lets -merge +onlyGivenNames pf_50 at hr
  have e_pf_50 : pf_50 = step43.f := rfl
  clear_value pf_50
  -- pg_50: divstep!(), invocation 43 output
  extract_lets -merge +onlyGivenNames pg_50 at hr
  have e_pg_50 : pg_50 = step43.g := rfl
  clear_value pg_50
  -- fl_43: divstep!(), invocation 43 output
  extract_lets -merge +onlyGivenNames fl_43 at hr
  have e_fl_43 : fl_43 = step43.fl := rfl
  clear_value fl_43
  -- step44: divstep!(), invocation 44
  extract_lets -merge +onlyGivenNames step44 at hr
  have e_step44 : step44 = divstepRound ⟨d_43, pf_50, pg_50, fl_43⟩ := rfl
  clear_value step44
  -- d_44: divstep!(), invocation 44 output
  extract_lets -merge +onlyGivenNames d_44 at hr
  have e_d_44 : d_44 = step44.d := rfl
  clear_value d_44
  -- pf_51: divstep!(), invocation 44 output
  extract_lets -merge +onlyGivenNames pf_51 at hr
  have e_pf_51 : pf_51 = step44.f := rfl
  clear_value pf_51
  -- pg_51: divstep!(), invocation 44 output
  extract_lets -merge +onlyGivenNames pg_51 at hr
  have e_pg_51 : pg_51 = step44.g := rfl
  clear_value pg_51
  -- fl_44: divstep!(), invocation 44 output
  extract_lets -merge +onlyGivenNames fl_44 at hr
  have e_fl_44 : fl_44 = step44.fl := rfl
  clear_value fl_44
  -- step45: divstep!(), invocation 45
  extract_lets -merge +onlyGivenNames step45 at hr
  have e_step45 : step45 = divstepRound ⟨d_44, pf_51, pg_51, fl_44⟩ := rfl
  clear_value step45
  -- d_45: divstep!(), invocation 45 output
  extract_lets -merge +onlyGivenNames d_45 at hr
  have e_d_45 : d_45 = step45.d := rfl
  clear_value d_45
  -- pf_52: divstep!(), invocation 45 output
  extract_lets -merge +onlyGivenNames pf_52 at hr
  have e_pf_52 : pf_52 = step45.f := rfl
  clear_value pf_52
  -- pg_52: divstep!(), invocation 45 output
  extract_lets -merge +onlyGivenNames pg_52 at hr
  have e_pg_52 : pg_52 = step45.g := rfl
  clear_value pg_52
  -- fl_45: divstep!(), invocation 45 output
  extract_lets -merge +onlyGivenNames fl_45 at hr
  have e_fl_45 : fl_45 = step45.fl := rfl
  clear_value fl_45
  -- step46: divstep!(), invocation 46
  extract_lets -merge +onlyGivenNames step46 at hr
  have e_step46 : step46 = divstepRound ⟨d_45, pf_52, pg_52, fl_45⟩ := rfl
  clear_value step46
  -- d_46: divstep!(), invocation 46 output
  extract_lets -merge +onlyGivenNames d_46 at hr
  have e_d_46 : d_46 = step46.d := rfl
  clear_value d_46
  -- pf_53: divstep!(), invocation 46 output
  extract_lets -merge +onlyGivenNames pf_53 at hr
  have e_pf_53 : pf_53 = step46.f := rfl
  clear_value pf_53
  -- pg_53: divstep!(), invocation 46 output
  extract_lets -merge +onlyGivenNames pg_53 at hr
  have e_pg_53 : pg_53 = step46.g := rfl
  clear_value pg_53
  -- fl_46: divstep!(), invocation 46 output
  extract_lets -merge +onlyGivenNames fl_46 at hr
  have e_fl_46 : fl_46 = step46.fl := rfl
  clear_value fl_46
  -- step47: divstep!(), invocation 47
  extract_lets -merge +onlyGivenNames step47 at hr
  have e_step47 : step47 = divstepRound ⟨d_46, pf_53, pg_53, fl_46⟩ := rfl
  clear_value step47
  -- d_47: divstep!(), invocation 47 output
  extract_lets -merge +onlyGivenNames d_47 at hr
  have e_d_47 : d_47 = step47.d := rfl
  clear_value d_47
  -- pf_54: divstep!(), invocation 47 output
  extract_lets -merge +onlyGivenNames pf_54 at hr
  have e_pf_54 : pf_54 = step47.f := rfl
  clear_value pf_54
  -- pg_54: divstep!(), invocation 47 output
  extract_lets -merge +onlyGivenNames pg_54 at hr
  have e_pg_54 : pg_54 = step47.g := rfl
  clear_value pg_54
  -- fl_47: divstep!(), invocation 47 output
  extract_lets -merge +onlyGivenNames fl_47 at hr
  have e_fl_47 : fl_47 = step47.fl := rfl
  clear_value fl_47
  -- step48: divstep!(), invocation 48
  extract_lets -merge +onlyGivenNames step48 at hr
  have e_step48 : step48 = divstepRound ⟨d_47, pf_54, pg_54, fl_47⟩ := rfl
  clear_value step48
  -- d_48: divstep!(), invocation 48 output
  extract_lets -merge +onlyGivenNames d_48 at hr
  have e_d_48 : d_48 = step48.d := rfl
  clear_value d_48
  -- pf_55: divstep!(), invocation 48 output
  extract_lets -merge +onlyGivenNames pf_55 at hr
  have e_pf_55 : pf_55 = step48.f := rfl
  clear_value pf_55
  -- pg_55: divstep!(), invocation 48 output
  extract_lets -merge +onlyGivenNames pg_55 at hr
  have e_pg_55 : pg_55 = step48.g := rfl
  clear_value pg_55
  -- fl_48: divstep!(), invocation 48 output
  extract_lets -merge +onlyGivenNames fl_48 at hr
  have e_fl_48 : fl_48 = step48.fl := rfl
  clear_value fl_48
  -- step49: divstep!(), invocation 49
  extract_lets -merge +onlyGivenNames step49 at hr
  have e_step49 : step49 = divstepRound ⟨d_48, pf_55, pg_55, fl_48⟩ := rfl
  clear_value step49
  -- d_49: divstep!(), invocation 49 output
  extract_lets -merge +onlyGivenNames d_49 at hr
  have e_d_49 : d_49 = step49.d := rfl
  clear_value d_49
  -- pf_56: divstep!(), invocation 49 output
  extract_lets -merge +onlyGivenNames pf_56 at hr
  have e_pf_56 : pf_56 = step49.f := rfl
  clear_value pf_56
  -- pg_56: divstep!(), invocation 49 output
  extract_lets -merge +onlyGivenNames pg_56 at hr
  have e_pg_56 : pg_56 = step49.g := rfl
  clear_value pg_56
  -- fl_49: divstep!(), invocation 49 output
  extract_lets -merge +onlyGivenNames fl_49 at hr
  have e_fl_49 : fl_49 = step49.fl := rfl
  clear_value fl_49
  -- step50: divstep!(), invocation 50
  extract_lets -merge +onlyGivenNames step50 at hr
  have e_step50 : step50 = divstepRound ⟨d_49, pf_56, pg_56, fl_49⟩ := rfl
  clear_value step50
  -- d_50: divstep!(), invocation 50 output
  extract_lets -merge +onlyGivenNames d_50 at hr
  have e_d_50 : d_50 = step50.d := rfl
  clear_value d_50
  -- pf_57: divstep!(), invocation 50 output
  extract_lets -merge +onlyGivenNames pf_57 at hr
  have e_pf_57 : pf_57 = step50.f := rfl
  clear_value pf_57
  -- pg_57: divstep!(), invocation 50 output
  extract_lets -merge +onlyGivenNames pg_57 at hr
  have e_pg_57 : pg_57 = step50.g := rfl
  clear_value pg_57
  -- fl_50: divstep!(), invocation 50 output
  extract_lets -merge +onlyGivenNames fl_50 at hr
  have e_fl_50 : fl_50 = step50.fl := rfl
  clear_value fl_50
  -- f_5: mul f,b00,a00
  extract_lets -merge +onlyGivenNames f_5 at hr
  have e_f_5 : f_5 = b00_1 * a00_1 % 2^64 := rfl
  clear_value f_5
  have b_f_5 : f_5 < 2^64 := by rw [e_f_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- g_5: mul g,b00,a01
  extract_lets -merge +onlyGivenNames g_5 at hr
  have e_g_5 : g_5 = b00_1 * a01_1 % 2^64 := rfl
  clear_value g_5
  have b_g_5 : g_5 < 2^64 := by rw [e_g_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t_2: mul t,b10,a00
  extract_lets -merge +onlyGivenNames t_2 at hr
  have e_t_2 : t_2 = b10_1 * a00_1 % 2^64 := rfl
  clear_value t_2
  have b_t_2 : t_2 < 2^64 := by rw [e_t_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t2_2: mul t2,b10,a01
  extract_lets -merge +onlyGivenNames t2_2 at hr
  have e_t2_2 : t2_2 = b10_1 * a01_1 % 2^64 := rfl
  clear_value t2_2
  have b_t2_2 : t2_2 < 2^64 := by rw [e_t2_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- a00_2: madd a00,b01,a10,f
  extract_lets -merge +onlyGivenNames a00_2 at hr
  have e_a00_2 : a00_2 = madd b01_1 a10_1 f_5 := rfl
  clear_value a00_2
  have b_a00_2 : a00_2 < 2^64 := by rw [e_a00_2]; exact madd_lt b01_1 a10_1 f_5
  -- a01_2: madd a01,b01,a11,g
  extract_lets -merge +onlyGivenNames a01_2 at hr
  have e_a01_2 : a01_2 = madd b01_1 a11_3 g_5 := rfl
  clear_value a01_2
  have b_a01_2 : a01_2 < 2^64 := by rw [e_a01_2]; exact madd_lt b01_1 a11_3 g_5
  -- c10: madd c10,b11,a10,t
  extract_lets -merge +onlyGivenNames c10 at hr
  have e_c10 : c10 = madd b11_3 a10_1 t_2 := rfl
  clear_value c10
  have b_c10 : c10 < 2^64 := by rw [e_c10]; exact madd_lt b11_3 a10_1 t_2
  -- c11: madd c11,b11,a11,t2
  extract_lets -merge +onlyGivenNames c11 at hr
  have e_c11 : c11 = madd b11_3 a11_3 t2_2 := rfl
  clear_value c11
  have b_c11 : c11 < 2^64 := by rw [e_c11]; exact madd_lt b11_3 a11_3 t2_2
  -- step51: divstep!(), invocation 51
  extract_lets -merge +onlyGivenNames step51 at hr
  have e_step51 : step51 = divstepRound ⟨d_50, pf_57, pg_57, fl_50⟩ := rfl
  clear_value step51
  -- d_51: divstep!(), invocation 51 output
  extract_lets -merge +onlyGivenNames d_51 at hr
  have e_d_51 : d_51 = step51.d := rfl
  clear_value d_51
  -- pf_58: divstep!(), invocation 51 output
  extract_lets -merge +onlyGivenNames pf_58 at hr
  have e_pf_58 : pf_58 = step51.f := rfl
  clear_value pf_58
  -- pg_58: divstep!(), invocation 51 output
  extract_lets -merge +onlyGivenNames pg_58 at hr
  have e_pg_58 : pg_58 = step51.g := rfl
  clear_value pg_58
  -- fl_51: divstep!(), invocation 51 output
  extract_lets -merge +onlyGivenNames fl_51 at hr
  have e_fl_51 : fl_51 = step51.fl := rfl
  clear_value fl_51
  -- step52: divstep!(), invocation 52
  extract_lets -merge +onlyGivenNames step52 at hr
  have e_step52 : step52 = divstepRound ⟨d_51, pf_58, pg_58, fl_51⟩ := rfl
  clear_value step52
  -- d_52: divstep!(), invocation 52 output
  extract_lets -merge +onlyGivenNames d_52 at hr
  have e_d_52 : d_52 = step52.d := rfl
  clear_value d_52
  -- pf_59: divstep!(), invocation 52 output
  extract_lets -merge +onlyGivenNames pf_59 at hr
  have e_pf_59 : pf_59 = step52.f := rfl
  clear_value pf_59
  -- pg_59: divstep!(), invocation 52 output
  extract_lets -merge +onlyGivenNames pg_59 at hr
  have e_pg_59 : pg_59 = step52.g := rfl
  clear_value pg_59
  -- fl_52: divstep!(), invocation 52 output
  extract_lets -merge +onlyGivenNames fl_52 at hr
  have e_fl_52 : fl_52 = step52.fl := rfl
  clear_value fl_52
  -- step53: divstep!(), invocation 53
  extract_lets -merge +onlyGivenNames step53 at hr
  have e_step53 : step53 = divstepRound ⟨d_52, pf_59, pg_59, fl_52⟩ := rfl
  clear_value step53
  -- d_53: divstep!(), invocation 53 output
  extract_lets -merge +onlyGivenNames d_53 at hr
  have e_d_53 : d_53 = step53.d := rfl
  clear_value d_53
  -- pf_60: divstep!(), invocation 53 output
  extract_lets -merge +onlyGivenNames pf_60 at hr
  have e_pf_60 : pf_60 = step53.f := rfl
  clear_value pf_60
  -- pg_60: divstep!(), invocation 53 output
  extract_lets -merge +onlyGivenNames pg_60 at hr
  have e_pg_60 : pg_60 = step53.g := rfl
  clear_value pg_60
  -- fl_53: divstep!(), invocation 53 output
  extract_lets -merge +onlyGivenNames fl_53 at hr
  have e_fl_53 : fl_53 = step53.fl := rfl
  clear_value fl_53
  -- step54: divstep!(), invocation 54
  extract_lets -merge +onlyGivenNames step54 at hr
  have e_step54 : step54 = divstepRound ⟨d_53, pf_60, pg_60, fl_53⟩ := rfl
  clear_value step54
  -- d_54: divstep!(), invocation 54 output
  extract_lets -merge +onlyGivenNames d_54 at hr
  have e_d_54 : d_54 = step54.d := rfl
  clear_value d_54
  -- pf_61: divstep!(), invocation 54 output
  extract_lets -merge +onlyGivenNames pf_61 at hr
  have e_pf_61 : pf_61 = step54.f := rfl
  clear_value pf_61
  -- pg_61: divstep!(), invocation 54 output
  extract_lets -merge +onlyGivenNames pg_61 at hr
  have e_pg_61 : pg_61 = step54.g := rfl
  clear_value pg_61
  -- fl_54: divstep!(), invocation 54 output
  extract_lets -merge +onlyGivenNames fl_54 at hr
  have e_fl_54 : fl_54 = step54.fl := rfl
  clear_value fl_54
  -- step55: divstep!(), invocation 55
  extract_lets -merge +onlyGivenNames step55 at hr
  have e_step55 : step55 = divstepRound ⟨d_54, pf_61, pg_61, fl_54⟩ := rfl
  clear_value step55
  -- d_55: divstep!(), invocation 55 output
  extract_lets -merge +onlyGivenNames d_55 at hr
  have e_d_55 : d_55 = step55.d := rfl
  clear_value d_55
  -- pf_62: divstep!(), invocation 55 output
  extract_lets -merge +onlyGivenNames pf_62 at hr
  have e_pf_62 : pf_62 = step55.f := rfl
  clear_value pf_62
  -- pg_62: divstep!(), invocation 55 output
  extract_lets -merge +onlyGivenNames pg_62 at hr
  have e_pg_62 : pg_62 = step55.g := rfl
  clear_value pg_62
  -- fl_55: divstep!(), invocation 55 output
  extract_lets -merge +onlyGivenNames fl_55 at hr
  have e_fl_55 : fl_55 = step55.fl := rfl
  clear_value fl_55
  -- step56: divstep!(), invocation 56
  extract_lets -merge +onlyGivenNames step56 at hr
  have e_step56 : step56 = divstepRound ⟨d_55, pf_62, pg_62, fl_55⟩ := rfl
  clear_value step56
  -- d_56: divstep!(), invocation 56 output
  extract_lets -merge +onlyGivenNames d_56 at hr
  have e_d_56 : d_56 = step56.d := rfl
  clear_value d_56
  -- pf_63: divstep!(), invocation 56 output
  extract_lets -merge +onlyGivenNames pf_63 at hr
  have e_pf_63 : pf_63 = step56.f := rfl
  clear_value pf_63
  -- pg_63: divstep!(), invocation 56 output
  extract_lets -merge +onlyGivenNames pg_63 at hr
  have e_pg_63 : pg_63 = step56.g := rfl
  clear_value pg_63
  -- fl_56: divstep!(), invocation 56 output
  extract_lets -merge +onlyGivenNames fl_56 at hr
  have e_fl_56 : fl_56 = step56.fl := rfl
  clear_value fl_56
  -- step57: divstep!(), invocation 57
  extract_lets -merge +onlyGivenNames step57 at hr
  have e_step57 : step57 = divstepRound ⟨d_56, pf_63, pg_63, fl_56⟩ := rfl
  clear_value step57
  -- d_57: divstep!(), invocation 57 output
  extract_lets -merge +onlyGivenNames d_57 at hr
  have e_d_57 : d_57 = step57.d := rfl
  clear_value d_57
  -- pf_64: divstep!(), invocation 57 output
  extract_lets -merge +onlyGivenNames pf_64 at hr
  have e_pf_64 : pf_64 = step57.f := rfl
  clear_value pf_64
  -- pg_64: divstep!(), invocation 57 output
  extract_lets -merge +onlyGivenNames pg_64 at hr
  have e_pg_64 : pg_64 = step57.g := rfl
  clear_value pg_64
  -- fl_57: divstep!(), invocation 57 output
  extract_lets -merge +onlyGivenNames fl_57 at hr
  have e_fl_57 : fl_57 = step57.fl := rfl
  clear_value fl_57
  -- step58: divstep!(), invocation 58
  extract_lets -merge +onlyGivenNames step58 at hr
  have e_step58 : step58 = divstepRound ⟨d_57, pf_64, pg_64, fl_57⟩ := rfl
  clear_value step58
  -- d_58: divstep!(), invocation 58 output
  extract_lets -merge +onlyGivenNames d_58 at hr
  have e_d_58 : d_58 = step58.d := rfl
  clear_value d_58
  -- pf_65: divstep!(), invocation 58 output
  extract_lets -merge +onlyGivenNames pf_65 at hr
  have e_pf_65 : pf_65 = step58.f := rfl
  clear_value pf_65
  -- pg_65: divstep!(), invocation 58 output
  extract_lets -merge +onlyGivenNames pg_65 at hr
  have e_pg_65 : pg_65 = step58.g := rfl
  clear_value pg_65
  -- fl_58: divstep!(), invocation 58 output
  extract_lets -merge +onlyGivenNames fl_58 at hr
  have e_fl_58 : fl_58 = step58.fl := rfl
  clear_value fl_58
  -- step59: divstep!(last), invocation 59
  extract_lets -merge +onlyGivenNames step59 at hr
  have e_step59 : step59 = divstepLast ⟨d_58, pf_65, pg_65, fl_58⟩ := rfl
  clear_value step59
  -- d_59: divstep!(last), invocation 59 output
  extract_lets -merge +onlyGivenNames d_59 at hr
  have e_d_59 : d_59 = step59.d := rfl
  clear_value d_59
  -- pf_66: divstep!(last), invocation 59 output
  extract_lets -merge +onlyGivenNames pf_66 at hr
  have e_pf_66 : pf_66 = step59.f := rfl
  clear_value pf_66
  -- pg_66: divstep!(last), invocation 59 output
  extract_lets -merge +onlyGivenNames pg_66 at hr
  have e_pg_66 : pg_66 = step59.g := rfl
  clear_value pg_66
  -- b00_2: add b00,pf,#0x100,lsl #12
  extract_lets -merge +onlyGivenNames b00_2 at hr
  have e_b00_2 : b00_2 = addw pf_66 0x100000 := rfl
  clear_value b00_2
  have b_b00_2 : b00_2 < 2^64 := by rw [e_b00_2]; exact addw_lt pf_66 0x100000
  -- b00_3: sbfx b00,b00,#22,#21
  extract_lets -merge +onlyGivenNames b00_3 at hr
  have e_b00_3 : b00_3 = sbfx b00_2 22 21 := rfl
  clear_value b00_3
  have b_b00_3 : b00_3 < 2^64 := by rw [e_b00_3]; exact sbfx_lt b00_2 22 21 b_b00_2
  -- b11_4: mov b11,#0x100000
  extract_lets -merge +onlyGivenNames b11_4 at hr
  have e_b11_4 : b11_4 = 0x100000 := rfl
  clear_value b11_4
  have b_b11_4 : b11_4 < 2^64 := by rw [e_b11_4]; decide
  -- b11_5: add b11,b11,b11,lsl #21
  extract_lets -merge +onlyGivenNames b11_5 at hr
  have e_b11_5 : b11_5 = addw b11_4 (lsl b11_4 21) := rfl
  clear_value b11_5
  have b_b11_5 : b11_5 < 2^64 := by rw [e_b11_5]; exact addw_lt b11_4 (lsl b11_4 21)
  -- b01_2: add b01,pf,b11
  extract_lets -merge +onlyGivenNames b01_2 at hr
  have e_b01_2 : b01_2 = addw pf_66 b11_5 := rfl
  clear_value b01_2
  have b_b01_2 : b01_2 < 2^64 := by rw [e_b01_2]; exact addw_lt pf_66 b11_5
  -- b01_3: asr b01,b01,#43
  extract_lets -merge +onlyGivenNames b01_3 at hr
  have e_b01_3 : b01_3 = asr b01_2 43 := rfl
  clear_value b01_3
  have b_b01_3 : b01_3 < 2^64 := by rw [e_b01_3]; exact asr_lt b01_2 43 b_b01_2
  -- b10_2: add b10,pg,#0x100,lsl #12
  extract_lets -merge +onlyGivenNames b10_2 at hr
  have e_b10_2 : b10_2 = addw pg_66 0x100000 := rfl
  clear_value b10_2
  have b_b10_2 : b10_2 < 2^64 := by rw [e_b10_2]; exact addw_lt pg_66 0x100000
  -- b10_3: sbfx b10,b10,#22,#21
  extract_lets -merge +onlyGivenNames b10_3 at hr
  have e_b10_3 : b10_3 = sbfx b10_2 22 21 := rfl
  clear_value b10_3
  have b_b10_3 : b10_3 < 2^64 := by rw [e_b10_3]; exact sbfx_lt b10_2 22 21 b_b10_2
  -- b11_6: add b11,pg,b11
  extract_lets -merge +onlyGivenNames b11_6 at hr
  have e_b11_6 : b11_6 = addw pg_66 b11_5 := rfl
  clear_value b11_6
  have b_b11_6 : b11_6 < 2^64 := by rw [e_b11_6]; exact addw_lt pg_66 b11_5
  -- b11_7: asr b11,b11,#43
  extract_lets -merge +onlyGivenNames b11_7 at hr
  have e_b11_7 : b11_7 = asr b11_6 43 := rfl
  clear_value b11_7
  have b_b11_7 : b11_7 < 2^64 := by rw [e_b11_7]; exact asr_lt b11_6 43 b_b11_6
  -- f_6: mneg f,b00,a00
  extract_lets -merge +onlyGivenNames f_6 at hr
  have e_f_6 : f_6 = mneg b00_3 a00_2 := rfl
  clear_value f_6
  have b_f_6 : f_6 < 2^64 := by rw [e_f_6]; exact mneg_lt b00_3 a00_2
  -- g_6: mneg g,b00,a01
  extract_lets -merge +onlyGivenNames g_6 at hr
  have e_g_6 : g_6 = mneg b00_3 a01_2 := rfl
  clear_value g_6
  have b_g_6 : g_6 < 2^64 := by rw [e_g_6]; exact mneg_lt b00_3 a01_2
  -- pf_67: mneg pf,b10,a00
  extract_lets -merge +onlyGivenNames pf_67 at hr
  have e_pf_67 : pf_67 = mneg b10_3 a00_2 := rfl
  clear_value pf_67
  have b_pf_67 : pf_67 < 2^64 := by rw [e_pf_67]; exact mneg_lt b10_3 a00_2
  -- pg_67: mneg pg,b10,a01
  extract_lets -merge +onlyGivenNames pg_67 at hr
  have e_pg_67 : pg_67 = mneg b10_3 a01_2 := rfl
  clear_value pg_67
  have b_pg_67 : pg_67 < 2^64 := by rw [e_pg_67]; exact mneg_lt b10_3 a01_2
  -- m00: msub m00,b01,c10,f
  extract_lets -merge +onlyGivenNames m00 at hr
  have e_m00 : m00 = msub b01_3 c10 f_6 := rfl
  clear_value m00
  have b_m00 : m00 < 2^64 := by rw [e_m00]; exact msub_lt b01_3 c10 f_6
  -- m01: msub m01,b01,c11,g
  extract_lets -merge +onlyGivenNames m01 at hr
  have e_m01 : m01 = msub b01_3 c11 g_6 := rfl
  clear_value m01
  have b_m01 : m01 < 2^64 := by rw [e_m01]; exact msub_lt b01_3 c11 g_6
  -- m10: msub m10,b11,c10,pf
  extract_lets -merge +onlyGivenNames m10 at hr
  have e_m10 : m10 = msub b11_7 c10 pf_67 := rfl
  clear_value m10
  have b_m10 : m10 < 2^64 := by rw [e_m10]; exact msub_lt b11_7 c10 pf_67
  -- m11: msub m11,b11,c11,pg
  extract_lets -merge +onlyGivenNames m11 at hr
  have e_m11 : m11 = msub b11_7 c11 pg_67 := rfl
  clear_value m11
  have b_m11 : m11 < 2^64 := by rw [e_m11]; exact msub_lt b11_7 c11 pg_67
  subst hr
  -- BEGIN conclusion
  -- The three chains of steps are iterates of the round, ending in the last step; then the
  -- intermediate steps' facts and words are dropped, and the rest works in a small context.
  have i1 : step1 = divstepRound^[1] ⟨d', pf_1, pg_1, fl⟩ := e_step1
  have i2 : step2 = divstepRound^[2] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step2, e_d_1, e_pf_2, e_pg_2, e_fl_1, i1]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i3 : step3 = divstepRound^[3] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step3, e_d_2, e_pf_3, e_pg_3, e_fl_2, i2]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i4 : step4 = divstepRound^[4] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step4, e_d_3, e_pf_4, e_pg_4, e_fl_3, i3]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i5 : step5 = divstepRound^[5] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step5, e_d_4, e_pf_5, e_pg_5, e_fl_4, i4]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i6 : step6 = divstepRound^[6] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step6, e_d_5, e_pf_6, e_pg_6, e_fl_5, i5]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i7 : step7 = divstepRound^[7] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step7, e_d_6, e_pf_7, e_pg_7, e_fl_6, i6]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i8 : step8 = divstepRound^[8] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step8, e_d_7, e_pf_8, e_pg_8, e_fl_7, i7]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i9 : step9 = divstepRound^[9] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step9, e_d_8, e_pf_9, e_pg_9, e_fl_8, i8]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i10 : step10 = divstepRound^[10] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step10, e_d_9, e_pf_10, e_pg_10, e_fl_9, i9]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i11 : step11 = divstepRound^[11] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step11, e_d_10, e_pf_11, e_pg_11, e_fl_10, i10]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i12 : step12 = divstepRound^[12] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step12, e_d_11, e_pf_12, e_pg_12, e_fl_11, i11]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i13 : step13 = divstepRound^[13] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step13, e_d_12, e_pf_13, e_pg_13, e_fl_12, i12]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i14 : step14 = divstepRound^[14] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step14, e_d_13, e_pf_14, e_pg_14, e_fl_13, i13]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i15 : step15 = divstepRound^[15] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step15, e_d_14, e_pf_15, e_pg_15, e_fl_14, i14]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i16 : step16 = divstepRound^[16] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step16, e_d_15, e_pf_16, e_pg_16, e_fl_15, i15]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i17 : step17 = divstepRound^[17] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step17, e_d_16, e_pf_17, e_pg_17, e_fl_16, i16]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i18 : step18 = divstepRound^[18] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step18, e_d_17, e_pf_18, e_pg_18, e_fl_17, i17]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i19 : step19 = divstepRound^[19] ⟨d', pf_1, pg_1, fl⟩ := by
    rw [e_step19, e_d_18, e_pf_19, e_pg_19, e_fl_18, i18]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i20 : step20 = divstepLast (divstepRound^[19] ⟨d', pf_1, pg_1, fl⟩) := by
    rw [e_step20, e_d_19, e_pf_20, e_pg_20, e_fl_19, i19]
  have i21 : step21 = divstepRound^[1] ⟨d_20, pf_24, pg_24, fl_20⟩ := e_step21
  have i22 : step22 = divstepRound^[2] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step22, e_d_21, e_pf_25, e_pg_25, e_fl_21, i21]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i23 : step23 = divstepRound^[3] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step23, e_d_22, e_pf_26, e_pg_26, e_fl_22, i22]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i24 : step24 = divstepRound^[4] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step24, e_d_23, e_pf_27, e_pg_27, e_fl_23, i23]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i25 : step25 = divstepRound^[5] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step25, e_d_24, e_pf_28, e_pg_28, e_fl_24, i24]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i26 : step26 = divstepRound^[6] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step26, e_d_25, e_pf_29, e_pg_29, e_fl_25, i25]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i27 : step27 = divstepRound^[7] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step27, e_d_26, e_pf_30, e_pg_30, e_fl_26, i26]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i28 : step28 = divstepRound^[8] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step28, e_d_27, e_pf_31, e_pg_31, e_fl_27, i27]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i29 : step29 = divstepRound^[9] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step29, e_d_28, e_pf_32, e_pg_32, e_fl_28, i28]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i30 : step30 = divstepRound^[10] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step30, e_d_29, e_pf_33, e_pg_33, e_fl_29, i29]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i31 : step31 = divstepRound^[11] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step31, e_d_30, e_pf_34, e_pg_34, e_fl_30, i30]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i32 : step32 = divstepRound^[12] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step32, e_d_31, e_pf_35, e_pg_35, e_fl_31, i31]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i33 : step33 = divstepRound^[13] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step33, e_d_32, e_pf_36, e_pg_36, e_fl_32, i32]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i34 : step34 = divstepRound^[14] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step34, e_d_33, e_pf_37, e_pg_37, e_fl_33, i33]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i35 : step35 = divstepRound^[15] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step35, e_d_34, e_pf_38, e_pg_38, e_fl_34, i34]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i36 : step36 = divstepRound^[16] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step36, e_d_35, e_pf_39, e_pg_39, e_fl_35, i35]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i37 : step37 = divstepRound^[17] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step37, e_d_36, e_pf_40, e_pg_40, e_fl_36, i36]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i38 : step38 = divstepRound^[18] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step38, e_d_37, e_pf_41, e_pg_41, e_fl_37, i37]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i39 : step39 = divstepRound^[19] ⟨d_20, pf_24, pg_24, fl_20⟩ := by
    rw [e_step39, e_d_38, e_pf_42, e_pg_42, e_fl_38, i38]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i40 : step40 = divstepLast (divstepRound^[19] ⟨d_20, pf_24, pg_24, fl_20⟩) := by
    rw [e_step40, e_d_39, e_pf_43, e_pg_43, e_fl_39, i39]
  have i41 : step41 = divstepRound^[1] ⟨d_40, pf_47, pg_47, fl_40⟩ := e_step41
  have i42 : step42 = divstepRound^[2] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step42, e_d_41, e_pf_48, e_pg_48, e_fl_41, i41]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i43 : step43 = divstepRound^[3] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step43, e_d_42, e_pf_49, e_pg_49, e_fl_42, i42]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i44 : step44 = divstepRound^[4] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step44, e_d_43, e_pf_50, e_pg_50, e_fl_43, i43]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i45 : step45 = divstepRound^[5] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step45, e_d_44, e_pf_51, e_pg_51, e_fl_44, i44]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i46 : step46 = divstepRound^[6] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step46, e_d_45, e_pf_52, e_pg_52, e_fl_45, i45]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i47 : step47 = divstepRound^[7] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step47, e_d_46, e_pf_53, e_pg_53, e_fl_46, i46]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i48 : step48 = divstepRound^[8] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step48, e_d_47, e_pf_54, e_pg_54, e_fl_47, i47]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i49 : step49 = divstepRound^[9] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step49, e_d_48, e_pf_55, e_pg_55, e_fl_48, i48]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i50 : step50 = divstepRound^[10] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step50, e_d_49, e_pf_56, e_pg_56, e_fl_49, i49]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i51 : step51 = divstepRound^[11] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step51, e_d_50, e_pf_57, e_pg_57, e_fl_50, i50]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i52 : step52 = divstepRound^[12] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step52, e_d_51, e_pf_58, e_pg_58, e_fl_51, i51]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i53 : step53 = divstepRound^[13] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step53, e_d_52, e_pf_59, e_pg_59, e_fl_52, i52]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i54 : step54 = divstepRound^[14] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step54, e_d_53, e_pf_60, e_pg_60, e_fl_53, i53]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i55 : step55 = divstepRound^[15] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step55, e_d_54, e_pf_61, e_pg_61, e_fl_54, i54]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i56 : step56 = divstepRound^[16] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step56, e_d_55, e_pf_62, e_pg_62, e_fl_55, i55]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i57 : step57 = divstepRound^[17] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step57, e_d_56, e_pf_63, e_pg_63, e_fl_56, i56]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i58 : step58 = divstepRound^[18] ⟨d_40, pf_47, pg_47, fl_40⟩ := by
    rw [e_step58, e_d_57, e_pf_64, e_pg_64, e_fl_57, i57]
    exact (Function.iterate_succ_apply' _ _ _).symm
  have i59 : step59 = divstepLast (divstepRound^[18] ⟨d_40, pf_47, pg_47, fl_40⟩) := by
    rw [e_step59, e_d_58, e_pf_65, e_pg_65, e_fl_58, i58]
  clear i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 i12 i13 i14 i15 i16 i17 i18 i19 e_step1 e_step2 e_step3
    e_step4 e_step5 e_step6 e_step7 e_step8 e_step9 e_step10 e_step11 e_step12 e_step13 e_step14
    e_step15 e_step16 e_step17 e_step18 e_step19 e_step20 e_d_1 e_d_2 e_d_3 e_d_4 e_d_5 e_d_6
    e_d_7 e_d_8 e_d_9 e_d_10 e_d_11 e_d_12 e_d_13 e_d_14 e_d_15 e_d_16 e_d_17 e_d_18 e_d_19 e_pf_2
    e_pf_3 e_pf_4 e_pf_5 e_pf_6 e_pf_7 e_pf_8 e_pf_9 e_pf_10 e_pf_11 e_pf_12 e_pf_13 e_pf_14
    e_pf_15 e_pf_16 e_pf_17 e_pf_18 e_pf_19 e_pf_20 e_pg_2 e_pg_3 e_pg_4 e_pg_5 e_pg_6 e_pg_7
    e_pg_8 e_pg_9 e_pg_10 e_pg_11 e_pg_12 e_pg_13 e_pg_14 e_pg_15 e_pg_16 e_pg_17 e_pg_18 e_pg_19
    e_pg_20 e_fl_1 e_fl_2 e_fl_3 e_fl_4 e_fl_5 e_fl_6 e_fl_7 e_fl_8 e_fl_9 e_fl_10 e_fl_11 e_fl_12
    e_fl_13 e_fl_14 e_fl_15 e_fl_16 e_fl_17 e_fl_18 e_fl_19 step1 step2 step3 step4 step5 step6
    step7 step8 step9 step10 step11 step12 step13 step14 step15 step16 step17 step18 step19 d_1
    d_2 d_3 d_4 d_5 d_6 d_7 d_8 d_9 d_10 d_11 d_12 d_13 d_14 d_15 d_16 d_17 d_18 d_19 pf_2 pf_3
    pf_4 pf_5 pf_6 pf_7 pf_8 pf_9 pf_10 pf_11 pf_12 pf_13 pf_14 pf_15 pf_16 pf_17 pf_18 pf_19
    pf_20 pg_2 pg_3 pg_4 pg_5 pg_6 pg_7 pg_8 pg_9 pg_10 pg_11 pg_12 pg_13 pg_14 pg_15 pg_16 pg_17
    pg_18 pg_19 pg_20 fl_1 fl_2 fl_3 fl_4 fl_5 fl_6 fl_7 fl_8 fl_9 fl_10 fl_11 fl_12 fl_13 fl_14
    fl_15 fl_16 fl_17 fl_18 fl_19
  clear i21 i22 i23 i24 i25 i26 i27 i28 i29 i30 i31 i32 i33 i34 i35 i36 i37 i38 i39 e_step21
    e_step22 e_step23 e_step24 e_step25 e_step26 e_step27 e_step28 e_step29 e_step30 e_step31
    e_step32 e_step33 e_step34 e_step35 e_step36 e_step37 e_step38 e_step39 e_step40 e_d_21 e_d_22
    e_d_23 e_d_24 e_d_25 e_d_26 e_d_27 e_d_28 e_d_29 e_d_30 e_d_31 e_d_32 e_d_33 e_d_34 e_d_35
    e_d_36 e_d_37 e_d_38 e_d_39 e_pf_25 e_pf_26 e_pf_27 e_pf_28 e_pf_29 e_pf_30 e_pf_31 e_pf_32
    e_pf_33 e_pf_34 e_pf_35 e_pf_36 e_pf_37 e_pf_38 e_pf_39 e_pf_40 e_pf_41 e_pf_42 e_pf_43
    e_pg_25 e_pg_26 e_pg_27 e_pg_28 e_pg_29 e_pg_30 e_pg_31 e_pg_32 e_pg_33 e_pg_34 e_pg_35
    e_pg_36 e_pg_37 e_pg_38 e_pg_39 e_pg_40 e_pg_41 e_pg_42 e_pg_43 e_fl_21 e_fl_22 e_fl_23
    e_fl_24 e_fl_25 e_fl_26 e_fl_27 e_fl_28 e_fl_29 e_fl_30 e_fl_31 e_fl_32 e_fl_33 e_fl_34
    e_fl_35 e_fl_36 e_fl_37 e_fl_38 e_fl_39 step21 step22 step23 step24 step25 step26 step27
    step28 step29 step30 step31 step32 step33 step34 step35 step36 step37 step38 step39 d_21 d_22
    d_23 d_24 d_25 d_26 d_27 d_28 d_29 d_30 d_31 d_32 d_33 d_34 d_35 d_36 d_37 d_38 d_39 pf_25
    pf_26 pf_27 pf_28 pf_29 pf_30 pf_31 pf_32 pf_33 pf_34 pf_35 pf_36 pf_37 pf_38 pf_39 pf_40
    pf_41 pf_42 pf_43 pg_25 pg_26 pg_27 pg_28 pg_29 pg_30 pg_31 pg_32 pg_33 pg_34 pg_35 pg_36
    pg_37 pg_38 pg_39 pg_40 pg_41 pg_42 pg_43 fl_21 fl_22 fl_23 fl_24 fl_25 fl_26 fl_27 fl_28
    fl_29 fl_30 fl_31 fl_32 fl_33 fl_34 fl_35 fl_36 fl_37 fl_38 fl_39
  clear i41 i42 i43 i44 i45 i46 i47 i48 i49 i50 i51 i52 i53 i54 i55 i56 i57 i58 e_step41 e_step42
    e_step43 e_step44 e_step45 e_step46 e_step47 e_step48 e_step49 e_step50 e_step51 e_step52
    e_step53 e_step54 e_step55 e_step56 e_step57 e_step58 e_step59 e_d_41 e_d_42 e_d_43 e_d_44
    e_d_45 e_d_46 e_d_47 e_d_48 e_d_49 e_d_50 e_d_51 e_d_52 e_d_53 e_d_54 e_d_55 e_d_56 e_d_57
    e_d_58 e_pf_48 e_pf_49 e_pf_50 e_pf_51 e_pf_52 e_pf_53 e_pf_54 e_pf_55 e_pf_56 e_pf_57 e_pf_58
    e_pf_59 e_pf_60 e_pf_61 e_pf_62 e_pf_63 e_pf_64 e_pf_65 e_pg_48 e_pg_49 e_pg_50 e_pg_51
    e_pg_52 e_pg_53 e_pg_54 e_pg_55 e_pg_56 e_pg_57 e_pg_58 e_pg_59 e_pg_60 e_pg_61 e_pg_62
    e_pg_63 e_pg_64 e_pg_65 e_fl_41 e_fl_42 e_fl_43 e_fl_44 e_fl_45 e_fl_46 e_fl_47 e_fl_48
    e_fl_49 e_fl_50 e_fl_51 e_fl_52 e_fl_53 e_fl_54 e_fl_55 e_fl_56 e_fl_57 e_fl_58 step41 step42
    step43 step44 step45 step46 step47 step48 step49 step50 step51 step52 step53 step54 step55
    step56 step57 step58 d_41 d_42 d_43 d_44 d_45 d_46 d_47 d_48 d_49 d_50 d_51 d_52 d_53 d_54
    d_55 d_56 d_57 d_58 pf_48 pf_49 pf_50 pf_51 pf_52 pf_53 pf_54 pf_55 pf_56 pf_57 pf_58 pf_59
    pf_60 pf_61 pf_62 pf_63 pf_64 pf_65 pg_48 pg_49 pg_50 pg_51 pg_52 pg_53 pg_54 pg_55 pg_56
    pg_57 pg_58 pg_59 pg_60 pg_61 pg_62 pg_63 pg_64 pg_65 fl_41 fl_42 fl_43 fl_44 fl_45 fl_46
    fl_47 fl_48 fl_49 fl_50 fl_51 fl_52 fl_53 fl_54 fl_55 fl_56 fl_57 fl_58
  have hpos64 : (0 : ℤ) < 2^64 := by norm_num
  have hsDa : |s.d| < 2^61 := hsD
  rw [abs_lt] at hsD
  -- Batch 1: the packed start is `packedStart s₀` for the truncation `s₀` of `s`.
  set s₀ : State := ⟨s.d, s.f % 2^20, s.g % 2^20⟩ with hs₀
  have hs₀f : s₀.f % 2 = 1 := by show (s.f % 2^20) % 2 = 1; omega
  have hs₀fb : 0 ≤ s₀.f ∧ s₀.f < 2^20 :=
    ⟨Int.emod_nonneg _ (by norm_num), Int.emod_lt_of_pos _ (by norm_num)⟩
  have hs₀gb : 0 ≤ s₀.g ∧ s₀.g < 2^20 :=
    ⟨Int.emod_nonneg _ (by norm_num), Int.emod_lt_of_pos _ (by norm_num)⟩
  set P0 : State := packedStart s₀ with hP0
  have hP0f : P0.f % 2 = 1 := by show (s.f % 2^20 - 2^41) % 2 = 1; omega
  have hP0g : |P0.g| < 2^63 := by
    show |s.g % 2^20 - 2^62| < 2^63; rw [abs_lt]; constructor <;> omega
  have hP0D : |P0.d| + 2 * (19 + 1) < 2^62 := by show |s.d| + 2 * (19 + 1) < 2^62; omega
  have hw_d : (d' : ℤ) = P0.d % 2^64 := by rw [e_d']; exact ed
  have hw_f : (pf_1 : ℤ) = P0.f % 2^64 := by
    rw [e_pf_1, e_pf, e_f, pack_f_word]; show _ = (s.f % 2^20 - 2^41) % 2^64; omega
  have hw_g : (pg_1 : ℤ) = P0.g % 2^64 := by
    rw [e_pg_1, e_pg, e_g, pack_g_word]; show _ = (s.g % 2^20 - 2^62) % 2^64; omega
  have hw_z : fl.z = if P0.g % 2 = 0 then 1 else 0 := by rw [e_fl]; exact tst_one_z pg_1 P0.g hw_g
  obtain ⟨hb1, hd1, hf1, hg1⟩ := batch_words 19 P0 hP0f hsd hP0D hP0g
    (fun j _ => Inversion.divsteps_packedStart_g_abs_lt j s₀ hs₀f hs₀fb hs₀gb)
    ⟨d', pf_1, pg_1, fl⟩ ⟨b_d', b_pf_1, b_pg_1⟩ hw_d hw_f hw_g hw_z
  have b_d_20 : d_20 < 2^64 := by rw [e_d_20, i20]; exact hb1.1
  rw [← i20] at hd1 hf1 hg1
  rw [← e_d_20] at hd1
  rw [← e_pf_21] at hf1
  rw [← e_pg_21] at hg1
  clear hb1 i20 e_d_20 e_pf_21 e_pg_21 step20
  -- The words after batch 1 carry `divsteps 20 P0`, which Lemma 6 writes in the true state and
  -- matrix of `s₀`, and Lemma 2 in those of `s`.
  have hpk1 := Inversion.divsteps_packedStart 20 (le_refl _) s₀ hs₀f
  obtain ⟨hφ1, hγ1⟩ := Inversion.divsteps_abs_le 20 s₀ (2^20 - 1)
    (by rw [abs_le]; constructor <;> omega) (by rw [abs_le]; constructor <;> omega)
  obtain ⟨hA1, hB1, hC1, hD1⟩ := Inversion.M_entry_range 20 s₀
  obtain ⟨hdl1, hMl1⟩ := Inversion.divsteps_local 20 s s₀ hsf rfl
    (by show s.f % 2^20 = (s.f % 2^20) % 2^20; rw [Int.emod_emod_of_dvd _ (dvd_refl _)])
    (by show s.g % 2^20 = (s.g % 2^20) % 2^20; rw [Int.emod_emod_of_dvd _ (dvd_refl _)])
  rw [← hMl1] at hA1 hB1 hC1 hD1
  generalize hM1 : M 20 s = M1 at hA1 hB1 hC1 hD1 hMl1
  generalize hs20 : divsteps 20 s = s20 at hdl1
  have hd20 : (d_20 : ℤ) = s20.d % 2^64 := by
    rw [hd1, hpk1]; exact congrArg (· % 2^64) hdl1.symm
  obtain ⟨hA00, hA01⟩ := decode20 pf_21 _ _ _ _ hf1 (by rw [hpk1, hMl1])
    (by rw [abs_le] at hφ1; rw [abs_lt]; constructor <;> omega) hA1 hB1
  obtain ⟨hA10, hA11⟩ := decode20 pg_21 _ _ _ _ hg1 (by rw [hpk1, hMl1])
    (by rw [abs_le] at hγ1; rw [abs_lt]; constructor <;> omega) hC1 hD1
  rw [← e_a00, ← e_a00_1] at hA00
  rw [← e_a11, ← e_a11_1, ← e_a01, ← e_a01_1] at hA01
  rw [← e_a10, ← e_a10_1] at hA10
  rw [← e_a11, ← e_a11_1, ← e_a11_2, ← e_a11_3] at hA11
  -- The next low words: `-(M1 (f, g))` over `2^20`, so they carry `-f_20`, `-g_20` modulo `2^44`.
  obtain ⟨hMf1, hMg1⟩ := Inversion.M_spec 20 s hsf
  rw [hM1, hs20] at hMf1 hMg1
  have hf20 : s20.f % 2 = 1 := by rw [← hs20]; exact Inversion.divsteps_f_odd 20 s hsf
  have ht : (t : ℤ) = ((-M1.a) * s.f) % 2^64 := by
    rw [e_t]; exact mul_word _ _ _ _ hA00 (by rw [e_f]; exact ef0)
  have ht2 : (t2 : ℤ) = ((-M1.b) * s.g) % 2^64 := by
    rw [e_t2]; exact mul_word _ _ _ _ hA01 (by rw [e_g]; exact eg0)
  have hf1' : (f_1 : ℤ) = ((-M1.c) * s.f) % 2^64 := by
    rw [e_f_1]; exact mul_word _ _ _ _ hA10 (by rw [e_f]; exact ef0)
  have hg1' : (g_1 : ℤ) = ((-M1.d) * s.g) % 2^64 := by
    rw [e_g_1]; exact mul_word _ _ _ _ hA11 (by rw [e_g]; exact eg0)
  have hf2 : (f_2 : ℤ) % 2^44 = (-s20.f) % 2^44 := by
    rw [e_f_2]; apply asr20_word
    rw [e_pf_22, addw_word _ _ _ _ ht ht2]
    exact congrArg (· % 2^64) (by linear_combination hMf1)
  have hg2 : (g_2 : ℤ) % 2^44 = (-s20.g) % 2^44 := by
    rw [e_g_2]; apply asr20_word
    rw [e_pg_22, addw_word _ _ _ _ hf1' hg1']
    exact congrArg (· % 2^64) (by linear_combination hMg1)
  -- Batch 2 runs on the negated low words: its packed start is `packedStart s₁` for the truncation
  -- `s₁` of the negated state `sN`.
  set sN : State := ⟨s20.d, -s20.f, -s20.g⟩ with hsN
  set s₁ : State := ⟨s20.d, (-s20.f) % 2^20, (-s20.g) % 2^20⟩ with hs₁
  have hs₁f : s₁.f % 2 = 1 := by show ((-s20.f) % 2^20) % 2 = 1; omega
  have hs₁fb : 0 ≤ s₁.f ∧ s₁.f < 2^20 :=
    ⟨Int.emod_nonneg _ (by norm_num), Int.emod_lt_of_pos _ (by norm_num)⟩
  have hs₁gb : 0 ≤ s₁.g ∧ s₁.g < 2^20 :=
    ⟨Int.emod_nonneg _ (by norm_num), Int.emod_lt_of_pos _ (by norm_num)⟩
  set P1 : State := packedStart s₁ with hP1
  have hP1f : P1.f % 2 = 1 := by show ((-s20.f) % 2^20 - 2^41) % 2 = 1; omega
  have hP1d : P1.d % 2 = 1 := by
    show s20.d % 2 = 1; rw [← hs20, Inversion.divsteps_d_emod_two]; exact hsd
  have hs20D := Inversion.divsteps_d_abs_le 20 s
  rw [hs20] at hs20D
  push_cast at hs20D
  have hs20Db := abs_le.mp (le_trans hs20D (le_refl _))
  have hP1g : |P1.g| < 2^63 := by
    show |(-s20.g) % 2^20 - 2^62| < 2^63; rw [abs_lt]; constructor <;> omega
  have hP1D : |P1.d| + 2 * (19 + 1) < 2^62 := by
    show |s20.d| + 2 * (19 + 1) < 2^62; omega
  have hw_f2 : (pf_24 : ℤ) = P1.f % 2^64 := by
    rw [e_pf_24, e_pf_23, pack_f_word]; show _ = ((-s20.f) % 2^20 - 2^41) % 2^64; omega
  have hw_g2 : (pg_24 : ℤ) = P1.g % 2^64 := by
    rw [e_pg_24, e_pg_23, pack_g_word]; show _ = ((-s20.g) % 2^20 - 2^62) % 2^64; omega
  have hw_z2 : fl_20.z = if P1.g % 2 = 0 then 1 else 0 := by
    rw [e_fl_20]; exact tst_one_z pg_24 P1.g hw_g2
  obtain ⟨hb2, hd2, hf2', hg2'⟩ := batch_words 19 P1 hP1f hP1d hP1D hP1g
    (fun j _ => Inversion.divsteps_packedStart_g_abs_lt j s₁ hs₁f hs₁fb hs₁gb)
    ⟨d_20, pf_24, pg_24, fl_20⟩ ⟨b_d_20, b_pf_24, b_pg_24⟩ hd20 hw_f2 hw_g2 hw_z2
  have b_d_40 : d_40 < 2^64 := by rw [e_d_40, i40]; exact hb2.1
  rw [← i40] at hd2 hf2' hg2'
  rw [← e_d_40] at hd2
  rw [← e_pf_44] at hf2'
  rw [← e_pg_44] at hg2'
  clear hb2 i40 e_d_40 e_pf_44 e_pg_44 step40
  -- Lemma 6 in `s₁`, Lemma 2 from `sN` to `s₁`, and the sign symmetry from `s20` to `sN`.
  have hpk2 := Inversion.divsteps_packedStart 20 (le_refl _) s₁ hs₁f
  obtain ⟨hφ2, hγ2⟩ := Inversion.divsteps_abs_le 20 s₁ (2^20 - 1)
    (by rw [abs_le]; constructor <;> omega) (by rw [abs_le]; constructor <;> omega)
  obtain ⟨hA2, hB2, hC2, hD2⟩ := Inversion.M_entry_range 20 s₁
  have hsNf : sN.f % 2 = 1 := by show (-s20.f) % 2 = 1; omega
  obtain ⟨hdl2, hMl2⟩ := Inversion.divsteps_local 20 sN s₁ hsNf rfl
    (by show (-s20.f) % 2^20 = ((-s20.f) % 2^20) % 2^20; rw [Int.emod_emod_of_dvd _ (dvd_refl _)])
    (by show (-s20.g) % 2^20 = ((-s20.g) % 2^20) % 2^20; rw [Int.emod_emod_of_dvd _ (dvd_refl _)])
  have hMN : M 20 sN = M 20 s20 := Inversion.M_neg 20 s20 hf20
  have hdN : (divsteps 20 sN).d = (divsteps 20 s20).d := by
    rw [hsN, Inversion.divsteps_neg 20 s20 hf20]
  rw [← hMl2, hMN] at hA2 hB2 hC2 hD2
  generalize hM2 : M 20 s20 = M2 at hA2 hB2 hC2 hD2 hMN
  generalize hs40 : divsteps 40 s = s40
  have hs40' : divsteps 20 s20 = s40 := by
    rw [← hs40, ← hs20]; exact (Inversion.divsteps_add 20 20 s).symm
  have hd40 : (d_40 : ℤ) = s40.d % 2^64 := by
    rw [hd2, hpk2]
    exact congrArg (· % 2^64) (by rw [← hdl2, hdN, hs40'] : (divsteps 20 s₁).d = s40.d)
  obtain ⟨hB00, hB01⟩ := decode20 pf_44 _ _ _ _ hf2' (by rw [hpk2, ← hMl2, hMN])
    (by rw [abs_le] at hφ2; rw [abs_lt]; constructor <;> omega) hA2 hB2
  obtain ⟨hB10, hB11⟩ := decode20 pg_44 _ _ _ _ hg2' (by rw [hpk2, ← hMl2, hMN])
    (by rw [abs_le] at hγ2; rw [abs_lt]; constructor <;> omega) hC2 hD2
  rw [← e_b00, ← e_b00_1] at hB00
  rw [← e_b11, ← e_b11_1, ← e_b01, ← e_b01_1] at hB01
  rw [← e_b10, ← e_b10_1] at hB10
  rw [← e_b11, ← e_b11_1, ← e_b11_2, ← e_b11_3] at hB11
  -- The third low words: `-M2` times the negated words is `M2 (f_20, g_20)` modulo `2^44`, so after
  -- the shift they carry `f_40`, `g_40` modulo `2^24`.
  obtain ⟨hMf2, hMg2⟩ := Inversion.M_spec 20 s20 hf20
  rw [hs40', hM2] at hMf2 hMg2
  have hf40 : s40.f % 2 = 1 := by rw [← hs40']; exact Inversion.divsteps_f_odd 20 s20 hf20
  have hf2c : (f_2 : ℤ) = (f_2 : ℤ) % 2^64 :=
    (Int.emod_eq_of_lt (by omega) (by omega)).symm
  have hg2c : (g_2 : ℤ) = (g_2 : ℤ) % 2^64 :=
    (Int.emod_eq_of_lt (by omega) (by omega)).symm
  have ht_1 : (t_1 : ℤ) = ((-M2.a) * f_2) % 2^64 := by
    rw [e_t_1]; exact mul_word _ _ _ _ hB00 hf2c
  have ht2_1 : (t2_1 : ℤ) = ((-M2.b) * g_2) % 2^64 := by
    rw [e_t2_1]; exact mul_word _ _ _ _ hB01 hg2c
  have hf_3 : (f_3 : ℤ) = ((-M2.c) * f_2) % 2^64 := by
    rw [e_f_3]; exact mul_word _ _ _ _ hB10 hf2c
  have hg_3 : (g_3 : ℤ) = ((-M2.d) * g_2) % 2^64 := by
    rw [e_g_3]; exact mul_word _ _ _ _ hB11 hg2c
  have hf2m : (f_2 : ℤ) ≡ -s20.f [ZMOD 2^44] := hf2
  have hg2m : (g_2 : ℤ) ≡ -s20.g [ZMOD 2^44] := hg2
  have h44 : (2 : ℤ)^44 ∣ 2^64 := pow_dvd_pow 2 (by norm_num)
  have hf4 : (f_4 : ℤ) % 2^24 = s40.f % 2^24 := by
    rw [e_f_4]; apply asr20_word44 _ _
    rw [e_pf_45, addw_word _ _ _ _ ht_1 ht2_1, Int.emod_emod_of_dvd _ h44]
    have h := (hf2m.mul_left (-M2.a)).add (hg2m.mul_left (-M2.b))
    unfold Int.ModEq at h; rw [h]
    exact congrArg (· % 2^44) (by linear_combination -hMf2)
  have hg4 : (g_4 : ℤ) % 2^24 = s40.g % 2^24 := by
    rw [e_g_4]; apply asr20_word44 _ _
    rw [e_pg_45, addw_word _ _ _ _ hf_3 hg_3, Int.emod_emod_of_dvd _ h44]
    have h := (hf2m.mul_left (-M2.c)).add (hg2m.mul_left (-M2.d))
    unfold Int.ModEq at h; rw [h]
    exact congrArg (· % 2^44) (by linear_combination -hMg2)
  -- The products of the two negated matrices are `M2 · M1`.
  set P : Inversion.Mat2 := M2.mul M1 with hPdef
  have hf5 : (f_5 : ℤ) = ((-M2.a) * (-M1.a)) % 2^64 := by
    rw [e_f_5]; exact mul_word _ _ _ _ hB00 hA00
  have hg5 : (g_5 : ℤ) = ((-M2.a) * (-M1.b)) % 2^64 := by
    rw [e_g_5]; exact mul_word _ _ _ _ hB00 hA01
  have ht_2 : (t_2 : ℤ) = ((-M2.c) * (-M1.a)) % 2^64 := by
    rw [e_t_2]; exact mul_word _ _ _ _ hB10 hA00
  have ht2_2 : (t2_2 : ℤ) = ((-M2.c) * (-M1.b)) % 2^64 := by
    rw [e_t2_2]; exact mul_word _ _ _ _ hB10 hA01
  have hPa : (a00_2 : ℤ) = P.a % 2^64 := by
    rw [e_a00_2, madd_word _ _ _ _ _ _ hB01 hA10 hf5]
    exact congrArg (· % 2^64) (by simp only [hPdef, Inversion.Mat2.mul]; ring)
  have hPb : (a01_2 : ℤ) = P.b % 2^64 := by
    rw [e_a01_2, madd_word _ _ _ _ _ _ hB01 hA11 hg5]
    exact congrArg (· % 2^64) (by simp only [hPdef, Inversion.Mat2.mul]; ring)
  have hPc : (c10 : ℤ) = P.c % 2^64 := by
    rw [e_c10, madd_word _ _ _ _ _ _ hB11 hA10 ht_2]
    exact congrArg (· % 2^64) (by simp only [hPdef, Inversion.Mat2.mul]; ring)
  have hPd : (c11 : ℤ) = P.d % 2^64 := by
    rw [e_c11, madd_word _ _ _ _ _ _ hB11 hA11 ht2_2]
    exact congrArg (· % 2^64) (by simp only [hPdef, Inversion.Mat2.mul]; ring)
  -- Batch 3: 19 steps from the truncation `s₂` of `s40`, with the true sign.
  set s₂ : State := ⟨s40.d, s40.f % 2^20, s40.g % 2^20⟩ with hs₂
  have hs₂f : s₂.f % 2 = 1 := by show (s40.f % 2^20) % 2 = 1; omega
  have hs₂fb : 0 ≤ s₂.f ∧ s₂.f < 2^20 :=
    ⟨Int.emod_nonneg _ (by norm_num), Int.emod_lt_of_pos _ (by norm_num)⟩
  have hs₂gb : 0 ≤ s₂.g ∧ s₂.g < 2^20 :=
    ⟨Int.emod_nonneg _ (by norm_num), Int.emod_lt_of_pos _ (by norm_num)⟩
  set P2 : State := packedStart s₂ with hP2
  have hP2f : P2.f % 2 = 1 := by show (s40.f % 2^20 - 2^41) % 2 = 1; omega
  have hP2d : P2.d % 2 = 1 := by
    show s40.d % 2 = 1; rw [← hs40, Inversion.divsteps_d_emod_two]; exact hsd
  have hs40D := Inversion.divsteps_d_abs_le 40 s
  rw [hs40] at hs40D
  push_cast at hs40D
  have hs40Db := abs_le.mp (le_trans hs40D (le_refl _))
  have hP2g : |P2.g| < 2^63 := by
    show |s40.g % 2^20 - 2^62| < 2^63; rw [abs_lt]; constructor <;> omega
  have hP2D : |P2.d| + 2 * (18 + 1) < 2^62 := by
    show |s40.d| + 2 * (18 + 1) < 2^62; omega
  have hw_f3 : (pf_47 : ℤ) = P2.f % 2^64 := by
    rw [e_pf_47, e_pf_46, pack_f_word]; show _ = (s40.f % 2^20 - 2^41) % 2^64; omega
  have hw_g3 : (pg_47 : ℤ) = P2.g % 2^64 := by
    rw [e_pg_47, e_pg_46, pack_g_word]; show _ = (s40.g % 2^20 - 2^62) % 2^64; omega
  have hw_z3 : fl_40.z = if P2.g % 2 = 0 then 1 else 0 := by
    rw [e_fl_40]; exact tst_one_z pg_47 P2.g hw_g3
  obtain ⟨hb3, hd3, hf3', hg3'⟩ := batch_words 18 P2 hP2f hP2d hP2D hP2g
    (fun j _ => Inversion.divsteps_packedStart_g_abs_lt j s₂ hs₂f hs₂fb hs₂gb)
    ⟨d_40, pf_47, pg_47, fl_40⟩ ⟨b_d_40, b_pf_47, b_pg_47⟩ hd40 hw_f3 hw_g3 hw_z3
  rw [← i59] at hd3 hf3' hg3'
  rw [← e_d_59] at hd3
  rw [← e_pf_66] at hf3'
  rw [← e_pg_66] at hg3'
  clear hb3 i59 e_d_59 e_pf_66 e_pg_66 step59
  have hpk3 := Inversion.divsteps_packedStart 19 (by norm_num) s₂ hs₂f
  obtain ⟨hφ3, hγ3⟩ := Inversion.divsteps_abs_le 19 s₂ (2^20 - 1)
    (by rw [abs_le]; constructor <;> omega) (by rw [abs_le]; constructor <;> omega)
  obtain ⟨hA3, hB3, hC3, hD3⟩ := Inversion.M_entry_range 19 s₂
  have h19 : (2 : ℤ)^19 ∣ 2^20 := pow_dvd_pow 2 (by norm_num)
  obtain ⟨hdl3, hMl3⟩ := Inversion.divsteps_local 19 s40 s₂ hf40 rfl
    (by show s40.f % 2^19 = (s40.f % 2^20) % 2^19; rw [Int.emod_emod_of_dvd _ h19])
    (by show s40.g % 2^19 = (s40.g % 2^20) % 2^19; rw [Int.emod_emod_of_dvd _ h19])
  rw [← hMl3] at hA3 hB3 hC3 hD3
  generalize hM3 : M 19 s40 = M3 at hA3 hB3 hC3 hD3 hMl3
  have hd59 : (d_59 : ℤ) = (divsteps 59 s).d % 2^64 := by
    rw [hd3, hpk3]
    exact congrArg (· % 2^64)
      (by rw [← hdl3, ← hs40, ← Inversion.divsteps_add] : (divsteps 19 s₂).d = (divsteps 59 s).d)
  obtain ⟨hC00, hC01⟩ := decode19 pf_66 _ _ _ _ hf3' (by rw [hpk3, hMl3])
    (by rw [abs_le] at hφ3; rw [abs_lt]; constructor <;> omega) hA3 hB3
  obtain ⟨hC10, hC11⟩ := decode19 pg_66 _ _ _ _ hg3' (by rw [hpk3, hMl3])
    (by rw [abs_le] at hγ3; rw [abs_lt]; constructor <;> omega) hC3 hD3
  rw [← e_b00_2, ← e_b00_3] at hC00
  rw [← e_b11_4, ← e_b11_5, ← e_b01_2, ← e_b01_3] at hC01
  rw [← e_b10_2, ← e_b10_3] at hC10
  rw [← e_b11_4, ← e_b11_5, ← e_b11_6, ← e_b11_7] at hC11
  -- The final products: `mneg` and `msub` restore the sign of the third matrix.
  have hM59 : M 59 s = M3.mul P := by
    rw [← hM3, hPdef, ← hM2, ← hM1, ← hs40, ← hs20, ← Inversion.M_add, ← Inversion.M_add]
  have hf6 : (f_6 : ℤ) = (-((-M3.a) * P.a)) % 2^64 := by
    rw [e_f_6]; exact mneg_word _ _ _ _ hC00 hPa
  have hg6 : (g_6 : ℤ) = (-((-M3.a) * P.b)) % 2^64 := by
    rw [e_g_6]; exact mneg_word _ _ _ _ hC00 hPb
  have hpf67 : (pf_67 : ℤ) = (-((-M3.c) * P.a)) % 2^64 := by
    rw [e_pf_67]; exact mneg_word _ _ _ _ hC10 hPa
  have hpg67 : (pg_67 : ℤ) = (-((-M3.c) * P.b)) % 2^64 := by
    rw [e_pg_67]; exact mneg_word _ _ _ _ hC10 hPb
  refine ⟨hd59, ?_, ?_, ?_, ?_⟩
  · rw [e_m00, msub_word _ _ _ _ _ _ hC01 hPc hf6, hM59]
    exact congrArg (· % 2^64) (by simp only [Inversion.Mat2.mul]; ring)
  · rw [e_m01, msub_word _ _ _ _ _ _ hC01 hPd hg6, hM59]
    exact congrArg (· % 2^64) (by simp only [Inversion.Mat2.mul]; ring)
  · rw [e_m10, msub_word _ _ _ _ _ _ hC11 hPc hpf67, hM59]
    exact congrArg (· % 2^64) (by simp only [Inversion.Mat2.mul]; ring)
  · rw [e_m11, msub_word _ _ _ _ _ _ hC11 hPd hpg67, hM59]
    exact congrArg (· % 2^64) (by simp only [Inversion.Mat2.mul]; ring)
  -- END conclusion

end PastaCurves.AArch64
