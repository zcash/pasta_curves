/-
Copyright Amazon.com, Inc. or its affiliates (the block, adapted from s2n-bignum).
Copyright (c) 2026 the pasta_curves contributors (the transcription and the proofs).
-/
import PastaCurves.AArch64.Spec.Words
import PastaCurves.AArch64.Transcription
import PastaCurves.Inversion.PackedWords

/-!
# Correctness of the inversion's packed divstep

See the parent module's documentation for details. `divstepRound` is one step of the packed
recurrence of `Inversion/Packed.lean`, which is the plain `divstep` on the packed state, on
two's-complement words. The flags on entry carry the parity of the packed `g` (`Z` set when it is
even); `ccmp` then tests the sign of `d` only when `g` is odd, so `ge` is the divstep's swap
condition `0 < d ∧ g odd`. The step ends by testing bit 1 of the unhalved `g`, which is the parity
of the halved one, for the next step. `divstepLast` is the same step without that test.

The one subtlety is the sum `g ± f`, which the step forms in a word before halving it. The halving
is exact only when the sum does not wrap, that is when the halved result is below `2^62` in
magnitude. The row-sum bound on the transition matrix does not give this by itself: both packed
words can approach `2^62` in magnitude, and their sum would reach `2^63` exactly when both rows of
the matrix are at their extreme, which the determinant of the matrix rules out. The step theorems
take the bound as the hypothesis `hG'`, and the batch proof supplies it for every step.
-/

namespace PastaCurves.AArch64

open Inversion (State divstep divstep_words_even divstep_words_swap divstep_words_add)

-- BEGIN divstep lemmas
/-- `cmp d, xzr`: `N` is the sign bit of `d`, and `V` is clear. -/
theorem cmpFlags_zero (d : ℕ) (hd : d < 2^64) :
    (cmpFlags d 0).n = d / 2^63 ∧ (cmpFlags d 0).v = 0 := by
  refine ⟨?_, ?_⟩ <;> simp [cmpFlags, regMod] <;> omega

/-- The immediate `#8` of `ccmp` sets `N` alone. -/
theorem immFlags_eight : immFlags 8 = ⟨1, 0, 0, 0⟩ := rfl

/-- `tst x, #2` finds bit 1 clear exactly when `x / 2` is even. -/
theorem andw_two_eq_zero_iff (x : ℕ) : andw x 2 = 0 ↔ x / 2 % 2 = 0 := by
  have h2 : ∀ i, Nat.testBit 2 i = decide (i = 1) := by
    intro i
    rw [show (2 : ℕ) = 2^1 by norm_num, Nat.testBit_two_pow]
    simp [eq_comm]
  have hx : x.testBit 1 = decide (x / 2 % 2 = 1) := by
    simp only [Nat.testBit_eq_decide_div_mod_eq, pow_one]
  unfold andw
  constructor
  · intro h
    have h1 := congrArg (fun y => Nat.testBit y 1) h
    simp only [Nat.testBit_and, h2, Nat.zero_testBit, hx] at h1
    simp at h1
    omega
  · intro h
    apply Nat.eq_of_testBit_eq
    intro i
    rw [Nat.testBit_and, h2, Nat.zero_testBit]
    by_cases hi : i = 1
    · subst hi; simp [hx, h]
    · simp [hi]

/-- The step on words, from its instruction equations: the new `d`, `f`, and `g` words carry the
divstep of `s`, and bit 1 of the unhalved `g` word is the parity of the new `g`. The three cases are
the divstep's: `g` even, the swap when `d > 0`, and the addition when `d < 0`. In each, the
conditional instructions' values are read off the flags, and the shared case lemma of
`Inversion/PackedWords.lean` does the arithmetic. -/
theorem divstep_words (s : State) (hf : s.f % 2 = 1) (hd : s.d % 2 = 1) (hD : |s.d| < 2^62)
    (hG : |s.g| < 2^63) (hG' : |(divstep s).g| < 2^62)
    (d pf pg : ℕ) (fl : Flags) (b_d : d < 2^64) (b_pf : pf < 2^64)
    (hd' : (d : ℤ) = s.d % 2^64) (hf' : (pf : ℤ) = s.f % 2^64) (hg' : (pg : ℤ) = s.g % 2^64)
    (hfz : fl.z = if s.g % 2 = 0 then 1 else 0)
    (t : ℕ) (fl_1 : Flags) (d_1 t_1 pf_1 pg_1 d_2 pg_2 : ℕ)
    (e_t : t = cselNe fl pf 0) (e_fl_1 : fl_1 = ccmpNe fl d 0 8) (e_d_1 : d_1 = cnegGe fl_1 d)
    (e_t_1 : t_1 = cnegGe fl_1 t) (e_pf_1 : pf_1 = cselGe fl_1 pg pf) (e_pg_1 : pg_1 = addw pg t_1)
    (e_d_2 : d_2 = addw d_1 2) (e_pg_2 : pg_2 = asr pg_1 1) :
    (d_2 : ℤ) = (divstep s).d % 2^64 ∧ (pf_1 : ℤ) = (divstep s).f % 2^64 ∧
      (pg_2 : ℤ) = (divstep s).g % 2^64 ∧ (pg_1 / 2 % 2 = 0 ↔ (divstep s).g % 2 = 0) := by
  obtain ⟨hDl, hDu⟩ := abs_lt.mp hD
  obtain ⟨hcn, hcv⟩ := cmpFlags_zero d b_d
  rcases Int.emod_two_eq_zero_or_one s.g with hg0 | hg1
  · -- `g` even: no swap, `d + 2`, and `g / 2`.
    have hz1 : fl.z = 1 := by rw [hfz, if_pos hg0]
    have ht : t = 0 := by rw [e_t, cselNe, hz1]; simp
    have hfl1 : fl_1 = immFlags 8 := by rw [e_fl_1, ccmpNe, hz1]; simp
    have hge : ¬ fl_1.n = fl_1.v := by rw [hfl1, immFlags_eight]; decide
    have hd1 : d_1 = d := by rw [e_d_1, cnegGe, if_neg hge]
    have ht1 : t_1 = 0 := by rw [e_t_1, cnegGe, if_neg hge, ht]
    have hpf1 : pf_1 = pf := by rw [e_pf_1, cselGe, if_neg hge]
    exact divstep_words_even s hd hG d pf pg hd' hf' hg' hg0 t_1 d_1 pf_1 pg_1 d_2 pg_2 ht1 hd1
      hpf1 e_pg_1 e_d_2 e_pg_2
  · rcases lt_or_ge 0 s.d with hdpos | hdnp
    · -- The swap: `2 - d`, `f := g`, and `(g - f) / 2`.
      have hsw : 0 < s.d ∧ s.g % 2 = 1 := ⟨hdpos, hg1⟩
      have hz0 : fl.z = 0 := by rw [hfz, if_neg (by omega)]
      have ht : t = pf := by rw [e_t, cselNe, hz0]; simp
      have hfl1 : fl_1 = cmpFlags d 0 := by rw [e_fl_1, ccmpNe, hz0]; simp
      have hge : fl_1.n = fl_1.v := by rw [hfl1, hcn, hcv]; omega
      have hd1 : d_1 = negw d := by rw [e_d_1, cnegGe, if_pos hge]
      have ht1 : t_1 = negw pf := by rw [e_t_1, cnegGe, if_pos hge, ht]
      have hpf1 : pf_1 = pg := by rw [e_pf_1, cselGe, if_pos hge]
      exact divstep_words_swap s hf hd hG' d pf pg b_d b_pf hd' hf' hg' hsw t_1 d_1 pf_1 pg_1
        d_2 pg_2 ht1 hd1 hpf1 e_pg_1 e_d_2 e_pg_2
    · -- The addition: `d + 2`, `f`, and `(g + f) / 2`.
      have hz0 : fl.z = 0 := by rw [hfz, if_neg (by omega)]
      have ht : t = pf := by rw [e_t, cselNe, hz0]; simp
      have hfl1 : fl_1 = cmpFlags d 0 := by rw [e_fl_1, ccmpNe, hz0]; simp
      have hge : ¬ fl_1.n = fl_1.v := by rw [hfl1, hcn, hcv]; omega
      have hd1 : d_1 = d := by rw [e_d_1, cnegGe, if_neg hge]
      have ht1 : t_1 = pf := by rw [e_t_1, cnegGe, if_neg hge, ht]
      have hpf1 : pf_1 = pf := by rw [e_pf_1, cselGe, if_neg hge]
      exact divstep_words_add s hf hd hD hG' d pf pg hd' hf' hg' hg1 hdnp t_1 d_1 pf_1 pg_1 d_2
        pg_2 ht1 hd1 hpf1 e_pg_1 e_d_2 e_pg_2
-- END divstep lemmas

-- BEGIN divstepRound_spec statement
/-- One packed divstep on words is the divstep of `Inversion/Divstep.lean` on the packed state `s`,
whose words `st` are, with the flags carrying the parity of `s.g`. `s.f` is odd, as the packed `f`
always is, so that the halving is exact; `hG'` is the no-wrap condition on the sum `g ± f` discussed
in the module documentation. The result's words and its flags carry the next state and its parity.
-/
theorem divstepRound_spec (st : DivstepState) (hst : st.Bounded) (s : State)
    (hf : s.f % 2 = 1) (hd : s.d % 2 = 1) (hD : |s.d| < 2^62) (hG : |s.g| < 2^63)
    (hG' : |(divstep s).g| < 2^62)
    (ed : (st.d : ℤ) = s.d % 2^64) (ef : (st.f : ℤ) = s.f % 2^64) (eg : (st.g : ℤ) = s.g % 2^64)
    (hz : st.fl.z = if s.g % 2 = 0 then 1 else 0) :
    ∀ r, r = divstepRound st →
      r.Bounded ∧ (r.d : ℤ) = (divstep s).d % 2^64 ∧ (r.f : ℤ) = (divstep s).f % 2^64 ∧
        (r.g : ℤ) = (divstep s).g % 2^64 ∧ r.fl.z = if (divstep s).g % 2 = 0 then 1 else 0 := by
  intro r hr
-- END divstepRound_spec statement
  -- generated skeleton for `divstepRound`: do not edit between the annotations
  unfold divstepRound at hr
  lift_lets -merge at hr
  -- d: argument
  word_step d := st.d using hst.1
  -- pf: argument
  word_step pf := st.f using hst.2.1
  -- pg: argument
  word_step pg := st.g using hst.2.2
  -- fl: argument
  word_step fl := st.fl
  -- t: csel t,pf,xzr,ne
  word_step t := cselNe fl pf 0 using cselNe_lt fl pf 0 b_pf (by decide)
  -- fl_1: ccmp d,xzr,#8,ne
  word_step fl_1 := ccmpNe fl d 0 8
  -- d_1: cneg d,d,ge
  word_step d_1 := cnegGe fl_1 d using cnegGe_lt fl_1 d b_d
  -- t_1: cneg t,t,ge
  word_step t_1 := cnegGe fl_1 t using cnegGe_lt fl_1 t b_t
  -- pf_1: csel pf,pg,pf,ge
  word_step pf_1 := cselGe fl_1 pg pf using cselGe_lt fl_1 pg pf b_pg b_pf
  -- pg_1: add pg,pg,t
  word_step pg_1 := addw pg t_1 using addw_lt pg t_1
  -- d_2: add d,d,#2
  word_step d_2 := addw d_1 2 using addw_lt d_1 2
  -- fl_2: tst pg,#2
  word_step fl_2 := tstFlags (andw pg_1 2)
  -- pg_2: asr pg,pg,#1
  word_step pg_2 := asr pg_1 1 using asr_lt pg_1 1 b_pg_1
  subst hr
  -- BEGIN conclusion
  obtain ⟨h1, h2, h3, h4⟩ := divstep_words s hf hd hD hG hG' d pf pg fl b_d b_pf
    (by rw [e_d]; exact ed) (by rw [e_pf]; exact ef) (by rw [e_pg]; exact eg)
    (by rw [e_fl]; exact hz) t fl_1 d_1 t_1 pf_1 pg_1 d_2 pg_2 e_t e_fl_1 e_d_1 e_t_1 e_pf_1
    e_pg_1 e_d_2 e_pg_2
  refine ⟨⟨b_d_2, b_pf_1, b_pg_2⟩, h1, h2, h3, ?_⟩
  rw [e_fl_2]
  show (if andw pg_1 2 = 0 then 1 else 0) = _
  simp only [andw_two_eq_zero_iff, h4]
  -- END conclusion

-- BEGIN divstepLast_spec statement
/-- The last step of a batch: as `divstepRound_spec`, without the parity flags, which the batch does
not read. -/
theorem divstepLast_spec (st : DivstepState) (hst : st.Bounded) (s : State)
    (hf : s.f % 2 = 1) (hd : s.d % 2 = 1) (hD : |s.d| < 2^62) (hG : |s.g| < 2^63)
    (hG' : |(divstep s).g| < 2^62)
    (ed : (st.d : ℤ) = s.d % 2^64) (ef : (st.f : ℤ) = s.f % 2^64) (eg : (st.g : ℤ) = s.g % 2^64)
    (hz : st.fl.z = if s.g % 2 = 0 then 1 else 0) :
    ∀ r, r = divstepLast st →
      r.Bounded ∧ (r.d : ℤ) = (divstep s).d % 2^64 ∧ (r.f : ℤ) = (divstep s).f % 2^64 ∧
        (r.g : ℤ) = (divstep s).g % 2^64 := by
  intro r hr
-- END divstepLast_spec statement
  -- generated skeleton for `divstepLast`: do not edit between the annotations
  unfold divstepLast at hr
  lift_lets -merge at hr
  -- d: argument
  word_step d := st.d using hst.1
  -- pf: argument
  word_step pf := st.f using hst.2.1
  -- pg: argument
  word_step pg := st.g using hst.2.2
  -- fl: argument
  word_step fl := st.fl
  -- t: csel t,pf,xzr,ne
  word_step t := cselNe fl pf 0 using cselNe_lt fl pf 0 b_pf (by decide)
  -- fl_1: ccmp d,xzr,#8,ne
  word_step fl_1 := ccmpNe fl d 0 8
  -- d_1: cneg d,d,ge
  word_step d_1 := cnegGe fl_1 d using cnegGe_lt fl_1 d b_d
  -- t_1: cneg t,t,ge
  word_step t_1 := cnegGe fl_1 t using cnegGe_lt fl_1 t b_t
  -- pf_1: csel pf,pg,pf,ge
  word_step pf_1 := cselGe fl_1 pg pf using cselGe_lt fl_1 pg pf b_pg b_pf
  -- pg_1: add pg,pg,t
  word_step pg_1 := addw pg t_1 using addw_lt pg t_1
  -- d_2: add d,d,#2
  word_step d_2 := addw d_1 2 using addw_lt d_1 2
  -- pg_2: asr pg,pg,#1
  word_step pg_2 := asr pg_1 1 using asr_lt pg_1 1 b_pg_1
  subst hr
  -- BEGIN conclusion
  obtain ⟨h1, h2, h3, -⟩ := divstep_words s hf hd hD hG hG' d pf pg fl b_d b_pf
    (by rw [e_d]; exact ed) (by rw [e_pf]; exact ef) (by rw [e_pg]; exact eg)
    (by rw [e_fl]; exact hz) t fl_1 d_1 t_1 pf_1 pg_1 d_2 pg_2 e_t e_fl_1 e_d_1 e_t_1 e_pf_1
    e_pg_1 e_d_2 e_pg_2
  exact ⟨⟨b_d_2, b_pf_1, b_pg_2⟩, h1, h2, h3⟩
  -- END conclusion

end PastaCurves.AArch64
