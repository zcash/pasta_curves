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
even); `ccmp` then tests the sign of `two_delta` only when `g` is odd, so `ge` is the divstep's swap
condition `0 < two_delta ∧ g odd`. The step ends by testing bit 1 of the unhalved `g`, which is the
parity of the halved one, for the next step. `divstepLast` is the same step without that test.

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
/-- `cmp two_delta, xzr`: `N` is the sign bit of `two_delta`, and `V` is clear. -/
theorem cmpFlags_zero (two_delta : ℕ) (hd : two_delta < 2^64) :
    (cmpFlags two_delta 0).n = two_delta / 2^63 ∧ (cmpFlags two_delta 0).v = 0 := by
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

/-- The step on words, from its instruction equations: the new `two_delta`, `f`, and `g` words carry
the divstep of `s`, and bit 1 of the unhalved `g` word is the parity of the new `g`. The three cases
are the divstep's: `g` even, the swap when `two_delta > 0`, and the addition when `two_delta < 0`.
In each, the conditional instructions' values are read off the flags, and the shared case lemma of
`Inversion/PackedWords.lean` does the arithmetic. -/
theorem divstep_words (s : State) (hf : s.f % 2 = 1) (hd : s.two_delta % 2 = 1) (hD : |s.two_delta| < 2^62)
    (hG : |s.g| < 2^63) (hG' : |(divstep s).g| < 2^62)
    (two_delta pf pg : ℕ) (fl : Flags) (b_two_delta : two_delta < 2^64) (b_pf : pf < 2^64)
    (hd' : (two_delta : ℤ) = s.two_delta % 2^64) (hf' : (pf : ℤ) = s.f % 2^64) (hg' : (pg : ℤ) = s.g % 2^64)
    (hfz : fl.z = if s.g % 2 = 0 then 1 else 0)
    (t : ℕ) (fl_1 : Flags) (two_delta_1 t_1 pf_1 pg_1 two_delta_2 pg_2 : ℕ)
    (e_t : t = cselNe fl pf 0) (e_fl_1 : fl_1 = ccmpNe fl two_delta 0 8) (e_two_delta_1 : two_delta_1 = cnegGe fl_1 two_delta)
    (e_t_1 : t_1 = cnegGe fl_1 t) (e_pf_1 : pf_1 = cselGe fl_1 pg pf) (e_pg_1 : pg_1 = addw pg t_1)
    (e_two_delta_2 : two_delta_2 = addw two_delta_1 2) (e_pg_2 : pg_2 = asr pg_1 1) :
    (two_delta_2 : ℤ) = (divstep s).two_delta % 2^64 ∧ (pf_1 : ℤ) = (divstep s).f % 2^64 ∧
      (pg_2 : ℤ) = (divstep s).g % 2^64 ∧ (pg_1 / 2 % 2 = 0 ↔ (divstep s).g % 2 = 0) := by
  obtain ⟨hDl, hDu⟩ := abs_lt.mp hD
  obtain ⟨hcn, hcv⟩ := cmpFlags_zero two_delta b_two_delta
  rcases Int.emod_two_eq_zero_or_one s.g with hg0 | hg1
  · -- `g` even: no swap, `two_delta + 2`, and `g / 2`.
    have hz1 : fl.z = 1 := by rw [hfz, if_pos hg0]
    have ht : t = 0 := by rw [e_t, cselNe, hz1]; simp
    have hfl1 : fl_1 = immFlags 8 := by rw [e_fl_1, ccmpNe, hz1]; simp
    have hge : ¬ fl_1.n = fl_1.v := by rw [hfl1, immFlags_eight]; decide
    have hd1 : two_delta_1 = two_delta := by rw [e_two_delta_1, cnegGe, if_neg hge]
    have ht1 : t_1 = 0 := by rw [e_t_1, cnegGe, if_neg hge, ht]
    have hpf1 : pf_1 = pf := by rw [e_pf_1, cselGe, if_neg hge]
    exact divstep_words_even s hd hG two_delta pf pg hd' hf' hg' hg0 t_1 two_delta_1 pf_1 pg_1 two_delta_2 pg_2 ht1 hd1
      hpf1 e_pg_1 e_two_delta_2 e_pg_2
  · rcases lt_or_ge 0 s.two_delta with hdpos | hdnp
    · -- The swap: `2 - two_delta`, `f := g`, and `(g - f) / 2`.
      have hsw : 0 < s.two_delta ∧ s.g % 2 = 1 := ⟨hdpos, hg1⟩
      have hz0 : fl.z = 0 := by rw [hfz, if_neg (by omega)]
      have ht : t = pf := by rw [e_t, cselNe, hz0]; simp
      have hfl1 : fl_1 = cmpFlags two_delta 0 := by rw [e_fl_1, ccmpNe, hz0]; simp
      have hge : fl_1.n = fl_1.v := by rw [hfl1, hcn, hcv]; omega
      have hd1 : two_delta_1 = negw two_delta := by rw [e_two_delta_1, cnegGe, if_pos hge]
      have ht1 : t_1 = negw pf := by rw [e_t_1, cnegGe, if_pos hge, ht]
      have hpf1 : pf_1 = pg := by rw [e_pf_1, cselGe, if_pos hge]
      exact divstep_words_swap s hf hd hG' two_delta pf pg b_two_delta b_pf hd' hf' hg' hsw t_1 two_delta_1 pf_1 pg_1
        two_delta_2 pg_2 ht1 hd1 hpf1 e_pg_1 e_two_delta_2 e_pg_2
    · -- The addition: `two_delta + 2`, `f`, and `(g + f) / 2`.
      have hz0 : fl.z = 0 := by rw [hfz, if_neg (by omega)]
      have ht : t = pf := by rw [e_t, cselNe, hz0]; simp
      have hfl1 : fl_1 = cmpFlags two_delta 0 := by rw [e_fl_1, ccmpNe, hz0]; simp
      have hge : ¬ fl_1.n = fl_1.v := by rw [hfl1, hcn, hcv]; omega
      have hd1 : two_delta_1 = two_delta := by rw [e_two_delta_1, cnegGe, if_neg hge]
      have ht1 : t_1 = pf := by rw [e_t_1, cnegGe, if_neg hge, ht]
      have hpf1 : pf_1 = pf := by rw [e_pf_1, cselGe, if_neg hge]
      exact divstep_words_add s hf hd hD hG' two_delta pf pg hd' hf' hg' hg1 hdnp t_1 two_delta_1 pf_1 pg_1 two_delta_2
        pg_2 ht1 hd1 hpf1 e_pg_1 e_two_delta_2 e_pg_2
-- END divstep lemmas

-- BEGIN divstepRound_spec statement
/-- One packed divstep on words is the divstep of `Inversion/Divstep.lean` on the packed state `s`,
whose words `st` are, with the flags carrying the parity of `s.g`. `s.f` is odd, as the packed `f`
always is, so that the halving is exact; `hG'` is the no-wrap condition on the sum `g ± f` discussed
in the module documentation. The result's words and its flags carry the next state and its parity.
-/
theorem divstepRound_spec (st : DivstepState) (hst : st.Bounded) (s : State)
    (hf : s.f % 2 = 1) (hd : s.two_delta % 2 = 1) (hD : |s.two_delta| < 2^62) (hG : |s.g| < 2^63)
    (hG' : |(divstep s).g| < 2^62)
    (ed : (st.two_delta : ℤ) = s.two_delta % 2^64) (ef : (st.f : ℤ) = s.f % 2^64) (eg : (st.g : ℤ) = s.g % 2^64)
    (hz : st.fl.z = if s.g % 2 = 0 then 1 else 0) :
    ∀ res, res = divstepRound st →
      res.Bounded ∧ (res.two_delta : ℤ) = (divstep s).two_delta % 2^64 ∧ (res.f : ℤ) = (divstep s).f % 2^64 ∧
        (res.g : ℤ) = (divstep s).g % 2^64 ∧ res.fl.z = if (divstep s).g % 2 = 0 then 1 else 0 := by
  intro res hres
-- END divstepRound_spec statement
  -- generated skeleton for `divstepRound`: do not edit between the annotations
  unfold divstepRound at hres
  lift_lets -merge at hres
  -- two_delta: argument
  extract_lets -merge +onlyGivenNames two_delta at hres
  have e_two_delta : two_delta = st.two_delta := rfl
  clear_value two_delta
  have b_two_delta : two_delta < 2^64 := by rw [e_two_delta]; exact hst.1
  -- pf: argument
  extract_lets -merge +onlyGivenNames pf at hres
  have e_pf : pf = st.f := rfl
  clear_value pf
  have b_pf : pf < 2^64 := by rw [e_pf]; exact hst.2.1
  -- pg: argument
  extract_lets -merge +onlyGivenNames pg at hres
  have e_pg : pg = st.g := rfl
  clear_value pg
  have b_pg : pg < 2^64 := by rw [e_pg]; exact hst.2.2
  -- fl: argument
  extract_lets -merge +onlyGivenNames fl at hres
  have e_fl : fl = st.fl := rfl
  clear_value fl
  -- t: csel t,pf,xzr,ne
  extract_lets -merge +onlyGivenNames t at hres
  have e_t : t = cselNe fl pf 0 := rfl
  clear_value t
  have b_t : t < 2^64 := by rw [e_t]; exact cselNe_lt fl pf 0 b_pf (by decide)
  -- fl_1: ccmp two_delta,xzr,#8,ne
  extract_lets -merge +onlyGivenNames fl_1 at hres
  have e_fl_1 : fl_1 = ccmpNe fl two_delta 0 8 := rfl
  clear_value fl_1
  -- two_delta_1: cneg two_delta,two_delta,ge
  extract_lets -merge +onlyGivenNames two_delta_1 at hres
  have e_two_delta_1 : two_delta_1 = cnegGe fl_1 two_delta := rfl
  clear_value two_delta_1
  have b_two_delta_1 : two_delta_1 < 2^64 := by rw [e_two_delta_1]; exact cnegGe_lt fl_1 two_delta b_two_delta
  -- t_1: cneg t,t,ge
  extract_lets -merge +onlyGivenNames t_1 at hres
  have e_t_1 : t_1 = cnegGe fl_1 t := rfl
  clear_value t_1
  have b_t_1 : t_1 < 2^64 := by rw [e_t_1]; exact cnegGe_lt fl_1 t b_t
  -- pf_1: csel pf,pg,pf,ge
  extract_lets -merge +onlyGivenNames pf_1 at hres
  have e_pf_1 : pf_1 = cselGe fl_1 pg pf := rfl
  clear_value pf_1
  have b_pf_1 : pf_1 < 2^64 := by rw [e_pf_1]; exact cselGe_lt fl_1 pg pf b_pg b_pf
  -- pg_1: add pg,pg,t
  extract_lets -merge +onlyGivenNames pg_1 at hres
  have e_pg_1 : pg_1 = addw pg t_1 := rfl
  clear_value pg_1
  have b_pg_1 : pg_1 < 2^64 := by rw [e_pg_1]; exact addw_lt pg t_1
  -- two_delta_2: add two_delta,two_delta,#2
  extract_lets -merge +onlyGivenNames two_delta_2 at hres
  have e_two_delta_2 : two_delta_2 = addw two_delta_1 2 := rfl
  clear_value two_delta_2
  have b_two_delta_2 : two_delta_2 < 2^64 := by rw [e_two_delta_2]; exact addw_lt two_delta_1 2
  -- fl_2: tst pg,#2
  extract_lets -merge +onlyGivenNames fl_2 at hres
  have e_fl_2 : fl_2 = tstFlags (andw pg_1 2) := rfl
  clear_value fl_2
  -- pg_2: asr pg,pg,#1
  extract_lets -merge +onlyGivenNames pg_2 at hres
  have e_pg_2 : pg_2 = asr pg_1 1 := rfl
  clear_value pg_2
  have b_pg_2 : pg_2 < 2^64 := by rw [e_pg_2]; exact asr_lt pg_1 1 b_pg_1
  subst hres
  -- BEGIN conclusion
  obtain ⟨h1, h2, h3, h4⟩ := divstep_words s hf hd hD hG hG' two_delta pf pg fl b_two_delta b_pf
    (by rw [e_two_delta]; exact ed) (by rw [e_pf]; exact ef) (by rw [e_pg]; exact eg)
    (by rw [e_fl]; exact hz) t fl_1 two_delta_1 t_1 pf_1 pg_1 two_delta_2 pg_2 e_t e_fl_1 e_two_delta_1 e_t_1 e_pf_1
    e_pg_1 e_two_delta_2 e_pg_2
  refine ⟨⟨b_two_delta_2, b_pf_1, b_pg_2⟩, h1, h2, h3, ?_⟩
  rw [e_fl_2]
  show (if andw pg_1 2 = 0 then 1 else 0) = _
  simp only [andw_two_eq_zero_iff, h4]
  -- END conclusion

-- BEGIN divstepLast_spec statement
/-- The last step of a batch: as `divstepRound_spec`, without the parity flags, which the batch does
not read. -/
theorem divstepLast_spec (st : DivstepState) (hst : st.Bounded) (s : State)
    (hf : s.f % 2 = 1) (hd : s.two_delta % 2 = 1) (hD : |s.two_delta| < 2^62) (hG : |s.g| < 2^63)
    (hG' : |(divstep s).g| < 2^62)
    (ed : (st.two_delta : ℤ) = s.two_delta % 2^64) (ef : (st.f : ℤ) = s.f % 2^64) (eg : (st.g : ℤ) = s.g % 2^64)
    (hz : st.fl.z = if s.g % 2 = 0 then 1 else 0) :
    ∀ res, res = divstepLast st →
      res.Bounded ∧ (res.two_delta : ℤ) = (divstep s).two_delta % 2^64 ∧ (res.f : ℤ) = (divstep s).f % 2^64 ∧
        (res.g : ℤ) = (divstep s).g % 2^64 := by
  intro res hres
-- END divstepLast_spec statement
  -- generated skeleton for `divstepLast`: do not edit between the annotations
  unfold divstepLast at hres
  lift_lets -merge at hres
  -- two_delta: argument
  extract_lets -merge +onlyGivenNames two_delta at hres
  have e_two_delta : two_delta = st.two_delta := rfl
  clear_value two_delta
  have b_two_delta : two_delta < 2^64 := by rw [e_two_delta]; exact hst.1
  -- pf: argument
  extract_lets -merge +onlyGivenNames pf at hres
  have e_pf : pf = st.f := rfl
  clear_value pf
  have b_pf : pf < 2^64 := by rw [e_pf]; exact hst.2.1
  -- pg: argument
  extract_lets -merge +onlyGivenNames pg at hres
  have e_pg : pg = st.g := rfl
  clear_value pg
  have b_pg : pg < 2^64 := by rw [e_pg]; exact hst.2.2
  -- fl: argument
  extract_lets -merge +onlyGivenNames fl at hres
  have e_fl : fl = st.fl := rfl
  clear_value fl
  -- t: csel t,pf,xzr,ne
  extract_lets -merge +onlyGivenNames t at hres
  have e_t : t = cselNe fl pf 0 := rfl
  clear_value t
  have b_t : t < 2^64 := by rw [e_t]; exact cselNe_lt fl pf 0 b_pf (by decide)
  -- fl_1: ccmp two_delta,xzr,#8,ne
  extract_lets -merge +onlyGivenNames fl_1 at hres
  have e_fl_1 : fl_1 = ccmpNe fl two_delta 0 8 := rfl
  clear_value fl_1
  -- two_delta_1: cneg two_delta,two_delta,ge
  extract_lets -merge +onlyGivenNames two_delta_1 at hres
  have e_two_delta_1 : two_delta_1 = cnegGe fl_1 two_delta := rfl
  clear_value two_delta_1
  have b_two_delta_1 : two_delta_1 < 2^64 := by rw [e_two_delta_1]; exact cnegGe_lt fl_1 two_delta b_two_delta
  -- t_1: cneg t,t,ge
  extract_lets -merge +onlyGivenNames t_1 at hres
  have e_t_1 : t_1 = cnegGe fl_1 t := rfl
  clear_value t_1
  have b_t_1 : t_1 < 2^64 := by rw [e_t_1]; exact cnegGe_lt fl_1 t b_t
  -- pf_1: csel pf,pg,pf,ge
  extract_lets -merge +onlyGivenNames pf_1 at hres
  have e_pf_1 : pf_1 = cselGe fl_1 pg pf := rfl
  clear_value pf_1
  have b_pf_1 : pf_1 < 2^64 := by rw [e_pf_1]; exact cselGe_lt fl_1 pg pf b_pg b_pf
  -- pg_1: add pg,pg,t
  extract_lets -merge +onlyGivenNames pg_1 at hres
  have e_pg_1 : pg_1 = addw pg t_1 := rfl
  clear_value pg_1
  have b_pg_1 : pg_1 < 2^64 := by rw [e_pg_1]; exact addw_lt pg t_1
  -- two_delta_2: add two_delta,two_delta,#2
  extract_lets -merge +onlyGivenNames two_delta_2 at hres
  have e_two_delta_2 : two_delta_2 = addw two_delta_1 2 := rfl
  clear_value two_delta_2
  have b_two_delta_2 : two_delta_2 < 2^64 := by rw [e_two_delta_2]; exact addw_lt two_delta_1 2
  -- pg_2: asr pg,pg,#1
  extract_lets -merge +onlyGivenNames pg_2 at hres
  have e_pg_2 : pg_2 = asr pg_1 1 := rfl
  clear_value pg_2
  have b_pg_2 : pg_2 < 2^64 := by rw [e_pg_2]; exact asr_lt pg_1 1 b_pg_1
  subst hres
  -- BEGIN conclusion
  obtain ⟨h1, h2, h3, -⟩ := divstep_words s hf hd hD hG hG' two_delta pf pg fl b_two_delta b_pf
    (by rw [e_two_delta]; exact ed) (by rw [e_pf]; exact ef) (by rw [e_pg]; exact eg)
    (by rw [e_fl]; exact hz) t fl_1 two_delta_1 t_1 pf_1 pg_1 two_delta_2 pg_2 e_t e_fl_1 e_two_delta_1 e_t_1 e_pf_1
    e_pg_1 e_two_delta_2 e_pg_2
  exact ⟨⟨b_two_delta_2, b_pf_1, b_pg_2⟩, h1, h2, h3⟩
  -- END conclusion

end PastaCurves.AArch64
