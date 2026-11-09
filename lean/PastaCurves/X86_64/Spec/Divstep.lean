/-
Copyright Amazon.com, Inc. or its affiliates (the block, adapted from s2n-bignum).
Copyright (c) 2026 the pasta_curves contributors (the transcription and the proofs).
-/
import PastaCurves.Spec
import PastaCurves.X86_64.Spec.Arithmetic
import PastaCurves.X86_64.Transcription
import PastaCurves.Inversion.PackedWords
import PastaCurves.Inversion.SignMag
import Mathlib.Data.Nat.Bitwise
import Mathlib.Tactic.ClearExcept

/-!
# Correctness of the inversion's packed divstep

`divstepRound` is one step of the packed recurrence of `Inversion/Packed.lean` on two's-complement
words, as s2n-bignum's x86-64 `divstep59` computes it: the step finishes the previous step and
decides the next. Between two steps the words therefore hold the packed state with the next step's
decision applied (`Decided`): on a swap, `two_delta` is already negated, `f` is already `g`, and the
term `t` that the step adds to `g` is `-f`; with `g` odd and no swap, `t` is `f`; with `g` even,
`t` is zero. The decision is a mask `k`, `-2` to swap and `0` otherwise, and xoring an odd word
with `-2` negates it (`eorw_neg_two`).

A step finishes as the AArch64 one does: `two_delta + 2`, and `(g + t) / 2`, which the shared case
lemmas of `Inversion/PackedWords.lean` relate to `divstep`, with the same no-wrap hypothesis `hG'`.
The batch's first decision is made from the packed words before any step (`decide_words`); its
last step is `finishStep`, which decides nothing.
-/

namespace PastaCurves.X86_64

open Inversion (State divstep divstep_words_even divstep_words_swap divstep_words_add
  divstep_f_odd divstep_two_delta_abs_le divstep_two_delta_emod_two andw_two_eq_zero_iff
  PackedStep)

-- BEGIN divstep lemmas
/-- Xoring an odd word with `-2` negates it: complementing all its bits gives `2^64 - 1 - x`,
which is even, and setting bit 0 adds one. The swap mask negates `f` and `two_delta` this way. -/
theorem eorw_neg_two (x : ℕ) (hx : x < 2^64) (ho : x % 2 = 1) :
    eorw x 18446744073709551614 = negw x := by
  have hm : (18446744073709551614 : ℕ) = (2^64 - 1) ^^^ 1 := by
    rw [Nat.xor_one_of_odd (Nat.odd_iff.mpr (by norm_num))]; norm_num
  have he : Even (2^64 - 1 - x) := by rw [Nat.even_iff]; omega
  unfold eorw negw
  rw [hm, ← Nat.xor_assoc, show x ^^^ (2^64 - 1) = eorw x (2^64 - 1) from rfl,
    Inversion.eorw_ones x hx, Nat.xor_one_of_even he]
  simp only [regMod]
  omega

/-- The words between two steps: the packed state `s`, with the next step's decision applied.
`td` and `pf` are the words of `s.two_delta` and `s.f`; on a swap the state holds their negations
and `f := g`, and `t` is what the step adds to `g`. -/
def Decided (st : DivstepState) (s : State) : Prop :=
  st.z = 0 ∧ (st.g : ℤ) = s.g % 2^64 ∧
  ∃ td pf : ℕ, td < 2^64 ∧ pf < 2^64 ∧ (td : ℤ) = s.two_delta % 2^64 ∧ (pf : ℤ) = s.f % 2^64 ∧
    ((s.g % 2 = 0 ∧ st.t = 0 ∧ st.two_delta = td ∧ st.f = pf) ∨
     (0 < s.two_delta ∧ s.g % 2 = 1 ∧ st.t = negw pf ∧ st.two_delta = negw td ∧ st.f = st.g) ∨
     (s.g % 2 = 1 ∧ s.two_delta ≤ 0 ∧ st.t = pf ∧ st.two_delta = td ∧ st.f = pf))

/-- The decision of a step, from words carrying `s`: the mask is `-2` unless `two_delta` is
negative or `g` is even, which `zs` tests, and then `t`, `two_delta`, and `f` follow it. The
conditional moves take the zero word, which the state keeps. -/
theorem decide_words (s : State) (hf : s.f % 2 = 1) (hd : s.two_delta % 2 = 1)
    (hD : |s.two_delta| < 2^63) (td pf pg : ℕ) (b_td : td < 2^64) (b_pf : pf < 2^64)
    (hd' : (td : ℤ) = s.two_delta % 2^64) (hf' : (pf : ℤ) = s.f % 2^64)
    (hg' : (pg : ℤ) = s.g % 2^64) (zs : ℕ) (hzs : zs = 0 ↔ s.g % 2 = 0) :
    Decided ⟨eorw td (cmovz zs (cmovs td 18446744073709551614 0) 0),
      cmovb (bt (cmovz zs (cmovs td 18446744073709551614 0) 0) 63) pf pg, pg,
      eorw (cmovz zs pf 0) (cmovz zs (cmovs td 18446744073709551614 0) 0), 0⟩ s := by
  rw [abs_lt] at hD
  have hpo : pf % 2 = 1 := by omega
  have hto : td % 2 = 1 := by omega
  refine ⟨rfl, hg', td, pf, b_td, b_pf, hd', hf', ?_⟩
  dsimp only
  have e0 : ∀ x, eorw x 0 = x := fun x => Nat.xor_zero x
  rcases Int.emod_two_eq_zero_or_one s.g with hg0 | hg1
  · have hz : zs = 0 := hzs.mpr hg0
    left
    simp only [cmovz, if_pos hz, e0, bt, cmovb]
    exact ⟨hg0, by norm_num, by norm_num, by norm_num⟩
  · have hz : zs ≠ 0 := fun h => by have := hzs.mp h; omega
    simp only [cmovz, if_neg hz]
    rcases lt_or_ge 0 s.two_delta with hpos | hnp
    · have hs : td < 2^63 := by omega
      right; left
      simp only [cmovs, if_pos hs, bt, cmovb]
      rw [eorw_neg_two pf b_pf hpo, eorw_neg_two td b_td hto]
      exact ⟨hpos, hg1, by norm_num, by norm_num, by norm_num⟩
    · have hs : ¬ td < 2^63 := by omega
      right; right
      simp only [cmovs, if_neg hs, e0, bt, cmovb]
      exact ⟨hg1, hnp, by norm_num, by norm_num, by norm_num⟩

/-- A step finished: `two_delta + 2`, `f`, and `(g + t) / 2`, from words with the decision
applied. This is the last step of a batch, which decides nothing more. -/
def finishStep (st : DivstepState) : DivstepState :=
  ⟨(addc st.two_delta 2 0).1, st.f, asr (addw st.g st.t) 1, st.t, st.z⟩

/-- Finishing a decided step: the words carry `divstep s`, by the shared case lemmas, and bit 1 of
the unhalved `g` is the parity of the new `g`. -/
theorem finish_words (st : DivstepState) (s : State) (hf : s.f % 2 = 1)
    (hd : s.two_delta % 2 = 1) (hD : |s.two_delta| < 2^62) (hG : |s.g| < 2^63)
    (hG' : |(divstep s).g| < 2^62) (hc : Decided st s) :
    ((addc st.two_delta 2 0).1 : ℤ) = (divstep s).two_delta % 2^64 ∧
      (st.f : ℤ) = (divstep s).f % 2^64 ∧
      (asr (addw st.g st.t) 1 : ℤ) = (divstep s).g % 2^64 ∧
      (addw st.g st.t / 2 % 2 = 0 ↔ (divstep s).g % 2 = 0) := by
  obtain ⟨-, hg', td, pf, b_td, b_pf, hd', hf', hcase⟩ := hc
  have ha : (addc st.two_delta 2 0).1 = addw st.two_delta 2 := rfl
  rw [ha]
  rcases hcase with ⟨hg0, ht, htd, hpf⟩ | ⟨hpos, hg1, ht, htd, hpf⟩ | ⟨hg1, hnp, ht, htd, hpf⟩
  · exact divstep_words_even s hd hG td pf st.g hd' hf' hg' hg0 st.t st.two_delta st.f _ _ _
      ht htd hpf rfl rfl rfl
  · exact divstep_words_swap s hf hd hG' td pf st.g b_td b_pf hd' hf' hg' ⟨hpos, hg1⟩ st.t
      st.two_delta st.f _ _ _ ht htd hpf rfl rfl rfl
  · exact divstep_words_add s hf hd hD hG' td pf st.g hd' hf' hg' hg1 hnp st.t st.two_delta
      st.f _ _ _ ht htd hpf rfl rfl rfl

-- END divstep lemmas

-- BEGIN divstepRound_spec statement
/-- One step on words, finishing the decided step on `σ` and deciding the next: the result is
bounded and carries `divstep s` with its decision applied. The hypotheses are the AArch64 step's:
`s.f` odd so that the halving is exact, and `hG'`, the no-wrap condition on the sum `g ± f`. -/
theorem divstepRound_spec (st : DivstepState) (hst : st.Bounded) (σ : State)
    (hf : σ.f % 2 = 1) (hd : σ.two_delta % 2 = 1) (hD : |σ.two_delta| < 2^62) (hG : |σ.g| < 2^63)
    (hG' : |(divstep σ).g| < 2^62) (hc : Decided st σ) :
    ∀ res, res = divstepRound st → res.Bounded ∧ Decided res (divstep σ) := by
  intro res hres
  obtain ⟨h1, h2, h3, h4⟩ := finish_words st σ hf hd hD hG hG' hc
  have hz0 : st.z = 0 := hc.1
-- END divstepRound_spec statement
  -- generated skeleton for `divstepRound`: do not edit between the annotations
  unfold divstepRound at hres
  lift_lets -merge at hres
  -- td: argument
  word_step td := st.two_delta using hst.1
  -- pf: argument
  word_step pf := st.f using hst.2.1
  -- pg: argument
  word_step pg := st.g using hst.2.2.1
  -- t: argument
  word_step t := st.t using hst.2.2.2.1
  -- z: argument
  word_step z := st.z using hst.2.2.2.2
  -- k: mov {k}, -2
  word_step k := 18446744073709551614 using (by decide)
  -- td_1: add {td}, 2
  word_step s, td_1 := (addc td 2 0).1 using addc_value_lt td 2 0, cf := (addc td 2 0).2
  have l_td_1 : td_1 + 2^64 * cf = td + 2 + 0 := by
    rw [e_td_1, e_cf]; exact addc_lin td 2 0
  have b_cf : cf ≤ 1 := by rw [e_cf]; exact addc_carry_le_one td 2 0 b_td (by decide) (by decide)
  clear e_td_1 e_cf
  -- pg_1: lea {pg}, [{pg} + {t}]
  word_step pg_1 := addw pg t using addw_lt pg t
  -- k_1: cmovs {k}, {z}
  word_step k_1 := cmovs td_1 k z using cmovs_lt td_1 k z b_k b_z
  -- t_1: mov {t}, {pf}
  word_step t_1 := pf using b_pf
  -- zs: test {pg}, 2
  word_step zs := andw pg_1 2 using andw_lt pg_1 2 b_pg_1 (by decide)
  -- k_2: cmove {k}, {z}
  word_step k_2 := cmovz zs k_1 z using cmovz_lt zs k_1 z b_k_1 b_z
  -- t_2: cmove {t}, {z}
  word_step t_2 := cmovz zs t_1 z using cmovz_lt zs t_1 z b_t_1 b_z
  -- pg_2: sar {pg}, 1
  word_step pg_2 := asr pg_1 1 using asr_lt pg_1 1 b_pg_1
  -- t_3: xor {t}, {k}
  word_step t_3 := eorw t_2 k_2 using eorw_lt t_2 k_2 b_t_2 b_k_2
  -- td_2: xor {td}, {k}
  word_step td_2 := eorw td_1 k_2 using eorw_lt td_1 k_2 b_td_1 b_k_2
  -- cf_1: bt {k}, 63
  word_step cf_1 := bt k_2 63
  have b_cf_1 : cf_1 ≤ 1 := by
    rw [e_cf_1]; exact Nat.le_of_lt_succ (Nat.mod_lt _ (by decide))
  -- pf_1: cmovb {pf}, {pg}
  word_step pf_1 := cmovb cf_1 pf pg_2 using cmovb_lt cf_1 pf pg_2 b_pf b_pg_2
  subst hres
  -- BEGIN conclusion
  refine ⟨⟨b_td_2, b_pf_1, b_pg_2, b_t_3, b_z⟩, ?_⟩
  have hz : z = 0 := by rw [e_z]; exact hz0
  have hs' : |(divstep σ).two_delta| < 2^63 := by
    have := divstep_two_delta_abs_le σ; linarith
  have htd1 : td_1 = (addc st.two_delta 2 0).1 := by
    have h := addc_lin st.two_delta 2 0
    have h' := addc_value_lt st.two_delta 2 0
    rw [← e_td] at h h' ⊢
    omega
  have key := decide_words (divstep σ) (divstep_f_odd σ hf)
    (by rw [divstep_two_delta_emod_two]; exact hd) hs' td_1 pf pg_2 b_td_1 b_pf
    (by rw [htd1]; exact h1) (by rw [e_pf]; exact h2)
    (by rw [e_pg_2, e_pg_1, e_pg, e_t]; exact h3) zs
    (by rw [e_zs, andw_two_eq_zero_iff, e_pg_1, e_pg, e_t]; exact h4)
  rw [e_td_2, e_t_3, e_pf_1, e_cf_1, e_t_2, e_t_1, e_k_2, e_k_1, e_k, hz]
  exact key
  -- END conclusion

/-- The x86-64 packed step, for the shared batch iteration: `divstepRound`, and `finishStep` to
end a batch, on `DivstepState`, whose words carry the packed state with the next step's decision
applied. -/
def packedStep : PackedStep :=
  ⟨DivstepState, DivstepState.Bounded, (·.two_delta), (·.f), (·.g), Decided, divstepRound,
    finishStep⟩

/-- The two step theorems, as the batch iteration takes them. -/
theorem packedStep_spec : packedStep.Spec where
  round := fun st hst s hf hd hD hG hG' hc => divstepRound_spec st hst s hf hd hD hG hG' hc _ rfl
  last := fun st hst s hf hd hD hG hG' hc => by
    obtain ⟨h1, h2, h3, -⟩ := finish_words st s hf hd hD hG hG' hc
    exact ⟨⟨addc_value_lt _ _ _, hst.2.1, asr_lt _ _ (addw_lt _ _), hst.2.2.2.1, hst.2.2.2.2⟩,
      h1, h2, h3⟩

end PastaCurves.X86_64
