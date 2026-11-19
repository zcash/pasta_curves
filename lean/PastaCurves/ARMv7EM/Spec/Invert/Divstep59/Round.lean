/-
Copyright (c) 2026 the pasta_curves contributors.
-/
import PastaCurves.ARMv7EM.Spec.Invert.Divstep59.Finish

/-!
# Correctness of one ARMv7E-M packed divstep round

The instruction prefix selects the addend, conditionally swaps the packed words, and updates the
32-bit `two_delta`. The factored suffix shifts the complete packed `g` word and updates the public
loop control. The arithmetic proof follows the shared three-case packed-divstep proof.
-/

namespace PastaCurves.ARMv7EM

open Inversion (State divstep divstep_words_even divstep_words_swap divstep_words_add)

-- BEGIN batchRound_spec statement
/-- One generated batch round implements one packed divstep on bounded words. The signed
`two_delta` budget is the width-31 contract of the ARMv7E-M register implementation. -/
theorem batchRound_spec (st : BatchState) (hst : packedStep.Bounded st) (state : State)
    (hf : state.f % 2 = 1) (hd : state.two_delta % 2 = 1)
    (hD : |state.two_delta| + 2 < 2^31) (hG : |state.g| < 2^63)
    (hG' : |(divstep state).g| < 2^62)
    (ed : (signExtend32 st.d : Int) = state.two_delta % 2^64)
    (ef : (join32 st.pflo st.pfhi : Int) = state.f % 2^64)
    (eg : (join32 st.pglo st.pghi : Int) = state.g % 2^64)
    (hz : (if st.pglo % 2 = 0 then 1 else 0) = if state.g % 2 = 0 then 1 else 0) :
    ∀ res, res = batchRound st →
      packedStep.Bounded res ∧
        (signExtend32 res.d : Int) = (divstep state).two_delta % 2^64 ∧
        (join32 res.pflo res.pfhi : Int) = (divstep state).f % 2^64 ∧
        (join32 res.pglo res.pghi : Int) = (divstep state).g % 2^64 ∧
        (if res.pglo % 2 = 0 then 1 else 0) =
          if (divstep state).g % 2 = 0 then 1 else 0 := by
  intro res hres
-- END batchRound_spec statement
  -- generated skeleton for `batchRound`: do not edit between the annotations
  unfold batchRound at hres
  -- pflo: input pflo
  word_step -clear pflo := st.pflo
  have b_pflo : pflo < 2^32 := by rw [e_pflo]; exact hst.pflo
  -- pfhi: input pfhi
  word_step -clear pfhi := st.pfhi
  have b_pfhi : pfhi < 2^32 := by rw [e_pfhi]; exact hst.pfhi
  -- pglo: input pglo
  word_step -clear pglo := st.pglo
  have b_pglo : pglo < 2^32 := by rw [e_pglo]; exact hst.pglo
  -- pghi: input pghi
  word_step -clear pghi := st.pghi
  have b_pghi : pghi < 2^32 := by rw [e_pghi]; exact hst.pghi
  -- d: input d
  word_step -clear d := st.d
  have b_d : d < 2^32 := by rw [e_d]; exact hst.d
  -- remaining: input remaining
  word_step -clear remaining := st.remaining
  have b_remaining : remaining < 2^32 := by rw [e_remaining]; exact hst.remaining
  -- odd: and {odd}, {pglo}, #1
  word_step -clear odd := andw pglo 1
  have b_odd : odd < 2^32 := by
    rw [e_odd]; exact Nat.and_lt_two_pow pglo (by decide)
  -- odd_1: rsb.w {odd}, {odd}, #0
  word_step -clear odd_1 := (0 + 2^32 - odd) % 2^32
  have b_odd_1 : odd_1 < 2^32 := by rw [e_odd_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- sub: rsbs {tmp}, {d}, #0
  word_step -clear sub, tmp := (0 + 2^32 - d - (1 - 1)) % 2^32
  have b_tmp : tmp < 2^32 := by rw [e_tmp]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- swap: asr.w {swap}, {tmp}, #31
  word_step -clear swap := asr32 tmp 31
  have b_swap : swap < 2^32 := by
    rw [e_swap, asr32]; split <;> norm_num [wordMod] at * <;> omega
  -- swap_1: and.w {swap}, {swap}, {odd}
  word_step -clear swap_1 := andw swap odd_1
  have b_swap_1 : swap_1 < 2^32 := by
    rw [e_swap_1]; exact Nat.and_lt_two_pow swap b_odd_1
  -- alo: and.w {alo}, {pflo}, {odd}
  word_step -clear alo := andw pflo odd_1
  have b_alo : alo < 2^32 := by
    rw [e_alo]; exact Nat.and_lt_two_pow pflo b_odd_1
  -- ahi: and.w {ahi}, {pfhi}, {odd}
  word_step -clear ahi := andw pfhi odd_1
  have b_ahi : ahi < 2^32 := by
    rw [e_ahi]; exact Nat.and_lt_two_pow pfhi b_odd_1
  -- alo_1: eor.w {alo}, {alo}, {swap}
  word_step -clear alo_1 := (alo ^^^ swap_1) % 2^32
  have b_alo_1 : alo_1 < 2^32 := by rw [e_alo_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have w_alo_1 : (alo ^^^ swap_1) < 2^32 := Nat.xor_lt_two_pow b_alo b_swap_1
  -- ahi_1: eor.w {ahi}, {ahi}, {swap}
  word_step -clear ahi_1 := (ahi ^^^ swap_1) % 2^32
  have b_ahi_1 : ahi_1 < 2^32 := by rw [e_ahi_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have w_ahi_1 : (ahi ^^^ swap_1) < 2^32 := Nat.xor_lt_two_pow b_ahi b_swap_1
  -- tmp_1: and.w {tmp}, {swap}, #1
  word_step -clear tmp_1 := andw swap_1 1
  have b_tmp_1 : tmp_1 < 2^32 := by
    rw [e_tmp_1]; exact Nat.and_lt_two_pow swap_1 (by decide)
  -- alo_2: adds {alo}, {alo}, {tmp}
  word_step -clear s, alo_2 := (alo_1 + tmp_1 + 0) % 2^32, c := (alo_1 + tmp_1 + 0) / 2^32
  have b_alo_2 : alo_2 < 2^32 := by rw [e_alo_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_alo_2 : alo_2 + 2^32 * c = alo_1 + tmp_1 + 0 := by
    rw [e_alo_2, e_c]; exact Nat.mod_add_div _ _
  have b_c : c ≤ 1 := by
    rw [e_c]; exact addc32_carry_le_one alo_1 tmp_1 0 b_alo_1 b_tmp_1 (by decide)
  clear e_alo_2 e_c
  -- tmp_2: mov.w {tmp}, #0
  word_step -clear tmp_2 := 0
  have b_tmp_2 : tmp_2 < 2^32 := by rw [e_tmp_2]; exact (by decide)
  -- ahi_2: adc.w {ahi}, {ahi}, {tmp}
  word_step -clear ahi_2 := (ahi_1 + tmp_2 + c) % 2^32
  have b_ahi_2 : ahi_2 < 2^32 := by rw [e_ahi_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_ahi_2, b_k_ahi_2, l_ahi_2⟩ :
      ∃ k, k ≤ 1 ∧ ahi_2 + 2^32 * k = ahi_1 + tmp_2 + c :=
    ⟨(ahi_1 + tmp_2 + c) / 2^32, addc32_carry_le_one ahi_1 tmp_2 c b_ahi_1 b_tmp_2 b_c,
      by rw [e_ahi_2]; exact Nat.mod_add_div _ _⟩
  clear e_ahi_2
  -- tmp_3: rsb.w {tmp}, {d}, #2
  word_step -clear tmp_3 := (2 + 2^32 - d) % 2^32
  have b_tmp_3 : tmp_3 < 2^32 := by rw [e_tmp_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- d_1: add.w {d}, {d}, #2
  word_step -clear d_1 := (d + 2) % 2^32
  have b_d_1 : d_1 < 2^32 := by rw [e_d_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- tmp_4: eor.w {tmp}, {tmp}, {d}
  word_step -clear tmp_4 := (tmp_3 ^^^ d_1) % 2^32
  have b_tmp_4 : tmp_4 < 2^32 := by rw [e_tmp_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have w_tmp_4 : (tmp_3 ^^^ d_1) < 2^32 := Nat.xor_lt_two_pow b_tmp_3 b_d_1
  -- tmp_5: and.w {tmp}, {tmp}, {swap}
  word_step -clear tmp_5 := andw tmp_4 swap_1
  have b_tmp_5 : tmp_5 < 2^32 := by
    rw [e_tmp_5]; exact Nat.and_lt_two_pow tmp_4 b_swap_1
  -- d_2: eor.w {d}, {d}, {tmp}
  word_step -clear d_2 := (d_1 ^^^ tmp_5) % 2^32
  have b_d_2 : d_2 < 2^32 := by rw [e_d_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have w_d_2 : (d_1 ^^^ tmp_5) < 2^32 := Nat.xor_lt_two_pow b_d_1 b_tmp_5
  -- tmp_6: eor.w {tmp}, {pflo}, {pglo}
  word_step -clear tmp_6 := (pflo ^^^ pglo) % 2^32
  have b_tmp_6 : tmp_6 < 2^32 := by rw [e_tmp_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have w_tmp_6 : (pflo ^^^ pglo) < 2^32 := Nat.xor_lt_two_pow b_pflo b_pglo
  -- tmp_7: and.w {tmp}, {tmp}, {swap}
  word_step -clear tmp_7 := andw tmp_6 swap_1
  have b_tmp_7 : tmp_7 < 2^32 := by
    rw [e_tmp_7]; exact Nat.and_lt_two_pow tmp_6 b_swap_1
  -- pflo_1: eor.w {pflo}, {pflo}, {tmp}
  word_step -clear pflo_1 := (pflo ^^^ tmp_7) % 2^32
  have b_pflo_1 : pflo_1 < 2^32 := by rw [e_pflo_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have w_pflo_1 : (pflo ^^^ tmp_7) < 2^32 := Nat.xor_lt_two_pow b_pflo b_tmp_7
  -- tmp_8: eor.w {tmp}, {pfhi}, {pghi}
  word_step -clear tmp_8 := (pfhi ^^^ pghi) % 2^32
  have b_tmp_8 : tmp_8 < 2^32 := by rw [e_tmp_8]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have w_tmp_8 : (pfhi ^^^ pghi) < 2^32 := Nat.xor_lt_two_pow b_pfhi b_pghi
  -- tmp_9: and.w {tmp}, {tmp}, {swap}
  word_step -clear tmp_9 := andw tmp_8 swap_1
  have b_tmp_9 : tmp_9 < 2^32 := by
    rw [e_tmp_9]; exact Nat.and_lt_two_pow tmp_8 b_swap_1
  -- pfhi_1: eor.w {pfhi}, {pfhi}, {tmp}
  word_step -clear pfhi_1 := (pfhi ^^^ tmp_9) % 2^32
  have b_pfhi_1 : pfhi_1 < 2^32 := by rw [e_pfhi_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have w_pfhi_1 : (pfhi ^^^ tmp_9) < 2^32 := Nat.xor_lt_two_pow b_pfhi b_tmp_9
  -- pglo_1: adds {pglo}, {pglo}, {alo}
  word_step -clear s_1, pglo_1 := (pglo + alo_2 + 0) % 2^32, c_1 := (pglo + alo_2 + 0) / 2^32
  have b_pglo_1 : pglo_1 < 2^32 := by rw [e_pglo_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have l_pglo_1 : pglo_1 + 2^32 * c_1 = pglo + alo_2 + 0 := by
    rw [e_pglo_1, e_c_1]; exact Nat.mod_add_div _ _
  have b_c_1 : c_1 ≤ 1 := by
    rw [e_c_1]; exact addc32_carry_le_one pglo alo_2 0 b_pglo b_alo_2 (by decide)
  clear e_pglo_1 e_c_1
  -- pghi_1: adc.w {pghi}, {pghi}, {ahi}
  word_step -clear pghi_1 := (pghi + ahi_2 + c_1) % 2^32
  have b_pghi_1 : pghi_1 < 2^32 := by rw [e_pghi_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_pghi_1, b_k_pghi_1, l_pghi_1⟩ :
      ∃ k, k ≤ 1 ∧ pghi_1 + 2^32 * k = pghi + ahi_2 + c_1 :=
    ⟨(pghi + ahi_2 + c_1) / 2^32, addc32_carry_le_one pghi ahi_2 c_1 b_pghi b_ahi_2 b_c_1,
      by rw [e_pghi_1]; exact Nat.mod_add_div _ _⟩
  clear e_pghi_1
  -- finish: mechanically factored PCs 27 through 31
  word_step -clear finish := batchRoundFinish pglo_1 pghi_1 remaining
  -- BEGIN finish specification
  have hfinishAtOne := batchRoundFinish_spec pglo_1 pghi_1 1 b_pglo_1 b_pghi_1
    (by norm_num [wordMod]) (by decide) (by decide) (batchRoundFinish pglo_1 pghi_1 1) rfl
  have hfinishData :
      finish.pglo < wordMod ∧ finish.pghi < wordMod ∧
        join32 finish.pglo finish.pghi = asr (join32 pglo_1 pghi_1) 1 := by
    rw [e_finish]
    exact ⟨hfinishAtOne.1, hfinishAtOne.2.1, hfinishAtOne.2.2.1⟩
  obtain ⟨b_finish_pglo, b_finish_pghi, hfinishShift⟩ := hfinishData
  -- END finish specification
  -- pglo_2: factored suffix output
  word_step -clear pglo_2 := finish.pglo
  -- BEGIN finish pglo bound
  have b_pglo_2 : pglo_2 < wordMod := by rw [e_pglo_2]; exact b_finish_pglo
  -- END finish pglo bound
  -- pghi_2: factored suffix output
  word_step -clear pghi_2 := finish.pghi
  -- BEGIN finish pghi bound
  have b_pghi_2 : pghi_2 < wordMod := by rw [e_pghi_2]; exact b_finish_pghi
  -- END finish pghi bound
  -- remaining_1: factored suffix output
  word_step -clear remaining_1 := finish.remaining
  -- c_2: factored suffix output
  word_step -clear c_2 := finish.c
  -- z: factored suffix output
  word_step -clear z := finish.z
  -- branch: factored suffix output
  word_step -clear branch := finish.branch
  subst hres
  -- BEGIN batchRound_spec conclusion
  have hDword : (d : Int) =
      if 0 ≤ state.two_delta then state.two_delta else state.two_delta + 2^32 := by
    apply signExtend32_small d state.two_delta hD
    rw [e_d]
    exact ed
  have hFword : (join32 pflo pfhi : Int) = state.f % 2^64 := by
    rw [e_pflo, e_pfhi]
    exact ef
  have hGword : (join32 pglo pghi : Int) = state.g % 2^64 := by
    rw [e_pglo, e_pghi]
    exact eg
  have hparity : pglo % 2 = 0 ↔ state.g % 2 = 0 := by
    have hz' : (if pglo % 2 = 0 then 1 else 0) = if state.g % 2 = 0 then 1 else 0 := by
      rw [e_pglo]
      exact hz
    split at hz' <;> split at hz' <;> simp_all
  rcases Int.emod_two_eq_zero_or_one state.g with hg0 | hg1
  · have hodd : odd_1 = 0 := by
      rw [e_odd_1, e_odd, andw_one_eq_mod_two]
      norm_num [wordMod]
      omega
    have hswap : swap_1 = 0 := by rw [e_swap_1, hodd, andw]; simp
    have halo1 : alo_1 = 0 := by rw [e_alo_1, e_alo, hodd, hswap]; simp [andw]
    have hahi1 : ahi_1 = 0 := by rw [e_ahi_1, e_ahi, hodd, hswap]; simp [andw]
    have htmp1 : tmp_1 = 0 := by rw [e_tmp_1, hswap]; simp [andw]
    have halo2 : alo_2 = 0 := by omega
    have hahi2 : ahi_2 = 0 := by omega
    have hd2 : d_2 = (d + 2) % wordMod := by
      rw [e_d_2, e_tmp_5, hswap, e_d_1]
      simp [andw, wordMod]
    have hpflo : pflo_1 = pflo := by
      rw [e_pflo_1, e_tmp_7, hswap]
      simp only [andw, Nat.and_zero, Nat.xor_zero, Nat.mod_eq_of_lt b_pflo]
    have hpfhi : pfhi_1 = pfhi := by
      rw [e_pfhi_1, e_tmp_9, hswap]
      simp only [andw, Nat.and_zero, Nat.xor_zero, Nat.mod_eq_of_lt b_pfhi]
    have hpf : join32 pflo_1 pfhi_1 = join32 pflo pfhi := by rw [hpflo, hpfhi]
    have hpre : join32 pglo_1 pghi_1 = addw (join32 pglo pghi) 0 := by
      have hjoin := join32_add_of_eq pglo pghi alo_2 ahi_2 0 pglo_1 c_1 pghi_1
        k_pghi_1 l_pglo_1 l_pghi_1 b_pglo_1 b_pghi_1
      rw [halo2, hahi2] at hjoin
      simpa [join32, addw, regMod,
        Nat.mod_eq_of_lt (join32_lt _ _ b_pglo b_pghi)] using hjoin
    have hdNext : signExtend32 d_2 = addw (signExtend32 d) 2 :=
      signExtend32_add_two d d_2 state.two_delta hd2 hD hDword
    have hgNext : join32 pglo_2 pghi_2 = asr (join32 pglo_1 pghi_1) 1 := by
      rw [e_pglo_2, e_pghi_2, hfinishShift]
    obtain ⟨hdOut, hfOut, hgOut, hnextParity⟩ :=
      divstep_words_even state hd hG (signExtend32 d) (join32 pflo pfhi) (join32 pglo pghi)
        ed hFword hGword hg0 0 (signExtend32 d) (join32 pflo_1 pfhi_1)
        (join32 pglo_1 pghi_1) (signExtend32 d_2) (join32 pglo_2 pghi_2)
        rfl rfl hpf hpre hdNext hgNext
    refine ⟨packedStep_round_bounded st hst, hdOut, hfOut, hgOut, ?_⟩
    apply if_congr _ rfl rfl
    rw [← join32_mod_two pglo_2 pghi_2, hgNext, asr_one_mod_two]
    exact hnextParity
  · have hodd : odd_1 = wordMod - 1 := by
      rw [e_odd_1, e_odd, andw_one_eq_mod_two]
      have hpgOdd : pglo % 2 = 1 := by omega
      rw [hpgOdd]
      norm_num [wordMod]
    have hoddMask : odd_1 = 0 ∨ odd_1 = wordMod - 1 := Or.inr hodd
    have halo : alo = pflo := by
      rw [e_alo, andw_mask_eq pflo odd_1 b_pflo hoddMask, hodd]
      norm_num [wordMod]
    have hahi : ahi = pfhi := by
      rw [e_ahi, andw_mask_eq pfhi odd_1 b_pfhi hoddMask, hodd]
      norm_num [wordMod]
    have hgNext : join32 pglo_2 pghi_2 = asr (join32 pglo_1 pghi_1) 1 := by
      rw [e_pglo_2, e_pghi_2, hfinishShift]
    by_cases hdpos : 0 < state.two_delta
    · have hdwordPos : (d : Int) = state.two_delta := by simpa [hdpos.le] using hDword
      have hdSmall : d < 2^31 := by
        rw [abs_of_pos hdpos] at hD
        omega
      have htmp : tmp = wordMod - d := by
        rw [e_tmp]
        norm_num [wordMod]
        omega
      have hswap : swap_1 = wordMod - 1 := by
        have hswap' : swap = wordMod - 1 := by
          rw [e_swap, htmp]
          unfold asr32
          rw [if_neg (by norm_num [wordMod]; omega)]
          norm_num [wordMod]
          omega
        rw [e_swap_1, hswap', hodd]
        simp [andw, wordMod]
      have hswapMask : swap_1 = 0 ∨ swap_1 = wordMod - 1 := Or.inr hswap
      have halo1 : alo_1 = wordMod - 1 - pflo := by
        calc
          alo_1 = eor32 pflo (wordMod - 1) := by rw [e_alo_1, halo, hswap]; rfl
          _ = wordMod - 1 - pflo := eor32_allOnes pflo b_pflo
      have hahi1 : ahi_1 = wordMod - 1 - pfhi := by
        calc
          ahi_1 = eor32 pfhi (wordMod - 1) := by rw [e_ahi_1, hahi, hswap]; rfl
          _ = wordMod - 1 - pfhi := eor32_allOnes pfhi b_pfhi
      have htmp1 : tmp_1 = 1 := by
        rw [e_tmp_1, hswap, andw_one_eq_mod_two]
        norm_num [wordMod]
      have hloNeg : alo_2 + wordMod * c = (wordMod - 1 - pflo) + 1 := by
        simpa [halo1, htmp1] using l_alo_2
      have hhiNeg : ahi_2 + wordMod * k_ahi_2 = (wordMod - 1 - pfhi) + c := by
        simpa [hahi1, e_tmp_2] using l_ahi_2
      have haddend : join32 alo_2 ahi_2 = negw (join32 pflo pfhi) :=
        join32_negate pflo pfhi alo_2 c ahi_2 k_ahi_2 b_pflo b_pfhi hloNeg hhiNeg
          b_alo_2 b_ahi_2
      have htmp6 : tmp_6 = eor32 pflo pglo := e_tmp_6
      have hpfloEq : pflo_1 = eor32 pflo tmp_7 := e_pflo_1
      have hpfloSel := eor32_mask_select_of_eq pflo pglo swap_1 tmp_6 tmp_7 pflo_1
        b_pflo b_pglo hswapMask htmp6 e_tmp_7 hpfloEq
      have hpflo : pflo_1 = pglo := by rw [hpfloSel, hswap]; norm_num [wordMod]
      have htmp8 : tmp_8 = eor32 pfhi pghi := e_tmp_8
      have hpfhiEq : pfhi_1 = eor32 pfhi tmp_9 := e_pfhi_1
      have hpfhiSel := eor32_mask_select_of_eq pfhi pghi swap_1 tmp_8 tmp_9 pfhi_1
        b_pfhi b_pghi hswapMask htmp8 e_tmp_9 hpfhiEq
      have hpfhi : pfhi_1 = pghi := by rw [hpfhiSel, hswap]; norm_num [wordMod]
      have hpf : join32 pflo_1 pfhi_1 = join32 pglo pghi := by rw [hpflo, hpfhi]
      have hpre : join32 pglo_1 pghi_1 =
          addw (join32 pglo pghi) (negw (join32 pflo pfhi)) := by
        have hjoin := join32_add_of_eq pglo pghi alo_2 ahi_2 0 pglo_1 c_1 pghi_1
          k_pghi_1 l_pglo_1 l_pghi_1 b_pglo_1 b_pghi_1
        rw [haddend] at hjoin
        simpa [addw, regMod] using hjoin
      have hdSelect := two_delta_mask_select d d_1 tmp_3 swap_1 tmp_4 tmp_5 d_2
        e_d_1 e_tmp_3 hswapMask e_tmp_4 e_tmp_5 e_d_2
      have hd2 : d_2 = (2 + wordMod - d) % wordMod := by
        rw [hdSelect, hswap, if_neg (by norm_num [wordMod])]
      have hdPositive : 1 ≤ d := by
        by_contra hzero
        have hdZero : d = 0 := by omega
        rw [hdZero] at hdwordPos
        omega
      have hdNext : signExtend32 d_2 = addw (negw (signExtend32 d)) 2 :=
        signExtend32_two_sub d d_2 hdPositive hdSmall hd2
      obtain ⟨hdOut, hfOut, hgOut, hnextParity⟩ :=
        divstep_words_swap state hf hd hG' (signExtend32 d) (join32 pflo pfhi)
          (join32 pglo pghi) (signExtend32_lt d b_d)
          (join32_lt pflo pfhi b_pflo b_pfhi) ed hFword hGword ⟨hdpos, hg1⟩
          (negw (join32 pflo pfhi)) (negw (signExtend32 d))
          (join32 pflo_1 pfhi_1) (join32 pglo_1 pghi_1) (signExtend32 d_2)
          (join32 pglo_2 pghi_2) rfl rfl hpf hpre hdNext hgNext
      refine ⟨packedStep_round_bounded st hst, hdOut, hfOut, hgOut, ?_⟩
      apply if_congr _ rfl rfl
      rw [← join32_mod_two pglo_2 pghi_2, hgNext, asr_one_mod_two]
      exact hnextParity
    · have hdnp : state.two_delta ≤ 0 := le_of_not_gt hdpos
      have hswap : swap_1 = 0 := by
        have hdLarge : 2^31 ≤ d := by
          rw [if_neg (by omega)] at hDword
          have hdAbs : |state.two_delta| < 2^31 := by omega
          have hdBounds := abs_lt.mp hdAbs
          norm_num [wordMod] at hDword
          omega
        have htmp : tmp = wordMod - d := by
          rw [e_tmp]
          norm_num [wordMod]
          omega
        have hswap' : swap = 0 := by
          rw [e_swap, htmp]
          unfold asr32
          rw [if_pos (by norm_num [wordMod]; omega)]
          norm_num [wordMod]
          omega
        rw [e_swap_1, hswap']
        simp [andw]
      have hswapMask : swap_1 = 0 ∨ swap_1 = wordMod - 1 := Or.inl hswap
      have halo1 : alo_1 = pflo := by
        calc
          alo_1 = eor32 pflo 0 := by rw [e_alo_1, halo, hswap]; rfl
          _ = pflo := by simp [eor32, Nat.mod_eq_of_lt b_pflo]
      have hahi1 : ahi_1 = pfhi := by
        calc
          ahi_1 = eor32 pfhi 0 := by rw [e_ahi_1, hahi, hswap]; rfl
          _ = pfhi := by simp [eor32, Nat.mod_eq_of_lt b_pfhi]
      have htmp1 : tmp_1 = 0 := by rw [e_tmp_1, hswap]; simp [andw]
      have halo2 : alo_2 = pflo := by omega
      have hahi2 : ahi_2 = pfhi := by rw [e_tmp_2] at l_ahi_2; omega
      have htmp6 : tmp_6 = eor32 pflo pglo := e_tmp_6
      have hpfloEq : pflo_1 = eor32 pflo tmp_7 := e_pflo_1
      have hpfloSel := eor32_mask_select_of_eq pflo pglo swap_1 tmp_6 tmp_7 pflo_1
        b_pflo b_pglo hswapMask htmp6 e_tmp_7 hpfloEq
      have hpflo : pflo_1 = pflo := by rw [hpfloSel, hswap]; simp
      have htmp8 : tmp_8 = eor32 pfhi pghi := e_tmp_8
      have hpfhiEq : pfhi_1 = eor32 pfhi tmp_9 := e_pfhi_1
      have hpfhiSel := eor32_mask_select_of_eq pfhi pghi swap_1 tmp_8 tmp_9 pfhi_1
        b_pfhi b_pghi hswapMask htmp8 e_tmp_9 hpfhiEq
      have hpfhi : pfhi_1 = pfhi := by rw [hpfhiSel, hswap]; simp
      have hpf : join32 pflo_1 pfhi_1 = join32 pflo pfhi := by rw [hpflo, hpfhi]
      have hpre : join32 pglo_1 pghi_1 = addw (join32 pglo pghi) (join32 pflo pfhi) := by
        have hjoin := join32_add_of_eq pglo pghi alo_2 ahi_2 0 pglo_1 c_1 pghi_1
          k_pghi_1 l_pglo_1 l_pghi_1 b_pglo_1 b_pghi_1
        rw [halo2, hahi2] at hjoin
        simpa [addw, regMod] using hjoin
      have hd2 : d_2 = (d + 2) % wordMod := by
        rw [e_d_2, e_tmp_5, hswap, e_d_1]
        simp [andw, wordMod]
      have hdNext : signExtend32 d_2 = addw (signExtend32 d) 2 :=
        signExtend32_add_two d d_2 state.two_delta hd2 hD hDword
      obtain ⟨hdOut, hfOut, hgOut, hnextParity⟩ :=
        divstep_words_add state hf hd (by omega) hG' (signExtend32 d)
          (join32 pflo pfhi) (join32 pglo pghi) ed hFword hGword hg1 hdnp
          (join32 pflo pfhi) (signExtend32 d) (join32 pflo_1 pfhi_1)
          (join32 pglo_1 pghi_1) (signExtend32 d_2) (join32 pglo_2 pghi_2)
          rfl rfl hpf hpre hdNext hgNext
      refine ⟨packedStep_round_bounded st hst, hdOut, hfOut, hgOut, ?_⟩
      apply if_congr _ rfl rfl
      rw [← join32_mod_two pglo_2 pghi_2, hgNext, asr_one_mod_two]
      exact hnextParity
  -- END batchRound_spec conclusion

-- BEGIN packedStep_widthSpec statement
/-- The generated ARMv7E-M round satisfies the shared signed-width iteration contract. -/
theorem packedStep_widthSpec : packedStep.WidthSpec 31 where
  round st hst s hf hd hD hG hG' hc := by
    exact batchRound_spec st hst s hf hd hD hG hG' hc.1 hc.2.1 hc.2.2.1 hc.2.2.2 _ rfl
  last st hst s hf hd hD hG hG' hc := by
    obtain ⟨hb, hd', hf', hg', -⟩ := batchRound_spec st hst s hf hd hD hG hG'
      hc.1 hc.2.1 hc.2.2.1 hc.2.2.2 _ rfl
    exact ⟨hb, hd', hf', hg'⟩
-- END packedStep_widthSpec statement

end PastaCurves.ARMv7EM
