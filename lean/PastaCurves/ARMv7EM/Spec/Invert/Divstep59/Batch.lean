/-
Copyright (c) 2026 the pasta_curves contributors.
-/
import PastaCurves.ARMv7EM.Spec.Invert.Divstep59.Round
import PastaCurves.Inversion.Divstep59

/-!
# Correctness of one ARMv7E-M packed divstep batch

The generated composition packs the low words, runs the proved ARMv7E-M round for its public
19- or 20-step count, and decodes the transition matrix from the resulting packed words.
-/

namespace PastaCurves.ARMv7EM

open Inversion (State packedStart)

-- BEGIN batchBlock_spec helpers
private theorem negw_word (w : Nat) (W : Int) (hw : (w : Int) = W % 2^64) :
    (negw w : Int) = (-W) % 2^64 := by
  have h : (negw w : Int) = (-(w : Int)) % 2^64 := by
    unfold negw regMod
    have bw : w < 2^64 := by
      have := Int.emod_nonneg W (by norm_num : (2 : Int)^64 ≠ 0)
      have := Int.emod_lt_of_pos W (by norm_num : (0 : Int) < 2^64)
      omega
    push_cast
    omega
  rw [h, hw]
  exact (Int.mod_modEq W _).neg

private theorem subw_word (x y : Nat) (X Y : Int) (hx : (x : Int) = X % 2^64)
    (hy : (y : Int) = Y % 2^64) : (subw x y : Int) = (X - Y) % 2^64 := by
  have bx : x < 2^64 := by
    have := Int.emod_lt_of_pos X (by norm_num : (0 : Int) < 2^64)
    omega
  have byBound : y < 2^64 := by
    have := Int.emod_lt_of_pos Y (by norm_num : (0 : Int) < 2^64)
    omega
  have h : (subw x y : Int) = ((x : Int) - y) % 2^64 := by
    unfold subw regMod
    push_cast
    omega
  rw [h, hx, hy]
  exact (Int.mod_modEq X _).sub (Int.mod_modEq Y _)

private theorem lsl_word (w k : Nat) (W : Int) (hw : (w : Int) = W % 2^64) :
    (lsl w k : Int) = (W * 2^k) % 2^64 := by
  have h : (lsl w k : Int) = ((w : Int) * 2^k) % 2^64 := by
    unfold lsl regMod
    push_cast
    norm_num
  rw [h, hw]
  exact (Int.mod_modEq W _).mul (Int.ModEq.refl (2^k))

private theorem asr21_word (w : Nat) (W : Int) (hw : (w : Int) = W % 2^64)
    (hW : |W| < 2^63) :
    (asr w 21 : Int) = (W / 2^21) % 2^64 := by
  have bw : w < 2^64 := by
    have := Int.emod_nonneg W (by norm_num : (2 : Int)^64 ≠ 0)
    have := Int.emod_lt_of_pos W (by norm_num : (0 : Int) < 2^64)
    omega
  rw [abs_lt] at hW
  by_cases hnonneg : 0 ≤ W
  · have hmod : W % 2^64 = W := Int.emod_eq_of_lt hnonneg (by omega)
    rw [hmod] at hw
    have hwsmall : w < 2^63 := by omega
    unfold asr
    rw [if_pos hwsmall]
    norm_num at hw ⊢
    omega
  · have hneg : W < 0 := lt_of_not_ge hnonneg
    have hmod : W % 2^64 = W + 2^64 := by
      calc
        W % 2^64 = (W + 2^64) % 2^64 := (Int.add_emod_right W (2^64)).symm
        _ = W + 2^64 := Int.emod_eq_of_lt (by omega) (by omega)
    rw [hmod] at hw
    have hwlarge : ¬w < 2^63 := by omega
    unfold asr
    rw [if_neg hwlarge]
    norm_num at hw ⊢
    omega

private theorem asr22_word (w : Nat) (W : Int) (hw : (w : Int) = W % 2^64)
    (hW : |W| < 2^63) :
    (asr w 22 : Int) = (W / 2^22) % 2^64 := by
  have bw : w < 2^64 := by
    have := Int.emod_nonneg W (by norm_num : (2 : Int)^64 ≠ 0)
    have := Int.emod_lt_of_pos W (by norm_num : (0 : Int) < 2^64)
    omega
  rw [abs_lt] at hW
  by_cases hnonneg : 0 ≤ W
  · have hmod : W % 2^64 = W := Int.emod_eq_of_lt hnonneg (by omega)
    rw [hmod] at hw
    have hwsmall : w < 2^63 := by omega
    unfold asr
    rw [if_pos hwsmall]
    norm_num at hw ⊢
    omega
  · have hneg : W < 0 := lt_of_not_ge hnonneg
    have hmod : W % 2^64 = W + 2^64 := by
      calc
        W % 2^64 = (W + 2^64) % 2^64 := (Int.add_emod_right W (2^64)).symm
        _ = W + 2^64 := Int.emod_eq_of_lt (by omega) (by omega)
    rw [hmod] at hw
    have hwlarge : ¬w < 2^63 := by omega
    unfold asr
    rw [if_neg hwlarge]
    norm_num at hw ⊢
    omega

/-- The word implementation of the packed decoder recovers entries in their two's-complement
representations. -/
private theorem unpackWords_spec (k w : Nat) (hk : k = 19 ∨ k = 20) (φ u v : Int)
    (hw : (w : Int) = (φ - 2^(41 - k) * u - 2^(62 - k) * v) % 2^64)
    (hφ : |φ| < 2^20) (hu1 : -(2 : Int)^k < u) (hu2 : u ≤ 2^k)
    (hv1 : -(2 : Int)^k < v) (hv2 : v ≤ 2^k) :
    ((unpackWords k w).1 : Int) = u % 2^64 ∧
      ((unpackWords k w).2 : Int) = v % 2^64 := by
  let W := φ - 2^(41 - k) * u - 2^(62 - k) * v
  let up : Int := (-W + 2^(40 - k)) / 2^(41 - k)
  have hW : W = φ - 2^(41 - k) * u - 2^(62 - k) * v := rfl
  have ht : (negw w : Int) = (-W) % 2^64 := negw_word w W (by simpa [W] using hw)
  have htAdd : (addw (negw w) (2^(40 - k)) : Int) =
      (-W + 2^(40 - k)) % 2^64 :=
    Inversion.addw_word _ _ _ _ ht (by
      rcases hk with rfl | rfl <;> norm_num)
  have hup : up = u + 2^21 * v := by
    rcases hk with rfl | rfl <;>
      simp only [up, W] <;>
      rw [abs_lt] at hφ <;>
      norm_num at * <;>
      omega
  have hfirst : (asr (addw (negw w) (2^(40 - k))) (41 - k) : Int) = up % 2^64 := by
    rcases hk with rfl | rfl
    · apply asr22_word _ (-W + 2^(40 - 19)) htAdd
      rw [hW]
      rw [abs_lt] at hφ ⊢
      norm_num at *
      omega
    · apply asr21_word _ (-W + 2^(40 - 20)) htAdd
      rw [hW]
      rw [abs_lt] at hφ ⊢
      norm_num at *
      omega
  let upWord := asr (addw (negw w) (2^(40 - k))) (41 - k)
  have hvDiv : (up + (2^20 - 1)) / 2^21 = v := by
    rw [hup]
    have hsmall : 0 ≤ u + 2^20 - 1 ∧ u + 2^20 - 1 < 2^21 := by
      rcases hk with rfl | rfl <;> norm_num at * <;> omega
    rw [show u + 2^21 * v + (2^20 - 1) = (u + 2^20 - 1) + v * 2^21 by ring,
      Int.add_mul_ediv_right _ _ (by norm_num), Int.ediv_eq_zero_of_lt hsmall.1 hsmall.2,
      zero_add]
  have hvAdd : (addw upWord (2^20 - 1) : Int) = (up + (2^20 - 1)) % 2^64 :=
    Inversion.addw_word _ _ _ _ (by simpa [upWord] using hfirst) (by norm_num)
  have hvWord : (asr (addw upWord (2^20 - 1)) 21 : Int) = v % 2^64 := by
    rw [← hvDiv]
    apply asr21_word _ (up + (2^20 - 1)) hvAdd
    rw [hup, abs_lt]
    rcases hk with rfl | rfl <;> norm_num at * <;> omega
  let vWord := asr (addw upWord (2^20 - 1)) 21
  have hlsl : (lsl vWord 21 : Int) = (v * 2^21) % 2^64 :=
    lsl_word _ _ _ (by simpa [vWord] using hvWord)
  have huWord : (subw upWord (lsl vWord 21) : Int) = u % 2^64 := by
    rw [← show up - v * 2^21 = u by rw [hup]; ring]
    exact subw_word _ _ up (v * 2^21) (by simpa [upWord] using hfirst) hlsl
  simpa only [unpackWords, negw, upWord, vWord] using And.intro huWord hvWord
-- END batchBlock_spec helpers

-- BEGIN batchBlock_spec statement
/-- One generated batch on the low words of an odd state returns the model's new `two_delta` and
transition matrix. The public count is exactly one of the two counts used by `divstep59Block`. -/
theorem batchBlock_spec (k delta f g : Nat) (hk : k = 19 ∨ k = 20) (TD : Int)
    (ed : (delta : Int) = TD % 2^64) (hf : (f : Int) % 2 = 1) (hd : TD % 2 = 1)
    (hD : |TD| + 2 * k < 2^31) :
    ∀ res, res = batchBlock k delta f g →
      (res.two_delta : Int) = (Inversion.batch k TD f g).1 % 2^64 ∧
        (res.u : Int) = (Inversion.batch k TD f g).2.1.u % 2^64 ∧
        (res.v : Int) = (Inversion.batch k TD f g).2.1.v % 2^64 ∧
        (res.q : Int) = (Inversion.batch k TD f g).2.1.q % 2^64 ∧
        (res.r : Int) = (Inversion.batch k TD f g).2.1.r % 2^64 := by
  intro res hres
  have edInput := ed
-- END batchBlock_spec statement
  -- generated skeleton for `batchBlock`: do not edit between the annotations
  unfold batchBlock at hres
  lift_lets -merge at hres
  -- pf: pack the low word of f
  word_step pf := orrw (andw f 0xfffff) 0xfffffe0000000000
  -- pg: pack the low word of g
  word_step pg := orrw (andw g 0xfffff) 0xc000000000000000
  -- pflo: Rust cast to the low u32
  word_step pflo := pf % wordMod
  -- pfhi: Rust shift and cast to the high u32
  word_step pfhi := pf / wordMod
  -- pglo: Rust cast to the low u32
  word_step pglo := pg % wordMod
  -- pghi: Rust shift and cast to the high u32
  word_step pghi := pg / wordMod
  -- d: Rust cast of two_delta to u32
  word_step d := delta % wordMod
  -- st: initial assembly registers and flags
  word_step st := (⟨pflo, pfhi, pglo, pghi, d, k, 0, 0, 0⟩ : BatchState)
  -- st_1: public-count assembly loop
  word_step st_1 := batchRound^[k] st
  -- pf_1: join the returned packed f words
  word_step pf_1 := join32 st_1.pflo st_1.pfhi
  -- pg_1: join the returned packed g words
  word_step pg_1 := join32 st_1.pglo st_1.pghi
  -- delta: sign-extend the returned u32
  word_step delta := signExtend32 st_1.d
  -- uv: Rust unpack(k, pf)
  word_step uv := unpackWords k pf_1
  -- qr: Rust unpack(k, pg)
  word_step qr := unpackWords k pg_1
  subst hres
  -- BEGIN batchBlock_spec conclusion
  have hk0 : 0 < k := by omega
  have hk20 : k ≤ 20 := by omega
  set n := k - 1 with hn
  have hnSucc : n + 1 = k := by omega
  set s₀ : State := ⟨TD, (f : Int) % 2^20, (g : Int) % 2^20⟩ with hs₀
  set P : State := packedStart s₀ with hP
  have hs₀f : s₀.f % 2 = 1 := by
    rw [hs₀]
    exact Int.emod_emod_of_dvd f (by norm_num) |>.trans hf
  have hs₀fb : 0 ≤ s₀.f ∧ s₀.f < 2^20 := by
    rw [hs₀]
    exact ⟨Int.emod_nonneg _ (by norm_num), Int.emod_lt_of_pos _ (by norm_num)⟩
  have hs₀gb : 0 ≤ s₀.g ∧ s₀.g < 2^20 := by
    rw [hs₀]
    exact ⟨Int.emod_nonneg _ (by norm_num), Int.emod_lt_of_pos _ (by norm_num)⟩
  have hPf : P.f % 2 = 1 := by
    rw [hP]
    show (s₀.f - 2^41) % 2 = 1
    omega
  have hPg : |P.g| < 2^63 := by
    rw [hP]
    show |s₀.g - 2^62| < 2^63
    rw [abs_lt]
    omega
  have hPD : |P.two_delta| + 2 * k < 2^31 := by
    rw [hP, hs₀]
    exact hD
  have bpf : pf < 2^64 := by
    rw [e_pf]
    exact orrw_lt _ _ (Nat.and_lt_two_pow _ (by norm_num)) (by norm_num)
  have bpg : pg < 2^64 := by
    rw [e_pg]
    exact orrw_lt _ _ (Nat.and_lt_two_pow _ (by norm_num)) (by norm_num)
  have bpflo : pflo < wordMod := by rw [e_pflo]; exact Nat.mod_lt _ (by norm_num [wordMod])
  have bpfhi : pfhi < wordMod := by
    rw [e_pfhi]
    norm_num [wordMod] at bpf ⊢
    omega
  have bpglo : pglo < wordMod := by rw [e_pglo]; exact Nat.mod_lt _ (by norm_num [wordMod])
  have bpghi : pghi < wordMod := by
    rw [e_pghi]
    norm_num [wordMod] at bpg ⊢
    omega
  have bd : d < wordMod := by rw [e_d]; exact Nat.mod_lt _ (by norm_num [wordMod])
  have hstBounded : packedStep.Bounded st := by
    rw [e_st]
    exact ⟨bpflo, bpfhi, bpglo, bpghi, bd, by norm_num [wordMod]; omega⟩
  have hDsmall : |TD| + 2 < 2^31 := by omega
  have hd32 : (d : Int) = TD % 2^32 := by
    rw [e_d]
    push_cast
    rw [edInput, Int.emod_emod_of_dvd]
    norm_num [wordMod]
  have hdRep : (signExtend32 d : Int) = TD % 2^64 := by
    have hTDbounds : -(2 : Int)^31 < TD ∧ TD < 2^31 := by
      have hlo := neg_abs_le TD
      have hhi := le_abs_self TD
      omega
    unfold signExtend32
    split
    · have hdSmall : d < 2^31 := by assumption
      have hTDnonneg : 0 ≤ TD := by
        by_contra h
        have hTDneg : TD < 0 := lt_of_not_ge h
        have hmod32 : TD % 2^32 = TD + 2^32 := by
          calc
            TD % 2^32 = (TD + 2^32) % 2^32 := (Int.add_emod_right TD (2^32)).symm
            _ = TD + 2^32 := Int.emod_eq_of_lt (by omega) (by omega)
        rw [hmod32] at hd32
        omega
      rw [Int.emod_eq_of_lt hTDnonneg (by omega)] at hd32
      rw [Int.emod_eq_of_lt hTDnonneg (by omega)]
      exact hd32
    · have hdLarge : ¬d < 2^31 := by assumption
      have hTDneg : TD < 0 := by
        by_contra h
        have hTDnonneg : 0 ≤ TD := le_of_not_gt h
        rw [Int.emod_eq_of_lt hTDnonneg (by omega)] at hd32
        omega
      have hmod32 : TD % 2^32 = TD + 2^32 := by
        calc
          TD % 2^32 = (TD + 2^32) % 2^32 := (Int.add_emod_right TD (2^32)).symm
          _ = TD + 2^32 := Int.emod_eq_of_lt (by omega) (by omega)
      have hmod64 : TD % 2^64 = TD + 2^64 := by
        calc
          TD % 2^64 = (TD + 2^64) % 2^64 := (Int.add_emod_right TD (2^64)).symm
          _ = TD + 2^64 := Int.emod_eq_of_lt (by omega) (by omega)
      rw [hmod32] at hd32
      rw [hmod64]
      push_cast
      norm_num [wordMod]
      omega
  have hstD : (packedStep.two_delta st : Int) = P.two_delta % 2^64 := by
    rw [e_st, packedStep_two_delta, hP, hs₀]
    exact hdRep
  have hpfJoin : join32 pflo pfhi = pf := by
    rw [e_pflo, e_pfhi, join32]
    exact Nat.mod_add_div pf wordMod
  have hpgJoin : join32 pglo pghi = pg := by
    rw [e_pglo, e_pghi, join32]
    exact Nat.mod_add_div pg wordMod
  have hstF : (packedStep.f st : Int) = P.f % 2^64 := by
    rw [e_st, packedStep_f, hpfJoin, e_pf, Inversion.pack_f_word, hP, Inversion.packedStart, hs₀]
  have hstG : (packedStep.g st : Int) = P.g % 2^64 := by
    rw [e_st, packedStep_g, hpgJoin, e_pg, Inversion.pack_g_word, hP, Inversion.packedStart, hs₀]
  have hstZ : packedStep.z st = if P.g % 2 = 0 then 1 else 0 :=
    packedStep_z_of_g_word st P.g hstG
  have hiter : batchRound (batchRound^[n] st) = st_1 := by
    rw [e_st_1, ← hnSucc, Function.iterate_succ_apply']
  have hbudget : |P.two_delta| + 2 * (n + 1) < 2^31 := by
    push_cast
    have hnSuccInt : (n : Int) + 1 = k := by omega
    rw [hnSuccInt]
    exact hPD
  have hbatch := packedStep_widthSpec.batch_words n P hPf
    (by rw [hP, hs₀]; exact hd) hbudget hPg
    (fun j _ => Inversion.divsteps_packedStart_g_abs_lt j s₀ hs₀f hs₀fb hs₀gb)
    st hstBounded hstD hstF hstG hstZ
  obtain ⟨hbounded, hdelta, hpf, hpg⟩ := hbatch
  change packedStep.Bounded (batchRound (batchRound^[n] st)) at hbounded
  change (signExtend32 (batchRound (batchRound^[n] st)).d : Int) = _ at hdelta
  change (join32 (batchRound (batchRound^[n] st)).pflo
    (batchRound (batchRound^[n] st)).pfhi : Int) = _ at hpf
  change (join32 (batchRound (batchRound^[n] st)).pglo
    (batchRound (batchRound^[n] st)).pghi : Int) = _ at hpg
  rw [hiter] at hbounded hdelta hpf hpg
  simp only [hnSucc] at hdelta hpf hpg
  have hpk := Inversion.divsteps_packedStart k hk20 s₀ hs₀f
  obtain ⟨hφ, hγ⟩ := Inversion.divsteps_abs_le k s₀ (2^20 - 1)
    (by rw [abs_le]; omega) (by rw [abs_le]; omega)
  have hφlt : |(Inversion.divsteps k s₀).f| < 2^20 := by omega
  have hγlt : |(Inversion.divsteps k s₀).g| < 2^20 := by omega
  obtain ⟨hU, hV, hQ, hR⟩ := Inversion.M_entry_range k s₀
  have hpfWord : (pf_1 : Int) = (Inversion.divsteps k P).f % 2^64 := by
    rw [e_pf_1]
    simpa [packedStep_f] using hpf
  have hpgWord : (pg_1 : Int) = (Inversion.divsteps k P).g % 2^64 := by
    rw [e_pg_1]
    simpa [packedStep_g] using hpg
  have huv := unpackWords_spec k pf_1 hk (Inversion.divsteps k s₀).f
    (Inversion.M k s₀).u (Inversion.M k s₀).v
    (by rw [hpfWord, hP, hpk]) hφlt hU.1 hU.2 hV.1 hV.2
  have hqr := unpackWords_spec k pg_1 hk (Inversion.divsteps k s₀).g
    (Inversion.M k s₀).q (Inversion.M k s₀).r
    (by rw [hpgWord, hP, hpk]) hγlt hQ.1 hQ.2 hR.1 hR.2
  have hmodelD : (Inversion.batch k TD f g).1 = (Inversion.divsteps k s₀).two_delta := by
    simp only [Inversion.batch, Inversion.packedDivsteps]
    rw [show (⟨TD, (Inversion.pack f g).1, (Inversion.pack f g).2⟩ : State) = P by rw [hP, hs₀]; rfl,
      hpk]
  have hmodelM : (Inversion.batch k TD f g).2.1 = Inversion.M k s₀ := by
    simp only [Inversion.batch, Inversion.unpackMat, Inversion.packedDivsteps]
    rw [show (⟨TD, (Inversion.pack f g).1, (Inversion.pack f g).2⟩ : State) = P by rw [hP, hs₀]; rfl,
      hpk, Inversion.unpack_spec k hk20 _ _ _ hφlt hU.1 hU.2,
      Inversion.unpack_spec k hk20 _ _ _ hγlt hQ.1 hQ.2]
  rw [e_delta]
  have hdelta' : (signExtend32 st_1.d : Int) = (Inversion.divsteps k s₀).two_delta % 2^64 := by
    have hdelta' := hdelta
    rw [hP, hpk] at hdelta'
    exact hdelta'
  rw [hmodelD, hmodelM]
  exact ⟨hdelta', by simpa [e_uv] using huv.1, by simpa [e_uv] using huv.2,
    by simpa [e_qr] using hqr.1, by simpa [e_qr] using hqr.2⟩
  -- END batchBlock_spec conclusion

end PastaCurves.ARMv7EM
