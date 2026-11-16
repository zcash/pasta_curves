import PastaCurves.ARMv7EM.Spec.Invert.Divstep59.Block
import PastaCurves.ARMv7EM.Backend
import PastaCurves.Portable.Divstep59
import PastaCurves.Portable.Entry

/-!
# ARMv7E-M divstep composition correspondence

This file connects the Aeneas translation of the ARMv7E-M `divstep59` Rust composition to the
mechanically transcribed `batchBlock` and `divstep59Block` models.
-/

namespace PastaCurves.ARMv7EM.Compositions.Invert.Divstep59

open _root_.Aeneas _root_.Aeneas.Std Result
open PastaCurves

private def mechanicalStart (k delta f g : Nat) : ARMv7EM.BatchState :=
  let pf := orrw (andw f 0xfffff) 0xfffffe0000000000
  let pg := orrw (andw g 0xfffff) 0xc000000000000000
  ⟨pf % ARMv7EM.wordMod, pf / ARMv7EM.wordMod, pg % ARMv7EM.wordMod,
    pg / ARMv7EM.wordMod, delta % ARMv7EM.wordMod, k, 0, 0, 0⟩

private theorem mechanicalStart_bounded (k delta f g : Nat) :
    (mechanicalStart k delta f g).Bounded := by
  let pf := orrw (andw f 0xfffff) 0xfffffe0000000000
  let pg := orrw (andw g 0xfffff) 0xc000000000000000
  have hpf : pf < 2^64 :=
    orrw_lt _ _ (Nat.and_lt_two_pow _ (by norm_num)) (by norm_num)
  have hpg : pg < 2^64 :=
    orrw_lt _ _ (Nat.and_lt_two_pow _ (by norm_num)) (by norm_num)
  have hword : 0 < ARMv7EM.wordMod := by norm_num [ARMv7EM.wordMod]
  have hpfhi : pf / ARMv7EM.wordMod < ARMv7EM.wordMod := by
    apply (Nat.div_lt_iff_lt_mul hword).2
    norm_num [ARMv7EM.wordMod] at hpf ⊢
    exact hpf
  have hpghi : pg / ARMv7EM.wordMod < ARMv7EM.wordMod := by
    apply (Nat.div_lt_iff_lt_mul hword).2
    norm_num [ARMv7EM.wordMod] at hpg ⊢
    exact hpg
  show ARMv7EM.BatchState.Bounded (mechanicalStart k delta f g)
  refine { pflo := ?_, pfhi := ?_, pglo := ?_, pghi := ?_, d := ?_ }
  · exact Nat.mod_lt _ hword
  · exact hpfhi
  · exact Nat.mod_lt _ hword
  · exact hpghi
  · exact Nat.mod_lt _ hword

private theorem batchRound_iterate_bounded (k : Nat) (st : ARMv7EM.BatchState)
    (hst : st.Bounded) : ((ARMv7EM.batchRound^[k]) st).Bounded := by
  induction k generalizing st with
  | zero => exact hst
  | succ k ih =>
      rw [Function.iterate_succ_apply]
      exact ih _ (ARMv7EM.batchRound_bounded st hst)

private def mechanicalFinish (k : Nat) (st : ARMv7EM.BatchState) :
    _root_.PastaCurves.Divstep59Result :=
  let uv := ARMv7EM.unpackWords k (ARMv7EM.join32 st.pflo st.pfhi)
  let qr := ARMv7EM.unpackWords k (ARMv7EM.join32 st.pglo st.pghi)
  ⟨ARMv7EM.signExtend32 st.d, uv.1, uv.2, qr.1, qr.2⟩

private theorem mechanicalFinish_bounded (k : Nat) (st : ARMv7EM.BatchState)
    (hst : st.Bounded) :
    let res := mechanicalFinish k st
    res.two_delta < 2^64 ∧ res.u < 2^64 ∧ res.v < 2^64 ∧ res.q < 2^64 ∧ res.r < 2^64 := by
  have hpf := ARMv7EM.join32_lt st.pflo st.pfhi hst.pflo hst.pfhi
  have hpg := ARMv7EM.join32_lt st.pglo st.pghi hst.pglo hst.pghi
  have hupf := asr_lt (addw ((2^64 - ARMv7EM.join32 st.pflo st.pfhi) % 2^64)
    (2^(40 - k))) (41 - k) (addw_lt _ _)
  have hupg := asr_lt (addw ((2^64 - ARMv7EM.join32 st.pglo st.pghi) % 2^64)
    (2^(40 - k))) (41 - k) (addw_lt _ _)
  have hvf := asr_lt (addw (asr (addw ((2^64 - ARMv7EM.join32 st.pflo st.pfhi) % 2^64)
    (2^(40 - k))) (41 - k)) (2^20 - 1)) 21 (addw_lt _ _)
  have hvg := asr_lt (addw (asr (addw ((2^64 - ARMv7EM.join32 st.pglo st.pghi) % 2^64)
    (2^(40 - k))) (41 - k)) (2^20 - 1)) 21 (addw_lt _ _)
  simp only [mechanicalFinish, ARMv7EM.unpackWords]
  exact ⟨ARMv7EM.signExtend32_lt st.d hst.d, subw_lt _ _, hvf, subw_lt _ _, hvg⟩

private theorem batchBlock_bounded (k delta f g : Nat) :
    let res := ARMv7EM.batchBlock k delta f g
    res.two_delta < 2^64 ∧ res.u < 2^64 ∧ res.v < 2^64 ∧ res.q < 2^64 ∧ res.r < 2^64 := by
  change let res := mechanicalFinish k ((ARMv7EM.batchRound^[k])
      (mechanicalStart k delta f g))
    res.two_delta < 2^64 ∧ res.u < 2^64 ∧ res.v < 2^64 ∧ res.q < 2^64 ∧ res.r < 2^64
  exact mechanicalFinish_bounded _ _
    (batchRound_iterate_bounded _ _ (mechanicalStart_bounded k delta f g))

private def resultArray (res : _root_.PastaCurves.Divstep59Result) :
    Std.Array Std.U64 5#usize :=
  Std.Array.make 5#usize [word res.two_delta, word res.u, word res.v, word res.q, word res.r]

/-- The Aeneas external for one ARM batch returns the mechanically transcribed batch result. -/
theorem batch_eq_batchBlock (k : Std.U32) (twoDelta f g : Std.U64) :
    asm.armv7em.batch k twoDelta f g =
      .ok (resultArray (ARMv7EM.batchBlock k.val twoDelta.val f.val g.val)) := by
  rfl

@[step]
private theorem batch_spec (k : Std.U32) (twoDelta f g : Std.U64) :
    asm.armv7em.batch k twoDelta f g ⦃ res =>
      Portable.divstep59OfArray res = ARMv7EM.batchBlock k.val twoDelta.val f.val g.val ⦄ := by
  rw [batch_eq_batchBlock]
  obtain ⟨h0, h1, h2, h3, h4⟩ := batchBlock_bounded k.val twoDelta.val f.val g.val
  simp [resultArray, Portable.divstep59OfArray, word_val h0, word_val h1, word_val h2,
    word_val h3, word_val h4]

@[step]
private theorem next_low_eq (k : Std.U32) (hk : k.val < 64) (a b f g : Std.U64) :
    pasta_curves.inversion.portable.next_low k a b f g ⦃ res =>
      res.val = ARMv7EM.nextLowWords k.val a.val b.val f.val g.val ⦄ := by
  unfold pasta_curves.inversion.portable.next_low
  step*
  rw [res_post, i2_post, i_post, i1_post]
  simp [ARMv7EM.nextLowWords, lsr, addw, core.num.U64.wrapping_mul_val_eq,
    core.num.U64.wrapping_add_val_eq, Nat.shiftRight_eq_div_pow, U64.size, U64.numBits]

@[step]
private theorem mat_mul_eq (m n : Std.Array Std.U64 4#usize) :
    pasta_curves.inversion.portable.mat_mul m n ⦃ res =>
      (res[0].val, res[1].val, res[2].val, res[3].val) =
        ARMv7EM.matMulWords
          (m[0].val, m[1].val, m[2].val, m[3].val)
          (n[0].val, n[1].val, n[2].val, n[3].val) ⦄ := by
  unfold pasta_curves.inversion.portable.mat_mul
  step*
  simp [ARMv7EM.matMulWords, addw, core.num.U64.wrapping_mul_val_eq,
    core.num.U64.wrapping_add_val_eq, U64.size, U64.numBits, *]

/-- The generated Rust `divstep59` composition returns the words of the mechanically transcribed
ARMv7E-M composition. -/
theorem divstep59_eq_divstep59Block (twoDelta f0 g0 : Std.U64) :
    pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.divstep59
      twoDelta f0 g0 ⦃ res =>
        Portable.divstep59OfArray res = ARMv7EM.divstep59Block twoDelta.val f0.val g0.val ⦄ := by
  unfold pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.divstep59
  unfold pasta_curves.asm.armv7em.divstep59
  step with batch_spec as ⟨a, ha⟩
  iterate 5 step
  iterate 2 step with next_low_eq 20#u32 (by decide)
  step with batch_spec as ⟨a1, ha1⟩
  iterate 5 step
  iterate 2 step with next_low_eq 20#u32 (by decide)
  step with batch_spec as ⟨a2, ha2⟩
  iterate 5 step
  iterate 2 step with mat_mul_eq
  step*
  have ha0 := congrArg Divstep59Result.two_delta ha
  have ha1u := congrArg Divstep59Result.u ha
  have ha1v := congrArg Divstep59Result.v ha
  have ha1q := congrArg Divstep59Result.q ha
  have ha1r := congrArg Divstep59Result.r ha
  have hb0 := congrArg Divstep59Result.two_delta ha1
  have hb1u := congrArg Divstep59Result.u ha1
  have hb1v := congrArg Divstep59Result.v ha1
  have hb1q := congrArg Divstep59Result.q ha1
  have hb1r := congrArg Divstep59Result.r ha1
  have hc0 := congrArg Divstep59Result.two_delta ha2
  have hc1u := congrArg Divstep59Result.u ha2
  have hc1v := congrArg Divstep59Result.v ha2
  have hc1q := congrArg Divstep59Result.q ha2
  have hc1r := congrArg Divstep59Result.r ha2
  simp only [Portable.divstep59OfArray] at ha0 ha1u ha1v ha1q ha1r hb0 hb1u hb1v hb1q hb1r hc0 hc1u hc1v hc1q hc1r ⊢
  simp only [Std.Array.make] at a3_post matrix_post ⊢
  have hm0 := congrArg (fun x => x.1) a3_post
  have hm1 := congrArg (fun x => x.2.1) a3_post
  have hm2 := congrArg (fun x => x.2.2.1) a3_post
  have hm3 := congrArg (fun x => x.2.2.2) a3_post
  have hn0 := congrArg (fun x => x.1) matrix_post
  have hn1 := congrArg (fun x => x.2.1) matrix_post
  have hn2 := congrArg (fun x => x.2.2.1) matrix_post
  have hn3 := congrArg (fun x => x.2.2.2) matrix_post
  simp only [ARMv7EM.divstep59Block]
  grind

end PastaCurves.ARMv7EM.Compositions.Invert.Divstep59
