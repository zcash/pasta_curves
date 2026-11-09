/-
Copyright Amazon.com, Inc. or its affiliates (the block, adapted from s2n-bignum).
Copyright (c) 2026 the pasta_curves contributors (the transcription and the proofs).
-/
import PastaCurves.X86_64.Spec.Divstep
import PastaCurves.Inversion.Divstep59
import PastaCurves.Inversion.PackedWords

/-!
# Correctness of the inversion's 59-step block

The statement is the AArch64 block's, and so is the proof's plan: the block packs the low words of
`f` and `g` with the identity matrix, runs 20 packed divsteps, decodes the matrix from the upper
bits, forms the next low words, and repeats for 20 and 19 steps, multiplying the three matrices
together. Each batch is the shared `batch_words` at the x86-64 step, which starts from the words with
the first step's decision applied (`decide_words`) and ends with `finishStep`. The x86-64 block
differs from the AArch64 one only in its words:

* the decoder takes `-u` as bits 21 to 41 (bits 22 to 42 for 19 steps) of `w + 2^20` with a shift
  left and an arithmetic shift right (`decode20`, `decode19`), where AArch64 uses `sbfx`;
* the packed `g` gets its top bits from two `bts`, which together are AArch64's one `orr`;
* the products of the third matrix with the first two are formed with `imul` and `add`, and their
  sign is restored with `neg`.
-/

set_option exponentiation.threshold 400

namespace PastaCurves.X86_64

open Inversion (State divstep divsteps M packedStart mul_word addw_word pack_f_word pack_g_word
  asr20_word asr20_word44 decode_v20 decode_v19 andw_one_eq_zero_iff PackedStep)

-- BEGIN divstep59Block_spec lemmas
/-! ## Words -/

/-- An `add` whose carry is dropped is `addw`: the skeleton keeps the add's linear equation, and the
products and sums of the block are taken modulo `2^64`. -/
theorem lin_addw (r a b c : ℕ) (hr : r < 2^64) (h : r + 2^64 * c = a + b + 0) : r = addw a b := by
  unfold addw regMod; omega

/-- `neg` carries `-X`: the block restores the sign of the third matrix's products this way. -/
theorem neg_word (x : ℕ) (X : ℤ) (hx : (x : ℤ) = X % 2^64) :
    ((neg x).1 : ℤ) = (-X) % 2^64 := by
  have hx64 : x < 2^64 := by omega
  have h : ((neg x).1 : ℤ) = (-(x : ℤ)) % 2^64 := by
    simp only [neg, sbb, regMod]; omega
  rw [h, hx]
  exact (Int.mod_modEq X _).neg

/-- The packed `g` word, its top bits set by two `bts`: the shared packing. -/
theorem pack_g_bts (g : ℕ) :
    ((orrw (orrw (andw g 1048575) 9223372036854775808) 4611686018427387904 : ℕ) : ℤ)
      = (g % 2^20 - 2^62) % 2^64 := by
  have h : orrw (orrw (andw g 1048575) 9223372036854775808) 4611686018427387904
      = orrw (andw g 0xfffff) 0xc000000000000000 := by
    unfold orrw; rw [Nat.or_assoc]; rfl
  rw [h]; exact pack_g_word g

/-! ## The decoder -/

/-- The decoder of a batch's packed word, for `k = 20`: bits 21 to 41 of `w + 2^20`, by `shl` 22 and
`sar` 43, give `-u`, and `sar` by 42 of `w + 2^20 + 2^41` gives `-v`, as words. The offsets round
the `|φ| < 2^20` disturbance away. The `v` half is the shared `decode_v20`. -/
theorem decode20 (w : ℕ) (W φ u v : ℤ) (hw : (w : ℤ) = W % 2^64)
    (hW : W = φ - 2^21 * u - 2^42 * v) (hφ : |φ| < 2^20)
    (hu : -(2 : ℤ)^20 < u ∧ u ≤ 2^20) (hv : -(2 : ℤ)^20 < v ∧ v ≤ 2^20) :
    ((asr (addw w 1048576 * 2^22 % 2^64) 43 : ℕ) : ℤ) = (-u) % 2^64 ∧
      ((asr (addw w 2199024304128) 42 : ℕ) : ℤ) = (-v) % 2^64 := by
  rw [show (2199024304128 : ℕ) = 2^20 + 2^41 by norm_num]
  refine ⟨?_, decode_v20 w W φ u v hw hW hφ hu hv⟩
  rw [abs_lt] at hφ
  have hw64 : w < 2^64 := by omega
  have ha1 : addw w 1048576 = (w + 2^20) % 2^64 := rfl
  unfold asr
  rw [ha1]
  norm_num
  split_ifs <;> omega

/-- The decoder for `k = 19`: bits 22 to 42, by `shl` 21 and `sar` 43, and `sar` by 43. -/
theorem decode19 (w : ℕ) (W φ u v : ℤ) (hw : (w : ℤ) = W % 2^64)
    (hW : W = φ - 2^22 * u - 2^43 * v) (hφ : |φ| < 2^20)
    (hu : -(2 : ℤ)^19 < u ∧ u ≤ 2^19) (hv : -(2 : ℤ)^19 < v ∧ v ≤ 2^19) :
    ((asr (addw w 1048576 * 2^21 % 2^64) 43 : ℕ) : ℤ) = (-u) % 2^64 ∧
      ((asr (addw w 2199024304128) 43 : ℕ) : ℤ) = (-v) % 2^64 := by
  rw [show (2199024304128 : ℕ) = 2^20 + 2^41 by norm_num]
  refine ⟨?_, decode_v19 w W φ u v hw hW hφ hu hv⟩
  rw [abs_lt] at hφ
  have hw64 : w < 2^64 := by omega
  have ha1 : addw w 1048576 = (w + 2^20) % 2^64 := rfl
  unfold asr
  rw [ha1]
  norm_num
  split_ifs <;> omega

/-! ## Iterating the step -/

/-- A step leaves its words below `2^64`, whatever they carry. The skeleton needs the bound on a
batch's `two_delta` before the batch's semantics are known. -/
theorem divstepRound_bounded (st : DivstepState) (hst : st.Bounded) : (divstepRound st).Bounded := by
  obtain ⟨-, h2, -, -, h5⟩ := hst
  exact ⟨eorw_lt _ _ (addc_value_lt _ _ _) (cmovz_lt _ _ _ (cmovs_lt _ _ _ (by decide) h5) h5),
    cmovb_lt _ _ _ h2 (asr_lt _ _ (addw_lt _ _)), asr_lt _ _ (addw_lt _ _),
    eorw_lt _ _ (cmovz_lt _ _ _ h2 h5) (cmovz_lt _ _ _ (cmovs_lt _ _ _ (by decide) h5) h5), h5⟩

/-- Steps leave their words below `2^64`. -/
theorem divstepRound_iterate_bounded (n : ℕ) (st : DivstepState) (hst : st.Bounded) :
    (divstepRound^[n] st).Bounded := by
  induction n with
  | zero => exact hst
  | succ n ih => rw [Function.iterate_succ_apply']; exact divstepRound_bounded _ ih

/-- A batch of `n + 1` steps: `n` rounds and the finishing step, the shared `batch_words` at the
x86-64 step, from words with the first step's decision applied. -/
theorem batch_words (n : ℕ) (P : State) (hf : P.f % 2 = 1) (hd : P.two_delta % 2 = 1)
    (hD : |P.two_delta| + 2 * (n + 1) < 2^62) (hg0 : |P.g| < 2^63)
    (hg : ∀ j, j < n + 1 → |(divsteps (j + 1) P).g| < 2^62)
    (st : DivstepState) (hst : st.Bounded) (hc : Decided st P) :
    (finishStep (divstepRound^[n] st)).Bounded ∧
      ((finishStep (divstepRound^[n] st)).two_delta : ℤ) = (divsteps (n + 1) P).two_delta % 2^64 ∧
      ((finishStep (divstepRound^[n] st)).f : ℤ) = (divsteps (n + 1) P).f % 2^64 ∧
      ((finishStep (divstepRound^[n] st)).g : ℤ) = (divsteps (n + 1) P).g % 2^64 :=
  packedStep_spec.batch_words n P hf hd hD hg0 hg st hst hc

/-- A batch's first decision, from its packed words: `decide_words` with the sign of `two_delta`
tested by `test td, td` and the parity of `g` by `test pg, 1`. -/
theorem decide_batch (P : State) (hf : P.f % 2 = 1) (hd : P.two_delta % 2 = 1)
    (hD : |P.two_delta| < 2^63) (td pf pg : ℕ) (b_td : td < 2^64) (b_pf : pf < 2^64)
    (hd' : (td : ℤ) = P.two_delta % 2^64) (hf' : (pf : ℤ) = P.f % 2^64)
    (hg' : (pg : ℤ) = P.g % 2^64) :
    Decided ⟨eorw td (cmovz (andw pg 1) (cmovs (andw td td) 18446744073709551614 0) 0),
      cmovb (bt (cmovz (andw pg 1) (cmovs (andw td td) 18446744073709551614 0) 0) 63) pf pg, pg,
      eorw (cmovz (andw pg 1) pf 0) (cmovz (andw pg 1) (cmovs (andw td td) 18446744073709551614 0) 0),
      0⟩ P := by
  have hs : andw td td = td := Nat.and_self td
  rw [hs]
  refine decide_words P hf hd hD td pf pg b_td b_pf hd' hf' hg' (andw pg 1) ?_
  rw [andw_one_eq_zero_iff]
  constructor <;> intro h <;> omega

-- END divstep59Block_spec lemmas

-- BEGIN divstep59Block_spec statement
-- The budget is the sum of the skeleton's bindings and of the conclusion's many small steps,
-- not any one of them.
set_option maxHeartbeats 800000 in
/-- The 59-step block on the words of a state `σ` with odd `f` and `two_delta`, and
`|two_delta| < 2^61`, which the ten rounds of the inversion keep: its `two_delta` word carries the
`two_delta` after 59 steps, and its matrix words the 59-step matrix `M 59 σ`, in two's complement.
-/
theorem divstep59Block_spec (two_delta f0 g0 : ℕ) (σ : State) (hsf : σ.f % 2 = 1) (hsd : σ.two_delta % 2 = 1)
    (hsD : |σ.two_delta| < 2^61) (ed : (two_delta : ℤ) = σ.two_delta % 2^64) (ef0 : (f0 : ℤ) = σ.f % 2^64)
    (eg0 : (g0 : ℤ) = σ.g % 2^64) :
    ∀ res, res = divstep59Block two_delta f0 g0 →
      (res.two_delta : ℤ) = (divsteps 59 σ).two_delta % 2^64 ∧
        (res.u : ℤ) = (M 59 σ).u % 2^64 ∧ (res.v : ℤ) = (M 59 σ).v % 2^64 ∧
        (res.q : ℤ) = (M 59 σ).q % 2^64 ∧ (res.r : ℤ) = (M 59 σ).r % 2^64 := by
  intro res hres
  have htwo_delta : two_delta < 2^64 := by omega
  have hf0 : f0 < 2^64 := by omega
  have hg0 : g0 < 2^64 := by omega
-- END divstep59Block_spec statement
  -- generated skeleton for `divstep59Block`: do not edit between the annotations
  unfold divstep59Block at hres
  lift_lets -merge at hres
  -- td: scalar input
  word_step td := two_delta using htwo_delta
  -- x: scalar input
  word_step x := f0 using hf0
  -- y: scalar input
  word_step y := g0 using hg0
  -- pg: movabs {pg}, 0xfffffe0000000000
  word_step pg := 18446741874686296064 using (by decide)
  -- pf: mov {pf}, {x}
  word_step pf := x using b_x
  -- pf_1: and {pf}, 0xfffff
  word_step pf_1 := andw pf 1048575 using andw_lt pf 1048575 b_pf (by decide)
  -- pf_2: or {pf}, {pg}
  word_step pf_2 := orrw pf_1 pg using orrw_lt pf_1 pg b_pf_1 b_pg
  -- pg_1: mov {pg}, {y}
  word_step pg_1 := y using b_y
  -- pg_2: and {pg}, 0xfffff
  word_step pg_2 := andw pg_1 1048575 using andw_lt pg_1 1048575 b_pg_1 (by decide)
  -- pg_3: bts {pg}, 63
  word_step pg_3 := orrw pg_2 9223372036854775808 using orrw_lt pg_2 9223372036854775808 b_pg_2 (by decide)
  -- pg_4: bts {pg}, 62
  word_step pg_4 := orrw pg_3 4611686018427387904 using orrw_lt pg_3 4611686018427387904 b_pg_3 (by decide)
  -- z: xor {z:e}, {z:e}
  word_step z := 0 using (by decide), cf := 0, ofl := 0
  have b_cf : cf ≤ 1 := by rw [e_cf]; decide
  have b_ofl : ofl ≤ 1 := by rw [e_ofl]; decide
  -- t: mov {t}, {pf}
  word_step t := pf_2 using b_pf_2
  -- k: mov {k}, -2
  word_step k := 18446744073709551614 using (by decide)
  -- zs: test {td}, {td}
  word_step zs := andw td td using andw_lt td td b_td b_td
  -- k_1: cmovs {k}, {z}
  word_step k_1 := cmovs zs k z using cmovs_lt zs k z b_k b_z
  -- zs_1: test {pg}, 1
  word_step zs_1 := andw pg_4 1 using andw_lt pg_4 1 b_pg_4 (by decide)
  -- k_2: cmove {k}, {z}
  word_step k_2 := cmovz zs_1 k_1 z using cmovz_lt zs_1 k_1 z b_k_1 b_z
  -- t_1: cmove {t}, {z}
  word_step t_1 := cmovz zs_1 t z using cmovz_lt zs_1 t z b_t b_z
  -- t_2: xor {t}, {k}
  word_step t_2 := eorw t_1 k_2 using eorw_lt t_1 k_2 b_t_1 b_k_2
  -- td_1: xor {td}, {k}
  word_step td_1 := eorw td k_2 using eorw_lt td k_2 b_td b_k_2
  -- cf_1: bt {k}, 63
  word_step cf_1 := bt k_2 63
  have b_cf_1 : cf_1 ≤ 1 := by
    rw [e_cf_1]; exact Nat.le_of_lt_succ (Nat.mod_lt _ (by decide))
  -- pf_3: cmovb {pf}, {pg}
  word_step pf_3 := cmovb cf_1 pf_2 pg_4 using cmovb_lt cf_1 pf_2 pg_4 b_pf_2 b_pg_4
  -- step20: divstep!(), invocations 2 to 20
  word_step step20 := divstepRound^[19] ⟨td_1, pf_3, pg_4, t_2, z⟩
  -- td_2: divstep!(), invocations 2 to 20 output
  word_step td_2 := step20.two_delta
  -- pf_4: divstep!(), invocations 2 to 20 output
  word_step pf_4 := step20.f
  -- pg_5: divstep!(), invocations 2 to 20 output
  word_step pg_5 := step20.g
  -- t_3: divstep!(), invocations 2 to 20 output
  word_step t_3 := step20.t
  -- z_1: divstep!(), invocations 2 to 20 output
  word_step z_1 := step20.z
  -- BEGIN batch 1 bound
  have b_td_2 : td_2 < 2^64 := by
    rw [e_td_2, e_step20]
    exact (divstepRound_iterate_bounded 19 _ ⟨b_td_1, b_pf_3, b_pg_4, b_t_2, b_z⟩).1
  -- END batch 1 bound
  -- td_3: add {td}, 2
  word_step s, td_3 := (addc td_2 2 0).1 using addc_value_lt td_2 2 0, cf_2 := (addc td_2 2 0).2
  have l_td_3 : td_3 + 2^64 * cf_2 = td_2 + 2 + 0 := by
    rw [e_td_3, e_cf_2]; exact addc_lin td_2 2 0
  have b_cf_2 : cf_2 ≤ 1 := by rw [e_cf_2]; exact addc_carry_le_one td_2 2 0 b_td_2 (by decide) (by decide)
  clear e_td_3 e_cf_2
  -- pg_6: lea {pg}, [{pg} + {t}]
  word_step pg_6 := addw pg_5 t_3 using addw_lt pg_5 t_3
  -- pg_7: sar {pg}, 1
  word_step pg_7 := asr pg_6 1 using asr_lt pg_6 1 b_pg_6
  -- a0: lea {a0}, [{pf} + 0x100000]
  word_step a0 := addw pf_4 1048576 using addw_lt pf_4 1048576
  -- a0_1: shl {a0}, 22
  word_step a0_1 := a0 * 2^22 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- a0_2: sar {a0}, 43
  word_step a0_2 := asr a0_1 43 using asr_lt a0_1 43 b_a0_1
  -- h: movabs {h}, 0x20000100000
  word_step h := 2199024304128 using (by decide)
  -- a1: lea {a1}, [{pf} + {h}]
  word_step a1 := addw pf_4 h using addw_lt pf_4 h
  -- a1_1: sar {a1}, 42
  word_step a1_1 := asr a1 42 using asr_lt a1 42 b_a1
  -- a2: lea {a2}, [{pg} + 0x100000]
  word_step a2 := addw pg_7 1048576 using addw_lt pg_7 1048576
  -- a2_1: shl {a2}, 22
  word_step a2_1 := a2 * 2^22 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- a2_2: sar {a2}, 43
  word_step a2_2 := asr a2_1 43 using asr_lt a2_1 43 b_a2_1
  -- a3: lea {a3}, [{pg} + {h}]
  word_step a3 := addw pg_7 h using addw_lt pg_7 h
  -- a3_1: sar {a3}, 42
  word_step a3_1 := asr a3 42 using asr_lt a3 42 b_a3
  -- pf_5: mov {pf}, {a0}
  word_step pf_5 := a0_2 using b_a0_2
  -- pf_6: imul {pf}, {x}
  word_step pf_6 := pf_5 * x % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- pg_8: mov {pg}, {a1}
  word_step pg_8 := a1_1 using b_a1_1
  -- pg_9: imul {pg}, {y}
  word_step pg_9 := pg_8 * y % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- pf_7: add {pf}, {pg}
  word_step s_1, pf_7 := (addc pf_6 pg_9 0).1 using addc_value_lt pf_6 pg_9 0,
      cf_3 := (addc pf_6 pg_9 0).2
  have l_pf_7 : pf_7 + 2^64 * cf_3 = pf_6 + pg_9 + 0 := by
    rw [e_pf_7, e_cf_3]; exact addc_lin pf_6 pg_9 0
  have b_cf_3 : cf_3 ≤ 1 := by rw [e_cf_3]; exact addc_carry_le_one pf_6 pg_9 0 b_pf_6 b_pg_9 (by decide)
  clear e_pf_7 e_cf_3
  -- x_1: imul {x}, {a2}
  word_step x_1 := x * a2_2 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- y_1: imul {y}, {a3}
  word_step y_1 := y * a3_1 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- y_2: add {y}, {x}
  word_step s_2, y_2 := (addc y_1 x_1 0).1 using addc_value_lt y_1 x_1 0, cf_4 := (addc y_1 x_1 0).2
  have l_y_2 : y_2 + 2^64 * cf_4 = y_1 + x_1 + 0 := by
    rw [e_y_2, e_cf_4]; exact addc_lin y_1 x_1 0
  have b_cf_4 : cf_4 ≤ 1 := by rw [e_cf_4]; exact addc_carry_le_one y_1 x_1 0 b_y_1 b_x_1 (by decide)
  clear e_y_2 e_cf_4
  -- pf_8: sar {pf}, 20
  word_step pf_8 := asr pf_7 20 using asr_lt pf_7 20 b_pf_7
  -- y_3: sar {y}, 20
  word_step y_3 := asr y_2 20 using asr_lt y_2 20 b_y_2
  -- x_2: mov {x}, {pf}
  word_step x_2 := pf_8 using b_pf_8
  -- pg_10: movabs {pg}, 0xfffffe0000000000
  word_step pg_10 := 18446741874686296064 using (by decide)
  -- pf_9: mov {pf}, {x}
  word_step pf_9 := x_2 using b_x_2
  -- pf_10: and {pf}, 0xfffff
  word_step pf_10 := andw pf_9 1048575 using andw_lt pf_9 1048575 b_pf_9 (by decide)
  -- pf_11: or {pf}, {pg}
  word_step pf_11 := orrw pf_10 pg_10 using orrw_lt pf_10 pg_10 b_pf_10 b_pg_10
  -- pg_11: mov {pg}, {y}
  word_step pg_11 := y_3 using b_y_3
  -- pg_12: and {pg}, 0xfffff
  word_step pg_12 := andw pg_11 1048575 using andw_lt pg_11 1048575 b_pg_11 (by decide)
  -- pg_13: bts {pg}, 63
  word_step pg_13 := orrw pg_12 9223372036854775808 using orrw_lt pg_12 9223372036854775808 b_pg_12 (by decide)
  -- pg_14: bts {pg}, 62
  word_step pg_14 := orrw pg_13 4611686018427387904 using orrw_lt pg_13 4611686018427387904 b_pg_13 (by decide)
  -- z_2: xor {z:e}, {z:e}
  word_step z_2 := 0 using (by decide), cf_5 := 0, ofl_1 := 0
  have b_cf_5 : cf_5 ≤ 1 := by rw [e_cf_5]; decide
  have b_ofl_1 : ofl_1 ≤ 1 := by rw [e_ofl_1]; decide
  -- t_4: mov {t}, {pf}
  word_step t_4 := pf_11 using b_pf_11
  -- k_3: mov {k}, -2
  word_step k_3 := 18446744073709551614 using (by decide)
  -- zs_2: test {td}, {td}
  word_step zs_2 := andw td_3 td_3 using andw_lt td_3 td_3 b_td_3 b_td_3
  -- k_4: cmovs {k}, {z}
  word_step k_4 := cmovs zs_2 k_3 z_2 using cmovs_lt zs_2 k_3 z_2 b_k_3 b_z_2
  -- zs_3: test {pg}, 1
  word_step zs_3 := andw pg_14 1 using andw_lt pg_14 1 b_pg_14 (by decide)
  -- k_5: cmove {k}, {z}
  word_step k_5 := cmovz zs_3 k_4 z_2 using cmovz_lt zs_3 k_4 z_2 b_k_4 b_z_2
  -- t_5: cmove {t}, {z}
  word_step t_5 := cmovz zs_3 t_4 z_2 using cmovz_lt zs_3 t_4 z_2 b_t_4 b_z_2
  -- t_6: xor {t}, {k}
  word_step t_6 := eorw t_5 k_5 using eorw_lt t_5 k_5 b_t_5 b_k_5
  -- td_4: xor {td}, {k}
  word_step td_4 := eorw td_3 k_5 using eorw_lt td_3 k_5 b_td_3 b_k_5
  -- cf_6: bt {k}, 63
  word_step cf_6 := bt k_5 63
  have b_cf_6 : cf_6 ≤ 1 := by
    rw [e_cf_6]; exact Nat.le_of_lt_succ (Nat.mod_lt _ (by decide))
  -- pf_12: cmovb {pf}, {pg}
  word_step pf_12 := cmovb cf_6 pf_11 pg_14 using cmovb_lt cf_6 pf_11 pg_14 b_pf_11 b_pg_14
  -- step41: divstep!(), invocations 23 to 41
  word_step step41 := divstepRound^[19] ⟨td_4, pf_12, pg_14, t_6, z_2⟩
  -- td_5: divstep!(), invocations 23 to 41 output
  word_step td_5 := step41.two_delta
  -- pf_13: divstep!(), invocations 23 to 41 output
  word_step pf_13 := step41.f
  -- pg_15: divstep!(), invocations 23 to 41 output
  word_step pg_15 := step41.g
  -- t_7: divstep!(), invocations 23 to 41 output
  word_step t_7 := step41.t
  -- z_3: divstep!(), invocations 23 to 41 output
  word_step z_3 := step41.z
  -- BEGIN batch 2 bound
  have b_td_5 : td_5 < 2^64 := by
    rw [e_td_5, e_step41]
    exact (divstepRound_iterate_bounded 19 _ ⟨b_td_4, b_pf_12, b_pg_14, b_t_6, b_z_2⟩).1
  -- END batch 2 bound
  -- td_6: add {td}, 2
  word_step s_3, td_6 := (addc td_5 2 0).1 using addc_value_lt td_5 2 0, cf_7 := (addc td_5 2 0).2
  have l_td_6 : td_6 + 2^64 * cf_7 = td_5 + 2 + 0 := by
    rw [e_td_6, e_cf_7]; exact addc_lin td_5 2 0
  have b_cf_7 : cf_7 ≤ 1 := by rw [e_cf_7]; exact addc_carry_le_one td_5 2 0 b_td_5 (by decide) (by decide)
  clear e_td_6 e_cf_7
  -- pg_16: lea {pg}, [{pg} + {t}]
  word_step pg_16 := addw pg_15 t_7 using addw_lt pg_15 t_7
  -- pg_17: sar {pg}, 1
  word_step pg_17 := asr pg_16 1 using asr_lt pg_16 1 b_pg_16
  -- h_1: movabs {h}, 0x20000100000
  word_step h_1 := 2199024304128 using (by decide)
  -- k_6: lea {k}, [{pf} + {h}]
  word_step k_6 := addw pf_13 h_1 using addw_lt pf_13 h_1
  -- k_7: sar {k}, 42
  word_step k_7 := asr k_6 42 using asr_lt k_6 42 b_k_6
  -- z_4: lea {z}, [{pg} + {h}]
  word_step z_4 := addw pg_17 h_1 using addw_lt pg_17 h_1
  -- z_5: sar {z}, 42
  word_step z_5 := asr z_4 42 using asr_lt z_4 42 b_z_4
  -- t_8: lea {t}, [{pf} + 0x100000]
  word_step t_8 := addw pf_13 1048576 using addw_lt pf_13 1048576
  -- t_9: shl {t}, 22
  word_step t_9 := t_8 * 2^22 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t_10: sar {t}, 43
  word_step t_10 := asr t_9 43 using asr_lt t_9 43 b_t_9
  -- h_2: lea {h}, [{pg} + 0x100000]
  word_step h_2 := addw pg_17 1048576 using addw_lt pg_17 1048576
  -- h_3: shl {h}, 22
  word_step h_3 := h_2 * 2^22 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- h_4: sar {h}, 43
  word_step h_4 := asr h_3 43 using asr_lt h_3 43 b_h_3
  -- pf_14: mov {pf}, {t}
  word_step pf_14 := t_10 using b_t_10
  -- pf_15: imul {pf}, {x}
  word_step pf_15 := pf_14 * x_2 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- pg_18: mov {pg}, {k}
  word_step pg_18 := k_7 using b_k_7
  -- pg_19: imul {pg}, {y}
  word_step pg_19 := pg_18 * y_3 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- pf_16: add {pf}, {pg}
  word_step s_4, pf_16 := (addc pf_15 pg_19 0).1 using addc_value_lt pf_15 pg_19 0,
      cf_8 := (addc pf_15 pg_19 0).2
  have l_pf_16 : pf_16 + 2^64 * cf_8 = pf_15 + pg_19 + 0 := by
    rw [e_pf_16, e_cf_8]; exact addc_lin pf_15 pg_19 0
  have b_cf_8 : cf_8 ≤ 1 := by rw [e_cf_8]; exact addc_carry_le_one pf_15 pg_19 0 b_pf_15 b_pg_19 (by decide)
  clear e_pf_16 e_cf_8
  -- pg_20: mov {pg}, {h}
  word_step pg_20 := h_4 using b_h_4
  -- pg_21: imul {pg}, {x}
  word_step pg_21 := pg_20 * x_2 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- y_4: imul {y}, {z}
  word_step y_4 := y_3 * z_5 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- y_5: add {y}, {pg}
  word_step s_5, y_5 := (addc y_4 pg_21 0).1 using addc_value_lt y_4 pg_21 0,
      cf_9 := (addc y_4 pg_21 0).2
  have l_y_5 : y_5 + 2^64 * cf_9 = y_4 + pg_21 + 0 := by
    rw [e_y_5, e_cf_9]; exact addc_lin y_4 pg_21 0
  have b_cf_9 : cf_9 ≤ 1 := by rw [e_cf_9]; exact addc_carry_le_one y_4 pg_21 0 b_y_4 b_pg_21 (by decide)
  clear e_y_5 e_cf_9
  -- pf_17: sar {pf}, 20
  word_step pf_17 := asr pf_16 20 using asr_lt pf_16 20 b_pf_16
  -- y_6: sar {y}, 20
  word_step y_6 := asr y_5 20 using asr_lt y_5 20 b_y_5
  -- x_3: mov {x}, {pf}
  word_step x_3 := pf_17 using b_pf_17
  -- pg_22: movabs {pg}, 0xfffffe0000000000
  word_step pg_22 := 18446741874686296064 using (by decide)
  -- pf_18: mov {pf}, {x}
  word_step pf_18 := x_3 using b_x_3
  -- pf_19: and {pf}, 0xfffff
  word_step pf_19 := andw pf_18 1048575 using andw_lt pf_18 1048575 b_pf_18 (by decide)
  -- pf_20: or {pf}, {pg}
  word_step pf_20 := orrw pf_19 pg_22 using orrw_lt pf_19 pg_22 b_pf_19 b_pg_22
  -- pg_23: mov {pg}, {y}
  word_step pg_23 := y_6 using b_y_6
  -- pg_24: and {pg}, 0xfffff
  word_step pg_24 := andw pg_23 1048575 using andw_lt pg_23 1048575 b_pg_23 (by decide)
  -- pg_25: bts {pg}, 63
  word_step pg_25 := orrw pg_24 9223372036854775808 using orrw_lt pg_24 9223372036854775808 b_pg_24 (by decide)
  -- pg_26: bts {pg}, 62
  word_step pg_26 := orrw pg_25 4611686018427387904 using orrw_lt pg_25 4611686018427387904 b_pg_25 (by decide)
  -- x_4: mov {x}, {t}
  word_step x_4 := t_10 using b_t_10
  -- x_5: imul {x}, {a0}
  word_step x_5 := x_4 * a0_2 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- y_7: mov {y}, {k}
  word_step y_7 := k_7 using b_k_7
  -- y_8: imul {y}, {a2}
  word_step y_8 := y_7 * a2_2 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- x_6: add {x}, {y}
  word_step s_6, x_6 := (addc x_5 y_8 0).1 using addc_value_lt x_5 y_8 0,
      cf_10 := (addc x_5 y_8 0).2
  have l_x_6 : x_6 + 2^64 * cf_10 = x_5 + y_8 + 0 := by
    rw [e_x_6, e_cf_10]; exact addc_lin x_5 y_8 0
  have b_cf_10 : cf_10 ≤ 1 := by rw [e_cf_10]; exact addc_carry_le_one x_5 y_8 0 b_x_5 b_y_8 (by decide)
  clear e_x_6 e_cf_10
  -- y_9: mov {y}, {h}
  word_step y_9 := h_4 using b_h_4
  -- y_10: imul {y}, {a0}
  word_step y_10 := y_9 * a0_2 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- a0_3: mov {a0}, {z}
  word_step a0_3 := z_5 using b_z_5
  -- a0_4: imul {a0}, {a2}
  word_step a0_4 := a0_3 * a2_2 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- y_11: add {y}, {a0}
  word_step s_7, y_11 := (addc y_10 a0_4 0).1 using addc_value_lt y_10 a0_4 0,
      cf_11 := (addc y_10 a0_4 0).2
  have l_y_11 : y_11 + 2^64 * cf_11 = y_10 + a0_4 + 0 := by
    rw [e_y_11, e_cf_11]; exact addc_lin y_10 a0_4 0
  have b_cf_11 : cf_11 ≤ 1 := by rw [e_cf_11]; exact addc_carry_le_one y_10 a0_4 0 b_y_10 b_a0_4 (by decide)
  clear e_y_11 e_cf_11
  -- a0_5: mov {a0}, {t}
  word_step a0_5 := t_10 using b_t_10
  -- a0_6: imul {a0}, {a1}
  word_step a0_6 := a0_5 * a1_1 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- a2_3: mov {a2}, {k}
  word_step a2_3 := k_7 using b_k_7
  -- a2_4: imul {a2}, {a3}
  word_step a2_4 := a2_3 * a3_1 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- a0_7: add {a0}, {a2}
  word_step s_8, a0_7 := (addc a0_6 a2_4 0).1 using addc_value_lt a0_6 a2_4 0,
      cf_12 := (addc a0_6 a2_4 0).2
  have l_a0_7 : a0_7 + 2^64 * cf_12 = a0_6 + a2_4 + 0 := by
    rw [e_a0_7, e_cf_12]; exact addc_lin a0_6 a2_4 0
  have b_cf_12 : cf_12 ≤ 1 := by rw [e_cf_12]; exact addc_carry_le_one a0_6 a2_4 0 b_a0_6 b_a2_4 (by decide)
  clear e_a0_7 e_cf_12
  -- a2_5: mov {a2}, {h}
  word_step a2_5 := h_4 using b_h_4
  -- a2_6: imul {a2}, {a1}
  word_step a2_6 := a2_5 * a1_1 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- a3_2: imul {a3}, {z}
  word_step a3_2 := a3_1 * z_5 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- a3_3: add {a3}, {a2}
  word_step s_9, a3_3 := (addc a3_2 a2_6 0).1 using addc_value_lt a3_2 a2_6 0,
      cf_13 := (addc a3_2 a2_6 0).2
  have l_a3_3 : a3_3 + 2^64 * cf_13 = a3_2 + a2_6 + 0 := by
    rw [e_a3_3, e_cf_13]; exact addc_lin a3_2 a2_6 0
  have b_cf_13 : cf_13 ≤ 1 := by rw [e_cf_13]; exact addc_carry_le_one a3_2 a2_6 0 b_a3_2 b_a2_6 (by decide)
  clear e_a3_3 e_cf_13
  -- z_6: xor {z:e}, {z:e}
  word_step z_6 := 0 using (by decide), cf_14 := 0, ofl_2 := 0
  have b_cf_14 : cf_14 ≤ 1 := by rw [e_cf_14]; decide
  have b_ofl_2 : ofl_2 ≤ 1 := by rw [e_ofl_2]; decide
  -- t_11: mov {t}, {pf}
  word_step t_11 := pf_20 using b_pf_20
  -- k_8: mov {k}, -2
  word_step k_8 := 18446744073709551614 using (by decide)
  -- zs_4: test {td}, {td}
  word_step zs_4 := andw td_6 td_6 using andw_lt td_6 td_6 b_td_6 b_td_6
  -- k_9: cmovs {k}, {z}
  word_step k_9 := cmovs zs_4 k_8 z_6 using cmovs_lt zs_4 k_8 z_6 b_k_8 b_z_6
  -- zs_5: test {pg}, 1
  word_step zs_5 := andw pg_26 1 using andw_lt pg_26 1 b_pg_26 (by decide)
  -- k_10: cmove {k}, {z}
  word_step k_10 := cmovz zs_5 k_9 z_6 using cmovz_lt zs_5 k_9 z_6 b_k_9 b_z_6
  -- t_12: cmove {t}, {z}
  word_step t_12 := cmovz zs_5 t_11 z_6 using cmovz_lt zs_5 t_11 z_6 b_t_11 b_z_6
  -- t_13: xor {t}, {k}
  word_step t_13 := eorw t_12 k_10 using eorw_lt t_12 k_10 b_t_12 b_k_10
  -- td_7: xor {td}, {k}
  word_step td_7 := eorw td_6 k_10 using eorw_lt td_6 k_10 b_td_6 b_k_10
  -- cf_15: bt {k}, 63
  word_step cf_15 := bt k_10 63
  have b_cf_15 : cf_15 ≤ 1 := by
    rw [e_cf_15]; exact Nat.le_of_lt_succ (Nat.mod_lt _ (by decide))
  -- pf_21: cmovb {pf}, {pg}
  word_step pf_21 := cmovb cf_15 pf_20 pg_26 using cmovb_lt cf_15 pf_20 pg_26 b_pf_20 b_pg_26
  -- step61: divstep!(), invocations 44 to 61
  word_step step61 := divstepRound^[18] ⟨td_7, pf_21, pg_26, t_13, z_6⟩
  -- td_8: divstep!(), invocations 44 to 61 output
  word_step td_8 := step61.two_delta
  -- pf_22: divstep!(), invocations 44 to 61 output
  word_step pf_22 := step61.f
  -- pg_27: divstep!(), invocations 44 to 61 output
  word_step pg_27 := step61.g
  -- t_14: divstep!(), invocations 44 to 61 output
  word_step t_14 := step61.t
  -- z_7: divstep!(), invocations 44 to 61 output
  word_step z_7 := step61.z
  -- BEGIN batch 3 bound
  have b_td_8 : td_8 < 2^64 := by
    rw [e_td_8, e_step61]
    exact (divstepRound_iterate_bounded 18 _ ⟨b_td_7, b_pf_21, b_pg_26, b_t_13, b_z_6⟩).1
  -- END batch 3 bound
  -- td_9: add {td}, 2
  word_step s_10, td_9 := (addc td_8 2 0).1 using addc_value_lt td_8 2 0, cf_16 := (addc td_8 2 0).2
  have l_td_9 : td_9 + 2^64 * cf_16 = td_8 + 2 + 0 := by
    rw [e_td_9, e_cf_16]; exact addc_lin td_8 2 0
  have b_cf_16 : cf_16 ≤ 1 := by rw [e_cf_16]; exact addc_carry_le_one td_8 2 0 b_td_8 (by decide) (by decide)
  clear e_td_9 e_cf_16
  -- pg_28: lea {pg}, [{pg} + {t}]
  word_step pg_28 := addw pg_27 t_14 using addw_lt pg_27 t_14
  -- pg_29: sar {pg}, 1
  word_step pg_29 := asr pg_28 1 using asr_lt pg_28 1 b_pg_28
  -- h_5: movabs {h}, 0x20000100000
  word_step h_5 := 2199024304128 using (by decide)
  -- a1_2: lea {a1}, [{pf} + {h}]
  word_step a1_2 := addw pf_22 h_5 using addw_lt pf_22 h_5
  -- a1_3: sar {a1}, 43
  word_step a1_3 := asr a1_2 43 using asr_lt a1_2 43 b_a1_2
  -- a2_7: lea {a2}, [{pg} + {h}]
  word_step a2_7 := addw pg_29 h_5 using addw_lt pg_29 h_5
  -- a2_8: sar {a2}, 43
  word_step a2_8 := asr a2_7 43 using asr_lt a2_7 43 b_a2_7
  -- t_15: lea {t}, [{pf} + 0x100000]
  word_step t_15 := addw pf_22 1048576 using addw_lt pf_22 1048576
  -- t_16: shl {t}, 21
  word_step t_16 := t_15 * 2^21 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t_17: sar {t}, 43
  word_step t_17 := asr t_16 43 using asr_lt t_16 43 b_t_16
  -- k_11: lea {k}, [{pg} + 0x100000]
  word_step k_11 := addw pg_29 1048576 using addw_lt pg_29 1048576
  -- k_12: shl {k}, 21
  word_step k_12 := k_11 * 2^21 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- k_13: sar {k}, 43
  word_step k_13 := asr k_12 43 using asr_lt k_12 43 b_k_12
  -- pf_23: mov {pf}, {t}
  word_step pf_23 := t_17 using b_t_17
  -- pf_24: imul {pf}, {x}
  word_step pf_24 := pf_23 * x_6 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- pg_30: mov {pg}, {a1}
  word_step pg_30 := a1_3 using b_a1_3
  -- pg_31: imul {pg}, {y}
  word_step pg_31 := pg_30 * y_11 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- pf_25: add {pf}, {pg}
  word_step s_11, pf_25 := (addc pf_24 pg_31 0).1 using addc_value_lt pf_24 pg_31 0,
      cf_17 := (addc pf_24 pg_31 0).2
  have l_pf_25 : pf_25 + 2^64 * cf_17 = pf_24 + pg_31 + 0 := by
    rw [e_pf_25, e_cf_17]; exact addc_lin pf_24 pg_31 0
  have b_cf_17 : cf_17 ≤ 1 := by rw [e_cf_17]; exact addc_carry_le_one pf_24 pg_31 0 b_pf_24 b_pg_31 (by decide)
  clear e_pf_25 e_cf_17
  -- n: neg {pf}
  word_step n, pf_26 := (neg pf_25).1 using neg_value_lt pf_25, cf_18 := (neg pf_25).2
  have b_cf_18 : cf_18 ≤ 1 := by
    rw [e_cf_18]; simp only [neg]; split <;> omega
  -- pg_32: mov {pg}, {t}
  word_step pg_32 := t_17 using b_t_17
  -- pg_33: imul {pg}, {a0}
  word_step pg_33 := pg_32 * a0_7 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- h_6: mov {h}, {a1}
  word_step h_6 := a1_3 using b_a1_3
  -- h_7: imul {h}, {a3}
  word_step h_7 := h_6 * a3_3 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- pg_34: add {pg}, {h}
  word_step s_12, pg_34 := (addc pg_33 h_7 0).1 using addc_value_lt pg_33 h_7 0,
      cf_19 := (addc pg_33 h_7 0).2
  have l_pg_34 : pg_34 + 2^64 * cf_19 = pg_33 + h_7 + 0 := by
    rw [e_pg_34, e_cf_19]; exact addc_lin pg_33 h_7 0
  have b_cf_19 : cf_19 ≤ 1 := by rw [e_cf_19]; exact addc_carry_le_one pg_33 h_7 0 b_pg_33 b_h_7 (by decide)
  clear e_pg_34 e_cf_19
  -- n_1: neg {pg}
  word_step n_1, pg_35 := (neg pg_34).1 using neg_value_lt pg_34, cf_20 := (neg pg_34).2
  have b_cf_20 : cf_20 ≤ 1 := by
    rw [e_cf_20]; simp only [neg]; split <;> omega
  -- z_8: mov {z}, {k}
  word_step z_8 := k_13 using b_k_13
  -- z_9: imul {z}, {x}
  word_step z_9 := z_8 * x_6 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t_18: mov {t}, {a2}
  word_step t_18 := a2_8 using b_a2_8
  -- t_19: imul {t}, {y}
  word_step t_19 := t_18 * y_11 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- z_10: add {z}, {t}
  word_step s_13, z_10 := (addc z_9 t_19 0).1 using addc_value_lt z_9 t_19 0,
      cf_21 := (addc z_9 t_19 0).2
  have l_z_10 : z_10 + 2^64 * cf_21 = z_9 + t_19 + 0 := by
    rw [e_z_10, e_cf_21]; exact addc_lin z_9 t_19 0
  have b_cf_21 : cf_21 ≤ 1 := by rw [e_cf_21]; exact addc_carry_le_one z_9 t_19 0 b_z_9 b_t_19 (by decide)
  clear e_z_10 e_cf_21
  -- n_2: neg {z}
  word_step n_2, z_11 := (neg z_10).1 using neg_value_lt z_10, cf_22 := (neg z_10).2
  have b_cf_22 : cf_22 ≤ 1 := by
    rw [e_cf_22]; simp only [neg]; split <;> omega
  -- k_14: imul {k}, {a0}
  word_step k_14 := k_13 * a0_7 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- a2_9: imul {a2}, {a3}
  word_step a2_9 := a2_8 * a3_3 % 2^64 using Nat.mod_lt _ (Nat.two_pow_pos _)
  -- k_15: add {k}, {a2}
  word_step s_14, k_15 := (addc k_14 a2_9 0).1 using addc_value_lt k_14 a2_9 0,
      cf_23 := (addc k_14 a2_9 0).2
  have l_k_15 : k_15 + 2^64 * cf_23 = k_14 + a2_9 + 0 := by
    rw [e_k_15, e_cf_23]; exact addc_lin k_14 a2_9 0
  have b_cf_23 : cf_23 ≤ 1 := by rw [e_cf_23]; exact addc_carry_le_one k_14 a2_9 0 b_k_14 b_a2_9 (by decide)
  clear e_k_15 e_cf_23
  -- n_3: neg {k}
  word_step n_3, k_16 := (neg k_15).1 using neg_value_lt k_15, cf_24 := (neg k_15).2
  have b_cf_24 : cf_24 ≤ 1 := by
    rw [e_cf_24]; simp only [neg]; split <;> omega
  subst hres
  -- BEGIN conclusion
  have hpos64 : (0 : ℤ) < 2^64 := by norm_num
  have hsDa : |σ.two_delta| < 2^61 := hsD
  rw [abs_lt] at hsD
  -- Batch 1: the packed start is `packedStart s₀` for the truncation `s₀` of `σ`.
  set s₀ : State := ⟨σ.two_delta, σ.f % 2^20, σ.g % 2^20⟩ with hs₀
  have hs₀f : s₀.f % 2 = 1 := by show (σ.f % 2^20) % 2 = 1; omega
  have hs₀fb : 0 ≤ s₀.f ∧ s₀.f < 2^20 :=
    ⟨Int.emod_nonneg _ (by norm_num), Int.emod_lt_of_pos _ (by norm_num)⟩
  have hs₀gb : 0 ≤ s₀.g ∧ s₀.g < 2^20 :=
    ⟨Int.emod_nonneg _ (by norm_num), Int.emod_lt_of_pos _ (by norm_num)⟩
  set P0 : State := packedStart s₀ with hP0
  have hP0f : P0.f % 2 = 1 := by show (σ.f % 2^20 - 2^41) % 2 = 1; omega
  have hP0g : |P0.g| < 2^63 := by
    show |σ.g % 2^20 - 2^62| < 2^63; rw [abs_lt]; constructor <;> omega
  have hP0D : |P0.two_delta| + 2 * (19 + 1) < 2^62 := by
    show |σ.two_delta| + 2 * (19 + 1) < 2^62; omega
  have hw_d : (td : ℤ) = P0.two_delta % 2^64 := by rw [e_td]; exact ed
  have hw_f : (pf_2 : ℤ) = P0.f % 2^64 := by
    rw [e_pf_2, e_pf_1, e_pg, e_pf, e_x, pack_f_word]; show _ = (σ.f % 2^20 - 2^41) % 2^64; omega
  have hw_g : (pg_4 : ℤ) = P0.g % 2^64 := by
    rw [e_pg_4, e_pg_3, e_pg_2, e_pg_1, e_y, pack_g_bts]; show _ = (σ.g % 2^20 - 2^62) % 2^64
    omega
  have hc0 : Decided ⟨td_1, pf_3, pg_4, t_2, z⟩ P0 := by
    have h := decide_batch P0 hP0f hsd (by show |σ.two_delta| < 2^63; rw [abs_lt]; omega) td pf_2
      pg_4 b_td b_pf_2 hw_d hw_f hw_g
    rwa [e_td_1, e_pf_3, e_cf_1, e_t_2, e_t_1, e_t, e_k_2, e_k_1, e_k, e_zs, e_zs_1, e_z]
  obtain ⟨-, hd1, hf1, hg1⟩ := batch_words 19 P0 hP0f hsd hP0D hP0g
    (fun j _ => Inversion.divsteps_packedStart_g_abs_lt j s₀ hs₀f hs₀fb hs₀gb)
    ⟨td_1, pf_3, pg_4, t_2, z⟩ ⟨b_td_1, b_pf_3, b_pg_4, b_t_2, b_z⟩ hc0
  rw [← e_step20] at hd1 hf1 hg1
  have hd1' : (td_3 : ℤ) = (divsteps 20 P0).two_delta % 2^64 := by
    rw [lin_addw td_3 td_2 2 cf_2 b_td_3 l_td_3, e_td_2]; exact hd1
  have hf1' : (pf_4 : ℤ) = (divsteps 20 P0).f % 2^64 := by rw [e_pf_4]; exact hf1
  have hg1' : (pg_7 : ℤ) = (divsteps 20 P0).g % 2^64 := by
    rw [e_pg_7, e_pg_6, e_pg_5, e_t_3]; exact hg1
  clear hd1 hf1 hg1
  -- The words after batch 1 carry `divsteps 20 P0`, which Lemma 6 writes in the true state and
  -- matrix of `s₀`, and Lemma 2 in those of `σ`.
  have hpk1 := Inversion.divsteps_packedStart 20 (le_refl _) s₀ hs₀f
  obtain ⟨hφ1, hγ1⟩ := Inversion.divsteps_abs_le 20 s₀ (2^20 - 1)
    (by rw [abs_le]; constructor <;> omega) (by rw [abs_le]; constructor <;> omega)
  obtain ⟨hA1, hB1, hC1, hD1⟩ := Inversion.M_entry_range 20 s₀
  obtain ⟨hdl1, hMl1⟩ := Inversion.divsteps_local 20 σ s₀ hsf rfl
    (by show σ.f % 2^20 = (σ.f % 2^20) % 2^20; rw [Int.emod_emod_of_dvd _ (dvd_refl _)])
    (by show σ.g % 2^20 = (σ.g % 2^20) % 2^20; rw [Int.emod_emod_of_dvd _ (dvd_refl _)])
  rw [← hMl1] at hA1 hB1 hC1 hD1
  generalize hM1 : M 20 σ = M1 at hA1 hB1 hC1 hD1 hMl1
  generalize hs20 : divsteps 20 σ = s20 at hdl1
  have hd20 : (td_3 : ℤ) = s20.two_delta % 2^64 := by
    rw [hd1', hpk1]; exact congrArg (· % 2^64) hdl1.symm
  obtain ⟨hA00, hA01⟩ := decode20 pf_4 _ _ _ _ hf1' (by rw [hpk1, hMl1])
    (by rw [abs_le] at hφ1; rw [abs_lt]; constructor <;> omega) hA1 hB1
  obtain ⟨hA10, hA11⟩ := decode20 pg_7 _ _ _ _ hg1' (by rw [hpk1, hMl1])
    (by rw [abs_le] at hγ1; rw [abs_lt]; constructor <;> omega) hC1 hD1
  rw [← e_a0, ← e_a0_1, ← e_a0_2] at hA00
  rw [← e_h, ← e_a1, ← e_a1_1] at hA01
  rw [← e_a2, ← e_a2_1, ← e_a2_2] at hA10
  rw [← e_h, ← e_a3, ← e_a3_1] at hA11
  -- The next low words: `-(M1 (f, g))` over `2^20`, so they carry `-f_20`, `-g_20` modulo `2^44`.
  obtain ⟨hMf1, hMg1⟩ := Inversion.M_spec 20 σ hsf
  rw [hM1, hs20] at hMf1 hMg1
  have hf20 : s20.f % 2 = 1 := by rw [← hs20]; exact Inversion.divsteps_f_odd 20 σ hsf
  have hx0 : (x : ℤ) = σ.f % 2^64 := by rw [e_x]; exact ef0
  have hy0 : (y : ℤ) = σ.g % 2^64 := by rw [e_y]; exact eg0
  have hpf6 : (pf_6 : ℤ) = ((-M1.u) * σ.f) % 2^64 := by
    rw [e_pf_6, e_pf_5]; exact mul_word _ _ _ _ hA00 hx0
  have hpg9 : (pg_9 : ℤ) = ((-M1.v) * σ.g) % 2^64 := by
    rw [e_pg_9, e_pg_8]; exact mul_word _ _ _ _ hA01 hy0
  have hx1 : (x_1 : ℤ) = (σ.f * (-M1.q)) % 2^64 := by
    rw [e_x_1]; exact mul_word _ _ _ _ hx0 hA10
  have hy1 : (y_1 : ℤ) = (σ.g * (-M1.r)) % 2^64 := by
    rw [e_y_1]; exact mul_word _ _ _ _ hy0 hA11
  have hf2 : (x_2 : ℤ) % 2^44 = (-s20.f) % 2^44 := by
    rw [e_x_2, e_pf_8]; apply asr20_word
    rw [lin_addw pf_7 pf_6 pg_9 cf_3 b_pf_7 l_pf_7, addw_word _ _ _ _ hpf6 hpg9]
    exact congrArg (· % 2^64) (by linear_combination hMf1)
  have hg2 : (y_3 : ℤ) % 2^44 = (-s20.g) % 2^44 := by
    rw [e_y_3]; apply asr20_word
    rw [lin_addw y_2 y_1 x_1 cf_4 b_y_2 l_y_2, addw_word _ _ _ _ hy1 hx1]
    exact congrArg (· % 2^64) (by linear_combination hMg1)
  -- Batch 2 runs on the negated low words: its packed start is `packedStart s₁` for the truncation
  -- `s₁` of the negated state `sN`.
  set sN : State := ⟨s20.two_delta, -s20.f, -s20.g⟩ with hsN
  set s₁ : State := ⟨s20.two_delta, (-s20.f) % 2^20, (-s20.g) % 2^20⟩ with hs₁
  have hs₁f : s₁.f % 2 = 1 := by show ((-s20.f) % 2^20) % 2 = 1; omega
  have hs₁fb : 0 ≤ s₁.f ∧ s₁.f < 2^20 :=
    ⟨Int.emod_nonneg _ (by norm_num), Int.emod_lt_of_pos _ (by norm_num)⟩
  have hs₁gb : 0 ≤ s₁.g ∧ s₁.g < 2^20 :=
    ⟨Int.emod_nonneg _ (by norm_num), Int.emod_lt_of_pos _ (by norm_num)⟩
  set P1 : State := packedStart s₁ with hP1
  have hP1f : P1.f % 2 = 1 := by show ((-s20.f) % 2^20 - 2^41) % 2 = 1; omega
  have hP1d : P1.two_delta % 2 = 1 := by
    show s20.two_delta % 2 = 1; rw [← hs20, Inversion.divsteps_two_delta_emod_two]; exact hsd
  have hs20D := Inversion.divsteps_two_delta_abs_le 20 σ
  rw [hs20] at hs20D
  push_cast at hs20D
  have hs20Db := abs_le.mp (le_trans hs20D (le_refl _))
  have hP1g : |P1.g| < 2^63 := by
    show |(-s20.g) % 2^20 - 2^62| < 2^63; rw [abs_lt]; constructor <;> omega
  have hP1D : |P1.two_delta| + 2 * (19 + 1) < 2^62 := by
    show |s20.two_delta| + 2 * (19 + 1) < 2^62; omega
  have hw_f2 : (pf_11 : ℤ) = P1.f % 2^64 := by
    rw [e_pf_11, e_pf_10, e_pg_10, e_pf_9, pack_f_word]
    show _ = ((-s20.f) % 2^20 - 2^41) % 2^64; clear * - hf2; omega
  have hw_g2 : (pg_14 : ℤ) = P1.g % 2^64 := by
    rw [e_pg_14, e_pg_13, e_pg_12, e_pg_11, pack_g_bts]
    show _ = ((-s20.g) % 2^20 - 2^62) % 2^64; clear * - hg2; omega
  have hc1 : Decided ⟨td_4, pf_12, pg_14, t_6, z_2⟩ P1 := by
    have h := decide_batch P1 hP1f hP1d (by show |s20.two_delta| < 2^63; rw [abs_lt]; omega)
      td_3 pf_11 pg_14 b_td_3 b_pf_11 hd20 hw_f2 hw_g2
    rwa [e_td_4, e_pf_12, e_cf_6, e_t_6, e_t_5, e_t_4, e_k_5, e_k_4, e_k_3, e_zs_2, e_zs_3, e_z_2]
  obtain ⟨-, hd2, hf2', hg2'⟩ := batch_words 19 P1 hP1f hP1d hP1D hP1g
    (fun j _ => Inversion.divsteps_packedStart_g_abs_lt j s₁ hs₁f hs₁fb hs₁gb)
    ⟨td_4, pf_12, pg_14, t_6, z_2⟩ ⟨b_td_4, b_pf_12, b_pg_14, b_t_6, b_z_2⟩ hc1
  rw [← e_step41] at hd2 hf2' hg2'
  have hd2'' : (td_6 : ℤ) = (divsteps 20 P1).two_delta % 2^64 := by
    rw [lin_addw td_6 td_5 2 cf_7 b_td_6 l_td_6, e_td_5]; exact hd2
  have hf2'' : (pf_13 : ℤ) = (divsteps 20 P1).f % 2^64 := by rw [e_pf_13]; exact hf2'
  have hg2'' : (pg_17 : ℤ) = (divsteps 20 P1).g % 2^64 := by
    rw [e_pg_17, e_pg_16, e_pg_15, e_t_7]; exact hg2'
  clear hd2 hf2' hg2'
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
  have hdN : (divsteps 20 sN).two_delta = (divsteps 20 s20).two_delta := by
    rw [hsN, Inversion.divsteps_neg 20 s20 hf20]
  rw [← hMl2, hMN] at hA2 hB2 hC2 hD2
  generalize hM2 : M 20 s20 = M2 at hA2 hB2 hC2 hD2 hMN
  generalize hs40 : divsteps 40 σ = s40
  have hs40' : divsteps 20 s20 = s40 := by
    rw [← hs40, ← hs20]; exact (Inversion.divsteps_add 20 20 σ).symm
  have hd40 : (td_6 : ℤ) = s40.two_delta % 2^64 := by
    rw [hd2'', hpk2]
    exact congrArg (· % 2^64) (by rw [← hdl2, hdN, hs40'] : (divsteps 20 s₁).two_delta = s40.two_delta)
  obtain ⟨hB00, hB01⟩ := decode20 pf_13 _ _ _ _ hf2'' (by rw [hpk2, ← hMl2, hMN])
    (by rw [abs_le] at hφ2; rw [abs_lt]; constructor <;> omega) hA2 hB2
  obtain ⟨hB10, hB11⟩ := decode20 pg_17 _ _ _ _ hg2'' (by rw [hpk2, ← hMl2, hMN])
    (by rw [abs_le] at hγ2; rw [abs_lt]; constructor <;> omega) hC2 hD2
  rw [← e_t_8, ← e_t_9, ← e_t_10] at hB00
  rw [← e_h_1, ← e_k_6, ← e_k_7] at hB01
  rw [← e_h_2, ← e_h_3, ← e_h_4] at hB10
  rw [← e_h_1, ← e_z_4, ← e_z_5] at hB11
  -- The third low words: `-M2` times the negated words is `M2 (f_20, g_20)` modulo `2^44`, so after
  -- the shift they carry `f_40`, `g_40` modulo `2^24`.
  obtain ⟨hMf2, hMg2⟩ := Inversion.M_spec 20 s20 hf20
  rw [hs40', hM2] at hMf2 hMg2
  have hf40 : s40.f % 2 = 1 := by rw [← hs40']; exact Inversion.divsteps_f_odd 20 s20 hf20
  have hf2c : (x_2 : ℤ) = (x_2 : ℤ) % 2^64 :=
    (Int.emod_eq_of_lt (by omega) (by omega)).symm
  have hg2c : (y_3 : ℤ) = (y_3 : ℤ) % 2^64 :=
    (Int.emod_eq_of_lt (by omega) (by omega)).symm
  have hpf15 : (pf_15 : ℤ) = ((-M2.u) * x_2) % 2^64 := by
    rw [e_pf_15, e_pf_14]; exact mul_word _ _ _ _ hB00 hf2c
  have hpg19 : (pg_19 : ℤ) = ((-M2.v) * y_3) % 2^64 := by
    rw [e_pg_19, e_pg_18]; exact mul_word _ _ _ _ hB01 hg2c
  have hpg21 : (pg_21 : ℤ) = ((-M2.q) * x_2) % 2^64 := by
    rw [e_pg_21, e_pg_20]; exact mul_word _ _ _ _ hB10 hf2c
  have hy4 : (y_4 : ℤ) = (y_3 * (-M2.r)) % 2^64 := by
    rw [e_y_4]; exact mul_word _ _ _ _ hg2c hB11
  have hf2m : (x_2 : ℤ) ≡ -s20.f [ZMOD 2^44] := hf2
  have hg2m : (y_3 : ℤ) ≡ -s20.g [ZMOD 2^44] := hg2
  have h44 : (2 : ℤ)^44 ∣ 2^64 := pow_dvd_pow 2 (by norm_num)
  have hf4 : (x_3 : ℤ) % 2^24 = s40.f % 2^24 := by
    rw [e_x_3, e_pf_17]; apply asr20_word44 _ _
    rw [lin_addw pf_16 pf_15 pg_19 cf_8 b_pf_16 l_pf_16, addw_word _ _ _ _ hpf15 hpg19,
      Int.emod_emod_of_dvd _ h44]
    have h := (hf2m.mul_left (-M2.u)).add (hg2m.mul_left (-M2.v))
    unfold Int.ModEq at h; rw [h]
    exact congrArg (· % 2^44) (by linear_combination -hMf2)
  have hg4 : (y_6 : ℤ) % 2^24 = s40.g % 2^24 := by
    rw [e_y_6]; apply asr20_word44 _ _
    rw [lin_addw y_5 y_4 pg_21 cf_9 b_y_5 l_y_5, addw_word _ _ _ _ hy4 hpg21,
      Int.emod_emod_of_dvd _ h44]
    have h := (hg2m.mul_left (-M2.r)).add (hf2m.mul_left (-M2.q))
    unfold Int.ModEq at h; rw [mul_comm (y_3 : ℤ), h]
    exact congrArg (· % 2^44) (by linear_combination -hMg2)
  -- The products of the two negated matrices are `M2 · M1`.
  set M2M1 : Inversion.Mat2 := M2.mul M1 with hM2M1def
  have hM2M1u : (x_6 : ℤ) = M2M1.u % 2^64 := by
    rw [lin_addw x_6 x_5 y_8 cf_10 b_x_6 l_x_6,
      addw_word _ _ _ _ (by rw [e_x_5, e_x_4]; exact mul_word _ _ _ _ hB00 hA00)
        (by rw [e_y_8, e_y_7]; exact mul_word _ _ _ _ hB01 hA10)]
    exact congrArg (· % 2^64) (by simp only [hM2M1def, Inversion.Mat2.mul]; ring)
  have hM2M1q : (y_11 : ℤ) = M2M1.q % 2^64 := by
    rw [lin_addw y_11 y_10 a0_4 cf_11 b_y_11 l_y_11,
      addw_word _ _ _ _ (by rw [e_y_10, e_y_9]; exact mul_word _ _ _ _ hB10 hA00)
        (by rw [e_a0_4, e_a0_3]; exact mul_word _ _ _ _ hB11 hA10)]
    exact congrArg (· % 2^64) (by simp only [hM2M1def, Inversion.Mat2.mul]; ring)
  have hM2M1v : (a0_7 : ℤ) = M2M1.v % 2^64 := by
    rw [lin_addw a0_7 a0_6 a2_4 cf_12 b_a0_7 l_a0_7,
      addw_word _ _ _ _ (by rw [e_a0_6, e_a0_5]; exact mul_word _ _ _ _ hB00 hA01)
        (by rw [e_a2_4, e_a2_3]; exact mul_word _ _ _ _ hB01 hA11)]
    exact congrArg (· % 2^64) (by simp only [hM2M1def, Inversion.Mat2.mul]; ring)
  have hM2M1r : (a3_3 : ℤ) = M2M1.r % 2^64 := by
    rw [lin_addw a3_3 a3_2 a2_6 cf_13 b_a3_3 l_a3_3,
      addw_word _ _ _ _ (by rw [e_a3_2]; exact mul_word _ _ _ _ hA11 hB11)
        (by rw [e_a2_6, e_a2_5]; exact mul_word _ _ _ _ hB10 hA01)]
    exact congrArg (· % 2^64) (by simp only [hM2M1def, Inversion.Mat2.mul]; ring)
  -- Batch 3: 19 steps from the truncation `s₂` of `s40`, with the true sign.
  set s₂ : State := ⟨s40.two_delta, s40.f % 2^20, s40.g % 2^20⟩ with hs₂
  have hs₂f : s₂.f % 2 = 1 := by show (s40.f % 2^20) % 2 = 1; omega
  have hs₂fb : 0 ≤ s₂.f ∧ s₂.f < 2^20 :=
    ⟨Int.emod_nonneg _ (by norm_num), Int.emod_lt_of_pos _ (by norm_num)⟩
  have hs₂gb : 0 ≤ s₂.g ∧ s₂.g < 2^20 :=
    ⟨Int.emod_nonneg _ (by norm_num), Int.emod_lt_of_pos _ (by norm_num)⟩
  set P2 : State := packedStart s₂ with hP2
  have hP2f : P2.f % 2 = 1 := by show (s40.f % 2^20 - 2^41) % 2 = 1; omega
  have hP2d : P2.two_delta % 2 = 1 := by
    show s40.two_delta % 2 = 1; rw [← hs40, Inversion.divsteps_two_delta_emod_two]; exact hsd
  have hs40D := Inversion.divsteps_two_delta_abs_le 40 σ
  rw [hs40] at hs40D
  push_cast at hs40D
  have hs40Db := abs_le.mp (le_trans hs40D (le_refl _))
  have hP2g : |P2.g| < 2^63 := by
    show |s40.g % 2^20 - 2^62| < 2^63; rw [abs_lt]; constructor <;> omega
  have hP2D : |P2.two_delta| + 2 * (18 + 1) < 2^62 := by
    show |s40.two_delta| + 2 * (18 + 1) < 2^62; omega
  have hw_f3 : (pf_20 : ℤ) = P2.f % 2^64 := by
    rw [e_pf_20, e_pf_19, e_pg_22, e_pf_18, pack_f_word]
    show _ = (s40.f % 2^20 - 2^41) % 2^64; clear * - hf4; omega
  have hw_g3 : (pg_26 : ℤ) = P2.g % 2^64 := by
    rw [e_pg_26, e_pg_25, e_pg_24, e_pg_23, pack_g_bts]
    show _ = (s40.g % 2^20 - 2^62) % 2^64; clear * - hg4; omega
  have hc2 : Decided ⟨td_7, pf_21, pg_26, t_13, z_6⟩ P2 := by
    have h := decide_batch P2 hP2f hP2d (by show |s40.two_delta| < 2^63; rw [abs_lt]; omega)
      td_6 pf_20 pg_26 b_td_6 b_pf_20 hd40 hw_f3 hw_g3
    rwa [e_td_7, e_pf_21, e_cf_15, e_t_13, e_t_12, e_t_11, e_k_10, e_k_9, e_k_8, e_zs_4, e_zs_5,
      e_z_6]
  obtain ⟨-, hd3, hf3', hg3'⟩ := batch_words 18 P2 hP2f hP2d hP2D hP2g
    (fun j _ => Inversion.divsteps_packedStart_g_abs_lt j s₂ hs₂f hs₂fb hs₂gb)
    ⟨td_7, pf_21, pg_26, t_13, z_6⟩ ⟨b_td_7, b_pf_21, b_pg_26, b_t_13, b_z_6⟩ hc2
  rw [← e_step61] at hd3 hf3' hg3'
  have hd3'' : (td_9 : ℤ) = (divsteps 19 P2).two_delta % 2^64 := by
    rw [lin_addw td_9 td_8 2 cf_16 b_td_9 l_td_9, e_td_8]; exact hd3
  have hf3'' : (pf_22 : ℤ) = (divsteps 19 P2).f % 2^64 := by rw [e_pf_22]; exact hf3'
  have hg3'' : (pg_29 : ℤ) = (divsteps 19 P2).g % 2^64 := by
    rw [e_pg_29, e_pg_28, e_pg_27, e_t_14]; exact hg3'
  clear hd3 hf3' hg3'
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
  have hd59 : (td_9 : ℤ) = (divsteps 59 σ).two_delta % 2^64 := by
    rw [hd3'', hpk3]
    exact congrArg (· % 2^64)
      (by rw [← hdl3, ← hs40, ← Inversion.divsteps_add] :
        (divsteps 19 s₂).two_delta = (divsteps 59 σ).two_delta)
  obtain ⟨hC00, hC01⟩ := decode19 pf_22 _ _ _ _ hf3'' (by rw [hpk3, hMl3])
    (by rw [abs_le] at hφ3; rw [abs_lt]; constructor <;> omega) hA3 hB3
  obtain ⟨hC10, hC11⟩ := decode19 pg_29 _ _ _ _ hg3'' (by rw [hpk3, hMl3])
    (by rw [abs_le] at hγ3; rw [abs_lt]; constructor <;> omega) hC3 hD3
  rw [← e_t_15, ← e_t_16, ← e_t_17] at hC00
  rw [← e_h_5, ← e_a1_2, ← e_a1_3] at hC01
  rw [← e_k_11, ← e_k_12, ← e_k_13] at hC10
  rw [← e_h_5, ← e_a2_7, ← e_a2_8] at hC11
  -- The final products: `neg` restores the sign of the third matrix.
  have hM59 : M 59 σ = M3.mul M2M1 := by
    rw [← hM3, hM2M1def, ← hM2, ← hM1, ← hs40, ← hs20, ← Inversion.M_add, ← Inversion.M_add]
  refine ⟨hd59, ?_, ?_, ?_, ?_⟩
  · show (pf_26 : ℤ) = _
    rw [e_pf_26]
    rw [neg_word _ _ (by
      rw [lin_addw pf_25 pf_24 pg_31 cf_17 b_pf_25 l_pf_25,
        addw_word _ _ _ _ (by rw [e_pf_24, e_pf_23]; exact mul_word _ _ _ _ hC00 hM2M1u)
          (by rw [e_pg_31, e_pg_30]; exact mul_word _ _ _ _ hC01 hM2M1q)]), hM59]
    exact congrArg (· % 2^64) (by simp only [Inversion.Mat2.mul]; ring)
  · show (pg_35 : ℤ) = _
    rw [e_pg_35]
    rw [neg_word _ _ (by
      rw [lin_addw pg_34 pg_33 h_7 cf_19 b_pg_34 l_pg_34,
        addw_word _ _ _ _ (by rw [e_pg_33, e_pg_32]; exact mul_word _ _ _ _ hC00 hM2M1v)
          (by rw [e_h_7, e_h_6]; exact mul_word _ _ _ _ hC01 hM2M1r)]), hM59]
    exact congrArg (· % 2^64) (by simp only [Inversion.Mat2.mul]; ring)
  · show (z_11 : ℤ) = _
    rw [e_z_11]
    rw [neg_word _ _ (by
      rw [lin_addw z_10 z_9 t_19 cf_21 b_z_10 l_z_10,
        addw_word _ _ _ _ (by rw [e_z_9, e_z_8]; exact mul_word _ _ _ _ hC10 hM2M1u)
          (by rw [e_t_19, e_t_18]; exact mul_word _ _ _ _ hC11 hM2M1q)]), hM59]
    exact congrArg (· % 2^64) (by simp only [Inversion.Mat2.mul]; ring)
  · show (k_16 : ℤ) = _
    rw [e_k_16]
    rw [neg_word _ _ (by
      rw [lin_addw k_15 k_14 a2_9 cf_23 b_k_15 l_k_15,
        addw_word _ _ _ _ (by rw [e_k_14]; exact mul_word _ _ _ _ hC10 hM2M1v)
          (by rw [e_a2_9]; exact mul_word _ _ _ _ hC11 hM2M1r)]), hM59]
    exact congrArg (· % 2^64) (by simp only [Inversion.Mat2.mul]; ring)
  -- END conclusion

end PastaCurves.X86_64
