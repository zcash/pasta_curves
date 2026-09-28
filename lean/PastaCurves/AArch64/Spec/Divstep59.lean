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
  asr20_word asr20_word44 decode_v20 decode_v19 PackedStep)

-- BEGIN divstep59Block_spec lemmas
/-! ## Words -/

/-- `madd` on words carrying `X`, `Y`, and `Z` carries `X * Y + Z`: the third batch accumulates its
matrix products through it. -/
theorem madd_word (x y z : ℕ) (X Y Z : ℤ) (hx : (x : ℤ) = X % 2^64) (hy : (y : ℤ) = Y % 2^64)
    (hz : (z : ℤ) = Z % 2^64) : ((madd x y z : ℕ) : ℤ) = (X * Y + Z) % 2^64 := by
  have h : ((madd x y z : ℕ) : ℤ) = ((x : ℤ) * y + z) % 2^64 := by
    unfold madd regMod; push_cast; norm_num
  rw [h, hx, hy, hz]
  exact ((Int.mod_modEq X _).mul (Int.mod_modEq Y _)).add (Int.mod_modEq Z _)

/-- `mneg` carries `-(X * Y)`: it restores the sign of a product of two negated entries. -/
theorem mneg_word (x y : ℕ) (X Y : ℤ) (hx : (x : ℤ) = X % 2^64) (hy : (y : ℤ) = Y % 2^64) :
    ((mneg x y : ℕ) : ℤ) = (-(X * Y)) % 2^64 := by
  have h : ((mneg x y : ℕ) : ℤ) = (-((x : ℤ) * y)) % 2^64 := by
    unfold mneg regMod
    have h1 : x * y % 2^64 < 2^64 := Nat.mod_lt _ (by norm_num)
    omega
  rw [h, hx, hy]
  exact ((Int.mod_modEq X _).mul (Int.mod_modEq Y _)).neg

/-- `msub` carries `Z - X * Y`: subtracting the product of a negated entry restores its sign. -/
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

/-- The AArch64 packed step, for the shared batch iteration: `divstepRound` and `divstepLast` on
`DivstepState`, whose `Z` flag is the parity flag. -/
def packedStep : PackedStep :=
  ⟨DivstepState, DivstepState.Bounded, (·.two_delta), (·.f), (·.g), (·.fl.z), divstepRound, divstepLast⟩

/-- The two step theorems, as the batch iteration takes them. -/
theorem packedStep_spec : packedStep.Spec where
  round := fun st hst s hf hd hD hG hG' ed ef eg hz =>
    divstepRound_spec st hst s hf hd hD hG hG' ed ef eg hz _ rfl
  last := fun st hst s hf hd hD hG hG' ed ef eg hz =>
    divstepLast_spec st hst s hf hd hD hG hG' ed ef eg hz _ rfl

/-- A batch of `n + 1` steps: `n` rounds and the last step, the shared `batch_words` at the AArch64
step. -/
theorem batch_words (n : ℕ) (P : State) (hf : P.f % 2 = 1) (hd : P.two_delta % 2 = 1)
    (hD : |P.two_delta| + 2 * (n + 1) < 2^62) (hg0 : |P.g| < 2^63)
    (hg : ∀ j, j < n + 1 → |(divsteps (j + 1) P).g| < 2^62)
    (st : DivstepState) (hst : st.Bounded)
    (ed : (st.two_delta : ℤ) = P.two_delta % 2^64) (ef : (st.f : ℤ) = P.f % 2^64) (eg : (st.g : ℤ) = P.g % 2^64)
    (hz : st.fl.z = if P.g % 2 = 0 then 1 else 0) :
    (divstepLast (divstepRound^[n] st)).Bounded ∧
      ((divstepLast (divstepRound^[n] st)).two_delta : ℤ) = (divsteps (n + 1) P).two_delta % 2^64 ∧
      ((divstepLast (divstepRound^[n] st)).f : ℤ) = (divsteps (n + 1) P).f % 2^64 ∧
      ((divstepLast (divstepRound^[n] st)).g : ℤ) = (divsteps (n + 1) P).g % 2^64 :=
  packedStep_spec.batch_words n P hf hd hD hg0 hg st hst ed ef eg hz

-- END divstep59Block_spec lemmas

-- BEGIN divstep59Block_spec statement
-- The budget is the sum of the skeleton's bindings and of the conclusion's many small steps,
-- not any one of them.
set_option maxHeartbeats 400000 in
/-- The 59-step block on the words of a state `s` with odd `f` and `two_delta`, and
`|two_delta| < 2^61`, which the ten rounds of the inversion keep: its `two_delta` word carries the
`two_delta` after 59 steps, and its matrix words the 59-step matrix `M 59 s`, in two's complement.
-/
theorem divstep59Block_spec (two_delta f0 g0 : ℕ) (s : State) (hsf : s.f % 2 = 1) (hsd : s.two_delta % 2 = 1)
    (hsD : |s.two_delta| < 2^61) (ed : (two_delta : ℤ) = s.two_delta % 2^64) (ef0 : (f0 : ℤ) = s.f % 2^64)
    (eg0 : (g0 : ℤ) = s.g % 2^64) :
    ∀ res, res = divstep59Block two_delta f0 g0 →
      (res.two_delta : ℤ) = (divsteps 59 s).two_delta % 2^64 ∧
        (res.u : ℤ) = (M 59 s).u % 2^64 ∧ (res.v : ℤ) = (M 59 s).v % 2^64 ∧
        (res.q : ℤ) = (M 59 s).q % 2^64 ∧ (res.r : ℤ) = (M 59 s).r % 2^64 := by
  intro res hres
  have htwo_delta : two_delta < 2^64 := by omega
  have hf0 : f0 < 2^64 := by omega
  have hg0 : g0 < 2^64 := by omega
-- END divstep59Block_spec statement
  -- generated skeleton for `divstep59Block`: do not edit between the annotations
  unfold divstep59Block at hres
  lift_lets -merge at hres
  -- two_delta': argument
  extract_lets -merge +onlyGivenNames two_delta' at hres
  have e_two_delta' : two_delta' = two_delta := rfl
  clear_value two_delta'
  have b_two_delta' : two_delta' < 2^64 := by rw [e_two_delta']; exact htwo_delta
  -- f: argument
  extract_lets -merge +onlyGivenNames f at hres
  have e_f : f = f0 := rfl
  clear_value f
  have b_f : f < 2^64 := by rw [e_f]; exact hf0
  -- g: argument
  extract_lets -merge +onlyGivenNames g at hres
  have e_g : g = g0 := rfl
  clear_value g
  have b_g : g < 2^64 := by rw [e_g]; exact hg0
  -- pf: and pf,f,#0xfffff
  extract_lets -merge +onlyGivenNames pf at hres
  have e_pf : pf = andw f 0xfffff := rfl
  clear_value pf
  have b_pf : pf < 2^64 := by rw [e_pf]; exact andw_lt f 0xfffff b_f (by decide)
  -- pf_1: orr pf,pf,#0xfffffe0000000000
  extract_lets -merge +onlyGivenNames pf_1 at hres
  have e_pf_1 : pf_1 = orrw pf 0xfffffe0000000000 := rfl
  clear_value pf_1
  have b_pf_1 : pf_1 < 2^64 := by rw [e_pf_1]; exact orrw_lt pf 0xfffffe0000000000 b_pf (by decide)
  -- pg: and pg,g,#0xfffff
  extract_lets -merge +onlyGivenNames pg at hres
  have e_pg : pg = andw g 0xfffff := rfl
  clear_value pg
  have b_pg : pg < 2^64 := by rw [e_pg]; exact andw_lt g 0xfffff b_g (by decide)
  -- pg_1: orr pg,pg,#0xc000000000000000
  extract_lets -merge +onlyGivenNames pg_1 at hres
  have e_pg_1 : pg_1 = orrw pg 0xc000000000000000 := rfl
  clear_value pg_1
  have b_pg_1 : pg_1 < 2^64 := by rw [e_pg_1]; exact orrw_lt pg 0xc000000000000000 b_pg (by decide)
  -- fl: tst pg,#1
  extract_lets -merge +onlyGivenNames fl at hres
  have e_fl : fl = tstFlags (andw pg_1 1) := rfl
  clear_value fl
  -- step19: divstep!(), invocations 1 to 19
  extract_lets -merge +onlyGivenNames step19 at hres
  have e_step19 : step19 = divstepRound^[19] ⟨two_delta', pf_1, pg_1, fl⟩ := rfl
  clear_value step19
  -- two_delta_1: divstep!(), invocations 1 to 19 output
  extract_lets -merge +onlyGivenNames two_delta_1 at hres
  have e_two_delta_1 : two_delta_1 = step19.two_delta := rfl
  clear_value two_delta_1
  -- pf_2: divstep!(), invocations 1 to 19 output
  extract_lets -merge +onlyGivenNames pf_2 at hres
  have e_pf_2 : pf_2 = step19.f := rfl
  clear_value pf_2
  -- pg_2: divstep!(), invocations 1 to 19 output
  extract_lets -merge +onlyGivenNames pg_2 at hres
  have e_pg_2 : pg_2 = step19.g := rfl
  clear_value pg_2
  -- fl_1: divstep!(), invocations 1 to 19 output
  extract_lets -merge +onlyGivenNames fl_1 at hres
  have e_fl_1 : fl_1 = step19.fl := rfl
  clear_value fl_1
  -- step20: divstep!(last), invocation 20
  extract_lets -merge +onlyGivenNames step20 at hres
  have e_step20 : step20 = divstepLast ⟨two_delta_1, pf_2, pg_2, fl_1⟩ := rfl
  clear_value step20
  -- two_delta_2: divstep!(last), invocation 20 output
  extract_lets -merge +onlyGivenNames two_delta_2 at hres
  have e_two_delta_2 : two_delta_2 = step20.two_delta := rfl
  clear_value two_delta_2
  -- pf_3: divstep!(last), invocation 20 output
  extract_lets -merge +onlyGivenNames pf_3 at hres
  have e_pf_3 : pf_3 = step20.f := rfl
  clear_value pf_3
  -- pg_3: divstep!(last), invocation 20 output
  extract_lets -merge +onlyGivenNames pg_3 at hres
  have e_pg_3 : pg_3 = step20.g := rfl
  clear_value pg_3
  -- a00: add a00,pf,#0x100,lsl #12
  extract_lets -merge +onlyGivenNames a00 at hres
  have e_a00 : a00 = addw pf_3 0x100000 := rfl
  clear_value a00
  have b_a00 : a00 < 2^64 := by rw [e_a00]; exact addw_lt pf_3 0x100000
  -- a00_1: sbfx a00,a00,#21,#21
  extract_lets -merge +onlyGivenNames a00_1 at hres
  have e_a00_1 : a00_1 = sbfx a00 21 21 := rfl
  clear_value a00_1
  have b_a00_1 : a00_1 < 2^64 := by rw [e_a00_1]; exact sbfx_lt a00 21 21 b_a00
  -- a11: mov a11,#0x100000
  extract_lets -merge +onlyGivenNames a11 at hres
  have e_a11 : a11 = 0x100000 := rfl
  clear_value a11
  have b_a11 : a11 < 2^64 := by rw [e_a11]; decide
  -- a11_1: add a11,a11,a11,lsl #21
  extract_lets -merge +onlyGivenNames a11_1 at hres
  have e_a11_1 : a11_1 = addw a11 (lsl a11 21) := rfl
  clear_value a11_1
  have b_a11_1 : a11_1 < 2^64 := by rw [e_a11_1]; exact addw_lt a11 (lsl a11 21)
  -- a01: add a01,pf,a11
  extract_lets -merge +onlyGivenNames a01 at hres
  have e_a01 : a01 = addw pf_3 a11_1 := rfl
  clear_value a01
  have b_a01 : a01 < 2^64 := by rw [e_a01]; exact addw_lt pf_3 a11_1
  -- a01_1: asr a01,a01,#42
  extract_lets -merge +onlyGivenNames a01_1 at hres
  have e_a01_1 : a01_1 = asr a01 42 := rfl
  clear_value a01_1
  have b_a01_1 : a01_1 < 2^64 := by rw [e_a01_1]; exact asr_lt a01 42 b_a01
  -- a10: add a10,pg,#0x100,lsl #12
  extract_lets -merge +onlyGivenNames a10 at hres
  have e_a10 : a10 = addw pg_3 0x100000 := rfl
  clear_value a10
  have b_a10 : a10 < 2^64 := by rw [e_a10]; exact addw_lt pg_3 0x100000
  -- a10_1: sbfx a10,a10,#21,#21
  extract_lets -merge +onlyGivenNames a10_1 at hres
  have e_a10_1 : a10_1 = sbfx a10 21 21 := rfl
  clear_value a10_1
  have b_a10_1 : a10_1 < 2^64 := by rw [e_a10_1]; exact sbfx_lt a10 21 21 b_a10
  -- a11_2: add a11,pg,a11
  extract_lets -merge +onlyGivenNames a11_2 at hres
  have e_a11_2 : a11_2 = addw pg_3 a11_1 := rfl
  clear_value a11_2
  have b_a11_2 : a11_2 < 2^64 := by rw [e_a11_2]; exact addw_lt pg_3 a11_1
  -- a11_3: asr a11,a11,#42
  extract_lets -merge +onlyGivenNames a11_3 at hres
  have e_a11_3 : a11_3 = asr a11_2 42 := rfl
  clear_value a11_3
  have b_a11_3 : a11_3 < 2^64 := by rw [e_a11_3]; exact asr_lt a11_2 42 b_a11_2
  -- t: mul t,a00,f
  extract_lets -merge +onlyGivenNames t at hres
  have e_t : t = a00_1 * f % 2^64 := rfl
  clear_value t
  have b_t : t < 2^64 := by rw [e_t]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t2: mul t2,a01,g
  extract_lets -merge +onlyGivenNames t2 at hres
  have e_t2 : t2 = a01_1 * g % 2^64 := rfl
  clear_value t2
  have b_t2 : t2 < 2^64 := by rw [e_t2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- f_1: mul f,a10,f
  extract_lets -merge +onlyGivenNames f_1 at hres
  have e_f_1 : f_1 = a10_1 * f % 2^64 := rfl
  clear_value f_1
  have b_f_1 : f_1 < 2^64 := by rw [e_f_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- g_1: mul g,a11,g
  extract_lets -merge +onlyGivenNames g_1 at hres
  have e_g_1 : g_1 = a11_3 * g % 2^64 := rfl
  clear_value g_1
  have b_g_1 : g_1 < 2^64 := by rw [e_g_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- pf_4: add pf,t,t2
  extract_lets -merge +onlyGivenNames pf_4 at hres
  have e_pf_4 : pf_4 = addw t t2 := rfl
  clear_value pf_4
  have b_pf_4 : pf_4 < 2^64 := by rw [e_pf_4]; exact addw_lt t t2
  -- pg_4: add pg,f,g
  extract_lets -merge +onlyGivenNames pg_4 at hres
  have e_pg_4 : pg_4 = addw f_1 g_1 := rfl
  clear_value pg_4
  have b_pg_4 : pg_4 < 2^64 := by rw [e_pg_4]; exact addw_lt f_1 g_1
  -- f_2: asr f,pf,#20
  extract_lets -merge +onlyGivenNames f_2 at hres
  have e_f_2 : f_2 = asr pf_4 20 := rfl
  clear_value f_2
  have b_f_2 : f_2 < 2^64 := by rw [e_f_2]; exact asr_lt pf_4 20 b_pf_4
  -- g_2: asr g,pg,#20
  extract_lets -merge +onlyGivenNames g_2 at hres
  have e_g_2 : g_2 = asr pg_4 20 := rfl
  clear_value g_2
  have b_g_2 : g_2 < 2^64 := by rw [e_g_2]; exact asr_lt pg_4 20 b_pg_4
  -- pf_5: and pf,f,#0xfffff
  extract_lets -merge +onlyGivenNames pf_5 at hres
  have e_pf_5 : pf_5 = andw f_2 0xfffff := rfl
  clear_value pf_5
  have b_pf_5 : pf_5 < 2^64 := by rw [e_pf_5]; exact andw_lt f_2 0xfffff b_f_2 (by decide)
  -- pf_6: orr pf,pf,#0xfffffe0000000000
  extract_lets -merge +onlyGivenNames pf_6 at hres
  have e_pf_6 : pf_6 = orrw pf_5 0xfffffe0000000000 := rfl
  clear_value pf_6
  have b_pf_6 : pf_6 < 2^64 := by rw [e_pf_6]; exact orrw_lt pf_5 0xfffffe0000000000 b_pf_5 (by decide)
  -- pg_5: and pg,g,#0xfffff
  extract_lets -merge +onlyGivenNames pg_5 at hres
  have e_pg_5 : pg_5 = andw g_2 0xfffff := rfl
  clear_value pg_5
  have b_pg_5 : pg_5 < 2^64 := by rw [e_pg_5]; exact andw_lt g_2 0xfffff b_g_2 (by decide)
  -- pg_6: orr pg,pg,#0xc000000000000000
  extract_lets -merge +onlyGivenNames pg_6 at hres
  have e_pg_6 : pg_6 = orrw pg_5 0xc000000000000000 := rfl
  clear_value pg_6
  have b_pg_6 : pg_6 < 2^64 := by rw [e_pg_6]; exact orrw_lt pg_5 0xc000000000000000 b_pg_5 (by decide)
  -- fl_2: tst pg,#1
  extract_lets -merge +onlyGivenNames fl_2 at hres
  have e_fl_2 : fl_2 = tstFlags (andw pg_6 1) := rfl
  clear_value fl_2
  -- step39: divstep!(), invocations 21 to 39
  extract_lets -merge +onlyGivenNames step39 at hres
  have e_step39 : step39 = divstepRound^[19] ⟨two_delta_2, pf_6, pg_6, fl_2⟩ := rfl
  clear_value step39
  -- two_delta_3: divstep!(), invocations 21 to 39 output
  extract_lets -merge +onlyGivenNames two_delta_3 at hres
  have e_two_delta_3 : two_delta_3 = step39.two_delta := rfl
  clear_value two_delta_3
  -- pf_7: divstep!(), invocations 21 to 39 output
  extract_lets -merge +onlyGivenNames pf_7 at hres
  have e_pf_7 : pf_7 = step39.f := rfl
  clear_value pf_7
  -- pg_7: divstep!(), invocations 21 to 39 output
  extract_lets -merge +onlyGivenNames pg_7 at hres
  have e_pg_7 : pg_7 = step39.g := rfl
  clear_value pg_7
  -- fl_3: divstep!(), invocations 21 to 39 output
  extract_lets -merge +onlyGivenNames fl_3 at hres
  have e_fl_3 : fl_3 = step39.fl := rfl
  clear_value fl_3
  -- step40: divstep!(last), invocation 40
  extract_lets -merge +onlyGivenNames step40 at hres
  have e_step40 : step40 = divstepLast ⟨two_delta_3, pf_7, pg_7, fl_3⟩ := rfl
  clear_value step40
  -- two_delta_4: divstep!(last), invocation 40 output
  extract_lets -merge +onlyGivenNames two_delta_4 at hres
  have e_two_delta_4 : two_delta_4 = step40.two_delta := rfl
  clear_value two_delta_4
  -- pf_8: divstep!(last), invocation 40 output
  extract_lets -merge +onlyGivenNames pf_8 at hres
  have e_pf_8 : pf_8 = step40.f := rfl
  clear_value pf_8
  -- pg_8: divstep!(last), invocation 40 output
  extract_lets -merge +onlyGivenNames pg_8 at hres
  have e_pg_8 : pg_8 = step40.g := rfl
  clear_value pg_8
  -- b00: add b00,pf,#0x100,lsl #12
  extract_lets -merge +onlyGivenNames b00 at hres
  have e_b00 : b00 = addw pf_8 0x100000 := rfl
  clear_value b00
  have b_b00 : b00 < 2^64 := by rw [e_b00]; exact addw_lt pf_8 0x100000
  -- b00_1: sbfx b00,b00,#21,#21
  extract_lets -merge +onlyGivenNames b00_1 at hres
  have e_b00_1 : b00_1 = sbfx b00 21 21 := rfl
  clear_value b00_1
  have b_b00_1 : b00_1 < 2^64 := by rw [e_b00_1]; exact sbfx_lt b00 21 21 b_b00
  -- b11: mov b11,#0x100000
  extract_lets -merge +onlyGivenNames b11 at hres
  have e_b11 : b11 = 0x100000 := rfl
  clear_value b11
  have b_b11 : b11 < 2^64 := by rw [e_b11]; decide
  -- b11_1: add b11,b11,b11,lsl #21
  extract_lets -merge +onlyGivenNames b11_1 at hres
  have e_b11_1 : b11_1 = addw b11 (lsl b11 21) := rfl
  clear_value b11_1
  have b_b11_1 : b11_1 < 2^64 := by rw [e_b11_1]; exact addw_lt b11 (lsl b11 21)
  -- b01: add b01,pf,b11
  extract_lets -merge +onlyGivenNames b01 at hres
  have e_b01 : b01 = addw pf_8 b11_1 := rfl
  clear_value b01
  have b_b01 : b01 < 2^64 := by rw [e_b01]; exact addw_lt pf_8 b11_1
  -- b01_1: asr b01,b01,#42
  extract_lets -merge +onlyGivenNames b01_1 at hres
  have e_b01_1 : b01_1 = asr b01 42 := rfl
  clear_value b01_1
  have b_b01_1 : b01_1 < 2^64 := by rw [e_b01_1]; exact asr_lt b01 42 b_b01
  -- b10: add b10,pg,#0x100,lsl #12
  extract_lets -merge +onlyGivenNames b10 at hres
  have e_b10 : b10 = addw pg_8 0x100000 := rfl
  clear_value b10
  have b_b10 : b10 < 2^64 := by rw [e_b10]; exact addw_lt pg_8 0x100000
  -- b10_1: sbfx b10,b10,#21,#21
  extract_lets -merge +onlyGivenNames b10_1 at hres
  have e_b10_1 : b10_1 = sbfx b10 21 21 := rfl
  clear_value b10_1
  have b_b10_1 : b10_1 < 2^64 := by rw [e_b10_1]; exact sbfx_lt b10 21 21 b_b10
  -- b11_2: add b11,pg,b11
  extract_lets -merge +onlyGivenNames b11_2 at hres
  have e_b11_2 : b11_2 = addw pg_8 b11_1 := rfl
  clear_value b11_2
  have b_b11_2 : b11_2 < 2^64 := by rw [e_b11_2]; exact addw_lt pg_8 b11_1
  -- b11_3: asr b11,b11,#42
  extract_lets -merge +onlyGivenNames b11_3 at hres
  have e_b11_3 : b11_3 = asr b11_2 42 := rfl
  clear_value b11_3
  have b_b11_3 : b11_3 < 2^64 := by rw [e_b11_3]; exact asr_lt b11_2 42 b_b11_2
  -- t_1: mul t,b00,f
  extract_lets -merge +onlyGivenNames t_1 at hres
  have e_t_1 : t_1 = b00_1 * f_2 % 2^64 := rfl
  clear_value t_1
  have b_t_1 : t_1 < 2^64 := by rw [e_t_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t2_1: mul t2,b01,g
  extract_lets -merge +onlyGivenNames t2_1 at hres
  have e_t2_1 : t2_1 = b01_1 * g_2 % 2^64 := rfl
  clear_value t2_1
  have b_t2_1 : t2_1 < 2^64 := by rw [e_t2_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- f_3: mul f,b10,f
  extract_lets -merge +onlyGivenNames f_3 at hres
  have e_f_3 : f_3 = b10_1 * f_2 % 2^64 := rfl
  clear_value f_3
  have b_f_3 : f_3 < 2^64 := by rw [e_f_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- g_3: mul g,b11,g
  extract_lets -merge +onlyGivenNames g_3 at hres
  have e_g_3 : g_3 = b11_3 * g_2 % 2^64 := rfl
  clear_value g_3
  have b_g_3 : g_3 < 2^64 := by rw [e_g_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- pf_9: add pf,t,t2
  extract_lets -merge +onlyGivenNames pf_9 at hres
  have e_pf_9 : pf_9 = addw t_1 t2_1 := rfl
  clear_value pf_9
  have b_pf_9 : pf_9 < 2^64 := by rw [e_pf_9]; exact addw_lt t_1 t2_1
  -- pg_9: add pg,f,g
  extract_lets -merge +onlyGivenNames pg_9 at hres
  have e_pg_9 : pg_9 = addw f_3 g_3 := rfl
  clear_value pg_9
  have b_pg_9 : pg_9 < 2^64 := by rw [e_pg_9]; exact addw_lt f_3 g_3
  -- f_4: asr f,pf,#20
  extract_lets -merge +onlyGivenNames f_4 at hres
  have e_f_4 : f_4 = asr pf_9 20 := rfl
  clear_value f_4
  have b_f_4 : f_4 < 2^64 := by rw [e_f_4]; exact asr_lt pf_9 20 b_pf_9
  -- g_4: asr g,pg,#20
  extract_lets -merge +onlyGivenNames g_4 at hres
  have e_g_4 : g_4 = asr pg_9 20 := rfl
  clear_value g_4
  have b_g_4 : g_4 < 2^64 := by rw [e_g_4]; exact asr_lt pg_9 20 b_pg_9
  -- pf_10: and pf,f,#0xfffff
  extract_lets -merge +onlyGivenNames pf_10 at hres
  have e_pf_10 : pf_10 = andw f_4 0xfffff := rfl
  clear_value pf_10
  have b_pf_10 : pf_10 < 2^64 := by rw [e_pf_10]; exact andw_lt f_4 0xfffff b_f_4 (by decide)
  -- pf_11: orr pf,pf,#0xfffffe0000000000
  extract_lets -merge +onlyGivenNames pf_11 at hres
  have e_pf_11 : pf_11 = orrw pf_10 0xfffffe0000000000 := rfl
  clear_value pf_11
  have b_pf_11 : pf_11 < 2^64 := by rw [e_pf_11]; exact orrw_lt pf_10 0xfffffe0000000000 b_pf_10 (by decide)
  -- pg_10: and pg,g,#0xfffff
  extract_lets -merge +onlyGivenNames pg_10 at hres
  have e_pg_10 : pg_10 = andw g_4 0xfffff := rfl
  clear_value pg_10
  have b_pg_10 : pg_10 < 2^64 := by rw [e_pg_10]; exact andw_lt g_4 0xfffff b_g_4 (by decide)
  -- pg_11: orr pg,pg,#0xc000000000000000
  extract_lets -merge +onlyGivenNames pg_11 at hres
  have e_pg_11 : pg_11 = orrw pg_10 0xc000000000000000 := rfl
  clear_value pg_11
  have b_pg_11 : pg_11 < 2^64 := by rw [e_pg_11]; exact orrw_lt pg_10 0xc000000000000000 b_pg_10 (by decide)
  -- fl_4: tst pg,#1
  extract_lets -merge +onlyGivenNames fl_4 at hres
  have e_fl_4 : fl_4 = tstFlags (andw pg_11 1) := rfl
  clear_value fl_4
  -- step50: divstep!(), invocations 41 to 50
  extract_lets -merge +onlyGivenNames step50 at hres
  have e_step50 : step50 = divstepRound^[10] ⟨two_delta_4, pf_11, pg_11, fl_4⟩ := rfl
  clear_value step50
  -- two_delta_5: divstep!(), invocations 41 to 50 output
  extract_lets -merge +onlyGivenNames two_delta_5 at hres
  have e_two_delta_5 : two_delta_5 = step50.two_delta := rfl
  clear_value two_delta_5
  -- pf_12: divstep!(), invocations 41 to 50 output
  extract_lets -merge +onlyGivenNames pf_12 at hres
  have e_pf_12 : pf_12 = step50.f := rfl
  clear_value pf_12
  -- pg_12: divstep!(), invocations 41 to 50 output
  extract_lets -merge +onlyGivenNames pg_12 at hres
  have e_pg_12 : pg_12 = step50.g := rfl
  clear_value pg_12
  -- fl_5: divstep!(), invocations 41 to 50 output
  extract_lets -merge +onlyGivenNames fl_5 at hres
  have e_fl_5 : fl_5 = step50.fl := rfl
  clear_value fl_5
  -- f_5: mul f,b00,a00
  extract_lets -merge +onlyGivenNames f_5 at hres
  have e_f_5 : f_5 = b00_1 * a00_1 % 2^64 := rfl
  clear_value f_5
  have b_f_5 : f_5 < 2^64 := by rw [e_f_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- g_5: mul g,b00,a01
  extract_lets -merge +onlyGivenNames g_5 at hres
  have e_g_5 : g_5 = b00_1 * a01_1 % 2^64 := rfl
  clear_value g_5
  have b_g_5 : g_5 < 2^64 := by rw [e_g_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t_2: mul t,b10,a00
  extract_lets -merge +onlyGivenNames t_2 at hres
  have e_t_2 : t_2 = b10_1 * a00_1 % 2^64 := rfl
  clear_value t_2
  have b_t_2 : t_2 < 2^64 := by rw [e_t_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t2_2: mul t2,b10,a01
  extract_lets -merge +onlyGivenNames t2_2 at hres
  have e_t2_2 : t2_2 = b10_1 * a01_1 % 2^64 := rfl
  clear_value t2_2
  have b_t2_2 : t2_2 < 2^64 := by rw [e_t2_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- a00_2: madd a00,b01,a10,f
  extract_lets -merge +onlyGivenNames a00_2 at hres
  have e_a00_2 : a00_2 = madd b01_1 a10_1 f_5 := rfl
  clear_value a00_2
  have b_a00_2 : a00_2 < 2^64 := by rw [e_a00_2]; exact madd_lt b01_1 a10_1 f_5
  -- a01_2: madd a01,b01,a11,g
  extract_lets -merge +onlyGivenNames a01_2 at hres
  have e_a01_2 : a01_2 = madd b01_1 a11_3 g_5 := rfl
  clear_value a01_2
  have b_a01_2 : a01_2 < 2^64 := by rw [e_a01_2]; exact madd_lt b01_1 a11_3 g_5
  -- c10: madd c10,b11,a10,t
  extract_lets -merge +onlyGivenNames c10 at hres
  have e_c10 : c10 = madd b11_3 a10_1 t_2 := rfl
  clear_value c10
  have b_c10 : c10 < 2^64 := by rw [e_c10]; exact madd_lt b11_3 a10_1 t_2
  -- c11: madd c11,b11,a11,t2
  extract_lets -merge +onlyGivenNames c11 at hres
  have e_c11 : c11 = madd b11_3 a11_3 t2_2 := rfl
  clear_value c11
  have b_c11 : c11 < 2^64 := by rw [e_c11]; exact madd_lt b11_3 a11_3 t2_2
  -- step58: divstep!(), invocations 51 to 58
  extract_lets -merge +onlyGivenNames step58 at hres
  have e_step58 : step58 = divstepRound^[8] ⟨two_delta_5, pf_12, pg_12, fl_5⟩ := rfl
  clear_value step58
  -- two_delta_6: divstep!(), invocations 51 to 58 output
  extract_lets -merge +onlyGivenNames two_delta_6 at hres
  have e_two_delta_6 : two_delta_6 = step58.two_delta := rfl
  clear_value two_delta_6
  -- pf_13: divstep!(), invocations 51 to 58 output
  extract_lets -merge +onlyGivenNames pf_13 at hres
  have e_pf_13 : pf_13 = step58.f := rfl
  clear_value pf_13
  -- pg_13: divstep!(), invocations 51 to 58 output
  extract_lets -merge +onlyGivenNames pg_13 at hres
  have e_pg_13 : pg_13 = step58.g := rfl
  clear_value pg_13
  -- fl_6: divstep!(), invocations 51 to 58 output
  extract_lets -merge +onlyGivenNames fl_6 at hres
  have e_fl_6 : fl_6 = step58.fl := rfl
  clear_value fl_6
  -- step59: divstep!(last), invocation 59
  extract_lets -merge +onlyGivenNames step59 at hres
  have e_step59 : step59 = divstepLast ⟨two_delta_6, pf_13, pg_13, fl_6⟩ := rfl
  clear_value step59
  -- two_delta_7: divstep!(last), invocation 59 output
  extract_lets -merge +onlyGivenNames two_delta_7 at hres
  have e_two_delta_7 : two_delta_7 = step59.two_delta := rfl
  clear_value two_delta_7
  -- pf_14: divstep!(last), invocation 59 output
  extract_lets -merge +onlyGivenNames pf_14 at hres
  have e_pf_14 : pf_14 = step59.f := rfl
  clear_value pf_14
  -- pg_14: divstep!(last), invocation 59 output
  extract_lets -merge +onlyGivenNames pg_14 at hres
  have e_pg_14 : pg_14 = step59.g := rfl
  clear_value pg_14
  -- b00_2: add b00,pf,#0x100,lsl #12
  extract_lets -merge +onlyGivenNames b00_2 at hres
  have e_b00_2 : b00_2 = addw pf_14 0x100000 := rfl
  clear_value b00_2
  have b_b00_2 : b00_2 < 2^64 := by rw [e_b00_2]; exact addw_lt pf_14 0x100000
  -- b00_3: sbfx b00,b00,#22,#21
  extract_lets -merge +onlyGivenNames b00_3 at hres
  have e_b00_3 : b00_3 = sbfx b00_2 22 21 := rfl
  clear_value b00_3
  have b_b00_3 : b00_3 < 2^64 := by rw [e_b00_3]; exact sbfx_lt b00_2 22 21 b_b00_2
  -- b11_4: mov b11,#0x100000
  extract_lets -merge +onlyGivenNames b11_4 at hres
  have e_b11_4 : b11_4 = 0x100000 := rfl
  clear_value b11_4
  have b_b11_4 : b11_4 < 2^64 := by rw [e_b11_4]; decide
  -- b11_5: add b11,b11,b11,lsl #21
  extract_lets -merge +onlyGivenNames b11_5 at hres
  have e_b11_5 : b11_5 = addw b11_4 (lsl b11_4 21) := rfl
  clear_value b11_5
  have b_b11_5 : b11_5 < 2^64 := by rw [e_b11_5]; exact addw_lt b11_4 (lsl b11_4 21)
  -- b01_2: add b01,pf,b11
  extract_lets -merge +onlyGivenNames b01_2 at hres
  have e_b01_2 : b01_2 = addw pf_14 b11_5 := rfl
  clear_value b01_2
  have b_b01_2 : b01_2 < 2^64 := by rw [e_b01_2]; exact addw_lt pf_14 b11_5
  -- b01_3: asr b01,b01,#43
  extract_lets -merge +onlyGivenNames b01_3 at hres
  have e_b01_3 : b01_3 = asr b01_2 43 := rfl
  clear_value b01_3
  have b_b01_3 : b01_3 < 2^64 := by rw [e_b01_3]; exact asr_lt b01_2 43 b_b01_2
  -- b10_2: add b10,pg,#0x100,lsl #12
  extract_lets -merge +onlyGivenNames b10_2 at hres
  have e_b10_2 : b10_2 = addw pg_14 0x100000 := rfl
  clear_value b10_2
  have b_b10_2 : b10_2 < 2^64 := by rw [e_b10_2]; exact addw_lt pg_14 0x100000
  -- b10_3: sbfx b10,b10,#22,#21
  extract_lets -merge +onlyGivenNames b10_3 at hres
  have e_b10_3 : b10_3 = sbfx b10_2 22 21 := rfl
  clear_value b10_3
  have b_b10_3 : b10_3 < 2^64 := by rw [e_b10_3]; exact sbfx_lt b10_2 22 21 b_b10_2
  -- b11_6: add b11,pg,b11
  extract_lets -merge +onlyGivenNames b11_6 at hres
  have e_b11_6 : b11_6 = addw pg_14 b11_5 := rfl
  clear_value b11_6
  have b_b11_6 : b11_6 < 2^64 := by rw [e_b11_6]; exact addw_lt pg_14 b11_5
  -- b11_7: asr b11,b11,#43
  extract_lets -merge +onlyGivenNames b11_7 at hres
  have e_b11_7 : b11_7 = asr b11_6 43 := rfl
  clear_value b11_7
  have b_b11_7 : b11_7 < 2^64 := by rw [e_b11_7]; exact asr_lt b11_6 43 b_b11_6
  -- f_6: mneg f,b00,a00
  extract_lets -merge +onlyGivenNames f_6 at hres
  have e_f_6 : f_6 = mneg b00_3 a00_2 := rfl
  clear_value f_6
  have b_f_6 : f_6 < 2^64 := by rw [e_f_6]; exact mneg_lt b00_3 a00_2
  -- g_6: mneg g,b00,a01
  extract_lets -merge +onlyGivenNames g_6 at hres
  have e_g_6 : g_6 = mneg b00_3 a01_2 := rfl
  clear_value g_6
  have b_g_6 : g_6 < 2^64 := by rw [e_g_6]; exact mneg_lt b00_3 a01_2
  -- pf_15: mneg pf,b10,a00
  extract_lets -merge +onlyGivenNames pf_15 at hres
  have e_pf_15 : pf_15 = mneg b10_3 a00_2 := rfl
  clear_value pf_15
  have b_pf_15 : pf_15 < 2^64 := by rw [e_pf_15]; exact mneg_lt b10_3 a00_2
  -- pg_15: mneg pg,b10,a01
  extract_lets -merge +onlyGivenNames pg_15 at hres
  have e_pg_15 : pg_15 = mneg b10_3 a01_2 := rfl
  clear_value pg_15
  have b_pg_15 : pg_15 < 2^64 := by rw [e_pg_15]; exact mneg_lt b10_3 a01_2
  -- u: msub u,b01,c10,f
  extract_lets -merge +onlyGivenNames u at hres
  have e_u : u = msub b01_3 c10 f_6 := rfl
  clear_value u
  have b_u : u < 2^64 := by rw [e_u]; exact msub_lt b01_3 c10 f_6
  -- v: msub v,b01,c11,g
  extract_lets -merge +onlyGivenNames v at hres
  have e_v : v = msub b01_3 c11 g_6 := rfl
  clear_value v
  have b_v : v < 2^64 := by rw [e_v]; exact msub_lt b01_3 c11 g_6
  -- q: msub q,b11,c10,pf
  extract_lets -merge +onlyGivenNames q at hres
  have e_q : q = msub b11_7 c10 pf_15 := rfl
  clear_value q
  have b_q : q < 2^64 := by rw [e_q]; exact msub_lt b11_7 c10 pf_15
  -- r: msub r,b11,c11,pg
  extract_lets -merge +onlyGivenNames r at hres
  have e_r : r = msub b11_7 c11 pg_15 := rfl
  clear_value r
  have b_r : r < 2^64 := by rw [e_r]; exact msub_lt b11_7 c11 pg_15
  subst hres
  -- BEGIN conclusion
  -- Each batch's bindings are the round iterated, then the last step; the third batch's run
  -- is split where the second batch's products are scheduled.
  have i20 : step20 = divstepLast (divstepRound^[19] ⟨two_delta', pf_1, pg_1, fl⟩) := by
    rw [e_step20, e_two_delta_1, e_pf_2, e_pg_2, e_fl_1, e_step19]
  have i40 : step40 = divstepLast (divstepRound^[19] ⟨two_delta_2, pf_6, pg_6, fl_2⟩) := by
    rw [e_step40, e_two_delta_3, e_pf_7, e_pg_7, e_fl_3, e_step39]
  have i59 : step59 = divstepLast (divstepRound^[18] ⟨two_delta_4, pf_11, pg_11, fl_4⟩) := by
    rw [e_step59, e_two_delta_6, e_pf_13, e_pg_13, e_fl_6, e_step58, e_two_delta_5, e_pf_12, e_pg_12, e_fl_5,
      e_step50]
    show divstepLast (divstepRound^[8] (divstepRound^[10] ⟨two_delta_4, pf_11, pg_11, fl_4⟩)) = _
    rw [← Function.iterate_add_apply]
  have hpos64 : (0 : ℤ) < 2^64 := by norm_num
  have hsDa : |s.two_delta| < 2^61 := hsD
  rw [abs_lt] at hsD
  -- Batch 1: the packed start is `packedStart s₀` for the truncation `s₀` of `s`.
  set s₀ : State := ⟨s.two_delta, s.f % 2^20, s.g % 2^20⟩ with hs₀
  have hs₀f : s₀.f % 2 = 1 := by show (s.f % 2^20) % 2 = 1; omega
  have hs₀fb : 0 ≤ s₀.f ∧ s₀.f < 2^20 :=
    ⟨Int.emod_nonneg _ (by norm_num), Int.emod_lt_of_pos _ (by norm_num)⟩
  have hs₀gb : 0 ≤ s₀.g ∧ s₀.g < 2^20 :=
    ⟨Int.emod_nonneg _ (by norm_num), Int.emod_lt_of_pos _ (by norm_num)⟩
  set P0 : State := packedStart s₀ with hP0
  have hP0f : P0.f % 2 = 1 := by show (s.f % 2^20 - 2^41) % 2 = 1; omega
  have hP0g : |P0.g| < 2^63 := by
    show |s.g % 2^20 - 2^62| < 2^63; rw [abs_lt]; constructor <;> omega
  have hP0D : |P0.two_delta| + 2 * (19 + 1) < 2^62 := by show |s.two_delta| + 2 * (19 + 1) < 2^62; omega
  have hw_d : (two_delta' : ℤ) = P0.two_delta % 2^64 := by rw [e_two_delta']; exact ed
  have hw_f : (pf_1 : ℤ) = P0.f % 2^64 := by
    rw [e_pf_1, e_pf, e_f, pack_f_word]; show _ = (s.f % 2^20 - 2^41) % 2^64; omega
  have hw_g : (pg_1 : ℤ) = P0.g % 2^64 := by
    rw [e_pg_1, e_pg, e_g, pack_g_word]; show _ = (s.g % 2^20 - 2^62) % 2^64; omega
  have hw_z : fl.z = if P0.g % 2 = 0 then 1 else 0 := by rw [e_fl]; exact tst_one_z pg_1 P0.g hw_g
  obtain ⟨hb1, hd1, hf1, hg1⟩ := batch_words 19 P0 hP0f hsd hP0D hP0g
    (fun j _ => Inversion.divsteps_packedStart_g_abs_lt j s₀ hs₀f hs₀fb hs₀gb)
    ⟨two_delta', pf_1, pg_1, fl⟩ ⟨b_two_delta', b_pf_1, b_pg_1⟩ hw_d hw_f hw_g hw_z
  have b_two_delta_2 : two_delta_2 < 2^64 := by rw [e_two_delta_2, i20]; exact hb1.1
  rw [← i20] at hd1 hf1 hg1
  rw [← e_two_delta_2] at hd1
  rw [← e_pf_3] at hf1
  rw [← e_pg_3] at hg1
  clear hb1 i20 e_two_delta_2 e_pf_3 e_pg_3 step20 e_step20 e_two_delta_1 e_pf_2 e_pg_2 e_fl_1 two_delta_1 pf_2 pg_2 fl_1
    e_step19 step19
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
  have hd20 : (two_delta_2 : ℤ) = s20.two_delta % 2^64 := by
    rw [hd1, hpk1]; exact congrArg (· % 2^64) hdl1.symm
  obtain ⟨hA00, hA01⟩ := decode20 pf_3 _ _ _ _ hf1 (by rw [hpk1, hMl1])
    (by rw [abs_le] at hφ1; rw [abs_lt]; constructor <;> omega) hA1 hB1
  obtain ⟨hA10, hA11⟩ := decode20 pg_3 _ _ _ _ hg1 (by rw [hpk1, hMl1])
    (by rw [abs_le] at hγ1; rw [abs_lt]; constructor <;> omega) hC1 hD1
  rw [← e_a00, ← e_a00_1] at hA00
  rw [← e_a11, ← e_a11_1, ← e_a01, ← e_a01_1] at hA01
  rw [← e_a10, ← e_a10_1] at hA10
  rw [← e_a11, ← e_a11_1, ← e_a11_2, ← e_a11_3] at hA11
  -- The next low words: `-(M1 (f, g))` over `2^20`, so they carry `-f_20`, `-g_20` modulo `2^44`.
  obtain ⟨hMf1, hMg1⟩ := Inversion.M_spec 20 s hsf
  rw [hM1, hs20] at hMf1 hMg1
  have hf20 : s20.f % 2 = 1 := by rw [← hs20]; exact Inversion.divsteps_f_odd 20 s hsf
  have ht : (t : ℤ) = ((-M1.u) * s.f) % 2^64 := by
    rw [e_t]; exact mul_word _ _ _ _ hA00 (by rw [e_f]; exact ef0)
  have ht2 : (t2 : ℤ) = ((-M1.v) * s.g) % 2^64 := by
    rw [e_t2]; exact mul_word _ _ _ _ hA01 (by rw [e_g]; exact eg0)
  have hf1' : (f_1 : ℤ) = ((-M1.q) * s.f) % 2^64 := by
    rw [e_f_1]; exact mul_word _ _ _ _ hA10 (by rw [e_f]; exact ef0)
  have hg1' : (g_1 : ℤ) = ((-M1.r) * s.g) % 2^64 := by
    rw [e_g_1]; exact mul_word _ _ _ _ hA11 (by rw [e_g]; exact eg0)
  have hf2 : (f_2 : ℤ) % 2^44 = (-s20.f) % 2^44 := by
    rw [e_f_2]; apply asr20_word
    rw [e_pf_4, addw_word _ _ _ _ ht ht2]
    exact congrArg (· % 2^64) (by linear_combination hMf1)
  have hg2 : (g_2 : ℤ) % 2^44 = (-s20.g) % 2^44 := by
    rw [e_g_2]; apply asr20_word
    rw [e_pg_4, addw_word _ _ _ _ hf1' hg1']
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
  have hs20D := Inversion.divsteps_two_delta_abs_le 20 s
  rw [hs20] at hs20D
  push_cast at hs20D
  have hs20Db := abs_le.mp (le_trans hs20D (le_refl _))
  have hP1g : |P1.g| < 2^63 := by
    show |(-s20.g) % 2^20 - 2^62| < 2^63; rw [abs_lt]; constructor <;> omega
  have hP1D : |P1.two_delta| + 2 * (19 + 1) < 2^62 := by
    show |s20.two_delta| + 2 * (19 + 1) < 2^62; omega
  have hw_f2 : (pf_6 : ℤ) = P1.f % 2^64 := by
    rw [e_pf_6, e_pf_5, pack_f_word]; show _ = ((-s20.f) % 2^20 - 2^41) % 2^64; omega
  have hw_g2 : (pg_6 : ℤ) = P1.g % 2^64 := by
    rw [e_pg_6, e_pg_5, pack_g_word]; show _ = ((-s20.g) % 2^20 - 2^62) % 2^64; omega
  have hw_z2 : fl_2.z = if P1.g % 2 = 0 then 1 else 0 := by
    rw [e_fl_2]; exact tst_one_z pg_6 P1.g hw_g2
  obtain ⟨hb2, hd2, hf2', hg2'⟩ := batch_words 19 P1 hP1f hP1d hP1D hP1g
    (fun j _ => Inversion.divsteps_packedStart_g_abs_lt j s₁ hs₁f hs₁fb hs₁gb)
    ⟨two_delta_2, pf_6, pg_6, fl_2⟩ ⟨b_two_delta_2, b_pf_6, b_pg_6⟩ hd20 hw_f2 hw_g2 hw_z2
  have b_two_delta_4 : two_delta_4 < 2^64 := by rw [e_two_delta_4, i40]; exact hb2.1
  rw [← i40] at hd2 hf2' hg2'
  rw [← e_two_delta_4] at hd2
  rw [← e_pf_8] at hf2'
  rw [← e_pg_8] at hg2'
  clear hb2 i40 e_two_delta_4 e_pf_8 e_pg_8 step40 e_step40 e_two_delta_3 e_pf_7 e_pg_7 e_fl_3 two_delta_3 pf_7 pg_7 fl_3
    e_step39 step39
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
  generalize hs40 : divsteps 40 s = s40
  have hs40' : divsteps 20 s20 = s40 := by
    rw [← hs40, ← hs20]; exact (Inversion.divsteps_add 20 20 s).symm
  have hd40 : (two_delta_4 : ℤ) = s40.two_delta % 2^64 := by
    rw [hd2, hpk2]
    exact congrArg (· % 2^64) (by rw [← hdl2, hdN, hs40'] : (divsteps 20 s₁).two_delta = s40.two_delta)
  obtain ⟨hB00, hB01⟩ := decode20 pf_8 _ _ _ _ hf2' (by rw [hpk2, ← hMl2, hMN])
    (by rw [abs_le] at hφ2; rw [abs_lt]; constructor <;> omega) hA2 hB2
  obtain ⟨hB10, hB11⟩ := decode20 pg_8 _ _ _ _ hg2' (by rw [hpk2, ← hMl2, hMN])
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
  have ht_1 : (t_1 : ℤ) = ((-M2.u) * f_2) % 2^64 := by
    rw [e_t_1]; exact mul_word _ _ _ _ hB00 hf2c
  have ht2_1 : (t2_1 : ℤ) = ((-M2.v) * g_2) % 2^64 := by
    rw [e_t2_1]; exact mul_word _ _ _ _ hB01 hg2c
  have hf_3 : (f_3 : ℤ) = ((-M2.q) * f_2) % 2^64 := by
    rw [e_f_3]; exact mul_word _ _ _ _ hB10 hf2c
  have hg_3 : (g_3 : ℤ) = ((-M2.r) * g_2) % 2^64 := by
    rw [e_g_3]; exact mul_word _ _ _ _ hB11 hg2c
  have hf2m : (f_2 : ℤ) ≡ -s20.f [ZMOD 2^44] := hf2
  have hg2m : (g_2 : ℤ) ≡ -s20.g [ZMOD 2^44] := hg2
  have h44 : (2 : ℤ)^44 ∣ 2^64 := pow_dvd_pow 2 (by norm_num)
  have hf4 : (f_4 : ℤ) % 2^24 = s40.f % 2^24 := by
    rw [e_f_4]; apply asr20_word44 _ _
    rw [e_pf_9, addw_word _ _ _ _ ht_1 ht2_1, Int.emod_emod_of_dvd _ h44]
    have h := (hf2m.mul_left (-M2.u)).add (hg2m.mul_left (-M2.v))
    unfold Int.ModEq at h; rw [h]
    exact congrArg (· % 2^44) (by linear_combination -hMf2)
  have hg4 : (g_4 : ℤ) % 2^24 = s40.g % 2^24 := by
    rw [e_g_4]; apply asr20_word44 _ _
    rw [e_pg_9, addw_word _ _ _ _ hf_3 hg_3, Int.emod_emod_of_dvd _ h44]
    have h := (hf2m.mul_left (-M2.q)).add (hg2m.mul_left (-M2.r))
    unfold Int.ModEq at h; rw [h]
    exact congrArg (· % 2^44) (by linear_combination -hMg2)
  -- The products of the two negated matrices are `M2 · M1`.
  set M2M1 : Inversion.Mat2 := M2.mul M1 with hM2M1def
  have hf5 : (f_5 : ℤ) = ((-M2.u) * (-M1.u)) % 2^64 := by
    rw [e_f_5]; exact mul_word _ _ _ _ hB00 hA00
  have hg5 : (g_5 : ℤ) = ((-M2.u) * (-M1.v)) % 2^64 := by
    rw [e_g_5]; exact mul_word _ _ _ _ hB00 hA01
  have ht_2 : (t_2 : ℤ) = ((-M2.q) * (-M1.u)) % 2^64 := by
    rw [e_t_2]; exact mul_word _ _ _ _ hB10 hA00
  have ht2_2 : (t2_2 : ℤ) = ((-M2.q) * (-M1.v)) % 2^64 := by
    rw [e_t2_2]; exact mul_word _ _ _ _ hB10 hA01
  have hM2M1u : (a00_2 : ℤ) = M2M1.u % 2^64 := by
    rw [e_a00_2, madd_word _ _ _ _ _ _ hB01 hA10 hf5]
    exact congrArg (· % 2^64) (by simp only [hM2M1def, Inversion.Mat2.mul]; ring)
  have hM2M1v : (a01_2 : ℤ) = M2M1.v % 2^64 := by
    rw [e_a01_2, madd_word _ _ _ _ _ _ hB01 hA11 hg5]
    exact congrArg (· % 2^64) (by simp only [hM2M1def, Inversion.Mat2.mul]; ring)
  have hM2M1q : (c10 : ℤ) = M2M1.q % 2^64 := by
    rw [e_c10, madd_word _ _ _ _ _ _ hB11 hA10 ht_2]
    exact congrArg (· % 2^64) (by simp only [hM2M1def, Inversion.Mat2.mul]; ring)
  have hM2M1r : (c11 : ℤ) = M2M1.r % 2^64 := by
    rw [e_c11, madd_word _ _ _ _ _ _ hB11 hA11 ht2_2]
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
  have hs40D := Inversion.divsteps_two_delta_abs_le 40 s
  rw [hs40] at hs40D
  push_cast at hs40D
  have hs40Db := abs_le.mp (le_trans hs40D (le_refl _))
  have hP2g : |P2.g| < 2^63 := by
    show |s40.g % 2^20 - 2^62| < 2^63; rw [abs_lt]; constructor <;> omega
  have hP2D : |P2.two_delta| + 2 * (18 + 1) < 2^62 := by
    show |s40.two_delta| + 2 * (18 + 1) < 2^62; omega
  have hw_f3 : (pf_11 : ℤ) = P2.f % 2^64 := by
    rw [e_pf_11, e_pf_10, pack_f_word]; show _ = (s40.f % 2^20 - 2^41) % 2^64; omega
  have hw_g3 : (pg_11 : ℤ) = P2.g % 2^64 := by
    rw [e_pg_11, e_pg_10, pack_g_word]; show _ = (s40.g % 2^20 - 2^62) % 2^64; omega
  have hw_z3 : fl_4.z = if P2.g % 2 = 0 then 1 else 0 := by
    rw [e_fl_4]; exact tst_one_z pg_11 P2.g hw_g3
  obtain ⟨hb3, hd3, hf3', hg3'⟩ := batch_words 18 P2 hP2f hP2d hP2D hP2g
    (fun j _ => Inversion.divsteps_packedStart_g_abs_lt j s₂ hs₂f hs₂fb hs₂gb)
    ⟨two_delta_4, pf_11, pg_11, fl_4⟩ ⟨b_two_delta_4, b_pf_11, b_pg_11⟩ hd40 hw_f3 hw_g3 hw_z3
  rw [← i59] at hd3 hf3' hg3'
  rw [← e_two_delta_7] at hd3
  rw [← e_pf_14] at hf3'
  rw [← e_pg_14] at hg3'
  clear hb3 i59 e_two_delta_7 e_pf_14 e_pg_14 step59 e_step59 e_two_delta_6 e_pf_13 e_pg_13 e_fl_6 two_delta_6 pf_13 pg_13
    fl_6 e_step58 step58 e_two_delta_5 e_pf_12 e_pg_12 e_fl_5 two_delta_5 pf_12 pg_12 fl_5 e_step50 step50
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
  have hd59 : (two_delta_7 : ℤ) = (divsteps 59 s).two_delta % 2^64 := by
    rw [hd3, hpk3]
    exact congrArg (· % 2^64)
      (by rw [← hdl3, ← hs40, ← Inversion.divsteps_add] : (divsteps 19 s₂).two_delta = (divsteps 59 s).two_delta)
  obtain ⟨hC00, hC01⟩ := decode19 pf_14 _ _ _ _ hf3' (by rw [hpk3, hMl3])
    (by rw [abs_le] at hφ3; rw [abs_lt]; constructor <;> omega) hA3 hB3
  obtain ⟨hC10, hC11⟩ := decode19 pg_14 _ _ _ _ hg3' (by rw [hpk3, hMl3])
    (by rw [abs_le] at hγ3; rw [abs_lt]; constructor <;> omega) hC3 hD3
  rw [← e_b00_2, ← e_b00_3] at hC00
  rw [← e_b11_4, ← e_b11_5, ← e_b01_2, ← e_b01_3] at hC01
  rw [← e_b10_2, ← e_b10_3] at hC10
  rw [← e_b11_4, ← e_b11_5, ← e_b11_6, ← e_b11_7] at hC11
  -- The final products: `mneg` and `msub` restore the sign of the third matrix.
  have hM59 : M 59 s = M3.mul M2M1 := by
    rw [← hM3, hM2M1def, ← hM2, ← hM1, ← hs40, ← hs20, ← Inversion.M_add, ← Inversion.M_add]
  have hf6 : (f_6 : ℤ) = (-((-M3.u) * M2M1.u)) % 2^64 := by
    rw [e_f_6]; exact mneg_word _ _ _ _ hC00 hM2M1u
  have hg6 : (g_6 : ℤ) = (-((-M3.u) * M2M1.v)) % 2^64 := by
    rw [e_g_6]; exact mneg_word _ _ _ _ hC00 hM2M1v
  have hpf67 : (pf_15 : ℤ) = (-((-M3.q) * M2M1.u)) % 2^64 := by
    rw [e_pf_15]; exact mneg_word _ _ _ _ hC10 hM2M1u
  have hpg67 : (pg_15 : ℤ) = (-((-M3.q) * M2M1.v)) % 2^64 := by
    rw [e_pg_15]; exact mneg_word _ _ _ _ hC10 hM2M1v
  refine ⟨hd59, ?_, ?_, ?_, ?_⟩
  · rw [e_u, msub_word _ _ _ _ _ _ hC01 hM2M1q hf6, hM59]
    exact congrArg (· % 2^64) (by simp only [Inversion.Mat2.mul]; ring)
  · rw [e_v, msub_word _ _ _ _ _ _ hC01 hM2M1r hg6, hM59]
    exact congrArg (· % 2^64) (by simp only [Inversion.Mat2.mul]; ring)
  · rw [e_q, msub_word _ _ _ _ _ _ hC11 hM2M1q hpf67, hM59]
    exact congrArg (· % 2^64) (by simp only [Inversion.Mat2.mul]; ring)
  · rw [e_r, msub_word _ _ _ _ _ _ hC11 hM2M1r hpg67, hM59]
    exact congrArg (· % 2^64) (by simp only [Inversion.Mat2.mul]; ring)
  -- END conclusion

end PastaCurves.AArch64
