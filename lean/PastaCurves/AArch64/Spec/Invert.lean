import PastaCurves.AArch64.Spec.Divstep59
import PastaCurves.AArch64.Spec.FgRow
import PastaCurves.AArch64.Spec.Amontred
import PastaCurves.AArch64.Spec.CondSub
import PastaCurves.AArch64.Compositions
import PastaCurves.Inversion.Model

/-!
# The inversion's composition is the model

`invert` of `Compositions.lean` mirrors the Rust driver: nine rounds of the six blocks, then the
last round. `invert_eq_model` shows that it computes `montInvModel` of `Inversion/Model.lean`,
whose Theorem 12 (`montInv_spec`) then applies to the crate's entry point.

The proof carries a relation between the word state of a round and the model's `RoundState`:
the same `f`, `g`, `d`, `e`, and `two_delta` as a word. Each round preserves it by the six block
theorems, whose hypotheses come from the model's invariant (`rounds_invariant`) and from the
bounds of the true divstep state after `59 i` steps. The last round is the same with the sign word
xored into the masks of its row, which negates the represented entries exactly when the model's
`finalD` negates the row.

The proofs name the true state, the model's state, the matrix, and the blocks' results by
`generalize` rather than `set`: those terms are iterates and long `let` chains, and a `let`-bound
name lets `omega`, `rw`, and `show` unfold them to the recursion limit. -/

set_option exponentiation.threshold 400

namespace PastaCurves.AArch64

open Inversion (State divsteps M RoundState rounds trueState round initState startE montInvModel
  signWordOf finalD updateFG updateDE amontredZ Mat2)

/-- The word state of a round carries the model's: the same `f`, `g`, `d`, `e`, and `two_delta` as a
word. -/
def Carries (st : InvertState) (rs : RoundState) : Prop :=
  (st.two_delta : ℤ) = rs.two_delta % 2^64 ∧ st.f = rs.f ∧ st.g = rs.g ∧ st.d = rs.d ∧ st.e = rs.e

/-- What the blocks need of the true divstep state after `59 i` steps, for `i ≤ 9`: `f` odd,
`two_delta` odd and small, and `f`, `g` below `2^256` in magnitude. -/
theorem trueState_facts (F : PastaField) (x : Limbs) (hx : x.Bounded) (i : ℕ) (hi : i ≤ 9) :
    (trueState F x i).f % 2 = 1 ∧ (trueState F x i).two_delta % 2 = 1 ∧ |(trueState F x i).two_delta| < 2^61 ∧
      |(trueState F x i).f| < 2^256 ∧ |(trueState F x i).g| < 2^256 := by
  have hp_odd : (F.modulus.toNat : ℤ) % 2 = 1 := by exact_mod_cast F.modulus_odd
  have hp255 : (F.modulus.toNat : ℤ) < 2^255 := by exact_mod_cast F.modulus_lt
  have hx256 : (x.toNat : ℤ) < 2^256 := by exact_mod_cast Limbs.toNat_lt x hx
  unfold trueState
  generalize hs0 : (⟨1, (F.modulus.toNat : ℤ), (x.toNat : ℤ)⟩ : State) = s0
  have hs0d : s0.two_delta = 1 := by rw [← hs0]
  have hpB : |s0.f| ≤ 2^256 - 1 := by
    rw [← hs0]; show |(F.modulus.toNat : ℤ)| ≤ 2^256 - 1; rw [abs_of_nonneg (by positivity)]; omega
  have hxB : |s0.g| ≤ 2^256 - 1 := by
    rw [← hs0]; show |(x.toNat : ℤ)| ≤ 2^256 - 1; rw [abs_of_nonneg (by positivity)]; omega
  obtain ⟨hfB, hgB⟩ := Inversion.divsteps_abs_le (59 * i) s0 _ hpB hxB
  have hdB := Inversion.divsteps_two_delta_abs_le (59 * i) s0
  rw [hs0d, abs_one] at hdB
  push_cast at hdB
  rw [abs_le] at hdB hfB hgB
  refine ⟨Inversion.divsteps_f_odd _ _ (by rw [← hs0]; exact hp_odd), ?_, ?_, ?_, ?_⟩
  · rw [Inversion.divsteps_two_delta_emod_two, hs0d]; rfl
  · rw [abs_lt]; constructor <;> omega
  · rw [abs_lt]; constructor <;> omega
  · rw [abs_lt]; constructor <;> omega

/-- `asr` by 63 is the sign mask of a word. -/
theorem asr_63 (w : ℕ) (hw : w < 2^64) : asr w 63 = if w < 2^63 then 0 else 2^64 - 1 := by
  unfold asr; norm_num; split_ifs <;> omega

/-- One round of `invert` on words carrying the model's state after `i` rounds carries its state
after `i + 1`. -/
theorem invertRound_carries (F : PastaField) (x : Limbs) (hx : x.Bounded) (i : ℕ) (hi : i + 1 ≤ 9)
    (st : InvertState) (h : Carries st (rounds F x i)) :
    Carries (invertRound F.modulus F.inv st) (rounds F x (i + 1)) := by
  obtain ⟨s_two_delta, sf, sg, sd, se⟩ := st
  obtain ⟨htwo_delta, hf, hg, hd, he⟩ := h
  simp only at htwo_delta hf hg hd he
  subst hf hg hd he
  obtain ⟨hdT, hfb, hgb, hfT, hgT, hdb, heb, -, -, -⟩ := Inversion.rounds_invariant F x hx i
  obtain ⟨hodd, hdodd, hdD, hfv, hgv⟩ := trueState_facts F x hx i (by omega)
  have hround : rounds F x (i + 1) = round F (rounds F x i) := by
    rw [Inversion.rounds, Inversion.rounds, Function.iterate_succ_apply']
  rw [hround]
  simp only [invertRound, round]
  generalize hrs : rounds F x i = rs at *
  generalize ht : trueState F x i = t at *
  -- The block's inputs carry the true state, and the model's round starts with `divstep59` on
  -- the same words.
  have hfl : (rs.f.l0 : ℤ) = t.f % 2^64 := by
    rw [← hfT]; exact (Inversion.Signed5.toInt_emod _ hfb).symm
  have hgl : (rs.g.l0 : ℤ) = t.g % 2^64 := by
    rw [← hgT]; exact (Inversion.Signed5.toInt_emod _ hgb).symm
  have hdw : (s_two_delta : ℤ) = t.two_delta % 2^64 := by rw [htwo_delta, hdT]
  have hfl' : rs.f.l0 = (t.f % 2^64).toNat := by rw [← hfl, Int.toNat_natCast]
  have hgl' : rs.g.l0 = (t.g % 2^64).toNat := by rw [← hgl, Int.toNat_natCast]
  have hdm := Inversion.divstep59_spec t hodd
  rw [← hdT, ← hfl', ← hgl'] at hdm
  rw [hdm.1, hdm.2]
  generalize hdm' : divstep59Block s_two_delta rs.f.l0 rs.g.l0 = dm
  obtain ⟨hD, hMu, hMv, hMq, hMr⟩ :=
    divstep59Block_spec s_two_delta rs.f.l0 rs.g.l0 t hodd hdodd hdD hdw hfl hgl dm hdm'.symm
  obtain ⟨hrow1, hrow2⟩ := Inversion.M_rowSum_le 59 t
  generalize hN : M 59 t = N at *
  have hu0 := abs_nonneg N.u
  have hv0 := abs_nonneg N.v
  have hq0 := abs_nonneg N.q
  have hr0 := abs_nonneg N.r
  -- The sign-magnitude form of the matrix.
  generalize hsm' : signMagBlock dm.u dm.v dm.q dm.r = sm
  have hsm : sm = ⟨N.u.natAbs, N.v.natAbs, N.q.natAbs, N.r.natAbs,
      signMask N.u, signMask N.v, signMask N.q, signMask N.r⟩ :=
    signMagBlock_spec N.u N.v N.q N.r dm.u dm.v dm.q dm.r (by omega) (by omega)
      (by omega) (by omega) hMu hMv hMq hMr sm hsm'.symm
  have hru : SignMagRep sm.u sm.su N.u := by rw [hsm]; exact SignMagRep.of_natAbs _
  have hrv : SignMagRep sm.v sm.sv N.v := by rw [hsm]; exact SignMagRep.of_natAbs _
  have hrq : SignMagRep sm.q sm.sq N.q := by rw [hsm]; exact SignMagRep.of_natAbs _
  have hrr : SignMagRep sm.r sm.sr N.r := by rw [hsm]; exact SignMagRep.of_natAbs _
  -- The two rows of `f`, `g`, which are `updateFG`'s.
  have hfv' : |rs.f.toInt| < 2^256 := by rw [hfT]; exact hfv
  have hgv' : |rs.g.toInt| < 2^256 := by rw [hgT]; exact hgv
  obtain ⟨hF1b, hF1⟩ := fgRowBlock_spec N.u N.v rs.f rs.g sm.u sm.v sm.su sm.sv hfb hgb
    hfv' hgv' (by omega) hru hrv _ rfl
  obtain ⟨hF2b, hF2⟩ := fgRowBlock_spec N.q N.r rs.f rs.g sm.q sm.r sm.sq sm.sr hfb hgb
    hfv' hgv' (by omega) hrq hrr _ rfl
  obtain ⟨hG1b, hG2b, hG1, hG2⟩ := Inversion.updateFG_spec N rs.f rs.g hfv' hgv' ⟨hrow1, hrow2⟩
  have hfeq : fgRowBlock rs.f rs.g sm.u sm.v sm.su sm.sv = (updateFG N rs.f rs.g).1 := by
    apply Signed5.ext_of_toInt _ _ hF1b hG1b
    rw [hF1, hG1]
  have hgeq : fgRowBlock rs.f rs.g sm.q sm.r sm.sq sm.sr = (updateFG N rs.f rs.g).2 := by
    apply Signed5.ext_of_toInt _ _ hF2b hG2b
    rw [hF2, hG2]
  -- The two rows of `d`, `e`, reduced, which are `updateDE`'s.
  have hd' : |(rs.d.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (Int.natCast_nonneg _)]; exact_mod_cast Limbs.toNat_lt _ hdb
  have he' : |(rs.e.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (Int.natCast_nonneg _)]; exact_mod_cast Limbs.toNat_lt _ heb
  obtain ⟨hD1b, hD1⟩ := deRowBlock_spec N.u N.v rs.d rs.e sm.u sm.v sm.su sm.sv hdb heb
    (by omega) hru hrv _ rfl
  obtain ⟨hD2b, hD2⟩ := deRowBlock_spec N.q N.r rs.d rs.e sm.q sm.r sm.sq sm.sr hdb heb
    (by omega) hrq hrr _ rfl
  have h315 : (2 : ℤ)^59 * 2^256 = 2^315 := by norm_num
  have htd : |(deRowBlock rs.d rs.e sm.u sm.v sm.su sm.sv).toInt| < 2^315 := by
    rw [hD1, ← h315]; exact Inversion.row_abs_lt N.u N.v _ _ _ _ hrow1 hd' he' (by positivity)
  have hte : |(deRowBlock rs.d rs.e sm.q sm.r sm.sq sm.sr).toInt| < 2^315 := by
    rw [hD2, ← h315]; exact Inversion.row_abs_lt N.q N.r _ _ _ _ hrow2 hd' he' (by positivity)
  have hdeq : amontredBlock (deRowBlock rs.d rs.e sm.u sm.v sm.su sm.sv) F.modulus F.inv
      = (updateDE N rs.d rs.e F.modulus F.inv).1 := by
    rw [amontredBlock_spec F _ F.modulus F.inv rfl rfl hD1b htd _ rfl]
    unfold Inversion.amontred Inversion.updateDE
    rw [hD1]
  have heeq : amontredBlock (deRowBlock rs.d rs.e sm.q sm.r sm.sq sm.sr) F.modulus F.inv
      = (updateDE N rs.d rs.e F.modulus F.inv).2 := by
    rw [amontredBlock_spec F _ F.modulus F.inv rfl rfl hD2b hte _ rfl]
    unfold Inversion.amontred Inversion.updateDE
    rw [hD2]
  unfold Carries
  dsimp only
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · exact hD
  · exact hfeq
  · exact hgeq
  · exact hdeq
  · exact heeq

/-- The composition computes the model. -/
theorem invert_eq_model (F : PastaField) (x : Limbs) (hx : x.Bounded) :
    invert x F.modulus F.inv (startE F) = montInvModel F x := by
  -- The nine rounds.
  simp only [invert, montInvModel, finalD]
  generalize hst0 : (⟨1, ⟨F.modulus.l0, F.modulus.l1, F.modulus.l2, F.modulus.l3, 0⟩,
    ⟨x.l0, x.l1, x.l2, x.l3, 0⟩, ⟨0, 0, 0, 0⟩, startE F⟩ : InvertState) = st0
  have hcarry : ∀ i, i ≤ 9 → Carries ((invertRound F.modulus F.inv)^[i] st0) (rounds F x i) := by
    intro i
    induction i with
    | zero =>
      intro _
      rw [← hst0]
      exact ⟨rfl, rfl, rfl, rfl, rfl⟩
    | succ i ih =>
      intro hi
      rw [Function.iterate_succ_apply']
      exact invertRound_carries F x hx i hi _ (ih (by omega))
  obtain ⟨htwo_delta, hf, hg, hd, he⟩ := hcarry 9 (le_refl _)
  generalize hst : (invertRound F.modulus F.inv)^[9] st0 = st at htwo_delta hf hg hd he ⊢
  rw [hf, hg, hd, he]
  obtain ⟨hdT, hfb, hgb, hfT, hgT, hdb, heb, -, -, -⟩ := Inversion.rounds_invariant F x hx 9
  obtain ⟨hodd, hdodd, hdD, hfv, hgv⟩ := trueState_facts F x hx 9 (le_refl _)
  generalize hrs : rounds F x 9 = rs at *
  generalize ht : trueState F x 9 = t at *
  -- The last `divstep59`.
  have hfl : (rs.f.l0 : ℤ) = t.f % 2^64 := by
    rw [← hfT]; exact (Inversion.Signed5.toInt_emod _ hfb).symm
  have hgl : (rs.g.l0 : ℤ) = t.g % 2^64 := by
    rw [← hgT]; exact (Inversion.Signed5.toInt_emod _ hgb).symm
  have hdw : (st.two_delta : ℤ) = t.two_delta % 2^64 := by rw [htwo_delta, hdT]
  have hfl' : rs.f.l0 = (t.f % 2^64).toNat := by rw [← hfl, Int.toNat_natCast]
  have hgl' : rs.g.l0 = (t.g % 2^64).toNat := by rw [← hgl, Int.toNat_natCast]
  have hdm := Inversion.divstep59_spec t hodd
  rw [← hdT, ← hfl', ← hgl'] at hdm
  rw [hdm.2]
  generalize hdm' : divstep59Block st.two_delta rs.f.l0 rs.g.l0 = dm
  obtain ⟨-, hMu, hMv, hMq, hMr⟩ :=
    divstep59Block_spec st.two_delta rs.f.l0 rs.g.l0 t hodd hdodd hdD hdw hfl hgl dm hdm'.symm
  obtain ⟨hrow1, hrow2⟩ := Inversion.M_rowSum_le 59 t
  generalize hN : M 59 t = N at *
  have hu0 := abs_nonneg N.u
  have hv0 := abs_nonneg N.v
  have hq0 := abs_nonneg N.q
  have hr0 := abs_nonneg N.r
  -- The sign word is the model's.
  set sw := signWordOf N rs.f rs.g with hsw
  have hsw' : (sw : ℤ) = (N.u * t.f + N.v * t.g) % 2^64 := by
    rw [hsw, signWordOf, Int.toNat_of_nonneg (Int.emod_nonneg _ (by norm_num))]
    have e1 : (rs.f.l0 : ℤ) ≡ t.f [ZMOD 2^64] := by rw [hfl]; exact Int.mod_modEq _ _
    have e2 : (rs.g.l0 : ℤ) ≡ t.g [ZMOD 2^64] := by rw [hgl]; exact Int.mod_modEq _ _
    exact (e1.mul_left N.u).add (e2.mul_left N.v)
  have hsw64 : sw < 2^64 := by
    have := Int.emod_lt_of_pos (N.u * t.f + N.v * t.g) (by norm_num : (0 : ℤ) < 2^64); omega
  have hsign : signWordBlock rs.f.l0 rs.g.l0 dm.u dm.v = if sw < 2^63 then 0 else 2^64 - 1 := by
    have h1 : ((mulLo rs.f.l0 dm.u : ℕ) : ℤ) = (t.f * N.u) % 2^64 := mul_word _ _ _ _ hfl hMu
    have h2 : ((mulLo rs.g.l0 dm.v : ℕ) : ℤ) = (t.g * N.v) % 2^64 := mul_word _ _ _ _ hgl hMv
    have hw : ((addw (mulLo rs.f.l0 dm.u) (mulLo rs.g.l0 dm.v) : ℕ) : ℤ)
        = (t.f * N.u + t.g * N.v) % 2^64 := addw_word _ _ _ _ h1 h2
    have hw' : addw (mulLo rs.f.l0 dm.u) (mulLo rs.g.l0 dm.v) = sw := by
      have : ((addw (mulLo rs.f.l0 dm.u) (mulLo rs.g.l0 dm.v) : ℕ) : ℤ) = sw := by
        rw [hw, hsw']; exact congrArg (· % 2^64) (by ring)
      exact_mod_cast this
    unfold signWordBlock
    rw [hw', asr_63 sw hsw64]
  rw [hsign]
  -- The masks of the row: the entries with the sign folded in.
  set σ : ℤ := if sw < 2^63 then 1 else -1 with hσ
  generalize hsm' : signMagBlock dm.u dm.v dm.q dm.r = sm
  have hsm : sm = ⟨N.u.natAbs, N.v.natAbs, N.q.natAbs, N.r.natAbs,
      signMask N.u, signMask N.v, signMask N.q, signMask N.r⟩ :=
    signMagBlock_spec N.u N.v N.q N.r dm.u dm.v dm.q dm.r (by omega) (by omega)
      (by omega) (by omega) hMu hMv hMq hMr sm hsm'.symm
  have hru : SignMagRep sm.u (eorw sm.su (if sw < 2^63 then 0 else 2^64 - 1)) (σ * N.u) := by
    rw [hsm, hσ]
    split_ifs
    · rw [one_mul]; exact (SignMagRep.of_natAbs _).eor_zero _ _ _
    · rw [neg_one_mul]; exact (SignMagRep.of_natAbs _).eor_ones _ _ _
  have hrv : SignMagRep sm.v (eorw sm.sv (if sw < 2^63 then 0 else 2^64 - 1)) (σ * N.v) := by
    rw [hsm, hσ]
    split_ifs
    · rw [one_mul]; exact (SignMagRep.of_natAbs _).eor_zero _ _ _
    · rw [neg_one_mul]; exact (SignMagRep.of_natAbs _).eor_ones _ _ _
  have hσ1 : |σ| = 1 := by rw [hσ]; split_ifs <;> simp
  have hrow' : |σ * N.u| + |σ * N.v| ≤ 2^59 := by
    rw [abs_mul, abs_mul, hσ1, one_mul, one_mul]; exact hrow1
  -- The row and its reduction.
  obtain ⟨hTb, hT⟩ := deRowBlock_spec (σ * N.u) (σ * N.v) rs.d rs.e sm.u sm.v _ _ hdb heb
    (le_trans hrow' (by norm_num)) hru hrv _ rfl
  have hd' : |(rs.d.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (Int.natCast_nonneg _)]; exact_mod_cast Limbs.toNat_lt _ hdb
  have he' : |(rs.e.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (Int.natCast_nonneg _)]; exact_mod_cast Limbs.toNat_lt _ heb
  have hval : σ * N.u * rs.d.toNat + σ * N.v * rs.e.toNat
      = σ * (N.u * rs.d.toNat + N.v * rs.e.toNat) := by ring
  have htv : |σ * (N.u * rs.d.toNat + N.v * rs.e.toNat)| < 2^315 := by
    rw [← hval, ← show (2 : ℤ)^59 * 2^256 = 2^315 by norm_num]
    exact Inversion.row_abs_lt (σ * N.u) (σ * N.v) _ _ _ _ hrow' hd' he' (by positivity)
  set res : ℤ := amontredZ (σ * (N.u * rs.d.toNat + N.v * rs.e.toNat)) F.modulus.toNat F.inv
    with hres
  have hA : amontredBlock (deRowBlock rs.d rs.e sm.u sm.v
      (eorw sm.su (if sw < 2^63 then 0 else 2^64 - 1))
      (eorw sm.sv (if sw < 2^63 then 0 else 2^64 - 1))) F.modulus F.inv
      = Limbs.ofNat res.toNat := by
    rw [amontredBlock_spec F _ F.modulus F.inv rfl rfl hTb (by rw [hT, hval]; exact htv) _ rfl]
    unfold Inversion.amontred
    rw [hT, hval]
  obtain ⟨hr0, -, hr2p, hr256, -⟩ := Inversion.amontredZ_spec F _ htv
  rw [← hres] at hr0 hr2p hr256
  -- The conditional subtraction is the model's.
  have hp254 : 2^254 ≤ F.modulus.toNat := F.two_pow_le_modulus
  have hp255 : F.modulus.toNat < 2^255 := F.modulus_lt
  have hAt : (Limbs.ofNat res.toNat).toNat = res.toNat := Limbs.toNat_ofNat _ (by omega)
  obtain ⟨hRb, hR⟩ := condSubBlock_spec (Limbs.ofNat res.toNat) F.modulus (Limbs.ofNat_bounded _)
    F.bounded F.shape _ rfl
  rw [hAt] at hR
  have hqlt : (if res < F.modulus.toNat then res else res - F.modulus.toNat).toNat < 2^256 := by
    split_ifs <;> omega
  rw [hA]
  apply Limbs.ext_of_toNat _ _ hRb (Limbs.ofNat_bounded _)
  rw [Limbs.toNat_ofNat _ hqlt]
  rcases hR with ⟨h1, h2⟩ | ⟨h1, h2⟩
  · rw [h2, if_pos (by omega)]
  · rw [if_neg (by omega)]; omega

end PastaCurves.AArch64
