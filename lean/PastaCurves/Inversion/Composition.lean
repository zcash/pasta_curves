import PastaCurves.Compositions
import PastaCurves.Inversion.SignMag
import PastaCurves.Inversion.PackedWords
import PastaCurves.Inversion.Model
import PastaCurves.Inversion.HullCert

/-!
# The inversion's composition is the model

`invert` of `Compositions.lean` mirrors the Rust driver over a backend's six blocks: nine rounds,
then the last round. `InvertBlocks.Spec` states what each block must compute, in terms of the
word-level functions of the shared layer, and `invert_eq_model` shows that a backend whose blocks
meet it computes `montInvModel` of `Model.lean`. `invert_entry_spec` then applies Theorem 12
(`montInv_spec`) to the crate's entry point, with the termination bound discharged by the hull
certificate. A backend proves its six block theorems and instantiates the record; nothing else
of the composition is per backend.

The proof carries a relation between the word state of a round and the model's `RoundState`:
the same `f`, `g`, `d`, `e`, and `two_delta` as a word. Each round preserves it by the six block
specifications, whose hypotheses come from the model's invariant (`rounds_invariant`) and from
the bounds of the true divstep state after `59 i` steps. The last round is the same with the sign
word xored into the masks of its row, which negates the represented entries exactly when the
model's `finalD` negates the row.

The proofs name the true state, the model's state, the matrix, and the blocks' results by
`generalize` rather than `set`: those terms are iterates and long `let` chains, and a `let`-bound
name lets `omega`, `rw`, and `show` unfold them to the recursion limit. -/

set_option exponentiation.threshold 400

namespace PastaCurves

open Inversion (State divsteps M RoundState rounds trueState round initState startE montInvModel
  signWordOf finalD updateFG updateDE amontredZ Mat2 SignMagRep signMask mul_word addw_word)

/-- What the composition needs of a backend's blocks at a field `F`: each block, under the
hypotheses of its field, computes the word-level function of the shared layer that the round model
composes. -/
structure InvertBlocks.Spec (B : InvertBlocks) (F : PastaField) : Prop where
  /-- `divstep59` on the words of a true state with `f` and `two_delta` odd and `two_delta` small
  returns `two_delta` and the entries of the 59-step matrix, modulo `2^64`. -/
  divstep59 : ∀ (two_delta f0 g0 : ℕ) (s : State), s.f % 2 = 1 → s.two_delta % 2 = 1 → |s.two_delta| < 2^61 →
    (two_delta : ℤ) = s.two_delta % 2^64 → (f0 : ℤ) = s.f % 2^64 → (g0 : ℤ) = s.g % 2^64 →
    ((B.divstep59 two_delta f0 g0).two_delta : ℤ) = (divsteps 59 s).two_delta % 2^64 ∧
      ((B.divstep59 two_delta f0 g0).u : ℤ) = (M 59 s).u % 2^64 ∧
      ((B.divstep59 two_delta f0 g0).v : ℤ) = (M 59 s).v % 2^64 ∧
      ((B.divstep59 two_delta f0 g0).q : ℤ) = (M 59 s).q % 2^64 ∧
      ((B.divstep59 two_delta f0 g0).r : ℤ) = (M 59 s).r % 2^64
  /-- `sign_mag` on the words of four entries below `2^63` in magnitude returns their
  magnitudes and sign masks. -/
  signMag : ∀ (zu zv zq zr : ℤ) (u v q r : ℕ), |zu| < 2^63 → |zv| < 2^63 → |zq| < 2^63 →
    |zr| < 2^63 → (u : ℤ) = zu % 2^64 → (v : ℤ) = zv % 2^64 → (q : ℤ) = zq % 2^64 →
    (r : ℤ) = zr % 2^64 →
    B.signMag u v q r =
      ⟨zu.natAbs, zv.natAbs, zq.natAbs, zr.natAbs, signMask zu, signMask zv, signMask zq, signMask zr⟩
  /-- `fg_row` on `f`, `g` below `2^256` in magnitude and a row in sign-magnitude form, under
  the row bound, is the exact `(a f + b g) / 2^59` in five words. -/
  fgRow : ∀ (a b : ℤ) (f g : Signed5) (m0 m1 s0 s1 : ℕ), f.Bounded → g.Bounded →
    |f.toInt| < 2^256 → |g.toInt| < 2^256 → |a| + |b| ≤ 2^63 →
    SignMagRep m0 s0 a → SignMagRep m1 s1 b →
    (B.fgRow f g m0 m1 s0 s1).Bounded ∧
      (B.fgRow f g m0 m1 s0 s1).toInt = (a * f.toInt + b * g.toInt) / 2^59
  /-- `de_row` on four-limb `d`, `e` and a row in sign-magnitude form, under the row bound, is
  the exact `a d + b e` in five words. -/
  deRow : ∀ (a b : ℤ) (d e : Limbs) (m0 m1 s0 s1 : ℕ), d.Bounded → e.Bounded →
    |a| + |b| ≤ 2^63 → SignMagRep m0 s0 a → SignMagRep m1 s1 b →
    (B.deRow d e m0 m1 s0 s1).Bounded ∧
      (B.deRow d e m0 m1 s0 s1).toInt = a * d.toNat + b * e.toNat
  /-- `amontred` at the field's modulus and `inv`, on a bounded five-word value below `2^315`
  in magnitude, is the round's `amontred`. -/
  amontred : ∀ (t : Signed5), t.Bounded → |t.toInt| < 2^315 →
    B.amontred t F.modulus F.inv = Inversion.amontred t F.modulus F.inv
  /-- `cond_sub` at the field's modulus subtracts it exactly when the value is not below it. -/
  condSub : ∀ (value : Limbs), value.Bounded →
    (B.condSub value F.modulus).Bounded ∧
      ((value.toNat < F.modulus.toNat ∧ (B.condSub value F.modulus).toNat = value.toNat) ∨
        (F.modulus.toNat ≤ value.toNat ∧
          (B.condSub value F.modulus).toNat + F.modulus.toNat = value.toNat))

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

/-- One round of `invert` on words carrying the model's state after `i` rounds carries its state
after `i + 1`. -/
theorem invertRound_carries (B : InvertBlocks) (F : PastaField) (hB : B.Spec F) (x : Limbs)
    (hx : x.Bounded) (i : ℕ) (hi : i + 1 ≤ 9) (st : InvertState) (h : Carries st (rounds F x i)) :
    Carries (invertRound B F.modulus F.inv st) (rounds F x (i + 1)) := by
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
  have hspec := hB.divstep59 s_two_delta rs.f.l0 rs.g.l0 t hodd hdodd hdD hdw hfl hgl
  generalize hdm' : B.divstep59 s_two_delta rs.f.l0 rs.g.l0 = dm at hspec ⊢
  obtain ⟨hD, hMu, hMv, hMq, hMr⟩ := hspec
  obtain ⟨hrow1, hrow2⟩ := Inversion.M_rowSum_le 59 t
  generalize hN : M 59 t = N at *
  have hu0 := abs_nonneg N.u
  have hv0 := abs_nonneg N.v
  have hq0 := abs_nonneg N.q
  have hr0 := abs_nonneg N.r
  -- The sign-magnitude form of the matrix.
  have hsm := hB.signMag N.u N.v N.q N.r dm.u dm.v dm.q dm.r (by omega) (by omega)
    (by omega) (by omega) hMu hMv hMq hMr
  generalize hsm' : B.signMag dm.u dm.v dm.q dm.r = sm at hsm ⊢
  have hru : SignMagRep sm.u sm.su N.u := by rw [hsm]; exact SignMagRep.of_natAbs _
  have hrv : SignMagRep sm.v sm.sv N.v := by rw [hsm]; exact SignMagRep.of_natAbs _
  have hrq : SignMagRep sm.q sm.sq N.q := by rw [hsm]; exact SignMagRep.of_natAbs _
  have hrr : SignMagRep sm.r sm.sr N.r := by rw [hsm]; exact SignMagRep.of_natAbs _
  -- The two rows of `f`, `g`, which are `updateFG`'s.
  have hfv' : |rs.f.toInt| < 2^256 := by rw [hfT]; exact hfv
  have hgv' : |rs.g.toInt| < 2^256 := by rw [hgT]; exact hgv
  obtain ⟨hF1b, hF1⟩ := hB.fgRow N.u N.v rs.f rs.g sm.u sm.v sm.su sm.sv hfb hgb hfv' hgv'
    (by omega) hru hrv
  obtain ⟨hF2b, hF2⟩ := hB.fgRow N.q N.r rs.f rs.g sm.q sm.r sm.sq sm.sr hfb hgb hfv' hgv'
    (by omega) hrq hrr
  obtain ⟨hG1b, hG2b, hG1, hG2⟩ := Inversion.updateFG_spec N rs.f rs.g hfv' hgv' ⟨hrow1, hrow2⟩
  have hfeq : B.fgRow rs.f rs.g sm.u sm.v sm.su sm.sv = (updateFG N rs.f rs.g).1 := by
    apply Signed5.ext_of_toInt _ _ hF1b hG1b
    rw [hF1, hG1]
  have hgeq : B.fgRow rs.f rs.g sm.q sm.r sm.sq sm.sr = (updateFG N rs.f rs.g).2 := by
    apply Signed5.ext_of_toInt _ _ hF2b hG2b
    rw [hF2, hG2]
  -- The two rows of `d`, `e`, reduced, which are `updateDE`'s.
  have hd' : |(rs.d.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (Int.natCast_nonneg _)]; exact_mod_cast Limbs.toNat_lt _ hdb
  have he' : |(rs.e.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (Int.natCast_nonneg _)]; exact_mod_cast Limbs.toNat_lt _ heb
  obtain ⟨hD1b, hD1⟩ := hB.deRow N.u N.v rs.d rs.e sm.u sm.v sm.su sm.sv hdb heb (by omega)
    hru hrv
  obtain ⟨hD2b, hD2⟩ := hB.deRow N.q N.r rs.d rs.e sm.q sm.r sm.sq sm.sr hdb heb (by omega)
    hrq hrr
  have h315 : (2 : ℤ)^59 * 2^256 = 2^315 := by norm_num
  have htd : |(B.deRow rs.d rs.e sm.u sm.v sm.su sm.sv).toInt| < 2^315 := by
    rw [hD1, ← h315]; exact Inversion.row_abs_lt N.u N.v _ _ _ _ hrow1 hd' he' (by positivity)
  have hte : |(B.deRow rs.d rs.e sm.q sm.r sm.sq sm.sr).toInt| < 2^315 := by
    rw [hD2, ← h315]; exact Inversion.row_abs_lt N.q N.r _ _ _ _ hrow2 hd' he' (by positivity)
  have hdeq : B.amontred (B.deRow rs.d rs.e sm.u sm.v sm.su sm.sv) F.modulus F.inv
      = (updateDE N rs.d rs.e F.modulus F.inv).1 := by
    rw [hB.amontred _ hD1b htd]
    unfold Inversion.amontred Inversion.updateDE
    rw [hD1]
  have heeq : B.amontred (B.deRow rs.d rs.e sm.q sm.r sm.sq sm.sr) F.modulus F.inv
      = (updateDE N rs.d rs.e F.modulus F.inv).2 := by
    rw [hB.amontred _ hD2b hte]
    unfold Inversion.amontred Inversion.updateDE
    rw [hD2]
  unfold Carries
  dsimp only
  exact ⟨hD, hfeq, hgeq, hdeq, heeq⟩

/-- The composition over blocks that meet their specification computes the model. -/
theorem invert_eq_model (B : InvertBlocks) (F : PastaField) (hB : B.Spec F) (x : Limbs)
    (hx : x.Bounded) : invert B x F.modulus F.inv (startE F) = montInvModel F x := by
  -- The nine rounds.
  simp only [invert, montInvModel, finalD]
  generalize hst0 : (⟨1, ⟨F.modulus.l0, F.modulus.l1, F.modulus.l2, F.modulus.l3, 0⟩,
    ⟨x.l0, x.l1, x.l2, x.l3, 0⟩, ⟨0, 0, 0, 0⟩, startE F⟩ : InvertState) = st0
  have hcarry : ∀ i, i ≤ 9 →
      Carries ((invertRound B F.modulus F.inv)^[i] st0) (rounds F x i) := by
    intro i
    induction i with
    | zero =>
      intro _
      rw [← hst0]
      exact ⟨rfl, rfl, rfl, rfl, rfl⟩
    | succ i ih =>
      intro hi
      rw [Function.iterate_succ_apply']
      exact invertRound_carries B F hB x hx i hi _ (ih (by omega))
  obtain ⟨htwo_delta, hf, hg, hd, he⟩ := hcarry 9 (le_refl _)
  generalize hst : (invertRound B F.modulus F.inv)^[9] st0 = st at htwo_delta hf hg hd he ⊢
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
  have hspec := hB.divstep59 st.two_delta rs.f.l0 rs.g.l0 t hodd hdodd hdD hdw hfl hgl
  generalize hdm' : B.divstep59 st.two_delta rs.f.l0 rs.g.l0 = dm at hspec ⊢
  obtain ⟨-, hMu, hMv, hMq, hMr⟩ := hspec
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
  have hsign : signWord rs.f.l0 rs.g.l0 dm.u dm.v = if sw < 2^63 then 0 else 2^64 - 1 := by
    have h1 : ((mulLo rs.f.l0 dm.u : ℕ) : ℤ) = (t.f * N.u) % 2^64 := mul_word _ _ _ _ hfl hMu
    have h2 : ((mulLo rs.g.l0 dm.v : ℕ) : ℤ) = (t.g * N.v) % 2^64 := mul_word _ _ _ _ hgl hMv
    have hw : ((addw (mulLo rs.f.l0 dm.u) (mulLo rs.g.l0 dm.v) : ℕ) : ℤ)
        = (t.f * N.u + t.g * N.v) % 2^64 := addw_word _ _ _ _ h1 h2
    have hw' : addw (mulLo rs.f.l0 dm.u) (mulLo rs.g.l0 dm.v) = sw := by
      have : ((addw (mulLo rs.f.l0 dm.u) (mulLo rs.g.l0 dm.v) : ℕ) : ℤ) = sw := by
        rw [hw, hsw']; exact congrArg (· % 2^64) (by ring)
      exact_mod_cast this
    unfold signWord regMod
    rw [hw']
  rw [hsign]
  -- The masks of the row: the entries with the sign folded in.
  set σ : ℤ := if sw < 2^63 then 1 else -1 with hσ
  have hsm := hB.signMag N.u N.v N.q N.r dm.u dm.v dm.q dm.r (by omega) (by omega)
    (by omega) (by omega) hMu hMv hMq hMr
  generalize hsm' : B.signMag dm.u dm.v dm.q dm.r = sm at hsm ⊢
  have hru : SignMagRep sm.u (sm.su ^^^ (if sw < 2^63 then 0 else 2^64 - 1)) (σ * N.u) := by
    rw [hsm, hσ]
    split_ifs
    · rw [one_mul]; exact (SignMagRep.of_natAbs _).xor_zero _ _ _
    · rw [neg_one_mul]; exact (SignMagRep.of_natAbs _).xor_ones _ _ _
  have hrv : SignMagRep sm.v (sm.sv ^^^ (if sw < 2^63 then 0 else 2^64 - 1)) (σ * N.v) := by
    rw [hsm, hσ]
    split_ifs
    · rw [one_mul]; exact (SignMagRep.of_natAbs _).xor_zero _ _ _
    · rw [neg_one_mul]; exact (SignMagRep.of_natAbs _).xor_ones _ _ _
  have hσ1 : |σ| = 1 := by rw [hσ]; split_ifs <;> simp
  have hrow' : |σ * N.u| + |σ * N.v| ≤ 2^59 := by
    rw [abs_mul, abs_mul, hσ1, one_mul, one_mul]; exact hrow1
  -- The row and its reduction.
  obtain ⟨hTb, hT⟩ := hB.deRow (σ * N.u) (σ * N.v) rs.d rs.e sm.u sm.v _ _ hdb heb
    (le_trans hrow' (by norm_num)) hru hrv
  have hd' : |(rs.d.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (Int.natCast_nonneg _)]; exact_mod_cast Limbs.toNat_lt _ hdb
  have he' : |(rs.e.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (Int.natCast_nonneg _)]; exact_mod_cast Limbs.toNat_lt _ heb
  have hval : σ * N.u * rs.d.toNat + σ * N.v * rs.e.toNat
      = σ * (N.u * rs.d.toNat + N.v * rs.e.toNat) := by ring
  have htv : |σ * (N.u * rs.d.toNat + N.v * rs.e.toNat)| < 2^315 := by
    rw [← hval, ← show (2 : ℤ)^59 * 2^256 = 2^315 by norm_num]
    exact Inversion.row_abs_lt (σ * N.u) (σ * N.v) _ _ _ _ hrow' hd' he' (by positivity)
  set t' : ℤ := amontredZ (σ * (N.u * rs.d.toNat + N.v * rs.e.toNat)) F.modulus.toNat F.inv
    with ht'
  have hA : B.amontred (B.deRow rs.d rs.e sm.u sm.v
      (sm.su ^^^ (if sw < 2^63 then 0 else 2^64 - 1))
      (sm.sv ^^^ (if sw < 2^63 then 0 else 2^64 - 1))) F.modulus F.inv
      = Limbs.ofNat t'.toNat := by
    rw [hB.amontred _ hTb (by rw [hT, hval]; exact htv)]
    unfold Inversion.amontred
    rw [hT, hval]
  obtain ⟨ht'0, -, ht'2p, ht'256, -⟩ := Inversion.amontredZ_spec F _ htv
  rw [← ht'] at ht'0 ht'2p ht'256
  -- The conditional subtraction is the model's.
  have hp254 : 2^254 ≤ F.modulus.toNat := F.two_pow_le_modulus
  have hp255 : F.modulus.toNat < 2^255 := F.modulus_lt
  have hAt : (Limbs.ofNat t'.toNat).toNat = t'.toNat := Limbs.toNat_ofNat _ (by omega)
  obtain ⟨hRb, hR⟩ := hB.condSub (Limbs.ofNat t'.toNat) (Limbs.ofNat_bounded _)
  rw [hAt] at hR
  have hqlt : (if t' < F.modulus.toNat then t' else t' - F.modulus.toNat).toNat < 2^256 := by
    split_ifs <;> omega
  rw [hA]
  apply Limbs.ext_of_toNat _ _ hRb (Limbs.ofNat_bounded _)
  rw [Limbs.toNat_ofNat _ hqlt]
  rcases hR with ⟨h1, h2⟩ | ⟨h1, h2⟩
  · rw [h2, if_pos (by omega)]
  · rw [if_neg (by omega)]; omega

/-- The crate's `invert` over blocks that meet their specification, at a Pasta field. The input
is canonical, as the entry point asserts, and `e0` is `2^562 mod p`, as its contract requires.
The result is canonical. For `x = 0` it is `0`; otherwise it is the Montgomery inverse, with
`x * result ≡ R^2 (mod p)`. The termination bound of the divstep recurrence is discharged by
the hull certificate's `terminationBound_256`, and the primality of `p` is `F.prime`. -/
theorem invert_entry_spec (B : InvertBlocks) (F : PastaField) (hB : B.Spec F) (x : Limbs)
    (hx : x.Bounded)
    (h : isCanonical x F.modulus = true) (e0 : Limbs) (he0 : e0 = Inversion.startE F) :
    (invert B x F.modulus F.inv e0).Bounded ∧
      (invert B x F.modulus F.inv e0).toNat < F.modulus.toNat ∧
      (x.toNat = 0 → invert B x F.modulus F.inv e0 = Limbs.ofNat 0) ∧
      (x.toNat ≠ 0 →
        x.toNat * (invert B x F.modulus F.inv e0).toNat ≡ R^2 [MOD F.modulus.toNat]) := by
  rw [he0, invert_eq_model B F hB x hx]
  exact Inversion.montInv_spec F Inversion.Hull.terminationBound_256 x hx
    ((isCanonical_iff x F.modulus hx F.bounded).1 h)

end PastaCurves
