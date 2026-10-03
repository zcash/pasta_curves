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
the same `f`, `g`, `u`, `v`, and `d` as a word. Each round preserves it by the six block
specifications, whose hypotheses come from the model's invariant (`rounds_invariant`) and from
the bounds of the true divstep state after `59 i` steps. The last round is the same with the sign
word xored into the masks of its row, which negates the represented entries exactly when the
model's `finalU` negates the row.

The proofs name the true state, the model's state, the matrix, and the blocks' results by
`generalize` rather than `set`: those terms are iterates and long `let` chains, and a `let`-bound
name lets `omega`, `rw`, and `show` unfold them to the recursion limit. -/

set_option exponentiation.threshold 400

namespace PastaCurves

open Inversion (State divsteps M RoundState rounds trueState round initState startV montInvModel
  signWordOf finalU updateFG updateUV amontredZ Mat2 SignMagRep signMask mul_word addw_word)

/-- What the composition needs of a backend's blocks at a field `F`: each block, on bounded
inputs, computes the word-level function of the shared layer that the round model composes. -/
structure InvertBlocks.Spec (B : InvertBlocks) (F : PastaField) : Prop where
  /-- `divstep59` on the words of a true state with `f` and `d` odd and `d` small returns `d`
  and the entries of the 59-step matrix, modulo `2^64`. -/
  divstep59 : ∀ (d f0 g0 : ℕ) (s : State), s.f % 2 = 1 → s.d % 2 = 1 → |s.d| < 2^61 →
    (d : ℤ) = s.d % 2^64 → (f0 : ℤ) = s.f % 2^64 → (g0 : ℤ) = s.g % 2^64 →
    ((B.divstep59 d f0 g0).d : ℤ) = (divsteps 59 s).d % 2^64 ∧
      ((B.divstep59 d f0 g0).m00 : ℤ) = (M 59 s).a % 2^64 ∧
      ((B.divstep59 d f0 g0).m01 : ℤ) = (M 59 s).b % 2^64 ∧
      ((B.divstep59 d f0 g0).m10 : ℤ) = (M 59 s).c % 2^64 ∧
      ((B.divstep59 d f0 g0).m11 : ℤ) = (M 59 s).d % 2^64
  /-- `sign_mag` on the words of four entries below `2^63` in magnitude returns their
  magnitudes and sign masks. -/
  signMag : ∀ (a b c d : ℤ) (m00 m01 m10 m11 : ℕ), |a| < 2^63 → |b| < 2^63 → |c| < 2^63 →
    |d| < 2^63 → (m00 : ℤ) = a % 2^64 → (m01 : ℤ) = b % 2^64 → (m10 : ℤ) = c % 2^64 →
    (m11 : ℤ) = d % 2^64 →
    B.signMag m00 m01 m10 m11 =
      ⟨a.natAbs, b.natAbs, c.natAbs, d.natAbs, signMask a, signMask b, signMask c, signMask d⟩
  /-- `fg_row` on `f`, `g` below `2^256` in magnitude and a row in sign-magnitude form, under
  the row bound, is the exact `(a f + b g) / 2^59` in five words. -/
  fgRow : ∀ (a b : ℤ) (f g : Signed5) (m0 m1 s0 s1 : ℕ), f.Bounded → g.Bounded →
    |f.toInt| < 2^256 → |g.toInt| < 2^256 → |a| + |b| ≤ 2^63 →
    SignMagRep m0 s0 a → SignMagRep m1 s1 b →
    (B.fgRow f g m0 m1 s0 s1).Bounded ∧
      (B.fgRow f g m0 m1 s0 s1).toInt = (a * f.toInt + b * g.toInt) / 2^59
  /-- `uv_row` on four-limb `u`, `v` and a row in sign-magnitude form, under the row bound, is
  the exact `a u + b v` in five words. -/
  uvRow : ∀ (a b : ℤ) (u v : Limbs) (m0 m1 s0 s1 : ℕ), u.Bounded → v.Bounded →
    |a| + |b| ≤ 2^63 → SignMagRep m0 s0 a → SignMagRep m1 s1 b →
    (B.uvRow u v m0 m1 s0 s1).Bounded ∧
      (B.uvRow u v m0 m1 s0 s1).toInt = a * u.toNat + b * v.toNat
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

/-- The word state of a round carries the model's: the same `f`, `g`, `u`, `v`, and `d` as a
word. -/
def Carries (st : InvertState) (rs : RoundState) : Prop :=
  (st.d : ℤ) = rs.d % 2^64 ∧ st.f = rs.f ∧ st.g = rs.g ∧ st.u = rs.u ∧ st.v = rs.v

/-- What the blocks need of the true divstep state after `59 i` steps, for `i ≤ 9`: `f` odd, `d`
odd and small, and `f`, `g` below `2^256` in magnitude. -/
theorem trueState_facts (F : PastaField) (x : Limbs) (hx : x.Bounded) (i : ℕ) (hi : i ≤ 9) :
    (trueState F x i).f % 2 = 1 ∧ (trueState F x i).d % 2 = 1 ∧ |(trueState F x i).d| < 2^61 ∧
      |(trueState F x i).f| < 2^256 ∧ |(trueState F x i).g| < 2^256 := by
  have hp_odd : (F.modulus.toNat : ℤ) % 2 = 1 := by exact_mod_cast F.modulus_odd
  have hp255 : (F.modulus.toNat : ℤ) < 2^255 := by exact_mod_cast F.modulus_lt
  have hx256 : (x.toNat : ℤ) < 2^256 := by exact_mod_cast Limbs.toNat_lt x hx
  unfold trueState
  generalize hs0 : (⟨1, (F.modulus.toNat : ℤ), (x.toNat : ℤ)⟩ : State) = s0
  have hs0d : s0.d = 1 := by rw [← hs0]
  have hpB : |s0.f| ≤ 2^256 - 1 := by
    rw [← hs0]; show |(F.modulus.toNat : ℤ)| ≤ 2^256 - 1; rw [abs_of_nonneg (by positivity)]; omega
  have hxB : |s0.g| ≤ 2^256 - 1 := by
    rw [← hs0]; show |(x.toNat : ℤ)| ≤ 2^256 - 1; rw [abs_of_nonneg (by positivity)]; omega
  obtain ⟨hfB, hgB⟩ := Inversion.divsteps_abs_le (59 * i) s0 _ hpB hxB
  have hdB := Inversion.divsteps_d_abs_le (59 * i) s0
  rw [hs0d, abs_one] at hdB
  push_cast at hdB
  rw [abs_le] at hdB hfB hgB
  refine ⟨Inversion.divsteps_f_odd _ _ (by rw [← hs0]; exact hp_odd), ?_, ?_, ?_, ?_⟩
  · rw [Inversion.divsteps_d_emod_two, hs0d]; rfl
  · rw [abs_lt]; constructor <;> omega
  · rw [abs_lt]; constructor <;> omega
  · rw [abs_lt]; constructor <;> omega

/-- One round of `invert` on words carrying the model's state after `i` rounds carries its state
after `i + 1`. -/
theorem invertRound_carries (B : InvertBlocks) (F : PastaField) (hB : B.Spec F) (x : Limbs)
    (hx : x.Bounded) (i : ℕ) (hi : i + 1 ≤ 9) (st : InvertState) (h : Carries st (rounds F x i)) :
    Carries (invertRound B F.modulus F.inv st) (rounds F x (i + 1)) := by
  obtain ⟨sd, sf, sg, su, sv⟩ := st
  obtain ⟨hd, hf, hg, hu, hv⟩ := h
  simp only at hd hf hg hu hv
  subst hf hg hu hv
  obtain ⟨hdT, hfb, hgb, hfT, hgT, hub, hvb, -, -, -⟩ := Inversion.rounds_invariant F x hx i
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
  have hdw : (sd : ℤ) = t.d % 2^64 := by rw [hd, hdT]
  have hfl' : rs.f.l0 = (t.f % 2^64).toNat := by rw [← hfl, Int.toNat_natCast]
  have hgl' : rs.g.l0 = (t.g % 2^64).toNat := by rw [← hgl, Int.toNat_natCast]
  have hdm := Inversion.divstep59_spec t hodd
  rw [← hdT, ← hfl', ← hgl'] at hdm
  rw [hdm.1, hdm.2]
  have hspec := hB.divstep59 sd rs.f.l0 rs.g.l0 t hodd hdodd hdD hdw hfl hgl
  generalize hdm' : B.divstep59 sd rs.f.l0 rs.g.l0 = dm at hspec ⊢
  obtain ⟨hD, hM00, hM01, hM10, hM11⟩ := hspec
  obtain ⟨hrow1, hrow2⟩ := Inversion.M_rowSum_le 59 t
  generalize hN : M 59 t = N at *
  have ha0 := abs_nonneg N.a
  have hb0 := abs_nonneg N.b
  have hc0 := abs_nonneg N.c
  have hd0 := abs_nonneg N.d
  -- The sign-magnitude form of the matrix.
  have hsm := hB.signMag N.a N.b N.c N.d dm.m00 dm.m01 dm.m10 dm.m11 (by omega) (by omega)
    (by omega) (by omega) hM00 hM01 hM10 hM11
  generalize hsm' : B.signMag dm.m00 dm.m01 dm.m10 dm.m11 = sm at hsm ⊢
  have hr00 : SignMagRep sm.m00 sm.s00 N.a := by rw [hsm]; exact SignMagRep.of_natAbs _
  have hr01 : SignMagRep sm.m01 sm.s01 N.b := by rw [hsm]; exact SignMagRep.of_natAbs _
  have hr10 : SignMagRep sm.m10 sm.s10 N.c := by rw [hsm]; exact SignMagRep.of_natAbs _
  have hr11 : SignMagRep sm.m11 sm.s11 N.d := by rw [hsm]; exact SignMagRep.of_natAbs _
  -- The two rows of `f`, `g`, which are `updateFG`'s.
  have hfv' : |rs.f.toInt| < 2^256 := by rw [hfT]; exact hfv
  have hgv' : |rs.g.toInt| < 2^256 := by rw [hgT]; exact hgv
  obtain ⟨hF1b, hF1⟩ := hB.fgRow N.a N.b rs.f rs.g sm.m00 sm.m01 sm.s00 sm.s01 hfb hgb hfv' hgv'
    (by omega) hr00 hr01
  obtain ⟨hF2b, hF2⟩ := hB.fgRow N.c N.d rs.f rs.g sm.m10 sm.m11 sm.s10 sm.s11 hfb hgb hfv' hgv'
    (by omega) hr10 hr11
  obtain ⟨hG1b, hG2b, hG1, hG2⟩ := Inversion.updateFG_spec N rs.f rs.g hfv' hgv' ⟨hrow1, hrow2⟩
  have hfeq : B.fgRow rs.f rs.g sm.m00 sm.m01 sm.s00 sm.s01 = (updateFG N rs.f rs.g).1 := by
    apply Signed5.ext_of_toInt _ _ hF1b hG1b
    rw [hF1, hG1]
  have hgeq : B.fgRow rs.f rs.g sm.m10 sm.m11 sm.s10 sm.s11 = (updateFG N rs.f rs.g).2 := by
    apply Signed5.ext_of_toInt _ _ hF2b hG2b
    rw [hF2, hG2]
  -- The two rows of `u`, `v`, reduced, which are `updateUV`'s.
  have hu' : |(rs.u.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (Int.natCast_nonneg _)]; exact_mod_cast Limbs.toNat_lt _ hub
  have hv' : |(rs.v.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (Int.natCast_nonneg _)]; exact_mod_cast Limbs.toNat_lt _ hvb
  obtain ⟨hU1b, hU1⟩ := hB.uvRow N.a N.b rs.u rs.v sm.m00 sm.m01 sm.s00 sm.s01 hub hvb (by omega)
    hr00 hr01
  obtain ⟨hU2b, hU2⟩ := hB.uvRow N.c N.d rs.u rs.v sm.m10 sm.m11 sm.s10 sm.s11 hub hvb (by omega)
    hr10 hr11
  have h315 : (2 : ℤ)^59 * 2^256 = 2^315 := by norm_num
  have htu : |(B.uvRow rs.u rs.v sm.m00 sm.m01 sm.s00 sm.s01).toInt| < 2^315 := by
    rw [hU1, ← h315]; exact Inversion.row_abs_lt N.a N.b _ _ _ _ hrow1 hu' hv' (by positivity)
  have htv : |(B.uvRow rs.u rs.v sm.m10 sm.m11 sm.s10 sm.s11).toInt| < 2^315 := by
    rw [hU2, ← h315]; exact Inversion.row_abs_lt N.c N.d _ _ _ _ hrow2 hu' hv' (by positivity)
  have hueq : B.amontred (B.uvRow rs.u rs.v sm.m00 sm.m01 sm.s00 sm.s01) F.modulus F.inv
      = (updateUV N rs.u rs.v F.modulus F.inv).1 := by
    rw [hB.amontred _ hU1b htu]
    unfold Inversion.amontred Inversion.updateUV
    rw [hU1]
  have hveq : B.amontred (B.uvRow rs.u rs.v sm.m10 sm.m11 sm.s10 sm.s11) F.modulus F.inv
      = (updateUV N rs.u rs.v F.modulus F.inv).2 := by
    rw [hB.amontred _ hU2b htv]
    unfold Inversion.amontred Inversion.updateUV
    rw [hU2]
  unfold Carries
  dsimp only
  exact ⟨hD, hfeq, hgeq, hueq, hveq⟩

/-- The composition over blocks that meet their specification computes the model. -/
theorem invert_eq_model (B : InvertBlocks) (F : PastaField) (hB : B.Spec F) (x : Limbs)
    (hx : x.Bounded) : invert B x F.modulus F.inv (startV F) = montInvModel F x := by
  -- The nine rounds.
  simp only [invert, montInvModel, finalU]
  generalize hst0 : (⟨1, ⟨F.modulus.l0, F.modulus.l1, F.modulus.l2, F.modulus.l3, 0⟩,
    ⟨x.l0, x.l1, x.l2, x.l3, 0⟩, ⟨0, 0, 0, 0⟩, startV F⟩ : InvertState) = st0
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
  obtain ⟨hd, hf, hg, hu, hv⟩ := hcarry 9 (le_refl _)
  generalize hst : (invertRound B F.modulus F.inv)^[9] st0 = st at hd hf hg hu hv ⊢
  rw [hf, hg, hu, hv]
  obtain ⟨hdT, hfb, hgb, hfT, hgT, hub, hvb, -, -, -⟩ := Inversion.rounds_invariant F x hx 9
  obtain ⟨hodd, hdodd, hdD, hfv, hgv⟩ := trueState_facts F x hx 9 (le_refl _)
  generalize hrs : rounds F x 9 = rs at *
  generalize ht : trueState F x 9 = t at *
  -- The last `divstep59`.
  have hfl : (rs.f.l0 : ℤ) = t.f % 2^64 := by
    rw [← hfT]; exact (Inversion.Signed5.toInt_emod _ hfb).symm
  have hgl : (rs.g.l0 : ℤ) = t.g % 2^64 := by
    rw [← hgT]; exact (Inversion.Signed5.toInt_emod _ hgb).symm
  have hdw : (st.d : ℤ) = t.d % 2^64 := by rw [hd, hdT]
  have hfl' : rs.f.l0 = (t.f % 2^64).toNat := by rw [← hfl, Int.toNat_natCast]
  have hgl' : rs.g.l0 = (t.g % 2^64).toNat := by rw [← hgl, Int.toNat_natCast]
  have hdm := Inversion.divstep59_spec t hodd
  rw [← hdT, ← hfl', ← hgl'] at hdm
  rw [hdm.2]
  have hspec := hB.divstep59 st.d rs.f.l0 rs.g.l0 t hodd hdodd hdD hdw hfl hgl
  generalize hdm' : B.divstep59 st.d rs.f.l0 rs.g.l0 = dm at hspec ⊢
  obtain ⟨-, hM00, hM01, hM10, hM11⟩ := hspec
  obtain ⟨hrow1, hrow2⟩ := Inversion.M_rowSum_le 59 t
  generalize hN : M 59 t = N at *
  have ha0 := abs_nonneg N.a
  have hb0 := abs_nonneg N.b
  have hc0 := abs_nonneg N.c
  have hd0 := abs_nonneg N.d
  -- The sign word is the model's.
  set sw := signWordOf N rs.f rs.g with hsw
  have hsw' : (sw : ℤ) = (N.a * t.f + N.b * t.g) % 2^64 := by
    rw [hsw, signWordOf, Int.toNat_of_nonneg (Int.emod_nonneg _ (by norm_num))]
    have e1 : (rs.f.l0 : ℤ) ≡ t.f [ZMOD 2^64] := by rw [hfl]; exact Int.mod_modEq _ _
    have e2 : (rs.g.l0 : ℤ) ≡ t.g [ZMOD 2^64] := by rw [hgl]; exact Int.mod_modEq _ _
    exact (e1.mul_left N.a).add (e2.mul_left N.b)
  have hsw64 : sw < 2^64 := by
    have := Int.emod_lt_of_pos (N.a * t.f + N.b * t.g) (by norm_num : (0 : ℤ) < 2^64); omega
  have hsign : signWord rs.f.l0 rs.g.l0 dm.m00 dm.m01 = if sw < 2^63 then 0 else 2^64 - 1 := by
    have h1 : ((mulLo rs.f.l0 dm.m00 : ℕ) : ℤ) = (t.f * N.a) % 2^64 := mul_word _ _ _ _ hfl hM00
    have h2 : ((mulLo rs.g.l0 dm.m01 : ℕ) : ℤ) = (t.g * N.b) % 2^64 := mul_word _ _ _ _ hgl hM01
    have hw : ((addw (mulLo rs.f.l0 dm.m00) (mulLo rs.g.l0 dm.m01) : ℕ) : ℤ)
        = (t.f * N.a + t.g * N.b) % 2^64 := addw_word _ _ _ _ h1 h2
    have hw' : addw (mulLo rs.f.l0 dm.m00) (mulLo rs.g.l0 dm.m01) = sw := by
      have : ((addw (mulLo rs.f.l0 dm.m00) (mulLo rs.g.l0 dm.m01) : ℕ) : ℤ) = sw := by
        rw [hw, hsw']; exact congrArg (· % 2^64) (by ring)
      exact_mod_cast this
    unfold signWord regMod
    rw [hw']
  rw [hsign]
  -- The masks of the row: the entries with the sign folded in.
  set e : ℤ := if sw < 2^63 then 1 else -1 with he
  have hsm := hB.signMag N.a N.b N.c N.d dm.m00 dm.m01 dm.m10 dm.m11 (by omega) (by omega)
    (by omega) (by omega) hM00 hM01 hM10 hM11
  generalize hsm' : B.signMag dm.m00 dm.m01 dm.m10 dm.m11 = sm at hsm ⊢
  have hr00 : SignMagRep sm.m00 (sm.s00 ^^^ (if sw < 2^63 then 0 else 2^64 - 1)) (e * N.a) := by
    rw [hsm, he]
    split_ifs
    · rw [one_mul]; exact (SignMagRep.of_natAbs _).xor_zero _ _ _
    · rw [neg_one_mul]; exact (SignMagRep.of_natAbs _).xor_ones _ _ _
  have hr01 : SignMagRep sm.m01 (sm.s01 ^^^ (if sw < 2^63 then 0 else 2^64 - 1)) (e * N.b) := by
    rw [hsm, he]
    split_ifs
    · rw [one_mul]; exact (SignMagRep.of_natAbs _).xor_zero _ _ _
    · rw [neg_one_mul]; exact (SignMagRep.of_natAbs _).xor_ones _ _ _
  have he1 : |e| = 1 := by rw [he]; split_ifs <;> simp
  have hrow' : |e * N.a| + |e * N.b| ≤ 2^59 := by
    rw [abs_mul, abs_mul, he1, one_mul, one_mul]; exact hrow1
  -- The row and its reduction.
  obtain ⟨hTb, hT⟩ := hB.uvRow (e * N.a) (e * N.b) rs.u rs.v sm.m00 sm.m01 _ _ hub hvb
    (le_trans hrow' (by norm_num)) hr00 hr01
  have hu' : |(rs.u.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (Int.natCast_nonneg _)]; exact_mod_cast Limbs.toNat_lt _ hub
  have hv' : |(rs.v.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (Int.natCast_nonneg _)]; exact_mod_cast Limbs.toNat_lt _ hvb
  have hval : e * N.a * rs.u.toNat + e * N.b * rs.v.toNat
      = e * (N.a * rs.u.toNat + N.b * rs.v.toNat) := by ring
  have htv : |e * (N.a * rs.u.toNat + N.b * rs.v.toNat)| < 2^315 := by
    rw [← hval, ← show (2 : ℤ)^59 * 2^256 = 2^315 by norm_num]
    exact Inversion.row_abs_lt (e * N.a) (e * N.b) _ _ _ _ hrow' hu' hv' (by positivity)
  set r : ℤ := amontredZ (e * (N.a * rs.u.toNat + N.b * rs.v.toNat)) F.modulus.toNat F.inv
    with hr
  have hA : B.amontred (B.uvRow rs.u rs.v sm.m00 sm.m01
      (sm.s00 ^^^ (if sw < 2^63 then 0 else 2^64 - 1))
      (sm.s01 ^^^ (if sw < 2^63 then 0 else 2^64 - 1))) F.modulus F.inv
      = Limbs.ofNat r.toNat := by
    rw [hB.amontred _ hTb (by rw [hT, hval]; exact htv)]
    unfold Inversion.amontred
    rw [hT, hval]
  obtain ⟨hr0, -, hr2p, hr256, -⟩ := Inversion.amontredZ_spec F _ htv
  rw [← hr] at hr0 hr2p hr256
  -- The conditional subtraction is the model's.
  have hp254 : 2^254 ≤ F.modulus.toNat := F.two_pow_le_modulus
  have hp255 : F.modulus.toNat < 2^255 := F.modulus_lt
  have hAt : (Limbs.ofNat r.toNat).toNat = r.toNat := Limbs.toNat_ofNat _ (by omega)
  obtain ⟨hRb, hR⟩ := hB.condSub (Limbs.ofNat r.toNat) (Limbs.ofNat_bounded _)
  rw [hAt] at hR
  have hqlt : (if r < F.modulus.toNat then r else r - F.modulus.toNat).toNat < 2^256 := by
    split_ifs <;> omega
  rw [hA]
  apply Limbs.ext_of_toNat _ _ hRb (Limbs.ofNat_bounded _)
  rw [Limbs.toNat_ofNat _ hqlt]
  rcases hR with ⟨h1, h2⟩ | ⟨h1, h2⟩
  · rw [h2, if_pos (by omega)]
  · rw [if_neg (by omega)]; omega

/-- The crate's `invert` over blocks that meet their specification, at a Pasta field. The input
is canonical, as the entry point asserts, and `v0` is `2^562 mod p`, as its contract requires.
The result is canonical. For `x = 0` it is `0`; otherwise it is the Montgomery inverse, with
`x * result ≡ R^2 (mod p)`. The termination bound of the divstep recurrence is discharged by
the hull certificate's `terminationBound_256`, and the primality of `p` is `F.prime`. -/
theorem invert_entry_spec (B : InvertBlocks) (F : PastaField) (hB : B.Spec F) (x : Limbs)
    (hx : x.Bounded)
    (h : isCanonical x F.modulus = true) (v0 : Limbs) (hv0 : v0 = Inversion.startV F) :
    (invert B x F.modulus F.inv v0).Bounded ∧
      (invert B x F.modulus F.inv v0).toNat < F.modulus.toNat ∧
      (x.toNat = 0 → invert B x F.modulus F.inv v0 = Limbs.ofNat 0) ∧
      (x.toNat ≠ 0 →
        x.toNat * (invert B x F.modulus F.inv v0).toNat ≡ R^2 [MOD F.modulus.toNat]) := by
  rw [hv0, invert_eq_model B F hB x hx]
  exact Inversion.montInv_spec F Inversion.Hull.terminationBound_256 x hx
    ((isCanonical_iff x F.modulus hx F.bounded).1 h)

end PastaCurves
