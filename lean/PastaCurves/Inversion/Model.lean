import Mathlib.Data.Nat.Prime.Basic
import PastaCurves.Inversion.Round
import PastaCurves.Inversion.Termination

/-!
# The composition and the correctness theorem

`montInvModel` is the whole algorithm on words: nine full rounds of `divstep59`, `updateFG`,
and `updateUV` from `(d, f, g, u, v) = (1, p, x, 0, 2^562 mod p)`, then the last round, which
computes only `u` with the sign of the final `f` folded in and reduces strictly.

`rounds_invariant` is Lemma 11 at the word level: after `i` rounds the word state carries the
true divstep state after `59 i` steps, and `f_i 2^562 ≡ x 2^(5 i) u_i`, `g_i 2^562 ≡ x 2^(5 i) v_i`
modulo `p`. `montInv_spec` is Theorem 12, with the termination bound and the primality of `p` as
hypotheses.
-/

namespace PastaCurves

/-- A bounded limb vector representing zero is the zero vector. -/
theorem Limbs.eq_ofNat_zero (x : Limbs) (hx : x.Bounded) (h : x.toNat = 0) : x = Limbs.ofNat 0 := by
  obtain ⟨l0, l1, l2, l3⟩ := x
  obtain ⟨h0, h1, h2, h3⟩ := hx
  simp only [Limbs.toNat] at h
  have : l0 = 0 ∧ l1 = 0 ∧ l2 = 0 ∧ l3 = 0 := by omega
  obtain ⟨rfl, rfl, rfl, rfl⟩ := this
  rfl

/-- The modulus is odd, since `inv` inverts its low limb modulo `2^64`. -/
theorem PastaField.modulus_odd (F : PastaField) : F.modulus.toNat % 2 = 1 := by
  have h := F.inv_spec
  have hl0 : F.modulus.l0 % 2 = 1 := by
    rcases Nat.even_or_odd F.modulus.l0 with ⟨k, hk⟩ | hodd
    · exfalso
      rw [hk, show F.inv * (k + k) = 2 * (F.inv * k) by ring] at h
      omega
    · exact Nat.odd_iff.mp hodd
  unfold Limbs.toNat
  omega

theorem PastaField.gcd_two_pow (F : PastaField) (n : ℕ) :
    Int.gcd (F.modulus.toNat : ℤ) (2^n) = 1 := by
  have hc : Nat.Coprime F.modulus.toNat (2^n) := by
    apply Nat.Coprime.pow_right
    rw [Nat.Coprime, Nat.gcd_comm, Nat.gcd_rec, F.modulus_odd]
    rfl
  have : ((2 : ℤ)^n) = ((2^n : ℕ) : ℤ) := by push_cast; rfl
  rw [this, Int.gcd_natCast_natCast]
  exact hc

theorem PastaField.modulus_pos (F : PastaField) : (0 : ℤ) < F.modulus.toNat := by
  have := F.two_pow_le_modulus
  exact_mod_cast lt_of_lt_of_le (by norm_num) this

/-- Cancel a power of two in a congruence modulo the odd `p`. -/
theorem PastaField.modEq_cancel_two_pow (F : PastaField) (n : ℕ) {a b : ℤ}
    (h : 2^n * a ≡ 2^n * b [ZMOD F.modulus.toNat]) : a ≡ b [ZMOD F.modulus.toNat] := by
  have := Int.ModEq.cancel_left_div_gcd F.modulus_pos h
  rwa [F.gcd_two_pow, Nat.cast_one, Int.ediv_one] at this

end PastaCurves

set_option exponentiation.threshold 600

namespace PastaCurves.Inversion

/-! ## Five words from four -/

/-- Four limbs as a five-word signed value, sign word zero. -/
def Signed5.ofLimbs (x : Limbs) : Signed5 := ⟨x.l0, x.l1, x.l2, x.l3, 0⟩

theorem Signed5.ofLimbs_bounded (x : Limbs) (hx : x.Bounded) : (Signed5.ofLimbs x).Bounded := by
  obtain ⟨h0, h1, h2, h3⟩ := hx
  exact ⟨h0, h1, h2, h3, by show (0 : ℕ) < 2^64; norm_num⟩

theorem Signed5.toInt_ofLimbs (x : Limbs) : (Signed5.ofLimbs x).toInt = x.toNat := by
  simp only [Signed5.toInt, Signed5.ofLimbs, Limbs.toNat]
  rw [if_pos (by norm_num)]
  push_cast
  ring

/-- The low word of a bounded five-word value is its value modulo `2^64`. -/
theorem Signed5.toInt_emod (x : Signed5) (hx : x.Bounded) : x.toInt % 2^64 = x.l0 := by
  obtain ⟨h0, h1, h2, h3, h4⟩ := hx
  unfold Signed5.toInt
  split_ifs <;> omega

/-! ## The composition -/

/-- `2^562 mod p`, the starting `v`. -/
def startV (F : PastaField) : Limbs := Limbs.ofNat (2^562 % F.modulus.toNat)

/-- The word-level state carried between rounds. -/
structure RoundState where
  d : ℤ
  f : Signed5
  g : Signed5
  u : Limbs
  v : Limbs

/-- One full round: `divstep59` on the low words, `updateFG`, `updateUV`. -/
def round (F : PastaField) (st : RoundState) : RoundState :=
  let dm := divstep59 st.d st.f.l0 st.g.l0
  let fg := updateFG dm.2 st.f st.g
  let uv := updateUV dm.2 st.u st.v F.modulus F.inv
  ⟨dm.1, fg.1, fg.2, uv.1, uv.2⟩

/-- The starting state `(1, p, x, 0, 2^562 mod p)`. -/
def initState (F : PastaField) (x : Limbs) : RoundState :=
  ⟨1, Signed5.ofLimbs F.modulus, Signed5.ofLimbs x, Limbs.ofNat 0, startV F⟩

/-- The state after `i` full rounds. -/
def rounds (F : PastaField) (x : Limbs) (i : ℕ) : RoundState := (round F)^[i] (initState F x)

/-- The sign word of the last round: the low word of `m00 f + m01 g`, from the low words. -/
def signWordOf (M : Mat2) (f g : Signed5) : ℕ := ((M.a * f.l0 + M.b * g.l0) % 2^64).toNat

/-- The whole algorithm on words: nine full rounds and the last one. -/
def montInvModel (F : PastaField) (x : Limbs) : Limbs :=
  let st := rounds F x 9
  let dm := divstep59 st.d st.f.l0 st.g.l0
  finalU dm.2 (signWordOf dm.2 st.f st.g) st.u st.v F.modulus F.inv

/-- The true divstep state after `59 i` steps from `(1, p, x)`. -/
def trueState (F : PastaField) (x : Limbs) (i : ℕ) : State :=
  divsteps (59 * i) ⟨1, (F.modulus.toNat : ℤ), (x.toNat : ℤ)⟩

/-! ## The invariant -/

/-- Lemma 11 at the word level. -/
theorem rounds_invariant (F : PastaField) (x : Limbs) (hx : x.Bounded) (i : ℕ) :
    (rounds F x i).d = (trueState F x i).d ∧
      (rounds F x i).f.Bounded ∧ (rounds F x i).g.Bounded ∧
      (rounds F x i).f.toInt = (trueState F x i).f ∧
      (rounds F x i).g.toInt = (trueState F x i).g ∧
      (rounds F x i).u.Bounded ∧ (rounds F x i).v.Bounded ∧
      (rounds F x i).f.toInt * 2^562 ≡ x.toNat * 2^(5 * i) * (rounds F x i).u.toNat
        [ZMOD F.modulus.toNat] ∧
      (rounds F x i).g.toInt * 2^562 ≡ x.toNat * 2^(5 * i) * (rounds F x i).v.toNat
        [ZMOD F.modulus.toNat] ∧
      (x.toNat = 0 → ((rounds F x i).u.toNat : ℤ) ≡ 0 [ZMOD F.modulus.toNat]) := by
  have hp_odd : (F.modulus.toNat : ℤ) % 2 = 1 := by exact_mod_cast F.modulus_odd
  have hp255 : (F.modulus.toNat : ℤ) < 2^255 := by exact_mod_cast F.modulus_lt
  have hx256 : (x.toNat : ℤ) < 2^256 := by exact_mod_cast Limbs.toNat_lt x hx
  induction i with
  | zero =>
    have hlt : 2^562 % F.modulus.toNat < 2^256 :=
      lt_trans (Nat.mod_lt _ (by have := F.two_pow_le_modulus; omega)) (by have := F.modulus_lt; omega)
    simp only [rounds, Function.iterate_zero, id, initState, trueState, Nat.mul_zero, divsteps_zero]
    refine ⟨trivial, Signed5.ofLimbs_bounded _ F.bounded, Signed5.ofLimbs_bounded _ hx,
      Signed5.toInt_ofLimbs _, Signed5.toInt_ofLimbs _, Limbs.ofNat_bounded _,
      Limbs.ofNat_bounded _, ?_, ?_, ?_⟩
    · rw [Signed5.toInt_ofLimbs, Limbs.toNat_ofNat 0 (by norm_num)]
      simp only [Nat.cast_zero, mul_zero]
      exact Int.modEq_zero_iff_dvd.2 (dvd_mul_right _ _)
    · rw [Signed5.toInt_ofLimbs, startV, Limbs.toNat_ofNat _ hlt]
      push_cast
      simp only [mul_one]
      exact Int.ModEq.mul_left _ (Int.mod_modEq _ _).symm
    · intro _
      rw [Limbs.toNat_ofNat 0 (by norm_num)]
      exact Int.ModEq.refl _
  | succ i ih =>
    obtain ⟨hd, hfb, hgb, hf, hg, hub, hvb, hcf, hcg, hu0⟩ := ih
    set st := rounds F x i with hst
    set t := trueState F x i with ht
    have hround : rounds F x (i + 1) = round F st := by
      rw [hst, rounds, rounds, Function.iterate_succ_apply']
    have ht' : trueState F x (i + 1) = divsteps 59 t := by
      rw [ht, trueState, trueState, show 59 * (i + 1) = 59 * i + 59 by ring, divsteps_add]
    have hodd : t.f % 2 = 1 := divsteps_f_odd _ _ hp_odd
    have hpB : |(F.modulus.toNat : ℤ)| ≤ 2^256 - 1 := by
      rw [abs_of_nonneg (by positivity)]; omega
    have hxB : |(x.toNat : ℤ)| ≤ 2^256 - 1 := by
      rw [abs_of_nonneg (by positivity)]; omega
    have hfB : |t.f| ≤ 2^256 - 1 := by
      rw [ht, trueState]; exact (divsteps_abs_le (59 * i) _ _ hpB hxB).1
    have hgB : |t.g| ≤ 2^256 - 1 := by
      rw [ht, trueState]; exact (divsteps_abs_le (59 * i) _ _ hpB hxB).2
    have hM := M_rowSum_le 59 t
    have hfl : st.f.l0 = (t.f % 2^64).toNat := by
      have h := Signed5.toInt_emod st.f hfb
      rw [hf] at h
      rw [h, Int.toNat_natCast]
    have hgl : st.g.l0 = (t.g % 2^64).toNat := by
      have h := Signed5.toInt_emod st.g hgb
      rw [hg] at h
      rw [h, Int.toNat_natCast]
    have hdm := divstep59_spec t hodd
    rw [← hd, ← hfl, ← hgl] at hdm
    obtain ⟨hs1, hs2⟩ := M_spec 59 t hodd
    have hfv : |st.f.toInt| < 2^256 := by rw [hf, abs_lt]; rw [abs_le] at hfB; constructor <;> omega
    have hgv : |st.g.toInt| < 2^256 := by rw [hg, abs_lt]; rw [abs_le] at hgB; constructor <;> omega
    obtain ⟨hf'b, hg'b, hf', hg'⟩ := updateFG_spec (M 59 t) st.f st.g hfv hgv hM
    have hF' : (updateFG (M 59 t) st.f st.g).1.toInt = (divsteps 59 t).f := by
      rw [hf', hf, hg, ← hs1, Int.mul_ediv_cancel_left _ (by positivity)]
    have hG' : (updateFG (M 59 t) st.f st.g).2.toInt = (divsteps 59 t).g := by
      rw [hg', hf, hg, ← hs2, Int.mul_ediv_cancel_left _ (by positivity)]
    obtain ⟨hu'b, hv'b, -, -, hcu, hcv⟩ := updateUV_spec F (M 59 t) st.u st.v hub hvb hM
    rw [hf] at hcf
    rw [hg] at hcg
    rw [hround, ht']
    simp only [round, hdm.1, hdm.2]
    refine ⟨trivial, hf'b, hg'b, hF', hG', hu'b, hv'b, ?_, ?_, ?_⟩
    · rw [hF']
      apply F.modEq_cancel_two_pow 59
      calc 2^59 * ((divsteps 59 t).f * 2^562)
          = (M 59 t).a * (t.f * 2^562) + (M 59 t).b * (t.g * 2^562) := by
            linear_combination 2^562 * hs1
        _ ≡ (M 59 t).a * (x.toNat * 2^(5 * i) * st.u.toNat)
            + (M 59 t).b * (x.toNat * 2^(5 * i) * st.v.toNat) [ZMOD F.modulus.toNat] :=
            (hcf.mul_left _).add (hcg.mul_left _)
        _ = x.toNat * 2^(5 * i) * ((M 59 t).a * st.u.toNat + (M 59 t).b * st.v.toNat) := by ring
        _ ≡ x.toNat * 2^(5 * i)
            * (((updateUV (M 59 t) st.u st.v F.modulus F.inv).1.toNat : ℤ) * 2^64)
            [ZMOD F.modulus.toNat] := hcu.symm.mul_left _
        _ = 2^59 * (x.toNat * 2^(5 * (i + 1))
            * ((updateUV (M 59 t) st.u st.v F.modulus F.inv).1.toNat : ℤ)) := by ring
    · rw [hG']
      apply F.modEq_cancel_two_pow 59
      calc 2^59 * ((divsteps 59 t).g * 2^562)
          = (M 59 t).c * (t.f * 2^562) + (M 59 t).d * (t.g * 2^562) := by
            linear_combination 2^562 * hs2
        _ ≡ (M 59 t).c * (x.toNat * 2^(5 * i) * st.u.toNat)
            + (M 59 t).d * (x.toNat * 2^(5 * i) * st.v.toNat) [ZMOD F.modulus.toNat] :=
            (hcf.mul_left _).add (hcg.mul_left _)
        _ = x.toNat * 2^(5 * i) * ((M 59 t).c * st.u.toNat + (M 59 t).d * st.v.toNat) := by ring
        _ ≡ x.toNat * 2^(5 * i)
            * (((updateUV (M 59 t) st.u st.v F.modulus F.inv).2.toNat : ℤ) * 2^64)
            [ZMOD F.modulus.toNat] := hcv.symm.mul_left _
        _ = 2^59 * (x.toNat * 2^(5 * (i + 1))
            * ((updateUV (M 59 t) st.u st.v F.modulus F.inv).2.toNat : ℤ)) := by ring
    · intro hx0
      have hu0' := hu0 hx0
      -- With `x = 0`, `g` is zero throughout and the matrix is `[[2^59, 0], [0, 1]]`.
      have htg : t.g = 0 := by
        rw [ht, trueState]
        exact (divsteps_of_g_zero _ _ (by simp [hx0])).2.1
      have hMt : M 59 t = ⟨2^59, 0, 0, 1⟩ := (divsteps_of_g_zero 59 t htg).2.2
      apply F.modEq_cancel_two_pow 64
      calc 2^64 * ((updateUV (M 59 t) st.u st.v F.modulus F.inv).1.toNat : ℤ)
          = ((updateUV (M 59 t) st.u st.v F.modulus F.inv).1.toNat : ℤ) * 2^64 := by ring
        _ ≡ (M 59 t).a * st.u.toNat + (M 59 t).b * st.v.toNat [ZMOD F.modulus.toNat] := hcu
        _ = 2^59 * (st.u.toNat : ℤ) := by rw [hMt]; ring
        _ ≡ 2^59 * 0 [ZMOD F.modulus.toNat] := hu0'.mul_left _
        _ = 2^64 * 0 := by ring

/-! ## The theorem -/

/-- Theorem 12: for a canonical input, the model returns the canonical Montgomery residue `z`
with `x z ≡ R^2 (mod p)`, and zero for zero. The termination bound and the primality of `p`
are hypotheses. -/
theorem montInv_spec (F : PastaField) (hbound : TerminationBound 256)
    (hprime : Nat.Prime F.modulus.toNat) (x : Limbs) (hx : x.Bounded)
    (hxlt : x.toNat < F.modulus.toNat) :
    (montInvModel F x).Bounded ∧ (montInvModel F x).toNat < F.modulus.toNat ∧
      (x.toNat = 0 → montInvModel F x = Limbs.ofNat 0) ∧
      (x.toNat ≠ 0 → x.toNat * (montInvModel F x).toNat ≡ R^2 [MOD F.modulus.toNat]) := by
  have hp_odd : (F.modulus.toNat : ℤ) % 2 = 1 := by exact_mod_cast F.modulus_odd
  have hp255 : (F.modulus.toNat : ℤ) < 2^255 := by exact_mod_cast F.modulus_lt
  obtain ⟨hd, hfb, hgb, hf, hg, hub, hvb, hcf, hcg, hu0⟩ := rounds_invariant F x hx 9
  set st := rounds F x 9 with hst
  set t := trueState F x 9 with ht
  have hodd : t.f % 2 = 1 := divsteps_f_odd _ _ hp_odd
  have hM := M_rowSum_le 59 t
  have hfl0 : t.f % 2^64 = st.f.l0 := by
    have h := Signed5.toInt_emod st.f hfb; rw [hf] at h; exact h
  have hgl0 : t.g % 2^64 = st.g.l0 := by
    have h := Signed5.toInt_emod st.g hgb; rw [hg] at h; exact h
  have hfl : st.f.l0 = (t.f % 2^64).toNat := by rw [hfl0, Int.toNat_natCast]
  have hgl : st.g.l0 = (t.g % 2^64).toNat := by rw [hgl0, Int.toNat_natCast]
  have hdm := divstep59_spec t hodd
  rw [← hd, ← hfl, ← hgl] at hdm
  obtain ⟨hs1, -⟩ := M_spec 59 t hodd
  set t10 := divsteps 59 t with ht10
  have ht10' : t10 = divsteps 590 ⟨1, (F.modulus.toNat : ℤ), (x.toNat : ℤ)⟩ := by
    rw [ht10, ht, trueState, ← divsteps_add]
  have h590 : iterations 256 = 590 := by decide
  have hg0 : t10.g = 0 := by
    have hb := hbound (F.modulus.toNat : ℤ) (x.toNat : ℤ) hp_odd (Int.natCast_nonneg _)
      (by exact_mod_cast hxlt.le) (by omega)
    rw [h590] at hb
    rw [ht10']
    exact hb
  -- The model's result is the last round on the state after nine.
  set sw := signWordOf (M 59 t) st.f st.g with hsw
  have hres : montInvModel F x = finalU (M 59 t) sw st.u st.v F.modulus F.inv := by
    show finalU (divstep59 st.d st.f.l0 st.g.l0).2
      (signWordOf (divstep59 st.d st.f.l0 st.g.l0).2 st.f st.g) st.u st.v F.modulus F.inv = _
    rw [hdm.2]
  obtain ⟨hrb, hrlt, hrc⟩ := finalU_spec F (M 59 t) sw st.u st.v hub hvb hM.1
  rw [hres]
  set r : ℤ := ((finalU (M 59 t) sw st.u st.v F.modulus F.inv).toNat : ℤ) with hr
  set e : ℤ := if sw < 2^63 then (1 : ℤ) else -1 with he
  -- The sign word is the low word of `2^59 f_10`.
  have hsw' : (sw : ℤ) = (2^59 * t10.f) % 2^64 := by
    rw [hsw, signWordOf, Int.toNat_of_nonneg (Int.emod_nonneg _ (by norm_num)), hs1]
    have h1 : (st.f.l0 : ℤ) ≡ t.f [ZMOD 2^64] := by rw [← hfl0]; exact Int.mod_modEq _ _
    have h2 : (st.g.l0 : ℤ) ≡ t.g [ZMOD 2^64] := by rw [← hgl0]; exact Int.mod_modEq _ _
    exact (h1.mul_left _).add (h2.mul_left _)
  refine ⟨hrb, hrlt, ?_, ?_⟩
  · -- `x = 0`: the matrices are `[[2^59, 0], [0, 1]]`, `u ≡ 0`, and the result is canonical.
    intro hx0
    have hu0' := hu0 hx0
    have htg : t.g = 0 := by
      rw [ht, trueState]
      exact (divsteps_of_g_zero _ _ (by simp [hx0])).2.1
    have hMt : M 59 t = ⟨2^59, 0, 0, 1⟩ := (divsteps_of_g_zero 59 t htg).2.2
    have hr0 : r ≡ 0 [ZMOD F.modulus.toNat] := by
      apply F.modEq_cancel_two_pow 64
      calc 2^64 * r = r * 2^64 := by ring
        _ ≡ e * ((M 59 t).a * st.u.toNat + (M 59 t).b * st.v.toNat) [ZMOD F.modulus.toNat] :=
            hrc
        _ = e * 2^59 * st.u.toNat := by rw [hMt]; ring
        _ ≡ e * 2^59 * 0 [ZMOD F.modulus.toNat] := hu0'.mul_left _
        _ = 2^64 * 0 := by ring
    have hdvd : F.modulus.toNat ∣ (finalU (M 59 t) sw st.u st.v F.modulus F.inv).toNat :=
      Int.natCast_dvd_natCast.mp (Int.modEq_zero_iff_dvd.1 hr0)
    exact Limbs.eq_ofNat_zero _ hrb (Nat.eq_zero_of_dvd_of_lt hdvd hrlt)
  · -- `x ≠ 0`: `f_10 = ±1`, the sign word reads it, and the invariant gives `x z ≡ 2^512`.
    intro hx0
    obtain ⟨hdp, hdx⟩ := f_dvd_of_g_eq_zero 590 _ hp_odd (ht10' ▸ hg0)
    rw [← ht10'] at hdp hdx
    have hnat : t10.f.natAbs ∣ F.modulus.toNat := by
      have h := Int.natAbs_dvd_natAbs.mpr hdp
      rwa [Int.natAbs_natCast] at h
    have hpm : t10.f = 1 ∨ t10.f = -1 := by
      rcases Nat.Prime.eq_one_or_self_of_dvd hprime _ hnat with h1 | hp
      · rcases Int.natAbs_eq_iff.mp h1 with h | h
        · exact Or.inl (by rw [h]; rfl)
        · exact Or.inr (by rw [h]; rfl)
      · exfalso
        have h : F.modulus.toNat ∣ x.toNat := by
          have h' := Int.natAbs_dvd_natAbs.mpr hdx
          rwa [hp, Int.natAbs_natCast] at h'
        exact hx0 (Nat.eq_zero_of_dvd_of_lt h hxlt)
    have he' : e = t10.f := by
      rcases hpm with h | h
      · rw [h] at hsw'
        have : sw < 2^63 := by omega
        rw [he, if_pos this, h]
      · rw [h] at hsw'
        have : ¬ sw < 2^63 := by omega
        rw [he, if_neg this, h]
    have hff : t10.f * t10.f = 1 := by rcases hpm with h | h <;> rw [h] <;> norm_num
    have hee : e * e = 1 := by rw [he']; exact hff
    rw [hf] at hcf
    rw [hg] at hcg
    have step1 : 2^59 * (t10.f * 2^562) ≡ x.toNat * 2^45
        * ((M 59 t).a * st.u.toNat + (M 59 t).b * st.v.toNat) [ZMOD F.modulus.toNat] := by
      calc 2^59 * (t10.f * 2^562)
          = (M 59 t).a * (t.f * 2^562) + (M 59 t).b * (t.g * 2^562) := by
            linear_combination 2^562 * hs1
        _ ≡ (M 59 t).a * (x.toNat * 2^(5 * 9) * st.u.toNat)
            + (M 59 t).b * (x.toNat * 2^(5 * 9) * st.v.toNat) [ZMOD F.modulus.toNat] :=
            (hcf.mul_left _).add (hcg.mul_left _)
        _ = x.toNat * 2^45 * ((M 59 t).a * st.u.toNat + (M 59 t).b * st.v.toNat) := by ring
    have step2 : e * (r * 2^64) ≡ (M 59 t).a * st.u.toNat + (M 59 t).b * st.v.toNat
        [ZMOD F.modulus.toNat] := by
      calc e * (r * 2^64) ≡ e * (e * ((M 59 t).a * st.u.toNat + (M 59 t).b * st.v.toNat))
            [ZMOD F.modulus.toNat] := hrc.mul_left e
        _ = (e * e) * ((M 59 t).a * st.u.toNat + (M 59 t).b * st.v.toNat) := by ring
        _ = (M 59 t).a * st.u.toNat + (M 59 t).b * st.v.toNat := by rw [hee]; ring
    have step3 : 2^109 * (2^512 * t10.f) ≡ 2^109 * (x.toNat * (e * r)) [ZMOD F.modulus.toNat] := by
      calc 2^109 * (2^512 * t10.f) = 2^59 * (t10.f * 2^562) := by ring
        _ ≡ x.toNat * 2^45 * ((M 59 t).a * st.u.toNat + (M 59 t).b * st.v.toNat)
            [ZMOD F.modulus.toNat] := step1
        _ ≡ x.toNat * 2^45 * (e * (r * 2^64)) [ZMOD F.modulus.toNat] := step2.symm.mul_left _
        _ = 2^109 * (x.toNat * (e * r)) := by ring
    have step4 := F.modEq_cancel_two_pow 109 step3
    have key : (x.toNat : ℤ) * r ≡ 2^512 [ZMOD F.modulus.toNat] := by
      calc (x.toNat : ℤ) * r = t10.f * (x.toNat * (t10.f * r)) := by
            linear_combination (-(x.toNat * r)) * hff
        _ = t10.f * (x.toNat * (e * r)) := by rw [he']
        _ ≡ t10.f * (2^512 * t10.f) [ZMOD F.modulus.toNat] := step4.symm.mul_left _
        _ = 2^512 := by linear_combination 2^512 * hff
    have hR : R^2 = 2^512 := by norm_num [R]
    rw [hr] at key
    rw [hR]
    unfold Nat.ModEq
    unfold Int.ModEq at key
    exact_mod_cast key

end PastaCurves.Inversion
