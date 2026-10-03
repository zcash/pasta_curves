import PastaCurves.Fields
import PastaCurves.Spec
import PastaCurves.Inversion.Divstep59

/-!
# The round arithmetic on words

The three word-level functions of a round besides `divstep59`, and the facts about them that the
round invariant needs (§3 of `book/src/design/inversion.md`). `f` and `g` are five-word signed
values (`Signed5`), `u` and `v` four-word unsigned values (`Limbs`). The functions are defined by
their integer effect with word-shaped inputs and outputs: `updateFG` encodes the exact quotients
`(m f + m' g) / 2^59`, and `amontred` is one word of Montgomery reduction after adding `2^61 p`. The
per-word carry structure belongs to an implementation's proof that it computes these functions.
-/

set_option exponentiation.threshold 400

namespace PastaCurves

/-- The limbs of a natural number are bounded. -/
theorem Limbs.ofNat_bounded (n : ℕ) : (Limbs.ofNat n).Bounded := by
  unfold Limbs.ofNat Limbs.Bounded
  refine ⟨?_, ?_, ?_, ?_⟩ <;> exact Nat.mod_lt _ (by norm_num)

/-- Below `2^256`, the limbs of a number give it back. -/
theorem Limbs.toNat_ofNat (n : ℕ) (h : n < 2^256) : (Limbs.ofNat n).toNat = n := by
  simp only [Limbs.ofNat, Limbs.toNat]
  omega

/-- A bounded five-word value is determined by the integer it represents. -/
theorem Signed5.ext_of_toInt (x y : Signed5) (hx : x.Bounded) (hy : y.Bounded)
    (h : x.toInt = y.toInt) : x = y := by
  obtain ⟨hx0, hx1, hx2, hx3, hx4⟩ := hx
  obtain ⟨hy0, hy1, hy2, hy3, hy4⟩ := hy
  unfold Signed5.toInt at h
  have e0 : x.l0 = y.l0 := by split_ifs at h <;> omega
  have e1 : x.l1 = y.l1 := by split_ifs at h <;> omega
  have e2 : x.l2 = y.l2 := by split_ifs at h <;> omega
  have e3 : x.l3 = y.l3 := by split_ifs at h <;> omega
  have e4 : x.l4 = y.l4 := by split_ifs at h <;> omega
  calc x = ⟨x.l0, x.l1, x.l2, x.l3, x.l4⟩ := rfl
    _ = ⟨y.l0, y.l1, y.l2, y.l3, y.l4⟩ := by rw [e0, e1, e2, e3, e4]
    _ = y := rfl

/-- A bounded four-limb value is determined by the natural number it represents. -/
theorem Limbs.ext_of_toNat (x y : Limbs) (hx : x.Bounded) (hy : y.Bounded)
    (h : x.toNat = y.toNat) : x = y := by
  obtain ⟨hx0, hx1, hx2, hx3⟩ := hx
  obtain ⟨hy0, hy1, hy2, hy3⟩ := hy
  unfold Limbs.toNat at h
  have e0 : x.l0 = y.l0 := by omega
  have e1 : x.l1 = y.l1 := by omega
  have e2 : x.l2 = y.l2 := by omega
  have e3 : x.l3 = y.l3 := by omega
  calc x = ⟨x.l0, x.l1, x.l2, x.l3⟩ := rfl
    _ = ⟨y.l0, y.l1, y.l2, y.l3⟩ := by rw [e0, e1, e2, e3]
    _ = y := rfl

/-- The modulus is at least `2^254`, by its shape. -/
theorem PastaField.two_pow_le_modulus (F : PastaField) : 2^254 ≤ F.modulus.toNat := by
  obtain ⟨h2, h3⟩ := F.shape
  unfold Limbs.toNat
  rw [h2, h3]
  omega

/-- The modulus is below `2^255`, by its shape. -/
theorem PastaField.modulus_lt (F : PastaField) : F.modulus.toNat < 2^255 :=
  Limbs.toNat_lt_of_shape F.modulus F.bounded F.shape

/-- `inv` is `-p⁻¹` modulo `2^64`, for the whole modulus and not only its low limb. -/
theorem PastaField.inv_mul_modulus (F : PastaField) :
    (F.inv * F.modulus.toNat + 1) % 2^64 = 0 := by
  obtain ⟨h2, h3⟩ := F.shape
  have hspec := F.inv_spec
  have hsplit : F.inv * F.modulus.toNat =
      F.inv * F.modulus.l0 + 2^64 * (F.inv * F.modulus.l1 + 2^190 * F.inv) := by
    unfold Limbs.toNat
    rw [h2, h3]
    ring
  rw [hsplit]
  omega

end PastaCurves

namespace PastaCurves.Inversion

/-! ## Five-word signed values

`Signed5`, its value `toInt`, and `Bounded` are in `PastaCurves/Semantics.lean`, beside `Limbs`,
since the transcribed blocks take and return them. -/

/-- The five words of a natural number; bits at and above `2^320` are dropped. -/
def Signed5.ofNat (n : ℕ) : Signed5 :=
  ⟨n % 2^64, n / 2^64 % 2^64, n / 2^128 % 2^64, n / 2^192 % 2^64, n / 2^256 % 2^64⟩

/-- The five words of an integer, in two's complement modulo `2^320`. -/
def Signed5.ofInt (z : ℤ) : Signed5 := Signed5.ofNat (z % 2^320).toNat

/-- The five words of a natural number are bounded. -/
theorem Signed5.ofNat_bounded (n : ℕ) : (Signed5.ofNat n).Bounded := by
  unfold Signed5.ofNat Signed5.Bounded
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> exact Nat.mod_lt _ (by norm_num)

/-- The five words of an integer are bounded. -/
theorem Signed5.ofInt_bounded (z : ℤ) : (Signed5.ofInt z).Bounded := Signed5.ofNat_bounded _

/-- The encoding is faithful below `2^319` in absolute value. -/
theorem Signed5.toInt_ofInt (z : ℤ) (hz : |z| < 2^319) : (Signed5.ofInt z).toInt = z := by
  have hpos : (0 : ℤ) < 2^320 := by positivity
  have hnn : 0 ≤ z % 2^320 := Int.emod_nonneg _ hpos.ne'
  obtain ⟨n, hn⟩ : ∃ n : ℕ, (n : ℤ) = z % 2^320 := ⟨_, Int.toNat_of_nonneg hnn⟩
  have hnt : (z % 2^320).toNat = n := by omega
  rw [abs_lt] at hz
  unfold Signed5.ofInt
  rw [hnt]
  unfold Signed5.ofNat Signed5.toInt
  dsimp only
  push_cast
  split_ifs <;> omega

/-! ## `updateFG` -/

/-- `updateFG`: `((m00 f + m01 g) / 2^59, (m10 f + m11 g) / 2^59)`, encoded in five words. -/
def updateFG (M : Mat2) (f g : Signed5) : Signed5 × Signed5 :=
  (Signed5.ofInt ((M.a * f.toInt + M.b * g.toInt) / 2^59),
   Signed5.ofInt ((M.c * f.toInt + M.d * g.toInt) / 2^59))

/-- A row combination is below `B · C` when the row sum is at most `B` and both values are
below `C`. -/
theorem row_abs_lt (a b f g B C : ℤ) (hab : |a| + |b| ≤ B) (hf : |f| < C) (hg : |g| < C)
    (hB : 0 < B) : |a * f + b * g| < B * C := by
  have hC : 0 < C := lt_of_le_of_lt (abs_nonneg f) hf
  calc |a * f + b * g| ≤ |a * f| + |b * g| := abs_add_le _ _
    _ = |a| * |f| + |b| * |g| := by rw [abs_mul, abs_mul]
    _ ≤ |a| * (C - 1) + |b| * (C - 1) :=
        add_le_add (mul_le_mul_of_nonneg_left (by omega) (abs_nonneg a))
          (mul_le_mul_of_nonneg_left (by omega) (abs_nonneg b))
    _ = (|a| + |b|) * (C - 1) := by ring
    _ ≤ B * (C - 1) := mul_le_mul_of_nonneg_right hab (by omega)
    _ < B * C := by linarith

/-- Lemma 9: the words are bounded and decode to the quotients, given the bounds that the
rounds maintain (`|f|, |g| < 2^256`, which admits a non-canonical four-word input). -/
theorem updateFG_spec (M : Mat2) (f g : Signed5)
    (hfv : |f.toInt| < 2^256) (hgv : |g.toInt| < 2^256)
    (hM : |M.a| + |M.b| ≤ 2^59 ∧ |M.c| + |M.d| ≤ 2^59) :
    (updateFG M f g).1.Bounded ∧ (updateFG M f g).2.Bounded ∧
      (updateFG M f g).1.toInt = (M.a * f.toInt + M.b * g.toInt) / 2^59 ∧
      (updateFG M f g).2.toInt = (M.c * f.toInt + M.d * g.toInt) / 2^59 := by
  have h1 := row_abs_lt _ _ _ _ _ _ hM.1 hfv hgv (by positivity)
  have h2 := row_abs_lt _ _ _ _ _ _ hM.2 hfv hgv (by positivity)
  rw [abs_lt] at h1 h2
  refine ⟨Signed5.ofInt_bounded _, Signed5.ofInt_bounded _, ?_, ?_⟩
  · apply Signed5.toInt_ofInt
    rw [abs_lt]; constructor <;> omega
  · apply Signed5.toInt_ofInt
    rw [abs_lt]; constructor <;> omega

/-! ## `amontred` -/

/-- The integer effect of `amontred`: add `2^61 p`, then one word of Montgomery reduction. -/
def amontredZ (t : ℤ) (p inv : ℕ) : ℤ :=
  let s : ℤ := t + 2^61 * (p : ℤ)
  let w : ℤ := (s * (inv : ℤ)) % 2^64
  (s + w * (p : ℤ)) / 2^64

/-- Lemma 10. The sharp bound is `8 r < 2^254 + 9 p`; it gives `r < 2 p` and `r < 2^256`. -/
theorem amontredZ_spec (F : PastaField) (t : ℤ) (ht : |t| < 2^315) :
    0 ≤ amontredZ t F.modulus.toNat F.inv ∧
      8 * amontredZ t F.modulus.toNat F.inv < 2^254 + 9 * F.modulus.toNat ∧
      amontredZ t F.modulus.toNat F.inv < 2 * F.modulus.toNat ∧
      amontredZ t F.modulus.toNat F.inv < 2^256 ∧
      amontredZ t F.modulus.toNat F.inv * 2^64 ≡ t [ZMOD F.modulus.toNat] := by
  have hp1 : 2^254 ≤ F.modulus.toNat := F.two_pow_le_modulus
  have hp2 : F.modulus.toNat < 2^255 := F.modulus_lt
  have hinv : ((F.inv : ℤ) * F.modulus.toNat + 1) % 2^64 = 0 := by
    exact_mod_cast F.inv_mul_modulus
  set p : ℤ := (F.modulus.toNat : ℤ) with hp
  have hp1' : (2 : ℤ)^254 ≤ p := by rw [hp]; exact_mod_cast hp1
  have hp2' : p < (2 : ℤ)^255 := by rw [hp]; exact_mod_cast hp2
  rw [abs_lt] at ht
  set s : ℤ := t + 2^61 * p with hs
  set w : ℤ := (s * (F.inv : ℤ)) % 2^64 with hw
  have hw0 : 0 ≤ w := Int.emod_nonneg _ (by norm_num)
  have hw1 : w < 2^64 := Int.emod_lt_of_pos _ (by norm_num)
  have hwp : w * p ≤ (2^64 - 1) * p := mul_le_mul_of_nonneg_right (by omega) (by omega)
  have hwp0 : 0 ≤ w * p := mul_nonneg hw0 (by omega)
  -- `s + w p` is a multiple of `2^64`.
  have h0 : (F.inv : ℤ) * p + 1 ≡ 0 [ZMOD 2^64] := by
    unfold Int.ModEq; rw [hinv, Int.zero_emod]
  have hwm : w ≡ s * F.inv [ZMOD 2^64] := Int.mod_modEq _ _
  have hmul : s + w * p ≡ 0 [ZMOD 2^64] := by
    calc s + w * p ≡ s + s * F.inv * p [ZMOD 2^64] := (hwm.mul_right p).add_left s
      _ = s * (F.inv * p + 1) := by ring
      _ ≡ s * 0 [ZMOD 2^64] := h0.mul_left s
      _ = 0 := mul_zero s
  have hdvd : (2 : ℤ)^64 ∣ s + w * p := Int.dvd_of_emod_eq_zero hmul
  have hexact : (s + w * p) / 2^64 * 2^64 = s + w * p := Int.ediv_mul_cancel hdvd
  have hr : amontredZ t F.modulus.toNat F.inv = (s + w * p) / 2^64 := rfl
  rw [hr]
  set r : ℤ := (s + w * p) / 2^64 with hrdef
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · exact Int.ediv_nonneg (by omega) (by norm_num)
  · omega
  · omega
  · omega
  · rw [Int.modEq_iff_dvd]
    exact ⟨-(2^61 + w), by linear_combination -hexact⟩

/-- `amontred` on five words: four words out, below `2^256`. -/
def amontred (t : Signed5) (modulus : Limbs) (inv : ℕ) : Limbs :=
  Limbs.ofNat (amontredZ t.toInt modulus.toNat inv).toNat

/-- Lemma 10 on words: the reduction is bounded, below `2p`, and congruent to `t / 2^64`. -/
theorem amontred_spec (F : PastaField) (t : Signed5) (htv : |t.toInt| < 2^315) :
    (amontred t F.modulus F.inv).Bounded ∧
      (amontred t F.modulus F.inv).toNat < 2 * F.modulus.toNat ∧
      ((amontred t F.modulus F.inv).toNat : ℤ) * 2^64 ≡ t.toInt [ZMOD F.modulus.toNat] := by
  obtain ⟨h0, -, h2, h3, h4⟩ := amontredZ_spec F t.toInt htv
  have hnat : (((amontredZ t.toInt F.modulus.toNat F.inv).toNat : ℕ) : ℤ) =
      amontredZ t.toInt F.modulus.toNat F.inv := Int.toNat_of_nonneg h0
  have hlt : (amontredZ t.toInt F.modulus.toNat F.inv).toNat < 2^256 := by omega
  unfold amontred
  rw [Limbs.toNat_ofNat _ hlt]
  refine ⟨Limbs.ofNat_bounded _, by omega, ?_⟩
  rw [hnat]; exact h4

/-! ## `updateUV` and the last round -/

/-- `updateUV`: the two row combinations, each reduced by `amontred`. -/
def updateUV (M : Mat2) (u v : Limbs) (modulus : Limbs) (inv : ℕ) : Limbs × Limbs :=
  (Limbs.ofNat (amontredZ (M.a * u.toNat + M.b * v.toNat) modulus.toNat inv).toNat,
   Limbs.ofNat (amontredZ (M.c * u.toNat + M.d * v.toNat) modulus.toNat inv).toNat)

/-- The integer form of the reduced row: bounded, below `2p`, congruent to the row over `2^64`. -/
theorem amontredZ_row (F : PastaField) (a b : ℤ) (u v : Limbs) (hu : u.Bounded) (hv : v.Bounded)
    (hab : |a| + |b| ≤ 2^59) :
    (Limbs.ofNat (amontredZ (a * u.toNat + b * v.toNat) F.modulus.toNat F.inv).toNat).Bounded ∧
      (Limbs.ofNat (amontredZ (a * u.toNat + b * v.toNat) F.modulus.toNat F.inv).toNat).toNat
        < 2 * F.modulus.toNat ∧
      ((Limbs.ofNat (amontredZ (a * u.toNat + b * v.toNat) F.modulus.toNat F.inv).toNat).toNat
        : ℤ) * 2^64 ≡ a * u.toNat + b * v.toNat [ZMOD F.modulus.toNat] := by
  have hu' : |(u.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (by positivity)]; exact_mod_cast Limbs.toNat_lt u hu
  have hv' : |(v.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (by positivity)]; exact_mod_cast Limbs.toNat_lt v hv
  have ht : |a * u.toNat + b * v.toNat| < 2^315 := by
    have := row_abs_lt a b _ _ _ _ hab hu' hv' (by positivity)
    norm_num at this ⊢; exact this
  obtain ⟨h0, -, h2, h3, h4⟩ := amontredZ_spec F _ ht
  have hnat : (((amontredZ (a * u.toNat + b * v.toNat) F.modulus.toNat F.inv).toNat : ℕ) : ℤ) =
      amontredZ (a * u.toNat + b * v.toNat) F.modulus.toNat F.inv := Int.toNat_of_nonneg h0
  have hlt : (amontredZ (a * u.toNat + b * v.toNat) F.modulus.toNat F.inv).toNat < 2^256 := by
    omega
  rw [Limbs.toNat_ofNat _ hlt]
  refine ⟨Limbs.ofNat_bounded _, by omega, ?_⟩
  rw [hnat]; exact h4

/-- Both reduced rows of `updateUV` are bounded and below `2p`, each congruent to its row over
`2^64`, under the row-sum bound. -/
theorem updateUV_spec (F : PastaField) (M : Mat2) (u v : Limbs)
    (hu : u.Bounded) (hv : v.Bounded)
    (hM : |M.a| + |M.b| ≤ 2^59 ∧ |M.c| + |M.d| ≤ 2^59) :
    (updateUV M u v F.modulus F.inv).1.Bounded ∧ (updateUV M u v F.modulus F.inv).2.Bounded ∧
      (updateUV M u v F.modulus F.inv).1.toNat < 2 * F.modulus.toNat ∧
      (updateUV M u v F.modulus F.inv).2.toNat < 2 * F.modulus.toNat ∧
      ((updateUV M u v F.modulus F.inv).1.toNat : ℤ) * 2^64
        ≡ M.a * u.toNat + M.b * v.toNat [ZMOD F.modulus.toNat] ∧
      ((updateUV M u v F.modulus F.inv).2.toNat : ℤ) * 2^64
        ≡ M.c * u.toNat + M.d * v.toNat [ZMOD F.modulus.toNat] := by
  obtain ⟨b1, l1, c1⟩ := amontredZ_row F M.a M.b u v hu hv hM.1
  obtain ⟨b2, l2, c2⟩ := amontredZ_row F M.c M.d u v hu hv hM.2
  exact ⟨b1, b2, l1, l2, c1, c2⟩

/-- The last round: only `u`, with the sign of the new `f` folded into the row, then one
conditional subtraction. `signWord` is the low word of `m00 f + m01 g`, whose bit 63 is the
sign. -/
def finalU (M : Mat2) (signWord : ℕ) (u v : Limbs) (modulus : Limbs) (inv : ℕ) : Limbs :=
  let t := (if signWord < 2^63 then (1 : ℤ) else -1) * (M.a * u.toNat + M.b * v.toNat)
  let r := amontredZ t modulus.toNat inv
  Limbs.ofNat (if r < modulus.toNat then r else r - modulus.toNat).toNat

/-- The last round's `u` is canonical and congruent to the signed row over `2^64`. -/
theorem finalU_spec (F : PastaField) (M : Mat2) (signWord : ℕ) (u v : Limbs)
    (hu : u.Bounded) (hv : v.Bounded) (hM : |M.a| + |M.b| ≤ 2^59) :
    (finalU M signWord u v F.modulus F.inv).Bounded ∧
      (finalU M signWord u v F.modulus F.inv).toNat < F.modulus.toNat ∧
      ((finalU M signWord u v F.modulus F.inv).toNat : ℤ) * 2^64
        ≡ (if signWord < 2^63 then (1 : ℤ) else -1) * (M.a * u.toNat + M.b * v.toNat)
          [ZMOD F.modulus.toNat] := by
  have hu' : |(u.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (by positivity)]; exact_mod_cast Limbs.toNat_lt u hu
  have hv' : |(v.toNat : ℤ)| < 2^256 := by
    rw [abs_of_nonneg (by positivity)]; exact_mod_cast Limbs.toNat_lt v hv
  set e : ℤ := if signWord < 2^63 then (1 : ℤ) else -1 with he
  have he1 : |e| = 1 := by rw [he]; split_ifs <;> simp
  have ht : |e * (M.a * u.toNat + M.b * v.toNat)| < 2^315 := by
    rw [abs_mul, he1, one_mul]
    have := row_abs_lt M.a M.b _ _ _ _ hM hu' hv' (by positivity)
    norm_num at this ⊢; exact this
  obtain ⟨h0, -, h2, h3, h4⟩ := amontredZ_spec F _ ht
  have hp1 : 2^254 ≤ F.modulus.toNat := F.two_pow_le_modulus
  set p : ℤ := (F.modulus.toNat : ℤ) with hp
  set r : ℤ := amontredZ (e * (M.a * u.toNat + M.b * v.toNat)) F.modulus.toNat F.inv with hr
  have hres : finalU M signWord u v F.modulus F.inv =
      Limbs.ofNat (if r < p then r else r - p).toNat := rfl
  rw [hres]
  have hq0 : 0 ≤ (if r < p then r else r - p) := by split_ifs <;> omega
  have hqp : (if r < p then r else r - p) < p := by split_ifs <;> omega
  have hnat : ((((if r < p then r else r - p)).toNat : ℕ) : ℤ) = if r < p then r else r - p :=
    Int.toNat_of_nonneg hq0
  have hlt : (if r < p then r else r - p).toNat < 2^256 := by
    have : (p : ℤ) < 2^255 := by rw [hp]; exact_mod_cast F.modulus_lt
    omega
  rw [Limbs.toNat_ofNat _ hlt]
  refine ⟨Limbs.ofNat_bounded _, by omega, ?_⟩
  rw [hnat]
  have hcong : (if r < p then r else r - p) ≡ r [ZMOD p] := by
    split_ifs
    · exact Int.ModEq.refl _
    · exact Int.modEq_iff_dvd.2 (by rw [sub_sub_cancel])
  exact (hcong.mul_right _).trans h4

end PastaCurves.Inversion
