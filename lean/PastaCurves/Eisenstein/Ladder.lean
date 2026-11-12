import PastaCurves.Eisenstein.Classes
import PastaCurves.Fields

/-!
# The batch-affine ladder has no exceptional case but the identity

`Table::batch_mul_affine` in `src/glv_eisenstein.rs` runs the Eisenstein ladder with affine
formulas. With `A` the accumulator and `D` the column's digit point, a column divides by zero in
three situations: a doubling's `2y` when `A = O`; a fused `2A + D`'s first slope when `A = ±D`; and
its second when `D = -2A`. On multiples of the base point `P ≠ O`, a point `x P` is `O` exactly when
the Eisenstein integer `x` lies in the kernel of `a + b ω ↦ a + b λ (mod n)`, `n` the group order
and `λ` the scalar-field `ZETA` (`InKernel`). The accumulator at a fused column `c` holds the
recoder's residual `r = r_(c+1)` and leaves holding `r_c = 2 r + d`, so the three cases are that
`r`, `r ∓ d`, or `2 r + d` lies in the kernel.

None does, for either Pasta curve, by a norm bound rather than by parity, since `2` is a unit mod
`n`:

1. An element of the kernel has norm divisible by `n` (`norm_dvd_of_inKernel`), using only
   `λ² + λ + 1 ≡ 0 (mod n)`. A nonzero element of the kernel therefore has norm at least `n`.
2. The recoder keeps every residual after the first column inside the box
   `|a| ≤ R₁, |b| ≤ R₂`, `Rᵢ = ⌊(Bᵢ + 5) / 2⌋`, from a decomposition inside the Babai bounds `Bᵢ`
   (`residual_le_of_babai`, `residual_le_of_residual`); a digit adds at most 5 to each coordinate.
3. Every element of the box `|a| ≤ R₁ + 5, |b| ≤ R₂ + 5` has norm below `n`, for each curve's
   constants (`pallas_box`, `vesta_box`).
4. The three quantities are nonzero: `r` is a residual before the last column; `r` is even while
   every digit is odd, so `r ∓ d` and `2 r + d` are odd.

The Babai bound on `decompose`'s halves, `|k₁| ≤ B₁` and `|k₂| ≤ B₂`, is a hypothesis here, as in
`sage/glv_eisenstein.sage` step `[6b]`, which checks the same argument numerically: it is the
rounding bound `decompose` documents in `src/glv.rs`, not proved from its limb arithmetic.
-/

namespace PastaCurves.Eisenstein

open E

/-- `x` lies in the kernel of `a + b ω ↦ a + b λ (mod n)`: on multiples of a point of order `n`,
`x P = O`. -/
def InKernel (n lam : ℤ) (x : E ℤ) : Prop := n ∣ x.a + x.b * lam

/-! ## The generic argument -/

/-- An element of the kernel has norm divisible by `n`: with `a ≡ -b λ`, the norm
`a² - ab + b²` is `b² (λ² + λ + 1) ≡ 0`. -/
theorem norm_dvd_of_inKernel {n lam : ℤ} (hlam : n ∣ lam ^ 2 + lam + 1) {x : E ℤ}
    (hx : InKernel n lam x) : n ∣ x.norm := by
  obtain ⟨k, hk⟩ := hx
  have ha : x.a = n * k - x.b * lam := by linarith
  have : x.norm = n * (n * k ^ 2 - 2 * k * x.b * lam - k * x.b) + x.b ^ 2 * (lam ^ 2 + lam + 1) := by
    simp only [norm]; rw [ha]; ring
  rw [this]
  exact dvd_add (dvd_mul_right n _) (dvd_mul_of_dvd_right hlam _)

/-- On the box `|a| ≤ A, |b| ≤ B`, the norm is at most `A² + AB + B²`. -/
theorem norm_le_box {x : E ℤ} {A B : ℤ} (ha : |x.a| ≤ A) (hb : |x.b| ≤ B) :
    x.norm ≤ A ^ 2 + A * B + B ^ 2 := by
  obtain ⟨ha1, ha2⟩ := abs_le.mp ha
  obtain ⟨hb1, hb2⟩ := abs_le.mp hb
  simp only [norm]
  nlinarith [mul_nonneg (by linarith : (0:ℤ) ≤ A - x.a) (by linarith : (0:ℤ) ≤ A + x.a),
    mul_nonneg (by linarith : (0:ℤ) ≤ B - x.b) (by linarith : (0:ℤ) ≤ B + x.b),
    mul_nonneg (by linarith : (0:ℤ) ≤ A - x.a) (by linarith : (0:ℤ) ≤ B + x.b),
    mul_nonneg (by linarith : (0:ℤ) ≤ A + x.a) (by linarith : (0:ℤ) ≤ B - x.b)]

/-- A nonzero element of the box whose norms all lie below `n` is not in the kernel. -/
theorem not_inKernel {n lam A B : ℤ} (hlam : n ∣ lam ^ 2 + lam + 1)
    (hbox : A ^ 2 + A * B + B ^ 2 < n) {x : E ℤ} (hx : x ≠ 0) (ha : |x.a| ≤ A) (hb : |x.b| ≤ B) :
    ¬ InKernel n lam x := by
  intro hk
  have hpos := norm_pos hx
  have hle := Int.le_of_dvd hpos (norm_dvd_of_inKernel hlam hk)
  have := norm_le_box ha hb
  omega

/-- One recoder column on one coordinate: `x = 2 x' + d` with `|d| ≤ 5` and `|x| ≤ X` give
`|x'| ≤ ⌊(X + 5) / 2⌋`. -/
theorem halve_le {x x' d X : ℤ} (h : x = 2 * x' + d) (hd : |d| ≤ 5) (hx : |x| ≤ X) :
    |x'| ≤ (X + 5) / 2 := by
  rw [abs_le] at hd hx ⊢
  omega

/-- Every digit is odd: not both coordinates even. -/
theorem digit_odd : ∀ d ∈ digits, ¬ TwoDvd d := by
  unfold TwoDvd; decide

/-! ## The recoder's residuals -/

/-- The recoder's column: `r = 2 r' + d`, `d` zero (a halving) or a digit. -/
def Column (r r' d : E ℤ) : Prop := r = r' + r' + d ∧ (d = 0 ∨ d ∈ digits)

private theorem column_coords {r r' d : E ℤ} (h : Column r r' d) :
    r.a = 2 * r'.a + d.a ∧ r.b = 2 * r'.b + d.b ∧ |d.a| ≤ 5 ∧ |d.b| ≤ 5 := by
  obtain ⟨rfl, hd⟩ := h
  refine ⟨by simp; ring, by simp; ring, ?_⟩
  rcases hd with rfl | hd
  · simp
  · exact digit_bound d hd

/-- The first column takes a decomposition inside the Babai bounds into the box of half-width
`Rᵢ = ⌊(Bᵢ + 5) / 2⌋`. -/
theorem residual_le_of_babai {r r' d : E ℤ} {B₁ B₂ : ℤ} (h : Column r r' d)
    (ha : |r.a| ≤ B₁) (hb : |r.b| ≤ B₂) :
    |r'.a| ≤ (B₁ + 5) / 2 ∧ |r'.b| ≤ (B₂ + 5) / 2 := by
  obtain ⟨h1, h2, h3, h4⟩ := column_coords h
  exact ⟨halve_le h1 h3 ha, halve_le h2 h4 hb⟩

/-- Every later column keeps a residual inside that box, which is absorbing once `Rᵢ ≥ 5`. -/
theorem residual_le_of_residual {r r' d : E ℤ} {R₁ R₂ : ℤ} (h : Column r r' d)
    (h₁ : 5 ≤ R₁) (h₂ : 5 ≤ R₂) (ha : |r.a| ≤ R₁) (hb : |r.b| ≤ R₂) :
    |r'.a| ≤ R₁ ∧ |r'.b| ≤ R₂ := by
  obtain ⟨h1, h2, h3, h4⟩ := column_coords h
  have := halve_le h1 h3 ha
  have := halve_le h2 h4 hb
  constructor <;> omega

/-! ## A ladder column -/

/-- The constants of one curve that the argument needs: its group order `n`, its `λ`, and the
Babai bounds `B₁`, `B₂` of its `decompose`. -/
structure LadderParams where
  /-- The group order. -/
  n : ℤ
  /-- The eigenvalue of `φ`, the scalar-field `ZETA`. -/
  lam : ℤ
  /-- The bound on `|k₁|`. -/
  B₁ : ℤ
  /-- The bound on `|k₂|`. -/
  B₂ : ℤ

namespace LadderParams

variable (P : LadderParams)

/-- The residual box after the first column, `⌊(B₁ + 5) / 2⌋`. -/
def R₁ : ℤ := (P.B₁ + 5) / 2

/-- The residual box after the first column, `⌊(B₂ + 5) / 2⌋`. -/
def R₂ : ℤ := (P.B₂ + 5) / 2

/-- The facts about the constants the argument consumes, each a closed check on the curve's own
numbers. -/
structure Valid : Prop where
  /-- `λ` is a root of `X² + X + 1` mod `n`. -/
  lam : P.n ∣ P.lam ^ 2 + P.lam + 1
  /-- The residual box is absorbing. -/
  R₁_ge : 5 ≤ P.R₁
  /-- The residual box is absorbing. -/
  R₂_ge : 5 ≤ P.R₂
  /-- Every element of the residual box widened by a digit has norm below `n`. -/
  box : (P.R₁ + 5) ^ 2 + (P.R₁ + 5) * (P.R₂ + 5) + (P.R₂ + 5) ^ 2 < P.n

variable {P}

/-- At a fused column, with the accumulator holding a residual `r` from inside the box, even
(it was halved at least twice since the previous digit) and nonzero, none of the three affine
denominators vanishes: `r` is not `O` (the doubling's `2y`), `r ∓ d` is not `O` (`A = ±D`, the
first slope), and the new residual `2 r + d`, when it is again inside the box, is not `O`
(`D = -2A`, the second slope). The last premise holds from the second column on, by
`residual_le_of_residual`; at the last column `2 r + d` is the decomposed scalar itself, which lies
in the kernel only for the scalar `0`, whose recoding is empty. -/
theorem column_safe (hP : P.Valid) {r d : E ℤ} (hr0 : r ≠ 0) (hra : |r.a| ≤ P.R₁)
    (hrb : |r.b| ≤ P.R₂) (heven : TwoDvd r) (hd : d ∈ digits) :
    ¬ InKernel P.n P.lam r ∧ ¬ InKernel P.n P.lam (r - d) ∧ ¬ InKernel P.n P.lam (r + d) ∧
      (|(r + r + d).a| ≤ P.R₁ → |(r + r + d).b| ≤ P.R₂ → ¬ InKernel P.n P.lam (r + r + d)) := by
  have hodd := digit_odd d hd
  obtain ⟨hda, hdb⟩ := digit_bound d hd
  have hlam := hP.lam
  have hbox := hP.box
  have hR₁ := hP.R₁_ge
  have hR₂ := hP.R₂_ge
  -- The box of a residual sits inside the widened box.
  have hsmall : P.R₁ ^ 2 + P.R₁ * P.R₂ + P.R₂ ^ 2 < P.n := by nlinarith
  obtain ⟨⟨k₁, hk₁⟩, ⟨k₂, hk₂⟩⟩ := heven
  refine ⟨not_inKernel hlam hsmall hr0 hra hrb, ?_, ?_, ?_⟩
  · refine not_inKernel hlam hbox ?_ ?_ ?_
    · intro h; apply hodd
      have ha : d.a = r.a := by have := congrArg E.a h; simp at this; linarith
      have hb : d.b = r.b := by have := congrArg E.b h; simp at this; linarith
      exact ⟨⟨k₁, by rw [ha, hk₁]⟩, ⟨k₂, by rw [hb, hk₂]⟩⟩
    · simp only [sub_a]; rw [abs_le] at hra hda ⊢; constructor <;> linarith
    · simp only [sub_b]; rw [abs_le] at hrb hdb ⊢; constructor <;> linarith
  · refine not_inKernel hlam hbox ?_ ?_ ?_
    · intro h; apply hodd
      have ha : d.a = -r.a := by have := congrArg E.a h; simp at this; linarith
      have hb : d.b = -r.b := by have := congrArg E.b h; simp at this; linarith
      exact ⟨⟨-k₁, by rw [ha, hk₁]; ring⟩, ⟨-k₂, by rw [hb, hk₂]; ring⟩⟩
    · simp only [add_a]; rw [abs_le] at hra hda ⊢; constructor <;> linarith
    · simp only [add_b]; rw [abs_le] at hrb hdb ⊢; constructor <;> linarith
  · intro ha hb
    refine not_inKernel hlam hsmall ?_ ha hb
    intro h; apply hodd
    have ha' : d.a = -(r.a + r.a) := by have := congrArg E.a h; simp at this; linarith
    have hb' : d.b = -(r.b + r.b) := by have := congrArg E.b h; simp at this; linarith
    exact ⟨⟨-(k₁ + k₁), by rw [ha', hk₁]; ring⟩, ⟨-(k₂ + k₂), by rw [hb', hk₂]; ring⟩⟩

end LadderParams

/-! ## The two curves -/

/-- A field element from its four little-endian limbs, as the crate's `from_raw` takes them. -/
def ofLimbs (l0 l1 l2 l3 : ℕ) : ℤ := (Limbs.toNat ⟨l0, l1, l2, l3⟩ : ℤ)

/-- Pallas: the group order is `q`, Vesta's base modulus; `λ` is `Fq::ZETA` (`src/fields/fq.rs`);
the bounds are those of `GlvParams for pallas::Point` (`src/glv.rs`), `(V1A + V2A) / 2 + 1` and
`(V1B_NEG + V2B) / 2 + 1`. -/
def pallas : LadderParams where
  n := (vestaBase.modulus.toNat : ℤ)
  lam := ofLimbs 0x2aa9d2e050aa0e4f 0x0fed467d47c033af 0x511db4d81cf70f5a 0x06819a58283e528e
  B₁ := (0x49e69d1640f049157fcae1c700000001 + 0x49e69d1640a899538cb1279300000000) / 2 + 1
  B₂ := (0x49e69d1640a899538cb1279300000000 + 0x93cd3a2c8198e2690c7c095a00000001) / 2 + 1

/-- Vesta: the group order is `p`, Pallas' base modulus; `λ` is `Fp::ZETA` (`src/fields/fp.rs`);
the bounds are those of `GlvParams for vesta::Point` (`src/glv.rs`). -/
def vesta : LadderParams where
  n := (pallasBase.modulus.toNat : ℤ)
  lam := ofLimbs 0x1dad5ebdfdfe4ab9 0x1d1f8bd237ad3149 0x2caad5dc57aab1b0 0x12ccca834acdba71
  B₁ := (0x49e69d1640f049157fcae1c700000000 + 0x49e69d1640a899538cb1279300000001) / 2 + 1
  B₂ := (0x49e69d1640a899538cb1279300000001 + 0x93cd3a2c8198e2690c7c095a00000001) / 2 + 1

/-- Pallas' constants satisfy the argument's premises. -/
theorem pallas_valid : pallas.Valid where
  lam := by decide +kernel
  R₁_ge := by decide +kernel
  R₂_ge := by decide +kernel
  box := by decide +kernel

/-- Vesta's constants satisfy the argument's premises. -/
theorem vesta_valid : vesta.Valid where
  lam := by decide +kernel
  R₁_ge := by decide +kernel
  R₂_ge := by decide +kernel
  box := by decide +kernel

end PastaCurves.Eisenstein
