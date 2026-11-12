import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# The Eisenstein integers, as `glv_eisenstein` computes with them

`src/glv_eisenstein.rs` reads a GLV split `(k1, k2)` as the Eisenstein integer `k1 + k2 ω`, an
element of `ℤ[ω] = ℤ[X]/(X² + X + 1)`, and holds such an element as its coefficient pair `(a, b)`.
This module models that pair, generically over the coefficient ring, so that the same
multiplication serves `ℤ`, and `ℤ/2` and `ℤ/8` for the residue classes the recoder works with.

## Main definitions

- `E R`: the pair `a + b ω` over a commutative ring `R`, with `ω² = -1 - ω`.
- `E.norm`: the norm `a² - ab + b²`.
- `E.map`: reduction of the coefficients along a ring homomorphism.

## Main statements

- `E.norm_mul`: the norm is multiplicative.
- `E.isUnit_iff`: the units of `ℤ[ω]` are exactly the six elements `±1, ±ω, ±ω²`, the group `μ₆`
  whose action `glv_eisenstein` reduces its table by.
- `E.two_dvd_mul`: 2 is prime in `ℤ[ω]` (it is inert), so "even" means both coordinates even,
  as the recoder's parity test assumes.
-/

namespace PastaCurves.Eisenstein

/-- The Eisenstein integer `a + b ω` over `R`, as the Rust code's coefficient pair `(a, b)`. -/
@[ext]
structure E (R : Type) where
  /-- The coefficient of `1`. -/
  a : R
  /-- The coefficient of `ω`. -/
  b : R
  deriving DecidableEq, Repr

namespace E

variable {R : Type} [CommRing R]

/-- `0`. -/
instance : Zero (E R) := ⟨⟨0, 0⟩⟩

/-- `1`. -/
instance : One (E R) := ⟨⟨1, 0⟩⟩

/-- Coefficient-wise addition. -/
instance : Add (E R) := ⟨fun x y => ⟨x.a + y.a, x.b + y.b⟩⟩

/-- Coefficient-wise negation. -/
instance : Neg (E R) := ⟨fun x => ⟨-x.a, -x.b⟩⟩

/-- Coefficient-wise subtraction. -/
instance : Sub (E R) := ⟨fun x y => ⟨x.a - y.a, x.b - y.b⟩⟩

/-- `(a + b ω)(c + d ω) = (ac - bd) + (ad + bc - bd) ω`, using `ω² = -1 - ω`. -/
instance : Mul (E R) := ⟨fun x y => ⟨x.a * y.a - x.b * y.b, x.a * y.b + x.b * y.a - x.b * y.b⟩⟩

@[simp] theorem zero_a : (0 : E R).a = 0 := rfl
@[simp] theorem zero_b : (0 : E R).b = 0 := rfl
@[simp] theorem one_a : (1 : E R).a = 1 := rfl
@[simp] theorem one_b : (1 : E R).b = 0 := rfl
@[simp] theorem add_a (x y : E R) : (x + y).a = x.a + y.a := rfl
@[simp] theorem add_b (x y : E R) : (x + y).b = x.b + y.b := rfl
@[simp] theorem neg_a (x : E R) : (-x).a = -x.a := rfl
@[simp] theorem neg_b (x : E R) : (-x).b = -x.b := rfl
@[simp] theorem sub_a (x y : E R) : (x - y).a = x.a - y.a := rfl
@[simp] theorem sub_b (x y : E R) : (x - y).b = x.b - y.b := rfl
@[simp] theorem mul_a (x y : E R) : (x * y).a = x.a * y.a - x.b * y.b := rfl
@[simp] theorem mul_b (x y : E R) : (x * y).b = x.a * y.b + x.b * y.a - x.b * y.b := rfl

/-- `ω`, the coefficient pair `(0, 1)`. -/
def ω : E R := ⟨0, 1⟩

/-- Multiplication by `ω`, `(a + b ω) ω = -b + (a - b) ω`: the Rust `rotate`, and on points the
endomorphism `φ`. -/
def rotate (x : E R) : E R := ⟨-x.b, x.a - x.b⟩

/-- `rotate` is multiplication by `ω`, as its Rust doc comment derives. -/
theorem rotate_eq_mul (x : E R) : rotate x = x * ω := by
  ext <;> simp [rotate, ω]

/-- `ω` is a root of `X² + X + 1`, which is what makes `E R` the ring `R[X]/(X² + X + 1)`. -/
theorem ω_sq_add : ω * ω + ω + (1 : E R) = 0 := by
  ext <;> simp [ω]

/-- Reduction of the coefficients along a ring homomorphism, the way the recoder reduces a
residual mod 8 coordinate-wise. -/
def map {S : Type} [CommRing S] (f : R →+* S) (x : E R) : E S := ⟨f x.a, f x.b⟩

/-- Reduction commutes with multiplication: the class of a product depends only on the classes of
its factors. -/
theorem map_mul {S : Type} [CommRing S] (f : R →+* S) (x y : E R) :
    (x * y).map f = x.map f * y.map f := by
  ext <;> simp [map]

/-- The norm `N(a + b ω) = a² - ab + b²`, the product of `a + b ω` with its conjugate. -/
def norm (x : E R) : R := x.a * x.a - x.a * x.b + x.b * x.b

/-- The norm is multiplicative. Used for the units: a unit's norm divides 1. -/
theorem norm_mul (x y : E R) : (x * y).norm = x.norm * y.norm := by
  simp only [norm, mul_a, mul_b]; ring

end E

/-! ## Over the integers -/

namespace E

/-- Four times the norm is a sum of squares, `(2a - b)² + 3b²`, so the norm of an integer pair is
nonnegative and vanishes only at zero. -/
theorem four_norm (x : E ℤ) : 4 * x.norm = (2 * x.a - x.b) ^ 2 + 3 * x.b ^ 2 := by
  simp only [norm]; ring

theorem norm_nonneg (x : E ℤ) : 0 ≤ x.norm := by
  have h := four_norm x
  nlinarith [sq_nonneg (2 * x.a - x.b), sq_nonneg x.b]

/-- Only `0` has norm zero. Used to turn "the norm is a multiple of `n` below `n`" into
"the element is zero". -/
theorem norm_eq_zero_iff (x : E ℤ) : x.norm = 0 ↔ x = 0 := by
  constructor
  · intro h
    have h4 := four_norm x
    rw [h] at h4
    have hb : x.b = 0 := by nlinarith [sq_nonneg (2 * x.a - x.b), sq_nonneg x.b]
    have ha : x.a = 0 := by
      rw [hb] at h4; nlinarith [sq_nonneg x.a]
    ext <;> simp [ha, hb]
  · rintro rfl; simp [norm]

/-- A nonzero element has positive norm. -/
theorem norm_pos {x : E ℤ} (hx : x ≠ 0) : 0 < x.norm :=
  lt_of_le_of_ne (norm_nonneg x) (fun h => hx ((norm_eq_zero_iff x).mp h.symm))

/-- The six units `1, ω, ω², -1, -ω, -ω²`, as coefficient pairs, in the order `glv_eisenstein`'s
look-up table produces them: three rotations, each also negated. -/
def units : List (E ℤ) := [⟨1, 0⟩, ⟨0, 1⟩, ⟨-1, -1⟩, ⟨-1, 0⟩, ⟨0, -1⟩, ⟨1, 1⟩]

/-- The elements of norm one are exactly the six units. -/
theorem norm_eq_one_iff (x : E ℤ) : x.norm = 1 ↔ x ∈ units := by
  constructor
  · intro h
    have h4 := four_norm x
    rw [h] at h4
    have hb1 : x.b ≤ 1 := by nlinarith [sq_nonneg (2 * x.a - x.b), sq_nonneg (x.b - 1)]
    have hb0 : -1 ≤ x.b := by nlinarith [sq_nonneg (2 * x.a - x.b), sq_nonneg (x.b + 1)]
    have ha1 : x.a ≤ 1 := by nlinarith [sq_nonneg (2 * x.a - x.b - 2), sq_nonneg x.b]
    have ha0 : -1 ≤ x.a := by nlinarith [sq_nonneg (2 * x.a - x.b + 2), sq_nonneg x.b]
    obtain ⟨a, b⟩ := x
    simp only at *
    simp only [norm] at h
    interval_cases a <;> interval_cases b <;> simp_all [units]
  · intro h
    simp only [units, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl | rfl | rfl <;> decide

/-- The units of `ℤ[ω]` are exactly the six of `μ₆`. This is the group whose free action on the
odd residue classes cuts the table to eight points (see `Classes.lean`). -/
theorem isUnit_iff (x : E ℤ) : (∃ y : E ℤ, x * y = 1) ↔ x ∈ units := by
  constructor
  · rintro ⟨y, hy⟩
    have hn : x.norm * y.norm = 1 := by rw [← norm_mul, hy]; decide
    have hx := norm_nonneg x
    have hy' := norm_nonneg y
    have : x.norm = 1 := by
      rcases Int.eq_one_or_neg_one_of_mul_eq_one hn with h | h
      · exact h
      · omega
    exact (norm_eq_one_iff x).mp this
  · intro h
    simp only [units, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl | rfl | rfl
    · exact ⟨⟨1, 0⟩, by decide⟩
    · exact ⟨⟨-1, -1⟩, by decide⟩
    · exact ⟨⟨0, 1⟩, by decide⟩
    · exact ⟨⟨-1, 0⟩, by decide⟩
    · exact ⟨⟨1, 1⟩, by decide⟩
    · exact ⟨⟨0, -1⟩, by decide⟩

/-- `2 ∣ x` in `ℤ[ω]`: both coordinates even. -/
def TwoDvd (x : E ℤ) : Prop := 2 ∣ x.a ∧ 2 ∣ x.b

/-- Over `ℤ/2`, a product vanishes only if a factor does: `ℤ[ω]/2 = 𝔽₂[X]/(X² + X + 1)` is the
field `𝔽₄`, since `X² + X + 1` has no root mod 2. Closed by evaluation. -/
theorem mod_two_no_zero_divisors :
    ∀ x y : E (ZMod 2), x * y = 0 → x = 0 ∨ y = 0 := by
  rintro ⟨a, b⟩ ⟨c, d⟩
  revert a b c d
  decide

private theorem twoDvd_iff_map (x : E ℤ) :
    TwoDvd x ↔ x.map (Int.castRingHom (ZMod 2)) = 0 := by
  simp only [TwoDvd, map, E.ext_iff, zero_a, zero_b, eq_intCast,
    ZMod.intCast_zmod_eq_zero_iff_dvd, Nat.cast_ofNat]

/-- 2 is prime in `ℤ[ω]`, which is to say inert: if `2` divides a product it divides a factor.
So "even" in the ring is "both coordinates even", which is the recoder's parity test, and
halving an even residual is exact and coordinate-wise. -/
theorem two_dvd_mul {x y : E ℤ} (h : TwoDvd (x * y)) : TwoDvd x ∨ TwoDvd y := by
  rw [twoDvd_iff_map, map_mul] at h
  rw [twoDvd_iff_map, twoDvd_iff_map]
  exact mod_two_no_zero_divisors _ _ h

end E

end PastaCurves.Eisenstein
