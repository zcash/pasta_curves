import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Ring.Int.Parity
import Mathlib.Data.Int.ModEq
import Mathlib.Data.List.Forall2
import PastaCurves.Glv.Model

/-!
# The definitions the GLV path's theorems are stated with

What a width-4 wNAF digit is (`IsWnafDigit`), the least magnitude whose recoding needs a 129th
digit (`leastFailing`), and the odd multiples a table holds (`oddMultiples`, `TableOf`); then what
the curve's points and scalars stand for, and the conditions of the end-to-end theorems.

The curve's points stand for elements of a group `G` through an abstraction `CurveAbs`, and its
scalars stand for natural numbers through an abstraction `ScalarAbs`. Both are data, not
assumptions. The end-to-end theorems (`Entry.lean`) take them as arguments, with conditions on
them that are hypotheses of the theorems, never axioms:

- `OrderSpec`: `n` kills every valid point;
- `EndoSpec`: `endo` keeps points valid and acts as multiplication by `λ`;
- `DecomposeSpec`: on a valid scalar, `decompose` returns halves below `2^127` that combine to the
  scalar modulo `n`;
- `CurveOps.Spec`: the curve primitives keep points valid and are the group operations.

The first three are hypotheses because the curve's order, its endomorphism and the
decomposition are not formalized here; `CurveOps.Spec`, the contract of the curve arithmetic, is
trusted. The model leaves
allocation out and so assumes it succeeds (`Table.batch`).
-/

namespace PastaCurves.Glv

open Aeneas Aeneas.Std

/-- A width-4 wNAF digit: zero, or odd and between `-7` and `7`. -/
def IsWnafDigit (d : ℤ) : Prop := d = 0 ∨ (Odd d ∧ -7 ≤ d ∧ d ≤ 7)

/-- `0x8888…8889` (32 hex digits), the least magnitude whose wNAF needs a 129th digit
(`wnaf_digits_below_least_failing` and `wnaf_digits_least_failing`, `Wnaf.lean`). -/
def leastFailing : ℕ := 0x88888888888888888888888888888889

/-! ## Odd multiples -/

/-- The odd multiples `{1, 3, 5, 7}·p` of a group element. -/
def oddMultiples {G : Type} [AddCommGroup G] (p : G) : Fin 4 → G :=
  fun j => (2 * (j : ℕ) + 1) • p

/-! ## The abstraction and the conditions -/

/-- What the curve's points stand for: validity predicates on projective and affine points, and
maps `α`, `αA` into a group `G`. For Pasta, the intended instance, not defined here: valid means
on the curve, and `α` takes the Jacobian `(X : Y : Z)` to the affine point `(X/Z², Y/Z³)`, and
`Z = 0` to the identity. -/
structure CurveAbs (P A G : Type) where
  /-- A projective point that the arithmetic accepts (on the curve, Rust's `C`). -/
  valid : P → Prop
  /-- An affine point that the arithmetic accepts (`C::AffineExt`). -/
  validA : A → Prop
  /-- The group element a projective point stands for; Rust's `==` is equality under `α`. -/
  α : P → G
  /-- The group element an affine point stands for. -/
  αA : A → G

/-- `n` kills every valid point, as the order of a group of order `n` does. Primality is not
needed. -/
def OrderSpec {P A G : Type} [AddCommGroup G] (V : CurveAbs P A G) (n : ℕ) : Prop :=
  ∀ p, V.valid p → n • V.α p = 0

/-- Rust's `endo` itself (`(ζX, Y, Z)` with `ζ = Base::ZETA`) keeps points valid and acts as
multiplication by `lam`. -/
def EndoSpec {P A G : Type} [AddCommGroup G] (O : CurveOps P A) (V : CurveAbs P A G)
    (lam : ℕ) : Prop :=
  ∀ p, V.valid p → V.valid (O.endo p) ∧ V.α (O.endo p) = lam • V.α p

/-- What the scalars stand for: which values of the scalar type are scalars the crate can hold, and
the number each stands for. For Pasta, the intended instance, not defined here: valid means
canonical Montgomery limbs, as in every scalar the crate builds, and `val` is the canonical value
in `[0, n)`. -/
structure ScalarAbs (Sc : Type) where
  /-- A value of the scalar type that the crate can hold. -/
  valid : Sc → Prop
  /-- The number a scalar stands for. -/
  val : Sc → ℕ

/-- A half `(negative, magnitude)` as an integer. -/
def decode (h : Bool × Std.U128) : ℤ := if h.1 then -(h.2.val : ℤ) else h.2.val

/-- `decompose`'s contract, per curve: on a valid scalar `k`, `decompose(k)` succeeds, with both
magnitudes below `2^127` and `k1 + k2·λ ≡ k (mod n)`. -/
def DecomposeSpec {Sc : Type} (n lam : ℕ) (S : ScalarAbs Sc) (f : DecomposeFn Sc) : Prop :=
  ∀ k, S.valid k → ∃ r, f k = .ok r ∧ r.1.2.val < 2 ^ 127 ∧ r.2.2.val < 2 ^ 127 ∧
    decode r.1 + decode r.2 * lam ≡ S.val k [ZMOD n]

/-- The contract of the curve primitives, which is trusted. The model takes them as total
functions (`CurveOps`), so that each returns rather than panics is assumed there; this adds that,
under the abstraction `V`, each keeps validity and is the group operation. It is stated for all
valid operands, so it covers each case the path produces: identity operands, equal operands
(`acc += a` where `acc` and `a` stand for the same point), negated operands, and
`batch_normalize` on points with `Z = 0`. `batch_normalize`'s `assert_eq!` on its two lengths is
not modelled: both callers pass equal lengths (8, and `8 n`), and nothing is assumed of other
lengths. `AffineExt::identity()` only fills the buffer that `batch_normalize` overwrites, so
nothing is assumed of it either. -/
structure CurveOps.Spec {P A G : Type} [AddCommGroup G] (O : CurveOps P A)
    (V : CurveAbs P A G) : Prop where
  /-- `C::identity()`. -/
  identity : V.valid O.identity ∧ V.α O.identity = 0
  /-- `is_identity`: a point it accepts stands for the identity. -/
  isIdentity : ∀ p, V.valid p → O.isIdentity p = true → V.α p = 0
  /-- `double`. -/
  double : ∀ p, V.valid p → V.valid (O.double p) ∧ V.α (O.double p) = 2 • V.α p
  /-- `C + C`. -/
  add : ∀ p q, V.valid p → V.valid q →
    V.valid (O.add p q) ∧ V.α (O.add p q) = V.α p + V.α q
  /-- `C += AffineExt`. -/
  addMixed : ∀ p a, V.valid p → V.validA a →
    V.valid (O.addMixed p a) ∧ V.α (O.addMixed p a) = V.α p + V.αA a
  /-- `-AffineExt`. -/
  negA : ∀ a, V.validA a → V.validA (O.negA a) ∧ V.αA (O.negA a) = -V.αA a
  /-- `batch_normalize(&ps, &mut qs)` on slices of equal length: entry by entry, the normalized
  point. Its `invert().unwrap()` inverts a product of nonzero `Z`s. -/
  batchNormalize : ∀ (ps : List P) (qs : List A), (∀ p ∈ ps, V.valid p) →
    ps.length = qs.length →
    List.Forall₂ (fun p a => V.validA a ∧ V.αA a = V.α p) ps (O.batchNormalize ps qs)
  /-- `CurveAffine::to_curve`. -/
  toCurve : ∀ a, V.validA a → V.valid (O.toCurve a) ∧ V.α (O.toCurve a) = V.αA a

/-! ## The table of a point -/

/-- `t` is the table of the point `p`: its entries are valid and stand for `(2j + 1)·p` (`t1`) and
`(2j + 1)·(λ·p)` (`t2`). -/
def TableOf {P A G : Type} [AddCommGroup G] (V : CurveAbs P A G) (lam : ℕ) (p : P)
    (t : Table A) : Prop :=
  ∀ j : Fin 4, V.validA (t.t1 j) ∧ V.αA (t.t1 j) = oddMultiples (V.α p) j ∧
    V.validA (t.t2 j) ∧ V.αA (t.t2 j) = oddMultiples (lam • V.α p) j

end PastaCurves.Glv
