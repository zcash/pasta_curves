import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Convex regions by half-planes, and inclusions by Farkas certificates

The certificate of the termination bound (Theorem 5 of `book/src/design/inversion.md`) describes
convex regions of the rational plane as finite conjunctions of half-planes `a x + b y ≤ c`. An
inclusion `M S ⊆ λ^k T`, that is `∀ p ∈ S, M p / λ^k ∈ T`, is certified edge by edge: every
half-plane of `T`, pulled back through the linear map, is a nonnegative combination of two
half-planes of `S`. That is the form of Bernstein's "hull light" certificate, whose Sage script
generates the HOL Light proof `Divstep/hull_light.ml` of `jrh13/hol-light`
(<https://github.com/jrh13/hol-light/tree/6f6ac17e8f4bff6c7f6897de0570a004744e4963/Divstep>).

Pulling a half-plane back through a map means taking its preimage: the points whose image satisfies
it. Through the linear map `(x, y) ↦ M (x, y) / d`, the preimage of `a x + b y ≤ c` is again a
half-plane, `a' x + b' y ≤ c` with `(a', b') = (a, b) M / d` (`pullback`, `pullback_holds`). The
image of `S` lies in `T` exactly when `S` lies in the preimage of every half-plane of `T`. So an
inclusion is a list of implications, one for each half-plane of `T`: every point that satisfies the
inequalities of `S` satisfies the pulled-back inequality.

Farkas' lemma says when such an implication holds. Its affine form (Alexander Schrijver, "Theory of
Linear and Integer Programming", Wiley, 1986, Corollary 7.1h) is: if a finite system of linear
inequalities has a solution, and every solution satisfies `c · v ≤ δ`, then for some `δ' ≤ δ` the
inequality `c · v ≤ δ'` is a nonnegative combination of the inequalities of the system. The lemma
goes back to Gyula Farkas, "Theorie der einfachen Ungleichungen", Journal für die reine und
angewandte Mathematik 124 (1902), 1–27; <https://gdz.sub.uni-goettingen.de/id/PPN243919689_0124>. A
Farkas record writes the combination down. Its multipliers `m` and `n` weight two half-planes of
`S`. The weighted sum equals `q` times the pulled-back half-plane in the coefficients of `x` and
`y`, for a scale `q > 0`, and falls short of it in the constant by a slack `p ≥ 0`, which accounts
for `δ' ≤ δ`. Two multipliers suffice here because a linear functional on a polygon is maximized at
a vertex, where two edges are active. Only the elementary direction is used: a nonnegative
combination of valid inequalities is valid (`farkas_sound`). The existence of the multipliers is
what makes the certificate searchable, and is not needed for soundness; no convexity theory is
needed either.

Everything here is over `ℚ`, where the checks are decidable and the points of the argument
(integers divided by rational scales) live.
-/

namespace PastaCurves.Inversion.Hull

/-- The half-plane `a x + b y ≤ c`. -/
structure HalfPlane where
  a : ℚ
  b : ℚ
  c : ℚ
  deriving DecidableEq, Repr

/-- A convex region: the points satisfying every half-plane of the list. -/
abbrev Region := List HalfPlane

/-- The point `(x, y)` satisfies the half-plane. -/
def HalfPlane.holds (h : HalfPlane) (x y : ℚ) : Prop := h.a * x + h.b * y ≤ h.c

/-- Decidable, so that the kernel can evaluate the checks. -/
instance (h : HalfPlane) (x y : ℚ) : Decidable (h.holds x y) := by
  unfold HalfPlane.holds; infer_instance

/-- The point `(x, y)` lies in the region: every half-plane holds. -/
def Region.mem (R : Region) (x y : ℚ) : Prop := ∀ h ∈ R, h.holds x y

/-- Decidable, so that the kernel can evaluate the checks. -/
instance (R : Region) (x y : ℚ) : Decidable (R.mem x y) := by
  unfold Region.mem; infer_instance

/-- A `2×2` rational matrix `[[m11, m12], [m21, m22]]` acting on column vectors. -/
structure Mat where
  m11 : ℚ
  m12 : ℚ
  m21 : ℚ
  m22 : ℚ
  deriving DecidableEq, Repr

/-- The first coordinate of `M (x, y)`. -/
def Mat.apX (M : Mat) (x y : ℚ) : ℚ := M.m11 * x + M.m12 * y

/-- The second coordinate of `M (x, y)`. -/
def Mat.apY (M : Mat) (x y : ℚ) : ℚ := M.m21 * x + M.m22 * y

/-- One Farkas record: the target half-plane follows from source half-planes `i` and `j` with
multipliers `m, n > 0`, slack `p ≥ 0`, and scale `q > 0`. -/
structure Farkas where
  i : ℕ
  j : ℕ
  m : ℚ
  n : ℚ
  p : ℚ
  q : ℚ
  deriving DecidableEq, Repr

/-- The pullback of the target half-plane `t` through `(x, y) ↦ M (x, y) / d`: its preimage, the
half-plane in `(x, y)` that says `t` holds at the image. -/
def pullback (t : HalfPlane) (M : Mat) (d : ℚ) : HalfPlane :=
  ⟨(t.a * M.m11 + t.b * M.m21) / d, (t.a * M.m12 + t.b * M.m22) / d, t.c⟩

/-- The pullback holds at `(x, y)` exactly when the target holds at the image. -/
theorem pullback_holds (t : HalfPlane) (M : Mat) (d : ℚ) (x y : ℚ) :
    (pullback t M d).holds x y ↔ t.holds (M.apX x y / d) (M.apY x y / d) := by
  unfold pullback HalfPlane.holds Mat.apX Mat.apY
  simp only
  have key : (t.a * M.m11 + t.b * M.m21) / d * x + (t.a * M.m12 + t.b * M.m22) / d * y
      = t.a * ((M.m11 * x + M.m12 * y) / d) + t.b * ((M.m21 * x + M.m22 * y) / d) := by ring
  rw [key]

/-- The one inequality behind every check: a nonnegative combination of two valid half-planes, with
a positive scale and nonnegative slack, is valid. -/
theorem farkas_sound (e₁ e₂ t : HalfPlane) (m n p q : ℚ) (hm : 0 ≤ m) (hn : 0 ≤ n) (hp : 0 ≤ p)
    (hq : 0 < q) (ha : m * e₁.a + n * e₂.a = q * t.a) (hb : m * e₁.b + n * e₂.b = q * t.b)
    (hc : m * e₁.c + n * e₂.c + p = q * t.c) (x y : ℚ) (h₁ : e₁.holds x y) (h₂ : e₂.holds x y) :
    t.holds x y := by
  unfold HalfPlane.holds at *
  have h : q * (t.a * x + t.b * y) ≤ q * t.c := by
    calc q * (t.a * x + t.b * y) = m * (e₁.a * x + e₁.b * y) + n * (e₂.a * x + e₂.b * y) := by
          linear_combination (-x) * ha - y * hb
      _ ≤ m * e₁.c + n * e₂.c := add_le_add (mul_le_mul_of_nonneg_left h₁ hm)
          (mul_le_mul_of_nonneg_left h₂ hn)
      _ ≤ q * t.c := by linarith
  exact le_of_mul_le_mul_left h hq

/-- The check of one Farkas record against a source region and a pulled-back target
half-plane. -/
def checkFarkas (S : Region) (t : HalfPlane) (f : Farkas) : Bool :=
  match S[f.i]?, S[f.j]? with
  | some e₁, some e₂ =>
    0 < f.m && 0 < f.n && 0 ≤ f.p && 0 < f.q &&
      f.m * e₁.a + f.n * e₂.a == f.q * t.a && f.m * e₁.b + f.n * e₂.b == f.q * t.b &&
      f.m * e₁.c + f.n * e₂.c + f.p == f.q * t.c
  | _, _ => false

/-- A record that passes the check proves its target half-plane on the source region, by
`farkas_sound`. -/
theorem checkFarkas_sound (S : Region) (t : HalfPlane) (f : Farkas)
    (h : checkFarkas S t f = true) (x y : ℚ) (hS : S.mem x y) : t.holds x y := by
  unfold checkFarkas at h
  split at h
  · rename_i e₁ e₂ h₁ h₂
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
    obtain ⟨⟨⟨⟨⟨⟨hm, hn⟩, hp⟩, hq⟩, ha⟩, hb⟩, hc⟩ := h
    exact farkas_sound e₁ e₂ t f.m f.n f.p f.q hm.le hn.le hp hq ha hb hc x y
      (hS e₁ (List.mem_of_getElem? h₁)) (hS e₂ (List.mem_of_getElem? h₂))
  · exact absurd h Bool.false_ne_true

/-- An inclusion certificate: `M S / d ⊆ T`, one Farkas record per half-plane of `T`. -/
def checkInclusion (S T : Region) (M : Mat) (d : ℚ) (fs : List Farkas) : Bool :=
  T.length == fs.length &&
    (List.zip T fs).all fun ⟨t, f⟩ => checkFarkas S (pullback t M d) f

/-- A certificate that passes the check proves the inclusion: every point of `S`, mapped and
scaled, lies in `T`. -/
theorem checkInclusion_sound (S T : Region) (M : Mat) (d : ℚ) (fs : List Farkas)
    (h : checkInclusion S T M d fs = true) (x y : ℚ) (hS : S.mem x y) :
    T.mem (M.apX x y / d) (M.apY x y / d) := by
  unfold checkInclusion at h
  simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true] at h
  obtain ⟨hlen, hall⟩ := h
  intro t ht
  obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem ht
  have hk' : k < fs.length := hlen ▸ hk
  have hmem : (T[k], fs[k]) ∈ List.zip T fs := by
    rw [← List.getElem_zip (h := by simp [hk, hk'])]
    exact List.getElem_mem _
  have := hall _ hmem
  rw [← pullback_holds]
  exact checkFarkas_sound S _ _ this x y hS

end PastaCurves.Inversion.Hull
