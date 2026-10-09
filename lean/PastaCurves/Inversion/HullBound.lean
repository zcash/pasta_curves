import Mathlib.Algebra.Order.Field.Power
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.IntervalCases
import PastaCurves.Inversion.Hull
import PastaCurves.Inversion.Termination

/-!
# The termination bound from a hull certificate

The argument of Bernstein's "hull light" certificate (its Sage script generates the HOL Light
proof `Divstep/hull_light.ml` of `jrh13/hol-light`,
<https://github.com/jrh13/hol-light/tree/6f6ac17e8f4bff6c7f6897de0570a004744e4963/Divstep>),
stated over two abstract regions `H0` and `H1` whose properties, `Certified`, are the finite
checks that the certificate data discharges:

* the eight inclusions `M S ⊆ λ^k T` between `H0`, `H1`, and their images under the step maps;
* `H1` lies in the outer box `|x| ≤ 8193/8192`, `|y| ≤ 379/512`, and the triangle
  `0 ≤ y ≤ x ≤ 1` scaled by `2753/4096` lies in `H1`;
* `0 ∈ H1`, and the few integer points with `y ≠ 0` that the outer box admits at the two smallest
  scales are outside `H1`.

With `λ = s = 30902639/41749730`, the state after `n` steps from `(f, g)` with `0 ≤ g ≤ f ≤ M`,
scaled by `M s^n / stretch`, stays in a region `W i` indexed by `i = δ - 1/2`. While `g ≠ 0` the
integer point `(f, g)` forces the scale to stay above `L = 3047/2048` (the lattice argument), so
once `M s^n ≤ L · stretch` some `g_k`, `k ≤ n`, is zero. The bound `9437 b + 1 ≤ 4096 n` gives
`2^b s^n ≤ L · stretch` from two integer inequalities.
-/

namespace PastaCurves.Inversion.Hull

/-! ## Constants -/

/-- The shrink factor `λ'`. -/
def s : ℚ := 30902639 / 41749730

/-- The lattice scale. -/
def L : ℚ := 3047 / 2048

/-- The initial scale: the triangle `0 ≤ y ≤ x ≤ 1` scaled by `stretch` lies in `H1`. -/
def stretch : ℚ := 2753 / 4096

/-- The shrink factor is positive, for the side conditions of the scalings. -/
theorem s_pos : 0 < s := by norm_num [s]

/-- The shrink factor is below one, so that the scalings shrink. -/
theorem s_lt_one : s < 1 := by norm_num [s]

/-- The lattice scale is positive. -/
theorem L_pos : 0 < L := by norm_num [L]

/-- The initial scale is positive. -/
theorem stretch_pos : 0 < stretch := by norm_num [stretch]

/-- `2 s^2 ≥ 1`, so that dividing by it is a shrink. -/
theorem two_s_sq_ge_one : 1 ≤ 2 * s^2 := by norm_num [s]

/-- `2 s^2` is positive. -/
theorem two_s_sq_pos : 0 < 2 * s^2 := mul_pos two_pos (pow_pos s_pos 2)

/-! ## The regions and the certificate's facts -/

/-- Star-shaped from the origin: a region containing `0` contains every shrink of its points. -/
theorem Region.mem_smul (R : Region) (h0 : R.mem 0 0) (x y t : ℚ) (hR : R.mem x y) (ht0 : 0 ≤ t)
    (ht1 : t ≤ 1) : R.mem (t * x) (t * y) := by
  intro h hh
  have hc : 0 ≤ h.c := by
    have := h0 h hh; unfold HalfPlane.holds at this; linarith
  have := hR h hh
  unfold HalfPlane.holds at *
  calc h.a * (t * x) + h.b * (t * y) = t * (h.a * x + h.b * y) := by ring
    _ ≤ t * h.c := mul_le_mul_of_nonneg_left this ht0
    _ ≤ h.c := by nlinarith

/-- The facts the certificate data establishes about `H0` and `H1`. -/
structure Certified (H0 H1 : Region) : Prop where
  inc0 : ∀ x y, H0.mem x y → H1.mem (x / s) ((y / 2) / s)
  inc1 : ∀ x y, H0.mem x y → H1.mem (x / s) (((x + y) / 2) / s)
  inc3 : ∀ x y, H1.mem x y → H1.mem (y / s) (((y - x) / 2) / s)
  inc5 : ∀ x y, H1.mem x y → H0.mem ((y / 2) / s^2) ((-x / 2 + y / 4) / s^2)
  inc_2 : ∀ x y, H1.mem x y → H0.mem ((y / 4) / s^4) ((-x / 4 + y / 16) / s^4)
  inc_1 : ∀ x y, H1.mem x y → H0.mem ((y / 4) / s^4) ((-x / 4 + 3 * y / 16) / s^4)
  inc_4s : ∀ x y, H1.mem x y → H1.mem ((33 / 64 * x + 33 / 512 * y) / s^2) ((33 / 64 * y) / s^2)
  inc_3s : ∀ x y, H1.mem x y → H1.mem ((33 / 64 * x - 33 / 512 * y) / s^2) ((33 / 64 * y) / s^2)
  outer : ∀ x y, H1.mem x y → |x| ≤ 8193 / 8192 ∧ |y| ≤ 379 / 512
  init : ∀ x y, 0 ≤ y → y ≤ x → x ≤ 1 → H1.mem (stretch * x) (stretch * y)
  zero : H1.mem 0 0
  lat0 : ∀ x y : ℤ, |x| ≤ 1 → |y| ≤ 1 → y ≠ 0 → ¬ H1.mem (x / L) (y / L)
  lat1 : ∀ x y : ℤ, |x| ≤ 2 → |y| ≤ 1 → y ≠ 0 → ¬ H1.mem (x * s^2 / L) (y * (2 * s^2) / L)

/-! ## The family `W` -/

/-- The scale factor `32/33` outside `|i| ≤ 2`. -/
def cf (i : ℤ) : ℚ := if -2 ≤ i ∧ i ≤ 2 then 1 else 32 / 33

/-- The region `W i`, as a predicate: `H0` at `i = -1`, otherwise a linear image of `H1`. For
`i = -k ≤ -2` the image of `(x, y)` is `(cf (x - 2y) s^(k+1), cf x s^(k+1) 2^k)`, for `i = k ≥ 0`
it is `(cf x s^k, cf y (2s)^k)`. -/
def W (H0 H1 : Region) (i : ℤ) (x y : ℚ) : Prop :=
  if i = -1 then H0.mem x y
  else if i ≤ -2 then
    H1.mem (cf i * (x - 2 * y) * s ^ ((-i).toNat + 1)) (cf i * x * s ^ ((-i).toNat + 1) * 2 ^ (-i).toNat)
  else H1.mem (cf i * x * s ^ i.toNat) (cf i * y * (2 * s) ^ i.toNat)

variable {H0 H1 : Region}

/-- The scale factor is positive. -/
theorem cf_pos (i : ℤ) : 0 < cf i := by unfold cf; split_ifs <;> norm_num

/-- The scale factor is at most one, so that it shrinks. -/
theorem cf_le_one (i : ℤ) : cf i ≤ 1 := by unfold cf; split_ifs <;> norm_num

/-- The scale factor is `1` on the central levels. -/
theorem cf_small {i : ℤ} (h1 : -2 ≤ i) (h2 : i ≤ 2) : cf i = 1 := by
  unfold cf; rw [if_pos ⟨h1, h2⟩]

/-- The scale factor is `32/33` below the central levels. -/
theorem cf_neg {i : ℤ} (h : i < -2) : cf i = 32 / 33 := by
  unfold cf; rw [if_neg (by omega)]

/-- The scale factor is `32/33` above the central levels. -/
theorem cf_big {i : ℤ} (h : 2 < i) : cf i = 32 / 33 := by
  unfold cf; rw [if_neg (by omega)]

/-- `W` at level `-1` is `H0`. -/
theorem W_neg_one (x y : ℚ) : W H0 H1 (-1) x y ↔ H0.mem x y := by
  unfold W; simp

/-- `W` at a level `-k ≤ -2`, unfolded to its image of `H1`. -/
theorem W_neg (k : ℕ) (hk : 2 ≤ k) (x y : ℚ) :
    W H0 H1 (-(k : ℤ)) x y ↔
      H1.mem (cf (-(k : ℤ)) * (x - 2 * y) * s ^ (k + 1)) (cf (-(k : ℤ)) * x * s ^ (k + 1) * 2 ^ k) := by
  unfold W
  rw [if_neg (by omega), if_pos (by omega)]
  simp

/-- `W` at a nonnegative level, unfolded to its image of `H1`. -/
theorem W_nonneg (k : ℕ) (x y : ℚ) :
    W H0 H1 (k : ℤ) x y ↔ H1.mem (cf k * x * s ^ k) (cf k * y * (2 * s) ^ k) := by
  unfold W
  rw [if_neg (by omega), if_neg (by omega)]
  simp

/-- Convexity, in the form the transitions need: the average of two points of `H1` is in `H1`. -/
theorem H1_mid (x₁ y₁ x₂ y₂ : ℚ) (h₁ : H1.mem x₁ y₁) (h₂ : H1.mem x₂ y₂) :
    H1.mem ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) := by
  intro h hh
  have a := h₁ h hh
  have b := h₂ h hh
  unfold HalfPlane.holds at *
  linarith

/-- `H1` shrinks. -/
theorem H1_smul (hc : Certified H0 H1) (x y t : ℚ) (h : H1.mem x y) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    H1.mem (t * x) (t * y) := Region.mem_smul H1 hc.zero x y t h ht0 ht1

/-! ## The maps at the far negative levels

For `i ≤ -4`, one step from level `i` to level `i + 1` acts on the point of `H1` that represents the
state as `(X, Y) ↦ ((X ± 2^i Y) / (2 s^2), Y / (2 s^2))`. The induction starts from the same map at
`i = -3`, which is the certified inclusion `inc_4s` (or `inc_3s`) followed by the shrink `32/33`.
Going one level further down replaces `2^i` by `2^(i-1)`, which is the midpoint of the previous map
and the shrink by `1 / (2 s^2) ≤ 1`. The step from level `-3` itself is `inc_4s` (or `inc_3s`)
alone, since the scale factor changes there. -/

/-- The map with the plus sign keeps `H1`: `inc_4s` with the shrink `32/33` at `n = 0`, then the
midpoint with the shrunk point at each further level. -/
theorem A_mem (hc : Certified H0 H1) (n : ℕ) (X Y : ℚ) (h : H1.mem X Y) :
    H1.mem ((X + Y / 2 ^ (n + 3)) / (2 * s ^ 2)) (Y / (2 * s ^ 2)) := by
  induction n with
  | zero =>
    have h1 := H1_smul hc _ _ (32 / 33) (hc.inc_4s X Y h) (by norm_num) (by norm_num)
    convert h1 using 1 <;> field_simp <;> ring
  | succ n ih =>
    have hsh : H1.mem ((1 / (2 * s ^ 2)) * X) ((1 / (2 * s ^ 2)) * Y) :=
      H1_smul hc _ _ _ h (div_pos one_pos two_s_sq_pos).le
        (by rw [div_le_one two_s_sq_pos]; exact two_s_sq_ge_one)
    have := H1_mid _ _ _ _ ih hsh
    convert this using 1 <;> field_simp <;> ring

/-- The map with the minus sign keeps `H1`, from `inc_3s` in the same way. -/
theorem A'_mem (hc : Certified H0 H1) (n : ℕ) (X Y : ℚ) (h : H1.mem X Y) :
    H1.mem ((X - Y / 2 ^ (n + 3)) / (2 * s ^ 2)) (Y / (2 * s ^ 2)) := by
  induction n with
  | zero =>
    have h1 := H1_smul hc _ _ (32 / 33) (hc.inc_3s X Y h) (by norm_num) (by norm_num)
    convert h1 using 1 <;> field_simp <;> ring
  | succ n ih =>
    have hsh : H1.mem ((1 / (2 * s ^ 2)) * X) ((1 / (2 * s ^ 2)) * Y) :=
      H1_smul hc _ _ _ h (div_pos one_pos two_s_sq_pos).le
        (by rw [div_le_one two_s_sq_pos]; exact two_s_sq_ge_one)
    have := H1_mid _ _ _ _ ih hsh
    convert this using 1 <;> field_simp <;> ring

/-! ## The two transitions -/

/-- The halving step, `g` even: from level `i` to level `i + 1`. -/
theorem w_transition0 (hc : Certified H0 H1) (i : ℤ) (x y : ℚ) (h : W H0 H1 i x y) :
    W H0 H1 (1 + i) (x / s) (y / (2 * s)) := by
  have hs := s_pos
  obtain ⟨k, rfl | rfl⟩ := Int.eq_nat_or_neg i
  · -- `i = k ≥ 0`: the same image, with the shrink `32/33` when crossing from `2` to `3`.
    rw [W_nonneg] at h
    rw [show (1 : ℤ) + k = ((k + 1 : ℕ) : ℤ) by push_cast; ring, W_nonneg]
    rcases Nat.lt_or_ge k 2 with hk | hk
    · rw [cf_small (by omega) (by omega)] at h ⊢
      convert h using 1 <;> field_simp <;> ring
    · rcases Nat.eq_or_lt_of_le hk with hk2 | hk3
      · subst hk2
        rw [cf_small (by omega) (by omega)] at h
        rw [cf_big (by omega)]
        have := H1_smul hc _ _ (32 / 33) h (by norm_num) (by norm_num)
        convert this using 1 <;> field_simp
        ring
      · rw [cf_big (by omega)] at h ⊢
        convert h using 1 <;> field_simp <;> ring
  · -- `i = -k`.
    rcases Nat.lt_or_ge k 4 with hk | hk
    · interval_cases k
      · -- `i = 0`
        simp only [Nat.cast_zero, neg_zero] at h ⊢
        rw [show (1 : ℤ) + 0 = ((1 : ℕ) : ℤ) by norm_num, W_nonneg]
        rw [show (0 : ℤ) = ((0 : ℕ) : ℤ) by norm_num, W_nonneg] at h
        rw [cf_small (by omega) (by omega)] at h ⊢
        convert h using 1 <;> field_simp
      · -- `i = -1`: `inc0`
        rw [show -((1 : ℕ) : ℤ) = -1 by norm_num, W_neg_one] at h
        rw [show (1 : ℤ) + -((1 : ℕ) : ℤ) = ((0 : ℕ) : ℤ) by norm_num, W_nonneg,
          cf_small (by omega) (by omega)]
        have := hc.inc0 x y h
        convert this using 1 <;> field_simp
      · -- `i = -2`: `inc_2`
        rw [W_neg 2 (le_refl _), cf_small (by omega) (by omega)] at h
        rw [show (1 : ℤ) + -((2 : ℕ) : ℤ) = -1 by norm_num, W_neg_one]
        have := hc.inc_2 _ _ h
        convert this using 1 <;> field_simp <;> ring
      · -- `i = -3`: `inc_4s`
        rw [W_neg 3 (by norm_num), cf_neg (by omega)] at h
        rw [show (1 : ℤ) + -((3 : ℕ) : ℤ) = -((2 : ℕ) : ℤ) by norm_num, W_neg 2 (le_refl _),
          cf_small (by omega) (by omega)]
        have := hc.inc_4s _ _ h
        convert this using 1 <;> field_simp <;> ring
    · -- `i = -k ≤ -4`: the midpoint map.
      obtain ⟨n, rfl⟩ : ∃ n, k = n + 4 := ⟨k - 4, by omega⟩
      rw [W_neg (n + 4) (by omega), cf_neg (by omega)] at h
      rw [show (1 : ℤ) + -((n + 4 : ℕ) : ℤ) = -((n + 3 : ℕ) : ℤ) by push_cast; ring,
        W_neg (n + 3) (by omega), cf_neg (by omega)]
      have := A_mem hc (n + 1) _ _ h
      convert this using 1 <;> field_simp <;> ring

/-- The step on odd `g`: from level `i < 0` to `i + 1`, or from `i ≥ 0` to `-i`. -/
theorem w_transition1 (hc : Certified H0 H1) (i : ℤ) (x y : ℚ) (h : W H0 H1 i x y) :
    (i < 0 → W H0 H1 (1 + i) (x / s) ((y + x) / (2 * s))) ∧
      (0 ≤ i → W H0 H1 (-i) (y / s) ((y - x) / (2 * s))) := by
  have hs := s_pos
  obtain ⟨k, rfl | rfl⟩ := Int.eq_nat_or_neg i
  · -- `i = k ≥ 0`: the swap.
    refine ⟨fun hlt => absurd hlt (by omega), fun _ => ?_⟩
    rw [W_nonneg] at h
    rcases Nat.lt_or_ge k 3 with hk | hk
    · interval_cases k
      · -- `i = 0`: `inc3`
        simp only [Nat.cast_zero, neg_zero]
        rw [show (0 : ℤ) = ((0 : ℕ) : ℤ) by norm_num, W_nonneg]
        rw [cf_small (by omega) (by omega)] at h ⊢
        have := hc.inc3 _ _ h
        convert this using 1 <;> field_simp
      · -- `i = 1`: `inc5`
        rw [cf_small (by omega) (by omega)] at h
        rw [show -((1 : ℕ) : ℤ) = -1 by norm_num, W_neg_one]
        have := hc.inc5 _ _ h
        convert this using 1 <;> field_simp
        ring
      · -- `i = 2`: an equality
        rw [cf_small (by omega) (by omega)] at h
        rw [W_neg 2 (le_refl _), cf_small (by omega) (by omega)]
        convert h using 1 <;> field_simp
        ring
    · -- `i = k ≥ 3`: an equality
      rw [cf_big (by omega)] at h
      rw [W_neg k (by omega), cf_neg (by omega)]
      convert h using 1 <;> field_simp <;> ring
  · -- `i = -k`.
    rcases Nat.eq_zero_or_pos k with rfl | hkpos
    · -- `i = 0`, as above
      refine ⟨fun hlt => absurd hlt (by omega), fun _ => ?_⟩
      simp only [Nat.cast_zero, neg_zero] at h ⊢
      rw [show (0 : ℤ) = ((0 : ℕ) : ℤ) by norm_num, W_nonneg, cf_small (by omega) (by omega)] at h ⊢
      have := hc.inc3 _ _ h
      convert this using 1 <;> field_simp
    refine ⟨fun _ => ?_, fun hge => absurd hge (by omega)⟩
    rcases Nat.lt_or_ge k 4 with hk | hk
    · interval_cases k
      · -- `i = -1`: `inc1`
        rw [show -((1 : ℕ) : ℤ) = -1 by norm_num, W_neg_one] at h
        rw [show (1 : ℤ) + -((1 : ℕ) : ℤ) = ((0 : ℕ) : ℤ) by norm_num, W_nonneg,
          cf_small (by omega) (by omega)]
        have := hc.inc1 x y h
        convert this using 1 <;> field_simp
        ring
      · -- `i = -2`: `inc_1`
        rw [W_neg 2 (le_refl _), cf_small (by omega) (by omega)] at h
        rw [show (1 : ℤ) + -((2 : ℕ) : ℤ) = -1 by norm_num, W_neg_one]
        have := hc.inc_1 _ _ h
        convert this using 1 <;> field_simp <;> ring
      · -- `i = -3`: `inc_3s`
        rw [W_neg 3 (by norm_num), cf_neg (by omega)] at h
        rw [show (1 : ℤ) + -((3 : ℕ) : ℤ) = -((2 : ℕ) : ℤ) by norm_num, W_neg 2 (le_refl _),
          cf_small (by omega) (by omega)]
        have := hc.inc_3s _ _ h
        convert this using 1 <;> field_simp <;> ring
    · -- `i = -k ≤ -4`: the midpoint map with the minus sign.
      obtain ⟨n, rfl⟩ : ∃ n, k = n + 4 := ⟨k - 4, by omega⟩
      rw [W_neg (n + 4) (by omega), cf_neg (by omega)] at h
      rw [show (1 : ℤ) + -((n + 4 : ℕ) : ℤ) = -((n + 3 : ℕ) : ℤ) by push_cast; ring,
        W_neg (n + 3) (by omega), cf_neg (by omega)]
      have := A'_mem hc (n + 1) _ _ h
      convert this using 1 <;> field_simp <;> ring

/-! ## The divstep iteration inside `W` -/

/-- `i = δ - 1/2 = (two_delta - 1) / 2`. -/
def idx (t : State) : ℤ := (t.two_delta - 1) / 2

/-- For odd `two_delta`, the index determines `two_delta`. -/
theorem idx_spec (t : State) (hd : t.two_delta % 2 = 1) : 2 * idx t + 1 = t.two_delta := by
  unfold idx; omega

/-- A step keeps `two_delta` odd, so the index stays defined. -/
theorem divstep_two_delta_odd (t : State) (hd : t.two_delta % 2 = 1) : (divstep t).two_delta % 2 = 1 := by
  unfold divstep; split_ifs <;> simp only <;> omega

/-- `two_delta` stays odd through the iteration. -/
theorem divsteps_two_delta_odd (n : ℕ) (t : State) (hd : t.two_delta % 2 = 1) : (divsteps n t).two_delta % 2 = 1 := by
  induction n generalizing t with
  | zero => simpa
  | succ n ih => rw [divsteps_succ]; exact ih _ (divstep_two_delta_odd t hd)

/-- The index moves by one step up, or flips sign from a nonnegative value. -/
theorem idx_divstep (t : State) (hd : t.two_delta % 2 = 1) :
    idx (divstep t) = 1 + idx t ∨ (0 ≤ idx t ∧ idx (divstep t) = -(idx t)) := by
  unfold divstep idx
  split_ifs with h
  · right; simp only; constructor <;> omega
  · left; simp only; omega

/-- One divstep is one transition of `W`, at the scale multiplied by `s`. -/
theorem W_step (hc : Certified H0 H1) (t : State) (hd : t.two_delta % 2 = 1) (hf : t.f % 2 = 1) (u : ℚ)
    (hu : 0 < u) (h : W H0 H1 (idx t) (t.f / u) (t.g / u)) :
    W H0 H1 (idx (divstep t)) ((divstep t).f / (u * s)) ((divstep t).g / (u * s)) := by
  have hs := s_pos
  by_cases hg : t.g % 2 = 1
  · by_cases hpos : 0 < t.two_delta
    · -- the swap
      have hstep : divstep t = ⟨2 - t.two_delta, t.g, (t.g - t.f) / 2⟩ := by
        simp only [divstep, if_pos (And.intro hpos hg)]
      have hidx : idx (divstep t) = -(idx t) := by rw [hstep]; unfold idx; simp only; omega
      have hcast : (((t.g - t.f) / 2 : ℤ) : ℚ) = ((t.g : ℚ) - t.f) / 2 := by
        rw [Int.cast_div (by omega) (by norm_num)]; push_cast; ring
      rw [hidx, hstep]
      simp only [hcast]
      have := (w_transition1 hc _ _ _ h).2 (by unfold idx; omega)
      convert this using 1 <;> field_simp
    · -- odd `g`, `i < 0`
      have hstep : divstep t = ⟨2 + t.two_delta, t.f, (t.g + t.f) / 2⟩ := by
        unfold divstep
        rw [if_neg (show ¬ (0 < t.two_delta ∧ t.g % 2 = 1) from fun h' => hpos h'.1), hg, one_mul]
      have hidx : idx (divstep t) = 1 + idx t := by rw [hstep]; unfold idx; simp only; omega
      have hcast : (((t.g + t.f) / 2 : ℤ) : ℚ) = ((t.g : ℚ) + t.f) / 2 := by
        rw [Int.cast_div (by omega) (by norm_num)]; push_cast; ring
      rw [hidx, hstep]
      simp only [hcast]
      have := (w_transition1 hc _ _ _ h).1 (by unfold idx; omega)
      convert this using 1 <;> field_simp
  · -- even `g`: the halving
    have hg0 : t.g % 2 = 0 := by omega
    have hstep : divstep t = ⟨2 + t.two_delta, t.f, t.g / 2⟩ := by
      unfold divstep
      rw [if_neg (show ¬ (0 < t.two_delta ∧ t.g % 2 = 1) from fun h' => hg h'.2), hg0, zero_mul, add_zero]
    have hidx : idx (divstep t) = 1 + idx t := by rw [hstep]; unfold idx; simp only; omega
    have hcast : ((t.g / 2 : ℤ) : ℚ) = (t.g : ℚ) / 2 := by
      rw [Int.cast_div (by omega) (by norm_num)]; push_cast; ring
    rw [hidx, hstep]
    simp only [hcast]
    have := w_transition0 hc _ _ _ h
    convert this using 1 <;> field_simp

/-- The invariant: after `n` steps from `two_delta = 1`, the state scaled by `u s^n` is in `W` at
the current index. -/
theorem iteration_W (hc : Certified H0 H1) (t₀ : State) (hd0 : t₀.two_delta = 1) (hf0 : t₀.f % 2 = 1)
    (u : ℚ) (hu : 0 < u) (h0 : H1.mem (t₀.f / u) (t₀.g / u)) (n : ℕ) :
    W H0 H1 (idx (divsteps n t₀)) ((divsteps n t₀).f / (u * s ^ n))
      ((divsteps n t₀).g / (u * s ^ n)) := by
  have hs := s_pos
  induction n with
  | zero =>
    simp only [divsteps_zero, pow_zero, mul_one]
    have hidx : idx t₀ = ((0 : ℕ) : ℤ) := by unfold idx; omega
    rw [hidx, W_nonneg, cf_small (by omega) (by omega)]
    simpa using h0
  | succ n ih =>
    rw [divsteps_succ']
    have hd : (divsteps n t₀).two_delta % 2 = 1 := divsteps_two_delta_odd n t₀ (by omega)
    have hf : (divsteps n t₀).f % 2 = 1 := divsteps_f_odd n t₀ hf0
    have := W_step hc _ hd hf (u * s ^ n) (mul_pos hu (pow_pos hs n)) ih
    rwa [pow_succ, ← mul_assoc]

/-! ## The lattice argument -/

/-- An integer point with `y ≠ 0` in `W k` at scale `t` forces `L < t s^k`. -/
theorem lattice_bound (hc : Certified H0 H1) (k : ℕ) (x y : ℤ) (hy : y ≠ 0) (t : ℚ) (ht : 0 < t)
    (h : W H0 H1 k (x / t) (y / t)) : L < t * s ^ k := by
  have hs := s_pos
  have hL := L_pos
  have hy1 : (1 : ℚ) ≤ |(y : ℚ)| := by
    have : (1 : ℤ) ≤ |y| := Int.one_le_abs hy
    exact_mod_cast this
  rw [W_nonneg] at h
  rcases Nat.lt_or_ge k 2 with hk | hk
  · interval_cases k
    · -- `k = 0`: shrink to scale `L` and enumerate
      rw [cf_small (by omega) (by omega)] at h
      simp only [pow_zero, mul_one, one_mul] at h ⊢
      by_contra hlt
      have hlt' : t ≤ L := not_lt.mp hlt
      have hsh := H1_smul hc _ _ (t / L) h (div_pos ht hL).le ((div_le_one hL).2 hlt')
      have hsh' : H1.mem (x / L) (y / L) := by
        convert hsh using 1 <;> field_simp
      obtain ⟨hx, hyb⟩ := hc.outer _ _ hsh'
      rw [abs_div, abs_of_pos hL, div_le_iff₀ hL] at hx hyb
      have hx1 : |x| ≤ 1 := by
        have h2 : ((|x| : ℤ) : ℚ) < 2 := by rw [Int.cast_abs]; exact lt_of_le_of_lt hx (by norm_num [L])
        have : |x| < 2 := by exact_mod_cast h2
        omega
      have hy1' : |y| ≤ 1 := by
        have h2 : ((|y| : ℤ) : ℚ) < 2 := by rw [Int.cast_abs]; exact lt_of_le_of_lt hyb (by norm_num [L])
        have : |y| < 2 := by exact_mod_cast h2
        omega
      exact hc.lat0 x y hx1 hy1' hy hsh'
    · -- `k = 1`: shrink to scale `L / s` and enumerate
      rw [cf_small (by omega) (by omega)] at h
      simp only [pow_one, one_mul] at h ⊢
      by_contra hlt
      have hlt' : t * s ≤ L := not_lt.mp hlt
      have hsh := H1_smul hc _ _ (t * s / L) h (div_pos (mul_pos ht hs) hL).le ((div_le_one hL).2 hlt')
      have hsh' : H1.mem (x * s ^ 2 / L) (y * (2 * s ^ 2) / L) := by
        convert hsh using 1 <;> field_simp
      obtain ⟨hx, hyb⟩ := hc.outer _ _ hsh'
      rw [abs_div, abs_of_pos hL, div_le_iff₀ hL, abs_mul, abs_of_pos (pow_pos hs 2)] at hx
      rw [abs_div, abs_of_pos hL, div_le_iff₀ hL, abs_mul, abs_of_pos two_s_sq_pos] at hyb
      have hx2 : |x| ≤ 2 := by
        have h3 : |(x : ℚ)| < 3 := by
          rw [← le_div_iff₀ (pow_pos hs 2)] at hx
          exact lt_of_le_of_lt hx (by norm_num [L, s])
        have h3' : ((|x| : ℤ) : ℚ) < 3 := by rw [Int.cast_abs]; exact h3
        have : |x| < 3 := by exact_mod_cast h3'
        omega
      have hy1' : |y| ≤ 1 := by
        have h2 : |(y : ℚ)| < 2 := by
          rw [← le_div_iff₀ two_s_sq_pos] at hyb
          exact lt_of_le_of_lt hyb (by norm_num [L, s])
        have h2' : ((|y| : ℤ) : ℚ) < 2 := by rw [Int.cast_abs]; exact h2
        have : |y| < 2 := by exact_mod_cast h2'
        omega
      exact hc.lat1 x y hx2 hy1' hy hsh'
  · -- `k ≥ 2`: the outer box's bound on `y`
    obtain ⟨-, hyb⟩ := hc.outer _ _ h
    have hcf : 32 / 33 ≤ cf k := by
      unfold cf; split_ifs <;> norm_num
    have hcfpos := cf_pos k
    rw [show cf k * ((y : ℚ) / t) * (2 * s) ^ k = (cf k * y * (2 * s) ^ k) / t by ring, abs_div,
      abs_of_pos ht, div_le_iff₀ ht, abs_mul, abs_mul, abs_of_pos hcfpos,
      abs_of_pos (pow_pos (mul_pos two_pos hs) k)] at hyb
    -- `cf (2s)^k ≤ cf |y| (2s)^k ≤ 379/512 t`, so `t s^k ≥ cf (2 s^2)^k 512/379`.
    have h1 : cf k * (2 * s) ^ k ≤ 379 / 512 * t := by
      calc cf k * (2 * s) ^ k = cf k * 1 * (2 * s) ^ k := by ring
        _ ≤ cf k * |(y : ℚ)| * (2 * s) ^ k := by gcongr
        _ ≤ 379 / 512 * t := hyb
    have hpow : (2 * s ^ 2) ^ 2 ≤ (2 * s ^ 2) ^ k := pow_le_pow_right₀ two_s_sq_ge_one hk
    have hkey : 32 / 33 * (2 * s ^ 2) ^ 2 ≤ 379 / 512 * (t * s ^ k) := by
      calc 32 / 33 * (2 * s ^ 2) ^ 2 ≤ cf k * (2 * s ^ 2) ^ k := by gcongr
        _ = (cf k * (2 * s) ^ k) * s ^ k := by rw [mul_pow, mul_pow, ← pow_mul]; ring_nf
        _ ≤ 379 / 512 * t * s ^ k := by gcongr
        _ = 379 / 512 * (t * s ^ k) := by ring
    have hnum : L < 32 / 33 * (2 * s ^ 2) ^ 2 * (512 / 379) := by norm_num [L, s]
    linarith

/-! ## The size bound and the end-to-end theorem -/

/-- The extra exponent that carries the bound through the levels below zero. -/
def ex (t : State) : ℕ := if idx t < 0 then (-(idx t) - 1).toNat else (idx t).toNat

/-- While `g` stays nonzero, the scale stays above `L`, with the exponent strengthened by the
current index. -/
theorem iteration_sizebound (hc : Certified H0 H1) (t₀ : State) (hd0 : t₀.two_delta = 1)
    (hf0 : t₀.f % 2 = 1) (u : ℚ) (hu : 0 < u) (h0 : H1.mem (t₀.f / u) (t₀.g / u)) (m : ℕ)
    (hg : ∀ n ≤ m, (divsteps n t₀).g ≠ 0) : L < u * s ^ (m + ex (divsteps m t₀)) := by
  have hs := s_pos
  induction m with
  | zero =>
    have hW := iteration_W hc t₀ hd0 hf0 u hu h0 0
    simp only [divsteps_zero, pow_zero, mul_one] at hW
    have hidx : idx t₀ = 0 := by unfold idx; omega
    have hex : ex t₀ = 0 := by unfold ex; rw [hidx]; simp
    simp only [divsteps_zero, hex, Nat.add_zero, pow_zero, mul_one]
    have := lattice_bound hc 0 t₀.f t₀.g (hg 0 (le_refl _)) u hu (by rw [hidx] at hW; exact hW)
    simpa using this
  | succ m ih =>
    set t := divsteps m t₀ with ht
    have hd : t.two_delta % 2 = 1 := divsteps_two_delta_odd m t₀ (by omega)
    have hstep : divsteps (m + 1) t₀ = divstep t := divsteps_succ' m t₀
    rw [hstep]
    rcases lt_or_ge (idx (divstep t)) 0 with hneg | hnn
    · -- carried from the previous step
      have ih' := ih (fun n hn => hg n (by omega))
      have hex : m + ex t = m + 1 + ex (divstep t) := by
        unfold ex
        rcases idx_divstep t hd with h1 | ⟨h2, h3⟩
        · rw [if_pos (by omega), if_pos hneg]; omega
        · rw [if_neg (by omega), if_pos hneg]; omega
      rw [← hex]; exact ih'
    · -- the lattice bound at the new level
      have hW := iteration_W hc t₀ hd0 hf0 u hu h0 (m + 1)
      rw [hstep] at hW
      obtain ⟨k, hk⟩ : ∃ k : ℕ, idx (divstep t) = k := ⟨(idx (divstep t)).toNat, by omega⟩
      rw [hk] at hW
      have hex : ex (divstep t) = k := by unfold ex; rw [hk]; simp
      rw [hex]
      have hgz : (divstep t).g ≠ 0 := by rw [← hstep]; exact hg (m + 1) (le_refl _)
      have := lattice_bound hc k _ _ hgz (u * s ^ (m + 1)) (mul_pos hu (pow_pos hs _)) hW
      rw [pow_add, ← mul_assoc]; exact this

/-- `s^n` is monotone decreasing in `n`. -/
theorem s_pow_le (m n : ℕ) (h : m ≤ n) : s ^ n ≤ s ^ m :=
  pow_le_pow_of_le_one s_pos.le s_lt_one.le h

/-- The end-to-end theorem: from `0 ≤ g ≤ f ≤ M`, odd `f`, and `M s^m ≤ L · stretch`, some `g_n`
with `n ≤ m` is zero. -/
theorem endtoend (hc : Certified H0 H1) (f g : ℤ) (M : ℚ) (hM : 0 < M) (hf : f % 2 = 1)
    (hg0 : 0 ≤ g) (hgf : g ≤ f) (hfM : (f : ℚ) ≤ M) (m : ℕ) (hm : M * s ^ m ≤ L * stretch) :
    ∃ n ≤ m, (divsteps n ⟨1, f, g⟩).g = 0 := by
  have hs := s_pos
  have hL := L_pos
  have hst := stretch_pos
  by_contra hno
  have hno' : ∀ n ≤ m, (divsteps n ⟨1, f, g⟩).g ≠ 0 := fun n hn hz => hno ⟨n, hn, hz⟩
  set u : ℚ := M / stretch with hu
  have hupos : 0 < u := div_pos hM hst
  have h0 : H1.mem ((f : ℚ) / u) ((g : ℚ) / u) := by
    have hg0' : (0 : ℚ) ≤ (g : ℚ) / M := div_nonneg (by exact_mod_cast hg0) hM.le
    have hgf' : (g : ℚ) / M ≤ (f : ℚ) / M :=
      div_le_div_of_nonneg_right (by exact_mod_cast hgf) hM.le
    have hf1 : (f : ℚ) / M ≤ 1 := (div_le_one hM).2 hfM
    have := hc.init _ _ hg0' hgf' hf1
    have e1 : (f : ℚ) / u = stretch * ((f : ℚ) / M) := by rw [hu]; field_simp
    have e2 : (g : ℚ) / u = stretch * ((g : ℚ) / M) := by rw [hu]; field_simp
    rw [e1, e2]; exact this
  have := iteration_sizebound hc ⟨1, f, g⟩ rfl hf u hupos h0 m hno'
  have hle : u * s ^ (m + ex (divsteps m ⟨1, f, g⟩)) ≤ u * s ^ m :=
    mul_le_mul_of_nonneg_left (s_pow_le m _ (Nat.le_add_right _ _)) hupos.le
  have : L < u * s ^ m := lt_of_lt_of_le this hle
  rw [hu, div_mul_eq_mul_div, lt_div_iff₀ hst] at this
  linarith

/-! ## The exponent bound -/

set_option exponentiation.threshold 10000 in
/-- The first half of the exponent bound, `2^4096 s^9437 ≤ 1`, from an integer inequality that
the kernel evaluates. -/
theorem big_one : (2 : ℚ) ^ 4096 * s ^ 9437 ≤ 1 := by
  have h : (2 : ℕ) ^ 4096 * 30902639 ^ 9437 ≤ 41749730 ^ 9437 := by decide
  have h' : ((2 : ℕ) ^ 4096 * 30902639 ^ 9437 : ℚ) ≤ (41749730 ^ 9437 : ℕ) := by exact_mod_cast h
  push_cast at h'
  unfold s
  rw [div_pow, ← mul_div_assoc, div_le_one (by positivity)]
  exact h'

set_option exponentiation.threshold 10000 in
/-- The second half of the exponent bound, `s ≤ (L · stretch)^4096`, likewise. -/
theorem big_two : s ≤ (L * stretch) ^ 4096 := by
  have h : (30902639 : ℕ) * 8388608 ^ 4096 ≤ 41749730 * 8388391 ^ 4096 := by decide
  have h' : ((30902639 : ℕ) * 8388608 ^ 4096 : ℚ) ≤ (41749730 * 8388391 ^ 4096 : ℕ) := by
    exact_mod_cast h
  push_cast at h'
  unfold s L stretch
  rw [show ((3047 : ℚ) / 2048 * (2753 / 4096)) = 8388391 / 8388608 by norm_num, div_pow,
    div_le_div_iff₀ (by norm_num) (by positivity), mul_comm ((8388391 : ℚ) ^ 4096)]
  exact h'.trans (le_of_eq (by norm_num))

/-- `9437 b + 1 ≤ 4096 m` gives `2^b s^m ≤ L · stretch`. -/
theorem spow_bound (b m : ℕ) (h : 9437 * b + 1 ≤ 4096 * m) : (2 : ℚ) ^ b * s ^ m ≤ L * stretch := by
  have hs := s_pos
  have hpos : (0 : ℚ) ≤ 2 ^ b * s ^ m := mul_nonneg (by positivity) (pow_nonneg hs.le m)
  have hfz : (0 : ℚ) ≤ L * stretch := (mul_pos L_pos stretch_pos).le
  refine (pow_le_pow_iff_left₀ hpos hfz (by norm_num : (4096 : ℕ) ≠ 0)).1 ?_
  calc (2 ^ b * s ^ m) ^ 4096 = 2 ^ (4096 * b) * s ^ (4096 * m) := by
        rw [mul_pow, ← pow_mul, ← pow_mul]; ring_nf
    _ ≤ 2 ^ (4096 * b) * s ^ (9437 * b + 1) :=
        mul_le_mul_of_nonneg_left (pow_le_pow_of_le_one hs.le s_lt_one.le h) (by positivity)
    _ = (2 ^ 4096 * s ^ 9437) ^ b * s := by
        rw [mul_pow, ← pow_mul, ← pow_mul, pow_succ]; ring_nf
    _ ≤ 1 ^ b * s :=
        mul_le_mul_of_nonneg_right
          (pow_le_pow_left₀ (mul_nonneg (by positivity) (pow_nonneg hs.le _)) big_one b) hs.le
    _ = s := by ring
    _ ≤ (L * stretch) ^ 4096 := big_two

/-! ## The bound -/

/-- Theorem 5 from a certificate. -/
theorem terminationBound_of_certified (hc : Certified H0 H1) (b : ℕ) : TerminationBound b := by
  intro f g hf hg0 hgf hfb
  have hm : 9437 * b + 1 ≤ 4096 * iterations b := by unfold iterations; omega
  have hpow : (0 : ℚ) < 2 ^ b := by positivity
  obtain ⟨n, hn, hgn⟩ := endtoend hc f g (2 ^ b) hpow hf hg0 hgf
    (by have : (f : ℚ) < 2 ^ b := by exact_mod_cast hfb
        exact this.le) (iterations b) (spow_bound b _ hm)
  obtain ⟨k, hk⟩ : ∃ k, iterations b = n + k := ⟨iterations b - n, by omega⟩
  rw [hk, divsteps_add]
  exact (divsteps_of_g_zero k _ hgn).2.1

end PastaCurves.Inversion.Hull
