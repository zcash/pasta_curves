import PastaCurves.Glv.Spec

/-!
# `wnaf_digits`: no panic, length, digit shape and value

Theorems about the model of `wnaf_digits` (`Wnaf.exec`, `Model.lean`), on a magnitude `a` and a
sign `neg`:

- below `2^127` no build panics: the assertion holds, no write is out of bounds, and no `u128` or
  `i8` operation overflows (`wnaf_digits_no_panic`);
- on every `u128` the loop makes at most `⌊log₂ a⌋ + 2` passes, so below `2^127` it returns at most
  128 digits (`wnaf_digits_length_le_log2`, `wnaf_digits_length_le_128`);
- every digit is a width-4 wNAF digit, so `|d| / 2` indexes a table of four entries
  (`wnaf_digits_shape`);
- below `2^127` the digits encode `a`, or `-a` when `neg`, and are zero from position `n` on
  (`wnaf_digits_value`);
- `Decomposed::new`, on two halves below `2^127`, does not panic and holds their digits
  (`decomposed_of_halves_spec`).

Two remarks about the model follow them: the least magnitude whose recoding needs a 129th digit
(`wnaf_digits_below_least_failing`, `wnaf_digits_least_failing` and its neighbours), and where
the assertion and the overflow checks fire on a `u128` (`wnaf_digits_checks`, with the value
returned where a `u128` operation wraps, `wnaf_digits_wrap_value`). Aeneas' translation of
`wnaf_digits` has the semantics of a debug build (`WnafRefines.lean`), so what the remarks say
about builds that lack debug assertions or overflow checks rests on the model's faithfulness to
the Rust, not on a proof.

The recoding's arithmetic is read through three functions of `k` alone: the digit `wdigit k`
written for an odd `k`, the next value `knext k`, and the number of passes `passes k`. A bound
`gmin m` on the magnitudes whose loop makes at least `m` passes comes with its recurrence, and
`gmin_le` gives the length bounds and the least failing magnitude.
-/

namespace PastaCurves.Glv

namespace Wnaf

/-! ## The recoding's arithmetic -/

/-- The digit `wnaf_digits` writes for an odd `k`, before the sign: `k mod 16`, taken in
`[-8, 8)`. -/
def wdigit (k : ℕ) : ℤ := if k % 16 < 8 then ((k % 16 : ℕ) : ℤ) else ((k % 16 : ℕ) : ℤ) - 16

/-- `k` after one pass of the loop: an odd `k` loses its digit, wrapped as a `u128`, and then `k`
is halved. -/
def knext (k : ℕ) : ℕ := if k % 2 = 1 then wrapU128 ((k : ℤ) - wdigit k) / 2 else k / 2

/-- A pass lowers a nonzero `k`. -/
private theorem knext_lt (k : ℕ) (h : k ≠ 0) : knext k < k := by
  have hp : (2 : ℤ) ^ 128 = 340282366920938463463374607431768211456 := by norm_num
  simp only [knext, wdigit, wrapU128, hp]
  split_ifs <;> omega

/-- The number of passes the loop makes from `k` until `k = 0`. -/
private def passes (k : ℕ) : ℕ := if h : k = 0 then 0 else passes (knext k) + 1
termination_by k
decreasing_by exact knext_lt k h

/-- No pass from `0`. -/
@[simp] private theorem passes_zero : passes 0 = 0 := by
  rw [passes]; simp

/-- From a nonzero `k`, one pass and then the passes from `knext k`. -/
private theorem passes_of_ne (k : ℕ) (h : k ≠ 0) : passes k = passes (knext k) + 1 := by
  rw [passes]; simp [h]

/-- `k` four passes after an odd `k`, when none of them reaches `0`: `(k - d) / 16`, read through
the `u128` wrap. -/
private def kq (k : ℕ) : ℕ := wrapU128 ((k : ℤ) - wdigit k) / 16

/-- The window `(gmin (j + 4), gmin (j + 3), gmin (j + 2), gmin (j + 1))` of the recurrence
below. -/
private def gminW : ℕ → ℕ × ℕ × ℕ × ℕ
  | 0 => (8, 4, 2, 1)
  | j + 1 => match gminW j with
    | (a, b, c, d) => (min (2 * a) (16 * d - 7), a, b, c)

/-- A lower bound on every magnitude whose loop makes at least `m` passes (`gmin_le`): `1, 2, 4,
8`, then `gmin m = min (2 · gmin (m - 1)) (16 · gmin (m - 4) - 7)`. A pass halves an even `k`; an
odd `k` takes four passes to reach `(k - d) / 16`, with `|d| ≤ 7`. -/
private def gmin (m : ℕ) : ℕ :=
  if m = 0 then 0 else if m ≤ 4 then 2 ^ (m - 1) else (gminW (m - 4)).1

/-- At 129 passes the bound is the least failing magnitude itself. -/
private theorem gmin_129 : gmin 129 = leastFailing := by decide +kernel

/-- What a run on `a < 2^127` keeps: `k` stays below `2^127`, the passes end by position 128, the
digits are zero from `n` on, and the digits written so far and what `k` still holds make `±a`. -/
private def ValInv (a : ℕ) (neg : Bool) (s : State) : Prop :=
  s.k < 2 ^ 127 ∧ s.n + passes s.k ≤ 128 ∧ (∀ i : Fin 128, s.n ≤ (i : ℕ) → s.digits i = 0) ∧
    ∑ i : Fin 128, s.digits i * 2 ^ (i : ℕ) + (if neg then -(s.k : ℤ) else s.k) * 2 ^ s.n =
      if neg then -(a : ℤ) else a

/-! ## One pass -/

/-- A pass takes `k` to `knext k`. -/
theorem step_k (neg : Bool) (s : State) : (step neg s).k = knext s.k := by
  have h1 : s.k &&& 1 = s.k % 2 := Nat.and_one_is_mod s.k
  have h15 : s.k &&& 0xF = s.k % 16 := Nat.and_two_pow_sub_one_eq_mod s.k 4
  have hp : (2 : ℤ) ^ 128 = 340282366920938463463374607431768211456 := by norm_num
  simp only [Wnaf.step, knext, wdigit, Nat.shiftRight_eq_div_pow, pow_one, h1]
  split
  · simp only [Wnaf.emit, Wnaf.write_k, wrapU128, wrapI8, h15, hp]
    split_ifs <;> omega
  · simp

/-- A pass adds one to `n`. -/
theorem step_n (neg : Bool) (s : State) : (step neg s).n = s.n + 1 := by
  unfold Wnaf.step
  split <;> simp [Wnaf.emit]

/-- The digit of an odd `k` lies in `[-7, 7]`, and `k - d` is not negative. -/
private theorem wdigit_bounds (k : ℕ) (h : k % 2 = 1) :
    -7 ≤ wdigit k ∧ wdigit k ≤ 7 ∧ 0 ≤ (k : ℤ) - wdigit k := by
  unfold wdigit; split_ifs <;> omega

/-- A pass leaves the assertion and `i8` flags alone; it sets the `u128` flag exactly when `k` is
odd and `k - d` reaches `2^128`, and the bounds flag exactly when `k` is odd and `n ≥ 128`. -/
theorem step_flags (neg : Bool) (s : State) :
    (step neg s).assertFailed = s.assertFailed ∧
    (step neg s).i8Overflow = s.i8Overflow ∧
    (step neg s).u128Overflow = (s.u128Overflow ||
      (decide (s.k % 2 = 1) && decide (2 ^ 128 ≤ (s.k : ℤ) - wdigit s.k))) ∧
    (step neg s).outOfBounds = (s.outOfBounds || (decide (s.k % 2 = 1) && decide (128 ≤ s.n))) := by
  have h1 : s.k &&& 1 = s.k % 2 := Nat.and_one_is_mod s.k
  have h15 : s.k &&& 0xF = s.k % 16 := Nat.and_two_pow_sub_one_eq_mod s.k 4
  simp only [Wnaf.step, h1]
  split
  · rename_i hodd
    have hlow : wrapI8 ((s.k &&& 0xF : ℕ) : ℤ) = ((s.k % 16 : ℕ) : ℤ) := by
      rw [h15]; unfold wrapI8; omega
    have hd : (if 8 ≤ ((s.k % 16 : ℕ) : ℤ) then wrapI8 (((s.k % 16 : ℕ) : ℤ) - 16)
        else ((s.k % 16 : ℕ) : ℤ)) = wdigit s.k := by
      unfold wdigit wrapI8; split_ifs <;> omega
    obtain ⟨hb1, hb2, hb3⟩ := wdigit_bounds s.k hodd
    have e1 : overI8 (((s.k % 16 : ℕ) : ℤ) - 16) = false := by unfold overI8; simp; omega
    have e2 : overI8 (-wdigit s.k) = false := by unfold overI8; simp; omega
    have hx : ∀ s' : Wnaf.State, s'.k = s.k →
        (if 0 ≤ wdigit s.k then (s'.k : ℤ) - wrapU128 (wdigit s.k)
          else (s'.k : ℤ) + wrapU128 (wrapI8 (-wdigit s.k))) = (s.k : ℤ) - wdigit s.k := by
      intro s' hs; rw [hs]; unfold wrapU128 wrapI8; split_ifs <;> omega
    have ho : overU128 ((s.k : ℤ) - wdigit s.k) = decide (2 ^ 128 ≤ (s.k : ℤ) - wdigit s.k) := by
      unfold overU128; simp; omega
    simp only [Wnaf.emit, hlow, hd, e1, e2, Bool.and_false, Bool.or_false,
      hx _ (Wnaf.write_k _ _), ho, hodd, decide_true, Bool.true_and]
    unfold Wnaf.write
    split <;> simp <;> omega
  · rename_i hev
    have : ¬ (s.k % 2 = 1) := hev
    simp [this]

/-- A pass on an odd `k` writes its signed digit at position `n`, and changes no other digit. -/
theorem step_digits (neg : Bool) (s : State) (i : Fin 128) :
    (step neg s).digits i =
      if s.k % 2 = 1 ∧ (i : ℕ) = s.n then (if neg then -wdigit s.k else wdigit s.k)
      else s.digits i := by
  have h1 : s.k &&& 1 = s.k % 2 := Nat.and_one_is_mod s.k
  have h15 : s.k &&& 0xF = s.k % 16 := Nat.and_two_pow_sub_one_eq_mod s.k 4
  simp only [Wnaf.step, h1]
  split
  · rename_i hodd
    simp only [Wnaf.emit, Wnaf.write]
    split
    · simp only [Function.update_apply, Fin.ext_iff]
      by_cases hi : (i : ℕ) = s.n
      · simp only [hi, hodd, and_self, if_true, wdigit, wrapI8, h15]
        split_ifs <;> omega
      · simp [hi]
    · rename_i hn
      have : ¬ ((i : ℕ) = s.n) := by omega
      simp [this]
  · rename_i hev
    have : ¬ (s.k % 2 = 1) := hev
    simp [this]

/-! ## The loop -/

/-- The loop adds `passes k` to `n`. -/
private theorem loop_n (neg : Bool) : ∀ (fuel : ℕ) (s : State), s.k ≤ fuel →
    (loop neg fuel s).n = s.n + passes s.k := by
  intro fuel
  induction fuel with
  | zero => intro s h; simp [Wnaf.loop, show s.k = 0 by omega]
  | succ f ih =>
    intro s h
    simp only [Wnaf.loop]
    split
    · rename_i h0; simp [h0]
    · rename_i h0
      have hl := knext_lt s.k h0
      rw [ih (Wnaf.step neg s) (by rw [step_k]; omega), step_n, step_k, passes_of_ne s.k h0]
      omega

/-- The loop leaves the assertion and `i8` flags alone, and sets the `u128` flag from no `k`
below `2^128 - 16`. -/
private theorem loop_flags (neg : Bool) : ∀ (fuel : ℕ) (s : State),
    (loop neg fuel s).assertFailed = s.assertFailed ∧
    (loop neg fuel s).i8Overflow = s.i8Overflow ∧
    (s.k < 2 ^ 128 - 16 → (loop neg fuel s).u128Overflow = s.u128Overflow) := by
  intro fuel
  induction fuel with
  | zero => intro s; simp [Wnaf.loop]
  | succ f ih =>
    intro s
    simp only [Wnaf.loop]
    split
    · exact ⟨rfl, rfl, fun _ => rfl⟩
    · rename_i h0
      obtain ⟨ha, hi, hu⟩ := ih (Wnaf.step neg s)
      obtain ⟨fa, fi, fu, -⟩ := step_flags neg s
      refine ⟨by rw [ha, fa], by rw [hi, fi], fun hlt => ?_⟩
      have hl := knext_lt s.k h0
      rw [hu (by rw [step_k]; omega), fu]
      have : -8 ≤ wdigit s.k := by unfold wdigit; split_ifs <;> omega
      have hp : (2 : ℤ) ^ 128 = 340282366920938463463374607431768211456 := by norm_num
      have : ¬ ((2 : ℤ) ^ 128 ≤ (s.k : ℤ) - wdigit s.k) := by
        rw [hp]; omega
      rw [hp] at this
      simp; intro _ h; omega

/-- The loop writes nothing out of bounds when its passes end by position 128. -/
private theorem loop_oob (neg : Bool) : ∀ (fuel : ℕ) (s : State),
    s.k ≤ fuel → s.n + passes s.k ≤ 128 → (loop neg fuel s).outOfBounds = s.outOfBounds := by
  intro fuel
  induction fuel with
  | zero => intro s _ _; simp [Wnaf.loop]
  | succ f ih =>
    intro s hle hp
    simp only [Wnaf.loop]
    split
    · rfl
    · rename_i h0
      have hl := knext_lt s.k h0
      have hps := passes_of_ne s.k h0
      rw [ih (Wnaf.step neg s) (by rw [step_k]; omega) (by rw [step_n, step_k]; omega),
        (step_flags neg s).2.2.2]
      have : ¬ (128 ≤ s.n) := by omega
      simp [this]

/-- The loop keeps every digit a width-4 wNAF digit. -/
private theorem loop_shape (neg : Bool) : ∀ (fuel : ℕ) (s : State),
    (∀ i, IsWnafDigit (s.digits i)) → ∀ i, IsWnafDigit ((loop neg fuel s).digits i) := by
  intro fuel
  induction fuel with
  | zero => intro s h; simpa [Wnaf.loop] using h
  | succ f ih =>
    intro s h
    simp only [Wnaf.loop]
    split
    · exact h
    · apply ih
      intro i
      rw [step_digits]
      split
      · have hw : IsWnafDigit (wdigit s.k) ∧ IsWnafDigit (-wdigit s.k) := by
          unfold IsWnafDigit; simp only [Int.odd_iff]
          unfold wdigit; split_ifs <;> omega
        split
        · exact hw.2
        · exact hw.1
      · exact h i

/-- Writing `x` over a zero digit at position `m` adds `x·2^m` to the digits' value. -/
private theorem sum_point (f g : Fin 128 → ℤ) (m : Fin 128) (x : ℤ) (hf : f m = 0) (hg : g m = x)
    (hne : ∀ i, i ≠ m → g i = f i) :
    ∑ i : Fin 128, g i * 2 ^ (i : ℕ) = ∑ i : Fin 128, f i * 2 ^ (i : ℕ) + x * 2 ^ (m : ℕ) := by
  have : ∀ i : Fin 128, g i * 2 ^ (i : ℕ) =
      f i * 2 ^ (i : ℕ) + (if m = i then x * 2 ^ (m : ℕ) else 0) := by
    intro i
    by_cases h : m = i
    · subst h; rw [hg, hf]; simp
    · rw [hne i (Ne.symm h), if_neg h, add_zero]
  rw [Finset.sum_congr rfl (fun i _ => this i), Finset.sum_add_distrib, Finset.sum_ite_eq]
  simp

/-- A pass from a nonzero `k` keeps the value invariant. -/
private theorem step_value (a : ℕ) (neg : Bool) (s : State) (hk0 : s.k ≠ 0)
    (hv : ValInv a neg s) : ValInv a neg (step neg s) := by
  obtain ⟨hlt, hpass, hzero, hsum⟩ := hv
  have hl := knext_lt s.k hk0
  have hps := passes_of_ne s.k hk0
  have hn128 : s.n < 128 := by omega
  refine ⟨by rw [step_k]; omega, by rw [step_n, step_k]; omega, ?_, ?_⟩
  · intro i hi
    rw [step_n] at hi
    rw [step_digits, if_neg (by omega)]
    exact hzero i (by omega)
  · rw [step_n, step_k]
    simp only [step_digits]
    rcases Nat.mod_two_eq_zero_or_one s.k with he | ho
    · have hkn : (s.k : ℤ) = 2 * (knext s.k : ℤ) := by
        have : knext s.k = s.k / 2 := by simp [knext, he]
        omega
      simp only [he, zero_ne_one, false_and, if_false]
      rw [← hsum, hkn, pow_succ]
      split <;> ring
    · have hp : (2 : ℤ) ^ 128 = 340282366920938463463374607431768211456 := by norm_num
      have hkn : (s.k : ℤ) = 2 * (knext s.k : ℤ) + wdigit s.k := by
        simp only [knext, ho, if_true, wrapU128, hp]
        unfold wdigit; split_ifs <;> omega
      set m : Fin 128 := ⟨s.n, hn128⟩
      set x : ℤ := if neg then -wdigit s.k else wdigit s.k
      have hs := sum_point s.digits
        (fun i => if s.k % 2 = 1 ∧ (i : ℕ) = s.n then x else s.digits i) m x
        (hzero m le_rfl) (if_pos ⟨ho, rfl⟩) (fun i hi => if_neg (fun hc => hi (Fin.ext hc.2)))
      rw [hs, ← hsum, hkn, pow_succ]
      simp only [x]
      split <;> ring

/-- The loop keeps the value invariant. -/
private theorem loop_value (a : ℕ) (neg : Bool) : ∀ (fuel : ℕ) (s : State),
    ValInv a neg s → ValInv a neg (loop neg fuel s) := by
  intro fuel
  induction fuel with
  | zero => intro s h; simpa [Wnaf.loop] using h
  | succ f ih =>
    intro s h
    simp only [Wnaf.loop]
    split
    · exact h
    · rename_i h0; exact ih _ (step_value a neg s h0 h)

/-! ## The number of passes -/

/-- The window `gminW j` holds `gmin (j + 4), …, gmin (j + 1)`. -/
private theorem gminW_eq :
    ∀ j, gminW j = (gmin (j + 4), gmin (j + 3), gmin (j + 2), gmin (j + 1))
  | 0 => by decide
  | j + 1 => by
    have ih := gminW_eq j
    have e : gmin (j + 1 + 4) = (gminW (j + 1)).1 := by
      simp [gmin]
    rw [e]
    simp only [gminW, ih]

/-- The recurrence of `gmin`. -/
private theorem gmin_rec (m : ℕ) (hm : 5 ≤ m) :
    gmin m = min (2 * gmin (m - 1)) (16 * gmin (m - 4) - 7) := by
  obtain ⟨j, rfl⟩ : ∃ j, m = j + 5 := ⟨m - 5, by omega⟩
  have h1 : gmin (j + 5) = (gminW (j + 1)).1 := by simp [gmin]
  rw [h1]
  simp only [gminW, gminW_eq j]
  simp

/-- `15 · gmin m ≥ 2^(m+2) + 7` for every `m ≥ 1`, by strong induction on the recurrence, so
`gmin m ≥ 2^(m-2)` (`gmin_ge`). The bound is tight at every `m ≡ 1 (mod 4)`, `m = 129` among them,
and the induction needs it: on the plain `2^(m-2)`, the recurrence's second branch would only give
`2^(m-2) - 7`. -/
private theorem gmin_lower (m : ℕ) : 1 ≤ m → 2 ^ (m + 2) + 7 ≤ 15 * gmin m := by
  induction m using Nat.strong_induction_on with
  | _ m ih =>
    intro hm
    by_cases h4 : m ≤ 4
    · obtain rfl | rfl | rfl | rfl : m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4 := by omega
      all_goals decide
    · rw [gmin_rec m (by omega)]
      have a := ih (m - 1) (by omega) (by omega)
      have b := ih (m - 4) (by omega) (by omega)
      have e1 : 2 ^ (m - 1 + 2) = 2 * 2 ^ (m - 2) * 4 := by
        rw [show m - 1 + 2 = (m - 2) + 1 + 2 by omega]; ring
      have e2 : 2 ^ (m - 4 + 2) = 2 ^ (m - 2) := by
        rw [show m - 4 + 2 = m - 2 by omega]
      have e3 : 2 ^ (m + 2) = 2 ^ (m - 2) * 16 := by
        rw [show m + 2 = (m - 2) + 4 by omega, pow_add]; norm_num
      rw [e1] at a; rw [e2] at b; rw [e3]
      omega

/-- From an odd `k` below `2^128`, either the next pass reaches `0` (one pass in all), or
`k = 16 q + d` with `q = kq k ≥ 1`, and four passes reach `q`. -/
private theorem passes_odd (k : ℕ) (hk : k < 2 ^ 128) (hodd : k % 2 = 1) :
    (kq k = 0 ∧ passes k = 1) ∨
      (kq k ≠ 0 ∧ (k : ℤ) = 16 * kq k + wdigit k ∧ passes k = passes (kq k) + 4) := by
  have hp : (2 : ℤ) ^ 128 = 340282366920938463463374607431768211456 := by norm_num
  have hk0 : k ≠ 0 := by omega
  have hnext : knext k = wrapU128 ((k : ℤ) - wdigit k) / 2 := by simp [knext, hodd]
  by_cases hq : kq k = 0
  · left
    refine ⟨hq, ?_⟩
    have : knext k = 0 := by
      rw [hnext]; unfold kq at hq; unfold wrapU128 wdigit at *; split_ifs at * <;> omega
    rw [passes_of_ne k hk0, this, passes_zero]
  · right
    have hval : (k : ℤ) = 16 * kq k + wdigit k := by
      unfold kq at *; unfold wrapU128 wdigit at *; split_ifs at * <;> omega
    have h8 : knext k = 8 * kq k := by
      rw [hnext]; unfold kq at *; unfold wrapU128 wdigit at *; split_ifs at * <;> omega
    have hev : ∀ x, x ≠ 0 → knext (2 * x) = x := by
      intro x hx; simp only [knext]; split_ifs <;> omega
    refine ⟨hq, hval, ?_⟩
    rw [passes_of_ne k hk0, h8, show 8 * kq k = 2 * (4 * kq k) by ring,
      passes_of_ne _ (by omega), hev _ (by omega), show 4 * kq k = 2 * (2 * kq k) by ring,
      passes_of_ne _ (by omega), hev _ (by omega), passes_of_ne _ (by omega), hev _ hq]

/-- A magnitude whose loop makes at least `m` passes is at least `gmin m`. -/
private theorem gmin_le (k : ℕ) : k < 2 ^ 128 → ∀ m, m ≤ passes k → gmin m ≤ k := by
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    intro hk m hm
    have g1 : gmin 1 = 1 := by decide
    have g4 : ∀ m, m ≤ 4 → gmin m ≤ 8 := by
      intro m h
      obtain rfl | rfl | rfl | rfl | rfl : m = 0 ∨ m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4 := by omega
      all_goals decide
    by_cases hk0 : k = 0
    · subst hk0; simp at hm; subst hm; decide
    by_cases hm0 : m = 0
    · subst hm0; simp [gmin]
    rcases Nat.mod_two_eq_zero_or_one k with he | ho
    · -- An even `k`: one pass halves it.
      have hn : knext k = k / 2 := by simp [knext, he]
      rw [passes_of_ne k hk0, hn] at hm
      by_cases hm1 : m = 1
      · subst hm1; rw [g1]; omega
      have h := ih (k / 2) (by omega) (by omega) (m - 1) (by omega)
      have hdbl : gmin m ≤ 2 * gmin (m - 1) := by
        by_cases h5 : 5 ≤ m
        · rw [gmin_rec m h5]; exact min_le_left _ _
        · obtain rfl | rfl | rfl : m = 2 ∨ m = 3 ∨ m = 4 := by omega
          all_goals decide
      omega
    · rcases passes_odd k hk ho with ⟨hq, h1⟩ | ⟨hq, hv, hpq⟩
      · rw [h1] at hm
        have : m = 1 := by omega
        subst this; rw [g1]; omega
      · have hdb : -7 ≤ wdigit k := by unfold wdigit; split_ifs <;> omega
        by_cases h5 : 5 ≤ m
        · have h := ih (kq k) (by omega) (by omega) (m - 4) (by omega)
          rw [gmin_rec m h5]
          have := min_le_right (2 * gmin (m - 1)) (16 * gmin (m - 4) - 7)
          omega
        · have := g4 m (by omega)
          omega

/-- `leastFailing` lies between `2^127` and `2^128 - 16`. -/
private theorem leastFailing_bounds : 2 ^ 127 < leastFailing ∧ leastFailing < 2 ^ 128 - 16 := by
  unfold leastFailing; norm_num

/-- `gmin m ≥ 2^(m-2)`, from `gmin_lower`'s sharper bound. -/
private theorem gmin_ge (m : ℕ) (hm : 2 ≤ m) : 2 ^ (m - 2) ≤ gmin m := by
  have h := gmin_lower m (by omega)
  have e : 2 ^ (m + 2) = 16 * 2 ^ (m - 2) := by
    rw [show m + 2 = (m - 2) + 4 by omega, pow_add]; ring
  omega

/-- Below `leastFailing` the loop makes at most 128 passes. -/
private theorem passes_le_128 (a : ℕ) (ha : a < leastFailing) : passes a ≤ 128 := by
  have hb := leastFailing_bounds
  by_contra h
  have h129 := gmin_le a (by omega) 129 (by omega)
  rw [gmin_129] at h129
  omega

/-- On a `u128` the loop makes at most `⌊log₂ a⌋ + 2` passes. -/
private theorem passes_le_log (a : ℕ) (ha : a < 2 ^ 128) : passes a ≤ Nat.log2 a + 2 := by
  by_cases hm : passes a ≤ 1
  · omega
  · have h := (gmin_ge (passes a) (by omega)).trans (gmin_le a ha (passes a) le_rfl)
    have h1 : 1 ≤ 2 ^ (passes a - 2) := Nat.one_le_two_pow
    have h2 := (Nat.le_log2 (by omega)).2 h
    omega

/-- `exec` returns the number of passes as `n`. -/
private theorem exec_n (a : ℕ) (neg : Bool) : (exec a neg).n = passes a := by
  show (Wnaf.loop neg a (Wnaf.init a)).n = passes a
  rw [loop_n neg a (Wnaf.init a) le_rfl]
  simp [Wnaf.init]

set_option maxRecDepth 100000 in
/-- The `u128` flag on the last sixteen magnitudes, evaluated by the kernel. -/
private theorem exec_top16 : ∀ (i : Fin 16) (neg : Bool),
    (Wnaf.exec (2 ^ 128 - 16 + i) neg).u128Overflow =
      decide ((i : ℕ) = 9 ∨ (i : ℕ) = 11 ∨ (i : ℕ) = 13 ∨ (i : ℕ) = 15) := by
  decide +kernel

end Wnaf

/-! ## Length -/

/-- `wnaf_digits(a, negate)` on a `u128` magnitude `a` makes at most `⌊log₂ a⌋ + 2` passes of its
loop, the count it returns as `n`. The bound is on the model's passes, which run on through every
failed check: a debug build panics from `2^127` on, at its assertion, and a build that lacks debug
assertions at `leastFailing` and at some magnitudes above it (`wnaf_digits_out_of_bounds_above`),
at the write past the array. -/
theorem wnaf_digits_length_le_log2 (a : ℕ) (neg : Bool) (ha : a < 2 ^ 128) :
    (Wnaf.exec a neg).n ≤ Nat.log2 a + 2 := by
  rw [Wnaf.exec_n]; exact Wnaf.passes_le_log a ha

/-- `wnaf_digits(a, negate)` on a magnitude below `2^127` returns at most 128 digits, the
`MAX_WNAF_DIGITS` that its array holds. -/
theorem wnaf_digits_length_le_128 (a : ℕ) (neg : Bool) (ha : a < 2 ^ 127) :
    (Wnaf.exec a neg).n ≤ 128 := by
  have hb := Wnaf.leastFailing_bounds
  rw [Wnaf.exec_n]; exact Wnaf.passes_le_128 a (by omega)

/-! ## Remarks about the model: the least failing magnitude, and where each check fires

The facts at single magnitudes are checked by kernel evaluation (`decide +kernel`). -/

/-- Below `leastFailing` there are at most 128 digits, no write out of bounds and no overflow, so a
build without debug assertions returns, with or without overflow checks. Below `2^127` a debug
build returns as well (`wnaf_digits_no_panic`). -/
theorem wnaf_digits_below_least_failing (a : ℕ) (neg : Bool) (ha : a < leastFailing) :
    (Wnaf.exec a neg).n ≤ 128 ∧ (Wnaf.exec a neg).outOfBounds = false ∧
      (Wnaf.exec a neg).u128Overflow = false ∧ (Wnaf.exec a neg).i8Overflow = false := by
  have hb := Wnaf.leastFailing_bounds
  have hp := Wnaf.passes_le_128 a ha
  obtain ⟨-, hi8, hu⟩ := Wnaf.loop_flags neg a (Wnaf.init a)
  refine ⟨by rw [Wnaf.exec_n]; exact hp, ?_, hu (by show a < _; omega), hi8⟩
  exact Wnaf.loop_oob neg a (Wnaf.init a) le_rfl (by simpa [Wnaf.init] using hp)

set_option maxRecDepth 100000 in
/-- `leastFailing` gets 129 digits, the last written to `digits[128]`, which panics in every build.
With `wnaf_digits_below_least_failing`, it is the least magnitude that does. The length is not
monotone in the magnitude (`leastFailing + 1` gets 128 digits, `wnaf_digits_least_failing_succ`),
so this is no threshold. -/
theorem wnaf_digits_least_failing : ∀ neg : Bool,
    (Wnaf.exec leastFailing neg).n = 129 ∧ (Wnaf.exec leastFailing neg).outOfBounds = true := by
  decide +kernel

set_option maxRecDepth 100000 in
/-- The length is not monotone in the magnitude, since `leastFailing + 1` gets 128 digits. -/
theorem wnaf_digits_least_failing_succ : ∀ neg : Bool,
    (Wnaf.exec (leastFailing + 1) neg).n = 128 := by
  decide +kernel

set_option maxRecDepth 100000 in
/-- Above `leastFailing` some magnitudes still write past the array, `leastFailing + 2` for one. -/
theorem wnaf_digits_out_of_bounds_above : ∀ neg : Bool,
    (Wnaf.exec (leastFailing + 2) neg).outOfBounds = true := by
  decide +kernel

/-- On every `u128` magnitude, the assertion fails exactly from `2^127` on, no `i8` operation
overflows, and a `u128` operation overflows exactly on `2^128 - 7`, `2^128 - 5`, `2^128 - 3` and
`2^128 - 1`, in the first pass's `k += (-d) as u128`. A build without debug assertions or overflow
checks wraps it without a panic and returns the digits of `a - 2^128` (`wnaf_digits_wrap_value`);
with debug assertions, the assertion fails first. -/
theorem wnaf_digits_checks (a : ℕ) (neg : Bool) (ha : a < 2 ^ 128) :
    ((Wnaf.exec a neg).assertFailed = true ↔ 2 ^ 127 ≤ a) ∧ (Wnaf.exec a neg).i8Overflow = false ∧
      ((Wnaf.exec a neg).u128Overflow = true ↔
        a = 2 ^ 128 - 7 ∨ a = 2 ^ 128 - 5 ∨ a = 2 ^ 128 - 3 ∨ a = 2 ^ 128 - 1) := by
  obtain ⟨has, hi8, hu⟩ := Wnaf.loop_flags neg a (Wnaf.init a)
  refine ⟨?_, hi8, ?_⟩
  · show (Wnaf.loop neg a (Wnaf.init a)).assertFailed = true ↔ _
    rw [has]
    simp only [Wnaf.init, Nat.shiftRight_eq_div_pow, bne_iff_ne, ne_eq, Nat.div_eq_zero_iff]
    omega
  · by_cases hlo : a < 2 ^ 128 - 16
    · show (Wnaf.loop neg a (Wnaf.init a)).u128Overflow = true ↔ _
      rw [hu hlo]
      simp only [Wnaf.init, Bool.false_eq_true, false_iff]
      omega
    · obtain ⟨i, rfl⟩ : ∃ i : Fin 16, a = 2 ^ 128 - 16 + i :=
        ⟨⟨a - (2 ^ 128 - 16), by omega⟩, by dsimp only; omega⟩
      have hi := i.isLt
      rw [Wnaf.exec_top16 i neg, decide_eq_true_iff]
      omega

set_option maxRecDepth 100000 in
/-- On the four magnitudes where `k += (-d) as u128` overflows, the digits encode `a - 2^128`,
negated when `neg`: what a build without debug assertions or overflow checks returns there. -/
theorem wnaf_digits_wrap_value : ∀ (i : Fin 4) (neg : Bool),
    ∑ j : Fin 128, (Wnaf.exec (2 ^ 128 - 7 + 2 * i) neg).digits j * 2 ^ (j : ℕ) =
      if neg then 2 ^ 128 - ((2 ^ 128 - 7 + 2 * i : ℕ) : ℤ)
      else ((2 ^ 128 - 7 + 2 * i : ℕ) : ℤ) - 2 ^ 128 := by
  decide +kernel

/-! ## No panic, digit shape and value -/

/-- `wnaf_digits(a, negate)` on a magnitude below `2^127` panics in no build: the assertion holds,
no write is out of bounds, and no `u128` or `i8` operation overflows. -/
theorem wnaf_digits_no_panic (a : ℕ) (neg : Bool) (ha : a < 2 ^ 127) :
    (Wnaf.exec a neg).assertFailed = false ∧ (Wnaf.exec a neg).outOfBounds = false ∧
      (Wnaf.exec a neg).u128Overflow = false ∧ (Wnaf.exec a neg).i8Overflow = false := by
  have hb := Wnaf.leastFailing_bounds
  obtain ⟨-, ho, hu, hi⟩ := wnaf_digits_below_least_failing a neg (by omega)
  have has := (wnaf_digits_checks a neg (by omega)).1
  refine ⟨?_, ho, hu, hi⟩
  cases h : (Wnaf.exec a neg).assertFailed
  · rfl
  · have := has.1 h
    omega

/-- Every digit of `wnaf_digits(a, negate)` is a width-4 wNAF digit, so `|d| / 2` indexes a table
of four entries. This holds on every magnitude, a `u128` or not. -/
theorem wnaf_digits_shape (a : ℕ) (neg : Bool) (i : Fin 128) :
    IsWnafDigit ((Wnaf.exec a neg).digits i) ∧ ((Wnaf.exec a neg).digits i).natAbs / 2 < 4 := by
  have h := Wnaf.loop_shape neg a (Wnaf.init a) (fun _ => Or.inl rfl) i
  refine ⟨h, ?_⟩
  show ((Wnaf.loop neg a (Wnaf.init a)).digits i).natAbs / 2 < 4
  rcases h with h | ⟨-, h1, h2⟩
  · rw [h]; decide
  · omega

/-- The digits of `wnaf_digits(a, negate)` on a magnitude below `2^127` encode `a`, or `-a` when
`neg`, and they are zero from position `n` on. -/
theorem wnaf_digits_value (a : ℕ) (neg : Bool) (ha : a < 2 ^ 127) :
    ∑ i : Fin 128, (Wnaf.exec a neg).digits i * 2 ^ (i : ℕ) = (if neg then -(a : ℤ) else a) ∧
      ∀ i : Fin 128, (Wnaf.exec a neg).n ≤ i → (Wnaf.exec a neg).digits i = 0 := by
  have hb := Wnaf.leastFailing_bounds
  have hinit : Wnaf.ValInv a neg (Wnaf.init a) := by
    unfold Wnaf.ValInv
    refine ⟨ha, ?_, fun _ _ => rfl, ?_⟩
    · simpa [Wnaf.init] using Wnaf.passes_le_128 a (by omega)
    · simp [Wnaf.init]
  obtain ⟨-, -, hz, hsum⟩ := Wnaf.loop_value a neg a (Wnaf.init a) hinit
  have hk0 : (Wnaf.loop neg a (Wnaf.init a)).k = 0 := Wnaf.exec_k a neg
  rw [hk0] at hsum
  simp only [Nat.cast_zero, neg_zero, ite_self, zero_mul, add_zero] at hsum
  exact ⟨hsum, hz⟩

/-! ## `Decomposed::new` on two halves -/

/-- `Decomposed::new` after its call to `decompose`, on two halves whose magnitudes are below
`2^127`: it does not panic; the digit arrays are the halves' wNAF digits, `len` is the larger of
their lengths and at most 128, and both arrays are zero from `len` on. -/
theorem decomposed_of_halves_spec (neg1 neg2 : Bool) (a1 a2 : ℕ) (h1 : a1 < 2 ^ 127)
    (h2 : a2 < 2 ^ 127) :
    ∃ D, Decomposed.ofHalves (neg1, a1) (neg2, a2) = some D ∧
      D.digits1 = (Wnaf.run a1 neg1).1 ∧ D.digits2 = (Wnaf.run a2 neg2).1 ∧
      D.len = max (Wnaf.run a1 neg1).2 (Wnaf.run a2 neg2).2 ∧ D.len ≤ 128 ∧
      ∀ i : Fin 128, D.len ≤ i → D.digits1 i = 0 ∧ D.digits2 i = 0 := by
  obtain ⟨af1, o1, u1, i1⟩ := wnaf_digits_no_panic a1 neg1 h1
  obtain ⟨af2, o2, u2, i2⟩ := wnaf_digits_no_panic a2 neg2 h2
  have hz1 := (wnaf_digits_value a1 neg1 h1).2
  have hz2 := (wnaf_digits_value a2 neg2 h2).2
  refine ⟨⟨(Wnaf.exec a1 neg1).digits, (Wnaf.exec a2 neg2).digits,
      max (Wnaf.exec a1 neg1).n (Wnaf.exec a2 neg2).n⟩, ?_, rfl, rfl, rfl,
    max_le (wnaf_digits_length_le_128 a1 neg1 h1) (wnaf_digits_length_le_128 a2 neg2 h2),
    fun i hi => ⟨hz1 i (le_of_max_le_left hi), hz2 i (le_of_max_le_right hi)⟩⟩
  simp [Decomposed.ofHalves, Wnaf.State.panics, Wnaf.State.panicsIn, af1, o1, u1, i1, af2, o2,
    u2, i2]

end PastaCurves.Glv
