import PastaCurves.Inversion.Divstep

/-!
# Divsteps on packed words

The inner loop of the inversion runs the divstep recurrence on two 64-bit words that carry the low
bits of `f` and `g` together with the two coefficients of a row of the transition matrix:
`w_f = f₀ - 2^41 · u - 2^62 · v`, and likewise `w_g` with the second row, starting from
`(u, v) = (1, 0)` and `(q, r) = (0, 1)` (§2 of `book/src/design/inversion.md`). The point of this
file is that the packed recurrence is the plain `divstep` on the packed state, so Lemma 6 is an
identity between `divsteps j` of the packed state and `divsteps j` of the true state, and Lemma 7
reads the matrix back out of the upper bits, which the half-open range of `M_entry_range` makes
unambiguous.
-/

namespace PastaCurves.Inversion

/-- The packed start for a state whose `f` and `g` are the low bits: `(1, 0)` and `(0, 1)` are
the identity matrix's rows, negated and shifted to bits 41 and 62. -/
def packedStart (s : State) : State := ⟨s.d, s.f - 2^41, s.g - 2^62⟩

/-- The initial packing: `(f mod 2^20) - 2^41` and `(g mod 2^20) - 2^62`. -/
def pack (f g : ℤ) : ℤ × ℤ := (f % 2^20 - 2^41, g % 2^20 - 2^62)

/-- `k` packed divsteps on the words `wf, wg` at `d`: the plain recurrence on the packed state,
returning the new `d` and the two words. -/
def packedDivsteps (k : ℕ) (d wf wg : ℤ) : ℤ × ℤ × ℤ :=
  let t := divsteps k ⟨d, wf, wg⟩
  (t.d, t.f, t.g)

/-- Lemma 7's decoder: the coefficient pair in the upper bits of a packed word after `k` steps,
with the first coefficient taken in `(-2^20, 2^20]` for every `k ≤ 20`. `unpack_spec` assumes
the narrower range `(-2^k, 2^k]`, which is what the entries of `M k s` satisfy
(`M_entry_range`). -/
def unpack (k : ℕ) (w : ℤ) : ℤ × ℤ :=
  let t := -w
  let up := (t + 2^(40 - k)) / 2^(41 - k)   -- = u + 2^21 v
  let v := (up + 2^20 - 1) / 2^21
  (up - 2^21 * v, v)

/-- Lemma 6: for `j ≤ 20`, `j` steps on the packed state are `j` steps on the true state with
the entries of `M j s` carried in the upper bits. The branch taken on the packed word is the true
branch, because the coefficient terms are even, and every halving is exact for the same reason. -/
theorem divsteps_packedStart (j : ℕ) (hj : j ≤ 20) (s : State) (hf : s.f % 2 = 1) :
    divsteps j (packedStart s) =
      ⟨(divsteps j s).d,
       (divsteps j s).f - 2^(41 - j) * (M j s).a - 2^(62 - j) * (M j s).b,
       (divsteps j s).g - 2^(41 - j) * (M j s).c - 2^(62 - j) * (M j s).d⟩ := by
  induction j with
  | zero => simp [packedStart, M, Mat2.one]
  | succ j ih =>
    rw [divsteps_succ', divsteps_succ', M_succ_left, ih (by omega)]
    set t := divsteps j s with ht
    set N := M j s with hN
    have hodd : t.f % 2 = 1 := divsteps_f_odd j s hf
    have e1 : (2 : ℤ)^(41 - j) = 2 * 2^(40 - j) := by
      rw [← pow_succ']; congr 1; omega
    have e2 : (2 : ℤ)^(62 - j) = 2 * 2^(61 - j) := by
      rw [← pow_succ']; congr 1; omega
    have e3 : 41 - (j + 1) = 40 - j := by omega
    have e4 : 62 - (j + 1) = 61 - j := by omega
    rw [e1, e2, e3, e4]
    set X : ℤ := 2^(40 - j) with hX
    set Y : ℤ := 2^(61 - j) with hY
    -- The packed word's parity is the true word's parity.
    have hpar : (t.g - 2 * X * N.c - 2 * Y * N.d) % 2 = t.g % 2 := by
      rw [mul_assoc, mul_assoc]; omega
    by_cases h : 0 < t.d ∧ t.g % 2 = 1
    · have h' : 0 < t.d ∧ (t.g - 2 * X * N.c - 2 * Y * N.d) % 2 = 1 := by rw [hpar]; exact h
      simp only [divstep, T, if_pos h, if_pos h', Mat2.mul]
      obtain ⟨G, hG⟩ : ∃ G, t.g - t.f = 2 * G := ⟨(t.g - t.f) / 2, by omega⟩
      have hdiv : (t.g - t.f) / 2 = G := by omega
      have hpk : t.g - 2 * X * N.c - 2 * Y * N.d - (t.f - 2 * X * N.a - 2 * Y * N.b) =
          2 * (G - X * (N.c - N.a) - Y * (N.d - N.b)) := by linear_combination hG
      rw [hpk, Int.mul_ediv_cancel_left _ two_ne_zero, hdiv]
      simp only [State.mk.injEq]
      refine ⟨trivial, ?_, ?_⟩ <;> ring
    · simp only [divstep, T, if_neg h, Mat2.mul, hpar]
      rcases Int.emod_two_eq_zero_or_one t.g with h2 | h2
      · rw [h2]
        obtain ⟨G, hG⟩ : ∃ G, t.g = 2 * G := ⟨t.g / 2, by omega⟩
        have hdiv : (t.g + 0 * t.f) / 2 = G := by omega
        have hpk : t.g - 2 * X * N.c - 2 * Y * N.d + 0 * (t.f - 2 * X * N.a - 2 * Y * N.b) =
            2 * (G - X * N.c - Y * N.d) := by linear_combination hG
        rw [hpk, Int.mul_ediv_cancel_left _ two_ne_zero, hdiv]
        simp only [State.mk.injEq]
        refine ⟨trivial, ?_, ?_⟩ <;> ring
      · rw [h2]
        obtain ⟨G, hG⟩ : ∃ G, t.g + t.f = 2 * G := ⟨(t.g + t.f) / 2, by omega⟩
        have hdiv : (t.g + 1 * t.f) / 2 = G := by omega
        have hpk : t.g - 2 * X * N.c - 2 * Y * N.d + 1 * (t.f - 2 * X * N.a - 2 * Y * N.b) =
            2 * (G - X * (N.c + N.a) - Y * (N.d + N.b)) := by linear_combination hG
        rw [hpk, Int.mul_ediv_cancel_left _ two_ne_zero, hdiv]
        simp only [State.mk.injEq]
        refine ⟨trivial, ?_, ?_⟩ <;> ring

/-- Lemma 6, magnitudes: for `|f|, |g| < 2^20` the packed words stay below `2^63` in absolute
value throughout a batch, so they fit a signed 64-bit word without wrapping. -/
theorem divsteps_packedStart_abs_lt (j : ℕ) (hj : j ≤ 20) (s : State) (hf : s.f % 2 = 1)
    (hsf : |s.f| < 2^20) (hsg : |s.g| < 2^20) :
    |(divsteps j (packedStart s)).f| < 2^63 ∧ |(divsteps j (packedStart s)).g| < 2^63 := by
  rw [divsteps_packedStart j hj s hf]
  obtain ⟨hφ, hγ⟩ := divsteps_abs_le j s (2^20 - 1) (by omega) (by omega)
  obtain ⟨⟨ha1, ha2⟩, ⟨hb1, hb2⟩, ⟨hc1, hc2⟩, ⟨hd1, hd2⟩⟩ := M_entry_range j s
  have hA : (2 : ℤ)^(41 - j) * 2^j = 2^41 := by rw [← pow_add]; congr 1; omega
  have hB : (2 : ℤ)^(62 - j) * 2^j = 2^62 := by rw [← pow_add]; congr 1; omega
  have hpA : (0 : ℤ) < 2^(41 - j) := by positivity
  have hpB : (0 : ℤ) < 2^(62 - j) := by positivity
  have bound : ∀ (P x : ℤ) (n : ℕ), 0 < P → P * 2^j = 2^n → -(2 : ℤ)^j < x → x ≤ 2^j →
      |P * x| ≤ 2^n := by
    intro P x n hP hPn hx1 hx2
    rw [abs_mul, abs_of_pos hP, ← hPn]
    exact mul_le_mul_of_nonneg_left (abs_le.mpr ⟨by linarith, hx2⟩) hP.le
  have ha := bound _ _ _ hpA hA ha1 ha2
  have hb := bound _ _ _ hpB hB hb1 hb2
  have hc := bound _ _ _ hpA hA hc1 hc2
  have hd := bound _ _ _ hpB hB hd1 hd2
  have hsum : (2 : ℤ)^20 - 1 + 2^41 + 2^62 < 2^63 := by norm_num
  constructor
  · calc |(divsteps j s).f - 2^(41 - j) * (M j s).a - 2^(62 - j) * (M j s).b|
        ≤ |(divsteps j s).f - 2^(41 - j) * (M j s).a| + |2^(62 - j) * (M j s).b| := abs_sub _ _
      _ ≤ |(divsteps j s).f| + |2^(41 - j) * (M j s).a| + |2^(62 - j) * (M j s).b| := by
          linarith [abs_sub (divsteps j s).f (2^(41 - j) * (M j s).a)]
      _ < 2^63 := by linarith
  · calc |(divsteps j s).g - 2^(41 - j) * (M j s).c - 2^(62 - j) * (M j s).d|
        ≤ |(divsteps j s).g - 2^(41 - j) * (M j s).c| + |2^(62 - j) * (M j s).d| := abs_sub _ _
      _ ≤ |(divsteps j s).g| + |2^(41 - j) * (M j s).c| + |2^(62 - j) * (M j s).d| := by
          linarith [abs_sub (divsteps j s).g (2^(41 - j) * (M j s).c)]
      _ < 2^63 := by linarith

/-- Lemma 7: the decoder recovers `(u, v)` from `φ - 2^(41-k) u - 2^(62-k) v` when `|φ| < 2^20`,
`k ≤ 20`, and `u ∈ (-2^k, 2^k]`. -/
theorem unpack_spec (k : ℕ) (hk : k ≤ 20) (φ u v : ℤ) (hφ : |φ| < 2^20)
    (hu1 : -(2 : ℤ)^k < u) (hu2 : u ≤ 2^k) :
    unpack k (φ - 2^(41 - k) * u - 2^(62 - k) * v) = (u, v) := by
  set w := φ - 2^(41 - k) * u - 2^(62 - k) * v with hw
  have e1 : (2 : ℤ)^(41 - k) = 2 * 2^(40 - k) := by rw [← pow_succ']; congr 1; omega
  have e2 : (2 : ℤ)^(62 - k) = 2^21 * 2^(41 - k) := by rw [← pow_add]; congr 1; omega
  have hX : (2 : ℤ)^20 ≤ 2^(40 - k) := pow_le_pow_right₀ (by norm_num) (by omega)
  have hK : (2 : ℤ)^k ≤ 2^20 := pow_le_pow_right₀ (by norm_num) hk
  rw [abs_lt] at hφ
  have hup : (-w + 2^(40 - k)) / 2^(41 - k) = u + 2^21 * v := by
    rw [show -w + 2^(40 - k) = (2^(40 - k) - φ) + (u + 2^21 * v) * 2^(41 - k) by
      rw [hw, e2]; ring]
    rw [Int.add_mul_ediv_right _ _ (by positivity), Int.ediv_eq_zero_of_lt (by omega) (by omega),
      zero_add]
  have hv : (u + 2^21 * v + 2^20 - 1) / 2^21 = v := by
    rw [show u + 2^21 * v + 2^20 - 1 = (u + 2^20 - 1) + v * 2^21 by ring]
    rw [Int.add_mul_ediv_right _ _ (by norm_num), Int.ediv_eq_zero_of_lt (by omega) (by omega),
      zero_add]
  show ((-w + 2^(40 - k)) / 2^(41 - k) -
      2^21 * (((-w + 2^(40 - k)) / 2^(41 - k) + 2^20 - 1) / 2^21),
    ((-w + 2^(40 - k)) / 2^(41 - k) + 2^20 - 1) / 2^21) = (u, v)
  rw [hup, hv]
  simp

/-- Lemmas 6 and 7 with Lemma 2: the packed batch on the low 20 bits yields the true `d` and,
through the decoder, the true matrix. -/
theorem packedDivsteps_spec (k : ℕ) (hk : k ≤ 20) (s : State) (hf : s.f % 2 = 1) :
    (packedDivsteps k s.d (pack s.f s.g).1 (pack s.f s.g).2).1 = (divsteps k s).d ∧
      unpack k (packedDivsteps k s.d (pack s.f s.g).1 (pack s.f s.g).2).2.1
        = ((M k s).a, (M k s).b) ∧
      unpack k (packedDivsteps k s.d (pack s.f s.g).1 (pack s.f s.g).2).2.2
        = ((M k s).c, (M k s).d) := by
  -- The truncated state, whose low `k` bits agree with `s`.
  set s₀ : State := ⟨s.d, s.f % 2^20, s.g % 2^20⟩ with hs₀
  have h2 : (2 : ℤ) ∣ 2^20 := dvd_pow_self 2 (by norm_num)
  have hf₀ : s₀.f % 2 = 1 := by
    show (s.f % 2^20) % 2 = 1
    rw [Int.emod_emod_of_dvd _ h2]; exact hf
  have hk2 : (2 : ℤ)^k ∣ 2^20 := pow_dvd_pow 2 hk
  obtain ⟨hd, hM⟩ := divsteps_local k s s₀ hf rfl
    (by show s.f % 2^k = (s.f % 2^20) % 2^k; rw [Int.emod_emod_of_dvd _ hk2])
    (by show s.g % 2^k = (s.g % 2^20) % 2^k; rw [Int.emod_emod_of_dvd _ hk2])
  -- The packed run is the packed start of the truncated state.
  have hpk : packedDivsteps k s.d (pack s.f s.g).1 (pack s.f s.g).2 =
      ((divsteps k (packedStart s₀)).d, (divsteps k (packedStart s₀)).f,
        (divsteps k (packedStart s₀)).g) := rfl
  rw [hpk, divsteps_packedStart k hk s₀ hf₀, hd, hM]
  -- Lemma 3 on the truncated state: the true words stay below `2^20`.
  have hpos : (0 : ℤ) < 2^20 := by positivity
  have hbf : |s₀.f| ≤ 2^20 - 1 := by
    rw [abs_le]; constructor
    · have := Int.emod_nonneg s.f hpos.ne'; show -(2^20 - 1) ≤ s.f % 2^20; omega
    · have := Int.emod_lt_of_pos s.f hpos; show s.f % 2^20 ≤ 2^20 - 1; omega
  have hbg : |s₀.g| ≤ 2^20 - 1 := by
    rw [abs_le]; constructor
    · have := Int.emod_nonneg s.g hpos.ne'; show -(2^20 - 1) ≤ s.g % 2^20; omega
    · have := Int.emod_lt_of_pos s.g hpos; show s.g % 2^20 ≤ 2^20 - 1; omega
  obtain ⟨hφ, hγ⟩ := divsteps_abs_le k s₀ (2^20 - 1) hbf hbg
  obtain ⟨⟨ha1, ha2⟩, -, ⟨hc1, hc2⟩, -⟩ := M_entry_range k s₀
  refine ⟨rfl, ?_, ?_⟩
  · exact unpack_spec k hk _ _ _ (by omega) ha1 ha2
  · exact unpack_spec k hk _ _ _ (by omega) hc1 hc2

end PastaCurves.Inversion
