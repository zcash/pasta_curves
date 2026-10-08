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
def packedStart (s : State) : State := ⟨s.two_delta, s.f - 2^41, s.g - 2^62⟩

/-- The initial packing: `(f mod 2^20) - 2^41` and `(g mod 2^20) - 2^62`. -/
def pack (f g : ℤ) : ℤ × ℤ := (f % 2^20 - 2^41, g % 2^20 - 2^62)

/-- `k` packed divsteps on the words `wf, wg` at `two_delta`: the plain recurrence on the packed
state, returning the new `two_delta` and the two words. -/
def packedDivsteps (k : ℕ) (two_delta wf wg : ℤ) : ℤ × ℤ × ℤ :=
  let t := divsteps k ⟨two_delta, wf, wg⟩
  (t.two_delta, t.f, t.g)

/-- Lemma 7's decoder: the coefficient pair in the upper bits of a packed word after `k` steps, with
the first coefficient taken in `(-2^20, 2^20]` for every `k ≤ 20`. `unpack_spec` assumes the
narrower range `(-2^k, 2^k]`, which is what the entries of `M k s` satisfy (`M_entry_range`). -/
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
      ⟨(divsteps j s).two_delta,
       (divsteps j s).f - 2^(41 - j) * (M j s).u - 2^(62 - j) * (M j s).v,
       (divsteps j s).g - 2^(41 - j) * (M j s).q - 2^(62 - j) * (M j s).r⟩ := by
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
    have hpar : (t.g - 2 * X * N.q - 2 * Y * N.r) % 2 = t.g % 2 := by
      rw [mul_assoc, mul_assoc]; omega
    by_cases h : 0 < t.two_delta ∧ t.g % 2 = 1
    · have h' : 0 < t.two_delta ∧ (t.g - 2 * X * N.q - 2 * Y * N.r) % 2 = 1 := by rw [hpar]; exact h
      simp only [divstep, T, if_pos h, if_pos h', Mat2.mul]
      obtain ⟨G, hG⟩ : ∃ G, t.g - t.f = 2 * G := ⟨(t.g - t.f) / 2, by omega⟩
      have hdiv : (t.g - t.f) / 2 = G := by omega
      have hpk : t.g - 2 * X * N.q - 2 * Y * N.r - (t.f - 2 * X * N.u - 2 * Y * N.v) =
          2 * (G - X * (N.q - N.u) - Y * (N.r - N.v)) := by linear_combination hG
      rw [hpk, Int.mul_ediv_cancel_left _ two_ne_zero, hdiv]
      simp only [State.mk.injEq]
      refine ⟨trivial, ?_, ?_⟩ <;> ring
    · simp only [divstep, T, if_neg h, Mat2.mul, hpar]
      rcases Int.emod_two_eq_zero_or_one t.g with h2 | h2
      · rw [h2]
        obtain ⟨G, hG⟩ : ∃ G, t.g = 2 * G := ⟨t.g / 2, by omega⟩
        have hdiv : (t.g + 0 * t.f) / 2 = G := by omega
        have hpk : t.g - 2 * X * N.q - 2 * Y * N.r + 0 * (t.f - 2 * X * N.u - 2 * Y * N.v) =
            2 * (G - X * N.q - Y * N.r) := by linear_combination hG
        rw [hpk, Int.mul_ediv_cancel_left _ two_ne_zero, hdiv]
        simp only [State.mk.injEq]
        refine ⟨trivial, ?_, ?_⟩ <;> ring
      · rw [h2]
        obtain ⟨G, hG⟩ : ∃ G, t.g + t.f = 2 * G := ⟨(t.g + t.f) / 2, by omega⟩
        have hdiv : (t.g + 1 * t.f) / 2 = G := by omega
        have hpk : t.g - 2 * X * N.q - 2 * Y * N.r + 1 * (t.f - 2 * X * N.u - 2 * Y * N.v) =
            2 * (G - X * (N.q + N.u) - Y * (N.r + N.v)) := by linear_combination hG
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
  · calc |(divsteps j s).f - 2^(41 - j) * (M j s).u - 2^(62 - j) * (M j s).v|
        ≤ |(divsteps j s).f - 2^(41 - j) * (M j s).u| + |2^(62 - j) * (M j s).v| := abs_sub _ _
      _ ≤ |(divsteps j s).f| + |2^(41 - j) * (M j s).u| + |2^(62 - j) * (M j s).v| := by
          linarith [abs_sub (divsteps j s).f (2^(41 - j) * (M j s).u)]
      _ < 2^63 := by linarith
  · calc |(divsteps j s).g - 2^(41 - j) * (M j s).q - 2^(62 - j) * (M j s).r|
        ≤ |(divsteps j s).g - 2^(41 - j) * (M j s).q| + |2^(62 - j) * (M j s).r| := abs_sub _ _
      _ ≤ |(divsteps j s).g| + |2^(41 - j) * (M j s).q| + |2^(62 - j) * (M j s).r| := by
          linarith [abs_sub (divsteps j s).g (2^(41 - j) * (M j s).q)]
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

/-- Lemmas 6 and 7 with Lemma 2: the packed batch on the low 20 bits yields the true `two_delta`
and, through the decoder, the true matrix. -/
theorem packedDivsteps_spec (k : ℕ) (hk : k ≤ 20) (s : State) (hf : s.f % 2 = 1) :
    (packedDivsteps k s.two_delta (pack s.f s.g).1 (pack s.f s.g).2).1 = (divsteps k s).two_delta ∧
      unpack k (packedDivsteps k s.two_delta (pack s.f s.g).1 (pack s.f s.g).2).2.1
        = ((M k s).u, (M k s).v) ∧
      unpack k (packedDivsteps k s.two_delta (pack s.f s.g).1 (pack s.f s.g).2).2.2
        = ((M k s).q, (M k s).r) := by
  -- The truncated state, whose low `k` bits agree with `s`.
  set s₀ : State := ⟨s.two_delta, s.f % 2^20, s.g % 2^20⟩ with hs₀
  have h2 : (2 : ℤ) ∣ 2^20 := dvd_pow_self 2 (by norm_num)
  have hf₀ : s₀.f % 2 = 1 := by
    show (s.f % 2^20) % 2 = 1
    rw [Int.emod_emod_of_dvd _ h2]; exact hf
  have hk2 : (2 : ℤ)^k ∣ 2^20 := pow_dvd_pow 2 hk
  obtain ⟨hd, hM⟩ := divsteps_local k s s₀ hf rfl
    (by show s.f % 2^k = (s.f % 2^20) % 2^k; rw [Int.emod_emod_of_dvd _ hk2])
    (by show s.g % 2^k = (s.g % 2^20) % 2^k; rw [Int.emod_emod_of_dvd _ hk2])
  -- The packed run is the packed start of the truncated state.
  have hpk : packedDivsteps k s.two_delta (pack s.f s.g).1 (pack s.f s.g).2 =
      ((divsteps k (packedStart s₀)).two_delta, (divsteps k (packedStart s₀)).f,
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
  obtain ⟨⟨hu1, hu2⟩, -, ⟨hq1, hq2⟩, -⟩ := M_entry_range k s₀
  refine ⟨rfl, ?_, ?_⟩
  · exact unpack_spec k hk _ _ _ (by omega) hu1 hu2
  · exact unpack_spec k hk _ _ _ (by omega) hq1 hq2

/-- Lemma 6′: after every step of a batch the packed `g` is below `2^62` in magnitude, so the sum
`g ± f` that the next step halves, which is twice this `g`, fits a signed 64-bit word. The row-sum
bound alone gives only `2^62`: `2^(j+1) g_{j+1} = q f₀ + r g₀` with `|f₀| ≤ 2^41`, `|g₀| ≤ 2^62`,
and `|q| + |r| ≤ 2^(j+1)`, so the extreme needs `q = 0` and `r = 2^(j+1)`. That row arises only as
the sum of the two rows of the previous matrix with both second entries equal to `2^j`, and then the
determinant `2^j` of that matrix forces its first entries to differ by one, so `q` is odd. -/
theorem divsteps_packedStart_g_abs_lt (j : ℕ) (s : State) (hf : s.f % 2 = 1)
    (hsf : 0 ≤ s.f ∧ s.f < 2^20) (hsg : 0 ≤ s.g ∧ s.g < 2^20) :
    |(divsteps (j + 1) (packedStart s)).g| < 2^62 := by
  set P := packedStart s with hP
  have hPf : P.f % 2 = 1 := by show (s.f - 2^41) % 2 = 1; omega
  have hPfb : |P.f| ≤ 2^41 := by show |s.f - 2^41| ≤ 2^41; rw [abs_le]; constructor <;> omega
  have hPgb : |P.g| ≤ 2^62 := by show |s.g - 2^62| ≤ 2^62; rw [abs_le]; constructor <;> omega
  obtain ⟨-, hg⟩ := M_spec (j + 1) P hPf
  have hrow := (M_rowSum_le (j + 1) P).2
  set q' := (M (j + 1) P).q with hq'
  set r' := (M (j + 1) P).r with hr'
  -- The row in terms of the previous matrix `N` and the step matrix.
  obtain ⟨⟨hu1, hu2⟩, ⟨hv1, hv2⟩, ⟨hq1, hq2⟩, ⟨hr1, hr2⟩⟩ := M_entry_range j P
  have hdet := M_det j P
  simp only [Mat2.det] at hdet
  set N := M j P with hN
  set Z : ℤ := 2^j with hZ
  have hZpos : 0 < Z := by positivity
  have hstep : q' = (T (divsteps j P)).q * N.u + (T (divsteps j P)).r * N.q ∧
      r' = (T (divsteps j P)).q * N.v + (T (divsteps j P)).r * N.r := by
    rw [hq', hr', M_succ_left]; exact ⟨rfl, rfl⟩
  -- Either `|r'| < 2^(j+1)`, or `q'` is odd.
  have hdisj : |r'| ≤ 2 * Z - 1 ∨ 1 ≤ |q'| := by
    obtain ⟨hq'', hr''⟩ := hstep
    by_cases h : 0 < (divsteps j P).two_delta ∧ (divsteps j P).g % 2 = 1
    · simp only [T, if_pos h] at hq'' hr''
      left; rw [hr'', abs_le]; constructor <;> omega
    · simp only [T, if_neg h] at hq'' hr''
      rcases Int.emod_two_eq_zero_or_one (divsteps j P).g with h2 | h2
      · rw [h2] at hr''; left; rw [hr'', abs_le]; constructor <;> omega
      · rw [h2] at hq'' hr''
        by_cases hv : N.v = Z ∧ N.r = Z
        · right
          obtain ⟨hv1', hr1'⟩ := hv
          rw [hv1', hr1'] at hdet
          have huq : N.u - N.q = 1 := by
            have h3 : Z * (N.u - N.q) = Z * 1 := by linear_combination hdet
            exact mul_left_cancel₀ hZpos.ne' h3
          have hodd : q' ≠ 0 := by rw [hq'']; omega
          exact Int.one_le_abs hodd
        · left; rw [hr'', abs_le]; constructor <;> omega
  -- The bound on the row combination.
  have hY : (2 : ℤ)^(j + 1) = 2 * Z := by rw [hZ, pow_succ]; ring
  rw [hY] at hg hrow
  have hq0 := abs_nonneg q'
  have hr0 := abs_nonneg r'
  have hcomb : |q' * P.f + r' * P.g| ≤ |q'| * 2^41 + |r'| * 2^62 := by
    calc |q' * P.f + r' * P.g| ≤ |q' * P.f| + |r' * P.g| := abs_add_le _ _
      _ = |q'| * |P.f| + |r'| * |P.g| := by rw [abs_mul, abs_mul]
      _ ≤ |q'| * 2^41 + |r'| * 2^62 :=
          add_le_add (mul_le_mul_of_nonneg_left hPfb hq0) (mul_le_mul_of_nonneg_left hPgb hr0)
  have hlt : |q'| * 2^41 + |r'| * 2^62 < 2^62 * (2 * Z) := by
    rcases hdisj with h | h <;> omega
  have h2g : |2 * Z * (divsteps (j + 1) P).g| < 2^62 * (2 * Z) := by
    rw [hg]; exact lt_of_le_of_lt hcomb hlt
  rw [abs_mul, abs_of_pos (by positivity : (0 : ℤ) < 2 * Z), mul_comm] at h2g
  exact lt_of_mul_lt_mul_right h2g (by positivity)

end PastaCurves.Inversion
