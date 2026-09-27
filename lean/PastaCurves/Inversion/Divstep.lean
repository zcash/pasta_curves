import Mathlib.Algebra.Order.Ring.Abs
import Mathlib.Data.Int.GCD
import Mathlib.Data.Int.ModEq
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination

/-!
# Half-delta divsteps on integers

The step function of the constant-time inversion and its transition matrices, with the facts
that hold for every number of steps: the matrix identity, locality in the low bits, the row-sum
bound, the invariance of `max (|f|, |g|)` and of the gcd, and the behaviour once `g = 0`.
Nothing here is specific to a modulus or to a word size.

The state is `(two_delta, f, g)` with `two_delta = 2δ` an odd integer (the start `two_delta = 1` is
`δ = 1/2`), `f` odd, and `g` any integer. One divstep is

    (2 - two_delta, g, (g - f) / 2)            if 0 < two_delta and g is odd,
    (2 + two_delta, f, (g + (g mod 2) f) / 2)  otherwise,

Bernstein and Yang's divstep with the half-delta start of Bernstein, Chen, Harrison, Huang,
Maxwell, Wang, Wuille, and Yang, "Accelerating and verifying constant-time modular inversion"
(EUROCRYPT 2026). The paper writes the step as a rational matrix `T(δ, f, g)` and the `n`-step
map as the product `T_{n-1} ⋯ T_0` with entries `u_n, v_n, q_n, r_n`; here `T` is twice the
paper's step matrix, so that it is integral, and `M n s = 2^n · T_{n-1} ⋯ T_0`.

The lemma numbers in the docstrings are those of the pen-and-paper argument in the book,
`book/src/design/inversion.md`.
-/

namespace PastaCurves.Inversion

/-- The divstep state. -/
structure State where
  two_delta : ℤ
  f : ℤ
  g : ℤ

/-- One half-delta divstep. Both divisions are exact when `f` is odd. -/
def divstep (s : State) : State :=
  if 0 < s.two_delta ∧ s.g % 2 = 1 then ⟨2 - s.two_delta, s.g, (s.g - s.f) / 2⟩
  else ⟨2 + s.two_delta, s.f, (s.g + (s.g % 2) * s.f) / 2⟩

/-- `n` divsteps. -/
def divsteps (n : ℕ) (s : State) : State := divstep^[n] s

@[simp] theorem divsteps_zero (s : State) : divsteps 0 s = s := rfl

theorem divsteps_succ (n : ℕ) (s : State) : divsteps (n + 1) s = divsteps n (divstep s) :=
  Function.iterate_succ_apply divstep n s

/-- A `2×2` integer matrix `[[u, v], [q, r]]`. -/
structure Mat2 where
  u : ℤ
  v : ℤ
  q : ℤ
  r : ℤ
  deriving DecidableEq

def Mat2.one : Mat2 := ⟨1, 0, 0, 1⟩

def Mat2.mul (m n : Mat2) : Mat2 :=
  ⟨m.u * n.u + m.v * n.q, m.u * n.v + m.v * n.r, m.q * n.u + m.r * n.q, m.q * n.v + m.r * n.r⟩

/-- The step matrix: `2 · (f', g')ᵀ = T s · (f, g)ᵀ`. -/
def T (s : State) : Mat2 :=
  if 0 < s.two_delta ∧ s.g % 2 = 1 then ⟨0, 2, -1, 1⟩ else ⟨2, 0, s.g % 2, 1⟩

/-- The `n`-step matrix `M n s = T (s_{n-1}) ⋯ T (s_0)`, so `2^n · (f_n, g_n)ᵀ = M n s · (f, g)ᵀ`. -/
def M : ℕ → State → Mat2
  | 0, _ => Mat2.one
  | n + 1, s => (M n (divstep s)).mul (T s)

theorem M_succ (n : ℕ) (s : State) : M (n + 1) s = (M n (divstep s)).mul (T s) := rfl

theorem divstep_f_odd (s : State) (hf : s.f % 2 = 1) : (divstep s).f % 2 = 1 := by
  unfold divstep
  split_ifs with h
  · exact h.2
  · exact hf

/-- One step: `2 f' = T.u f + T.v g` and `2 g' = T.q f + T.r g`. -/
theorem divstep_eq (s : State) (hf : s.f % 2 = 1) :
    2 * (divstep s).f = (T s).u * s.f + (T s).v * s.g ∧
      2 * (divstep s).g = (T s).q * s.f + (T s).r * s.g := by
  by_cases h : 0 < s.two_delta ∧ s.g % 2 = 1
  · simp only [divstep, T, if_pos h]
    constructor <;> omega
  · simp only [divstep, T, if_neg h]
    rcases Int.emod_two_eq_zero_or_one s.g with h2 | h2 <;>
      simp only [h2, zero_mul, one_mul, add_zero] <;> constructor <;> first | trivial | omega

theorem divsteps_f_odd (n : ℕ) (s : State) (hf : s.f % 2 = 1) : (divsteps n s).f % 2 = 1 := by
  induction n generalizing s with
  | zero => simpa
  | succ n ih => rw [divsteps_succ]; exact ih _ (divstep_f_odd s hf)

/-- Lemma 1: `2^n (f_n, g_n) = M_n (f_0, g_0)`. -/
theorem M_spec (n : ℕ) (s : State) (hf : s.f % 2 = 1) :
    (2 : ℤ)^n * (divsteps n s).f = (M n s).u * s.f + (M n s).v * s.g ∧
      (2 : ℤ)^n * (divsteps n s).g = (M n s).q * s.f + (M n s).r * s.g := by
  induction n generalizing s with
  | zero => simp [M, Mat2.one]
  | succ n ih =>
    rw [divsteps_succ]
    obtain ⟨ihf, ihg⟩ := ih (divstep s) (divstep_f_odd s hf)
    obtain ⟨hsf, hsg⟩ := divstep_eq s hf
    simp only [M_succ, Mat2.mul, pow_succ]
    constructor
    · linear_combination 2 * ihf + (M n (divstep s)).u * hsf + (M n (divstep s)).v * hsg
    · linear_combination 2 * ihg + (M n (divstep s)).q * hsf + (M n (divstep s)).r * hsg

/-- One step never increases `max (|f|, |g|)`. -/
theorem divstep_abs_le (s : State) (B : ℤ) (hf : |s.f| ≤ B) (hg : |s.g| ≤ B) :
    |(divstep s).f| ≤ B ∧ |(divstep s).g| ≤ B := by
  rw [abs_le] at hf hg
  by_cases h : 0 < s.two_delta ∧ s.g % 2 = 1
  · simp only [divstep, if_pos h]
    constructor <;> rw [abs_le] <;> constructor <;> omega
  · simp only [divstep, if_neg h]
    rcases Int.emod_two_eq_zero_or_one s.g with h2 | h2 <;>
      simp only [h2, zero_mul, one_mul, add_zero] <;> constructor <;> rw [abs_le] <;>
      constructor <;> omega

/-- Lemma 3: `max (|f|, |g|)` never grows. -/
theorem divsteps_abs_le (n : ℕ) (s : State) (B : ℤ) (hf : |s.f| ≤ B) (hg : |s.g| ≤ B) :
    |(divsteps n s).f| ≤ B ∧ |(divsteps n s).g| ≤ B := by
  induction n generalizing s with
  | zero => exact ⟨hf, hg⟩
  | succ n ih =>
    rw [divsteps_succ]
    obtain ⟨hf', hg'⟩ := divstep_abs_le s B hf hg
    exact ih _ hf' hg'

/-- Lemma 4: from `g = 0` nothing moves but `two_delta`, and the matrix is `[[2^n, 0], [0, 1]]`. -/
theorem divsteps_of_g_zero (n : ℕ) (s : State) (hg : s.g = 0) :
    (divsteps n s).f = s.f ∧ (divsteps n s).g = 0 ∧ M n s = ⟨2^n, 0, 0, 1⟩ := by
  induction n generalizing s with
  | zero => exact ⟨rfl, hg, rfl⟩
  | succ n ih =>
    have hstep : divstep s = ⟨2 + s.two_delta, s.f, 0⟩ := by
      simp [divstep, hg]
    have hT : T s = ⟨2, 0, 0, 1⟩ := by
      simp [T, hg]
    obtain ⟨h1, h2, h3⟩ := ih ⟨2 + s.two_delta, s.f, 0⟩ rfl
    refine ⟨?_, ?_, ?_⟩
    · rw [divsteps_succ, hstep]; exact h1
    · rw [divsteps_succ, hstep]; exact h2
    · rw [M_succ, hstep, hT, h3]
      simp only [Mat2.mul, Mat2.mk.injEq]
      refine ⟨?_, ?_, ?_, ?_⟩ <;> ring

/-- Lemma 3, row sums: `|u_n| + |v_n| ≤ 2^n` and `|q_n| + |r_n| ≤ 2^n`. -/
theorem M_rowSum_le (n : ℕ) (s : State) :
    |(M n s).u| + |(M n s).v| ≤ 2^n ∧ |(M n s).q| + |(M n s).r| ≤ 2^n := by
  induction n generalizing s with
  | zero => simp [M, Mat2.one]
  | succ n ih =>
    obtain ⟨ih1, ih2⟩ := ih (divstep s)
    rw [M_succ]
    set N := M n (divstep s) with hN
    have hu := abs_nonneg N.u
    have hv := abs_nonneg N.v
    have hq := abs_nonneg N.q
    have hr := abs_nonneg N.r
    by_cases h : 0 < s.two_delta ∧ s.g % 2 = 1
    · simp only [T, if_pos h, Mat2.mul, mul_zero, zero_add, mul_neg, mul_one, abs_neg, pow_succ]
      have h1 := abs_add_le (N.u * 2) N.v
      have h2 := abs_add_le (N.q * 2) N.r
      rw [abs_mul, abs_two] at h1 h2
      constructor <;> linarith
    · simp only [T, if_neg h, Mat2.mul, mul_zero, zero_add, mul_one, pow_succ]
      have hbit : |s.g % 2| ≤ 1 := by rw [abs_le]; omega
      have h1 := abs_add_le (N.u * 2) (N.v * (s.g % 2))
      have h2 := abs_add_le (N.q * 2) (N.r * (s.g % 2))
      rw [abs_mul, abs_mul, abs_two] at h1 h2
      have h3 : |N.v| * |s.g % 2| ≤ |N.v| := mul_le_of_le_one_right hv hbit
      have h4 : |N.r| * |s.g % 2| ≤ |N.r| := mul_le_of_le_one_right hr hbit
      constructor <;> linarith

/-! ## Locality: `n` steps see only the low `n` bits -/

/-- Halving preserves a congruence between even numbers, at half the modulus. -/
theorem half_emod {a b m : ℤ} (ha : 2 ∣ a) (hb : 2 ∣ b) (h : a % (2 * m) = b % (2 * m)) :
    (a / 2) % m = (b / 2) % m := by
  obtain ⟨a', rfl⟩ := ha
  obtain ⟨b', rfl⟩ := hb
  rw [Int.mul_emod_mul_of_pos _ _ (by norm_num), Int.mul_emod_mul_of_pos _ _ (by norm_num)] at h
  rw [Int.mul_ediv_cancel_left _ (by norm_num), Int.mul_ediv_cancel_left _ (by norm_num)]
  omega

theorem emod_two_of_emod_pow {a b : ℤ} (n : ℕ) (h : a % 2^(n + 1) = b % 2^(n + 1)) :
    a % 2 = b % 2 := by
  have h2 : (2 : ℤ) ∣ 2^(n + 1) := dvd_pow_self 2 (Nat.succ_ne_zero n)
  rw [← Int.emod_emod_of_dvd a h2, h, Int.emod_emod_of_dvd b h2]

theorem emod_pow_of_emod_pow_succ {a b : ℤ} (n : ℕ) (h : a % 2^(n + 1) = b % 2^(n + 1)) :
    a % 2^n = b % 2^n := by
  have h2 : (2 : ℤ)^n ∣ 2^(n + 1) := pow_dvd_pow 2 (Nat.le_succ n)
  rw [← Int.emod_emod_of_dvd a h2, h, Int.emod_emod_of_dvd b h2]

/-- One step of Lemma 2: states congruent modulo `2^(n+1)` take the same branch and stay
congruent modulo `2^n`. -/
theorem divstep_local (n : ℕ) (s s' : State) (hf0 : s.f % 2 = 1) (hd : s.two_delta = s'.two_delta)
    (hf : s.f % 2^(n + 1) = s'.f % 2^(n + 1)) (hg : s.g % 2^(n + 1) = s'.g % 2^(n + 1)) :
    T s = T s' ∧ (divstep s).two_delta = (divstep s').two_delta ∧
      (divstep s).f % 2^n = (divstep s').f % 2^n ∧ (divstep s).g % 2^n = (divstep s').g % 2^n := by
  have hf2 : s.f % 2 = s'.f % 2 := emod_two_of_emod_pow n hf
  have hg2 : s.g % 2 = s'.g % 2 := emod_two_of_emod_pow n hg
  have hpow : (2 : ℤ)^(n + 1) = 2 * 2^n := by ring
  rw [hpow] at hf hg
  have hfn : s.f % 2^n = s'.f % 2^n := by
    have h2 : (2 : ℤ)^n ∣ 2 * 2^n := dvd_mul_left _ _
    rw [← Int.emod_emod_of_dvd s.f h2, hf, Int.emod_emod_of_dvd s'.f h2]
  have hgn : s.g % 2^n = s'.g % 2^n := by
    have h2 : (2 : ℤ)^n ∣ 2 * 2^n := dvd_mul_left _ _
    rw [← Int.emod_emod_of_dvd s.g h2, hg, Int.emod_emod_of_dvd s'.g h2]
  by_cases h : 0 < s.two_delta ∧ s.g % 2 = 1
  · have h' : 0 < s'.two_delta ∧ s'.g % 2 = 1 := by rw [← hd, ← hg2]; exact h
    have hT : T s = T s' := by simp only [T, if_pos h, if_pos h']
    refine ⟨hT, ?_, ?_, ?_⟩ <;> simp only [divstep, if_pos h, if_pos h']
    · rw [hd]
    · exact hgn
    have hsub : (s.g - s.f) % (2 * 2^n) = (s'.g - s'.f) % (2 * 2^n) := Int.ModEq.sub hg hf
    exact half_emod (Int.dvd_of_emod_eq_zero (by omega)) (Int.dvd_of_emod_eq_zero (by omega)) hsub
  · have h' : ¬ (0 < s'.two_delta ∧ s'.g % 2 = 1) := by rw [← hd, ← hg2]; exact h
    have hT : T s = T s' := by simp only [T, if_neg h, if_neg h']; rw [hg2]
    refine ⟨hT, ?_, ?_, ?_⟩ <;> simp only [divstep, if_neg h, if_neg h']
    · rw [hd]
    · exact hfn
    rw [← hg2]
    have hadd : (s.g + s.g % 2 * s.f) % (2 * 2^n) = (s'.g + s.g % 2 * s'.f) % (2 * 2^n) :=
      Int.ModEq.add hg (Int.ModEq.mul_left _ hf)
    rcases Int.emod_two_eq_zero_or_one s.g with h2 | h2
    · rw [h2] at hadd ⊢
      simp only [zero_mul, add_zero] at hadd ⊢
      exact half_emod (Int.dvd_of_emod_eq_zero h2) (Int.dvd_of_emod_eq_zero (by omega)) hadd
    · rw [h2] at hadd ⊢
      simp only [one_mul] at hadd ⊢
      exact half_emod (Int.dvd_of_emod_eq_zero (by omega)) (Int.dvd_of_emod_eq_zero (by omega)) hadd

/-- Lemma 2: `δ_n` and `M_n` depend only on `δ_0` and the low `n` bits of `f_0, g_0`. -/
theorem divsteps_local (n : ℕ) (s s' : State) (hf0 : s.f % 2 = 1) (hd : s.two_delta = s'.two_delta)
    (hf : s.f % 2^n = s'.f % 2^n) (hg : s.g % 2^n = s'.g % 2^n) :
    (divsteps n s).two_delta = (divsteps n s').two_delta ∧ M n s = M n s' := by
  induction n generalizing s s' with
  | zero => exact ⟨hd, rfl⟩
  | succ n ih =>
    obtain ⟨hT, hd', hf', hg'⟩ := divstep_local n s s' hf0 hd hf hg
    obtain ⟨ihd, ihM⟩ := ih (divstep s) (divstep s') (divstep_f_odd s hf0) hd' hf' hg'
    rw [divsteps_succ, divsteps_succ, M_succ, M_succ, hT, ihM]
    exact ⟨ihd, rfl⟩

/-! ## The determinant, the inverse identity, and the end state -/

def Mat2.det (m : Mat2) : ℤ := m.u * m.r - m.v * m.q

theorem Mat2.det_mul (m n : Mat2) : (m.mul n).det = m.det * n.det := by
  simp only [Mat2.mul, Mat2.det]; ring

theorem T_det (s : State) : (T s).det = 2 := by
  unfold T; split_ifs <;> simp [Mat2.det]

/-- `det (M n s) = 2^n`. -/
theorem M_det (n : ℕ) (s : State) : (M n s).det = 2^n := by
  induction n generalizing s with
  | zero => simp [M, Mat2.one, Mat2.det]
  | succ n ih => rw [M_succ, Mat2.det_mul, ih, T_det, pow_succ]

/-- The adjugate identity: the inputs are integer combinations of the outputs, `f = r f_n - v g_n`
and `g = u g_n - q f_n`. -/
theorem M_inv_spec (n : ℕ) (s : State) (hf : s.f % 2 = 1) :
    s.f = (M n s).r * (divsteps n s).f - (M n s).v * (divsteps n s).g ∧
      s.g = (M n s).u * (divsteps n s).g - (M n s).q * (divsteps n s).f := by
  obtain ⟨h1, h2⟩ := M_spec n s hf
  have hdet := M_det n s
  simp only [Mat2.det] at hdet
  have hpos : (0 : ℤ) < 2^n := by positivity
  constructor
  · apply mul_left_cancel₀ hpos.ne'
    linear_combination (-(M n s).r) * h1 + (M n s).v * h2 - s.f * hdet
  · apply mul_left_cancel₀ hpos.ne'
    linear_combination (-(M n s).u) * h2 + (M n s).q * h1 - s.g * hdet

/-- Lemma 4, in the form Theorem 12 uses: once `g_n = 0`, `f_n` divides both inputs. -/
theorem f_dvd_of_g_eq_zero (n : ℕ) (s : State) (hf : s.f % 2 = 1) (hg : (divsteps n s).g = 0) :
    (divsteps n s).f ∣ s.f ∧ (divsteps n s).f ∣ s.g := by
  obtain ⟨h1, h2⟩ := M_inv_spec n s hf
  rw [hg, mul_zero, sub_zero] at h1
  rw [hg, mul_zero, zero_sub] at h2
  exact ⟨⟨(M n s).r, by rw [h1]; ring⟩, ⟨-(M n s).q, by rw [h2]; ring⟩⟩

end PastaCurves.Inversion
