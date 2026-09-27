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

The state is `(d, f, g)` with `d = 2δ` an odd integer (the start `d = 1` is `δ = 1/2`), `f`
odd, and `g` any integer. One divstep is

    (2 - d, g, (g - f) / 2)            if 0 < d and g is odd,
    (2 + d, f, (g + (g mod 2) f) / 2)  otherwise,

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
  d : ℤ
  f : ℤ
  g : ℤ

/-- One half-delta divstep. Both divisions are exact when `f` is odd. -/
def divstep (s : State) : State :=
  if 0 < s.d ∧ s.g % 2 = 1 then ⟨2 - s.d, s.g, (s.g - s.f) / 2⟩
  else ⟨2 + s.d, s.f, (s.g + (s.g % 2) * s.f) / 2⟩

/-- `n` divsteps. -/
def divsteps (n : ℕ) (s : State) : State := divstep^[n] s

@[simp] theorem divsteps_zero (s : State) : divsteps 0 s = s := rfl

theorem divsteps_succ (n : ℕ) (s : State) : divsteps (n + 1) s = divsteps n (divstep s) :=
  Function.iterate_succ_apply divstep n s

/-- A `2×2` integer matrix `[[a, b], [c, d]]`. -/
structure Mat2 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  deriving DecidableEq

def Mat2.one : Mat2 := ⟨1, 0, 0, 1⟩

def Mat2.mul (m n : Mat2) : Mat2 :=
  ⟨m.a * n.a + m.b * n.c, m.a * n.b + m.b * n.d, m.c * n.a + m.d * n.c, m.c * n.b + m.d * n.d⟩

/-- The step matrix: `2 · (f', g')ᵀ = T s · (f, g)ᵀ`. -/
def T (s : State) : Mat2 :=
  if 0 < s.d ∧ s.g % 2 = 1 then ⟨0, 2, -1, 1⟩ else ⟨2, 0, s.g % 2, 1⟩

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

/-- One step: `2 f' = T.a f + T.b g` and `2 g' = T.c f + T.d g`. -/
theorem divstep_eq (s : State) (hf : s.f % 2 = 1) :
    2 * (divstep s).f = (T s).a * s.f + (T s).b * s.g ∧
      2 * (divstep s).g = (T s).c * s.f + (T s).d * s.g := by
  by_cases h : 0 < s.d ∧ s.g % 2 = 1
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
    (2 : ℤ)^n * (divsteps n s).f = (M n s).a * s.f + (M n s).b * s.g ∧
      (2 : ℤ)^n * (divsteps n s).g = (M n s).c * s.f + (M n s).d * s.g := by
  induction n generalizing s with
  | zero => simp [M, Mat2.one]
  | succ n ih =>
    rw [divsteps_succ]
    obtain ⟨ihf, ihg⟩ := ih (divstep s) (divstep_f_odd s hf)
    obtain ⟨hsf, hsg⟩ := divstep_eq s hf
    simp only [M_succ, Mat2.mul, pow_succ]
    constructor
    · linear_combination 2 * ihf + (M n (divstep s)).a * hsf + (M n (divstep s)).b * hsg
    · linear_combination 2 * ihg + (M n (divstep s)).c * hsf + (M n (divstep s)).d * hsg

/-- One step never increases `max (|f|, |g|)`. -/
theorem divstep_abs_le (s : State) (B : ℤ) (hf : |s.f| ≤ B) (hg : |s.g| ≤ B) :
    |(divstep s).f| ≤ B ∧ |(divstep s).g| ≤ B := by
  rw [abs_le] at hf hg
  by_cases h : 0 < s.d ∧ s.g % 2 = 1
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

/-- Lemma 4: from `g = 0` nothing moves but `d`, and the matrix is `[[2^n, 0], [0, 1]]`. -/
theorem divsteps_of_g_zero (n : ℕ) (s : State) (hg : s.g = 0) :
    (divsteps n s).f = s.f ∧ (divsteps n s).g = 0 ∧ M n s = ⟨2^n, 0, 0, 1⟩ := by
  induction n generalizing s with
  | zero => exact ⟨rfl, hg, rfl⟩
  | succ n ih =>
    have hstep : divstep s = ⟨2 + s.d, s.f, 0⟩ := by
      simp [divstep, hg]
    have hT : T s = ⟨2, 0, 0, 1⟩ := by
      simp [T, hg]
    obtain ⟨h1, h2, h3⟩ := ih ⟨2 + s.d, s.f, 0⟩ rfl
    refine ⟨?_, ?_, ?_⟩
    · rw [divsteps_succ, hstep]; exact h1
    · rw [divsteps_succ, hstep]; exact h2
    · rw [M_succ, hstep, hT, h3]
      simp only [Mat2.mul, Mat2.mk.injEq]
      refine ⟨?_, ?_, ?_, ?_⟩ <;> ring

/-- Lemma 3, row sums: `|u_n| + |v_n| ≤ 2^n` and `|q_n| + |r_n| ≤ 2^n`. -/
theorem M_rowSum_le (n : ℕ) (s : State) :
    |(M n s).a| + |(M n s).b| ≤ 2^n ∧ |(M n s).c| + |(M n s).d| ≤ 2^n := by
  induction n generalizing s with
  | zero => simp [M, Mat2.one]
  | succ n ih =>
    obtain ⟨ih1, ih2⟩ := ih (divstep s)
    rw [M_succ]
    set N := M n (divstep s) with hN
    have ha := abs_nonneg N.a
    have hb := abs_nonneg N.b
    have hc := abs_nonneg N.c
    have hd := abs_nonneg N.d
    by_cases h : 0 < s.d ∧ s.g % 2 = 1
    · simp only [T, if_pos h, Mat2.mul, mul_zero, zero_add, mul_neg, mul_one, abs_neg, pow_succ]
      have h1 := abs_add_le (N.a * 2) N.b
      have h2 := abs_add_le (N.c * 2) N.d
      rw [abs_mul, abs_two] at h1 h2
      constructor <;> linarith
    · simp only [T, if_neg h, Mat2.mul, mul_zero, zero_add, mul_one, pow_succ]
      have hbit : |s.g % 2| ≤ 1 := by rw [abs_le]; omega
      have h1 := abs_add_le (N.a * 2) (N.b * (s.g % 2))
      have h2 := abs_add_le (N.c * 2) (N.d * (s.g % 2))
      rw [abs_mul, abs_mul, abs_two] at h1 h2
      have h3 : |N.b| * |s.g % 2| ≤ |N.b| := mul_le_of_le_one_right hb hbit
      have h4 : |N.d| * |s.g % 2| ≤ |N.d| := mul_le_of_le_one_right hd hbit
      constructor <;> linarith

end PastaCurves.Inversion
