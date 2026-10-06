import Mathlib.Algebra.Order.Ring.Abs
import Mathlib.Data.Int.GCD
import Mathlib.Data.Int.ModEq
import Mathlib.RingTheory.Coprime.Basic
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
paper's step matrix, so that it is integral, and `M n s` is the product of these integral
matrices, `2^n` times the paper's product.

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

/-- Peeling the first step off `n + 1` steps. -/
theorem divsteps_succ (n : ℕ) (s : State) : divsteps (n + 1) s = divsteps n (divstep s) :=
  Function.iterate_succ_apply divstep n s

/-- Peeling the last step off `n + 1` steps. -/
theorem divsteps_succ' (n : ℕ) (s : State) : divsteps (n + 1) s = divstep (divsteps n s) :=
  Function.iterate_succ_apply' divstep n s

/-- Steps compose: `m` steps then `n` more. -/
theorem divsteps_add (m n : ℕ) (s : State) : divsteps (m + n) s = divsteps n (divsteps m s) := by
  unfold divsteps
  rw [Nat.add_comm]
  exact Function.iterate_add_apply divstep n m s

/-- A `2×2` integer matrix `[[a, b], [c, d]]`. -/
structure Mat2 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  deriving DecidableEq

/-- The identity matrix. -/
def Mat2.one : Mat2 := ⟨1, 0, 0, 1⟩

/-- The matrix product. -/
def Mat2.mul (m n : Mat2) : Mat2 :=
  ⟨m.a * n.a + m.b * n.c, m.a * n.b + m.b * n.d, m.c * n.a + m.d * n.c, m.c * n.b + m.d * n.d⟩

/-- The step matrix: `2 · (f', g')ᵀ = T s · (f, g)ᵀ`. -/
def T (s : State) : Mat2 :=
  if 0 < s.d ∧ s.g % 2 = 1 then ⟨0, 2, -1, 1⟩ else ⟨2, 0, s.g % 2, 1⟩

/-- The `n`-step matrix `M n s = T (s_{n-1}) ⋯ T (s_0)`, so `2^n · (f_n, g_n)ᵀ = M n s · (f, g)ᵀ`. -/
def M : ℕ → State → Mat2
  | 0, _ => Mat2.one
  | n + 1, s => (M n (divstep s)).mul (T s)

/-- The definition of `M` at `n + 1` as a rewrite rule: the first step's matrix on the right of
the rest. -/
theorem M_succ (n : ℕ) (s : State) : M (n + 1) s = (M n (divstep s)).mul (T s) := rfl

/-- A step keeps `f` odd: the new `f` is `f` or the odd `g`. -/
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

/-- `f` stays odd through the iteration. -/
theorem divsteps_f_odd (n : ℕ) (s : State) (hf : s.f % 2 = 1) : (divsteps n s).f % 2 = 1 := by
  induction n generalizing s with
  | zero => exact hf
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

/-! ## Locality: `n` steps see only the low `n` bits -/

/-- Halving preserves a congruence between even numbers, at half the modulus. -/
theorem half_emod {a b m : ℤ} (ha : 2 ∣ a) (hb : 2 ∣ b) (h : a % (2 * m) = b % (2 * m)) :
    (a / 2) % m = (b / 2) % m := by
  obtain ⟨a', rfl⟩ := ha
  obtain ⟨b', rfl⟩ := hb
  rw [Int.mul_emod_mul_of_pos _ _ (by norm_num), Int.mul_emod_mul_of_pos _ _ (by norm_num)] at h
  rw [Int.mul_ediv_cancel_left _ (by norm_num), Int.mul_ediv_cancel_left _ (by norm_num)]
  omega

/-- A congruence modulo `2^(n + 1)` gives one modulo `2`. -/
theorem emod_two_of_emod_pow {a b : ℤ} (n : ℕ) (h : a % 2^(n + 1) = b % 2^(n + 1)) :
    a % 2 = b % 2 := by
  have h2 : (2 : ℤ) ∣ 2^(n + 1) := dvd_pow_self 2 (Nat.succ_ne_zero n)
  rw [← Int.emod_emod_of_dvd a h2, h, Int.emod_emod_of_dvd b h2]

/-- One step of Lemma 2: states congruent modulo `2^(n+1)` take the same branch and stay
congruent modulo `2^n`. -/
theorem divstep_local (n : ℕ) (s s' : State) (hf0 : s.f % 2 = 1) (hd : s.d = s'.d)
    (hf : s.f % 2^(n + 1) = s'.f % 2^(n + 1)) (hg : s.g % 2^(n + 1) = s'.g % 2^(n + 1)) :
    T s = T s' ∧ (divstep s).d = (divstep s').d ∧
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
  by_cases h : 0 < s.d ∧ s.g % 2 = 1
  · have h' : 0 < s'.d ∧ s'.g % 2 = 1 := by rw [← hd, ← hg2]; exact h
    have hT : T s = T s' := by simp only [T, if_pos h, if_pos h']
    refine ⟨hT, ?_, ?_, ?_⟩ <;> simp only [divstep, if_pos h, if_pos h']
    · rw [hd]
    · exact hgn
    have hsub : (s.g - s.f) % (2 * 2^n) = (s'.g - s'.f) % (2 * 2^n) := Int.ModEq.sub hg hf
    exact half_emod (Int.dvd_of_emod_eq_zero (by omega)) (Int.dvd_of_emod_eq_zero (by omega)) hsub
  · have h' : ¬ (0 < s'.d ∧ s'.g % 2 = 1) := by rw [← hd, ← hg2]; exact h
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

/-- Lemma 2: `d_n` and `M_n` depend only on `d_0` and the low `n` bits of `f_0, g_0`. -/
theorem divsteps_local (n : ℕ) (s s' : State) (hf0 : s.f % 2 = 1) (hd : s.d = s'.d)
    (hf : s.f % 2^n = s'.f % 2^n) (hg : s.g % 2^n = s'.g % 2^n) :
    (divsteps n s).d = (divsteps n s').d ∧ M n s = M n s' := by
  induction n generalizing s s' with
  | zero => exact ⟨hd, rfl⟩
  | succ n ih =>
    obtain ⟨hT, hd', hf', hg'⟩ := divstep_local n s s' hf0 hd hf hg
    obtain ⟨ihd, ihM⟩ := ih (divstep s) (divstep s') (divstep_f_odd s hf0) hd' hf' hg'
    rw [divsteps_succ, divsteps_succ, M_succ, M_succ, hT, ihM]
    exact ⟨ihd, rfl⟩

/-! ## The determinant, the inverse identity, and the end state -/

/-- The determinant. -/
def Mat2.det (m : Mat2) : ℤ := m.a * m.d - m.b * m.c

/-- The determinant is multiplicative. -/
theorem Mat2.det_mul (m n : Mat2) : (m.mul n).det = m.det * n.det := by
  simp only [Mat2.mul, Mat2.det]; ring

/-- A step matrix has determinant `2`: it is twice the paper's step matrix, whose determinant is
`1/2`. -/
theorem T_det (s : State) : (T s).det = 2 := by
  unfold T; split_ifs <;> simp [Mat2.det]

/-- `det (M n s) = 2^n`. -/
theorem M_det (n : ℕ) (s : State) : (M n s).det = 2^n := by
  induction n generalizing s with
  | zero => simp [M, Mat2.one, Mat2.det]
  | succ n ih => rw [M_succ, Mat2.det_mul, ih, T_det, pow_succ]

/-- The adjugate identity: the inputs are integer combinations of the outputs,
`f = d f_n - b g_n` and `g = a g_n - c f_n`. -/
theorem M_inv_spec (n : ℕ) (s : State) (hf : s.f % 2 = 1) :
    s.f = (M n s).d * (divsteps n s).f - (M n s).b * (divsteps n s).g ∧
      s.g = (M n s).a * (divsteps n s).g - (M n s).c * (divsteps n s).f := by
  obtain ⟨h1, h2⟩ := M_spec n s hf
  have hdet := M_det n s
  simp only [Mat2.det] at hdet
  have hpos : (0 : ℤ) < 2^n := by positivity
  constructor
  · apply mul_left_cancel₀ hpos.ne'
    linear_combination (-(M n s).d) * h1 + (M n s).b * h2 - s.f * hdet
  · apply mul_left_cancel₀ hpos.ne'
    linear_combination (-(M n s).a) * h2 + (M n s).c * h1 - s.g * hdet

/-- Lemma 4′, in the form Theorem 12 uses: once `g_n = 0`, `f_n` divides both inputs. -/
theorem f_dvd_of_g_eq_zero (n : ℕ) (s : State) (hf : s.f % 2 = 1) (hg : (divsteps n s).g = 0) :
    (divsteps n s).f ∣ s.f ∧ (divsteps n s).f ∣ s.g := by
  obtain ⟨h1, h2⟩ := M_inv_spec n s hf
  rw [hg, mul_zero, sub_zero] at h1
  rw [hg, mul_zero, zero_sub] at h2
  exact ⟨⟨(M n s).d, by rw [h1]; ring⟩, ⟨-(M n s).c, by rw [h2]; ring⟩⟩

/-- Lemma 4, the gcd: `gcd (f_n, g_n) = gcd (f_0, g_0)`. A common divisor of the outputs divides
the inputs by the adjugate identity; a common divisor of the inputs divides `2^n f_n` and
`2^n g_n` by Lemma 1, and it is odd because `f_0` is, so it divides `f_n` and `g_n`. -/
theorem divsteps_gcd (n : ℕ) (s : State) (hf : s.f % 2 = 1) :
    Int.gcd (divsteps n s).f (divsteps n s).g = Int.gcd s.f s.g := by
  obtain ⟨hMf, hMg⟩ := M_spec n s hf
  obtain ⟨hIf, hIg⟩ := M_inv_spec n s hf
  set fn := (divsteps n s).f
  set gn := (divsteps n s).g
  set G : ℤ := (Int.gcd s.f s.g : ℤ)
  have hGf : G ∣ s.f := Int.gcd_dvd_left _ _
  have hGg : G ∣ s.g := Int.gcd_dvd_right _ _
  -- `G` divides the odd `f_0`, so it is odd too, and cancels every power of `2`.
  have hcop : IsCoprime G 2 := IsCoprime.of_isCoprime_of_dvd_left ⟨1, -(s.f / 2), by omega⟩ hGf
  have hcancel : ∀ (m : ℕ) (x : ℤ), G ∣ 2^m * x → G ∣ x := by
    intro m
    induction m with
    | zero => simp
    | succ m ih =>
      intro x h
      rw [pow_succ', mul_assoc] at h
      exact ih x (hcop.dvd_of_dvd_mul_left h)
  apply Nat.dvd_antisymm
  · apply Int.dvd_gcd
    · rw [hIf]
      exact dvd_sub (dvd_mul_of_dvd_right (Int.gcd_dvd_left _ _) _)
        (dvd_mul_of_dvd_right (Int.gcd_dvd_right _ _) _)
    · rw [hIg]
      exact dvd_sub (dvd_mul_of_dvd_right (Int.gcd_dvd_right _ _) _)
        (dvd_mul_of_dvd_right (Int.gcd_dvd_left _ _) _)
  · apply Int.dvd_gcd
    · apply hcancel n
      rw [hMf]
      exact dvd_add (dvd_mul_of_dvd_right hGf _) (dvd_mul_of_dvd_right hGg _)
    · apply hcancel n
      rw [hMg]
      exact dvd_add (dvd_mul_of_dvd_right hGf _) (dvd_mul_of_dvd_right hGg _)

/-- Lemma 4, the end state: once `g_n = 0`, `f_n = ± gcd (f_0, g_0)`. -/
theorem f_natAbs_of_g_eq_zero (n : ℕ) (s : State) (hf : s.f % 2 = 1)
    (hg : (divsteps n s).g = 0) : (divsteps n s).f.natAbs = Int.gcd s.f s.g := by
  rw [← divsteps_gcd n s hf, hg, Int.gcd_zero_right]

/-! ## The half-open entry range -/

/-- Associativity. -/
theorem Mat2.mul_assoc (a b c : Mat2) : (a.mul b).mul c = a.mul (b.mul c) := by
  simp only [Mat2.mul, Mat2.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> ring

/-- The identity is a left unit. -/
theorem Mat2.one_mul (a : Mat2) : Mat2.one.mul a = a := by
  simp [Mat2.mul, Mat2.one]

/-- The identity is a right unit. -/
theorem Mat2.mul_one (a : Mat2) : a.mul Mat2.one = a := by
  simp [Mat2.mul, Mat2.one]

/-- `M` built up from the left: the newest step multiplies on the left. -/
theorem M_succ_left (n : ℕ) (s : State) : M (n + 1) s = (T (divsteps n s)).mul (M n s) := by
  induction n generalizing s with
  | zero => simp [M, Mat2.one_mul, Mat2.mul_one]
  | succ n ih =>
    rw [M_succ, ih (divstep s), Mat2.mul_assoc, ← M_succ, divsteps_succ]

/-- Lemma 3, the half-open range: every entry of `M n s` lies in `(-2^n, 2^n]`. With the newest
step on the left, each new entry is either twice an old one or an old one plus or minus another,
so the strict lower bound and the closed upper bound both propagate. This is the range that
makes the packed-word decoding unambiguous. -/
theorem M_entry_range (n : ℕ) (s : State) :
    (-(2 : ℤ)^n < (M n s).a ∧ (M n s).a ≤ 2^n) ∧
      (-(2 : ℤ)^n < (M n s).b ∧ (M n s).b ≤ 2^n) ∧
      (-(2 : ℤ)^n < (M n s).c ∧ (M n s).c ≤ 2^n) ∧
      (-(2 : ℤ)^n < (M n s).d ∧ (M n s).d ≤ 2^n) := by
  induction n with
  | zero => simp [M, Mat2.one]
  | succ n ih =>
    obtain ⟨⟨ha1, ha2⟩, ⟨hb1, hb2⟩, ⟨hc1, hc2⟩, ⟨hd1, hd2⟩⟩ := ih
    rw [M_succ_left]
    set N := M n s with hN
    set t := divsteps n s with ht
    have hpow : (2 : ℤ)^(n + 1) = 2^n * 2 := pow_succ 2 n
    rw [hpow]
    by_cases h : 0 < t.d ∧ t.g % 2 = 1
    · simp only [T, if_pos h, Mat2.mul]
      refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩⟩ <;> omega
    · simp only [T, if_neg h, Mat2.mul]
      have hb : 0 ≤ t.g % 2 ∧ t.g % 2 ≤ 1 := by omega
      refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩⟩ <;> nlinarith

end PastaCurves.Inversion
