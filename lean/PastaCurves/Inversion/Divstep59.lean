import PastaCurves.Inversion.Packed

/-!
# The 59-step block on low words

`divstep59` is the block interface of the inversion's inner loop: from `d` and the low 64 bits of
`f` and `g` it returns the 59-step transition matrix and the new `d`, computed as three packed
batches of 20, 20, and 19 steps (Corollary 8 of the proof). Between batches the next low words
are `(a f + b g) / 2^k` on 64-bit words, which carry the true next state modulo `2^44` after the
first batch and modulo `2^24` after the second, enough for the 20 low bits the next batch reads.
The three matrices multiply with the newest on the left.
-/

namespace PastaCurves.Inversion

/-! ## Composing step counts -/

/-- The later steps' matrix, computed at the earlier state, multiplies on the left. -/
theorem M_add (m n : ℕ) (s : State) : M (m + n) s = (M n (divsteps m s)).mul (M m s) := by
  induction m generalizing s with
  | zero => simp [M, Mat2.mul_one]
  | succ m ih =>
    rw [show m + 1 + n = (m + n) + 1 by omega, M_succ, ih (divstep s), Mat2.mul_assoc, ← M_succ,
      ← divsteps_succ]

/-- The integer form of `Nat.mod_mul_right_div_self`. -/
theorem emod_mul_ediv (y m c : ℤ) (hm : 0 < m) (hc : 0 < c) :
    (y % (m * c)) / m = (y / m) % c := by
  have hmc : 0 < m * c := mul_pos hm hc
  have hy := Int.emod_add_mul_ediv y (m * c)
  set r := y % (m * c) with hr
  set q := y / (m * c) with hq
  have hr0 : 0 ≤ r := Int.emod_nonneg y hmc.ne'
  have hr1 : r < m * c := Int.emod_lt_of_pos y hmc
  have hdiv : y / m = r / m + c * q := by
    rw [← hy, show r + m * c * q = r + (c * q) * m by ring, Int.add_mul_ediv_right _ _ hm.ne']
  have hrm : r / m < c := Int.ediv_lt_of_lt_mul hm (by linarith)
  have hrm0 : 0 ≤ r / m := Int.ediv_nonneg hr0 hm.le
  rw [hdiv, Int.add_mul_emod_self_left, Int.emod_eq_of_lt hrm0 hrm]

/-! ## The block on low words -/

/-- The next low word from a matrix row and the current low words: `(a f + b g) / 2^k`, taken
modulo `2^64` first. An abstract model of the 64-bit computation; each backend's proofs relate
its instructions to it. -/
def nextLow (k : ℕ) (a b f g : ℤ) : ℤ := ((a * f + b * g) % 2^64) / 2^k

/-- The matrix read from the two packed words. -/
def unpackMat (k : ℕ) (wf wg : ℤ) : Mat2 :=
  ⟨(unpack k wf).1, (unpack k wf).2, (unpack k wg).1, (unpack k wg).2⟩

/-- One batch on low words: the new `d`, the matrix, and the next low words. -/
def batch (k : ℕ) (d f g : ℤ) : ℤ × Mat2 × ℤ × ℤ :=
  let r := packedDivsteps k d (pack f g).1 (pack f g).2
  let m := unpackMat k r.2.1 r.2.2
  (r.1, m, nextLow k m.a m.b f g, nextLow k m.c m.d f g)

/-- `divstep59`, the block interface: from `d` and the low 64 bits of `f` and `g`, the 59-step
matrix and the new `d`, as batches of 20, 20, and 19. -/
def divstep59 (d : ℤ) (f0 g0 : ℕ) : ℤ × Mat2 :=
  let b1 := batch 20 d f0 g0
  let b2 := batch 20 b1.1 b1.2.2.1 b1.2.2.2
  let b3 := batch 19 b2.1 b2.2.2.1 b2.2.2.2
  (b3.1, b3.2.1.mul (b2.2.1.mul b1.2.1))

/-- The next low words carry the true next state modulo `2^(n-k)` when the current ones carry
the state modulo `2^n`, for `k ≤ n ≤ 64`: the word computation agrees with `2^k f_k = a f + b g`
(Lemma 1) modulo `2^n`, and dividing by `2^k` keeps `n - k` bits. -/
theorem nextLow_spec (k n : ℕ) (hk : k ≤ n) (hn64 : n ≤ 64) (s : State) (hf : s.f % 2 = 1)
    (f g : ℤ) (hfn : f % 2^n = s.f % 2^n) (hgn : g % 2^n = s.g % 2^n) :
    nextLow k (M k s).a (M k s).b f g % 2^(n - k) = (divsteps k s).f % 2^(n - k) ∧
      nextLow k (M k s).c (M k s).d f g % 2^(n - k) = (divsteps k s).g % 2^(n - k) := by
  obtain ⟨hF, hG⟩ := M_spec k s hf
  have hn' : (2 : ℤ)^n ∣ 2^64 := pow_dvd_pow 2 hn64
  have hsplit : (2 : ℤ)^n = 2^k * 2^(n - k) := by rw [← pow_add]; congr 1; omega
  have key : ∀ a b F : ℤ, 2^k * F = a * s.f + b * s.g →
      nextLow k a b f g % 2^(n - k) = F % 2^(n - k) := by
    intro a b F hF
    have hcong : (a * f + b * g) % 2^n = (2^k * F) % 2^n := by
      rw [hF]
      exact Int.ModEq.add (Int.ModEq.mul_left a hfn) (Int.ModEq.mul_left b hgn)
    have hY : ((a * f + b * g) % 2^64) % 2^n = (2^k * F) % 2^n := by
      rw [Int.emod_emod_of_dvd _ hn', hcong]
    rw [hsplit] at hY
    unfold nextLow
    rw [← emod_mul_ediv _ _ _ (by positivity) (by positivity), hY,
      Int.mul_emod_mul_of_pos _ _ (by positivity),
      Int.mul_ediv_cancel_left _ (by positivity : (0 : ℤ) < 2^k).ne']
  exact ⟨key _ _ _ hF, key _ _ _ hG⟩

/-- One batch on low words that carry the state modulo `2^n`, `20 ≤ n ≤ 64`, `k ≤ 20`: the
true `d` and matrix, and next words that carry the true next state modulo `2^(n-k)`. -/
theorem batch_spec (k n : ℕ) (hk : k ≤ 20) (hn20 : 20 ≤ n) (hn64 : n ≤ 64) (s : State)
    (hf : s.f % 2 = 1) (f g : ℤ) (hfn : f % 2^n = s.f % 2^n) (hgn : g % 2^n = s.g % 2^n) :
    (batch k s.d f g).1 = (divsteps k s).d ∧ (batch k s.d f g).2.1 = M k s ∧
      (batch k s.d f g).2.2.1 % 2^(n - k) = (divsteps k s).f % 2^(n - k) ∧
      (batch k s.d f g).2.2.2 % 2^(n - k) = (divsteps k s).g % 2^(n - k) := by
  have h20 : (2 : ℤ)^20 ∣ 2^n := pow_dvd_pow 2 hn20
  have hpack : pack f g = pack s.f s.g := by
    simp only [pack, Prod.mk.injEq]
    constructor
    · rw [← Int.emod_emod_of_dvd f h20, hfn, Int.emod_emod_of_dvd _ h20]
    · rw [← Int.emod_emod_of_dvd g h20, hgn, Int.emod_emod_of_dvd _ h20]
  obtain ⟨hd, hM1, hM2⟩ := packedDivsteps_spec k hk s hf
  have hm : unpackMat k (packedDivsteps k s.d (pack f g).1 (pack f g).2).2.1
      (packedDivsteps k s.d (pack f g).1 (pack f g).2).2.2 = M k s := by
    rw [hpack]
    simp only [unpackMat, hM1, hM2]
  obtain ⟨hnf, hng⟩ := nextLow_spec k n (by omega) hn64 s hf f g hfn hgn
  simp only [batch, hm]
  refine ⟨?_, trivial, hnf, hng⟩
  rw [hpack]
  exact hd

/-- Corollary 8: `divstep59` on the low words of a state is the 59-step matrix and `d`. -/
theorem divstep59_spec (s : State) (hf : s.f % 2 = 1) :
    (divstep59 s.d (s.f % 2^64).toNat (s.g % 2^64).toNat).1 = (divsteps 59 s).d ∧
      (divstep59 s.d (s.f % 2^64).toNat (s.g % 2^64).toNat).2 = M 59 s := by
  have h64 : (0 : ℤ) < 2^64 := by positivity
  have hf0 : (((s.f % 2^64).toNat : ℕ) : ℤ) % 2^64 = s.f % 2^64 := by
    rw [Int.toNat_of_nonneg (Int.emod_nonneg _ h64.ne'), Int.emod_emod]
  have hg0 : (((s.g % 2^64).toNat : ℕ) : ℤ) % 2^64 = s.g % 2^64 := by
    rw [Int.toNat_of_nonneg (Int.emod_nonneg _ h64.ne'), Int.emod_emod]
  obtain ⟨d1, M1, f1, g1⟩ := batch_spec 20 64 (le_refl _) (by norm_num) (le_refl _) s hf _ _ hf0 hg0
  set s1 := divsteps 20 s with hs1
  have hf1 : s1.f % 2 = 1 := divsteps_f_odd 20 s hf
  obtain ⟨d2, M2, f2, g2⟩ :=
    batch_spec 20 44 (le_refl _) (by norm_num) (by norm_num) s1 hf1 _ _ f1 g1
  set s2 := divsteps 20 s1 with hs2
  have hf2 : s2.f % 2 = 1 := divsteps_f_odd 20 s1 hf1
  obtain ⟨d3, M3, -, -⟩ :=
    batch_spec 19 24 (by norm_num) (by norm_num) (by norm_num) s2 hf2 _ _ f2 g2
  simp only [divstep59]
  rw [d1, d2, M1, M2, M3, d3]
  refine ⟨?_, ?_⟩
  · rw [hs2, hs1, ← divsteps_add, ← divsteps_add]
  · rw [hs2, hs1, ← M_add, ← divsteps_add, ← M_add]

end PastaCurves.Inversion
