import PastaCurves.Inversion.HullData

/-!
# The certificate checked

`Certified H0 H1` from the generated data: each inclusion by `checkInclusion_sound` on a check
that the kernel evaluates, the outer box and the initial triangle likewise, `0 ∈ H1` and the
lattice points by evaluation as well. Then the termination bound for every bit size, and for
`256` in particular.
-/

namespace PastaCurves.Inversion.Hull

set_option maxRecDepth 8192

/-! ## The checks, evaluated by the kernel -/

theorem check_theorem0 :
    checkInclusion H0 H1 ⟨1, 0, 0, 1 / 2⟩ (s ^ 1) farkas_theorem0 = true := by decide +kernel
theorem check_theorem1 :
    checkInclusion H0 H1 ⟨1, 0, 1 / 2, 1 / 2⟩ (s ^ 1) farkas_theorem1 = true := by decide +kernel
theorem check_theorem3 :
    checkInclusion H1 H1 ⟨0, 1, -1 / 2, 1 / 2⟩ (s ^ 1) farkas_theorem3 = true := by decide +kernel
theorem check_theorem5 :
    checkInclusion H1 H0 ⟨0, 1 / 2, -1 / 2, 1 / 4⟩ (s ^ 2) farkas_theorem5 = true := by
  decide +kernel
theorem check_theorem_2 :
    checkInclusion H1 H0 ⟨0, 1 / 4, -1 / 4, 1 / 16⟩ (s ^ 4) farkas_theorem_2 = true := by
  decide +kernel
theorem check_theorem_1 :
    checkInclusion H1 H0 ⟨0, 1 / 4, -1 / 4, 3 / 16⟩ (s ^ 4) farkas_theorem_1 = true := by
  decide +kernel
theorem check_theorem_4scale :
    checkInclusion H1 H1 ⟨33 / 64, 33 / 512, 0, 33 / 64⟩ (s ^ 2) farkas_theorem_4scale = true := by
  decide +kernel
theorem check_theorem_3scale :
    checkInclusion H1 H1 ⟨33 / 64, -33 / 512, 0, 33 / 64⟩ (s ^ 2) farkas_theorem_3scale = true := by
  decide +kernel
theorem check_init2stable :
    checkInclusion hinit H1 ⟨stretch, 0, 0, stretch⟩ 1 farkas_init2stable = true := by
  decide +kernel
theorem check_theoremouter :
    checkInclusion H1 houter ⟨1, 0, 0, 1⟩ 1 farkas_theoremouter = true := by decide +kernel

theorem H1_zero : H1.mem 0 0 := by decide +kernel

/-! ## The certified facts -/

theorem houter_mem (x y : ℚ) (h : houter.mem x y) : |x| ≤ 8193 / 8192 ∧ |y| ≤ 379 / 512 := by
  have h1 := h ⟨0, -512, 379⟩ (by simp [houter])
  have h2 := h ⟨8192, 0, 8193⟩ (by simp [houter])
  have h3 := h ⟨0, 512, 379⟩ (by simp [houter])
  have h4 := h ⟨-8192, 0, 8193⟩ (by simp [houter])
  simp only [HalfPlane.holds] at h1 h2 h3 h4
  constructor <;> rw [abs_le] <;> constructor <;> linarith

theorem hinit_mem (x y : ℚ) (h0 : 0 ≤ y) (hxy : y ≤ x) (hx1 : x ≤ 1) : hinit.mem x y := by
  intro h hh
  simp only [hinit, List.mem_cons, List.not_mem_nil, or_false] at hh
  rcases hh with rfl | rfl | rfl <;> simp only [HalfPlane.holds] <;> linarith

theorem lat0 (x y : ℤ) (hx : |x| ≤ 1) (hy : |y| ≤ 1) (hne : y ≠ 0) :
    ¬ H1.mem (x / L) (y / L) := by
  intro hmem
  rw [abs_le] at hx hy
  obtain ⟨hx1, hx2⟩ := hx
  obtain ⟨hy1, hy2⟩ := hy
  interval_cases x <;> interval_cases y <;>
    first
    | exact absurd rfl hne
    | exact absurd hmem (by decide +kernel)

theorem lat1 (x y : ℤ) (hx : |x| ≤ 2) (hy : |y| ≤ 1) (hne : y ≠ 0) :
    ¬ H1.mem (x * s ^ 2 / L) (y * (2 * s ^ 2) / L) := by
  intro hmem
  rw [abs_le] at hx hy
  obtain ⟨hx1, hx2⟩ := hx
  obtain ⟨hy1, hy2⟩ := hy
  interval_cases x <;> interval_cases y <;>
    first
    | exact absurd rfl hne
    | exact absurd hmem (by decide +kernel)

/-- The certificate's facts, from the checks. -/
theorem certified : Certified H0 H1 where
  inc0 := by
    intro x y h
    have := checkInclusion_sound H0 H1 _ _ farkas_theorem0 check_theorem0 x y h
    convert this using 1 <;> simp only [Mat.apX, Mat.apY] <;> ring
  inc1 := by
    intro x y h
    have := checkInclusion_sound H0 H1 _ _ farkas_theorem1 check_theorem1 x y h
    convert this using 1 <;> simp only [Mat.apX, Mat.apY] <;> ring
  inc3 := by
    intro x y h
    have := checkInclusion_sound H1 H1 _ _ farkas_theorem3 check_theorem3 x y h
    convert this using 1 <;> simp only [Mat.apX, Mat.apY] <;> ring
  inc5 := by
    intro x y h
    have := checkInclusion_sound H1 H0 _ _ farkas_theorem5 check_theorem5 x y h
    convert this using 1 <;> simp only [Mat.apX, Mat.apY] <;> ring
  inc_2 := by
    intro x y h
    have := checkInclusion_sound H1 H0 _ _ farkas_theorem_2 check_theorem_2 x y h
    convert this using 1 <;> simp only [Mat.apX, Mat.apY] <;> ring
  inc_1 := by
    intro x y h
    have := checkInclusion_sound H1 H0 _ _ farkas_theorem_1 check_theorem_1 x y h
    convert this using 1 <;> simp only [Mat.apX, Mat.apY] <;> ring
  inc_4s := by
    intro x y h
    have := checkInclusion_sound H1 H1 _ _ farkas_theorem_4scale check_theorem_4scale x y h
    convert this using 1
    simp only [Mat.apY]
    ring
  inc_3s := by
    intro x y h
    have := checkInclusion_sound H1 H1 _ _ farkas_theorem_3scale check_theorem_3scale x y h
    convert this using 1 <;> simp only [Mat.apX, Mat.apY] <;> ring
  outer := by
    intro x y h
    have := checkInclusion_sound H1 houter _ _ farkas_theoremouter check_theoremouter x y h
    simp only [Mat.apX, Mat.apY, one_mul, zero_mul, add_zero, zero_add, div_one] at this
    exact houter_mem x y this
  init := by
    intro x y h0 hxy hx1
    have := checkInclusion_sound hinit H1 _ _ farkas_init2stable check_init2stable x y
      (hinit_mem x y h0 hxy hx1)
    convert this using 1 <;> simp only [Mat.apX, Mat.apY] <;> ring
  zero := H1_zero
  lat0 := lat0
  lat1 := lat1

/-- Theorem 5 for every bit size. -/
theorem terminationBound (b : ℕ) : TerminationBound b := terminationBound_of_certified certified b

/-- Theorem 5 for `b = 256`: `590` half-delta divsteps, ten rounds of `59`. -/
theorem terminationBound_256 : TerminationBound 256 := terminationBound 256

end PastaCurves.Inversion.Hull
