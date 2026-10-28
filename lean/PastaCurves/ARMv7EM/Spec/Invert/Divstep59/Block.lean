/-
Copyright (c) 2026 the pasta_curves contributors.
-/
import PastaCurves.ARMv7EM.Spec.Invert.Divstep59.Batch

/-!
# Correctness of the ARMv7E-M 59-step divstep composition

The generated composition runs three proved batches of 20, 20, and 19 packed divsteps, computes the
next low words between them, and multiplies the three transition matrices with the newest on the
left.
-/

namespace PastaCurves.ARMv7EM

open Inversion (State divsteps M)

-- BEGIN divstep59Block_spec helpers
/-- The generated next-low-word helper agrees with the shared integer model. -/
private theorem nextLowWords_spec (k a b f g : Nat) (A B : Int) (hf : f < 2^64)
    (hg : g < 2^64) (ha : (a : Int) = A % 2^64) (hb : (b : Int) = B % 2^64) :
    (nextLowWords k a b f g : Int) = Inversion.nextLow k A B f g := by
  have ef : (f : Int) = (f : Int) % 2^64 := by
    rw [Int.emod_eq_of_lt (by positivity) (by exact_mod_cast hf)]
  have eg : (g : Int) = (g : Int) % 2^64 := by
    rw [Int.emod_eq_of_lt (by positivity) (by exact_mod_cast hg)]
  have hp := Inversion.mul_word a f A f ha ef
  have hq := Inversion.mul_word b g B g hb eg
  have hs := Inversion.addw_word _ _ (A * f) (B * g) hp hq
  unfold nextLowWords lsr Inversion.nextLow
  rw [Int.natCast_ediv, hs]
  norm_num

/-- Four generated words carry a matrix's entries modulo `2^64`. -/
private def CarriesMatWords (m : Nat × Nat × Nat × Nat) (N : Inversion.Mat2) : Prop :=
  (m.1 : Int) = N.u % 2^64 ∧ (m.2.1 : Int) = N.v % 2^64 ∧
    (m.2.2.1 : Int) = N.q % 2^64 ∧ (m.2.2.2 : Int) = N.r % 2^64

/-- The generated matrix multiplication helper agrees with the shared matrix product. -/
private theorem matMulWords_spec (m n : Nat × Nat × Nat × Nat) (M N : Inversion.Mat2)
    (hm : CarriesMatWords m M) (hn : CarriesMatWords n N) :
    CarriesMatWords (matMulWords m n) (M.mul N) := by
  rcases m with ⟨mu, mv, mq, mr⟩
  rcases n with ⟨nu, nv, nq, nr⟩
  rcases hm with ⟨hmu, hmv, hmq, hmr⟩
  rcases hn with ⟨hnu, hnv, hnq, hnr⟩
  simp only [CarriesMatWords, matMulWords, Inversion.Mat2.mul]
  exact ⟨Inversion.addw_word _ _ _ _ (Inversion.mul_word _ _ _ _ hmu hnu)
      (Inversion.mul_word _ _ _ _ hmv hnq),
    Inversion.addw_word _ _ _ _ (Inversion.mul_word _ _ _ _ hmu hnv)
      (Inversion.mul_word _ _ _ _ hmv hnr),
    Inversion.addw_word _ _ _ _ (Inversion.mul_word _ _ _ _ hmq hnu)
      (Inversion.mul_word _ _ _ _ hmr hnq),
    Inversion.addw_word _ _ _ _ (Inversion.mul_word _ _ _ _ hmq hnv)
      (Inversion.mul_word _ _ _ _ hmr hnr)⟩
-- END divstep59Block_spec helpers

-- BEGIN divstep59Block_spec statement
/-- The generated ARMv7E-M composition returns the 59-step `two_delta` and transition matrix. -/
theorem divstep59Block_spec (two_delta f0 g0 : Nat) (s : State) (hsf : s.f % 2 = 1)
    (hsd : s.two_delta % 2 = 1) (hsD : |s.two_delta| + 118 < 2^31)
    (ed : (two_delta : Int) = s.two_delta % 2^64) (ef0 : (f0 : Int) = s.f % 2^64)
    (eg0 : (g0 : Int) = s.g % 2^64) :
    ∀ res, res = divstep59Block two_delta f0 g0 →
      (res.two_delta : Int) = (divsteps 59 s).two_delta % 2^64 ∧
        (res.u : Int) = (M 59 s).u % 2^64 ∧ (res.v : Int) = (M 59 s).v % 2^64 ∧
        (res.q : Int) = (M 59 s).q % 2^64 ∧ (res.r : Int) = (M 59 s).r % 2^64 := by
  intro res hres
-- END divstep59Block_spec statement
  -- generated skeleton for `divstep59Block`: do not edit between the annotations
  unfold divstep59Block at hres
  lift_lets -merge at hres
  -- b1: batch(20, two_delta, f0, g0)
  word_step b1 := batchBlock 20 two_delta f0 g0
  -- d1: first batch output
  word_step d1 := b1.two_delta
  -- u1: first batch output
  word_step u1 := b1.u
  -- v1: first batch output
  word_step v1 := b1.v
  -- q1: first batch output
  word_step q1 := b1.q
  -- r1: first batch output
  word_step r1 := b1.r
  -- f1: next_low for f after batch one
  word_step f1 := nextLowWords 20 u1 v1 f0 g0
  -- g1: next_low for g after batch one
  word_step g1 := nextLowWords 20 q1 r1 f0 g0
  -- b2: batch(20, d1, f1, g1)
  word_step b2 := batchBlock 20 d1 f1 g1
  -- d2: second batch output
  word_step d2 := b2.two_delta
  -- u2: second batch output
  word_step u2 := b2.u
  -- v2: second batch output
  word_step v2 := b2.v
  -- q2: second batch output
  word_step q2 := b2.q
  -- r2: second batch output
  word_step r2 := b2.r
  -- f2: next_low for f after batch two
  word_step f2 := nextLowWords 20 u2 v2 f1 g1
  -- g2: next_low for g after batch two
  word_step g2 := nextLowWords 20 q2 r2 f1 g1
  -- b3: batch(19, d2, f2, g2)
  word_step b3 := batchBlock 19 d2 f2 g2
  -- m21: inner Rust mat_mul
  word_step m21 := matMulWords (u2, v2, q2, r2) (u1, v1, q1, r1)
  -- matrix: outer Rust mat_mul
  word_step matrix := matMulWords (b3.u, b3.v, b3.q, b3.r) m21
  subst hres
  -- BEGIN divstep59Block_spec conclusion
  have hf0w : f0 < 2^64 := by
    have := Int.emod_lt_of_pos s.f (by norm_num : (0 : Int) < 2^64)
    omega
  have hg0w : g0 < 2^64 := by
    have := Int.emod_lt_of_pos s.g (by norm_num : (0 : Int) < 2^64)
    omega
  have hf0odd : (f0 : Int) % 2 = 1 := by
    rw [ef0, Int.emod_emod_of_dvd s.f (by norm_num)]
    exact hsf
  obtain ⟨ha0, ha1, ha2, ha3, ha4⟩ := batchBlock_spec 20 two_delta f0 g0
    (Or.inr rfl) s.two_delta ed hf0odd hsd (by omega) b1 e_b1
  have hf1m : (f1 : Int) = (Inversion.batch 20 s.two_delta f0 g0).2.2.1 := by
    rw [e_f1]
    change (nextLowWords 20 u1 v1 f0 g0 : Int) = Inversion.nextLow 20
      (Inversion.batch 20 s.two_delta f0 g0).2.1.u
      (Inversion.batch 20 s.two_delta f0 g0).2.1.v f0 g0
    exact nextLowWords_spec 20 u1 v1 f0 g0 _ _ hf0w hg0w
      (by rw [e_u1]; exact ha1) (by rw [e_v1]; exact ha2)
  have hg1m : (g1 : Int) = (Inversion.batch 20 s.two_delta f0 g0).2.2.2 := by
    rw [e_g1]
    change (nextLowWords 20 q1 r1 f0 g0 : Int) = Inversion.nextLow 20
      (Inversion.batch 20 s.two_delta f0 g0).2.1.q
      (Inversion.batch 20 s.two_delta f0 g0).2.1.r f0 g0
    exact nextLowWords_spec 20 q1 r1 f0 g0 _ _ hf0w hg0w
      (by rw [e_q1]; exact ha3) (by rw [e_r1]; exact ha4)
  have hf0m : (f0 : Int) % 2^64 = s.f % 2^64 := by rw [ef0, Int.emod_emod]
  have hg0m : (g0 : Int) % 2^64 = s.g % 2^64 := by rw [eg0, Int.emod_emod]
  obtain ⟨hd1, hM1, hF1, hG1⟩ := Inversion.batch_spec 20 64 (le_refl _) (by norm_num)
    (le_refl _) s hsf f0 g0 hf0m hg0m
  generalize hbatch1 : Inversion.batch 20 s.two_delta f0 g0 = B1 at *
  have hs1f := Inversion.divsteps_f_odd 20 s hsf
  have hs1d : B1.1 % 2 = 1 := by
    rw [hd1, Inversion.divsteps_two_delta_emod_two]
    exact hsd
  have hs1D : |B1.1| + 2 * 20 < 2^31 := by
    rw [hd1]
    have h := Inversion.divsteps_two_delta_abs_le 20 s
    omega
  have hf1odd : (f1 : Int) % 2 = 1 := by
    rw [hf1m]
    calc
      B1.2.2.1 % 2 = (B1.2.2.1 % 2^(64 - 20)) % 2 :=
        (Int.emod_emod_of_dvd B1.2.2.1 (by norm_num)).symm
      _ = ((divsteps 20 s).f % 2^(64 - 20)) % 2 := by rw [hF1]
      _ = (divsteps 20 s).f % 2 :=
        Int.emod_emod_of_dvd _ (by norm_num)
      _ = 1 := hs1f
  have hf1w : f1 < 2^64 := by
    rw [e_f1]
    unfold nextLowWords lsr
    have h := addw_lt (u1 * f0 % 2^64) (v1 * g0 % 2^64)
    omega
  have hg1w : g1 < 2^64 := by
    rw [e_g1]
    unfold nextLowWords lsr
    have h := addw_lt (q1 * f0 % 2^64) (r1 * g0 % 2^64)
    omega
  obtain ⟨hb0, hb1, hb2, hb3, hb4⟩ := batchBlock_spec 20 d1 f1 g1
    (Or.inr rfl) B1.1 (by rw [e_d1]; exact ha0) hf1odd hs1d hs1D b2 e_b2
  have hf2m : (f2 : Int) = (Inversion.batch 20 B1.1 f1 g1).2.2.1 := by
    rw [e_f2]
    change (nextLowWords 20 u2 v2 f1 g1 : Int) = Inversion.nextLow 20
      (Inversion.batch 20 B1.1 f1 g1).2.1.u (Inversion.batch 20 B1.1 f1 g1).2.1.v f1 g1
    exact nextLowWords_spec 20 u2 v2 f1 g1 _ _ hf1w hg1w
      (by rw [e_u2]; exact hb1) (by rw [e_v2]; exact hb2)
  have hg2m : (g2 : Int) = (Inversion.batch 20 B1.1 f1 g1).2.2.2 := by
    rw [e_g2]
    change (nextLowWords 20 q2 r2 f1 g1 : Int) = Inversion.nextLow 20
      (Inversion.batch 20 B1.1 f1 g1).2.1.q (Inversion.batch 20 B1.1 f1 g1).2.1.r f1 g1
    exact nextLowWords_spec 20 q2 r2 f1 g1 _ _ hf1w hg1w
      (by rw [e_q2]; exact hb3) (by rw [e_r2]; exact hb4)
  obtain ⟨hd2, hM2, hF2, hG2⟩ := Inversion.batch_spec 20 44 (le_refl _) (by norm_num)
    (by norm_num) (divsteps 20 s) hs1f f1 g1 (by rw [hf1m]; norm_num at hF1 ⊢; exact hF1)
      (by rw [hg1m]; norm_num at hG1 ⊢; exact hG1)
  rw [← hd1] at hd2 hM2 hF2 hG2
  generalize hbatch2 : Inversion.batch 20 B1.1 f1 g1 = B2 at *
  have hs2f := Inversion.divsteps_f_odd 20 (divsteps 20 s) hs1f
  have hs2d : B2.1 % 2 = 1 := by
    rw [hd2, Inversion.divsteps_two_delta_emod_two,
      Inversion.divsteps_two_delta_emod_two]
    exact hsd
  have hs2D : |B2.1| + 2 * 19 < 2^31 := by
    rw [hd2]
    have h1 := Inversion.divsteps_two_delta_abs_le 20 s
    have h2 := Inversion.divsteps_two_delta_abs_le 20 (divsteps 20 s)
    omega
  have hf2odd : (f2 : Int) % 2 = 1 := by
    rw [hf2m]
    calc
      B2.2.2.1 % 2 = (B2.2.2.1 % 2^(44 - 20)) % 2 :=
        (Int.emod_emod_of_dvd B2.2.2.1 (by norm_num)).symm
      _ = ((divsteps 20 (divsteps 20 s)).f % 2^(44 - 20)) % 2 := by rw [hF2]
      _ = (divsteps 20 (divsteps 20 s)).f % 2 :=
        Int.emod_emod_of_dvd _ (by norm_num)
      _ = 1 := hs2f
  obtain ⟨hc0, hc1, hc2, hc3, hc4⟩ := batchBlock_spec 19 d2 f2 g2
    (Or.inl rfl) B2.1 (by rw [e_d2]; exact hb0) hf2odd hs2d hs2D b3 e_b3
  generalize hbatch3 : Inversion.batch 19 B2.1 f2 g2 = B3 at *
  have hm21 : CarriesMatWords m21 (B2.2.1.mul B1.2.1) := by
    rw [e_m21]
    apply matMulWords_spec
    · exact ⟨by rw [e_u2]; exact hb1, by rw [e_v2]; exact hb2,
        by rw [e_q2]; exact hb3, by rw [e_r2]; exact hb4⟩
    · exact ⟨by rw [e_u1]; exact ha1, by rw [e_v1]; exact ha2,
        by rw [e_q1]; exact ha3, by rw [e_r1]; exact ha4⟩
  have hm : CarriesMatWords matrix (B3.2.1.mul (B2.2.1.mul B1.2.1)) := by
    rw [e_matrix]
    apply matMulWords_spec
    · exact ⟨hc1, hc2, hc3, hc4⟩
    · exact hm21
  have hmodel : Inversion.divstep59 s.two_delta f0 g0 =
      (B3.1, B3.2.1.mul (B2.2.1.mul B1.2.1)) := by
    simp only [Inversion.divstep59, hbatch1]
    rw [← hf1m, ← hg1m, hbatch2, ← hf2m, ← hg2m, hbatch3]
  have hf0n : f0 = (s.f % 2^64).toNat := by rw [← ef0]; rfl
  have hg0n : g0 = (s.g % 2^64).toNat := by rw [← eg0]; rfl
  obtain ⟨hd59, hM59⟩ := Inversion.divstep59_spec s hsf
  rw [← hf0n, ← hg0n, hmodel] at hd59 hM59
  simp only at hd59 hM59
  obtain ⟨hmu, hmv, hmq, hmr⟩ := hm
  rw [hM59] at hmu hmv hmq hmr
  exact ⟨by rw [← hd59]; simpa using hc0, hmu, hmv, hmq, hmr⟩
  -- END divstep59Block_spec conclusion

end PastaCurves.ARMv7EM
