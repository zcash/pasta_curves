/-
Copyright Amazon.com, Inc. or its affiliates (the block, adapted from s2n-bignum).
Copyright (c) 2026 the pasta_curves contributors (the transcription and the proofs).
-/
import PastaCurves.Spec
import PastaCurves.X86_64.Spec.Arithmetic
import PastaCurves.X86_64.Transcription
import Mathlib.Tactic.ClearExcept

/-!
# Correctness of the inversion's conditional subtraction block

The x86-64 block subtracts the modulus from the input into temporaries, and takes them with
`cmovnc` when the last subtraction did not borrow. The statement is the AArch64 block's. On
x86-64, CF is set on borrow, so a set final CF keeps the input, which is then below `p`.
-/

namespace PastaCurves.X86_64

-- BEGIN condSubBlock_spec statement
/-- The last round's reduction leaves a value below `2p`, and this block makes it canonical: it
subtracts `p` speculatively and, when that borrows, keeps the input instead. The theorem states
both outcomes exactly, so it holds for every bounded input, not only those below `2p`. -/
theorem condSubBlock_spec (value modulus : Limbs) (hv : value.Bounded) (hm : modulus.Bounded)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62) :
    ∀ res, res = condSubBlock value modulus →
      res.Bounded ∧
        ((value.toNat < modulus.toNat ∧ res.toNat = value.toNat) ∨
          (modulus.toNat ≤ value.toNat ∧ res.toNat + modulus.toNat = value.toNat)) := by
  intro res hres
-- END condSubBlock_spec statement
  -- generated skeleton for `condSubBlock`: do not edit between the annotations
  unfold condSubBlock at hres
  lift_lets -merge at hres
  -- r0: input r0
  word_step r0 := value.l0 using hv.1
  -- r1: input r1
  word_step r1 := value.l1 using hv.2.1
  -- r2: input r2
  word_step r2 := value.l2 using hv.2.2.1
  -- r3: input r3
  word_step r3 := value.l3 using hv.2.2.2
  -- p0: input modulus[0]
  word_step p0 := modulus.l0 using hm.1
  -- p1: input modulus[1]
  word_step p1 := modulus.l1 using hm.2.1
  -- p3: input modulus[3]
  word_step p3 := modulus.l3 using hm.2.2.2
  -- t0: mov {t0}, {r0}
  word_step t0 := r0 using b_r0
  -- t0_1: sub {t0}, {p0}
  word_step d, t0_1 := (sbb t0 p0 0).1 using sbb_value_lt t0 p0 0, cf := (sbb t0 p0 0).2
  have l_t0_1 : t0_1 + p0 + 0 = t0 + 2^64 * cf := by
    rw [e_t0_1, e_cf]; exact sbb_lin t0 p0 0 b_t0 b_p0 (by decide)
  have b_cf : cf ≤ 1 := by rw [e_cf]; exact sbb_borrow_le_one t0 p0 0
  clear e_t0_1 e_cf
  -- t1: mov {t1}, {r1}
  word_step t1 := r1 using b_r1
  -- t1_1: sbb {t1}, {p1}
  word_step d_1, t1_1 := (sbb t1 p1 cf).1 using sbb_value_lt t1 p1 cf, cf_1 := (sbb t1 p1 cf).2
  have l_t1_1 : t1_1 + p1 + cf = t1 + 2^64 * cf_1 := by
    rw [e_t1_1, e_cf_1]; exact sbb_lin t1 p1 cf b_t1 b_p1 b_cf
  have b_cf_1 : cf_1 ≤ 1 := by rw [e_cf_1]; exact sbb_borrow_le_one t1 p1 cf
  clear e_t1_1 e_cf_1
  -- t2: mov {t2}, {r2}
  word_step t2 := r2 using b_r2
  -- t2_1: sbb {t2}, 0
  word_step d_2, t2_1 := (sbb t2 0 cf_1).1 using sbb_value_lt t2 0 cf_1, cf_2 := (sbb t2 0 cf_1).2
  have l_t2_1 : t2_1 + 0 + cf_1 = t2 + 2^64 * cf_2 := by
    rw [e_t2_1, e_cf_2]; exact sbb_lin t2 0 cf_1 b_t2 (by decide) b_cf_1
  have b_cf_2 : cf_2 ≤ 1 := by rw [e_cf_2]; exact sbb_borrow_le_one t2 0 cf_1
  clear e_t2_1 e_cf_2
  -- t3: mov {t3}, {r3}
  word_step t3 := r3 using b_r3
  -- t3_1: sbb {t3}, {p3}
  word_step d_3, t3_1 := (sbb t3 p3 cf_2).1 using sbb_value_lt t3 p3 cf_2,
      cf_3 := (sbb t3 p3 cf_2).2
  have l_t3_1 : t3_1 + p3 + cf_2 = t3 + 2^64 * cf_3 := by
    rw [e_t3_1, e_cf_3]; exact sbb_lin t3 p3 cf_2 b_t3 b_p3 b_cf_2
  have b_cf_3 : cf_3 ≤ 1 := by rw [e_cf_3]; exact sbb_borrow_le_one t3 p3 cf_2
  clear e_t3_1 e_cf_3
  -- r0_1: cmovnc {r0}, {t0}
  word_step r0_1 := (if cf_3 = 0 then t0_1 else r0) using ite_lt b_t0_1 b_r0
  -- r1_1: cmovnc {r1}, {t1}
  word_step r1_1 := (if cf_3 = 0 then t1_1 else r1) using ite_lt b_t1_1 b_r1
  -- r2_1: cmovnc {r2}, {t2}
  word_step r2_1 := (if cf_3 = 0 then t2_1 else r2) using ite_lt b_t2_1 b_r2
  -- r3_1: cmovnc {r3}, {t3}
  word_step r3_1 := (if cf_3 = 0 then t3_1 else r3) using ite_lt b_t3_1 b_r3
  subst hres
  -- BEGIN conclusion
  have hV : value.toNat = r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3 := by
    rw [e_r0, e_r1, e_r2, e_r3]; rfl
  have hP : modulus.toNat = p0 + 2^64 * p1 + 2^192 * p3 := by
    rw [e_p0, e_p1, e_p3]; simp only [Limbs.toNat, hshape.1]; omega
  -- The subtraction of `p`: its final borrow is set exactly when the input is below `p`.
  have hD : t0_1 + 2^64 * t1_1 + 2^128 * t2_1 + 2^192 * t3_1 + modulus.toNat
      = value.toNat + 2^256 * cf_3 := by
    rw [hP, hV]; clear * - l_t0_1 l_t1_1 l_t2_1 l_t3_1 e_t0 e_t1 e_t2 e_t3; subst_vars; omega
  refine ⟨⟨b_r0_1, b_r1_1, b_r2_1, b_r3_1⟩, ?_⟩
  show (value.toNat < modulus.toNat ∧
      r0_1 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 = value.toNat) ∨
    (modulus.toNat ≤ value.toNat ∧
      r0_1 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 + modulus.toNat = value.toNat)
  obtain hc | hc : cf_3 = 0 ∨ cf_3 = 1 := by clear * - b_cf_3; omega
  · -- No borrow: the input is at least `p`, and the block keeps the difference.
    rw [if_pos hc] at e_r0_1 e_r1_1 e_r2_1 e_r3_1
    right
    clear * - hD hc e_r0_1 e_r1_1 e_r2_1 e_r3_1
    subst_vars; omega
  · -- A borrow: the input is below `p`, and the block keeps it.
    rw [if_neg (by clear * - hc; omega)] at e_r0_1 e_r1_1 e_r2_1 e_r3_1
    left
    clear * - hV hD hc e_r0_1 e_r1_1 e_r2_1 e_r3_1 b_t0_1 b_t1_1 b_t2_1 b_t3_1
    subst_vars; omega
  -- END conclusion

end PastaCurves.X86_64
