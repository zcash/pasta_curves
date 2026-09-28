/-
Copyright Amazon.com, Inc. or its affiliates (the block, adapted from s2n-bignum).
Copyright (c) 2026 the pasta_curves contributors (the transcription and the proofs).
-/
import PastaCurves.AArch64.Spec.Words
import PastaCurves.AArch64.Transcription

/-!
# Correctness of the inversion's conditional subtraction block

See the parent module's documentation for details.
-/

namespace PastaCurves.AArch64

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
  -- r0: argument
  extract_lets -merge +onlyGivenNames r0 at hres
  have e_r0 : r0 = value.l0 := rfl
  clear_value r0
  have b_r0 : r0 < 2^64 := by rw [e_r0]; exact hv.1
  -- r1: argument
  extract_lets -merge +onlyGivenNames r1 at hres
  have e_r1 : r1 = value.l1 := rfl
  clear_value r1
  have b_r1 : r1 < 2^64 := by rw [e_r1]; exact hv.2.1
  -- r2: argument
  extract_lets -merge +onlyGivenNames r2 at hres
  have e_r2 : r2 = value.l2 := rfl
  clear_value r2
  have b_r2 : r2 < 2^64 := by rw [e_r2]; exact hv.2.2.1
  -- r3: argument
  extract_lets -merge +onlyGivenNames r3 at hres
  have e_r3 : r3 = value.l3 := rfl
  clear_value r3
  have b_r3 : r3 < 2^64 := by rw [e_r3]; exact hv.2.2.2
  -- p0: argument
  extract_lets -merge +onlyGivenNames p0 at hres
  have e_p0 : p0 = modulus.l0 := rfl
  clear_value p0
  have b_p0 : p0 < 2^64 := by rw [e_p0]; exact hm.1
  -- p1: argument
  extract_lets -merge +onlyGivenNames p1 at hres
  have e_p1 : p1 = modulus.l1 := rfl
  clear_value p1
  have b_p1 : p1 < 2^64 := by rw [e_p1]; exact hm.2.1
  -- q: mov q,#0x4000000000000000
  extract_lets -merge +onlyGivenNames q at hres
  have e_q : q = 4611686018427387904 := rfl
  clear_value q
  have b_q : q < 2^64 := by rw [e_q]; decide
  -- t0: subs t0,r0,p0
  extract_lets -merge +onlyGivenNames s t0 c at hres
  have e_t0 : t0 = (r0 + 2^64 - p0 - (1 - 1)) % 2^64 := rfl
  have e_c : c = (r0 + 2^64 - p0 - (1 - 1)) / 2^64 := rfl
  clear_value s t0 c
  have l_t0 : t0 + 2^64 * c + p0 + 1 = r0 + 2^64 + 1 := by
    rw [e_t0, e_c]; exact subc_lin r0 p0 1 b_p0 (by decide)
  have b_t0 : t0 < 2^64 := by rw [e_t0]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c : c ≤ 1 := by
    rw [e_c]; exact subc_carry_le_one r0 p0 1 b_r0
  clear e_t0 e_c
  -- t1: sbcs t1,r1,p1
  extract_lets -merge +onlyGivenNames s_1 t1 c_1 at hres
  have e_t1 : t1 = (r1 + 2^64 - p1 - (1 - c)) % 2^64 := rfl
  have e_c_1 : c_1 = (r1 + 2^64 - p1 - (1 - c)) / 2^64 := rfl
  clear_value s_1 t1 c_1
  have l_t1 : t1 + 2^64 * c_1 + p1 + 1 = r1 + 2^64 + c := by
    rw [e_t1, e_c_1]; exact subc_lin r1 p1 c b_p1 b_c
  have b_t1 : t1 < 2^64 := by rw [e_t1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_1 : c_1 ≤ 1 := by
    rw [e_c_1]; exact subc_carry_le_one r1 p1 c b_r1
  clear e_t1 e_c_1
  -- t2: sbcs t2,r2,xzr
  extract_lets -merge +onlyGivenNames s_2 t2 c_2 at hres
  have e_t2 : t2 = (r2 + 2^64 - 0 - (1 - c_1)) % 2^64 := rfl
  have e_c_2 : c_2 = (r2 + 2^64 - 0 - (1 - c_1)) / 2^64 := rfl
  clear_value s_2 t2 c_2
  have l_t2 : t2 + 2^64 * c_2 + 0 + 1 = r2 + 2^64 + c_1 := by
    rw [e_t2, e_c_2]; exact subc_lin r2 0 c_1 (by decide) b_c_1
  have b_t2 : t2 < 2^64 := by rw [e_t2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_2 : c_2 ≤ 1 := by
    rw [e_c_2]; exact subc_carry_le_one r2 0 c_1 b_r2
  clear e_t2 e_c_2
  -- t3: sbcs t3,r3,q
  extract_lets -merge +onlyGivenNames s_3 t3 c_3 at hres
  have e_t3 : t3 = (r3 + 2^64 - q - (1 - c_2)) % 2^64 := rfl
  have e_c_3 : c_3 = (r3 + 2^64 - q - (1 - c_2)) / 2^64 := rfl
  clear_value s_3 t3 c_3
  have l_t3 : t3 + 2^64 * c_3 + q + 1 = r3 + 2^64 + c_2 := by
    rw [e_t3, e_c_3]; exact subc_lin r3 q c_2 b_q b_c_2
  have b_t3 : t3 < 2^64 := by rw [e_t3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_3 : c_3 ≤ 1 := by
    rw [e_c_3]; exact subc_carry_le_one r3 q c_2 b_r3
  clear e_t3 e_c_3
  -- r0_1: csel r0,r0,t0,lo
  extract_lets -merge +onlyGivenNames r0_1 at hres
  have e_r0_1 : r0_1 = (if c_3 = 0 then r0 else t0) := rfl
  clear_value r0_1
  have b_r0_1 : r0_1 < 2^64 := by
    rw [e_r0_1]; split <;> first | exact b_r0 | exact b_t0
  -- r1_1: csel r1,r1,t1,lo
  extract_lets -merge +onlyGivenNames r1_1 at hres
  have e_r1_1 : r1_1 = (if c_3 = 0 then r1 else t1) := rfl
  clear_value r1_1
  have b_r1_1 : r1_1 < 2^64 := by
    rw [e_r1_1]; split <;> first | exact b_r1 | exact b_t1
  -- r2_1: csel r2,r2,t2,lo
  extract_lets -merge +onlyGivenNames r2_1 at hres
  have e_r2_1 : r2_1 = (if c_3 = 0 then r2 else t2) := rfl
  clear_value r2_1
  have b_r2_1 : r2_1 < 2^64 := by
    rw [e_r2_1]; split <;> first | exact b_r2 | exact b_t2
  -- r3_1: csel r3,r3,t3,lo
  extract_lets -merge +onlyGivenNames r3_1 at hres
  have e_r3_1 : r3_1 = (if c_3 = 0 then r3 else t3) := rfl
  clear_value r3_1
  have b_r3_1 : r3_1 < 2^64 := by
    rw [e_r3_1]; split <;> first | exact b_r3 | exact b_t3
  subst hres
  -- BEGIN conclusion
  have hV : value.toNat = r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3 := by
    rw [e_r0, e_r1, e_r2, e_r3]; rfl
  have hP : modulus.toNat = p0 + 2^64 * p1 + 2^192 * q := by
    rw [e_p0, e_p1, e_q]; simp only [Limbs.toNat, hshape.1, hshape.2]; omega
  -- The subtraction of `p`: its carry is set exactly when the input is at least `p`.
  have hD : t0 + 2^64 * t1 + 2^128 * t2 + 2^192 * t3 + modulus.toNat + 2^256 * c_3
      = value.toNat + 2^256 := by
    rw [hP, hV]; clear * - l_t0 l_t1 l_t2 l_t3; omega
  refine ⟨⟨b_r0_1, b_r1_1, b_r2_1, b_r3_1⟩, ?_⟩
  show (value.toNat < modulus.toNat ∧
      r0_1 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 = value.toNat) ∨
    (modulus.toNat ≤ value.toNat ∧
      r0_1 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 + modulus.toNat = value.toNat)
  obtain hc | hc : c_3 = 0 ∨ c_3 = 1 := by clear * - b_c_3; omega
  · -- Carry clear: the input is below `p`, and the block keeps it.
    rw [if_pos hc] at e_r0_1 e_r1_1 e_r2_1 e_r3_1
    left
    clear * - hV hD hc e_r0_1 e_r1_1 e_r2_1 e_r3_1 b_t0 b_t1 b_t2 b_t3
    omega
  · -- Carry set: the input is at least `p`, and the block keeps the difference.
    rw [if_neg (by clear * - hc; omega)] at e_r0_1 e_r1_1 e_r2_1 e_r3_1
    right
    clear * - hV hD hc e_r0_1 e_r1_1 e_r2_1 e_r3_1
    omega
  -- END conclusion

end PastaCurves.AArch64
