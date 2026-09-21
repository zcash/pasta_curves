/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.Spec
import PastaAsm.X86_64.Spec.Arithmetic
import PastaAsm.X86_64.Transcription
import Mathlib.Tactic.NormNum

/-!
# Correctness of x86-64 Montgomery reduction

-/

namespace PastaAsm.X86_64

-- BEGIN fromMont arithmetic helpers
/-- One cancellation-and-shift step of the standalone Montgomery reduction block. -/
theorem fromMont_step {x0 x1 x2 x3 q p0 p1 lo lohi p1lo p1hi p3lo p3hi
    n1 c1 n2 c2 n3 c3 n4 c4 o0 c5 o1 c6 o2 c7 o3 c8 c9 : Nat}
    (hc : x0 + lo = 2^64 * c1)
    (d0 : lo + 2^64 * lohi = q * p0)
    (d1 : p1lo + 2^64 * p1hi = q * p1)
    (sh : p3lo + 2^64 * p3hi = q * 2^62)
    (l1 : n1 + 2^64 * c2 = x1 + p1lo + c1)
    (l2 : n2 + 2^64 * c3 = x2 + 0 + c2)
    (l3 : n3 + 2^64 * c4 = x3 + p3lo + c3)
    (l4 : n4 + 2^64 * c5 = 0 + 0 + c4)
    (o0lin : o0 + 2^64 * c6 = n1 + lohi + 0)
    (o1lin : o1 + 2^64 * c7 = n2 + p1hi + c6)
    (o2lin : o2 + 2^64 * c8 = n3 + 0 + c7)
    (o3lin : o3 + 2^64 * c9 = n4 + p3hi + c8)
    (hk : c5 = 0 ∧ c9 = 0) :
    2^64 * (o0 + 2^64 * o1 + 2^128 * o2 + 2^192 * o3) =
      (x0 + 2^64 * x1 + 2^128 * x2 + 2^192 * x3) +
        (p0 * q + 2^64 * (p1 * q) + 2^254 * q) := by
  rw [Nat.mul_comm q p0] at d0
  rw [Nat.mul_comm q p1] at d1
  omega

/-- Final subtraction and x86 `cmovnc`: borrow keeps the candidate, no borrow selects `A - p`. -/
theorem fromMont_conclude {a0 a1 a2 a3 d0 d1 d2 d3 b0 b1 b2 b3
    r0 r1 r2 r3 Q : Nat} {value modulus : Limbs}
    (hmain : 2^256 * (a0 + 2^64 * a1 + 2^128 * a2 + 2^192 * a3) =
      value.toNat + Q * modulus.toNat)
    (hA : a0 + 2^64 * a1 + 2^128 * a2 + 2^192 * a3 < 2 * modulus.toNat)
    (hd0 : d0 + modulus.l0 = a0 + 2^64 * b0)
    (hd1 : d1 + modulus.l1 + b0 = a1 + 2^64 * b1)
    (hd2 : d2 + 0 + b1 = a2 + 2^64 * b2)
    (hd3 : d3 + 2^62 + b2 = a3 + 2^64 * b3)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62) (hb3 : b3 ≤ 1)
    (bd0 : d0 < 2^64) (bd1 : d1 < 2^64) (bd2 : d2 < 2^64) (bd3 : d3 < 2^64)
    (br0 : r0 < 2^64) (br1 : r1 < 2^64) (br2 : r2 < 2^64) (br3 : r3 < 2^64)
    (er0 : r0 = if b3 = 0 then d0 else a0) (er1 : r1 = if b3 = 0 then d1 else a1)
    (er2 : r2 = if b3 = 0 then d2 else a2) (er3 : r3 = if b3 = 0 then d3 else a3) :
    (⟨r0, r1, r2, r3⟩ : Limbs).Bounded ∧ (⟨r0, r1, r2, r3⟩ : Limbs).toNat < modulus.toNat ∧
      2^256 * (⟨r0, r1, r2, r3⟩ : Limbs).toNat ≡ value.toNat [MOD modulus.toNat] := by
  have hP : modulus.toNat = modulus.l0 + 2^64 * modulus.l1 + 2^254 := by
    simp only [Limbs.toNat, hshape.1, hshape.2]; ring
  have hD : d0 + 2^64 * d1 + 2^128 * d2 + 2^192 * d3 + modulus.toNat =
      a0 + 2^64 * a1 + 2^128 * a2 + 2^192 * a3 + 2^256 * b3 := by
    rw [hP]
    omega
  refine ⟨⟨br0, br1, br2, br3⟩, ?_⟩
  show r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3 < modulus.toNat ∧
    2^256 * (r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3) ≡ value.toNat [MOD modulus.toNat]
  obtain hb | hb : b3 = 0 ∨ b3 = 1 := by omega
  · rw [if_pos hb] at er0 er1 er2 er3
    have hout : r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3 + modulus.toNat =
        a0 + 2^64 * a1 + 2^128 * a2 + 2^192 * a3 := by omega
    refine ⟨by omega, modEq_of_add_mul _ _ (2^256) Q _ ?_⟩
    calc
      2^256 * (r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3) + 2^256 * modulus.toNat =
          2^256 * (a0 + 2^64 * a1 + 2^128 * a2 + 2^192 * a3) := by omega
      _ = value.toNat + Q * modulus.toNat := hmain
  · rw [if_neg (by omega)] at er0 er1 er2 er3
    have hout : r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3 =
        a0 + 2^64 * a1 + 2^128 * a2 + 2^192 * a3 := by omega
    have hdiff : d0 + 2^64 * d1 + 2^128 * d2 + 2^192 * d3 < 2^256 := by
      clear * - bd0 bd1 bd2 bd3
      omega
    have hbelow : a0 + 2^64 * a1 + 2^128 * a2 + 2^192 * a3 < modulus.toNat := by
      clear * - hD hb hdiff
      omega
    refine ⟨by omega, modEq_of_add_mul _ _ 0 Q _ ?_⟩
    simp only [zero_mul, add_zero]
    rw [hout]
    exact hmain
-- END fromMont arithmetic helpers

-- BEGIN fromMont_spec statement
/-- The standalone x86-64 conversion block performs four Montgomery cancellation steps and one
conditional subtraction: for every four-limb `value`, the result is below `p` and
`2^256 * result ≡ value (mod p)`. -/
theorem fromMont_spec (value modulus : Limbs) (inv : Nat) (hv : value.Bounded)
    (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinv_lt : inv < 2^64) (hinv : (inv * modulus.l0 + 1) % 2^64 = 0) :
    ∀ r, r = fromMont value modulus inv →
      r.Bounded ∧ r.toNat < modulus.toNat ∧
        2^256 * r.toNat ≡ value.toNat [MOD modulus.toNat] := by
  intro r hr
-- END fromMont_spec statement
  -- generated skeleton for `fromMont`: do not edit between the annotations
  unfold fromMont at hr
  lift_lets -merge at hr
  -- z0: input value[0]
  extract_lets -merge +onlyGivenNames z0 at hr
  have e_z0 : z0 = value.l0 := rfl
  clear_value z0
  have b_z0 : z0 < 2^64 := by rw [e_z0]; exact hv.1
  -- z1: input value[1]
  extract_lets -merge +onlyGivenNames z1 at hr
  have e_z1 : z1 = value.l1 := rfl
  clear_value z1
  have b_z1 : z1 < 2^64 := by rw [e_z1]; exact hv.2.1
  -- z2: input value[2]
  extract_lets -merge +onlyGivenNames z2 at hr
  have e_z2 : z2 = value.l2 := rfl
  clear_value z2
  have b_z2 : z2 < 2^64 := by rw [e_z2]; exact hv.2.2.1
  -- z3: input value[3]
  extract_lets -merge +onlyGivenNames z3 at hr
  have e_z3 : z3 = value.l3 := rfl
  clear_value z3
  have b_z3 : z3 < 2^64 := by rw [e_z3]; exact hv.2.2.2
  -- p0: input modulus[0]
  extract_lets -merge +onlyGivenNames p0 at hr
  have e_p0 : p0 = modulus.l0 := rfl
  clear_value p0
  have b_p0 : p0 < 2^64 := by rw [e_p0]; exact hm.1
  -- p1: input modulus[1]
  extract_lets -merge +onlyGivenNames p1 at hr
  have e_p1 : p1 = modulus.l1 := rfl
  clear_value p1
  have b_p1 : p1 < 2^64 := by rw [e_p1]; exact hm.2.1
  -- inv': scalar input
  extract_lets -merge +onlyGivenNames inv' at hr
  have e_inv' : inv' = inv := rfl
  clear_value inv'
  have b_inv' : inv' < 2^64 := by rw [e_inv']; exact hinv_lt
  -- p3: operand p3 = const PASTA_HIGH_LIMB
  extract_lets -merge +onlyGivenNames p3 at hr
  have e_p3 : p3 = 4611686018427387904 := rfl
  clear_value p3
  have b_p3 : p3 < 2^64 := by rw [e_p3]; decide
  -- rdx: mov rdx, {z0}
  extract_lets -merge +onlyGivenNames rdx at hr
  have e_rdx : rdx = z0 := rfl
  clear_value rdx
  have b_rdx : rdx < 2^64 := by rw [e_rdx]; exact b_z0
  -- rdx_1: imul rdx, {inv}
  extract_lets -merge +onlyGivenNames rdx_1 at hr
  have e_rdx_1 : rdx_1 = rdx * inv' % 2^64 := rfl
  clear_value rdx_1
  have b_rdx_1 : rdx_1 < 2^64 := by rw [e_rdx_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- m: mulx {t2}, {t1}, {p1}
  extract_lets -merge +onlyGivenNames m t2 t1 at hr
  have e_t2 : t2 = (mulx rdx_1 p1).1 := rfl
  have e_t1 : t1 = (mulx rdx_1 p1).2 := rfl
  clear_value m t2 t1
  have b_t2 : t2 < 2^64 := by rw [e_t2]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_1 b_p1)
  have b_t1 : t1 < 2^64 := by rw [e_t1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t2 : t1 + 2^64 * t2 = rdx_1 * p1 := by
    rw [e_t1, e_t2]; exact Nat.mod_add_div _ _
  -- a: mov {a}, rdx
  extract_lets -merge +onlyGivenNames a at hr
  have e_a : a = rdx_1 := rfl
  clear_value a
  have b_a : a < 2^64 := by rw [e_a]; exact b_rdx_1
  -- a_1: shl {a}, 62
  extract_lets -merge +onlyGivenNames a_1 at hr
  have e_a_1 : a_1 = a * 2^62 % 2^64 := rfl
  clear_value a_1
  have b_a_1 : a_1 < 2^64 := by rw [e_a_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_a_1 : a_1 + 2^64 * (a / 2^2) = a * 2^62 := by
    rw [e_a_1]; exact lsl62_lsr2_split _
  -- n: neg {z0}
  extract_lets -merge +onlyGivenNames n z0_1 cf at hr
  have e_z0_1 : z0_1 = (neg z0).1 := rfl
  have e_cf : cf = (neg z0).2 := rfl
  clear_value n z0_1 cf
  have b_z0_1 : z0_1 < 2^64 := by rw [e_z0_1]; exact sbb_value_lt 0 z0 0
  have b_cf : cf ≤ 1 := by
    rw [e_cf]; simp only [neg]; split <;> omega
  -- z1_1: adc {z1}, {t1}
  extract_lets -merge +onlyGivenNames s z1_1 cf_1 at hr
  have e_z1_1 : z1_1 = (addc z1 t1 cf).1 := rfl
  have e_cf_1 : cf_1 = (addc z1 t1 cf).2 := rfl
  clear_value s z1_1 cf_1
  have l_z1_1 : z1_1 + 2^64 * cf_1 = z1 + t1 + cf := by
    rw [e_z1_1, e_cf_1]; exact addc_lin z1 t1 cf
  have b_z1_1 : z1_1 < 2^64 := by rw [e_z1_1]; exact addc_value_lt z1 t1 cf
  have b_cf_1 : cf_1 ≤ 1 := by rw [e_cf_1]; exact addc_carry_le_one z1 t1 cf b_z1 b_t1 b_cf
  clear e_z1_1 e_cf_1
  -- z2_1: adc {z2}, 0
  extract_lets -merge +onlyGivenNames s_1 z2_1 cf_2 at hr
  have e_z2_1 : z2_1 = (addc z2 0 cf_1).1 := rfl
  have e_cf_2 : cf_2 = (addc z2 0 cf_1).2 := rfl
  clear_value s_1 z2_1 cf_2
  have l_z2_1 : z2_1 + 2^64 * cf_2 = z2 + 0 + cf_1 := by
    rw [e_z2_1, e_cf_2]; exact addc_lin z2 0 cf_1
  have b_z2_1 : z2_1 < 2^64 := by rw [e_z2_1]; exact addc_value_lt z2 0 cf_1
  have b_cf_2 : cf_2 ≤ 1 := by rw [e_cf_2]; exact addc_carry_le_one z2 0 cf_1 b_z2 (by decide) b_cf_1
  clear e_z2_1 e_cf_2
  -- z3_1: adc {z3}, {a}
  extract_lets -merge +onlyGivenNames s_2 z3_1 cf_3 at hr
  have e_z3_1 : z3_1 = (addc z3 a_1 cf_2).1 := rfl
  have e_cf_3 : cf_3 = (addc z3 a_1 cf_2).2 := rfl
  clear_value s_2 z3_1 cf_3
  have l_z3_1 : z3_1 + 2^64 * cf_3 = z3 + a_1 + cf_2 := by
    rw [e_z3_1, e_cf_3]; exact addc_lin z3 a_1 cf_2
  have b_z3_1 : z3_1 < 2^64 := by rw [e_z3_1]; exact addc_value_lt z3 a_1 cf_2
  have b_cf_3 : cf_3 ≤ 1 := by rw [e_cf_3]; exact addc_carry_le_one z3 a_1 cf_2 b_z3 b_a_1 b_cf_2
  clear e_z3_1 e_cf_3
  -- a_2: mov {a}, 0
  extract_lets -merge +onlyGivenNames a_2 at hr
  have e_a_2 : a_2 = 0 := rfl
  clear_value a_2
  have b_a_2 : a_2 < 2^64 := by rw [e_a_2]; decide
  -- a_3: adc {a}, 0
  extract_lets -merge +onlyGivenNames s_3 a_3 cf_4 at hr
  have e_a_3 : a_3 = (addc a_2 0 cf_3).1 := rfl
  have e_cf_4 : cf_4 = (addc a_2 0 cf_3).2 := rfl
  clear_value s_3 a_3 cf_4
  have l_a_3 : a_3 + 2^64 * cf_4 = a_2 + 0 + cf_3 := by
    rw [e_a_3, e_cf_4]; exact addc_lin a_2 0 cf_3
  have b_a_3 : a_3 < 2^64 := by rw [e_a_3]; exact addc_value_lt a_2 0 cf_3
  have b_cf_4 : cf_4 ≤ 1 := by rw [e_cf_4]; exact addc_carry_le_one a_2 0 cf_3 b_a_2 (by decide) b_cf_3
  clear e_a_3 e_cf_4
  -- m_1: mulx {t1}, {z0}, {p0}
  extract_lets -merge +onlyGivenNames m_1 t1_1 z0_2 at hr
  have e_t1_1 : t1_1 = (mulx rdx_1 p0).1 := rfl
  have e_z0_2 : z0_2 = (mulx rdx_1 p0).2 := rfl
  clear_value m_1 t1_1 z0_2
  have b_t1_1 : t1_1 < 2^64 := by rw [e_t1_1]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_1 b_p0)
  have b_z0_2 : z0_2 < 2^64 := by rw [e_z0_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t1_1 : z0_2 + 2^64 * t1_1 = rdx_1 * p0 := by
    rw [e_z0_2, e_t1_1]; exact Nat.mod_add_div _ _
  -- z0_3: mov {z0}, rdx
  extract_lets -merge +onlyGivenNames z0_3 at hr
  have e_z0_3 : z0_3 = rdx_1 := rfl
  clear_value z0_3
  have b_z0_3 : z0_3 < 2^64 := by rw [e_z0_3]; exact b_rdx_1
  -- z0_4: shr {z0}, 2
  extract_lets -merge +onlyGivenNames z0_4 at hr
  have e_z0_4 : z0_4 = z0_3 / 2^2 := rfl
  clear_value z0_4
  have b_z0_4 : z0_4 < 2^62 := by
    rw [e_z0_4]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_z0_3 (by norm_num))
  -- z1_2: add {z1}, {t1}
  extract_lets -merge +onlyGivenNames s_4 z1_2 cf_5 at hr
  have e_z1_2 : z1_2 = (addc z1_1 t1_1 0).1 := rfl
  have e_cf_5 : cf_5 = (addc z1_1 t1_1 0).2 := rfl
  clear_value s_4 z1_2 cf_5
  have l_z1_2 : z1_2 + 2^64 * cf_5 = z1_1 + t1_1 + 0 := by
    rw [e_z1_2, e_cf_5]; exact addc_lin z1_1 t1_1 0
  have b_z1_2 : z1_2 < 2^64 := by rw [e_z1_2]; exact addc_value_lt z1_1 t1_1 0
  have b_cf_5 : cf_5 ≤ 1 := by rw [e_cf_5]; exact addc_carry_le_one z1_1 t1_1 0 b_z1_1 b_t1_1 (by decide)
  clear e_z1_2 e_cf_5
  -- z2_2: adc {z2}, {t2}
  extract_lets -merge +onlyGivenNames s_5 z2_2 cf_6 at hr
  have e_z2_2 : z2_2 = (addc z2_1 t2 cf_5).1 := rfl
  have e_cf_6 : cf_6 = (addc z2_1 t2 cf_5).2 := rfl
  clear_value s_5 z2_2 cf_6
  have l_z2_2 : z2_2 + 2^64 * cf_6 = z2_1 + t2 + cf_5 := by
    rw [e_z2_2, e_cf_6]; exact addc_lin z2_1 t2 cf_5
  have b_z2_2 : z2_2 < 2^64 := by rw [e_z2_2]; exact addc_value_lt z2_1 t2 cf_5
  have b_cf_6 : cf_6 ≤ 1 := by rw [e_cf_6]; exact addc_carry_le_one z2_1 t2 cf_5 b_z2_1 b_t2 b_cf_5
  clear e_z2_2 e_cf_6
  -- z3_2: adc {z3}, 0
  extract_lets -merge +onlyGivenNames s_6 z3_2 cf_7 at hr
  have e_z3_2 : z3_2 = (addc z3_1 0 cf_6).1 := rfl
  have e_cf_7 : cf_7 = (addc z3_1 0 cf_6).2 := rfl
  clear_value s_6 z3_2 cf_7
  have l_z3_2 : z3_2 + 2^64 * cf_7 = z3_1 + 0 + cf_6 := by
    rw [e_z3_2, e_cf_7]; exact addc_lin z3_1 0 cf_6
  have b_z3_2 : z3_2 < 2^64 := by rw [e_z3_2]; exact addc_value_lt z3_1 0 cf_6
  have b_cf_7 : cf_7 ≤ 1 := by rw [e_cf_7]; exact addc_carry_le_one z3_1 0 cf_6 b_z3_1 (by decide) b_cf_6
  clear e_z3_2 e_cf_7
  -- a_4: adc {a}, {z0}
  extract_lets -merge +onlyGivenNames s_7 a_4 cf_8 at hr
  have e_a_4 : a_4 = (addc a_3 z0_4 cf_7).1 := rfl
  have e_cf_8 : cf_8 = (addc a_3 z0_4 cf_7).2 := rfl
  clear_value s_7 a_4 cf_8
  have l_a_4 : a_4 + 2^64 * cf_8 = a_3 + z0_4 + cf_7 := by
    rw [e_a_4, e_cf_8]; exact addc_lin a_3 z0_4 cf_7
  have b_a_4 : a_4 < 2^64 := by rw [e_a_4]; exact addc_value_lt a_3 z0_4 cf_7
  have b_cf_8 : cf_8 ≤ 1 := by rw [e_cf_8]; exact addc_carry_le_one a_3 z0_4 cf_7 b_a_3 (lt_of_lt_of_le b_z0_4 (by norm_num)) b_cf_7
  clear e_a_4 e_cf_8
  -- BEGIN fromMont round 0
  have hq_0 : rdx_1 = mulLo inv' z0 := by
    simp only [mulLo, e_rdx_1, e_rdx, Nat.mul_comm]
  have hc_0 : z0 + z0_2 = 2^64 * cf := by
    have h := neg_carry_cancel z0 inv' p0 rdx_1 b_z0 b_inv' b_p0
      (by rw [e_inv', e_p0]; exact hinv) hq_0
    have hlo : mulLo rdx_1 p0 = z0_2 := by rw [e_z0_2]; rfl
    have hcf : cf = if z0 = 0 then 0 else 1 := by rw [e_cf]; rfl
    rwa [hlo, ← hcf] at h
  have z_cf_4 : cf_4 = 0 := by clear * - l_a_3 e_a_2 b_cf_3; omega
  have z_cf_8 : cf_8 = 0 := by
    clear * - l_a_4 z_cf_4 l_a_3 e_a_2 b_cf_3 b_z0_4 b_cf_7
    omega
  have I_0 : 2^64 * (z1_2 + 2^64 * z2_2 + 2^128 * z3_2 + 2^192 * a_4) =
      (z0 + 2^64 * z1 + 2^128 * z2 + 2^192 * z3) +
        (p0 * rdx_1 + 2^64 * (p1 * rdx_1) + 2^254 * rdx_1) :=
    fromMont_step hc_0 d_t1_1 d_t2 (by rw [e_z0_4, e_z0_3]; simpa only [e_a] using sh_a_1)
      l_z1_1 l_z2_1 l_z3_1 (by simpa only [e_a_2] using l_a_3)
      l_z1_2 l_z2_2 l_z3_2 l_a_4 ⟨z_cf_4, z_cf_8⟩
  -- END fromMont round 0
  -- rdx_2: mov rdx, {z1}
  extract_lets -merge +onlyGivenNames rdx_2 at hr
  have e_rdx_2 : rdx_2 = z1_2 := rfl
  clear_value rdx_2
  have b_rdx_2 : rdx_2 < 2^64 := by rw [e_rdx_2]; exact b_z1_2
  -- rdx_3: imul rdx, {inv}
  extract_lets -merge +onlyGivenNames rdx_3 at hr
  have e_rdx_3 : rdx_3 = rdx_2 * inv' % 2^64 := rfl
  clear_value rdx_3
  have b_rdx_3 : rdx_3 < 2^64 := by rw [e_rdx_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- m_2: mulx {t2}, {t1}, {p1}
  extract_lets -merge +onlyGivenNames m_2 t2_1 t1_2 at hr
  have e_t2_1 : t2_1 = (mulx rdx_3 p1).1 := rfl
  have e_t1_2 : t1_2 = (mulx rdx_3 p1).2 := rfl
  clear_value m_2 t2_1 t1_2
  have b_t2_1 : t2_1 < 2^64 := by rw [e_t2_1]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_3 b_p1)
  have b_t1_2 : t1_2 < 2^64 := by rw [e_t1_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t2_1 : t1_2 + 2^64 * t2_1 = rdx_3 * p1 := by
    rw [e_t1_2, e_t2_1]; exact Nat.mod_add_div _ _
  -- z0_5: mov {z0}, rdx
  extract_lets -merge +onlyGivenNames z0_5 at hr
  have e_z0_5 : z0_5 = rdx_3 := rfl
  clear_value z0_5
  have b_z0_5 : z0_5 < 2^64 := by rw [e_z0_5]; exact b_rdx_3
  -- z0_6: shl {z0}, 62
  extract_lets -merge +onlyGivenNames z0_6 at hr
  have e_z0_6 : z0_6 = z0_5 * 2^62 % 2^64 := rfl
  clear_value z0_6
  have b_z0_6 : z0_6 < 2^64 := by rw [e_z0_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_z0_6 : z0_6 + 2^64 * (z0_5 / 2^2) = z0_5 * 2^62 := by
    rw [e_z0_6]; exact lsl62_lsr2_split _
  -- n_1: neg {z1}
  extract_lets -merge +onlyGivenNames n_1 z1_3 cf_9 at hr
  have e_z1_3 : z1_3 = (neg z1_2).1 := rfl
  have e_cf_9 : cf_9 = (neg z1_2).2 := rfl
  clear_value n_1 z1_3 cf_9
  have b_z1_3 : z1_3 < 2^64 := by rw [e_z1_3]; exact sbb_value_lt 0 z1_2 0
  have b_cf_9 : cf_9 ≤ 1 := by
    rw [e_cf_9]; simp only [neg]; split <;> omega
  -- z2_3: adc {z2}, {t1}
  extract_lets -merge +onlyGivenNames s_8 z2_3 cf_10 at hr
  have e_z2_3 : z2_3 = (addc z2_2 t1_2 cf_9).1 := rfl
  have e_cf_10 : cf_10 = (addc z2_2 t1_2 cf_9).2 := rfl
  clear_value s_8 z2_3 cf_10
  have l_z2_3 : z2_3 + 2^64 * cf_10 = z2_2 + t1_2 + cf_9 := by
    rw [e_z2_3, e_cf_10]; exact addc_lin z2_2 t1_2 cf_9
  have b_z2_3 : z2_3 < 2^64 := by rw [e_z2_3]; exact addc_value_lt z2_2 t1_2 cf_9
  have b_cf_10 : cf_10 ≤ 1 := by rw [e_cf_10]; exact addc_carry_le_one z2_2 t1_2 cf_9 b_z2_2 b_t1_2 b_cf_9
  clear e_z2_3 e_cf_10
  -- z3_3: adc {z3}, 0
  extract_lets -merge +onlyGivenNames s_9 z3_3 cf_11 at hr
  have e_z3_3 : z3_3 = (addc z3_2 0 cf_10).1 := rfl
  have e_cf_11 : cf_11 = (addc z3_2 0 cf_10).2 := rfl
  clear_value s_9 z3_3 cf_11
  have l_z3_3 : z3_3 + 2^64 * cf_11 = z3_2 + 0 + cf_10 := by
    rw [e_z3_3, e_cf_11]; exact addc_lin z3_2 0 cf_10
  have b_z3_3 : z3_3 < 2^64 := by rw [e_z3_3]; exact addc_value_lt z3_2 0 cf_10
  have b_cf_11 : cf_11 ≤ 1 := by rw [e_cf_11]; exact addc_carry_le_one z3_2 0 cf_10 b_z3_2 (by decide) b_cf_10
  clear e_z3_3 e_cf_11
  -- a_5: adc {a}, {z0}
  extract_lets -merge +onlyGivenNames s_10 a_5 cf_12 at hr
  have e_a_5 : a_5 = (addc a_4 z0_6 cf_11).1 := rfl
  have e_cf_12 : cf_12 = (addc a_4 z0_6 cf_11).2 := rfl
  clear_value s_10 a_5 cf_12
  have l_a_5 : a_5 + 2^64 * cf_12 = a_4 + z0_6 + cf_11 := by
    rw [e_a_5, e_cf_12]; exact addc_lin a_4 z0_6 cf_11
  have b_a_5 : a_5 < 2^64 := by rw [e_a_5]; exact addc_value_lt a_4 z0_6 cf_11
  have b_cf_12 : cf_12 ≤ 1 := by rw [e_cf_12]; exact addc_carry_le_one a_4 z0_6 cf_11 b_a_4 b_z0_6 b_cf_11
  clear e_a_5 e_cf_12
  -- z0_7: mov {z0}, 0
  extract_lets -merge +onlyGivenNames z0_7 at hr
  have e_z0_7 : z0_7 = 0 := rfl
  clear_value z0_7
  have b_z0_7 : z0_7 < 2^64 := by rw [e_z0_7]; decide
  -- z0_8: adc {z0}, 0
  extract_lets -merge +onlyGivenNames s_11 z0_8 cf_13 at hr
  have e_z0_8 : z0_8 = (addc z0_7 0 cf_12).1 := rfl
  have e_cf_13 : cf_13 = (addc z0_7 0 cf_12).2 := rfl
  clear_value s_11 z0_8 cf_13
  have l_z0_8 : z0_8 + 2^64 * cf_13 = z0_7 + 0 + cf_12 := by
    rw [e_z0_8, e_cf_13]; exact addc_lin z0_7 0 cf_12
  have b_z0_8 : z0_8 < 2^64 := by rw [e_z0_8]; exact addc_value_lt z0_7 0 cf_12
  have b_cf_13 : cf_13 ≤ 1 := by rw [e_cf_13]; exact addc_carry_le_one z0_7 0 cf_12 b_z0_7 (by decide) b_cf_12
  clear e_z0_8 e_cf_13
  -- m_3: mulx {t1}, {z1}, {p0}
  extract_lets -merge +onlyGivenNames m_3 t1_3 z1_4 at hr
  have e_t1_3 : t1_3 = (mulx rdx_3 p0).1 := rfl
  have e_z1_4 : z1_4 = (mulx rdx_3 p0).2 := rfl
  clear_value m_3 t1_3 z1_4
  have b_t1_3 : t1_3 < 2^64 := by rw [e_t1_3]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_3 b_p0)
  have b_z1_4 : z1_4 < 2^64 := by rw [e_z1_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t1_3 : z1_4 + 2^64 * t1_3 = rdx_3 * p0 := by
    rw [e_z1_4, e_t1_3]; exact Nat.mod_add_div _ _
  -- z1_5: mov {z1}, rdx
  extract_lets -merge +onlyGivenNames z1_5 at hr
  have e_z1_5 : z1_5 = rdx_3 := rfl
  clear_value z1_5
  have b_z1_5 : z1_5 < 2^64 := by rw [e_z1_5]; exact b_rdx_3
  -- z1_6: shr {z1}, 2
  extract_lets -merge +onlyGivenNames z1_6 at hr
  have e_z1_6 : z1_6 = z1_5 / 2^2 := rfl
  clear_value z1_6
  have b_z1_6 : z1_6 < 2^62 := by
    rw [e_z1_6]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_z1_5 (by norm_num))
  -- z2_4: add {z2}, {t1}
  extract_lets -merge +onlyGivenNames s_12 z2_4 cf_14 at hr
  have e_z2_4 : z2_4 = (addc z2_3 t1_3 0).1 := rfl
  have e_cf_14 : cf_14 = (addc z2_3 t1_3 0).2 := rfl
  clear_value s_12 z2_4 cf_14
  have l_z2_4 : z2_4 + 2^64 * cf_14 = z2_3 + t1_3 + 0 := by
    rw [e_z2_4, e_cf_14]; exact addc_lin z2_3 t1_3 0
  have b_z2_4 : z2_4 < 2^64 := by rw [e_z2_4]; exact addc_value_lt z2_3 t1_3 0
  have b_cf_14 : cf_14 ≤ 1 := by rw [e_cf_14]; exact addc_carry_le_one z2_3 t1_3 0 b_z2_3 b_t1_3 (by decide)
  clear e_z2_4 e_cf_14
  -- z3_4: adc {z3}, {t2}
  extract_lets -merge +onlyGivenNames s_13 z3_4 cf_15 at hr
  have e_z3_4 : z3_4 = (addc z3_3 t2_1 cf_14).1 := rfl
  have e_cf_15 : cf_15 = (addc z3_3 t2_1 cf_14).2 := rfl
  clear_value s_13 z3_4 cf_15
  have l_z3_4 : z3_4 + 2^64 * cf_15 = z3_3 + t2_1 + cf_14 := by
    rw [e_z3_4, e_cf_15]; exact addc_lin z3_3 t2_1 cf_14
  have b_z3_4 : z3_4 < 2^64 := by rw [e_z3_4]; exact addc_value_lt z3_3 t2_1 cf_14
  have b_cf_15 : cf_15 ≤ 1 := by rw [e_cf_15]; exact addc_carry_le_one z3_3 t2_1 cf_14 b_z3_3 b_t2_1 b_cf_14
  clear e_z3_4 e_cf_15
  -- a_6: adc {a}, 0
  extract_lets -merge +onlyGivenNames s_14 a_6 cf_16 at hr
  have e_a_6 : a_6 = (addc a_5 0 cf_15).1 := rfl
  have e_cf_16 : cf_16 = (addc a_5 0 cf_15).2 := rfl
  clear_value s_14 a_6 cf_16
  have l_a_6 : a_6 + 2^64 * cf_16 = a_5 + 0 + cf_15 := by
    rw [e_a_6, e_cf_16]; exact addc_lin a_5 0 cf_15
  have b_a_6 : a_6 < 2^64 := by rw [e_a_6]; exact addc_value_lt a_5 0 cf_15
  have b_cf_16 : cf_16 ≤ 1 := by rw [e_cf_16]; exact addc_carry_le_one a_5 0 cf_15 b_a_5 (by decide) b_cf_15
  clear e_a_6 e_cf_16
  -- z0_9: adc {z0}, {z1}
  extract_lets -merge +onlyGivenNames s_15 z0_9 cf_17 at hr
  have e_z0_9 : z0_9 = (addc z0_8 z1_6 cf_16).1 := rfl
  have e_cf_17 : cf_17 = (addc z0_8 z1_6 cf_16).2 := rfl
  clear_value s_15 z0_9 cf_17
  have l_z0_9 : z0_9 + 2^64 * cf_17 = z0_8 + z1_6 + cf_16 := by
    rw [e_z0_9, e_cf_17]; exact addc_lin z0_8 z1_6 cf_16
  have b_z0_9 : z0_9 < 2^64 := by rw [e_z0_9]; exact addc_value_lt z0_8 z1_6 cf_16
  have b_cf_17 : cf_17 ≤ 1 := by rw [e_cf_17]; exact addc_carry_le_one z0_8 z1_6 cf_16 b_z0_8 (lt_of_lt_of_le b_z1_6 (by norm_num)) b_cf_16
  clear e_z0_9 e_cf_17
  -- BEGIN fromMont round 1
  have hq_1 : rdx_3 = mulLo inv' z1_2 := by
    simp only [mulLo, e_rdx_3, e_rdx_2, Nat.mul_comm]
  have hc_1 : z1_2 + z1_4 = 2^64 * cf_9 := by
    have h := neg_carry_cancel z1_2 inv' p0 rdx_3 b_z1_2 b_inv' b_p0
      (by rw [e_inv', e_p0]; exact hinv) hq_1
    have hlo : mulLo rdx_3 p0 = z1_4 := by rw [e_z1_4]; rfl
    have hcf : cf_9 = if z1_2 = 0 then 0 else 1 := by rw [e_cf_9]; rfl
    rwa [hlo, ← hcf] at h
  have z_cf_13 : cf_13 = 0 := by clear * - l_z0_8 e_z0_7 b_cf_12; omega
  have z_cf_17 : cf_17 = 0 := by
    clear * - l_z0_9 z_cf_13 l_z0_8 e_z0_7 b_cf_12 b_z1_6 b_cf_16
    omega
  have I_1 : 2^64 * (z2_4 + 2^64 * z3_4 + 2^128 * a_6 + 2^192 * z0_9) =
      (z1_2 + 2^64 * z2_2 + 2^128 * z3_2 + 2^192 * a_4) +
        (p0 * rdx_3 + 2^64 * (p1 * rdx_3) + 2^254 * rdx_3) :=
    fromMont_step hc_1 d_t1_3 d_t2_1 (by rw [e_z1_6, e_z1_5]; simpa only [e_z0_5] using sh_z0_6)
      l_z2_3 l_z3_3 l_a_5 (by simpa only [e_z0_7] using l_z0_8)
      l_z2_4 l_z3_4 l_a_6 l_z0_9 ⟨z_cf_13, z_cf_17⟩
  -- END fromMont round 1
  -- rdx_4: mov rdx, {z2}
  extract_lets -merge +onlyGivenNames rdx_4 at hr
  have e_rdx_4 : rdx_4 = z2_4 := rfl
  clear_value rdx_4
  have b_rdx_4 : rdx_4 < 2^64 := by rw [e_rdx_4]; exact b_z2_4
  -- rdx_5: imul rdx, {inv}
  extract_lets -merge +onlyGivenNames rdx_5 at hr
  have e_rdx_5 : rdx_5 = rdx_4 * inv' % 2^64 := rfl
  clear_value rdx_5
  have b_rdx_5 : rdx_5 < 2^64 := by rw [e_rdx_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- m_4: mulx {t2}, {t1}, {p1}
  extract_lets -merge +onlyGivenNames m_4 t2_2 t1_4 at hr
  have e_t2_2 : t2_2 = (mulx rdx_5 p1).1 := rfl
  have e_t1_4 : t1_4 = (mulx rdx_5 p1).2 := rfl
  clear_value m_4 t2_2 t1_4
  have b_t2_2 : t2_2 < 2^64 := by rw [e_t2_2]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_5 b_p1)
  have b_t1_4 : t1_4 < 2^64 := by rw [e_t1_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t2_2 : t1_4 + 2^64 * t2_2 = rdx_5 * p1 := by
    rw [e_t1_4, e_t2_2]; exact Nat.mod_add_div _ _
  -- z1_7: mov {z1}, rdx
  extract_lets -merge +onlyGivenNames z1_7 at hr
  have e_z1_7 : z1_7 = rdx_5 := rfl
  clear_value z1_7
  have b_z1_7 : z1_7 < 2^64 := by rw [e_z1_7]; exact b_rdx_5
  -- z1_8: shl {z1}, 62
  extract_lets -merge +onlyGivenNames z1_8 at hr
  have e_z1_8 : z1_8 = z1_7 * 2^62 % 2^64 := rfl
  clear_value z1_8
  have b_z1_8 : z1_8 < 2^64 := by rw [e_z1_8]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_z1_8 : z1_8 + 2^64 * (z1_7 / 2^2) = z1_7 * 2^62 := by
    rw [e_z1_8]; exact lsl62_lsr2_split _
  -- n_2: neg {z2}
  extract_lets -merge +onlyGivenNames n_2 z2_5 cf_18 at hr
  have e_z2_5 : z2_5 = (neg z2_4).1 := rfl
  have e_cf_18 : cf_18 = (neg z2_4).2 := rfl
  clear_value n_2 z2_5 cf_18
  have b_z2_5 : z2_5 < 2^64 := by rw [e_z2_5]; exact sbb_value_lt 0 z2_4 0
  have b_cf_18 : cf_18 ≤ 1 := by
    rw [e_cf_18]; simp only [neg]; split <;> omega
  -- z3_5: adc {z3}, {t1}
  extract_lets -merge +onlyGivenNames s_16 z3_5 cf_19 at hr
  have e_z3_5 : z3_5 = (addc z3_4 t1_4 cf_18).1 := rfl
  have e_cf_19 : cf_19 = (addc z3_4 t1_4 cf_18).2 := rfl
  clear_value s_16 z3_5 cf_19
  have l_z3_5 : z3_5 + 2^64 * cf_19 = z3_4 + t1_4 + cf_18 := by
    rw [e_z3_5, e_cf_19]; exact addc_lin z3_4 t1_4 cf_18
  have b_z3_5 : z3_5 < 2^64 := by rw [e_z3_5]; exact addc_value_lt z3_4 t1_4 cf_18
  have b_cf_19 : cf_19 ≤ 1 := by rw [e_cf_19]; exact addc_carry_le_one z3_4 t1_4 cf_18 b_z3_4 b_t1_4 b_cf_18
  clear e_z3_5 e_cf_19
  -- a_7: adc {a}, 0
  extract_lets -merge +onlyGivenNames s_17 a_7 cf_20 at hr
  have e_a_7 : a_7 = (addc a_6 0 cf_19).1 := rfl
  have e_cf_20 : cf_20 = (addc a_6 0 cf_19).2 := rfl
  clear_value s_17 a_7 cf_20
  have l_a_7 : a_7 + 2^64 * cf_20 = a_6 + 0 + cf_19 := by
    rw [e_a_7, e_cf_20]; exact addc_lin a_6 0 cf_19
  have b_a_7 : a_7 < 2^64 := by rw [e_a_7]; exact addc_value_lt a_6 0 cf_19
  have b_cf_20 : cf_20 ≤ 1 := by rw [e_cf_20]; exact addc_carry_le_one a_6 0 cf_19 b_a_6 (by decide) b_cf_19
  clear e_a_7 e_cf_20
  -- z0_10: adc {z0}, {z1}
  extract_lets -merge +onlyGivenNames s_18 z0_10 cf_21 at hr
  have e_z0_10 : z0_10 = (addc z0_9 z1_8 cf_20).1 := rfl
  have e_cf_21 : cf_21 = (addc z0_9 z1_8 cf_20).2 := rfl
  clear_value s_18 z0_10 cf_21
  have l_z0_10 : z0_10 + 2^64 * cf_21 = z0_9 + z1_8 + cf_20 := by
    rw [e_z0_10, e_cf_21]; exact addc_lin z0_9 z1_8 cf_20
  have b_z0_10 : z0_10 < 2^64 := by rw [e_z0_10]; exact addc_value_lt z0_9 z1_8 cf_20
  have b_cf_21 : cf_21 ≤ 1 := by rw [e_cf_21]; exact addc_carry_le_one z0_9 z1_8 cf_20 b_z0_9 b_z1_8 b_cf_20
  clear e_z0_10 e_cf_21
  -- z1_9: mov {z1}, 0
  extract_lets -merge +onlyGivenNames z1_9 at hr
  have e_z1_9 : z1_9 = 0 := rfl
  clear_value z1_9
  have b_z1_9 : z1_9 < 2^64 := by rw [e_z1_9]; decide
  -- z1_10: adc {z1}, 0
  extract_lets -merge +onlyGivenNames s_19 z1_10 cf_22 at hr
  have e_z1_10 : z1_10 = (addc z1_9 0 cf_21).1 := rfl
  have e_cf_22 : cf_22 = (addc z1_9 0 cf_21).2 := rfl
  clear_value s_19 z1_10 cf_22
  have l_z1_10 : z1_10 + 2^64 * cf_22 = z1_9 + 0 + cf_21 := by
    rw [e_z1_10, e_cf_22]; exact addc_lin z1_9 0 cf_21
  have b_z1_10 : z1_10 < 2^64 := by rw [e_z1_10]; exact addc_value_lt z1_9 0 cf_21
  have b_cf_22 : cf_22 ≤ 1 := by rw [e_cf_22]; exact addc_carry_le_one z1_9 0 cf_21 b_z1_9 (by decide) b_cf_21
  clear e_z1_10 e_cf_22
  -- m_5: mulx {t1}, {z2}, {p0}
  extract_lets -merge +onlyGivenNames m_5 t1_5 z2_6 at hr
  have e_t1_5 : t1_5 = (mulx rdx_5 p0).1 := rfl
  have e_z2_6 : z2_6 = (mulx rdx_5 p0).2 := rfl
  clear_value m_5 t1_5 z2_6
  have b_t1_5 : t1_5 < 2^64 := by rw [e_t1_5]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_5 b_p0)
  have b_z2_6 : z2_6 < 2^64 := by rw [e_z2_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t1_5 : z2_6 + 2^64 * t1_5 = rdx_5 * p0 := by
    rw [e_z2_6, e_t1_5]; exact Nat.mod_add_div _ _
  -- z2_7: mov {z2}, rdx
  extract_lets -merge +onlyGivenNames z2_7 at hr
  have e_z2_7 : z2_7 = rdx_5 := rfl
  clear_value z2_7
  have b_z2_7 : z2_7 < 2^64 := by rw [e_z2_7]; exact b_rdx_5
  -- z2_8: shr {z2}, 2
  extract_lets -merge +onlyGivenNames z2_8 at hr
  have e_z2_8 : z2_8 = z2_7 / 2^2 := rfl
  clear_value z2_8
  have b_z2_8 : z2_8 < 2^62 := by
    rw [e_z2_8]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_z2_7 (by norm_num))
  -- z3_6: add {z3}, {t1}
  extract_lets -merge +onlyGivenNames s_20 z3_6 cf_23 at hr
  have e_z3_6 : z3_6 = (addc z3_5 t1_5 0).1 := rfl
  have e_cf_23 : cf_23 = (addc z3_5 t1_5 0).2 := rfl
  clear_value s_20 z3_6 cf_23
  have l_z3_6 : z3_6 + 2^64 * cf_23 = z3_5 + t1_5 + 0 := by
    rw [e_z3_6, e_cf_23]; exact addc_lin z3_5 t1_5 0
  have b_z3_6 : z3_6 < 2^64 := by rw [e_z3_6]; exact addc_value_lt z3_5 t1_5 0
  have b_cf_23 : cf_23 ≤ 1 := by rw [e_cf_23]; exact addc_carry_le_one z3_5 t1_5 0 b_z3_5 b_t1_5 (by decide)
  clear e_z3_6 e_cf_23
  -- a_8: adc {a}, {t2}
  extract_lets -merge +onlyGivenNames s_21 a_8 cf_24 at hr
  have e_a_8 : a_8 = (addc a_7 t2_2 cf_23).1 := rfl
  have e_cf_24 : cf_24 = (addc a_7 t2_2 cf_23).2 := rfl
  clear_value s_21 a_8 cf_24
  have l_a_8 : a_8 + 2^64 * cf_24 = a_7 + t2_2 + cf_23 := by
    rw [e_a_8, e_cf_24]; exact addc_lin a_7 t2_2 cf_23
  have b_a_8 : a_8 < 2^64 := by rw [e_a_8]; exact addc_value_lt a_7 t2_2 cf_23
  have b_cf_24 : cf_24 ≤ 1 := by rw [e_cf_24]; exact addc_carry_le_one a_7 t2_2 cf_23 b_a_7 b_t2_2 b_cf_23
  clear e_a_8 e_cf_24
  -- z0_11: adc {z0}, 0
  extract_lets -merge +onlyGivenNames s_22 z0_11 cf_25 at hr
  have e_z0_11 : z0_11 = (addc z0_10 0 cf_24).1 := rfl
  have e_cf_25 : cf_25 = (addc z0_10 0 cf_24).2 := rfl
  clear_value s_22 z0_11 cf_25
  have l_z0_11 : z0_11 + 2^64 * cf_25 = z0_10 + 0 + cf_24 := by
    rw [e_z0_11, e_cf_25]; exact addc_lin z0_10 0 cf_24
  have b_z0_11 : z0_11 < 2^64 := by rw [e_z0_11]; exact addc_value_lt z0_10 0 cf_24
  have b_cf_25 : cf_25 ≤ 1 := by rw [e_cf_25]; exact addc_carry_le_one z0_10 0 cf_24 b_z0_10 (by decide) b_cf_24
  clear e_z0_11 e_cf_25
  -- z1_11: adc {z1}, {z2}
  extract_lets -merge +onlyGivenNames s_23 z1_11 cf_26 at hr
  have e_z1_11 : z1_11 = (addc z1_10 z2_8 cf_25).1 := rfl
  have e_cf_26 : cf_26 = (addc z1_10 z2_8 cf_25).2 := rfl
  clear_value s_23 z1_11 cf_26
  have l_z1_11 : z1_11 + 2^64 * cf_26 = z1_10 + z2_8 + cf_25 := by
    rw [e_z1_11, e_cf_26]; exact addc_lin z1_10 z2_8 cf_25
  have b_z1_11 : z1_11 < 2^64 := by rw [e_z1_11]; exact addc_value_lt z1_10 z2_8 cf_25
  have b_cf_26 : cf_26 ≤ 1 := by rw [e_cf_26]; exact addc_carry_le_one z1_10 z2_8 cf_25 b_z1_10 (lt_of_lt_of_le b_z2_8 (by norm_num)) b_cf_25
  clear e_z1_11 e_cf_26
  -- BEGIN fromMont round 2
  have hq_2 : rdx_5 = mulLo inv' z2_4 := by
    simp only [mulLo, e_rdx_5, e_rdx_4, Nat.mul_comm]
  have hc_2 : z2_4 + z2_6 = 2^64 * cf_18 := by
    have h := neg_carry_cancel z2_4 inv' p0 rdx_5 b_z2_4 b_inv' b_p0
      (by rw [e_inv', e_p0]; exact hinv) hq_2
    have hlo : mulLo rdx_5 p0 = z2_6 := by rw [e_z2_6]; rfl
    have hcf : cf_18 = if z2_4 = 0 then 0 else 1 := by rw [e_cf_18]; rfl
    rwa [hlo, ← hcf] at h
  have z_cf_22 : cf_22 = 0 := by clear * - l_z1_10 e_z1_9 b_cf_21; omega
  have z_cf_26 : cf_26 = 0 := by
    clear * - l_z1_11 z_cf_22 l_z1_10 e_z1_9 b_cf_21 b_z2_8 b_cf_25
    omega
  have I_2 : 2^64 * (z3_6 + 2^64 * a_8 + 2^128 * z0_11 + 2^192 * z1_11) =
      (z2_4 + 2^64 * z3_4 + 2^128 * a_6 + 2^192 * z0_9) +
        (p0 * rdx_5 + 2^64 * (p1 * rdx_5) + 2^254 * rdx_5) :=
    fromMont_step hc_2 d_t1_5 d_t2_2 (by rw [e_z2_8, e_z2_7]; simpa only [e_z1_7] using sh_z1_8)
      l_z3_5 l_a_7 l_z0_10 (by simpa only [e_z1_9] using l_z1_10)
      l_z3_6 l_a_8 l_z0_11 l_z1_11 ⟨z_cf_22, z_cf_26⟩
  -- END fromMont round 2
  -- rdx_6: mov rdx, {z3}
  extract_lets -merge +onlyGivenNames rdx_6 at hr
  have e_rdx_6 : rdx_6 = z3_6 := rfl
  clear_value rdx_6
  have b_rdx_6 : rdx_6 < 2^64 := by rw [e_rdx_6]; exact b_z3_6
  -- rdx_7: imul rdx, {inv}
  extract_lets -merge +onlyGivenNames rdx_7 at hr
  have e_rdx_7 : rdx_7 = rdx_6 * inv' % 2^64 := rfl
  clear_value rdx_7
  have b_rdx_7 : rdx_7 < 2^64 := by rw [e_rdx_7]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- m_6: mulx {t2}, {t1}, {p1}
  extract_lets -merge +onlyGivenNames m_6 t2_3 t1_6 at hr
  have e_t2_3 : t2_3 = (mulx rdx_7 p1).1 := rfl
  have e_t1_6 : t1_6 = (mulx rdx_7 p1).2 := rfl
  clear_value m_6 t2_3 t1_6
  have b_t2_3 : t2_3 < 2^64 := by rw [e_t2_3]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_7 b_p1)
  have b_t1_6 : t1_6 < 2^64 := by rw [e_t1_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t2_3 : t1_6 + 2^64 * t2_3 = rdx_7 * p1 := by
    rw [e_t1_6, e_t2_3]; exact Nat.mod_add_div _ _
  -- z2_9: mov {z2}, rdx
  extract_lets -merge +onlyGivenNames z2_9 at hr
  have e_z2_9 : z2_9 = rdx_7 := rfl
  clear_value z2_9
  have b_z2_9 : z2_9 < 2^64 := by rw [e_z2_9]; exact b_rdx_7
  -- z2_10: shl {z2}, 62
  extract_lets -merge +onlyGivenNames z2_10 at hr
  have e_z2_10 : z2_10 = z2_9 * 2^62 % 2^64 := rfl
  clear_value z2_10
  have b_z2_10 : z2_10 < 2^64 := by rw [e_z2_10]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_z2_10 : z2_10 + 2^64 * (z2_9 / 2^2) = z2_9 * 2^62 := by
    rw [e_z2_10]; exact lsl62_lsr2_split _
  -- n_3: neg {z3}
  extract_lets -merge +onlyGivenNames n_3 z3_7 cf_27 at hr
  have e_z3_7 : z3_7 = (neg z3_6).1 := rfl
  have e_cf_27 : cf_27 = (neg z3_6).2 := rfl
  clear_value n_3 z3_7 cf_27
  have b_z3_7 : z3_7 < 2^64 := by rw [e_z3_7]; exact sbb_value_lt 0 z3_6 0
  have b_cf_27 : cf_27 ≤ 1 := by
    rw [e_cf_27]; simp only [neg]; split <;> omega
  -- a_9: adc {a}, {t1}
  extract_lets -merge +onlyGivenNames s_24 a_9 cf_28 at hr
  have e_a_9 : a_9 = (addc a_8 t1_6 cf_27).1 := rfl
  have e_cf_28 : cf_28 = (addc a_8 t1_6 cf_27).2 := rfl
  clear_value s_24 a_9 cf_28
  have l_a_9 : a_9 + 2^64 * cf_28 = a_8 + t1_6 + cf_27 := by
    rw [e_a_9, e_cf_28]; exact addc_lin a_8 t1_6 cf_27
  have b_a_9 : a_9 < 2^64 := by rw [e_a_9]; exact addc_value_lt a_8 t1_6 cf_27
  have b_cf_28 : cf_28 ≤ 1 := by rw [e_cf_28]; exact addc_carry_le_one a_8 t1_6 cf_27 b_a_8 b_t1_6 b_cf_27
  clear e_a_9 e_cf_28
  -- z0_12: adc {z0}, 0
  extract_lets -merge +onlyGivenNames s_25 z0_12 cf_29 at hr
  have e_z0_12 : z0_12 = (addc z0_11 0 cf_28).1 := rfl
  have e_cf_29 : cf_29 = (addc z0_11 0 cf_28).2 := rfl
  clear_value s_25 z0_12 cf_29
  have l_z0_12 : z0_12 + 2^64 * cf_29 = z0_11 + 0 + cf_28 := by
    rw [e_z0_12, e_cf_29]; exact addc_lin z0_11 0 cf_28
  have b_z0_12 : z0_12 < 2^64 := by rw [e_z0_12]; exact addc_value_lt z0_11 0 cf_28
  have b_cf_29 : cf_29 ≤ 1 := by rw [e_cf_29]; exact addc_carry_le_one z0_11 0 cf_28 b_z0_11 (by decide) b_cf_28
  clear e_z0_12 e_cf_29
  -- z1_12: adc {z1}, {z2}
  extract_lets -merge +onlyGivenNames s_26 z1_12 cf_30 at hr
  have e_z1_12 : z1_12 = (addc z1_11 z2_10 cf_29).1 := rfl
  have e_cf_30 : cf_30 = (addc z1_11 z2_10 cf_29).2 := rfl
  clear_value s_26 z1_12 cf_30
  have l_z1_12 : z1_12 + 2^64 * cf_30 = z1_11 + z2_10 + cf_29 := by
    rw [e_z1_12, e_cf_30]; exact addc_lin z1_11 z2_10 cf_29
  have b_z1_12 : z1_12 < 2^64 := by rw [e_z1_12]; exact addc_value_lt z1_11 z2_10 cf_29
  have b_cf_30 : cf_30 ≤ 1 := by rw [e_cf_30]; exact addc_carry_le_one z1_11 z2_10 cf_29 b_z1_11 b_z2_10 b_cf_29
  clear e_z1_12 e_cf_30
  -- z2_11: mov {z2}, 0
  extract_lets -merge +onlyGivenNames z2_11 at hr
  have e_z2_11 : z2_11 = 0 := rfl
  clear_value z2_11
  have b_z2_11 : z2_11 < 2^64 := by rw [e_z2_11]; decide
  -- z2_12: adc {z2}, 0
  extract_lets -merge +onlyGivenNames s_27 z2_12 cf_31 at hr
  have e_z2_12 : z2_12 = (addc z2_11 0 cf_30).1 := rfl
  have e_cf_31 : cf_31 = (addc z2_11 0 cf_30).2 := rfl
  clear_value s_27 z2_12 cf_31
  have l_z2_12 : z2_12 + 2^64 * cf_31 = z2_11 + 0 + cf_30 := by
    rw [e_z2_12, e_cf_31]; exact addc_lin z2_11 0 cf_30
  have b_z2_12 : z2_12 < 2^64 := by rw [e_z2_12]; exact addc_value_lt z2_11 0 cf_30
  have b_cf_31 : cf_31 ≤ 1 := by rw [e_cf_31]; exact addc_carry_le_one z2_11 0 cf_30 b_z2_11 (by decide) b_cf_30
  clear e_z2_12 e_cf_31
  -- m_7: mulx {t1}, {z3}, {p0}
  extract_lets -merge +onlyGivenNames m_7 t1_7 z3_8 at hr
  have e_t1_7 : t1_7 = (mulx rdx_7 p0).1 := rfl
  have e_z3_8 : z3_8 = (mulx rdx_7 p0).2 := rfl
  clear_value m_7 t1_7 z3_8
  have b_t1_7 : t1_7 < 2^64 := by rw [e_t1_7]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_7 b_p0)
  have b_z3_8 : z3_8 < 2^64 := by rw [e_z3_8]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_t1_7 : z3_8 + 2^64 * t1_7 = rdx_7 * p0 := by
    rw [e_z3_8, e_t1_7]; exact Nat.mod_add_div _ _
  -- z3_9: mov {z3}, rdx
  extract_lets -merge +onlyGivenNames z3_9 at hr
  have e_z3_9 : z3_9 = rdx_7 := rfl
  clear_value z3_9
  have b_z3_9 : z3_9 < 2^64 := by rw [e_z3_9]; exact b_rdx_7
  -- z3_10: shr {z3}, 2
  extract_lets -merge +onlyGivenNames z3_10 at hr
  have e_z3_10 : z3_10 = z3_9 / 2^2 := rfl
  clear_value z3_10
  have b_z3_10 : z3_10 < 2^62 := by
    rw [e_z3_10]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_z3_9 (by norm_num))
  -- a_10: add {a}, {t1}
  extract_lets -merge +onlyGivenNames s_28 a_10 cf_32 at hr
  have e_a_10 : a_10 = (addc a_9 t1_7 0).1 := rfl
  have e_cf_32 : cf_32 = (addc a_9 t1_7 0).2 := rfl
  clear_value s_28 a_10 cf_32
  have l_a_10 : a_10 + 2^64 * cf_32 = a_9 + t1_7 + 0 := by
    rw [e_a_10, e_cf_32]; exact addc_lin a_9 t1_7 0
  have b_a_10 : a_10 < 2^64 := by rw [e_a_10]; exact addc_value_lt a_9 t1_7 0
  have b_cf_32 : cf_32 ≤ 1 := by rw [e_cf_32]; exact addc_carry_le_one a_9 t1_7 0 b_a_9 b_t1_7 (by decide)
  clear e_a_10 e_cf_32
  -- z0_13: adc {z0}, {t2}
  extract_lets -merge +onlyGivenNames s_29 z0_13 cf_33 at hr
  have e_z0_13 : z0_13 = (addc z0_12 t2_3 cf_32).1 := rfl
  have e_cf_33 : cf_33 = (addc z0_12 t2_3 cf_32).2 := rfl
  clear_value s_29 z0_13 cf_33
  have l_z0_13 : z0_13 + 2^64 * cf_33 = z0_12 + t2_3 + cf_32 := by
    rw [e_z0_13, e_cf_33]; exact addc_lin z0_12 t2_3 cf_32
  have b_z0_13 : z0_13 < 2^64 := by rw [e_z0_13]; exact addc_value_lt z0_12 t2_3 cf_32
  have b_cf_33 : cf_33 ≤ 1 := by rw [e_cf_33]; exact addc_carry_le_one z0_12 t2_3 cf_32 b_z0_12 b_t2_3 b_cf_32
  clear e_z0_13 e_cf_33
  -- z1_13: adc {z1}, 0
  extract_lets -merge +onlyGivenNames s_30 z1_13 cf_34 at hr
  have e_z1_13 : z1_13 = (addc z1_12 0 cf_33).1 := rfl
  have e_cf_34 : cf_34 = (addc z1_12 0 cf_33).2 := rfl
  clear_value s_30 z1_13 cf_34
  have l_z1_13 : z1_13 + 2^64 * cf_34 = z1_12 + 0 + cf_33 := by
    rw [e_z1_13, e_cf_34]; exact addc_lin z1_12 0 cf_33
  have b_z1_13 : z1_13 < 2^64 := by rw [e_z1_13]; exact addc_value_lt z1_12 0 cf_33
  have b_cf_34 : cf_34 ≤ 1 := by rw [e_cf_34]; exact addc_carry_le_one z1_12 0 cf_33 b_z1_12 (by decide) b_cf_33
  clear e_z1_13 e_cf_34
  -- z2_13: adc {z2}, {z3}
  extract_lets -merge +onlyGivenNames s_31 z2_13 cf_35 at hr
  have e_z2_13 : z2_13 = (addc z2_12 z3_10 cf_34).1 := rfl
  have e_cf_35 : cf_35 = (addc z2_12 z3_10 cf_34).2 := rfl
  clear_value s_31 z2_13 cf_35
  have l_z2_13 : z2_13 + 2^64 * cf_35 = z2_12 + z3_10 + cf_34 := by
    rw [e_z2_13, e_cf_35]; exact addc_lin z2_12 z3_10 cf_34
  have b_z2_13 : z2_13 < 2^64 := by rw [e_z2_13]; exact addc_value_lt z2_12 z3_10 cf_34
  have b_cf_35 : cf_35 ≤ 1 := by rw [e_cf_35]; exact addc_carry_le_one z2_12 z3_10 cf_34 b_z2_12 (lt_of_lt_of_le b_z3_10 (by norm_num)) b_cf_34
  clear e_z2_13 e_cf_35
  -- BEGIN fromMont round 3 and candidate
  have hq_3 : rdx_7 = mulLo inv' z3_6 := by
    simp only [mulLo, e_rdx_7, e_rdx_6, Nat.mul_comm]
  have hc_3 : z3_6 + z3_8 = 2^64 * cf_27 := by
    have h := neg_carry_cancel z3_6 inv' p0 rdx_7 b_z3_6 b_inv' b_p0
      (by rw [e_inv', e_p0]; exact hinv) hq_3
    have hlo : mulLo rdx_7 p0 = z3_8 := by rw [e_z3_8]; rfl
    have hcf : cf_27 = if z3_6 = 0 then 0 else 1 := by rw [e_cf_27]; rfl
    rwa [hlo, ← hcf] at h
  have z_cf_31 : cf_31 = 0 := by clear * - l_z2_12 e_z2_11 b_cf_30; omega
  have z_cf_35 : cf_35 = 0 := by
    clear * - l_z2_13 z_cf_31 l_z2_12 e_z2_11 b_cf_30 b_z3_10 b_cf_34
    omega
  have I_3 : 2^64 * (a_10 + 2^64 * z0_13 + 2^128 * z1_13 + 2^192 * z2_13) =
      (z3_6 + 2^64 * a_8 + 2^128 * z0_11 + 2^192 * z1_11) +
        (p0 * rdx_7 + 2^64 * (p1 * rdx_7) + 2^254 * rdx_7) :=
    fromMont_step hc_3 d_t1_7 d_t2_3 (by rw [e_z3_10, e_z3_9]; simpa only [e_z2_9] using sh_z2_10)
      l_a_9 l_z0_12 l_z1_12 (by simpa only [e_z2_11] using l_z2_12)
      l_a_10 l_z0_13 l_z1_13 l_z2_13 ⟨z_cf_31, z_cf_35⟩
  have hQ : rdx_1 + 2^64 * rdx_3 + 2^128 * rdx_5 + 2^192 * rdx_7 < 2^256 := by
    clear * - b_rdx_1 b_rdx_3 b_rdx_5 b_rdx_7
    omega
  have hQp : (rdx_1 + 2^64 * rdx_3 + 2^128 * rdx_5 + 2^192 * rdx_7) * modulus.toNat =
      (p0 * rdx_1 + 2^64 * (p1 * rdx_1) + 2^254 * rdx_1) +
      2^64 * (p0 * rdx_3 + 2^64 * (p1 * rdx_3) + 2^254 * rdx_3) +
      2^128 * (p0 * rdx_5 + 2^64 * (p1 * rdx_5) + 2^254 * rdx_5) +
      2^192 * (p0 * rdx_7 + 2^64 * (p1 * rdx_7) + 2^254 * rdx_7) := by
    rw [e_p0, e_p1]
    simp only [Limbs.toNat, hshape.1, hshape.2]
    ring
  have hV : value.toNat = z0 + 2^64 * z1 + 2^128 * z2 + 2^192 * z3 := by
    rw [e_z0, e_z1, e_z2, e_z3]
    rfl
  have hmain : 2^256 * (a_10 + 2^64 * z0_13 + 2^128 * z1_13 + 2^192 * z2_13) =
      value.toNat + (rdx_1 + 2^64 * rdx_3 + 2^128 * rdx_5 + 2^192 * rdx_7) * modulus.toNat := by
    rw [hV]
    clear * - I_0 I_1 I_2 I_3 hQp
    omega
  have hValue := Limbs.toNat_lt value hv
  have hP : 0 < modulus.toNat := by
    simp only [Limbs.toNat, hshape.1, hshape.2]
    positivity
  have hQPle :
      (rdx_1 + 2^64 * rdx_3 + 2^128 * rdx_5 + 2^192 * rdx_7) * modulus.toNat + modulus.toNat ≤
        2^256 * modulus.toNat := by
    rw [← Nat.succ_mul]
    exact Nat.mul_le_mul_right _ hQ
  have hA : a_10 + 2^64 * z0_13 + 2^128 * z1_13 + 2^192 * z2_13 < 2 * modulus.toNat := by
    clear * - hmain hValue hQPle hP
    omega
  -- END fromMont round 3 and candidate
  -- rdx_8: movabs rdx, {p3}
  extract_lets -merge +onlyGivenNames rdx_8 at hr
  have e_rdx_8 : rdx_8 = p3 := rfl
  clear_value rdx_8
  have b_rdx_8 : rdx_8 < 2^64 := by rw [e_rdx_8]; exact b_p3
  -- t1_8: mov {t1}, {a}
  extract_lets -merge +onlyGivenNames t1_8 at hr
  have e_t1_8 : t1_8 = a_10 := rfl
  clear_value t1_8
  have b_t1_8 : t1_8 < 2^64 := by rw [e_t1_8]; exact b_a_10
  -- t2_4: mov {t2}, {z0}
  extract_lets -merge +onlyGivenNames t2_4 at hr
  have e_t2_4 : t2_4 = z0_13 := rfl
  clear_value t2_4
  have b_t2_4 : t2_4 < 2^64 := by rw [e_t2_4]; exact b_z0_13
  -- z3_11: mov {z3}, {z1}
  extract_lets -merge +onlyGivenNames z3_11 at hr
  have e_z3_11 : z3_11 = z1_13 := rfl
  clear_value z3_11
  have b_z3_11 : z3_11 < 2^64 := by rw [e_z3_11]; exact b_z1_13
  -- z4: mov {z4}, {z2}
  extract_lets -merge +onlyGivenNames z4 at hr
  have e_z4 : z4 = z2_13 := rfl
  clear_value z4
  have b_z4 : z4 < 2^64 := by rw [e_z4]; exact b_z2_13
  -- t1_9: sub {t1}, {p0}
  extract_lets -merge +onlyGivenNames d t1_9 cf_36 at hr
  have e_t1_9 : t1_9 = (sbb t1_8 p0 0).1 := rfl
  have e_cf_36 : cf_36 = (sbb t1_8 p0 0).2 := rfl
  clear_value d t1_9 cf_36
  have l_t1_9 : t1_9 + p0 + 0 = t1_8 + 2^64 * cf_36 := by
    rw [e_t1_9, e_cf_36]; exact sbb_lin t1_8 p0 0 b_t1_8 b_p0 (by decide)
  have b_t1_9 : t1_9 < 2^64 := by rw [e_t1_9]; exact sbb_value_lt t1_8 p0 0
  have b_cf_36 : cf_36 ≤ 1 := by rw [e_cf_36]; exact sbb_borrow_le_one t1_8 p0 0
  clear e_t1_9 e_cf_36
  -- t2_5: sbb {t2}, {p1}
  extract_lets -merge +onlyGivenNames d_1 t2_5 cf_37 at hr
  have e_t2_5 : t2_5 = (sbb t2_4 p1 cf_36).1 := rfl
  have e_cf_37 : cf_37 = (sbb t2_4 p1 cf_36).2 := rfl
  clear_value d_1 t2_5 cf_37
  have l_t2_5 : t2_5 + p1 + cf_36 = t2_4 + 2^64 * cf_37 := by
    rw [e_t2_5, e_cf_37]; exact sbb_lin t2_4 p1 cf_36 b_t2_4 b_p1 b_cf_36
  have b_t2_5 : t2_5 < 2^64 := by rw [e_t2_5]; exact sbb_value_lt t2_4 p1 cf_36
  have b_cf_37 : cf_37 ≤ 1 := by rw [e_cf_37]; exact sbb_borrow_le_one t2_4 p1 cf_36
  clear e_t2_5 e_cf_37
  -- z3_12: sbb {z3}, 0
  extract_lets -merge +onlyGivenNames d_2 z3_12 cf_38 at hr
  have e_z3_12 : z3_12 = (sbb z3_11 0 cf_37).1 := rfl
  have e_cf_38 : cf_38 = (sbb z3_11 0 cf_37).2 := rfl
  clear_value d_2 z3_12 cf_38
  have l_z3_12 : z3_12 + 0 + cf_37 = z3_11 + 2^64 * cf_38 := by
    rw [e_z3_12, e_cf_38]; exact sbb_lin z3_11 0 cf_37 b_z3_11 (by decide) b_cf_37
  have b_z3_12 : z3_12 < 2^64 := by rw [e_z3_12]; exact sbb_value_lt z3_11 0 cf_37
  have b_cf_38 : cf_38 ≤ 1 := by rw [e_cf_38]; exact sbb_borrow_le_one z3_11 0 cf_37
  clear e_z3_12 e_cf_38
  -- z4_1: sbb {z4}, rdx
  extract_lets -merge +onlyGivenNames d_3 z4_1 cf_39 at hr
  have e_z4_1 : z4_1 = (sbb z4 rdx_8 cf_38).1 := rfl
  have e_cf_39 : cf_39 = (sbb z4 rdx_8 cf_38).2 := rfl
  clear_value d_3 z4_1 cf_39
  have l_z4_1 : z4_1 + rdx_8 + cf_38 = z4 + 2^64 * cf_39 := by
    rw [e_z4_1, e_cf_39]; exact sbb_lin z4 rdx_8 cf_38 b_z4 b_rdx_8 b_cf_38
  have b_z4_1 : z4_1 < 2^64 := by rw [e_z4_1]; exact sbb_value_lt z4 rdx_8 cf_38
  have b_cf_39 : cf_39 ≤ 1 := by rw [e_cf_39]; exact sbb_borrow_le_one z4 rdx_8 cf_38
  clear e_z4_1 e_cf_39
  -- a_11: cmovnc {a}, {t1}
  extract_lets -merge +onlyGivenNames a_11 at hr
  have e_a_11 : a_11 = (if cf_39 = 0 then t1_9 else a_10) := rfl
  clear_value a_11
  have b_a_11 : a_11 < 2^64 := by
    rw [e_a_11]; split <;> first | exact b_t1_9 | exact b_a_10
  -- z0_14: cmovnc {z0}, {t2}
  extract_lets -merge +onlyGivenNames z0_14 at hr
  have e_z0_14 : z0_14 = (if cf_39 = 0 then t2_5 else z0_13) := rfl
  clear_value z0_14
  have b_z0_14 : z0_14 < 2^64 := by
    rw [e_z0_14]; split <;> first | exact b_t2_5 | exact b_z0_13
  -- z1_14: cmovnc {z1}, {z3}
  extract_lets -merge +onlyGivenNames z1_14 at hr
  have e_z1_14 : z1_14 = (if cf_39 = 0 then z3_12 else z1_13) := rfl
  clear_value z1_14
  have b_z1_14 : z1_14 < 2^64 := by
    rw [e_z1_14]; split <;> first | exact b_z3_12 | exact b_z1_13
  -- z2_14: cmovnc {z2}, {z4}
  extract_lets -merge +onlyGivenNames z2_14 at hr
  have e_z2_14 : z2_14 = (if cf_39 = 0 then z4_1 else z2_13) := rfl
  clear_value z2_14
  have b_z2_14 : z2_14 < 2^64 := by
    rw [e_z2_14]; split <;> first | exact b_z4_1 | exact b_z2_13
  subst hr
  -- BEGIN fromMont conclusion
  rw [e_p0, e_t1_8] at l_t1_9
  rw [e_p1, e_t2_4] at l_t2_5
  rw [e_z3_11] at l_z3_12
  rw [e_z4, e_rdx_8, e_p3] at l_z4_1
  exact fromMont_conclude hmain hA l_t1_9 l_t2_5 l_z3_12 l_z4_1 hshape b_cf_39
    b_t1_9 b_t2_5 b_z3_12 b_z4_1 b_a_11 b_z0_14 b_z1_14 b_z2_14
    e_a_11 e_z0_14 e_z1_14 e_z2_14
  -- END fromMont conclusion

end PastaAsm.X86_64
