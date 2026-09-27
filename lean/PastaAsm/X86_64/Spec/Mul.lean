/-
Copyright Supranational LLC (the routines, transcribed from Semolina v0.1.4).
Copyright (c) 2026 the pasta-asm contributors (the transcription and the proofs).
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.Spec
import PastaAsm.X86_64.Transcription
import PastaAsm.X86_64.Spec.Arithmetic
import Mathlib.Tactic.NormNum

/-!
# Correctness of the transcribed Pasta multiplication block

See the parent module's documentation for details.
-/

namespace PastaAsm.X86_64

-- BEGIN mulMontRound_spec statement
/-- One complete full internal multiplication round. The accumulator first receives `lhs * b`,
then its low limb is cancelled by the Montgomery quotient and the result is shifted by one limb. -/
theorem mulMontRound_spec (lhs modulus : Limbs) (inv b : Nat) (acc : MulMontAcc)
    (hlhs : lhs.Bounded) (hm : modulus.Bounded)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinv_lt : inv < 2^64) (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hb : b < 2^64) (hacc : acc.Bounded)
    (hH1 : acc.toNat + lhs.toNat * b + 2^128 + 3 * 2^254 < 2^320) :
    ∀ s', s' = mulMontRound lhs modulus inv b acc →
      s'.Bounded ∧
        2^64 * s'.toNat = acc.toNat + lhs.toNat * b + s'.q * modulus.toNat := by
  intro s' hr
-- END mulMontRound_spec statement
  -- generated skeleton for `mulMontRound`: do not edit between the annotations
  unfold mulMontRound at hr
  lift_lets -merge at hr
  -- inv': scalar argument
  extract_lets -merge +onlyGivenNames inv' at hr
  have e_inv' : inv' = inv := rfl
  have b_inv' : inv' < 2^64 := by rw [e_inv']; exact hinv_lt
  -- b': mov rdx, qword ptr [{b} + 8]
  extract_lets -merge +onlyGivenNames b' at hr
  have e_b' : b' = b := rfl
  have b_b' : b' < 2^64 := by rw [e_b']; exact hb
  -- r0: accumulator argument
  extract_lets -merge +onlyGivenNames r0 at hr
  have e_r0 : r0 = acc.r0 := rfl
  have b_r0 : r0 < 2^64 := by rw [e_r0]; exact hacc.1
  -- r1: accumulator argument
  extract_lets -merge +onlyGivenNames r1 at hr
  have e_r1 : r1 = acc.r1 := rfl
  have b_r1 : r1 < 2^64 := by rw [e_r1]; exact hacc.2.1
  -- r2: accumulator argument
  extract_lets -merge +onlyGivenNames r2 at hr
  have e_r2 : r2 = acc.r2 := rfl
  have b_r2 : r2 < 2^64 := by rw [e_r2]; exact hacc.2.2.1
  -- r3: accumulator argument
  extract_lets -merge +onlyGivenNames r3 at hr
  have e_r3 : r3 = acc.r3 := rfl
  have b_r3 : r3 < 2^64 := by rw [e_r3]; exact hacc.2.2.2.1
  -- r4: accumulator argument
  extract_lets -merge +onlyGivenNames r4 at hr
  have e_r4 : r4 = acc.r4 := rfl
  have b_r4 : r4 < 2^64 := by rw [e_r4]; exact hacc.2.2.2.2.1
  -- s1: xor {s1}, {s1}
  extract_lets -merge +onlyGivenNames s1 cf ofl at hr
  have e_s1 : s1 = 0 := rfl
  have e_cf : cf = 0 := rfl
  have e_ofl : ofl = 0 := rfl
  have b_s1 : s1 < 2^64 := by rw [e_s1]; decide
  have b_cf : cf ≤ 1 := by rw [e_cf]; decide
  have b_ofl : ofl ≤ 1 := by rw [e_ofl]; decide
  -- m: mulx {s2}, {s1}, qword ptr [{a}]
  extract_lets -merge +onlyGivenNames m s2 s1_1 at hr
  have e_s2 : s2 = (mulx b' lhs.l0).1 := rfl
  have e_s1_1 : s1_1 = (mulx b' lhs.l0).2 := rfl
  have b_s2 : s2 < 2^64 := by rw [e_s2]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_b' hlhs.1)
  have b_s1_1 : s1_1 < 2^64 := by rw [e_s1_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_s2 : s1_1 + 2^64 * s2 = b' * lhs.l0 := by
    rw [e_s1_1, e_s2]; exact Nat.mod_add_div _ _
  -- r0_1: adcx {be}, {s1}
  extract_lets -merge +onlyGivenNames s r0_1 cf_1 at hr
  have e_r0_1 : r0_1 = (addc r0 s1_1 cf).1 := rfl
  have e_cf_1 : cf_1 = (addc r0 s1_1 cf).2 := rfl
  have l_r0_1 : r0_1 + 2^64 * cf_1 = r0 + s1_1 + cf := by
    rw [e_r0_1, e_cf_1]; exact addc_lin r0 s1_1 cf
  have b_r0_1 : r0_1 < 2^64 := by rw [e_r0_1]; exact addc_value_lt r0 s1_1 cf
  have b_cf_1 : cf_1 ≤ 1 := by rw [e_cf_1]; exact addc_carry_le_one r0 s1_1 cf b_r0 b_s1_1 b_cf
  clear e_r0_1 e_cf_1
  -- r1_1: adox {ce}, {s2}
  extract_lets -merge +onlyGivenNames s_1 r1_1 ofl_1 at hr
  have e_r1_1 : r1_1 = (addc r1 s2 ofl).1 := rfl
  have e_ofl_1 : ofl_1 = (addc r1 s2 ofl).2 := rfl
  have l_r1_1 : r1_1 + 2^64 * ofl_1 = r1 + s2 + ofl := by
    rw [e_r1_1, e_ofl_1]; exact addc_lin r1 s2 ofl
  have b_r1_1 : r1_1 < 2^64 := by rw [e_r1_1]; exact addc_value_lt r1 s2 ofl
  have b_ofl_1 : ofl_1 ≤ 1 := by rw [e_ofl_1]; exact addc_carry_le_one r1 s2 ofl b_r1 b_s2 b_ofl
  clear e_r1_1 e_ofl_1
  -- m_1: mulx {s2}, {s1}, qword ptr [{a} + 8]
  extract_lets -merge +onlyGivenNames m_1 s2_1 s1_2 at hr
  have e_s2_1 : s2_1 = (mulx b' lhs.l1).1 := rfl
  have e_s1_2 : s1_2 = (mulx b' lhs.l1).2 := rfl
  have b_s2_1 : s2_1 < 2^64 := by rw [e_s2_1]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_b' hlhs.2.1)
  have b_s1_2 : s1_2 < 2^64 := by rw [e_s1_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_s2_1 : s1_2 + 2^64 * s2_1 = b' * lhs.l1 := by
    rw [e_s1_2, e_s2_1]; exact Nat.mod_add_div _ _
  -- r1_2: adcx {ce}, {s1}
  extract_lets -merge +onlyGivenNames s_2 r1_2 cf_2 at hr
  have e_r1_2 : r1_2 = (addc r1_1 s1_2 cf_1).1 := rfl
  have e_cf_2 : cf_2 = (addc r1_1 s1_2 cf_1).2 := rfl
  have l_r1_2 : r1_2 + 2^64 * cf_2 = r1_1 + s1_2 + cf_1 := by
    rw [e_r1_2, e_cf_2]; exact addc_lin r1_1 s1_2 cf_1
  have b_r1_2 : r1_2 < 2^64 := by rw [e_r1_2]; exact addc_value_lt r1_1 s1_2 cf_1
  have b_cf_2 : cf_2 ≤ 1 := by rw [e_cf_2]; exact addc_carry_le_one r1_1 s1_2 cf_1 b_r1_1 b_s1_2 b_cf_1
  clear e_r1_2 e_cf_2
  -- r2_1: adox {de}, {s2}
  extract_lets -merge +onlyGivenNames s_3 r2_1 ofl_2 at hr
  have e_r2_1 : r2_1 = (addc r2 s2_1 ofl_1).1 := rfl
  have e_ofl_2 : ofl_2 = (addc r2 s2_1 ofl_1).2 := rfl
  have l_r2_1 : r2_1 + 2^64 * ofl_2 = r2 + s2_1 + ofl_1 := by
    rw [e_r2_1, e_ofl_2]; exact addc_lin r2 s2_1 ofl_1
  have b_r2_1 : r2_1 < 2^64 := by rw [e_r2_1]; exact addc_value_lt r2 s2_1 ofl_1
  have b_ofl_2 : ofl_2 ≤ 1 := by rw [e_ofl_2]; exact addc_carry_le_one r2 s2_1 ofl_1 b_r2 b_s2_1 b_ofl_1
  clear e_r2_1 e_ofl_2
  -- m_2: mulx {s2}, {s1}, qword ptr [{a} + 16]
  extract_lets -merge +onlyGivenNames m_2 s2_2 s1_3 at hr
  have e_s2_2 : s2_2 = (mulx b' lhs.l2).1 := rfl
  have e_s1_3 : s1_3 = (mulx b' lhs.l2).2 := rfl
  have b_s2_2 : s2_2 < 2^64 := by rw [e_s2_2]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_b' hlhs.2.2.1)
  have b_s1_3 : s1_3 < 2^64 := by rw [e_s1_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_s2_2 : s1_3 + 2^64 * s2_2 = b' * lhs.l2 := by
    rw [e_s1_3, e_s2_2]; exact Nat.mod_add_div _ _
  -- r2_2: adcx {de}, {s1}
  extract_lets -merge +onlyGivenNames s_4 r2_2 cf_3 at hr
  have e_r2_2 : r2_2 = (addc r2_1 s1_3 cf_2).1 := rfl
  have e_cf_3 : cf_3 = (addc r2_1 s1_3 cf_2).2 := rfl
  have l_r2_2 : r2_2 + 2^64 * cf_3 = r2_1 + s1_3 + cf_2 := by
    rw [e_r2_2, e_cf_3]; exact addc_lin r2_1 s1_3 cf_2
  have b_r2_2 : r2_2 < 2^64 := by rw [e_r2_2]; exact addc_value_lt r2_1 s1_3 cf_2
  have b_cf_3 : cf_3 ≤ 1 := by rw [e_cf_3]; exact addc_carry_le_one r2_1 s1_3 cf_2 b_r2_1 b_s1_3 b_cf_2
  clear e_r2_2 e_cf_3
  -- r3_1: adox {ee}, {s2}
  extract_lets -merge +onlyGivenNames s_5 r3_1 ofl_3 at hr
  have e_r3_1 : r3_1 = (addc r3 s2_2 ofl_2).1 := rfl
  have e_ofl_3 : ofl_3 = (addc r3 s2_2 ofl_2).2 := rfl
  have l_r3_1 : r3_1 + 2^64 * ofl_3 = r3 + s2_2 + ofl_2 := by
    rw [e_r3_1, e_ofl_3]; exact addc_lin r3 s2_2 ofl_2
  have b_r3_1 : r3_1 < 2^64 := by rw [e_r3_1]; exact addc_value_lt r3 s2_2 ofl_2
  have b_ofl_3 : ofl_3 ≤ 1 := by rw [e_ofl_3]; exact addc_carry_le_one r3 s2_2 ofl_2 b_r3 b_s2_2 b_ofl_2
  clear e_r3_1 e_ofl_3
  -- m_3: mulx {s2}, {s1}, qword ptr [{a} + 24]
  extract_lets -merge +onlyGivenNames m_3 s2_3 s1_4 at hr
  have e_s2_3 : s2_3 = (mulx b' lhs.l3).1 := rfl
  have e_s1_4 : s1_4 = (mulx b' lhs.l3).2 := rfl
  have b_s2_3 : s2_3 < 2^64 := by rw [e_s2_3]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_b' hlhs.2.2.2)
  have b_s1_4 : s1_4 < 2^64 := by rw [e_s1_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_s2_3 : s1_4 + 2^64 * s2_3 = b' * lhs.l3 := by
    rw [e_s1_4, e_s2_3]; exact Nat.mod_add_div _ _
  -- r3_2: adcx {ee}, {s1}
  extract_lets -merge +onlyGivenNames s_6 r3_2 cf_4 at hr
  have e_r3_2 : r3_2 = (addc r3_1 s1_4 cf_3).1 := rfl
  have e_cf_4 : cf_4 = (addc r3_1 s1_4 cf_3).2 := rfl
  have l_r3_2 : r3_2 + 2^64 * cf_4 = r3_1 + s1_4 + cf_3 := by
    rw [e_r3_2, e_cf_4]; exact addc_lin r3_1 s1_4 cf_3
  have b_r3_2 : r3_2 < 2^64 := by rw [e_r3_2]; exact addc_value_lt r3_1 s1_4 cf_3
  have b_cf_4 : cf_4 ≤ 1 := by rw [e_cf_4]; exact addc_carry_le_one r3_1 s1_4 cf_3 b_r3_1 b_s1_4 b_cf_3
  clear e_r3_2 e_cf_4
  -- r4_1: adox {ae}, {s2}
  extract_lets -merge +onlyGivenNames s_7 r4_1 ofl_4 at hr
  have e_r4_1 : r4_1 = (addc r4 s2_3 ofl_3).1 := rfl
  have e_ofl_4 : ofl_4 = (addc r4 s2_3 ofl_3).2 := rfl
  have l_r4_1 : r4_1 + 2^64 * ofl_4 = r4 + s2_3 + ofl_3 := by
    rw [e_r4_1, e_ofl_4]; exact addc_lin r4 s2_3 ofl_3
  have b_r4_1 : r4_1 < 2^64 := by rw [e_r4_1]; exact addc_value_lt r4 s2_3 ofl_3
  have b_ofl_4 : ofl_4 ≤ 1 := by rw [e_ofl_4]; exact addc_carry_le_one r4 s2_3 ofl_3 b_r4 b_s2_3 b_ofl_3
  clear e_r4_1 e_ofl_4
  -- s1_5: mov {s1}, 0
  extract_lets -merge +onlyGivenNames s1_5 at hr
  have e_s1_5 : s1_5 = 0 := rfl
  have b_s1_5 : s1_5 < 2^64 := by rw [e_s1_5]; decide
  -- r4_2: adcx {ae}, {s1}
  extract_lets -merge +onlyGivenNames s_8 r4_2 cf_5 at hr
  have e_r4_2 : r4_2 = (addc r4_1 s1_5 cf_4).1 := rfl
  have e_cf_5 : cf_5 = (addc r4_1 s1_5 cf_4).2 := rfl
  have l_r4_2 : r4_2 + 2^64 * cf_5 = r4_1 + s1_5 + cf_4 := by
    rw [e_r4_2, e_cf_5]; exact addc_lin r4_1 s1_5 cf_4
  have b_r4_2 : r4_2 < 2^64 := by rw [e_r4_2]; exact addc_value_lt r4_1 s1_5 cf_4
  have b_cf_5 : cf_5 ≤ 1 := by rw [e_cf_5]; exact addc_carry_le_one r4_1 s1_5 cf_4 b_r4_1 b_s1_5 b_cf_4
  clear e_r4_2 e_cf_5
  -- r4_3: adox {ae}, {s1}
  extract_lets -merge +onlyGivenNames s_9 r4_3 ofl_5 at hr
  have e_r4_3 : r4_3 = (addc r4_2 s1_5 ofl_4).1 := rfl
  have e_ofl_5 : ofl_5 = (addc r4_2 s1_5 ofl_4).2 := rfl
  have l_r4_3 : r4_3 + 2^64 * ofl_5 = r4_2 + s1_5 + ofl_4 := by
    rw [e_r4_3, e_ofl_5]; exact addc_lin r4_2 s1_5 ofl_4
  have b_r4_3 : r4_3 < 2^64 := by rw [e_r4_3]; exact addc_value_lt r4_2 s1_5 ofl_4
  have b_ofl_5 : ofl_5 ≤ 1 := by rw [e_ofl_5]; exact addc_carry_le_one r4_2 s1_5 ofl_4 b_r4_2 b_s1_5 b_ofl_4
  clear e_r4_3 e_ofl_5
  -- BEGIN round product fold
  clear_value inv' b' r0 r1 r2 r3 r4 s1 s1_1 s2 cf_1 r0_1 ofl_1 r1_1 s1_2 s2_1 cf_2 r1_2 ofl_2 r2_1 s1_3 s2_2 cf_3 r2_2 ofl_3 r3_1 s1_4 s2_3 cf_4 r3_2 ofl_4 r4_1 cf_5 r4_2 ofl_5 r4_3
  have hA : acc.toNat = r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3 + 2^256 * r4 := by
    simp only [MulMontAcc.toNat, e_r0, e_r1, e_r2, e_r3, e_r4]
  have hL : lhs.toNat * b
      = b' * lhs.l0 + 2^64 * (b' * lhs.l1) + 2^128 * (b' * lhs.l2)
        + 2^192 * (b' * lhs.l3) := by
    rw [e_b']; simp only [Limbs.toNat]; ring
  have hpre : (r0_1 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + 2^256 * r4_2)
        + 2^320 * (ofl_4 + cf_5) = acc.toNat + lhs.toNat * b := by
    rw [hA, hL]
    clear * - e_s1 d_s2 d_s2_1 d_s2_2 d_s2_3 l_r0_1 l_r1_1 l_r1_2
      l_r2_1 l_r2_2 l_r3_1 l_r3_2 l_r4_1 l_r4_2
    omega
  have zofl4 : ofl_4 = 0 := by
    clear * - hpre hH1
    omega
  have hsum : (r0_1 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + 2^256 * r4_3)
        + 2^320 * (cf_5 + ofl_5)
      = (r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3 + 2^256 * r4)
        + (s1_1 + 2^64 * s1_2 + 2^128 * s1_3 + 2^192 * s1_4)
        + 2^64 * (s2 + 2^64 * s2_1 + 2^128 * s2_2 + 2^192 * s2_3) := by
    clear * - e_s1 l_r0_1 l_r1_1 l_r1_2 l_r2_1 l_r2_2 l_r3_1 l_r3_2 l_r4_1
      l_r4_2 l_r4_3 zofl4
    omega
  have hmul : (s1_1 + 2^64 * s1_2 + 2^128 * s1_3 + 2^192 * s1_4)
        + 2^64 * (s2 + 2^64 * s2_1 + 2^128 * s2_2 + 2^192 * s2_3)
      = lhs.toNat * b := by
    clear * - hL d_s2 d_s2_1 d_s2_2 d_s2_3
    omega
  have hkprod : cf_5 = 0 ∧ ofl_5 = 0 := by
    clear * - hsum hmul hA hH1 b_cf_5 b_ofl_5 b_r0_1 b_r1_2 b_r2_2 b_r3_2 b_r4_3
    omega
  have Fprod : r0_1 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + 2^256 * r4_3
      = acc.toNat + lhs.toNat * b := by
    clear * - hsum hmul hA hkprod
    omega
  -- END round product fold
  -- rdx: mov rdx, {be}
  extract_lets -merge +onlyGivenNames rdx at hr
  have e_rdx : rdx = r0_1 := rfl
  have b_rdx : rdx < 2^64 := by rw [e_rdx]; exact b_r0_1
  -- rdx_1: imul rdx, {inv}
  extract_lets -merge +onlyGivenNames rdx_1 at hr
  have e_rdx_1 : rdx_1 = rdx * inv' % 2^64 := rfl
  have b_rdx_1 : rdx_1 < 2^64 := by rw [e_rdx_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- BEGIN round cancellation quotient
  have hq : rdx_1 = inv * r0_1 % 2^64 := by rw [e_rdx_1, e_rdx, e_inv']; ring_nf
  -- END round cancellation quotient
  -- m_4: mulx {s2}, {s1}, qword ptr [{p} + 8]
  extract_lets -merge +onlyGivenNames m_4 s2_4 s1_6 at hr
  have e_s2_4 : s2_4 = (mulx rdx_1 modulus.l1).1 := rfl
  have e_s1_6 : s1_6 = (mulx rdx_1 modulus.l1).2 := rfl
  have b_s2_4 : s2_4 < 2^64 := by rw [e_s2_4]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_1 hm.2.1)
  have b_s1_6 : s1_6 < 2^64 := by rw [e_s1_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_s2_4 : s1_6 + 2^64 * s2_4 = rdx_1 * modulus.l1 := by
    rw [e_s1_6, e_s2_4]; exact Nat.mod_add_div _ _
  -- s3: mov {s3}, rdx
  extract_lets -merge +onlyGivenNames s3 at hr
  have e_s3 : s3 = rdx_1 := rfl
  have b_s3 : s3 < 2^64 := by rw [e_s3]; exact b_rdx_1
  -- s3_1: shl {s3}, 62
  extract_lets -merge +onlyGivenNames s3_1 at hr
  have e_s3_1 : s3_1 = s3 * 2^62 % 2^64 := rfl
  have b_s3_1 : s3_1 < 2^64 := by rw [e_s3_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_s3_1 : s3_1 + 2^64 * (s3 / 2^2) = s3 * 2^62 := by
    rw [e_s3_1]; exact lsl62_lsr2_split _
  -- n: neg {be}
  extract_lets -merge +onlyGivenNames n r0_2 cf_6 at hr
  have e_r0_2 : r0_2 = (neg r0_1).1 := rfl
  have e_cf_6 : cf_6 = (neg r0_1).2 := rfl
  have b_r0_2 : r0_2 < 2^64 := by rw [e_r0_2]; exact sbb_value_lt 0 r0_1 0
  have b_cf_6 : cf_6 ≤ 1 := by
    rw [e_cf_6]; simp only [neg]; split <;> omega
  -- r1_3: adc {ce}, {s1}
  extract_lets -merge +onlyGivenNames s_10 r1_3 cf_7 at hr
  have e_r1_3 : r1_3 = (addc r1_2 s1_6 cf_6).1 := rfl
  have e_cf_7 : cf_7 = (addc r1_2 s1_6 cf_6).2 := rfl
  have l_r1_3 : r1_3 + 2^64 * cf_7 = r1_2 + s1_6 + cf_6 := by
    rw [e_r1_3, e_cf_7]; exact addc_lin r1_2 s1_6 cf_6
  have b_r1_3 : r1_3 < 2^64 := by rw [e_r1_3]; exact addc_value_lt r1_2 s1_6 cf_6
  have b_cf_7 : cf_7 ≤ 1 := by rw [e_cf_7]; exact addc_carry_le_one r1_2 s1_6 cf_6 b_r1_2 b_s1_6 b_cf_6
  clear e_r1_3 e_cf_7
  -- r2_3: adc {de}, 0
  extract_lets -merge +onlyGivenNames s_11 r2_3 cf_8 at hr
  have e_r2_3 : r2_3 = (addc r2_2 0 cf_7).1 := rfl
  have e_cf_8 : cf_8 = (addc r2_2 0 cf_7).2 := rfl
  have l_r2_3 : r2_3 + 2^64 * cf_8 = r2_2 + 0 + cf_7 := by
    rw [e_r2_3, e_cf_8]; exact addc_lin r2_2 0 cf_7
  have b_r2_3 : r2_3 < 2^64 := by rw [e_r2_3]; exact addc_value_lt r2_2 0 cf_7
  have b_cf_8 : cf_8 ≤ 1 := by rw [e_cf_8]; exact addc_carry_le_one r2_2 0 cf_7 b_r2_2 (by decide) b_cf_7
  clear e_r2_3 e_cf_8
  -- r3_3: adc {ee}, {s3}
  extract_lets -merge +onlyGivenNames s_12 r3_3 cf_9 at hr
  have e_r3_3 : r3_3 = (addc r3_2 s3_1 cf_8).1 := rfl
  have e_cf_9 : cf_9 = (addc r3_2 s3_1 cf_8).2 := rfl
  have l_r3_3 : r3_3 + 2^64 * cf_9 = r3_2 + s3_1 + cf_8 := by
    rw [e_r3_3, e_cf_9]; exact addc_lin r3_2 s3_1 cf_8
  have b_r3_3 : r3_3 < 2^64 := by rw [e_r3_3]; exact addc_value_lt r3_2 s3_1 cf_8
  have b_cf_9 : cf_9 ≤ 1 := by rw [e_cf_9]; exact addc_carry_le_one r3_2 s3_1 cf_8 b_r3_2 b_s3_1 b_cf_8
  clear e_r3_3 e_cf_9
  -- r4_4: adc {ae}, 0
  extract_lets -merge +onlyGivenNames s_13 r4_4 cf_10 at hr
  have e_r4_4 : r4_4 = (addc r4_3 0 cf_9).1 := rfl
  have e_cf_10 : cf_10 = (addc r4_3 0 cf_9).2 := rfl
  have l_r4_4 : r4_4 + 2^64 * cf_10 = r4_3 + 0 + cf_9 := by
    rw [e_r4_4, e_cf_10]; exact addc_lin r4_3 0 cf_9
  have b_r4_4 : r4_4 < 2^64 := by rw [e_r4_4]; exact addc_value_lt r4_3 0 cf_9
  have b_cf_10 : cf_10 ≤ 1 := by rw [e_cf_10]; exact addc_carry_le_one r4_3 0 cf_9 b_r4_3 (by decide) b_cf_9
  clear e_r4_4 e_cf_10
  -- m_5: mulx {s1}, {s3}, qword ptr [{p}]
  extract_lets -merge +onlyGivenNames m_5 s1_7 s3_2 at hr
  have e_s1_7 : s1_7 = (mulx rdx_1 modulus.l0).1 := rfl
  have e_s3_2 : s3_2 = (mulx rdx_1 modulus.l0).2 := rfl
  have b_s1_7 : s1_7 < 2^64 := by rw [e_s1_7]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_1 hm.1)
  have b_s3_2 : s3_2 < 2^64 := by rw [e_s3_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_s1_7 : s3_2 + 2^64 * s1_7 = rdx_1 * modulus.l0 := by
    rw [e_s3_2, e_s1_7]; exact Nat.mod_add_div _ _
  -- s3_3: mov {s3}, rdx
  extract_lets -merge +onlyGivenNames s3_3 at hr
  have e_s3_3 : s3_3 = rdx_1 := rfl
  have b_s3_3 : s3_3 < 2^64 := by rw [e_s3_3]; exact b_rdx_1
  -- s3_4: shr {s3}, 2
  extract_lets -merge +onlyGivenNames s3_4 at hr
  have e_s3_4 : s3_4 = s3_3 / 2^2 := rfl
  have b_s3_4 : s3_4 < 2^62 := by
    rw [e_s3_4]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_s3_3 (by norm_num))
  -- r0_3: mov {be}, 0
  extract_lets -merge +onlyGivenNames r0_3 at hr
  have e_r0_3 : r0_3 = 0 := rfl
  have b_r0_3 : r0_3 < 2^64 := by rw [e_r0_3]; decide
  -- r1_4: add {ce}, {s1}
  extract_lets -merge +onlyGivenNames s_14 r1_4 cf_11 at hr
  have e_r1_4 : r1_4 = (addc r1_3 s1_7 0).1 := rfl
  have e_cf_11 : cf_11 = (addc r1_3 s1_7 0).2 := rfl
  have l_r1_4 : r1_4 + 2^64 * cf_11 = r1_3 + s1_7 + 0 := by
    rw [e_r1_4, e_cf_11]; exact addc_lin r1_3 s1_7 0
  have b_r1_4 : r1_4 < 2^64 := by rw [e_r1_4]; exact addc_value_lt r1_3 s1_7 0
  have b_cf_11 : cf_11 ≤ 1 := by rw [e_cf_11]; exact addc_carry_le_one r1_3 s1_7 0 b_r1_3 b_s1_7 (by decide)
  clear e_r1_4 e_cf_11
  -- r2_4: adc {de}, {s2}
  extract_lets -merge +onlyGivenNames s_15 r2_4 cf_12 at hr
  have e_r2_4 : r2_4 = (addc r2_3 s2_4 cf_11).1 := rfl
  have e_cf_12 : cf_12 = (addc r2_3 s2_4 cf_11).2 := rfl
  have l_r2_4 : r2_4 + 2^64 * cf_12 = r2_3 + s2_4 + cf_11 := by
    rw [e_r2_4, e_cf_12]; exact addc_lin r2_3 s2_4 cf_11
  have b_r2_4 : r2_4 < 2^64 := by rw [e_r2_4]; exact addc_value_lt r2_3 s2_4 cf_11
  have b_cf_12 : cf_12 ≤ 1 := by rw [e_cf_12]; exact addc_carry_le_one r2_3 s2_4 cf_11 b_r2_3 b_s2_4 b_cf_11
  clear e_r2_4 e_cf_12
  -- r3_4: adc {ee}, 0
  extract_lets -merge +onlyGivenNames s_16 r3_4 cf_13 at hr
  have e_r3_4 : r3_4 = (addc r3_3 0 cf_12).1 := rfl
  have e_cf_13 : cf_13 = (addc r3_3 0 cf_12).2 := rfl
  have l_r3_4 : r3_4 + 2^64 * cf_13 = r3_3 + 0 + cf_12 := by
    rw [e_r3_4, e_cf_13]; exact addc_lin r3_3 0 cf_12
  have b_r3_4 : r3_4 < 2^64 := by rw [e_r3_4]; exact addc_value_lt r3_3 0 cf_12
  have b_cf_13 : cf_13 ≤ 1 := by rw [e_cf_13]; exact addc_carry_le_one r3_3 0 cf_12 b_r3_3 (by decide) b_cf_12
  clear e_r3_4 e_cf_13
  -- r4_5: adc {ae}, {s3}
  extract_lets -merge +onlyGivenNames s_17 r4_5 cf_14 at hr
  have e_r4_5 : r4_5 = (addc r4_4 s3_4 cf_13).1 := rfl
  have e_cf_14 : cf_14 = (addc r4_4 s3_4 cf_13).2 := rfl
  have l_r4_5 : r4_5 + 2^64 * cf_14 = r4_4 + s3_4 + cf_13 := by
    rw [e_r4_5, e_cf_14]; exact addc_lin r4_4 s3_4 cf_13
  have b_r4_5 : r4_5 < 2^64 := by rw [e_r4_5]; exact addc_value_lt r4_4 s3_4 cf_13
  have b_cf_14 : cf_14 ≤ 1 := by rw [e_cf_14]; exact addc_carry_le_one r4_4 s3_4 cf_13 b_r4_4 (lt_of_lt_of_le b_s3_4 (by norm_num)) b_cf_13
  clear e_r4_5 e_cf_14
  -- r0_4: adc {be}, 0
  extract_lets -merge +onlyGivenNames s_18 r0_4 cf_15 at hr
  have e_r0_4 : r0_4 = (addc r0_3 0 cf_14).1 := rfl
  have e_cf_15 : cf_15 = (addc r0_3 0 cf_14).2 := rfl
  have l_r0_4 : r0_4 + 2^64 * cf_15 = r0_3 + 0 + cf_14 := by
    rw [e_r0_4, e_cf_15]; exact addc_lin r0_3 0 cf_14
  have b_r0_4 : r0_4 < 2^64 := by rw [e_r0_4]; exact addc_value_lt r0_3 0 cf_14
  have b_cf_15 : cf_15 ≤ 1 := by rw [e_cf_15]; exact addc_carry_le_one r0_3 0 cf_14 b_r0_3 (by decide) b_cf_14
  clear e_r0_4 e_cf_15
  -- BEGIN round reduction fold
  clear_value rdx rdx_1 s1_6 s2_4 s3 s3_1 cf_6 r0_2 cf_7 r1_3 cf_8 r2_3 cf_9 r3_3 cf_10 r4_4 s3_2 s1_7 s3_4 cf_11 r1_4 cf_12 r2_4 cf_13 r3_4 cf_14 r4_5 cf_15 r0_4
  have hc : r0_1 + s3_2 = 2^64 * cf_6 := by
    have h := neg_carry_cancel r0_1 inv modulus.l0 rdx_1 b_r0_1 hinv_lt hm.1
      hinv hq
    simpa only [e_s3_2, e_cf_6, mulx, neg] using h
  have hP : modulus.toNat = modulus.l0 + 2^64 * modulus.l1 + 2^192 * 2^62 := by
    simp only [Limbs.toNat, hshape.1, hshape.2, Nat.mul_zero, Nat.add_zero]
  have hqP : rdx_1 * modulus.toNat
      = rdx_1 * modulus.l0 + 2^64 * (rdx_1 * modulus.l1) + 2^192 * (rdx_1 * 2^62) := by
    rw [hP]; ring
  have hlow : (r0_1 + 2^64 * r1_3 + 2^128 * r2_3 + 2^192 * r3_3 + 2^256 * r4_4)
        + 2^320 * cf_10
      = (r0_1 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + 2^256 * r4_3)
        + 2^64 * s1_6 + 2^64 * cf_6 + 2^192 * s3_1 := by
    clear * - l_r1_3 l_r2_3 l_r3_3 l_r4_4
    omega
  have hhigh : (r1_4 + 2^64 * r2_4 + 2^128 * r3_4 + 2^192 * r4_5 + 2^256 * r0_4)
        + 2^320 * cf_15
      = (r1_3 + 2^64 * r2_3 + 2^128 * r3_3 + 2^192 * r4_4)
        + s1_7 + 2^64 * s2_4 + 2^192 * s3_4 := by
    clear * - l_r1_4 l_r2_4 l_r3_4 l_r4_5 l_r0_4 e_s1
    omega
  have hs3 : s3_1 ≤ 3 * 2^62 := by
    clear * - sh_s3_1 b_s3_1
    omega
  have hk : cf_10 = 0 ∧ cf_15 = 0 := by
    clear * - hlow Fprod hH1 l_r0_4 e_s1 b_cf_14 b_cf_6 b_s1_6 hs3
    omega
  have I : 2^64 * (r1_4 + 2^64 * r2_4 + 2^128 * r3_4 + 2^192 * r4_5 + 2^256 * r0_4)
      = acc.toNat + lhs.toNat * b + rdx_1 * modulus.toNat := by
    clear * - Fprod hc d_s2_4 d_s1_7 sh_s3_1 e_s3_4 e_s3_3 e_s3 hqP hlow hhigh hk
    omega
  -- END round reduction fold
  subst hr
  -- BEGIN round conclusion
  refine ⟨⟨b_r1_4, b_r2_4, b_r3_4, b_r4_5, b_r0_4, b_rdx_1⟩, ?_⟩
  simpa only [MulMontAcc.toNat] using I
  -- END round conclusion

-- BEGIN multiplication congruence helper
private theorem conclude_mod (out candidate input q p k : Nat)
    (h : 2^256 * candidate = input + q * p) (ho : out + k * p = candidate) :
    2^256 * out ≡ input [MOD p] := by
  apply modEq_of_add_mul _ _ (2^256 * k) q p
  calc
    2^256 * out + (2^256 * k) * p = 2^256 * (out + k * p) := by ring
    _ = _ := by rw [ho]; exact h
-- END multiplication congruence helper

-- BEGIN multiplication conclusion helper
private theorem mul_conclude {a0 a1 a2 a3 d0 d1 d2 d3 c r0 r1 r2 r3 input q p : Nat}
    (hi : 2^256 * (a0 + 2^64*a1 + 2^128*a2 + 2^192*a3) = input + q*p)
    (ha : a0 + 2^64*a1 + 2^128*a2 + 2^192*a3 < 2*p)
    (hd : d0 + 2^64*d1 + 2^128*d2 + 2^192*d3 + p = a0 + 2^64*a1 + 2^128*a2 + 2^192*a3 + 2^256*c)
    (bc : c ≤ 1) (bd0 : d0 < 2^64) (bd1 : d1 < 2^64) (bd2 : d2 < 2^64) (bd3 : d3 < 2^64)
    (e0 : r0 = if c = 0 then d0 else a0) (e1 : r1 = if c = 0 then d1 else a1)
    (e2 : r2 = if c = 0 then d2 else a2) (e3 : r3 = if c = 0 then d3 else a3) :
    r0 + 2^64*r1 + 2^128*r2 + 2^192*r3 < p ∧
    2^256*(r0 + 2^64*r1 + 2^128*r2 + 2^192*r3) ≡ input [MOD p] := by
  obtain hc | hc : c = 0 ∨ c = 1 := by omega
  · rw [if_pos hc] at e0 e1 e2 e3
    have ho : r0 + 2^64*r1 + 2^128*r2 + 2^192*r3 + p = a0 + 2^64*a1 + 2^128*a2 + 2^192*a3 := by omega
    refine ⟨by omega, conclude_mod _ _ _ _ _ 1 hi ?_⟩
    simpa only [Nat.one_mul] using ho
  · rw [if_neg (by omega)] at e0 e1 e2 e3
    have ho : r0 + 2^64*r1 + 2^128*r2 + 2^192*r3 = a0 + 2^64*a1 + 2^128*a2 + 2^192*a3 := by omega
    refine ⟨by omega, conclude_mod _ _ _ _ _ 0 hi ?_⟩
    simpa only [Nat.zero_mul, Nat.add_zero] using ho
-- END multiplication conclusion helper

-- BEGIN mulMont_spec statement
set_option exponentiation.threshold 512 in
/-- Montgomery multiplication under the round-safety and final reduction bounds. -/
theorem mulMont_spec (lhs rhs modulus : Limbs) (inv : Nat) (hlhs : lhs.Bounded)
    (hrhs : rhs.Bounded) (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinv_lt : inv < 2^64) (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hsafe : lhs.toNat * (rhs.l1 + 1) + modulus.toNat + 3 * 2^254 + 2^128 ≤ 2^320 ∧
      lhs.toNat * (rhs.l2 + 1) + modulus.toNat + 3 * 2^254 + 2^128 ≤ 2^320 ∧
      lhs.toNat * (rhs.l3 + 1) + modulus.toNat + 3 * 2^254 + 2^128 ≤ 2^320)
    (hfinal : lhs.toNat * rhs.toNat < 2^256 * modulus.toNat) :
    ∀ r : Limbs, r = mulMont lhs rhs modulus inv →
      r.Bounded ∧ r.toNat < modulus.toNat ∧
        2^256 * r.toNat ≡ lhs.toNat * rhs.toNat [MOD modulus.toNat] := by
  intro r hr
-- END mulMont_spec statement
  -- generated skeleton for `mulMont`: do not edit between the annotations
  unfold mulMont at hr
  lift_lets -merge at hr
  -- inv': scalar input
  extract_lets -merge +onlyGivenNames inv' at hr
  have e_inv' : inv' = inv := rfl
  have b_inv' : inv' < 2^64 := by rw [e_inv']; exact hinv_lt
  -- p3: operand p3 = const PASTA_HIGH_LIMB
  extract_lets -merge +onlyGivenNames p3 at hr
  have e_p3 : p3 = 4611686018427387904 := rfl
  have b_p3 : p3 < 2^64 := by rw [e_p3]; decide
  -- rdx: mov rdx, qword ptr [{b}]
  extract_lets -merge +onlyGivenNames rdx at hr
  have e_rdx : rdx = rhs.l0 := rfl
  have b_rdx : rdx < 2^64 := by rw [e_rdx]; exact hrhs.1
  -- m: mulx {be}, {ae}, qword ptr [{a}]
  extract_lets -merge +onlyGivenNames m be ae at hr
  have e_be : be = (mulx rdx lhs.l0).1 := rfl
  have e_ae : ae = (mulx rdx lhs.l0).2 := rfl
  have b_be : be < 2^64 := by rw [e_be]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx hlhs.1)
  have b_ae : ae < 2^64 := by rw [e_ae]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_be : ae + 2^64 * be = rdx * lhs.l0 := by
    rw [e_ae, e_be]; exact Nat.mod_add_div _ _
  -- m_1: mulx {ce}, {s1}, qword ptr [{a} + 8]
  extract_lets -merge +onlyGivenNames m_1 ce s1 at hr
  have e_ce : ce = (mulx rdx lhs.l1).1 := rfl
  have e_s1 : s1 = (mulx rdx lhs.l1).2 := rfl
  have b_ce : ce < 2^64 := by rw [e_ce]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx hlhs.2.1)
  have b_s1 : s1 < 2^64 := by rw [e_s1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_ce : s1 + 2^64 * ce = rdx * lhs.l1 := by
    rw [e_s1, e_ce]; exact Nat.mod_add_div _ _
  -- be_1: add {be}, {s1}
  extract_lets -merge +onlyGivenNames s be_1 cf at hr
  have e_be_1 : be_1 = (addc be s1 0).1 := rfl
  have e_cf : cf = (addc be s1 0).2 := rfl
  have l_be_1 : be_1 + 2^64 * cf = be + s1 + 0 := by
    rw [e_be_1, e_cf]; exact addc_lin be s1 0
  have b_be_1 : be_1 < 2^64 := by rw [e_be_1]; exact addc_value_lt be s1 0
  have b_cf : cf ≤ 1 := by rw [e_cf]; exact addc_carry_le_one be s1 0 b_be b_s1 (by decide)
  clear e_be_1 e_cf
  -- m_2: mulx {de}, {s1}, qword ptr [{a} + 16]
  extract_lets -merge +onlyGivenNames m_2 de s1_1 at hr
  have e_de : de = (mulx rdx lhs.l2).1 := rfl
  have e_s1_1 : s1_1 = (mulx rdx lhs.l2).2 := rfl
  have b_de : de < 2^64 := by rw [e_de]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx hlhs.2.2.1)
  have b_s1_1 : s1_1 < 2^64 := by rw [e_s1_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_de : s1_1 + 2^64 * de = rdx * lhs.l2 := by
    rw [e_s1_1, e_de]; exact Nat.mod_add_div _ _
  -- ce_1: adc {ce}, {s1}
  extract_lets -merge +onlyGivenNames s_1 ce_1 cf_1 at hr
  have e_ce_1 : ce_1 = (addc ce s1_1 cf).1 := rfl
  have e_cf_1 : cf_1 = (addc ce s1_1 cf).2 := rfl
  have l_ce_1 : ce_1 + 2^64 * cf_1 = ce + s1_1 + cf := by
    rw [e_ce_1, e_cf_1]; exact addc_lin ce s1_1 cf
  have b_ce_1 : ce_1 < 2^64 := by rw [e_ce_1]; exact addc_value_lt ce s1_1 cf
  have b_cf_1 : cf_1 ≤ 1 := by rw [e_cf_1]; exact addc_carry_le_one ce s1_1 cf b_ce b_s1_1 b_cf
  clear e_ce_1 e_cf_1
  -- m_3: mulx {ee}, {s1}, qword ptr [{a} + 24]
  extract_lets -merge +onlyGivenNames m_3 ee s1_2 at hr
  have e_ee : ee = (mulx rdx lhs.l3).1 := rfl
  have e_s1_2 : s1_2 = (mulx rdx lhs.l3).2 := rfl
  have b_ee : ee < 2^64 := by rw [e_ee]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx hlhs.2.2.2)
  have b_s1_2 : s1_2 < 2^64 := by rw [e_s1_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_ee : s1_2 + 2^64 * ee = rdx * lhs.l3 := by
    rw [e_s1_2, e_ee]; exact Nat.mod_add_div _ _
  -- de_1: adc {de}, {s1}
  extract_lets -merge +onlyGivenNames s_2 de_1 cf_2 at hr
  have e_de_1 : de_1 = (addc de s1_2 cf_1).1 := rfl
  have e_cf_2 : cf_2 = (addc de s1_2 cf_1).2 := rfl
  have l_de_1 : de_1 + 2^64 * cf_2 = de + s1_2 + cf_1 := by
    rw [e_de_1, e_cf_2]; exact addc_lin de s1_2 cf_1
  have b_de_1 : de_1 < 2^64 := by rw [e_de_1]; exact addc_value_lt de s1_2 cf_1
  have b_cf_2 : cf_2 ≤ 1 := by rw [e_cf_2]; exact addc_carry_le_one de s1_2 cf_1 b_de b_s1_2 b_cf_1
  clear e_de_1 e_cf_2
  -- ee_1: adc {ee}, 0
  extract_lets -merge +onlyGivenNames s_3 ee_1 cf_3 at hr
  have e_ee_1 : ee_1 = (addc ee 0 cf_2).1 := rfl
  have e_cf_3 : cf_3 = (addc ee 0 cf_2).2 := rfl
  have l_ee_1 : ee_1 + 2^64 * cf_3 = ee + 0 + cf_2 := by
    rw [e_ee_1, e_cf_3]; exact addc_lin ee 0 cf_2
  have b_ee_1 : ee_1 < 2^64 := by rw [e_ee_1]; exact addc_value_lt ee 0 cf_2
  have b_cf_3 : cf_3 ≤ 1 := by rw [e_cf_3]; exact addc_carry_le_one ee 0 cf_2 b_ee (by decide) b_cf_2
  clear e_ee_1 e_cf_3
  -- BEGIN initial product
  clear_value inv' p3 rdx m be ae m_1 ce s1 s be_1 cf m_2 de s1_1 s_1 ce_1 cf_1 m_3 ee s1_2 s_2 de_1 cf_2 s_3 ee_1 cf_3
  have hL : lhs.toNat * rhs.l0 = rdx * lhs.l0 + 2^64 * (rdx * lhs.l1) +
      2^128 * (rdx * lhs.l2) + 2^192 * (rdx * lhs.l3) := by
    rw [e_rdx]; simp only [Limbs.toNat]; ring
  have Fmul : ae + 2^64 * be_1 + 2^128 * ce_1 + 2^192 * de_1 + 2^256 * ee_1 +
      2^320 * cf_3 = lhs.toNat * rhs.l0 := by
    clear * - hL d_be d_ce d_de d_ee l_be_1 l_ce_1 l_de_1 l_ee_1
    omega
  have hprod_lt : lhs.toNat * rhs.l0 < 2^320 := by
    have h := Nat.mul_lt_mul'' (Limbs.toNat_lt lhs hlhs) hrhs.1
    norm_num at h ⊢
    exact h
  have zcf3 : cf_3 = 0 := by
    clear * - Fmul hprod_lt
    omega
  -- END initial product
  -- rdx_1: mov rdx, {ae}
  extract_lets -merge +onlyGivenNames rdx_1 at hr
  have e_rdx_1 : rdx_1 = ae := rfl
  have b_rdx_1 : rdx_1 < 2^64 := by rw [e_rdx_1]; exact b_ae
  -- rdx_2: imul rdx, {inv}
  extract_lets -merge +onlyGivenNames rdx_2 at hr
  have e_rdx_2 : rdx_2 = rdx_1 * inv' % 2^64 := rfl
  have b_rdx_2 : rdx_2 < 2^64 := by rw [e_rdx_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- m_4: mulx {s2}, {s1}, qword ptr [{p} + 8]
  extract_lets -merge +onlyGivenNames m_4 s2 s1_3 at hr
  have e_s2 : s2 = (mulx rdx_2 modulus.l1).1 := rfl
  have e_s1_3 : s1_3 = (mulx rdx_2 modulus.l1).2 := rfl
  have b_s2 : s2 < 2^64 := by rw [e_s2]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_2 hm.2.1)
  have b_s1_3 : s1_3 < 2^64 := by rw [e_s1_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_s2 : s1_3 + 2^64 * s2 = rdx_2 * modulus.l1 := by
    rw [e_s1_3, e_s2]; exact Nat.mod_add_div _ _
  -- s3: mov {s3}, rdx
  extract_lets -merge +onlyGivenNames s3 at hr
  have e_s3 : s3 = rdx_2 := rfl
  have b_s3 : s3 < 2^64 := by rw [e_s3]; exact b_rdx_2
  -- s3_1: shl {s3}, 62
  extract_lets -merge +onlyGivenNames s3_1 at hr
  have e_s3_1 : s3_1 = s3 * 2^62 % 2^64 := rfl
  have b_s3_1 : s3_1 < 2^64 := by rw [e_s3_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_s3_1 : s3_1 + 2^64 * (s3 / 2^2) = s3 * 2^62 := by
    rw [e_s3_1]; exact lsl62_lsr2_split _
  -- n: neg {ae}
  extract_lets -merge +onlyGivenNames n ae_1 cf_4 at hr
  have e_ae_1 : ae_1 = (neg ae).1 := rfl
  have e_cf_4 : cf_4 = (neg ae).2 := rfl
  have b_ae_1 : ae_1 < 2^64 := by rw [e_ae_1]; exact sbb_value_lt 0 ae 0
  have b_cf_4 : cf_4 ≤ 1 := by
    rw [e_cf_4]; simp only [neg]; split <;> omega
  -- be_2: adc {be}, {s1}
  extract_lets -merge +onlyGivenNames s_4 be_2 cf_5 at hr
  have e_be_2 : be_2 = (addc be_1 s1_3 cf_4).1 := rfl
  have e_cf_5 : cf_5 = (addc be_1 s1_3 cf_4).2 := rfl
  have l_be_2 : be_2 + 2^64 * cf_5 = be_1 + s1_3 + cf_4 := by
    rw [e_be_2, e_cf_5]; exact addc_lin be_1 s1_3 cf_4
  have b_be_2 : be_2 < 2^64 := by rw [e_be_2]; exact addc_value_lt be_1 s1_3 cf_4
  have b_cf_5 : cf_5 ≤ 1 := by rw [e_cf_5]; exact addc_carry_le_one be_1 s1_3 cf_4 b_be_1 b_s1_3 b_cf_4
  clear e_be_2 e_cf_5
  -- ce_2: adc {ce}, 0
  extract_lets -merge +onlyGivenNames s_5 ce_2 cf_6 at hr
  have e_ce_2 : ce_2 = (addc ce_1 0 cf_5).1 := rfl
  have e_cf_6 : cf_6 = (addc ce_1 0 cf_5).2 := rfl
  have l_ce_2 : ce_2 + 2^64 * cf_6 = ce_1 + 0 + cf_5 := by
    rw [e_ce_2, e_cf_6]; exact addc_lin ce_1 0 cf_5
  have b_ce_2 : ce_2 < 2^64 := by rw [e_ce_2]; exact addc_value_lt ce_1 0 cf_5
  have b_cf_6 : cf_6 ≤ 1 := by rw [e_cf_6]; exact addc_carry_le_one ce_1 0 cf_5 b_ce_1 (by decide) b_cf_5
  clear e_ce_2 e_cf_6
  -- de_2: adc {de}, {s3}
  extract_lets -merge +onlyGivenNames s_6 de_2 cf_7 at hr
  have e_de_2 : de_2 = (addc de_1 s3_1 cf_6).1 := rfl
  have e_cf_7 : cf_7 = (addc de_1 s3_1 cf_6).2 := rfl
  have l_de_2 : de_2 + 2^64 * cf_7 = de_1 + s3_1 + cf_6 := by
    rw [e_de_2, e_cf_7]; exact addc_lin de_1 s3_1 cf_6
  have b_de_2 : de_2 < 2^64 := by rw [e_de_2]; exact addc_value_lt de_1 s3_1 cf_6
  have b_cf_7 : cf_7 ≤ 1 := by rw [e_cf_7]; exact addc_carry_le_one de_1 s3_1 cf_6 b_de_1 b_s3_1 b_cf_6
  clear e_de_2 e_cf_7
  -- ee_2: adc {ee}, 0
  extract_lets -merge +onlyGivenNames s_7 ee_2 cf_8 at hr
  have e_ee_2 : ee_2 = (addc ee_1 0 cf_7).1 := rfl
  have e_cf_8 : cf_8 = (addc ee_1 0 cf_7).2 := rfl
  have l_ee_2 : ee_2 + 2^64 * cf_8 = ee_1 + 0 + cf_7 := by
    rw [e_ee_2, e_cf_8]; exact addc_lin ee_1 0 cf_7
  have b_ee_2 : ee_2 < 2^64 := by rw [e_ee_2]; exact addc_value_lt ee_1 0 cf_7
  have b_cf_8 : cf_8 ≤ 1 := by rw [e_cf_8]; exact addc_carry_le_one ee_1 0 cf_7 b_ee_1 (by decide) b_cf_7
  clear e_ee_2 e_cf_8
  -- m_5: mulx {s1}, {s3}, qword ptr [{p}]
  extract_lets -merge +onlyGivenNames m_5 s1_4 s3_2 at hr
  have e_s1_4 : s1_4 = (mulx rdx_2 modulus.l0).1 := rfl
  have e_s3_2 : s3_2 = (mulx rdx_2 modulus.l0).2 := rfl
  have b_s1_4 : s1_4 < 2^64 := by rw [e_s1_4]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_2 hm.1)
  have b_s3_2 : s3_2 < 2^64 := by rw [e_s3_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_s1_4 : s3_2 + 2^64 * s1_4 = rdx_2 * modulus.l0 := by
    rw [e_s3_2, e_s1_4]; exact Nat.mod_add_div _ _
  -- s3_3: mov {s3}, rdx
  extract_lets -merge +onlyGivenNames s3_3 at hr
  have e_s3_3 : s3_3 = rdx_2 := rfl
  have b_s3_3 : s3_3 < 2^64 := by rw [e_s3_3]; exact b_rdx_2
  -- s3_4: shr {s3}, 2
  extract_lets -merge +onlyGivenNames s3_4 at hr
  have e_s3_4 : s3_4 = s3_3 / 2^2 := rfl
  have b_s3_4 : s3_4 < 2^62 := by
    rw [e_s3_4]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_s3_3 (by norm_num))
  -- ae_2: mov {ae}, 0
  extract_lets -merge +onlyGivenNames ae_2 at hr
  have e_ae_2 : ae_2 = 0 := rfl
  have b_ae_2 : ae_2 < 2^64 := by rw [e_ae_2]; decide
  -- be_3: add {be}, {s1}
  extract_lets -merge +onlyGivenNames s_8 be_3 cf_9 at hr
  have e_be_3 : be_3 = (addc be_2 s1_4 0).1 := rfl
  have e_cf_9 : cf_9 = (addc be_2 s1_4 0).2 := rfl
  have l_be_3 : be_3 + 2^64 * cf_9 = be_2 + s1_4 + 0 := by
    rw [e_be_3, e_cf_9]; exact addc_lin be_2 s1_4 0
  have b_be_3 : be_3 < 2^64 := by rw [e_be_3]; exact addc_value_lt be_2 s1_4 0
  have b_cf_9 : cf_9 ≤ 1 := by rw [e_cf_9]; exact addc_carry_le_one be_2 s1_4 0 b_be_2 b_s1_4 (by decide)
  clear e_be_3 e_cf_9
  -- ce_3: adc {ce}, {s2}
  extract_lets -merge +onlyGivenNames s_9 ce_3 cf_10 at hr
  have e_ce_3 : ce_3 = (addc ce_2 s2 cf_9).1 := rfl
  have e_cf_10 : cf_10 = (addc ce_2 s2 cf_9).2 := rfl
  have l_ce_3 : ce_3 + 2^64 * cf_10 = ce_2 + s2 + cf_9 := by
    rw [e_ce_3, e_cf_10]; exact addc_lin ce_2 s2 cf_9
  have b_ce_3 : ce_3 < 2^64 := by rw [e_ce_3]; exact addc_value_lt ce_2 s2 cf_9
  have b_cf_10 : cf_10 ≤ 1 := by rw [e_cf_10]; exact addc_carry_le_one ce_2 s2 cf_9 b_ce_2 b_s2 b_cf_9
  clear e_ce_3 e_cf_10
  -- de_3: adc {de}, 0
  extract_lets -merge +onlyGivenNames s_10 de_3 cf_11 at hr
  have e_de_3 : de_3 = (addc de_2 0 cf_10).1 := rfl
  have e_cf_11 : cf_11 = (addc de_2 0 cf_10).2 := rfl
  have l_de_3 : de_3 + 2^64 * cf_11 = de_2 + 0 + cf_10 := by
    rw [e_de_3, e_cf_11]; exact addc_lin de_2 0 cf_10
  have b_de_3 : de_3 < 2^64 := by rw [e_de_3]; exact addc_value_lt de_2 0 cf_10
  have b_cf_11 : cf_11 ≤ 1 := by rw [e_cf_11]; exact addc_carry_le_one de_2 0 cf_10 b_de_2 (by decide) b_cf_10
  clear e_de_3 e_cf_11
  -- ee_3: adc {ee}, {s3}
  extract_lets -merge +onlyGivenNames s_11 ee_3 cf_12 at hr
  have e_ee_3 : ee_3 = (addc ee_2 s3_4 cf_11).1 := rfl
  have e_cf_12 : cf_12 = (addc ee_2 s3_4 cf_11).2 := rfl
  have l_ee_3 : ee_3 + 2^64 * cf_12 = ee_2 + s3_4 + cf_11 := by
    rw [e_ee_3, e_cf_12]; exact addc_lin ee_2 s3_4 cf_11
  have b_ee_3 : ee_3 < 2^64 := by rw [e_ee_3]; exact addc_value_lt ee_2 s3_4 cf_11
  have b_cf_12 : cf_12 ≤ 1 := by rw [e_cf_12]; exact addc_carry_le_one ee_2 s3_4 cf_11 b_ee_2 (lt_of_lt_of_le b_s3_4 (by norm_num)) b_cf_11
  clear e_ee_3 e_cf_12
  -- ae_3: adc {ae}, 0
  extract_lets -merge +onlyGivenNames s_12 ae_3 cf_13 at hr
  have e_ae_3 : ae_3 = (addc ae_2 0 cf_12).1 := rfl
  have e_cf_13 : cf_13 = (addc ae_2 0 cf_12).2 := rfl
  have l_ae_3 : ae_3 + 2^64 * cf_13 = ae_2 + 0 + cf_12 := by
    rw [e_ae_3, e_cf_13]; exact addc_lin ae_2 0 cf_12
  have b_ae_3 : ae_3 < 2^64 := by rw [e_ae_3]; exact addc_value_lt ae_2 0 cf_12
  have b_cf_13 : cf_13 ≤ 1 := by rw [e_cf_13]; exact addc_carry_le_one ae_2 0 cf_12 b_ae_2 (by decide) b_cf_12
  clear e_ae_3 e_cf_13
  -- BEGIN initial reduction
  clear_value rdx_1 rdx_2 m_4 s2 s1_3 s3 s3_1 n ae_1 cf_4 s_4 be_2 cf_5 s_5 ce_2 cf_6 s_6 de_2 cf_7 s_7 ee_2 cf_8 m_5 s1_4 s3_2 s3_4 ae_2 s_8 be_3 cf_9 s_9 ce_3 cf_10 s_10 de_3 cf_11 s_11 ee_3 cf_12 s_12 ae_3 cf_13
  have hq0 : rdx_2 = mulLo inv ae := by
    rw [e_rdx_2, e_rdx_1, e_inv']; simp only [mulLo, Nat.mul_comm]
  have hc0 : ae + s3_2 = 2^64 * cf_4 := by
    have h := neg_carry_cancel ae inv modulus.l0 rdx_2 b_ae hinv_lt hm.1 hinv hq0
    simpa only [e_s3_2, e_cf_4, mulx, neg] using h
  have hp : modulus.toNat = modulus.l0 + 2^64 * modulus.l1 + 2^254 := by
    simp only [Limbs.toNat, hshape.1, hshape.2]; ring
  have hqp : rdx_2 * modulus.toNat = rdx_2 * modulus.l0 +
      2^64 * (rdx_2 * modulus.l1) + 2^192 * (rdx_2 * 2^62) := by rw [hp]; ring
  have htophigh : ee ≤ 2^64 - 2 := by
    have h := Nat.mul_le_mul (Nat.le_sub_one_of_lt b_rdx) (Nat.le_sub_one_of_lt hlhs.2.2.2)
    norm_num at h
    clear * - h d_ee
    omega
  have hprod_margin : lhs.toNat * rhs.l0 < 2^320 - 2^256 := by
    have h := Nat.mul_le_mul (Nat.le_sub_one_of_lt (Limbs.toNat_lt lhs hlhs))
      (Nat.le_sub_one_of_lt hrhs.1)
    norm_num at h ⊢
    clear * - h
    omega
  have hee1 : ee_1 ≤ 2^64 - 2 := by
    clear * - Fmul hprod_margin
    omega
  have zcf8 : cf_8 = 0 := by
    clear * - l_ee_2 hee1 b_cf_7
    omega
  have zcf13 : cf_13 = 0 := by
    clear * - l_ae_3 e_ae_2 b_cf_12
    omega
  have I0 : 2^64 * (be_3 + 2^64 * ce_3 + 2^128 * de_3 + 2^192 * ee_3 + 2^256 * ae_3)
      = lhs.toNat * rhs.l0 + rdx_2 * modulus.toNat := by
    clear * - Fmul zcf3 hc0 hqp d_s2 d_s1_4 sh_s3_1 e_s3_4 e_s3_3 e_s3
      l_be_2 l_ce_2 l_de_2 l_ee_2 l_be_3 l_ce_3 l_de_3 l_ee_3 l_ae_3
      e_ae_2 zcf8 zcf13
    omega
  have H0 : be_3 + 2^64 * ce_3 + 2^128 * de_3 + 2^192 * ee_3 + 2^256 * ae_3 <
      lhs.toNat + modulus.toNat := by
    have hb0 := Nat.mul_le_mul_left lhs.toNat (Nat.le_sub_one_of_lt hrhs.1)
    have hq := Nat.mul_le_mul_right modulus.toNat (Nat.le_sub_one_of_lt b_rdx_2)
    have hpos : 0 < modulus.toNat := by rw [hp]; clear * -; omega
    have hbl : rhs.l0 + 1 ≤ 2^64 := by have h := hrhs.1; clear * - h; omega
    have hql : rdx_2 + 1 ≤ 2^64 := by clear * - b_rdx_2; omega
    have h1 := Nat.mul_le_mul_left lhs.toNat hbl
    have h2 := Nat.mul_le_mul_right modulus.toNat hql
    simp only [Nat.mul_add, Nat.add_mul, Nat.mul_one, Nat.one_mul] at h1 h2
    clear * - I0 h1 h2 hpos
    omega
  -- END initial reduction
  -- round1: factored round 1
  extract_lets -merge +onlyGivenNames round1 at hr
  have e_round1 : round1 = mulMontRound lhs modulus inv' rhs.l1 ⟨be_3, ce_3, de_3, ee_3, ae_3, rdx_2⟩ := rfl
  -- BEGIN round 1 application
  clear_value round1
  have hs1 : (⟨be_3, ce_3, de_3, ee_3, ae_3, rdx_2⟩ : MulMontAcc).toNat +
      lhs.toNat * rhs.l1 + 2^128 + 3 * 2^254 < 2^320 := by
    have h := hsafe.1
    simp only [MulMontAcc.toNat, Nat.mul_add, Nat.mul_one] at h ⊢
    clear * - h H0
    omega
  obtain ⟨B1, I1⟩ := mulMontRound_spec lhs modulus inv' rhs.l1
    ⟨be_3, ce_3, de_3, ee_3, ae_3, rdx_2⟩ hlhs hm hshape b_inv'
    (by rw [e_inv']; exact hinv) hrhs.2.1
    ⟨b_be_3, b_ce_3, b_de_3, b_ee_3, b_ae_3, b_rdx_2⟩ hs1 round1 e_round1
  have H1 : round1.toNat < lhs.toNat + modulus.toNat := by
    have hb1 : rhs.l1 + 1 ≤ 2^64 := by have h := hrhs.2.1; clear * - h; omega
    have hq1 : round1.q + 1 ≤ 2^64 := by have h := B1.2.2.2.2.2; clear * - h; omega
    have h1 := Nat.mul_le_mul_left lhs.toNat hb1
    have h2 := Nat.mul_le_mul_right modulus.toNat hq1
    simp only [MulMontAcc.toNat] at I1
    simp only [Nat.mul_add, Nat.add_mul, Nat.mul_one, Nat.one_mul] at h1 h2
    change 2^64 * round1.toNat =
      (be_3 + 2^64 * ce_3 + 2^128 * de_3 + 2^192 * ee_3 + 2^256 * ae_3) +
        lhs.toNat * rhs.l1 + round1.q * modulus.toNat at I1
    clear * - I1 H0 h1 h2
    omega
  -- END round 1 application
  -- ce_4: round 1 output
  extract_lets -merge +onlyGivenNames ce_4 at hr
  have e_ce_4 : ce_4 = round1.r0 := rfl
  -- de_4: round 1 output
  extract_lets -merge +onlyGivenNames de_4 at hr
  have e_de_4 : de_4 = round1.r1 := rfl
  -- ee_4: round 1 output
  extract_lets -merge +onlyGivenNames ee_4 at hr
  have e_ee_4 : ee_4 = round1.r2 := rfl
  -- ae_4: round 1 output
  extract_lets -merge +onlyGivenNames ae_4 at hr
  have e_ae_4 : ae_4 = round1.r3 := rfl
  -- be_4: round 1 output
  extract_lets -merge +onlyGivenNames be_4 at hr
  have e_be_4 : be_4 = round1.r4 := rfl
  -- rdx_3: round 1 output
  extract_lets -merge +onlyGivenNames rdx_3 at hr
  have e_rdx_3 : rdx_3 = round1.q := rfl
  -- BEGIN round 1 outputs
  clear_value ce_4 de_4 ee_4 ae_4 be_4 rdx_3
  have b_ce_4 : ce_4 < 2^64 := by rw [e_ce_4]; exact B1.1
  have b_de_4 : de_4 < 2^64 := by rw [e_de_4]; exact B1.2.1
  have b_ee_4 : ee_4 < 2^64 := by rw [e_ee_4]; exact B1.2.2.1
  have b_ae_4 : ae_4 < 2^64 := by rw [e_ae_4]; exact B1.2.2.2.1
  have b_be_4 : be_4 < 2^64 := by rw [e_be_4]; exact B1.2.2.2.2.1
  have b_rdx_3 : rdx_3 < 2^64 := by rw [e_rdx_3]; exact B1.2.2.2.2.2
  have E1 : (⟨ce_4, de_4, ee_4, ae_4, be_4, rdx_3⟩ : MulMontAcc).toNat = round1.toNat := by
    simp only [MulMontAcc.toNat, e_ce_4, e_de_4, e_ee_4, e_ae_4, e_be_4]
  -- END round 1 outputs
  -- round2: factored round 2
  extract_lets -merge +onlyGivenNames round2 at hr
  have e_round2 : round2 = mulMontRound lhs modulus inv' rhs.l2 ⟨ce_4, de_4, ee_4, ae_4, be_4, rdx_3⟩ := rfl
  -- BEGIN round 2 application
  clear_value round2
  have hs2 : (⟨ce_4, de_4, ee_4, ae_4, be_4, rdx_3⟩ : MulMontAcc).toNat +
      lhs.toNat * rhs.l2 + 2^128 + 3 * 2^254 < 2^320 := by
    rw [E1]
    have h := hsafe.2.1
    simp only [Nat.mul_add, Nat.mul_one] at h
    clear * - h H1
    omega
  obtain ⟨B2, I2⟩ := mulMontRound_spec lhs modulus inv' rhs.l2
    ⟨ce_4, de_4, ee_4, ae_4, be_4, rdx_3⟩ hlhs hm hshape b_inv'
    (by rw [e_inv']; exact hinv) hrhs.2.2.1
    ⟨b_ce_4, b_de_4, b_ee_4, b_ae_4, b_be_4, b_rdx_3⟩ hs2 round2 e_round2
  rw [E1] at I2
  have H2 : round2.toNat < lhs.toNat + modulus.toNat := by
    have hb2 : rhs.l2 + 1 ≤ 2^64 := by have h := hrhs.2.2.1; clear * - h; omega
    have hq2 : round2.q + 1 ≤ 2^64 := by have h := B2.2.2.2.2.2; clear * - h; omega
    have h1 := Nat.mul_le_mul_left lhs.toNat hb2
    have h2 := Nat.mul_le_mul_right modulus.toNat hq2
    simp only [Nat.mul_add, Nat.add_mul, Nat.mul_one, Nat.one_mul] at h1 h2
    clear * - I2 H1 h1 h2
    omega
  -- END round 2 application
  -- de_5: round 2 output
  extract_lets -merge +onlyGivenNames de_5 at hr
  have e_de_5 : de_5 = round2.r0 := rfl
  -- ee_5: round 2 output
  extract_lets -merge +onlyGivenNames ee_5 at hr
  have e_ee_5 : ee_5 = round2.r1 := rfl
  -- ae_5: round 2 output
  extract_lets -merge +onlyGivenNames ae_5 at hr
  have e_ae_5 : ae_5 = round2.r2 := rfl
  -- be_5: round 2 output
  extract_lets -merge +onlyGivenNames be_5 at hr
  have e_be_5 : be_5 = round2.r3 := rfl
  -- ce_5: round 2 output
  extract_lets -merge +onlyGivenNames ce_5 at hr
  have e_ce_5 : ce_5 = round2.r4 := rfl
  -- rdx_4: round 2 output
  extract_lets -merge +onlyGivenNames rdx_4 at hr
  have e_rdx_4 : rdx_4 = round2.q := rfl
  -- BEGIN round 2 outputs
  clear_value de_5 ee_5 ae_5 be_5 ce_5 rdx_4
  have b_de_5 : de_5 < 2^64 := by rw [e_de_5]; exact B2.1
  have b_ee_5 : ee_5 < 2^64 := by rw [e_ee_5]; exact B2.2.1
  have b_ae_5 : ae_5 < 2^64 := by rw [e_ae_5]; exact B2.2.2.1
  have b_be_5 : be_5 < 2^64 := by rw [e_be_5]; exact B2.2.2.2.1
  have b_ce_5 : ce_5 < 2^64 := by rw [e_ce_5]; exact B2.2.2.2.2.1
  have b_rdx_4 : rdx_4 < 2^64 := by rw [e_rdx_4]; exact B2.2.2.2.2.2
  -- END round 2 outputs
  -- rdx_5: mov rdx, qword ptr [{b} + 24]
  extract_lets -merge +onlyGivenNames rdx_5 at hr
  have e_rdx_5 : rdx_5 = rhs.l3 := rfl
  have b_rdx_5 : rdx_5 < 2^64 := by rw [e_rdx_5]; exact hrhs.2.2.2
  -- s1_5: xor {s1}, {s1}
  extract_lets -merge +onlyGivenNames s1_5 cf_14 ofl at hr
  have e_s1_5 : s1_5 = 0 := rfl
  have e_cf_14 : cf_14 = 0 := rfl
  have e_ofl : ofl = 0 := rfl
  have b_s1_5 : s1_5 < 2^64 := by rw [e_s1_5]; decide
  have b_cf_14 : cf_14 ≤ 1 := by rw [e_cf_14]; decide
  have b_ofl : ofl ≤ 1 := by rw [e_ofl]; decide
  -- m_6: mulx {s2}, {s1}, qword ptr [{a}]
  extract_lets -merge +onlyGivenNames m_6 s2_1 s1_6 at hr
  have e_s2_1 : s2_1 = (mulx rdx_5 lhs.l0).1 := rfl
  have e_s1_6 : s1_6 = (mulx rdx_5 lhs.l0).2 := rfl
  have b_s2_1 : s2_1 < 2^64 := by rw [e_s2_1]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_5 hlhs.1)
  have b_s1_6 : s1_6 < 2^64 := by rw [e_s1_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_s2_1 : s1_6 + 2^64 * s2_1 = rdx_5 * lhs.l0 := by
    rw [e_s1_6, e_s2_1]; exact Nat.mod_add_div _ _
  -- de_6: adcx {de}, {s1}
  extract_lets -merge +onlyGivenNames s_13 de_6 cf_15 at hr
  have e_de_6 : de_6 = (addc de_5 s1_6 cf_14).1 := rfl
  have e_cf_15 : cf_15 = (addc de_5 s1_6 cf_14).2 := rfl
  have l_de_6 : de_6 + 2^64 * cf_15 = de_5 + s1_6 + cf_14 := by
    rw [e_de_6, e_cf_15]; exact addc_lin de_5 s1_6 cf_14
  have b_de_6 : de_6 < 2^64 := by rw [e_de_6]; exact addc_value_lt de_5 s1_6 cf_14
  have b_cf_15 : cf_15 ≤ 1 := by rw [e_cf_15]; exact addc_carry_le_one de_5 s1_6 cf_14 b_de_5 b_s1_6 b_cf_14
  clear e_de_6 e_cf_15
  -- ee_6: adox {ee}, {s2}
  extract_lets -merge +onlyGivenNames s_14 ee_6 ofl_1 at hr
  have e_ee_6 : ee_6 = (addc ee_5 s2_1 ofl).1 := rfl
  have e_ofl_1 : ofl_1 = (addc ee_5 s2_1 ofl).2 := rfl
  have l_ee_6 : ee_6 + 2^64 * ofl_1 = ee_5 + s2_1 + ofl := by
    rw [e_ee_6, e_ofl_1]; exact addc_lin ee_5 s2_1 ofl
  have b_ee_6 : ee_6 < 2^64 := by rw [e_ee_6]; exact addc_value_lt ee_5 s2_1 ofl
  have b_ofl_1 : ofl_1 ≤ 1 := by rw [e_ofl_1]; exact addc_carry_le_one ee_5 s2_1 ofl b_ee_5 b_s2_1 b_ofl
  clear e_ee_6 e_ofl_1
  -- m_7: mulx {s2}, {s1}, qword ptr [{a} + 8]
  extract_lets -merge +onlyGivenNames m_7 s2_2 s1_7 at hr
  have e_s2_2 : s2_2 = (mulx rdx_5 lhs.l1).1 := rfl
  have e_s1_7 : s1_7 = (mulx rdx_5 lhs.l1).2 := rfl
  have b_s2_2 : s2_2 < 2^64 := by rw [e_s2_2]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_5 hlhs.2.1)
  have b_s1_7 : s1_7 < 2^64 := by rw [e_s1_7]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_s2_2 : s1_7 + 2^64 * s2_2 = rdx_5 * lhs.l1 := by
    rw [e_s1_7, e_s2_2]; exact Nat.mod_add_div _ _
  -- ee_7: adcx {ee}, {s1}
  extract_lets -merge +onlyGivenNames s_15 ee_7 cf_16 at hr
  have e_ee_7 : ee_7 = (addc ee_6 s1_7 cf_15).1 := rfl
  have e_cf_16 : cf_16 = (addc ee_6 s1_7 cf_15).2 := rfl
  have l_ee_7 : ee_7 + 2^64 * cf_16 = ee_6 + s1_7 + cf_15 := by
    rw [e_ee_7, e_cf_16]; exact addc_lin ee_6 s1_7 cf_15
  have b_ee_7 : ee_7 < 2^64 := by rw [e_ee_7]; exact addc_value_lt ee_6 s1_7 cf_15
  have b_cf_16 : cf_16 ≤ 1 := by rw [e_cf_16]; exact addc_carry_le_one ee_6 s1_7 cf_15 b_ee_6 b_s1_7 b_cf_15
  clear e_ee_7 e_cf_16
  -- ae_6: adox {ae}, {s2}
  extract_lets -merge +onlyGivenNames s_16 ae_6 ofl_2 at hr
  have e_ae_6 : ae_6 = (addc ae_5 s2_2 ofl_1).1 := rfl
  have e_ofl_2 : ofl_2 = (addc ae_5 s2_2 ofl_1).2 := rfl
  have l_ae_6 : ae_6 + 2^64 * ofl_2 = ae_5 + s2_2 + ofl_1 := by
    rw [e_ae_6, e_ofl_2]; exact addc_lin ae_5 s2_2 ofl_1
  have b_ae_6 : ae_6 < 2^64 := by rw [e_ae_6]; exact addc_value_lt ae_5 s2_2 ofl_1
  have b_ofl_2 : ofl_2 ≤ 1 := by rw [e_ofl_2]; exact addc_carry_le_one ae_5 s2_2 ofl_1 b_ae_5 b_s2_2 b_ofl_1
  clear e_ae_6 e_ofl_2
  -- m_8: mulx {s2}, {s1}, qword ptr [{a} + 16]
  extract_lets -merge +onlyGivenNames m_8 s2_3 s1_8 at hr
  have e_s2_3 : s2_3 = (mulx rdx_5 lhs.l2).1 := rfl
  have e_s1_8 : s1_8 = (mulx rdx_5 lhs.l2).2 := rfl
  have b_s2_3 : s2_3 < 2^64 := by rw [e_s2_3]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_5 hlhs.2.2.1)
  have b_s1_8 : s1_8 < 2^64 := by rw [e_s1_8]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_s2_3 : s1_8 + 2^64 * s2_3 = rdx_5 * lhs.l2 := by
    rw [e_s1_8, e_s2_3]; exact Nat.mod_add_div _ _
  -- ae_7: adcx {ae}, {s1}
  extract_lets -merge +onlyGivenNames s_17 ae_7 cf_17 at hr
  have e_ae_7 : ae_7 = (addc ae_6 s1_8 cf_16).1 := rfl
  have e_cf_17 : cf_17 = (addc ae_6 s1_8 cf_16).2 := rfl
  have l_ae_7 : ae_7 + 2^64 * cf_17 = ae_6 + s1_8 + cf_16 := by
    rw [e_ae_7, e_cf_17]; exact addc_lin ae_6 s1_8 cf_16
  have b_ae_7 : ae_7 < 2^64 := by rw [e_ae_7]; exact addc_value_lt ae_6 s1_8 cf_16
  have b_cf_17 : cf_17 ≤ 1 := by rw [e_cf_17]; exact addc_carry_le_one ae_6 s1_8 cf_16 b_ae_6 b_s1_8 b_cf_16
  clear e_ae_7 e_cf_17
  -- be_6: adox {be}, {s2}
  extract_lets -merge +onlyGivenNames s_18 be_6 ofl_3 at hr
  have e_be_6 : be_6 = (addc be_5 s2_3 ofl_2).1 := rfl
  have e_ofl_3 : ofl_3 = (addc be_5 s2_3 ofl_2).2 := rfl
  have l_be_6 : be_6 + 2^64 * ofl_3 = be_5 + s2_3 + ofl_2 := by
    rw [e_be_6, e_ofl_3]; exact addc_lin be_5 s2_3 ofl_2
  have b_be_6 : be_6 < 2^64 := by rw [e_be_6]; exact addc_value_lt be_5 s2_3 ofl_2
  have b_ofl_3 : ofl_3 ≤ 1 := by rw [e_ofl_3]; exact addc_carry_le_one be_5 s2_3 ofl_2 b_be_5 b_s2_3 b_ofl_2
  clear e_be_6 e_ofl_3
  -- m_9: mulx {s2}, {s1}, qword ptr [{a} + 24]
  extract_lets -merge +onlyGivenNames m_9 s2_4 s1_9 at hr
  have e_s2_4 : s2_4 = (mulx rdx_5 lhs.l3).1 := rfl
  have e_s1_9 : s1_9 = (mulx rdx_5 lhs.l3).2 := rfl
  have b_s2_4 : s2_4 < 2^64 := by rw [e_s2_4]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_5 hlhs.2.2.2)
  have b_s1_9 : s1_9 < 2^64 := by rw [e_s1_9]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_s2_4 : s1_9 + 2^64 * s2_4 = rdx_5 * lhs.l3 := by
    rw [e_s1_9, e_s2_4]; exact Nat.mod_add_div _ _
  -- be_7: adcx {be}, {s1}
  extract_lets -merge +onlyGivenNames s_19 be_7 cf_18 at hr
  have e_be_7 : be_7 = (addc be_6 s1_9 cf_17).1 := rfl
  have e_cf_18 : cf_18 = (addc be_6 s1_9 cf_17).2 := rfl
  have l_be_7 : be_7 + 2^64 * cf_18 = be_6 + s1_9 + cf_17 := by
    rw [e_be_7, e_cf_18]; exact addc_lin be_6 s1_9 cf_17
  have b_be_7 : be_7 < 2^64 := by rw [e_be_7]; exact addc_value_lt be_6 s1_9 cf_17
  have b_cf_18 : cf_18 ≤ 1 := by rw [e_cf_18]; exact addc_carry_le_one be_6 s1_9 cf_17 b_be_6 b_s1_9 b_cf_17
  clear e_be_7 e_cf_18
  -- ce_6: adox {ce}, {s2}
  extract_lets -merge +onlyGivenNames s_20 ce_6 ofl_4 at hr
  have e_ce_6 : ce_6 = (addc ce_5 s2_4 ofl_3).1 := rfl
  have e_ofl_4 : ofl_4 = (addc ce_5 s2_4 ofl_3).2 := rfl
  have l_ce_6 : ce_6 + 2^64 * ofl_4 = ce_5 + s2_4 + ofl_3 := by
    rw [e_ce_6, e_ofl_4]; exact addc_lin ce_5 s2_4 ofl_3
  have b_ce_6 : ce_6 < 2^64 := by rw [e_ce_6]; exact addc_value_lt ce_5 s2_4 ofl_3
  have b_ofl_4 : ofl_4 ≤ 1 := by rw [e_ofl_4]; exact addc_carry_le_one ce_5 s2_4 ofl_3 b_ce_5 b_s2_4 b_ofl_3
  clear e_ce_6 e_ofl_4
  -- s1_10: mov {s1}, 0
  extract_lets -merge +onlyGivenNames s1_10 at hr
  have e_s1_10 : s1_10 = 0 := rfl
  have b_s1_10 : s1_10 < 2^64 := by rw [e_s1_10]; decide
  -- ce_7: adcx {ce}, {s1}
  extract_lets -merge +onlyGivenNames s_21 ce_7 cf_19 at hr
  have e_ce_7 : ce_7 = (addc ce_6 s1_10 cf_18).1 := rfl
  have e_cf_19 : cf_19 = (addc ce_6 s1_10 cf_18).2 := rfl
  have l_ce_7 : ce_7 + 2^64 * cf_19 = ce_6 + s1_10 + cf_18 := by
    rw [e_ce_7, e_cf_19]; exact addc_lin ce_6 s1_10 cf_18
  have b_ce_7 : ce_7 < 2^64 := by rw [e_ce_7]; exact addc_value_lt ce_6 s1_10 cf_18
  have b_cf_19 : cf_19 ≤ 1 := by rw [e_cf_19]; exact addc_carry_le_one ce_6 s1_10 cf_18 b_ce_6 b_s1_10 b_cf_18
  clear e_ce_7 e_cf_19
  -- ce_8: adox {ce}, {s1}
  extract_lets -merge +onlyGivenNames s_22 ce_8 ofl_5 at hr
  have e_ce_8 : ce_8 = (addc ce_7 s1_10 ofl_4).1 := rfl
  have e_ofl_5 : ofl_5 = (addc ce_7 s1_10 ofl_4).2 := rfl
  have l_ce_8 : ce_8 + 2^64 * ofl_5 = ce_7 + s1_10 + ofl_4 := by
    rw [e_ce_8, e_ofl_5]; exact addc_lin ce_7 s1_10 ofl_4
  have b_ce_8 : ce_8 < 2^64 := by rw [e_ce_8]; exact addc_value_lt ce_7 s1_10 ofl_4
  have b_ofl_5 : ofl_5 ≤ 1 := by rw [e_ofl_5]; exact addc_carry_le_one ce_7 s1_10 ofl_4 b_ce_7 b_s1_10 b_ofl_4
  clear e_ce_8 e_ofl_5
  -- BEGIN final product
  clear_value rdx_5 m_6 s2_1 s1_6 s_13 de_6 cf_15 s_14 ee_6 ofl_1 m_7 s2_2 s1_7 s_15 ee_7 cf_16 s_16 ae_6 ofl_2 m_8 s2_3 s1_8 s_17 ae_7 cf_17 s_18 be_6 ofl_3 m_9 s2_4 s1_9 s_19 be_7 cf_18 s_20 ce_6 ofl_4 s_21 ce_7 cf_19 s_22 ce_8 ofl_5
  have EA2 : round2.toNat = de_5 + 2^64 * ee_5 + 2^128 * ae_5 + 2^192 * be_5 + 2^256 * ce_5 := by
    simp only [MulMontAcc.toNat, e_de_5, e_ee_5, e_ae_5, e_be_5, e_ce_5]
  have hL3 : lhs.toNat * rhs.l3 = rdx_5 * lhs.l0 + 2^64 * (rdx_5 * lhs.l1) +
      2^128 * (rdx_5 * lhs.l2) + 2^192 * (rdx_5 * lhs.l3) := by
    rw [e_rdx_5]; simp only [Limbs.toNat]; ring
  have hs3 : round2.toNat + lhs.toNat * rhs.l3 + 2^128 + 3 * 2^254 < 2^320 := by
    have h := hsafe.2.2
    simp only [Nat.mul_add, Nat.mul_one] at h
    clear * - h H2
    omega
  have pre3 : de_6 + 2^64 * ee_7 + 2^128 * ae_7 + 2^192 * be_7 + 2^256 * ce_7 +
      2^320 * (ofl_4 + cf_19) = round2.toNat + lhs.toNat * rhs.l3 := by
    rw [EA2, hL3]
    clear * - e_ae_2 d_s2_1 d_s2_2 d_s2_3 d_s2_4 l_de_6 l_ee_6 l_ee_7
      l_ae_6 l_ae_7 l_be_6 l_be_7 l_ce_6 l_ce_7
    omega
  have ztop3 : ofl_4 = 0 ∧ cf_19 = 0 := by clear * - pre3 hs3; omega
  have zlast3 : ofl_5 = 0 := by clear * - l_ce_8 e_ae_2 ztop3 b_ce_7; omega
  have P3 : de_6 + 2^64 * ee_7 + 2^128 * ae_7 + 2^192 * be_7 + 2^256 * ce_8 =
      round2.toNat + lhs.toNat * rhs.l3 := by
    clear * - pre3 l_ce_8 e_ae_2 ztop3 zlast3
    omega
  -- END final product
  -- rdx_6: mov rdx, {de}
  extract_lets -merge +onlyGivenNames rdx_6 at hr
  have e_rdx_6 : rdx_6 = de_6 := rfl
  have b_rdx_6 : rdx_6 < 2^64 := by rw [e_rdx_6]; exact b_de_6
  -- rdx_7: imul rdx, {inv}
  extract_lets -merge +onlyGivenNames rdx_7 at hr
  have e_rdx_7 : rdx_7 = rdx_6 * inv' % 2^64 := rfl
  have b_rdx_7 : rdx_7 < 2^64 := by rw [e_rdx_7]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- m_10: mulx {s2}, {s1}, qword ptr [{p} + 8]
  extract_lets -merge +onlyGivenNames m_10 s2_5 s1_11 at hr
  have e_s2_5 : s2_5 = (mulx rdx_7 modulus.l1).1 := rfl
  have e_s1_11 : s1_11 = (mulx rdx_7 modulus.l1).2 := rfl
  have b_s2_5 : s2_5 < 2^64 := by rw [e_s2_5]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_7 hm.2.1)
  have b_s1_11 : s1_11 < 2^64 := by rw [e_s1_11]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_s2_5 : s1_11 + 2^64 * s2_5 = rdx_7 * modulus.l1 := by
    rw [e_s1_11, e_s2_5]; exact Nat.mod_add_div _ _
  -- s3_5: mov {s3}, rdx
  extract_lets -merge +onlyGivenNames s3_5 at hr
  have e_s3_5 : s3_5 = rdx_7 := rfl
  have b_s3_5 : s3_5 < 2^64 := by rw [e_s3_5]; exact b_rdx_7
  -- s3_6: shl {s3}, 62
  extract_lets -merge +onlyGivenNames s3_6 at hr
  have e_s3_6 : s3_6 = s3_5 * 2^62 % 2^64 := rfl
  have b_s3_6 : s3_6 < 2^64 := by rw [e_s3_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_s3_6 : s3_6 + 2^64 * (s3_5 / 2^2) = s3_5 * 2^62 := by
    rw [e_s3_6]; exact lsl62_lsr2_split _
  -- n_1: neg {de}
  extract_lets -merge +onlyGivenNames n_1 de_7 cf_20 at hr
  have e_de_7 : de_7 = (neg de_6).1 := rfl
  have e_cf_20 : cf_20 = (neg de_6).2 := rfl
  have b_de_7 : de_7 < 2^64 := by rw [e_de_7]; exact sbb_value_lt 0 de_6 0
  have b_cf_20 : cf_20 ≤ 1 := by
    rw [e_cf_20]; simp only [neg]; split <;> omega
  -- ee_8: adc {ee}, {s1}
  extract_lets -merge +onlyGivenNames s_23 ee_8 cf_21 at hr
  have e_ee_8 : ee_8 = (addc ee_7 s1_11 cf_20).1 := rfl
  have e_cf_21 : cf_21 = (addc ee_7 s1_11 cf_20).2 := rfl
  have l_ee_8 : ee_8 + 2^64 * cf_21 = ee_7 + s1_11 + cf_20 := by
    rw [e_ee_8, e_cf_21]; exact addc_lin ee_7 s1_11 cf_20
  have b_ee_8 : ee_8 < 2^64 := by rw [e_ee_8]; exact addc_value_lt ee_7 s1_11 cf_20
  have b_cf_21 : cf_21 ≤ 1 := by rw [e_cf_21]; exact addc_carry_le_one ee_7 s1_11 cf_20 b_ee_7 b_s1_11 b_cf_20
  clear e_ee_8 e_cf_21
  -- ae_8: adc {ae}, 0
  extract_lets -merge +onlyGivenNames s_24 ae_8 cf_22 at hr
  have e_ae_8 : ae_8 = (addc ae_7 0 cf_21).1 := rfl
  have e_cf_22 : cf_22 = (addc ae_7 0 cf_21).2 := rfl
  have l_ae_8 : ae_8 + 2^64 * cf_22 = ae_7 + 0 + cf_21 := by
    rw [e_ae_8, e_cf_22]; exact addc_lin ae_7 0 cf_21
  have b_ae_8 : ae_8 < 2^64 := by rw [e_ae_8]; exact addc_value_lt ae_7 0 cf_21
  have b_cf_22 : cf_22 ≤ 1 := by rw [e_cf_22]; exact addc_carry_le_one ae_7 0 cf_21 b_ae_7 (by decide) b_cf_21
  clear e_ae_8 e_cf_22
  -- be_8: adc {be}, {s3}
  extract_lets -merge +onlyGivenNames s_25 be_8 cf_23 at hr
  have e_be_8 : be_8 = (addc be_7 s3_6 cf_22).1 := rfl
  have e_cf_23 : cf_23 = (addc be_7 s3_6 cf_22).2 := rfl
  have l_be_8 : be_8 + 2^64 * cf_23 = be_7 + s3_6 + cf_22 := by
    rw [e_be_8, e_cf_23]; exact addc_lin be_7 s3_6 cf_22
  have b_be_8 : be_8 < 2^64 := by rw [e_be_8]; exact addc_value_lt be_7 s3_6 cf_22
  have b_cf_23 : cf_23 ≤ 1 := by rw [e_cf_23]; exact addc_carry_le_one be_7 s3_6 cf_22 b_be_7 b_s3_6 b_cf_22
  clear e_be_8 e_cf_23
  -- ce_9: adc {ce}, 0
  extract_lets -merge +onlyGivenNames s_26 ce_9 cf_24 at hr
  have e_ce_9 : ce_9 = (addc ce_8 0 cf_23).1 := rfl
  have e_cf_24 : cf_24 = (addc ce_8 0 cf_23).2 := rfl
  have l_ce_9 : ce_9 + 2^64 * cf_24 = ce_8 + 0 + cf_23 := by
    rw [e_ce_9, e_cf_24]; exact addc_lin ce_8 0 cf_23
  have b_ce_9 : ce_9 < 2^64 := by rw [e_ce_9]; exact addc_value_lt ce_8 0 cf_23
  have b_cf_24 : cf_24 ≤ 1 := by rw [e_cf_24]; exact addc_carry_le_one ce_8 0 cf_23 b_ce_8 (by decide) b_cf_23
  clear e_ce_9 e_cf_24
  -- m_11: mulx {s1}, {s3}, qword ptr [{p}]
  extract_lets -merge +onlyGivenNames m_11 s1_12 s3_7 at hr
  have e_s1_12 : s1_12 = (mulx rdx_7 modulus.l0).1 := rfl
  have e_s3_7 : s3_7 = (mulx rdx_7 modulus.l0).2 := rfl
  have b_s1_12 : s1_12 < 2^64 := by rw [e_s1_12]; exact Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' b_rdx_7 hm.1)
  have b_s3_7 : s3_7 < 2^64 := by rw [e_s3_7]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have d_s1_12 : s3_7 + 2^64 * s1_12 = rdx_7 * modulus.l0 := by
    rw [e_s3_7, e_s1_12]; exact Nat.mod_add_div _ _
  -- s3_8: mov {s3}, rdx
  extract_lets -merge +onlyGivenNames s3_8 at hr
  have e_s3_8 : s3_8 = rdx_7 := rfl
  have b_s3_8 : s3_8 < 2^64 := by rw [e_s3_8]; exact b_rdx_7
  -- s3_9: shr {s3}, 2
  extract_lets -merge +onlyGivenNames s3_9 at hr
  have e_s3_9 : s3_9 = s3_8 / 2^2 := rfl
  have b_s3_9 : s3_9 < 2^62 := by
    rw [e_s3_9]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_s3_8 (by norm_num))
  -- ee_9: add {ee}, {s1}
  extract_lets -merge +onlyGivenNames s_27 ee_9 cf_25 at hr
  have e_ee_9 : ee_9 = (addc ee_8 s1_12 0).1 := rfl
  have e_cf_25 : cf_25 = (addc ee_8 s1_12 0).2 := rfl
  have l_ee_9 : ee_9 + 2^64 * cf_25 = ee_8 + s1_12 + 0 := by
    rw [e_ee_9, e_cf_25]; exact addc_lin ee_8 s1_12 0
  have b_ee_9 : ee_9 < 2^64 := by rw [e_ee_9]; exact addc_value_lt ee_8 s1_12 0
  have b_cf_25 : cf_25 ≤ 1 := by rw [e_cf_25]; exact addc_carry_le_one ee_8 s1_12 0 b_ee_8 b_s1_12 (by decide)
  clear e_ee_9 e_cf_25
  -- ae_9: adc {ae}, {s2}
  extract_lets -merge +onlyGivenNames s_28 ae_9 cf_26 at hr
  have e_ae_9 : ae_9 = (addc ae_8 s2_5 cf_25).1 := rfl
  have e_cf_26 : cf_26 = (addc ae_8 s2_5 cf_25).2 := rfl
  have l_ae_9 : ae_9 + 2^64 * cf_26 = ae_8 + s2_5 + cf_25 := by
    rw [e_ae_9, e_cf_26]; exact addc_lin ae_8 s2_5 cf_25
  have b_ae_9 : ae_9 < 2^64 := by rw [e_ae_9]; exact addc_value_lt ae_8 s2_5 cf_25
  have b_cf_26 : cf_26 ≤ 1 := by rw [e_cf_26]; exact addc_carry_le_one ae_8 s2_5 cf_25 b_ae_8 b_s2_5 b_cf_25
  clear e_ae_9 e_cf_26
  -- be_9: adc {be}, 0
  extract_lets -merge +onlyGivenNames s_29 be_9 cf_27 at hr
  have e_be_9 : be_9 = (addc be_8 0 cf_26).1 := rfl
  have e_cf_27 : cf_27 = (addc be_8 0 cf_26).2 := rfl
  have l_be_9 : be_9 + 2^64 * cf_27 = be_8 + 0 + cf_26 := by
    rw [e_be_9, e_cf_27]; exact addc_lin be_8 0 cf_26
  have b_be_9 : be_9 < 2^64 := by rw [e_be_9]; exact addc_value_lt be_8 0 cf_26
  have b_cf_27 : cf_27 ≤ 1 := by rw [e_cf_27]; exact addc_carry_le_one be_8 0 cf_26 b_be_8 (by decide) b_cf_26
  clear e_be_9 e_cf_27
  -- ce_10: adc {ce}, {s3}
  extract_lets -merge +onlyGivenNames s_30 ce_10 cf_28 at hr
  have e_ce_10 : ce_10 = (addc ce_9 s3_9 cf_27).1 := rfl
  have e_cf_28 : cf_28 = (addc ce_9 s3_9 cf_27).2 := rfl
  have l_ce_10 : ce_10 + 2^64 * cf_28 = ce_9 + s3_9 + cf_27 := by
    rw [e_ce_10, e_cf_28]; exact addc_lin ce_9 s3_9 cf_27
  have b_ce_10 : ce_10 < 2^64 := by rw [e_ce_10]; exact addc_value_lt ce_9 s3_9 cf_27
  have b_cf_28 : cf_28 ≤ 1 := by rw [e_cf_28]; exact addc_carry_le_one ce_9 s3_9 cf_27 b_ce_9 (lt_of_lt_of_le b_s3_9 (by norm_num)) b_cf_27
  clear e_ce_10 e_cf_28
  -- BEGIN final reduction identity
  clear_value rdx_6 rdx_7 m_10 s2_5 s1_11 s3_5 s3_6 n_1 de_7 cf_20 s_23 ee_8 cf_21 s_24 ae_8 cf_22 s_25 be_8 cf_23 s_26 ce_9 cf_24 m_11 s1_12 s3_7 s3_9 s_27 ee_9 cf_25 s_28 ae_9 cf_26 s_29 be_9 cf_27 s_30 ce_10 cf_28
  have hq3 : rdx_7 = mulLo inv de_6 := by
    rw [e_rdx_7, e_rdx_6, e_inv']; simp only [mulLo, Nat.mul_comm]
  have hc3 : de_6 + s3_7 = 2^64 * cf_20 := by
    have h := neg_carry_cancel de_6 inv modulus.l0 rdx_7 b_de_6 hinv_lt hm.1 hinv hq3
    simpa only [e_s3_7, e_cf_20, mulx, neg] using h
  have hqp3 : rdx_7 * modulus.toNat = rdx_7 * modulus.l0 +
      2^64 * (rdx_7 * modulus.l1) + 2^192 * (rdx_7 * 2^62) := by rw [hp]; ring
  have hshift3 : s3_6 ≤ 3 * 2^62 := by clear * - sh_s3_6 b_s3_6; omega
  have low3 : de_6 + 2^64 * ee_8 + 2^128 * ae_8 + 2^192 * be_8 + 2^256 * ce_9 + 2^320 * cf_24 =
      round2.toNat + lhs.toNat * rhs.l3 + 2^64 * s1_11 + 2^64 * cf_20 + 2^192 * s3_6 := by
    clear * - P3 l_ee_8 l_ae_8 l_be_8 l_ce_9
    omega
  have zcf24 : cf_24 = 0 := by clear * - low3 hs3 b_s1_11 b_cf_20 hshift3; omega
  have I3 : 2^64 * (ee_9 + 2^64 * ae_9 + 2^128 * be_9 + 2^192 * ce_10 + 2^256 * cf_28) =
      round2.toNat + lhs.toNat * rhs.l3 + rdx_7 * modulus.toNat := by
    clear * - low3 zcf24 hc3 hqp3 d_s2_5 d_s1_12 sh_s3_6 e_s3_9 e_s3_8 e_s3_5
      l_ee_9 l_ae_9 l_be_9 l_ce_10
    omega
  have hR : lhs.toNat * rhs.toNat = lhs.toNat * rhs.l0 + 2^64 * (lhs.toNat * rhs.l1) +
      2^128 * (lhs.toNat * rhs.l2) + 2^192 * (lhs.toNat * rhs.l3) := by
    simp only [Limbs.toNat]; ring
  have Iall : 2^256 * (ee_9 + 2^64 * ae_9 + 2^128 * be_9 + 2^192 * ce_10 + 2^256 * cf_28) =
      lhs.toNat * rhs.toNat + (rdx_2 + 2^64 * round1.q + 2^128 * round2.q + 2^192 * rdx_7) * modulus.toNat := by
    simp only [MulMontAcc.toNat] at I1
    rw [Nat.add_mul, Nat.add_mul, Nat.add_mul]
    simp only [Nat.mul_assoc]
    simp only [MulMontAcc.toNat] at I2 I3
    clear * - I0 I1 I2 I3 hR
    omega
  have hQ : rdx_2 + 2^64 * round1.q + 2^128 * round2.q + 2^192 * rdx_7 < 2^256 := by
    have h1 := B1.2.2.2.2.2
    have h2 := B2.2.2.2.2.2
    clear * - b_rdx_2 b_rdx_7 h1 h2
    omega
  have hA : ee_9 + 2^64 * ae_9 + 2^128 * be_9 + 2^192 * ce_10 + 2^256 * cf_28 < 2 * modulus.toNat := by
    have hq := Nat.mul_le_mul_right modulus.toNat (Nat.le_of_lt hQ)
    clear * - Iall hq hfinal
    omega
  have zcf28 : cf_28 = 0 := by
    have hp_lt := Limbs.toNat_lt_of_shape modulus hm hshape
    clear * - hA hp_lt
    omega
  -- END final reduction identity
  -- rdx_8: movabs rdx, {p3}
  extract_lets -merge +onlyGivenNames rdx_8 at hr
  have e_rdx_8 : rdx_8 = p3 := rfl
  have b_rdx_8 : rdx_8 < 2^64 := by rw [e_rdx_8]; exact b_p3
  -- s1_13: mov {s1}, {ee}
  extract_lets -merge +onlyGivenNames s1_13 at hr
  have e_s1_13 : s1_13 = ee_9 := rfl
  have b_s1_13 : s1_13 < 2^64 := by rw [e_s1_13]; exact b_ee_9
  -- s2_6: mov {s2}, {ae}
  extract_lets -merge +onlyGivenNames s2_6 at hr
  have e_s2_6 : s2_6 = ae_9 := rfl
  have b_s2_6 : s2_6 < 2^64 := by rw [e_s2_6]; exact b_ae_9
  -- s3_10: mov {s3}, {be}
  extract_lets -merge +onlyGivenNames s3_10 at hr
  have e_s3_10 : s3_10 = be_9 := rfl
  have b_s3_10 : s3_10 < 2^64 := by rw [e_s3_10]; exact b_be_9
  -- de_8: mov {de}, {ce}
  extract_lets -merge +onlyGivenNames de_8 at hr
  have e_de_8 : de_8 = ce_10 := rfl
  have b_de_8 : de_8 < 2^64 := by rw [e_de_8]; exact b_ce_10
  -- s1_14: sub {s1}, qword ptr [{p}]
  extract_lets -merge +onlyGivenNames d s1_14 cf_29 at hr
  have e_s1_14 : s1_14 = (sbb s1_13 modulus.l0 0).1 := rfl
  have e_cf_29 : cf_29 = (sbb s1_13 modulus.l0 0).2 := rfl
  have l_s1_14 : s1_14 + modulus.l0 + 0 = s1_13 + 2^64 * cf_29 := by
    rw [e_s1_14, e_cf_29]; exact sbb_lin s1_13 modulus.l0 0 b_s1_13 hm.1 (by decide)
  have b_s1_14 : s1_14 < 2^64 := by rw [e_s1_14]; exact sbb_value_lt s1_13 modulus.l0 0
  have b_cf_29 : cf_29 ≤ 1 := by rw [e_cf_29]; exact sbb_borrow_le_one s1_13 modulus.l0 0
  clear e_s1_14 e_cf_29
  -- s2_7: sbb {s2}, qword ptr [{p} + 8]
  extract_lets -merge +onlyGivenNames d_1 s2_7 cf_30 at hr
  have e_s2_7 : s2_7 = (sbb s2_6 modulus.l1 cf_29).1 := rfl
  have e_cf_30 : cf_30 = (sbb s2_6 modulus.l1 cf_29).2 := rfl
  have l_s2_7 : s2_7 + modulus.l1 + cf_29 = s2_6 + 2^64 * cf_30 := by
    rw [e_s2_7, e_cf_30]; exact sbb_lin s2_6 modulus.l1 cf_29 b_s2_6 hm.2.1 b_cf_29
  have b_s2_7 : s2_7 < 2^64 := by rw [e_s2_7]; exact sbb_value_lt s2_6 modulus.l1 cf_29
  have b_cf_30 : cf_30 ≤ 1 := by rw [e_cf_30]; exact sbb_borrow_le_one s2_6 modulus.l1 cf_29
  clear e_s2_7 e_cf_30
  -- s3_11: sbb {s3}, 0
  extract_lets -merge +onlyGivenNames d_2 s3_11 cf_31 at hr
  have e_s3_11 : s3_11 = (sbb s3_10 0 cf_30).1 := rfl
  have e_cf_31 : cf_31 = (sbb s3_10 0 cf_30).2 := rfl
  have l_s3_11 : s3_11 + 0 + cf_30 = s3_10 + 2^64 * cf_31 := by
    rw [e_s3_11, e_cf_31]; exact sbb_lin s3_10 0 cf_30 b_s3_10 (by decide) b_cf_30
  have b_s3_11 : s3_11 < 2^64 := by rw [e_s3_11]; exact sbb_value_lt s3_10 0 cf_30
  have b_cf_31 : cf_31 ≤ 1 := by rw [e_cf_31]; exact sbb_borrow_le_one s3_10 0 cf_30
  clear e_s3_11 e_cf_31
  -- de_9: sbb {de}, rdx
  extract_lets -merge +onlyGivenNames d_3 de_9 cf_32 at hr
  have e_de_9 : de_9 = (sbb de_8 rdx_8 cf_31).1 := rfl
  have e_cf_32 : cf_32 = (sbb de_8 rdx_8 cf_31).2 := rfl
  have l_de_9 : de_9 + rdx_8 + cf_31 = de_8 + 2^64 * cf_32 := by
    rw [e_de_9, e_cf_32]; exact sbb_lin de_8 rdx_8 cf_31 b_de_8 b_rdx_8 b_cf_31
  have b_de_9 : de_9 < 2^64 := by rw [e_de_9]; exact sbb_value_lt de_8 rdx_8 cf_31
  have b_cf_32 : cf_32 ≤ 1 := by rw [e_cf_32]; exact sbb_borrow_le_one de_8 rdx_8 cf_31
  clear e_de_9 e_cf_32
  -- ee_10: cmovnc {ee}, {s1}
  extract_lets -merge +onlyGivenNames ee_10 at hr
  have e_ee_10 : ee_10 = (if cf_32 = 0 then s1_14 else ee_9) := rfl
  have b_ee_10 : ee_10 < 2^64 := by
    rw [e_ee_10]; split <;> first | exact b_s1_14 | exact b_ee_9
  -- ae_10: cmovnc {ae}, {s2}
  extract_lets -merge +onlyGivenNames ae_10 at hr
  have e_ae_10 : ae_10 = (if cf_32 = 0 then s2_7 else ae_9) := rfl
  have b_ae_10 : ae_10 < 2^64 := by
    rw [e_ae_10]; split <;> first | exact b_s2_7 | exact b_ae_9
  -- be_10: cmovnc {be}, {s3}
  extract_lets -merge +onlyGivenNames be_10 at hr
  have e_be_10 : be_10 = (if cf_32 = 0 then s3_11 else be_9) := rfl
  have b_be_10 : be_10 < 2^64 := by
    rw [e_be_10]; split <;> first | exact b_s3_11 | exact b_be_9
  -- ce_11: cmovnc {ce}, {de}
  extract_lets -merge +onlyGivenNames ce_11 at hr
  have e_ce_11 : ce_11 = (if cf_32 = 0 then de_9 else ce_10) := rfl
  have b_ce_11 : ce_11 < 2^64 := by
    rw [e_ce_11]; split <;> first | exact b_de_9 | exact b_ce_10
  -- BEGIN final opacity
  clear_value rdx_8 s1_13 s2_6 s3_10 de_8 d s1_14 cf_29 d_1 s2_7 cf_30 d_2 s3_11 cf_31 d_3 de_9 cf_32 ee_10 ae_10 be_10 ce_11
  -- END final opacity
  subst hr
  -- BEGIN final conclusion
  have hD : s1_14 + 2^64 * s2_7 + 2^128 * s3_11 + 2^192 * de_9 + modulus.toNat =
      ee_9 + 2^64 * ae_9 + 2^128 * be_9 + 2^192 * ce_10 + 2^256 * cf_32 := by
    rw [hp]
    clear * - l_s1_14 l_s2_7 l_s3_11 l_de_9 e_s1_13 e_s2_6 e_s3_10 e_de_8 e_rdx_8 e_p3
    omega
  rw [zcf28, Nat.mul_zero, Nat.add_zero] at Iall hA
  exact ⟨⟨b_ee_10, b_ae_10, b_be_10, b_ce_11⟩,
    mul_conclude Iall hA hD b_cf_32 b_s1_14 b_s2_7 b_s3_11 b_de_9
      e_ee_10 e_ae_10 e_be_10 e_ce_11⟩
  -- END final conclusion

-- BEGIN mulMont_spec corollaries
/-- The contract the crate's callers use: a canonical left operand and any four-limb right
operand. -/
theorem mulMont_spec_of_lhs_lt (lhs rhs modulus : Limbs) (inv : Nat) (hlhs : lhs.Bounded)
    (hrhs : rhs.Bounded) (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinv_lt : inv < 2^64) (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hlt : lhs.toNat < modulus.toNat) :
    ∀ r, r = mulMont lhs rhs modulus inv →
      r.Bounded ∧ r.toNat < modulus.toNat ∧
        2^256 * r.toNat ≡ lhs.toNat * rhs.toNat [MOD modulus.toNat] := by
  have hP := Limbs.toNat_lt_of_shape modulus hm hshape
  have hR := Limbs.toNat_lt rhs hrhs
  -- `lhs * (rhs_i + 1) ≤ (p - 1) * 2^64`, and `lhs * rhs < p * 2^256`.
  have s1 : lhs.toNat * (rhs.l1 + 1) ≤ (modulus.toNat - 1) * 2^64 :=
    Nat.mul_le_mul (Nat.le_sub_one_of_lt hlt) hrhs.2.1
  have s2 : lhs.toNat * (rhs.l2 + 1) ≤ (modulus.toNat - 1) * 2^64 :=
    Nat.mul_le_mul (Nat.le_sub_one_of_lt hlt) hrhs.2.2.1
  have s3 : lhs.toNat * (rhs.l3 + 1) ≤ (modulus.toNat - 1) * 2^64 :=
    Nat.mul_le_mul (Nat.le_sub_one_of_lt hlt) hrhs.2.2.2
  have f : (lhs.toNat + 1) * (rhs.toNat + 1) ≤ modulus.toNat * 2^256 := Nat.mul_le_mul hlt hR
  rw [Nat.add_one_mul, Nat.mul_add_one] at f
  exact mulMont_spec lhs rhs modulus inv hlhs hrhs hm hshape hinv_lt hinv
    ⟨by omega, by omega, by omega⟩ (by omega)

/-- The other safe contract: any four-limb left operand, a canonical right operand whose limbs 1
to 3 are at most `2^64 - 3`. -/
theorem mulMont_spec_of_rhs_lt (lhs rhs modulus : Limbs) (inv : Nat) (hlhs : lhs.Bounded)
    (hrhs : rhs.Bounded) (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinv_lt : inv < 2^64) (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hlt : rhs.toNat < modulus.toNat)
    (hlimbs : rhs.l1 + 3 ≤ 2^64 ∧ rhs.l2 + 3 ≤ 2^64 ∧ rhs.l3 + 3 ≤ 2^64) :
    ∀ r, r = mulMont lhs rhs modulus inv →
      r.Bounded ∧ r.toNat < modulus.toNat ∧
        2^256 * r.toNat ≡ lhs.toNat * rhs.toNat [MOD modulus.toNat] := by
  have hP := Limbs.toNat_lt_of_shape modulus hm hshape
  have hL := Limbs.toNat_lt lhs hlhs
  -- `lhs * (rhs_i + 1) ≤ (2^256 - 1) * (2^64 - 2)`, and `lhs * rhs < 2^256 * p`.
  have c1 : rhs.l1 + 1 ≤ 2^64 - 2 := by omega
  have c2 : rhs.l2 + 1 ≤ 2^64 - 2 := by omega
  have c3 : rhs.l3 + 1 ≤ 2^64 - 2 := by omega
  have s1 : lhs.toNat * (rhs.l1 + 1) ≤ (2^256 - 1) * (2^64 - 2) :=
    Nat.mul_le_mul (Nat.le_sub_one_of_lt hL) c1
  have s2 : lhs.toNat * (rhs.l2 + 1) ≤ (2^256 - 1) * (2^64 - 2) :=
    Nat.mul_le_mul (Nat.le_sub_one_of_lt hL) c2
  have s3 : lhs.toNat * (rhs.l3 + 1) ≤ (2^256 - 1) * (2^64 - 2) :=
    Nat.mul_le_mul (Nat.le_sub_one_of_lt hL) c3
  norm_num at s1 s2 s3
  have f : (lhs.toNat + 1) * (rhs.toNat + 1) ≤ 2^256 * modulus.toNat := Nat.mul_le_mul hL hlt
  rw [Nat.add_one_mul, Nat.mul_add_one] at f
  exact mulMont_spec lhs rhs modulus inv hlhs hrhs hm hshape hinv_lt hinv
    ⟨by omega, by omega, by omega⟩ (by omega)


-- END mulMont_spec corollaries

end PastaAsm.X86_64
