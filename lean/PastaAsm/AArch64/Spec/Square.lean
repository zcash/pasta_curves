/-
Copyright Supranational LLC (the routines, transcribed from Semolina v0.1.4).
Copyright (c) 2026 the pasta-asm contributors (the transcription and the proofs).
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.Spec
import PastaAsm.AArch64.Transcription
import PastaAsm.AArch64.Compositions
import PastaAsm.AArch64.Spec.Mul
import Mathlib.Tactic.NormNum

/-!
# Correctness of the transcribed Pasta squaring block

See the parent module's documentation for details.
-/

namespace PastaAsm.AArch64

-- BEGIN sqrMont lemmas
/-- One Montgomery cancellation step of the squaring block, on limbs `t0..t3` with quotient `q`:
after the reduction chain, the shift, and the carry `cy` above limb 3, the shifted limbs satisfy
`2^64 * new = old + q * p` (with `p = p0 + 2^64 * p1 + 2^254`), given the cancellation of the low
limb and that neither `adc` wraps. Stated over free variables and proved once, so that the
squaring theorem instantiates it four times without repeating the linear arithmetic. -/
theorem sqrMont_step {t0 t1 t2 t3 q p0 p1 lo w0 w1lo w1hi z1n c1 z2n c2 z3n c3 w3lsl w3lsr cy kcy
    z0o c4 z1o c5 z2o c6 z3o kz3 cs : Nat}
    (hc : t0 + lo = 2^64 * cs) (d_w0 : lo + 2^64 * w0 = p0 * q) (d_w1 : w1lo + 2^64 * w1hi = p1 * q)
    (l_z1 : z1n + 2^64 * c1 = t1 + w1lo + cs) (l_z2 : z2n + 2^64 * c2 = t2 + 0 + c1)
    (l_z3 : z3n + 2^64 * c3 = t3 + w3lsl + c2) (sh : w3lsl + 2^64 * w3lsr = q * 2^62)
    (l_cy : cy + 2^64 * kcy = 0 + 0 + c3) (l_z0o : z0o + 2^64 * c4 = z1n + w0 + 0)
    (l_z1o : z1o + 2^64 * c5 = z2n + w1hi + c4) (l_z2o : z2o + 2^64 * c6 = z3n + 0 + c5)
    (l_z3o : z3o + 2^64 * kz3 = cy + w3lsr + c6) (hk : kcy = 0 ∧ kz3 = 0) :
    2^64 * (z0o + 2^64 * z1o + 2^128 * z2o + 2^192 * z3o)
      = (t0 + 2^64 * t1 + 2^128 * t2 + 2^192 * t3) + (p0 * q + 2^64 * (p1 * q) + 2^254 * q) := by
  omega

set_option exponentiation.threshold 512 in
/-- The schoolbook cross products, their doubling, and the diagonal squares of the squaring
block, as the generated skeleton records them, make up the square of `a0 + 2^64 a1 + 2^128 a2 +
2^192 a3` exactly, with the carry out of the top limb (`k_z7_1`) at weight `2^512`. Stated over the
skeleton's own fact statements (this lemma is generated with the section) and proved once. -/
theorem sqrMont_square {a0 a0_1 a1 a1_1 a2 a2_1 a3 a3_1 c c_1 c_10 c_11 c_12 c_13 c_14 c_15 c_16
    c_17 c_2 c_3 c_4 c_5 c_6 c_7 c_8 c_9 k_w2_2 k_z4_1 k_z6_1 k_z7 k_z7_1 w0 w1 w1_1 w1_2 w1_3 w2
    w2_1 w2_2 w2_3 w3 w3_1 z0 z1 z1_1 z1_2 z2 z2_1 z2_2 z2_3 z3 z3_1 z3_2 z3_3 z3_4 z4 z4_1 z4_2
    z4_3 z4_4 z5 z5_1 z5_2 z5_3 z6 z6_1 z6_2 z6_3 z7 z7_1 : Nat}
    (d_w1 : z1 + 2^64 * w1 = a1 * a0)
    (d_w2 : z2 + 2^64 * w2 = a2 * a0)
    (d_z4 : z3 + 2^64 * z4 = a3 * a0)
    (d_w1_1 : w0 + 2^64 * w1_1 = a2 * a1)
    (d_w3 : w2_1 + 2^64 * w3 = a3 * a1)
    (d_z6 : z5 + 2^64 * z6 = a3 * a2)
    (d_a0_1 : z0 + 2^64 * a0_1 = a0 * a0)
    (d_a1_1 : w1_3 + 2^64 * a1_1 = a1 * a1)
    (d_a2_1 : w2_3 + 2^64 * a2_1 = a2 * a2)
    (d_a3_1 : w3_1 + 2^64 * a3_1 = a3 * a3)
    (l_z2_1 : z2_1 + 2^64 * c = z2 + w1 + 0)
    (l_z3_1 : z3_1 + 2^64 * c_1 = z3 + w2 + c)
    (l_z4_1 : z4_1 + 2^64 * k_z4_1 = z4 + 0 + c_1)
    (l_w1_2 : w1_2 + 2^64 * c_2 = w1_1 + w2_1 + 0)
    (l_w2_2 : w2_2 + 2^64 * k_w2_2 = w3 + 0 + c_2)
    (l_z3_2 : z3_2 + 2^64 * c_3 = z3_1 + w0 + 0)
    (l_z4_2 : z4_2 + 2^64 * c_4 = z4_1 + w1_2 + c_3)
    (l_z5_1 : z5_1 + 2^64 * c_5 = z5 + w2_2 + c_4)
    (l_z6_1 : z6_1 + 2^64 * k_z6_1 = z6 + 0 + c_5)
    (l_z1_1 : z1_1 + 2^64 * c_6 = z1 + z1 + 0)
    (l_z2_2 : z2_2 + 2^64 * c_7 = z2_1 + z2_1 + c_6)
    (l_z3_3 : z3_3 + 2^64 * c_8 = z3_2 + z3_2 + c_7)
    (l_z4_3 : z4_3 + 2^64 * c_9 = z4_2 + z4_2 + c_8)
    (l_z5_2 : z5_2 + 2^64 * c_10 = z5_1 + z5_1 + c_9)
    (l_z6_2 : z6_2 + 2^64 * c_11 = z6_1 + z6_1 + c_10)
    (l_z7 : z7 + 2^64 * k_z7 = 0 + 0 + c_11)
    (l_z1_2 : z1_2 + 2^64 * c_12 = z1_1 + a0_1 + 0)
    (l_z2_3 : z2_3 + 2^64 * c_13 = z2_2 + w1_3 + c_12)
    (l_z3_4 : z3_4 + 2^64 * c_14 = z3_3 + a1_1 + c_13)
    (l_z4_4 : z4_4 + 2^64 * c_15 = z4_3 + w2_3 + c_14)
    (l_z5_3 : z5_3 + 2^64 * c_16 = z5_2 + a2_1 + c_15)
    (l_z6_3 : z6_3 + 2^64 * c_17 = z6_2 + w3_1 + c_16)
    (l_z7_1 : z7_1 + 2^64 * k_z7_1 = z7 + a3_1 + c_17)
    (hk1 : k_z4_1 = 0)
    (hk2 : k_w2_2 = 0)
    (hk3 : k_z6_1 = 0)
    (hk4 : k_z7 = 0) :
    z0 + 2^64 * z1_2 + 2^128 * z2_3 + 2^192 * z3_4 + 2^256 * z4_4 + 2^320 * z5_3
        + 2^384 * z6_3 + 2^448 * z7_1 + 2^512 * k_z7_1
      = a0 * a0 + 2^64 * (2 * (a1 * a0)) + 2^128 * (2 * (a2 * a0) + a1 * a1)
        + 2^192 * (2 * (a3 * a0) + 2 * (a2 * a1)) + 2^256 * (2 * (a3 * a1) + a2 * a2)
        + 2^320 * (2 * (a3 * a2)) + 2^384 * (a3 * a3) := by
  omega

/-- The squaring block's final conditional subtraction and select, as a standalone lemma so its
case analysis has its own heartbeat budget. From `hmain` (the candidate `A` satisfies
`2^256 * A = value^2 + Q * p`), `hA` (`A < 2 * p`, so the dropped carry `ka = 0`), and the
subtraction chain `hD` and select equations, the four output limbs are `A` or `A - p` and are
below `p` and congruent to `value^2 * 2^-256` modulo `p`. -/
theorem sqrMont_conclude {a0 a1 a2 a3 ka z0 z1 z2 z3 c0 dz0 dz1 dz2 r0 r1 r2 r3 q4 vv Q : Nat}
    {modulus : Limbs} (hmain : 2^256 * (a0 + 2^64 * a1 + 2^128 * a2 + 2^192 * a3 + 2^256 * ka)
        = vv + Q * modulus.toNat)
    (hA : a0 + 2^64 * a1 + 2^128 * a2 + 2^192 * a3 + 2^256 * ka < 2 * modulus.toNat) (hk : ka = 0)
    (lz0 : z0 + 2^64 * dz0 + modulus.l0 + 1 = a0 + 2^64 + 1)
    (lz1 : z1 + 2^64 * dz1 + modulus.l1 + 1 = a1 + 2^64 + dz0)
    (lz2 : z2 + 2^64 * dz2 + 0 + 1 = a2 + 2^64 + dz1)
    (lz3 : z3 + 2^64 * c0 + q4 + 1 = a3 + 2^64 + dz2)
    (hq4 : q4 = 4611686018427387904) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62) (hc1 : c0 ≤ 1)
    (bz0 : z0 < 2^64) (bz1 : z1 < 2^64) (bz2 : z2 < 2^64) (bz3 : z3 < 2^64)
    (br0 : r0 < 2^64) (br1 : r1 < 2^64) (br2 : r2 < 2^64) (br3 : r3 < 2^64)
    (er0 : r0 = if c0 = 0 then a0 else z0) (er1 : r1 = if c0 = 0 then a1 else z1)
    (er2 : r2 = if c0 = 0 then a2 else z2) (er3 : r3 = if c0 = 0 then a3 else z3) :
    (⟨r0, r1, r2, r3⟩ : Limbs).Bounded ∧ (⟨r0, r1, r2, r3⟩ : Limbs).toNat < modulus.toNat ∧
      2^256 * (⟨r0, r1, r2, r3⟩ : Limbs).toNat ≡ vv [MOD modulus.toNat] := by
  have hPv : modulus.toNat = modulus.l0 + 2^64 * modulus.l1 + 2^254 := by
    simp only [Limbs.toNat, hshape.1, hshape.2]; ring
  -- The four subtraction facts compose to the candidate-minus-`p` identity.
  have hD : z0 + 2^64 * z1 + 2^128 * z2 + 2^192 * z3
        + (modulus.l0 + 2^64 * modulus.l1 + 2^192 * q4) + 2^256 * c0
      = a0 + 2^64 * a1 + 2^128 * a2 + 2^192 * a3 + 2^256 := by omega
  refine ⟨⟨br0, br1, br2, br3⟩, ?_⟩
  show r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3 < modulus.toNat ∧
    2^256 * (r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3) ≡ vv [MOD modulus.toNat]
  obtain hc | hc : c0 = 0 ∨ c0 = 1 := by omega
  · rw [if_pos hc] at er0 er1 er2 er3
    exact ⟨by omega, modEq_of_add_mul _ _ 0 Q _ (by omega)⟩
  · rw [if_neg (by omega)] at er0 er1 er2 er3
    exact ⟨by omega, modEq_of_add_mul _ _ (2^256) Q _ (by omega)⟩
-- END sqrMont lemmas

-- BEGIN sqrMont_spec statement
set_option exponentiation.threshold 512 in
/-- Montgomery squaring by the inline block: for a canonical `value`, the result is below `p` and
`2^256 * result ≡ value * value (mod p)`. The block forms the eight-limb square exactly, reduces its
low half by four Montgomery cancellation steps, adds the high half, and reduces once conditionally.
The candidate is below `2 * p`, so the carry the block drops there is `0`. The square's limbs have
weights up to `2^512`, beyond the default threshold for evaluating powers, hence the option. -/
theorem sqrMont_spec (value modulus : Limbs) (inv : Nat) (hv : value.Bounded)
    (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinv_lt : inv < 2^64) (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hlt : value.toNat < modulus.toNat) :
    ∀ r, r = sqrMont value modulus inv →
      r.Bounded ∧ r.toNat < modulus.toNat ∧
        2^256 * r.toNat ≡ value.toNat * value.toNat [MOD modulus.toNat] := by
  intro r hr
-- END sqrMont_spec statement
  -- generated skeleton for `sqrMont`: do not edit between the annotations
  unfold sqrMont at hr
  lift_lets -merge at hr
  -- a0: argument
  extract_lets -merge +onlyGivenNames a0 at hr
  have e_a0 : a0 = value.l0 := rfl
  clear_value a0
  have b_a0 : a0 < 2^64 := by rw [e_a0]; exact hv.1
  -- a1: argument
  extract_lets -merge +onlyGivenNames a1 at hr
  have e_a1 : a1 = value.l1 := rfl
  clear_value a1
  have b_a1 : a1 < 2^64 := by rw [e_a1]; exact hv.2.1
  -- a2: argument
  extract_lets -merge +onlyGivenNames a2 at hr
  have e_a2 : a2 = value.l2 := rfl
  clear_value a2
  have b_a2 : a2 < 2^64 := by rw [e_a2]; exact hv.2.2.1
  -- a3: argument
  extract_lets -merge +onlyGivenNames a3 at hr
  have e_a3 : a3 = value.l3 := rfl
  clear_value a3
  have b_a3 : a3 < 2^64 := by rw [e_a3]; exact hv.2.2.2
  -- p0: argument
  extract_lets -merge +onlyGivenNames p0 at hr
  have e_p0 : p0 = modulus.l0 := rfl
  clear_value p0
  have b_p0 : p0 < 2^64 := by rw [e_p0]; exact hm.1
  -- p1: argument
  extract_lets -merge +onlyGivenNames p1 at hr
  have e_p1 : p1 = modulus.l1 := rfl
  clear_value p1
  have b_p1 : p1 < 2^64 := by rw [e_p1]; exact hm.2.1
  -- inv': argument
  extract_lets -merge +onlyGivenNames inv' at hr
  have e_inv' : inv' = inv := rfl
  clear_value inv'
  have b_inv' : inv' < 2^64 := by rw [e_inv']; exact hinv_lt
  -- z1: mul z1,a1,a0
  extract_lets -merge +onlyGivenNames z1 at hr
  have e_z1 : z1 = a1 * a0 % 2^64 := rfl
  clear_value z1
  have b_z1 : z1 < 2^64 := by rw [e_z1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w1: umulh w1,a1,a0
  extract_lets -merge +onlyGivenNames w1 at hr
  have e_w1 : w1 = a1 * a0 / 2^64 := rfl
  clear_value w1
  have p_w1 : a1 * a0 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a1 b_a0
  have b_w1 : w1 < 2^64 := by rw [e_w1]; exact Nat.div_lt_of_lt_mul p_w1
  have d_w1 : z1 + 2^64 * w1 = a1 * a0 := by
    rw [e_z1, e_w1]; exact Nat.mod_add_div _ _
  clear e_z1 e_w1
  -- z2: mul z2,a2,a0
  extract_lets -merge +onlyGivenNames z2 at hr
  have e_z2 : z2 = a2 * a0 % 2^64 := rfl
  clear_value z2
  have b_z2 : z2 < 2^64 := by rw [e_z2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w2: umulh w2,a2,a0
  extract_lets -merge +onlyGivenNames w2 at hr
  have e_w2 : w2 = a2 * a0 / 2^64 := rfl
  clear_value w2
  have p_w2 : a2 * a0 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a2 b_a0
  have b_w2 : w2 < 2^64 := by rw [e_w2]; exact Nat.div_lt_of_lt_mul p_w2
  have d_w2 : z2 + 2^64 * w2 = a2 * a0 := by
    rw [e_z2, e_w2]; exact Nat.mod_add_div _ _
  clear e_z2 e_w2
  -- z3: mul z3,a3,a0
  extract_lets -merge +onlyGivenNames z3 at hr
  have e_z3 : z3 = a3 * a0 % 2^64 := rfl
  clear_value z3
  have b_z3 : z3 < 2^64 := by rw [e_z3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- z4: umulh z4,a3,a0
  extract_lets -merge +onlyGivenNames z4 at hr
  have e_z4 : z4 = a3 * a0 / 2^64 := rfl
  clear_value z4
  have p_z4 : a3 * a0 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a3 b_a0
  have b_z4 : z4 < 2^64 := by rw [e_z4]; exact Nat.div_lt_of_lt_mul p_z4
  have d_z4 : z3 + 2^64 * z4 = a3 * a0 := by
    rw [e_z3, e_z4]; exact Nat.mod_add_div _ _
  clear e_z3 e_z4
  -- z2_1: adds z2,z2,w1
  extract_lets -merge +onlyGivenNames s z2_1 c at hr
  have e_z2_1 : z2_1 = (z2 + w1 + 0) % 2^64 := rfl
  have e_c : c = (z2 + w1 + 0) / 2^64 := rfl
  clear_value s z2_1 c
  have l_z2_1 : z2_1 + 2^64 * c = z2 + w1 + 0 := by
    rw [e_z2_1, e_c]; exact Nat.mod_add_div _ _
  have b_z2_1 : z2_1 < 2^64 := by rw [e_z2_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c : c ≤ 1 := by
    rw [e_c]; exact addc_carry_le_one z2 w1 0 b_z2 b_w1 (by decide)
  clear e_z2_1 e_c
  -- w0: mul w0,a2,a1
  extract_lets -merge +onlyGivenNames w0 at hr
  have e_w0 : w0 = a2 * a1 % 2^64 := rfl
  clear_value w0
  have b_w0 : w0 < 2^64 := by rw [e_w0]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w1_1: umulh w1,a2,a1
  extract_lets -merge +onlyGivenNames w1_1 at hr
  have e_w1_1 : w1_1 = a2 * a1 / 2^64 := rfl
  clear_value w1_1
  have p_w1_1 : a2 * a1 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a2 b_a1
  have b_w1_1 : w1_1 < 2^64 := by rw [e_w1_1]; exact Nat.div_lt_of_lt_mul p_w1_1
  have d_w1_1 : w0 + 2^64 * w1_1 = a2 * a1 := by
    rw [e_w0, e_w1_1]; exact Nat.mod_add_div _ _
  clear e_w0 e_w1_1
  -- z3_1: adcs z3,z3,w2
  extract_lets -merge +onlyGivenNames s_1 z3_1 c_1 at hr
  have e_z3_1 : z3_1 = (z3 + w2 + c) % 2^64 := rfl
  have e_c_1 : c_1 = (z3 + w2 + c) / 2^64 := rfl
  clear_value s_1 z3_1 c_1
  have l_z3_1 : z3_1 + 2^64 * c_1 = z3 + w2 + c := by
    rw [e_z3_1, e_c_1]; exact Nat.mod_add_div _ _
  have b_z3_1 : z3_1 < 2^64 := by rw [e_z3_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_1 : c_1 ≤ 1 := by
    rw [e_c_1]; exact addc_carry_le_one z3 w2 c b_z3 b_w2 b_c
  clear e_z3_1 e_c_1
  -- w2_1: mul w2,a3,a1
  extract_lets -merge +onlyGivenNames w2_1 at hr
  have e_w2_1 : w2_1 = a3 * a1 % 2^64 := rfl
  clear_value w2_1
  have b_w2_1 : w2_1 < 2^64 := by rw [e_w2_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w3: umulh w3,a3,a1
  extract_lets -merge +onlyGivenNames w3 at hr
  have e_w3 : w3 = a3 * a1 / 2^64 := rfl
  clear_value w3
  have p_w3 : a3 * a1 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a3 b_a1
  have b_w3 : w3 < 2^64 := by rw [e_w3]; exact Nat.div_lt_of_lt_mul p_w3
  have d_w3 : w2_1 + 2^64 * w3 = a3 * a1 := by
    rw [e_w2_1, e_w3]; exact Nat.mod_add_div _ _
  clear e_w2_1 e_w3
  -- z4_1: adc z4,z4,xzr
  extract_lets -merge +onlyGivenNames z4_1 at hr
  have e_z4_1 : z4_1 = (z4 + 0 + c_1) % 2^64 := rfl
  clear_value z4_1
  have b_z4_1 : z4_1 < 2^64 := by rw [e_z4_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_z4_1, b_k_z4_1, l_z4_1⟩ :
      ∃ k, k ≤ 1 ∧ z4_1 + 2^64 * k = z4 + 0 + c_1 :=
    ⟨(z4 + 0 + c_1) / 2^64, addc_carry_le_one z4 0 c_1 b_z4 (by decide) b_c_1,
      by rw [e_z4_1]; exact Nat.mod_add_div _ _⟩
  clear e_z4_1
  -- z5: mul z5,a3,a2
  extract_lets -merge +onlyGivenNames z5 at hr
  have e_z5 : z5 = a3 * a2 % 2^64 := rfl
  clear_value z5
  have b_z5 : z5 < 2^64 := by rw [e_z5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- z6: umulh z6,a3,a2
  extract_lets -merge +onlyGivenNames z6 at hr
  have e_z6 : z6 = a3 * a2 / 2^64 := rfl
  clear_value z6
  have p_z6 : a3 * a2 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a3 b_a2
  have b_z6 : z6 < 2^64 := by rw [e_z6]; exact Nat.div_lt_of_lt_mul p_z6
  have d_z6 : z5 + 2^64 * z6 = a3 * a2 := by
    rw [e_z5, e_z6]; exact Nat.mod_add_div _ _
  clear e_z5 e_z6
  -- w1_2: adds w1,w1,w2
  extract_lets -merge +onlyGivenNames s_2 w1_2 c_2 at hr
  have e_w1_2 : w1_2 = (w1_1 + w2_1 + 0) % 2^64 := rfl
  have e_c_2 : c_2 = (w1_1 + w2_1 + 0) / 2^64 := rfl
  clear_value s_2 w1_2 c_2
  have l_w1_2 : w1_2 + 2^64 * c_2 = w1_1 + w2_1 + 0 := by
    rw [e_w1_2, e_c_2]; exact Nat.mod_add_div _ _
  have b_w1_2 : w1_2 < 2^64 := by rw [e_w1_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_2 : c_2 ≤ 1 := by
    rw [e_c_2]; exact addc_carry_le_one w1_1 w2_1 0 b_w1_1 b_w2_1 (by decide)
  clear e_w1_2 e_c_2
  -- z0: mul z0,a0,a0
  extract_lets -merge +onlyGivenNames z0 at hr
  have e_z0 : z0 = a0 * a0 % 2^64 := rfl
  clear_value z0
  have b_z0 : z0 < 2^64 := by rw [e_z0]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w2_2: adc w2,w3,xzr
  extract_lets -merge +onlyGivenNames w2_2 at hr
  have e_w2_2 : w2_2 = (w3 + 0 + c_2) % 2^64 := rfl
  clear_value w2_2
  have b_w2_2 : w2_2 < 2^64 := by rw [e_w2_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_w2_2, b_k_w2_2, l_w2_2⟩ :
      ∃ k, k ≤ 1 ∧ w2_2 + 2^64 * k = w3 + 0 + c_2 :=
    ⟨(w3 + 0 + c_2) / 2^64, addc_carry_le_one w3 0 c_2 b_w3 (by decide) b_c_2,
      by rw [e_w2_2]; exact Nat.mod_add_div _ _⟩
  clear e_w2_2
  -- z3_2: adds z3,z3,w0
  extract_lets -merge +onlyGivenNames s_3 z3_2 c_3 at hr
  have e_z3_2 : z3_2 = (z3_1 + w0 + 0) % 2^64 := rfl
  have e_c_3 : c_3 = (z3_1 + w0 + 0) / 2^64 := rfl
  clear_value s_3 z3_2 c_3
  have l_z3_2 : z3_2 + 2^64 * c_3 = z3_1 + w0 + 0 := by
    rw [e_z3_2, e_c_3]; exact Nat.mod_add_div _ _
  have b_z3_2 : z3_2 < 2^64 := by rw [e_z3_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_3 : c_3 ≤ 1 := by
    rw [e_c_3]; exact addc_carry_le_one z3_1 w0 0 b_z3_1 b_w0 (by decide)
  clear e_z3_2 e_c_3
  -- a0_1: umulh a0,a0,a0
  extract_lets -merge +onlyGivenNames a0_1 at hr
  have e_a0_1 : a0_1 = a0 * a0 / 2^64 := rfl
  clear_value a0_1
  have p_a0_1 : a0 * a0 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a0 b_a0
  have b_a0_1 : a0_1 < 2^64 := by rw [e_a0_1]; exact Nat.div_lt_of_lt_mul p_a0_1
  have d_a0_1 : z0 + 2^64 * a0_1 = a0 * a0 := by
    rw [e_z0, e_a0_1]; exact Nat.mod_add_div _ _
  clear e_z0 e_a0_1
  -- z4_2: adcs z4,z4,w1
  extract_lets -merge +onlyGivenNames s_4 z4_2 c_4 at hr
  have e_z4_2 : z4_2 = (z4_1 + w1_2 + c_3) % 2^64 := rfl
  have e_c_4 : c_4 = (z4_1 + w1_2 + c_3) / 2^64 := rfl
  clear_value s_4 z4_2 c_4
  have l_z4_2 : z4_2 + 2^64 * c_4 = z4_1 + w1_2 + c_3 := by
    rw [e_z4_2, e_c_4]; exact Nat.mod_add_div _ _
  have b_z4_2 : z4_2 < 2^64 := by rw [e_z4_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_4 : c_4 ≤ 1 := by
    rw [e_c_4]; exact addc_carry_le_one z4_1 w1_2 c_3 b_z4_1 b_w1_2 b_c_3
  clear e_z4_2 e_c_4
  -- w1_3: mul w1,a1,a1
  extract_lets -merge +onlyGivenNames w1_3 at hr
  have e_w1_3 : w1_3 = a1 * a1 % 2^64 := rfl
  clear_value w1_3
  have b_w1_3 : w1_3 < 2^64 := by rw [e_w1_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- z5_1: adcs z5,z5,w2
  extract_lets -merge +onlyGivenNames s_5 z5_1 c_5 at hr
  have e_z5_1 : z5_1 = (z5 + w2_2 + c_4) % 2^64 := rfl
  have e_c_5 : c_5 = (z5 + w2_2 + c_4) / 2^64 := rfl
  clear_value s_5 z5_1 c_5
  have l_z5_1 : z5_1 + 2^64 * c_5 = z5 + w2_2 + c_4 := by
    rw [e_z5_1, e_c_5]; exact Nat.mod_add_div _ _
  have b_z5_1 : z5_1 < 2^64 := by rw [e_z5_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_5 : c_5 ≤ 1 := by
    rw [e_c_5]; exact addc_carry_le_one z5 w2_2 c_4 b_z5 b_w2_2 b_c_4
  clear e_z5_1 e_c_5
  -- a1_1: umulh a1,a1,a1
  extract_lets -merge +onlyGivenNames a1_1 at hr
  have e_a1_1 : a1_1 = a1 * a1 / 2^64 := rfl
  clear_value a1_1
  have p_a1_1 : a1 * a1 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a1 b_a1
  have b_a1_1 : a1_1 < 2^64 := by rw [e_a1_1]; exact Nat.div_lt_of_lt_mul p_a1_1
  have d_a1_1 : w1_3 + 2^64 * a1_1 = a1 * a1 := by
    rw [e_w1_3, e_a1_1]; exact Nat.mod_add_div _ _
  clear e_w1_3 e_a1_1
  -- z6_1: adc z6,z6,xzr
  extract_lets -merge +onlyGivenNames z6_1 at hr
  have e_z6_1 : z6_1 = (z6 + 0 + c_5) % 2^64 := rfl
  clear_value z6_1
  have b_z6_1 : z6_1 < 2^64 := by rw [e_z6_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_z6_1, b_k_z6_1, l_z6_1⟩ :
      ∃ k, k ≤ 1 ∧ z6_1 + 2^64 * k = z6 + 0 + c_5 :=
    ⟨(z6 + 0 + c_5) / 2^64, addc_carry_le_one z6 0 c_5 b_z6 (by decide) b_c_5,
      by rw [e_z6_1]; exact Nat.mod_add_div _ _⟩
  clear e_z6_1
  -- z1_1: adds z1,z1,z1
  extract_lets -merge +onlyGivenNames s_6 z1_1 c_6 at hr
  have e_z1_1 : z1_1 = (z1 + z1 + 0) % 2^64 := rfl
  have e_c_6 : c_6 = (z1 + z1 + 0) / 2^64 := rfl
  clear_value s_6 z1_1 c_6
  have l_z1_1 : z1_1 + 2^64 * c_6 = z1 + z1 + 0 := by
    rw [e_z1_1, e_c_6]; exact Nat.mod_add_div _ _
  have b_z1_1 : z1_1 < 2^64 := by rw [e_z1_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_6 : c_6 ≤ 1 := by
    rw [e_c_6]; exact addc_carry_le_one z1 z1 0 b_z1 b_z1 (by decide)
  clear e_z1_1 e_c_6
  -- w2_3: mul w2,a2,a2
  extract_lets -merge +onlyGivenNames w2_3 at hr
  have e_w2_3 : w2_3 = a2 * a2 % 2^64 := rfl
  clear_value w2_3
  have b_w2_3 : w2_3 < 2^64 := by rw [e_w2_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- z2_2: adcs z2,z2,z2
  extract_lets -merge +onlyGivenNames s_7 z2_2 c_7 at hr
  have e_z2_2 : z2_2 = (z2_1 + z2_1 + c_6) % 2^64 := rfl
  have e_c_7 : c_7 = (z2_1 + z2_1 + c_6) / 2^64 := rfl
  clear_value s_7 z2_2 c_7
  have l_z2_2 : z2_2 + 2^64 * c_7 = z2_1 + z2_1 + c_6 := by
    rw [e_z2_2, e_c_7]; exact Nat.mod_add_div _ _
  have b_z2_2 : z2_2 < 2^64 := by rw [e_z2_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_7 : c_7 ≤ 1 := by
    rw [e_c_7]; exact addc_carry_le_one z2_1 z2_1 c_6 b_z2_1 b_z2_1 b_c_6
  clear e_z2_2 e_c_7
  -- a2_1: umulh a2,a2,a2
  extract_lets -merge +onlyGivenNames a2_1 at hr
  have e_a2_1 : a2_1 = a2 * a2 / 2^64 := rfl
  clear_value a2_1
  have p_a2_1 : a2 * a2 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a2 b_a2
  have b_a2_1 : a2_1 < 2^64 := by rw [e_a2_1]; exact Nat.div_lt_of_lt_mul p_a2_1
  have d_a2_1 : w2_3 + 2^64 * a2_1 = a2 * a2 := by
    rw [e_w2_3, e_a2_1]; exact Nat.mod_add_div _ _
  clear e_w2_3 e_a2_1
  -- z3_3: adcs z3,z3,z3
  extract_lets -merge +onlyGivenNames s_8 z3_3 c_8 at hr
  have e_z3_3 : z3_3 = (z3_2 + z3_2 + c_7) % 2^64 := rfl
  have e_c_8 : c_8 = (z3_2 + z3_2 + c_7) / 2^64 := rfl
  clear_value s_8 z3_3 c_8
  have l_z3_3 : z3_3 + 2^64 * c_8 = z3_2 + z3_2 + c_7 := by
    rw [e_z3_3, e_c_8]; exact Nat.mod_add_div _ _
  have b_z3_3 : z3_3 < 2^64 := by rw [e_z3_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_8 : c_8 ≤ 1 := by
    rw [e_c_8]; exact addc_carry_le_one z3_2 z3_2 c_7 b_z3_2 b_z3_2 b_c_7
  clear e_z3_3 e_c_8
  -- w3_1: mul w3,a3,a3
  extract_lets -merge +onlyGivenNames w3_1 at hr
  have e_w3_1 : w3_1 = a3 * a3 % 2^64 := rfl
  clear_value w3_1
  have b_w3_1 : w3_1 < 2^64 := by rw [e_w3_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- z4_3: adcs z4,z4,z4
  extract_lets -merge +onlyGivenNames s_9 z4_3 c_9 at hr
  have e_z4_3 : z4_3 = (z4_2 + z4_2 + c_8) % 2^64 := rfl
  have e_c_9 : c_9 = (z4_2 + z4_2 + c_8) / 2^64 := rfl
  clear_value s_9 z4_3 c_9
  have l_z4_3 : z4_3 + 2^64 * c_9 = z4_2 + z4_2 + c_8 := by
    rw [e_z4_3, e_c_9]; exact Nat.mod_add_div _ _
  have b_z4_3 : z4_3 < 2^64 := by rw [e_z4_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_9 : c_9 ≤ 1 := by
    rw [e_c_9]; exact addc_carry_le_one z4_2 z4_2 c_8 b_z4_2 b_z4_2 b_c_8
  clear e_z4_3 e_c_9
  -- a3_1: umulh a3,a3,a3
  extract_lets -merge +onlyGivenNames a3_1 at hr
  have e_a3_1 : a3_1 = a3 * a3 / 2^64 := rfl
  clear_value a3_1
  have p_a3_1 : a3 * a3 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_a3 b_a3
  have b_a3_1 : a3_1 < 2^64 := by rw [e_a3_1]; exact Nat.div_lt_of_lt_mul p_a3_1
  have d_a3_1 : w3_1 + 2^64 * a3_1 = a3 * a3 := by
    rw [e_w3_1, e_a3_1]; exact Nat.mod_add_div _ _
  clear e_w3_1 e_a3_1
  -- z5_2: adcs z5,z5,z5
  extract_lets -merge +onlyGivenNames s_10 z5_2 c_10 at hr
  have e_z5_2 : z5_2 = (z5_1 + z5_1 + c_9) % 2^64 := rfl
  have e_c_10 : c_10 = (z5_1 + z5_1 + c_9) / 2^64 := rfl
  clear_value s_10 z5_2 c_10
  have l_z5_2 : z5_2 + 2^64 * c_10 = z5_1 + z5_1 + c_9 := by
    rw [e_z5_2, e_c_10]; exact Nat.mod_add_div _ _
  have b_z5_2 : z5_2 < 2^64 := by rw [e_z5_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_10 : c_10 ≤ 1 := by
    rw [e_c_10]; exact addc_carry_le_one z5_1 z5_1 c_9 b_z5_1 b_z5_1 b_c_9
  clear e_z5_2 e_c_10
  -- z6_2: adcs z6,z6,z6
  extract_lets -merge +onlyGivenNames s_11 z6_2 c_11 at hr
  have e_z6_2 : z6_2 = (z6_1 + z6_1 + c_10) % 2^64 := rfl
  have e_c_11 : c_11 = (z6_1 + z6_1 + c_10) / 2^64 := rfl
  clear_value s_11 z6_2 c_11
  have l_z6_2 : z6_2 + 2^64 * c_11 = z6_1 + z6_1 + c_10 := by
    rw [e_z6_2, e_c_11]; exact Nat.mod_add_div _ _
  have b_z6_2 : z6_2 < 2^64 := by rw [e_z6_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_11 : c_11 ≤ 1 := by
    rw [e_c_11]; exact addc_carry_le_one z6_1 z6_1 c_10 b_z6_1 b_z6_1 b_c_10
  clear e_z6_2 e_c_11
  -- z7: adc z7,xzr,xzr
  extract_lets -merge +onlyGivenNames z7 at hr
  have e_z7 : z7 = (0 + 0 + c_11) % 2^64 := rfl
  clear_value z7
  have b_z7 : z7 < 2^64 := by rw [e_z7]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_z7, b_k_z7, l_z7⟩ :
      ∃ k, k ≤ 1 ∧ z7 + 2^64 * k = 0 + 0 + c_11 :=
    ⟨(0 + 0 + c_11) / 2^64, addc_carry_le_one 0 0 c_11 (by decide) (by decide) b_c_11,
      by rw [e_z7]; exact Nat.mod_add_div _ _⟩
  clear e_z7
  -- q: mul q,inv,z0
  extract_lets -merge +onlyGivenNames q at hr
  have e_q : q = inv' * z0 % 2^64 := rfl
  clear_value q
  have b_q : q < 2^64 := by rw [e_q]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- z1_2: adds z1,z1,a0
  extract_lets -merge +onlyGivenNames s_12 z1_2 c_12 at hr
  have e_z1_2 : z1_2 = (z1_1 + a0_1 + 0) % 2^64 := rfl
  have e_c_12 : c_12 = (z1_1 + a0_1 + 0) / 2^64 := rfl
  clear_value s_12 z1_2 c_12
  have l_z1_2 : z1_2 + 2^64 * c_12 = z1_1 + a0_1 + 0 := by
    rw [e_z1_2, e_c_12]; exact Nat.mod_add_div _ _
  have b_z1_2 : z1_2 < 2^64 := by rw [e_z1_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_12 : c_12 ≤ 1 := by
    rw [e_c_12]; exact addc_carry_le_one z1_1 a0_1 0 b_z1_1 b_a0_1 (by decide)
  clear e_z1_2 e_c_12
  -- z2_3: adcs z2,z2,w1
  extract_lets -merge +onlyGivenNames s_13 z2_3 c_13 at hr
  have e_z2_3 : z2_3 = (z2_2 + w1_3 + c_12) % 2^64 := rfl
  have e_c_13 : c_13 = (z2_2 + w1_3 + c_12) / 2^64 := rfl
  clear_value s_13 z2_3 c_13
  have l_z2_3 : z2_3 + 2^64 * c_13 = z2_2 + w1_3 + c_12 := by
    rw [e_z2_3, e_c_13]; exact Nat.mod_add_div _ _
  have b_z2_3 : z2_3 < 2^64 := by rw [e_z2_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_13 : c_13 ≤ 1 := by
    rw [e_c_13]; exact addc_carry_le_one z2_2 w1_3 c_12 b_z2_2 b_w1_3 b_c_12
  clear e_z2_3 e_c_13
  -- z3_4: adcs z3,z3,a1
  extract_lets -merge +onlyGivenNames s_14 z3_4 c_14 at hr
  have e_z3_4 : z3_4 = (z3_3 + a1_1 + c_13) % 2^64 := rfl
  have e_c_14 : c_14 = (z3_3 + a1_1 + c_13) / 2^64 := rfl
  clear_value s_14 z3_4 c_14
  have l_z3_4 : z3_4 + 2^64 * c_14 = z3_3 + a1_1 + c_13 := by
    rw [e_z3_4, e_c_14]; exact Nat.mod_add_div _ _
  have b_z3_4 : z3_4 < 2^64 := by rw [e_z3_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_14 : c_14 ≤ 1 := by
    rw [e_c_14]; exact addc_carry_le_one z3_3 a1_1 c_13 b_z3_3 b_a1_1 b_c_13
  clear e_z3_4 e_c_14
  -- z4_4: adcs z4,z4,w2
  extract_lets -merge +onlyGivenNames s_15 z4_4 c_15 at hr
  have e_z4_4 : z4_4 = (z4_3 + w2_3 + c_14) % 2^64 := rfl
  have e_c_15 : c_15 = (z4_3 + w2_3 + c_14) / 2^64 := rfl
  clear_value s_15 z4_4 c_15
  have l_z4_4 : z4_4 + 2^64 * c_15 = z4_3 + w2_3 + c_14 := by
    rw [e_z4_4, e_c_15]; exact Nat.mod_add_div _ _
  have b_z4_4 : z4_4 < 2^64 := by rw [e_z4_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_15 : c_15 ≤ 1 := by
    rw [e_c_15]; exact addc_carry_le_one z4_3 w2_3 c_14 b_z4_3 b_w2_3 b_c_14
  clear e_z4_4 e_c_15
  -- z5_3: adcs z5,z5,a2
  extract_lets -merge +onlyGivenNames s_16 z5_3 c_16 at hr
  have e_z5_3 : z5_3 = (z5_2 + a2_1 + c_15) % 2^64 := rfl
  have e_c_16 : c_16 = (z5_2 + a2_1 + c_15) / 2^64 := rfl
  clear_value s_16 z5_3 c_16
  have l_z5_3 : z5_3 + 2^64 * c_16 = z5_2 + a2_1 + c_15 := by
    rw [e_z5_3, e_c_16]; exact Nat.mod_add_div _ _
  have b_z5_3 : z5_3 < 2^64 := by rw [e_z5_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_16 : c_16 ≤ 1 := by
    rw [e_c_16]; exact addc_carry_le_one z5_2 a2_1 c_15 b_z5_2 b_a2_1 b_c_15
  clear e_z5_3 e_c_16
  -- z6_3: adcs z6,z6,w3
  extract_lets -merge +onlyGivenNames s_17 z6_3 c_17 at hr
  have e_z6_3 : z6_3 = (z6_2 + w3_1 + c_16) % 2^64 := rfl
  have e_c_17 : c_17 = (z6_2 + w3_1 + c_16) / 2^64 := rfl
  clear_value s_17 z6_3 c_17
  have l_z6_3 : z6_3 + 2^64 * c_17 = z6_2 + w3_1 + c_16 := by
    rw [e_z6_3, e_c_17]; exact Nat.mod_add_div _ _
  have b_z6_3 : z6_3 < 2^64 := by rw [e_z6_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_17 : c_17 ≤ 1 := by
    rw [e_c_17]; exact addc_carry_le_one z6_2 w3_1 c_16 b_z6_2 b_w3_1 b_c_16
  clear e_z6_3 e_c_17
  -- z7_1: adc z7,z7,a3
  extract_lets -merge +onlyGivenNames z7_1 at hr
  have e_z7_1 : z7_1 = (z7 + a3_1 + c_17) % 2^64 := rfl
  clear_value z7_1
  have b_z7_1 : z7_1 < 2^64 := by rw [e_z7_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_z7_1, b_k_z7_1, l_z7_1⟩ :
      ∃ k, k ≤ 1 ∧ z7_1 + 2^64 * k = z7 + a3_1 + c_17 :=
    ⟨(z7 + a3_1 + c_17) / 2^64, addc_carry_le_one z7 a3_1 c_17 b_z7 b_a3_1 b_c_17,
      by rw [e_z7_1]; exact Nat.mod_add_div _ _⟩
  clear e_z7_1
  -- BEGIN square
  -- The `umulh` results that the schoolbook's `adc`s add a carry to are at most `2^64 - 2`, so
  -- those `adc`s do not wrap.
  have t_z4 : z4 ≤ 2^64 - 2 := by
    have h := Nat.mul_le_mul (Nat.le_sub_one_of_lt b_a3) (Nat.le_sub_one_of_lt b_a0)
    norm_num at h; clear * - h d_z4; omega
  have t_w3 : w3 ≤ 2^64 - 2 := by
    have h := Nat.mul_le_mul (Nat.le_sub_one_of_lt b_a3) (Nat.le_sub_one_of_lt b_a1)
    norm_num at h; clear * - h d_w3; omega
  have t_z6 : z6 ≤ 2^64 - 2 := by
    have h := Nat.mul_le_mul (Nat.le_sub_one_of_lt b_a3) (Nat.le_sub_one_of_lt b_a2)
    norm_num at h; clear * - h d_z6; omega
  have hk_sq : k_z4_1 = 0 ∧ k_w2_2 = 0 ∧ k_z6_1 = 0 ∧ k_z7 = 0 := by
    clear * - l_z4_1 t_z4 b_c_1 l_w2_2 t_w3 b_c_2 l_z6_1 t_z6 b_c_5 l_z7 b_c_11
    omega
  -- The square of `value` as the weighted sum of the ten limb products, in the orientation the
  -- block computes them.
  have hL : value.toNat * value.toNat
      = a0 * a0 + 2^64 * (2 * (a1 * a0)) + 2^128 * (2 * (a2 * a0) + a1 * a1)
        + 2^192 * (2 * (a3 * a0) + 2 * (a2 * a1)) + 2^256 * (2 * (a3 * a1) + a2 * a2)
        + 2^320 * (2 * (a3 * a2)) + 2^384 * (a3 * a3) := by
    simp only [Limbs.toNat]; rw [← e_a0, ← e_a1, ← e_a2, ← e_a3]; ring
  -- The eight limbs and the carry out of the top limb make up the square exactly.
  have hT : z0 + 2^64 * z1_2 + 2^128 * z2_3 + 2^192 * z3_4 + 2^256 * z4_4 + 2^320 * z5_3
        + 2^384 * z6_3 + 2^448 * z7_1 + 2^512 * k_z7_1
      = value.toNat * value.toNat := by
    rw [hL]
    exact sqrMont_square d_w1 d_w2 d_z4 d_w1_1 d_w3 d_z6 d_a0_1 d_a1_1 d_a2_1 d_a3_1
      l_z2_1 l_z3_1 l_z4_1 l_w1_2 l_w2_2 l_z3_2 l_z4_2 l_z5_1 l_z6_1
      l_z1_1 l_z2_2 l_z3_3 l_z4_3 l_z5_2 l_z6_2 l_z7
      l_z1_2 l_z2_3 l_z3_4 l_z4_4 l_z5_3 l_z6_3 l_z7_1 hk_sq.1 hk_sq.2.1 hk_sq.2.2.1 hk_sq.2.2.2
  -- `value < 2^256`, so the square is below `2^512` and the carry out of the top limb is `0`.
  have hk_top : k_z7_1 = 0 := by
    have hAA := Nat.mul_lt_mul'' (Limbs.toNat_lt value hv) (Limbs.toNat_lt value hv)
    clear * - hT hAA
    omega
  -- END square
  -- w1_4: mul w1,p1,q
  extract_lets -merge +onlyGivenNames w1_4 at hr
  have e_w1_4 : w1_4 = p1 * q % 2^64 := rfl
  clear_value w1_4
  have b_w1_4 : w1_4 < 2^64 := by rw [e_w1_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w3_2: lsl w3,q,#62
  extract_lets -merge +onlyGivenNames w3_2 at hr
  have e_w3_2 : w3_2 = q * 2^62 % 2^64 := rfl
  clear_value w3_2
  have b_w3_2 : w3_2 < 2^64 := by rw [e_w3_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_w3_2 : w3_2 + 2^64 * (q / 2^2) = q * 2^62 := by
    rw [e_w3_2]; exact lsl62_lsr2_split _
  -- c_18: subs xzr,z0,#1
  extract_lets -merge +onlyGivenNames c_18 at hr
  have e_c_18 : c_18 = (z0 + 2^64 - 1 - (1 - 1)) / 2^64 := rfl
  clear_value c_18
  have b_c_18 : c_18 ≤ 1 := by rw [e_c_18]; exact subc_carry_le_one z0 1 1 b_z0
  have l_c_18 : (c_18 = 1 ∧ 1 + 1 ≤ z0 + 1) ∨ (c_18 = 0 ∧ z0 + 1 < 1 + 1) :=
    subc_carry_cases z0 1 1 _ e_c_18 b_z0 (by decide) (by decide)
  clear e_c_18
  -- w0_1: umulh w0,p0,q
  extract_lets -merge +onlyGivenNames w0_1 at hr
  have e_w0_1 : w0_1 = p0 * q / 2^64 := rfl
  clear_value w0_1
  have p_w0_1 : p0 * q < 2^64 * 2^64 := Nat.mul_lt_mul'' b_p0 b_q
  have b_w0_1 : w0_1 < 2^64 := by rw [e_w0_1]; exact Nat.div_lt_of_lt_mul p_w0_1
  obtain ⟨lo_w0_1, b_lo_w0_1, d_w0_1⟩ :
      ∃ lo, lo < 2^64 ∧ lo + 2^64 * w0_1 = p0 * q :=
    ⟨p0 * q % 2^64, Nat.mod_lt _ (Nat.two_pow_pos _),
      by rw [e_w0_1]; exact Nat.mod_add_div _ _⟩
  clear e_w0_1
  -- z1_3: adcs z1,z1,w1
  extract_lets -merge +onlyGivenNames s_18 z1_3 c_19 at hr
  have e_z1_3 : z1_3 = (z1_2 + w1_4 + c_18) % 2^64 := rfl
  have e_c_19 : c_19 = (z1_2 + w1_4 + c_18) / 2^64 := rfl
  clear_value s_18 z1_3 c_19
  have l_z1_3 : z1_3 + 2^64 * c_19 = z1_2 + w1_4 + c_18 := by
    rw [e_z1_3, e_c_19]; exact Nat.mod_add_div _ _
  have b_z1_3 : z1_3 < 2^64 := by rw [e_z1_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_19 : c_19 ≤ 1 := by
    rw [e_c_19]; exact addc_carry_le_one z1_2 w1_4 c_18 b_z1_2 b_w1_4 b_c_18
  clear e_z1_3 e_c_19
  -- w1_5: umulh w1,p1,q
  extract_lets -merge +onlyGivenNames w1_5 at hr
  have e_w1_5 : w1_5 = p1 * q / 2^64 := rfl
  clear_value w1_5
  have p_w1_5 : p1 * q < 2^64 * 2^64 := Nat.mul_lt_mul'' b_p1 b_q
  have b_w1_5 : w1_5 < 2^64 := by rw [e_w1_5]; exact Nat.div_lt_of_lt_mul p_w1_5
  have d_w1_5 : w1_4 + 2^64 * w1_5 = p1 * q := by
    rw [e_w1_4, e_w1_5]; exact Nat.mod_add_div _ _
  clear e_w1_4 e_w1_5
  -- z2_4: adcs z2,z2,xzr
  extract_lets -merge +onlyGivenNames s_19 z2_4 c_20 at hr
  have e_z2_4 : z2_4 = (z2_3 + 0 + c_19) % 2^64 := rfl
  have e_c_20 : c_20 = (z2_3 + 0 + c_19) / 2^64 := rfl
  clear_value s_19 z2_4 c_20
  have l_z2_4 : z2_4 + 2^64 * c_20 = z2_3 + 0 + c_19 := by
    rw [e_z2_4, e_c_20]; exact Nat.mod_add_div _ _
  have b_z2_4 : z2_4 < 2^64 := by rw [e_z2_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_20 : c_20 ≤ 1 := by
    rw [e_c_20]; exact addc_carry_le_one z2_3 0 c_19 b_z2_3 (by decide) b_c_19
  clear e_z2_4 e_c_20
  -- z3_5: adcs z3,z3,w3
  extract_lets -merge +onlyGivenNames s_20 z3_5 c_21 at hr
  have e_z3_5 : z3_5 = (z3_4 + w3_2 + c_20) % 2^64 := rfl
  have e_c_21 : c_21 = (z3_4 + w3_2 + c_20) / 2^64 := rfl
  clear_value s_20 z3_5 c_21
  have l_z3_5 : z3_5 + 2^64 * c_21 = z3_4 + w3_2 + c_20 := by
    rw [e_z3_5, e_c_21]; exact Nat.mod_add_div _ _
  have b_z3_5 : z3_5 < 2^64 := by rw [e_z3_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_21 : c_21 ≤ 1 := by
    rw [e_c_21]; exact addc_carry_le_one z3_4 w3_2 c_20 b_z3_4 b_w3_2 b_c_20
  clear e_z3_5 e_c_21
  -- w3_3: lsr w3,q,#2
  extract_lets -merge +onlyGivenNames w3_3 at hr
  have e_w3_3 : w3_3 = q / 2^2 := rfl
  clear_value w3_3
  have b_w3_3 : w3_3 < 2^62 := by
    rw [e_w3_3]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_q (by norm_num))
  -- cy: adc cy,xzr,xzr
  extract_lets -merge +onlyGivenNames cy at hr
  have e_cy : cy = (0 + 0 + c_21) % 2^64 := rfl
  clear_value cy
  have b_cy : cy < 2^64 := by rw [e_cy]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_cy, b_k_cy, l_cy⟩ :
      ∃ k, k ≤ 1 ∧ cy + 2^64 * k = 0 + 0 + c_21 :=
    ⟨(0 + 0 + c_21) / 2^64, addc_carry_le_one 0 0 c_21 (by decide) (by decide) b_c_21,
      by rw [e_cy]; exact Nat.mod_add_div _ _⟩
  clear e_cy
  -- z0_1: adds z0,z1,w0
  extract_lets -merge +onlyGivenNames s_21 z0_1 c_22 at hr
  have e_z0_1 : z0_1 = (z1_3 + w0_1 + 0) % 2^64 := rfl
  have e_c_22 : c_22 = (z1_3 + w0_1 + 0) / 2^64 := rfl
  clear_value s_21 z0_1 c_22
  have l_z0_1 : z0_1 + 2^64 * c_22 = z1_3 + w0_1 + 0 := by
    rw [e_z0_1, e_c_22]; exact Nat.mod_add_div _ _
  have b_z0_1 : z0_1 < 2^64 := by rw [e_z0_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_22 : c_22 ≤ 1 := by
    rw [e_c_22]; exact addc_carry_le_one z1_3 w0_1 0 b_z1_3 b_w0_1 (by decide)
  clear e_z0_1 e_c_22
  -- z1_4: adcs z1,z2,w1
  extract_lets -merge +onlyGivenNames s_22 z1_4 c_23 at hr
  have e_z1_4 : z1_4 = (z2_4 + w1_5 + c_22) % 2^64 := rfl
  have e_c_23 : c_23 = (z2_4 + w1_5 + c_22) / 2^64 := rfl
  clear_value s_22 z1_4 c_23
  have l_z1_4 : z1_4 + 2^64 * c_23 = z2_4 + w1_5 + c_22 := by
    rw [e_z1_4, e_c_23]; exact Nat.mod_add_div _ _
  have b_z1_4 : z1_4 < 2^64 := by rw [e_z1_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_23 : c_23 ≤ 1 := by
    rw [e_c_23]; exact addc_carry_le_one z2_4 w1_5 c_22 b_z2_4 b_w1_5 b_c_22
  clear e_z1_4 e_c_23
  -- z2_5: adcs z2,z3,xzr
  extract_lets -merge +onlyGivenNames s_23 z2_5 c_24 at hr
  have e_z2_5 : z2_5 = (z3_5 + 0 + c_23) % 2^64 := rfl
  have e_c_24 : c_24 = (z3_5 + 0 + c_23) / 2^64 := rfl
  clear_value s_23 z2_5 c_24
  have l_z2_5 : z2_5 + 2^64 * c_24 = z3_5 + 0 + c_23 := by
    rw [e_z2_5, e_c_24]; exact Nat.mod_add_div _ _
  have b_z2_5 : z2_5 < 2^64 := by rw [e_z2_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_24 : c_24 ≤ 1 := by
    rw [e_c_24]; exact addc_carry_le_one z3_5 0 c_23 b_z3_5 (by decide) b_c_23
  clear e_z2_5 e_c_24
  -- q_1: mul q,inv,z0
  extract_lets -merge +onlyGivenNames q_1 at hr
  have e_q_1 : q_1 = inv' * z0_1 % 2^64 := rfl
  clear_value q_1
  have b_q_1 : q_1 < 2^64 := by rw [e_q_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- z3_6: adc z3,cy,w3
  extract_lets -merge +onlyGivenNames z3_6 at hr
  have e_z3_6 : z3_6 = (cy + w3_3 + c_24) % 2^64 := rfl
  clear_value z3_6
  have b_z3_6 : z3_6 < 2^64 := by rw [e_z3_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_z3_6, b_k_z3_6, l_z3_6⟩ :
      ∃ k, k ≤ 1 ∧ z3_6 + 2^64 * k = cy + w3_3 + c_24 :=
    ⟨(cy + w3_3 + c_24) / 2^64, addc_carry_le_one cy w3_3 c_24 b_cy (lt_of_lt_of_le b_w3_3 (by norm_num)) b_c_24,
      by rw [e_z3_6]; exact Nat.mod_add_div _ _⟩
  clear e_z3_6
  -- BEGIN step 0
  -- Cancellation: the low limb of `z0 + p0 * q` is zero, so `z0 + lo_w0_1` is `0` or `2^64`,
  -- and `subs xzr, z0, #1` set the carry exactly when it is `2^64`.
  have hc_0 : z0 + lo_w0_1 = 2^64 * c_18 := by
    have h := cancel_low z0 inv modulus.l0 hinv
    rw [← e_p0, ← e_inv', ← e_q, ← d_w0_1, Nat.add_mul_mod_self_left,
      Nat.mod_eq_of_lt b_lo_w0_1] at h
    clear * - h b_z0 b_lo_w0_1 l_c_18
    omega
  -- Neither `adc` wraps: the carry above limb 3 is at most `1`, and the shifted quotient is
  -- below `2^62`.
  have hk_0 : k_cy = 0 ∧ k_z3_6 = 0 := by
    clear * - l_cy b_c_21 l_z3_6 b_w3_3 b_c_24
    omega
  -- The step's invariant is a linear combination of the instruction equations.
  have I_0 : 2^64 * (z0_1 + 2^64 * z1_4 + 2^128 * z2_5 + 2^192 * z3_6)
      = (z0 + 2^64 * z1_2 + 2^128 * z2_3 + 2^192 * z3_4)
        + (p0 * q + 2^64 * (p1 * q) + 2^254 * q) :=
    sqrMont_step hc_0 d_w0_1 d_w1_5 l_z1_3 l_z2_4 l_z3_5 (by rw [e_w3_3]; exact sh_w3_2)
      l_cy l_z0_1 l_z1_4 l_z2_5 l_z3_6 hk_0
  -- END step 0
  -- w1_6: mul w1,p1,q
  extract_lets -merge +onlyGivenNames w1_6 at hr
  have e_w1_6 : w1_6 = p1 * q_1 % 2^64 := rfl
  clear_value w1_6
  have b_w1_6 : w1_6 < 2^64 := by rw [e_w1_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w3_4: lsl w3,q,#62
  extract_lets -merge +onlyGivenNames w3_4 at hr
  have e_w3_4 : w3_4 = q_1 * 2^62 % 2^64 := rfl
  clear_value w3_4
  have b_w3_4 : w3_4 < 2^64 := by rw [e_w3_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_w3_4 : w3_4 + 2^64 * (q_1 / 2^2) = q_1 * 2^62 := by
    rw [e_w3_4]; exact lsl62_lsr2_split _
  -- c_25: subs xzr,z0,#1
  extract_lets -merge +onlyGivenNames c_25 at hr
  have e_c_25 : c_25 = (z0_1 + 2^64 - 1 - (1 - 1)) / 2^64 := rfl
  clear_value c_25
  have b_c_25 : c_25 ≤ 1 := by rw [e_c_25]; exact subc_carry_le_one z0_1 1 1 b_z0_1
  have l_c_25 : (c_25 = 1 ∧ 1 + 1 ≤ z0_1 + 1) ∨ (c_25 = 0 ∧ z0_1 + 1 < 1 + 1) :=
    subc_carry_cases z0_1 1 1 _ e_c_25 b_z0_1 (by decide) (by decide)
  clear e_c_25
  -- w0_2: umulh w0,p0,q
  extract_lets -merge +onlyGivenNames w0_2 at hr
  have e_w0_2 : w0_2 = p0 * q_1 / 2^64 := rfl
  clear_value w0_2
  have p_w0_2 : p0 * q_1 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_p0 b_q_1
  have b_w0_2 : w0_2 < 2^64 := by rw [e_w0_2]; exact Nat.div_lt_of_lt_mul p_w0_2
  obtain ⟨lo_w0_2, b_lo_w0_2, d_w0_2⟩ :
      ∃ lo, lo < 2^64 ∧ lo + 2^64 * w0_2 = p0 * q_1 :=
    ⟨p0 * q_1 % 2^64, Nat.mod_lt _ (Nat.two_pow_pos _),
      by rw [e_w0_2]; exact Nat.mod_add_div _ _⟩
  clear e_w0_2
  -- z1_5: adcs z1,z1,w1
  extract_lets -merge +onlyGivenNames s_24 z1_5 c_26 at hr
  have e_z1_5 : z1_5 = (z1_4 + w1_6 + c_25) % 2^64 := rfl
  have e_c_26 : c_26 = (z1_4 + w1_6 + c_25) / 2^64 := rfl
  clear_value s_24 z1_5 c_26
  have l_z1_5 : z1_5 + 2^64 * c_26 = z1_4 + w1_6 + c_25 := by
    rw [e_z1_5, e_c_26]; exact Nat.mod_add_div _ _
  have b_z1_5 : z1_5 < 2^64 := by rw [e_z1_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_26 : c_26 ≤ 1 := by
    rw [e_c_26]; exact addc_carry_le_one z1_4 w1_6 c_25 b_z1_4 b_w1_6 b_c_25
  clear e_z1_5 e_c_26
  -- w1_7: umulh w1,p1,q
  extract_lets -merge +onlyGivenNames w1_7 at hr
  have e_w1_7 : w1_7 = p1 * q_1 / 2^64 := rfl
  clear_value w1_7
  have p_w1_7 : p1 * q_1 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_p1 b_q_1
  have b_w1_7 : w1_7 < 2^64 := by rw [e_w1_7]; exact Nat.div_lt_of_lt_mul p_w1_7
  have d_w1_7 : w1_6 + 2^64 * w1_7 = p1 * q_1 := by
    rw [e_w1_6, e_w1_7]; exact Nat.mod_add_div _ _
  clear e_w1_6 e_w1_7
  -- z2_6: adcs z2,z2,xzr
  extract_lets -merge +onlyGivenNames s_25 z2_6 c_27 at hr
  have e_z2_6 : z2_6 = (z2_5 + 0 + c_26) % 2^64 := rfl
  have e_c_27 : c_27 = (z2_5 + 0 + c_26) / 2^64 := rfl
  clear_value s_25 z2_6 c_27
  have l_z2_6 : z2_6 + 2^64 * c_27 = z2_5 + 0 + c_26 := by
    rw [e_z2_6, e_c_27]; exact Nat.mod_add_div _ _
  have b_z2_6 : z2_6 < 2^64 := by rw [e_z2_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_27 : c_27 ≤ 1 := by
    rw [e_c_27]; exact addc_carry_le_one z2_5 0 c_26 b_z2_5 (by decide) b_c_26
  clear e_z2_6 e_c_27
  -- z3_7: adcs z3,z3,w3
  extract_lets -merge +onlyGivenNames s_26 z3_7 c_28 at hr
  have e_z3_7 : z3_7 = (z3_6 + w3_4 + c_27) % 2^64 := rfl
  have e_c_28 : c_28 = (z3_6 + w3_4 + c_27) / 2^64 := rfl
  clear_value s_26 z3_7 c_28
  have l_z3_7 : z3_7 + 2^64 * c_28 = z3_6 + w3_4 + c_27 := by
    rw [e_z3_7, e_c_28]; exact Nat.mod_add_div _ _
  have b_z3_7 : z3_7 < 2^64 := by rw [e_z3_7]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_28 : c_28 ≤ 1 := by
    rw [e_c_28]; exact addc_carry_le_one z3_6 w3_4 c_27 b_z3_6 b_w3_4 b_c_27
  clear e_z3_7 e_c_28
  -- w3_5: lsr w3,q,#2
  extract_lets -merge +onlyGivenNames w3_5 at hr
  have e_w3_5 : w3_5 = q_1 / 2^2 := rfl
  clear_value w3_5
  have b_w3_5 : w3_5 < 2^62 := by
    rw [e_w3_5]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_q_1 (by norm_num))
  -- cy_1: adc cy,xzr,xzr
  extract_lets -merge +onlyGivenNames cy_1 at hr
  have e_cy_1 : cy_1 = (0 + 0 + c_28) % 2^64 := rfl
  clear_value cy_1
  have b_cy_1 : cy_1 < 2^64 := by rw [e_cy_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_cy_1, b_k_cy_1, l_cy_1⟩ :
      ∃ k, k ≤ 1 ∧ cy_1 + 2^64 * k = 0 + 0 + c_28 :=
    ⟨(0 + 0 + c_28) / 2^64, addc_carry_le_one 0 0 c_28 (by decide) (by decide) b_c_28,
      by rw [e_cy_1]; exact Nat.mod_add_div _ _⟩
  clear e_cy_1
  -- z0_2: adds z0,z1,w0
  extract_lets -merge +onlyGivenNames s_27 z0_2 c_29 at hr
  have e_z0_2 : z0_2 = (z1_5 + w0_2 + 0) % 2^64 := rfl
  have e_c_29 : c_29 = (z1_5 + w0_2 + 0) / 2^64 := rfl
  clear_value s_27 z0_2 c_29
  have l_z0_2 : z0_2 + 2^64 * c_29 = z1_5 + w0_2 + 0 := by
    rw [e_z0_2, e_c_29]; exact Nat.mod_add_div _ _
  have b_z0_2 : z0_2 < 2^64 := by rw [e_z0_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_29 : c_29 ≤ 1 := by
    rw [e_c_29]; exact addc_carry_le_one z1_5 w0_2 0 b_z1_5 b_w0_2 (by decide)
  clear e_z0_2 e_c_29
  -- z1_6: adcs z1,z2,w1
  extract_lets -merge +onlyGivenNames s_28 z1_6 c_30 at hr
  have e_z1_6 : z1_6 = (z2_6 + w1_7 + c_29) % 2^64 := rfl
  have e_c_30 : c_30 = (z2_6 + w1_7 + c_29) / 2^64 := rfl
  clear_value s_28 z1_6 c_30
  have l_z1_6 : z1_6 + 2^64 * c_30 = z2_6 + w1_7 + c_29 := by
    rw [e_z1_6, e_c_30]; exact Nat.mod_add_div _ _
  have b_z1_6 : z1_6 < 2^64 := by rw [e_z1_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_30 : c_30 ≤ 1 := by
    rw [e_c_30]; exact addc_carry_le_one z2_6 w1_7 c_29 b_z2_6 b_w1_7 b_c_29
  clear e_z1_6 e_c_30
  -- z2_7: adcs z2,z3,xzr
  extract_lets -merge +onlyGivenNames s_29 z2_7 c_31 at hr
  have e_z2_7 : z2_7 = (z3_7 + 0 + c_30) % 2^64 := rfl
  have e_c_31 : c_31 = (z3_7 + 0 + c_30) / 2^64 := rfl
  clear_value s_29 z2_7 c_31
  have l_z2_7 : z2_7 + 2^64 * c_31 = z3_7 + 0 + c_30 := by
    rw [e_z2_7, e_c_31]; exact Nat.mod_add_div _ _
  have b_z2_7 : z2_7 < 2^64 := by rw [e_z2_7]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_31 : c_31 ≤ 1 := by
    rw [e_c_31]; exact addc_carry_le_one z3_7 0 c_30 b_z3_7 (by decide) b_c_30
  clear e_z2_7 e_c_31
  -- q_2: mul q,inv,z0
  extract_lets -merge +onlyGivenNames q_2 at hr
  have e_q_2 : q_2 = inv' * z0_2 % 2^64 := rfl
  clear_value q_2
  have b_q_2 : q_2 < 2^64 := by rw [e_q_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- z3_8: adc z3,cy,w3
  extract_lets -merge +onlyGivenNames z3_8 at hr
  have e_z3_8 : z3_8 = (cy_1 + w3_5 + c_31) % 2^64 := rfl
  clear_value z3_8
  have b_z3_8 : z3_8 < 2^64 := by rw [e_z3_8]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_z3_8, b_k_z3_8, l_z3_8⟩ :
      ∃ k, k ≤ 1 ∧ z3_8 + 2^64 * k = cy_1 + w3_5 + c_31 :=
    ⟨(cy_1 + w3_5 + c_31) / 2^64, addc_carry_le_one cy_1 w3_5 c_31 b_cy_1 (lt_of_lt_of_le b_w3_5 (by norm_num)) b_c_31,
      by rw [e_z3_8]; exact Nat.mod_add_div _ _⟩
  clear e_z3_8
  -- BEGIN step 1
  -- Cancellation: the low limb of `z0_1 + p0 * q_1` is zero, so `z0_1 + lo_w0_2` is `0` or `2^64`,
  -- and `subs xzr, z0_1, #1` set the carry exactly when it is `2^64`.
  have hc_1 : z0_1 + lo_w0_2 = 2^64 * c_25 := by
    have h := cancel_low z0_1 inv modulus.l0 hinv
    rw [← e_p0, ← e_inv', ← e_q_1, ← d_w0_2, Nat.add_mul_mod_self_left,
      Nat.mod_eq_of_lt b_lo_w0_2] at h
    clear * - h b_z0_1 b_lo_w0_2 l_c_25
    omega
  -- Neither `adc` wraps: the carry above limb 3 is at most `1`, and the shifted quotient is
  -- below `2^62`.
  have hk_1 : k_cy_1 = 0 ∧ k_z3_8 = 0 := by
    clear * - l_cy_1 b_c_28 l_z3_8 b_w3_5 b_c_31
    omega
  -- The step's invariant is a linear combination of the instruction equations.
  have I_1 : 2^64 * (z0_2 + 2^64 * z1_6 + 2^128 * z2_7 + 2^192 * z3_8)
      = (z0_1 + 2^64 * z1_4 + 2^128 * z2_5 + 2^192 * z3_6)
        + (p0 * q_1 + 2^64 * (p1 * q_1) + 2^254 * q_1) :=
    sqrMont_step hc_1 d_w0_2 d_w1_7 l_z1_5 l_z2_6 l_z3_7 (by rw [e_w3_5]; exact sh_w3_4)
      l_cy_1 l_z0_2 l_z1_6 l_z2_7 l_z3_8 hk_1
  -- END step 1
  -- w1_8: mul w1,p1,q
  extract_lets -merge +onlyGivenNames w1_8 at hr
  have e_w1_8 : w1_8 = p1 * q_2 % 2^64 := rfl
  clear_value w1_8
  have b_w1_8 : w1_8 < 2^64 := by rw [e_w1_8]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w3_6: lsl w3,q,#62
  extract_lets -merge +onlyGivenNames w3_6 at hr
  have e_w3_6 : w3_6 = q_2 * 2^62 % 2^64 := rfl
  clear_value w3_6
  have b_w3_6 : w3_6 < 2^64 := by rw [e_w3_6]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_w3_6 : w3_6 + 2^64 * (q_2 / 2^2) = q_2 * 2^62 := by
    rw [e_w3_6]; exact lsl62_lsr2_split _
  -- c_32: subs xzr,z0,#1
  extract_lets -merge +onlyGivenNames c_32 at hr
  have e_c_32 : c_32 = (z0_2 + 2^64 - 1 - (1 - 1)) / 2^64 := rfl
  clear_value c_32
  have b_c_32 : c_32 ≤ 1 := by rw [e_c_32]; exact subc_carry_le_one z0_2 1 1 b_z0_2
  have l_c_32 : (c_32 = 1 ∧ 1 + 1 ≤ z0_2 + 1) ∨ (c_32 = 0 ∧ z0_2 + 1 < 1 + 1) :=
    subc_carry_cases z0_2 1 1 _ e_c_32 b_z0_2 (by decide) (by decide)
  clear e_c_32
  -- w0_3: umulh w0,p0,q
  extract_lets -merge +onlyGivenNames w0_3 at hr
  have e_w0_3 : w0_3 = p0 * q_2 / 2^64 := rfl
  clear_value w0_3
  have p_w0_3 : p0 * q_2 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_p0 b_q_2
  have b_w0_3 : w0_3 < 2^64 := by rw [e_w0_3]; exact Nat.div_lt_of_lt_mul p_w0_3
  obtain ⟨lo_w0_3, b_lo_w0_3, d_w0_3⟩ :
      ∃ lo, lo < 2^64 ∧ lo + 2^64 * w0_3 = p0 * q_2 :=
    ⟨p0 * q_2 % 2^64, Nat.mod_lt _ (Nat.two_pow_pos _),
      by rw [e_w0_3]; exact Nat.mod_add_div _ _⟩
  clear e_w0_3
  -- z1_7: adcs z1,z1,w1
  extract_lets -merge +onlyGivenNames s_30 z1_7 c_33 at hr
  have e_z1_7 : z1_7 = (z1_6 + w1_8 + c_32) % 2^64 := rfl
  have e_c_33 : c_33 = (z1_6 + w1_8 + c_32) / 2^64 := rfl
  clear_value s_30 z1_7 c_33
  have l_z1_7 : z1_7 + 2^64 * c_33 = z1_6 + w1_8 + c_32 := by
    rw [e_z1_7, e_c_33]; exact Nat.mod_add_div _ _
  have b_z1_7 : z1_7 < 2^64 := by rw [e_z1_7]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_33 : c_33 ≤ 1 := by
    rw [e_c_33]; exact addc_carry_le_one z1_6 w1_8 c_32 b_z1_6 b_w1_8 b_c_32
  clear e_z1_7 e_c_33
  -- w1_9: umulh w1,p1,q
  extract_lets -merge +onlyGivenNames w1_9 at hr
  have e_w1_9 : w1_9 = p1 * q_2 / 2^64 := rfl
  clear_value w1_9
  have p_w1_9 : p1 * q_2 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_p1 b_q_2
  have b_w1_9 : w1_9 < 2^64 := by rw [e_w1_9]; exact Nat.div_lt_of_lt_mul p_w1_9
  have d_w1_9 : w1_8 + 2^64 * w1_9 = p1 * q_2 := by
    rw [e_w1_8, e_w1_9]; exact Nat.mod_add_div _ _
  clear e_w1_8 e_w1_9
  -- z2_8: adcs z2,z2,xzr
  extract_lets -merge +onlyGivenNames s_31 z2_8 c_34 at hr
  have e_z2_8 : z2_8 = (z2_7 + 0 + c_33) % 2^64 := rfl
  have e_c_34 : c_34 = (z2_7 + 0 + c_33) / 2^64 := rfl
  clear_value s_31 z2_8 c_34
  have l_z2_8 : z2_8 + 2^64 * c_34 = z2_7 + 0 + c_33 := by
    rw [e_z2_8, e_c_34]; exact Nat.mod_add_div _ _
  have b_z2_8 : z2_8 < 2^64 := by rw [e_z2_8]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_34 : c_34 ≤ 1 := by
    rw [e_c_34]; exact addc_carry_le_one z2_7 0 c_33 b_z2_7 (by decide) b_c_33
  clear e_z2_8 e_c_34
  -- z3_9: adcs z3,z3,w3
  extract_lets -merge +onlyGivenNames s_32 z3_9 c_35 at hr
  have e_z3_9 : z3_9 = (z3_8 + w3_6 + c_34) % 2^64 := rfl
  have e_c_35 : c_35 = (z3_8 + w3_6 + c_34) / 2^64 := rfl
  clear_value s_32 z3_9 c_35
  have l_z3_9 : z3_9 + 2^64 * c_35 = z3_8 + w3_6 + c_34 := by
    rw [e_z3_9, e_c_35]; exact Nat.mod_add_div _ _
  have b_z3_9 : z3_9 < 2^64 := by rw [e_z3_9]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_35 : c_35 ≤ 1 := by
    rw [e_c_35]; exact addc_carry_le_one z3_8 w3_6 c_34 b_z3_8 b_w3_6 b_c_34
  clear e_z3_9 e_c_35
  -- w3_7: lsr w3,q,#2
  extract_lets -merge +onlyGivenNames w3_7 at hr
  have e_w3_7 : w3_7 = q_2 / 2^2 := rfl
  clear_value w3_7
  have b_w3_7 : w3_7 < 2^62 := by
    rw [e_w3_7]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_q_2 (by norm_num))
  -- cy_2: adc cy,xzr,xzr
  extract_lets -merge +onlyGivenNames cy_2 at hr
  have e_cy_2 : cy_2 = (0 + 0 + c_35) % 2^64 := rfl
  clear_value cy_2
  have b_cy_2 : cy_2 < 2^64 := by rw [e_cy_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_cy_2, b_k_cy_2, l_cy_2⟩ :
      ∃ k, k ≤ 1 ∧ cy_2 + 2^64 * k = 0 + 0 + c_35 :=
    ⟨(0 + 0 + c_35) / 2^64, addc_carry_le_one 0 0 c_35 (by decide) (by decide) b_c_35,
      by rw [e_cy_2]; exact Nat.mod_add_div _ _⟩
  clear e_cy_2
  -- z0_3: adds z0,z1,w0
  extract_lets -merge +onlyGivenNames s_33 z0_3 c_36 at hr
  have e_z0_3 : z0_3 = (z1_7 + w0_3 + 0) % 2^64 := rfl
  have e_c_36 : c_36 = (z1_7 + w0_3 + 0) / 2^64 := rfl
  clear_value s_33 z0_3 c_36
  have l_z0_3 : z0_3 + 2^64 * c_36 = z1_7 + w0_3 + 0 := by
    rw [e_z0_3, e_c_36]; exact Nat.mod_add_div _ _
  have b_z0_3 : z0_3 < 2^64 := by rw [e_z0_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_36 : c_36 ≤ 1 := by
    rw [e_c_36]; exact addc_carry_le_one z1_7 w0_3 0 b_z1_7 b_w0_3 (by decide)
  clear e_z0_3 e_c_36
  -- z1_8: adcs z1,z2,w1
  extract_lets -merge +onlyGivenNames s_34 z1_8 c_37 at hr
  have e_z1_8 : z1_8 = (z2_8 + w1_9 + c_36) % 2^64 := rfl
  have e_c_37 : c_37 = (z2_8 + w1_9 + c_36) / 2^64 := rfl
  clear_value s_34 z1_8 c_37
  have l_z1_8 : z1_8 + 2^64 * c_37 = z2_8 + w1_9 + c_36 := by
    rw [e_z1_8, e_c_37]; exact Nat.mod_add_div _ _
  have b_z1_8 : z1_8 < 2^64 := by rw [e_z1_8]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_37 : c_37 ≤ 1 := by
    rw [e_c_37]; exact addc_carry_le_one z2_8 w1_9 c_36 b_z2_8 b_w1_9 b_c_36
  clear e_z1_8 e_c_37
  -- z2_9: adcs z2,z3,xzr
  extract_lets -merge +onlyGivenNames s_35 z2_9 c_38 at hr
  have e_z2_9 : z2_9 = (z3_9 + 0 + c_37) % 2^64 := rfl
  have e_c_38 : c_38 = (z3_9 + 0 + c_37) / 2^64 := rfl
  clear_value s_35 z2_9 c_38
  have l_z2_9 : z2_9 + 2^64 * c_38 = z3_9 + 0 + c_37 := by
    rw [e_z2_9, e_c_38]; exact Nat.mod_add_div _ _
  have b_z2_9 : z2_9 < 2^64 := by rw [e_z2_9]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_38 : c_38 ≤ 1 := by
    rw [e_c_38]; exact addc_carry_le_one z3_9 0 c_37 b_z3_9 (by decide) b_c_37
  clear e_z2_9 e_c_38
  -- q_3: mul q,inv,z0
  extract_lets -merge +onlyGivenNames q_3 at hr
  have e_q_3 : q_3 = inv' * z0_3 % 2^64 := rfl
  clear_value q_3
  have b_q_3 : q_3 < 2^64 := by rw [e_q_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- z3_10: adc z3,cy,w3
  extract_lets -merge +onlyGivenNames z3_10 at hr
  have e_z3_10 : z3_10 = (cy_2 + w3_7 + c_38) % 2^64 := rfl
  clear_value z3_10
  have b_z3_10 : z3_10 < 2^64 := by rw [e_z3_10]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_z3_10, b_k_z3_10, l_z3_10⟩ :
      ∃ k, k ≤ 1 ∧ z3_10 + 2^64 * k = cy_2 + w3_7 + c_38 :=
    ⟨(cy_2 + w3_7 + c_38) / 2^64, addc_carry_le_one cy_2 w3_7 c_38 b_cy_2 (lt_of_lt_of_le b_w3_7 (by norm_num)) b_c_38,
      by rw [e_z3_10]; exact Nat.mod_add_div _ _⟩
  clear e_z3_10
  -- BEGIN step 2
  -- Cancellation: the low limb of `z0_2 + p0 * q_2` is zero, so `z0_2 + lo_w0_3` is `0` or `2^64`,
  -- and `subs xzr, z0_2, #1` set the carry exactly when it is `2^64`.
  have hc_2 : z0_2 + lo_w0_3 = 2^64 * c_32 := by
    have h := cancel_low z0_2 inv modulus.l0 hinv
    rw [← e_p0, ← e_inv', ← e_q_2, ← d_w0_3, Nat.add_mul_mod_self_left,
      Nat.mod_eq_of_lt b_lo_w0_3] at h
    clear * - h b_z0_2 b_lo_w0_3 l_c_32
    omega
  -- Neither `adc` wraps: the carry above limb 3 is at most `1`, and the shifted quotient is
  -- below `2^62`.
  have hk_2 : k_cy_2 = 0 ∧ k_z3_10 = 0 := by
    clear * - l_cy_2 b_c_35 l_z3_10 b_w3_7 b_c_38
    omega
  -- The step's invariant is a linear combination of the instruction equations.
  have I_2 : 2^64 * (z0_3 + 2^64 * z1_8 + 2^128 * z2_9 + 2^192 * z3_10)
      = (z0_2 + 2^64 * z1_6 + 2^128 * z2_7 + 2^192 * z3_8)
        + (p0 * q_2 + 2^64 * (p1 * q_2) + 2^254 * q_2) :=
    sqrMont_step hc_2 d_w0_3 d_w1_9 l_z1_7 l_z2_8 l_z3_9 (by rw [e_w3_7]; exact sh_w3_6)
      l_cy_2 l_z0_3 l_z1_8 l_z2_9 l_z3_10 hk_2
  -- END step 2
  -- w1_10: mul w1,p1,q
  extract_lets -merge +onlyGivenNames w1_10 at hr
  have e_w1_10 : w1_10 = p1 * q_3 % 2^64 := rfl
  clear_value w1_10
  have b_w1_10 : w1_10 < 2^64 := by rw [e_w1_10]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w3_8: lsl w3,q,#62
  extract_lets -merge +onlyGivenNames w3_8 at hr
  have e_w3_8 : w3_8 = q_3 * 2^62 % 2^64 := rfl
  clear_value w3_8
  have b_w3_8 : w3_8 < 2^64 := by rw [e_w3_8]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_w3_8 : w3_8 + 2^64 * (q_3 / 2^2) = q_3 * 2^62 := by
    rw [e_w3_8]; exact lsl62_lsr2_split _
  -- c_39: subs xzr,z0,#1
  extract_lets -merge +onlyGivenNames c_39 at hr
  have e_c_39 : c_39 = (z0_3 + 2^64 - 1 - (1 - 1)) / 2^64 := rfl
  clear_value c_39
  have b_c_39 : c_39 ≤ 1 := by rw [e_c_39]; exact subc_carry_le_one z0_3 1 1 b_z0_3
  have l_c_39 : (c_39 = 1 ∧ 1 + 1 ≤ z0_3 + 1) ∨ (c_39 = 0 ∧ z0_3 + 1 < 1 + 1) :=
    subc_carry_cases z0_3 1 1 _ e_c_39 b_z0_3 (by decide) (by decide)
  clear e_c_39
  -- w0_4: umulh w0,p0,q
  extract_lets -merge +onlyGivenNames w0_4 at hr
  have e_w0_4 : w0_4 = p0 * q_3 / 2^64 := rfl
  clear_value w0_4
  have p_w0_4 : p0 * q_3 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_p0 b_q_3
  have b_w0_4 : w0_4 < 2^64 := by rw [e_w0_4]; exact Nat.div_lt_of_lt_mul p_w0_4
  obtain ⟨lo_w0_4, b_lo_w0_4, d_w0_4⟩ :
      ∃ lo, lo < 2^64 ∧ lo + 2^64 * w0_4 = p0 * q_3 :=
    ⟨p0 * q_3 % 2^64, Nat.mod_lt _ (Nat.two_pow_pos _),
      by rw [e_w0_4]; exact Nat.mod_add_div _ _⟩
  clear e_w0_4
  -- z1_9: adcs z1,z1,w1
  extract_lets -merge +onlyGivenNames s_36 z1_9 c_40 at hr
  have e_z1_9 : z1_9 = (z1_8 + w1_10 + c_39) % 2^64 := rfl
  have e_c_40 : c_40 = (z1_8 + w1_10 + c_39) / 2^64 := rfl
  clear_value s_36 z1_9 c_40
  have l_z1_9 : z1_9 + 2^64 * c_40 = z1_8 + w1_10 + c_39 := by
    rw [e_z1_9, e_c_40]; exact Nat.mod_add_div _ _
  have b_z1_9 : z1_9 < 2^64 := by rw [e_z1_9]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_40 : c_40 ≤ 1 := by
    rw [e_c_40]; exact addc_carry_le_one z1_8 w1_10 c_39 b_z1_8 b_w1_10 b_c_39
  clear e_z1_9 e_c_40
  -- w1_11: umulh w1,p1,q
  extract_lets -merge +onlyGivenNames w1_11 at hr
  have e_w1_11 : w1_11 = p1 * q_3 / 2^64 := rfl
  clear_value w1_11
  have p_w1_11 : p1 * q_3 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_p1 b_q_3
  have b_w1_11 : w1_11 < 2^64 := by rw [e_w1_11]; exact Nat.div_lt_of_lt_mul p_w1_11
  have d_w1_11 : w1_10 + 2^64 * w1_11 = p1 * q_3 := by
    rw [e_w1_10, e_w1_11]; exact Nat.mod_add_div _ _
  clear e_w1_10 e_w1_11
  -- z2_10: adcs z2,z2,xzr
  extract_lets -merge +onlyGivenNames s_37 z2_10 c_41 at hr
  have e_z2_10 : z2_10 = (z2_9 + 0 + c_40) % 2^64 := rfl
  have e_c_41 : c_41 = (z2_9 + 0 + c_40) / 2^64 := rfl
  clear_value s_37 z2_10 c_41
  have l_z2_10 : z2_10 + 2^64 * c_41 = z2_9 + 0 + c_40 := by
    rw [e_z2_10, e_c_41]; exact Nat.mod_add_div _ _
  have b_z2_10 : z2_10 < 2^64 := by rw [e_z2_10]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_41 : c_41 ≤ 1 := by
    rw [e_c_41]; exact addc_carry_le_one z2_9 0 c_40 b_z2_9 (by decide) b_c_40
  clear e_z2_10 e_c_41
  -- z3_11: adcs z3,z3,w3
  extract_lets -merge +onlyGivenNames s_38 z3_11 c_42 at hr
  have e_z3_11 : z3_11 = (z3_10 + w3_8 + c_41) % 2^64 := rfl
  have e_c_42 : c_42 = (z3_10 + w3_8 + c_41) / 2^64 := rfl
  clear_value s_38 z3_11 c_42
  have l_z3_11 : z3_11 + 2^64 * c_42 = z3_10 + w3_8 + c_41 := by
    rw [e_z3_11, e_c_42]; exact Nat.mod_add_div _ _
  have b_z3_11 : z3_11 < 2^64 := by rw [e_z3_11]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_42 : c_42 ≤ 1 := by
    rw [e_c_42]; exact addc_carry_le_one z3_10 w3_8 c_41 b_z3_10 b_w3_8 b_c_41
  clear e_z3_11 e_c_42
  -- w3_9: lsr w3,q,#2
  extract_lets -merge +onlyGivenNames w3_9 at hr
  have e_w3_9 : w3_9 = q_3 / 2^2 := rfl
  clear_value w3_9
  have b_w3_9 : w3_9 < 2^62 := by
    rw [e_w3_9]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_q_3 (by norm_num))
  -- cy_3: adc cy,xzr,xzr
  extract_lets -merge +onlyGivenNames cy_3 at hr
  have e_cy_3 : cy_3 = (0 + 0 + c_42) % 2^64 := rfl
  clear_value cy_3
  have b_cy_3 : cy_3 < 2^64 := by rw [e_cy_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_cy_3, b_k_cy_3, l_cy_3⟩ :
      ∃ k, k ≤ 1 ∧ cy_3 + 2^64 * k = 0 + 0 + c_42 :=
    ⟨(0 + 0 + c_42) / 2^64, addc_carry_le_one 0 0 c_42 (by decide) (by decide) b_c_42,
      by rw [e_cy_3]; exact Nat.mod_add_div _ _⟩
  clear e_cy_3
  -- z0_4: adds z0,z1,w0
  extract_lets -merge +onlyGivenNames s_39 z0_4 c_43 at hr
  have e_z0_4 : z0_4 = (z1_9 + w0_4 + 0) % 2^64 := rfl
  have e_c_43 : c_43 = (z1_9 + w0_4 + 0) / 2^64 := rfl
  clear_value s_39 z0_4 c_43
  have l_z0_4 : z0_4 + 2^64 * c_43 = z1_9 + w0_4 + 0 := by
    rw [e_z0_4, e_c_43]; exact Nat.mod_add_div _ _
  have b_z0_4 : z0_4 < 2^64 := by rw [e_z0_4]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_43 : c_43 ≤ 1 := by
    rw [e_c_43]; exact addc_carry_le_one z1_9 w0_4 0 b_z1_9 b_w0_4 (by decide)
  clear e_z0_4 e_c_43
  -- z1_10: adcs z1,z2,w1
  extract_lets -merge +onlyGivenNames s_40 z1_10 c_44 at hr
  have e_z1_10 : z1_10 = (z2_10 + w1_11 + c_43) % 2^64 := rfl
  have e_c_44 : c_44 = (z2_10 + w1_11 + c_43) / 2^64 := rfl
  clear_value s_40 z1_10 c_44
  have l_z1_10 : z1_10 + 2^64 * c_44 = z2_10 + w1_11 + c_43 := by
    rw [e_z1_10, e_c_44]; exact Nat.mod_add_div _ _
  have b_z1_10 : z1_10 < 2^64 := by rw [e_z1_10]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_44 : c_44 ≤ 1 := by
    rw [e_c_44]; exact addc_carry_le_one z2_10 w1_11 c_43 b_z2_10 b_w1_11 b_c_43
  clear e_z1_10 e_c_44
  -- z2_11: adcs z2,z3,xzr
  extract_lets -merge +onlyGivenNames s_41 z2_11 c_45 at hr
  have e_z2_11 : z2_11 = (z3_11 + 0 + c_44) % 2^64 := rfl
  have e_c_45 : c_45 = (z3_11 + 0 + c_44) / 2^64 := rfl
  clear_value s_41 z2_11 c_45
  have l_z2_11 : z2_11 + 2^64 * c_45 = z3_11 + 0 + c_44 := by
    rw [e_z2_11, e_c_45]; exact Nat.mod_add_div _ _
  have b_z2_11 : z2_11 < 2^64 := by rw [e_z2_11]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_45 : c_45 ≤ 1 := by
    rw [e_c_45]; exact addc_carry_le_one z3_11 0 c_44 b_z3_11 (by decide) b_c_44
  clear e_z2_11 e_c_45
  -- z3_12: adc z3,cy,w3
  extract_lets -merge +onlyGivenNames z3_12 at hr
  have e_z3_12 : z3_12 = (cy_3 + w3_9 + c_45) % 2^64 := rfl
  clear_value z3_12
  have b_z3_12 : z3_12 < 2^64 := by rw [e_z3_12]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_z3_12, b_k_z3_12, l_z3_12⟩ :
      ∃ k, k ≤ 1 ∧ z3_12 + 2^64 * k = cy_3 + w3_9 + c_45 :=
    ⟨(cy_3 + w3_9 + c_45) / 2^64, addc_carry_le_one cy_3 w3_9 c_45 b_cy_3 (lt_of_lt_of_le b_w3_9 (by norm_num)) b_c_45,
      by rw [e_z3_12]; exact Nat.mod_add_div _ _⟩
  clear e_z3_12
  -- BEGIN step 3
  -- Cancellation: the low limb of `z0_3 + p0 * q_3` is zero, so `z0_3 + lo_w0_4` is `0` or `2^64`,
  -- and `subs xzr, z0_3, #1` set the carry exactly when it is `2^64`.
  have hc_3 : z0_3 + lo_w0_4 = 2^64 * c_39 := by
    have h := cancel_low z0_3 inv modulus.l0 hinv
    rw [← e_p0, ← e_inv', ← e_q_3, ← d_w0_4, Nat.add_mul_mod_self_left,
      Nat.mod_eq_of_lt b_lo_w0_4] at h
    clear * - h b_z0_3 b_lo_w0_4 l_c_39
    omega
  -- Neither `adc` wraps: the carry above limb 3 is at most `1`, and the shifted quotient is
  -- below `2^62`.
  have hk_3 : k_cy_3 = 0 ∧ k_z3_12 = 0 := by
    clear * - l_cy_3 b_c_42 l_z3_12 b_w3_9 b_c_45
    omega
  -- The step's invariant is a linear combination of the instruction equations.
  have I_3 : 2^64 * (z0_4 + 2^64 * z1_10 + 2^128 * z2_11 + 2^192 * z3_12)
      = (z0_3 + 2^64 * z1_8 + 2^128 * z2_9 + 2^192 * z3_10)
        + (p0 * q_3 + 2^64 * (p1 * q_3) + 2^254 * q_3) :=
    sqrMont_step hc_3 d_w0_4 d_w1_11 l_z1_9 l_z2_10 l_z3_11 (by rw [e_w3_9]; exact sh_w3_8)
      l_cy_3 l_z0_4 l_z1_10 l_z2_11 l_z3_12 hk_3
  -- END step 3
  -- BEGIN reduction
  have hQ : q + 2^64 * q_1 + 2^128 * q_2 + 2^192 * q_3 < 2^256 := by
    clear * - b_q b_q_1 b_q_2 b_q_3; omega
  -- `Q * p`, with the modulus in the shape the code assumes, as the sum of the four steps'
  -- contributions.
  have hQp : (q + 2^64 * q_1 + 2^128 * q_2 + 2^192 * q_3) * modulus.toNat
      = (modulus.l0 * q + 2^64 * (modulus.l1 * q) + 2^254 * q)
        + 2^64 * (modulus.l0 * q_1 + 2^64 * (modulus.l1 * q_1) + 2^254 * q_1)
        + 2^128 * (modulus.l0 * q_2 + 2^64 * (modulus.l1 * q_2) + 2^254 * q_2)
        + 2^192 * (modulus.l0 * q_3 + 2^64 * (modulus.l1 * q_3) + 2^254 * q_3) := by
    simp only [Limbs.toNat, hshape.1, hshape.2]; ring
  -- The four steps compose to `2^256 * R = T_lo + Q * p` on the square's low half.
  have hR : 2^256 * (z0_4 + 2^64 * z1_10 + 2^128 * z2_11 + 2^192 * z3_12)
      = (z0 + 2^64 * z1_2 + 2^128 * z2_3 + 2^192 * z3_4)
        + (q + 2^64 * q_1 + 2^128 * q_2 + 2^192 * q_3) * modulus.toNat := by
    clear * - I_0 I_1 I_2 I_3 hQp e_p0 e_p1
    subst e_p0 e_p1
    omega
  -- END reduction
  -- a0_2: adds a0,z0,z4
  extract_lets -merge +onlyGivenNames s_42 a0_2 c_46 at hr
  have e_a0_2 : a0_2 = (z0_4 + z4_4 + 0) % 2^64 := rfl
  have e_c_46 : c_46 = (z0_4 + z4_4 + 0) / 2^64 := rfl
  clear_value s_42 a0_2 c_46
  have l_a0_2 : a0_2 + 2^64 * c_46 = z0_4 + z4_4 + 0 := by
    rw [e_a0_2, e_c_46]; exact Nat.mod_add_div _ _
  have b_a0_2 : a0_2 < 2^64 := by rw [e_a0_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_46 : c_46 ≤ 1 := by
    rw [e_c_46]; exact addc_carry_le_one z0_4 z4_4 0 b_z0_4 b_z4_4 (by decide)
  clear e_a0_2 e_c_46
  -- a1_2: adcs a1,z1,z5
  extract_lets -merge +onlyGivenNames s_43 a1_2 c_47 at hr
  have e_a1_2 : a1_2 = (z1_10 + z5_3 + c_46) % 2^64 := rfl
  have e_c_47 : c_47 = (z1_10 + z5_3 + c_46) / 2^64 := rfl
  clear_value s_43 a1_2 c_47
  have l_a1_2 : a1_2 + 2^64 * c_47 = z1_10 + z5_3 + c_46 := by
    rw [e_a1_2, e_c_47]; exact Nat.mod_add_div _ _
  have b_a1_2 : a1_2 < 2^64 := by rw [e_a1_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_47 : c_47 ≤ 1 := by
    rw [e_c_47]; exact addc_carry_le_one z1_10 z5_3 c_46 b_z1_10 b_z5_3 b_c_46
  clear e_a1_2 e_c_47
  -- a2_2: adcs a2,z2,z6
  extract_lets -merge +onlyGivenNames s_44 a2_2 c_48 at hr
  have e_a2_2 : a2_2 = (z2_11 + z6_3 + c_47) % 2^64 := rfl
  have e_c_48 : c_48 = (z2_11 + z6_3 + c_47) / 2^64 := rfl
  clear_value s_44 a2_2 c_48
  have l_a2_2 : a2_2 + 2^64 * c_48 = z2_11 + z6_3 + c_47 := by
    rw [e_a2_2, e_c_48]; exact Nat.mod_add_div _ _
  have b_a2_2 : a2_2 < 2^64 := by rw [e_a2_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_48 : c_48 ≤ 1 := by
    rw [e_c_48]; exact addc_carry_le_one z2_11 z6_3 c_47 b_z2_11 b_z6_3 b_c_47
  clear e_a2_2 e_c_48
  -- a3_2: adc a3,z3,z7
  extract_lets -merge +onlyGivenNames a3_2 at hr
  have e_a3_2 : a3_2 = (z3_12 + z7_1 + c_48) % 2^64 := rfl
  clear_value a3_2
  have b_a3_2 : a3_2 < 2^64 := by rw [e_a3_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_a3_2, b_k_a3_2, l_a3_2⟩ :
      ∃ k, k ≤ 1 ∧ a3_2 + 2^64 * k = z3_12 + z7_1 + c_48 :=
    ⟨(z3_12 + z7_1 + c_48) / 2^64, addc_carry_le_one z3_12 z7_1 c_48 b_z3_12 b_z7_1 b_c_48,
      by rw [e_a3_2]; exact Nat.mod_add_div _ _⟩
  clear e_a3_2
  -- BEGIN candidate
  -- The candidate is the reduced low half plus the high half, with the carry the block drops.
  have hC : a0_2 + 2^64 * a1_2 + 2^128 * a2_2 + 2^192 * a3_2 + 2^256 * k_a3_2
      = (z0_4 + 2^64 * z1_10 + 2^128 * z2_11 + 2^192 * z3_12)
        + (z4_4 + 2^64 * z5_3 + 2^128 * z6_3 + 2^192 * z7_1) := by
    clear * - l_a0_2 l_a1_2 l_a2_2 l_a3_2; omega
  -- `2^256 * candidate = value^2 + Q * p`.
  have hmain : 2^256 * (a0_2 + 2^64 * a1_2 + 2^128 * a2_2 + 2^192 * a3_2 + 2^256 * k_a3_2)
      = value.toNat * value.toNat
        + (q + 2^64 * q_1 + 2^128 * q_2 + 2^192 * q_3) * modulus.toNat := by
    clear * - hC hR hT hk_top; omega
  -- `value < p` and `Q < 2^256` put the candidate below `2 * p`, hence below `2^256`: the carry
  -- the block drops is `0`.
  have hP := Limbs.toNat_lt_of_shape modulus hm hshape
  have hAA : value.toNat * value.toNat < 2^256 * modulus.toNat := by
    have h := Nat.mul_lt_mul'' hlt (Limbs.toNat_lt value hv)
    clear * - h; omega
  have hQPle : (q + 2^64 * q_1 + 2^128 * q_2 + 2^192 * q_3) * modulus.toNat + modulus.toNat
      ≤ 2^256 * modulus.toNat := by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hQ
  have hA2 : a0_2 + 2^64 * a1_2 + 2^128 * a2_2 + 2^192 * a3_2 + 2^256 * k_a3_2
      < 2 * modulus.toNat := by
    clear * - hmain hAA hQPle; omega
  have hk : k_a3_2 = 0 := by clear * - hA2 hP; omega
  -- END candidate
  -- q_4: mov q,#0x4000000000000000
  extract_lets -merge +onlyGivenNames q_4 at hr
  have e_q_4 : q_4 = 4611686018427387904 := rfl
  clear_value q_4
  have b_q_4 : q_4 < 2^64 := by rw [e_q_4]; decide
  -- z0_5: subs z0,a0,p0
  extract_lets -merge +onlyGivenNames s_45 z0_5 c_49 at hr
  have e_z0_5 : z0_5 = (a0_2 + 2^64 - p0 - (1 - 1)) % 2^64 := rfl
  have e_c_49 : c_49 = (a0_2 + 2^64 - p0 - (1 - 1)) / 2^64 := rfl
  clear_value s_45 z0_5 c_49
  have l_z0_5 : z0_5 + 2^64 * c_49 + p0 + 1 = a0_2 + 2^64 + 1 := by
    rw [e_z0_5, e_c_49]; exact subc_lin a0_2 p0 1 b_p0 (by decide)
  have b_z0_5 : z0_5 < 2^64 := by rw [e_z0_5]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_49 : c_49 ≤ 1 := by
    rw [e_c_49]; exact subc_carry_le_one a0_2 p0 1 b_a0_2
  clear e_z0_5 e_c_49
  -- z1_11: sbcs z1,a1,p1
  extract_lets -merge +onlyGivenNames s_46 z1_11 c_50 at hr
  have e_z1_11 : z1_11 = (a1_2 + 2^64 - p1 - (1 - c_49)) % 2^64 := rfl
  have e_c_50 : c_50 = (a1_2 + 2^64 - p1 - (1 - c_49)) / 2^64 := rfl
  clear_value s_46 z1_11 c_50
  have l_z1_11 : z1_11 + 2^64 * c_50 + p1 + 1 = a1_2 + 2^64 + c_49 := by
    rw [e_z1_11, e_c_50]; exact subc_lin a1_2 p1 c_49 b_p1 b_c_49
  have b_z1_11 : z1_11 < 2^64 := by rw [e_z1_11]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_50 : c_50 ≤ 1 := by
    rw [e_c_50]; exact subc_carry_le_one a1_2 p1 c_49 b_a1_2
  clear e_z1_11 e_c_50
  -- z2_12: sbcs z2,a2,xzr
  extract_lets -merge +onlyGivenNames s_47 z2_12 c_51 at hr
  have e_z2_12 : z2_12 = (a2_2 + 2^64 - 0 - (1 - c_50)) % 2^64 := rfl
  have e_c_51 : c_51 = (a2_2 + 2^64 - 0 - (1 - c_50)) / 2^64 := rfl
  clear_value s_47 z2_12 c_51
  have l_z2_12 : z2_12 + 2^64 * c_51 + 0 + 1 = a2_2 + 2^64 + c_50 := by
    rw [e_z2_12, e_c_51]; exact subc_lin a2_2 0 c_50 (by decide) b_c_50
  have b_z2_12 : z2_12 < 2^64 := by rw [e_z2_12]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_51 : c_51 ≤ 1 := by
    rw [e_c_51]; exact subc_carry_le_one a2_2 0 c_50 b_a2_2
  clear e_z2_12 e_c_51
  -- z3_13: sbcs z3,a3,q
  extract_lets -merge +onlyGivenNames s_48 z3_13 c_52 at hr
  have e_z3_13 : z3_13 = (a3_2 + 2^64 - q_4 - (1 - c_51)) % 2^64 := rfl
  have e_c_52 : c_52 = (a3_2 + 2^64 - q_4 - (1 - c_51)) / 2^64 := rfl
  clear_value s_48 z3_13 c_52
  have l_z3_13 : z3_13 + 2^64 * c_52 + q_4 + 1 = a3_2 + 2^64 + c_51 := by
    rw [e_z3_13, e_c_52]; exact subc_lin a3_2 q_4 c_51 b_q_4 b_c_51
  have b_z3_13 : z3_13 < 2^64 := by rw [e_z3_13]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_52 : c_52 ≤ 1 := by
    rw [e_c_52]; exact subc_carry_le_one a3_2 q_4 c_51 b_a3_2
  clear e_z3_13 e_c_52
  -- a0_3: csel a0,a0,z0,lo
  extract_lets -merge +onlyGivenNames a0_3 at hr
  have e_a0_3 : a0_3 = (if c_52 = 0 then a0_2 else z0_5) := rfl
  clear_value a0_3
  have b_a0_3 : a0_3 < 2^64 := by
    rw [e_a0_3]; split <;> first | exact b_a0_2 | exact b_z0_5
  -- a1_3: csel a1,a1,z1,lo
  extract_lets -merge +onlyGivenNames a1_3 at hr
  have e_a1_3 : a1_3 = (if c_52 = 0 then a1_2 else z1_11) := rfl
  clear_value a1_3
  have b_a1_3 : a1_3 < 2^64 := by
    rw [e_a1_3]; split <;> first | exact b_a1_2 | exact b_z1_11
  -- a2_3: csel a2,a2,z2,lo
  extract_lets -merge +onlyGivenNames a2_3 at hr
  have e_a2_3 : a2_3 = (if c_52 = 0 then a2_2 else z2_12) := rfl
  clear_value a2_3
  have b_a2_3 : a2_3 < 2^64 := by
    rw [e_a2_3]; split <;> first | exact b_a2_2 | exact b_z2_12
  -- a3_3: csel a3,a3,z3,lo
  extract_lets -merge +onlyGivenNames a3_3 at hr
  have e_a3_3 : a3_3 = (if c_52 = 0 then a3_2 else z3_13) := rfl
  clear_value a3_3
  have b_a3_3 : a3_3 < 2^64 := by
    rw [e_a3_3]; split <;> first | exact b_a3_2 | exact b_z3_13
  subst hr
  -- BEGIN conclusion
  -- The subtraction chain computes the candidate minus `p`, its final carry `c_52` set (no borrow)
  -- exactly when the candidate is at least `p`; the select returns the candidate or the difference.
  -- `sqrMont_conclude` takes the candidate identity `hmain`, the four subtraction facts, and the
  -- select equations and does the case analysis in its own small context, so nothing here runs a
  -- tactic over the full proof context. `p0`/`p1` are rewritten to the modulus limbs first.
  rw [e_p0] at l_z0_5
  rw [e_p1] at l_z1_11
  exact sqrMont_conclude hmain hA2 hk l_z0_5 l_z1_11 l_z2_12 l_z3_13 e_q_4 hshape b_c_52
    b_z0_5 b_z1_11 b_z2_12 b_z3_13 b_a0_3 b_a1_3 b_a2_3 b_a3_3
    e_a0_3 e_a1_3 e_a2_3 e_a3_3
  -- END conclusion

-- BEGIN sqrMont_spec corollaries
/-- The squaring block applied `count` times to a canonical `value`: the output stays below `p`,
and `2^(256 * (2^count - 1)) * output ≡ value^(2^count) (mod p)`, one factor `2^-256` per
squaring. -/
theorem sqrN_spec (value modulus : Limbs) (inv : Nat) (hv : value.Bounded)
    (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinv_lt : inv < 2^64) (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hlt : value.toNat < modulus.toNat) (count : Nat) :
    ∀ r, r = sqrN value modulus inv count →
      r.Bounded ∧ r.toNat < modulus.toNat ∧
        2^(256 * (2^count - 1)) * r.toNat ≡ value.toNat^(2^count) [MOD modulus.toNat] := by
  induction count with
  | zero =>
    intro r hr
    subst hr
    simp only [pow_zero, Nat.sub_self, mul_zero, one_mul, pow_one]
    exact ⟨hv, hlt, Nat.ModEq.refl _⟩
  | succ n ih =>
    intro r hr
    obtain ⟨hb, hl, hc⟩ := ih _ rfl
    obtain ⟨hb', hl', hc'⟩ := sqrMont_spec (sqrN value modulus inv n) modulus inv hb hm hshape
      hinv_lt hinv hl r (hr.trans rfl)
    refine ⟨hb', hl', ?_⟩
    -- `2^(n+1) - 1 = 2 * (2^n - 1) + 1`: the weight is two copies of the previous weight and one
    -- `2^256`, and the power is the square of the previous power.
    have hpos := Nat.two_pow_pos n
    have e1 : 2^(256 * (2^(n+1) - 1)) = 2^(256 * (2^n - 1)) * 2^(256 * (2^n - 1)) * 2^256 := by
      rw [← pow_add, ← pow_add, pow_succ]
      congr 1
      generalize 2^n = m at hpos ⊢
      omega
    have e2 : value.toNat^(2^(n+1)) = value.toNat^(2^n) * value.toNat^(2^n) := by
      rw [pow_succ, pow_mul, sq]
    rw [e1, e2]
    calc 2^(256 * (2^n - 1)) * 2^(256 * (2^n - 1)) * 2^256 * r.toNat
        = 2^(256 * (2^n - 1)) * 2^(256 * (2^n - 1)) * (2^256 * r.toNat) := by ring
      _ ≡ 2^(256 * (2^n - 1)) * 2^(256 * (2^n - 1))
            * ((sqrN value modulus inv n).toNat * (sqrN value modulus inv n).toNat)
            [MOD modulus.toNat] := Nat.ModEq.mul_left _ hc'
      _ = (2^(256 * (2^n - 1)) * (sqrN value modulus inv n).toNat)
            * (2^(256 * (2^n - 1)) * (sqrN value modulus inv n).toNat) := by ring
      _ ≡ value.toNat^(2^n) * value.toNat^(2^n) [MOD modulus.toNat] := Nat.ModEq.mul hc hc

/-- The crate's `sqr_n_mul`: the squaring block `count` times, then the multiplication block by
any four-limb `rhs`. For a canonical `value` the output is below `p` and
`2^(256 * 2^count) * output ≡ value^(2^count) * rhs (mod p)`. The chain keeps its value
canonical, so the multiplication is under its first contract. -/
theorem sqrNMul_spec (value : Limbs) (count : Nat) (rhs modulus : Limbs) (inv : Nat)
    (hv : value.Bounded) (hrhs : rhs.Bounded) (hm : modulus.Bounded)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hinv_lt : inv < 2^64) (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hlt : value.toNat < modulus.toNat) :
    ∀ r, r = sqrNMul value count rhs modulus inv →
      r.Bounded ∧ r.toNat < modulus.toNat ∧
        2^(256 * 2^count) * r.toNat ≡ value.toNat^(2^count) * rhs.toNat [MOD modulus.toNat] := by
  intro r hr
  obtain ⟨hb, hl, hc⟩ := sqrN_spec value modulus inv hv hm hshape hinv_lt hinv hlt count _ rfl
  obtain ⟨hb', hl', hc'⟩ := mulMont_spec_of_lhs_lt (sqrN value modulus inv count) rhs modulus
    inv hb hrhs hm hshape hinv_lt hinv hl r (hr.trans rfl)
  refine ⟨hb', hl', ?_⟩
  have hpos := Nat.two_pow_pos count
  have e : 2^(256 * 2^count) = 2^(256 * (2^count - 1)) * 2^256 := by
    rw [← pow_add]
    congr 1
    generalize 2^count = m at hpos ⊢
    omega
  rw [e]
  calc 2^(256 * (2^count - 1)) * 2^256 * r.toNat
      = 2^(256 * (2^count - 1)) * (2^256 * r.toNat) := by ring
    _ ≡ 2^(256 * (2^count - 1)) * ((sqrN value modulus inv count).toNat * rhs.toNat)
          [MOD modulus.toNat] := Nat.ModEq.mul_left _ hc'
    _ = (2^(256 * (2^count - 1)) * (sqrN value modulus inv count).toNat) * rhs.toNat := by ring
    _ ≡ value.toNat^(2^count) * rhs.toNat [MOD modulus.toNat] := Nat.ModEq.mul_right _ hc
-- END sqrMont_spec corollaries

end PastaAsm.AArch64
