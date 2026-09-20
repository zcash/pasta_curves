/-
Copyright Supranational LLC (the routines, transcribed from Semolina v0.1.4).
Copyright (c) 2026 the pasta-asm contributors (the transcription and the proofs).
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.Spec
import PastaAsm.AArch64.Transcription

/-!
# Correctness of the transcribed Pasta subtraction block

See the parent module's documentation for details.
-/

namespace PastaAsm.AArch64

-- BEGIN subMod_spec statement
/-- Modular subtraction by the inline block, for operands with `rhs ≤ lhs + p` (so the add-back of
`p` does not wrap): the result is the difference when `lhs ≥ rhs`, and the difference plus `p`
otherwise, since the subtraction borrows exactly when `lhs < rhs` and the block then adds `p`
back. -/
theorem subMod_spec (lhs rhs modulus : Limbs) (hlhs : lhs.Bounded) (hrhs : rhs.Bounded)
    (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hle : rhs.toNat ≤ lhs.toNat + modulus.toNat) :
    ∀ r, r = subMod lhs rhs modulus →
      r.Bounded ∧
        ((rhs.toNat ≤ lhs.toNat ∧ r.toNat + rhs.toNat = lhs.toNat) ∨
          (lhs.toNat < rhs.toNat ∧ r.toNat + rhs.toNat = lhs.toNat + modulus.toNat)) := by
  intro r hr
-- END subMod_spec statement
  -- generated skeleton for `subMod`: do not edit between the annotations
  unfold subMod at hr
  lift_lets at hr
  -- r0: argument
  extract_lets +onlyGivenNames r0 at hr
  have e_r0 : r0 = lhs.l0 := rfl
  clear_value r0
  have b_r0 : r0 < 2^64 := by rw [e_r0]; exact hlhs.1
  -- r1: argument
  extract_lets +onlyGivenNames r1 at hr
  have e_r1 : r1 = lhs.l1 := rfl
  clear_value r1
  have b_r1 : r1 < 2^64 := by rw [e_r1]; exact hlhs.2.1
  -- r2: argument
  extract_lets +onlyGivenNames r2 at hr
  have e_r2 : r2 = lhs.l2 := rfl
  clear_value r2
  have b_r2 : r2 < 2^64 := by rw [e_r2]; exact hlhs.2.2.1
  -- r3: argument
  extract_lets +onlyGivenNames r3 at hr
  have e_r3 : r3 = lhs.l3 := rfl
  clear_value r3
  have b_r3 : r3 < 2^64 := by rw [e_r3]; exact hlhs.2.2.2
  -- b0: argument
  extract_lets +onlyGivenNames b0 at hr
  have e_b0 : b0 = rhs.l0 := rfl
  clear_value b0
  have b_b0 : b0 < 2^64 := by rw [e_b0]; exact hrhs.1
  -- b1: argument
  extract_lets +onlyGivenNames b1 at hr
  have e_b1 : b1 = rhs.l1 := rfl
  clear_value b1
  have b_b1 : b1 < 2^64 := by rw [e_b1]; exact hrhs.2.1
  -- b2: argument
  extract_lets +onlyGivenNames b2 at hr
  have e_b2 : b2 = rhs.l2 := rfl
  clear_value b2
  have b_b2 : b2 < 2^64 := by rw [e_b2]; exact hrhs.2.2.1
  -- b3: argument
  extract_lets +onlyGivenNames b3 at hr
  have e_b3 : b3 = rhs.l3 := rfl
  clear_value b3
  have b_b3 : b3 < 2^64 := by rw [e_b3]; exact hrhs.2.2.2
  -- p0: argument
  extract_lets +onlyGivenNames p0 at hr
  have e_p0 : p0 = modulus.l0 := rfl
  clear_value p0
  have b_p0 : p0 < 2^64 := by rw [e_p0]; exact hm.1
  -- p1: argument
  extract_lets +onlyGivenNames p1 at hr
  have e_p1 : p1 = modulus.l1 := rfl
  clear_value p1
  have b_p1 : p1 < 2^64 := by rw [e_p1]; exact hm.2.1
  -- p3: argument
  extract_lets +onlyGivenNames p3 at hr
  have e_p3 : p3 = modulus.l3 := rfl
  clear_value p3
  have b_p3 : p3 < 2^64 := by rw [e_p3]; exact hm.2.2.2
  -- r0_1: subs r0,r0,b0
  extract_lets +onlyGivenNames s r0_1 c at hr
  have e_r0_1 : r0_1 = (r0 + 2^64 - b0 - (1 - 1)) % 2^64 := rfl
  have e_c : c = (r0 + 2^64 - b0 - (1 - 1)) / 2^64 := rfl
  clear_value s r0_1 c
  have l_r0_1 : r0_1 + 2^64 * c + b0 + 1 = r0 + 2^64 + 1 := by
    rw [e_r0_1, e_c]; exact subc_lin r0 b0 1 b_b0 (by decide)
  have b_r0_1 : r0_1 < 2^64 := by rw [e_r0_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c : c ≤ 1 := by
    rw [e_c]; exact subc_carry_le_one r0 b0 1 b_r0
  clear e_r0_1 e_c
  -- r1_1: sbcs r1,r1,b1
  extract_lets +onlyGivenNames s_1 r1_1 c_1 at hr
  have e_r1_1 : r1_1 = (r1 + 2^64 - b1 - (1 - c)) % 2^64 := rfl
  have e_c_1 : c_1 = (r1 + 2^64 - b1 - (1 - c)) / 2^64 := rfl
  clear_value s_1 r1_1 c_1
  have l_r1_1 : r1_1 + 2^64 * c_1 + b1 + 1 = r1 + 2^64 + c := by
    rw [e_r1_1, e_c_1]; exact subc_lin r1 b1 c b_b1 b_c
  have b_r1_1 : r1_1 < 2^64 := by rw [e_r1_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_1 : c_1 ≤ 1 := by
    rw [e_c_1]; exact subc_carry_le_one r1 b1 c b_r1
  clear e_r1_1 e_c_1
  -- r2_1: sbcs r2,r2,b2
  extract_lets +onlyGivenNames s_2 r2_1 c_2 at hr
  have e_r2_1 : r2_1 = (r2 + 2^64 - b2 - (1 - c_1)) % 2^64 := rfl
  have e_c_2 : c_2 = (r2 + 2^64 - b2 - (1 - c_1)) / 2^64 := rfl
  clear_value s_2 r2_1 c_2
  have l_r2_1 : r2_1 + 2^64 * c_2 + b2 + 1 = r2 + 2^64 + c_1 := by
    rw [e_r2_1, e_c_2]; exact subc_lin r2 b2 c_1 b_b2 b_c_1
  have b_r2_1 : r2_1 < 2^64 := by rw [e_r2_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_2 : c_2 ≤ 1 := by
    rw [e_c_2]; exact subc_carry_le_one r2 b2 c_1 b_r2
  clear e_r2_1 e_c_2
  -- r3_1: sbcs r3,r3,b3
  extract_lets +onlyGivenNames s_3 r3_1 c_3 at hr
  have e_r3_1 : r3_1 = (r3 + 2^64 - b3 - (1 - c_2)) % 2^64 := rfl
  have e_c_3 : c_3 = (r3 + 2^64 - b3 - (1 - c_2)) / 2^64 := rfl
  clear_value s_3 r3_1 c_3
  have l_r3_1 : r3_1 + 2^64 * c_3 + b3 + 1 = r3 + 2^64 + c_2 := by
    rw [e_r3_1, e_c_3]; exact subc_lin r3 b3 c_2 b_b3 b_c_2
  have b_r3_1 : r3_1 < 2^64 := by rw [e_r3_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_3 : c_3 ≤ 1 := by
    rw [e_c_3]; exact subc_carry_le_one r3 b3 c_2 b_r3
  clear e_r3_1 e_c_3
  -- t0: csel t0,p0,xzr,cc
  extract_lets +onlyGivenNames t0 at hr
  have e_t0 : t0 = (if c_3 = 0 then p0 else 0) := rfl
  clear_value t0
  have b_t0 : t0 < 2^64 := by
    rw [e_t0]; split <;> first | exact b_p0 | exact (by decide)
  -- t1: csel t1,p1,xzr,cc
  extract_lets +onlyGivenNames t1 at hr
  have e_t1 : t1 = (if c_3 = 0 then p1 else 0) := rfl
  clear_value t1
  have b_t1 : t1 < 2^64 := by
    rw [e_t1]; split <;> first | exact b_p1 | exact (by decide)
  -- t3: csel t3,p3,xzr,cc
  extract_lets +onlyGivenNames t3 at hr
  have e_t3 : t3 = (if c_3 = 0 then p3 else 0) := rfl
  clear_value t3
  have b_t3 : t3 < 2^64 := by
    rw [e_t3]; split <;> first | exact b_p3 | exact (by decide)
  -- r0_2: adds r0,r0,t0
  extract_lets +onlyGivenNames s_4 r0_2 c_4 at hr
  have e_r0_2 : r0_2 = (r0_1 + t0 + 0) % 2^64 := rfl
  have e_c_4 : c_4 = (r0_1 + t0 + 0) / 2^64 := rfl
  clear_value s_4 r0_2 c_4
  have l_r0_2 : r0_2 + 2^64 * c_4 = r0_1 + t0 + 0 := by
    rw [e_r0_2, e_c_4]; exact Nat.mod_add_div _ _
  have b_r0_2 : r0_2 < 2^64 := by rw [e_r0_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_4 : c_4 ≤ 1 := by
    rw [e_c_4]; exact addc_carry_le_one r0_1 t0 0 b_r0_1 b_t0 (by decide)
  clear e_r0_2 e_c_4
  -- r1_2: adcs r1,r1,t1
  extract_lets +onlyGivenNames s_5 r1_2 c_5 at hr
  have e_r1_2 : r1_2 = (r1_1 + t1 + c_4) % 2^64 := rfl
  have e_c_5 : c_5 = (r1_1 + t1 + c_4) / 2^64 := rfl
  clear_value s_5 r1_2 c_5
  have l_r1_2 : r1_2 + 2^64 * c_5 = r1_1 + t1 + c_4 := by
    rw [e_r1_2, e_c_5]; exact Nat.mod_add_div _ _
  have b_r1_2 : r1_2 < 2^64 := by rw [e_r1_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_5 : c_5 ≤ 1 := by
    rw [e_c_5]; exact addc_carry_le_one r1_1 t1 c_4 b_r1_1 b_t1 b_c_4
  clear e_r1_2 e_c_5
  -- r2_2: adcs r2,r2,xzr
  extract_lets +onlyGivenNames s_6 r2_2 c_6 at hr
  have e_r2_2 : r2_2 = (r2_1 + 0 + c_5) % 2^64 := rfl
  have e_c_6 : c_6 = (r2_1 + 0 + c_5) / 2^64 := rfl
  clear_value s_6 r2_2 c_6
  have l_r2_2 : r2_2 + 2^64 * c_6 = r2_1 + 0 + c_5 := by
    rw [e_r2_2, e_c_6]; exact Nat.mod_add_div _ _
  have b_r2_2 : r2_2 < 2^64 := by rw [e_r2_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_6 : c_6 ≤ 1 := by
    rw [e_c_6]; exact addc_carry_le_one r2_1 0 c_5 b_r2_1 (by decide) b_c_5
  clear e_r2_2 e_c_6
  -- r3_2: adc r3,r3,t3
  extract_lets +onlyGivenNames r3_2 at hr
  have e_r3_2 : r3_2 = (r3_1 + t3 + c_6) % 2^64 := rfl
  clear_value r3_2
  have b_r3_2 : r3_2 < 2^64 := by rw [e_r3_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_r3_2, b_k_r3_2, l_r3_2⟩ :
      ∃ k, k ≤ 1 ∧ r3_2 + 2^64 * k = r3_1 + t3 + c_6 :=
    ⟨(r3_1 + t3 + c_6) / 2^64, addc_carry_le_one r3_1 t3 c_6 b_r3_1 b_t3 b_c_6,
      by rw [e_r3_2]; exact Nat.mod_add_div _ _⟩
  clear e_r3_2
  subst hr
  -- BEGIN conclusion
  have hL : lhs.toNat = r0 + 2^64 * r1 + 2^128 * r2 + 2^192 * r3 := by
    rw [e_r0, e_r1, e_r2, e_r3]; rfl
  have hR : rhs.toNat = b0 + 2^64 * b1 + 2^128 * b2 + 2^192 * b3 := by
    rw [e_b0, e_b1, e_b2, e_b3]; rfl
  have hP : modulus.toNat = p0 + 2^64 * p1 + 2^192 * p3 := by
    rw [e_p0, e_p1, e_p3]; simp only [Limbs.toNat, hshape.1, mul_zero, add_zero]
  -- The subtraction, wrapped modulo `2^256`: its carry is clear exactly when `lhs < rhs`.
  have hD : r0_1 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 + rhs.toNat + 2^256 * c_3
      = lhs.toNat + 2^256 := by
    rw [hL, hR]; clear * - l_r0_1 l_r1_1 l_r2_1 l_r3_1; omega
  -- The add-back of the selected limbs, with the carry that the block drops.
  have hA : r0_2 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + 2^256 * k_r3_2
      = r0_1 + 2^64 * r1_1 + 2^128 * r2_1 + 2^192 * r3_1 + (t0 + 2^64 * t1 + 2^192 * t3) := by
    clear * - l_r0_2 l_r1_2 l_r2_2 l_r3_2; omega
  refine ⟨⟨b_r0_2, b_r1_2, b_r2_2, b_r3_2⟩, ?_⟩
  show (rhs.toNat ≤ lhs.toNat ∧
      r0_2 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + rhs.toNat = lhs.toNat) ∨
    (lhs.toNat < rhs.toNat ∧
      r0_2 + 2^64 * r1_2 + 2^128 * r2_2 + 2^192 * r3_2 + rhs.toNat = lhs.toNat + modulus.toNat)
  obtain hc | hc : c_3 = 0 ∨ c_3 = 1 := by clear * - b_c_3; omega
  · -- Borrow: `lhs < rhs`, the selected limbs are `p`, and the dropped carry undoes the wrap.
    rw [if_pos hc] at e_t0 e_t1 e_t3
    right
    clear * - hD hA hP hc hle e_t0 e_t1 e_t3 b_k_r3_2 b_r0_2 b_r1_2 b_r2_2 b_r3_2
      b_r0_1 b_r1_1 b_r2_1 b_r3_1
    omega
  · -- No borrow: the difference is exact, and the selected limbs are `0`.
    rw [if_neg (by clear * - hc; omega)] at e_t0 e_t1 e_t3
    left
    clear * - hD hA hc e_t0 e_t1 e_t3 b_k_r3_2 b_r0_1 b_r1_1 b_r2_1 b_r3_1
    omega
  -- END conclusion

-- BEGIN subMod corollaries
/-- Subtraction with a lazily reduced left operand: for `lhs < 2^255` and a canonical `rhs`, the
result is below `2^255` and `result + rhs ≡ lhs (mod p)`. -/
theorem subMod_spec_of_rhs_lt (lhs rhs modulus : Limbs) (hlhs : lhs.Bounded)
    (hrhs : rhs.Bounded) (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hlhs_lt : lhs.toNat < 2^255) (hrhs_lt : rhs.toNat < modulus.toNat) :
    ∀ r, r = subMod lhs rhs modulus →
      r.Bounded ∧ r.toNat < 2^255 ∧ r.toNat + rhs.toNat ≡ lhs.toNat [MOD modulus.toNat] := by
  intro r hr
  have hP := Limbs.toNat_lt_of_shape modulus hm hshape
  obtain ⟨hb, h⟩ := subMod_spec lhs rhs modulus hlhs hrhs hm hshape (by omega) r hr
  refine ⟨hb, ?_⟩
  rcases h with ⟨hge, he⟩ | ⟨hlt, he⟩
  · exact ⟨by omega, modEq_of_add_mul _ _ 0 0 _ (by omega)⟩
  · exact ⟨by omega, modEq_of_add_mul _ _ 0 1 _ (by omega)⟩

/-- Subtraction of canonical operands: the result is canonical, `result < p` and
`result + rhs ≡ lhs (mod p)`. This is the contract that the crate's `sub` asserts. -/
theorem subMod_spec_of_lt (lhs rhs modulus : Limbs) (hlhs : lhs.Bounded) (hrhs : rhs.Bounded)
    (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hlhs_lt : lhs.toNat < modulus.toNat) (hrhs_lt : rhs.toNat < modulus.toNat) :
    ∀ r, r = subMod lhs rhs modulus →
      r.Bounded ∧ r.toNat < modulus.toNat ∧
        r.toNat + rhs.toNat ≡ lhs.toNat [MOD modulus.toNat] := by
  intro r hr
  obtain ⟨hb, h⟩ := subMod_spec lhs rhs modulus hlhs hrhs hm hshape (by omega) r hr
  refine ⟨hb, ?_⟩
  rcases h with ⟨hge, he⟩ | ⟨hlt, he⟩
  · exact ⟨by omega, modEq_of_add_mul _ _ 0 0 _ (by omega)⟩
  · exact ⟨by omega, modEq_of_add_mul _ _ 0 1 _ (by omega)⟩
-- END subMod corollaries

end PastaAsm.AArch64
