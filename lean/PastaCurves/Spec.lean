import PastaCurves.Compositions
import Mathlib.Data.Nat.ModEq
import Mathlib.Tactic.Ring

/-!
# Generic arithmetic and limb lemmas

These results are independent of any instruction transcription and are shared by architecture-
specific correctness proofs.
-/

namespace PastaCurves

-- BEGIN cancel_low
/-- The Montgomery cancellation: with `inv * p0 ≡ -1 (mod 2^64)` and `q = inv * t0 mod 2^64`, the
low limb of `t0 + p0 * q` is zero. -/
theorem cancel_low (t0 inv p0 : Nat) (h : (inv * p0 + 1) % 2^64 = 0) :
    (t0 + p0 * (inv * t0 % 2^64) % 2^64) % 2^64 = 0 := by
  have key : (t0 + p0 * (inv * t0)) % 2^64 = 0 := by
    have : t0 + p0 * (inv * t0) = t0 * (inv * p0 + 1) := by ring
    rw [this, Nat.mul_mod, h]; simp
  have h1 : inv * t0 % 2^64 ≡ inv * t0 [MOD 2^64] := Nat.mod_modEq _ _
  have h2 : p0 * (inv * t0 % 2^64) % 2^64 ≡ p0 * (inv * t0) [MOD 2^64] :=
    (Nat.mod_modEq _ _).trans (Nat.ModEq.mul_left p0 h1)
  have h3 : t0 + p0 * (inv * t0 % 2^64) % 2^64 ≡ t0 + p0 * (inv * t0) [MOD 2^64] :=
    Nat.ModEq.add_left t0 h2
  exact Eq.trans h3 key
-- END cancel_low

-- BEGIN skeleton lemmas
/-! The generated skeleton derives its per-instruction facts from these lemmas and from
`Nat.mod_add_div`, `Nat.mod_lt`, and `Nat.div_lt_of_lt_mul`. -/

/-- The carry out of an addition of two limbs and a carry is at most `1`. -/
theorem addc_carry_le_one (a b cin : Nat) (ha : a < 2^64) (hb : b < 2^64) (hc : cin ≤ 1) :
    (a + b + cin) / 2^64 ≤ 1 := by omega

/-- A subtraction with borrow, in the form used by machine instruction semantics: its result and carry
satisfy `result + 2^64 * carry + b + 1 = a + 2^64 + cin`, since `b + (1 - cin) ≤ a + 2^64` keeps
the difference from truncating. -/
theorem subc_lin (a b cin : Nat) (hb : b < 2^64) (hc : cin ≤ 1) :
    (a + 2^64 - b - (1 - cin)) % 2^64 + 2^64 * ((a + 2^64 - b - (1 - cin)) / 2^64) + b + 1
      = a + 2^64 + cin := by omega

/-- The carry (no-borrow flag) of a subtraction is at most `1`. -/
theorem subc_carry_le_one (a b cin : Nat) (ha : a < 2^64) :
    (a + 2^64 - b - (1 - cin)) / 2^64 ≤ 1 := by omega

/-- The carry of a subtraction is set exactly when no borrow occurs. -/
theorem subc_carry_cases (a b cin c : Nat) (hc : c = (a + 2^64 - b - (1 - cin)) / 2^64)
    (ha : a < 2^64) (hb : b < 2^64) (hcin : cin ≤ 1) :
    (c = 1 ∧ b + 1 ≤ a + cin) ∨ (c = 0 ∧ a + cin < b + 1) := by omega

/-- `lsl #62` and `lsr #2` split a limb at its second bit. -/
theorem lsl62_lsr2_split (a : Nat) : a * 2^62 % 2^64 + 2^64 * (a / 2^2) = a * 2^62 := by omega
-- END skeleton lemmas

-- BEGIN modEq_of_add_mul
/-- `a ≡ b (mod n)` from `a + k * n = b + l * n`, the form in which a final conditional
subtraction leaves a result. -/
theorem modEq_of_add_mul (a b k l n : Nat) (h : a + k * n = b + l * n) : a ≡ b [MOD n] := by
  unfold Nat.ModEq
  rw [← Nat.add_mul_mod_self_right a k n, h, Nat.add_mul_mod_self_right]
-- END modEq_of_add_mul

-- BEGIN mulMont_spec corollaries
/-- Below `2^256`, by the limb bounds. -/
theorem Limbs.toNat_lt (x : Limbs) (hx : x.Bounded) : x.toNat < 2^256 := by
  obtain ⟨h0, h1, h2, h3⟩ := hx
  simp only [Limbs.toNat]; omega

/-- With the modulus in the shape the code assumes, `p < 2^255`. -/
theorem Limbs.toNat_lt_of_shape (modulus : Limbs) (hm : modulus.Bounded)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62) : modulus.toNat < 2^255 := by
  obtain ⟨h0, h1, _, _⟩ := hm
  simp only [Limbs.toNat, hshape.1, hshape.2]; omega
-- END mulMont_spec corollaries

/-! ## Bounds of the shared word operations

The generated skeletons of the inversion's block proofs bound each instruction's result below
`2^64` by one of these lemmas. -/

/-- The skeleton's bound for an `add` result, by the reduction modulo `2^64`. -/
theorem addw_lt (a b : Nat) : addw a b < 2^64 := Nat.mod_lt _ (by decide)

/-- The bound for a `sub` result, by the reduction. -/
theorem subw_lt (a b : Nat) : subw a b < 2^64 := Nat.mod_lt _ (by decide)

/-- The bound for a `neg` result, by the reduction. -/
theorem negw_lt (a : Nat) : negw a < 2^64 := Nat.mod_lt _ (by decide)

/-- The bound for an `extr` result, by the reduction. -/
theorem extr_lt (hi lo k : Nat) : extr hi lo k < 2^64 := Nat.mod_lt _ (by decide)

/-- The bound for an `and` result: no bit above the second operand's. -/
theorem andw_lt (a b : Nat) (_ha : a < 2^64) (hb : b < 2^64) : andw a b < 2^64 :=
  Nat.and_lt_two_pow a hb

/-- The bound for an `orr` result: no bit above either operand's. -/
theorem orrw_lt (a b : Nat) (ha : a < 2^64) (hb : b < 2^64) : orrw a b < 2^64 :=
  Nat.or_lt_two_pow ha hb

/-- The bound for an `eor` result: no bit above either operand's. -/
theorem eorw_lt (a b : Nat) (ha : a < 2^64) (hb : b < 2^64) : eorw a b < 2^64 :=
  Nat.xor_lt_two_pow ha hb

/-- `a / 2^k < 2^(64 - k)` for `a < 2^64`, the fact behind the two branches of `asr`. -/
theorem div_two_pow_lt (a k : Nat) (ha : a < 2^64) : a / 2^k < 2^(64 - k) := by
  rcases Nat.lt_or_ge 64 k with hk | hk
  · have h : a < 2^k := lt_of_lt_of_le ha (Nat.pow_le_pow_right (by decide) hk.le)
    rw [Nat.div_eq_of_lt h]
    exact Nat.two_pow_pos _
  · have h : (2 : Nat)^64 = 2^k * 2^(64 - k) := by rw [← pow_add]; congr 1; omega
    exact Nat.div_lt_of_lt_mul (h ▸ ha)

/-- The bound for an `asr` result: both branches of the shift, by `div_two_pow_lt`. -/
theorem asr_lt (a k : Nat) (ha : a < 2^64) : asr a k < 2^64 := by
  have h := div_two_pow_lt a k ha
  have hle : (2 : Nat)^(64 - k) ≤ 2^64 := Nat.pow_le_pow_right (by decide) (by omega)
  unfold asr
  by_cases hs : a < 2^63
  · rw [if_pos hs]; exact lt_of_lt_of_le h hle
  · rw [if_neg hs]
    calc a / 2^k + (2^64 - 2^(64 - k)) < 2^(64 - k) + (2^64 - 2^(64 - k)) :=
          Nat.add_lt_add_right h _
      _ = 2^64 := Nat.add_sub_of_le hle

end PastaCurves
