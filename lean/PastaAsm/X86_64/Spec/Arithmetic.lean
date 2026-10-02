/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.Spec
import PastaAsm.X86_64.Semantics
import Mathlib.Tactic.NormNum

/-!
# Arithmetic lemmas for x86-64 instruction proofs

These lemmas expose the bounded arithmetic facts attached to individual x86-64 instructions.
In particular, x86 subtraction sets CF on borrow, unlike the AArch64 subtraction convention.
-/

namespace PastaAsm.X86_64

/-- The value written by an addition instruction is a 64-bit word. -/
theorem addc_value_lt (a b cin : Nat) : (addc a b cin).1 < 2^64 := by
  simp only [addc]
  exact Nat.mod_lt _ (Nat.two_pow_pos 64)

/-- The carry written by an addition of two words and a carry bit is a bit. -/
theorem addc_carry_le_one (a b cin : Nat) (ha : a < 2^64) (hb : b < 2^64)
    (hcin : cin ≤ 1) : (addc a b cin).2 ≤ 1 := by
  simpa only [addc] using PastaAsm.addc_carry_le_one a b cin ha hb hcin

/-- An addition instruction splits its full sum into its low word and carry. -/
theorem addc_lin (a b cin : Nat) :
    (addc a b cin).1 + 2^64 * (addc a b cin).2 = a + b + cin := by
  simp only [addc]
  exact Nat.mod_add_div _ _

/-- The value written by `sub` or `sbb` is a 64-bit word. -/
theorem sbb_value_lt (a b cin : Nat) : (sbb a b cin).1 < 2^64 := by
  simp only [sbb]
  exact Nat.mod_lt _ (Nat.two_pow_pos 64)

/-- The x86 subtraction borrow-out is a bit. -/
theorem sbb_borrow_le_one (a b cin : Nat) : (sbb a b cin).2 ≤ 1 := by
  simp only [sbb]
  exact Nat.sub_le 1 _

/-- The low result and borrow-out of x86 subtraction reconstruct the integer subtraction.

The bounds ensure that adding `2^64` before the natural-number subtractions prevents
truncation.  Since x86 CF is set on borrow, the modulus is added on the right exactly when
`(sbb a b cin).2 = 1`.
-/
theorem sbb_lin (a b cin : Nat) (ha : a < 2^64) (hb : b < 2^64) (hcin : cin ≤ 1) :
    (sbb a b cin).1 + b + cin = a + 2^64 * (sbb a b cin).2 := by
  simp only [sbb, regMod]
  have hsplit := Nat.mod_add_div (a + 2^64 - b - cin) (2^64)
  have hsub : a + 2^64 - b - cin + b + cin = a + 2^64 := by omega
  have hquot : (a + 2^64 - b - cin) / 2^64 ≤ 1 := by omega
  omega

/-- x86 subtraction sets CF exactly when `a - b - cin` borrows. -/
theorem sbb_borrow_iff (a b cin : Nat) (ha : a < 2^64) (hb : b < 2^64)
    (hcin : cin ≤ 1) : (sbb a b cin).2 = 1 ↔ a < b + cin := by
  have hlin := sbb_lin a b cin ha hb hcin
  have hresult := sbb_value_lt a b cin
  have hborrow := sbb_borrow_le_one a b cin
  omega

/-- Montgomery cancellation produces either zero or exactly one register modulus.

The conditional is also the CF produced by `neg t0`.  The explicit word bounds match the
contracts available at an instruction site; the low-product operation itself supplies the bound
on its result.
-/
theorem mulLo_cancel_eq_neg_carry (t0 inv p0 : Nat) (ht0 : t0 < 2^64)
    (_hinv : inv < 2^64) (_hp0 : p0 < 2^64)
    (hinvP : (inv * p0 + 1) % 2^64 = 0) :
    t0 + mulLo (mulLo inv t0) p0 = 2^64 * (if t0 = 0 then 0 else 1) := by
  have hcancel := PastaAsm.cancel_low t0 inv p0 hinvP
  have hlo : mulLo (mulLo inv t0) p0 < 2^64 := by
    exact Nat.mod_lt _ (Nat.two_pow_pos 64)
  have hmod : (t0 + mulLo (mulLo inv t0) p0) % 2^64 = 0 := by
    change (t0 + (inv * t0 % 2^64) * p0 % 2^64) % 2^64 = 0
    rw [Nat.mul_comm (inv * t0 % 2^64) p0]
    exact hcancel
  split_ifs with ht
  · subst t0
    simp [mulLo]
  · have hpos : 0 < t0 + mulLo (mulLo inv t0) p0 := by omega
    have hlt : t0 + mulLo (mulLo inv t0) p0 < 2 * 2^64 := by omega
    omega

/-- Instruction-site form of `mulLo_cancel_eq_neg_carry`, with the generated value `q` named. -/
theorem neg_carry_cancel (t0 inv p0 q : Nat) (ht0 : t0 < 2^64) (hinv : inv < 2^64)
    (hp0 : p0 < 2^64) (hinvP : (inv * p0 + 1) % 2^64 = 0)
    (hq : q = mulLo inv t0) :
    t0 + mulLo q p0 = 2^64 * (if t0 = 0 then 0 else 1) := by
  rw [hq]
  exact mulLo_cancel_eq_neg_carry t0 inv p0 ht0 hinv hp0 hinvP

end PastaAsm.X86_64
