import PastaCurves.AArch64.Semantics
import PastaCurves.X86_64.Semantics

/-!
# The two architectures' subtraction flags correspond

x86-64 sets CF when a subtraction borrows; AArch64 sets its carry when a subtraction does not
borrow. Otherwise the instructions that the blocks use agree. This module states the
correspondence for every in-range input. The boundary examples of `PastaCurves.X86_64.Semantics`
and their mirrors in `PastaCurves.AArch64.Semantics` illustrate it on a few inputs:

- `sub`, `sbb` and `subs`, `sbcs`: the same difference, with complementary flags.
- `neg r` and `subs` from zero: the same word, with complementary flags; and `neg r` and
  `subs xzr, r, #1`: the same flag, set exactly when `r` is nonzero.
- `cmovnc d, s` and `csel d, s, d, cs` (equivalently `csel d, d, s, lo`): both take `s` exactly
  when the subtraction did not borrow, so they agree under the complemented flag.

The products need no theorem: x86-64's `mulx` is defined as the pair of the shared `umulh` and
`mulLo`, which model AArch64's `umulh` and `mul`.
-/

namespace PastaCurves

/-- `subs`/`sbcs` and `sub`/`sbb` compute the same difference, and AArch64's carry is the
complement of x86-64's CF when the carry-in is complemented likewise. A carry-in above `1` acts
as `1` on both sides, so the carry-in needs no bound. -/
theorem subc_eq_sbb {a b : Nat} (c : Nat) (ha : a < 2^64) (hb : b < 2^64) :
    AArch64.subc a b c = ((X86_64.sbb a b (1 - c)).1, 1 - (X86_64.sbb a b (1 - c)).2) := by
  have hquot : (a + 2^64 - b - (1 - c)) / 2^64 ≤ 1 := by omega
  simp only [AArch64.subc, X86_64.sbb, regMod, Prod.mk.injEq, true_and]
  omega

/-- `subs` from zero computes the word of `neg`, with the complementary flag. -/
theorem subc_zero_eq_neg (a : Nat) :
    AArch64.subc 0 a 1 = ((X86_64.neg a).1, 1 - (X86_64.neg a).2) := by
  by_cases h : a = 0
  · subst h
    decide
  · simp only [AArch64.subc, X86_64.neg, X86_64.sbb, regMod, Prod.mk.injEq, h, ite_false,
      Nat.zero_add, Nat.sub_self, Nat.sub_zero, true_and]
    exact Nat.div_eq_of_lt (Nat.sub_lt (by decide) (Nat.pos_of_ne_zero h))

/-- `subs xzr, r, #1` sets the carry exactly as `neg r` sets CF: when `r` is nonzero. -/
theorem subc_one_carry_eq_neg {r : Nat} (hr : r < 2^64) :
    (AArch64.subc r 1 1).2 = (X86_64.neg r).2 := by
  by_cases h : r = 0
  · subst h
    decide
  · simp only [AArch64.subc, X86_64.neg, regMod, h, ite_false]
    omega

/-- `csel d, s, d, cs` is `cmovnc d, s` under the complemented flag. -/
theorem cselCs_eq_cmovnc {c : Nat} (hc : c ≤ 1) (d s : Nat) :
    AArch64.cselCs c s d = X86_64.cmovnc (1 - c) d s := by
  unfold AArch64.cselCs X86_64.cmovnc
  split <;> split <;> omega

/-- `csel d, d, s, lo` is `cmovnc d, s` under the complemented flag. -/
theorem cselLo_eq_cmovnc {c : Nat} (hc : c ≤ 1) (d s : Nat) :
    AArch64.cselLo c d s = X86_64.cmovnc (1 - c) d s := by
  unfold AArch64.cselLo X86_64.cmovnc
  split <;> split <;> omega

end PastaCurves
