import Aeneas
import PastaCurves.Semantics

/-!
# Values as the translations' words

Aeneas' translations of the crate's Rust (`Portable/` and `Glue/`) take and return Aeneas' 64-bit
words and arrays of them, where the Lean model has natural-number limbs. The definitions here write
the model's values in the translations' types, and read four limbs back from an array.
-/

namespace PastaCurves

open Aeneas Aeneas.Std

/-- A word as Aeneas' `U64`, reduced modulo `2^64`. -/
def word (x : Nat) : Std.U64 := ⟨BitVec.ofNat 64 x⟩

/-- Four limbs as Aeneas' array of four words. -/
def limbsArray (x : Limbs) : Std.Array Std.U64 4#usize :=
  Std.Array.from [word x.l0, word x.l1, word x.l2, word x.l3] rfl

/-- A five-word signed value as Aeneas' array of five words. -/
def signed5Array (x : Signed5) : Std.Array Std.U64 5#usize :=
  Std.Array.from [word x.l0, word x.l1, word x.l2, word x.l3, word x.l4] rfl

/-- Four limbs from Aeneas' array of four words. -/
def limbsOfArray (a : Std.Array Std.U64 4#usize) : Limbs :=
  ⟨a[0].val, a[1].val, a[2].val, a[3].val⟩

/-- A number below `2^64` is the value of its word. -/
theorem word_val (x : Nat) (hx : x < 2^64) : (word x).val = x := by
  show (BitVec.ofNat 64 x).toNat = x
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hx]

/-- The value of a Boolean as a word: `1` for true, `0` for false. -/
@[step_pure core.convert.num.FromU64Bool.from b]
theorem fromU64Bool_val (b : Bool) :
    (core.convert.num.FromU64Bool.from b).val = if b then 1 else 0 := by
  cases b <;> rfl

/-- One limb of a borrow chain, from the post-conditions of its two subtractions and of the
borrow's conversion: the difference word, the subtrahend, and the borrow in make the minuend and
the borrow out, which is zero or one. -/
theorem sbb_step (a b cin : Nat) (word word1 bout : Std.U64) (u1 u2 : Bool) (hb : b < 2^64)
    (hcin : cin ≤ 1)
    (h1 : if a < b then word.val + b = a + U64.size ∧ u1 = true else word.val = a - b ∧ u1 = false)
    (h2 : if word.val < cin then word1.val + cin = word.val + U64.size ∧ u2 = true
      else word1.val = word.val - cin ∧ u2 = false)
    (h3 : bout.val = if (u1 || u2) = true then 1 else 0) :
    word1.val + b + cin = a + 2^64 * bout.val ∧ bout.val ≤ 1 := by
  have hw := word.hBounds
  have hw1 := word1.hBounds
  simp only [U64.size] at h1 h2
  cases u1 <;> cases u2 <;> split_ifs at h1 h2 <;> simp_all <;> scalar_tac

/-- The limbs read from an array of words are bounded. -/
theorem limbsOfArray_bounded (a : Std.Array Std.U64 4#usize) : (limbsOfArray a).Bounded :=
  ⟨a[0].hBounds, a[1].hBounds, a[2].hBounds, a[3].hBounds⟩

/-- Bounded limbs, written as an array of words, read back as themselves. -/
theorem limbsOfArray_limbsArray (x : Limbs) (hx : x.Bounded) : limbsOfArray (limbsArray x) = x := by
  obtain ⟨h0, h1, h2, h3⟩ := hx
  simp [limbsOfArray, limbsArray, word_val _ h0, word_val _ h1, word_val _ h2, word_val _ h3]

end PastaCurves
