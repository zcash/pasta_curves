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
  ⟨a.val[0]!.val, a.val[1]!.val, a.val[2]!.val, a.val[3]!.val⟩

/-- A number below `2^64` is the value of its word. -/
theorem word_val (x : Nat) (hx : x < 2^64) : (word x).val = x := by
  show (BitVec.ofNat 64 x).toNat = x
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hx]

end PastaCurves
