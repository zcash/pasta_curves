import PastaCurves.Portable.Funs
import PastaCurves.Semantics

/-!
# Values as the translation's words

Aeneas' translation of the portable blocks (`Funs.lean`) takes and returns Aeneas' 64-bit words
and arrays of them, where the Lean model has natural-number limbs. The definitions here write the
model's values in the translation's types.
-/

namespace PastaCurves.Portable

open Aeneas Aeneas.Std

/-- A word as Aeneas' `U64`, reduced modulo `2^64`. -/
def word (x : Nat) : Std.U64 := ⟨BitVec.ofNat 64 x⟩

/-- Four limbs as Aeneas' array of four words. -/
def limbsArray (x : Limbs) : Std.Array Std.U64 4#usize :=
  Std.Array.from [word x.l0, word x.l1, word x.l2, word x.l3] rfl

/-- A five-word signed value as Aeneas' array of five words. -/
def signed5Array (x : Signed5) : Std.Array Std.U64 5#usize :=
  Std.Array.from [word x.l0, word x.l1, word x.l2, word x.l3, word x.l4] rfl

/-- The translated record of the portable blocks. -/
abbrev blocks := pasta_curves.inversion.portable.Backend.Insts.Pasta_curvesInversionInvertBlocks

end PastaCurves.Portable
