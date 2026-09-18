/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.Fields
import PastaAsm.Spec
import PastaAsm.X86_64.Compositions
import PastaAsm.X86_64.Spec
import PastaAsm.X86_64.Transcription

/-!
# The crate's x86-64 entry points at its fields

`PastaAsm.X86_64.Spec` proves the blocks and compositions for any modulus of the assumed shape,
under arithmetic conditions on the operands. The theorems here restate them for the crate's
entry points as `src/asm/mod.rs` exposes them: at either of its fields (a `PastaField`, whose facts
discharge the hypotheses on the modulus), and under the condition that the entry point checks
in a debug build.

These theorems are intentionally identical to those in `PastaAsm.AArch64.Entry` (other than
calling the x86-64 assembly transcription), because all architecture-specific assembly is
exposed through the same crate API. This ensures that the architecture-specific proofs apply
to the architecture-agnostic interface.
-/

namespace PastaAsm.X86_64

/-- The crate's `add` at a Pasta field: for canonical operands, as it asserts, the result is the
canonical sum. -/
theorem add_entry_spec (F : PastaField) (lhs rhs : Limbs) (hlhs : lhs.Bounded)
    (hrhs : rhs.Bounded) (hl : isCanonical lhs F.modulus = true)
    (hr : isCanonical rhs F.modulus = true) :
    (addMod lhs rhs F.modulus).Bounded ∧
      (addMod lhs rhs F.modulus).toNat < F.modulus.toNat ∧
      (addMod lhs rhs F.modulus).toNat ≡ lhs.toNat + rhs.toNat [MOD F.modulus.toNat] :=
  addMod_spec_of_lt lhs rhs F.modulus hlhs hrhs F.bounded F.shape
    ((isCanonical_iff lhs F.modulus hlhs F.bounded).1 hl)
    ((isCanonical_iff rhs F.modulus hrhs F.bounded).1 hr) _ rfl

end PastaAsm.X86_64
