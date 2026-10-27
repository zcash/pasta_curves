/-
Copyright (c) 2026 the pasta_curves contributors.
-/
module
public import PastaCurves.Semantics
@[expose] public section

/-!
# ARMv7E-M instruction semantics

The subset of ARMv7E-M word operations used by the crate's inline assembly. Registers hold natural
numbers, with the low word and carry behavior modelled explicitly at 32 bits. These definitions are
kept separate from the generated Rust composition and assembly transcription.
-/

namespace PastaCurves.ARMv7EM

/-- The modulus of an ARMv7E-M word. -/
abbrev wordMod : Nat := 2^32

/-- The low word and carry-out of a 32-bit addition. -/
def addc32 (a b c : Nat) : Nat × Nat :=
  ((a + b + c) % wordMod, (a + b + c) / wordMod)

/-- The low word and ARM carry-out of a 32-bit subtraction. A carry of `1` means no borrow. -/
def subc32 (a b c : Nat) : Nat × Nat :=
  ((a + wordMod - b - (1 - c)) % wordMod, (a + wordMod - b - (1 - c)) / wordMod)

/-- The low and high words produced by `umaal`. -/
def umaal32 (low high a b : Nat) : Nat × Nat :=
  ((a * b + low + high) % wordMod, (a * b + low + high) / wordMod)

/-- The 32-bit result of `eor`. -/
def eor32 (a b : Nat) : Nat := (a ^^^ b) % wordMod

/-- The 32-bit result of `orr`. -/
def orr32 (a b : Nat) : Nat := (a ||| b) % wordMod

/-- `lsl` in a 32-bit ARM register. -/
def lsl32 (a k : Nat) : Nat := a * 2^k % wordMod

/-- `lsr` in a 32-bit ARM register. -/
def lsr32 (a k : Nat) : Nat := a / 2^k

/-- `asr` in a 32-bit ARM register. -/
def asr32 (a k : Nat) : Nat :=
  if a < 2^31 then a / 2^k else a / 2^k + (wordMod - 2^(32 - k))

/-- Two 32-bit words joined into one 64-bit word. -/
def join32 (low high : Nat) : Nat := low + wordMod * high

/-- The 64-bit sign extension of a 32-bit word. -/
def signExtend32 (a : Nat) : Nat := if a < 2^31 then a else a + (2^64 - wordMod)

end PastaCurves.ARMv7EM
