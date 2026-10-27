module
public import Aeneas
public import PastaCurves.ARMv7EM.Compositions.Types
public import PastaCurves.ARMv7EM.Transcription
public import PastaCurves.Portable.Funs
@[expose] public section
open Aeneas Aeneas.Std Result ControlFlow Error

/-!
# External functions called by the translated ARMv7E-M arithmetic

Written by hand, not generated. The carry-sensitive word primitives are opaque to the main
translation because their ARM implementations use inline assembly. The batch, conversion, and
conditional-subtraction externals execute the generated instruction transcription directly.
-/

open pasta_curves

namespace pasta_curves.asm.armv7em.external

/-- Read the translated words into the instruction model's limbs. -/
def limbsOfArray (a : Array Std.U64 4#usize) : _root_.PastaCurves.Limbs :=
  ⟨a[0].val, a[1].val, a[2].val, a[3].val⟩

/-- Return a model value as a translated word. -/
def word (x : Nat) : Std.U64 := ⟨BitVec.ofNat 64 x⟩

/-- Return model limbs as translated words. -/
def limbsArray (x : _root_.PastaCurves.Limbs) : Array Std.U64 4#usize :=
  Array.make 4#usize [word x.l0, word x.l1, word x.l2, word x.l3]

end pasta_curves.asm.armv7em.external

/-- The modulus shifted by 61 bits. This external model mirrors the translated portable
helper; its equality to that translation must be established before relying on the ARM
amontred composition proof. -/
def inversion.portable.shifted_modulus
    (modulus : Array Std.U64 4#usize) : Result (Array Std.U64 5#usize) := do
  let i ← Array.index_usize modulus 0#usize
  let i1 ← i <<< 61#i32
  let i2 ← i >>> 3#i32
  let i3 ← Array.index_usize modulus 1#usize
  let i4 ← i3 <<< 61#i32
  let i5 ← lift (i2 ||| i4)
  let i6 ← i3 >>> 3#i32
  let i7 ← Array.index_usize modulus 2#usize
  let i8 ← i7 <<< 61#i32
  let i9 ← lift (i6 ||| i8)
  let i10 ← i7 >>> 3#i32
  let i11 ← Array.index_usize modulus 3#usize
  let i12 ← i11 <<< 61#i32
  let i13 ← lift (i10 ||| i12)
  let i14 ← i11 >>> 3#i32
  ok (Array.make 5#usize [i1, i5, i9, i13, i14])

/-- Unsigned wrapping negation, computed as subtraction from zero. -/
@[rust_fun "core::num::{u32}::wrapping_neg"]
def core.num.U32.wrapping_neg (x : Std.U32) : Result Std.U32 :=
  ok (core.num.U32.wrapping_sub 0#u32 x)

/-- Addition with carry, matching the host implementation of `armv7em::adc`. -/
def asm.armv7em.adc
    (a : Std.U32) (b : Std.U32) (carry : Std.U32) :
    Result (Std.U32 × Std.U32) := do
  let i ← lift (core.convert.num.FromU64U32.from a)
  let i1 ← lift (core.convert.num.FromU64U32.from b)
  let i2 ← i + i1
  let i3 ← lift (core.convert.num.FromU64U32.from carry)
  let sum ← i2 + i3
  let i4 ← lift (UScalar.cast .U32 sum)
  let i5 ← sum >>> 32#i32
  let i6 ← lift (UScalar.cast .U32 i5)
  ok (i4, i6)

/-- Addition of two words, matching the host implementation of `armv7em::add_word`. -/
def asm.armv7em.add_word
    (a : Std.U32) (b : Std.U32) : Result (Std.U32 × Std.U32) := do
  let i ← lift (core.convert.num.FromU64U32.from a)
  let i1 ← lift (core.convert.num.FromU64U32.from b)
  let sum ← i + i1
  let i2 ← lift (UScalar.cast .U32 sum)
  let i3 ← sum >>> 32#i32
  let i4 ← lift (UScalar.cast .U32 i3)
  ok (i2, i4)

/-- Subtraction with borrow, matching the ARM primitive's borrow-bit behavior. -/
def asm.armv7em.sbb
    (a : Std.U32) (b : Std.U32) (borrow : Std.U32) :
    Result (Std.U32 × Std.U32) := do
  let borrowBit := if borrow = 0#u32 then 0#u32 else 1#u32
  let i ← lift (core.convert.num.FromU64U32.from b)
  let i1 ← lift (core.convert.num.FromU64U32.from borrowBit)
  let subtrahend ← i + i1
  let i2 ← lift (UScalar.cast .U32 subtrahend)
  let i3 ← lift (core.num.U32.wrapping_sub a i2)
  let i4 ← lift (core.convert.num.FromU64U32.from a)
  let i5 ← lift (core.convert.num.FromU32Bool.from (i4 < subtrahend))
  ok (i3, i5)

/-- Multiply-accumulate with carry, matching the host implementation of `armv7em::mac_word`. -/
def asm.armv7em.mac_word
    (acc : Std.U32) (a : Std.U32) (b : Std.U32) (carry : Std.U32) :
    Result (Std.U32 × Std.U32) := do
  let i ← lift (core.convert.num.FromU64U32.from a)
  let i1 ← lift (core.convert.num.FromU64U32.from b)
  let i2 ← i * i1
  let i3 ← lift (core.convert.num.FromU64U32.from acc)
  let i4 ← i2 + i3
  let i5 ← lift (core.convert.num.FromU64U32.from carry)
  let product ← i4 + i5
  let i6 ← lift (UScalar.cast .U32 product)
  let i7 ← product >>> 32#i32
  let i8 ← lift (UScalar.cast .U32 i7)
  ok (i6, i8)

/-- The eight-word ARM multiply-accumulate block, modeled through the translated 64-bit portable
multiply-accumulate with the same arithmetic result. Instruction-level equivalence remains to be proved from a
mechanical transcription of the assembly block; Aeneas treats the block as opaque. -/
def asm.armv7em.inversion_mac_words
    (r0 r1 b0 b1 c0 c1 r2 r3 : Std.U32) :
    Result (Std.U32 × Std.U32 × Std.U32 × Std.U32) := do
  let r1High ← core.convert.num.FromU64U32.from r1 <<< 32#i32
  let a ← lift (core.convert.num.FromU64U32.from r0 ||| r1High)
  let b1High ← core.convert.num.FromU64U32.from b1 <<< 32#i32
  let b ← lift (core.convert.num.FromU64U32.from b0 ||| b1High)
  let c1High ← core.convert.num.FromU64U32.from c1 <<< 32#i32
  let c ← lift (core.convert.num.FromU64U32.from c0 ||| c1High)
  let r3High ← core.convert.num.FromU64U32.from r3 <<< 32#i32
  let carry ← lift (core.convert.num.FromU64U32.from r2 ||| r3High)
  let (low, high) ← inversion.portable.mac a b c carry
  let lowHigh ← low >>> 32#i32
  let highHigh ← high >>> 32#i32
  ok (UScalar.cast .U32 low, UScalar.cast .U32 lowHigh,
    UScalar.cast .U32 high, UScalar.cast .U32 highHigh)

/-- The ARM batch call, evaluated by the generated instruction transcription. -/
def asm.armv7em.batch
    (k : Std.U32) (twoDelta f g : Std.U64) : Result (Array Std.U64 5#usize) :=
  let result := _root_.PastaCurves.ARMv7EM.batchBlock k.val twoDelta.val f.val g.val
  ok (Array.make 5#usize [asm.armv7em.external.word result.two_delta,
    asm.armv7em.external.word result.u, asm.armv7em.external.word result.v,
    asm.armv7em.external.word result.q, asm.armv7em.external.word result.r])

/-- The dedicated ARM conversion, evaluated by the generated instruction transcription.
The block does not read `inv`; the Rust wrapper checks its required value separately. -/
def asm.armv7em.from_mont
    (value modulus : Array Std.U64 4#usize) (_inv : Std.U64) :
    Result (Array Std.U64 4#usize) :=
  ok (asm.armv7em.external.limbsArray (_root_.PastaCurves.ARMv7EM.fromMontBlock
    (asm.armv7em.external.limbsOfArray value) (asm.armv7em.external.limbsOfArray modulus)))

/-- The ARM conditional subtraction, evaluated by the generated instruction transcription. -/
def asm.armv7em.cond_sub
    (value modulus : Array Std.U64 4#usize) : Result (Array Std.U64 4#usize) :=
  ok (asm.armv7em.external.limbsArray (_root_.PastaCurves.ARMv7EM.condSub
    (asm.armv7em.external.limbsOfArray value) (asm.armv7em.external.limbsOfArray modulus)))
