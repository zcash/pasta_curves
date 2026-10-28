import PastaCurves.ARMv7EM.Spec.Arithmetic

/-!
# The ARMv7E-M Rust around the word primitives

These definitions mirror the fixed-bound Rust loops and reductions in `src/asm/armv7em.rs`.
Operands are split into eight little-endian 32-bit words. Multiplication scans the right operand
in a ten-word rolling accumulator and specializes reduction to the sparse Pasta modulus shape. It
calls the mechanically transcribed, proved `adc`, `addWord`, `sbb`, and `macWord` primitives for
every word operation.
-/

namespace PastaCurves.ARMv7EM

/-- The number of 32-bit words in a field element. -/
abbrev words : Nat := 8

/-- The number of 32-bit words in the multiplication's rolling accumulator. -/
abbrev accumulatorWords : Nat := words + 2

/-- Split four little-endian 64-bit limbs into eight little-endian 32-bit words. -/
def wordsOfLimbsArray (value : Limbs) : Array Nat :=
  #[value.l0 % wordMod, value.l0 / wordMod % wordMod,
    value.l1 % wordMod, value.l1 / wordMod % wordMod,
    value.l2 % wordMod, value.l2 / wordMod % wordMod,
    value.l3 % wordMod, value.l3 / wordMod % wordMod]

/-- Reassemble eight little-endian 32-bit words into four little-endian 64-bit limbs. -/
def limbsOfWordsArray (value : Array Nat) : Limbs :=
  ⟨value.getD 0 0 + wordMod * value.getD 1 0,
   value.getD 2 0 + wordMod * value.getD 3 0,
   value.getD 4 0 + wordMod * value.getD 5 0,
   value.getD 6 0 + wordMod * value.getD 7 0⟩

/-- Subtract the modulus and select it exactly when the candidate is not reduced or its high word
is nonzero, using the same 32-bit mask calculation as the Rust routine. -/
def selectReduced (candidate modulus : Array Nat) (highNonzero : Nat) : Array Nat :=
  let sub := (List.range words).foldl (fun st i =>
    let res := sbb (st.1.getD i 0) (modulus.getD i 0) st.2
    (st.1.setIfInBounds i res.1, res.2)) (candidate, 0)
  let subtract := highNonzero ||| (sub.2 ^^^ 1)
  let mask := (wordMod - subtract) % wordMod
  let keepMask := (wordMod - 1) ^^^ mask
  (List.range words).foldl (fun result i =>
    result.setIfInBounds i
      (((sub.1.getD i 0) &&& mask) ||| ((candidate.getD i 0) &&& keepMask)))
    (Array.replicate words 0)

/-- Eight little-endian 32-bit words. This explicit form makes the fixed Rust loops inexpensive for
the kernel to evaluate while retaining each individual word operation. -/
structure FieldWords where
  w0 : Nat
  w1 : Nat
  w2 : Nat
  w3 : Nat
  w4 : Nat
  w5 : Nat
  w6 : Nat
  w7 : Nat

/-- Split four little-endian 64-bit limbs into the explicit word form used by multiplication. -/
def fieldWordsOfLimbs (value : Limbs) : FieldWords :=
  ⟨value.l0 % wordMod, value.l0 / wordMod % wordMod,
   value.l1 % wordMod, value.l1 / wordMod % wordMod,
   value.l2 % wordMod, value.l2 / wordMod % wordMod,
   value.l3 % wordMod, value.l3 / wordMod % wordMod⟩

/-- Convert explicit field words to the unchanged array model used by final reduction. -/
def FieldWords.toArray (value : FieldWords) : Array Nat :=
  #[value.w0, value.w1, value.w2, value.w3, value.w4, value.w5, value.w6, value.w7]

/-- The ten words of the rolling multiplication accumulator. -/
structure MulAcc where
  w0 : Nat
  w1 : Nat
  w2 : Nat
  w3 : Nat
  w4 : Nat
  w5 : Nat
  w6 : Nat
  w7 : Nat
  w8 : Nat
  w9 : Nat

/-- `mac_product`: add one eight-word product to the ten-word rolling accumulator, retaining carry
in its two high words exactly as the Rust helper does. -/
def macProduct (acc : MulAcc) (multiplicand : FieldWords) (multiplier : Nat) : MulAcc :=
  let r0 := macWord acc.w0 multiplicand.w0 multiplier 0
  let r1 := macWord acc.w1 multiplicand.w1 multiplier r0.2
  let r2 := macWord acc.w2 multiplicand.w2 multiplier r1.2
  let r3 := macWord acc.w3 multiplicand.w3 multiplier r2.2
  let r4 := macWord acc.w4 multiplicand.w4 multiplier r3.2
  let r5 := macWord acc.w5 multiplicand.w5 multiplier r4.2
  let r6 := macWord acc.w6 multiplicand.w6 multiplier r5.2
  let r7 := macWord acc.w7 multiplicand.w7 multiplier r6.2
  let r8 := addWord acc.w8 r7.2
  let r9 := addWord acc.w9 r8.2
  ⟨r0.1, r1.1, r2.1, r3.1, r4.1, r5.1, r6.1, r7.1, r8.1, r9.1⟩

/-- `mac_sparse_modulus`: add `q * modulus` using the Pasta modulus's 32-bit word shape
`[1, p1, p2, p3, 0, 0, 0, 2^30]`. -/
def macSparseModulus (acc : MulAcc) (modulus : FieldWords) (q : Nat) : MulAcc :=
  let r0 := adc acc.w0 q 0
  let r1 := macWord acc.w1 modulus.w1 q r0.2
  let r2 := macWord acc.w2 modulus.w2 q r1.2
  let r3 := macWord acc.w3 modulus.w3 q r2.2
  let r4 := addWord acc.w4 r3.2
  let r5 := addWord acc.w5 r4.2
  let r6 := addWord acc.w6 r5.2
  let r7 := macWord acc.w7 modulus.w7 q r6.2
  let r8 := addWord acc.w8 r7.2
  let r9 := addWord acc.w9 r8.2
  ⟨r0.1, r1.1, r2.1, r3.1, r4.1, r5.1, r6.1, r7.1, r8.1, r9.1⟩

/-- `shift_round`: drop the cancelled low word and clear the top of the rolling accumulator. -/
def shiftRound (acc : MulAcc) : MulAcc :=
  ⟨acc.w1, acc.w2, acc.w3, acc.w4, acc.w5, acc.w6, acc.w7, acc.w8, acc.w9, 0⟩

/-- One round of the rolling CIOS schedule on one word of the right operand. -/
def mulRound (lhs modulus : FieldWords) (inv b : Nat) (acc : MulAcc) : MulAcc :=
  let acc := macProduct acc lhs b
  let q := (acc.w0 * inv) % wordMod
  shiftRound (macSparseModulus acc modulus q)

/-- Eight-word rolling CIOS Montgomery multiplication specialized to the sparse Pasta moduli,
followed by the Rust routine's high-word-aware conditional subtraction. -/
def mulMont (lhs rhs modulus : Limbs) (inv : Nat) : Limbs :=
  let lhs := fieldWordsOfLimbs lhs
  let rhs := fieldWordsOfLimbs rhs
  let modulus := fieldWordsOfLimbs modulus
  let inv := inv % wordMod
  let acc : MulAcc := ⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩
  let acc := mulRound lhs modulus inv rhs.w0 acc
  let acc := mulRound lhs modulus inv rhs.w1 acc
  let acc := mulRound lhs modulus inv rhs.w2 acc
  let acc := mulRound lhs modulus inv rhs.w3 acc
  let acc := mulRound lhs modulus inv rhs.w4 acc
  let acc := mulRound lhs modulus inv rhs.w5 acc
  let acc := mulRound lhs modulus inv rhs.w6 acc
  let acc := mulRound lhs modulus inv rhs.w7 acc
  let candidate := FieldWords.toArray ⟨acc.w0, acc.w1, acc.w2, acc.w3, acc.w4, acc.w5, acc.w6, acc.w7⟩
  let high := acc.w8
  let highNeg := (wordMod - high) % wordMod
  let highNonzero := (high ||| highNeg) / 2^31
  limbsOfWordsArray (selectReduced candidate modulus.toArray highNonzero)

/-- `square`: Montgomery multiplication with the same input twice. -/
def sqrMont (value modulus : Limbs) (inv : Nat) : Limbs :=
  mulMont value value modulus inv

end PastaCurves.ARMv7EM
