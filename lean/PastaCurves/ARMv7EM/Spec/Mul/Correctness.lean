import PastaCurves.ARMv7EM.Spec.Mul.Invariant
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination

/-!
# Pure arithmetic correctness infrastructure for ARMv7E-M multiplication

This file reasons about the ten-word rolling accumulator used by `ARMv7EM.mulMont`.  In
particular, the equations below account for the two high accumulator words and prove that the
carry discarded by each helper is zero under the bounds established here.
-/

namespace PastaCurves.ARMv7EM

/-- The little-endian natural-number value of explicit field words. -/
def FieldWords.value (x : FieldWords) : Nat :=
  ListArithmetic.wordsValue [x.w0, x.w1, x.w2, x.w3, x.w4, x.w5, x.w6, x.w7]

/-- Every component of explicit field words is a 32-bit word. -/
def FieldWords.Bounded (x : FieldWords) : Prop :=
  x.w0 < wordMod ∧ x.w1 < wordMod ∧ x.w2 < wordMod ∧ x.w3 < wordMod ∧
    x.w4 < wordMod ∧ x.w5 < wordMod ∧ x.w6 < wordMod ∧ x.w7 < wordMod

/-- The little-endian natural-number value of the complete ten-word rolling accumulator. -/
def MulAcc.value (x : MulAcc) : Nat :=
  ListArithmetic.wordsValue
    [x.w0, x.w1, x.w2, x.w3, x.w4, x.w5, x.w6, x.w7, x.w8, x.w9]

/-- Every component of the rolling accumulator is a 32-bit word. -/
def MulAcc.Bounded (x : MulAcc) : Prop :=
  x.w0 < wordMod ∧ x.w1 < wordMod ∧ x.w2 < wordMod ∧ x.w3 < wordMod ∧
    x.w4 < wordMod ∧ x.w5 < wordMod ∧ x.w6 < wordMod ∧ x.w7 < wordMod ∧
    x.w8 < wordMod ∧ x.w9 < wordMod

/-- Splitting bounded limbs into explicit words preserves their value. -/
theorem fieldWordsOfLimbs_value (x : Limbs) (hx : x.Bounded) :
    (fieldWordsOfLimbs x).value = x.toNat := by
  obtain ⟨h0, h1, h2, h3⟩ := hx
  have e0 := Nat.mod_add_div x.l0 wordMod
  have e1 := Nat.mod_add_div x.l1 wordMod
  have e2 := Nat.mod_add_div x.l2 wordMod
  have e3 := Nat.mod_add_div x.l3 wordMod
  simp only [FieldWords.value, fieldWordsOfLimbs, ListArithmetic.wordsValue, Limbs.toNat]
  norm_num [wordMod] at h0 h1 h2 h3 e0 e1 e2 e3 ⊢
  omega

/-- Splitting bounded limbs produces eight bounded words. -/
theorem fieldWordsOfLimbs_bounded (x : Limbs) (hx : x.Bounded) :
    (fieldWordsOfLimbs x).Bounded := by
  obtain ⟨h0, h1, h2, h3⟩ := hx
  simp only [FieldWords.Bounded, fieldWordsOfLimbs]
  norm_num [wordMod] at h0 h1 h2 h3 ⊢
  omega

/-- The sparse word shape used by `macSparseModulus`, derived from the actual limb shape and
its separately asserted low word. -/
theorem fieldWordsOfLimbs_sparse_shape (modulus : Limbs)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62) (hlow : modulus.l0 % wordMod = 1) :
    let p := fieldWordsOfLimbs modulus
    p.w0 = 1 ∧ p.w4 = 0 ∧ p.w5 = 0 ∧ p.w6 = 0 ∧ p.w7 = 2^30 := by
  simp [fieldWordsOfLimbs, hshape.1, hshape.2, hlow, wordMod]

/-- The 64-bit Montgomery inverse relation implies its 32-bit low-word counterpart. -/
theorem inverse_relation_word (inv p0 : Nat)
    (hinv : (inv * p0 + 1) % 2^64 = 0) :
    ((inv % wordMod) * (p0 % wordMod) + 1) % wordMod = 0 := by
  have hdiv : 2^32 ∣ 2^64 := by norm_num
  have h := Nat.mod_mod_of_dvd (inv * p0 + 1) hdiv
  rw [hinv, Nat.zero_mod] at h
  simpa [wordMod, Nat.add_mod, Nat.mul_mod] using h.symm

/-- A bounded explicit field element is below the eight-word radix. -/
theorem FieldWords.value_lt (x : FieldWords) (hx : x.Bounded) :
    x.value < wordMod^8 := by
  apply ListArithmetic.wordsValue_lt
  simpa [FieldWords.value, FieldWords.Bounded, ListArithmetic.WordsBounded] using hx

/-- A bounded rolling accumulator is below the ten-word radix. -/
theorem MulAcc.value_lt (x : MulAcc) (hx : x.Bounded) :
    x.value < wordMod^10 := by
  apply ListArithmetic.wordsValue_lt
  simpa [MulAcc.value, MulAcc.Bounded, ListArithmetic.WordsBounded] using hx

/-- Clearing the tenth word strengthens the accumulator bound to nine words. -/
theorem MulAcc.value_lt_nine (x : MulAcc) (hx : x.Bounded) (h9 : x.w9 = 0) :
    x.value < wordMod^9 := by
  have hwords : ListArithmetic.WordsBounded
      [x.w0, x.w1, x.w2, x.w3, x.w4, x.w5, x.w6, x.w7, x.w8] := by
    simpa [ListArithmetic.WordsBounded] using
      ⟨hx.1, hx.2.1, hx.2.2.1, hx.2.2.2.1, hx.2.2.2.2.1,
        hx.2.2.2.2.2.1, hx.2.2.2.2.2.2.1, hx.2.2.2.2.2.2.2.1,
        hx.2.2.2.2.2.2.2.2.1⟩
  have h := ListArithmetic.wordsValue_lt
    [x.w0, x.w1, x.w2, x.w3, x.w4, x.w5, x.w6, x.w7, x.w8] hwords
  simpa [MulAcc.value, ListArithmetic.wordsValue, h9] using h

set_option maxHeartbeats 1000000 in
/-- `macProduct` has its exact ten-word value when the input's top word is zero.  The proof also
shows that the helper's discarded carry is zero. -/
theorem macProduct_spec (acc : MulAcc) (lhs : FieldWords) (b : Nat)
    (hacc : acc.Bounded) (h9 : acc.w9 = 0) (hlhs : lhs.Bounded) (hb : b < wordMod) :
    let out := macProduct acc lhs b
    out.Bounded ∧ out.value = acc.value + lhs.value * b := by
  let r0 := macWord acc.w0 lhs.w0 b 0
  let r1 := macWord acc.w1 lhs.w1 b r0.2
  let r2 := macWord acc.w2 lhs.w2 b r1.2
  let r3 := macWord acc.w3 lhs.w3 b r2.2
  let r4 := macWord acc.w4 lhs.w4 b r3.2
  let r5 := macWord acc.w5 lhs.w5 b r4.2
  let r6 := macWord acc.w6 lhs.w6 b r5.2
  let r7 := macWord acc.w7 lhs.w7 b r6.2
  let r8 := addWord acc.w8 r7.2
  let r9 := addWord acc.w9 r8.2
  obtain ⟨e0, b00, b01⟩ := macWord_spec acc.w0 lhs.w0 b 0 hacc.1 hlhs.1 hb
    (by norm_num [wordMod]) r0 rfl
  obtain ⟨e1, b10, b11⟩ := macWord_spec acc.w1 lhs.w1 b r0.2 hacc.2.1 hlhs.2.1 hb b01 r1 rfl
  obtain ⟨e2, b20, b21⟩ := macWord_spec acc.w2 lhs.w2 b r1.2 hacc.2.2.1 hlhs.2.2.1 hb b11 r2 rfl
  obtain ⟨e3, b30, b31⟩ := macWord_spec acc.w3 lhs.w3 b r2.2 hacc.2.2.2.1 hlhs.2.2.2.1 hb b21 r3 rfl
  obtain ⟨e4, b40, b41⟩ := macWord_spec acc.w4 lhs.w4 b r3.2 hacc.2.2.2.2.1 hlhs.2.2.2.2.1 hb b31 r4 rfl
  obtain ⟨e5, b50, b51⟩ := macWord_spec acc.w5 lhs.w5 b r4.2 hacc.2.2.2.2.2.1 hlhs.2.2.2.2.2.1 hb b41 r5 rfl
  obtain ⟨e6, b60, b61⟩ := macWord_spec acc.w6 lhs.w6 b r5.2 hacc.2.2.2.2.2.2.1 hlhs.2.2.2.2.2.2.1 hb b51 r6 rfl
  obtain ⟨e7, b70, b71⟩ := macWord_spec acc.w7 lhs.w7 b r6.2 hacc.2.2.2.2.2.2.2.1 hlhs.2.2.2.2.2.2.2 hb b61 r7 rfl
  obtain ⟨e8, b80, b81⟩ := addWord_spec acc.w8 r7.2 hacc.2.2.2.2.2.2.2.2.1 b71 r8 rfl
  obtain ⟨e9, b90, b91⟩ := addWord_spec acc.w9 r8.2 hacc.2.2.2.2.2.2.2.2.2
    (lt_of_le_of_lt b81 (by norm_num [wordMod])) r9 rfl
  have ha := acc.value_lt_nine hacc h9
  have hl := lhs.value_lt hlhs
  have hp : lhs.value * b < wordMod^9 := by
    have hp' := Nat.mul_le_mul (Nat.le_sub_one_of_lt hl) (Nat.le_sub_one_of_lt hb)
    calc
      lhs.value * b ≤ (wordMod^8 - 1) * (wordMod - 1) := hp'
      _ < wordMod^9 := by norm_num [wordMod]
  have hsum : acc.value + lhs.value * b < wordMod^10 := by
    calc
      acc.value + lhs.value * b < wordMod^9 + wordMod^9 := Nat.add_lt_add ha hp
      _ < wordMod^10 := by norm_num [wordMod]
  have hall : r0.1 + wordMod * r1.1 + wordMod^2 * r2.1 + wordMod^3 * r3.1 +
      wordMod^4 * r4.1 + wordMod^5 * r5.1 + wordMod^6 * r6.1 +
      wordMod^7 * r7.1 + wordMod^8 * r8.1 + wordMod^9 * r9.1 + wordMod^10 * r9.2 =
      acc.value + lhs.value * b := by
    simp only [MulAcc.value, FieldWords.value, ListArithmetic.wordsValue]
    linear_combination e0 + wordMod * e1 + wordMod^2 * e2 + wordMod^3 * e3 +
      wordMod^4 * e4 + wordMod^5 * e5 + wordMod^6 * e6 + wordMod^7 * e7 +
      wordMod^8 * e8 + wordMod^9 * e9
  have hover : r9.2 = 0 := by
    by_contra hn
    have : wordMod^10 ≤ wordMod^10 * r9.2 := by
      exact Nat.le_mul_of_pos_right _ (Nat.pos_of_ne_zero hn)
    omega
  have hvalue : r0.1 + wordMod * r1.1 + wordMod^2 * r2.1 + wordMod^3 * r3.1 +
      wordMod^4 * r4.1 + wordMod^5 * r5.1 + wordMod^6 * r6.1 +
      wordMod^7 * r7.1 + wordMod^8 * r8.1 + wordMod^9 * r9.1 =
      acc.value + lhs.value * b := by
    simpa only [hover, Nat.mul_zero, Nat.add_zero] using hall
  change (⟨r0.1, r1.1, r2.1, r3.1, r4.1, r5.1, r6.1, r7.1, r8.1, r9.1⟩ : MulAcc).Bounded ∧
    (⟨r0.1, r1.1, r2.1, r3.1, r4.1, r5.1, r6.1, r7.1, r8.1, r9.1⟩ : MulAcc).value =
      acc.value + lhs.value * b
  refine ⟨⟨b00, b10, b20, b30, b40, b50, b60, b70, b80, b90⟩, ?_⟩
  change ListArithmetic.wordsValue
    [r0.1, r1.1, r2.1, r3.1, r4.1, r5.1, r6.1, r7.1, r8.1, r9.1] =
      acc.value + lhs.value * b
  calc
    ListArithmetic.wordsValue
        [r0.1, r1.1, r2.1, r3.1, r4.1, r5.1, r6.1, r7.1, r8.1, r9.1] =
      r0.1 + wordMod * r1.1 + wordMod^2 * r2.1 + wordMod^3 * r3.1 +
        wordMod^4 * r4.1 + wordMod^5 * r5.1 + wordMod^6 * r6.1 +
        wordMod^7 * r7.1 + wordMod^8 * r8.1 + wordMod^9 * r9.1 := by
          simp only [ListArithmetic.wordsValue]
          ring
    _ = acc.value + lhs.value * b := hvalue

/-- The value of words in the sparse shape used by the ARMv7E-M reduction helper. -/
theorem FieldWords.value_of_sparse_shape (modulus : FieldWords)
    (hshape : modulus.w0 = 1 ∧ modulus.w4 = 0 ∧ modulus.w5 = 0 ∧
      modulus.w6 = 0 ∧ modulus.w7 = 2^30) :
    modulus.value = 1 + wordMod * modulus.w1 + wordMod^2 * modulus.w2 +
      wordMod^3 * modulus.w3 + wordMod^7 * 2^30 := by
  rcases hshape with ⟨h0, h4, h5, h6, h7⟩
  simp only [FieldWords.value, ListArithmetic.wordsValue, h0, h4, h5, h6, h7]
  ring

set_option maxHeartbeats 1000000 in
/-- Under the sparse Pasta modulus shape, `macSparseModulus` has its exact ten-word value.
The strict value bound forces the discarded top carry to zero and, in particular, derives the
final top-word inequality checked by the helper. -/
theorem macSparseModulus_spec (acc : MulAcc) (modulus : FieldWords) (q : Nat)
    (hacc : acc.Bounded) (hmodulus : modulus.Bounded) (hq : q < wordMod)
    (hshape : modulus.w0 = 1 ∧ modulus.w4 = 0 ∧ modulus.w5 = 0 ∧
      modulus.w6 = 0 ∧ modulus.w7 = 2^30)
    (hsum : acc.value + modulus.value * q < wordMod^10) :
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
    let out := macSparseModulus acc modulus q
    r9.2 = 0 ∧ acc.w9 + r8.2 < wordMod ∧ out.Bounded ∧
      out.value = acc.value + modulus.value * q ∧ out.value < wordMod^10 := by
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
  have hmodulusValue := modulus.value_of_sparse_shape hshape
  obtain ⟨h0, h4, h5, h6, h7⟩ := hshape
  obtain ⟨e0, b00, b01⟩ := adc_spec acc.w0 q 0 hacc.1 hq
    (by norm_num [wordMod]) (by norm_num) r0 rfl
  obtain ⟨e1, b10, b11⟩ := macWord_spec acc.w1 modulus.w1 q r0.2
    hacc.2.1 hmodulus.2.1 hq (lt_of_le_of_lt b01 (by norm_num [wordMod])) r1 rfl
  obtain ⟨e2, b20, b21⟩ := macWord_spec acc.w2 modulus.w2 q r1.2
    hacc.2.2.1 hmodulus.2.2.1 hq b11 r2 rfl
  obtain ⟨e3, b30, b31⟩ := macWord_spec acc.w3 modulus.w3 q r2.2
    hacc.2.2.2.1 hmodulus.2.2.2.1 hq b21 r3 rfl
  obtain ⟨e4, b40, b41⟩ := addWord_spec acc.w4 r3.2 hacc.2.2.2.2.1 b31 r4 rfl
  obtain ⟨e5, b50, b51⟩ := addWord_spec acc.w5 r4.2 hacc.2.2.2.2.2.1
    (lt_of_le_of_lt b41 (by norm_num [wordMod])) r5 rfl
  obtain ⟨e6, b60, b61⟩ := addWord_spec acc.w6 r5.2 hacc.2.2.2.2.2.2.1
    (lt_of_le_of_lt b51 (by norm_num [wordMod])) r6 rfl
  obtain ⟨e7, b70, b71⟩ := macWord_spec acc.w7 modulus.w7 q r6.2
    hacc.2.2.2.2.2.2.2.1 hmodulus.2.2.2.2.2.2.2 hq
    (lt_of_le_of_lt b61 (by norm_num [wordMod])) r7 rfl
  obtain ⟨e8, b80, b81⟩ := addWord_spec acc.w8 r7.2
    hacc.2.2.2.2.2.2.2.2.1 b71 r8 rfl
  obtain ⟨e9, b90, b91⟩ := addWord_spec acc.w9 r8.2
    hacc.2.2.2.2.2.2.2.2.2 (lt_of_le_of_lt b81 (by norm_num [wordMod])) r9 rfl
  rw [h7] at e7
  have hall : r0.1 + wordMod * r1.1 + wordMod^2 * r2.1 + wordMod^3 * r3.1 +
      wordMod^4 * r4.1 + wordMod^5 * r5.1 + wordMod^6 * r6.1 +
      wordMod^7 * r7.1 + wordMod^8 * r8.1 + wordMod^9 * r9.1 + wordMod^10 * r9.2 =
      acc.value + modulus.value * q := by
    rw [hmodulusValue]
    simp only [MulAcc.value, ListArithmetic.wordsValue]
    linear_combination e0 + wordMod * e1 + wordMod^2 * e2 + wordMod^3 * e3 +
      wordMod^4 * e4 + wordMod^5 * e5 + wordMod^6 * e6 + wordMod^7 * e7 +
      wordMod^8 * e8 + wordMod^9 * e9
  have hover : r9.2 = 0 := by
    by_contra hn
    have hcarry : wordMod^10 ≤ wordMod^10 * r9.2 := by
      exact Nat.le_mul_of_pos_right _ (Nat.pos_of_ne_zero hn)
    have hcarryTotal : wordMod^10 * r9.2 ≤ acc.value + modulus.value * q := by
      rw [← hall]
      exact Nat.le_add_left _ _
    exact (Nat.not_lt_of_ge (hcarry.trans hcarryTotal)) hsum
  have htop : acc.w9 + r8.2 < wordMod := by
    rw [hover, Nat.mul_zero, Nat.add_zero] at e9
    rw [← e9]
    simpa only [wordMod] using b90
  have hvalue : r0.1 + wordMod * r1.1 + wordMod^2 * r2.1 + wordMod^3 * r3.1 +
      wordMod^4 * r4.1 + wordMod^5 * r5.1 + wordMod^6 * r6.1 +
      wordMod^7 * r7.1 + wordMod^8 * r8.1 + wordMod^9 * r9.1 =
      acc.value + modulus.value * q := by
    simpa only [hover, Nat.mul_zero, Nat.add_zero] using hall
  have houtBounded :
      (⟨r0.1, r1.1, r2.1, r3.1, r4.1, r5.1, r6.1, r7.1, r8.1, r9.1⟩ : MulAcc).Bounded :=
    ⟨b00, b10, b20, b30, b40, b50, b60, b70, b80, b90⟩
  have houtValue :
      (⟨r0.1, r1.1, r2.1, r3.1, r4.1, r5.1, r6.1, r7.1, r8.1, r9.1⟩ : MulAcc).value =
        acc.value + modulus.value * q := by
    change ListArithmetic.wordsValue
      [r0.1, r1.1, r2.1, r3.1, r4.1, r5.1, r6.1, r7.1, r8.1, r9.1] =
        acc.value + modulus.value * q
    calc
      ListArithmetic.wordsValue
          [r0.1, r1.1, r2.1, r3.1, r4.1, r5.1, r6.1, r7.1, r8.1, r9.1] =
        r0.1 + wordMod * r1.1 + wordMod^2 * r2.1 + wordMod^3 * r3.1 +
          wordMod^4 * r4.1 + wordMod^5 * r5.1 + wordMod^6 * r6.1 +
          wordMod^7 * r7.1 + wordMod^8 * r8.1 + wordMod^9 * r9.1 := by
            simp only [ListArithmetic.wordsValue]
            ring
      _ = acc.value + modulus.value * q := hvalue
  change r9.2 = 0 ∧ acc.w9 + r8.2 < wordMod ∧
    (⟨r0.1, r1.1, r2.1, r3.1, r4.1, r5.1, r6.1, r7.1, r8.1, r9.1⟩ : MulAcc).Bounded ∧
    (⟨r0.1, r1.1, r2.1, r3.1, r4.1, r5.1, r6.1, r7.1, r8.1, r9.1⟩ : MulAcc).value =
      acc.value + modulus.value * q ∧
    (⟨r0.1, r1.1, r2.1, r3.1, r4.1, r5.1, r6.1, r7.1, r8.1, r9.1⟩ : MulAcc).value <
      wordMod^10
  refine ⟨hover, htop, houtBounded, houtValue, ?_⟩
  rw [houtValue]
  exact hsum

/-!
## Remaining arithmetic obligations

An end-to-end theorem for `mulMont` still needs:

* the resulting exact division equation for `shiftRound`, using `cancel_low_word`;
* eight applications of that round equation connected to `cios_eight_round_invariant` and
  `mulContract_product_lt`; and
* a value theorem for the concrete masked `selectReduced` fold, connected to `single_reduction`.

No theorem in this file claims those obligations are complete.
-/

end PastaCurves.ARMv7EM
