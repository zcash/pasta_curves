import PastaCurves.ARMv7EM.Compositions.Funs
import PastaCurves.ARMv7EM.Compositions.Shared.Primitives
import PastaCurves.ARMv7EM.Compositions.Model

/-!
# The translated ARMv7E-M rolling helpers

This file relates the checked Aeneas translations of the rolling multiplication helpers to the
explicit ten-word models in `ARMv7EM.Compositions`. It proves the three generated loops terminate
and agree with pure word-operation models under their exact index bounds, and proves `mac_product`
and `mac_sparse_modulus` under their explicit top-word overflow bounds and `shift_round` under the
Rust assertion that reduction cancelled the low word.
-/

namespace PastaCurves.ARMv7EM.Compositions.Shared.Loops

open _root_.Aeneas _root_.Aeneas.Std Result ControlFlow
open PastaCurves.ARMv7EM

/-- Forget the fixed length and scalar wrappers of an Aeneas array. -/
def natArray {n : Std.Usize} (a : Std.Array Std.U32 n) : Array Nat :=
  (a.val.map (fun x => x.val)).toArray

@[simp]
theorem natArray_size {n : Std.Usize} (a : Std.Array Std.U32 n) :
    (natArray a).size = n.val := by
  simp [natArray]

@[simp]
theorem natArray_getD {n : Std.Usize} (a : Std.Array Std.U32 n) (i : Nat) :
    (natArray a).getD i 0 = (a.val.getD i 0#u32).val := by
  by_cases hi : i < n.val
  · simp [natArray, Array.getD, hi]
  · simp [natArray, Array.getD, hi]

theorem natArray_getElem {n : Std.Usize} (a : Std.Array Std.U32 n) (i : Nat)
    (hi : i < n.val) : (a.val[i]'(by simpa using hi)).val = (natArray a).getD i 0 := by
  simpa [Array.getD, hi] using (natArray_getD a i).symm

@[simp]
theorem natArray_set {n : Std.Usize} (a : Std.Array Std.U32 n) (i : Std.Usize)
    (x : Std.U32) :
    natArray (a.set i x) = (natArray a).setIfInBounds i.val x.val := by
  rw [← Array.toList_inj]
  simp [natArray, List.map_set]

/-- The eight natural-number words represented by a translated field-element array. -/
def fieldWords (a : Std.Array Std.U32 8#usize) : FieldWords :=
  let a := natArray a
  ⟨a.getD 0 0, a.getD 1 0, a.getD 2 0, a.getD 3 0,
   a.getD 4 0, a.getD 5 0, a.getD 6 0, a.getD 7 0⟩

/-- The ten natural-number words represented by a translated rolling accumulator. -/
def mulAcc (a : Std.Array Std.U32 10#usize) : MulAcc :=
  let a := natArray a
  ⟨a.getD 0 0, a.getD 1 0, a.getD 2 0, a.getD 3 0, a.getD 4 0,
   a.getD 5 0, a.getD 6 0, a.getD 7 0, a.getD 8 0, a.getD 9 0⟩

/-- One translated multiply-accumulate operation agrees with the instruction-level model. -/
@[step]
theorem macWord_step (acc lhs multiplier carry : Std.U32) :
    asm.armv7em.mac_word acc lhs multiplier carry ⦃ low high =>
      (low.val, high.val) = macWord acc.val lhs.val multiplier.val carry.val ⦄ := by
  obtain ⟨low, high, heq, hmodel⟩ :=
    Primitives.macWord_eq acc lhs multiplier carry
  rw [heq]
  simpa using hmodel

/-- One translated addition of two words agrees with the instruction-level model. -/
@[step]
theorem addWord_step (a b : Std.U32) :
    asm.armv7em.add_word a b ⦃ sum carry =>
      (sum.val, carry.val) = addWord a.val b.val ⦄ := by
  obtain ⟨sum, carry, heq, hmodel⟩ := Primitives.addWord_eq a b
  rw [heq]
  simpa using hmodel

/-- One translated addition with carry agrees with the instruction-level model. -/
@[step]
theorem adc_step (a b carry : Std.U32) :
    asm.armv7em.adc a b carry ⦃ sum carryOut =>
      (sum.val, carryOut.val) = adc a.val b.val carry.val ⦄ := by
  obtain ⟨sum, carryOut, heq, hmodel⟩ := Primitives.adc_eq a b carry
  rw [heq]
  simpa using hmodel

/-- The pure suffix of the eight-word product accumulation. -/
def macProductWords (acc multiplicand : Array Nat) (multiplier carry i count : Nat) :
    Array Nat × Nat :=
  match count with
  | 0 => (acc, carry)
  | count + 1 =>
      let result := macWord (acc.getD i 0) (multiplicand.getD i 0) multiplier carry
      macProductWords (acc.setIfInBounds i result.1) multiplicand multiplier result.2
        (i + 1) count

/-- The generated product loop terminates from every index at most eight and agrees with the
corresponding suffix of the pure eight-word accumulation. -/
theorem macProductLoop_eq (acc : Std.Array Std.U32 10#usize)
    (multiplicand : Std.Array Std.U32 8#usize) (multiplier carry : Std.U32)
    (i : Std.Usize) (hi : i.val ≤ 8) :
    pasta_curves.asm.armv7em.mac_product_loop acc multiplicand multiplier carry i
      ⦃ out finalCarry =>
        (natArray out, finalCarry.val) =
          macProductWords (natArray acc) (natArray multiplicand) multiplier.val carry.val i.val
            (8 - i.val) ⦄ := by
  unfold pasta_curves.asm.armv7em.mac_product_loop
  apply loop.spec_decr_nat (fun (_, _, j) => 8 - j.val)
    (fun (a, c, j) => j.val ≤ 8 ∧
      macProductWords (natArray a) (natArray multiplicand) multiplier.val c.val j.val
          (8 - j.val) =
        macProductWords (natArray acc) (natArray multiplicand) multiplier.val carry.val i.val
          (8 - i.val))
  · rintro ⟨before, c, j⟩ ⟨hj, hinv⟩
    unfold pasta_curves.asm.armv7em.mac_product_loop.body
    by_cases hlt : j < pasta_curves.asm.armv7em.WORDS
    · simp only [hlt, if_true]
      have hjlt : j.val < 8 := by
        simpa [pasta_curves.asm.armv7em.WORDS] using hlt
      step
      step
      step with macWord_step i1 i2 multiplier c
      step
      step
      have hi1 : i1.val = (natArray before).getD j.val 0 := by
        rw [i1_post]
        simp [natArray, Array.getD, show j.val < 10 by omega]
      have hi2 : i2.val = (natArray multiplicand).getD j.val 0 := by
        rw [i2_post]
        simp [natArray, Array.getD, hjlt]
      have hlow := congrArg Prod.fst i3_post
      have hhigh := congrArg Prod.snd i3_post
      simp only at hlow hhigh
      rw [hi1, hi2] at hlow hhigh
      refine ⟨?_, ?_, ?_⟩
      · omega
      · rw [← hinv]
        rw [show 8 - j.val = (8 - i4.val) + 1 by omega]
        simp only [macProductWords]
        rw [a_post, natArray_set, hlow, hhigh, i4_post]
      · omega
    · simp only [hlt, if_false, WP.spec_ok]
      have hj8 : j.val = 8 := by
        have : ¬j.val < 8 := by
          simpa [pasta_curves.asm.armv7em.WORDS] using hlt
        omega
      simp only [hj8, Nat.sub_self, macProductWords] at hinv
      simpa using hinv
  · exact ⟨hi, rfl⟩

/-- The pure suffix of the low sparse-modulus multiply-accumulate loop. -/
def macSparseWords (acc modulus : Array Nat) (q carry i count : Nat) : Array Nat × Nat :=
  match count with
  | 0 => (acc, carry)
  | count + 1 =>
      let result := macWord (acc.getD i 0) (modulus.getD i 0) q carry
      macSparseWords (acc.setIfInBounds i result.1) modulus q result.2 (i + 1) count

/-- The generated sparse-modulus multiply-accumulate loop terminates from every index at most
four and agrees with the corresponding pure suffix. -/
theorem macSparseModulusLoop0_eq (acc : Std.Array Std.U32 10#usize)
    (modulus : Std.Array Std.U32 8#usize) (q carry : Std.U32) (i : Std.Usize)
    (hi : i.val ≤ 4) :
    pasta_curves.asm.armv7em.mac_sparse_modulus_loop0 acc modulus q carry i
      ⦃ out finalCarry j =>
        (natArray out, finalCarry.val) =
            macSparseWords (natArray acc) (natArray modulus) q.val carry.val i.val
              (4 - i.val) ∧
          j.val = 4 ⦄ := by
  unfold pasta_curves.asm.armv7em.mac_sparse_modulus_loop0
  apply loop.spec_decr_nat (fun (_, _, j) => 4 - j.val)
    (fun (a, c, j) => j.val ≤ 4 ∧
      macSparseWords (natArray a) (natArray modulus) q.val c.val j.val (4 - j.val) =
        macSparseWords (natArray acc) (natArray modulus) q.val carry.val i.val (4 - i.val))
  · rintro ⟨before, c, j⟩ ⟨hj, hinv⟩
    unfold pasta_curves.asm.armv7em.mac_sparse_modulus_loop0.body
    by_cases hlt : j < 4#usize
    · simp only [hlt, if_true]
      have hjlt : j.val < 4 := by simpa using hlt
      step
      step
      step with macWord_step i1 i2 q c
      step
      step
      have hi1 : i1.val = (natArray before).getD j.val 0 := by
        rw [i1_post]
        simp [natArray, Array.getD, show j.val < 10 by omega]
      have hi2 : i2.val = (natArray modulus).getD j.val 0 := by
        rw [i2_post]
        simp [natArray, Array.getD, show j.val < 8 by omega]
      have hlow := congrArg Prod.fst i3_post
      have hhigh := congrArg Prod.snd i3_post
      simp only at hlow hhigh
      rw [hi1, hi2] at hlow hhigh
      refine ⟨?_, ?_, ?_⟩
      · omega
      · rw [← hinv]
        rw [show 4 - j.val = (4 - i4.val) + 1 by omega]
        simp only [macSparseWords]
        rw [a_post, natArray_set, hlow, hhigh, i4_post]
      · omega
    · simp only [hlt, if_false, WP.spec_ok]
      have hj4 : j.val = 4 := by
        have : ¬j.val < 4 := by simpa using hlt
        omega
      simp only [hj4, Nat.sub_self, macSparseWords] at hinv
      exact ⟨hinv, hj4⟩
  · exact ⟨hi, rfl⟩

/-- The pure suffix of the sparse-modulus carry-propagation loop. -/
def addWordSuffix (acc : Array Nat) (carry i count : Nat) : Array Nat × Nat :=
  match count with
  | 0 => (acc, carry)
  | count + 1 =>
      let result := addWord (acc.getD i 0) carry
      addWordSuffix (acc.setIfInBounds i result.1) result.2 (i + 1) count

/-- The generated sparse-modulus carry loop terminates from every index at most seven and agrees
with the corresponding pure suffix. -/
theorem macSparseModulusLoop1_eq (acc : Std.Array Std.U32 10#usize)
    (carry : Std.U32) (i : Std.Usize) (hi : i.val ≤ 7) :
    pasta_curves.asm.armv7em.mac_sparse_modulus_loop1 acc carry i
      ⦃ out finalCarry =>
        (natArray out, finalCarry.val) =
          addWordSuffix (natArray acc) carry.val i.val (7 - i.val) ⦄ := by
  unfold pasta_curves.asm.armv7em.mac_sparse_modulus_loop1
  apply loop.spec_decr_nat (fun (_, _, j) => 7 - j.val)
    (fun (a, c, j) => j.val ≤ 7 ∧
      addWordSuffix (natArray a) c.val j.val (7 - j.val) =
        addWordSuffix (natArray acc) carry.val i.val (7 - i.val))
  · rintro ⟨before, c, j⟩ ⟨hj, hinv⟩
    unfold pasta_curves.asm.armv7em.mac_sparse_modulus_loop1.body
    by_cases hlt : j < 7#usize
    · simp only [hlt, if_true]
      have hjlt : j.val < 7 := by simpa using hlt
      step
      step with addWord_step i1 c
      step
      step
      have hi1 : i1.val = (natArray before).getD j.val 0 := by
        rw [i1_post]
        simp [natArray, Array.getD, show j.val < 10 by omega]
      have hlow := congrArg Prod.fst i2_post
      have hhigh := congrArg Prod.snd i2_post
      simp only at hlow hhigh
      rw [hi1] at hlow hhigh
      refine ⟨?_, ?_, ?_⟩
      · omega
      · rw [← hinv]
        rw [show 7 - j.val = (7 - i3.val) + 1 by omega]
        simp only [addWordSuffix]
        rw [a_post, natArray_set, hlow, hhigh, i3_post]
      · omega
    · simp only [hlt, if_false, WP.spec_ok]
      have hj7 : j.val = 7 := by
        have : ¬j.val < 7 := by simpa using hlt
        omega
      simp only [hj7, Nat.sub_self, addWordSuffix] at hinv
      simpa using hinv
  · exact ⟨hi, rfl⟩

/-- Finishing the pure product-loop result with the two high-word additions gives the explicit
rolling-accumulator composition. -/
private theorem macProductWords_finish (acc : Std.Array Std.U32 10#usize)
    (multiplicand : Std.Array Std.U32 8#usize) (multiplier : Std.U32) :
    let loopResult :=
      macProductWords (natArray acc) (natArray multiplicand) multiplier.val 0 0 8
    let r8 := addWord (loopResult.1.getD 8 0) loopResult.2
    let r9 := addWord (loopResult.1.getD 9 0) r8.2
    (⟨loopResult.1.getD 0 0, loopResult.1.getD 1 0, loopResult.1.getD 2 0,
      loopResult.1.getD 3 0, loopResult.1.getD 4 0, loopResult.1.getD 5 0,
      loopResult.1.getD 6 0, loopResult.1.getD 7 0, r8.1, r9.1⟩ : MulAcc) =
      macProduct (mulAcc acc) (fieldWords multiplicand) multiplier.val := by
  simp only [macProductWords, macProduct, mulAcc, fieldWords]
  simp

private theorem macSparseWords_finish (acc : Std.Array Std.U32 10#usize)
    (modulus : Std.Array Std.U32 8#usize) (q : Std.U32) :
    let r0 := adc ((natArray acc).getD 0 0) q.val 0
    let lowResult :=
      macSparseWords ((natArray acc).setIfInBounds 0 r0.1) (natArray modulus) q.val r0.2 1 3
    let carryResult := addWordSuffix lowResult.1 lowResult.2 4 3
    let r7 := macWord (carryResult.1.getD 7 0) ((natArray modulus).getD 7 0) q.val carryResult.2
    let r8 := addWord (carryResult.1.getD 8 0) r7.2
    let r9 := addWord (carryResult.1.getD 9 0) r8.2
    (⟨carryResult.1.getD 0 0, carryResult.1.getD 1 0,
      carryResult.1.getD 2 0, carryResult.1.getD 3 0,
      carryResult.1.getD 4 0, carryResult.1.getD 5 0,
      carryResult.1.getD 6 0, r7.1, r8.1, r9.1⟩ : MulAcc) =
      macSparseModulus (mulAcc acc) (fieldWords modulus) q.val := by
  simp only [macSparseWords, addWordSuffix, macSparseModulus, mulAcc, fieldWords]
  simp

private theorem addWord_snd_eq_zero (a b : Nat) (ha : a < wordMod) (hb : b < wordMod)
    (hsum : a + b < wordMod) : (addWord a b).2 = 0 := by
  obtain ⟨hexact, _, _⟩ := addWord_spec a b ha hb (addWord a b) rfl
  simp only [wordMod] at hsum ⊢
  omega

private theorem addWord_u32_snd_eq_zero (a b : Std.U32)
    (hsum : a.val + b.val < wordMod) : (addWord a.val b.val).2 = 0 :=
  addWord_snd_eq_zero a.val b.val a.hBounds b.hBounds hsum

private theorem mulAcc_set_high (acc : Std.Array Std.U32 10#usize) (word8 word9 : Std.U32) :
    mulAcc ((acc.set 8#usize word8).set 9#usize word9) =
      ⟨(natArray acc).getD 0 0, (natArray acc).getD 1 0,
       (natArray acc).getD 2 0, (natArray acc).getD 3 0,
       (natArray acc).getD 4 0, (natArray acc).getD 5 0,
       (natArray acc).getD 6 0, (natArray acc).getD 7 0,
       word8.val, word9.val⟩ := by
  simp [mulAcc, natArray, Std.Array.set_val_eq, List.map_set, Array.getD]

private theorem mulAcc_set_tail (acc : Std.Array Std.U32 10#usize)
    (word7 word8 word9 : Std.U32) :
    mulAcc (((acc.set 7#usize word7).set 8#usize word8).set 9#usize word9) =
      ⟨(natArray acc).getD 0 0, (natArray acc).getD 1 0,
       (natArray acc).getD 2 0, (natArray acc).getD 3 0,
       (natArray acc).getD 4 0, (natArray acc).getD 5 0,
       (natArray acc).getD 6 0, word7.val, word8.val, word9.val⟩ := by
  simp [mulAcc, natArray, Std.Array.set_val_eq, List.map_set, Array.getD]

/-- The translated product helper implements the explicit rolling-accumulator composition whenever
the exact bound required by its final overflow assertion holds. -/
theorem macProduct_eq (acc : Std.Array Std.U32 10#usize)
    (multiplicand : Std.Array Std.U32 8#usize) (multiplier : Std.U32)
    (hnoOverflow :
      let loopResult :=
        macProductWords (natArray acc) (natArray multiplicand) multiplier.val 0 0 8
      let r8 := addWord (loopResult.1.getD 8 0) loopResult.2
      loopResult.1.getD 9 0 + r8.2 < wordMod) :
    pasta_curves.asm.armv7em.mac_product acc multiplicand multiplier ⦃ result =>
      mulAcc result = macProduct (mulAcc acc) (fieldWords multiplicand) multiplier.val ⦄ := by
  let loopResult :=
    macProductWords (natArray acc) (natArray multiplicand) multiplier.val 0 0 8
  change loopResult.1.getD 9 0 + (addWord (loopResult.1.getD 8 0) loopResult.2).2 < wordMod at hnoOverflow
  unfold pasta_curves.asm.armv7em.mac_product
  step with macProductLoop_eq acc multiplicand multiplier 0#u32 0#usize (by norm_num)
    as ⟨afterLoop, carry, hloop⟩
  step
  · simp [pasta_curves.asm.armv7em.WORDS]
  step with addWord_step i carry
  step
  step
  step
  · have hi2 : i2.val = 9 := by
      simpa [pasta_curves.asm.armv7em.WORDS] using i2_post
    have hlength : acc2.length = 10 := by
      simpa only [Std.Array.length, UScalar.ofNatCore_val_eq] using acc2.property
    omega
  obtain ⟨top, overflow, haddWord, htopModel⟩ := Primitives.addWord_eq i3 carry1
  rw [haddWord]
  simp only [Std.bind_ok]
  have hi : i.val = (natArray afterLoop).getD 8 0 := by
    rw [i_post]
    simpa [pasta_curves.asm.armv7em.WORDS] using natArray_getElem afterLoop 8 (by norm_num)
  have hi2 : i2 = 9#usize := by
    apply UScalar.eq_of_val_eq
    simpa [pasta_curves.asm.armv7em.WORDS] using i2_post
  have hi2Val : i2.val = 9 := congrArg UScalar.val hi2
  have hi3 : i3.val = (natArray acc2).getD 9 0 := by
    rw [i3_post]
    calc
      (acc2.val[i2.val]'(by
          rw [acc2.property, hi2Val]
          norm_num)).val = (natArray acc2).getD i2.val 0 :=
        natArray_getElem acc2 i2.val (by
          rw [hi2Val]
          norm_num)
      _ = (natArray acc2).getD 9 0 := by rw [hi2Val]
  have hword8Low := congrArg Prod.fst i1_post
  have hword8Carry := congrArg Prod.snd i1_post
  have htopLow := congrArg Prod.fst htopModel
  have htopCarry := congrArg Prod.snd htopModel
  simp only at hword8Low hword8Carry htopLow htopCarry
  rw [hi] at hword8Low hword8Carry
  rw [hi3] at htopLow htopCarry
  have hafterLoop : natArray afterLoop = loopResult.1 := by
    simpa only [loopResult] using congrArg Prod.fst hloop
  have hcarry : carry.val = loopResult.2 := by
    simpa only [loopResult] using congrArg Prod.snd hloop
  have hacc2 : natArray acc2 = (natArray afterLoop).setIfInBounds 8 i1.val := by
    rw [acc2_post, natArray_set]
    simp [pasta_curves.asm.armv7em.WORDS]
  have hword9 : (natArray acc2).getD 9 0 = loopResult.1.getD 9 0 := by
    calc
      (natArray acc2).getD 9 0 = (natArray afterLoop).getD 9 0 := by
        rw [hacc2]
        simp
      _ = _ := congrArg (fun a : Array Nat => a.getD 9 0) hafterLoop
  have hcarry8 : carry1.val = (addWord (loopResult.1.getD 8 0) loopResult.2).2 := by
    rw [hword8Carry, hafterLoop, hcarry]
  have htopSum : i3.val + carry1.val < wordMod := by
    rw [hi3, hword9, hcarry8]
    exact hnoOverflow
  have hoverflow : overflow.val = 0 := by
    rw [htopCarry]
    rw [← hi3]
    exact addWord_u32_snd_eq_zero i3 carry1 htopSum
  have hoverflowScalar : overflow = 0#u32 :=
    UScalar.eq_of_val_eq (by simpa using hoverflow)
  have hi2Word : i2 = 9#usize := by
    apply UScalar.eq_of_val_eq
    simpa [pasta_curves.asm.armv7em.WORDS] using i2_post
  have hwords : pasta_curves.asm.armv7em.WORDS = 8#usize := by
    apply UScalar.eq_of_val_eq
    simp [pasta_curves.asm.armv7em.WORDS]
  have hresult : mulAcc (acc2.set i2 top) =
      macProduct (mulAcc acc) (fieldWords multiplicand) multiplier.val := by
    rw [hi2Word, acc2_post, hwords, mulAcc_set_high]
    rw [hword8Low, hafterLoop, hcarry, htopLow, hword9, hcarry8]
    simpa only [loopResult] using macProductWords_finish acc multiplicand multiplier
  have hbound : i2.val < acc2.length := by
    rw [hi2Word]
    simp [Std.Array.length, acc2.property]
  have hsome : acc2.val[i2.val]? = some (acc2.val[i2.val]'(by simpa [Std.Array.length] using hbound)) := by
    exact List.getElem?_eq_getElem (by simpa [Std.Array.length] using hbound)
  have hupdate : acc2.update i2 top = ok (acc2.set i2 top) := by
    simp [Std.Array.update, Std.Array.set, Std.Array.getElem?_Usize_eq, hsome]
  have hassert : massert (overflow = 0#u32) = ok () :=
    (massert_ok _).2 hoverflowScalar
  change (do
    massert (overflow = 0#u32)
    acc2.update i2 top) ⦃ result =>
      mulAcc result = macProduct (mulAcc acc) (fieldWords multiplicand) multiplier.val ⦄
  simp only [hassert, hupdate, Std.bind_ok, WP.spec_ok]
  exact hresult

/-- The translated sparse-modulus helper implements the explicit rolling-accumulator composition
when the modulus has the word shape asserted by the Rust code and the final top-word addition does
not overflow. -/
theorem macSparseModulus_eq (acc : Std.Array Std.U32 10#usize)
    (modulus : Std.Array Std.U32 8#usize) (q : Std.U32)
    (hshape :
      (fieldWords modulus).w0 = 1 ∧
      (fieldWords modulus).w4 = 0 ∧
      (fieldWords modulus).w5 = 0 ∧
      (fieldWords modulus).w6 = 0 ∧
      (fieldWords modulus).w7 = 2^30)
    (hnoOverflow :
      let r0 := adc ((natArray acc).getD 0 0) q.val 0
      let lowResult :=
        macSparseWords ((natArray acc).setIfInBounds 0 r0.1) (natArray modulus) q.val r0.2 1 3
      let carryResult := addWordSuffix lowResult.1 lowResult.2 4 3
      let r7 :=
        macWord (carryResult.1.getD 7 0) ((natArray modulus).getD 7 0) q.val carryResult.2
      let r8 := addWord (carryResult.1.getD 8 0) r7.2
      carryResult.1.getD 9 0 + r8.2 < wordMod) :
    pasta_curves.asm.armv7em.mac_sparse_modulus acc modulus q ⦃ result =>
      mulAcc result = macSparseModulus (mulAcc acc) (fieldWords modulus) q.val ⦄ := by
  let r0 := adc ((natArray acc).getD 0 0) q.val 0
  let lowResult :=
    macSparseWords ((natArray acc).setIfInBounds 0 r0.1) (natArray modulus) q.val r0.2 1 3
  let carryResult := addWordSuffix lowResult.1 lowResult.2 4 3
  let r7 := macWord (carryResult.1.getD 7 0) ((natArray modulus).getD 7 0) q.val carryResult.2
  let r8 := addWord (carryResult.1.getD 8 0) r7.2
  change carryResult.1.getD 9 0 + r8.2 < wordMod at hnoOverflow
  unfold pasta_curves.asm.armv7em.mac_sparse_modulus
  step
  have hleft : left_val = 1#u32 := by
    apply UScalar.eq_of_val_eq
    simpa [fieldWords, natArray, left_val_post] using hshape.1
  step
  step
  have hleft1 : left_val1 = 0#u32 := by
    apply UScalar.eq_of_val_eq
    simpa [fieldWords, natArray, left_val1_post] using hshape.2.1
  step
  step
  have hleft2 : left_val2 = 0#u32 := by
    apply UScalar.eq_of_val_eq
    simpa [fieldWords, natArray, left_val2_post] using hshape.2.2.1
  step
  step
  have hleft3 : left_val3 = 0#u32 := by
    apply UScalar.eq_of_val_eq
    simpa [fieldWords, natArray, left_val3_post] using hshape.2.2.2.1
  step
  step
  step
  have hright : right_val.val = 2^30 := by
    rw [right_val_post]
    norm_num [U32.size, U32.numBits_eq, Nat.shiftLeft_eq]
  have hleft4 : left_val4 = right_val := by
    apply UScalar.eq_of_val_eq
    rw [left_val4_post, hright]
    simpa [fieldWords, natArray] using hshape.2.2.2.2
  step
  step
  step with adc_step i q 0#u32
  step
  step with macSparseModulusLoop0_eq a modulus q carry 1#usize (by norm_num)
    as ⟨acc1, carry1, i2, hloop0, hi2⟩
  step with macSparseModulusLoop1_eq acc1 carry1 i2 (by omega)
    as ⟨acc2, carry2, hloop1⟩
  step
  step
  step with macWord_step i3 i4 q carry2
  step
  step
  · simp [pasta_curves.asm.armv7em.WORDS]
  step with addWord_step i6 carry3
  step
  step
  step
  · have hi8 : i8.val = 9 := by
      simpa [pasta_curves.asm.armv7em.WORDS] using i8_post
    have hlength : acc4.length = 10 := by
      simpa only [Std.Array.length, UScalar.ofNatCore_val_eq] using acc4.property
    omega
  obtain ⟨top, overflow, haddWord, htopModel⟩ := Primitives.addWord_eq i9 carry4
  rw [haddWord]
  simp only [Std.bind_ok]
  have hi : i.val = (natArray acc).getD 0 0 := by
    rw [i_post]
    simpa using natArray_getElem acc 0 (by norm_num)
  have hi1Low := congrArg Prod.fst i1_post
  have hi1Carry := congrArg Prod.snd i1_post
  simp only at hi1Low hi1Carry
  rw [hi] at hi1Low hi1Carry
  have ha : natArray a = (natArray acc).setIfInBounds 0 i1.val := by
    rw [a_post, natArray_set]
    norm_num
  have hloop0Acc : natArray acc1 =
      (macSparseWords (natArray a) (natArray modulus) q.val carry.val 1 (4 - 1)).1 :=
    congrArg Prod.fst hloop0
  have hloop0Carry : carry1.val =
      (macSparseWords (natArray a) (natArray modulus) q.val carry.val 1 (4 - 1)).2 :=
    congrArg Prod.snd hloop0
  have hlowResult : natArray acc1 = lowResult.1 := by
    rw [hloop0Acc, ha, hi1Low, hi1Carry]
  have hlowCarry : carry1.val = lowResult.2 := by
    rw [hloop0Carry, ha, hi1Low, hi1Carry]
  have hloop1Acc : natArray acc2 =
      (addWordSuffix (natArray acc1) carry1.val i2.val (7 - i2.val)).1 :=
    congrArg Prod.fst hloop1
  have hloop1Carry : carry2.val =
      (addWordSuffix (natArray acc1) carry1.val i2.val (7 - i2.val)).2 :=
    congrArg Prod.snd hloop1
  have hcarryResult : natArray acc2 = carryResult.1 := by
    rw [hloop1Acc, hlowResult, hlowCarry, hi2]
  have hcarry : carry2.val = carryResult.2 := by
    rw [hloop1Carry, hlowResult, hlowCarry, hi2]
  have hi3Acc : i3.val = (natArray acc2).getD 7 0 := by
    rw [i3_post]
    simpa using natArray_getElem acc2 7 (by norm_num)
  have hi3 : i3.val = carryResult.1.getD 7 0 := by
    rw [hi3Acc, hcarryResult]
  have hi4 : i4.val = (natArray modulus).getD 7 0 := by
    rw [i4_post]
    simpa using natArray_getElem modulus 7 (by norm_num)
  have hi5Low := congrArg Prod.fst i5_post
  have hi5Carry := congrArg Prod.snd i5_post
  simp only at hi5Low hi5Carry
  rw [hi3, hi4, hcarry] at hi5Low hi5Carry
  have hacc3 : natArray acc3 = (natArray acc2).setIfInBounds 7 i5.val := by
    rw [acc3_post, natArray_set]
    norm_num
  have hwords : pasta_curves.asm.armv7em.WORDS = 8#usize := by
    apply UScalar.eq_of_val_eq
    simp [pasta_curves.asm.armv7em.WORDS]
  have hi6Acc : i6.val = (natArray acc3).getD 8 0 := by
    rw [i6_post]
    simpa [pasta_curves.asm.armv7em.WORDS] using natArray_getElem acc3 8 (by norm_num)
  have hi6 : i6.val = carryResult.1.getD 8 0 := by
    calc
      i6.val = (natArray acc3).getD 8 0 := hi6Acc
      _ = (natArray acc2).getD 8 0 := by rw [hacc3]; simp
      _ = _ := congrArg (fun a : Array Nat => a.getD 8 0) hcarryResult
  have hi7Low := congrArg Prod.fst i7_post
  have hi7Carry := congrArg Prod.snd i7_post
  simp only at hi7Low hi7Carry
  rw [hi6, hi5Carry] at hi7Low hi7Carry
  have hi8 : i8 = 9#usize := by
    apply UScalar.eq_of_val_eq
    rw [i8_post, congrArg UScalar.val hwords]
    norm_num
  have hi8Val : i8.val = 9 := congrArg UScalar.val hi8
  have hacc4 : natArray acc4 = (natArray acc3).setIfInBounds 8 i7.val := by
    rw [acc4_post, hwords, natArray_set]
    norm_num
  have hi9 : i9.val = carryResult.1.getD 9 0 := by
    rw [i9_post]
    calc
      (acc4.val[i8.val]'(by
          rw [acc4.property, hi8Val]
          norm_num)).val = (natArray acc4).getD i8.val 0 :=
        natArray_getElem acc4 i8.val (by
          rw [hi8Val]
          norm_num)
      _ = (natArray acc4).getD 9 0 := by rw [hi8Val]
      _ = (natArray acc3).getD 9 0 := by rw [hacc4]; simp
      _ = (natArray acc2).getD 9 0 := by rw [hacc3]; simp
      _ = _ := congrArg (fun a : Array Nat => a.getD 9 0) hcarryResult
  have htopLow := congrArg Prod.fst htopModel
  have htopCarry := congrArg Prod.snd htopModel
  simp only at htopLow htopCarry
  rw [hi9, hi7Carry] at htopLow htopCarry
  have htopSum : i9.val + carry4.val < wordMod := by
    rw [hi9, hi7Carry]
    exact hnoOverflow
  have hoverflow : overflow.val = 0 := by
    rw [htopCarry]
    rw [← hi9, ← hi7Carry]
    exact addWord_u32_snd_eq_zero i9 carry4 htopSum
  have hoverflowScalar : overflow = 0#u32 :=
    UScalar.eq_of_val_eq (by simpa using hoverflow)
  have hi8Word : i8 = 9#usize := by
    apply UScalar.eq_of_val_eq
    simpa [pasta_curves.asm.armv7em.WORDS] using i8_post
  have hresult : mulAcc (acc4.set i8 top) =
      macSparseModulus (mulAcc acc) (fieldWords modulus) q.val := by
    rw [hi8Word, acc4_post, hwords, acc3_post, mulAcc_set_tail]
    rw [hcarryResult, hi5Low, hi7Low, htopLow]
    simpa only [r0, lowResult, carryResult, r7, r8] using
      macSparseWords_finish acc modulus q
  have hbound : i8.val < acc4.length := by
    rw [hi8Word]
    simp [Std.Array.length, acc4.property]
  have hsome : acc4.val[i8.val]? = some (acc4.val[i8.val]'(by simpa [Std.Array.length] using hbound)) := by
    exact List.getElem?_eq_getElem (by simpa [Std.Array.length] using hbound)
  have hupdate : acc4.update i8 top = ok (acc4.set i8 top) := by
    simp [Std.Array.update, Std.Array.set, Std.Array.getElem?_Usize_eq, hsome]
  have hassert : massert (overflow = 0#u32) = ok () :=
    (massert_ok _).2 hoverflowScalar
  change (do
    massert (overflow = 0#u32)
    acc4.update i8 top) ⦃ result =>
      mulAcc result = macSparseModulus (mulAcc acc) (fieldWords modulus) q.val ⦄
  simp only [hassert, hupdate, Std.bind_ok, WP.spec_ok]
  exact hresult

/-- The translated shift succeeds once reduction has cancelled the low word, and implements the
explicit rolling-accumulator shift. -/
theorem shiftRound_eq (acc : Std.Array Std.U32 10#usize)
    (hlow : (mulAcc acc).w0 = 0) :
    pasta_curves.asm.armv7em.shift_round acc ⦃ result =>
      mulAcc result = shiftRound (mulAcc acc) ⦄ := by
  unfold pasta_curves.asm.armv7em.shift_round
  step
  have hleft : left_val = 0#u32 := by
    apply UScalar.eq_of_val_eq
    simpa [mulAcc, natArray, left_val_post] using hlow
  step
  step*
  simp_all [mulAcc, shiftRound, natArray, Std.Array.set_val_eq, List.map_set]

end PastaCurves.ARMv7EM.Compositions.Shared.Loops
