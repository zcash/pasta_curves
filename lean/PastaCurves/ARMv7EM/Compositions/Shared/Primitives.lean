import PastaCurves.ARMv7EM.Spec.Arithmetic
import PastaCurves.ARMv7EM.Compositions.FunsExternal

/-!
# The translated ARMv7E-M word primitives agree with the assembly transcription

The handwritten Aeneas models use checked 64-bit arithmetic. The theorems below prove that
those checks succeed for the covered 32-bit inputs and that the returned words agree with the
generated instruction-level transcription. Thus the equations also rule out every `Result`
failure and divergence case.
-/

namespace PastaCurves.ARMv7EM.Compositions.Shared.Primitives

open _root_.Aeneas _root_.Aeneas.Std

private abbrev modelAdc := PastaCurves.ARMv7EM.adc
private abbrev modelAddWord := PastaCurves.ARMv7EM.addWord
private abbrev modelSbb := PastaCurves.ARMv7EM.sbb
private abbrev modelMacWord := PastaCurves.ARMv7EM.macWord

private theorem wordPair_unique {lo hi lo' hi' total : Nat}
    (h : lo + 2^32 * hi = total) (h' : lo' + 2^32 * hi' = total)
    (hlo : lo < 2^32) (hlo' : lo' < 2^32) : (lo, hi) = (lo', hi') := by
  apply Prod.ext <;> omega

private theorem modelAdc_eq (a b carry : Nat) (ha : a < 2^32) (hb : b < 2^32)
    (hcarry : carry < 2^32) :
    modelAdc a b carry = ((a + b + carry) % 2^32, (a + b + carry) / 2^32) := by
  simp only [modelAdc, PastaCurves.ARMv7EM.adc, PastaCurves.ARMv7EM.addc32,
    PastaCurves.ARMv7EM.wordMod, Nat.add_zero, Nat.zero_add]
  have hab := Nat.mod_add_div (a + b) (2^32)
  have hsum := Nat.mod_add_div ((a + b) % 2^32 + carry) (2^32)
  have habMod := Nat.mod_lt (a + b) (Nat.two_pow_pos 32)
  have hsumMod := Nat.mod_lt ((a + b) % 2^32 + carry) (Nat.two_pow_pos 32)
  have habDiv : (a + b) / 2^32 ≤ 1 := by omega
  have hsumDiv : ((a + b) % 2^32 + carry) / 2^32 ≤ 1 := by omega
  rw [Nat.mod_eq_of_lt (by omega : (a + b) / 2^32 < 2^32)]
  rw [Nat.mod_eq_of_lt (by omega : (a + b) / 2^32 +
    ((a + b) % 2^32 + carry) / 2^32 < 2^32)]
  apply wordPair_unique (total := a + b + carry)
  · omega
  · exact Nat.mod_add_div _ _
  · exact hsumMod
  · exact Nat.mod_lt _ (Nat.two_pow_pos 32)

private theorem subtractionPair_spec (a b borrow : Nat) (ha : a < 2^32)
    (hb : b < 2^32) (hborrow : borrow ≤ 1) :
    let pair :=
      ((a + (2^32 - (b + borrow) % 2^32)) % 2^32,
        if a < b + borrow then 1 else 0)
    pair.1 + b + borrow = a + 2^32 * pair.2 ∧ pair.1 < 2^32 ∧ pair.2 ≤ 1 := by
  dsimp only
  constructor
  · split
    · rename_i hab
      have hsum : b + borrow ≤ 2^32 := by omega
      have hmod : (b + borrow) % 2^32 = b + borrow ∨
          (b + borrow) % 2^32 = 0 := by
        rcases lt_or_eq_of_le hsum with hlt | heq
        · exact Or.inl (Nat.mod_eq_of_lt hlt)
        · exact Or.inr (by simp [heq])
      rcases hmod with hmod | hmod <;> rw [hmod] <;> omega
    · rename_i hab
      have hsum : b + borrow < 2^32 := by omega
      rw [Nat.mod_eq_of_lt hsum]
      omega
  · constructor
    · exact Nat.mod_lt _ (Nat.two_pow_pos 32)
    · split <;> omega

private theorem modelSbb_eq (a b borrow : Nat) (ha : a < 2^32) (hb : b < 2^32)
    (hborrow : borrow ≤ 1) :
    modelSbb a b borrow =
      ((a + (2^32 - (b + borrow) % 2^32)) % 2^32, if a < b + borrow then 1 else 0) := by
  let pair :=
    ((a + (2^32 - (b + borrow) % 2^32)) % 2^32, if a < b + borrow then 1 else 0)
  obtain ⟨hpair, hpairLow, hpairBorrow⟩ := subtractionPair_spec a b borrow ha hb hborrow
  obtain ⟨hmodel, hmodelLow, hmodelBorrow⟩ :=
    PastaCurves.ARMv7EM.sbb_spec a b borrow ha hb (by omega) hborrow
      (modelSbb a b borrow) rfl
  apply Prod.ext <;> dsimp only [pair] at *
  · omega
  · omega

/-- The wrapping-negation external model is total and is unsigned negation modulo `2^32`. -/
theorem wrappingNeg_eq (x : Std.U32) :
    core.num.U32.wrapping_neg x = Result.ok ⟨-x.bv⟩ := by
  unfold core.num.U32.wrapping_neg
  apply congrArg Result.ok
  apply U32.bv_eq_imp_eq
  simp

/-- The handwritten `adc` model always succeeds and returns the generated transcription's words. -/
theorem adc_eq (a b carry : Std.U32) :
    ∃ sum carryOut : Std.U32,
      asm.armv7em.adc a b carry = Result.ok (sum, carryOut) ∧
      (sum.val, carryOut.val) = modelAdc a.val b.val carry.val := by
  let a64 := core.convert.num.FromU64U32.from a
  let b64 := core.convert.num.FromU64U32.from b
  obtain ⟨ab, hab, habVal⟩ := WP.spec_imp_exists
    (U64.add_spec (x := a64) (y := b64) (by simp [a64, b64]; scalar_tac))
  let carry64 := core.convert.num.FromU64U32.from carry
  obtain ⟨sum64, hsum, hsumVal⟩ := WP.spec_imp_exists
    (U64.add_spec (x := ab) (y := carry64) (by
      simp only [habVal, a64, b64, carry64,
        core.convert.num.FromU64U32.from_val_eq, U64.max_eq]
      scalar_tac))
  obtain ⟨high64, hhigh, hhighVal⟩ := WP.spec_imp_exists
    (U64.ShiftRight_IScalar_spec sum64 32#i32 (by norm_num) (by norm_num))
  let sum := UScalar.cast .U32 sum64
  let carryOut := UScalar.cast .U32 high64
  refine ⟨sum, carryOut, ?_, ?_⟩
  · unfold asm.armv7em.adc
    simp only [lift, bind_ok]
    rw [hab]
    simp only [bind_ok]
    rw [hsum]
    simp only [bind_ok]
    rw [hhigh]
    simp only [bind_ok]
    rfl
  · rw [modelAdc_eq a.val b.val carry.val (by scalar_tac) (by scalar_tac) (by scalar_tac)]
    simp only [sum, carryOut, UScalar.cast_val_eq, UScalarTy.U32_numBits_eq,
      Prod.mk.injEq]
    rw [hhighVal.1, Nat.shiftRight_eq_div_pow, hsumVal, habVal]
    have hshift : 32#i32.toNat = 32 := by
      simp only [IScalar.toNat]
      rfl
    simp only [a64, b64, carry64, core.convert.num.FromU64U32.from_val_eq, hshift]
    constructor
    · trivial
    · rw [Nat.mod_eq_of_lt]
      have hsumBound : a.val + b.val + carry.val < 2^34 := by scalar_tac
      omega

/-- The handwritten `add_word` model always succeeds and returns the generated transcription's
words. -/
theorem addWord_eq (a b : Std.U32) :
    ∃ sum carry : Std.U32,
      asm.armv7em.add_word a b = Result.ok (sum, carry) ∧
      (sum.val, carry.val) = modelAddWord a.val b.val := by
  let a64 := core.convert.num.FromU64U32.from a
  let b64 := core.convert.num.FromU64U32.from b
  obtain ⟨sum64, hsum, hsumVal⟩ := WP.spec_imp_exists
    (U64.add_spec (x := a64) (y := b64) (by
      simp only [a64, b64, core.convert.num.FromU64U32.from_val_eq, U64.max_eq]
      scalar_tac))
  obtain ⟨high64, hhigh, hhighVal⟩ := WP.spec_imp_exists
    (U64.ShiftRight_IScalar_spec sum64 32#i32 (by norm_num) (by norm_num))
  let sum := UScalar.cast .U32 sum64
  let carry := UScalar.cast .U32 high64
  refine ⟨sum, carry, ?_, ?_⟩
  · unfold asm.armv7em.add_word
    simp only [lift, bind_ok]
    rw [hsum]
    simp only [bind_ok]
    rw [hhigh]
    simp only [bind_ok]
    rfl
  · obtain ⟨hmodel, hmodelLow, hmodelCarry⟩ :=
      PastaCurves.ARMv7EM.addWord_spec a.val b.val (by scalar_tac) (by scalar_tac)
        (modelAddWord a.val b.val) rfl
    apply wordPair_unique (total := a.val + b.val)
    · simp only [sum, carry, UScalar.cast_val_eq, UScalarTy.U32_numBits_eq]
      rw [hhighVal.1, Nat.shiftRight_eq_div_pow, hsumVal]
      have hshift : 32#i32.toNat = 32 := by
        simp only [IScalar.toNat]
        rfl
      simp only [a64, b64, core.convert.num.FromU64U32.from_val_eq, hshift]
      rw [Nat.mod_eq_of_lt (by
        have hsumBound : a.val + b.val < 2^33 := by scalar_tac
        omega : (a.val + b.val) / 2^32 < 2^32)]
      exact Nat.mod_add_div _ _
    · exact hmodel
    · exact Nat.mod_lt _ (Nat.two_pow_pos 32)
    · exact hmodelLow

/-- For a borrow input that is a bit, the handwritten `sbb` model always succeeds and returns the
exact generated transcription words. -/
theorem sbb_eq (a b borrow : Std.U32) (hborrow : borrow.val ≤ 1) :
    ∃ difference borrowOut : Std.U32,
      asm.armv7em.sbb a b borrow = Result.ok (difference, borrowOut) ∧
      (difference.val, borrowOut.val) = modelSbb a.val b.val borrow.val := by
  let borrowBit := if borrow = 0#u32 then 0#u32 else 1#u32
  have hborrowBit : borrowBit = borrow := by
    apply UScalar.eq_of_val_eq
    simp only [borrowBit]
    split
    · rename_i hzero
      rw [hzero]
    · rename_i hne
      have hpos : 0 < borrow.val := by
        have : borrow.val ≠ 0 := by
          intro hzero
          apply hne
          exact UScalar.eq_of_val_eq hzero
        omega
      simp only [UScalar.ofNatCore_val_eq]
      omega
  let b64 := core.convert.num.FromU64U32.from b
  let borrow64 := core.convert.num.FromU64U32.from borrowBit
  obtain ⟨subtrahend, hsubtrahend, hsubtrahendVal⟩ := WP.spec_imp_exists
    (U64.add_spec (x := b64) (y := borrow64) (by
      simp only [b64, borrow64, core.convert.num.FromU64U32.from_val_eq, hborrowBit,
        U64.max_eq]
      scalar_tac))
  let subtrahend32 := UScalar.cast .U32 subtrahend
  let difference := core.num.U32.wrapping_sub a subtrahend32
  let a64 := core.convert.num.FromU64U32.from a
  let borrowOut := core.convert.num.FromU32Bool.from (a64 < subtrahend)
  refine ⟨difference, borrowOut, ?_, ?_⟩
  · unfold asm.armv7em.sbb
    simp only [lift, bind_ok]
    rw [hsubtrahend]
    simp only [bind_ok]
    rfl
  · rw [modelSbb_eq a.val b.val borrow.val (by scalar_tac) (by scalar_tac) hborrow]
    apply Prod.ext
    · simp only [difference, subtrahend32, core.num.U32.wrapping_sub_val_eq,
        UScalar.size_def, UScalarTy.U32_numBits_eq, UScalar.cast_val_eq,
        hsubtrahendVal, b64, borrow64, core.convert.num.FromU64U32.from_val_eq, hborrowBit]
    · simp only [borrowOut, core.convert.num.FromU32Bool.from]
      have hcompare : (a64 < subtrahend) ↔ (a.val < b.val + borrow.val) := by
        simp only [UScalar.lt_equiv, hsubtrahendVal, a64, b64, borrow64,
          core.convert.num.FromU64U32.from_val_eq, hborrowBit]
      simp only [decide_eq_true_eq, hcompare]
      split <;> rfl

/-- The handwritten `mac_word` model always succeeds and returns the generated transcription's
low and high words. -/
theorem macWord_eq (acc a b carry : Std.U32) :
    ∃ low high : Std.U32,
      asm.armv7em.mac_word acc a b carry = Result.ok (low, high) ∧
      (low.val, high.val) = modelMacWord acc.val a.val b.val carry.val := by
  have htotal : a.val * b.val + acc.val + carry.val ≤ U64.max := by
    have ha : a.val ≤ 4294967295 := by scalar_tac
    have hb : b.val ≤ 4294967295 := by scalar_tac
    have hacc : acc.val ≤ 4294967295 := by scalar_tac
    have hcarry : carry.val ≤ 4294967295 := by scalar_tac
    have hmul := Nat.mul_le_mul ha hb
    simp only [U64.max_eq]
    omega
  let a64 := core.convert.num.FromU64U32.from a
  let b64 := core.convert.num.FromU64U32.from b
  obtain ⟨product0, hproduct0, hproduct0Val⟩ := WP.spec_imp_exists
    (U64.mul_spec (x := a64) (y := b64) (by
      simp only [a64, b64, core.convert.num.FromU64U32.from_val_eq]
      omega))
  let acc64 := core.convert.num.FromU64U32.from acc
  obtain ⟨product1, hproduct1, hproduct1Val⟩ := WP.spec_imp_exists
    (U64.add_spec (x := product0) (y := acc64) (by
      simp only [hproduct0Val, a64, b64, acc64,
        core.convert.num.FromU64U32.from_val_eq]
      omega))
  let carry64 := core.convert.num.FromU64U32.from carry
  obtain ⟨product, hproduct, hproductVal⟩ := WP.spec_imp_exists
    (U64.add_spec (x := product1) (y := carry64) (by
      simp only [hproduct1Val, hproduct0Val, a64, b64, acc64, carry64,
        core.convert.num.FromU64U32.from_val_eq]
      exact htotal))
  obtain ⟨high64, hhigh, hhighVal⟩ := WP.spec_imp_exists
    (U64.ShiftRight_IScalar_spec product 32#i32 (by norm_num) (by norm_num))
  let low := UScalar.cast .U32 product
  let high := UScalar.cast .U32 high64
  refine ⟨low, high, ?_, ?_⟩
  · unfold asm.armv7em.mac_word
    simp only [lift, bind_ok]
    rw [hproduct0]
    simp only [bind_ok]
    rw [hproduct1]
    simp only [bind_ok]
    rw [hproduct]
    simp only [bind_ok]
    rw [hhigh]
    simp only [bind_ok]
    rfl
  · obtain ⟨hmodel, hmodelLow, hmodelHigh⟩ :=
      PastaCurves.ARMv7EM.macWord_spec acc.val a.val b.val carry.val
        (by scalar_tac) (by scalar_tac) (by scalar_tac) (by scalar_tac)
        (modelMacWord acc.val a.val b.val carry.val) rfl
    apply wordPair_unique (total := a.val * b.val + acc.val + carry.val)
    · simp only [low, high, UScalar.cast_val_eq, UScalarTy.U32_numBits_eq]
      rw [hhighVal.1, Nat.shiftRight_eq_div_pow, hproductVal, hproduct1Val,
        hproduct0Val]
      have hshift : 32#i32.toNat = 32 := by
        simp only [IScalar.toNat]
        rfl
      simp only [a64, b64, acc64, carry64,
        core.convert.num.FromU64U32.from_val_eq, hshift]
      rw [Nat.mod_eq_of_lt (by
        simp only [U64.max_eq] at htotal
        omega : (a.val * b.val + acc.val + carry.val) / 2^32 < 2^32)]
      exact Nat.mod_add_div _ _
    · exact hmodel
    · exact Nat.mod_lt _ (Nat.two_pow_pos 32)
    · exact hmodelLow

end PastaCurves.ARMv7EM.Compositions.Shared.Primitives
