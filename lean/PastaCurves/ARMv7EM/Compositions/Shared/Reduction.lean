import PastaCurves.ARMv7EM.Compositions.Model
import PastaCurves.ARMv7EM.Compositions.Funs
import PastaCurves.ARMv7EM.Compositions.Shared.Primitives

/-!
# The translated ARMv7E-M conditional reduction

This file connects the Aeneas translation of the Rust `select_reduced` helper to the
natural-number composition used by the ARMv7E-M arithmetic proofs.
-/

namespace PastaCurves.ARMv7EM.Compositions.Shared.Reduction

open _root_.Aeneas _root_.Aeneas.Std Result ControlFlow
open PastaCurves.ARMv7EM
open PastaCurves.ARMv7EM.Compositions.Shared.Primitives

/-- The natural-number array represented by an Aeneas array of 32-bit words. -/
def wordsValue (a : Std.Array Std.U32 8#usize) : Array Nat :=
  (a.val.map (·.val)).toArray

private theorem wordsValue_size (a : Std.Array Std.U32 8#usize) : (wordsValue a).size = 8 := by
  simp [wordsValue]

private theorem wordsValue_getD (a : Std.Array Std.U32 8#usize) (i : Std.Usize)
    (hi : i.val < 8) :
    (wordsValue a).getD i.val 0 = a.val[i.val]!.val := by
  simp [wordsValue, Array.getD, hi]

private theorem wordsValue_set (a : Std.Array Std.U32 8#usize) (i : Std.Usize)
    (x : Std.U32) :
    wordsValue (a.set i x) = (wordsValue a).setIfInBounds i.val x.val := by
  rw [← Array.toList_inj]
  simp [wordsValue]

private def writePrefix (value : Nat → Nat) (initial : Array Nat) (count : Nat) : Array Nat :=
  (List.range count).foldl (fun result i => result.setIfInBounds i (value i)) initial

private theorem writePrefix_succ (value : Nat → Nat) (initial : Array Nat) (i : Nat) :
    writePrefix value initial (i + 1) =
      (writePrefix value initial i).setIfInBounds i (value i) := by
  simp [writePrefix, List.range_succ]

private theorem writePrefix_size (value : Nat → Nat) (initial : Array Nat) (count : Nat) :
    (writePrefix value initial count).size = initial.size := by
  induction count with
  | zero => rfl
  | succ i ih => simp [writePrefix_succ, ih]

private theorem writePrefix_getElem?_of_lt (value : Nat → Nat) (initial : Array Nat)
    (count j : Nat) (hj : j < count) (hsize : j < initial.size) :
    (writePrefix value initial count)[j]? = some (value j) := by
  induction count with
  | zero => omega
  | succ i ih =>
      rw [writePrefix_succ, Array.getElem?_setIfInBounds]
      by_cases h : i = j
      · subst i
        simp [hsize, writePrefix_size]
      · simp only [h, if_false]
        exact ih (by omega)

private theorem writePrefix_getElem?_of_ge (value : Nat → Nat) (initial : Array Nat)
    (count j : Nat) (hj : count ≤ j) :
    (writePrefix value initial count)[j]? = initial[j]? := by
  induction count with
  | zero => rfl
  | succ i ih =>
      rw [writePrefix_succ, Array.getElem?_setIfInBounds]
      have hne : i ≠ j := by omega
      simp only [hne, if_false]
      exact ih (by omega)

private theorem writePrefix_getD_of_ge (value : Nat → Nat) (initial : Array Nat)
    (count j : Nat) (hj : count ≤ j) :
    (writePrefix value initial count).getD j 0 = initial.getD j 0 := by
  simp only [Array.getD_eq_getD_getElem?]
  rw [writePrefix_getElem?_of_ge value initial count j hj]

private theorem writePrefix_eight (value : Nat → Nat) (a b : Array Nat)
    (ha : a.size = 8) (hb : b.size = 8) : writePrefix value a 8 = writePrefix value b 8 := by
  apply Array.ext_getElem?
  intro j
  by_cases hj : j < 8
  · rw [writePrefix_getElem?_of_lt value a 8 j hj (by simpa [ha] using hj)]
    rw [writePrefix_getElem?_of_lt value b 8 j hj (by simpa [hb] using hj)]
  · rw [Array.getElem?_eq_none (by simpa [writePrefix_size, ha] using hj)]
    rw [Array.getElem?_eq_none (by simpa [writePrefix_size, hb] using hj)]

private def subBorrow (candidate modulus : Std.Array Std.U32 8#usize) : Nat → Nat
  | 0 => 0
  | i + 1 =>
      (sbb ((wordsValue candidate).getD i 0) ((wordsValue modulus).getD i 0)
        (subBorrow candidate modulus i)).2

private def subWord (candidate modulus : Std.Array Std.U32 8#usize) (i : Nat) : Nat :=
  (sbb ((wordsValue candidate).getD i 0) ((wordsValue modulus).getD i 0)
    (subBorrow candidate modulus i)).1

private def subPrefix (candidate modulus : Std.Array Std.U32 8#usize) (count : Nat) :
    Array Nat × Nat :=
  (writePrefix (subWord candidate modulus) (Array.replicate 8 0) count,
    subBorrow candidate modulus count)

private theorem subPrefix_succ (candidate modulus : Std.Array Std.U32 8#usize) (i : Nat) :
    subPrefix candidate modulus (i + 1) =
      let state := subPrefix candidate modulus i
      let result := sbb ((wordsValue candidate).getD i 0)
        ((wordsValue modulus).getD i 0) state.2
      (state.1.setIfInBounds i result.1, result.2) := by
  simp [subPrefix, subBorrow, subWord, writePrefix_succ]

@[step]
private theorem translatedSbb_spec (a b borrow : Std.U32) (hborrow : borrow.val ≤ 1) :
    asm.armv7em.sbb a b borrow ⦃ (difference borrowOut : Std.U32) =>
      (difference.val, borrowOut.val) = sbb a.val b.val borrow.val ∧ borrowOut.val ≤ 1 ⦄ := by
  obtain ⟨difference, borrowOut, hcall, hmodel⟩ := sbb_eq a b borrow hborrow
  rw [hcall]
  refine ⟨hmodel, ?_⟩
  have hborrowOut : borrowOut.val = (sbb a.val b.val borrow.val).2 := by
    simpa only using congrArg Prod.snd hmodel
  rw [hborrowOut]
  exact (sbb_spec a.val b.val borrow.val (by scalar_tac) (by scalar_tac)
    (by scalar_tac) hborrow _ rfl).2.2

/-- The translated subtraction loop equals the corresponding prefix of the pure subtraction
fold, provided its incoming borrow is a bit. -/
theorem subLoop_spec (candidate modulus reduced : Std.Array Std.U32 8#usize)
    (borrow : Std.U32) (i : Std.Usize) (hi : i.val ≤ 8)
    (hstate : wordsValue reduced = (subPrefix candidate modulus i.val).1)
    (hborrow : borrow.val = (subPrefix candidate modulus i.val).2)
    (hborrowBit : borrow.val ≤ 1) :
    pasta_curves.asm.armv7em.select_reduced_loop0 candidate modulus reduced borrow i
      ⦃ (result : Std.Array Std.U32 8#usize) (borrowOut : Std.U32) =>
        wordsValue result = (subPrefix candidate modulus 8).1 ∧
        borrowOut.val = (subPrefix candidate modulus 8).2 ∧ borrowOut.val ≤ 1 ⦄ := by
  unfold pasta_curves.asm.armv7em.select_reduced_loop0
  apply loop.spec_decr_nat (fun (_, _, i) => 8 - i.val)
    (fun (reduced, borrow, i) =>
      i.val ≤ 8 ∧
      wordsValue reduced = (subPrefix candidate modulus i.val).1 ∧
      borrow.val = (subPrefix candidate modulus i.val).2 ∧ borrow.val ≤ 1)
  · rintro ⟨reduced, borrow, i⟩ ⟨hi, hstate, hborrow, hborrowBit⟩
    unfold pasta_curves.asm.armv7em.select_reduced_loop0.body
    by_cases hlt : i < pasta_curves.asm.armv7em.WORDS
    · simp only [hlt, if_true]
      have hi8 : i.val < 8 := by
        simpa [pasta_curves.asm.armv7em.WORDS] using hlt
      step
      have hiCandidate : i.val < candidate.val.length := by simpa using hi8
      have hi1 : i1.val = (wordsValue candidate).getD i.val 0 := by
        rw [i1_post, wordsValue_getD candidate i hi8]
        exact congrArg UScalar.val
          (List.Inhabited_getElem_eq_getElem! candidate.val i.val hiCandidate)
      step
      have hiModulus : i.val < modulus.val.length := by simpa using hi8
      have hi2 : i2.val = (wordsValue modulus).getD i.val 0 := by
        rw [i2_post, wordsValue_getD modulus i hi8]
        exact congrArg UScalar.val
          (List.Inhabited_getElem_eq_getElem! modulus.val i.val hiModulus)
      step with translatedSbb_spec i1 i2 borrow hborrowBit
      have hi3 : i3.val = (sbb i1.val i2.val borrow.val).1 :=
        congrArg Prod.fst i3_post
      have hborrow1 : borrow1.val = (sbb i1.val i2.val borrow.val).2 :=
        congrArg Prod.snd i3_post
      step
      step
      split_conjs
      · scalar_tac
      · rw [i4_post, subPrefix_succ]
        dsimp only
        rw [← hstate, ← hborrow, ← hi1, ← hi2, ← hi3, a_post, wordsValue_set]
      · rw [i4_post, subPrefix_succ]
        dsimp only
        rw [← hborrow, ← hi1, ← hi2, ← hborrow1]
      · omega
      · scalar_tac
    · simp only [hlt, if_false, WP.spec_ok]
      have hieq : i.val = 8 := by
        have : ¬i.val < 8 := by
          simpa [pasta_curves.asm.armv7em.WORDS] using hlt
        scalar_tac
      simpa [hieq] using And.intro hstate (And.intro hborrow hborrowBit)
  · exact ⟨hi, hstate, hborrow, hborrowBit⟩

private def selectPrefix (candidate : Std.Array Std.U32 8#usize) (mask keepMask : Nat)
    (source initial : Array Nat) (count : Nat) : Array Nat :=
  writePrefix (fun i =>
    ((source.getD i 0) &&& mask) ||| (((wordsValue candidate).getD i 0) &&& keepMask))
    initial count

private theorem selectPrefix_succ (candidate : Std.Array Std.U32 8#usize)
    (mask keepMask : Nat) (source initial : Array Nat) (i : Nat) :
    selectPrefix candidate mask keepMask source initial (i + 1) =
      let state := selectPrefix candidate mask keepMask source initial i
      state.setIfInBounds i
        (((source.getD i 0) &&& mask) ||| (((wordsValue candidate).getD i 0) &&& keepMask)) := by
  exact writePrefix_succ _ _ _

private theorem selectPrefix_size (candidate : Std.Array Std.U32 8#usize)
    (mask keepMask : Nat) (source initial : Array Nat) (i : Nat) :
    (selectPrefix candidate mask keepMask source initial i).size = initial.size := by
  exact writePrefix_size _ _ _

private theorem selectPrefix_getD_current (candidate : Std.Array Std.U32 8#usize)
    (mask keepMask : Nat) (source initial : Array Nat) (i : Nat) :
    (selectPrefix candidate mask keepMask source initial i).getD i 0 = initial.getD i 0 := by
  exact writePrefix_getD_of_ge _ _ i i (by omega)

/-- The translated mask-selection loop equals the corresponding prefix of the pure selection
fold. -/
theorem selectLoop_spec (candidate source reduced : Std.Array Std.U32 8#usize)
    (i : Std.Usize) (mask : Std.U32) (hi : i.val ≤ 8)
    (hstate : wordsValue reduced =
      selectPrefix candidate mask.val (~~~mask).val (wordsValue source) (wordsValue source) i.val) :
    pasta_curves.asm.armv7em.select_reduced_loop1 candidate reduced i mask
      ⦃ (result : Std.Array Std.U32 8#usize) =>
        wordsValue result =
          selectPrefix candidate mask.val (~~~mask).val (wordsValue source) (wordsValue source) 8 ⦄ := by
  unfold pasta_curves.asm.armv7em.select_reduced_loop1
  apply loop.spec_decr_nat (fun (_, i) => 8 - i.val)
    (fun (reduced, i) =>
      i.val ≤ 8 ∧ wordsValue reduced =
        selectPrefix candidate mask.val (~~~mask).val (wordsValue source) (wordsValue source) i.val)
  · rintro ⟨reduced, i⟩ ⟨hi, hstate⟩
    unfold pasta_curves.asm.armv7em.select_reduced_loop1.body
    by_cases hlt : i < pasta_curves.asm.armv7em.WORDS
    · simp only [hlt, if_true]
      have hi8 : i.val < 8 := by
        simpa [pasta_curves.asm.armv7em.WORDS] using hlt
      step
      have hiReduced : i.val < reduced.val.length := by simpa using hi8
      have hi1 : i1.val = (wordsValue reduced).getD i.val 0 := by
        rw [i1_post, wordsValue_getD reduced i hi8]
        exact congrArg UScalar.val
          (List.Inhabited_getElem_eq_getElem! reduced.val i.val hiReduced)
      step
      step
      have hiCandidate : i.val < candidate.val.length := by simpa using hi8
      have hi3 : i3.val = (wordsValue candidate).getD i.val 0 := by
        rw [i3_post, wordsValue_getD candidate i hi8]
        exact congrArg UScalar.val
          (List.Inhabited_getElem_eq_getElem! candidate.val i.val hiCandidate)
      step
      step
      step
      step
      step
      split_conjs
      · scalar_tac
      · rw [i7_post, selectPrefix_succ]
        dsimp only
        rw [← hstate, ← hi3, a_post, wordsValue_set]
        have hsource : (wordsValue source).getD i.val 0 = i1.val := by
          rw [hi1, hstate, selectPrefix_getD_current]
        rw [hsource]
        simp only [i2_post, i4_post, i5_post, i6_post, UScalar.val_and, UScalar.val_or]
      · scalar_tac
    · simp only [hlt, if_false, WP.spec_ok]
      have hieq : i.val = 8 := by
        have : ¬i.val < 8 := by
          simpa [pasta_curves.asm.armv7em.WORDS] using hlt
        scalar_tac
      simpa [hieq] using hstate
  · exact ⟨hi, hstate⟩

private theorem subFold_eq (candidate modulus : Std.Array Std.U32 8#usize) (count : Nat) :
    (List.range count).foldl (fun st i =>
      let res := sbb (st.1.getD i 0) ((wordsValue modulus).getD i 0) st.2
      (st.1.setIfInBounds i res.1, res.2)) (wordsValue candidate, 0) =
      (writePrefix (subWord candidate modulus) (wordsValue candidate) count,
        subBorrow candidate modulus count) := by
  induction count with
  | zero => rfl
  | succ i ih =>
      rw [List.range_succ, List.foldl_append, ih]
      have hcurrent :
          (writePrefix (subWord candidate modulus) (wordsValue candidate) i).getD i 0 =
            (wordsValue candidate).getD i 0 :=
        writePrefix_getD_of_ge _ _ i i (by omega)
      simp only [List.foldl_cons, List.foldl_nil, subBorrow, subWord, writePrefix_succ]
      rw [hcurrent]

private theorem selectPrefix_eight (candidate : Std.Array Std.U32 8#usize)
    (mask keepMask : Nat) (source initial : Array Nat) (hinitial : initial.size = 8) :
    selectPrefix candidate mask keepMask source initial 8 =
      selectPrefix candidate mask keepMask source (Array.replicate 8 0) 8 := by
  apply writePrefix_eight
  · exact hinitial
  · simp

private theorem selectReduced_eq_prefix (candidate modulus : Std.Array Std.U32 8#usize)
    (highNonzero : Nat) :
    selectReduced (wordsValue candidate) (wordsValue modulus) highNonzero =
      let sub := subPrefix candidate modulus 8
      let subtract := highNonzero ||| (sub.2 ^^^ 1)
      let mask := (wordMod - subtract) % wordMod
      let keepMask := (wordMod - 1) ^^^ mask
      selectPrefix candidate mask keepMask sub.1 (Array.replicate words 0) 8 := by
  unfold selectReduced subPrefix
  rw [subFold_eq candidate modulus 8]
  rw [writePrefix_eight (subWord candidate modulus) (wordsValue candidate)
    (Array.replicate 8 0) (wordsValue_size candidate) (by simp)]
  rfl

/-- The translated conditional-reduction helper is exactly the ARMv7E-M composition.  The high
word indicator has the bit-valued contract established by its translated callers. -/
theorem selectReduced_eq (candidate modulus : Std.Array Std.U32 8#usize)
    (highNonzero : Std.U32) (hhigh : highNonzero.val ≤ 1) :
    pasta_curves.asm.armv7em.select_reduced candidate modulus highNonzero ⦃ result =>
      wordsValue result = selectReduced (wordsValue candidate) (wordsValue modulus)
        highNonzero.val ⦄ := by
  unfold pasta_curves.asm.armv7em.select_reduced
  step with subLoop_spec candidate modulus (Std.Array.repeat 8#usize 0#u32) 0#u32 0#usize
    (by norm_num)
    (by
      unfold subPrefix writePrefix
      rw [← Array.toList_inj]
      simp [wordsValue])
    (by rfl) (by norm_num)
    as ⟨reduced, borrow, hreduced, hborrow, hborrowBit⟩
  step
  step
  step
  have hiBit : i.val ≤ 1 := by
    rw [i_post, UScalar.val_xor]
    obtain hzero | hone : borrow.val = 0 ∨ borrow.val = 1 := by omega
    · simp [hzero]
    · simp [hone]
  have hsubtract : subtract.val ≤ 1 := by
    rw [subtract_post, UScalar.val_or]
    obtain hhighZero | hhighOne : highNonzero.val = 0 ∨ highNonzero.val = 1 := by omega
    · simp [hhighZero, hiBit]
    · obtain hiZero | hiOne : i.val = 0 ∨ i.val = 1 := by omega
      · simp [hhighOne, hiZero]
      · simp [hhighOne, hiOne]
  have hmask : mask.val = (wordMod - subtract.val) % wordMod := by
    rw [mask_post, core.num.U32.wrapping_sub_val_eq]
    simp only [UScalar.ofNatCore_val_eq, UScalar.size, UScalarTy.U32_numBits_eq, Nat.zero_add,
      wordMod]
  have hkeepMask : (~~~mask).val = (wordMod - 1) ^^^ mask.val := by
    rw [UScalar.val, UScalar.bv_not, ← BitVec.allOnes_xor]
    rw [BitVec.toNat_xor, BitVec.toNat_allOnes]
    rfl
  step with selectLoop_spec candidate reduced reduced 0#usize mask (by norm_num) (by
    rfl) as ⟨result, hresult⟩
  rw [hresult, selectPrefix_eight _ _ _ _ _ (wordsValue_size reduced)]
  rw [selectReduced_eq_prefix]
  dsimp only
  have hiVal : i.val = borrow.val ^^^ 1 := by
    simpa only [UScalar.val_xor, UScalar.ofNatCore_val_eq] using i_post
  have hsubtractVal : subtract.val = highNonzero.val ||| i.val := by
    simpa only [UScalar.val_or] using subtract_post
  rw [← hreduced, ← hborrow, ← hiVal, ← hsubtractVal, ← hmask, ← hkeepMask]

end PastaCurves.ARMv7EM.Compositions.Shared.Reduction
