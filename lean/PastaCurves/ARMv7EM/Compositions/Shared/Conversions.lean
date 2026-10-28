import PastaCurves.ARMv7EM.Compositions.Funs
import PastaCurves.ARMv7EM.Compositions.Shared.Primitives
import PastaCurves.ARMv7EM.Compositions.Model
import PastaCurves.Words

/-!
# The translated ARMv7E-M array conversions

The generated Rust translations use checked operations in `Result` and a partial fixed-point
loop.  These proofs show that the loops terminate successfully on their statically sized arrays
and that their values are exactly the arrays used by the ARMv7E-M composition model.
-/

namespace PastaCurves.ARMv7EM.Compositions.Shared.Conversions

open _root_.Aeneas _root_.Aeneas.Std Result ControlFlow
open PastaCurves

/-- Read the natural-number values of an Aeneas array of eight 32-bit words. -/
def wordValues (words : Std.Array Std.U32 8#usize) : Array Nat :=
  (words.val.map fun word => word.val).toArray

/-- Every natural value read from an Aeneas array of 32-bit words is below `2^32`. -/
theorem wordValues_bounded (words : Std.Array Std.U32 8#usize) :
    ∀ word ∈ (wordValues words).toList, word < 2^32 := by
  intro word hword
  rw [wordValues, List.toList_toArray] at hword
  obtain ⟨source, _, rfl⟩ := List.mem_map.mp hword
  exact source.hBounds

private def splitState (value : Std.Array Std.U64 4#usize) (i : Nat) : List Nat :=
  let target := (ARMv7EM.wordsOfLimbsArray (limbsOfArray value)).toList
  target.take (2 * i) ++ List.replicate (8 - 2 * i) 0

private theorem toWordsLoop_spec (value : Std.Array Std.U64 4#usize)
    (words : Std.Array Std.U32 8#usize) (i : Std.Usize)
    (hi : i.val ≤ 4)
    (hwords : (wordValues words).toList = splitState value i.val) :
    pasta_curves.asm.armv7em.to_words_loop value words i ⦃ result =>
      wordValues result = ARMv7EM.wordsOfLimbsArray (limbsOfArray value) ⦄ := by
  unfold pasta_curves.asm.armv7em.to_words_loop
  apply loop.spec_decr_nat (fun (_, i) => 4 - i.val)
    (fun (words, i) => i.val ≤ 4 ∧
      (wordValues words).toList = splitState value i.val)
  · rintro ⟨words, i⟩ ⟨hi, hwords⟩
    unfold pasta_curves.asm.armv7em.to_words_loop.body
    step*
    · split_conjs
      · agrind
      · have hiCases : i.val = 0 ∨ i.val = 1 ∨ i.val = 2 ∨ i.val = 3 := by
          scalar_tac
        rcases hiCases with h0 | h1 | h2 | h3
        · simp [wordValues, splitState, ARMv7EM.wordsOfLimbsArray, limbsOfArray, h0]
            at hwords
          simp only [wordValues, List.toList_toArray]
          rw [a_post, words1_post, Std.Array.set_val_eq, Std.Array.set_val_eq,
            List.map_set, List.map_set, hwords]
          simp [splitState, i6_post, i4_post, i5_post, i3_post, i2_post, i1_post,
            i7_post, h0, ARMv7EM.wordsOfLimbsArray, limbsOfArray, UScalar.cast_val_eq,
            Nat.shiftRight_eq_div_pow]
        · simp [wordValues, splitState, ARMv7EM.wordsOfLimbsArray, limbsOfArray, h1]
            at hwords
          simp only [wordValues, List.toList_toArray]
          rw [a_post, words1_post, Std.Array.set_val_eq, Std.Array.set_val_eq,
            List.map_set, List.map_set, hwords]
          simp [splitState, i6_post, i4_post, i5_post, i3_post, i2_post, i1_post,
            i7_post, h1, ARMv7EM.wordsOfLimbsArray, limbsOfArray, UScalar.cast_val_eq,
            Nat.shiftRight_eq_div_pow]
        · simp [wordValues, splitState, ARMv7EM.wordsOfLimbsArray, limbsOfArray, h2]
            at hwords
          simp only [wordValues, List.toList_toArray]
          rw [a_post, words1_post, Std.Array.set_val_eq, Std.Array.set_val_eq,
            List.map_set, List.map_set, hwords]
          simp [splitState, i6_post, i4_post, i5_post, i3_post, i2_post, i1_post,
            i7_post, h2, ARMv7EM.wordsOfLimbsArray, limbsOfArray, UScalar.cast_val_eq,
            Nat.shiftRight_eq_div_pow]
        · simp [wordValues, splitState, ARMv7EM.wordsOfLimbsArray, limbsOfArray, h3]
            at hwords
          simp only [wordValues, List.toList_toArray]
          rw [a_post, words1_post, Std.Array.set_val_eq, Std.Array.set_val_eq,
            List.map_set, List.map_set, hwords]
          simp [splitState, i6_post, i4_post, i5_post, i3_post, i2_post, i1_post,
            i7_post, h3, ARMv7EM.wordsOfLimbsArray, limbsOfArray, UScalar.cast_val_eq,
            Nat.shiftRight_eq_div_pow]
      · agrind
    · rename_i hdone
      have hi4 : i.val = 4 := by scalar_tac
      apply Array.toList_inj.mp
      simp [wordValues, splitState, ARMv7EM.wordsOfLimbsArray, hi4] at hwords
      exact hwords
  · exact ⟨hi, hwords⟩

/-- The translated `to_words` helper terminates successfully and returns exactly the eight
32-bit words of the ARMv7E-M composition model.  The input limbs are explicitly bounded by the
`U64` representation. -/
theorem toWords_eq (value : Std.Array Std.U64 4#usize) :
    pasta_curves.asm.armv7em.to_words value ⦃ result =>
      wordValues result = ARMv7EM.wordsOfLimbsArray (limbsOfArray value) ∧
      (limbsOfArray value).Bounded ⦄ := by
  unfold pasta_curves.asm.armv7em.to_words
  apply WP.spec_mono
  · apply toWordsLoop_spec
    · norm_num
    · simp [wordValues, splitState]
  · intro result hresult
    exact ⟨hresult, limbsOfArray_bounded value⟩

private theorem orShifted_eq (low high : Nat) (hlow : low < 2^32) :
    low ||| (high * 2^32) = low + 2^32 * high := by
  calc
    _ = (low ||| (high * 2^32)) % 2^32 +
          2^32 * ((low ||| (high * 2^32)) / 2^32) :=
      (Nat.mod_add_div (low ||| (high * 2^32)) (2^32)).symm
    _ = low + 2^32 * high := by
      rw [Nat.or_mod_two_pow, Nat.or_div_two_pow]
      norm_num at hlow ⊢
      rw [Nat.mod_eq_of_lt hlow, Nat.div_eq_of_lt hlow]
      simp

private theorem joinBitVec_eq (low high : BitVec 32) :
    ((low.setWidth 64) ||| ((high.setWidth 64) <<< 32)).toNat =
      low.toNat + 2^32 * high.toNat := by
  rw [BitVec.toNat_or, BitVec.toNat_shiftLeft,
    BitVec.toNat_setWidth, BitVec.toNat_setWidth]
  have hlow : low.toNat < 2^32 := by omega
  have hhigh : high.toNat < 2^32 := by omega
  have hlow64 : low.toNat % 2^64 = low.toNat := Nat.mod_eq_of_lt (by omega)
  have hhigh64 : high.toNat % 2^64 = high.toNat := Nat.mod_eq_of_lt (by omega)
  have hshift64 : high.toNat * 2^32 % 2^64 = high.toNat * 2^32 :=
    Nat.mod_eq_of_lt (by omega)
  rw [hlow64, hhigh64]
  simp only [Nat.shiftLeft_eq]
  rw [hshift64]
  simpa [Nat.mul_comm] using
    orShifted_eq low.toNat high.toNat hlow

private theorem joinU32_eq (low high : Std.U32) (shifted : Std.U64)
    (hshifted : shifted.bv = (core.convert.num.FromU64U32.from high).bv <<< 32) :
    (core.convert.num.FromU64U32.from low ||| shifted).val =
      low.val + 2^32 * high.val := by
  calc
    _ = ((core.convert.num.FromU64U32.from low).bv ||| shifted.bv).toNat := rfl
    _ = ((core.convert.num.FromU64U32.from low).bv |||
          ((core.convert.num.FromU64U32.from high).bv <<< 32)).toNat := by rw [hshifted]
    _ = low.val + 2^32 * high.val := by
      simpa only [core.convert.num.FromU64U32.from_bv_eq, UScalar.val] using
        joinBitVec_eq low.bv high.bv

/-- Read the natural-number values of an Aeneas array of four 64-bit limbs. -/
def limbValues (value : Std.Array Std.U64 4#usize) : Array Nat :=
  (value.val.map fun limb => limb.val).toArray

private def joinedValues (words : Std.Array Std.U32 8#usize) : List Nat :=
  [words[0].val + 2^32 * words[1].val,
   words[2].val + 2^32 * words[3].val,
   words[4].val + 2^32 * words[5].val,
   words[6].val + 2^32 * words[7].val]

private def joinState (words : Std.Array Std.U32 8#usize) (i : Nat) : List Nat :=
  (joinedValues words).take i ++ List.replicate (4 - i) 0

private theorem joinedValues_eq (words : Std.Array Std.U32 8#usize) :
    ARMv7EM.limbsOfWordsArray (wordValues words) =
      ⟨(joinedValues words)[0]!, (joinedValues words)[1]!,
       (joinedValues words)[2]!, (joinedValues words)[3]!⟩ := by
  simp [ARMv7EM.limbsOfWordsArray, wordValues, joinedValues, ARMv7EM.wordMod]

private theorem fromWordsLoop_spec (words : Std.Array Std.U32 8#usize)
    (value : Std.Array Std.U64 4#usize) (i : Std.Usize)
    (hi : i.val ≤ 4)
    (hvalue : (limbValues value).toList = joinState words i.val) :
    pasta_curves.asm.armv7em.from_words_loop words value i ⦃ result =>
      limbsOfArray result = ARMv7EM.limbsOfWordsArray (wordValues words) ⦄ := by
  unfold pasta_curves.asm.armv7em.from_words_loop
  apply loop.spec_decr_nat (fun (_, i) => 4 - i.val)
    (fun (value, i) => i.val ≤ 4 ∧
      (limbValues value).toList = joinState words i.val)
  · rintro ⟨value, i⟩ ⟨hi, hvalue⟩
    unfold pasta_curves.asm.armv7em.from_words_loop.body
    by_cases hlt : i < 4#usize
    · have hiCases : i.val = 0 ∨ i.val = 1 ∨ i.val = 2 ∨ i.val = 3 := by
        scalar_tac
      rcases hiCases with h0 | h1 | h2 | h3
      · have hiEq : i = 0#usize := by agrind
        subst i
        simp only [hlt, ↓reduceIte]
        step*
        have hi1Eq : i1 = 0#usize := by agrind
        subst i1
        simp only [Std.lift, Std.bind_ok]
        step*
        have hjoin := joinU32_eq i2 i5 i7 i7_post1
        simp_all [limbValues, joinState, joinedValues]
      · have hiEq : i = 1#usize := by agrind
        subst i
        simp only [hlt, ↓reduceIte]
        step*
        have hi1Eq : i1 = 2#usize := by agrind
        subst i1
        simp only [Std.lift, Std.bind_ok]
        step*
        have hjoin := joinU32_eq i2 i5 i7 i7_post1
        simp_all [limbValues, joinState, joinedValues]
      · have hiEq : i = 2#usize := by agrind
        subst i
        simp only [hlt, ↓reduceIte]
        step*
        have hi1Eq : i1 = 4#usize := by agrind
        subst i1
        simp only [Std.lift, Std.bind_ok]
        step*
        have hjoin := joinU32_eq i2 i5 i7 i7_post1
        simp_all [limbValues, joinState, joinedValues]
      · have hiEq : i = 3#usize := by agrind
        subst i
        simp only [hlt, ↓reduceIte]
        step*
        have hi1Eq : i1 = 6#usize := by agrind
        subst i1
        simp only [Std.lift, Std.bind_ok]
        step*
        have hjoin := joinU32_eq i2 i5 i7 i7_post1
        simp_all [limbValues, joinState, joinedValues]
    · simp only [hlt, ↓reduceIte, WP.spec_ok]
      have hi4 : i.val = 4 := by scalar_tac
      have harray : limbValues value = (joinedValues words).toArray := by
        apply Array.toList_inj.mp
        simpa [limbValues, joinState, joinedValues, hi4] using hvalue
      rw [joinedValues_eq]
      simp only [limbsOfArray]
      have h0 := congrArg (fun a : Array Nat => a[0]!) harray
      have h1 := congrArg (fun a : Array Nat => a[1]!) harray
      have h2 := congrArg (fun a : Array Nat => a[2]!) harray
      have h3 := congrArg (fun a : Array Nat => a[3]!) harray
      simpa [limbValues] using And.intro h0 (And.intro h1 (And.intro h2 h3))
  · exact ⟨hi, hvalue⟩

/-- The translated `from_words` helper terminates successfully and reassembles exactly the four
64-bit limbs of the ARMv7E-M composition model.  The input words and returned limbs carry their
explicit representation bounds. -/
theorem fromWords_eq (words : Std.Array Std.U32 8#usize) :
    pasta_curves.asm.armv7em.from_words words ⦃ result =>
      limbsOfArray result = ARMv7EM.limbsOfWordsArray (wordValues words) ∧
      (∀ word ∈ (wordValues words).toList, word < 2^32) ∧
      (limbsOfArray result).Bounded ⦄ := by
  unfold pasta_curves.asm.armv7em.from_words
  apply WP.spec_mono
  · apply fromWordsLoop_spec
    · norm_num
    · simp [limbValues, joinState, joinedValues]
  · intro result hresult
    exact ⟨hresult, wordValues_bounded words, limbsOfArray_bounded result⟩

end PastaCurves.ARMv7EM.Compositions.Shared.Conversions
