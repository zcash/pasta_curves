import PastaCurves.ARMv7EM.Compositions.Model
import PastaCurves.ARMv7EM.Spec.Arithmetic.List
import PastaCurves.Fields
import PastaCurves.Glue.Spec

/-!
# ARMv7E-M word-array arithmetic

Array/list correspondence, fixed eight-word carry and borrow folds, mask selection, and bounded
reassembly lemmas. The shared word operations and reassembly lemmas are used by the live
multiplication and Montgomery-conversion proofs.
-/

namespace PastaCurves.ARMv7EM

open ListArithmetic

/-- Splitting bounded limbs with the array model agrees with the list arithmetic model. -/
theorem wordsOfLimbsArray_toList (x : Limbs) (hx : x.Bounded) :
    (wordsOfLimbsArray x).toList = toWords x := by
  obtain ⟨h0, h1, h2, h3⟩ := hx
  simp only [wordsOfLimbsArray, toWords]
  norm_num [wordMod] at h0 h1 h2 h3 ⊢
  omega

/-- Reassembling an exact eight-word list agrees between the array and list models. -/
theorem limbsOfWordsArray_toArray (xs : List Nat) (hlen : xs.length = 8) :
    limbsOfWordsArray xs.toArray = fromWords xs := by
  rcases xs with _ | ⟨w0, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨w1, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨w2, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨w3, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨w4, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨w5, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨w6, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨w7, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨w8, xs⟩ <;> simp at hlen
  simp_all [limbsOfWordsArray, fromWords]

private theorem adcFold_eight (x0 x1 x2 x3 x4 x5 x6 x7
    y0 y1 y2 y3 y4 y5 y6 y7 : Nat) :
    (List.range words).foldl (fun st i =>
      let res := adc (st.1.getD i 0) (#[y0, y1, y2, y3, y4, y5, y6, y7].getD i 0) st.2
      (st.1.setIfInBounds i res.1, res.2))
      (#[x0, x1, x2, x3, x4, x5, x6, x7], 0) =
    let res := adcChain 0 [x0, x1, x2, x3, x4, x5, x6, x7]
      [y0, y1, y2, y3, y4, y5, y6, y7]
    (res.1.toArray, res.2) := by
  rw [show List.range words = [0, 1, 2, 3, 4, 5, 6, 7] by decide]
  simp [adcChain]

private theorem sbbFold_eight (x0 x1 x2 x3 x4 x5 x6 x7
    y0 y1 y2 y3 y4 y5 y6 y7 : Nat) :
    (List.range words).foldl (fun st i =>
      let res := sbb (st.1.getD i 0) (#[y0, y1, y2, y3, y4, y5, y6, y7].getD i 0) st.2
      (st.1.setIfInBounds i res.1, res.2))
      (#[x0, x1, x2, x3, x4, x5, x6, x7], 0) =
    let res := sbbChain 0 [x0, x1, x2, x3, x4, x5, x6, x7]
      [y0, y1, y2, y3, y4, y5, y6, y7]
    (res.1.toArray, res.2) := by
  rw [show List.range words = [0, 1, 2, 3, 4, 5, 6, 7] by decide]
  simp [sbbChain]

private theorem setEight (x0 x1 x2 x3 x4 x5 x6 x7 : Nat) :
    let a := Array.replicate 8 0
    let a := a.setIfInBounds 0 x0
    let a := a.setIfInBounds 1 x1
    let a := a.setIfInBounds 2 x2
    let a := a.setIfInBounds 3 x3
    let a := a.setIfInBounds 4 x4
    let a := a.setIfInBounds 5 x5
    let a := a.setIfInBounds 6 x6
    let a := a.setIfInBounds 7 x7
    a = #[x0, x1, x2, x3, x4, x5, x6, x7] := by
  rw [← Array.toList_inj]
  simp

private theorem maskedFold_left (xs ys : List Nat) (hlenx : xs.length = 8)
    (hleny : ys.length = 8) (hxs : WordsBounded xs) :
    (List.range words).foldl (fun result i =>
      result.setIfInBounds i
        (((xs.toArray.getD i 0) &&& (wordMod - 1)) ||| ((ys.toArray.getD i 0) &&& 0)))
      (Array.replicate words 0) = xs.toArray := by
  rcases xs with _ | ⟨x0, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x1, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x2, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x3, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x4, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x5, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x6, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x7, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x8, xs⟩ <;> simp at hlenx
  rcases ys with _ | ⟨y0, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y1, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y2, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y3, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y4, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y5, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y6, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y7, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y8, ys⟩ <;> simp at hleny
  simp only [WordsBounded, List.mem_cons, forall_eq_or_imp] at hxs
  have hx0 : x0 &&& (wordMod - 1) = x0 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hxs.1]
  have hx1 : x1 &&& (wordMod - 1) = x1 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hxs.2.1]
  have hx2 : x2 &&& (wordMod - 1) = x2 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hxs.2.2.1]
  have hx3 : x3 &&& (wordMod - 1) = x3 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hxs.2.2.2.1]
  have hx4 : x4 &&& (wordMod - 1) = x4 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hxs.2.2.2.2.1]
  have hx5 : x5 &&& (wordMod - 1) = x5 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hxs.2.2.2.2.2.1]
  have hx6 : x6 &&& (wordMod - 1) = x6 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hxs.2.2.2.2.2.2.1]
  have hx7 : x7 &&& (wordMod - 1) = x7 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hxs.2.2.2.2.2.2.2.1]
  norm_num [wordMod] at hx0 hx1 hx2 hx3 hx4 hx5 hx6 hx7
  rw [show List.range words = [0, 1, 2, 3, 4, 5, 6, 7] by decide]
  simp only [List.foldl_cons, List.foldl_nil]
  simp only [Array.getD]
  simp
  rw [hx0, hx1, hx2, hx3, hx4, hx5, hx6, hx7]
  simpa [words] using setEight x0 x1 x2 x3 x4 x5 x6 x7

private theorem maskedFold_right (xs ys : List Nat) (hlenx : xs.length = 8)
    (hleny : ys.length = 8) (hys : WordsBounded ys) :
    (List.range words).foldl (fun result i =>
      result.setIfInBounds i
        (((xs.toArray.getD i 0) &&& 0) ||| ((ys.toArray.getD i 0) &&& (wordMod - 1))))
      (Array.replicate words 0) = ys.toArray := by
  rcases xs with _ | ⟨x0, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x1, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x2, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x3, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x4, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x5, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x6, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x7, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x8, xs⟩ <;> simp at hlenx
  rcases ys with _ | ⟨y0, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y1, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y2, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y3, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y4, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y5, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y6, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y7, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y8, ys⟩ <;> simp at hleny
  simp only [WordsBounded, List.mem_cons, forall_eq_or_imp] at hys
  rw [show List.range words = [0, 1, 2, 3, 4, 5, 6, 7] by decide]
  simp only [List.foldl_cons, List.foldl_nil]
  have hy0 : y0 &&& (wordMod - 1) = y0 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hys.1]
  have hy1 : y1 &&& (wordMod - 1) = y1 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hys.2.1]
  have hy2 : y2 &&& (wordMod - 1) = y2 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hys.2.2.1]
  have hy3 : y3 &&& (wordMod - 1) = y3 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hys.2.2.2.1]
  have hy4 : y4 &&& (wordMod - 1) = y4 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hys.2.2.2.2.1]
  have hy5 : y5 &&& (wordMod - 1) = y5 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hys.2.2.2.2.2.1]
  have hy6 : y6 &&& (wordMod - 1) = y6 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hys.2.2.2.2.2.2.1]
  have hy7 : y7 &&& (wordMod - 1) = y7 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hys.2.2.2.2.2.2.2.1]
  norm_num [wordMod] at hy0 hy1 hy2 hy3 hy4 hy5 hy6 hy7
  simp only [Array.getD]
  simp
  rw [hy0, hy1, hy2, hy3, hy4, hy5, hy6, hy7]
  simpa [words] using setEight y0 y1 y2 y3 y4 y5 y6 y7

private theorem wordsOfLimbsArray_eq (x : Limbs) (hx : x.Bounded) :
    wordsOfLimbsArray x = (toWords x).toArray := by
  rw [← Array.toList_inj, wordsOfLimbsArray_toList x hx, List.toList_toArray]

/-- The array addition fold on exact eight-word lists agrees with `adcChain`. -/
theorem adcFold_toArray (xs ys : List Nat) (hlenx : xs.length = 8)
    (hleny : ys.length = 8) :
    (List.range words).foldl (fun st i =>
      let res := adc (st.1.getD i 0) (ys.toArray.getD i 0) st.2
      (st.1.setIfInBounds i res.1, res.2)) (xs.toArray, 0) =
    let res := adcChain 0 xs ys
    (res.1.toArray, res.2) := by
  rcases xs with _ | ⟨x0, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x1, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x2, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x3, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x4, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x5, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x6, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x7, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x8, xs⟩ <;> simp at hlenx
  rcases ys with _ | ⟨y0, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y1, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y2, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y3, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y4, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y5, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y6, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y7, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y8, ys⟩ <;> simp at hleny
  exact adcFold_eight x0 x1 x2 x3 x4 x5 x6 x7 y0 y1 y2 y3 y4 y5 y6 y7

/-- The array subtraction fold on exact eight-word lists agrees with `sbbChain`. -/
theorem sbbFold_toArray (xs ys : List Nat) (hlenx : xs.length = 8)
    (hleny : ys.length = 8) :
    (List.range words).foldl (fun st i =>
      let res := sbb (st.1.getD i 0) (ys.toArray.getD i 0) st.2
      (st.1.setIfInBounds i res.1, res.2)) (xs.toArray, 0) =
    let res := sbbChain 0 xs ys
    (res.1.toArray, res.2) := by
  rcases xs with _ | ⟨x0, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x1, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x2, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x3, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x4, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x5, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x6, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x7, xs⟩ <;> simp at hlenx
  rcases xs with _ | ⟨x8, xs⟩ <;> simp at hlenx
  rcases ys with _ | ⟨y0, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y1, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y2, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y3, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y4, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y5, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y6, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y7, ys⟩ <;> simp at hleny
  rcases ys with _ | ⟨y8, ys⟩ <;> simp at hleny
  exact sbbFold_eight x0 x1 x2 x3 x4 x5 x6 x7 y0 y1 y2 y3 y4 y5 y6 y7

private theorem maskedFold_left_array (xs ys : Array Nat) (hsizex : xs.size = 8)
    (hsizey : ys.size = 8) (hxs : WordsBounded xs.toList) :
    (List.range words).foldl (fun result i =>
      result.setIfInBounds i (((xs.getD i 0) &&& (wordMod - 1)) ||| ((ys.getD i 0) &&& 0)))
      (Array.replicate words 0) = xs := by
  rw [← Array.toList_inj]
  have hlenx : xs.toList.length = 8 := by simpa using hsizex
  have hleny : ys.toList.length = 8 := by simpa using hsizey
  simpa using congrArg Array.toList (maskedFold_left xs.toList ys.toList hlenx hleny hxs)

private theorem maskedFold_right_array (xs ys : Array Nat) (hsizex : xs.size = 8)
    (hsizey : ys.size = 8) (hys : WordsBounded ys.toList) :
    (List.range words).foldl (fun result i =>
      result.setIfInBounds i (((xs.getD i 0) &&& 0) ||| ((ys.getD i 0) &&& (wordMod - 1))))
      (Array.replicate words 0) = ys := by
  rw [← Array.toList_inj]
  have hlenx : xs.toList.length = 8 := by simpa using hsizex
  have hleny : ys.toList.length = 8 := by simpa using hsizey
  simpa using congrArg Array.toList (maskedFold_right xs.toList ys.toList hlenx hleny hys)

/-- Reassembling eight bounded 32-bit words produces bounded 64-bit limbs. -/
theorem fromWords_bounded (xs : List Nat) (hlen : xs.length = 8)
    (hxs : WordsBounded xs) : (fromWords xs).Bounded := by
  rcases xs with _ | ⟨x0, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨x1, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨x2, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨x3, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨x4, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨x5, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨x6, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨x7, xs⟩ <;> simp at hlen
  rcases xs with _ | ⟨x8, xs⟩ <;> simp at hlen
  simp only [WordsBounded, List.mem_cons, forall_eq_or_imp] at hxs
  unfold fromWords Limbs.Bounded
  norm_num [wordMod] at hxs ⊢
  omega

end PastaCurves.ARMv7EM
