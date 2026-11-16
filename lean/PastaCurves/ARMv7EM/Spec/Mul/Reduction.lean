import PastaCurves.ARMv7EM.Spec.Arithmetic.AddSub
import PastaCurves.ARMv7EM.Spec.Mul.Invariant

/-!
# Correctness of the ARMv7E-M masked final reduction

The pure composition subtracts the modulus with an eight-word borrow chain, then selects either
that subtraction or the original candidate with all-zero or all-one masks.  This file proves the
selected value and, under the bounds supplied by a Pasta multiplication, connects it to
`single_reduction`.
-/

namespace PastaCurves.ARMv7EM

open ListArithmetic

private def subtraction (candidate modulus : Array Nat) : Array Nat × Nat :=
  (List.range words).foldl (fun st i =>
    let res := sbb (st.1.getD i 0) (modulus.getD i 0) st.2
    (st.1.setIfInBounds i res.1, res.2)) (candidate, 0)

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

private theorem maskedFold_left_fast (xs ys : Array Nat) (hsizex : xs.size = 8)
    (hxs : WordsBounded xs.toList) :
    (List.range words).foldl (fun result i =>
      result.setIfInBounds i (((xs.getD i 0) &&& (wordMod - 1)) ||| ((ys.getD i 0) &&& 0)))
      (Array.replicate words 0) = xs := by
  change writePrefix (fun i =>
    ((xs.getD i 0) &&& (wordMod - 1)) ||| ((ys.getD i 0) &&& 0))
    (Array.replicate words 0) words = xs
  apply Array.ext_getElem?
  intro j
  by_cases hj : j < 8
  · rw [writePrefix_getElem?_of_lt _ _ words j (by simpa [words] using hj) (by simp [words]; omega)]
    have hjx : j < xs.size := by simpa [hsizex] using hj
    have hx := getD_lt_of_bounded xs j hxs hjx
    rw [Array.getElem?_eq_getElem hjx]
    simp only [Nat.and_zero, Nat.or_zero]
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hx]
    simp [Array.getD, hjx]
  · rw [Array.getElem?_eq_none (by simpa [writePrefix_size, words] using hj)]
    rw [Array.getElem?_eq_none (by simpa [hsizex] using hj)]

private theorem maskedFold_right_fast (xs ys : Array Nat) (hsizey : ys.size = 8)
    (hys : WordsBounded ys.toList) :
    (List.range words).foldl (fun result i =>
      result.setIfInBounds i (((xs.getD i 0) &&& 0) ||| ((ys.getD i 0) &&& (wordMod - 1))))
      (Array.replicate words 0) = ys := by
  change writePrefix (fun i =>
    ((xs.getD i 0) &&& 0) ||| ((ys.getD i 0) &&& (wordMod - 1)))
    (Array.replicate words 0) words = ys
  apply Array.ext_getElem?
  intro j
  by_cases hj : j < 8
  · rw [writePrefix_getElem?_of_lt _ _ words j (by simpa [words] using hj) (by simp [words]; omega)]
    have hjy : j < ys.size := by simpa [hsizey] using hj
    have hy := getD_lt_of_bounded ys j hys hjy
    rw [Array.getElem?_eq_getElem hjy]
    simp only [Nat.zero_or, Nat.and_zero]
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hy]
    simp [Array.getD, hjy]
  · rw [Array.getElem?_eq_none (by simpa [writePrefix_size, words] using hj)]
    rw [Array.getElem?_eq_none (by simpa [hsizey] using hj)]

private theorem selectReduced_eq_subtraction (candidate modulus : Array Nat)
    (highNonzero : Nat) :
    selectReduced candidate modulus highNonzero =
      let sub := subtraction candidate modulus
      let subtract := highNonzero ||| (sub.2 ^^^ 1)
      let mask := (wordMod - subtract) % wordMod
      let keepMask := (wordMod - 1) ^^^ mask
      (List.range words).foldl (fun result i =>
        result.setIfInBounds i
          (((sub.1.getD i 0) &&& mask) ||| ((candidate.getD i 0) &&& keepMask)))
        (Array.replicate words 0) := rfl

private theorem subtraction_eq_chain (candidate modulus : Array Nat)
    (hcsize : candidate.size = 8) (hmsize : modulus.size = 8) :
    subtraction candidate modulus =
      let res := sbbChain 0 candidate.toList modulus.toList
      (res.1.toArray, res.2) := by
  have hclen : candidate.toList.length = 8 := by simpa using hcsize
  have hmlen : modulus.toList.length = 8 := by simpa using hmsize
  simpa [subtraction] using sbbFold_toArray candidate.toList modulus.toList hclen hmlen

/-- With no ninth candidate word, the masked fold implements one ordinary conditional
subtraction.  It also preserves the invariant that every array entry is a 32-bit word. -/
theorem selectReduced_zero_value (candidate modulus : Array Nat)
    (hcsize : candidate.size = 8) (hmsize : modulus.size = 8)
    (hc : ArrayWordsBounded candidate) (hm : ArrayWordsBounded modulus) :
    ArrayWordsBounded (selectReduced candidate modulus 0) ∧
      arrayValue (selectReduced candidate modulus 0) =
        if arrayValue candidate < arrayValue modulus then
          arrayValue candidate
        else
          arrayValue candidate - arrayValue modulus := by
  set sub := subtraction candidate modulus with hsubDef
  set chain := sbbChain 0 candidate.toList modulus.toList with hchainDef
  have hsub : sub = (chain.1.toArray, chain.2) := by
    rw [hsubDef, hchainDef]
    exact subtraction_eq_chain candidate modulus hcsize hmsize
  have hclen : candidate.toList.length = 8 := by simpa using hcsize
  have hmlen : modulus.toList.length = 8 := by simpa using hmsize
  have hspec := sbbChain_spec candidate.toList modulus.toList 0 (by omega) hc hm (by omega)
  change WordsBounded chain.1 ∧ chain.1.length = candidate.toList.length ∧ chain.2 ≤ 1 ∧
    wordsValue chain.1 + wordsValue modulus.toList + 0 =
      wordsValue candidate.toList + wordMod ^ candidate.toList.length * chain.2 at hspec
  have hsubBounded : ArrayWordsBounded sub.1 := by
    rw [hsub]
    simpa [ArrayWordsBounded] using hspec.1
  have hsubSize : sub.1.size = 8 := by
    rw [hsub]
    simpa [hclen] using hspec.2.1
  have hborrowBit : sub.2 ≤ 1 := by simpa [hsub] using hspec.2.2.1
  have hvalue : arrayValue sub.1 + arrayValue modulus =
      arrayValue candidate + wordMod ^ 8 * sub.2 := by
    rw [hsub]
    simpa [arrayValue, hclen] using hspec.2.2.2
  have hsubLt : arrayValue sub.1 < wordMod ^ 8 := by
    rw [hsub]
    simpa [arrayValue, hspec.2.1, hclen] using wordsValue_lt chain.1 hspec.1
  have hborrow : sub.2 = if arrayValue candidate < arrayValue modulus then 1 else 0 := by
    by_cases hlt : arrayValue candidate < arrayValue modulus
    · rw [if_pos hlt]
      have hnzero : sub.2 ≠ 0 := by
        intro hz
        rw [hz] at hvalue
        simp only [Nat.mul_zero, Nat.add_zero] at hvalue
        omega
      omega
    · rw [if_neg hlt]
      have hnone : sub.2 ≠ 1 := by
        intro hone
        rw [hone, Nat.mul_one] at hvalue
        omega
      omega
  have hselected : selectReduced candidate modulus 0 =
      if sub.2 = 0 then sub.1 else candidate := by
    rw [selectReduced_eq_subtraction]
    rw [← hsubDef]
    dsimp only
    obtain hborrowZero | hborrowOne : sub.2 = 0 ∨ sub.2 = 1 := by omega
    · rw [hborrowZero]
      have hmask : (wordMod - (0 ||| (0 ^^^ 1))) % wordMod = wordMod - 1 := by
        norm_num [wordMod]
      have hkeep : (wordMod - 1) ^^^ (wordMod - 1) = 0 := by
        exact Nat.xor_self _
      rw [hmask, hkeep]
      simpa using maskedFold_left_fast sub.1 candidate hsubSize hsubBounded
    · rw [hborrowOne]
      have hmask : (wordMod - (0 ||| (1 ^^^ 1))) % wordMod = 0 := by
        norm_num [wordMod]
      have hkeep : (wordMod - 1) ^^^ 0 = wordMod - 1 := by
        exact Nat.xor_zero _
      rw [hmask, hkeep]
      simpa using maskedFold_right_fast sub.1 candidate hcsize hc
  rw [hselected, hborrow]
  by_cases hlt : arrayValue candidate < arrayValue modulus
  · simp [hlt, hc]
  · simp only [if_neg hlt]
    have hsubValue : arrayValue sub.1 = arrayValue candidate - arrayValue modulus := by
      rw [if_neg hlt] at hborrow
      rw [hborrow, Nat.mul_zero, Nat.add_zero] at hvalue
      omega
    exact ⟨hsubBounded, hsubValue⟩

/-- Under the actual ninth-word encoding used by `mulMont`, a candidate below twice a Pasta
modulus has a zero ninth word, so the masked fold has the value of one arithmetic reduction. -/
theorem selectReduced_actual_high_value (candidate modulus : Array Nat) (high : Nat)
    (hcsize : candidate.size = 8) (hmsize : modulus.size = 8)
    (hc : ArrayWordsBounded candidate) (hm : ArrayWordsBounded modulus)
    (hhigh : high < wordMod)
    (hmodulus : 2^254 ≤ arrayValue modulus ∧ arrayValue modulus < 2^255)
    (hcandidate : arrayValue candidate + wordMod ^ 8 * high < 2 * arrayValue modulus) :
    ArrayWordsBounded (selectReduced candidate modulus
      ((high ||| ((wordMod - high) % wordMod)) / 2^31)) ∧
    arrayValue (selectReduced candidate modulus
      ((high ||| ((wordMod - high) % wordMod)) / 2^31)) =
      let fullCandidate := arrayValue candidate + wordMod ^ 8 * high
      if fullCandidate < arrayValue modulus then
        fullCandidate
      else
        fullCandidate - arrayValue modulus := by
  have hhighZero : high = 0 := by
    norm_num [wordMod] at hcandidate hmodulus hhigh ⊢
    omega
  subst high
  norm_num [wordMod]
  exact selectReduced_zero_value candidate modulus hcsize hmsize hc hm

/-- For an actual low candidate and ninth word below `2p`, where `p` has the Pasta field bounds,
the masked output is bounded, canonical, and congruent to the full candidate.  The value equality
exhibits that this is exactly `single_reduction`. -/
theorem selectReduced_actual_high_canonical (candidate modulus : Array Nat) (high : Nat)
    (hcsize : candidate.size = 8) (hmsize : modulus.size = 8)
    (hc : ArrayWordsBounded candidate) (hm : ArrayWordsBounded modulus)
    (hhigh : high < wordMod)
    (hmodulus : 2^254 ≤ arrayValue modulus ∧ arrayValue modulus < 2^255)
    (hcandidate : arrayValue candidate + wordMod ^ 8 * high < 2 * arrayValue modulus) :
    let output := selectReduced candidate modulus
      ((high ||| ((wordMod - high) % wordMod)) / 2^31)
    ArrayWordsBounded output ∧
      let fullCandidate := arrayValue candidate + wordMod ^ 8 * high
      arrayValue output =
        (if fullCandidate < arrayValue modulus then
          fullCandidate
        else
          fullCandidate - arrayValue modulus) ∧
      arrayValue output < arrayValue modulus ∧
      arrayValue output ≡ fullCandidate [MOD arrayValue modulus] := by
  dsimp only
  obtain ⟨hbounded, hvalue⟩ := selectReduced_actual_high_value candidate modulus high
    hcsize hmsize hc hm hhigh hmodulus hcandidate
  obtain ⟨hreduced, hcongruent⟩ := single_reduction
    (arrayValue candidate + wordMod ^ 8 * high) (arrayValue modulus) hcandidate
  rw [← hvalue] at hreduced hcongruent
  exact ⟨hbounded, hvalue, hreduced, hcongruent⟩

end PastaCurves.ARMv7EM
