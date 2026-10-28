import PastaCurves.ARMv7EM.Compositions.Model
import PastaCurves.ARMv7EM.Spec.Arithmetic.List
import Mathlib.Tactic.Ring

/-!
# Correctness of the ARMv7E-M Montgomery multiplication composition

This file connects the array loops in `Compositions.lean` to exact arithmetic.  In particular,
the accumulator value below includes all eighteen transient words; no carry beyond that array is
silently discarded.
-/

namespace PastaCurves.ARMv7EM

/-- The natural number represented by all words of an array in little-endian order. -/
def arrayValue (a : Array Nat) : Nat :=
  ListArithmetic.wordsValue a.toList

/-- Every entry of an array is a 32-bit word. -/
def ArrayWordsBounded (a : Array Nat) : Prop :=
  ListArithmetic.WordsBounded a.toList

/-- The value of the concrete eight-word splitting used by the composition. -/
theorem arrayValue_wordsOfLimbsArray (x : Limbs) (hx : x.Bounded) :
    arrayValue (wordsOfLimbsArray x) = x.toNat := by
  obtain ⟨h0, h1, h2, h3⟩ := hx
  have e0 := Nat.mod_add_div x.l0 wordMod
  have e1 := Nat.mod_add_div x.l1 wordMod
  have e2 := Nat.mod_add_div x.l2 wordMod
  have e3 := Nat.mod_add_div x.l3 wordMod
  simp only [arrayValue, wordsOfLimbsArray, ListArithmetic.wordsValue, Limbs.toNat]
  norm_num [wordMod] at h0 h1 h2 h3 e0 e1 e2 e3 ⊢
  omega

/-- The concrete eight-word splitting used by the composition is bounded. -/
theorem wordsOfLimbsArray_bounded (x : Limbs) (hx : x.Bounded) :
    ArrayWordsBounded (wordsOfLimbsArray x) := by
  obtain ⟨h0, h1, h2, h3⟩ := hx
  simp only [ArrayWordsBounded, wordsOfLimbsArray, ListArithmetic.WordsBounded,
    List.mem_cons, forall_eq_or_imp]
  norm_num [wordMod] at h0 h1 h2 h3 ⊢
  omega

/-- Reassembling a concrete array of eight words preserves its represented value. -/
theorem limbsOfWordsArray_value (w0 w1 w2 w3 w4 w5 w6 w7 : Nat) :
    (limbsOfWordsArray #[w0, w1, w2, w3, w4, w5, w6, w7]).toNat =
      arrayValue #[w0, w1, w2, w3, w4, w5, w6, w7] := by
  simp only [limbsOfWordsArray, arrayValue, ListArithmetic.wordsValue, Limbs.toNat]
  norm_num [wordMod]
  ring

/-- Reassembling eight bounded words produces bounded 64-bit limbs. -/
theorem limbsOfWordsArray_bounded (w0 w1 w2 w3 w4 w5 w6 w7 : Nat)
    (h0 : w0 < wordMod) (h1 : w1 < wordMod) (h2 : w2 < wordMod) (h3 : w3 < wordMod)
    (h4 : w4 < wordMod) (h5 : w5 < wordMod) (h6 : w6 < wordMod) (h7 : w7 < wordMod) :
    (limbsOfWordsArray #[w0, w1, w2, w3, w4, w5, w6, w7]).Bounded := by
  simp only [limbsOfWordsArray, Limbs.Bounded]
  norm_num [wordMod] at h0 h1 h2 h3 h4 h5 h6 h7 ⊢
  constructor
  · omega
  constructor
  · omega
  constructor <;> omega

/-- Updating one in-bounds list word changes exactly its radix-weighted contribution. -/
theorem ListArithmetic.wordsValue_set (xs : List Nat) (i x : Nat) (hi : i < xs.length) :
    wordsValue (xs.set i x) + xs.getD i 0 * wordMod ^ i =
      wordsValue xs + x * wordMod ^ i := by
  induction xs generalizing i with
  | nil => simp at hi
  | cons y ys ih =>
      cases i with
      | zero =>
          simp only [List.set, List.getD, List.getElem?_cons_zero, Option.getD_some,
            wordsValue, pow_zero]
          omega
      | succ i =>
          simp only [List.length_cons, Nat.succ_lt_succ_iff] at hi
          simp only [List.set, List.getD_cons_succ, wordsValue, pow_succ]
          calc
            y + wordMod * wordsValue (ys.set i x) +
                ys.getD i 0 * (wordMod ^ i * wordMod) =
              y + wordMod * (wordsValue (ys.set i x) + ys.getD i 0 * wordMod ^ i) := by
                rw [Nat.mul_add]
                ac_rfl
            _ = y + wordMod * (wordsValue ys + x * wordMod ^ i) := by rw [ih i hi]
            _ = y + wordMod * wordsValue ys + x * (wordMod ^ i * wordMod) := by
              rw [Nat.mul_add]
              ac_rfl

/-- Reading an array through its list representation has the same default behavior. -/
theorem array_toList_getD (a : Array Nat) (i fallback : Nat) :
    a.toList.getD i fallback = a.getD i fallback := by
  rw [List.getD_eq_getElem?_getD, Array.getD_eq_getD_getElem?, Array.getElem?_toList]

/-- Updating one in-bounds array word changes exactly its radix-weighted contribution. -/
theorem arrayValue_setIfInBounds (a : Array Nat) (i x : Nat) (hi : i < a.size) :
    arrayValue (a.setIfInBounds i x) + a.getD i 0 * wordMod ^ i =
      arrayValue a + x * wordMod ^ i := by
  simp only [arrayValue, Array.toList_setIfInBounds]
  rw [← array_toList_getD a i 0]
  exact ListArithmetic.wordsValue_set a.toList i x (by simpa using hi)

/-- A bounded array has a bounded word at every in-bounds index. -/
theorem getD_lt_of_bounded (a : Array Nat) (i : Nat) (ha : ArrayWordsBounded a)
    (hi : i < a.size) : a.getD i 0 < wordMod := by
  rw [← Array.getElem_eq_getD (xs := a) (i := i) (h := hi) (fallback := 0)]
  exact ha _ (by
    rw [← Array.getElem_toList hi]
    exact List.getElem_mem (by simpa using hi))

/-- Replacing an in-bounds entry by a word preserves array boundedness. -/
theorem setIfInBounds_bounded (a : Array Nat) (i x : Nat) (ha : ArrayWordsBounded a)
    (hx : x < wordMod) : ArrayWordsBounded (a.setIfInBounds i x) := by
  intro y hy
  rw [Array.toList_setIfInBounds] at hy
  rcases List.mem_or_eq_of_mem_set hy with hy | rfl
  · exact ha y hy
  · exact hx

/-- One actual `macWord` call followed by the array update used in `macWords` preserves the exact
weighted multiply-accumulate equation; its returned carry and updated array remain bounded. -/
theorem macWord_array_step (acc : Array Nat) (lhs multiplier carry index : Nat)
    (hacc : ArrayWordsBounded acc) (hindex : index < acc.size)
    (hlhs : lhs < wordMod) (hmultiplier : multiplier < wordMod) (hcarry : carry < wordMod) :
    let res := macWord (acc.getD index 0) lhs multiplier carry
    ArrayWordsBounded (acc.setIfInBounds index res.1) ∧ res.2 < wordMod ∧
      arrayValue (acc.setIfInBounds index res.1) +
          wordMod ^ index * (wordMod * res.2 + acc.getD index 0) =
        arrayValue acc + wordMod ^ index * (lhs * multiplier + acc.getD index 0 + carry) := by
  let res := macWord (acc.getD index 0) lhs multiplier carry
  obtain ⟨hvalue, hlow, hhigh⟩ := macWord_spec (acc.getD index 0) lhs multiplier carry
    (getD_lt_of_bounded acc index hacc hindex) hlhs hmultiplier hcarry res rfl
  refine ⟨setIfInBounds_bounded acc index res.1 hacc hlow, hhigh, ?_⟩
  have hset := arrayValue_setIfInBounds acc index res.1 hindex
  change arrayValue (acc.setIfInBounds index res.1) +
      wordMod ^ index * (wordMod * res.2 + acc.getD index 0) =
    arrayValue acc + wordMod ^ index * (lhs * multiplier + acc.getD index 0 + carry)
  calc
    arrayValue (acc.setIfInBounds index res.1) +
        wordMod ^ index * (wordMod * res.2 + acc.getD index 0) =
      (arrayValue (acc.setIfInBounds index res.1) + acc.getD index 0 * wordMod ^ index) +
        wordMod ^ index * (wordMod * res.2) := by ring
    _ = (arrayValue acc + res.1 * wordMod ^ index) +
        wordMod ^ index * (wordMod * res.2) := by rw [hset]
    _ = arrayValue acc + wordMod ^ index * (res.1 + wordMod * res.2) := by ring
    _ = arrayValue acc + wordMod ^ index * (lhs * multiplier + acc.getD index 0 + carry) := by
      have hw : res.1 + wordMod * res.2 = lhs * multiplier + acc.getD index 0 + carry := by
        simpa only [wordMod] using hvalue
      rw [hw]

/-- Montgomery cancellation at the 32-bit word radix used by the CIOS loops. -/
theorem cancel_low_word (t inv p0 : Nat) (h : (inv * p0 + 1) % wordMod = 0) :
    (t + p0 * (inv * t % wordMod) % wordMod) % wordMod = 0 := by
  have key : (t + p0 * (inv * t)) % wordMod = 0 := by
    have heq : t + p0 * (inv * t) = t * (inv * p0 + 1) := by ring
    rw [heq, Nat.mul_mod, h]
    simp
  have h1 : inv * t % wordMod ≡ inv * t [MOD wordMod] := Nat.mod_modEq _ _
  have h2 : p0 * (inv * t % wordMod) % wordMod ≡ p0 * (inv * t) [MOD wordMod] :=
    (Nat.mod_modEq _ _).trans (Nat.ModEq.mul_left p0 h1)
  have h3 : t + p0 * (inv * t % wordMod) % wordMod ≡ t + p0 * (inv * t) [MOD wordMod] :=
    Nat.ModEq.add_left t h2
  exact Eq.trans h3 key

/-- The little-endian value of the first `n` words supplied by `f`. -/
def prefixValue (f : Nat → Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => prefixValue f n + f n * wordMod ^ n

/-- Successive CIOS round equations telescope to the standard Montgomery invariant. -/
theorem cios_invariant (lhs modulus : Nat) (b q acc : Nat → Nat) (n : Nat)
    (hstep : ∀ i < n,
      wordMod * acc (i + 1) = acc i + lhs * b i + modulus * q i) :
    wordMod ^ n * acc n =
      acc 0 + lhs * prefixValue b n + modulus * prefixValue q n := by
  let W := wordMod
  change W ^ n * acc n = acc 0 + lhs * prefixValue b n + modulus * prefixValue q n
  change ∀ i < n, W * acc (i + 1) = acc i + lhs * b i + modulus * q i at hstep
  induction n with
  | zero => simp [prefixValue]
  | succ n ih =>
      have hs := hstep n (by omega)
      have hp : ∀ i < n,
          W * acc (i + 1) = acc i + lhs * b i + modulus * q i := by
        intro i hi
        exact hstep i (by omega)
      specialize ih hp
      change W ^ (n + 1) * acc (n + 1) =
        acc 0 + lhs * (prefixValue b n + b n * W ^ n) +
          modulus * (prefixValue q n + q n * W ^ n)
      rw [pow_succ, Nat.mul_assoc, hs, Nat.mul_add, Nat.mul_add, ih]
      ring

/-- The invariant after the ARMv7E-M composition's eight CIOS rounds. -/
theorem cios_eight_round_invariant (lhs modulus : Nat) (b q acc : Nat → Nat)
    (hstep : ∀ i < 8,
      wordMod * acc (i + 1) = acc i + lhs * b i + modulus * q i) :
    wordMod ^ 8 * acc 8 =
      acc 0 + lhs * prefixValue b 8 + modulus * prefixValue q 8 :=
  cios_invariant lhs modulus b q acc 8 hstep

/-- Eight bounded quotient words represent a number below the Montgomery radix. -/
theorem prefixValue_eight_lt (q : Nat → Nat) (hq : ∀ i < 8, q i < wordMod) :
    prefixValue q 8 < wordMod ^ 8 := by
  simp only [prefixValue]
  have h0 := hq 0 (by omega)
  have h1 := hq 1 (by omega)
  have h2 := hq 2 (by omega)
  have h3 := hq 3 (by omega)
  have h4 := hq 4 (by omega)
  have h5 := hq 5 (by omega)
  have h6 := hq 6 (by omega)
  have h7 := hq 7 (by omega)
  norm_num [wordMod] at h0 h1 h2 h3 h4 h5 h6 h7 ⊢
  omega

/-- The accumulated CIOS candidate is below twice the modulus. This also proves that, for a Pasta
modulus below `2^255`, it fits in the eight candidate words and both extra high words are zero. -/
theorem cios_candidate_lt_two_mul (radix candidate product quotient modulus : Nat)
    (hradix : 0 < radix) (hmodulus : 0 < modulus)
    (hmain : radix * candidate = product + modulus * quotient)
    (hproduct : product < radix * modulus) (hquotient : quotient < radix) :
    candidate < 2 * modulus := by
  have hq : modulus * quotient < modulus * radix :=
    (Nat.mul_lt_mul_left hmodulus).2 hquotient
  have hsum : product + modulus * quotient < radix * (2 * modulus) := by
    calc
      product + modulus * quotient < radix * modulus + modulus * radix :=
        Nat.add_lt_add hproduct hq
      _ = radix * (2 * modulus) := by ring
  rw [← hmain] at hsum
  exact (Nat.mul_lt_mul_left hradix).mp hsum

/-- One subtraction canonically reduces any candidate below `2p`. -/
theorem single_reduction (candidate modulus : Nat)
    (hcandidate : candidate < 2 * modulus) :
    let reduced := if candidate < modulus then candidate else candidate - modulus
    reduced < modulus ∧ reduced ≡ candidate [MOD modulus] := by
  dsimp
  split_ifs with hlt
  · exact ⟨hlt, Nat.ModEq.rfl⟩
  · have hle : modulus ≤ candidate := Nat.le_of_not_gt hlt
    constructor
    · omega
    · have hcong : candidate - modulus ≡ candidate [MOD modulus] := by
        apply modEq_of_add_mul (candidate - modulus) candidate 1 0 modulus
        omega
      exact hcong

/-- Either branch of the Rust multiplication contract gives the product bound needed by the final
Montgomery reduction. The extra upper-limb conditions of the right-canonical branch are retained
by `mulContract_iff` for the loop-overflow proof, but are not needed for this final bound. -/
theorem mulContract_product_lt (lhs rhs modulus : Limbs)
    (hlhs : lhs.Bounded) (hrhs : rhs.Bounded) (hm : modulus.Bounded)
    (hcontract : mulContract lhs rhs modulus = true) :
    lhs.toNat * rhs.toNat < 2^256 * modulus.toNat := by
  rcases (mulContract_iff lhs rhs modulus hlhs hrhs hm).1 hcontract with hleft | hright
  · have hr := Limbs.toNat_lt rhs hrhs
    by_cases hz : rhs.toNat = 0
    · simp [hz, Nat.zero_lt_of_lt hleft]
    · calc
        lhs.toNat * rhs.toNat < modulus.toNat * rhs.toNat :=
          Nat.mul_lt_mul_of_pos_right hleft (Nat.pos_of_ne_zero hz)
        _ < modulus.toNat * 2^256 := Nat.mul_lt_mul_of_pos_left hr (Nat.zero_lt_of_lt hleft)
        _ = 2^256 * modulus.toNat := Nat.mul_comm _ _
  · have hr := hright.1
    have hl := Limbs.toNat_lt lhs hlhs
    by_cases hz : lhs.toNat = 0
    · simp [hz, Nat.zero_lt_of_lt hr]
    · calc
        lhs.toNat * rhs.toNat < lhs.toNat * modulus.toNat :=
          Nat.mul_lt_mul_of_pos_left hr (Nat.pos_of_ne_zero hz)
        _ < 2^256 * modulus.toNat := Nat.mul_lt_mul_of_pos_right hl (Nat.zero_lt_of_lt hr)

end PastaCurves.ARMv7EM
