import PastaCurves.ARMv7EM.Spec.Arithmetic
import Mathlib.Tactic.Ring

/-!
# Arithmetic for the ARMv7E-M word routines

This file gives pure models of the fixed-bound Rust loops around the generated models of
`adc`, `sbb`, and `mac_word`.  A word vector is represented as a little-endian list; the
Rust routines use lists of length eight.  The proofs call the instruction theorems from
`ARMv7EM.Spec` at every loop step.

These definitions are mathematical models, not generated transcriptions of the Rust loops.
Consequently the theorems establish the arithmetic of the modeled loops; a separate
translation-correspondence proof is required before applying them directly to compiled Rust.
-/

namespace PastaCurves.ARMv7EM.ListArithmetic

/-- The value of a little-endian list of 32-bit words. -/
def wordsValue : List Nat → Nat
  | [] => 0
  | x :: xs => x + wordMod * wordsValue xs

/-- Every member of a list is a 32-bit word. -/
def WordsBounded (xs : List Nat) : Prop := ∀ x ∈ xs, x < wordMod

/-- Split each 64-bit limb into its low and high 32-bit words, as Rust's `to_words` does. -/
def toWords (x : Limbs) : List Nat :=
  [x.l0 % wordMod, x.l0 / wordMod,
   x.l1 % wordMod, x.l1 / wordMod,
   x.l2 % wordMod, x.l2 / wordMod,
   x.l3 % wordMod, x.l3 / wordMod]

/-- Join eight 32-bit words in pairs, as Rust's `from_words` does.

The fallback is unreachable for `toWords` and for the eight-word arithmetic models below.
-/
def fromWords : List Nat → Limbs
  | [w0, w1, w2, w3, w4, w5, w6, w7] =>
      ⟨w0 + wordMod * w1, w2 + wordMod * w3,
       w4 + wordMod * w5, w6 + wordMod * w7⟩
  | _ => ⟨0, 0, 0, 0⟩

/-- Splitting four bounded 64-bit limbs produces exactly eight bounded 32-bit words. -/
theorem toWords_bounded (x : Limbs) (hx : x.Bounded) :
    (toWords x).length = 8 ∧ WordsBounded (toWords x) := by
  obtain ⟨h0, h1, h2, h3⟩ := hx
  constructor
  · rfl
  · simp only [WordsBounded, toWords, List.mem_cons, forall_eq_or_imp]
    constructor
    · exact Nat.mod_lt _ (Nat.two_pow_pos _)
    constructor
    · exact Nat.div_lt_of_lt_mul (by norm_num at h0 ⊢; exact h0)
    constructor
    · exact Nat.mod_lt _ (Nat.two_pow_pos _)
    constructor
    · exact Nat.div_lt_of_lt_mul (by norm_num at h1 ⊢; exact h1)
    constructor
    · exact Nat.mod_lt _ (Nat.two_pow_pos _)
    constructor
    · exact Nat.div_lt_of_lt_mul (by norm_num at h2 ⊢; exact h2)
    constructor
    · exact Nat.mod_lt _ (Nat.two_pow_pos _)
    constructor
    · apply Nat.div_lt_of_lt_mul
      norm_num [wordMod] at h3 ⊢
      exact h3
    · simp

/-- The eight words obtained from four bounded limbs have the same integer value. -/
theorem wordsValue_toWords (x : Limbs) (hx : x.Bounded) :
    wordsValue (toWords x) = x.toNat := by
  obtain ⟨h0, h1, h2, h3⟩ := hx
  have e0 := Nat.mod_add_div x.l0 wordMod
  have e1 := Nat.mod_add_div x.l1 wordMod
  have e2 := Nat.mod_add_div x.l2 wordMod
  have e3 := Nat.mod_add_div x.l3 wordMod
  simp only [wordsValue, toWords, Limbs.toNat]
  norm_num [wordMod] at e0 e1 e2 e3 ⊢
  omega

/-- Joining a bounded eight-word list preserves its integer value. -/
theorem fromWords_value (w0 w1 w2 w3 w4 w5 w6 w7 : Nat) :
    (fromWords [w0, w1, w2, w3, w4, w5, w6, w7]).toNat =
      wordsValue [w0, w1, w2, w3, w4, w5, w6, w7] := by
  simp only [fromWords, Limbs.toNat, wordsValue]
  norm_num [wordMod]
  ring

/-- Splitting after joining eight bounded words returns those words. -/
theorem toWords_fromWords (w0 w1 w2 w3 w4 w5 w6 w7 : Nat)
    (h0 : w0 < wordMod) (_h1 : w1 < wordMod) (h2 : w2 < wordMod) (_h3 : w3 < wordMod)
    (h4 : w4 < wordMod) (_h5 : w5 < wordMod) (h6 : w6 < wordMod) (_h7 : w7 < wordMod) :
    toWords (fromWords [w0, w1, w2, w3, w4, w5, w6, w7]) =
      [w0, w1, w2, w3, w4, w5, w6, w7] := by
  simp only [fromWords, toWords]
  norm_num [wordMod] at h0 _h1 h2 _h3 h4 _h5 h6 _h7 ⊢
  omega

/-- The exact call pattern of an `adc` loop over two equally long word lists. -/
def adcChain : Nat → List Nat → List Nat → List Nat × Nat
  | carry, x :: xs, y :: ys =>
      let head := adc x y carry
      let tail := adcChain head.2 xs ys
      (head.1 :: tail.1, tail.2)
  | carry, _, _ => ([], carry)

/-- The exact call pattern of an `sbb` loop over two equally long word lists. -/
def sbbChain : Nat → List Nat → List Nat → List Nat × Nat
  | borrow, x :: xs, y :: ys =>
      let head := sbb x y borrow
      let tail := sbbChain head.2 xs ys
      (head.1 :: tail.1, tail.2)
  | borrow, _, _ => ([], borrow)

/-- A bounded word list represents a number below the radix raised to its length. -/
theorem wordsValue_lt (xs : List Nat) (hxs : WordsBounded xs) :
    wordsValue xs < wordMod ^ xs.length := by
  induction xs with
  | nil => simp [wordsValue]
  | cons x xs ih =>
      have hx : x < wordMod := hxs x (by simp)
      have htail : WordsBounded xs := by
        intro y hy
        exact hxs y (by simp [hy])
      have hi := ih htail
      calc
        wordsValue (x :: xs) = x + wordMod * wordsValue xs := rfl
        _ < wordMod + wordMod * wordsValue xs := Nat.add_lt_add_right hx _
        _ = wordMod * (wordsValue xs + 1) := by ring
        _ ≤ wordMod * wordMod ^ xs.length :=
          Nat.mul_le_mul_left wordMod (Nat.succ_le_of_lt hi)
        _ = wordMod ^ (x :: xs).length := by
          simp only [List.length_cons, pow_succ]
          ring

/-- An `adc` chain reconstructs the full addition, including its final carry word. -/
theorem adcChain_spec (xs ys : List Nat) (carry : Nat)
    (hlen : xs.length = ys.length) (hxs : WordsBounded xs) (hys : WordsBounded ys)
    (hcarry : carry ≤ 1) :
    let res := adcChain carry xs ys
    WordsBounded res.1 ∧ res.1.length = xs.length ∧ res.2 ≤ 1 ∧
      wordsValue res.1 + wordMod ^ xs.length * res.2 =
        wordsValue xs + wordsValue ys + carry := by
  induction xs generalizing ys carry with
  | nil =>
      cases ys <;> simp_all [adcChain, wordsValue, WordsBounded]
  | cons x xs ih =>
      cases ys with
      | nil => simp at hlen
      | cons y ys =>
          have hx : x < wordMod := hxs x (by simp)
          have hy : y < wordMod := hys y (by simp)
          have hxs' : WordsBounded xs := by
            intro z hz; exact hxs z (by simp [hz])
          have hys' : WordsBounded ys := by
            intro z hz; exact hys z (by simp [hz])
          let head := adc x y carry
          obtain ⟨hhead, hbhead, hchead⟩ :=
            adc_spec x y carry hx hy (lt_of_le_of_lt hcarry (by norm_num [wordMod])) hcarry
              head rfl
          let tail := adcChain head.2 xs ys
          obtain ⟨hbtail, hltail, hctail, htail⟩ :=
            ih ys head.2 (by simpa using hlen) hxs' hys' hchead
          change WordsBounded (head.1 :: tail.1) ∧
            (head.1 :: tail.1).length = (x :: xs).length ∧ tail.2 ≤ 1 ∧
            wordsValue (head.1 :: tail.1) + wordMod ^ (x :: xs).length * tail.2 =
              wordsValue (x :: xs) + wordsValue (y :: ys) + carry
          have hbtail' : WordsBounded tail.1 := by simpa only [tail]
            using hbtail
          have hltail' : tail.1.length = xs.length := by simpa only [tail]
            using hltail
          have htail' : wordsValue tail.1 + wordMod ^ xs.length * tail.2 =
              wordsValue xs + wordsValue ys + head.2 := by
            simpa only [tail, hltail] using htail
          have hhead' : head.1 + wordMod * head.2 = x + y + carry := by
            simpa only [wordMod] using hhead
          refine ⟨?_, by simp only [List.length_cons, hltail'], hctail, ?_⟩
          · intro z hz
            simp only [List.mem_cons] at hz
            rcases hz with rfl | hz
            · exact hbhead
            · exact hbtail' z hz
          · have hfinal : head.1 + wordMod * wordsValue tail.1 +
                (wordMod * wordMod ^ xs.length) * tail.2 =
              (x + wordMod * wordsValue xs) +
                (y + wordMod * wordsValue ys) + carry := by
              calc
                _ = head.1 + wordMod *
                      (wordsValue tail.1 + wordMod ^ xs.length * tail.2) := by ring
                _ = head.1 + wordMod *
                      (wordsValue xs + wordsValue ys + head.2) := by rw [htail']
                _ = wordMod * (wordsValue xs + wordsValue ys) +
                      (head.1 + wordMod * head.2) := by ring
                _ = wordMod * (wordsValue xs + wordsValue ys) +
                      (x + y + carry) := by rw [hhead']
                _ = (x + wordMod * wordsValue xs) +
                      (y + wordMod * wordsValue ys) + carry := by ring
            simp only [wordsValue, List.length_cons, pow_succ]
            convert hfinal using 1 <;> ring

/-- An `sbb` chain reconstructs the full subtraction, including its final borrow word. -/
theorem sbbChain_spec (xs ys : List Nat) (borrow : Nat)
    (hlen : xs.length = ys.length) (hxs : WordsBounded xs) (hys : WordsBounded ys)
    (hborrow : borrow ≤ 1) :
    let res := sbbChain borrow xs ys
    WordsBounded res.1 ∧ res.1.length = xs.length ∧ res.2 ≤ 1 ∧
      wordsValue res.1 + wordsValue ys + borrow =
        wordsValue xs + wordMod ^ xs.length * res.2 := by
  induction xs generalizing ys borrow with
  | nil =>
      cases ys <;> simp_all [sbbChain, wordsValue, WordsBounded]
  | cons x xs ih =>
      cases ys with
      | nil => simp at hlen
      | cons y ys =>
          have hx : x < wordMod := hxs x (by simp)
          have hy : y < wordMod := hys y (by simp)
          have hxs' : WordsBounded xs := by
            intro z hz; exact hxs z (by simp [hz])
          have hys' : WordsBounded ys := by
            intro z hz; exact hys z (by simp [hz])
          let head := sbb x y borrow
          obtain ⟨hhead, hbhead, hchead⟩ :=
            sbb_spec x y borrow hx hy (lt_of_le_of_lt hborrow (by norm_num [wordMod])) hborrow
              head rfl
          let tail := sbbChain head.2 xs ys
          obtain ⟨hbtail, hltail, hctail, htail⟩ :=
            ih ys head.2 (by simpa using hlen) hxs' hys' hchead
          change WordsBounded (head.1 :: tail.1) ∧
            (head.1 :: tail.1).length = (x :: xs).length ∧ tail.2 ≤ 1 ∧
            wordsValue (head.1 :: tail.1) + wordsValue (y :: ys) + borrow =
              wordsValue (x :: xs) + wordMod ^ (x :: xs).length * tail.2
          have hbtail' : WordsBounded tail.1 := by simpa only [tail]
            using hbtail
          have hltail' : tail.1.length = xs.length := by simpa only [tail]
            using hltail
          have htail' : wordsValue tail.1 + wordsValue ys + head.2 =
              wordsValue xs + wordMod ^ xs.length * tail.2 := by
            simpa only [tail, hltail] using htail
          have hhead' : head.1 + y + borrow = x + wordMod * head.2 := by
            simpa only [wordMod] using hhead
          refine ⟨?_, by simp only [List.length_cons, hltail'], hctail, ?_⟩
          · intro z hz
            simp only [List.mem_cons] at hz
            rcases hz with rfl | hz
            · exact hbhead
            · exact hbtail' z hz
          · have hfinal : (head.1 + wordMod * wordsValue tail.1) +
                (y + wordMod * wordsValue ys) + borrow =
              x + wordMod * wordsValue xs +
                (wordMod * wordMod ^ xs.length) * tail.2 := by
              calc
                _ = (head.1 + y + borrow) +
                      wordMod * (wordsValue tail.1 + wordsValue ys) := by ring
                _ = (x + wordMod * head.2) +
                      wordMod * (wordsValue tail.1 + wordsValue ys) := by rw [hhead']
                _ = x + wordMod *
                      (wordsValue tail.1 + wordsValue ys + head.2) := by ring
                _ = x + wordMod *
                      (wordsValue xs + wordMod ^ xs.length * tail.2) := by rw [htail']
                _ = x + wordMod * wordsValue xs +
                      (wordMod * wordMod ^ xs.length) * tail.2 := by ring
            simp only [wordsValue, List.length_cons, pow_succ]
            convert hfinal using 1 <;> ring

end PastaCurves.ARMv7EM.ListArithmetic
