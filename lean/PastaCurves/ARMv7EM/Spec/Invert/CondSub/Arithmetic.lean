import PastaCurves.ARMv7EM.Spec.Arithmetic.List

/-!
# Arithmetic lemmas for the ARMv7E-M conditional subtraction

The generated `condSub` block threads ARM's no-borrow carry through eight 32-bit
subtractions, then uses the complemented carry as an all-zero or all-one mask.  This
module connects that instruction-level form to the list arithmetic model and isolates
the bitwise selection identity used for every output word.
-/

namespace PastaCurves.ARMv7EM

open ListArithmetic

/-- The exact call pattern of a `subc32` chain, threading ARM's no-borrow carry. -/
def subcChain : Nat → List Nat → List Nat → List Nat × Nat
  | carry, x :: xs, y :: ys =>
      let head := subc32 x y carry
      let tail := subcChain head.2 xs ys
      (head.1 :: tail.1, tail.2)
  | carry, _, _ => ([], carry)

/-- One `subc32` with no-borrow carry `carry` is the same subtraction as `sbb` with
borrow `1 - carry`; their output bits are complements. -/
theorem subc32_eq_sbb (a b carry : Nat) (ha : a < wordMod) (hb : b < wordMod)
    (hcarry : carry ≤ 1) :
    let difference := subc32 a b carry
    let result := sbb a b (1 - carry)
    difference.1 = result.1 ∧ difference.2 + result.2 = 1 := by
  let difference := subc32 a b carry
  let result := sbb a b (1 - carry)
  have hdifference : difference.1 + wordMod * difference.2 + b + 1 =
      a + wordMod + carry := by
    dsimp only [difference, subc32]
    exact subc32_lin a b carry hb hcarry
  have hdifferenceBound : difference.1 < wordMod := by
    dsimp only [difference, subc32]
    exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have hdifferenceCarry : difference.2 ≤ 1 := by
    dsimp only [difference, subc32]
    exact subc32_carry_le_one a b carry ha
  have hborrow : 1 - carry ≤ 1 := by omega
  obtain ⟨hresult, hresultBound, hresultBorrow⟩ :=
    sbb_spec a b (1 - carry) ha hb (by omega) hborrow result rfl
  change difference.1 = result.1 ∧ difference.2 + result.2 = 1
  norm_num [wordMod] at hdifference hdifferenceBound hresult hresultBound
  clear * - hdifference hdifferenceBound hdifferenceCarry hresult hresultBound hresultBorrow
    hcarry
  omega

/-- A no-borrow carry chain and the corresponding `sbbChain` produce the same words,
with complementary final carry and borrow bits. -/
theorem subcChain_eq_sbbChain (xs ys : List Nat) (carry : Nat)
    (hlen : xs.length = ys.length) (hxs : WordsBounded xs) (hys : WordsBounded ys)
    (hcarry : carry ≤ 1) :
    let difference := subcChain carry xs ys
    let result := sbbChain (1 - carry) xs ys
    difference.1 = result.1 ∧ difference.2 + result.2 = 1 := by
  induction xs generalizing ys carry with
  | nil =>
      cases ys <;> simp_all [subcChain, sbbChain]
  | cons x xs ih =>
      cases ys with
      | nil => simp at hlen
      | cons y ys =>
          have hx : x < wordMod := hxs x (by simp)
          have hy : y < wordMod := hys y (by simp)
          have hxs' : WordsBounded xs := by
            intro z hz
            exact hxs z (by simp [hz])
          have hys' : WordsBounded ys := by
            intro z hz
            exact hys z (by simp [hz])
          let head := subc32 x y carry
          let result := sbb x y (1 - carry)
          have hhead := subc32_eq_sbb x y carry hx hy hcarry
          change head.1 = result.1 ∧ head.2 + result.2 = 1 at hhead
          have hheadCarry : head.2 ≤ 1 := by
            dsimp only [head]
            exact subc32_carry_le_one x y carry hx
          have htail := ih ys head.2 (by simpa using hlen) hxs' hys' hheadCarry
          change
            (head.1 :: (subcChain head.2 xs ys).1) =
                (result.1 :: (sbbChain result.2 xs ys).1) ∧
              (subcChain head.2 xs ys).2 + (sbbChain result.2 xs ys).2 = 1
          rw [hhead.1]
          have hborrow : result.2 = 1 - head.2 := by omega
          rw [hborrow]
          constructor
          · congr 1
            exact htail.1
          · exact htail.2

/-- Eight bounded 32-bit subtractions, starting with carry set, reconstruct the full
256-bit subtraction.  The final carry is one exactly in the no-borrow representation:
adding it to the wrapped difference balances the extra `2^256` on the input side. -/
theorem subcChain_eight_spec (value modulus difference : List Nat) (finalCarry : Nat)
    (hvalueLength : value.length = 8) (hmodulusLength : modulus.length = 8)
    (hvalue : WordsBounded value) (hmodulus : WordsBounded modulus)
    (hchain : subcChain 1 value modulus = (difference, finalCarry)) :
    WordsBounded difference ∧ difference.length = 8 ∧ finalCarry ≤ 1 ∧
      wordsValue difference + wordsValue modulus + 2^256 * finalCarry =
        wordsValue value + 2^256 := by
  have hlen : value.length = modulus.length := by omega
  have hrelation' := subcChain_eq_sbbChain value modulus 1 hlen hvalue hmodulus (by omega)
  let result := sbbChain 0 value modulus
  have hrelation : difference = result.1 ∧ finalCarry + result.2 = 1 := by
    rw [hchain] at hrelation'
    simpa only [Nat.sub_self] using hrelation'
  have hsbb := sbbChain_spec value modulus 0 hlen hvalue hmodulus (by omega)
  change WordsBounded result.1 ∧ result.1.length = value.length ∧ result.2 ≤ 1 ∧
    wordsValue result.1 + wordsValue modulus + 0 =
      wordsValue value + wordMod ^ value.length * result.2 at hsbb
  rcases hrelation with ⟨rfl, hrelation⟩
  have hfinalCarry : finalCarry ≤ 1 := by omega
  refine ⟨hsbb.1, by simpa [hvalueLength] using hsbb.2.1, hfinalCarry, ?_⟩
  have harithmetic := hsbb.2.2.2
  simp only [Nat.add_zero, hvalueLength, wordMod] at harithmetic
  clear * - harithmetic hrelation
  omega

/-- `sbc keep, keep, keep` turns a no-borrow carry bit into the 32-bit selection mask:
all ones after a borrow, and zero after no borrow. -/
theorem subc32_zero_mask (carry : Nat) (hcarry : carry ≤ 1) :
    (subc32 0 0 carry).1 = if carry = 0 then wordMod - 1 else 0 := by
  obtain hzero | hone : carry = 0 ∨ carry = 1 := by omega
  · simp [hzero, subc32, wordMod]
  · simp [hone, subc32, wordMod]

/-- With the all-ones mask produced by a clear carry, the XOR selector returns the
original value. -/
theorem xorMask_select_zero (value difference : Nat) (hvalue : value < wordMod)
    (hdifference : difference < wordMod) :
    eor32 difference (andw (eor32 value difference) (subc32 0 0 0).1) = value := by
  have hxor : value ^^^ difference < wordMod := by
    exact Nat.xor_lt_two_pow hvalue hdifference
  have hand : andw (eor32 value difference) (wordMod - 1) =
      value ^^^ difference := by
    unfold andw eor32
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_mod, Nat.mod_eq_of_lt hxor]
  rw [subc32_zero_mask 0 (by omega), if_pos rfl, hand]
  unfold eor32
  rw [Nat.xor_comm value difference, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
  exact Nat.mod_eq_of_lt hvalue

/-- With the zero mask produced by a set carry, the XOR selector keeps the subtraction
result. -/
theorem xorMask_select_one (value difference : Nat) (hdifference : difference < wordMod) :
    eor32 difference (andw (eor32 value difference) (subc32 0 0 1).1) = difference := by
  norm_num [subc32, eor32, andw, wordMod, Nat.mod_eq_of_lt hdifference]

/-- The three-instruction XOR-mask selector used by `condSub`: a clear final carry
selects the original value, while a set final carry keeps the subtraction result. -/
theorem xorMask_select (value difference carry : Nat) (hvalue : value < wordMod)
    (hdifference : difference < wordMod) (hcarry : carry ≤ 1) :
    let keep := (subc32 0 0 carry).1
    eor32 difference (andw (eor32 value difference) keep) =
      if carry = 0 then value else difference := by
  obtain hzero | hone : carry = 0 ∨ carry = 1 := by omega
  · subst carry
    change eor32 difference (andw (eor32 value difference) (subc32 0 0 0).1) = value
    exact xorMask_select_zero value difference hvalue hdifference
  · subst carry
    change eor32 difference (andw (eor32 value difference) (subc32 0 0 1).1) = difference
    exact xorMask_select_one value difference hdifference

end PastaCurves.ARMv7EM
