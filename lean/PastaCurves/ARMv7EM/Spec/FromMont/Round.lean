import PastaCurves.ARMv7EM.Spec.Mul.Round

/-!
# Arithmetic for one ARMv7E-M Montgomery conversion round

The dedicated conversion block carries nine 32-bit words between rounds. This file embeds that
state in the ten-word rolling accumulator used by the pure multiplication model and proves the
sparse cancellation and shift performed by one round when the modulus low word is one.
-/

namespace PastaCurves.ARMv7EM

/-- Embed the nine words carried by `fromMontRound` in the multiplication accumulator. -/
def FromMontState.toMulAcc (st : FromMontState) : MulAcc :=
  ⟨st.w0, st.w1, st.w2, st.w3, st.w4, st.w5, st.w6, st.w7, st.w8, 0⟩

/-- Project the low nine words of a rolling accumulator into the state carried between rounds. -/
def FromMontState.ofMulAcc (acc : MulAcc) : FromMontState :=
  ⟨acc.w0, acc.w1, acc.w2, acc.w3, acc.w4, acc.w5, acc.w6, acc.w7, acc.w8⟩

/-- The conversion block uses `-a₀` modulo the word radix as its Montgomery quotient. -/
def fromMontQuotient (st : FromMontState) : Nat :=
  (wordMod - st.w0) % wordMod

/-- A bounded nine-word state embeds as a bounded accumulator with zero top word. -/
theorem FromMontState.toMulAcc_bounded (st : FromMontState) (hst : st.Bounded) :
    st.toMulAcc.Bounded ∧ st.toMulAcc.w9 = 0 := by
  rcases hst with ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8⟩
  exact ⟨⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, by
    change 0 < wordMod
    norm_num [wordMod]⟩, rfl⟩

/-- Projecting a bounded accumulator produces a bounded nine-word state. -/
theorem FromMontState.ofMulAcc_bounded (acc : MulAcc) (hacc : acc.Bounded) :
    (FromMontState.ofMulAcc acc).Bounded :=
  ⟨hacc.1, hacc.2.1, hacc.2.2.1, hacc.2.2.2.1, hacc.2.2.2.2.1,
    hacc.2.2.2.2.2.1, hacc.2.2.2.2.2.2.1, hacc.2.2.2.2.2.2.2.1,
    hacc.2.2.2.2.2.2.2.2.1⟩

/-- An exact value equation and bounded low word uniquely identify an `adc` result pair. -/
theorem adc_eq_pair (a b carry low high : Nat) (ha : a < wordMod) (hb : b < wordMod)
    (hcarry : carry < wordMod) (hcarryBit : carry ≤ 1) (hlow : low < wordMod)
    (hvalue : low + wordMod * high = a + b + carry) :
    adc a b carry = (low, high) := by
  obtain ⟨hsum, hsumLow, _⟩ := adc_spec a b carry ha hb hcarry hcarryBit (adc a b carry) rfl
  apply Prod.ext
  · norm_num [wordMod] at hsum hsumLow hlow hvalue ⊢
    omega
  · norm_num [wordMod] at hsum hsumLow hlow hvalue ⊢
    omega

/-- An exact value equation and bounded low word uniquely identify an `addWord` result pair. -/
theorem addWord_eq_pair (a b low high : Nat) (ha : a < wordMod) (hb : b < wordMod)
    (hlow : low < wordMod) (hvalue : low + wordMod * high = a + b) :
    addWord a b = (low, high) := by
  obtain ⟨hsum, hsumLow, _⟩ := addWord_spec a b ha hb (addWord a b) rfl
  apply Prod.ext
  · norm_num [wordMod] at hsum hsumLow hlow hvalue ⊢
    omega
  · norm_num [wordMod] at hsum hsumLow hlow hvalue ⊢
    omega

/-- An exact value equation and bounded low word uniquely identify a `macWord` result pair. -/
theorem macWord_eq_pair (acc a b carry low high : Nat) (hacc : acc < wordMod)
    (ha : a < wordMod) (hb : b < wordMod) (hcarry : carry < wordMod)
    (hlow : low < wordMod) (hvalue : low + wordMod * high = a * b + acc + carry) :
    macWord acc a b carry = (low, high) := by
  obtain ⟨hsum, hsumLow, _⟩ :=
    macWord_spec acc a b carry hacc ha hb hcarry (macWord acc a b carry) rfl
  apply Prod.ext
  · norm_num [wordMod] at hsum hsumLow hlow hvalue ⊢
    omega
  · norm_num [wordMod] at hsum hsumLow hlow hvalue ⊢
    omega

/-- Negating a bounded low word modulo the radix cancels it and remains a word. -/
theorem fromMontQuotient_spec (st : FromMontState) (hst : st.Bounded) :
    fromMontQuotient st < wordMod ∧
      (st.w0 + fromMontQuotient st) % wordMod = 0 := by
  have hword : st.w0 < wordMod := hst.1
  constructor
  · exact Nat.mod_lt _ (by norm_num [wordMod])
  · by_cases hzero : st.w0 = 0
    · simp [fromMontQuotient, hzero]
    · have hpos : 0 < st.w0 := Nat.pos_of_ne_zero hzero
      have hsub : wordMod - st.w0 < wordMod := Nat.sub_lt (by norm_num [wordMod]) hpos
      rw [fromMontQuotient, Nat.mod_eq_of_lt hsub, Nat.add_sub_of_le (Nat.le_of_lt hword),
        Nat.mod_self]

/-- The first subtraction computes the negated low word and its zero-test carry. -/
theorem subc32_zero_sub_one (a : Nat) (ha : a < wordMod) :
    subc32 0 a 1 = ((wordMod - a) % wordMod, if a = 0 then 1 else 0) := by
  unfold subc32
  simp only [Nat.zero_add, Nat.sub_self, Nat.sub_zero]
  apply Prod.ext
  · rfl
  · by_cases hzero : a = 0
    · subst a
      simp only [if_pos, Nat.sub_zero]
      exact Nat.div_self (Nat.two_pow_pos 32)
    · have hpos : 0 < a := Nat.pos_of_ne_zero hzero
      have hlt : wordMod - a < wordMod := Nat.sub_lt (Nat.two_pow_pos 32) hpos
      have hdiv : (wordMod - a) / wordMod = 0 := Nat.div_eq_of_lt hlt
      simpa only [if_neg hzero] using hdiv

/-- Masking the low word of a self-subtraction turns its carry bit into `1` or `0`. -/
theorem subc32_self_and_one (a c : Nat) (hc : c ≤ 1) :
    andw (subc32 a a c).1 1 = if c = 0 then 1 else 0 := by
  obtain hzero | hone : c = 0 ∨ c = 1 := by omega
  · subst c
    rw [if_pos rfl]
    have hlt : wordMod - 1 < wordMod := Nat.sub_lt (Nat.two_pow_pos 32) (by decide)
    have hsub : (subc32 a a 0).1 = wordMod - 1 := by
      change (a + wordMod - a - 1) % wordMod = wordMod - 1
      rw [Nat.add_sub_cancel_left, Nat.mod_eq_of_lt hlt]
    rw [hsub]
    change (wordMod - 1) &&& (2^1 - 1) = 1
    rw [Nat.and_two_pow_sub_one_eq_mod]
    have hpos : 0 < 2^31 := Nat.pow_pos (by decide)
    have heq : wordMod - 1 = 1 + 2 * (2^31 - 1) := by
      rw [show wordMod = 2 * 2^31 by rfl, Nat.mul_sub_left_distrib, Nat.mul_one]
      omega
    rw [heq, Nat.add_mul_mod_self_left]
  · subst c
    rw [if_neg (by decide)]
    have hsub : (subc32 a a 1).1 = 0 := by
      change (a + wordMod - a) % wordMod = 0
      rw [Nat.add_sub_cancel_left, Nat.mod_self]
    rw [hsub]
    rfl

/-- The carry of adding a bounded word to its Montgomery quotient is the zero-test carry. -/
theorem adc_fromMontQuotient_carry (st : FromMontState) (hst : st.Bounded) :
    (adc st.w0 (fromMontQuotient st) 0).2 = if st.w0 = 0 then 0 else 1 := by
  have hquotient := fromMontQuotient_spec st hst
  have hword := hst.1
  obtain ⟨hvalue, hlow, hcarry⟩ :=
    adc_spec st.w0 (fromMontQuotient st) 0 hword hquotient.1
      (Nat.two_pow_pos 32) (by decide) (adc st.w0 (fromMontQuotient st) 0) rfl
  by_cases hzero : st.w0 = 0
  · have hquotient_zero : fromMontQuotient st = 0 := by
      rw [fromMontQuotient, hzero, Nat.sub_zero, Nat.mod_self]
    rw [hzero, hquotient_zero] at hvalue hcarry ⊢
    simp only [Nat.add_zero] at hvalue
    have hcarry_cases : (adc 0 0 0).2 = 0 ∨ (adc 0 0 0).2 = 1 := by omega
    rcases hcarry_cases with hcarry_zero | hcarry_one
    · exact hcarry_zero
    · rw [hcarry_one, Nat.mul_one] at hvalue
      exact (Nat.ne_of_gt (Nat.add_pos_right _ (Nat.two_pow_pos 32)) hvalue).elim
  · have hpos : 0 < st.w0 := Nat.pos_of_ne_zero hzero
    have hquotient_eq : fromMontQuotient st = wordMod - st.w0 := by
      unfold fromMontQuotient
      exact Nat.mod_eq_of_lt (Nat.sub_lt (Nat.two_pow_pos 32) hpos)
    have hsum : st.w0 + fromMontQuotient st + 0 = wordMod := by
      rw [hquotient_eq, Nat.add_zero, Nat.add_sub_of_le (Nat.le_of_lt hword)]
    rw [hsum] at hvalue
    have hcarry_one : (adc st.w0 (fromMontQuotient st) 0).2 = 1 := by
      have hcarry_cases :
          (adc st.w0 (fromMontQuotient st) 0).2 = 0 ∨
            (adc st.w0 (fromMontQuotient st) 0).2 = 1 := by omega
      rcases hcarry_cases with hcarry_zero | hcarry_one
      · rw [hcarry_zero, Nat.mul_zero, Nat.add_zero] at hvalue
        exact (Nat.ne_of_lt hlow hvalue).elim
      · exact hcarry_one
    rw [if_neg hzero]
    exact hcarry_one

/-- The exact generated prefix computes the Montgomery quotient and its carry indicator. -/
theorem fromMontRound_quotient_prefix (st : FromMontState) (hst : st.Bounded) :
    let d := subc32 0 st.w0 1
    let w0 := d.1
    let c := d.2
    let r12 := (subc32 w0 w0 c).1
    let r12 := andw r12 1
    w0 = (wordMod - st.w0) % wordMod ∧
      w0 = fromMontQuotient st ∧
      r12 = (if st.w0 = 0 then 0 else 1) ∧
      r12 = (adc st.w0 (fromMontQuotient st) 0).2 := by
  have hquotient := fromMontQuotient_spec st hst
  have hfirst := subc32_zero_sub_one st.w0 hst.1
  have hfirstLow := congrArg Prod.fst hfirst
  have hfirstCarry := congrArg Prod.snd hfirst
  have hcarry : (subc32 0 st.w0 1).2 ≤ 1 := by
    rw [hfirstCarry]
    by_cases hzero : st.w0 = 0
    · rw [if_pos hzero]
    · rw [if_neg hzero]
      exact Nat.zero_le 1
  have hmasked := subc32_self_and_one (subc32 0 st.w0 1).1
    (subc32 0 st.w0 1).2 hcarry
  have hadc := adc_fromMontQuotient_carry st hst
  clear hquotient hcarry hfirst
  dsimp
  constructor
  · exact hfirstLow
  constructor
  · exact hfirstLow
  constructor
  · rw [hmasked, hfirstCarry]
    by_cases hzero : st.w0 = 0
    · rw [if_pos hzero, if_neg Nat.one_ne_zero, if_pos hzero]
    · rw [if_neg hzero, if_pos rfl, if_neg hzero]
  · rw [hmasked, hfirstCarry]
    by_cases hzero : st.w0 = 0
    · rw [if_pos hzero, if_neg Nat.one_ne_zero, hadc, if_pos hzero]
    · rw [if_neg hzero, if_pos rfl, hadc, if_neg hzero]

/-- The nine result words of the sparse multiply-accumulate chain, given its low-word carry. -/
def fromMontSparseChain (p : FieldWords) (st : FromMontState) (q carry : Nat) : FromMontState :=
  let r1 := macWord st.w1 p.w1 q carry
  let r2 := macWord st.w2 p.w2 q r1.2
  let r3 := macWord st.w3 p.w3 q r2.2
  let r4 := addWord st.w4 r3.2
  let r5 := addWord st.w5 r4.2
  let r6 := addWord st.w6 r5.2
  let r7 := macWord st.w7 p.w7 q r6.2
  let r8 := addWord st.w8 r7.2
  let r9 := addWord 0 r8.2
  ⟨r1.1, r2.1, r3.1, r4.1, r5.1, r6.1, r7.1, r8.1, r9.1⟩

/-- Pair equalities for every word operation identify the explicit result with the sparse chain. -/
theorem fromMontSparseChain_of_eqs (p : FieldWords) (st : FromMontState) (q carry : Nat)
    (w1 w2 w3 w4 w5 w6 w7 w8 w9 c1 c2 c3 c4 c5 c6 c7 : Nat)
    (h1 : macWord st.w1 p.w1 q carry = (w1, c1))
    (h2 : macWord st.w2 p.w2 q c1 = (w2, c2))
    (h3 : macWord st.w3 p.w3 q c2 = (w3, c3))
    (h4 : addWord st.w4 c3 = (w4, c4))
    (h5 : addWord st.w5 c4 = (w5, c5))
    (h6 : addWord st.w6 c5 = (w6, c6))
    (h7 : macWord st.w7 p.w7 q c6 = (w7, c7))
    (h8 : addWord st.w8 c7 = (w8, w9))
    (h9 : addWord 0 w9 = (w9, 0)) :
    ⟨w1, w2, w3, w4, w5, w6, w7, w8, w9⟩ = fromMontSparseChain p st q carry := by
  simp only [fromMontSparseChain, h1, h2, h3, h4, h5, h6, h7, h8, h9]

/-- Once the low-word addition is identified, the sparse chain is exactly the shifted pure model. -/
theorem fromMontSparseChain_eq (p : FieldWords) (st : FromMontState) (q carry : Nat)
    (hlow : adc st.w0 q 0 = (0, carry)) :
    fromMontSparseChain p st q carry =
      FromMontState.ofMulAcc (shiftRound (macSparseModulus st.toMulAcc p q)) := by
  simp only [fromMontSparseChain, FromMontState.toMulAcc, FromMontState.ofMulAcc, shiftRound,
    macSparseModulus, hlow]

/-- One pure sparse conversion round cancels its low word, shifts exactly, and leaves a bounded
nine-word result. -/
theorem fromMontSparseRound_spec (modulus : Limbs) (st : FromMontState)
    (hm : modulus.Bounded) (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hlow : modulus.l0 % wordMod = 1) :
    let p := fieldWordsOfLimbs modulus
    let acc := st.toMulAcc
    let q := fromMontQuotient st
    let reduced := macSparseModulus acc p q
    let out := shiftRound reduced
    st.Bounded →
      reduced.w0 = 0 ∧ out.Bounded ∧ out.w9 = 0 ∧ out.value < wordMod^9 ∧
        wordMod * out.value = acc.value + p.value * q := by
  intro p acc q reduced out hst
  obtain ⟨hacc, haccTop⟩ := st.toMulAcc_bounded hst
  have hp : p.Bounded := fieldWordsOfLimbs_bounded modulus hm
  have hpShape : p.w0 = 1 ∧ p.w4 = 0 ∧ p.w5 = 0 ∧ p.w6 = 0 ∧ p.w7 = 2^30 :=
    fieldWordsOfLimbs_sparse_shape modulus hshape hlow
  obtain ⟨hq, hcancel⟩ := fromMontQuotient_spec st hst
  have haccValue : acc.value < wordMod^9 := MulAcc.value_lt_nine acc hacc haccTop
  have hpValue : p.value < wordMod^8 := p.value_lt hp
  have hpq : p.value * q < wordMod^9 := by
    have hwordPos : 0 < wordMod := by norm_num [wordMod]
    have := Nat.mul_lt_mul_of_pos_left hq (Nat.zero_lt_of_lt hpValue)
    have := Nat.mul_lt_mul_of_pos_right hpValue hwordPos
    nlinarith [pow_succ wordMod 8]
  have hsum : acc.value + p.value * q < wordMod^10 := by
    calc
      acc.value + p.value * q < wordMod^9 + wordMod^9 := Nat.add_lt_add haccValue hpq
      _ ≤ wordMod^9 * wordMod := by
        have : 2 ≤ wordMod := by norm_num [wordMod]
        nlinarith
      _ = wordMod^10 := by ring
  obtain ⟨_, _, hreduced, hreducedValue, _⟩ :=
    macSparseModulus_spec acc p q hacc hp hq hpShape hsum
  have hreducedLow : reduced.w0 = 0 := by
    change (macSparseModulus acc p q).w0 = 0
    simp only [macSparseModulus]
    obtain ⟨hadc, hadcLow, _⟩ := adc_spec acc.w0 q 0 hacc.1 hq
      (by norm_num [wordMod]) (by norm_num) (adc acc.w0 q 0) rfl
    calc
      (adc acc.w0 q 0).1 = (adc acc.w0 q 0).1 % wordMod :=
        (Nat.mod_eq_of_lt hadcLow).symm
      _ = ((adc acc.w0 q 0).1 + wordMod * (adc acc.w0 q 0).2) % wordMod := by simp
      _ = (acc.w0 + q + 0) % wordMod := by rw [hadc]
      _ = 0 := by
        change (st.w0 + fromMontQuotient st + 0) % wordMod = 0
        simpa only [Nat.add_zero] using hcancel
  obtain ⟨hout, houtTop, houtLt, houtValue, _⟩ :=
    shiftRound_spec reduced hreduced hreducedLow
  exact ⟨hreducedLow, hout, houtTop, houtLt, by rw [houtValue, hreducedValue]⟩
