import PastaCurves.ARMv7EM.Spec.Mul.Round

/-!
# The eight multiplication rounds of the ARMv7E-M composition

This file connects `mulRound_spec` to the eight concrete calls made by `mulMont`.  The
32-bit implementation has two accumulator words above the eight-word field value.  Consequently,
the strict ten-word bound required by one round follows just from the preceding round's nine-word
bound and bounded words; it does not require an assumption stronger than `mulContract`.

The modulus shape and Montgomery inverse relation are explicit premises because they describe the
constants specialized by `macSparseModulus`.  The actual multiplication contract is used for the
product bound that puts the final CIOS candidate below twice the modulus.
-/

namespace PastaCurves.ARMv7EM

/-- The zero ten-word accumulator with which `mulMont` starts. -/
def zeroMulAcc : MulAcc :=
  ⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩

/-- Read one of the eight explicit words.  The fallback is unused in the eight rounds. -/
def FieldWords.word (x : FieldWords) : Nat → Nat
  | 0 => x.w0
  | 1 => x.w1
  | 2 => x.w2
  | 3 => x.w3
  | 4 => x.w4
  | 5 => x.w5
  | 6 => x.w6
  | 7 => x.w7
  | _ => 0

/-- The rolling accumulator after `n` calls to `mulRound`. -/
def mulRounds (lhs rhs modulus : FieldWords) (inv : Nat) : Nat → MulAcc
  | 0 => zeroMulAcc
  | n + 1 => mulRound lhs modulus inv (rhs.word n) (mulRounds lhs rhs modulus inv n)

/-- The quotient word selected in round `i`. -/
def roundQuotient (lhs rhs modulus : FieldWords) (inv i : Nat) : Nat :=
  (macProduct (mulRounds lhs rhs modulus inv i) lhs (rhs.word i)).w0 * inv % wordMod

/-- The part of `mulMont` after its eight rolling rounds. -/
def finishMulMont (acc : MulAcc) (modulus : FieldWords) : Limbs :=
  let candidate :=
    FieldWords.toArray ⟨acc.w0, acc.w1, acc.w2, acc.w3, acc.w4, acc.w5, acc.w6, acc.w7⟩
  let high := acc.w8
  let highNeg := (wordMod - high) % wordMod
  let highNonzero := (high ||| highNeg) / 2^31
  limbsOfWordsArray (selectReduced candidate modulus.toArray highNonzero)

/-- The indexed model unfolds to exactly the eight calls and epilogue in `Compositions.mulMont`. -/
theorem mulMont_eq_finishMulMont_mulRounds (lhs rhs modulus : Limbs) (inv : Nat) :
    mulMont lhs rhs modulus inv =
      finishMulMont
        (mulRounds (fieldWordsOfLimbs lhs) (fieldWordsOfLimbs rhs)
          (fieldWordsOfLimbs modulus) (inv % wordMod) 8)
        (fieldWordsOfLimbs modulus) := by
  rfl

/-- Every in-range projection of bounded field words is a word. -/
theorem FieldWords.word_lt (x : FieldWords) (hx : x.Bounded) (i : Nat) (hi : i < 8) :
    x.word i < wordMod := by
  interval_cases i <;> simp_all [FieldWords.word, FieldWords.Bounded]

/-- The eight projected words reassemble to the value of the explicit field words. -/
theorem prefixValue_word_eight (x : FieldWords) :
    prefixValue x.word 8 = x.value := by
  simp only [prefixValue, FieldWords.word, FieldWords.value, ListArithmetic.wordsValue]
  ring

/-- The quotient selected by every round is a word. -/
theorem roundQuotient_lt (lhs rhs modulus : FieldWords) (inv i : Nat) :
    roundQuotient lhs rhs modulus inv i < wordMod := by
  exact Nat.mod_lt _ (by norm_num [wordMod])

/-- The strict bound needed by `mulRound_spec` follows from the ordinary word bounds.  In
particular, no branch of `mulContract` needs to be strengthened for the 32-bit rolling schedule. -/
theorem mulRound_total_lt (acc : MulAcc) (lhs modulus : FieldWords) (b q : Nat)
    (hacc : acc.Bounded) (haccTop : acc.w9 = 0)
    (hlhs : lhs.Bounded) (hmodulus : modulus.Bounded)
    (hb : b < wordMod) (hq : q < wordMod) :
    acc.value + lhs.value * b + modulus.value * q < wordMod^10 := by
  have ha : acc.value < wordMod^9 := acc.value_lt_nine hacc haccTop
  have hl : lhs.value < wordMod^8 := lhs.value_lt hlhs
  have hp : modulus.value < wordMod^8 := modulus.value_lt hmodulus
  have hlb : lhs.value * b < wordMod^9 := by
    calc
      lhs.value * b < wordMod^8 * wordMod := Nat.mul_lt_mul'' hl hb
      _ = wordMod^9 := by ring
  have hpq : modulus.value * q < wordMod^9 := by
    calc
      modulus.value * q < wordMod^8 * wordMod := Nat.mul_lt_mul'' hp hq
      _ = wordMod^9 := by ring
  have hradix : 3 * wordMod^9 < wordMod^10 := by norm_num [wordMod]
  omega

/-- Every prefix of the eight concrete rounds is bounded, has a cleared top word, and represents
less than nine words. -/
theorem mulRounds_state (lhs rhs modulus : FieldWords) (inv : Nat)
    (hlhs : lhs.Bounded) (hrhs : rhs.Bounded) (hmodulus : modulus.Bounded)
    (hshape : modulus.w0 = 1 ∧ modulus.w4 = 0 ∧ modulus.w5 = 0 ∧
      modulus.w6 = 0 ∧ modulus.w7 = 2^30)
    (hinv : (inv * modulus.w0 + 1) % wordMod = 0) :
    ∀ n ≤ 8,
      (mulRounds lhs rhs modulus inv n).Bounded ∧
      (mulRounds lhs rhs modulus inv n).w9 = 0 ∧
      (mulRounds lhs rhs modulus inv n).value < wordMod^9 := by
  intro n hn
  induction n with
  | zero =>
      simp [mulRounds, zeroMulAcc, MulAcc.Bounded, MulAcc.value,
        ListArithmetic.wordsValue, wordMod]
  | succ n ih =>
      have hn8 : n < 8 := by omega
      obtain ⟨hacc, haccTop, _⟩ := ih (by omega)
      have hb := rhs.word_lt hrhs n hn8
      have hq := roundQuotient_lt lhs rhs modulus inv n
      have htotal :
          (mulRounds lhs rhs modulus inv n).value + lhs.value * rhs.word n +
              modulus.value * roundQuotient lhs rhs modulus inv n < wordMod^10 :=
        mulRound_total_lt (mulRounds lhs rhs modulus inv n) lhs modulus
          (rhs.word n) (roundQuotient lhs rhs modulus inv n)
          hacc haccTop hlhs hmodulus hb hq
      obtain ⟨hout, htop, hlt, _, _⟩ :=
        mulRound_spec lhs modulus inv (rhs.word n) (mulRounds lhs rhs modulus inv n)
          hacc haccTop hlhs hb hmodulus hshape hinv htotal
      simpa only [mulRounds] using And.intro hout (And.intro htop hlt)

/-- One indexed step is the exact value equation proved for the corresponding concrete
`mulRound` call. -/
theorem mulRounds_step_spec (lhs rhs modulus : FieldWords) (inv i : Nat)
    (hlhs : lhs.Bounded) (hrhs : rhs.Bounded) (hmodulus : modulus.Bounded)
    (hshape : modulus.w0 = 1 ∧ modulus.w4 = 0 ∧ modulus.w5 = 0 ∧
      modulus.w6 = 0 ∧ modulus.w7 = 2^30)
    (hinv : (inv * modulus.w0 + 1) % wordMod = 0) (hi : i < 8) :
    wordMod * (mulRounds lhs rhs modulus inv (i + 1)).value =
      (mulRounds lhs rhs modulus inv i).value + lhs.value * rhs.word i +
        modulus.value * roundQuotient lhs rhs modulus inv i := by
  obtain ⟨hacc, haccTop, _⟩ :=
    mulRounds_state lhs rhs modulus inv hlhs hrhs hmodulus hshape hinv i (by omega)
  have hb := rhs.word_lt hrhs i hi
  have hq := roundQuotient_lt lhs rhs modulus inv i
  have htotal :
      (mulRounds lhs rhs modulus inv i).value + lhs.value * rhs.word i +
          modulus.value * roundQuotient lhs rhs modulus inv i < wordMod^10 :=
    mulRound_total_lt (mulRounds lhs rhs modulus inv i) lhs modulus
      (rhs.word i) (roundQuotient lhs rhs modulus inv i)
      hacc haccTop hlhs hmodulus hb hq
  obtain ⟨_, _, _, hvalue, _⟩ :=
    mulRound_spec lhs modulus inv (rhs.word i) (mulRounds lhs rhs modulus inv i)
      hacc haccTop hlhs hb hmodulus hshape hinv htotal
  simpa only [mulRounds, roundQuotient] using hvalue

/-- The eight concrete round equations telescope to the CIOS invariant. -/
theorem mulRounds_eight_invariant (lhs rhs modulus : FieldWords) (inv : Nat)
    (hlhs : lhs.Bounded) (hrhs : rhs.Bounded) (hmodulus : modulus.Bounded)
    (hshape : modulus.w0 = 1 ∧ modulus.w4 = 0 ∧ modulus.w5 = 0 ∧
      modulus.w6 = 0 ∧ modulus.w7 = 2^30)
    (hinv : (inv * modulus.w0 + 1) % wordMod = 0) :
    wordMod^8 * (mulRounds lhs rhs modulus inv 8).value =
      lhs.value * rhs.value +
        modulus.value * prefixValue (roundQuotient lhs rhs modulus inv) 8 := by
  have h := cios_eight_round_invariant lhs.value modulus.value rhs.word
    (roundQuotient lhs rhs modulus inv)
    (fun i => (mulRounds lhs rhs modulus inv i).value) (by
      intro i hi
      exact mulRounds_step_spec lhs rhs modulus inv i hlhs hrhs hmodulus hshape hinv hi)
  rw [prefixValue_word_eight] at h
  have hzero : (mulRounds lhs rhs modulus inv 0).value = 0 := by
    change ListArithmetic.wordsValue [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = 0
    rfl
  rw [hzero, Nat.zero_add] at h
  exact h

/-- A value below the eight-word radix has zero in each higher accumulator word. -/
theorem MulAcc.w8_eq_zero_of_value_lt (acc : MulAcc) (hvalue : acc.value < wordMod^8) :
    acc.w8 = 0 := by
  have hdecomp : acc.value =
      acc.w0 + wordMod * acc.w1 + wordMod^2 * acc.w2 + wordMod^3 * acc.w3 +
        wordMod^4 * acc.w4 + wordMod^5 * acc.w5 + wordMod^6 * acc.w6 +
        wordMod^7 * acc.w7 + wordMod^8 * acc.w8 + wordMod^9 * acc.w9 := by
    simp only [MulAcc.value, ListArithmetic.wordsValue]
    ring
  have hterm : wordMod^8 * acc.w8 ≤ acc.value := by
    rw [hdecomp]
    omega
  by_contra hn
  have hone : 1 ≤ acc.w8 := Nat.one_le_iff_ne_zero.mpr hn
  have hradix : wordMod^8 ≤ wordMod^8 * acc.w8 := by
    exact Nat.mul_le_mul_left (wordMod^8) hone
  omega

/-- The eight calls inside `mulMont` satisfy the Montgomery invariant.  The real Rust multiplication
contract supplies the final product bound, so the complete candidate is below `2p`; the Pasta
modulus bound then proves that both words above the eight-word candidate are zero. -/
theorem mulMont_eight_rounds_spec (lhs rhs modulus : Limbs) (inv : Nat)
    (hlhs : lhs.Bounded) (hrhs : rhs.Bounded) (hmodulus : modulus.Bounded)
    (hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62)
    (hlow : modulus.l0 % wordMod = 1)
    (hinv : (inv * modulus.l0 + 1) % 2^64 = 0)
    (hcontract : mulContract lhs rhs modulus = true) :
    let lhsWords := fieldWordsOfLimbs lhs
    let rhsWords := fieldWordsOfLimbs rhs
    let modulusWords := fieldWordsOfLimbs modulus
    let wordInv := inv % wordMod
    let acc := mulRounds lhsWords rhsWords modulusWords wordInv 8
    let quotient := prefixValue (roundQuotient lhsWords rhsWords modulusWords wordInv) 8
    acc.Bounded ∧ acc.w9 = 0 ∧ acc.w8 = 0 ∧
      quotient < wordMod^8 ∧
      wordMod^8 * acc.value = lhs.toNat * rhs.toNat + modulus.toNat * quotient ∧
      acc.value < 2 * modulus.toNat ∧
      mulMont lhs rhs modulus inv = finishMulMont acc modulusWords := by
  dsimp only
  have hlhsWords := fieldWordsOfLimbs_bounded lhs hlhs
  have hrhsWords := fieldWordsOfLimbs_bounded rhs hrhs
  have hmodulusWords := fieldWordsOfLimbs_bounded modulus hmodulus
  have hwordShape := fieldWordsOfLimbs_sparse_shape modulus hshape hlow
  have hwordInv :
      ((inv % wordMod) * (fieldWordsOfLimbs modulus).w0 + 1) % wordMod = 0 := by
    simpa [fieldWordsOfLimbs] using inverse_relation_word inv modulus.l0 hinv
  obtain ⟨hacc, haccTop, _⟩ :=
    mulRounds_state (fieldWordsOfLimbs lhs) (fieldWordsOfLimbs rhs)
      (fieldWordsOfLimbs modulus) (inv % wordMod)
      hlhsWords hrhsWords hmodulusWords hwordShape hwordInv 8 (by omega)
  have hmain :=
    mulRounds_eight_invariant (fieldWordsOfLimbs lhs) (fieldWordsOfLimbs rhs)
      (fieldWordsOfLimbs modulus) (inv % wordMod)
      hlhsWords hrhsWords hmodulusWords hwordShape hwordInv
  rw [fieldWordsOfLimbs_value lhs hlhs, fieldWordsOfLimbs_value rhs hrhs,
    fieldWordsOfLimbs_value modulus hmodulus] at hmain
  have hquotient :
      prefixValue
          (roundQuotient (fieldWordsOfLimbs lhs) (fieldWordsOfLimbs rhs)
            (fieldWordsOfLimbs modulus) (inv % wordMod)) 8 < wordMod^8 := by
    apply prefixValue_eight_lt
    intro i hi
    exact roundQuotient_lt _ _ _ _ i
  have hproduct := mulContract_product_lt lhs rhs modulus hlhs hrhs hmodulus hcontract
  have hproduct' : lhs.toNat * rhs.toNat < wordMod^8 * modulus.toNat := by
    norm_num [wordMod] at hproduct ⊢
    exact hproduct
  have hmodulusPos : 0 < modulus.toNat := by
    simp only [Limbs.toNat, hshape.2]
    omega
  have hcandidate :
      (mulRounds (fieldWordsOfLimbs lhs) (fieldWordsOfLimbs rhs)
        (fieldWordsOfLimbs modulus) (inv % wordMod) 8).value < 2 * modulus.toNat :=
    cios_candidate_lt_two_mul (wordMod^8)
      (mulRounds (fieldWordsOfLimbs lhs) (fieldWordsOfLimbs rhs)
        (fieldWordsOfLimbs modulus) (inv % wordMod) 8).value
      (lhs.toNat * rhs.toNat)
      (prefixValue
        (roundQuotient (fieldWordsOfLimbs lhs) (fieldWordsOfLimbs rhs)
          (fieldWordsOfLimbs modulus) (inv % wordMod)) 8)
      modulus.toNat (by norm_num [wordMod]) hmodulusPos hmain hproduct' hquotient
  have hpUpper := Limbs.toNat_lt_of_shape modulus hmodulus hshape
  have hcandidateRadix :
      (mulRounds (fieldWordsOfLimbs lhs) (fieldWordsOfLimbs rhs)
        (fieldWordsOfLimbs modulus) (inv % wordMod) 8).value < wordMod^8 := by
    norm_num [wordMod] at hcandidate hpUpper ⊢
    omega
  have haccHigh := MulAcc.w8_eq_zero_of_value_lt _ hcandidateRadix
  refine ⟨hacc, haccTop, haccHigh, hquotient, hmain, hcandidate, ?_⟩
  exact mulMont_eq_finishMulMont_mulRounds lhs rhs modulus inv

end PastaCurves.ARMv7EM
