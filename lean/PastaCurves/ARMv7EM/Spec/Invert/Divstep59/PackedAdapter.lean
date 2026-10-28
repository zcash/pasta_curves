/-
Copyright (c) 2026 the pasta_curves contributors.
-/
import PastaCurves.ARMv7EM.Spec.Invert.Divstep59.Control
import PastaCurves.Inversion.PackedWords

/-!
# The ARMv7E-M packed-divstep state adapter

This module presents the generated `BatchState` and `batchRound` as the word-level state expected
by `Inversion.PackedStep`. ARMv7E-M keeps each packed 64-bit value in two 32-bit registers, and
keeps `two_delta` in one 32-bit register. The adapter therefore joins the packed words and
sign-extends `d`. Unlike the AArch64 step, this implementation has no carried parity flag, so its
`z` value is read directly from the low bit of `pglo`.

The structural facts here prepare the signed-width step contract. The checked `batchRound` proof
in `Round.lean` establishes `packedStep.WidthSpec 31`, which the batch proof iterates.
-/

namespace PastaCurves.ARMv7EM

open Inversion (PackedStep)

/-- The bounded words carried between packed rounds, including the public 32-bit loop count. -/
structure PackedBounded (st : BatchState) : Prop where
  pflo : st.pflo < wordMod
  pfhi : st.pfhi < wordMod
  pglo : st.pglo < wordMod
  pghi : st.pghi < wordMod
  d : st.d < wordMod
  remaining : st.remaining < wordMod

/-- The ARMv7E-M batch round as a shared packed step. The generated loop has one round body, so the
ordinary and last steps are the same function. -/
def packedStep : PackedStep where
  Words := BatchState
  Bounded := PackedBounded
  two_delta := fun st => signExtend32 st.d
  f := fun st => join32 st.pflo st.pfhi
  g := fun st => join32 st.pglo st.pghi
  z := fun st => if st.pglo % 2 = 0 then 1 else 0
  round := batchRound
  last := batchRound

/-- Joining bounded 32-bit words produces a bounded 64-bit word. -/
theorem join32_lt (low high : Nat) (hlow : low < wordMod) (hhigh : high < wordMod) :
    join32 low high < 2^64 := by
  norm_num [join32, wordMod] at hlow hhigh ⊢
  omega

/-- The low half of a joined word is the original low word. -/
theorem join32_mod_wordMod (low high : Nat) (hlow : low < wordMod) :
    join32 low high % wordMod = low := by
  simp [join32, Nat.add_mod, Nat.mod_eq_of_lt hlow]

/-- The high half of a joined word is the original high word. -/
theorem join32_div_wordMod (low high : Nat) (hlow : low < wordMod) :
    join32 low high / wordMod = high := by
  unfold join32
  rw [Nat.add_mul_div_left _ _ (by norm_num [wordMod]), Nat.div_eq_of_lt hlow, zero_add]

/-- Joining a high 32-bit word does not change the low word's parity. -/
theorem join32_mod_two (low high : Nat) : join32 low high % 2 = low % 2 := by
  unfold join32
  rw [show wordMod * high = 2 * (2^31 * high) by norm_num [wordMod]; ring,
    Nat.add_mul_mod_self_left]

/-- Sign-extending a bounded 32-bit word produces a bounded 64-bit word. -/
theorem signExtend32_lt (a : Nat) (ha : a < wordMod) : signExtend32 a < 2^64 := by
  unfold signExtend32
  split <;> norm_num [wordMod] at ha ⊢ <;> omega

/-- Two chained 32-bit additions implement addition modulo `2^64` after joining limbs. -/
theorem join32_add_of_eq (aLo aHi bLo bHi cin lo carry hi overflow : Nat)
    (hlo : lo + wordMod * carry = aLo + bLo + cin)
    (hhi : hi + wordMod * overflow = aHi + bHi + carry)
    (hloBound : lo < wordMod) (hhiBound : hi < wordMod) :
    join32 lo hi = (join32 aLo aHi + join32 bLo bHi + cin) % 2^64 := by
  have hout : join32 lo hi < 2^64 := join32_lt lo hi hloBound hhiBound
  have htotal : join32 lo hi + 2^64 * overflow =
      join32 aLo aHi + join32 bLo bHi + cin := by
    unfold join32 at *
    norm_num [wordMod] at *
    omega
  rw [← htotal, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hout]

/-- A bounded sign-extended word is the low 32-bit representation of a small signed integer. -/
theorem signExtend32_small (d : Nat) (D : Int)
    (hD : |D| + 2 < 2^31) (hd : (signExtend32 d : Int) = D % 2^64) :
    (d : Int) = if 0 ≤ D then D else D + 2^32 := by
  have hAbs : |D| < 2^31 := by omega
  rw [abs_lt] at hAbs
  have hmodPos (h : 0 ≤ D) : D % 2^64 = D :=
    Int.emod_eq_of_lt h (by omega)
  have hmodNeg (h : D < 0) : D % 2^64 = D + 2^64 := by
    calc
      D % 2^64 = (D + 2^64 * 1) % 2^64 :=
        (Int.add_mul_emod_self_left D (2^64) 1).symm
      _ = D + 2^64 := Int.emod_eq_of_lt (by omega) (by omega)
  unfold signExtend32 at hd
  split at hd
  · split
    · rw [hmodPos ‹0 ≤ D›] at hd
      exact hd
    · rw [hmodNeg (by omega)] at hd
      norm_num [wordMod] at hd
      omega
  · split
    · rw [hmodPos ‹0 ≤ D›] at hd
      norm_num [wordMod] at hd
      omega
    · rw [hmodNeg (by omega)] at hd
      norm_num [wordMod] at hd
      omega

/-- Adding two modulo 32 bits and sign-extending agrees with adding two modulo 64 bits when the
signed input and output remain within the 32-bit range. -/
theorem signExtend32_add_two (d out : Nat) (D : Int)
    (hout : out = (d + 2) % wordMod) (hD : |D| + 2 < 2^31)
    (hword : (d : Int) = if 0 ≤ D then D else D + 2^32) :
    signExtend32 out = addw (signExtend32 d) 2 := by
  have hAbs : |D| < 2^31 := by omega
  have hBounds := abs_lt.mp hAbs
  by_cases hnonneg : 0 ≤ D
  · rw [if_pos hnonneg] at hword
    rw [abs_of_nonneg hnonneg] at hD
    have hdSmall : d + 2 < 2^31 := by omega
    rw [hout, Nat.mod_eq_of_lt (by norm_num [wordMod]; omega : d + 2 < wordMod)]
    unfold signExtend32 addw
    rw [if_pos (by omega), if_pos (by omega)]
    norm_num [regMod]
    omega
  · have hneg : D < 0 := lt_of_not_ge hnonneg
    rw [if_neg hnonneg] at hword
    rw [abs_of_neg hneg] at hD
    norm_num [wordMod] at hword
    by_cases hstill : D + 2 < 0
    · have hdLarge : 2^31 ≤ d := by omega
      have hdSumLarge : 2^31 ≤ d + 2 := by omega
      have hdSumSmall : d + 2 < wordMod := by norm_num [wordMod]; omega
      rw [hout, Nat.mod_eq_of_lt hdSumSmall]
      unfold signExtend32 addw
      rw [if_neg (by omega), if_neg (by omega)]
      norm_num [regMod, wordMod]
      omega
    · have hboundary : D = -2 ∨ D = -1 := by omega
      rcases hboundary with rfl | rfl
      · have hdval : d = wordMod - 2 := by norm_num [wordMod] at hword ⊢; omega
        simp [hout, hdval, signExtend32, addw, wordMod, regMod]
      · have hdval : d = wordMod - 1 := by norm_num [wordMod] at hword ⊢; omega
        simp [hout, hdval, signExtend32, addw, wordMod, regMod]

/-- For a positive signed 32-bit input, the wrapped word `2 - d` represents the 64-bit word
`-d + 2`. Keeping this calculation separate avoids exposing the full round trace to arithmetic
normalization. -/
theorem signExtend32_two_sub (d out : Nat) (hpositive : 1 ≤ d) (hd : d < 2^31)
    (hout : out = (2 + wordMod - d) % wordMod) :
    signExtend32 out = addw (negw (signExtend32 d)) 2 := by
  by_cases hsmall : d ≤ 2
  · have hdCases : d = 1 ∨ d = 2 := by omega
    rcases hdCases with rfl | rfl <;>
      norm_num [hout, signExtend32, addw, negw, wordMod, regMod]
  · have houtVal : (2 + wordMod - d) % wordMod = wordMod + 2 - d := by
      apply Nat.mod_eq_of_lt
      norm_num [wordMod] at hd ⊢
      omega
    rw [hout, houtVal]
    unfold signExtend32 negw addw
    rw [if_neg (by norm_num [wordMod] at hd ⊢; omega), if_pos hd]
    have hneg : (2^64 - d) % 2^64 = 2^64 - d := Nat.mod_eq_of_lt (by omega)
    have haddBound : 2^64 - d + 2 < 2^64 := by
      norm_num at hsmall hd ⊢
      omega
    rw [hneg, Nat.mod_eq_of_lt haddBound]
    norm_num [wordMod] at hsmall hd ⊢
    omega

/-- XOR with the all-ones word computes the finite-width complement. -/
theorem eor32_allOnes (a : Nat) (ha : a < wordMod) :
    eor32 a (wordMod - 1) = wordMod - 1 - a := by
  unfold eor32
  have hxor : a ^^^ (2^32 - 1) < 2^32 :=
    Nat.xor_lt_two_pow ha (by omega)
  rw [show wordMod - 1 = 2^32 - 1 by rfl, Nat.mod_eq_of_lt hxor]
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_xor, Nat.testBit_two_pow_sub_one]
  have hsub : 2^32 - 1 - a = 2^32 - (a + 1) := by omega
  rw [hsub, Nat.testBit_two_pow_sub_succ ha]
  by_cases hi : i < 32
  · simp [hi]
  · have haiBound : a < 2^i :=
      lt_of_lt_of_le ha (Nat.pow_le_pow_right (by decide) (by omega))
    have hai : a.testBit i = false := Nat.testBit_lt_two_pow haiBound
    simp [hi, hai]

/-- Adding one to the two-limb complement computes 64-bit two's-complement negation. -/
theorem join32_negate (lo hi outLo carry outHi overflow : Nat)
    (hlo : lo < wordMod) (hhi : hi < wordMod)
    (houtLo : outLo + wordMod * carry = (wordMod - 1 - lo) + 1)
    (houtHi : outHi + wordMod * overflow = (wordMod - 1 - hi) + carry)
    (boutLo : outLo < wordMod) (boutHi : outHi < wordMod) :
    join32 outLo outHi = negw (join32 lo hi) := by
  have hjoin := join32_add_of_eq (wordMod - 1 - lo) (wordMod - 1 - hi)
    0 0 1 outLo carry outHi overflow houtLo houtHi boutLo boutHi
  unfold join32 negw regMod at *
  norm_num [wordMod] at *
  rw [hjoin]
  have hbound : lo + 2^32 * hi < 2^64 := by omega
  omega

/-- Two bounded complementary 32-bit words XOR to the all-ones word. -/
theorem eor32_complements (a b : Nat) (ha : a < wordMod) (_hb : b < wordMod)
    (hsum : a + b = wordMod - 1) : eor32 a b = wordMod - 1 := by
  have hcomp : b = wordMod - 1 - a := by omega
  subst b
  unfold eor32
  have hxor : a ^^^ (wordMod - 1 - a) = wordMod - 1 := by
    apply Nat.eq_of_testBit_eq
    intro i
    rw [Nat.testBit_xor, show wordMod - 1 - a = 2^32 - (a + 1) by
      norm_num [wordMod]; omega, Nat.testBit_two_pow_sub_succ ha,
      Nat.testBit_two_pow_sub_one]
    by_cases hi : i < 32
    · simp [hi]
    · have haiBound : a < 2^i :=
        lt_of_lt_of_le ha (Nat.pow_le_pow_right (by decide) (by omega))
      have hai : a.testBit i = false := Nat.testBit_lt_two_pow haiBound
      simp [hi, hai]
  rw [hxor]
  norm_num [wordMod]

/-- A generated 32-bit exclusive-or result is bounded. -/
theorem eor32_lt (a b : Nat) : eor32 a b < wordMod := by
  exact Nat.mod_lt _ (by norm_num [wordMod])

/-- A generated 32-bit or result is bounded. -/
theorem orr32_lt (a b : Nat) : orr32 a b < wordMod := by
  exact Nat.mod_lt _ (by norm_num [wordMod])

/-- Arithmetic shift right by one preserves the 32-bit word bound. -/
theorem asr32_one_lt (a : Nat) (ha : a < wordMod) : asr32 a 1 < wordMod := by
  unfold asr32
  split <;> norm_num [wordMod] at ha ⊢ <;> omega

/-- The sign-fill bit added by a 64-bit arithmetic shift is even, so shifting right by one has the
same parity as ordinary division by two. -/
theorem asr_one_mod_two (a : Nat) : asr a 1 % 2 = a / 2 % 2 := by
  unfold asr
  split <;> norm_num [regMod, Nat.add_mod]

/-- Masking by one reads the low bit. -/
theorem andw_one_eq_mod_two (a : Nat) : andw a 1 = a % 2 := by
  unfold andw
  rw [show (1 : Nat) = 2^1 - 1 by norm_num, Nat.and_two_pow_sub_one_eq_mod]

/-- A zero or all-ones 32-bit mask selects zero or a bounded word. -/
theorem andw_mask_eq (a mask : Nat) (ha : a < wordMod)
    (hmask : mask = 0 ∨ mask = wordMod - 1) :
    andw a mask = if mask = 0 then 0 else a := by
  rcases hmask with rfl | rfl
  · simp [andw]
  · have hnonzero : wordMod - 1 ≠ 0 := by norm_num [wordMod]
    rw [if_neg hnonzero]
    unfold andw
    rw [show wordMod - 1 = 2^32 - 1 by rfl, Nat.and_two_pow_sub_one_eq_mod,
      Nat.mod_eq_of_lt ha]

/-- Applying the same zero or all-ones mask to both halves selects zero or the joined word. -/
theorem join32_and_mask (low high mask : Nat) (hlow : low < wordMod)
    (hhigh : high < wordMod) (hmask : mask = 0 ∨ mask = wordMod - 1) :
    join32 (andw low mask) (andw high mask) =
      if mask = 0 then 0 else join32 low high := by
  rw [andw_mask_eq low mask hlow hmask, andw_mask_eq high mask hhigh hmask]
  split <;> simp_all [join32]

/-- The XOR-mask selector used by the generated conditional swap selects its first or second
bounded word. -/
theorem eor32_mask_select (x y mask : Nat) (hx : x < wordMod) (hy : y < wordMod)
    (hmask : mask = 0 ∨ mask = wordMod - 1) :
    eor32 x (andw (eor32 x y) mask) = if mask = 0 then x else y := by
  rw [andw_mask_eq (eor32 x y) mask (eor32_lt x y) hmask]
  rcases hmask with rfl | rfl
  · simp [eor32, Nat.mod_eq_of_lt hx]
  · have hnonzero : wordMod - 1 ≠ 0 := by norm_num [wordMod]
    simp only [if_neg hnonzero]
    unfold eor32
    have hxy : x ^^^ y < wordMod := Nat.xor_lt_two_pow hx hy
    rw [Nat.mod_eq_of_lt hxy, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor,
      Nat.mod_eq_of_lt hy]

/-- The three generated equations for one half of the conditional swap select the first or second
bounded word according to the zero or all-ones mask. -/
theorem eor32_mask_select_of_eq (x y mask xy masked out : Nat) (hx : x < wordMod)
    (hy : y < wordMod) (hmask : mask = 0 ∨ mask = wordMod - 1)
    (hxy : xy = eor32 x y) (hmasked : masked = andw xy mask)
    (hout : out = eor32 x masked) : out = if mask = 0 then x else y := by
  rw [hout, hmasked, hxy]
  exact eor32_mask_select x y mask hx hy hmask

/-- The three equations of the generated `d` update select `d + 2` or `2 - d` according to the
swap mask. -/
theorem two_delta_mask_select (d plus twoSub mask x masked out : Nat)
    (hplus : plus = (d + 2) % wordMod) (htwoSub : twoSub = (2 + wordMod - d) % wordMod)
    (hmask : mask = 0 ∨ mask = wordMod - 1) (hx : x = eor32 twoSub plus)
    (hmasked : masked = andw x mask) (hout : out = eor32 plus masked) :
    out = if mask = 0 then plus else twoSub := by
  have hplusBound : plus < wordMod := by
    rw [hplus]
    exact Nat.mod_lt _ (by norm_num [wordMod])
  have htwoSubBound : twoSub < wordMod := by
    rw [htwoSub]
    exact Nat.mod_lt _ (by norm_num [wordMod])
  exact eor32_mask_select_of_eq plus twoSub mask x masked out hplusBound htwoSubBound hmask
    (by rw [hx]; unfold eor32; rw [Nat.xor_comm]) hmasked hout

/-- Applying the generated XOR-mask selector to both halves selects one of two joined words. -/
theorem join32_eor_mask_select (xlo xhi ylo yhi mask : Nat)
    (hxlo : xlo < wordMod) (hxhi : xhi < wordMod) (hylo : ylo < wordMod)
    (hyhi : yhi < wordMod) (hmask : mask = 0 ∨ mask = wordMod - 1) :
    join32 (eor32 xlo (andw (eor32 xlo ylo) mask))
        (eor32 xhi (andw (eor32 xhi yhi) mask)) =
      if mask = 0 then join32 xlo xhi else join32 ylo yhi := by
  rw [eor32_mask_select xlo ylo mask hxlo hylo hmask,
    eor32_mask_select xhi yhi mask hxhi hyhi hmask]
  split <;> rfl

/-- Negating the low bit as a 32-bit word gives the all-zero or all-one mask used by
`batchRound`. -/
theorem negated_parity_mask (a : Nat) :
    (wordMod - andw a 1) % wordMod = if a % 2 = 0 then 0 else wordMod - 1 := by
  rw [andw_one_eq_mod_two]
  by_cases h : a % 2 = 0
  · simp [h, wordMod]
  · have hlt : a % 2 < 2 := Nat.mod_lt _ (by decide)
    have h1 : a % 2 = 1 := by omega
    simp [h1, wordMod]

@[simp] theorem packedStep_bounded (st : BatchState) :
    packedStep.Bounded st ↔ PackedBounded st := Iff.rfl

@[simp] theorem packedStep_two_delta (st : BatchState) :
    packedStep.two_delta st = signExtend32 st.d := rfl

@[simp] theorem packedStep_f (st : BatchState) :
    packedStep.f st = join32 st.pflo st.pfhi := rfl

@[simp] theorem packedStep_g (st : BatchState) :
    packedStep.g st = join32 st.pglo st.pghi := rfl

@[simp] theorem packedStep_z (st : BatchState) :
    packedStep.z st = if st.pglo % 2 = 0 then 1 else 0 := rfl

@[simp] theorem packedStep_round (st : BatchState) : packedStep.round st = batchRound st := rfl

@[simp] theorem packedStep_last (st : BatchState) : packedStep.last st = batchRound st := rfl

/-- The generated round preserves the five register-word bounds. This is only a width fact; it does
not assert the arithmetic meaning of the round. -/
theorem batchRound_bounded (st : BatchState) (_hst : st.Bounded) :
    (batchRound st).Bounded := by
  simp only [batchRound]
  exact ⟨eor32_lt _ _, eor32_lt _ _, orr32_lt _ _,
    asr32_one_lt _ (Nat.mod_lt _ (by norm_num [wordMod])), eor32_lt _ _⟩

/-- The generated round preserves the adapter's word bounds, including its arbitrary 32-bit
public count. -/
theorem packedStep_round_bounded (st : BatchState) (hst : packedStep.Bounded st) :
    packedStep.Bounded (batchRound st) := by
  have hwords : st.Bounded := ⟨hst.pflo, hst.pfhi, hst.pglo, hst.pghi, hst.d⟩
  obtain ⟨hpflo, hpfhi, hpglo, hpghi, hd⟩ := batchRound_bounded st hwords
  exact ⟨hpflo, hpfhi, hpglo, hpghi, hd, by
    simp only [batchRound, batchRoundFinish, subc32]
    exact Nat.mod_lt _ (by norm_num [wordMod])⟩

/-- Every word exposed through the adapter is a bounded 64-bit word. -/
theorem packedStep_words_lt (st : BatchState) (hst : packedStep.Bounded st) :
    packedStep.two_delta st < 2^64 ∧ packedStep.f st < 2^64 ∧ packedStep.g st < 2^64 := by
  obtain ⟨hfl, hfh, hgl, hgh, hd, -⟩ := hst
  exact ⟨signExtend32_lt st.d hd, join32_lt st.pflo st.pfhi hfl hfh,
    join32_lt st.pglo st.pghi hgl hgh⟩

/-- The flagless adapter's `z` value is exactly the shared parity predicate on its joined `g`
word. -/
theorem packedStep_z_eq_g_parity (st : BatchState) :
    packedStep.z st = if packedStep.g st % 2 = 0 then 1 else 0 := by
  rw [packedStep_z, packedStep_g, join32_mod_two]

/-- If the joined `g` word represents an integer modulo `2^64`, the flagless adapter's `z` value is
that integer's parity predicate, as required on entry to a packed step. -/
theorem packedStep_z_of_g_word (st : BatchState) (G : Int)
    (hg : (packedStep.g st : Int) = G % 2^64) :
    packedStep.z st = if G % 2 = 0 then 1 else 0 := by
  rw [packedStep_z_eq_g_parity]
  have hmod : ((packedStep.g st : Int) % 2) = G % 2 := by
    rw [hg, Int.emod_emod_of_dvd _ (by norm_num)]
  have hiff : packedStep.g st % 2 = 0 ↔ ((packedStep.g st : Int) % 2) = 0 := by
    omega
  have hpar : packedStep.g st % 2 = 0 ↔ G % 2 = 0 := by
    exact hiff.trans (by rw [hmod])
  by_cases h : packedStep.g st % 2 = 0
  · rw [if_pos h, if_pos (hpar.mp h)]
  · rw [if_neg h, if_neg (fun hG => h (hpar.mpr hG))]

end PastaCurves.ARMv7EM
