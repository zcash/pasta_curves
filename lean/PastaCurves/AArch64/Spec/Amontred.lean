/-
Copyright Amazon.com, Inc. or its affiliates (the block, adapted from s2n-bignum).
Copyright (c) 2026 the pasta_curves contributors (the transcription and the proofs).
-/
import PastaCurves.AArch64.Spec.Words
import PastaCurves.AArch64.Transcription
import PastaCurves.Inversion.Round

/-!
# Correctness of the inversion's almost-Montgomery reduction block

See the parent module's documentation for details. The block is proved equal to the shared
word-level function `amontred` of `Inversion/Round.lean`, whose Lemma 10 (`amontredZ_spec`)
supplies the bound that lets the five-word arithmetic modulo `2^320` stand for the integers.
-/

-- The conclusion's arithmetic reaches `2^320`, which the default threshold leaves unevaluated.
set_option exponentiation.threshold 400

namespace PastaCurves.AArch64

-- BEGIN amontredBlock_spec statement
/-- The almost-Montgomery reduction by the inline block is the shared `amontred`, for a bounded
five-word input below `2^315` in magnitude and the modulus and `inv` of a Pasta field. The words
compute modulo `2^320`; the input's sign word and the dropped top carry are accounted for by
`amontredZ_spec`'s bound, which places the true result in `[0, 2^256)`. -/
theorem amontredBlock_spec (F : PastaField) (t : Signed5) (modulus : Limbs) (inv : Nat)
    (hmod : modulus = F.modulus) (hinv' : inv = F.inv)
    (ht : t.Bounded) (htv : |t.toInt| < 2^315) :
    ∀ r, r = amontredBlock t modulus inv → r = Inversion.amontred t F.modulus F.inv := by
  intro r hr
  have hm : modulus.Bounded := hmod ▸ F.bounded
  have hinv_lt : inv < 2^64 := hinv' ▸ F.inv_lt
  have hshape : modulus.l2 = 0 ∧ modulus.l3 = 2^62 := hmod ▸ F.shape
  have hinv : (inv * modulus.l0 + 1) % 2^64 = 0 := by rw [hmod, hinv']; exact F.inv_spec
-- END amontredBlock_spec statement
  -- generated skeleton for `amontredBlock`: do not edit between the annotations
  unfold amontredBlock at hr
  lift_lets -merge at hr
  -- t0: argument
  extract_lets -merge +onlyGivenNames t0 at hr
  have e_t0 : t0 = t.l0 := rfl
  clear_value t0
  have b_t0 : t0 < 2^64 := by rw [e_t0]; exact ht.1
  -- t1: argument
  extract_lets -merge +onlyGivenNames t1 at hr
  have e_t1 : t1 = t.l1 := rfl
  clear_value t1
  have b_t1 : t1 < 2^64 := by rw [e_t1]; exact ht.2.1
  -- t2: argument
  extract_lets -merge +onlyGivenNames t2 at hr
  have e_t2 : t2 = t.l2 := rfl
  clear_value t2
  have b_t2 : t2 < 2^64 := by rw [e_t2]; exact ht.2.2.1
  -- t3: argument
  extract_lets -merge +onlyGivenNames t3 at hr
  have e_t3 : t3 = t.l3 := rfl
  clear_value t3
  have b_t3 : t3 < 2^64 := by rw [e_t3]; exact ht.2.2.2.1
  -- t4: argument
  extract_lets -merge +onlyGivenNames t4 at hr
  have e_t4 : t4 = t.l4 := rfl
  clear_value t4
  have b_t4 : t4 < 2^64 := by rw [e_t4]; exact ht.2.2.2.2
  -- p0: argument
  extract_lets -merge +onlyGivenNames p0 at hr
  have e_p0 : p0 = modulus.l0 := rfl
  clear_value p0
  have b_p0 : p0 < 2^64 := by rw [e_p0]; exact hm.1
  -- p1: argument
  extract_lets -merge +onlyGivenNames p1 at hr
  have e_p1 : p1 = modulus.l1 := rfl
  clear_value p1
  have b_p1 : p1 < 2^64 := by rw [e_p1]; exact hm.2.1
  -- inv': argument
  extract_lets -merge +onlyGivenNames inv' at hr
  have e_inv' : inv' = inv := rfl
  clear_value inv'
  have b_inv' : inv' < 2^64 := by rw [e_inv']; exact hinv_lt
  -- w: lsl w,p0,#61
  extract_lets -merge +onlyGivenNames w at hr
  have e_w : w = p0 * 2^61 % 2^64 := rfl
  clear_value w
  have b_w : w < 2^64 := by rw [e_w]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- t0_1: adds t0,t0,w
  extract_lets -merge +onlyGivenNames s t0_1 c at hr
  have e_t0_1 : t0_1 = (t0 + w + 0) % 2^64 := rfl
  have e_c : c = (t0 + w + 0) / 2^64 := rfl
  clear_value s t0_1 c
  have l_t0_1 : t0_1 + 2^64 * c = t0 + w + 0 := by
    rw [e_t0_1, e_c]; exact Nat.mod_add_div _ _
  have b_t0_1 : t0_1 < 2^64 := by rw [e_t0_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c : c ≤ 1 := by
    rw [e_c]; exact addc_carry_le_one t0 w 0 b_t0 b_w (by decide)
  clear e_t0_1 e_c
  -- w_1: extr w,p1,p0,#3
  extract_lets -merge +onlyGivenNames w_1 at hr
  have e_w_1 : w_1 = extr p1 p0 3 := rfl
  clear_value w_1
  have b_w_1 : w_1 < 2^64 := by rw [e_w_1]; exact extr_lt p1 p0 3
  -- t1_1: adcs t1,t1,w
  extract_lets -merge +onlyGivenNames s_1 t1_1 c_1 at hr
  have e_t1_1 : t1_1 = (t1 + w_1 + c) % 2^64 := rfl
  have e_c_1 : c_1 = (t1 + w_1 + c) / 2^64 := rfl
  clear_value s_1 t1_1 c_1
  have l_t1_1 : t1_1 + 2^64 * c_1 = t1 + w_1 + c := by
    rw [e_t1_1, e_c_1]; exact Nat.mod_add_div _ _
  have b_t1_1 : t1_1 < 2^64 := by rw [e_t1_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_1 : c_1 ≤ 1 := by
    rw [e_c_1]; exact addc_carry_le_one t1 w_1 c b_t1 b_w_1 b_c
  clear e_t1_1 e_c_1
  -- w_2: lsr w,p1,#3
  extract_lets -merge +onlyGivenNames w_2 at hr
  have e_w_2 : w_2 = p1 / 2^3 := rfl
  clear_value w_2
  have b_w_2 : w_2 < 2^61 := by
    rw [e_w_2]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_p1 (by norm_num))
  -- t2_1: adcs t2,t2,w
  extract_lets -merge +onlyGivenNames s_2 t2_1 c_2 at hr
  have e_t2_1 : t2_1 = (t2 + w_2 + c_1) % 2^64 := rfl
  have e_c_2 : c_2 = (t2 + w_2 + c_1) / 2^64 := rfl
  clear_value s_2 t2_1 c_2
  have l_t2_1 : t2_1 + 2^64 * c_2 = t2 + w_2 + c_1 := by
    rw [e_t2_1, e_c_2]; exact Nat.mod_add_div _ _
  have b_t2_1 : t2_1 < 2^64 := by rw [e_t2_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_2 : c_2 ≤ 1 := by
    rw [e_c_2]; exact addc_carry_le_one t2 w_2 c_1 b_t2 (lt_of_lt_of_le b_w_2 (by norm_num)) b_c_1
  clear e_t2_1 e_c_2
  -- t3_1: adcs t3,t3,xzr
  extract_lets -merge +onlyGivenNames s_3 t3_1 c_3 at hr
  have e_t3_1 : t3_1 = (t3 + 0 + c_2) % 2^64 := rfl
  have e_c_3 : c_3 = (t3 + 0 + c_2) / 2^64 := rfl
  clear_value s_3 t3_1 c_3
  have l_t3_1 : t3_1 + 2^64 * c_3 = t3 + 0 + c_2 := by
    rw [e_t3_1, e_c_3]; exact Nat.mod_add_div _ _
  have b_t3_1 : t3_1 < 2^64 := by rw [e_t3_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_3 : c_3 ≤ 1 := by
    rw [e_c_3]; exact addc_carry_le_one t3 0 c_2 b_t3 (by decide) b_c_2
  clear e_t3_1 e_c_3
  -- w_3: mov w,#0x800000000000000
  extract_lets -merge +onlyGivenNames w_3 at hr
  have e_w_3 : w_3 = 576460752303423488 := rfl
  clear_value w_3
  have b_w_3 : w_3 < 2^64 := by rw [e_w_3]; decide
  -- t4_1: adc t4,t4,w
  extract_lets -merge +onlyGivenNames t4_1 at hr
  have e_t4_1 : t4_1 = (t4 + w_3 + c_3) % 2^64 := rfl
  clear_value t4_1
  have b_t4_1 : t4_1 < 2^64 := by rw [e_t4_1]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t4_1, b_k_t4_1, l_t4_1⟩ :
      ∃ k, k ≤ 1 ∧ t4_1 + 2^64 * k = t4 + w_3 + c_3 :=
    ⟨(t4 + w_3 + c_3) / 2^64, addc_carry_le_one t4 w_3 c_3 b_t4 b_w_3 b_c_3,
      by rw [e_t4_1]; exact Nat.mod_add_div _ _⟩
  clear e_t4_1
  -- q: mul q,t0,inv
  extract_lets -merge +onlyGivenNames q at hr
  have e_q : q = t0_1 * inv' % 2^64 := rfl
  clear_value q
  have b_q : q < 2^64 := by rw [e_q]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- hi: umulh hi,q,p0
  extract_lets -merge +onlyGivenNames hi at hr
  have e_hi : hi = q * p0 / 2^64 := rfl
  clear_value hi
  have p_hi : q * p0 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_q b_p0
  have b_hi : hi < 2^64 := by rw [e_hi]; exact Nat.div_lt_of_lt_mul p_hi
  obtain ⟨lo_hi, b_lo_hi, d_hi⟩ :
      ∃ lo, lo < 2^64 ∧ lo + 2^64 * hi = q * p0 :=
    ⟨q * p0 % 2^64, Nat.mod_lt _ (Nat.two_pow_pos _),
      by rw [e_hi]; exact Nat.mod_add_div _ _⟩
  clear e_hi
  -- lo: mul lo,q,p1
  extract_lets -merge +onlyGivenNames lo at hr
  have e_lo : lo = q * p1 % 2^64 := rfl
  clear_value lo
  have b_lo : lo < 2^64 := by rw [e_lo]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  -- w_4: umulh w,q,p1
  extract_lets -merge +onlyGivenNames w_4 at hr
  have e_w_4 : w_4 = q * p1 / 2^64 := rfl
  clear_value w_4
  have p_w_4 : q * p1 < 2^64 * 2^64 := Nat.mul_lt_mul'' b_q b_p1
  have b_w_4 : w_4 < 2^64 := by rw [e_w_4]; exact Nat.div_lt_of_lt_mul p_w_4
  have d_w_4 : lo + 2^64 * w_4 = q * p1 := by
    rw [e_lo, e_w_4]; exact Nat.mod_add_div _ _
  clear e_lo e_w_4
  -- l3: lsl l3,q,#62
  extract_lets -merge +onlyGivenNames l3 at hr
  have e_l3 : l3 = q * 2^62 % 2^64 := rfl
  clear_value l3
  have b_l3 : l3 < 2^64 := by rw [e_l3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have sh_l3 : l3 + 2^64 * (q / 2^2) = q * 2^62 := by
    rw [e_l3]; exact lsl62_lsr2_split _
  -- h3: lsr h3,q,#2
  extract_lets -merge +onlyGivenNames h3 at hr
  have e_h3 : h3 = q / 2^2 := rfl
  clear_value h3
  have b_h3 : h3 < 2^62 := by
    rw [e_h3]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq b_q (by norm_num))
  -- c_4: subs xzr,t0,#1
  extract_lets -merge +onlyGivenNames c_4 at hr
  have e_c_4 : c_4 = (t0_1 + 2^64 - 1 - (1 - 1)) / 2^64 := rfl
  clear_value c_4
  have b_c_4 : c_4 ≤ 1 := by rw [e_c_4]; exact subc_carry_le_one t0_1 1 1 b_t0_1
  have l_c_4 : (c_4 = 1 ∧ 1 + 1 ≤ t0_1 + 1) ∨ (c_4 = 0 ∧ t0_1 + 1 < 1 + 1) :=
    subc_carry_cases t0_1 1 1 _ e_c_4 b_t0_1 (by decide) (by decide)
  clear e_c_4
  -- t1_2: adcs t1,t1,hi
  extract_lets -merge +onlyGivenNames s_4 t1_2 c_5 at hr
  have e_t1_2 : t1_2 = (t1_1 + hi + c_4) % 2^64 := rfl
  have e_c_5 : c_5 = (t1_1 + hi + c_4) / 2^64 := rfl
  clear_value s_4 t1_2 c_5
  have l_t1_2 : t1_2 + 2^64 * c_5 = t1_1 + hi + c_4 := by
    rw [e_t1_2, e_c_5]; exact Nat.mod_add_div _ _
  have b_t1_2 : t1_2 < 2^64 := by rw [e_t1_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_5 : c_5 ≤ 1 := by
    rw [e_c_5]; exact addc_carry_le_one t1_1 hi c_4 b_t1_1 b_hi b_c_4
  clear e_t1_2 e_c_5
  -- t2_2: adcs t2,t2,w
  extract_lets -merge +onlyGivenNames s_5 t2_2 c_6 at hr
  have e_t2_2 : t2_2 = (t2_1 + w_4 + c_5) % 2^64 := rfl
  have e_c_6 : c_6 = (t2_1 + w_4 + c_5) / 2^64 := rfl
  clear_value s_5 t2_2 c_6
  have l_t2_2 : t2_2 + 2^64 * c_6 = t2_1 + w_4 + c_5 := by
    rw [e_t2_2, e_c_6]; exact Nat.mod_add_div _ _
  have b_t2_2 : t2_2 < 2^64 := by rw [e_t2_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_6 : c_6 ≤ 1 := by
    rw [e_c_6]; exact addc_carry_le_one t2_1 w_4 c_5 b_t2_1 b_w_4 b_c_5
  clear e_t2_2 e_c_6
  -- t3_2: adcs t3,t3,l3
  extract_lets -merge +onlyGivenNames s_6 t3_2 c_7 at hr
  have e_t3_2 : t3_2 = (t3_1 + l3 + c_6) % 2^64 := rfl
  have e_c_7 : c_7 = (t3_1 + l3 + c_6) / 2^64 := rfl
  clear_value s_6 t3_2 c_7
  have l_t3_2 : t3_2 + 2^64 * c_7 = t3_1 + l3 + c_6 := by
    rw [e_t3_2, e_c_7]; exact Nat.mod_add_div _ _
  have b_t3_2 : t3_2 < 2^64 := by rw [e_t3_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_7 : c_7 ≤ 1 := by
    rw [e_c_7]; exact addc_carry_le_one t3_1 l3 c_6 b_t3_1 b_l3 b_c_6
  clear e_t3_2 e_c_7
  -- t4_2: adc t4,t4,h3
  extract_lets -merge +onlyGivenNames t4_2 at hr
  have e_t4_2 : t4_2 = (t4_1 + h3 + c_7) % 2^64 := rfl
  clear_value t4_2
  have b_t4_2 : t4_2 < 2^64 := by rw [e_t4_2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t4_2, b_k_t4_2, l_t4_2⟩ :
      ∃ k, k ≤ 1 ∧ t4_2 + 2^64 * k = t4_1 + h3 + c_7 :=
    ⟨(t4_1 + h3 + c_7) / 2^64, addc_carry_le_one t4_1 h3 c_7 b_t4_1 (lt_of_lt_of_le b_h3 (by norm_num)) b_c_7,
      by rw [e_t4_2]; exact Nat.mod_add_div _ _⟩
  clear e_t4_2
  -- t1_3: adds t1,t1,lo
  extract_lets -merge +onlyGivenNames s_7 t1_3 c_8 at hr
  have e_t1_3 : t1_3 = (t1_2 + lo + 0) % 2^64 := rfl
  have e_c_8 : c_8 = (t1_2 + lo + 0) / 2^64 := rfl
  clear_value s_7 t1_3 c_8
  have l_t1_3 : t1_3 + 2^64 * c_8 = t1_2 + lo + 0 := by
    rw [e_t1_3, e_c_8]; exact Nat.mod_add_div _ _
  have b_t1_3 : t1_3 < 2^64 := by rw [e_t1_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_8 : c_8 ≤ 1 := by
    rw [e_c_8]; exact addc_carry_le_one t1_2 lo 0 b_t1_2 b_lo (by decide)
  clear e_t1_3 e_c_8
  -- t2_3: adcs t2,t2,xzr
  extract_lets -merge +onlyGivenNames s_8 t2_3 c_9 at hr
  have e_t2_3 : t2_3 = (t2_2 + 0 + c_8) % 2^64 := rfl
  have e_c_9 : c_9 = (t2_2 + 0 + c_8) / 2^64 := rfl
  clear_value s_8 t2_3 c_9
  have l_t2_3 : t2_3 + 2^64 * c_9 = t2_2 + 0 + c_8 := by
    rw [e_t2_3, e_c_9]; exact Nat.mod_add_div _ _
  have b_t2_3 : t2_3 < 2^64 := by rw [e_t2_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_9 : c_9 ≤ 1 := by
    rw [e_c_9]; exact addc_carry_le_one t2_2 0 c_8 b_t2_2 (by decide) b_c_8
  clear e_t2_3 e_c_9
  -- t3_3: adcs t3,t3,xzr
  extract_lets -merge +onlyGivenNames s_9 t3_3 c_10 at hr
  have e_t3_3 : t3_3 = (t3_2 + 0 + c_9) % 2^64 := rfl
  have e_c_10 : c_10 = (t3_2 + 0 + c_9) / 2^64 := rfl
  clear_value s_9 t3_3 c_10
  have l_t3_3 : t3_3 + 2^64 * c_10 = t3_2 + 0 + c_9 := by
    rw [e_t3_3, e_c_10]; exact Nat.mod_add_div _ _
  have b_t3_3 : t3_3 < 2^64 := by rw [e_t3_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have b_c_10 : c_10 ≤ 1 := by
    rw [e_c_10]; exact addc_carry_le_one t3_2 0 c_9 b_t3_2 (by decide) b_c_9
  clear e_t3_3 e_c_10
  -- t4_3: adc t4,t4,xzr
  extract_lets -merge +onlyGivenNames t4_3 at hr
  have e_t4_3 : t4_3 = (t4_2 + 0 + c_10) % 2^64 := rfl
  clear_value t4_3
  have b_t4_3 : t4_3 < 2^64 := by rw [e_t4_3]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  obtain ⟨k_t4_3, b_k_t4_3, l_t4_3⟩ :
      ∃ k, k ≤ 1 ∧ t4_3 + 2^64 * k = t4_2 + 0 + c_10 :=
    ⟨(t4_2 + 0 + c_10) / 2^64, addc_carry_le_one t4_2 0 c_10 b_t4_2 (by decide) b_c_10,
      by rw [e_t4_3]; exact Nat.mod_add_div _ _⟩
  clear e_t4_3
  subst hr
  -- BEGIN conclusion
  subst hmod hinv'
  have hP : F.modulus.toNat = p0 + 2^64 * p1 + 2^254 := by
    rw [e_p0, e_p1]; simp only [Limbs.toNat, hshape.1, hshape.2]; norm_num
  have hP61 : 2^61 * F.modulus.toNat = 2^61 * p0 + 2^125 * p1 + 2^315 := by rw [hP]; ring
  -- The input's words and its value: the sign word subtracts `2^320` when set.
  obtain ⟨σ, hσ, hT⟩ : ∃ σ : ℤ, (σ = 0 ∨ σ = 1) ∧
      t.toInt = ((t0 + 2^64 * t1 + 2^128 * t2 + 2^192 * t3 + 2^256 * t4 : ℕ) : ℤ) - 2^320 * σ := by
    unfold Signed5.toInt
    rw [← e_t0, ← e_t1, ← e_t2, ← e_t3, ← e_t4]
    split_ifs
    · exact ⟨0, Or.inl rfl, by push_cast; ring⟩
    · exact ⟨1, Or.inr rfl, by push_cast; ring⟩
  -- Adding `2^61 p`: the words added are `2^61 p0 + 2^125 p1` split at the word boundaries, and
  -- `2^59` at word 4 is `2^315`.
  have hw1 : w_1 = (p0 / 2^3 + p1 * 2^61) % 2^64 := by rw [e_w_1]; rfl
  have hS : t0_1 + 2^64 * t1_1 + 2^128 * t2_1 + 2^192 * t3_1 + 2^256 * t4_1 + 2^320 * k_t4_1
      = (t0 + 2^64 * t1 + 2^128 * t2 + 2^192 * t3 + 2^256 * t4) + 2^61 * F.modulus.toNat := by
    rw [hP61]; clear * - l_t0_1 l_t1_1 l_t2_1 l_t3_1 l_t4_1 e_w hw1 e_w_2 e_w_3 b_p0 b_p1; omega
  -- The cancellation: the low word of `t0_1 + p0 * q` is zero, so `t0_1 + lo_hi` is `0` or
  -- `2^64`, and `subs xzr, t0, #1` set the carry exactly when it is `2^64`.
  have hq' : q = inv' * t0_1 % 2^64 := by rw [e_q, Nat.mul_comm]
  have d_hi' : lo_hi + 2^64 * hi = p0 * q := by rw [d_hi, Nat.mul_comm]
  have hc4 : t0_1 + lo_hi = 2^64 * c_4 := by
    have h := cancel_low t0_1 inv' p0 (by rw [e_inv', e_p0]; exact hinv)
    rw [← hq', ← d_hi', Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt b_lo_hi] at h
    clear * - h b_t0_1 b_lo_hi l_c_4
    omega
  -- The two chains add `q p` and drop the cancelled low word.
  have hqp : q * F.modulus.toNat
      = (lo_hi + 2^64 * hi) + 2^64 * (lo + 2^64 * w_4) + 2^192 * (l3 + 2^64 * h3) := by
    rw [hP, d_hi, d_w_4, e_h3, sh_l3]; ring
  have hR : 2^64 * (t1_3 + 2^64 * t2_3 + 2^128 * t3_3 + 2^192 * t4_3) + 2^320 * (k_t4_2 + k_t4_3)
      = (t0_1 + 2^64 * t1_1 + 2^128 * t2_1 + 2^192 * t3_1 + 2^256 * t4_1) + q * F.modulus.toNat := by
    rw [hqp]
    clear * - hc4 l_t1_2 l_t2_2 l_t3_2 l_t4_2 l_t1_3 l_t2_3 l_t3_3 l_t4_3
    omega
  -- On the integers: the block's multiplier is the model's, and the block's four words are the
  -- model's result, which `amontredZ_spec` places in `[0, 2^256)`.
  set T : ℕ := t0 + 2^64 * t1 + 2^128 * t2 + 2^192 * t3 + 2^256 * t4 with hTdef
  set V : ℕ := t1_3 + 2^64 * t2_3 + 2^128 * t3_3 + 2^192 * t4_3 with hVdef
  set A : ℕ := t1_1 + 2^64 * t2_1 + 2^128 * t3_1 + 2^192 * t4_1 + 2^256 * k_t4_1 with hAdef
  set p : ℕ := F.modulus.toNat with hp
  set K : ℕ := k_t4_1 + k_t4_2 + k_t4_3 with hKdef
  have hS' : T + 2^61 * p = t0_1 + 2^64 * A := by clear * - hS hAdef; omega
  have hR1 : t0_1 + 2^64 * A + q * p = 2^64 * V + 2^320 * K := by
    clear * - hR hAdef hVdef hKdef; omega
  have hV : V < 2^256 := by clear * - hVdef b_t1_3 b_t2_3 b_t3_3 b_t4_3; omega
  have hS'z : ((T : ℕ) : ℤ) + 2^61 * (p : ℤ) = t0_1 + 2^64 * A := by exact_mod_cast hS'
  have hR1z : (t0_1 : ℤ) + 2^64 * A + q * p = 2^64 * V + 2^320 * K := by exact_mod_cast hR1
  have hs : t.toInt + 2^61 * (p : ℤ) = (t0_1 : ℤ) + 2^64 * ((A : ℤ) - 2^256 * σ) := by
    rw [hT]; linear_combination hS'z
  have hw : ((t.toInt + 2^61 * (p : ℤ)) * (F.inv : ℤ)) % 2^64 = (q : ℤ) := by
    rw [hs, show ((t0_1 : ℤ) + 2^64 * ((A : ℤ) - 2^256 * σ)) * F.inv
        = t0_1 * F.inv + 2^64 * (((A : ℤ) - 2^256 * σ) * F.inv) by ring,
      Int.add_mul_emod_self_left, e_q, e_inv']
    norm_cast
  have hsum : t.toInt + 2^61 * (p : ℤ) + (q : ℤ) * p = 2^64 * ((V : ℤ) + 2^256 * ((K : ℤ) - σ)) := by
    rw [hs]; linear_combination hR1z
  have hval : Inversion.amontredZ t.toInt p F.inv = (V : ℤ) + 2^256 * ((K : ℤ) - σ) := by
    show (t.toInt + 2^61 * (p : ℤ) + (t.toInt + 2^61 * (p : ℤ)) * F.inv % 2^64 * p) / 2^64 = _
    rw [hw, hsum, Int.mul_ediv_cancel_left _ (by norm_num)]
  obtain ⟨h0, -, -, h256, -⟩ := Inversion.amontredZ_spec F t.toInt htv
  rw [← hp, hval] at h0 h256
  have hD : (K : ℤ) - σ = 0 := by clear * - h0 h256 hV hσ; omega
  have hres : Inversion.amontred t F.modulus F.inv = Limbs.ofNat V := by
    unfold Inversion.amontred
    rw [← hp, hval, hD, mul_zero, add_zero, Int.toNat_natCast]
  rw [hres]
  simp only [Limbs.ofNat, hVdef, Limbs.mk.injEq]
  clear * - b_t1_3 b_t2_3 b_t3_3 b_t4_3
  omega
  -- END conclusion

end PastaCurves.AArch64
