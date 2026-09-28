import PastaCurves.Spec
import PastaCurves.Inversion.Packed

/-!
# The packed divstep on words, independently of the instruction set

What a `divstep59` block computes with its words, stated over the shared word operations so
that any backend's block proof reduces to it: the words that carry a product or a sum modulo
`2^64`; the packing of a low word with the identity matrix; the arithmetic shifts that form the
next low words between the batches; the arithmetic shift of the decoder, which recovers a matrix
entry from the upper bits of a packed word; and the three cases of one packed step, on the words
that the step's conditional instructions produce. The bitfield extract of the decoder and the
flags are the backend's own.
-/

set_option exponentiation.threshold 400

namespace PastaCurves.Inversion

/-! ## Words -/

theorem mul_word (x y : ℕ) (X Y : ℤ) (hx : (x : ℤ) = X % 2^64) (hy : (y : ℤ) = Y % 2^64) :
    ((x * y % 2^64 : ℕ) : ℤ) = (X * Y) % 2^64 := by
  have h : ((x * y % 2^64 : ℕ) : ℤ) = ((x : ℤ) * y) % 2^64 := by push_cast; norm_num
  rw [h, hx, hy]
  exact (Int.mod_modEq X _).mul (Int.mod_modEq Y _)

theorem addw_word (x y : ℕ) (X Y : ℤ) (hx : (x : ℤ) = X % 2^64) (hy : (y : ℤ) = Y % 2^64) :
    ((addw x y : ℕ) : ℤ) = (X + Y) % 2^64 := by
  have h : ((addw x y : ℕ) : ℤ) = ((x : ℤ) + y) % 2^64 := by
    unfold addw regMod; push_cast; norm_num
  rw [h, hx, hy]
  exact (Int.mod_modEq X _).add (Int.mod_modEq Y _)

/-- A bitwise or of a value below `2^k` with a multiple of `2^k` is their sum. -/
theorem or_low_high (x k m : ℕ) (hx : x < 2^k) : x ||| 2^k * m = 2^k * m + x := by
  apply Nat.eq_of_testBit_eq
  intro j
  rw [Nat.testBit_or, Nat.testBit_two_pow_mul, Nat.testBit_two_pow_mul_add m hx]
  by_cases hj : j < k
  · simp [hj, Nat.not_le.mpr hj]
  · have hkj : k ≤ j := Nat.le_of_not_lt hj
    have hxj : x < 2^j := lt_of_lt_of_le hx (Nat.pow_le_pow_right (by decide) hkj)
    simp [hj, hkj, Nat.testBit_lt_two_pow hxj]

/-! ## Packing -/

/-- The packing of a low word: its low 20 bits with `-2^41` (resp. `-2^62`) in two's complement. -/
theorem pack_f_word (f : ℕ) :
    ((orrw (andw f 0xfffff) 0xfffffe0000000000 : ℕ) : ℤ) = (f % 2^20 - 2^41) % 2^64 := by
  unfold orrw andw
  rw [show (0xfffff : ℕ) = 2^20 - 1 by norm_num, Nat.and_two_pow_sub_one_eq_mod,
    show (0xfffffe0000000000 : ℕ) = 2^20 * (2^21 * (2^23 - 1)) by norm_num,
    or_low_high _ _ _ (Nat.mod_lt _ (by norm_num))]
  push_cast
  omega

theorem pack_g_word (g : ℕ) :
    ((orrw (andw g 0xfffff) 0xc000000000000000 : ℕ) : ℤ) = (g % 2^20 - 2^62) % 2^64 := by
  unfold orrw andw
  rw [show (0xfffff : ℕ) = 2^20 - 1 by norm_num, Nat.and_two_pow_sub_one_eq_mod,
    show (0xc000000000000000 : ℕ) = 2^20 * (2^42 * 3) by norm_num,
    or_low_high _ _ _ (Nat.mod_lt _ (by norm_num))]
  push_cast
  omega

/-! ## The next low words -/

/-- `asr` by 20 of a word that is `2^20 X` modulo `2^64` leaves `X` modulo `2^44`. -/
theorem asr20_word (w : ℕ) (X : ℤ) (hw : (w : ℤ) = (2^20 * X) % 2^64) :
    ((asr w 20 : ℕ) : ℤ) % 2^44 = X % 2^44 := by
  unfold asr; norm_num
  split_ifs <;> omega

/-- `asr` by 20 of a word that is `2^20 X` modulo `2^44` leaves `X` modulo `2^24`. -/
theorem asr20_word44 (w : ℕ) (X : ℤ) (hw : (w : ℤ) % 2^44 = (2^20 * X) % 2^44) :
    ((asr w 20 : ℕ) : ℤ) % 2^24 = X % 2^24 := by
  unfold asr; norm_num
  split_ifs <;> omega

/-! ## The decoder -/

/-- The `v` entry of a batch's packed word, for `k = 20`: `asr` by 42 of `w + 2^20 + 2^41` gives
`-v`, as a word. The offset rounds the `|φ| < 2^20` disturbance away, as Lemma 7 does with the
opposite sign. -/
theorem decode_v20 (w : ℕ) (W φ u v : ℤ) (hw : (w : ℤ) = W % 2^64)
    (hW : W = φ - 2^21 * u - 2^42 * v) (hφ : |φ| < 2^20)
    (hu : -(2 : ℤ)^20 < u ∧ u ≤ 2^20) (hv : -(2 : ℤ)^20 < v ∧ v ≤ 2^20) :
    ((asr (addw w (2^20 + 2^41)) 42 : ℕ) : ℤ) = (-v) % 2^64 := by
  rw [abs_lt] at hφ
  have hw64 : w < 2^64 := by omega
  have ha : addw w (2^20 + 2^41) = (w + (2^20 + 2^41)) % 2^64 := rfl
  unfold asr
  rw [ha]
  norm_num
  split_ifs <;> omega

/-- The `v` entry for `k = 19`: `asr` by 43. -/
theorem decode_v19 (w : ℕ) (W φ u v : ℤ) (hw : (w : ℤ) = W % 2^64)
    (hW : W = φ - 2^22 * u - 2^43 * v) (hφ : |φ| < 2^20)
    (hu : -(2 : ℤ)^19 < u ∧ u ≤ 2^19) (hv : -(2 : ℤ)^19 < v ∧ v ≤ 2^19) :
    ((asr (addw w (2^20 + 2^41)) 43 : ℕ) : ℤ) = (-v) % 2^64 := by
  rw [abs_lt] at hφ
  have hw64 : w < 2^64 := by omega
  have ha : addw w (2^20 + 2^41) = (w + (2^20 + 2^41)) % 2^64 := rfl
  unfold asr
  rw [ha]
  norm_num
  split_ifs <;> omega

/-! ## The three cases of a packed step

`divstep` on the packed state, as a backend computes it in words. The swap condition
`0 < two_delta ∧ g odd` selects one of three cases, and in each the step's conditional instructions
produce a negation or a copy of `two_delta`, of `f`, and of the addend `t`; the new `two_delta` is
`d' + 2`, the new `g` is the arithmetic shift of `g + t`, and bit 1 of the unhalved `g + t` is the
parity of the new `g`. Each case lemma takes the words that the instructions produced and says which
state they carry. The bound `hG'` is the no-wrap condition on the sum, Lemma 6′. -/

/-- `g` even: `t = 0`, `two_delta` and `f` unchanged. -/
theorem divstep_words_even (s : State) (hd : s.two_delta % 2 = 1) (hG : |s.g| < 2^63) (two_delta pf pg : ℕ)
    (hd' : (two_delta : ℤ) = s.two_delta % 2^64) (hf' : (pf : ℤ) = s.f % 2^64) (hg' : (pg : ℤ) = s.g % 2^64)
    (hg0 : s.g % 2 = 0) (t1 two_delta1 pf1 pg1 two_delta2 pg2 : ℕ) (ht1 : t1 = 0) (hd1 : two_delta1 = two_delta)
    (hpf1 : pf1 = pf) (e_pg1 : pg1 = addw pg t1) (e_two_delta2 : two_delta2 = addw two_delta1 2)
    (e_pg2 : pg2 = asr pg1 1) :
    (two_delta2 : ℤ) = (divstep s).two_delta % 2^64 ∧ (pf1 : ℤ) = (divstep s).f % 2^64 ∧
      (pg2 : ℤ) = (divstep s).g % 2^64 ∧ (pg1 / 2 % 2 = 0 ↔ (divstep s).g % 2 = 0) := by
  rw [abs_lt] at hG
  have hne : ¬ (0 < s.two_delta ∧ s.g % 2 = 1) := by omega
  simp only [divstep, if_neg hne]
  simp only [hg0, zero_mul, add_zero]
  have hpg1 : pg1 = (pg + t1) % 2^64 := e_pg1
  have hd2 : two_delta2 = (two_delta1 + 2) % 2^64 := e_two_delta2
  have hpg2 : pg2 = if pg1 < 2^63 then pg1 / 2 else pg1 / 2 + 2^63 := by
    rw [e_pg2]; unfold asr; norm_num [regMod]
  rw [hd2, hd1, hpf1, hpg2, hpg1, ht1]
  refine ⟨by omega, hf', ?_, ?_⟩
  · split_ifs <;> omega
  · omega

/-- The swap, `0 < two_delta` and `g` odd: `t = -f`, `two_delta` negated, `f := g`. -/
theorem divstep_words_swap (s : State) (hf : s.f % 2 = 1) (hd : s.two_delta % 2 = 1)
    (hG' : |(divstep s).g| < 2^62) (two_delta pf pg : ℕ) (b_two_delta : two_delta < 2^64) (b_pf : pf < 2^64)
    (hd' : (two_delta : ℤ) = s.two_delta % 2^64) (hf' : (pf : ℤ) = s.f % 2^64) (hg' : (pg : ℤ) = s.g % 2^64)
    (hsw : 0 < s.two_delta ∧ s.g % 2 = 1) (t1 two_delta1 pf1 pg1 two_delta2 pg2 : ℕ) (ht1 : t1 = negw pf)
    (hd1 : two_delta1 = negw two_delta) (hpf1 : pf1 = pg) (e_pg1 : pg1 = addw pg t1) (e_two_delta2 : two_delta2 = addw two_delta1 2)
    (e_pg2 : pg2 = asr pg1 1) :
    (two_delta2 : ℤ) = (divstep s).two_delta % 2^64 ∧ (pf1 : ℤ) = (divstep s).f % 2^64 ∧
      (pg2 : ℤ) = (divstep s).g % 2^64 ∧ (pg1 / 2 % 2 = 0 ↔ (divstep s).g % 2 = 0) := by
  rw [abs_lt] at hG'
  simp only [divstep, if_pos hsw] at hG' ⊢
  have hpg1 : pg1 = (pg + t1) % 2^64 := e_pg1
  have hd2 : two_delta2 = (two_delta1 + 2) % 2^64 := e_two_delta2
  have hpg2 : pg2 = if pg1 < 2^63 then pg1 / 2 else pg1 / 2 + 2^63 := by
    rw [e_pg2]; unfold asr; norm_num [regMod]
  have hd1' : two_delta1 = (2^64 - two_delta) % 2^64 := hd1
  have ht1' : t1 = (2^64 - pf) % 2^64 := ht1
  rw [hd2, hd1', hpf1, hpg2, hpg1, ht1']
  refine ⟨by omega, hg', ?_, ?_⟩
  · split_ifs <;> omega
  · omega

/-- The addition, `two_delta ≤ 0` and `g` odd: `t = f`, `two_delta` and `f` unchanged. -/
theorem divstep_words_add (s : State) (hf : s.f % 2 = 1) (hd : s.two_delta % 2 = 1) (hD : |s.two_delta| < 2^62)
    (hG' : |(divstep s).g| < 2^62) (two_delta pf pg : ℕ) (hd' : (two_delta : ℤ) = s.two_delta % 2^64)
    (hf' : (pf : ℤ) = s.f % 2^64) (hg' : (pg : ℤ) = s.g % 2^64) (hg1 : s.g % 2 = 1)
    (hdnp : s.two_delta ≤ 0) (t1 two_delta1 pf1 pg1 two_delta2 pg2 : ℕ)
    (ht1 : t1 = pf) (hd1 : two_delta1 = two_delta) (hpf1 : pf1 = pf) (e_pg1 : pg1 = addw pg t1)
    (e_two_delta2 : two_delta2 = addw two_delta1 2) (e_pg2 : pg2 = asr pg1 1) :
    (two_delta2 : ℤ) = (divstep s).two_delta % 2^64 ∧ (pf1 : ℤ) = (divstep s).f % 2^64 ∧
      (pg2 : ℤ) = (divstep s).g % 2^64 ∧ (pg1 / 2 % 2 = 0 ↔ (divstep s).g % 2 = 0) := by
  rw [abs_lt] at hD hG'
  have hne : ¬ (0 < s.two_delta ∧ s.g % 2 = 1) := by omega
  simp only [divstep, if_neg hne] at hG' ⊢
  simp only [hg1, one_mul] at hG' ⊢
  have hpg1 : pg1 = (pg + t1) % 2^64 := e_pg1
  have hd2 : two_delta2 = (two_delta1 + 2) % 2^64 := e_two_delta2
  have hpg2 : pg2 = if pg1 < 2^63 then pg1 / 2 else pg1 / 2 + 2^63 := by
    rw [e_pg2]; unfold asr; norm_num [regMod]
  rw [hd2, hd1, hpf1, hpg2, hpg1, ht1]
  refine ⟨by omega, hf', ?_, ?_⟩
  · split_ifs <;> omega
  · omega

end PastaCurves.Inversion
