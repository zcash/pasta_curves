import PastaCurves.Portable.Spec

/-!
# The translated rows satisfy their contracts

`fg_row` and `de_row` of `src/inversion/portable.rs` compute a row `a x + b y` of the transition
matrix applied to two values, with `a` and `b` given in sign-magnitude form (`SignMagRep`). Both go
through `row`, which negates each of `x` and `y` under its entry's sign mask (`negate`), then
multiplies by the magnitudes column by column. The words compute modulo `2^320`, and the bound
`|a| + |b| ≤ 2^63` keeps the row below `2^319` in magnitude, so its five words read as two's
complement are the row itself. `de_row` takes four-limb `d` and `e`; `fg_row` takes five-word
signed `f` and `g` and shifts the row right by `59` (`shift59_spec`).

The proofs read five words as one unsigned number (`Signed5.toNat`), which agrees with the signed
reading modulo `2^320`. A carry chain of five limbs adds up by `chain5`, for the carry chains of
`negate` and the columns of `row` alike.
-/

set_option exponentiation.threshold 400

namespace PastaCurves

/-- Five words read as an unsigned number, the sign word as a plain word; it is below `2^320` when
the words are bounded. -/
def Signed5.toNat (x : Signed5) : ℕ :=
  x.l0 + 2^64 * x.l1 + 2^128 * x.l2 + 2^192 * x.l3 + 2^256 * x.l4

/-- The signed and the unsigned reading of five words agree modulo `2^320`. -/
theorem Signed5.toInt_modEq (x : Signed5) : x.toInt ≡ x.toNat [ZMOD 2^320] := by
  unfold Int.ModEq Signed5.toInt Signed5.toNat
  split_ifs <;> agrind

/-- Five bounded words that agree modulo `2^320` with an integer in `[-2^319, 2^319)` represent
that integer. -/
theorem Signed5.toInt_eq_of_modEq {x : Signed5} (hx : x.Bounded) {z : ℤ}
    (h : (x.toNat : ℤ) ≡ z [ZMOD 2^320]) (hz : -2^319 ≤ z ∧ z < 2^319) : x.toInt = z := by
  obtain ⟨h0, h1, h2, h3, h4⟩ := hx
  unfold Int.ModEq Signed5.toNat at h
  unfold Signed5.toInt
  split_ifs <;> agrind

end PastaCurves

namespace PastaCurves.Portable

open Aeneas Aeneas.Std
open Inversion (SignMagRep)

/-- A carry chain of five limbs, added up. Each limb's word and carry out make up its addend plus
its carry in. So the words at their weights, with the last carry out at weight `2^320`, make up the
addends at their weights plus the first carry in. -/
theorem chain5 {o0 o1 o2 o3 o4 a0 a1 a2 a3 a4 c0 c1 c2 c3 c4 c5 : ℕ}
    (h0 : o0 + 2^64 * c1 = a0 + c0) (h1 : o1 + 2^64 * c2 = a1 + c1)
    (h2 : o2 + 2^64 * c3 = a2 + c2) (h3 : o3 + 2^64 * c4 = a3 + c3)
    (h4 : o4 + 2^64 * c5 = a4 + c4) :
    o0 + 2^64 * o1 + 2^128 * o2 + 2^192 * o3 + 2^256 * o4 + 2^320 * c5 =
      a0 + 2^64 * a1 + 2^128 * a2 + 2^192 * a3 + 2^256 * a4 + c0 := by
  agrind

open pasta_curves.inversion.portable in
/-- `negate` under a mask of zero or all ones returns the five words of `x`, or of `-x` modulo
`2^320`. -/
@[step]
theorem negate_spec (x : Std.Array Std.U64 5#usize) (s : Std.U64)
    (hs : s.val = 0 ∨ s.val = 2^64 - 1) :
    negate x s ⦃ (res : Std.Array Std.U64 5#usize) =>
      ((signed5OfArray res).toNat : ℤ) ≡
        (if s.val = 0 then 1 else -1) * (signed5OfArray x).toNat [ZMOD 2^320] ⦄ := by
  unfold negate
  step*
  · rw [carry_post, UScalar.val_and]
    exact Nat.and_le_right
  have hchain := chain5 i2_post i5_post i8_post i11_post i14_post
  have hres : signed5OfArray res = ⟨i2.val, i5.val, i8.val, i11.val, i14.val⟩ := by
    simp [signed5OfArray, res_post, out4_post, out3_post, out2_post, out1_post]
  have hx : signed5OfArray x = ⟨i.val, i3.val, i6.val, i9.val, i12.val⟩ := by
    simp [signed5OfArray, i_post, i3_post, i6_post, i9_post, i12_post]
  rw [hres, hx]
  rw [UScalar.val_xor] at i1_post i4_post i7_post i10_post i13_post
  rw [UScalar.val_and] at carry_post
  unfold Signed5.toNat Int.ModEq
  dsimp only
  rcases hs with h | h
  · simp only [h, Nat.xor_zero, Nat.zero_and]
      at i1_post i4_post i7_post i10_post i13_post carry_post
    simp only [h, if_true, one_mul]
    agrind
  · have hones (w : Std.U64) : w.val ^^^ (2^64 - 1) = 2^64 - 1 - w.val :=
      Inversion.eorw_ones w.val w.hBounds
    simp only [h, hones] at i1_post i4_post i7_post i10_post i13_post
    have hc : carry.val = 1 := by rw [carry_post, h]; rfl
    rw [hc, i1_post, i4_post, i7_post, i10_post, i13_post] at hchain
    have b0 : i.val < 2^64 := by scalar_tac
    have b1 : i3.val < 2^64 := by scalar_tac
    have b2 : i6.val < 2^64 := by scalar_tac
    have b3 : i9.val < 2^64 := by scalar_tac
    have b4 : i12.val < 2^64 := by scalar_tac
    simp only [h, if_neg (show (2:ℕ)^64 - 1 ≠ 0 by norm_num), neg_one_mul]
    clear * - hchain b0 b1 b2 b3 b4
    agrind

/-- An entry in sign-magnitude form is its magnitude times the sign that `negate` applies under
its mask. -/
theorem signMagRep_sign {m s : ℕ} {a : ℤ} (h : SignMagRep m s a) :
    (m : ℤ) * (if s = 0 then 1 else -1) = a := by
  rcases h with ⟨hs, hm⟩ | ⟨hs, hm⟩
  · rw [hs, if_pos rfl, mul_one, hm]
  · rw [hs, if_neg (by norm_num), hm, mul_neg_one, neg_neg]

open pasta_curves.inversion.portable in
/-- `row` on two sides with entries `a` and `b` in sign-magnitude form, whose magnitudes sum to at
most `2^63`, returns the five words of `a x + b y` modulo `2^320`. -/
@[step]
theorem row_spec (x y : Std.Array Std.U64 5#usize) (m0 m1 s0 s1 : Std.U64) (a b : ℤ)
    (hm : m0.val + m1.val ≤ 2^63) (ha : SignMagRep m0.val s0.val a)
    (hb : SignMagRep m1.val s1.val b) :
    row x y m0 m1 s0 s1 ⦃ (res : Std.Array Std.U64 5#usize) =>
      ((signed5OfArray res).toNat : ℤ) ≡
        a * (signed5OfArray x).toNat + b * (signed5OfArray y).toNat [ZMOD 2^320] ⦄ := by
  unfold row
  have hs0 : s0.val = 0 ∨ s0.val = 2^64 - 1 := by rcases ha with ⟨h, -⟩ | ⟨h, -⟩ <;> simp [h]
  have hs1 : s1.val = 0 ∨ s1.val = 2^64 - 1 := by rcases hb with ⟨h, -⟩ | ⟨h, -⟩ <;> simp [h]
  step*
  have hchain := chain5 i2_post i5_post i8_post i11_post i14_post
  have hres : signed5OfArray res = ⟨i2.val, i5.val, i8.val, i11.val, i14.val⟩ := by
    simp [signed5OfArray, res_post, out4_post, out3_post, out2_post, out1_post]
  have hx1 : signed5OfArray x1 = ⟨i.val, i3.val, i6.val, i9.val, i12.val⟩ := by
    simp [signed5OfArray, i_post, i3_post, i6_post, i9_post, i12_post]
  have hy1 : signed5OfArray y1 = ⟨i1.val, i4.val, i7.val, i10.val, i13.val⟩ := by
    simp [signed5OfArray, i1_post, i4_post, i7_post, i10_post, i13_post]
  have hW : ((signed5OfArray res).toNat : ℤ) ≡
      m0.val * (signed5OfArray x1).toNat + m1.val * (signed5OfArray y1).toNat [ZMOD 2^320] := by
    rw [hres, hx1, hy1]
    unfold Signed5.toNat Int.ModEq
    dsimp only
    clear * - hchain
    agrind
  calc _ ≡ m0.val * ((if s0.val = 0 then 1 else -1) * (signed5OfArray x).toNat)
        + m1.val * ((if s1.val = 0 then 1 else -1) * (signed5OfArray y).toNat) [ZMOD 2^320] :=
        hW.trans ((x1_post.mul_left _).add (y1_post.mul_left _))
    _ = _ := by rw [← signMagRep_sign ha, ← signMagRep_sign hb]; ring

/-- The magnitudes of a row in sign-magnitude form sum to `|a| + |b|`, so the row bound bounds
them. -/
theorem magnitudes_le {m0 m1 s0 s1 : ℕ} {a b : ℤ} (ha : SignMagRep m0 s0 a)
    (hb : SignMagRep m1 s1 b) (hab : |a| + |b| ≤ 2^63) : m0 + m1 ≤ 2^63 := by
  have h0 := Inversion.SignMagRep.natCast_eq_abs _ _ _ ha
  have h1 := Inversion.SignMagRep.natCast_eq_abs _ _ _ hb
  zify
  rw [h0, h1]
  exact hab

/-- A row whose entries' magnitudes sum to at most `2^63`, applied to two values below `2^256` in
magnitude, lies in `[-2^319, 2^319)`, the range of five words read as two's complement. -/
theorem row_range {a b x y : ℤ} (hab : |a| + |b| ≤ 2^63) (hx : |x| < 2^256) (hy : |y| < 2^256) :
    -2^319 ≤ a * x + b * y ∧ a * x + b * y < 2^319 := by
  have hax : |a * x| ≤ |a| * (2^256 - 1) := by
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_left (Int.le_sub_one_of_lt hx) (abs_nonneg a)
  have hby : |b * y| ≤ |b| * (2^256 - 1) := by
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_left (Int.le_sub_one_of_lt hy) (abs_nonneg b)
  have hrow : |a * x + b * y| ≤ 2^63 * (2^256 - 1) :=
    calc |a * x + b * y| ≤ |a * x| + |b * y| := abs_add_le _ _
      _ ≤ (|a| + |b|) * (2^256 - 1) := by rw [add_mul]; exact add_le_add hax hby
      _ ≤ 2^63 * (2^256 - 1) := mul_le_mul_of_nonneg_right hab (by norm_num)
  rw [abs_le] at hrow
  constructor <;> agrind

/-- A word of a shift right by `k` across two words, `(lo >> k) | (hi << j)` with `k + j = 64`, is
`extr` of `Semantics.lean`. `down` and `up` are the two shifted words, as `step` describes them;
their bits do not overlap, so the or is a sum. -/
theorem extr_word {k j : ℕ} {lo hi down up word : Std.U64} (hdown : down.val = lo.val >>> k)
    (hup : up.val = hi.val <<< j % U64.size) (hword : word.val = (down ||| up).val)
    (hkj : k + j = 64) :
    word.val = extr hi.val lo.val k := by
  simp only [U64.size, U64.numBits, UScalarTy.U64_numBits_eq] at hup
  have h64 : (2:ℕ)^64 = 2^k * 2^j := by rw [← pow_add, hkj]
  have hq : hi.val <<< j % 2^64 = (hi.val % 2^k) <<< j := by
    rw [Nat.shiftLeft_eq, Nat.shiftLeft_eq, h64, Nat.mul_mod_mul_right]
  have hlo : lo.val >>> k < 2^j := by
    rw [Nat.shiftRight_eq_div_pow]
    exact Nat.div_lt_of_lt_mul (by rw [← h64]; scalar_tac)
  rw [hword, UScalar.val_or, hdown, hup, hq, Nat.or_comm, ← Nat.shiftLeft_add_eq_or_of_lt hlo,
    extr, Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow]
  rw [Nat.shiftRight_eq_div_pow] at hlo
  -- `hi` splits at bit `k`; its upper part lands at `2^64` and drops out.
  have hsplit : lo.val / 2^k + hi.val * 2^j =
      (lo.val / 2^k + hi.val % 2^k * 2^j) + 2^k * 2^j * (hi.val / 2^k) := by
    conv_lhs => rw [← Nat.mod_add_div hi.val (2^k)]
    ring
  have hlt : lo.val / 2^k + hi.val % 2^k * 2^j < 2^k * 2^j :=
    calc _ < (hi.val % 2^k + 1) * 2^j := by rw [add_mul, one_mul]; agrind
      _ ≤ 2^k * 2^j := Nat.mul_le_mul_right _ (Nat.mod_lt _ (by positivity))
  rw [show 64 - k = j by agrind, regMod, h64, hsplit, Nat.add_mul_mod_self_left,
    Nat.mod_eq_of_lt hlt, add_comm]

/-- The top word of the shift right by `59`, `((w as i64) >> 59) as u64`, is `asr` of
`Semantics.lean`. `m` is the shifted value, as `step` describes it. -/
theorem asr59_val {w : Std.U64} {m : Std.I64} (hm : m.val = (UScalar.hcast .I64 w).val >>> 59) :
    (IScalar.hcast .U64 m).val = asr w.val 59 := by
  rw [IScalar.hcast_val_eq, hm, UScalar.hcast_val_eq, Int.shiftRight_eq_div_pow]
  agrind [asr, Int.bmod]

open pasta_curves.inversion.portable in
/-- `de_row` satisfies its contract, the `deRow` field of `InvertBlocks.Spec`. On four-limb `d` and
`e`, and a row in sign-magnitude form whose magnitudes sum to at most `2^63`, it returns the exact
`a d + b e` in five words. -/
theorem de_row_spec (a b : ℤ) (d e : Limbs) (m0 m1 s0 s1 : Std.U64) (hd : d.Bounded)
    (he : e.Bounded) (hab : |a| + |b| ≤ 2^63) (ha : SignMagRep m0.val s0.val a)
    (hb : SignMagRep m1.val s1.val b) :
    Backend.Insts.Pasta_curvesInversionInvertBlocks.de_row (limbsArray d) (limbsArray e) m0 m1 s0 s1
    ⦃ (res : Std.Array Std.U64 5#usize) =>
      (signed5OfArray res).Bounded ∧
        (signed5OfArray res).toInt = a * d.toNat + b * e.toNat ⦄ := by
  unfold Backend.Insts.Pasta_curvesInversionInvertBlocks.de_row
  have hm := magnitudes_le ha hb hab
  step*
  have hd_lt := Limbs.toNat_lt d hd
  have he_lt := Limbs.toNat_lt e he
  obtain ⟨hd0, hd1, hd2, hd3⟩ := hd
  obtain ⟨he0, he1, he2, he3⟩ := he
  have hx : (signed5OfArray (Std.Array.make 5#usize [i, i1, i2, i3, 0#u64] (by rfl))).toNat =
      d.toNat := by
    simp [signed5OfArray, Signed5.toNat, Limbs.toNat, i_post, i1_post, i2_post, i3_post,
      limbsArray, word_val hd0, word_val hd1, word_val hd2, word_val hd3]
  have hy : (signed5OfArray (Std.Array.make 5#usize [i4, i5, i6, i7, 0#u64] (by rfl))).toNat =
      e.toNat := by
    simp [signed5OfArray, Signed5.toNat, Limbs.toNat, i4_post, i5_post, i6_post, i7_post,
      limbsArray, word_val he0, word_val he1, word_val he2, word_val he3]
  rw [hx, hy] at res_post
  exact ⟨signed5OfArray_bounded res, Signed5.toInt_eq_of_modEq (signed5OfArray_bounded res)
    res_post (row_range hab (by simpa using hd_lt) (by simpa using he_lt))⟩

open pasta_curves.inversion.portable in
/-- `fg_row` satisfies its contract, the `fgRow` field of `InvertBlocks.Spec`. On five-word `f` and
`g` below `2^256` in magnitude, and a row in sign-magnitude form whose magnitudes sum to at most
`2^63`, it returns the exact `(a f + b g) / 2^59` in five words. -/
theorem fg_row_spec (a b : ℤ) (f g : Signed5) (m0 m1 s0 s1 : Std.U64) (hf : f.Bounded)
    (hg : g.Bounded) (hfv : |f.toInt| < 2^256) (hgv : |g.toInt| < 2^256) (hab : |a| + |b| ≤ 2^63)
    (ha : SignMagRep m0.val s0.val a) (hb : SignMagRep m1.val s1.val b) :
    Backend.Insts.Pasta_curvesInversionInvertBlocks.fg_row (signed5Array f) (signed5Array g) m0 m1
      s0 s1
    ⦃ (res : Std.Array Std.U64 5#usize) =>
      (signed5OfArray res).Bounded ∧
        (signed5OfArray res).toInt = (a * f.toInt + b * g.toInt) / 2^59 ⦄ := by
  unfold Backend.Insts.Pasta_curvesInversionInvertBlocks.fg_row
  have hm := magnitudes_le ha hb hab
  step*
  -- The row, exactly: its words agree with `a f + b g` modulo `2^320`, which is in range.
  rw [signed5OfArray_signed5Array hf, signed5OfArray_signed5Array hg] at t_post
  have ht : (signed5OfArray t).toInt = a * f.toInt + b * g.toInt :=
    Signed5.toInt_eq_of_modEq (signed5OfArray_bounded t)
      (t_post.trans ((f.toInt_modEq.symm.mul_left a).add (g.toInt_modEq.symm.mul_left b)))
      (row_range hab hfv hgv)
  -- The shift right by `59`, word by word.
  have e0 := extr_word i1_post i3_post i4_post rfl
  have e1 := extr_word i5_post i7_post i8_post rfl
  have e2 := extr_word i9_post i11_post i12_post rfl
  have e3 := extr_word i13_post i15_post i16_post rfl
  have e4 : i19.val = asr i14.val 59 := by
    subst i17
    rw [i19_post]
    exact asr59_val i18_post
  have hres : signed5OfArray (Std.Array.make 5#usize [i4, i8, i12, i16, i19] (by rfl)) =
      ⟨extr i2.val i.val 59, extr i6.val i2.val 59, extr i10.val i6.val 59,
        extr i14.val i10.val 59, asr i14.val 59⟩ := by
    simp [signed5OfArray, e0, e1, e2, e3, e4]
  have hwords : signed5OfArray t = ⟨i.val, i2.val, i6.val, i10.val, i14.val⟩ := by
    simp [signed5OfArray, i_post, i2_post, i6_post, i10_post, i14_post]
  rw [hres]
  obtain ⟨hbounded, hshift⟩ := Inversion.shift59_spec _ _ _ _ _ i.hBounds i2.hBounds i6.hBounds
    i10.hBounds i14.hBounds
  rw [← hwords, ht] at hshift
  exact ⟨hbounded, hshift⟩

end PastaCurves.Portable
