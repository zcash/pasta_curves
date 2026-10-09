import PastaCurves.Portable.Row

/-!
# The translated `amontred` satisfies its contract

`amontred` of `src/inversion/portable.rs` adds `2^61 p` to a five-word signed value `t`, which makes
it nonnegative, then divides it by `2^64` with one word of Montgomery reduction. `2^61 p` is `p`
shifted left by 61 bits. Each of its middle words holds the top 61 bits of one limb, with the low
three bits of the next limb above them. `extr_word` reads such a word as `extr` of `Semantics.lean`:
the 64 bits of a two-limb number from bit `3` up. `add5` adds the words to `t` with a carry chain
(`add5_spec`). Each column of the reduction is a `mac`, and together the columns represent
`(s + w p) / 2^64` exactly. Lemma 10 (`amontredZ_spec`) bounds that quotient below `2^256`, so the
addition into the top word does not wrap.
-/

set_option exponentiation.threshold 400

namespace PastaCurves.Portable

open Aeneas Aeneas.Std

open pasta_curves.inversion.portable in
/-- `add5` returns the five words of `x + y` modulo `2^320`. -/
@[step]
theorem add5_spec (x y : Std.Array Std.U64 5#usize) :
    add5 x y ⦃ (res : Std.Array Std.U64 5#usize) =>
      ((signed5OfArray res).toNat : ℤ) ≡
        (signed5OfArray x).toNat + (signed5OfArray y).toNat [ZMOD 2^320] ⦄ := by
  unfold add5
  step*
  have hchain := chain5 i2_post i5_post i8_post i11_post i14_post
  have hres : signed5OfArray res = ⟨i2.val, i5.val, i8.val, i11.val, i14.val⟩ := by
    simp [signed5OfArray, res_post, out4_post, out3_post, out2_post, out1_post]
  have hx : signed5OfArray x = ⟨i.val, i3.val, i6.val, i9.val, i12.val⟩ := by
    simp [signed5OfArray, i_post, i3_post, i6_post, i9_post, i12_post]
  have hy : signed5OfArray y = ⟨i1.val, i4.val, i7.val, i10.val, i13.val⟩ := by
    simp [signed5OfArray, i1_post, i4_post, i7_post, i10_post, i13_post]
  rw [hres, hx, hy]
  unfold Signed5.toNat Int.ModEq
  dsimp only
  clear * - hchain
  agrind

open pasta_curves.inversion.portable in
/-- `amontred` satisfies its contract, the `amontred` field of `InvertBlocks.Spec`. On five bounded
words `t` below `2^315` in magnitude, it returns the four limbs of the round's `amontred`. -/
theorem amontred_spec (F : PastaField) (t : Signed5) (ht : t.Bounded) (htv : |t.toInt| < 2^315) :
    Backend.Insts.Pasta_curvesInversionInvertBlocks.amontred (signed5Array t)
      (limbsArray F.modulus) (word F.inv)
    ⦃ res => limbsOfArray res = Inversion.amontred t F.modulus F.inv ⦄ := by
  unfold Backend.Insts.Pasta_curvesInversionInvertBlocks.amontred
  step*
  -- The middle words of `2^61 p`: the top 61 bits of one limb, the low three of the next above.
  have e5 := extr_word i2_post i4_post i5_post rfl
  have e9 := extr_word i6_post i8_post i9_post rfl
  have e13 := extr_word i10_post i12_post i13_post rfl
  -- The words read from the modulus are its limbs.
  obtain ⟨hp0, hp1, hp2, hp3⟩ := F.bounded
  have hp : F.modulus.toNat = i.val + 2^64 * i3.val + 2^128 * i7.val + 2^192 * i11.val := by
    simp [Limbs.toNat, i_post, i3_post, i7_post, i11_post, limbsArray, word_val hp0,
      word_val hp1, word_val hp2, word_val hp3]
  have hp61 : (signed5OfArray (Std.Array.make 5#usize [i1, i5, i9, i13, i14] (by rfl))).toNat =
      2^61 * F.modulus.toNat := by
    simp only [U64.size, U64.numBits, UScalarTy.U64_numBits_eq, Nat.shiftLeft_eq,
      Nat.shiftRight_eq_div_pow] at i1_post i14_post
    simp [signed5OfArray, Signed5.toNat, hp, e5, e9, e13, extr, i1_post, i14_post]
    agrind
  -- `s = t + 2^61 p` exactly: the sum is positive and below `2^320`.
  rw [signed5OfArray_signed5Array ht, hp61] at s_post
  have hp254 : 2^254 ≤ F.modulus.toNat := F.two_pow_le_modulus
  have hp255 : F.modulus.toNat < 2^255 := F.modulus_lt
  have hs_lt : (signed5OfArray s).toNat < 2^320 := by
    obtain ⟨b0, b1, b2, b3, b4⟩ := signed5OfArray_bounded s
    unfold Signed5.toNat
    agrind
  have hS : ((signed5OfArray s).toNat : ℤ) = t.toInt + 2^61 * F.modulus.toNat := by
    have h := s_post.trans (t.toInt_modEq.symm.add_right _)
    unfold Int.ModEq at h
    rw [abs_lt] at htv
    scalar_tac
  have hwords : signed5OfArray s = ⟨i15.val, i16.val, i17.val, i18.val, i19.val⟩ := by
    simp [signed5OfArray, i15_post, i16_post, i17_post, i18_post, i19_post]
  have hw : w.val = i15.val * F.inv % 2^64 := by
    rw [w_post, core.num.U64.wrapping_mul_val_eq, word_val F.inv_lt, UScalar.size,
      UScalarTy.U64_numBits_eq]
  have hi20 : i20.val = (i19.val + carry3.val) % 2^64 := by
    rw [i20_post, core.num.U64.wrapping_add_val_eq, UScalar.size, UScalarTy.U64_numBits_eq]
  obtain ⟨lo, hlo⟩ : ∃ lo : Std.U64, lo.val + 2^64 * carry.val = i15.val + w.val * i.val + 0 :=
    ⟨_, __post⟩
  -- The columns: `s + w p` is the low word plus `2^64` times the result `t'`, whose top word is
  -- `s[4]` plus the last carry.
  set t' := r0.val + 2^64 * r1.val + 2^128 * r2.val + 2^192 * (i19.val + carry3.val) with ht'
  have hred : (signed5OfArray s).toNat + w.val * F.modulus.toNat = lo.val + 2^64 * t' := by
    rw [hwords, ht', hp]
    unfold Signed5.toNat
    clear * - hlo r0_post r1_post r2_post
    agrind
  clear * - hS hwords hw hi20 hlo hred ht htv ht'
  -- `amontredZ`'s `w` is the word `w`, since `s` and its low word agree modulo `2^64`.
  have hSmod : (signed5OfArray s).toNat * F.inv % 2^64 = w.val := by
    have h : (signed5OfArray s).toNat ≡ i15.val [MOD 2^64] := by
      rw [hwords]
      unfold Signed5.toNat Nat.ModEq
      agrind
    rw [hw]
    exact h.mul_right _
  have hZ : Inversion.amontredZ t.toInt F.modulus.toNat F.inv = (t' : ℤ) := by
    unfold Inversion.amontredZ
    dsimp only
    rw [← hS]
    have e1 : ((signed5OfArray s).toNat : ℤ) * F.inv % 2^64 =
        ((signed5OfArray s).toNat * F.inv % 2^64 : ℕ) := by norm_cast
    have e2 : ((signed5OfArray s).toNat : ℤ) + (w.val : ℤ) * F.modulus.toNat =
        ((signed5OfArray s).toNat + w.val * F.modulus.toNat : ℕ) := by norm_cast
    rw [e1, hSmod, e2, hred]
    scalar_tac
  -- By Lemma 10 the result is below `2^256`, so the top word's addition does not wrap.
  have ht'_lt : t' < 2^256 := by
    obtain ⟨-, -, -, h, -⟩ := Inversion.amontredZ_spec F t.toInt htv
    rw [hZ] at h
    exact_mod_cast h
  have htop : i19.val + carry3.val < 2^64 := by agrind
  have hres : limbsOfArray (Std.Array.make 4#usize [r0, r1, r2, i20] (by rfl)) =
      ⟨r0.val, r1.val, r2.val, i20.val⟩ := by
    simp [limbsOfArray]
  unfold Inversion.amontred
  rw [hres, hZ, Int.toNat_natCast, hi20, Nat.mod_eq_of_lt htop, ht']
  simp only [Limbs.ofNat, Limbs.mk.injEq]
  scalar_tac

end PastaCurves.Portable
