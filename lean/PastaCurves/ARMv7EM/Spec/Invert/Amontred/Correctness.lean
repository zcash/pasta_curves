import PastaCurves.ARMv7EM.Spec.Invert.InversionMacWords
import PastaCurves.ARMv7EM.Backend
import PastaCurves.ARMv7EM.Compositions.Shared.Conversions
import PastaCurves.ARMv7EM.Compositions.Invert.Amontred
import PastaCurves.Portable.Amontred

/-!
# Correctness of the translated ARMv7E-M almost-Montgomery reduction

The ARMv7E-M implementation splits each 64-bit multiply-accumulate into four 32-bit words.  The
first theorem connects the Aeneas model of that opaque helper to the mechanically transcribed
instruction block.  The second theorem proves that the Rust wrapper reassembles those words into
the low and high halves of the exact 128-bit sum.  The final theorem follows the translated
`amontred` composition through its portable shift and addition helpers and the sparse Pasta modulus.
-/

set_option exponentiation.threshold 400

namespace PastaCurves.ARMv7EM

open _root_.Aeneas _root_.Aeneas.Std

private theorem orShifted_eq (low high : Nat) (hlow : low < 2^32) :
    low ||| (high * 2^32) = low + 2^32 * high := by
  calc
    _ = (low ||| (high * 2^32)) % 2^32 +
          2^32 * ((low ||| (high * 2^32)) / 2^32) :=
      (Nat.mod_add_div (low ||| (high * 2^32)) (2^32)).symm
    _ = low + 2^32 * high := by
      rw [Nat.or_mod_two_pow, Nat.or_div_two_pow]
      norm_num at hlow ⊢
      rw [Nat.mod_eq_of_lt hlow, Nat.div_eq_of_lt hlow]
      simp

private theorem joinBitVec_eq (low high : BitVec 32) :
    ((low.setWidth 64) ||| ((high.setWidth 64) <<< 32)).toNat =
      low.toNat + 2^32 * high.toNat := by
  rw [BitVec.toNat_or, BitVec.toNat_shiftLeft, BitVec.toNat_setWidth,
    BitVec.toNat_setWidth]
  have hlow : low.toNat < 2^32 := by omega
  have hhigh : high.toNat < 2^32 := by omega
  have hlow64 : low.toNat % 2^64 = low.toNat := Nat.mod_eq_of_lt (by omega)
  have hhigh64 : high.toNat % 2^64 = high.toNat := Nat.mod_eq_of_lt (by omega)
  have hshift64 : high.toNat * 2^32 % 2^64 = high.toNat * 2^32 :=
    Nat.mod_eq_of_lt (by omega)
  rw [hlow64, hhigh64]
  simp only [Nat.shiftLeft_eq]
  rw [hshift64]
  simpa [Nat.mul_comm] using orShifted_eq low.toNat high.toNat hlow

private theorem joinU32_eq (low high : Std.U32) (shifted joined : Std.U64)
    (hshifted : shifted.bv = (core.convert.num.FromU64U32.from high).bv <<< 32)
    (hjoined : joined.bv = (core.convert.num.FromU64U32.from low).bv ||| shifted.bv) :
    joined.val = low.val + 2^32 * high.val := by
  calc
    _ = ((core.convert.num.FromU64U32.from low).bv ||| shifted.bv).toNat := by
      simpa only [UScalar.val] using congrArg BitVec.toNat hjoined
    _ = ((core.convert.num.FromU64U32.from low).bv |||
          ((core.convert.num.FromU64U32.from high).bv <<< 32)).toNat := by rw [hshifted]
    _ = low.val + 2^32 * high.val := by
      simpa only [core.convert.num.FromU64U32.from_bv_eq, UScalar.val] using
        joinBitVec_eq low.bv high.bv

private theorem wordQuad_unique {x0 x1 x2 x3 y0 y1 y2 y3 total : Nat}
    (hx : x0 + 2^32 * x1 + 2^64 * x2 + 2^96 * x3 = total)
    (hy : y0 + 2^32 * y1 + 2^64 * y2 + 2^96 * y3 = total)
    (hx0 : x0 < 2^32) (hx1 : x1 < 2^32) (hx2 : x2 < 2^32) (_hx3 : x3 < 2^32)
    (hy0 : y0 < 2^32) (hy1 : y1 < 2^32) (hy2 : y2 < 2^32) (hy3 : y3 < 2^32) :
    (x0, x1, x2, x3) = (y0, y1, y2, y3) := by
  have h0 : x0 = y0 := by omega
  subst y0
  have h1 : x1 = y1 := by omega
  subst y1
  have h2 : x2 = y2 := by omega
  subst y2
  have h3 : x3 = y3 := by omega
  subst y3
  rfl

open pasta_curves in
/-- The external model of the opaque assembly helper returns exactly the words computed by the
mechanically transcribed ARM block whenever the complete multiply-accumulate fits in 128 bits. -/
@[step]
theorem inversion_mac_words_spec (r0 r1 b0 b1 c0 c1 r2 r3 : Std.U32) :
    asm.armv7em.inversion_mac_words r0 r1 b0 b1 c0 c1 r2 r3 ⦃ res =>
      (res.1.val, res.2.1.val, res.2.2.1.val, res.2.2.2.val) =
        inversionMacWords r0.val r1.val b0.val b1.val c0.val c1.val r2.val r3.val ⦄ := by
  have hb : b0.val + 2^32 * b1.val < 2^64 := by scalar_tac
  have hc : c0.val + 2^32 * c1.val < 2^64 := by scalar_tac
  have hbc := Nat.mul_le_mul (Nat.le_sub_one_of_lt hb) (Nat.le_sub_one_of_lt hc)
  have hsum : (r0.val + 2^32 * r1.val) + (b0.val + 2^32 * b1.val) *
      (c0.val + 2^32 * c1.val) + (r2.val + 2^32 * r3.val) < 2^128 := by
    norm_num at hbc ⊢
    scalar_tac
  unfold asm.armv7em.inversion_mac_words
  step*
  have ha : a.val = r0.val + 2^32 * r1.val :=
    joinU32_eq r0 r1 r1High a r1High_post1 a_post1
  have hb : b.val = b0.val + 2^32 * b1.val :=
    joinU32_eq b0 b1 b1High b b1High_post1 b_post1
  have hc : c.val = c0.val + 2^32 * c1.val :=
    joinU32_eq c0 c1 c1High c c1High_post1 c_post1
  have hcarry : carry.val = r2.val + 2^32 * r3.val :=
    joinU32_eq r2 r3 r3High carry r3High_post1 carry_post1
  let model := inversionMacWords r0.val r1.val b0.val b1.val c0.val c1.val r2.val r3.val
  obtain ⟨hmodel, hm0, hm1, hm2, hm3⟩ :=
    inversionMacWords_spec r0.val r1.val b0.val b1.val c0.val c1.val r2.val r3.val
      (by scalar_tac) (by scalar_tac) (by scalar_tac) (by scalar_tac)
      (by scalar_tac) (by scalar_tac) (by scalar_tac) (by scalar_tac) hsum model rfl
  have hlowHigh : lowHigh.val < 2^32 := by
    rw [lowHigh_post, Nat.shiftRight_eq_div_pow]
    apply Nat.div_lt_of_lt_mul
    simpa using low.hBounds
  have hlow : low.val = (UScalar.cast .U32 low).val +
      2^32 * (UScalar.cast .U32 lowHigh).val := by
    rw [UScalar.cast_val_eq, UScalar.cast_val_eq, UScalarTy.U32_numBits_eq,
      Nat.mod_eq_of_lt hlowHigh, lowHigh_post, Nat.shiftRight_eq_div_pow]
    exact (Nat.mod_add_div low.val (2^32)).symm
  have hhighHigh : highHigh.val < 2^32 := by
    rw [highHigh_post, Nat.shiftRight_eq_div_pow]
    apply Nat.div_lt_of_lt_mul
    simpa using high.hBounds
  have hhigh : high.val = (UScalar.cast .U32 high).val +
      2^32 * (UScalar.cast .U32 highHigh).val := by
    rw [UScalar.cast_val_eq, UScalar.cast_val_eq, UScalarTy.U32_numBits_eq,
      Nat.mod_eq_of_lt hhighHigh, highHigh_post, Nat.shiftRight_eq_div_pow]
    exact (Nat.mod_add_div high.val (2^32)).symm
  apply wordQuad_unique (total :=
    (r0.val + 2^32 * r1.val) + (b0.val + 2^32 * b1.val) *
      (c0.val + 2^32 * c1.val) + (r2.val + 2^32 * r3.val))
  · rw [← ha, ← hb, ← hc, ← hcarry, ← low_post, hlow, hhigh]
    ring
  · exact hmodel
  · rw [UScalar.cast_val_eq, UScalarTy.U32_numBits_eq]
    exact Nat.mod_lt _ (Nat.two_pow_pos _)
  · rw [UScalar.cast_val_eq, UScalarTy.U32_numBits_eq]
    exact Nat.mod_lt _ (Nat.two_pow_pos _)
  · rw [UScalar.cast_val_eq, UScalarTy.U32_numBits_eq]
    exact Nat.mod_lt _ (Nat.two_pow_pos _)
  · rw [UScalar.cast_val_eq, UScalarTy.U32_numBits_eq]
    exact Nat.mod_lt _ (Nat.two_pow_pos _)
  · exact hm0
  · exact hm1
  · exact hm2
  · exact hm3

open pasta_curves in
/-- The translated Rust wrapper splits its four 64-bit inputs into words, calls the transcribed
multiply-accumulate block, and reassembles the exact 128-bit result. -/
@[step]
theorem inversion_mac_spec (a b c carry : Std.U64)
    (hsum : a.val + b.val * c.val + carry.val < 2^128) :
    asm.armv7em.inversion_mac a b c carry ⦃ res =>
      res.1.val + 2^64 * res.2.val = a.val + b.val * c.val + carry.val ⦄ := by
  unfold asm.armv7em.inversion_mac
  step*
  have split (x : Std.U64) (lo : Std.U32) (hi64 : Std.U64) (hi : Std.U32)
      (hlo : lo = UScalar.cast .U32 x) (hhi64 : hi64.val = x.val >>> 32)
      (hhi : hi = UScalar.cast .U32 hi64) : x.val = lo.val + 2^32 * hi.val := by
    have hhi64lt : hi64.val < 2^32 := by
      rw [hhi64, Nat.shiftRight_eq_div_pow]
      apply Nat.div_lt_of_lt_mul
      simpa using x.hBounds
    rw [hlo, hhi, UScalar.cast_val_eq, UScalar.cast_val_eq,
      UScalarTy.U32_numBits_eq, Nat.mod_eq_of_lt hhi64lt, hhi64,
      Nat.shiftRight_eq_div_pow]
    exact (Nat.mod_add_div x.val (2^32)).symm
  have ha := split a r0 i r1 r0_post i_post r1_post
  have hb := split b b0 i1 b1 b0_post i1_post b1_post
  have hc := split c c0 i2 c1 c0_post i2_post c1_post
  have hcarry := split carry r2 i3 r3 r2_post i3_post r3_post
  have hwordSum : (r0.val + 2^32 * r1.val) + (b0.val + 2^32 * b1.val) *
      (c0.val + 2^32 * c1.val) + (r2.val + 2^32 * r3.val) < 2^128 := by
    rw [← ha, ← hb, ← hc, ← hcarry]
    exact hsum
  have hmodel := inversionMacWords_spec r0.val r1.val b0.val b1.val c0.val c1.val
    r2.val r3.val r0.hBounds r1.hBounds b0.hBounds b1.hBounds
    c0.hBounds c1.hBounds r2.hBounds r3.hBounds hwordSum
    (inversionMacWords r0.val r1.val b0.val b1.val c0.val c1.val r2.val r3.val) rfl
  obtain ⟨hmodelValue, -⟩ := hmodel
  simp only [Std.lift, Std.bind_ok]
  step*
  have hlo := joinU32_eq r01 r11 i6
    (core.convert.num.FromU64U32.from r01 ||| i6) i6_post1 rfl
  have hhi := joinU32_eq r21 r31 i10
    (core.convert.num.FromU64U32.from r21 ||| i10) i10_post1 rfl
  rw [hlo, hhi]
  have hwords : r01.val + 2^32 * r11.val + 2^64 * r21.val + 2^96 * r31.val =
      a.val + b.val * c.val + carry.val := by
    rw [← r01_post] at hmodelValue
    simpa only [ha, hb, hc, hcarry] using hmodelValue
  calc
    r01.val + 2^32 * r11.val + 2^64 * (r21.val + 2^32 * r31.val) =
        r01.val + 2^32 * r11.val + 2^64 * r21.val + 2^96 * r31.val := by ring
    _ = _ := hwords

open pasta_curves in
/-- At a Pasta field, the generated ARMv7E-M almost-Montgomery reduction method succeeds for every
five-word input. Its only explicit failure path tests the field's third modulus limb, which is zero
by the Pasta modulus shape. -/
theorem amontred_ok (F : PastaField) (t : Std.Array Std.U64 5#usize) :
    asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.amontred
        t (limbsArray F.modulus) (word F.inv) ⦃ _ => True ⦄ := by
  unfold asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.amontred
  unfold asm.armv7em.amontred
  step*
  have hzero : i4.val = 0 := by
    rw [i4_post]
    simpa [limbsArray, F.shape.1] using (word_val (x := 0) (by norm_num))
  exact UScalar.eq_of_val_eq hzero

open pasta_curves in
/-- The generated ARMv7E-M almost-Montgomery reduction method returns the shared arithmetic model
on the bounded inputs required by the inversion composition. -/
theorem amontred_spec (F : PastaField) (t : Signed5) (ht : t.Bounded)
    (htv : |t.toInt| < 2^315) :
    asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.amontred
        (signed5Array t) (limbsArray F.modulus) (word F.inv)
      ⦃ res => limbsOfArray res = Inversion.amontred t F.modulus F.inv ⦄ := by
  unfold asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.amontred
  unfold asm.armv7em.amontred
  step*
  have hzero : i4.val = 0 := by
    rw [i4_post]
    simpa [limbsArray, F.shape.1] using (word_val (x := 0) (by norm_num))
  have hi4 : i4 = 0#u64 := UScalar.eq_of_val_eq hzero
  simp only [hi4]
  · -- The shifted modulus is the exact five-word representation of `2^61 p`.
    have hp61 : (signed5OfArray a).toNat = 2^61 * F.modulus.toNat := by
      rw [a_post, limbsOfArray_limbsArray F.bounded]
    obtain ⟨hp0, hp1, hp2, hp3⟩ := F.bounded
    have hp : F.modulus.toNat =
        i1.val + 2^64 * i3.val + 2^128 * i4.val + 2^192 * i7.val := by
      simp [Limbs.toNat, i1_post, i3_post, i4_post, i7_post, limbsArray,
        word_val hp0, word_val hp1, word_val hp2, word_val hp3]
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
    have hwords : signed5OfArray s = ⟨i.val, i2.val, i5.val, i6.val, i8.val⟩ := by
      simp [signed5OfArray, i_post, i2_post, i5_post, i6_post, i8_post]
    have hw : w.val = i.val * F.inv % 2^64 := by
      rw [w_post, core.num.U64.wrapping_mul_val_eq, word_val F.inv_lt, UScalar.size,
        UScalarTy.U64_numBits_eq]
    have hi9 : i9.val = (i8.val + carry3.val) % 2^64 := by
      rw [i9_post, core.num.U64.wrapping_add_val_eq, UScalar.size,
        UScalarTy.U64_numBits_eq]
    obtain ⟨lo, hlo⟩ : ∃ lo : Std.U64,
        lo.val + 2^64 * carry.val = i.val + w.val * i1.val + 0 := ⟨_, __post⟩
    have hr1 : r1.val + 2^64 * carry2.val = i5.val + carry1.val := by
      by_cases hover : i5.val + carry1.val > UScalar.max UScalarTy.U64
      · rw [if_pos hover] at r1_post
        rw [carry2_post, if_pos r1_post.2]
        simpa [U64.size, U64.numBits, UScalarTy.U64_numBits_eq] using r1_post.1
      · rw [if_neg hover] at r1_post
        have hfalse : ¬overflow = true := by simp [r1_post.2]
        rw [carry2_post, if_neg hfalse, r1_post.1]
        simp
    set t' := r0.val + 2^64 * r1.val + 2^128 * r2.val +
      2^192 * (i8.val + carry3.val) with ht'
    have hred : (signed5OfArray s).toNat + w.val * F.modulus.toNat =
        lo.val + 2^64 * t' := by
      rw [hwords, ht', hp]
      have hmodulusZero : i4.val = 0 := by
        rw [i4_post]
        simpa [limbsArray, F.shape.1] using (word_val (x := 0) (by norm_num))
      rw [hmodulusZero]
      unfold Signed5.toNat
      clear * - hlo r0_post hr1 r2_post
      agrind
    clear * - hS hwords hw hi9 hlo hred ht htv ht'
    have hSmod : (signed5OfArray s).toNat * F.inv % 2^64 = w.val := by
      have h : (signed5OfArray s).toNat ≡ i.val [MOD 2^64] := by
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
    have ht'_lt : t' < 2^256 := by
      obtain ⟨-, -, -, h, -⟩ := Inversion.amontredZ_spec F t.toInt htv
      rw [hZ] at h
      exact_mod_cast h
    have htop : i8.val + carry3.val < 2^64 := by agrind
    have hres : limbsOfArray (Std.Array.make 4#usize [r0, r1, r2, i9] (by rfl)) =
        ⟨r0.val, r1.val, r2.val, i9.val⟩ := by
      simp [limbsOfArray]
    unfold Inversion.amontred
    rw [hres, hZ, Int.toNat_natCast, hi9, Nat.mod_eq_of_lt htop, ht']
    simp only [Limbs.ofNat, Limbs.mk.injEq]
    scalar_tac

open pasta_curves in
/-- At a Pasta field, the generated ARMv7E-M method succeeds on every bounded input in the
almost-Montgomery reduction contract and returns exactly the model's words. -/
theorem amontred_eq_ok (F : PastaField) (t : Signed5) (ht : t.Bounded)
    (htv : |t.toInt| < 2^315) :
    asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.amontred
        (signed5Array t) (limbsArray F.modulus) (word F.inv) =
      .ok (limbsArray (Inversion.amontred t F.modulus F.inv)) := by
  obtain ⟨res, hres, hvalue⟩ := WP.spec_imp_exists (amontred_spec F t ht htv)
  calc
    _ = .ok res := hres
    _ = .ok (limbsArray (limbsOfArray res)) := by rw [limbsArray_limbsOfArray]
    _ = _ := by rw [hvalue]

/-- The generated ARMv7E-M pure block computes the shared almost-Montgomery reduction on the
bounded inputs required by the inversion composition. -/
theorem generatedInvertBlocks_amontred_spec (F : PastaField) (t : Signed5) (ht : t.Bounded)
    (htv : |t.toInt| < 2^315) :
    generatedInvertBlocks.amontred t F.modulus F.inv =
      Inversion.amontred t F.modulus F.inv := by
  rw [show generatedInvertBlocks.amontred t F.modulus F.inv =
      limbsOfArray ((Option.ofResult
        (pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.amontred
          (signed5Array t) (limbsArray F.modulus) (word F.inv))).getD
            (Std.Array.repeat 4#usize 0#u64)) by rfl]
  rw [amontred_eq_ok F t ht htv]
  simp only [Option.ofResult]
  exact limbsOfArray_limbsArray (Inversion.amontred_spec F t htv).1

end PastaCurves.ARMv7EM
