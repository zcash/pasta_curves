import PastaCurves.Portable.Spec
import PastaCurves.Inversion.SignMag

/-!
# The translated `sign_mag` satisfies its contract

`sign_mag` of `src/inversion/portable.rs` takes the four entries of a transition matrix as
two's-complement words, and returns each entry's magnitude and sign mask, the form in which the row
blocks take the matrix (`Inversion/SignMag.lean`). An entry below `2^63` in magnitude has its sign
in bit 63, so `sign_mask` gives its sign mask (`sign_mask_eq`). Xoring the word with that mask and
subtracting the mask then negates a negative entry modulo `2^64`, and leaves a nonnegative one
unchanged (`magnitude_val`). `sign_mag_spec` shows that `sign_mag` satisfies the `signMag` field of
`InvertBlocks.Spec`.
-/

namespace PastaCurves.Portable

open Aeneas Aeneas.Std
open Inversion (signMask)

/-- A word that carries `z`, with `|z| < 2^63`, is below `2^63` exactly when `z` is nonnegative,
so the mask `s` of its sign is `z`'s sign mask. -/
theorem sign_mask_eq {w s : Std.U64} {z : ℤ} (hz : |z| < 2^63) (ew : (w.val : ℤ) = z % 2^64)
    (hs : s.val = if w.val < 2^63 then 0 else 2^64 - 1) :
    s.val = signMask z := by
  rw [abs_lt] at hz
  agrind [signMask]

/-- Xoring a word `w` that carries `z` with `z`'s sign mask `s`, then subtracting the mask, gives
the magnitude of `z`: `w` itself for a nonnegative `z`, and `2^64 - w = -z` for a negative one. -/
theorem magnitude_val {w s x : Std.U64} {z : ℤ} (hz : |z| < 2^63) (ew : (w.val : ℤ) = z % 2^64)
    (hs : s.val = signMask z) (hx : x.val = w.val ^^^ s.val) :
    (core.num.U64.wrapping_sub x s).val = z.natAbs := by
  rw [abs_lt] at hz
  rw [core.num.U64.wrapping_sub_val_eq, hx, hs]
  simp only [UScalar.size, UScalarTy.U64_numBits_eq]
  unfold signMask
  split_ifs with hneg
  · have hones := Inversion.eorw_ones w.val (by scalar_tac)
    unfold eorw at hones
    rw [hones]
    agrind
  · rw [Nat.xor_zero]
    agrind

open pasta_curves.inversion.portable in
/-- `sign_mag` satisfies its contract, the `signMag` field of `InvertBlocks.Spec`: on words that
carry four entries below `2^63` in magnitude, it returns their magnitudes and sign masks. -/
theorem sign_mag_spec (u v q r : Std.U64) (zu zv zq zr : ℤ) (hzu : |zu| < 2^63)
    (hzv : |zv| < 2^63) (hzq : |zq| < 2^63) (hzr : |zr| < 2^63)
    (eu : (u.val : ℤ) = zu % 2^64) (ev : (v.val : ℤ) = zv % 2^64)
    (eq : (q.val : ℤ) = zq % 2^64) (er : (r.val : ℤ) = zr % 2^64) :
    Backend.Insts.Pasta_curvesInversionInvertBlocks.sign_mag u v q r
    ⦃ (res : Std.Array Std.U64 8#usize) =>
      res[0].val = zu.natAbs ∧ res[1].val = zv.natAbs ∧ res[2].val = zq.natAbs ∧
        res[3].val = zr.natAbs ∧ res[4].val = signMask zu ∧ res[5].val = signMask zv ∧
        res[6].val = signMask zq ∧ res[7].val = signMask zr ⦄ := by
  unfold Backend.Insts.Pasta_curvesInversionInvertBlocks.sign_mag
  step*
  have hmu := sign_mask_eq hzu eu i_post
  have hmv := sign_mask_eq hzv ev i1_post
  have hmq := sign_mask_eq hzq eq i2_post
  have hmr := sign_mask_eq hzr er i3_post
  subst su sv sq sr i5 i7 i9 i11
  exact ⟨magnitude_val hzu eu hmu i4_post, magnitude_val hzv ev hmv i6_post,
    magnitude_val hzq eq hmq i8_post, magnitude_val hzr er hmr i10_post,
    hmu, hmv, hmq, hmr⟩

end PastaCurves.Portable
