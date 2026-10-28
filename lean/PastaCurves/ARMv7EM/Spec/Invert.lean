import PastaCurves.ARMv7EM.Backend
import PastaCurves.ARMv7EM.Spec.Invert.Divstep59
import PastaCurves.ARMv7EM.Spec.Invert.InversionMacWords
import PastaCurves.ARMv7EM.Spec.Invert.Amontred
import PastaCurves.ARMv7EM.Compositions.Invert.ConditionalSub
import PastaCurves.ARMv7EM.Compositions.Invert.Divstep59
import PastaCurves.Portable.SignMag
import PastaCurves.Portable.Row

/-!
# The ARMv7E-M inversion blocks meet the composition's specification

The ARMv7E-M backend delegates `sign_mag`, `fg_row`, and `de_row` to the portable Rust backend.
Their fields of `InvertBlocks.Spec` therefore follow from the corresponding Aeneas proofs
`PastaCurves.Portable.sign_mag_spec`, `PastaCurves.Portable.fg_row_spec`, and
`PastaCurves.Portable.de_row_spec`. The generated `cond_sub` method is connected to its checked
instruction model by `Compositions.Invert.ConditionalSub`; `amontred` is proved by its translated
Rust method and the checked multiply-accumulate block. `generatedInvertBlocks_divstep59_spec`
connects the generated Rust method to the mechanically transcribed ARM composition and its block
proof.

`generatedInvertBlocks_spec` packages all six method contracts for the shared inversion
composition, using its signed-32 divstep state bound.
-/

namespace PastaCurves.ARMv7EM

open _root_.Aeneas _root_.Aeneas.Std
open Inversion (State divsteps M SignMagRep)

/-- A natural number equal to an integer residue modulo `2^64` is a word. -/
private theorem lt_of_eq_emod {n : ℕ} {z : ℤ} (h : (n : ℤ) = z % 2^64) : n < 2^64 := by
  have := Int.emod_lt_of_pos z (by norm_num : (0 : ℤ) < 2^64)
  agrind

/-- The generated ARMv7E-M `divstep59` method satisfies the arithmetic contract used by the
shared inversion composition. -/
theorem generatedInvertBlocks_divstep59_spec :
    ∀ (two_delta f0 g0 : ℕ) (s : State), s.f % 2 = 1 → s.two_delta % 2 = 1 →
      InvertBlocks.divstep59StateBound s → (two_delta : ℤ) = s.two_delta % 2^64 →
      (f0 : ℤ) = s.f % 2^64 → (g0 : ℤ) = s.g % 2^64 →
      ((generatedInvertBlocks.divstep59 two_delta f0 g0).two_delta : ℤ) =
          (divsteps 59 s).two_delta % 2^64 ∧
        ((generatedInvertBlocks.divstep59 two_delta f0 g0).u : ℤ) = (M 59 s).u % 2^64 ∧
        ((generatedInvertBlocks.divstep59 two_delta f0 g0).v : ℤ) = (M 59 s).v % 2^64 ∧
        ((generatedInvertBlocks.divstep59 two_delta f0 g0).q : ℤ) = (M 59 s).q % 2^64 ∧
        ((generatedInvertBlocks.divstep59 two_delta f0 g0).r : ℤ) = (M 59 s).r % 2^64 := by
  intro two_delta f0 g0 s hsf hsd hsD ed ef0 eg0
  rw [show generatedInvertBlocks.divstep59 two_delta f0 g0 =
      Portable.divstep59OfArray ((Option.ofResult
        (pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.divstep59
          (word two_delta) (word f0) (word g0))).getD
            (Std.Array.repeat 5#usize 0#u64)) by rfl]
  obtain ⟨res, hres, hvalue⟩ := WP.spec_imp_exists
    (Compositions.Invert.Divstep59.divstep59_eq_divstep59Block
      (word two_delta) (word f0) (word g0))
  rw [hres]
  change ((Portable.divstep59OfArray res).two_delta : ℤ) = _ ∧
    ((Portable.divstep59OfArray res).u : ℤ) = _ ∧
    ((Portable.divstep59OfArray res).v : ℤ) = _ ∧
    ((Portable.divstep59OfArray res).q : ℤ) = _ ∧
    ((Portable.divstep59OfArray res).r : ℤ) = _
  rw [hvalue, word_val (lt_of_eq_emod ed), word_val (lt_of_eq_emod ef0),
    word_val (lt_of_eq_emod eg0)]
  exact divstep59Block_spec two_delta f0 g0 s hsf hsd (by
    unfold InvertBlocks.divstep59StateBound at hsD
    omega) ed ef0 eg0 _ rfl

/-- The result of a translated call satisfying a Hoare triple, read back with any default,
satisfies the triple's postcondition. -/
private theorem getD_of_spec {α : Type} {m : Result α} {P : α → Prop} (h : m ⦃ a => P a ⦄)
    (d : α) : P ((Option.ofResult m).getD d) := by
  obtain ⟨a, rfl, ha⟩ := WP.spec_imp_exists h
  simpa [Option.ofResult] using ha

/-- The generated delegated `sign_mag` satisfies its field of `InvertBlocks.Spec`, by the portable
`sign_mag_spec` theorem. -/
theorem generatedInvertBlocks_signMag_spec :
    ∀ (zu zv zq zr : ℤ) (u v q r : ℕ), |zu| < 2^63 → |zv| < 2^63 → |zq| < 2^63 →
      |zr| < 2^63 → (u : ℤ) = zu % 2^64 → (v : ℤ) = zv % 2^64 →
      (q : ℤ) = zq % 2^64 → (r : ℤ) = zr % 2^64 →
      generatedInvertBlocks.signMag u v q r =
        ⟨zu.natAbs, zv.natAbs, zq.natAbs, zr.natAbs, Inversion.signMask zu,
          Inversion.signMask zv, Inversion.signMask zq, Inversion.signMask zr⟩ := by
  intro zu zv zq zr u v q r hzu hzv hzq hzr eu ev eq er
  rw [← word_val (lt_of_eq_emod eu)] at eu
  rw [← word_val (lt_of_eq_emod ev)] at ev
  rw [← word_val (lt_of_eq_emod eq)] at eq
  rw [← word_val (lt_of_eq_emod er)] at er
  have h := getD_of_spec
    (Portable.sign_mag_spec _ _ _ _ zu zv zq zr hzu hzv hzq hzr eu ev eq er)
    (Std.Array.repeat 8#usize 0#u64)
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7⟩ := h
  simp only [generatedInvertBlocks, Portable.pureBlocks, inversionBlocks,
    pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.sign_mag,
    Portable.signMagOfArray]
  rw [h0, h1, h2, h3, h4, h5, h6, h7]

/-- The generated delegated `fg_row` satisfies its field of `InvertBlocks.Spec`, by the portable
`fg_row_spec` theorem. -/
theorem generatedInvertBlocks_fgRow_spec :
    ∀ (a b : ℤ) (f g : Signed5) (m0 m1 s0 s1 : ℕ), f.Bounded → g.Bounded →
      |f.toInt| < 2^256 → |g.toInt| < 2^256 → |a| + |b| ≤ 2^63 →
      SignMagRep m0 s0 a → SignMagRep m1 s1 b →
      (generatedInvertBlocks.fgRow f g m0 m1 s0 s1).Bounded ∧
        (generatedInvertBlocks.fgRow f g m0 m1 s0 s1).toInt =
          (a * f.toInt + b * g.toInt) / 2^59 := by
  intro a b f g m0 m1 s0 s1 hf hg hfv hgv hab ha hb
  have := abs_nonneg a
  have := abs_nonneg b
  obtain ⟨hm0, hs0⟩ := Inversion.SignMagRep.lt m0 s0 a ha (by agrind)
  obtain ⟨hm1, hs1⟩ := Inversion.SignMagRep.lt m1 s1 b hb (by agrind)
  rw [← word_val hm0, ← word_val hs0] at ha
  rw [← word_val hm1, ← word_val hs1] at hb
  exact getD_of_spec (Portable.fg_row_spec a b f g _ _ _ _ hf hg hfv hgv hab ha hb) _

/-- The generated delegated `de_row` satisfies its field of `InvertBlocks.Spec`, by the portable
`de_row_spec` theorem. -/
theorem generatedInvertBlocks_deRow_spec :
    ∀ (a b : ℤ) (d e : Limbs) (m0 m1 s0 s1 : ℕ), d.Bounded → e.Bounded →
      |a| + |b| ≤ 2^63 → SignMagRep m0 s0 a → SignMagRep m1 s1 b →
      (generatedInvertBlocks.deRow d e m0 m1 s0 s1).Bounded ∧
        (generatedInvertBlocks.deRow d e m0 m1 s0 s1).toInt = a * d.toNat + b * e.toNat := by
  intro a b d e m0 m1 s0 s1 hd he hab ha hb
  have := abs_nonneg a
  have := abs_nonneg b
  obtain ⟨hm0, hs0⟩ := Inversion.SignMagRep.lt m0 s0 a ha (by agrind)
  obtain ⟨hm1, hs1⟩ := Inversion.SignMagRep.lt m1 s1 b hb (by agrind)
  rw [← word_val hm0, ← word_val hs0] at ha
  rw [← word_val hm1, ← word_val hs1] at hb
  exact getD_of_spec (Portable.de_row_spec a b d e _ _ _ _ hd he hab ha hb) _

/-- The generated ARMv7E-M blocks compute what the shared inversion composition needs at either
field. -/
theorem generatedInvertBlocks_spec (F : PastaField) : generatedInvertBlocks.Spec F where
  divstep59 := generatedInvertBlocks_divstep59_spec
  signMag := generatedInvertBlocks_signMag_spec
  fgRow := generatedInvertBlocks_fgRow_spec
  deRow := generatedInvertBlocks_deRow_spec
  amontred := generatedInvertBlocks_amontred_spec F
  condSub :=
    PastaCurves.ARMv7EM.Compositions.Invert.ConditionalSub.generatedInvertBlocks_condSub_spec F

end PastaCurves.ARMv7EM
