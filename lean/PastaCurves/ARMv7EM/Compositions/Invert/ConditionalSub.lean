import PastaCurves.ARMv7EM.Spec.Invert.CondSub
import PastaCurves.ARMv7EM.Backend
import PastaCurves.Inversion.Round
import PastaCurves.Portable.Spec

/-!
# ARMv7E-M conditional-subtraction model correspondence

The Aeneas external `asm.armv7em.cond_sub` calls the generated ARM block directly. The theorems
below carry that equality through the generated `InvertBlocks::cond_sub` method and establish the
arithmetic contract used by the inversion composition.
-/

namespace PastaCurves.ARMv7EM.Compositions.Invert.ConditionalSub

open _root_.Aeneas _root_.Aeneas.Std Result
open PastaCurves

/-- On arbitrary bounded values and moduli, the Aeneas external returns exactly
the words computed by the mechanically generated ARMv7E-M conditional-subtraction block.

This theorem identifies the two block models; it is not a correctness theorem for the translated
Rust inversion glue that calls the external model. -/
theorem cond_sub_eq_pure (value modulus : Limbs) (hv : value.Bounded) (hm : modulus.Bounded) :
    asm.armv7em.cond_sub (limbsArray value) (limbsArray modulus) =
      .ok (limbsArray (ARMv7EM.condSub value modulus)) := by
  unfold asm.armv7em.cond_sub
  rw [show pasta_curves.asm.armv7em.external.limbsOfArray (limbsArray value) = value by
    rw [show pasta_curves.asm.armv7em.external.limbsOfArray (limbsArray value) =
      limbsOfArray (limbsArray value) by rfl, limbsOfArray_limbsArray hv]]
  rw [show pasta_curves.asm.armv7em.external.limbsOfArray (limbsArray modulus) = modulus by
    rw [show pasta_curves.asm.armv7em.external.limbsOfArray (limbsArray modulus) =
      limbsOfArray (limbsArray modulus) by rfl, limbsOfArray_limbsArray hm]]
  rfl

/-- The generated ARMv7E-M `InvertBlocks::cond_sub` method returns the checked instruction model's
result. -/
theorem generated_cond_sub_eq_pure (value modulus : Limbs) (hv : value.Bounded)
    (hm : modulus.Bounded) :
    pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.cond_sub
        (limbsArray value) (limbsArray modulus) =
      .ok (limbsArray (ARMv7EM.condSub value modulus)) := by
  simpa only [pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.cond_sub]
    using cond_sub_eq_pure value modulus hv hm

/-- On arbitrary word arrays, the generated ARMv7E-M `InvertBlocks::cond_sub` method succeeds and
returns the checked instruction model applied to their limb readbacks. -/
theorem generated_cond_sub_eq_pure_of_array (value modulus : Std.Array Std.U64 4#usize) :
    pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.cond_sub value modulus =
      .ok (limbsArray (ARMv7EM.condSub (limbsOfArray value) (limbsOfArray modulus))) := by
  calc
    _ = pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.cond_sub
        (limbsArray (limbsOfArray value)) (limbsArray (limbsOfArray modulus)) := by
      rw [limbsArray_limbsOfArray value, limbsArray_limbsOfArray modulus]
    _ = _ := generated_cond_sub_eq_pure _ _ (limbsOfArray_bounded value)
      (limbsOfArray_bounded modulus)

/-- Read as a pure inversion block, the generated ARMv7E-M conditional subtraction is exactly the
checked instruction model. -/
theorem generatedInvertBlocks_condSub_eq (value modulus : Limbs) (hv : value.Bounded)
    (hm : modulus.Bounded) :
    ARMv7EM.generatedInvertBlocks.condSub value modulus = ARMv7EM.condSub value modulus := by
  rw [show ARMv7EM.generatedInvertBlocks.condSub value modulus =
      limbsOfArray ((Option.ofResult
        (pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesInversionInvertBlocks.cond_sub
          (limbsArray value) (limbsArray modulus))).getD
            (Std.Array.repeat 4#usize 0#u64)) by rfl]
  rw [generated_cond_sub_eq_pure value modulus hv hm]
  simp [Option.ofResult,
    limbsOfArray_limbsArray (ARMv7EM.condSub_spec value modulus hv hm _ rfl).1]

/-- The generated ARMv7E-M conditional-subtraction method satisfies the arithmetic contract used by
the inversion composition. -/
theorem generatedInvertBlocks_condSub_spec (F : PastaField) (value : Limbs)
    (hv : value.Bounded) :
    (ARMv7EM.generatedInvertBlocks.condSub value F.modulus).Bounded ∧
      ((value.toNat < F.modulus.toNat ∧
          (ARMv7EM.generatedInvertBlocks.condSub value F.modulus).toNat = value.toNat) ∨
        (F.modulus.toNat ≤ value.toNat ∧
          (ARMv7EM.generatedInvertBlocks.condSub value F.modulus).toNat +
            F.modulus.toNat = value.toNat)) := by
  rw [generatedInvertBlocks_condSub_eq value F.modulus hv F.bounded]
  exact ARMv7EM.condSub_pasta_spec F value hv

end PastaCurves.ARMv7EM.Compositions.Invert.ConditionalSub
