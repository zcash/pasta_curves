import PastaCurves.AArch64.Spec.Divstep59
import PastaCurves.AArch64.Spec.FgRow
import PastaCurves.AArch64.Spec.Amontred
import PastaCurves.AArch64.Spec.CondSub
import PastaCurves.AArch64.Compositions
import PastaCurves.Inversion.Composition

/-!
# The AArch64 blocks meet the composition's specification

`invertBlocks_spec` packages the six block theorems as `InvertBlocks.Spec`, the record that the
shared `invert_eq_model` and `invert_entry_spec` of `Inversion/Composition.lean` take. Each field
is the block's theorem with the block's result named by `rfl`.
-/

namespace PastaCurves.AArch64

/-- The AArch64 blocks compute what the composition needs, at either field. -/
theorem invertBlocks_spec (F : PastaField) : invertBlocks.Spec F where
  divstep59 := fun d f0 g0 s hsf hsd hsD ed ef0 eg0 =>
    divstep59Block_spec d f0 g0 s hsf hsd hsD ed ef0 eg0 _ rfl
  signMag := fun a b c d m00 m01 m10 m11 ha hb hc hd em00 em01 em10 em11 =>
    signMagBlock_spec a b c d m00 m01 m10 m11 ha hb hc hd em00 em01 em10 em11 _ rfl
  fgRow := fun a b f g m0 m1 s0 s1 hf hg hfv hgv hab hrep0 hrep1 =>
    fgRowBlock_spec a b f g m0 m1 s0 s1 hf hg hfv hgv hab hrep0 hrep1 _ rfl
  uvRow := fun a b u v m0 m1 s0 s1 hu hv hab hrep0 hrep1 =>
    uvRowBlock_spec a b u v m0 m1 s0 s1 hu hv hab hrep0 hrep1 _ rfl
  amontred := fun t ht htv => amontredBlock_spec F t F.modulus F.inv rfl rfl ht htv _ rfl
  condSub := fun value hv => condSubBlock_spec value F.modulus hv F.bounded F.shape _ rfl

end PastaCurves.AArch64
