import PastaCurves.ARMv7EM.Spec.FromMont.Round
import PastaCurves.ARMv7EM.Spec.FromMont.QuotientStep
import PastaCurves.ARMv7EM.Spec.FromMont.Prefix
import PastaCurves.ARMv7EM.Spec.FromMont.Suffix
import PastaCurves.ARMv7EM.Spec.FromMont.RoundComposition
import PastaCurves.ARMv7EM.Spec.FromMont.FinalReduction
import PastaCurves.ARMv7EM.Spec.FromMont.FinalReduction.Block
import PastaCurves.ARMv7EM.Spec.FromMont.Correctness
import PastaCurves.ARMv7EM.Spec.FromMont.Block

/-!
# ARMv7E-M conversion from Montgomery form

The prefix and suffix instruction proofs establish the two halves of a conversion round, and
`RoundComposition` proves their generated caller. The final-reduction modules similarly prove the
subtraction and add-back helpers and their generated caller. `Block` proves the outer generated
`fromMontBlock` by composing eight checked rounds with the checked final reduction.
-/
