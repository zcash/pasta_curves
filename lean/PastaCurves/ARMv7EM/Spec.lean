import PastaCurves.ARMv7EM.Spec.Arithmetic
import PastaCurves.ARMv7EM.Spec.Mul
import PastaCurves.ARMv7EM.Spec.Square
import PastaCurves.ARMv7EM.Spec.FromMont
import PastaCurves.ARMv7EM.Spec.Invert

/-!
# ARMv7E-M correctness proofs

Checked generated instruction traces prove the arithmetic primitives, `inversionMacWords`,
`condSub`, the divstep batches and their composition, and the factored conversion round. The
subject modules also contain pure arithmetic and Aeneas-wrapper proofs; those results are not
additional assembly proofs.

The conversion proof composes its checked prefix, suffix, round, and final-reduction helpers to
establish the generated `fromMontBlock` contract.
-/
