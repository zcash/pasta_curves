import PastaCurves.ARMv7EM.Compositions.Model
import PastaCurves.ARMv7EM.Compositions.Funs
import PastaCurves.ARMv7EM.Compositions.Shared.Primitives
import PastaCurves.ARMv7EM.Compositions.Shared.Conversions
import PastaCurves.ARMv7EM.Compositions.Shared.Reduction
import PastaCurves.ARMv7EM.Compositions.Shared.Loops
import PastaCurves.ARMv7EM.Compositions.Mul.Cancellation
import PastaCurves.ARMv7EM.Compositions.Mul.Rounds
import PastaCurves.ARMv7EM.Compositions.Mul.Loop
import PastaCurves.ARMv7EM.Compositions.Mul.Wrapper
import PastaCurves.ARMv7EM.Compositions.Mul.Overflow
import PastaCurves.ARMv7EM.Compositions.Mul.Value
import PastaCurves.ARMv7EM.Compositions.Mul.Entry
import PastaCurves.ARMv7EM.Compositions.Mul.SquareConversion
import PastaCurves.ARMv7EM.Compositions.Invert.ConditionalSub

/-!
# The ARMv7E-M Rust around its assembly blocks

`Funs` is the generated translation of the ARM-specific Rust functions and backend methods.
`Model` is a Nat/limb view of their arithmetic, used only as an intermediate target of
correspondence proofs; it is not an alternative backend or the field-operation specification.
The handwritten correctness specification is the canonical-result and Montgomery-congruence
contract in `Glue.Spec`, instantiated for the generated method record in `ARMv7EM.Entry`.
The operation-oriented modules under `Compositions/` connect translated Rust loops to the
arithmetic view and its instruction primitives. Opaque assembly calls require separate checked
transcription proofs; no theorem about an intermediate model alone discharges that obligation.
-/
