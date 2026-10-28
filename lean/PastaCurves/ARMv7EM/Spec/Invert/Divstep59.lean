import PastaCurves.ARMv7EM.Spec.Invert.Divstep59.Batch
import PastaCurves.ARMv7EM.Spec.Invert.Divstep59.Block
import PastaCurves.ARMv7EM.Spec.Invert.Divstep59.Control
import PastaCurves.ARMv7EM.Spec.Invert.Divstep59.Finish
import PastaCurves.ARMv7EM.Spec.Invert.Divstep59.PackedAdapter
import PastaCurves.ARMv7EM.Spec.Invert.Divstep59.Round

/-!
# ARMv7E-M 59-step divstep support

These modules prove the generated `batchRoundFinish` and `batchRound` instruction routines,
compose the checked batch, and establish the generated `divstep59Block` contract under the signed
32-bit state bound. Control-flow and packed-step iteration facts justify the three public batches.
-/
