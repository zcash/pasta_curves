import PastaCurves.ARMv7EM.Semantics
import PastaCurves.ARMv7EM.Transcription
import PastaCurves.ARMv7EM.Compositions
import PastaCurves.ARMv7EM.Vectors
import PastaCurves.ARMv7EM.Spec
import PastaCurves.ARMv7EM.Backend
import PastaCurves.ARMv7EM.Entry

/-!
# The ARMv7E-M arithmetic routines, formalized

The generated instruction-level transcription and primitive proofs are composed according to the
fixed Rust schedule and checked against the shared AArch64 reference vectors.
-/
