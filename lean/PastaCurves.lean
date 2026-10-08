import PastaCurves.Semantics
import PastaCurves.Pratt
import PastaCurves.Primality
import PastaCurves.Fields
import PastaCurves.FieldTypes
import PastaCurves.KnownAnswers
import PastaCurves.Compositions
import PastaCurves.Spec
import PastaCurves.Inversion.Divstep
import PastaCurves.Inversion.Packed
import PastaCurves.Inversion.Divstep59
import PastaCurves.Inversion.Round
import PastaCurves.Inversion.Termination
import PastaCurves.Inversion.Model
import PastaCurves.Inversion.Hull
import PastaCurves.Inversion.HullBound
import PastaCurves.Inversion.HullData
import PastaCurves.Inversion.HullCert
import PastaCurves.Inversion.Correctness
import PastaCurves.Inversion.SignMag
import PastaCurves.Inversion.PackedWords
import PastaCurves.Inversion.Composition
import PastaCurves.Inversion.Vectors
import PastaCurves.AArch64
import PastaCurves.X86_64
import PastaCurves.Portable.Funs
import PastaCurves.Portable.Words
import PastaCurves.Portable.Vectors
import PastaCurves.Portable.Spec
import PastaCurves.Portable.Driver
import PastaCurves.Words
import PastaCurves.Glue.Funs
import PastaCurves.Glue.Blocks
import PastaCurves.Glue.Spec
import PastaCurves.Glue.Vectors

/-!
# The assembly routines, formalized

Generic arithmetic is defined in the top-level `PastaCurves` modules. Architecture-specific
transcriptions and proofs live under their corresponding namespaces.

Every module of the development is imported here, so that a build of this root builds all of
them and the nanoda re-check (`lean/scripts/check_nanoda.sh`) exports all of them.
-/
