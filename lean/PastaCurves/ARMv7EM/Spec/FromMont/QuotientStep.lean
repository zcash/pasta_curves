/-
Copyright (c) 2026 the pasta_curves contributors.
-/
import PastaCurves.ARMv7EM.Spec.FromMont.Round

/-!
# The quotient prefix of one ARMv7E-M Montgomery conversion round

The quotient-prefix lemmas are defined in `Round` so the checked generated round proof can use
them without an import cycle. This module remains as a compatibility import for existing users.
-/
