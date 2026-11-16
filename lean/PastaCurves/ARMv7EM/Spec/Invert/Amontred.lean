import PastaCurves.ARMv7EM.Spec.Invert.Amontred.Correctness

/-!
# ARMv7E-M almost-Montgomery reduction

This subject module connects the checked `inversionMacWords` instruction block to the
Aeneas-translated Rust wrapper and proves the wrapper's almost-Montgomery reduction result. The
wrapper proof is not a separate generated assembly-block proof.
-/
