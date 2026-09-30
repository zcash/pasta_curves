import PastaCurves.X86_64.Spec.Arithmetic
import PastaCurves.X86_64.Spec.Add
import PastaCurves.X86_64.Spec.Sub
import PastaCurves.X86_64.Spec.Mul
import PastaCurves.X86_64.Spec.Square
import PastaCurves.X86_64.Spec.FromMont

/-!
# Correctness proofs for the x86-64 blocks

The per-block proofs live under `PastaCurves.X86_64.Spec` and are imported here. Shared
instruction-level arithmetic lemmas support those proofs.
-/
