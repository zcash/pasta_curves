/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.X86_64.Spec.Arithmetic
import PastaAsm.X86_64.Spec.Add
import PastaAsm.X86_64.Spec.Sub
import PastaAsm.X86_64.Spec.Mul
import PastaAsm.X86_64.Spec.Square
import PastaAsm.X86_64.Spec.FromMont

/-!
# Correctness proofs for the x86-64 blocks

The per-block proofs live under `PastaAsm.X86_64.Spec` and are imported here as they are
completed. Shared instruction-level arithmetic lemmas support those proofs.
-/
