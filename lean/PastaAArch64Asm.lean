/-
Copyright (c) 2026 the pasta-aarch64-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAArch64Asm.Semantics
import PastaAArch64Asm.Transcription
import PastaAArch64Asm.Compositions
import PastaAArch64Asm.Fields
import PastaAArch64Asm.Vectors
import PastaAArch64Asm.Spec
import PastaAArch64Asm.Entry

/-!
# The crate's AArch64 Pasta Montgomery routines, formalized

Every module of the development is imported here, so that a build of this root builds all of
them and the nanoda re-check (`lean/scripts/check_nanoda.sh`) exports all of them.
-/
