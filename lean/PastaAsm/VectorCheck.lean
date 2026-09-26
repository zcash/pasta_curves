/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAsm.Vectors

/-!
# Checking a backend against the reference vectors

`Vectors.lean` holds the hardware corpus as data. A backend supplies its three routines as a
`VectorBackend`, and `VectorBackend.failures` lists the vectors that the routines do not
reproduce. Each backend's `Vectors.lean` proves that the list is empty by kernel evaluation
(`decide +kernel`), and also evaluates it under `#guard_msgs`, so that a failure names the
vectors that failed.
-/

namespace PastaAsm

/-- The routines that the vectors exercise, as one backend transcribes them: Montgomery
multiplication, Montgomery squaring, and the conversion out of Montgomery form, each taking the
modulus limbs and `inv` last. -/
structure VectorBackend where
  mulMont : Limbs → Limbs → Limbs → Nat → Limbs
  sqrMont : Limbs → Limbs → Nat → Limbs
  fromMont : Limbs → Limbs → Nat → Limbs

/-- The indices of the vectors that `reproduced` rejects. -/
def failing {α : Type} (vectors : List α) (reproduced : α → Bool) : List Nat :=
  (vectors.zipIdx.filter fun (v, _) => !reproduced v).map (·.2)

/-- The indices, into `mulVectors`, `sqrVectors`, and `fromVectors`, of the vectors that the
backend does not reproduce. -/
def VectorBackend.failures (B : VectorBackend) : List Nat × List Nat × List Nat :=
  (failing mulVectors fun (F, a, b, r) => B.mulMont a b F.modulus F.inv == r,
   failing sqrVectors fun (F, a, r) => B.sqrMont a F.modulus F.inv == r,
   failing fromVectors fun (F, a, r) => B.fromMont a F.modulus F.inv == r)

end PastaAsm
