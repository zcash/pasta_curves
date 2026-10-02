/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import Mathlib.Tactic.IntervalCases
import PastaAsm.Vectors

/-!
# Checking a backend against the reference vectors

`Vectors.lean` holds the hardware corpus as data. A backend supplies its three routines as a
`VectorBackend`, and `VectorBackend.failures` lists the vectors that the routines do not
reproduce. Each backend's `Vectors.lean` proves that the lists are empty by kernel evaluation
(`decide +kernel`), and also evaluates them under `#guard_msgs`, so that a failure names the
vectors that failed.

The kernel checks a few dozen vectors several times faster, per vector, than the whole
multiplication list at once, so `VectorBackend.failures_eq_nil` takes the multiplication check
in `pieces` pieces, cut by index modulo `pieces`, and `failing_eq_nil_of_pieces` says that empty
pieces mean an empty list.
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
def failing {α : Type} (vectors : List (Nat × α)) (reproduced : α → Bool) : List Nat :=
  (vectors.filter fun (_, v) => !reproduced v).map (·.1)

/-- The number of pieces that the multiplication vectors are checked in. -/
def pieces : Nat := 32

/-- The indices of the vectors that `reproduced` rejects among those whose index is `k` modulo
`pieces`. -/
def failingPiece {α : Type} (vectors : List (Nat × α)) (reproduced : α → Bool) (k : Nat) :
    List Nat :=
  (vectors.filter fun (i, v) => i % pieces == k && !reproduced v).map (·.1)

/-- Empty pieces mean that `reproduced` rejects no vector. -/
theorem failing_eq_nil_of_pieces {α : Type} (vectors : List (Nat × α)) (reproduced : α → Bool)
    (h : ∀ k, k < pieces → failingPiece vectors reproduced k = []) :
    failing vectors reproduced = [] := by
  unfold failing
  rw [List.map_eq_nil_iff, List.filter_eq_nil_iff]
  intro ⟨i, v⟩ hmem hfail
  have hk := h (i % pieces) (Nat.mod_lt _ (by decide))
  unfold failingPiece at hk
  rw [List.map_eq_nil_iff, List.filter_eq_nil_iff] at hk
  exact hk ⟨i, v⟩ hmem (by simp_all)

namespace VectorBackend

/-- Whether the backend reproduces a multiplication vector. -/
def mulOk (B : VectorBackend) : PastaField × Limbs × Limbs × Limbs → Bool :=
  fun (F, a, b, r) => B.mulMont a b F.modulus F.inv == r

/-- Whether the backend reproduces a squaring vector. -/
def sqrOk (B : VectorBackend) : PastaField × Limbs × Limbs → Bool :=
  fun (F, a, r) => B.sqrMont a F.modulus F.inv == r

/-- Whether the backend reproduces a conversion vector. -/
def fromOk (B : VectorBackend) : PastaField × Limbs × Limbs → Bool :=
  fun (F, a, r) => B.fromMont a F.modulus F.inv == r

/-- The indices, into `mulVectors`, `sqrVectors`, and `fromVectors`, of the vectors that the
backend does not reproduce. -/
def failures (B : VectorBackend) : List Nat × List Nat × List Nat :=
  (failing mulVectors B.mulOk, failing sqrVectors B.sqrOk, failing fromVectors B.fromOk)

/-- The backend reproduces every vector when each piece of the multiplication vectors, the
squaring vectors, and the conversion vectors come out without failures; a backend proves the
three premisses by kernel evaluation. -/
theorem failures_eq_nil (B : VectorBackend)
    (hmul : ∀ k, k < pieces → failingPiece mulVectors B.mulOk k = [])
    (hsqr : failing sqrVectors B.sqrOk = []) (hfrom : failing fromVectors B.fromOk = []) :
    B.failures = ([], [], []) := by
  unfold failures
  rw [failing_eq_nil_of_pieces _ _ hmul, hsqr, hfrom]

end VectorBackend

end PastaAsm
