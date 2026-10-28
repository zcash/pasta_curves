import PastaCurves.ARMv7EM.Compositions.Model
import PastaCurves.VectorCheck

/-!
# The ARMv7E-M routines on the reference vectors

These are cross-backend checks: the vectors in `Vectors.lean` were produced by the real AArch64
assembly, and the theorem below has the kernel evaluate the ARMv7E-M proof model on the same
operands, not the Aeneas translation; they are not captures from ARMv7E-M hardware. Multiplication
and squaring use the model of the fixed Rust composition around the generated, proved ARMv7E-M
word primitives. Conversion uses `fromMontBlock`, the mechanically generated transcription of the
dedicated inline assembly.

The 874 checks comprise 806 multiplications, 34 squarings, and 34 conversions. The denser 32-bit
schedule makes one proof over all multiplication vectors exceed Lean's per-declaration heartbeat
budget, so the shared `VectorCheck` machinery splits it into 32 exhaustive residue classes. The
square and conversion checks are split at index 17, the boundary between the Pallas and Vesta
vectors. Every piece is checked by kernel reduction before the results are combined.

The conversion vector theorem establishes agreement with this finite corpus. General correctness
of `fromMontBlock` is proved separately by its checked instruction skeleton and entry theorem.
-/

namespace PastaCurves.ARMv7EM

/-- The ARMv7E-M routines that the vectors exercise. -/
def vectorBackend : VectorBackend :=
  ⟨mulMont, sqrMont, fun value modulus _ => fromMontBlock value modulus⟩

private theorem mulPieces : ∀ k, k < pieces →
    failingPiece mulVectors vectorBackend.mulOk k = [] := by
  intro k hk
  unfold pieces at hk
  interval_cases k <;> decide +kernel

private theorem failing_append {α : Type} (xs ys : List (Nat × α)) (reproduced : α → Bool) :
    failing (xs ++ ys) reproduced = failing xs reproduced ++ failing ys reproduced := by
  simp [failing]

-- Index 17 is the boundary between the Pallas and Vesta vectors in each of these lists.
private theorem sqrPallas :
    failing (sqrVectors.take 17) vectorBackend.sqrOk = [] := by decide +kernel
private theorem sqrVesta :
    failing (sqrVectors.drop 17) vectorBackend.sqrOk = [] := by decide +kernel
private theorem fromPallas :
    failing (fromVectors.take 17) vectorBackend.fromOk = [] := by decide +kernel
private theorem fromVesta :
    failing (fromVectors.drop 17) vectorBackend.fromOk = [] := by decide +kernel

/-- The ARMv7E-M routines reproduce every AArch64 reference vector for both Pasta fields. -/
theorem vectors_reproduced : vectorBackend.failures = ([], [], []) :=
  vectorBackend.failures_eq_nil mulPieces
    (by
      rw [← List.take_append_drop 17 sqrVectors, failing_append, sqrPallas, sqrVesta]
      simp)
    (by
      rw [← List.take_append_drop 17 fromVectors, failing_append, fromPallas, fromVesta]
      simp)

-- The evaluation names the vectors that fail, should any.
/-- info: ([], [], []) -/
#guard_msgs in
#eval vectorBackend.failures

end PastaCurves.ARMv7EM
