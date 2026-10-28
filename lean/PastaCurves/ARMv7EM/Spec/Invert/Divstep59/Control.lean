/-
Copyright (c) 2026 the pasta_curves contributors.
-/
import PastaCurves.ARMv7EM.Transcription

/-!
# Control-flow facts for one ARMv7E-M batch round

The generated round carries a number of 32-bit registers, but its public loop control is the final
`subc32 remaining 1 1`, followed by the zero test and the backward-branch test. This file proves
those projections and the bounded iterations used by the batch driver.
-/

set_option maxRecDepth 4096

namespace PastaCurves.ARMv7EM

/-- The register words of a batch state are bounded 32-bit words. -/
structure BatchState.Bounded (st : BatchState) : Prop where
  pflo : st.pflo < wordMod
  pfhi : st.pfhi < wordMod
  pglo : st.pglo < wordMod
  pghi : st.pghi < wordMod
  d : st.d < wordMod

/-- Subtracting one with carry-in one decrements a positive small count and sets carry. -/
lemma subc32_one_one (n : Nat) (hn : 1 ≤ n) (hn20 : n ≤ 20) :
    subc32 n 1 1 = (n - 1, 1) := by
  unfold subc32 wordMod
  simp only [Nat.sub_self, Nat.sub_zero]
  have hrepr : n + 2 ^ 32 - 1 = 2 ^ 32 + (n - 1) := by omega
  rw [hrepr]
  have hlt : n - 1 < 2 ^ 32 := by omega
  have hmod : (2 ^ 32 + (n - 1)) % 2 ^ 32 = n - 1 := by omega
  have hdiv : (2 ^ 32 + (n - 1)) / 2 ^ 32 = 1 := by omega
  exact Prod.ext hmod hdiv

/-- A positive count at most twenty is decremented by one by `batchRound`. -/
lemma batchRound_remaining_of_count (st : BatchState) (h1 : 1 ≤ st.remaining)
    (h20 : st.remaining ≤ 20) :
    (batchRound st).remaining = st.remaining - 1 := by
  have hsub := subc32_one_one st.remaining h1 h20
  change (subc32 st.remaining 1 1).1 = st.remaining - 1
  rw [hsub]

/-- The decrement in `batchRound` leaves carry set. -/
lemma batchRound_c_of_count (st : BatchState) (h1 : 1 ≤ st.remaining)
    (h20 : st.remaining ≤ 20) :
    (batchRound st).c = 1 := by
  have hsub := subc32_one_one st.remaining h1 h20
  change (subc32 st.remaining 1 1).2 = 1
  rw [hsub]

/-- The zero flag is set exactly on the last round of a positive small count. -/
lemma batchRound_z_of_count (st : BatchState) (h1 : 1 ≤ st.remaining)
    (h20 : st.remaining ≤ 20) :
    (batchRound st).z = 1 ↔ st.remaining = 1 := by
  have hsub := subc32_one_one st.remaining h1 h20
  simp only [batchRound, batchRoundFinish, hsub]
  by_cases h : st.remaining = 1
  · simp [h]
  · have hgt : 1 < st.remaining := by omega
    simp [Nat.ne_of_gt (Nat.sub_pos_of_lt hgt)]
    omega

/-- The backward branch is taken exactly while more than one round remains. -/
lemma batchRound_branch_of_count (st : BatchState) (h1 : 1 ≤ st.remaining)
    (h20 : st.remaining ≤ 20) :
    (batchRound st).branch = 1 ↔ st.remaining > 1 := by
  have hsub := subc32_one_one st.remaining h1 h20
  simp only [batchRound, batchRoundFinish, hsub]
  by_cases h : st.remaining = 1
  · simp [h]
  · have hgt : 1 < st.remaining := by omega
    simp [Nat.ne_of_gt (Nat.sub_pos_of_lt hgt)]
    omega

/-- A positive bounded state exposes the decrement, carry, zero flag, and branch decision. -/
theorem batchRound_control (st : BatchState) (h1 : 1 ≤ st.remaining)
    (h20 : st.remaining ≤ 20) (_hb : st.Bounded) :
    let out := batchRound st
    out.remaining = st.remaining - 1 ∧ out.c = 1 ∧
      (out.branch = 1 ↔ st.remaining > 1) ∧ (out.z = 1 ↔ st.remaining = 1) := by
  dsimp
  exact ⟨batchRound_remaining_of_count st h1 h20, batchRound_c_of_count st h1 h20,
    batchRound_branch_of_count st h1 h20, batchRound_z_of_count st h1 h20⟩

/-- After at most twenty rounds, the remaining count is the initial count minus the number of
rounds already executed. -/
lemma batchRound_iterate_remaining (n j : Nat) (st : BatchState) (hrem : st.remaining = n)
    (hn : n ≤ 20) (hj : j ≤ n) :
    (batchRound^[j] st).remaining = n - j := by
  induction j with
  | zero => simp [hrem]
  | succ j ih =>
      have hjn : j < n := by omega
      have hmid := ih (by omega)
      rw [Function.iterate_succ_apply']
      rw [batchRound_remaining_of_count _ (by omega) (by omega)]
      rw [hmid]
      omega

/-- Iterating the round from count `k` reaches zero after exactly `k` rounds. For every executed
round, the branch is taken precisely when another round remains, so the loop does not terminate
early. The theorem is stated for the two batch sizes used by the driver. -/
theorem batchRound_iterate (k : Nat) (hk : k = 19 ∨ k = 20)
    (st : BatchState) (hstate : st.remaining = k) (_hb : st.Bounded) :
    (batchRound^[k] st).remaining = 0 ∧
      ∀ j, j < k → ((batchRound^[j + 1] st).branch = 1 ↔ j + 1 < k) := by
  have hk20 : k ≤ 20 := by omega
  have hk1 : 1 ≤ k := by omega
  constructor
  · simpa using batchRound_iterate_remaining k k st hstate hk20 (Nat.le_refl k)
  · intro j hj
    have hmid := batchRound_iterate_remaining k j st hstate hk20 (by omega)
    rw [Function.iterate_succ_apply']
    rw [batchRound_branch_of_count _ (by omega) (by omega)]
    rw [hmid]
    omega

end PastaCurves.ARMv7EM
