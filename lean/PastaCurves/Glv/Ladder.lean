import PastaCurves.Glv.Spec

/-!
# The ladder of `Table::mul_decomposed`

In any abelian group, over the tables of the odd multiples of `p` and `q`, the ladder of
`mul_decomposed`, on digits of wNAF shape that are zero from `len ≤ 128` on, returns
`(Σ d1ᵢ 2ⁱ)·p + (Σ d2ᵢ 2ⁱ)·q` without a panic (`mul_decomposed_spec`). Over curve operations that
meet `CurveOps.Spec`, the ladder follows the group's under the abstraction, through valid points
(`addDigit_transport`, `ladderLoop_transport`). `Entry.lean` puts the two together.
-/

namespace PastaCurves.Glv

/-! ## In a group -/

/-- `add_digit` in a group, over the odd multiples of `p`, adds `d·p`. -/
private theorem addDigit_group {G : Type} [AddCommGroup G] (acc p : G) (d : ℤ)
    (hd : IsWnafDigit d) : addDigit (groupOps G) acc (oddMultiples p) d = some (acc + d • p) := by
  rcases hd with rfl | ⟨hodd, h1, h2⟩
  · simp [addDigit]
  · rw [Int.odd_iff] at hodd
    have cases : d = -7 ∨ d = -5 ∨ d = -3 ∨ d = -1 ∨ d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 := by omega
    rcases cases with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [addDigit, groupOps, oddMultiples] <;> abel

/-- The positions below `j + 1` are those below `j`, and `j`. -/
private theorem sum_below_succ (f : Fin 128 → ℤ) (j : ℕ) (hj : j < 128) :
    (∑ i : Fin 128, if (i : ℕ) < j + 1 then f i else 0) =
      (∑ i : Fin 128, if (i : ℕ) < j then f i else 0) + f ⟨j, hj⟩ := by
  have : ∀ i : Fin 128, (if (i : ℕ) < j + 1 then f i else 0) =
      (if (i : ℕ) < j then f i else 0) + (if i = ⟨j, hj⟩ then f i else 0) := by
    intro i
    by_cases h : (i : ℕ) = j
    · have : i = ⟨j, hj⟩ := Fin.ext h
      subst this; simp
    · have : i ≠ ⟨j, hj⟩ := fun e => h (by rw [e])
      simp only [this, if_false, add_zero]
      by_cases h' : (i : ℕ) < j
      · simp [h', show (i : ℕ) < j + 1 by omega]
      · simp [h', show ¬ (i : ℕ) < j + 1 by omega]
  rw [Finset.sum_congr rfl (fun i _ => this i), Finset.sum_add_distrib]
  simp

/-- One step of the ladder's loop, rearranged in the group. -/
private theorem ladder_step {G : Type} [AddCommGroup G] (a p q : G) (d1 d2 S1 S2 : ℤ) (j : ℕ) :
    (2 ^ j : ℕ) • (2 • a + d1 • p + d2 • q) + S1 • p + S2 • q =
      (2 ^ (j + 1) : ℕ) • a + (S1 + d1 * 2 ^ j) • p + (S2 + d2 * 2 ^ j) • q := by
  rw [smul_add, smul_add, smul_smul, ← pow_succ, add_smul, add_smul]
  rw [← natCast_zsmul (d1 • p) (2 ^ j), ← natCast_zsmul (d2 • q) (2 ^ j), smul_smul, smul_smul]
  push_cast
  rw [mul_comm ((2 : ℤ) ^ j) d1, mul_comm ((2 : ℤ) ^ j) d2]
  abel

/-- The ladder's loop in a group, from position `j ≤ len`: the accumulator is doubled `j` times,
and the digits at positions `i < j` count `2^i` times. At `j = len` the skipped doubling acts on
`acc = 0`. -/
private theorem ladderLoop_group {G : Type} [AddCommGroup G] (p q : G) (D : Decomposed)
    (h1 : ∀ i, IsWnafDigit (D.digits1 i)) (h2 : ∀ i, IsWnafDigit (D.digits2 i))
    (hlen : D.len ≤ 128) :
    ∀ (j : ℕ) (acc : G), j ≤ D.len → (j < D.len ∨ acc = 0) →
      ladderLoop (groupOps G) ⟨oddMultiples p, oddMultiples q⟩ D j acc =
        some ((2 ^ j : ℕ) • acc +
          (∑ i : Fin 128, if (i : ℕ) < j then D.digits1 i * 2 ^ (i : ℕ) else 0) • p +
          (∑ i : Fin 128, if (i : ℕ) < j then D.digits2 i * 2 ^ (i : ℕ) else 0) • q) := by
  intro j
  induction j with
  | zero => intro acc _ _; simp [ladderLoop]
  | succ j ih =>
    intro acc hj hor
    have hj128 : j < 128 := by omega
    have hdbl : (if j + 1 < D.len then (groupOps G).double acc else acc) = 2 • acc := by
      split
      · rfl
      · rcases hor with h | h
        · omega
        · subst h; simp
    simp only [ladderLoop, hdbl, digitAt, hj128, dif_pos]
    simp only [Option.bind_eq_bind, Option.bind_some, addDigit_group _ _ _ (h1 _),
      addDigit_group _ _ _ (h2 _)]
    rw [ih _ (by omega) (Or.inl (by omega)), sum_below_succ _ j hj128, sum_below_succ _ j hj128]
    congr 1
    exact ladder_step acc p q _ _ _ _ j

/-- The ladder of `Table::mul_decomposed`, in any abelian group: over the tables of the odd
multiples of `p` and `q`, digits of wNAF shape that are zero from `len ≤ 128` on give
`(Σ d1ᵢ 2ⁱ)·p + (Σ d2ᵢ 2ⁱ)·q`, without a panic. With `len = 0` that is the identity. -/
theorem mul_decomposed_spec {G : Type} [AddCommGroup G] (p q : G) (D : Decomposed)
    (h1 : ∀ i, IsWnafDigit (D.digits1 i)) (h2 : ∀ i, IsWnafDigit (D.digits2 i))
    (hlen : D.len ≤ 128) (hz : ∀ i : Fin 128, D.len ≤ i → D.digits1 i = 0 ∧ D.digits2 i = 0) :
    Table.mulDecomposed (groupOps G) ⟨oddMultiples p, oddMultiples q⟩ D =
      some ((∑ i : Fin 128, D.digits1 i * 2 ^ (i : ℕ)) • p +
        (∑ i : Fin 128, D.digits2 i * 2 ^ (i : ℕ)) • q) := by
  show ladderLoop (groupOps G) ⟨oddMultiples p, oddMultiples q⟩ D D.len 0 = _
  have e1 : (∑ i : Fin 128, if (i : ℕ) < D.len then D.digits1 i * 2 ^ (i : ℕ) else 0) =
      ∑ i : Fin 128, D.digits1 i * 2 ^ (i : ℕ) :=
    Finset.sum_congr rfl fun i _ => by
      split_ifs with h
      · rfl
      · rw [(hz i (by omega)).1, zero_mul]
  have e2 : (∑ i : Fin 128, if (i : ℕ) < D.len then D.digits2 i * 2 ^ (i : ℕ) else 0) =
      ∑ i : Fin 128, D.digits2 i * 2 ^ (i : ℕ) :=
    Finset.sum_congr rfl fun i _ => by
      split_ifs with h
      · rfl
      · rw [(hz i (by omega)).2, zero_mul]
  rw [ladderLoop_group p q D h1 h2 hlen D.len 0 le_rfl (Or.inr rfl), smul_zero, zero_add, e1, e2]

/-! ## Over the curve operations -/

/-- One `add_digit` over curve operations that meet `CurveOps.Spec` follows `add_digit` over the
group under `α`: the same outcome, through valid points. -/
theorem addDigit_transport {P A G : Type} [AddCommGroup G] (O : CurveOps P A)
    (V : CurveAbs P A G) (hO : CurveOps.Spec O V) (acc : P) (tab : Fin 4 → A) (d : ℤ)
    (hacc : V.valid acc) (htab : ∀ j, V.validA (tab j)) :
    (addDigit O.toLadderOps acc tab d).map V.α =
        addDigit (groupOps G) (V.α acc) (fun j => V.αA (tab j)) d ∧
      ∀ r, addDigit O.toLadderOps acc tab d = some r → V.valid r := by
  unfold addDigit
  split
  · split
    · rename_i _ hlt
      have hv := htab ⟨d.natAbs / 2, hlt⟩
      split
      · have hn := hO.negA _ hv
        have hm := hO.addMixed _ _ hacc hn.1
        exact ⟨by simp [groupOps, hm.2, hn.2], by intro r hr; simp at hr; rw [← hr]; exact hm.1⟩
      · have hm := hO.addMixed _ _ hacc hv
        exact ⟨by simp [groupOps, hm.2], by intro r hr; simp at hr; rw [← hr]; exact hm.1⟩
    · simp
  · exact ⟨by simp, by intro r hr; simp at hr; rw [← hr]; exact hacc⟩

/-- The ladder's loop over curve operations that meet `CurveOps.Spec` follows the loop over the
group under `α`: the same outcome, through valid points. -/
theorem ladderLoop_transport {P A G : Type} [AddCommGroup G] (O : CurveOps P A)
    (V : CurveAbs P A G) (hO : CurveOps.Spec O V) (t : Table A) (D : Decomposed)
    (ht : ∀ j, V.validA (t.t1 j) ∧ V.validA (t.t2 j)) :
    ∀ (j : ℕ) (acc : P), V.valid acc →
      (ladderLoop O.toLadderOps t D j acc).map V.α =
        ladderLoop (groupOps G) ⟨fun j => V.αA (t.t1 j), fun j => V.αA (t.t2 j)⟩ D j
          (V.α acc) ∧
      ∀ r, ladderLoop O.toLadderOps t D j acc = some r → V.valid r := by
  intro j
  induction j with
  | zero =>
    intro acc hacc
    exact ⟨by simp [ladderLoop], by intro r hr; simp [ladderLoop] at hr; rw [← hr]; exact hacc⟩
  | succ j ih =>
    intro acc hacc
    have hd : V.valid (if j + 1 < D.len then O.double acc else acc) := by
      split
      · exact (hO.double acc hacc).1
      · exact hacc
    have hdα : V.α (if j + 1 < D.len then O.double acc else acc) =
        (if j + 1 < D.len then (groupOps G).double (V.α acc) else V.α acc) := by
      split
      · exact (hO.double acc hacc).2
      · rfl
    simp only [ladderLoop]
    cases hd1 : digitAt D.digits1 j with
    | none => simp
    | some d1 =>
      simp only [Option.bind_eq_bind, Option.bind_some]
      obtain ⟨e1, v1⟩ := addDigit_transport O V hO _ t.t1 d1 hd (fun i => (ht i).1)
      rw [hdα] at e1
      cases ha1 : addDigit O.toLadderOps (if j + 1 < D.len then O.double acc else acc) t.t1 d1 with
      | none =>
        rw [ha1] at e1; simp at e1
        simp [← e1]
      | some r1 =>
        rw [ha1] at e1; simp at e1
        rw [← e1]
        simp only [Option.bind_some]
        cases hd2 : digitAt D.digits2 j with
        | none => simp
        | some d2 =>
          simp only [Option.bind_some]
          obtain ⟨e2, v2⟩ := addDigit_transport O V hO r1 t.t2 d2 (v1 r1 ha1) (fun i => (ht i).2)
          cases ha2 : addDigit O.toLadderOps r1 t.t2 d2 with
          | none => rw [ha2] at e2; simp at e2; simp [← e2]
          | some r2 =>
            rw [ha2] at e2; simp at e2
            rw [← e2]
            simp only [Option.bind_some]
            exact ih r2 (v2 r2 ha2)

end PastaCurves.Glv
