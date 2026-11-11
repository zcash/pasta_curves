import PastaCurves.Glv.Wnaf
import PastaCurves.Glv.Ladder
import PastaCurves.Glv.Table

/-!
# The GLV path's entry points compute `[k]P`

For every valid scalar `k`: `Decomposed::new(k)` does not panic, and its digits encode `k` modulo
`n` (`decomposed_new_spec`); one `Decomposed` serves every table, `mul_decomposed` taking the table
of any valid point `p` to a valid point standing for `val k · p`
(`decomposed_new_mul_decomposed_spec`); and `p.mul_glv(&k)`, `Table::new(p).mul(k)`,
`Table::batch(ps)[i].mul(k)` and `batch_mul_same_scalar_vartime` do not panic and return valid
points standing for `val k · p` (`mul_glv_spec`, `table_mul_spec`, `table_batch_mul_spec`,
`batch_mul_same_scalar_spec`).

The theorems are stated against the abstractions `CurveAbs` and `ScalarAbs` of `Spec.lean`, and
each takes as hypotheses the conditions of `Spec.lean` that its proof uses.
-/

namespace PastaCurves.Glv

variable {P A G Sc : Type} [AddCommGroup G]

/-! ## `Decomposed::new` -/

/-- `Decomposed::new(k)` on a valid scalar: no panic, digits of wNAF shape that are zero from
`len ≤ 128` on, and the two halves' values combine to `k` modulo `n`. -/
theorem decomposed_new_spec (S : ScalarAbs Sc) (n lam : ℕ) (f : DecomposeFn Sc)
    (hdec : DecomposeSpec n lam S f) (k : Sc) (hk : S.valid k) :
    ∃ D, Decomposed.new f k = some D ∧
      (∀ i, IsWnafDigit (D.digits1 i)) ∧ (∀ i, IsWnafDigit (D.digits2 i)) ∧ D.len ≤ 128 ∧
      (∀ i : Fin 128, D.len ≤ i → D.digits1 i = 0 ∧ D.digits2 i = 0) ∧
      (∑ i : Fin 128, D.digits1 i * 2 ^ (i : ℕ)) +
        (∑ i : Fin 128, D.digits2 i * 2 ^ (i : ℕ)) * lam ≡ S.val k [ZMOD n] := by
  obtain ⟨⟨⟨neg1, a1⟩, ⟨neg2, a2⟩⟩, hf, h1, h2, hcong⟩ := hdec k hk
  dsimp only at h1 h2 hcong
  obtain ⟨D, hD1, hdig1, hdig2, -, hlen, hz⟩ :=
    decomposed_of_halves_spec neg1 neg2 a1.val a2.val h1 h2
  have hnew : Decomposed.new f k = some D := by
    simp [Decomposed.new, hf, Aeneas.Std.Option.ofResult, hD1]
  refine ⟨D, hnew, ?_, ?_, hlen, hz, ?_⟩
  · intro i; rw [hdig1]; exact (wnaf_digits_shape a1.val neg1 i).1
  · intro i; rw [hdig2]; exact (wnaf_digits_shape a2.val neg2 i).1
  · rw [hdig1, hdig2]
    simp only [Wnaf.run]
    rw [(wnaf_digits_value a1.val neg1 h1).1, (wnaf_digits_value a2.val neg2 h2).1]
    simpa [decode] using hcong

/-! ## `mul_decomposed` over the curve -/

/-- In a group, `n • x = 0` makes an integer multiple of `x` depend on the integer modulo `n`
only. -/
private theorem smul_congr (x : G) (n : ℕ) (a b : ℤ) (hx : n • x = 0) (hab : a ≡ b [ZMOD n]) :
    a • x = b • x := by
  obtain ⟨c, hc⟩ := (Int.ModEq.dvd hab)
  have : b = a + n * c := by omega
  rw [this, add_smul, mul_comm, mul_smul, natCast_zsmul, hx, smul_zero, add_zero]

/-- The ladder over the curve, from the table of a valid point `p` and digits that encode `kv`
modulo `n`, returns a valid point standing for `kv · p`: the step that every entry point shares. -/
private theorem mulDecomposed_curve (O : CurveOps P A) (V : CurveAbs P A G) (n lam kv : ℕ)
    (hord : OrderSpec V n) (hO : CurveOps.Spec O V) (D : Decomposed)
    (h1 : ∀ i, IsWnafDigit (D.digits1 i)) (h2 : ∀ i, IsWnafDigit (D.digits2 i))
    (hlen : D.len ≤ 128) (hz : ∀ i : Fin 128, D.len ≤ i → D.digits1 i = 0 ∧ D.digits2 i = 0)
    (hcong : (∑ i : Fin 128, D.digits1 i * 2 ^ (i : ℕ)) +
      (∑ i : Fin 128, D.digits2 i * 2 ^ (i : ℕ)) * lam ≡ kv [ZMOD n])
    (p : P) (t : Table A) (hp : V.valid p) (ht : TableOf V lam p t) :
    ∃ r, t.mulDecomposed O.toLadderOps D = some r ∧ V.valid r ∧ V.α r = kv • V.α p := by
  obtain ⟨hmap, hvalid⟩ := ladderLoop_transport O V hO t D
    (fun j => ⟨(ht j).1, (ht j).2.2.1⟩) D.len O.identity hO.identity.1
  have htab : (⟨fun j => V.αA (t.t1 j), fun j => V.αA (t.t2 j)⟩ : Table G) =
      ⟨oddMultiples (V.α p), oddMultiples (lam • V.α p)⟩ := by
    congr 1 <;> funext j
    · exact (ht j).2.1
    · exact (ht j).2.2.2
  have hL : ladderLoop (groupOps G) ⟨oddMultiples (V.α p), oddMultiples (lam • V.α p)⟩ D D.len 0 =
      some ((∑ i : Fin 128, D.digits1 i * 2 ^ (i : ℕ)) • V.α p +
        (∑ i : Fin 128, D.digits2 i * 2 ^ (i : ℕ)) • (lam • V.α p)) :=
    mul_decomposed_spec (V.α p) (lam • V.α p) D h1 h2 hlen hz
  rw [hO.identity.2, htab, hL] at hmap
  cases hr : ladderLoop O.toLadderOps t D D.len O.identity with
  | none => simp [hr] at hmap
  | some r =>
    rw [hr, Option.map_some, Option.some.injEq] at hmap
    refine ⟨r, hr, hvalid r hr, ?_⟩
    rw [hmap]
    calc _ = ((∑ i : Fin 128, D.digits1 i * 2 ^ (i : ℕ)) +
          (∑ i : Fin 128, D.digits2 i * 2 ^ (i : ℕ)) * (lam : ℤ)) • V.α p := by
          rw [add_smul, mul_smul, natCast_zsmul]
      _ = (kv : ℤ) • V.α p := smul_congr (V.α p) n _ _ (hord p hp) hcong
      _ = kv • V.α p := natCast_zsmul _ _

/-- `Decomposed::new(k)` on a valid scalar does not panic, and `mul_decomposed` with its result
takes the table of any valid point `p` to a valid point standing for `val k · p`, so one
`Decomposed` serves every table. -/
theorem decomposed_new_mul_decomposed_spec (O : CurveOps P A) (V : CurveAbs P A G)
    (S : ScalarAbs Sc) (n lam : ℕ) (f : DecomposeFn Sc) (hn : OrderSpec V n)
    (hdec : DecomposeSpec n lam S f) (hO : CurveOps.Spec O V) (k : Sc) (hk : S.valid k) :
    ∃ D, Decomposed.new f k = some D ∧ ∀ p t, V.valid p → TableOf V lam p t →
      ∃ r, t.mulDecomposed O.toLadderOps D = some r ∧ V.valid r ∧
        V.α r = S.val k • V.α p := by
  obtain ⟨D, hDnew, h1, h2, hlen, hz, hcong⟩ := decomposed_new_spec S n lam f hdec k hk
  exact ⟨D, hDnew, fun p t hp ht =>
    mulDecomposed_curve O V n lam (S.val k) hn hO D h1 h2 hlen hz hcong p t hp ht⟩

/-! ## The entry points -/

/-- `Table::mul` is `mul_decomposed` with the result of `Decomposed::new`. -/
private theorem table_mul_of (O : CurveOps P A) (f : DecomposeFn Sc) (t : Table A) (k : Sc)
    {D : Decomposed} {r : P} (hD : Decomposed.new f k = some D)
    (hr : t.mulDecomposed O.toLadderOps D = some r) : t.mul O f k = some r := by
  simp [Table.mul, hD, hr]

/-- `p.mul_glv(&k)` on a valid point and a valid scalar does not panic and returns a valid point
standing for `val k · p`. -/
theorem mul_glv_spec (O : CurveOps P A) (V : CurveAbs P A G) (S : ScalarAbs Sc) (n lam : ℕ)
    (f : DecomposeFn Sc) (hn : OrderSpec V n) (hendo : EndoSpec O V lam)
    (hdec : DecomposeSpec n lam S f) (hO : CurveOps.Spec O V) (k : Sc) (hk : S.valid k) (p : P)
    (hp : V.valid p) :
    ∃ r, mulGlv O f p k = some r ∧ V.valid r ∧ V.α r = S.val k • V.α p := by
  obtain ⟨D, hDnew, hcore⟩ := decomposed_new_mul_decomposed_spec O V S n lam f hn hdec hO k hk
  unfold mulGlv
  split
  · rename_i hid
    exact ⟨O.identity, rfl, hO.identity.1,
      by rw [hO.identity.2, hO.isIdentity p hp hid, smul_zero]⟩
  · obtain ⟨t, ht, htab⟩ := table_new_spec O V lam hO hendo p hp
    obtain ⟨r, hr, hv, hα⟩ := hcore p t hp htab
    exact ⟨r, by simp [ht, table_mul_of O f t k hDnew hr], hv, hα⟩

/-- `Table::new(p).mul(k)` on a valid point and a valid scalar does not panic and returns a valid
point standing for `val k · p`. -/
theorem table_mul_spec (O : CurveOps P A) (V : CurveAbs P A G) (S : ScalarAbs Sc) (n lam : ℕ)
    (f : DecomposeFn Sc) (hn : OrderSpec V n) (hendo : EndoSpec O V lam)
    (hdec : DecomposeSpec n lam S f) (hO : CurveOps.Spec O V) (k : Sc) (hk : S.valid k) (p : P)
    (hp : V.valid p) :
    ∃ t r, Table.new O p = some t ∧ t.mul O f k = some r ∧ V.valid r ∧
      V.α r = S.val k • V.α p := by
  obtain ⟨D, hDnew, hcore⟩ := decomposed_new_mul_decomposed_spec O V S n lam f hn hdec hO k hk
  obtain ⟨t, ht, htab⟩ := table_new_spec O V lam hO hendo p hp
  obtain ⟨r, hr, hv, hα⟩ := hcore p t hp htab
  exact ⟨t, r, ht, table_mul_of O f t k hDnew hr, hv, hα⟩

/-- `Table::batch(ps)` on valid points, then `.mul(k)` on each table with a valid scalar: no
panic, and the `i`th table gives a valid point standing for `val k · ps[i]`. -/
theorem table_batch_mul_spec (O : CurveOps P A) (V : CurveAbs P A G) (S : ScalarAbs Sc)
    (n lam : ℕ) (f : DecomposeFn Sc) (hn : OrderSpec V n) (hendo : EndoSpec O V lam)
    (hdec : DecomposeSpec n lam S f) (hO : CurveOps.Spec O V) (k : Sc) (hk : S.valid k)
    (ps : List P) (hps : ∀ p ∈ ps, V.valid p) :
    ∃ ts, Table.batch O ps = some ts ∧
      List.Forall₂ (fun p t => ∃ r, t.mul O f k = some r ∧ V.valid r ∧
        V.α r = S.val k • V.α p) ps ts := by
  obtain ⟨D, hDnew, hcore⟩ := decomposed_new_mul_decomposed_spec O V S n lam f hn hdec hO k hk
  obtain ⟨ts, hts, hall⟩ := table_batch_spec O V lam hO hendo ps hps
  refine ⟨ts, hts, ((List.forall₂_and_left _ _).2 ⟨hps, hall⟩).imp fun p t h => ?_⟩
  obtain ⟨r, hr, hv, hα⟩ := hcore p t h.1 h.2
  exact ⟨r, table_mul_of O f t k hDnew hr, hv, hα⟩

/-- The last loop of `batch_mul_same_scalar_vartime`, over lists of equal length, each table
related to an element of `xs`, succeeds and relates its results to `xs`. -/
private theorem mulEach_forall₂ {X : Type} (O : LadderOps P A) (D : Decomposed)
    (R : X → P → Prop) (xs : List X) (outs : List P) (ts : List (Table A))
    (hlen : outs.length = xs.length)
    (hall : List.Forall₂ (fun x t => ∃ r, t.mulDecomposed O D = some r ∧ R x r) xs ts) :
    ∃ rs, mulEach O D outs ts = some rs ∧ List.Forall₂ R xs rs := by
  induction hall generalizing outs with
  | nil =>
    obtain rfl : outs = [] := List.eq_nil_of_length_eq_zero hlen
    exact ⟨[], by simp [PastaCurves.Glv.mulEach], List.Forall₂.nil⟩
  | cons h _ ih =>
    obtain ⟨r, hr, hR⟩ := h
    cases outs with
    | nil => simp at hlen
    | cons o outs =>
      obtain ⟨rs, hrs, hR'⟩ := ih outs (by simpa using hlen)
      exact ⟨r :: rs, by simp [PastaCurves.Glv.mulEach, hr, hrs], List.Forall₂.cons hR hR'⟩

/-- `batch_mul_same_scalar_vartime(points, k, output)` on valid affine points, a valid scalar and
slices of equal length: no panic, and each `output[i]` becomes a valid point standing for
`val k · points[i]`. -/
theorem batch_mul_same_scalar_spec (O : CurveOps P A) (V : CurveAbs P A G) (S : ScalarAbs Sc)
    (n lam : ℕ) (f : DecomposeFn Sc) (hn : OrderSpec V n) (hendo : EndoSpec O V lam)
    (hdec : DecomposeSpec n lam S f) (hO : CurveOps.Spec O V) (k : Sc) (hk : S.valid k)
    (points : List A) (output : List P) (hpts : ∀ a ∈ points, V.validA a)
    (hlen : points.length = output.length) :
    ∃ rs, batchMulSameScalar O f points k output = some rs ∧
      List.Forall₂ (fun a r => V.valid r ∧ V.α r = S.val k • V.αA a) points rs := by
  obtain ⟨D, hDnew, h1, h2, hl, hz, hcong⟩ := decomposed_new_spec S n lam f hdec k hk
  unfold batchMulSameScalar
  rw [if_neg (not_not.mpr hlen)]
  split
  · rename_i he
    obtain rfl := List.isEmpty_iff.mp he
    obtain rfl := List.eq_nil_of_length_eq_zero hlen.symm
    exact ⟨[], rfl, List.Forall₂.nil⟩
  · have hv : ∀ q ∈ points.map O.toCurve, V.valid q := by
      intro q hq
      obtain ⟨a, ha, rfl⟩ := List.mem_map.mp hq
      exact (hO.toCurve a (hpts a ha)).1
    obtain ⟨ts, hts, hall⟩ := table_batch_spec O V lam hO hendo _ hv
    rw [List.forall₂_map_left_iff] at hall
    have hall' : List.Forall₂ (fun a t => ∃ r, t.mulDecomposed O.toLadderOps D = some r ∧
        (V.valid r ∧ V.α r = S.val k • V.αA a)) points ts :=
      ((List.forall₂_and_left _ _).2 ⟨hpts, hall⟩).imp fun a t h => by
        obtain ⟨r, hr, hvr, hα⟩ := mulDecomposed_curve O V n lam (S.val k) hn hO
          D h1 h2 hl hz hcong _ t (hO.toCurve a h.1).1 h.2
        exact ⟨r, hr, hvr, by rw [hα, (hO.toCurve a h.1).2]⟩
    obtain ⟨rs, hrs, hR⟩ := mulEach_forall₂ O.toLadderOps D
      (fun a r => V.valid r ∧ V.α r = S.val k • V.αA a) points (points.map O.toCurve) ts
      (by simp) hall'
    exact ⟨rs, by simp [hDnew, hts, hrs], hR⟩

end PastaCurves.Glv
