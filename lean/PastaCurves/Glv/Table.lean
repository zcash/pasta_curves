import PastaCurves.Glv.Spec

/-!
# The tables of `Table::new` and `Table::batch`

Under `CurveOps.Spec`, `window_proj` of a valid point holds the valid points `p, 3p, 5p, 7p`,
followed by their images under `endo` (`window_proj_spec`). Under `EndoSpec` as well,
`Table::new(p)` is the table of `p` (`table_new_spec`), and `Table::batch(ps)` is one table per
point, in order, as `chunks_exact(8)` cuts the normalized windows (`table_batch_spec`). Valid
points include the identity, whose table stands for the identity throughout.
-/

namespace PastaCurves.Glv

variable {P A G : Type} [AddCommGroup G]

/-! ## The window -/

/-- `Table::window_proj(p)` on a valid point holds eight points: the odd multiples `p, 3p, 5p, 7p`,
valid, followed by their images under `endo`. -/
theorem window_proj_spec (O : CurveOps P A) (V : CurveAbs P A G) (hO : CurveOps.Spec O V) (p : P)
    (hp : V.valid p) :
    (windowProj O p).length = 8 ∧ ∀ j : Fin 4, ∃ w,
      (windowProj O p)[(j : ℕ)]? = some w ∧ V.valid w ∧ V.α w = oddMultiples (V.α p) j ∧
      (windowProj O p)[(j : ℕ) + 4]? = some (O.endo w) := by
  have hd := hO.double p hp
  have h1 := hO.add p (O.double p) hp hd.1
  have h2 := hO.add _ (O.double p) h1.1 hd.1
  have h3 := hO.add _ (O.double p) h2.1 hd.1
  refine ⟨rfl, fun j => ?_⟩
  fin_cases j
  · exact ⟨p, rfl, hp, by simp [oddMultiples], rfl⟩
  · refine ⟨_, rfl, h1.1, ?_, rfl⟩
    rw [h1.2, hd.2]; simp [oddMultiples]; abel
  · refine ⟨_, rfl, h2.1, ?_, rfl⟩
    rw [h2.2, h1.2, hd.2]; simp [oddMultiples]; abel
  · refine ⟨_, rfl, h3.1, ?_, rfl⟩
    rw [h3.2, h2.2, h1.2, hd.2]; simp [oddMultiples]; abel

/-- Every point of the window of a valid point is valid. -/
private theorem windowProj_valid (O : CurveOps P A) (V : CurveAbs P A G) (lam : ℕ)
    (hO : CurveOps.Spec O V) (hE : EndoSpec O V lam) (p : P) (hp : V.valid p) :
    ∀ q ∈ windowProj O p, V.valid q := by
  have hd := hO.double p hp
  have h1 := hO.add p (O.double p) hp hd.1
  have h2 := hO.add _ (O.double p) h1.1 hd.1
  have h3 := hO.add _ (O.double p) h2.1 hd.1
  intro q hq
  simp only [windowProj, List.mem_cons, List.not_mem_nil, or_false] at hq
  rcases hq with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  exacts [hp, h1.1, h2.1, h3.1, (hE _ hp).1, (hE _ h1.1).1, (hE _ h2.1).1, (hE _ h3.1).1]

/-- Affine points that normalize the window of a valid point, entry by entry, make its table. -/
private theorem ofWindow_tableOf (O : CurveOps P A) (V : CurveAbs P A G) (lam : ℕ)
    (hO : CurveOps.Spec O V) (hE : EndoSpec O V lam) (p : P) (hp : V.valid p) (c : List A)
    (hc : List.Forall₂ (fun q a => V.validA a ∧ V.αA a = V.α q) (windowProj O p) c) :
    ∃ t, Table.ofWindow c = some t ∧ TableOf V lam p t := by
  have hd := hO.double p hp
  have h1 := hO.add p (O.double p) hp hd.1
  have h2 := hO.add _ (O.double p) h1.1 hd.1
  have h3 := hO.add _ (O.double p) h2.1 hd.1
  simp only [windowProj] at hc
  obtain ⟨a0, c0, r0, hc0, rfl⟩ := List.forall₂_cons_left_iff.mp hc
  obtain ⟨a1, c1, r1, hc1, rfl⟩ := List.forall₂_cons_left_iff.mp hc0
  obtain ⟨a2, c2, r2, hc2, rfl⟩ := List.forall₂_cons_left_iff.mp hc1
  obtain ⟨a3, c3, r3, hc3, rfl⟩ := List.forall₂_cons_left_iff.mp hc2
  obtain ⟨a4, c4, r4, hc4, rfl⟩ := List.forall₂_cons_left_iff.mp hc3
  obtain ⟨a5, c5, r5, hc5, rfl⟩ := List.forall₂_cons_left_iff.mp hc4
  obtain ⟨a6, c6, r6, hc6, rfl⟩ := List.forall₂_cons_left_iff.mp hc5
  obtain ⟨a7, c7, r7, hc7, rfl⟩ := List.forall₂_cons_left_iff.mp hc6
  obtain rfl := List.forall₂_nil_left_iff.mp hc7
  refine ⟨_, rfl, fun j => ?_⟩
  have e0 := (hE p hp).2
  have e1 := (hE _ h1.1).2
  have e2 := (hE _ h2.1).2
  have e3 := (hE _ h3.1).2
  fin_cases j <;> simp [oddMultiples] at *
  · exact ⟨r0.1, r0.2, r4.1, by rw [r4.2, e0]⟩
  · refine ⟨r1.1, by rw [r1.2, h1.2, hd.2]; abel, r5.1, ?_⟩
    rw [r5.2, e1, h1.2, hd.2]; simp only [smul_add, smul_comm lam 2]; abel
  · refine ⟨r2.1, by rw [r2.2, h2.2, h1.2, hd.2]; abel, r6.1, ?_⟩
    rw [r6.2, e2, h2.2, h1.2, hd.2]; simp only [smul_add, smul_comm lam 2]; abel
  · refine ⟨r3.1, by rw [r3.2, h3.2, h2.2, h1.2, hd.2]; abel, r7.1, ?_⟩
    rw [r7.2, e3, h3.2, h2.2, h1.2, hd.2]; simp only [smul_add, smul_comm lam 2]; abel

/-! ## `Table::new` -/

/-- `Table::new(p)` on a valid point builds the table of `p`: the window is valid, the batch
normalization relates it entry by entry to affine points, and those make the table. -/
theorem table_new_spec (O : CurveOps P A) (V : CurveAbs P A G) (lam : ℕ) (hO : CurveOps.Spec O V)
    (hE : EndoSpec O V lam) (p : P) (hp : V.valid p) :
    ∃ t, Table.new O p = some t ∧ TableOf V lam p t := by
  have hb := hO.batchNormalize (windowProj O p) (List.replicate 8 O.identityA)
    (windowProj_valid O V lam hO hE p hp) (by simp [windowProj])
  exact ofWindow_tableOf O V lam hO hE p hp _ hb

/-! ## `Table::batch` -/

/-- `chunksExact8` on a list that starts with an eight-entry chunk. -/
private theorem chunksExact8_append {Z : Type} (c : List Z) (rest : List Z) (hc : c.length = 8) :
    chunksExact8 (c ++ rest) = c :: chunksExact8 rest := by
  match c, hc with
  | [a0, a1, a2, a3, a4, a5, a6, a7], _ => simp [chunksExact8]

/-- `chunks_exact(8)` after a map that gives eight entries per element: chunk `i` is related,
entry by entry, to the eight entries of element `i`. -/
private theorem chunksExact8_flatMap {X Y Z : Type} (R : Y → Z → Prop) (g : X → List Y)
    (xs : List X) (zs : List Z) (hg : ∀ x, (g x).length = 8)
    (hF : List.Forall₂ R (xs.flatMap g) zs) :
    List.Forall₂ (fun x c => List.Forall₂ R (g x) c) xs (chunksExact8 zs) := by
  induction xs generalizing zs with
  | nil =>
    simp at hF
    subst hF
    simp [chunksExact8]
  | cons x xs ih =>
    simp only [List.flatMap_cons] at hF
    have h8 := hg x
    have hc : List.Forall₂ R (g x) (zs.take 8) := by
      have := List.forall₂_take 8 hF
      rwa [List.take_append_of_le_length (by omega), List.take_of_length_le (by omega)] at this
    have hr : List.Forall₂ R (xs.flatMap g) (zs.drop 8) := by
      have := List.forall₂_drop 8 hF
      rwa [List.drop_append_of_le_length (by omega), List.drop_of_length_le (by omega),
        List.nil_append] at this
    have hlen : (zs.take 8).length = 8 := by rw [← hc.length_eq, h8]
    rw [← List.take_append_drop 8 zs, chunksExact8_append _ _ hlen]
    exact List.Forall₂.cons hc (ih _ hr)

/-- `mapM` over a list related entry by entry to `xs` succeeds when `f` succeeds on each related
pair, and relates its results to `xs`. -/
private theorem mapM_forall₂ {X Y Z : Type} (f : Y → Option Z) (Q : X → Y → Prop)
    (R : X → Z → Prop) (xs : List X) (ys : List Y) (h : List.Forall₂ Q xs ys)
    (hf : ∀ x y, Q x y → ∃ z, f y = some z ∧ R x z) :
    ∃ zs, ys.mapM f = some zs ∧ List.Forall₂ R xs zs := by
  induction h with
  | nil => exact ⟨[], rfl, List.Forall₂.nil⟩
  | cons hq _ ih =>
    obtain ⟨z, hz, hR⟩ := hf _ _ hq
    obtain ⟨zs, hzs, hR'⟩ := ih
    exact ⟨z :: zs, by simp [hz, hzs], List.Forall₂.cons hR hR'⟩

/-- `Table::batch(ps)` on valid points builds their tables, one per point and in order: the
`chunks_exact(8)` of the normalized windows are the windows of the points. -/
theorem table_batch_spec (O : CurveOps P A) (V : CurveAbs P A G) (lam : ℕ)
    (hO : CurveOps.Spec O V) (hE : EndoSpec O V lam) (ps : List P) (hps : ∀ p ∈ ps, V.valid p) :
    ∃ ts, Table.batch O ps = some ts ∧ List.Forall₂ (TableOf V lam) ps ts := by
  unfold Table.batch
  dsimp only
  split
  · rename_i h0
    obtain rfl := List.eq_nil_of_length_eq_zero h0
    exact ⟨[], rfl, List.Forall₂.nil⟩
  · have hv : ∀ q ∈ ps.flatMap (windowProj O), V.valid q := by
      intro q hq
      obtain ⟨p, hp, hq⟩ := List.mem_flatMap.mp hq
      exact windowProj_valid O V lam hO hE p (hps p hp) q hq
    have hlen : (ps.flatMap (windowProj O)).length =
        (List.replicate (ps.length * 8) O.identityA).length := by
      rw [List.length_replicate, List.length_flatMap]
      have : ∀ p, (windowProj O p).length = 8 := fun p => rfl
      simp [this]
    have hb := hO.batchNormalize _ _ hv hlen
    have hc := chunksExact8_flatMap _ (windowProj O) ps _ (fun p => rfl) hb
    obtain ⟨ts, hts, hR⟩ := mapM_forall₂ Table.ofWindow
      (fun p c => V.valid p ∧
        List.Forall₂ (fun q a => V.validA a ∧ V.αA a = V.α q) (windowProj O p) c)
      (TableOf V lam) ps _ ((List.forall₂_and_left _ _).2 ⟨hps, hc⟩)
      (fun p c h => ofWindow_tableOf O V lam hO hE p h.1 c h.2)
    exact ⟨ts, hts, hR⟩

end PastaCurves.Glv
