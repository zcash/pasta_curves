import PastaCurves.Glue.Blocks
import PastaCurves.Words
import PastaCurves.Compositions
import PastaCurves.Fields

/-!
# The translated glue meets the entry points' contracts
-/

namespace PastaCurves.Glue

open Aeneas Aeneas.Std pasta_curves

/-- `black_box` returns its value. -/
@[step]
theorem black_box_spec {T : Type} (x : T) : core.hint.black_box x ⦃ res => res = x ⦄ := by
  simp [core.hint.black_box]

/-- The translated borrow chain of `is_canonical_word`: one when `value < modulus` as four-limb
numbers, and zero otherwise. -/
@[step]
theorem is_canonical_word_spec (value modulus : Std.Array Std.U64 4#usize) :
    limbs.is_canonical_word value modulus ⦃ w =>
      w.val = if (limbsOfArray value).toNat < (limbsOfArray modulus).toNat then 1 else 0 ⦄ := by
  unfold limbs.is_canonical_word
  step*
  subst borrow_post borrow1_post borrow2_post w_post
  obtain ⟨c0, b0⟩ := sbb_step (by scalar_tac) (Nat.zero_le 1) difference_post
    (by assumption) i2_post
  obtain ⟨c1, b1⟩ := sbb_step (by scalar_tac) b0 difference1_post (by assumption) i5_post
  obtain ⟨c2, b2⟩ := sbb_step (by scalar_tac) b1 difference2_post (by assumption) i8_post
  obtain ⟨c3, b3⟩ := sbb_step (by scalar_tac) b2 difference3_post (by assumption) i11_post
  have hv : limbsOfArray value = ⟨i.val, i3.val, i6.val, i9.val⟩ := by
    simp [limbsOfArray, i_post, i3_post, i6_post, i9_post]
  have hm : limbsOfArray modulus = ⟨i1.val, i4.val, i7.val, i10.val⟩ := by
    simp [limbsOfArray, i1_post, i4_post, i7_post, i10_post]
  rw [hv, hm]
  simp only [Limbs.toNat]
  split_ifs <;> scalar_tac

/-- The translated `is_canonical` is the model's `isCanonical` on the limbs that it reads. -/
@[step]
theorem is_canonical_spec (value modulus : Std.Array Std.U64 4#usize) :
    limbs.is_canonical value modulus ⦃ b =>
      b = isCanonical (limbsOfArray value) (limbsOfArray modulus) ⦄ := by
  unfold limbs.is_canonical
  step*
  have hiff := isCanonical_iff (limbsOfArray value) (limbsOfArray modulus)
    (limbsOfArray_bounded value) (limbsOfArray_bounded modulus)
  by_cases h : (limbsOfArray value).toNat < (limbsOfArray modulus).toNat
  · rw [if_pos h] at i_post
    rw [hiff.2 h]
    simp only [decide_eq_true_eq]
    scalar_tac
  · rw [if_neg h] at i_post
    have hf : isCanonical (limbsOfArray value) (limbsOfArray modulus) = false :=
      Bool.eq_false_iff.mpr fun ht => h (hiff.1 ht)
    rw [hf]
    simp only [decide_eq_false_iff_not]
    scalar_tac

/-- The word that the glue uses for a Boolean: one for `true`, and zero for `false`. -/
def bword (b : Bool) : Std.U64 := if b then 1#u64 else 0#u64

/-- The bitwise and of two Boolean words is the word of the conjunction. -/
theorem bword_and (p q : Bool) : (bword p &&& bword q) = bword (p && q) := by
  cases p <;> cases q <;> decide

/-- The bitwise or of two Boolean words is the word of the disjunction. -/
theorem bword_or (p q : Bool) : (bword p ||| bword q) = bword (p || q) := by
  cases p <;> cases q <;> decide

/-- A Boolean word is one exactly when its Boolean is true. -/
theorem bword_eq_one (p : Bool) : decide (bword p = 1#u64) = p := by
  cases p <;> decide

/-- A word whose value is one or zero as a proposition holds is that proposition's word. -/
theorem eq_bword (w : Std.U64) (P : Prop) [Decidable P] (h : w.val = if P then 1 else 0) :
    w = bword (decide P) := by
  apply UScalar.eq_of_val_eq
  by_cases hP : P <;> simp_all [bword]

/-- A comparison with `U64::MAX - 2`, the limb bound of the multiplication's second contract. -/
theorem le_max_sub_two {w m : Std.U64} (hm : m.val = U64.rMax - 2) : w ≤ m ↔ w.val ≤ 2^64 - 3 := by
  rw [UScalar.le_equiv, hm]; simp [U64.rMax]

/-- The translated `mul_contract` is the model's `mulContract` on the limbs that it reads. -/
@[step]
theorem mul_contract_spec (lhs rhs modulus : Std.Array Std.U64 4#usize) :
    montgomery.mul_contract lhs rhs modulus ⦃ b =>
      b = mulContract (limbsOfArray lhs) (limbsOfArray rhs) (limbsOfArray modulus) ⦄ := by
  unfold montgomery.mul_contract
  step*
  · decide
  have e14 : i14 = i11 ||| i13 := UScalar.eq_of_val_eq i14_post
  have e13 : i13 = i12 &&& rhs_limbs_ok := UScalar.eq_of_val_eq i13_post
  have eok : rhs_limbs_ok = i7 &&& i10 := UScalar.eq_of_val_eq rhs_limbs_ok_post
  have e7 : i7 = i3 &&& i6 := UScalar.eq_of_val_eq i7_post
  rw [e14, e13, eok, e7, i3_post, i6_post, i10_post, eq_bword _ _ i2_post, eq_bword _ _ i5_post,
    eq_bword _ _ i9_post, eq_bword _ _ i11_post, eq_bword _ _ i12_post, bword_and, bword_and,
    bword_and, bword_or, bword_eq_one]
  obtain ⟨h1, h2, h3⟩ : (limbsOfArray rhs).l1 = i.val ∧ (limbsOfArray rhs).l2 = i4.val ∧
      (limbsOfArray rhs).l3 = i8.val := by
    simp [limbsOfArray, i_post, i4_post, i8_post]
  have hl := isCanonical_iff (limbsOfArray lhs) (limbsOfArray modulus)
    (limbsOfArray_bounded lhs) (limbsOfArray_bounded modulus)
  have hr := isCanonical_iff (limbsOfArray rhs) (limbsOfArray modulus)
    (limbsOfArray_bounded rhs) (limbsOfArray_bounded modulus)
  rw [Bool.eq_iff_iff]
  simp only [mulContract, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, hl, hr,
    le_max_sub_two i1_post, h1, h2, h3, and_assoc]

/-- The shape of the entry points' results: `res` is canonical at the field `F`, and its value times
the weight `k` (`1`, `R`, or a power of `R`) is congruent to `expr` modulo `p`. -/
abbrev Canonical (F : PastaField) (res : Std.Array Std.U64 4#usize) (k expr : ℕ) : Prop :=
  (limbsOfArray res).toNat < F.modulus.toNat ∧ k * (limbsOfArray res).toNat ≡ expr [MOD F.modulus.toNat]

/-- What the entry points' compositions need of a backend's Montgomery blocks at the field `F`.
Each block, at the field's modulus and `inv`, and under the condition that its entry point
asserts, returns a canonical result with the block's congruence. -/
structure BlocksSpec {B : Type} (Blocks : montgomery.MontgomeryBlocks B) (F : PastaField) :
    Prop where
  /-- `add` on canonical operands returns their canonical sum. -/
  add : ∀ lhs rhs, isCanonical (limbsOfArray lhs) F.modulus = true →
    isCanonical (limbsOfArray rhs) F.modulus = true →
    Blocks.add lhs rhs (limbsArray F.modulus) ⦃ res =>
      Canonical F res 1 ((limbsOfArray lhs).toNat + (limbsOfArray rhs).toNat) ⦄
  /-- `sub` on canonical operands returns their canonical difference. -/
  sub : ∀ lhs rhs, isCanonical (limbsOfArray lhs) F.modulus = true →
    isCanonical (limbsOfArray rhs) F.modulus = true →
    Blocks.sub lhs rhs (limbsArray F.modulus) ⦃ res =>
      (limbsOfArray res).toNat < F.modulus.toNat ∧
        (limbsOfArray res).toNat + (limbsOfArray rhs).toNat ≡ (limbsOfArray lhs).toNat
          [MOD F.modulus.toNat] ⦄
  /-- `mul` under the condition that `mul_with` asserts returns a canonical `res` with
  `R * res ≡ lhs * rhs (mod p)`. -/
  mul : ∀ lhs rhs, mulContract (limbsOfArray lhs) (limbsOfArray rhs) F.modulus = true →
    Blocks.mul lhs rhs (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res R ((limbsOfArray lhs).toNat * (limbsOfArray rhs).toNat) ⦄
  /-- `square` on a canonical operand returns a canonical `res` with `R * res ≡ value² (mod p)`. -/
  square : ∀ value, isCanonical (limbsOfArray value) F.modulus = true →
    Blocks.square value (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res R ((limbsOfArray value).toNat * (limbsOfArray value).toNat) ⦄
  /-- `from_mont` on any operand returns a canonical `res` with `R * res ≡ value (mod p)`. -/
  from_mont : ∀ value, Blocks.from_mont value (limbsArray F.modulus) (word F.inv) ⦃ res =>
    Canonical F res R (limbsOfArray value).toNat ⦄

/-- `add_with` at a Pasta field: for canonical operands, as it asserts, the canonical sum. -/
theorem add_with_spec {B : Type} (Blocks : montgomery.MontgomeryBlocks B) (F : PastaField)
    (hS : BlocksSpec Blocks F) (lhs rhs : Std.Array Std.U64 4#usize)
    (hl : isCanonical (limbsOfArray lhs) F.modulus = true)
    (hr : isCanonical (limbsOfArray rhs) F.modulus = true) :
    montgomery.add_with Blocks lhs rhs (limbsArray F.modulus) ⦃ res =>
      Canonical F res 1 ((limbsOfArray lhs).toNat + (limbsOfArray rhs).toNat) ⦄ := by
  unfold montgomery.add_with
  have hm := limbsOfArray_limbsArray F.bounded
  step*
  exact hS.add lhs rhs hl hr

/-- `sub_with` at a Pasta field: for canonical operands, as it asserts, the canonical
difference. -/
theorem sub_with_spec {B : Type} (Blocks : montgomery.MontgomeryBlocks B) (F : PastaField)
    (hS : BlocksSpec Blocks F) (lhs rhs : Std.Array Std.U64 4#usize)
    (hl : isCanonical (limbsOfArray lhs) F.modulus = true)
    (hr : isCanonical (limbsOfArray rhs) F.modulus = true) :
    montgomery.sub_with Blocks lhs rhs (limbsArray F.modulus) ⦃ res =>
      (limbsOfArray res).toNat < F.modulus.toNat ∧
        (limbsOfArray res).toNat + (limbsOfArray rhs).toNat ≡ (limbsOfArray lhs).toNat
          [MOD F.modulus.toNat] ⦄ := by
  unfold montgomery.sub_with
  have hm := limbsOfArray_limbsArray F.bounded
  step*
  exact hS.sub lhs rhs hl hr

/-- `mul_with` at a Pasta field: when the condition that it asserts holds, a canonical `res` with
`R * res ≡ lhs * rhs (mod p)`. -/
theorem mul_with_spec {B : Type} (Blocks : montgomery.MontgomeryBlocks B) (F : PastaField)
    (hS : BlocksSpec Blocks F) (lhs rhs : Std.Array Std.U64 4#usize)
    (h : mulContract (limbsOfArray lhs) (limbsOfArray rhs) F.modulus = true) :
    montgomery.mul_with Blocks lhs rhs (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res R ((limbsOfArray lhs).toNat * (limbsOfArray rhs).toNat) ⦄ := by
  unfold montgomery.mul_with
  have hm := limbsOfArray_limbsArray F.bounded
  step*
  exact hS.mul lhs rhs h

/-- `square_with` at a Pasta field: for a canonical operand, as it asserts, a canonical `res` with
`R * res ≡ value² (mod p)`. -/
theorem square_with_spec {B : Type} (Blocks : montgomery.MontgomeryBlocks B) (F : PastaField)
    (hS : BlocksSpec Blocks F) (value : Std.Array Std.U64 4#usize)
    (h : isCanonical (limbsOfArray value) F.modulus = true) :
    montgomery.square_with Blocks value (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res R ((limbsOfArray value).toNat * (limbsOfArray value).toNat) ⦄ := by
  unfold montgomery.square_with
  have hm := limbsOfArray_limbsArray F.bounded
  step*
  exact hS.square value h

/-- `from_mont_with` at a Pasta field: for any operand, a canonical `res` with
`R * res ≡ value (mod p)`. -/
theorem from_mont_with_spec {B : Type} (Blocks : montgomery.MontgomeryBlocks B) (F : PastaField)
    (hS : BlocksSpec Blocks F) (value : Std.Array Std.U64 4#usize) :
    montgomery.from_mont_with Blocks value (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res R (limbsOfArray value).toNat ⦄ := by
  unfold montgomery.from_mont_with
  exact hS.from_mont value

/-- One squaring of the accumulator of `sqr_n_mul`: a Montgomery square of an accumulator with
weight `R^(2^s - 1)` has weight `R^(2^(s+1) - 1)`, which is two copies of the previous weight
and one `R`, against the square of the previous power. -/
theorem sqr_weight_step {p a a' v : ℕ} (s : ℕ) (hc : R^(2^s - 1) * a ≡ v^(2^s) [MOD p])
    (hsq : R * a' ≡ a * a [MOD p]) : R^(2^(s+1) - 1) * a' ≡ v^(2^(s+1)) [MOD p] := by
  have hpos := Nat.two_pow_pos s
  have e1 : R^(2^(s+1) - 1) = R^(2^s - 1) * R^(2^s - 1) * R := by
    rw [← pow_add, ← pow_succ, pow_succ 2]
    congr 1
    generalize 2^s = m at hpos ⊢
    omega
  have e2 : v^(2^(s+1)) = v^(2^s) * v^(2^s) := by rw [pow_succ, pow_mul, sq]
  rw [e1, e2]
  calc R^(2^s - 1) * R^(2^s - 1) * R * a'
      = R^(2^s - 1) * R^(2^s - 1) * (R * a') := by ring
    _ ≡ R^(2^s - 1) * R^(2^s - 1) * (a * a) [MOD p] := Nat.ModEq.mul_left _ hsq
    _ = (R^(2^s - 1) * a) * (R^(2^s - 1) * a) := by ring
    _ ≡ v^(2^s) * v^(2^s) [MOD p] := Nat.ModEq.mul hc hc

/-- The multiplication that ends `sqr_n_mul`: a Montgomery product of an accumulator with weight
`R^(2^k - 1)` and any `w` has weight `R^(2^k)`. -/
theorem mul_weight_step {p a res v w : ℕ} (k : ℕ) (hc : R^(2^k - 1) * a ≡ v^(2^k) [MOD p])
    (hm : R * res ≡ a * w [MOD p]) : R^(2^k) * res ≡ v^(2^k) * w [MOD p] := by
  have hpos := Nat.two_pow_pos k
  have e : R^(2^k) = R^(2^k - 1) * R := by
    rw [← pow_succ]
    congr 1
    omega
  rw [e]
  calc R^(2^k - 1) * R * res
      = R^(2^k - 1) * (R * res) := by ring
    _ ≡ R^(2^k - 1) * (a * w) [MOD p] := Nat.ModEq.mul_left _ hm
    _ = (R^(2^k - 1) * a) * w := by ring
    _ ≡ v^(2^k) * w [MOD p] := Nat.ModEq.mul_right _ hc

/-- The loop of `sqr_n_mul_with`, from any point of its range: a canonical accumulator whose
weight relation with `v` holds after `iter.start` squarings ends canonical, with the relation
after `iter.end` squarings. -/
theorem sqr_n_mul_with_loop_spec {B : Type} (Blocks : montgomery.MontgomeryBlocks B)
    (F : PastaField) (hS : BlocksSpec Blocks F) (v : ℕ) (iter : core.ops.range.Range Std.Usize)
    (acc : Std.Array Std.U64 4#usize) (hle : iter.start.val ≤ iter.«end».val)
    (hacc : Canonical F acc (R^(2^iter.start.val - 1)) (v^(2^iter.start.val))) :
    montgomery.sqr_n_mul_with_loop Blocks iter (limbsArray F.modulus) (word F.inv) acc ⦃ res =>
      Canonical F res (R^(2^iter.«end».val - 1)) (v^(2^iter.«end».val)) ⦄ := by
  unfold montgomery.sqr_n_mul_with_loop
  apply loop.spec_decr_nat (fun x => x.1.«end».val - x.1.start.val)
    (fun x => x.1.«end» = iter.«end» ∧ x.1.start.val ≤ x.1.«end».val ∧
      Canonical F x.2 (R^(2^x.1.start.val - 1)) (v^(2^x.1.start.val)))
  · rintro ⟨it, a⟩ ⟨hend, hle', ha, hca⟩
    simp only at hend hle' ha hca
    unfold montgomery.sqr_n_mul_with_loop.body
    step*
    · -- The range is exhausted, so `it.start` is `iter.end`.
      rename_i hnone
      split_ifs at o_post with hlt
      · simp [o_post.1] at hnone
      · have e : it.start.val = iter.«end».val := by rw [← hend]; omega
        rw [e] at hca
        exact ⟨ha, hca⟩
    · -- One more squaring: the accumulator stays canonical, and its weight relation moves on.
      rename_i hsome
      split_ifs at o_post with hlt
      · obtain ⟨-, hstart⟩ := o_post
        step with square_with_spec Blocks F hS a
          ((isCanonical_iff _ _ (limbsOfArray_bounded a) F.bounded).2 ha) as ⟨acc1, hlt1, hc1⟩
        have hendv : iter1.«end».val = it.«end».val := by rw [o_post1]
        refine ⟨by rw [o_post1, hend], by omega, hlt1, ?_, by omega⟩
        rw [hstart]
        exact sqr_weight_step _ hca hc1
      · simp [o_post.1] at hsome
  · exact ⟨rfl, hle, hacc⟩

/-- `sqr_n_mul_with` at a Pasta field: for a canonical `value`, as it asserts, and any `rhs`, a
canonical `res` with `R^(2^count) * res ≡ value^(2^count) * rhs (mod p)`. The squarings keep the
accumulator canonical, so the multiplication is under its first contract. -/
theorem sqr_n_mul_with_spec {B : Type} (Blocks : montgomery.MontgomeryBlocks B) (F : PastaField)
    (hS : BlocksSpec Blocks F) (value : Std.Array Std.U64 4#usize) (count : Std.Usize)
    (rhs : Std.Array Std.U64 4#usize) (h : isCanonical (limbsOfArray value) F.modulus = true) :
    montgomery.sqr_n_mul_with Blocks value count rhs (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res (R^(2^count.val))
        ((limbsOfArray value).toNat^(2^count.val) * (limbsOfArray rhs).toNat) ⦄ := by
  unfold montgomery.sqr_n_mul_with
  have hm := limbsOfArray_limbsArray F.bounded
  have hv := (isCanonical_iff _ _ (limbsOfArray_bounded value) F.bounded).1 h
  step*
  step with sqr_n_mul_with_loop_spec Blocks F hS (limbsOfArray value).toNat
    { start := 0#usize, «end» := count } value (by simp) ⟨hv, by simp; rfl⟩ as ⟨acc, hacc, hcacc⟩
  have hacc' := (isCanonical_iff _ _ (limbsOfArray_bounded acc) F.bounded).2 hacc
  step with mul_with_spec Blocks F hS acc rhs (by simp [mulContract, hacc']) as ⟨res, hres, hcres⟩
  exact ⟨hres, mul_weight_step _ hcacc hcres⟩

/-- The block theorems' form of a result: `res` is bounded, canonical at the field `F`, and
`k * res ≡ expr (mod p)`. -/
abbrev LimbsCanonical (F : PastaField) (res : Limbs) (k expr : ℕ) : Prop :=
  res.Bounded ∧ res.toNat < F.modulus.toNat ∧ k * res.toNat ≡ expr [MOD F.modulus.toNat]

/-- A block's result written back as an array: a result of the block theorems' form gives the
entry points' form. -/
theorem ok_limbsArray_spec {F : PastaField} {res : Limbs} {k expr : ℕ} (h : LimbsCanonical F res k expr) :
    (.ok (limbsArray res) : Result (Std.Array Std.U64 4#usize)) ⦃ s => Canonical F s k expr ⦄ := by
  simp only [WP.spec_ok, Canonical, limbsOfArray_limbsArray h.1]
  exact h.2

/-- The arithmetic meaning of the canonicity check on an array's limbs. -/
theorem lt_of_isCanonical {F : PastaField} {x : Std.Array Std.U64 4#usize}
    (h : isCanonical (limbsOfArray x) F.modulus = true) :
    (limbsOfArray x).toNat < F.modulus.toNat :=
  (isCanonical_iff _ _ (limbsOfArray_bounded x) F.bounded).1 h

/-- What a backend's block theorems give at the field `F`, on limbs, in the form that they state
it: the contracts of `BlocksSpec` for the blocks' models, with `mul` under each of its two
contracts. -/
structure LimbsSpec (add sub : Limbs → Limbs → Limbs → Limbs)
    (mul : Limbs → Limbs → Limbs → Nat → Limbs) (square fromMont : Limbs → Limbs → Nat → Limbs)
    (F : PastaField) : Prop where
  /-- `add` on canonical operands is their canonical sum. -/
  add : ∀ lhs rhs : Limbs, lhs.Bounded → rhs.Bounded → lhs.toNat < F.modulus.toNat →
    rhs.toNat < F.modulus.toNat → ∀ res, res = add lhs rhs F.modulus →
      res.Bounded ∧ res.toNat < F.modulus.toNat ∧
        res.toNat ≡ lhs.toNat + rhs.toNat [MOD F.modulus.toNat]
  /-- `sub` on canonical operands is their canonical difference. -/
  sub : ∀ lhs rhs : Limbs, lhs.Bounded → rhs.Bounded → lhs.toNat < F.modulus.toNat →
    rhs.toNat < F.modulus.toNat → ∀ res, res = sub lhs rhs F.modulus →
      res.Bounded ∧ res.toNat < F.modulus.toNat ∧
        res.toNat + rhs.toNat ≡ lhs.toNat [MOD F.modulus.toNat]
  /-- `mul` with a canonical `lhs` and any `rhs`. -/
  mul_of_lhs_lt : ∀ lhs rhs : Limbs, lhs.Bounded → rhs.Bounded → lhs.toNat < F.modulus.toNat →
    ∀ res, res = mul lhs rhs F.modulus F.inv → LimbsCanonical F res R (lhs.toNat * rhs.toNat)
  /-- `mul` with any `lhs` and a canonical `rhs` whose limbs 1 to 3 are at most `2^64 - 3`. -/
  mul_of_rhs_lt : ∀ lhs rhs : Limbs, lhs.Bounded → rhs.Bounded → rhs.toNat < F.modulus.toNat →
    rhs.l1 + 3 ≤ 2^64 ∧ rhs.l2 + 3 ≤ 2^64 ∧ rhs.l3 + 3 ≤ 2^64 →
    ∀ res, res = mul lhs rhs F.modulus F.inv → LimbsCanonical F res R (lhs.toNat * rhs.toNat)
  /-- `square` on a canonical operand. -/
  square : ∀ value : Limbs, value.Bounded → value.toNat < F.modulus.toNat →
    ∀ res, res = square value F.modulus F.inv → LimbsCanonical F res R (value.toNat * value.toNat)
  /-- The conversion out of Montgomery form on any operand. -/
  fromMont : ∀ value : Limbs, value.Bounded →
    ∀ res, res = fromMont value F.modulus F.inv → LimbsCanonical F res R value.toNat

/-- A backend's record meets the blocks' contracts at the field `F` when its block theorems give
their contracts there. `mul`'s contract is the one of its two that the condition that `mul_with`
asserts selects. -/
theorem blocksOf_spec {add sub : Limbs → Limbs → Limbs → Limbs}
    {mul : Limbs → Limbs → Limbs → Nat → Limbs} {square fromMont : Limbs → Limbs → Nat → Limbs}
    (F : PastaField) (h : LimbsSpec add sub mul square fromMont F) :
    BlocksSpec (blocksOf add sub mul square fromMont) F where
  add lhs rhs hl hr := by
    obtain ⟨hr', hlt, hc⟩ := h.add _ _ (limbsOfArray_bounded lhs) (limbsOfArray_bounded rhs)
      (lt_of_isCanonical hl) (lt_of_isCanonical hr) _ rfl
    simp only [blocksOf, limbsOfArray_limbsArray F.bounded]
    exact ok_limbsArray_spec ⟨hr', hlt, by rwa [one_mul]⟩
  sub lhs rhs hl hr := by
    obtain ⟨hr', hlt, hc⟩ := h.sub _ _ (limbsOfArray_bounded lhs) (limbsOfArray_bounded rhs)
      (lt_of_isCanonical hl) (lt_of_isCanonical hr) _ rfl
    simp only [blocksOf, WP.spec_ok, limbsOfArray_limbsArray F.bounded,
      limbsOfArray_limbsArray hr']
    exact ⟨hlt, hc⟩
  mul lhs rhs hc := by
    have hb := limbsOfArray_bounded
    have hc' := (mulContract_iff _ _ F.modulus (hb lhs) (hb rhs) F.bounded).1 hc
    simp only [blocksOf, limbsOfArray_limbsArray F.bounded, word_val F.inv_lt]
    exact ok_limbsArray_spec (hc'.elim
      (fun hlt => h.mul_of_lhs_lt _ _ (hb lhs) (hb rhs) hlt _ rfl)
      (fun ⟨hlt, hlimbs⟩ => h.mul_of_rhs_lt _ _ (hb lhs) (hb rhs) hlt hlimbs _ rfl))
  square value hv := by
    simp only [blocksOf, limbsOfArray_limbsArray F.bounded, word_val F.inv_lt]
    exact ok_limbsArray_spec (h.square _ (limbsOfArray_bounded value) (lt_of_isCanonical hv) _ rfl)
  from_mont value := by
    simp only [blocksOf, limbsOfArray_limbsArray F.bounded, word_val F.inv_lt]
    exact ok_limbsArray_spec (h.fromMont _ (limbsOfArray_bounded value) _ rfl)

end PastaCurves.Glue
