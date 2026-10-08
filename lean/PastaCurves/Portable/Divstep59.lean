import PastaCurves.Portable.Spec
import PastaCurves.Inversion.PackedWords
import PastaCurves.Inversion.Divstep59

/-!
# The translated `divstep59` satisfies its contract

`divstep59` of `src/inversion/portable.rs` runs the packed recurrence of `Inversion/Packed.lean` on
64-bit words, in three batches of 20, 20, and 19 steps, as `Inversion.divstep59` does on integers.
`divstep59_spec` shows that it satisfies the `divstep59` field of `InvertBlocks.Spec`: its words
agree with the model's modulo `2^64`, and Corollary 8 (`Inversion.divstep59_spec`) relates the model
to the true state.

The proof goes bottom up. One packed step is the plain `divstep` on the packed state
(`divstep_spec`), through the shared case lemmas of `Inversion/PackedWords.lean`. A batch's loop
iterates it (`batch_loop_spec`), as the shared `rounds_words` does. The decoder is Lemma 7 on words
(`unpack_spec`), where `up = a + 2^21 b` keeps its 64-bit operations from wrapping. The next low
words and the matrix products are the model's (`next_low_spec`, `mat_mul_spec`), and a whole batch
is the model's `batch` through Lemma 6 (`batch_spec`).
-/

namespace PastaCurves.Portable

open Aeneas Aeneas.Std
open Inversion (State)

/-- The bitwise and of two masks, each zero or all ones, is a mask. -/
theorem mask_and {a b : ℕ} (ha : a = 0 ∨ a = 2^64 - 1) (hb : b = 0 ∨ b = 2^64 - 1) :
    a &&& b = 0 ∨ a &&& b = 2^64 - 1 := by
  rcases ha with rfl | rfl
  · simp
  · rcases hb with rfl | rfl
    · simp
    · simp

/-- Aeneas' wrapping subtraction is `addw` of the negation, in the words of `Semantics.lean`. -/
theorem wrapping_sub_val_addw (x y : Std.U64) :
    (core.num.U64.wrapping_sub x y).val = addw x.val (negw y.val) := by
  rw [core.num.U64.wrapping_sub_val_eq]
  simp [negw, addw, regMod, U64.size, U64.numBits, Nat.add_mod]

/-- Aeneas' wrapping addition is `addw`. -/
theorem wrapping_add_val_addw (x y : Std.U64) :
    (core.num.U64.wrapping_add x y).val = addw x.val y.val := by
  simp [core.num.U64.wrapping_add_val_eq, addw, regMod, U64.size, U64.numBits]

/-- The all-ones mask leaves a word unchanged. -/
theorem ones_and {x : ℕ} (hx : x < 2^64) : (2^64 - 1) &&& x = x := by
  rw [Nat.and_comm, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hx]

open pasta_curves.inversion.portable in
/-- One packed divstep on words is `divstep` on the packed state `s` that the words carry, as two's
complement words. `s.f` and `s.two_delta` are odd, as they are throughout the recurrence, and
`hG'` is the no-wrap condition on the sum `g ± f` before it is halved. -/
@[step]
theorem divstep_spec (s : State) (hf : s.f % 2 = 1) (hd : s.two_delta % 2 = 1)
    (hD : |s.two_delta| < 2^62) (hG : |s.g| < 2^63) (hG' : |(Inversion.divstep s).g| < 2^62)
    (two_delta f g : Std.U64) (ed : (two_delta.val : ℤ) = s.two_delta % 2^64)
    (ef : (f.val : ℤ) = s.f % 2^64) (eg : (g.val : ℤ) = s.g % 2^64) :
    divstep two_delta f g ⦃ (two_delta' f' g' : Std.U64) =>
      (two_delta'.val : ℤ) = (Inversion.divstep s).two_delta % 2^64 ∧
        (f'.val : ℤ) = (Inversion.divstep s).f % 2^64 ∧
        (g'.val : ℤ) = (Inversion.divstep s).g % 2^64 ⦄ := by
  unfold divstep
  step*
  · agrind [Nat.and_one_is_mod]
  · agrind [mask_and]
  · agrind [mask_and]
  · agrind [mask_and]
  · -- The step's words, in the word operations of the shared case lemmas.
    have hi : i.val = g.val % 2 := by rw [i_post, UScalar.val_and]; agrind [Nat.and_one_is_mod]
    have h1 : i1.val = negw two_delta.val := by
      rw [i1_post, wrapping_sub_val_addw]
      simp [addw, negw, regMod]
    have h3 : i3.val = addw (negw two_delta.val) 2 := by
      rw [i3_post, wrapping_sub_val_addw]
      simp [addw, Nat.add_comm]
    have h4 : i4.val = addw two_delta.val 2 := by rw [i4_post, wrapping_add_val_addw]; rfl
    have h5 : i5.val = addw g.val (negw f.val) := by rw [i5_post, wrapping_sub_val_addw]
    have h6 : i6.val = odd.val &&& f.val := by rw [i6_post, UScalar.val_and]
    have h7 : i7.val = addw g.val i6.val := by rw [i7_post, wrapping_add_val_addw]
    have hsw : swap.val = odd.val &&& i2.val := by rw [swap_post, UScalar.val_and]
    have hgi : (i.val : ℤ) = s.g % 2 := by
      rw [hi, Int.natCast_mod, eg, Int.emod_emod_of_dvd _ (by norm_num)]
      rfl
    have htd := two_delta.hBounds
    obtain ⟨hDl, hDu⟩ := abs_lt.mp hD
    have hfb := f.hBounds
    rcases Int.emod_two_eq_zero_or_one s.g with hg0 | hg1
    · -- `g` even: no swap, `two_delta + 2`, `f`, and `g / 2`.
      have hodd : odd.val = 0 := by agrind
      have hswap : swap.val = 0 := by rw [hsw, hodd]; simp
      have hi6 : i6.val = 0 := by rw [h6, hodd]; simp
      simp only [hswap, if_true] at two_delta_new_post f_new_post sum_post
      subst two_delta_new f_new sum
      obtain ⟨r1, r2, r3, -⟩ := Inversion.divstep_words_even s hd hG two_delta.val f.val g.val
        ed ef eg hg0 0 two_delta.val f.val i7.val i4.val i8.val rfl rfl rfl (by rw [h7, hi6]) h4
        i8_post
      exact ⟨r1, r2, r3⟩
    · rcases lt_or_ge 0 s.two_delta with hdpos | hdnp
      · -- The swap: `2 - two_delta`, `f := g`, and `(g - f) / 2`.
        have hodd : odd.val = 2^64 - 1 := by agrind
        have hi2 : i2.val = 2^64 - 1 := by
          rw [i2_post, if_neg]
          rw [h1]
          simp only [negw, regMod]
          agrind
        have hswap : swap.val ≠ 0 := by rw [hsw, hodd, hi2]; simp
        simp only [hswap, if_false] at two_delta_new_post f_new_post sum_post
        subst two_delta_new f_new sum
        obtain ⟨r1, r2, r3, -⟩ := Inversion.divstep_words_swap s hf hd hG' two_delta.val f.val
          g.val htd hfb ed ef eg ⟨hdpos, hg1⟩ (negw f.val) (negw two_delta.val) g.val i5.val
          i3.val i8.val rfl rfl rfl h5 h3 i8_post
        exact ⟨r1, r2, r3⟩
      · -- The addition: `two_delta + 2`, `f`, and `(g + f) / 2`.
        have hodd : odd.val = 2^64 - 1 := by agrind
        have hi2 : i2.val = 0 := by
          rw [i2_post, if_pos]
          rw [h1]
          simp only [negw, regMod]
          agrind
        have hswap : swap.val = 0 := by rw [hsw, hi2]; simp
        have hi6 : i6.val = f.val := by rw [h6, hodd, ones_and hfb]
        simp only [hswap, if_true] at two_delta_new_post f_new_post sum_post
        subst two_delta_new f_new sum
        obtain ⟨r1, r2, r3, -⟩ := Inversion.divstep_words_add s hf hd hD hG' two_delta.val f.val
          g.val ed ef eg hg1 hdnp f.val two_delta.val f.val i7.val i4.val i8.val rfl rfl rfl
          (by rw [h7, hi6]) h4 i8_post
        exact ⟨r1, r2, r3⟩

open pasta_curves.inversion.portable in
/-- The loop of `batch`, from any point of its range: words that carry `divsteps iter.start P` end
carrying `divsteps iter.end P`. The hypotheses are those of `PackedStep.Spec.rounds_words`: `f` and
`two_delta` odd, `two_delta` small enough for every step, and Lemma 6′'s no-wrap bound on each new
`g`. -/
theorem batch_loop_spec (P : State) (hf : P.f % 2 = 1) (hd : P.two_delta % 2 = 1)
    (iter : core.ops.range.Range Std.U32) (hle : iter.start.val ≤ iter.«end».val)
    (hD : |P.two_delta| + 2 * iter.«end».val < 2^62) (hg0 : |P.g| < 2^63)
    (hg : ∀ j, j < iter.«end».val → |(Inversion.divsteps (j + 1) P).g| < 2^62)
    (two_delta f g : Std.U64)
    (ed : (two_delta.val : ℤ) = (Inversion.divsteps iter.start.val P).two_delta % 2^64)
    (ef : (f.val : ℤ) = (Inversion.divsteps iter.start.val P).f % 2^64)
    (eg : (g.val : ℤ) = (Inversion.divsteps iter.start.val P).g % 2^64) :
    batch_loop iter two_delta f g ⦃ (two_delta' f' g' : Std.U64) =>
      (two_delta'.val : ℤ) = (Inversion.divsteps iter.«end».val P).two_delta % 2^64 ∧
        (f'.val : ℤ) = (Inversion.divsteps iter.«end».val P).f % 2^64 ∧
        (g'.val : ℤ) = (Inversion.divsteps iter.«end».val P).g % 2^64 ⦄ := by
  unfold batch_loop
  apply loop.spec_decr_nat (fun (it, _) => it.«end».val - it.start.val)
    (fun (it, two_delta, f, g) => it.«end» = iter.«end» ∧ it.start.val ≤ it.«end».val ∧
      (two_delta.val : ℤ) = (Inversion.divsteps it.start.val P).two_delta % 2^64 ∧
      (f.val : ℤ) = (Inversion.divsteps it.start.val P).f % 2^64 ∧
      (g.val : ℤ) = (Inversion.divsteps it.start.val P).g % 2^64)
  · clear hle ed ef eg
    rintro ⟨it, two_delta, f, g⟩ ⟨hend, hle, ed, ef, eg⟩
    unfold batch_loop.body
    step*
    · exact Inversion.divsteps_f_odd _ P hf
    · rw [Inversion.divsteps_two_delta_emod_two]; exact hd
    · have := Inversion.divsteps_two_delta_abs_le it.start.val P
      agrind
    · rcases Nat.eq_zero_or_pos it.start.val with h0 | hpos
      · rw [h0]; exact hg0
      · have := hg (it.start.val - 1) (by agrind)
        rw [show it.start.val - 1 + 1 = it.start.val by agrind] at this
        agrind
    · rw [← Inversion.divsteps_succ']; exact hg _ (by agrind)
    · agrind [Inversion.divsteps_succ']
  · exact ⟨rfl, hle, ed, ef, eg⟩

/-- A value below `2^63` in magnitude is its own balanced residue modulo `2^64`, so the signed
64-bit operation that computes it does not wrap. -/
theorem bmod_of_small {x : ℤ} (h : |x| < 2^63) : x.bmod (2^64) = x := by
  rw [abs_lt] at h
  exact Int.bmod_eq_of_le (by agrind) (by agrind)

/-- The balanced residue modulo `2^64` of a value congruent to a small `y` is `y`. -/
theorem bmod_of_emod {x y : ℤ} (h : x % 2^64 = y % 2^64) (hy : |y| < 2^63) :
    x.bmod (2^64) = y := by
  rw [← bmod_of_small hy]
  agrind [Int.bmod]

open pasta_curves.inversion.portable in
/-- Lemma 7 on words: the decoder, on a word that carries `φ - 2^(41-k) a - 2^(62-k) b` with
`|φ| < 2^20` and `a` and `b` in `(-2^k, 2^k]`, returns words that carry `a` and `b`. None of its
64-bit operations wrap. -/
@[step]
theorem unpack_spec (k : Std.U32) (hk : k.val ≤ 20) (w : Std.U64) (φ a b : ℤ)
    (hφ : |φ| < 2^20)
    (ha : -(2 : ℤ)^k.val < a ∧ a ≤ 2^k.val) (hb : -(2 : ℤ)^k.val < b ∧ b ≤ 2^k.val)
    (ew : (w.val : ℤ) = (φ - 2^(41 - k.val) * a - 2^(62 - k.val) * b) % 2^64) :
    unpack k w ⦃ (a' b' : Std.U64) =>
      (a'.val : ℤ) = a % 2^64 ∧ (b'.val : ℤ) = b % 2^64 ⦄ := by
  unfold unpack
  step*
  -- The model's decoder recovers `(a, b)`, through `up = a + 2^21 b`.
  have hmodel := Inversion.unpack_spec k.val hk φ a b hφ ha.1 ha.2
  simp only [Inversion.unpack, Prod.mk.injEq] at hmodel
  generalize hW : φ - 2^(41 - k.val) * a - 2^(62 - k.val) * b = W at hmodel ew
  generalize hU : (-W + 2^(40 - k.val)) / 2^(41 - k.val) = U at hmodel
  obtain ⟨hmu, hmv⟩ := hmodel
  -- The bounds that keep the 64-bit operations from wrapping.
  have hK : (2 : ℤ)^k.val ≤ 2^20 := pow_le_pow_right₀ (by norm_num) hk
  have h40 : (2 : ℤ)^(40 - k.val) ≤ 2^40 := pow_le_pow_right₀ (by norm_num) (by agrind)
  have h40' : (0 : ℤ) < 2^(40 - k.val) := by positivity
  have hb1 : |(2 : ℤ)^(41 - k.val) * a| ≤ 2^41 := by
    rw [abs_mul, abs_of_pos (by positivity)]
    calc (2 : ℤ)^(41 - k.val) * |a| ≤ 2^(41 - k.val) * 2^k.val :=
          mul_le_mul_of_nonneg_left (abs_le.mpr ⟨by agrind, ha.2⟩) (by positivity)
      _ = 2^41 := by rw [← pow_add]; congr 1; agrind
  have hb2 : |(2 : ℤ)^(62 - k.val) * b| ≤ 2^62 := by
    rw [abs_mul, abs_of_pos (by positivity)]
    calc (2 : ℤ)^(62 - k.val) * |b| ≤ 2^(62 - k.val) * 2^k.val :=
          mul_le_mul_of_nonneg_left (abs_le.mpr ⟨by agrind, hb.2⟩) (by positivity)
      _ = 2^62 := by rw [← pow_add]; congr 1; agrind
  rw [abs_le] at hb1 hb2
  rw [abs_lt] at hφ
  have hW' : -W + 2^(40 - k.val) < 2^63 ∧ -(2^63) ≤ -W + 2^(40 - k.val) := by
    rw [← hW]; constructor <;> agrind
  have hUv : U = a + 2^21 * b := by agrind
  -- The decoder's values, none of which wrap.
  have hi : (i.val : ℤ) % 2^64 = (-W) % 2^64 := by
    rw [i_post, core.num.U64.wrapping_sub_val_eq]
    scalar_tac
  have ht : t.val = -W := by
    rw [t_post, UScalar.hcast_val_eq]
    exact bmod_of_emod hi (by rw [abs_lt]; constructor <;> agrind)
  have hi2 : i2.val = 2^(40 - k.val) := by
    rw [i2_post, i1_post, Int.shiftLeft_eq, one_mul]
    simp only [I64.size, I64.numBits, IScalarTy.I64_numBits_eq]
    exact bmod_of_small (by rw [abs_of_pos h40']; agrind)
  have hi3 : i3.val = -W + 2^(40 - k.val) := by
    rw [i3_post, core.num.I64.wrapping_add_val_eq, ht, hi2]
    exact bmod_of_small (by rw [abs_lt]; constructor <;> agrind)
  have hup : up.val = U := by
    rw [up_post, hi3, i4_post, Int.shiftRight_eq_div_pow, ← hU]
    push_cast
    rfl
  have hi5 : i5.val = 2^20 := by
    rw [i5_post, Int.shiftLeft_eq, one_mul]
    simp only [I64.size, I64.numBits, IScalarTy.I64_numBits_eq]
    exact bmod_of_small (by norm_num)
  have hi7 : i7.val = U + 2^20 - 1 := by
    rw [i7_post, core.num.I64.wrapping_add_val_eq, i6_post, hi5, hup]
    rw [show U + (2^20 - 1) = U + 2^20 - 1 by ring]
    exact bmod_of_small (by rw [abs_lt]; constructor <;> agrind)
  have hvb : v.val = b := by
    rw [v_post, hi7, Int.shiftRight_eq_div_pow]
    push_cast
    exact hmv
  have hi8 : i8.val = b * 2^21 := by
    rw [i8_post, Int.shiftLeft_eq, hvb]
    simp only [I64.size, I64.numBits, IScalarTy.I64_numBits_eq]
    exact bmod_of_small (by rw [abs_lt]; constructor <;> agrind)
  have hi9 : i9.val = a := by
    rw [i9_post, core.num.I64.wrapping_sub_val_eq, hi8, hup]
    rw [show U - b * 2^21 = a by agrind]
    exact bmod_of_small (by rw [abs_lt]; constructor <;> agrind)
  -- The words returned carry `a` and `b`.
  rw [i10_post, i11_post, IScalar.hcast_val_eq, IScalar.hcast_val_eq, hi9, hvb]
  simp only [UScalarTy.U64_numBits_eq]
  constructor
  · rw [Int.toNat_of_nonneg (Int.emod_nonneg _ (by norm_num))]
  · rw [Int.toNat_of_nonneg (Int.emod_nonneg _ (by norm_num))]

/-- A word carries itself modulo `2^64`. -/
theorem word_emod (w : Std.U64) : (w.val : ℤ) = (w.val : ℤ) % 2^64 :=
  (Int.emod_eq_of_lt (by positivity) (by exact_mod_cast w.hBounds)).symm

/-- Aeneas' wrapping product, as a word that carries the product of what its operands carry. -/
theorem wrapping_mul_word {x y : Std.U64} {X Y : ℤ} (ex : (x.val : ℤ) = X % 2^64)
    (ey : (y.val : ℤ) = Y % 2^64) :
    ((core.num.U64.wrapping_mul x y).val : ℤ) = (X * Y) % 2^64 := by
  rw [core.num.U64.wrapping_mul_val_eq]
  simp only [UScalar.size, UScalarTy.U64_numBits_eq]
  exact Inversion.mul_word x.val y.val X Y ex ey

/-- Aeneas' wrapping sum, as a word that carries the sum of what its operands carry. -/
theorem wrapping_add_word {x y : Std.U64} {X Y : ℤ} (ex : (x.val : ℤ) = X % 2^64)
    (ey : (y.val : ℤ) = Y % 2^64) :
    ((core.num.U64.wrapping_add x y).val : ℤ) = (X + Y) % 2^64 := by
  rw [wrapping_add_val_addw]
  exact Inversion.addw_word x.val y.val X Y ex ey

/-- The word `a b + c d`, with wrapping products and sum, carries `A B + C D` when `a`, `b`, `c`,
and `d` carry `A`, `B`, `C`, and `D`. -/
theorem dot_word {a b c d : Std.U64} {A B C D : ℤ} (ea : (a.val : ℤ) = A % 2^64)
    (eb : (b.val : ℤ) = B % 2^64) (ec : (c.val : ℤ) = C % 2^64)
    (ed : (d.val : ℤ) = D % 2^64) :
    ((core.num.U64.wrapping_add (core.num.U64.wrapping_mul a b)
      (core.num.U64.wrapping_mul c d)).val : ℤ) = (A * B + C * D) % 2^64 :=
  wrapping_add_word (wrapping_mul_word ea eb) (wrapping_mul_word ec ed)

open pasta_curves.inversion.portable in
/-- `next_low` is the model's `nextLow`, on words that carry the row `(A, B)`. -/
@[step]
theorem next_low_spec (k : Std.U32) (hk : k.val < 64) (a b f g : Std.U64) (A B : ℤ)
    (ea : (a.val : ℤ) = A % 2^64) (eb : (b.val : ℤ) = B % 2^64) :
    next_low k a b f g ⦃ (res : Std.U64) =>
      (res.val : ℤ) = Inversion.nextLow k.val A B f.val g.val ⦄ := by
  unfold next_low
  step*
  have hi2 := dot_word ea (word_emod f) eb (word_emod g)
  rw [← i_post, ← i1_post, ← i2_post] at hi2
  rw [res_post, Inversion.nextLow, ← hi2, Nat.shiftRight_eq_div_pow]
  push_cast
  rfl

/-- Four words carry a matrix's entries `u`, `v`, `q`, and `r`, modulo `2^64`. -/
def CarriesMat (m : Std.Array Std.U64 4#usize) (M : Inversion.Mat2) : Prop :=
  (m[0].val : ℤ) = M.u % 2^64 ∧ (m[1].val : ℤ) = M.v % 2^64 ∧
    (m[2].val : ℤ) = M.q % 2^64 ∧ (m[3].val : ℤ) = M.r % 2^64

open pasta_curves.inversion.portable in
/-- `mat_mul` is the model's product of matrices, on words that carry their entries. -/
@[step]
theorem mat_mul_spec (m n : Std.Array Std.U64 4#usize) (M N : Inversion.Mat2)
    (hm : CarriesMat m M) (hn : CarriesMat n N) :
    mat_mul m n ⦃ (res : Std.Array Std.U64 4#usize) => CarriesMat res (M.mul N) ⦄ := by
  unfold mat_mul
  step*
  obtain ⟨hm0, hm1, hm2, hm3⟩ := hm
  obtain ⟨hn0, hn1, hn2, hn3⟩ := hn
  subst_vars
  simp only [CarriesMat, Std.Array.make, Inversion.Mat2.mul]
  exact ⟨dot_word hm0 hn0 hm1 hn2, dot_word hm0 hn1 hm1 hn3,
    dot_word hm2 hn0 hm3 hn2, dot_word hm2 hn1 hm3 hn3⟩

open pasta_curves.inversion.portable in
/-- One batch on words: from `two_delta` and the low words `f` and `g`, with `f` odd, the words of
the model's `batch`, the new `two_delta` and the matrix of `k` steps. -/
@[step]
theorem batch_spec (k : Std.U32) (hk : k.val ≤ 20) (two_delta f g : Std.U64) (TD : ℤ)
    (ed : (two_delta.val : ℤ) = TD % 2^64) (hf : (f.val : ℤ) % 2 = 1) (hd : TD % 2 = 1)
    (hD : |TD| + 2 * k.val < 2^62) :
    batch k two_delta f g ⦃ (res : Std.Array Std.U64 5#usize) =>
      (res[0].val : ℤ) = (Inversion.batch k.val TD f.val g.val).1 % 2^64 ∧
        (res[1].val : ℤ) = (Inversion.batch k.val TD f.val g.val).2.1.u % 2^64 ∧
        (res[2].val : ℤ) = (Inversion.batch k.val TD f.val g.val).2.1.v % 2^64 ∧
        (res[3].val : ℤ) = (Inversion.batch k.val TD f.val g.val).2.1.q % 2^64 ∧
        (res[4].val : ℤ) = (Inversion.batch k.val TD f.val g.val).2.1.r % 2^64 ⦄ := by
  unfold batch
  -- The packed start: the low 20 bits of `f` and `g`, with the identity's rows above them.
  generalize hs₀ : (⟨TD, (f.val : ℤ) % 2^20, (g.val : ℤ) % 2^20⟩ : State) = s₀
  have hs₀f : s₀.f % 2 = 1 := by rw [← hs₀]; agrind
  have hs₀fb : 0 ≤ s₀.f ∧ s₀.f < 2^20 := by rw [← hs₀]; constructor <;> agrind
  have hs₀gb : 0 ≤ s₀.g ∧ s₀.g < 2^20 := by rw [← hs₀]; constructor <;> agrind
  step*
  -- The packed words carry the packed start.
  have hpf : (pf.val : ℤ) = (Inversion.packedStart s₀).f % 2^64 := by
    rw [pf_post, UScalar.val_or, i_post, UScalar.val_and, ← hs₀]
    exact Inversion.pack_f_word f.val
  have hpg : (pg.val : ℤ) = (Inversion.packedStart s₀).g % 2^64 := by
    rw [pg_post, UScalar.val_or, i1_post, UScalar.val_and, ← hs₀]
    exact Inversion.pack_g_word g.val
  have hpd : (two_delta.val : ℤ) = (Inversion.packedStart s₀).two_delta % 2^64 := by
    rw [← hs₀]; exact ed
  have hPf : (Inversion.packedStart s₀).f % 2 = 1 := by
    show (s₀.f - 2^41) % 2 = 1; agrind
  have hPd : (Inversion.packedStart s₀).two_delta % 2 = 1 := by rw [← hs₀]; exact hd
  -- `k` packed steps.
  step with batch_loop_spec (Inversion.packedStart s₀) hPf hPd
    as ⟨two_delta1, pf1, pg1, hd1, hf1, hg1⟩
  · simp only [Inversion.packedStart, ← hs₀]
    exact hD
  · show |s₀.g - 2^62| < 2^63
    rw [abs_lt]; constructor <;> agrind
  · intro j _
    exact Inversion.divsteps_packedStart_g_abs_lt j s₀ hs₀f hs₀fb hs₀gb
  · simpa [Inversion.divsteps] using hpd
  · simpa [Inversion.divsteps] using hpf
  · simpa [Inversion.divsteps] using hpg
  -- Lemma 6 writes the words after `k` steps in the true state and matrix of `s₀`.
  rw [Inversion.divsteps_packedStart k.val hk s₀ hs₀f] at hd1 hf1 hg1
  obtain ⟨hφ, hγ⟩ := Inversion.divsteps_abs_le k.val s₀ (2^20 - 1)
    (by rw [abs_le]; constructor <;> agrind) (by rw [abs_le]; constructor <;> agrind)
  obtain ⟨hU, hV, hQ, hR⟩ := Inversion.M_entry_range k.val s₀
  -- The decoder reads the matrix back out of the two words.
  step with unpack_spec k hk pf1 _ _ _ (lt_of_le_of_lt hφ (by norm_num)) hU hV hf1
    as ⟨u, v, hu, hv⟩
  step with unpack_spec k hk pg1 _ _ _ (lt_of_le_of_lt hγ (by norm_num)) hQ hR hg1
    as ⟨q, r, hq, hr⟩
  -- The model's batch runs on the same packed state and decodes the same matrix.
  have hP : (⟨TD, (Inversion.pack f.val g.val).1, (Inversion.pack f.val g.val).2⟩ : State) =
      Inversion.packedStart s₀ := by
    rw [← hs₀]; rfl
  have hm1 :
      (Inversion.batch k.val TD f.val g.val).1 = (Inversion.divsteps k.val s₀).two_delta := by
    simp only [Inversion.batch, Inversion.packedDivsteps, hP,
      Inversion.divsteps_packedStart k.val hk s₀ hs₀f]
  have hm2 : (Inversion.batch k.val TD f.val g.val).2.1 = Inversion.M k.val s₀ := by
    simp only [Inversion.batch, Inversion.unpackMat, Inversion.packedDivsteps, hP,
      Inversion.divsteps_packedStart k.val hk s₀ hs₀f,
      Inversion.unpack_spec k.val hk _ _ _ (lt_of_le_of_lt hφ (by norm_num)) hU.1 hU.2,
      Inversion.unpack_spec k.val hk _ _ _ (lt_of_le_of_lt hγ (by norm_num)) hQ.1 hQ.2]
  simp only [hm1, hm2, Std.Array.make]
  exact ⟨hd1, hu, hv, hq, hr⟩

open pasta_curves.inversion.portable in
/-- `divstep59` satisfies its contract, the `divstep59` field of `InvertBlocks.Spec`: on words that
carry a true state `s` with `f` and `two_delta` odd and `two_delta` small, it returns words that
carry `two_delta` and the matrix after 59 steps. The words agree with the model's `divstep59`, and
Corollary 8 (`Inversion.divstep59_spec`) does the rest. -/
theorem divstep59_spec (two_delta f0 g0 : Std.U64) (s : State) (hf : s.f % 2 = 1)
    (hd : s.two_delta % 2 = 1) (hD : |s.two_delta| < 2^61)
    (ed : (two_delta.val : ℤ) = s.two_delta % 2^64) (ef : (f0.val : ℤ) = s.f % 2^64)
    (eg : (g0.val : ℤ) = s.g % 2^64) :
    Backend.Insts.Pasta_curvesInversionInvertBlocks.divstep59 two_delta f0 g0
    ⦃ (res : Std.Array Std.U64 5#usize) =>
      (res[0].val : ℤ) = (Inversion.divsteps 59 s).two_delta % 2^64 ∧
        (res[1].val : ℤ) = (Inversion.M 59 s).u % 2^64 ∧
        (res[2].val : ℤ) = (Inversion.M 59 s).v % 2^64 ∧
        (res[3].val : ℤ) = (Inversion.M 59 s).q % 2^64 ∧
        (res[4].val : ℤ) = (Inversion.M 59 s).r % 2^64 ⦄ := by
  unfold Backend.Insts.Pasta_curvesInversionInvertBlocks.divstep59
  have hfo : (f0.val : ℤ) % 2 = 1 := by rw [ef]; agrind
  -- Batch 1, on the true state `s`.
  step with batch_spec 20#u32 (by decide) two_delta f0 g0 s.two_delta ed hfo hd
    (by rw [show (20#u32 : Std.U32).val = 20 from rfl]; agrind) as ⟨a, ha0, ha1, ha2, ha3, ha4⟩
  step as ⟨two_delta1, e_td1⟩
  step as ⟨u1, e_u1⟩
  step as ⟨v1, e_v1⟩
  step as ⟨q1, e_q1⟩
  step as ⟨r1, e_r1⟩
  step with next_low_spec 20#u32 (by decide) u1 v1 f0 g0 _ _ (by rw [e_u1]; exact ha1)
    (by rw [e_v1]; exact ha2) as ⟨f1, hf1⟩
  step with next_low_spec 20#u32 (by decide) q1 r1 f0 g0 _ _ (by rw [e_q1]; exact ha3)
    (by rw [e_r1]; exact ha4) as ⟨g1, hg1⟩
  have h20 : (20#u32 : Std.U32).val = 20 := rfl
  have h19 : (19#u32 : Std.U32).val = 19 := rfl
  -- The model's batch 1 on `s`: the true state after 20 steps, and next low words that carry it
  -- modulo `2^44`.
  have hf0' : ((f0.val : ℕ) : ℤ) % 2^64 = s.f % 2^64 := by rw [ef, Int.emod_emod]
  have hg0' : ((g0.val : ℕ) : ℤ) % 2^64 = s.g % 2^64 := by rw [eg, Int.emod_emod]
  obtain ⟨d1, -, F1, G1⟩ :=
    Inversion.batch_spec 20 64 (le_refl _) (by norm_num) (le_refl _) s hf _ _ hf0' hg0'
  have hs1f := Inversion.divsteps_f_odd 20 s hf
  have hs1d : (Inversion.divsteps 20 s).two_delta % 2 = 1 := by
    rw [Inversion.divsteps_two_delta_emod_two]; exact hd
  have hs1D := Inversion.divsteps_two_delta_abs_le 20 s
  generalize hbatch1 : Inversion.batch 20 s.two_delta f0.val g0.val = b1 at *
  -- Batch 2, on the next low words, which are the model's.
  have ef1 : (f1.val : ℤ) = b1.2.2.1 := by rw [hf1, ← hbatch1]; rfl
  have eg1 : (g1.val : ℤ) = b1.2.2.2 := by rw [hg1, ← hbatch1]; rfl
  step with batch_spec 20#u32 (by decide) two_delta1 f1 g1 b1.1 (by rw [e_td1]; exact ha0)
    (by rw [ef1]; agrind) (by rw [d1]; exact hs1d) (by rw [h20, d1]; agrind)
    as ⟨a1, hb0, hb1', hb2, hb3, hb4⟩
  rw [ef1, eg1] at hb0 hb1' hb2 hb3 hb4
  step as ⟨two_delta2, e_td2⟩
  step as ⟨u2, e_u2⟩
  step as ⟨v2, e_v2⟩
  step as ⟨q2, e_q2⟩
  step as ⟨r2, e_r2⟩
  step with next_low_spec 20#u32 (by decide) u2 v2 f1 g1 _ _ (by rw [e_u2]; exact hb1')
    (by rw [e_v2]; exact hb2) as ⟨f2, hf2⟩
  step with next_low_spec 20#u32 (by decide) q2 r2 f1 g1 _ _ (by rw [e_q2]; exact hb3)
    (by rw [e_r2]; exact hb4) as ⟨g2, hg2⟩
  rw [ef1, eg1] at hf2 hg2
  -- The model's batch 2 on the state after 20 steps.
  obtain ⟨d2, -, F2, G2⟩ := Inversion.batch_spec 20 44 (le_refl _) (by norm_num) (by norm_num)
    (Inversion.divsteps 20 s) hs1f _ _ F1 G1
  rw [← d1] at d2 F2 G2
  have hs2f := Inversion.divsteps_f_odd 20 _ hs1f
  have hs2d : (Inversion.divsteps 20 (Inversion.divsteps 20 s)).two_delta % 2 = 1 := by
    rw [Inversion.divsteps_two_delta_emod_two]; exact hs1d
  have hs2D := Inversion.divsteps_two_delta_abs_le 20 (Inversion.divsteps 20 s)
  generalize hbatch2 : Inversion.batch 20 b1.1 b1.2.2.1 b1.2.2.2 = b2 at *
  -- Batch 3, of 19 steps.
  have ef2 : (f2.val : ℤ) = b2.2.2.1 := by rw [hf2, ← hbatch2]; rfl
  have eg2 : (g2.val : ℤ) = b2.2.2.2 := by rw [hg2, ← hbatch2]; rfl
  step with batch_spec 19#u32 (by decide) two_delta2 f2 g2 b2.1 (by rw [e_td2]; exact hb0)
    (by rw [ef2]; agrind) (by rw [d2]; exact hs2d) (by rw [h19, d2]; agrind)
    as ⟨a2, hc0, hc1, hc2, hc3, hc4⟩
  rw [ef2, eg2] at hc0 hc1 hc2 hc3 hc4
  generalize hbatch3 : Inversion.batch 19 b2.1 b2.2.2.1 b2.2.2.2 = b3 at *
  step as ⟨two_delta3, e_td3⟩
  step as ⟨u3, e_u3⟩
  step as ⟨v3, e_v3⟩
  step as ⟨q3, e_q3⟩
  step as ⟨r3, e_r3⟩
  -- The product of the three matrices, newest on the left.
  step with mat_mul_spec _ _ b2.2.1 b1.2.1 as ⟨a3, hm3⟩
  · simp only [CarriesMat, Std.Array.make]
    exact ⟨by rw [e_u2]; exact hb1', by rw [e_v2]; exact hb2, by rw [e_q2]; exact hb3,
      by rw [e_r2]; exact hb4⟩
  · simp only [CarriesMat, Std.Array.make]
    exact ⟨by rw [e_u1]; exact ha1, by rw [e_v1]; exact ha2, by rw [e_q1]; exact ha3,
      by rw [e_r1]; exact ha4⟩
  step with mat_mul_spec _ _ b3.2.1 (b2.2.1.mul b1.2.1) as ⟨a4, hm4⟩
  · simp only [CarriesMat, Std.Array.make]
    exact ⟨by rw [e_u3]; exact hc1, by rw [e_v3]; exact hc2, by rw [e_q3]; exact hc3,
      by rw [e_r3]; exact hc4⟩
  step*
  -- The words are the model's `divstep59`, which Corollary 8 computes.
  have hmodel :
      Inversion.divstep59 s.two_delta f0.val g0.val = (b3.1, b3.2.1.mul (b2.2.1.mul b1.2.1)) := by
    simp only [Inversion.divstep59, hbatch1, hbatch2, hbatch3]
  have hf0n : f0.val = (s.f % 2^64).toNat := by rw [← ef]; rfl
  have hg0n : g0.val = (s.g % 2^64).toNat := by rw [← eg]; rfl
  obtain ⟨hd59, hM59⟩ := Inversion.divstep59_spec s hf
  rw [← hf0n, ← hg0n, hmodel] at hd59 hM59
  simp only at hd59 hM59
  obtain ⟨hm4u, hm4v, hm4q, hm4r⟩ := hm4
  rw [hM59] at hm4u hm4v hm4q hm4r
  simp only [Std.Array.make]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [← hd59, e_td3]; exact hc0
  · rw [u_post]; exact hm4u
  · rw [v_post]; exact hm4v
  · rw [q_post]; exact hm4q
  · rw [r_post]; exact hm4r

end PastaCurves.Portable
