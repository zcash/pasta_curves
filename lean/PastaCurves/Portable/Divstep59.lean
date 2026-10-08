import PastaCurves.Portable.Spec
import PastaCurves.Inversion.PackedWords
import PastaCurves.Inversion.Divstep59

/-!
# The translated `divstep59`

`divstep59` of `src/inversion/portable.rs` runs the packed recurrence of `Inversion/Packed.lean` on
64-bit words, in three batches of 20, 20, and 19 steps, as `Inversion.divstep59` does on integers.
Each packed step is the plain `divstep` on the packed state. `divstep_spec` relates one step on
words to it, through the shared case lemmas of `Inversion/PackedWords.lean`.
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

end PastaCurves.Portable
