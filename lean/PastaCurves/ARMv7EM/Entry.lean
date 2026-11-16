import PastaCurves.Fields
import PastaCurves.Inversion.Model
import PastaCurves.Glue.Spec
import PastaCurves.ARMv7EM.Backend
import PastaCurves.ARMv7EM.Spec.Invert
import PastaCurves.ARMv7EM.Compositions.Mul.Entry
import PastaCurves.ARMv7EM.Compositions.Mul.SquareConversion

/-!
# The ARMv7E-M backend at the crate's fields

The records in `Backend.lean` contain the methods that Aeneas generated from the ARMv7E-M Rust
backend. The entry theorems therefore describe the translated Rust execution, rather than a
separately written pure ARM model. The extra low-word hypothesis is required by the generated
multiplication proof and holds for both Pasta moduli.

The generated Montgomery conversion and inversion methods are connected to their checked
instruction proofs; the field-level theorems cover the translated Rust methods.
-/

namespace PastaCurves.ARMv7EM

open _root_.Aeneas _root_.Aeneas.Std pasta_curves Glue

private theorem divstep59_success (twoDelta f0 g0 : Std.U64) :
    ∃ a, inversionBlocks.divstep59 twoDelta f0 g0 = .ok a := by
  obtain ⟨a, ha, -⟩ := WP.spec_imp_exists
    (Compositions.Invert.Divstep59.divstep59_eq_divstep59Block twoDelta f0 g0)
  exact ⟨a, ha⟩

private theorem signMag_success (u v q r : Std.U64) :
    ∃ a, inversionBlocks.sign_mag u v q r = .ok a :=
  Portable.ok_of_spec (Portable.sign_mag_ok u v q r)

private theorem fgRow_success (f g : Std.Array Std.U64 5#usize) (m0 m1 s0 s1 : Std.U64) :
    ∃ a, inversionBlocks.fg_row f g m0 m1 s0 s1 = .ok a :=
  Portable.ok_of_spec (Portable.fg_row_ok f g m0 m1 s0 s1)

private theorem deRow_success (d e : Std.Array Std.U64 4#usize) (m0 m1 s0 s1 : Std.U64) :
    ∃ a, inversionBlocks.de_row d e m0 m1 s0 s1 = .ok a :=
  Portable.ok_of_spec (Portable.de_row_ok d e m0 m1 s0 s1)

private theorem amontred_success (F : PastaField) (t : Std.Array Std.U64 5#usize) :
    ∃ a, inversionBlocks.amontred t (limbsArray F.modulus) (word F.inv) = .ok a :=
  Portable.ok_of_spec (ARMv7EM.amontred_ok F t)

private theorem condSub_success (value modulus : Std.Array Std.U64 4#usize) :
    ∃ a, inversionBlocks.cond_sub value modulus = .ok a := by
  rw [inversionBlocks]
  exact ⟨_, Compositions.Invert.ConditionalSub.generated_cond_sub_eq_pure_of_array value modulus⟩

@[step]
private theorem entry_divstep59_spec (twoDelta f0 g0 : Std.U64) :
    inversionBlocks.divstep59 twoDelta f0 g0 ⦃ a =>
      generatedInvertBlocks.divstep59 twoDelta.val f0.val g0.val =
        Portable.divstep59OfArray a ⦄ := by
  obtain ⟨a, ha⟩ := divstep59_success twoDelta f0 g0
  simp [ha, generatedInvertBlocks, Portable.pureBlocks, Option.ofResult, word_val_self]

@[step]
private theorem entry_signMag_spec (u v q r : Std.U64) :
    inversionBlocks.sign_mag u v q r ⦃ a =>
      generatedInvertBlocks.signMag u.val v.val q.val r.val = Portable.signMagOfArray a ⦄ := by
  obtain ⟨a, ha⟩ := signMag_success u v q r
  simp [ha, generatedInvertBlocks, Portable.pureBlocks, Option.ofResult, word_val_self]

@[step]
private theorem entry_fgRow_spec (f g : Std.Array Std.U64 5#usize) (m0 m1 s0 s1 : Std.U64) :
    inversionBlocks.fg_row f g m0 m1 s0 s1 ⦃ a =>
      generatedInvertBlocks.fgRow (signed5OfArray f) (signed5OfArray g)
        m0.val m1.val s0.val s1.val = signed5OfArray a ⦄ := by
  obtain ⟨a, ha⟩ := fgRow_success f g m0 m1 s0 s1
  simp [ha, generatedInvertBlocks, Portable.pureBlocks, Option.ofResult, word_val_self,
    signed5Array_signed5OfArray]

@[step]
private theorem entry_deRow_spec (d e : Std.Array Std.U64 4#usize) (m0 m1 s0 s1 : Std.U64) :
    inversionBlocks.de_row d e m0 m1 s0 s1 ⦃ a =>
      generatedInvertBlocks.deRow (limbsOfArray d) (limbsOfArray e)
        m0.val m1.val s0.val s1.val = signed5OfArray a ⦄ := by
  obtain ⟨a, ha⟩ := deRow_success d e m0 m1 s0 s1
  simp [ha, generatedInvertBlocks, Portable.pureBlocks, Option.ofResult, word_val_self,
    limbsArray_limbsOfArray]

@[step]
private theorem entry_amontred_spec (F : PastaField) (t : Std.Array Std.U64 5#usize) :
    inversionBlocks.amontred t (limbsArray F.modulus) (word F.inv) ⦃ a =>
      generatedInvertBlocks.amontred (signed5OfArray t) F.modulus F.inv = limbsOfArray a ⦄ := by
  obtain ⟨a, ha⟩ := amontred_success F t
  simp [ha, generatedInvertBlocks, Portable.pureBlocks, Option.ofResult,
    signed5Array_signed5OfArray]

@[step]
private theorem entry_condSub_spec (value modulus : Std.Array Std.U64 4#usize) :
    inversionBlocks.cond_sub value modulus ⦃ a =>
      generatedInvertBlocks.condSub (limbsOfArray value) (limbsOfArray modulus) =
        limbsOfArray a ⦄ := by
  obtain ⟨a, ha⟩ := condSub_success value modulus
  simp [ha, generatedInvertBlocks, Portable.pureBlocks, Option.ofResult,
    limbsArray_limbsOfArray]

attribute [local grind =] Portable.divstep59OfArray Portable.signMagOfArray signed5OfArray

private theorem invert_with_loop_spec (F : PastaField) (x : Limbs)
    (iter : core.ops.range.Range Std.I32) (twoDelta : Std.U64)
    (f g : Std.Array Std.U64 5#usize) (d e : Std.Array Std.U64 4#usize)
    (hend : iter.«end».val = 9) (h0 : 0 ≤ iter.start.val) (h9 : iter.start.val ≤ 9)
    (hst : Portable.stateOf twoDelta f g d e =
      (invertRound generatedInvertBlocks F.modulus F.inv)^[iter.start.val.toNat]
        (Portable.startState F x)) :
    inversion.invert_with_loop inversionBlocks iter (limbsArray F.modulus) (word F.inv)
      twoDelta f g d e
    ⦃ (twoDelta' : Std.U64) (f' g' : Std.Array Std.U64 5#usize)
      (d' e' : Std.Array Std.U64 4#usize) =>
      Portable.stateOf twoDelta' f' g' d' e' =
        (invertRound generatedInvertBlocks F.modulus F.inv)^[9]
          (Portable.startState F x) ⦄ := by
  unfold inversion.invert_with_loop
  apply loop.spec_decr_nat (fun (iter, _) => (iter.«end».val - iter.start.val).toNat)
    (fun (iter, twoDelta, f, g, d, e) =>
      iter.«end».val = 9 ∧ 0 ≤ iter.start.val ∧ iter.start.val ≤ 9 ∧
        Portable.stateOf twoDelta f g d e =
          (invertRound generatedInvertBlocks F.modulus F.inv)^[iter.start.val.toNat]
            (Portable.startState F x))
  · clear hend h0 h9 hst
    rintro ⟨iter, twoDelta, f, g, d, e⟩ ⟨hend, h0, h9, hst⟩
    unfold inversion.invert_with_loop.body
    step*
    · agrind
    · split_conjs
      · agrind
      · agrind
      · agrind
      · have hsucc : iter1.start.val.toNat = iter.start.val.toNat + 1 := by agrind
        have hm : limbsOfArray (limbsArray F.modulus) = F.modulus :=
          limbsOfArray_limbsArray F.bounded
        have hinv : (word F.inv).val = F.inv := word_val F.inv_lt
        rw [hsucc, Function.iterate_succ_apply', ← hst]
        grind [invertRound, Portable.stateOf]
      · agrind
  · exact ⟨hend, h0, h9, hst⟩

private theorem invert_with_spec (F : PastaField) (x : Limbs) (hx : x.Bounded) :
    inversion.invert_with inversionBlocks (limbsArray x) (limbsArray F.modulus) (word F.inv)
      (limbsArray (Inversion.startE F))
    ⦃ (res : Std.Array Std.U64 4#usize) =>
      limbsOfArray res = invert generatedInvertBlocks x F.modulus F.inv (Inversion.startE F) ⦄ := by
  unfold inversion.invert_with
  step*
  step with invert_with_loop_spec F x as ⟨twoDelta, f, g, d, e, hloop⟩
  · obtain ⟨hm0, hm1, hm2, hm3⟩ := F.bounded
    obtain ⟨hx0, hx1, hx2, hx3⟩ := hx
    obtain ⟨he0, he1, he2, he3⟩ : (Inversion.startE F).Bounded := Limbs.ofNat_bounded _
    simp [Portable.startState, Portable.stateOf, signed5OfArray, limbsOfArray, limbsArray,
      word_val, *]
  step*
  have hm : limbsOfArray (limbsArray F.modulus) = F.modulus :=
    limbsOfArray_limbsArray F.bounded
  have hinv : (word F.inv).val = F.inv := word_val F.inv_lt
  have hsign : sign.val = signWord i8.val i9.val u.val v.val := by
    subst i13
    rw [sign_post, sign_mask_val i14_post]
    simp [signWord, addw, mulLo, regMod, U64.size, U64.numBits, i12_post, i10_post, i11_post]
  simp only [Portable.startState] at hloop
  simp only [invert]
  rw [← hloop]
  grind [Portable.stateOf]

/-- The translated ARM inversion entry follows from all-input success of the actual generated ARM
method record. Its arithmetic block specification is proved independently. -/
theorem invert_entry_spec_of_blocks (hS : Portable.Succeeds inversionBlocks)
    (F : PastaField) (x : Limbs) (hx : x.Bounded) (h : isCanonical x F.modulus = true) :
    inversion.invert_with inversionBlocks (limbsArray x) (limbsArray F.modulus) (word F.inv)
      (limbsArray (Inversion.startE F))
    ⦃ (res : Std.Array Std.U64 4#usize) =>
      (limbsOfArray res).toNat < F.modulus.toNat ∧
        (x.toNat = 0 → limbsOfArray res = Limbs.ofNat 0) ∧
        (x.toNat ≠ 0 →
          x.toNat * (limbsOfArray res).toNat ≡ R^2 [MOD F.modulus.toNat]) ⦄ := by
  exact Portable.invert_with_entry_spec hS F
    (by simpa [generatedInvertBlocks] using generatedInvertBlocks_spec F) x hx h

/-- The crate's `invert` at a Pasta field on the generated ARMv7E-M inversion methods. The input is
canonical, as the entry point asserts. The result is canonical; for `x = 0` it is `0`, and otherwise
it is the Montgomery inverse with `x * result ≡ R² (mod p)`. -/
theorem invert_entry_spec (F : PastaField) (x : Limbs) (hx : x.Bounded)
    (h : isCanonical x F.modulus = true) :
    inversion.invert_with inversionBlocks (limbsArray x) (limbsArray F.modulus) (word F.inv)
      (limbsArray (Inversion.startE F))
    ⦃ (res : Std.Array Std.U64 4#usize) =>
      (limbsOfArray res).toNat < F.modulus.toNat ∧
        (x.toNat = 0 → limbsOfArray res = Limbs.ofNat 0) ∧
        (x.toNat ≠ 0 →
          x.toNat * (limbsOfArray res).toNat ≡ R^2 [MOD F.modulus.toNat]) ⦄ := by
  apply WP.spec_mono (invert_with_spec F x hx)
  intro res hres
  obtain ⟨-, hlt, hzero, hinverse⟩ := PastaCurves.invert_entry_spec generatedInvertBlocks F
    (generatedInvertBlocks_spec F) x hx h _ rfl
  rw [hres]
  exact ⟨hlt, hzero, hinverse⟩

/-- The generated ARMv7E-M multiplication, squaring, and conversion methods meet their entry-point
contracts at a Pasta field whose modulus has low 32-bit word one. The generated addition and
subtraction methods panic and therefore cannot satisfy the full Montgomery-block specification. -/
theorem montgomeryBlocks_spec (F : PastaField) (hlow : F.modulus.l0 % wordMod = 1) :
    Glue.MulBlocksSpec montgomeryBlocks F where
  mul lhs rhs h := by
    change pasta_curves.asm.armv7em.mul lhs rhs (limbsArray F.modulus) (word F.inv) ⦃ result =>
      Canonical F result R ((limbsOfArray lhs).toNat * (limbsOfArray rhs).toNat) ⦄
    exact Compositions.Mul.Entry.mul_entry_spec F hlow lhs rhs h
  square value h := by
    change pasta_curves.asm.armv7em.Backend.Insts.Pasta_curvesAsmEntryMontgomeryBlocks.square
      value (limbsArray F.modulus) (word F.inv) ⦃ result =>
        Canonical F result R ((limbsOfArray value).toNat * (limbsOfArray value).toNat) ⦄
    exact Compositions.Mul.SquareConversion.square_entry_spec F hlow value h
  from_mont value := by
    exact Compositions.Mul.SquareConversion.from_mont_entry_spec F hlow value

section EntryPoints

/-- The crate's `mul` on the generated ARMv7E-M backend, under the operand contract asserted by
Rust. -/
theorem mul_entry_spec (F : PastaField) (hlow : F.modulus.l0 % wordMod = 1)
    (lhs rhs : Std.Array Std.U64 4#usize)
    (h : mulContract (limbsOfArray lhs) (limbsOfArray rhs) F.modulus = true) :
    asm.entry.mul_with montgomeryBlocks lhs rhs (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res R ((limbsOfArray lhs).toNat * (limbsOfArray rhs).toNat) ⦄ :=
  mul_with_spec montgomeryBlocks F (montgomeryBlocks_spec F hlow) lhs rhs h

/-- The crate's `square` on the generated ARMv7E-M backend, for a canonical operand. -/
theorem square_entry_spec (F : PastaField) (hlow : F.modulus.l0 % wordMod = 1)
    (value : Std.Array Std.U64 4#usize)
    (h : isCanonical (limbsOfArray value) F.modulus = true) :
    asm.entry.square_with montgomeryBlocks value (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res R ((limbsOfArray value).toNat * (limbsOfArray value).toNat) ⦄ :=
  square_with_spec montgomeryBlocks F (montgomeryBlocks_spec F hlow) value h

/-- The crate's `sqr_n_mul` on the generated ARMv7E-M backend, for a canonical initial value. -/
theorem sqr_n_mul_entry_spec (F : PastaField) (hlow : F.modulus.l0 % wordMod = 1)
    (value : Std.Array Std.U64 4#usize) (count : Std.Usize)
    (rhs : Std.Array Std.U64 4#usize)
    (h : isCanonical (limbsOfArray value) F.modulus = true) :
    asm.entry.sqr_n_mul_with montgomeryBlocks value count rhs (limbsArray F.modulus)
      (word F.inv) ⦃ res =>
      Canonical F res (R^(2^count.val))
        ((limbsOfArray value).toNat^(2^count.val) * (limbsOfArray rhs).toNat) ⦄ :=
  sqr_n_mul_with_spec montgomeryBlocks F (montgomeryBlocks_spec F hlow) value count rhs h

/-- The crate's `from_mont` on the generated ARMv7E-M backend, for any four-limb operand. -/
theorem from_mont_entry_spec (F : PastaField) (hlow : F.modulus.l0 % wordMod = 1)
    (value : Std.Array Std.U64 4#usize) :
    asm.entry.from_mont_with montgomeryBlocks value (limbsArray F.modulus) (word F.inv) ⦃ res =>
      Canonical F res R (limbsOfArray value).toNat ⦄ :=
  from_mont_with_spec montgomeryBlocks F (montgomeryBlocks_spec F hlow) value

end EntryPoints

end PastaCurves.ARMv7EM
