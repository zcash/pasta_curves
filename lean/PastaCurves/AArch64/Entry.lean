import PastaCurves.Fields
import PastaCurves.Glue.Spec
import PastaCurves.Spec
import PastaCurves.AArch64.Backend
import PastaCurves.AArch64.Spec

/-!
# The AArch64 backend at the crate's fields

`PastaCurves.AArch64.Spec` proves the blocks for any modulus of the assumed shape, under arithmetic
conditions on the operands. The theorems here instantiate them at either of the crate's fields (a
`PastaField`, whose facts discharge the hypotheses on the modulus).

`montgomeryBlocks_spec` shows that the backend's record of Montgomery blocks (`Backend.lean`) meets
the contracts that `Glue.BlocksSpec` states, under the conditions that the entry points assert in a
debug build. The theorems of `Glue/Spec.lean` carry those contracts through Aeneas' translation of
the entry points' compositions, so they hold for the crate's `add`, `sub`, `mul`, `square`,
`sqr_n_mul`, and `from_mont` on this backend. `invert_entry_spec` is the crate's `invert` on the
AArch64 inversion blocks. The results are stated against the Montgomery radix `R = 2^256` of
`Fields.lean`.
-/

namespace PastaCurves.AArch64

/-- The crate's `invert` at a Pasta field, on the AArch64 blocks: the shared `invert_entry_spec`
at `invertBlocks`. The input is canonical, as the entry point asserts, and `e0` is `2^562 mod p`,
as its contract requires. The result is canonical. For `x = 0` it is `0`; otherwise it is the
Montgomery inverse, with `x * result ≡ R^2 (mod p)`. -/
theorem invert_entry_spec (F : PastaField) (x : Limbs) (hx : x.Bounded)
    (h : isCanonical x F.modulus = true) (e0 : Limbs) (he0 : e0 = Inversion.startE F) :
    (invert invertBlocks x F.modulus F.inv e0).Bounded ∧
      (invert invertBlocks x F.modulus F.inv e0).toNat < F.modulus.toNat ∧
      (x.toNat = 0 → invert invertBlocks x F.modulus F.inv e0 = Limbs.ofNat 0) ∧
      (x.toNat ≠ 0 →
        x.toNat * (invert invertBlocks x F.modulus F.inv e0).toNat ≡ R^2 [MOD F.modulus.toNat]) :=
  PastaCurves.invert_entry_spec invertBlocks F (invertBlocks_spec F) x hx h e0 he0

/-- The AArch64 record meets the blocks' contracts at a Pasta field, by the blocks' theorems: `mul`
under either of its proved contracts, and `from_mont`, the multiplication by one, for every
operand. -/
theorem montgomeryBlocks_spec (F : PastaField) : Glue.BlocksSpec montgomeryBlocks F where
  add lhs rhs hl hr := by
    have hb := limbsOfArray_bounded
    obtain ⟨hr', hlt, hc⟩ := addMod_spec_of_lt _ _ F.modulus (hb lhs) (hb rhs) F.bounded F.shape
      ((isCanonical_iff _ _ (hb lhs) F.bounded).1 hl)
      ((isCanonical_iff _ _ (hb rhs) F.bounded).1 hr) _ rfl
    simp only [montgomeryBlocks, Aeneas.Std.WP.spec_ok, limbsOfArray_limbsArray _ F.bounded,
      limbsOfArray_limbsArray _ hr']
    exact ⟨hlt, hc⟩
  sub lhs rhs hl hr := by
    have hb := limbsOfArray_bounded
    obtain ⟨hr', hlt, hc⟩ := subMod_spec_of_lt _ _ F.modulus (hb lhs) (hb rhs) F.bounded F.shape
      ((isCanonical_iff _ _ (hb lhs) F.bounded).1 hl)
      ((isCanonical_iff _ _ (hb rhs) F.bounded).1 hr) _ rfl
    simp only [montgomeryBlocks, Aeneas.Std.WP.spec_ok, limbsOfArray_limbsArray _ F.bounded,
      limbsOfArray_limbsArray _ hr']
    exact ⟨hlt, hc⟩
  mul lhs rhs h := by
    have hb := limbsOfArray_bounded
    obtain ⟨hr', hlt, hc⟩ :
        (mulMont (limbsOfArray lhs) (limbsOfArray rhs) F.modulus F.inv).Bounded ∧
          (mulMont (limbsOfArray lhs) (limbsOfArray rhs) F.modulus F.inv).toNat <
            F.modulus.toNat ∧
          R * (mulMont (limbsOfArray lhs) (limbsOfArray rhs) F.modulus F.inv).toNat ≡
            (limbsOfArray lhs).toNat * (limbsOfArray rhs).toNat [MOD F.modulus.toNat] := by
      rcases (mulContract_iff _ _ F.modulus (hb lhs) (hb rhs) F.bounded).1 h with
        hlt | ⟨hlt, hlimbs⟩
      · exact mulMont_spec_of_lhs_lt _ _ F.modulus F.inv (hb lhs) (hb rhs) F.bounded F.shape
          F.inv_lt F.inv_spec hlt _ rfl
      · exact mulMont_spec_of_rhs_lt _ _ F.modulus F.inv (hb lhs) (hb rhs) F.bounded F.shape
          F.inv_lt F.inv_spec hlt hlimbs _ rfl
    simp only [montgomeryBlocks, Aeneas.Std.WP.spec_ok, limbsOfArray_limbsArray _ F.bounded,
      word_val _ F.inv_lt, limbsOfArray_limbsArray _ hr']
    exact ⟨hlt, hc⟩
  square value h := by
    have hb := limbsOfArray_bounded
    obtain ⟨hr', hlt, hc⟩ := sqrMont_spec _ F.modulus F.inv (hb value) F.bounded F.shape F.inv_lt
      F.inv_spec ((isCanonical_iff _ _ (hb value) F.bounded).1 h) _ rfl
    simp only [montgomeryBlocks, Aeneas.Std.WP.spec_ok, limbsOfArray_limbsArray _ F.bounded,
      word_val _ F.inv_lt, limbsOfArray_limbsArray _ hr']
    exact ⟨hlt, hc⟩
  from_mont value := by
    obtain ⟨hr', hlt, hc⟩ := fromMont_spec _ F.modulus F.inv (limbsOfArray_bounded value)
      F.bounded F.shape F.inv_lt F.inv_spec _ rfl
    simp only [montgomeryBlocks, Aeneas.Std.WP.spec_ok, limbsOfArray_limbsArray _ F.bounded,
      word_val _ F.inv_lt, limbsOfArray_limbsArray _ hr']
    exact ⟨hlt, hc⟩

end PastaCurves.AArch64
