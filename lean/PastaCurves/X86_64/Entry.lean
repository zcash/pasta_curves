import PastaCurves.Fields
import PastaCurves.Glue.Spec
import PastaCurves.Spec
import PastaCurves.X86_64.Backend
import PastaCurves.X86_64.Spec

/-!
# The x86-64 backend at the crate's fields

`PastaCurves.X86_64.Spec` proves the blocks for any modulus of the assumed shape, under arithmetic
conditions on the operands. `montgomeryBlocks_spec` instantiates them at either of the crate's
fields (a `PastaField`, whose facts discharge the hypotheses on the modulus). It shows that the
backend's record of Montgomery blocks (`Backend.lean`) meets the contracts that `Glue.BlocksSpec`
states, under the conditions that the entry points assert in a debug build. The theorems of
`Glue/Spec.lean` carry those contracts through Aeneas' translation of the entry points'
compositions, so they hold for the crate's `add`, `sub`, `mul`, `square`, `sqr_n_mul`, and
`from_mont` on this backend.
-/

namespace PastaCurves.X86_64

/-- The x86-64 record meets the blocks' contracts at a Pasta field, by the blocks' theorems: `mul`
under either of its proved contracts, and the backend's own `from_mont` for every operand. -/
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

end PastaCurves.X86_64
