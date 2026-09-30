import PastaAsm.Fields
import PastaAsm.Spec
import PastaAsm.AArch64.Spec

/-!
# The crate's entry points at its fields

`PastaAsm.AArch64.Spec` proves the blocks and the compositions for any modulus of the assumed shape,
under arithmetic conditions on the operands. The theorems here restate them for the crate's four
entry points as `src/asm/mod.rs` exposes them: at either of its fields (a `PastaField`, whose facts
discharge the hypotheses on the modulus), and under the condition that the entry point checks
in a debug build (`mulContract` for `mul`, `isCanonical` for `square` and for the squarings of
`sqr_n_mul`). The conversion out of Montgomery form checks nothing and holds for every input.
The results are stated against the Montgomery radix `R = 2^256` of `Fields.lean`.

These theorems are intentionally identical to those in `PastaAsm.X86_64.Entry` (other than
calling the AArch64 assembly transcription), because all architecture-specific assembly is
exposed through the same crate API. This ensures that the architecture-specific proofs apply
to the architecture-agnostic interface.
-/

namespace PastaAsm.AArch64

/-- The crate's `mul` at a Pasta field: when the condition it asserts holds, the result is
canonical, with `R * result ≡ lhs * rhs (mod p)`. -/
theorem mul_entry_spec (F : PastaField) (lhs rhs : Limbs) (hlhs : lhs.Bounded)
    (hrhs : rhs.Bounded) (h : mulContract lhs rhs F.modulus = true) :
    (mulMont lhs rhs F.modulus F.inv).Bounded ∧
      (mulMont lhs rhs F.modulus F.inv).toNat < F.modulus.toNat ∧
      R * (mulMont lhs rhs F.modulus F.inv).toNat ≡ lhs.toNat * rhs.toNat
        [MOD F.modulus.toNat] := by
  rcases (mulContract_iff lhs rhs F.modulus hlhs hrhs F.bounded).1 h with hlt | ⟨hlt, hlimbs⟩
  · exact mulMont_spec_of_lhs_lt lhs rhs F.modulus F.inv hlhs hrhs F.bounded F.shape F.inv_lt
      F.inv_spec hlt _ rfl
  · exact mulMont_spec_of_rhs_lt lhs rhs F.modulus F.inv hlhs hrhs F.bounded F.shape F.inv_lt
      F.inv_spec hlt hlimbs _ rfl

/-- The crate's `square` at a Pasta field: for a canonical input, as it asserts, the result is
canonical, with `R * result ≡ value² (mod p)`. -/
theorem square_entry_spec (F : PastaField) (value : Limbs) (hv : value.Bounded)
    (h : isCanonical value F.modulus = true) :
    (sqrMont value F.modulus F.inv).Bounded ∧
      (sqrMont value F.modulus F.inv).toNat < F.modulus.toNat ∧
      R * (sqrMont value F.modulus F.inv).toNat ≡ value.toNat * value.toNat
        [MOD F.modulus.toNat] :=
  sqrMont_spec value F.modulus F.inv hv F.bounded F.shape F.inv_lt F.inv_spec
    ((isCanonical_iff value F.modulus hv F.bounded).1 h) _ rfl

/-- The crate's `sqr_n_mul` at a Pasta field: for a canonical `value`, which its squarings
assert, and any four-limb `rhs`, the result is canonical with
`R^(2^count) * result ≡ value^(2^count) * rhs (mod p)`. -/
theorem sqrNMul_entry_spec (F : PastaField) (value : Limbs) (count : Nat) (rhs : Limbs)
    (hv : value.Bounded) (hrhs : rhs.Bounded) (h : isCanonical value F.modulus = true) :
    (sqrNMul value count rhs F.modulus F.inv).Bounded ∧
      (sqrNMul value count rhs F.modulus F.inv).toNat < F.modulus.toNat ∧
      R^(2^count) * (sqrNMul value count rhs F.modulus F.inv).toNat ≡
        value.toNat^(2^count) * rhs.toNat [MOD F.modulus.toNat] := by
  have hspec := sqrNMul_spec value count rhs F.modulus F.inv hv hrhs F.bounded F.shape F.inv_lt
    F.inv_spec ((isCanonical_iff value F.modulus hv F.bounded).1 h) _ rfl
  rwa [Nat.pow_mul] at hspec

/-- The crate's `from_mont` at a Pasta field: for every four-limb `value`, the result is
canonical with `R * result ≡ value (mod p)`. -/
theorem fromMont_entry_spec (F : PastaField) (value : Limbs) (hv : value.Bounded) :
    (fromMont value F.modulus F.inv).Bounded ∧
      (fromMont value F.modulus F.inv).toNat < F.modulus.toNat ∧
      R * (fromMont value F.modulus F.inv).toNat ≡ value.toNat [MOD F.modulus.toNat] :=
  fromMont_spec value F.modulus F.inv hv F.bounded F.shape F.inv_lt F.inv_spec _ rfl

/-- The crate's `add` at a Pasta field: for canonical operands, as it asserts, the result is the
canonical sum. -/
theorem add_entry_spec (F : PastaField) (lhs rhs : Limbs) (hlhs : lhs.Bounded)
    (hrhs : rhs.Bounded) (hl : isCanonical lhs F.modulus = true)
    (hr : isCanonical rhs F.modulus = true) :
    (addMod lhs rhs F.modulus).Bounded ∧
      (addMod lhs rhs F.modulus).toNat < F.modulus.toNat ∧
      (addMod lhs rhs F.modulus).toNat ≡ lhs.toNat + rhs.toNat [MOD F.modulus.toNat] :=
  addMod_spec_of_lt lhs rhs F.modulus hlhs hrhs F.bounded F.shape
    ((isCanonical_iff lhs F.modulus hlhs F.bounded).1 hl)
    ((isCanonical_iff rhs F.modulus hrhs F.bounded).1 hr) _ rfl

/-- The crate's `sub` at a Pasta field: for canonical operands, as it asserts, the result is the
canonical difference. -/
theorem sub_entry_spec (F : PastaField) (lhs rhs : Limbs) (hlhs : lhs.Bounded)
    (hrhs : rhs.Bounded) (hl : isCanonical lhs F.modulus = true)
    (hr : isCanonical rhs F.modulus = true) :
    (subMod lhs rhs F.modulus).Bounded ∧
      (subMod lhs rhs F.modulus).toNat < F.modulus.toNat ∧
      (subMod lhs rhs F.modulus).toNat + rhs.toNat ≡ lhs.toNat [MOD F.modulus.toNat] :=
  subMod_spec_of_lt lhs rhs F.modulus hlhs hrhs F.bounded F.shape
    ((isCanonical_iff lhs F.modulus hlhs F.bounded).1 hl)
    ((isCanonical_iff rhs F.modulus hrhs F.bounded).1 hr) _ rfl

end PastaAsm.AArch64
