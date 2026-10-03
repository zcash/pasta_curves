import PastaCurves.Portable.Words
import PastaCurves.Inversion.Vectors
import PastaCurves.Inversion.Model

/-!
# The translated portable blocks on the known answers

Aeneas' translation of the portable blocks (`Funs.lean`) is run on the six blocks' known answers
of `Inversion/Vectors.lean`, and its translation of the driver `invert_with`, over the portable
blocks, is run against the word-level model `montInvModel` on a few inputs at each field.

These are tests, not proofs. The translation's loops are Aeneas' `loop` combinator, a
`partial_fixpoint` that the kernel cannot unfold, so the checks run as compiled code with
`#guard`, which fails the build if a check is false; they add no declaration to the
development. They catch a mismatch between the translation and the values the crate's tests
record, which a proof about the translation would otherwise be the first to find.
-/

namespace PastaCurves.Portable

open Aeneas Aeneas.Std Inversion

/-- The words of a result that succeeds, as natural numbers, and `none` for one that fails,
diverges, or panics. -/
def resultWords {n} (r : Result (Std.Array Std.U64 n)) : Option (List Nat) :=
  (Option.ofResult r).map fun a => a.val.map (·.val)

#guard divstep59Vectors.all fun (d, f0, g0, r) =>
  resultWords (blocks.divstep59 (word d) (word f0) (word g0)) ==
    some [r.d, r.m00, r.m01, r.m10, r.m11]

#guard signMagVectors.all fun (m00, m01, m10, m11, r) =>
  resultWords (blocks.sign_mag (word m00) (word m01) (word m10) (word m11)) ==
    some [r.m00, r.m01, r.m10, r.m11, r.s00, r.s01, r.s10, r.s11]

#guard fgRowVectors.all fun (f, g, m0, m1, s0, s1, r) =>
  resultWords (blocks.fg_row (signed5Array f) (signed5Array g) (word m0) (word m1) (word s0)
    (word s1)) == some [r.l0, r.l1, r.l2, r.l3, r.l4]

#guard uvRowVectors.all fun (u, v, m0, m1, s0, s1, r) =>
  resultWords (blocks.uv_row (limbsArray u) (limbsArray v) (word m0) (word m1) (word s0)
    (word s1)) == some [r.l0, r.l1, r.l2, r.l3, r.l4]

#guard amontredVectors.all fun (t, F, r) =>
  resultWords (blocks.amontred (signed5Array t) (limbsArray F.modulus) (word F.inv)) ==
    some [r.l0, r.l1, r.l2, r.l3]

#guard condSubVectors.all fun (x, F, r) =>
  resultWords (blocks.cond_sub (limbsArray x) (limbsArray F.modulus)) ==
    some [r.l0, r.l1, r.l2, r.l3]

/-- Canonical inputs for the inversion at the field `F`: `0`, `1`, `7`, `p - 1`, `p / 3`, and
`2^200 + 1`. -/
def invertInputs (F : PastaField) : List Nat :=
  [0, 1, 7, F.modulus.toNat - 1, F.modulus.toNat / 3, 2^200 + 1]

#guard [pallasBase, vestaBase].all fun F => (invertInputs F).all fun x =>
  let r := montInvModel F (Limbs.ofNat x)
  resultWords (pasta_curves.inversion.invert_with blocks (limbsArray (Limbs.ofNat x))
    (limbsArray F.modulus) (word F.inv) (limbsArray (startV F))) == some [r.l0, r.l1, r.l2, r.l3]

end PastaCurves.Portable
