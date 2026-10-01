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
def resultWords {n} (res : Result (Std.Array Std.U64 n)) : Option (List Nat) :=
  (Option.ofResult res).map fun a => a.val.map (·.val)

#guard divstep59Vectors.all fun (two_delta, f0, g0, res) =>
  resultWords (blocks.divstep59 (word two_delta) (word f0) (word g0)) ==
    some [res.two_delta, res.u, res.v, res.q, res.r]

#guard signMagVectors.all fun (u, v, q, r, res) =>
  resultWords (blocks.sign_mag (word u) (word v) (word q) (word r)) ==
    some [res.u, res.v, res.q, res.r, res.su, res.sv, res.sq, res.sr]

#guard fgRowVectors.all fun (f, g, m0, m1, s0, s1, res) =>
  resultWords (blocks.fg_row (signed5Array f) (signed5Array g) (word m0) (word m1) (word s0)
    (word s1)) == some [res.l0, res.l1, res.l2, res.l3, res.l4]

#guard deRowVectors.all fun (d, e, m0, m1, s0, s1, res) =>
  resultWords (blocks.de_row (limbsArray d) (limbsArray e) (word m0) (word m1) (word s0)
    (word s1)) == some [res.l0, res.l1, res.l2, res.l3, res.l4]

#guard amontredVectors.all fun (t, F, res) =>
  resultWords (blocks.amontred (signed5Array t) (limbsArray F.modulus) (word F.inv)) ==
    some [res.l0, res.l1, res.l2, res.l3]

#guard condSubVectors.all fun (x, F, res) =>
  resultWords (blocks.cond_sub (limbsArray x) (limbsArray F.modulus)) ==
    some [res.l0, res.l1, res.l2, res.l3]

/-- Canonical inputs for the inversion at the field `F`: `0`, `1`, `7`, `p - 1`, `p / 3`, and
`2^200 + 1`. -/
def invertInputs (F : PastaField) : List Nat :=
  [0, 1, 7, F.modulus.toNat - 1, F.modulus.toNat / 3, 2^200 + 1]

#guard [pallasBase, vestaBase].all fun F => (invertInputs F).all fun x =>
  let res := montInvModel F (Limbs.ofNat x)
  resultWords (pasta_curves.inversion.invert_with blocks (limbsArray (Limbs.ofNat x))
    (limbsArray F.modulus) (word F.inv) (limbsArray (startE F))) == some [res.l0, res.l1, res.l2, res.l3]

end PastaCurves.Portable
