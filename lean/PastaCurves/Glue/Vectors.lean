import PastaCurves.AArch64.Backend
import PastaCurves.X86_64.Backend
import PastaCurves.Vectors

/-!
# The translated glue on the vectors and known answers

Aeneas' translation of the entry points' generic compositions (`Funs.lean`) is run over both
backends' records (`AArch64/Backend.lean` and `X86_64/Backend.lean`):

- on every reference vector of `Vectors.lean`, through `mul_with`, `square_with`, and
  `from_mont_with`;
- through `sqr_n_mul_with`, on the known answers of the crate's tests;
- through `add_with` and `sub_with`, on pairs of canonical inputs, against the sum and the
  difference modulo `p`;
- on operands outside the contracts, where the translated debug assertions must make the
  composition fail.

These are tests, not proofs. The loop of `sqr_n_mul_with` is Aeneas' `loop` combinator, which the
kernel cannot unfold, so the checks run as compiled code with `#guard`, which fails the build if a
check is false. They add no declaration to the development.
-/

namespace PastaCurves.Glue

open Aeneas Aeneas.Std

/-- The two backends' records of the Montgomery blocks. -/
def backends : List (pasta_curves.montgomery.MontgomeryBlocks Unit) :=
  [AArch64.montgomeryBlocks, X86_64.montgomeryBlocks]

/-- The limbs of a result that succeeds, and `none` for one that fails, diverges, or panics. -/
def resultLimbs (res : Result (Std.Array Std.U64 4#usize)) : Option Limbs :=
  (Option.ofResult res).map limbsOfArray

#guard backends.all fun B => mulVectors.all fun (_, F, a, b, res) =>
  resultLimbs (pasta_curves.montgomery.mul_with B (limbsArray a) (limbsArray b)
    (limbsArray F.modulus) (word F.inv)) == some res

#guard backends.all fun B => sqrVectors.all fun (_, F, a, res) =>
  resultLimbs (pasta_curves.montgomery.square_with B (limbsArray a) (limbsArray F.modulus)
    (word F.inv)) == some res

#guard backends.all fun B => fromVectors.all fun (_, F, a, res) =>
  resultLimbs (pasta_curves.montgomery.from_mont_with B (limbsArray a) (limbsArray F.modulus)
    (word F.inv)) == some res

/-- `R^k mod p` at the field `F`, the Montgomery form of `R^(k-1)`. -/
def rPow (F : PastaField) (k : Nat) : Limbs := Limbs.ofNat (R^k % F.modulus.toNat)

-- The known answers of the crate's `sqr_n_mul_known_answers` test.
#guard backends.all fun B => [pallasBase, vestaBase].all fun F =>
  [(2, 0, 3, 4), (1, 1, 2, 2), (2, 1, 1, 3), (2, 2, 3, 7)].all fun (v, n, w, res) =>
    resultLimbs (pasta_curves.montgomery.sqr_n_mul_with B (limbsArray (rPow F v))
      ⟨BitVec.ofNat _ n⟩ (limbsArray (rPow F w)) (limbsArray F.modulus) (word F.inv)) ==
      some (rPow F res)

/-- Canonical inputs at the field `F`: `0`, `1`, `7`, `p - 1`, `p / 3`, and `2^200 + 1`. -/
def inputs (F : PastaField) : List Nat :=
  [0, 1, 7, F.modulus.toNat - 1, F.modulus.toNat / 3, 2^200 + 1]

#guard backends.all fun B => [pallasBase, vestaBase].all fun F =>
  let p := F.modulus.toNat
  (inputs F).all fun x => (inputs F).all fun y =>
    resultLimbs (pasta_curves.montgomery.add_with B (limbsArray (Limbs.ofNat x))
      (limbsArray (Limbs.ofNat y)) (limbsArray F.modulus)) == some (Limbs.ofNat ((x + y) % p)) &&
    resultLimbs (pasta_curves.montgomery.sub_with B (limbsArray (Limbs.ofNat x))
      (limbsArray (Limbs.ofNat y)) (limbsArray F.modulus)) ==
      some (Limbs.ofNat ((x + p - y) % p))

-- Outside the contracts, the translated assertions make the composition fail. The cases: `p` itself
-- as an operand of `add`, `sub`, `square`, and `sqr_n_mul`, and two unreduced operands of `mul`.
#guard backends.all fun B => [pallasBase, vestaBase].all fun F =>
  let m := limbsArray F.modulus
  let one := limbsArray (Limbs.ofNat 1)
  let ones := limbsArray (Limbs.ofNat (2^256 - 1))
  resultLimbs (pasta_curves.montgomery.add_with B m one m) == none &&
    resultLimbs (pasta_curves.montgomery.sub_with B one m m) == none &&
    resultLimbs (pasta_curves.montgomery.square_with B m m (word F.inv)) == none &&
    resultLimbs (pasta_curves.montgomery.sqr_n_mul_with B m ⟨BitVec.ofNat _ 0⟩ one m
      (word F.inv)) == none &&
    resultLimbs (pasta_curves.montgomery.mul_with B ones ones m (word F.inv)) == none

end PastaCurves.Glue
