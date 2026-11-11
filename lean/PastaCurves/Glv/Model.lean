import Aeneas
import Mathlib.Algebra.Group.Defs
import Mathlib.Logic.Function.Basic

/-!
# The GLV path's wNAF recoding and Straus ladder, modelled

This file mirrors, line by line, the parts of `src/glv.rs` that recode a scalar's two halves into
wNAF digits and run the ladder over the two tables, and the one caller of them outside that file:

- `wnaf_digits`: `Wnaf.exec`, with its return value `Wnaf.run`;
- `Decomposed::new`: `Decomposed.new`;
- `Table::add_digit` and `Table::mul_decomposed`: `addDigit` and `Table.mulDecomposed`;
- `Table::window_proj`, `Table::from_window`, `Table::new` and `Table::batch`: `windowProj`,
  `Table.ofWindow`, `Table.new` and `Table.batch`;
- `Table::mul` and `GlvParams::mul_glv`: `Table.mul` and `mulGlv`;
- `batch_mul_same_scalar_vartime`, as Pallas and Vesta implement it with the `glv` feature
  (`src/curves.rs`): `batchMulSameScalar`.

Rust's integers are Lean's: a `u128` is a natural number and an `i8` an integer. Every operation
that a debug build checks computes the mathematical result, wraps it as a release build with
`overflow-checks = false` does, and records in a flag whether the result left the type's range.
A write past the end of `digits` writes nothing and sets a flag; Rust panics there in every build.
So one run of the model gives the outcome of every build profile (`Wnaf.State.panicsIn`).

The curve arithmetic is a record of the primitives the path calls (`CurveOps`), the way Aeneas
translates a trait; `CurveOps.Spec` (`Spec.lean`) states what is assumed of them. Allocation is
outside the model, which assumes it succeeds (see `Table.batch`). The scalar has any type, and
`decompose` (with the `scalar_limbs(k)` it calls) is an argument on it of the type Aeneas would
give `decompose::<C>` at one curve (`DecomposeFn`); `DecomposeSpec` (`Spec.lean`) states its
contract, which is assumed. Elsewhere a `none` result is a panic.
-/

namespace PastaCurves.Glv

open Aeneas Aeneas.Std

/-! ## Rust's integer semantics -/

/-- `x` reduced into `[-128, 128)`: an `i8` result as a release build keeps it, and `x as i8`. -/
def wrapI8 (x : ℤ) : ℤ := (x + 128) % 256 - 128

/-- Whether `x` lies outside `[-128, 127]`: an `i8` result on which a debug build panics. -/
def overI8 (x : ℤ) : Bool := decide (x < -128) || decide (127 < x)

/-- `x` reduced into `[0, 2^128)`: a `u128` result as a release build keeps it, and
`x as u128`. -/
def wrapU128 (x : ℤ) : ℕ := (x % 2 ^ 128).toNat

/-- Whether `x` lies outside `[0, 2^128)`: a `u128` result on which a debug build panics. -/
def overU128 (x : ℤ) : Bool := decide (x < 0) || decide (2 ^ 128 ≤ x)

/-! ## `wnaf_digits` -/

namespace Wnaf

/-- The state of `wnaf_digits` between passes of its loop. -/
structure State where
  /-- `digits`. -/
  digits : Fin 128 → ℤ
  /-- `n`. `n += 1` is not checked here: `n` grows by one per pass, and the passes number at most
  129 for every `u128` (`wnaf_digits_length_le_log2`, `Wnaf.lean`), far below any `usize::MAX`. -/
  n : ℕ
  /-- `k`. -/
  k : ℕ
  /-- `debug_assert!(a >> 127 == 0, …)` failed: a debug build panics. -/
  assertFailed : Bool
  /-- Some `digits[n] = …` had `n ≥ 128`: every build panics. -/
  outOfBounds : Bool
  /-- Some `k -= …` or `k += …` left `[0, 2^128)`: a debug build panics, a release build wraps. -/
  u128Overflow : Bool
  /-- Some `low - 16` or `-d` left `[-128, 127]`: a debug build panics, a release build wraps. -/
  i8Overflow : Bool

/-- The start of `wnaf_digits`: the assertion, `digits = [0i8; 128]`, `n = 0` and `k = a`. -/
def init (a : ℕ) : State where
  digits := fun _ => 0
  n := 0
  k := a
  assertFailed := a >>> 127 != 0
  outOfBounds := false
  u128Overflow := false
  i8Overflow := false

/-- `digits[n] = v`: an index past the array writes nothing and sets the flag. -/
def write (s : State) (v : ℤ) : State :=
  if h : s.n < 128 then { s with digits := Function.update s.digits ⟨s.n, h⟩ v }
  else { s with outOfBounds := true }

/-- The body of `if k & 1 == 1 { … }`. -/
def emit (neg : Bool) (s : State) : State :=
  -- `let low = (k & 0xF) as i8;`
  let low : ℤ := wrapI8 (s.k &&& 0xF : ℕ)
  -- `let d = if low >= 8 { low - 16 } else { low };`
  let d : ℤ := if 8 ≤ low then wrapI8 (low - 16) else low
  -- `digits[n] = if negate { -d } else { d };`
  let s' := write s (if neg then wrapI8 (-d) else d)
  -- `if d >= 0 { k -= d as u128; } else { k += (-d) as u128; }`, before the wrap
  let x : ℤ := if 0 ≤ d then (s'.k : ℤ) - wrapU128 d else (s'.k : ℤ) + wrapU128 (wrapI8 (-d))
  { s' with
    k := wrapU128 x
    u128Overflow := s'.u128Overflow || overU128 x
    i8Overflow := s'.i8Overflow || (decide (8 ≤ low) && overI8 (low - 16)) ||
      (neg && overI8 (-d)) || (!decide (0 ≤ d) && overI8 (-d)) }

/-- One pass of the loop. -/
def step (neg : Bool) (s : State) : State :=
  -- `if k & 1 == 1 { … }`
  let s := if s.k &&& 1 = 1 then emit neg s else s
  -- `n += 1; k >>= 1;`
  { s with n := s.n + 1, k := s.k >>> 1 }

/-- The loop `while k != 0 { … }`, for at most `fuel` passes. Each pass lowers a nonzero `k`
(`step_k_lt`), so from `fuel ≥ k` the loop always stops at `k = 0` (`loop_k`): the fuel makes the
recursion structural and never cuts the loop short. -/
def loop (neg : Bool) : ℕ → State → State
  | 0, s => s
  | fuel + 1, s => if s.k = 0 then s else loop neg fuel (step neg s)

/-- `wnaf_digits(a, negate)`: the loop's final state, flags included. -/
def exec (a : ℕ) (neg : Bool) : State := loop neg a (init a)

/-- `wnaf_digits(a, negate)`: its return value `(digits, n)`. -/
def run (a : ℕ) (neg : Bool) : (Fin 128 → ℤ) × ℕ := ((exec a neg).digits, (exec a neg).n)

/-- Whether a build panics, given whether it has debug assertions and overflow checks. A debug
build has both; a release build has neither unless its profile turns them on. The value
`(digits, n)` is the one a build returns when it does not panic. -/
def State.panicsIn (debugAssertions overflowChecks : Bool) (s : State) : Bool :=
  (debugAssertions && s.assertFailed) || (overflowChecks && (s.u128Overflow || s.i8Overflow)) ||
    s.outOfBounds

/-- Whether some build panics. -/
def State.panics (s : State) : Bool := s.panicsIn true true

/-- A write leaves `k` alone. -/
@[simp] theorem write_k (s : State) (v : ℤ) : (write s v).k = s.k := by
  unfold write; split <;> rfl

/-- A write leaves `n` alone. -/
@[simp] theorem write_n (s : State) (v : ℤ) : (write s v).n = s.n := by
  unfold write; split <;> rfl

/-- A pass lowers a nonzero `k`. -/
theorem step_k_lt (neg : Bool) (s : State) (h : s.k ≠ 0) : (step neg s).k < s.k := by
  have h1 : s.k &&& 1 = s.k % 2 := Nat.and_one_is_mod s.k
  have h15 : s.k &&& 0xF = s.k % 16 := Nat.and_two_pow_sub_one_eq_mod s.k 4
  have hp : (2 : ℤ) ^ 128 = 340282366920938463463374607431768211456 := by norm_num
  simp only [step, Nat.shiftRight_eq_div_pow, pow_one]
  split
  · simp only [emit, write_k, wrapU128, wrapI8, h15, hp]
    split_ifs <;> omega
  · omega

/-- From `fuel ≥ k`, the loop stops at `k = 0`. -/
theorem loop_k (neg : Bool) : ∀ (fuel : ℕ) (s : State), s.k ≤ fuel → (loop neg fuel s).k = 0
  | 0, s, h => by simp only [loop]; omega
  | fuel + 1, s, h => by
    simp only [loop]
    split
    · assumption
    · rename_i hk
      exact loop_k neg fuel (step neg s) (by have := step_k_lt neg s hk; omega)

/-- `exec` ends at `k = 0`, as the `while` loop does: it returns the first state of the passes
from `init a` with `k = 0`, which is the state the Rust loop exits with. -/
theorem exec_k (a : ℕ) (neg : Bool) : (exec a neg).k = 0 :=
  loop_k neg a (init a) le_rfl

end Wnaf

/-! ## `Decomposed::new` -/

/-- A `Decomposed`: both halves' digits and the positions in use. -/
structure Decomposed where
  /-- `digits1`. -/
  digits1 : Fin 128 → ℤ
  /-- `digits2`. -/
  digits2 : Fin 128 → ℤ
  /-- `len`. -/
  len : ℕ

/-- `Decomposed::new` after its call to `decompose`, from the halves `(negative, magnitude)` that
`decompose` returns. `none`: `wnaf_digits` panics in some build. -/
def Decomposed.ofHalves (h1 h2 : Bool × ℕ) : Option Decomposed :=
  -- `let (digits1, len1) = wnaf_digits(a1, neg1);`
  let s1 := Wnaf.exec h1.2 h1.1
  -- `let (digits2, len2) = wnaf_digits(a2, neg2);`
  let s2 := Wnaf.exec h2.2 h2.1
  if s1.panics || s2.panics then none
  -- `Decomposed { digits1, digits2, len: len1.max(len2), … }`
  else some ⟨s1.digits, s2.digits, max s1.n s2.n⟩

/-- `decompose::<C>` at one curve, from a scalar of type `Sc` (Rust's `C::ScalarExt`) to the
halves `(negative, magnitude)`, with the type Aeneas would give it. -/
abbrev DecomposeFn (Sc : Type) :=
  Sc → Result ((Bool × Std.U128) × (Bool × Std.U128))

/-- `Decomposed::new`. `none`: `decompose` fails or `wnaf_digits` panics. -/
def Decomposed.new {Sc : Type} (f : DecomposeFn Sc) (k : Sc) : Option Decomposed :=
  -- `let ((neg1, a1), (neg2, a2)) = decompose::<C>(k);`
  match Option.ofResult (f k) with
  | some ((neg1, a1), (neg2, a2)) => Decomposed.ofHalves (neg1, a1.val) (neg2, a2.val)
  | none => none

/-! ## The ladder -/

/-- The point operations `mul_decomposed` and `add_digit` call, over the projective type `P`
(Rust's `C`) and the affine type `A` (`C::AffineExt`). -/
structure LadderOps (P A : Type) where
  /-- `C::identity()`. -/
  identity : P
  /-- `C::double`. -/
  double : P → P
  /-- `acc += a` with `a : C::AffineExt` (mixed addition). -/
  addMixed : P → A → P
  /-- `-a` with `a : C::AffineExt`. -/
  negA : A → A

/-- A `Table`: `t1` holds `{1, 3, 5, 7}·P` and `t2` holds `{1, 3, 5, 7}·φ(P)`. -/
structure Table (A : Type) where
  /-- `t1`. -/
  t1 : Fin 4 → A
  /-- `t2`. -/
  t2 : Fin 4 → A

variable {P A : Type}

/-- `Table::add_digit`. `none`: the table index is out of bounds, a panic. -/
def addDigit (O : LadderOps P A) (acc : P) (table : Fin 4 → A) (d : ℤ) : Option P :=
  if d ≠ 0 then
    -- `let mut a = table[(d.unsigned_abs() / 2) as usize];`
    if h : d.natAbs / 2 < 4 then
      let a := table ⟨d.natAbs / 2, h⟩
      -- `if d < 0 { a = -a; }`
      let a := if d < 0 then O.negA a else a
      -- `*acc += a;`
      some (O.addMixed acc a)
    else none
  else some acc

/-- `k.digits1[i]` or `k.digits2[i]`. `none`: the index is past the array, a panic. -/
def digitAt (d : Fin 128 → ℤ) (i : ℕ) : Option ℤ :=
  if h : i < 128 then some (d ⟨i, h⟩) else none

/-- The loop of `Table::mul_decomposed`, at positions `i - 1` down to `0`. -/
def ladderLoop (O : LadderOps P A) (t : Table A) (D : Decomposed) : ℕ → P → Option P
  | 0, acc => some acc
  | i + 1, acc => do
    -- `if i + 1 < k.len { acc = acc.double(); }`
    let acc := if i + 1 < D.len then O.double acc else acc
    -- `Self::add_digit(&mut acc, &self.t1, k.digits1[i]);`
    let d1 ← digitAt D.digits1 i
    let acc ← addDigit O acc t.t1 d1
    -- `Self::add_digit(&mut acc, &self.t2, k.digits2[i]);`
    let d2 ← digitAt D.digits2 i
    let acc ← addDigit O acc t.t2 d2
    ladderLoop O t D i acc

/-- `Table::mul_decomposed`: `acc = C::identity()`, then the loop over `(0..k.len).rev()`. -/
def Table.mulDecomposed (O : LadderOps P A) (t : Table A) (D : Decomposed) : Option P :=
  ladderLoop O t D D.len O.identity

/-- The ladder's operations read in an additive group, where `mul_decomposed_spec`
(`Ladder.lean`) states the ladder's result. -/
def groupOps (G : Type) [AddCommGroup G] : LadderOps G G where
  identity := 0
  double x := 2 • x
  addMixed x a := x + a
  negA a := -a

/-! ## The tables and the entry points -/

/-- Every curve operation the path calls: the ladder's, and those of `mul_glv`, the table
builders and `batch_mul_same_scalar_vartime`. They are total functions, so the model assumes that
each returns; `CurveOps.Spec` (`Spec.lean`) states what else is assumed of them.
`batch_normalize`'s `assert_eq!` on its two lengths is not modelled: both callers pass equal
lengths. -/
structure CurveOps (P A : Type) extends LadderOps P A where
  /-- `bool::from(C::is_identity())`. -/
  isIdentity : P → Bool
  /-- `C + C`. -/
  add : P → P → P
  /-- `CurveExt::endo`. -/
  endo : P → P
  /-- `C::AffineExt::identity()`. -/
  identityA : A
  /-- `C::batch_normalize(&proj, &mut affine)`: the new contents of `affine`. -/
  batchNormalize : List P → List A → List A
  /-- `CurveAffine::to_curve`: an affine point as a projective one. -/
  toCurve : A → P

/-- `Table::window_proj`: `[1P, 3P, 5P, 7P, φ(1P), φ(3P), φ(5P), φ(7P)]`, with the two loops
written out pass by pass. -/
def windowProj (O : CurveOps P A) (p : P) : List P :=
  -- `let two_p = p.double();`
  let two_p := O.double p
  -- `let mut w = [*p; 8];`, then `for i in 1..4 { w[i] = w[i - 1] + two_p; }`
  let w0 := p
  let w1 := O.add w0 two_p
  let w2 := O.add w1 two_p
  let w3 := O.add w2 two_p
  -- `for i in 0..4 { w[i + 4] = w[i].endo(); }`
  [w0, w1, w2, w3, O.endo w0, O.endo w1, O.endo w2, O.endo w3]

/-- `Table::from_window`: `t1` from `w[..4]` and `t2` from `w[4..8]`. `none`: fewer than eight
entries, where the slicing panics. -/
def Table.ofWindow (w : List A) : Option (Table A) :=
  if h : 8 ≤ w.length then
    some ⟨fun j => w[j.val]'(by omega), fun j => w[j.val + 4]'(by omega)⟩
  else none

/-- `chunks_exact(8)`: the consecutive eight-entry chunks; a shorter tail is dropped. -/
def chunksExact8 : List A → List (List A)
  | a0 :: a1 :: a2 :: a3 :: a4 :: a5 :: a6 :: a7 :: rest =>
    [a0, a1, a2, a3, a4, a5, a6, a7] :: chunksExact8 rest
  | _ => []

/-- `Table::new`. -/
def Table.new (O : CurveOps P A) (p : P) : Option (Table A) :=
  -- `let proj = Self::window_proj(p);`
  let proj := windowProj O p
  -- `let mut affine = [C::AffineExt::identity(); 8];`
  -- `C::batch_normalize(&proj, &mut affine);`
  let affine := O.batchNormalize proj (List.replicate 8 O.identityA)
  -- `Self::from_window(&affine)`
  Table.ofWindow affine

/-- `Table::batch`. Allocation (`Vec::with_capacity`, `vec!`, `collect`) is outside the model,
which assumes it succeeds; in Rust it aborts when memory runs out, and on a 32-bit target
`Vec::with_capacity(n * 8)` panics from 2,796,203 points on. -/
def Table.batch (O : CurveOps P A) (points : List P) : Option (List (Table A)) :=
  let n := points.length
  -- `if n == 0 { return Vec::new(); }`
  if n = 0 then some []
  else
    -- `proj.extend_from_slice(&Self::window_proj(p))` for each `p`
    let proj := points.flatMap (windowProj O)
    -- `let mut affine = vec![C::AffineExt::identity(); n * 8];`, then `C::batch_normalize`
    let affine := O.batchNormalize proj (List.replicate (n * 8) O.identityA)
    -- `affine.chunks_exact(8).map(Self::from_window).collect()`
    (chunksExact8 affine).mapM Table.ofWindow

/-- `Table::mul`: `self.mul_decomposed(&Decomposed::new(k))`. -/
def Table.mul {Sc : Type} (O : CurveOps P A) (f : DecomposeFn Sc) (t : Table A) (k : Sc) :
    Option P := do
  let D ← Decomposed.new f k
  t.mulDecomposed O.toLadderOps D

/-- `GlvParams::mul_glv`. -/
def mulGlv {Sc : Type} (O : CurveOps P A) (f : DecomposeFn Sc) (p : P) (k : Sc) : Option P :=
  -- `if bool::from(self.is_identity()) { return Self::identity(); }`
  if O.isIdentity p then some O.identity
  -- `Table::new(self).mul(k)`
  else do
    let t ← Table.new O p
    t.mul O f k

/-- The loop `for (output, table) in output.iter_mut().zip(tables) { … }` of
`batch_mul_same_scalar_vartime`: an entry of `output` paired with a table becomes
`table.mul_decomposed(&scalar)`; entries past the last table keep their value. -/
def mulEach (O : LadderOps P A) (D : Decomposed) : List P → List (Table A) → Option (List P)
  | _ :: outs, t :: ts => do
    -- `*output = table.mul_decomposed(&scalar);`
    let r ← t.mulDecomposed O D
    let rs ← mulEach O D outs ts
    some (r :: rs)
  | outs, _ => some outs

/-- `batch_mul_same_scalar_vartime(points, scalar, output)` (`src/curves.rs`), as Pallas and
Vesta implement it with the `glv` feature: the new contents of `output`. `none`: a panic, such as
the `assert_eq!` on the two lengths. -/
def batchMulSameScalar {Sc : Type} (O : CurveOps P A) (f : DecomposeFn Sc) (points : List A)
    (k : Sc) (output : List P) : Option (List P) :=
  -- `assert_eq!(points.len(), output.len());`
  if points.length ≠ output.length then none
  -- `if points.is_empty() { return; }`
  else if points.isEmpty then some output
  else do
    -- `*output = point.to_curve();` for each pair of `points` and `output`
    let output := points.map O.toCurve
    -- `let scalar = crate::glv::Decomposed::<$name>::new(scalar);`
    let D ← Decomposed.new f k
    -- `let tables = crate::glv::Table::batch(output);`
    let tables ← Table.batch O output
    -- `for (output, table) in output.iter_mut().zip(tables) { … }`
    mulEach O.toLadderOps D output tables

end PastaCurves.Glv
