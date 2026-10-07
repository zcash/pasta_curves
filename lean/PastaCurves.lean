import PastaCurves.Semantics
import PastaCurves.Pratt
import PastaCurves.Primality
import PastaCurves.Fields
import PastaCurves.FieldTypes
import PastaCurves.KnownAnswers
import PastaCurves.Compositions
import PastaCurves.Spec
import PastaCurves.ArithmeticChecks
import PastaCurves.Inversion.Divstep
import PastaCurves.Inversion.Packed
import PastaCurves.Inversion.Divstep59
import PastaCurves.Inversion.Round
import PastaCurves.Inversion.Termination
import PastaCurves.Inversion.Model
import PastaCurves.Inversion.Hull
import PastaCurves.Inversion.HullBound
import PastaCurves.Inversion.HullData
import PastaCurves.Inversion.HullCert
import PastaCurves.Inversion.Correctness
import PastaCurves.Inversion.SignMag
import PastaCurves.Inversion.PackedWords
import PastaCurves.Inversion.Composition
import PastaCurves.Inversion.Vectors
import PastaCurves.Inversion.VectorCheck
import PastaCurves.Entry
import PastaCurves.AArch64
import PastaCurves.X86_64
import PastaCurves.FlagConventions
import PastaCurves.Portable.Funs
import PastaCurves.Portable.Words
import PastaCurves.Portable.Vectors
import PastaCurves.Portable.Spec
import PastaCurves.Portable.Divstep59
import PastaCurves.Portable.SignMag
import PastaCurves.Portable.Row
import PastaCurves.Portable.Amontred
import PastaCurves.Portable.Driver
import PastaCurves.Portable.Entry
import PastaCurves.Words
import PastaCurves.Glue.Funs
import PastaCurves.Glue.Blocks
import PastaCurves.Glue.Spec
import PastaCurves.Glue.Vectors

/-!
# The assembly routines, formalized

Generic arithmetic is defined in the top-level `PastaCurves` modules. Architecture-specific
transcriptions and proofs live under their corresponding namespaces.

Every module of the development is imported here, so that a build of this root builds all of
them and the nanoda re-check (`lean/scripts/check_nanoda.sh`) exports all of them.
-/

/-!
## The inversion: values, encodings, and names

The constant-time inversion is specified in `book/src/design/inversion.md`. The docstrings under
`PastaCurves.Inversion` cite that page's lemma numbers, and the lemma numbers below refer to it.
The values the inversion handles are defined across three layers: the book states their ranges, the
Rust lays them out in words (`src/inversion.rs` for the driver over any backend's blocks, each
assembly backend's file under `src/asm` for its blocks, at present `aarch64.rs` and `x86_64.rs`,
and `src/inversion/portable.rs` for the portable blocks, which every other target runs), and the
Lean models both. The three layers share their names: `two_delta` for
the half-delta, `u`, `v`, `q`, `r` for the transition matrix, and `d`, `e` for the coefficients.
This section explains each value: its range and where the bound comes from, its encoding in words,
and its names. The next section, "Recap", tabulates the same values for reference.

Throughout, `p` is one of the two Pasta primes, `R = 2^256` is the Montgomery radix, and a word
is 64 bits. Congruences are modulo `p`.

### Encodings

* **Four limbs** (`Limbs`): an unsigned integer below `2^256`, as four words `l0` to `l3` of
  weight `2^0`, `2^64`, `2^128`, `2^192`. The Rust type is `Limbs = [u64; 4]`. Field elements,
  the modulus, and the coefficients `d` and `e` are four limbs.
* **Five words** (`Signed5`): a signed integer as four limbs and a top word `l4` of weight
  `2^256`, read as a two's-complement word, which together form a 320-bit two's-complement
  integer (`Signed5.toInt`). The Rust type is `[u64; 5]`. A value `z` with `abs(z) < 2^k` has
  bits `k` to `319` all equal to its sign. For the row combination `t` of Lemma 10, `k = 315`,
  so the top word is laid out as:

  ```
  bits 256..314 : value (up to 59 bits)
  bits 315..319 : sign extension, all 0 if t >= 0, all 1 if t < 0
  ```

  Because the lemmas bound every such value well below `2^319`, a block may compute modulo
  `2^320` and drop its final carry without losing the sign.
* **Signed word**: one word read as two's complement, for `two_delta`, the matrix entries, and
  the packed words. A negative entry `a` is stored as `2^64 + a`.
* **Mask**: a word that is all ones (`2^64 - 1`) or `0`, used as a sign, so that a selection is
  an `and` or `xor` rather than a branch.
* **Sign-magnitude**: a signed entry `a` as the pair `(m, s)` with `m = abs(a)` and `s` the mask
  of `a < 0` (`SignMagRep`). The multiplication instructions are unsigned, so the row blocks
  multiply an operand `y` by a negative entry as `m * ((y xor s) + 1)`, which is `m * (-y)` in
  two's complement, and fold the `+ m` into the initial carry. `0` has both forms, `(0, 0)` and
  `(0, 2^64 - 1)`, which the last round relies on when it flips a row's masks.
* **Packed word** (Lemma 6): inside `divstep59`, one signed word carries the low 20 bits of `f`
  (or `g`) together with two entries of the matrix being built, at weights `2^41` and `2^62`
  (`packedStart` subtracts `2^41` from `f` and `2^62` from `g`). Every divstep then updates the
  state and the matrix at once. Lemma 7 unpacks the entries after a batch.

### Inputs, constants, and output

* `p`: both Pasta primes have limbs `l2 = 0` and `l3 = 2^62` (`PastaField.shape`). Lemma 10
  needs `p > 2^254` to keep `t + 2^61 p` nonnegative, and the blocks add multiples of `p` limb
  by limb, so this shape is part of their contract.
* `inv`: the Montgomery constant, chosen so that `s + (s * inv mod 2^64) * p` is divisible by
  `2^64` for every `s`.
* `x`: the input, the Montgomery form `X R mod p` of the element `X` to invert. It must be
  canonical; the Rust entry point debug-asserts this.
* `e0`: the start of `e`. Each round divides `(f, g)` by `2^59` exactly but `(d, e)` by `2^64`
  (one Montgomery word), so the invariant drifts by `2^5` per round. Starting `e` at
  `2^(512 + 5 * 10) = 2^562` compensates the ten rounds, so that the output is `X^-1 R`. The
  caller passes it, as it passes `modulus` and `inv`; the model computes it as `startE`.
* `z`: the output, canonical, with `x * z = 2^512 (mod p)`: the Montgomery form of `X^-1`. For
  `x = 0` it is `0`, so a caller that needs an optional inverse checks for zero separately. The
  Lean statements are `montInv_spec` on the model, the generic `invert_entry_spec` of
  `Inversion/Composition.lean` over any blocks that meet `InvertBlocks.Spec`, and its instances:
  the `invert` field of each assembly backend's `entrySpecs` (`AArch64.entrySpecs`,
  `X86_64.entrySpecs`), and `Portable.invert_entry_spec`.

### The state of the rounds

The state is `InvertState` in `PastaCurves.Compositions`, over a backend's blocks, and
`RoundState` in `Inversion/Model.lean`, over the integer model (where `two_delta` is an integer).
Both have the fields `two_delta`, `f`, `g`, `d`, `e`, the names of the Rust driver's variables.

* `two_delta`: twice the half-delta `delta` of Bernstein and Yang, a half integer, so
  `two_delta` is odd. With the parity of `g`, it decides which branch each divstep takes (the
  definition of `divstep`, section 1 of the book), and it stays small.
* `f`, `g`: the divstep state, starting at `(p, x)`. Divsteps never increase the larger
  magnitude (Lemma 3), so `abs(f), abs(g) <= p < 2^255`; they are signed, hence five words.
* `d`, `e`: the coefficients, starting at `(0, e0)`. They are unsigned: `amontred` returns a
  nonnegative value below `2 p`, hence below `2^256` (Lemma 10). They are not reduced below `p`
  between rounds, since the next round only needs them below `2^256`.

The invariant after round `i` is `(f, g) = x * 2^(5 i - 562) * (d, e) (mod p)` (Lemma 11).
After ten rounds (590 divsteps, Theorem 5) `g = 0` and `f = +-1`, so `x * (+-d) = 2^512`. The
tenth round therefore computes only `d`, folds in the sign of `f`, and reduces strictly with
`cond_sub` (`finalD` in the model).

### Values inside a round

* Entries of `M`: the transition matrix of 59 divsteps, `2^59 (f', g') = M (f, g)` (Lemma 1).
  Which branch each step takes depends only on `two_delta` and the low bits of `f` and `g`
  (Lemma 2), so `divstep59` computes `M` exactly from the low words, in batches of 20, 20, and
  19 (Corollary 8). Each row's absolute values sum to at most `2^59` and each entry lies in
  `(-2^59, 2^59]` (Lemma 3); `2^59` itself occurs, as for `g = 0`. In Lean the matrix is
  `M 59 s : Mat2` and the block's result is `Divstep59Result`.
* `w_f`, `w_g`: the packed words of a batch of `k <= 20` steps (Lemma 6); `abs(w) < 2^63`, and
  after each step `abs(w_g) < 2^62` (Lemma 6'), so the sum `w_g +- w_f` that the next step
  halves fits a signed word. In Lean, `packedStart` builds the start, and each architecture's
  transcription carries the words in its `DivstepState` (`.f`, `.g`), in the layout of its
  instruction schedule: AArch64 carries the packed state with the flags of the next step's parity
  test, and x86-64 carries it with the next step's decision already applied (`X86_64.Decided`).
  The shared batch iteration takes either through `PackedStep.Carries`.
* `(m, s)`: the sign-magnitude form of each entry, computed once per round by `sign_mag` and
  used by both row blocks. The Rust result is `[u, v, q, r, su, sv, sq, sr]`: the magnitudes,
  under the entries' own names, then the masks. In Lean, `SignMag`, `signMask`, and
  `SignMagRep`.
* Row of `f`, `g`: `u f + v g` or `q f + r g`. It is divisible by `2^59` and below `2^314` in
  absolute value (Lemma 9), so `fg_row` computes it in five words and shifts it right by 59
  exactly. In Lean, `updateFG`.
* `t`: a row of `d`, `e`, `u d + v e` or `q d + r e`, from `de_row` (the Rust driver's `td` and
  `te`). Since `abs(u) + abs(v) <= 2^59` and `d, e < 2^256`, `abs(t) < 2^315` (Lemma 10); this is
  the hypothesis `htv` of `amontred_spec` and `amontredBlock_spec`.
* `s`, `w`, `amontred(t)`: the almost-Montgomery reduction by one word (`amontredZ`). Adding
  `2^61 p` makes `s = t + 2^61 p` nonnegative; `w = s * inv mod 2^64` makes `s + w p` divisible
  by `2^64`; the result `(s + w p) / 2^64` is congruent to `t * 2^-64`, below `2 p` and below
  `2^256`. "Almost" means it is not reduced below `p`. In Lean, `amontred` and
  `amontred_spec`; `updateDE` applies it to both rows.
* `sign`: the sign of the last `f`, which is `+-1`. The low word of `u f + v g` equals the low
  word of `2^59 f'`, so its top bit is the sign of `f'`. The Rust takes that bit as a mask and
  xors it into the row's masks. The Lean model keeps the whole low word (`signWordOf`) and
  tests it against `2^63` (`finalD`).

### Names of the matrix entries

The book, the Rust, `Mat2` (`Inversion/Divstep.lean`), `Divstep59Result`, and `SignMag` all name
the four entries `u`, `v`, `q`, `r`, in reading order, rows first; the book writes `u_n`, `v_n`,
`q_n`, `r_n` after `n` steps. `Mat` (`Inversion/Hull.lean`), the rational matrix of the
termination certificate's step maps rather than the integer transition matrix, names the same
layout `m11`, `m12`, `m21`, `m22`.

Names that mean different things in different places:

* `t` is `-w_f` in Lemma 7, a row of `d`, `e` in Lemma 10, and, in the x86-64 transcription,
  the term that a divstep adds to the packed `g` (`DivstepState.t`).
* `w` is a packed word in Lemmas 6 and 7 and the Montgomery multiplier in Lemma 10.
* `s` is `t + 2^61 p` in Lemma 10, a sign mask (`su` to `sr`) in the Rust, and a divstep
  `State` in the Lean.
* `z` is the inversion's output and, in the x86-64 transcription, a register holding zero for
  the conditional moves (`DivstepState.z`).
* `u`, `v`, `q`, `r` are the signed entries in the Rust driver and, after `sign_mag`, their
  magnitudes, which the driver binds to the same names.
-/

/-!
## Recap: the inversion's values and names

The values of the previous section in tables, for reference. "Range" is the bound proved by
the lemma cited there; "encoding" is as defined under "Encodings" there.

### Inputs, constants, and output

| value | range               | encoding | Rust      | Lean                    |
|-------|---------------------|----------|-----------|-------------------------|
| `p`   | `2^254 < p < 2^255` | 4 limbs  | `modulus` | `F.modulus`             |
| `inv` | `-p^-1 mod 2^64`    | 1 word   | `inv`     | `F.inv`                 |
| `x`   | `< p`               | 4 limbs  | `x`       | `x`                     |
| `e0`  | `2^562 mod p`       | 4 limbs  | `e0`      | `e0`, `startE F`        |
| `z`   | `< p`               | 4 limbs  | result    | `montInvModel`          |

### The state of the rounds

The Rust driver's variables and the fields of `InvertState`, `RoundState`, and `State` carry the
values' own names.

| value       | range              | encoding    | Lean                                       |
|-------------|--------------------|-------------|--------------------------------------------|
| `two_delta` | odd, starts at `1` | signed word | `InvertState`, `RoundState`, `State`       |
| `f`, `g`    | `abs <= p`         | 5 words     | `InvertState`, `RoundState`, `State`       |
| `d`, `e`    | `< 2 p`            | 4 limbs     | `InvertState`, `RoundState`                |

### Values inside a round

| value           | range             | encoding     | Rust           | Lean                      |
|-----------------|-------------------|--------------|----------------|---------------------------|
| entries of `M`  | `(-2^59, 2^59]`   | signed words | `u` to `r`     | `Divstep59Result`, `Mat2` |
| `w_f`, `w_g`    | `abs < 2^63`      | signed words | in `divstep59` | each `DivstepState`       |
| `(m, s)`        | `m <= 2^59`, mask | 2 words      | `sign_mag`     | `SignMag`                 |
| row of `f`, `g` | `abs < 2^314`     | 5 words      | `fg_row`       | `updateFG`                |
| `t`             | `abs < 2^315`     | 5 words      | `td`, `te`     | `Signed5`                 |
| `s`             | `0 <= s < 2^317`  | not stored   | in `amontred`  | `amontredZ`               |
| `w`             | `< 2^64`          | 1 word       | in `amontred`  | `amontredZ`               |
| `amontred(t)`   | `< 2 p`           | 4 limbs      | `amontred`     | `amontred`                |
| `sign`          | mask              | 1 word       | `sign`         | `signWordOf`              |

### Names of the matrix entries

"Rust and Lean" covers the Rust, `Mat2`, `Divstep59Result`, and `SignMag`.

| position     | book, Lemma 1 | Rust and Lean | `Mat` |
|--------------|---------------|---------------|-------|
| top left     | `u_n`         | `u`           | `m11` |
| top right    | `v_n`         | `v`           | `m12` |
| bottom left  | `q_n`         | `q`           | `m21` |
| bottom right | `r_n`         | `r`           | `m22` |
-/
