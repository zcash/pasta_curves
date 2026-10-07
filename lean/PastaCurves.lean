import PastaCurves.Semantics
import PastaCurves.Pratt
import PastaCurves.Primality
import PastaCurves.Fields
import PastaCurves.FieldTypes
import PastaCurves.KnownAnswers
import PastaCurves.Compositions
import PastaCurves.Spec
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
import PastaCurves.AArch64
import PastaCurves.X86_64

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
The values the inversion handles are defined across three layers: the book states their ranges,
the Rust (`src/asm/inversion.rs` for the driver, `src/asm/aarch64.rs` for the blocks) lays them
out in words, and the Lean models both. This section explains each value: its range and where the
bound comes from, its encoding in words, and its names in the Rust and here. The next section,
"Recap", tabulates the same values for reference.

Throughout, `p` is one of the two Pasta primes, `R = 2^256` is the Montgomery radix, and a word
is 64 bits. Congruences are modulo `p`.

### Encodings

* **Four limbs** (`Limbs`): an unsigned integer below `2^256`, as four words `l0` to `l3` of
  weight `2^0`, `2^64`, `2^128`, `2^192`. The Rust type is `Limbs = [u64; 4]`. Field elements,
  the modulus, and the coefficients `u` and `v` are four limbs.
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
* **Signed word**: one word read as two's complement, for `d`, the matrix entries, and the
  packed words. A negative entry `e` is stored as `2^64 + e`.
* **Mask**: a word that is all ones (`2^64 - 1`) or `0`, used as a sign, so that a selection is
  an `and` or `xor` rather than a branch.
* **Sign-magnitude**: a signed entry `e` as the pair `(m, s)` with `m = abs(e)` and `s` the mask
  of `e < 0` (`SignMagRep`). The multiplication instructions are unsigned, so the row blocks
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
* `v0`: the start of `v`. Each round divides `(f, g)` by `2^59` exactly but `(u, v)` by `2^64`
  (one Montgomery word), so the invariant drifts by `2^5` per round. Starting `v` at
  `2^(512 + 5 * 10) = 2^562` compensates the ten rounds, so that the output is `X^-1 R`. The
  caller passes it, as it passes `modulus` and `inv`.
* `z`: the output, canonical, with `x * z = 2^512 (mod p)`: the Montgomery form of `X^-1`. For
  `x = 0` it is `0`, so a caller that needs an optional inverse checks for zero separately. The
  Lean statements are `montInv_spec` on the model and `invert_entry_spec` on the blocks.

### The state of the rounds

The state is `InvertState` in `PastaCurves.Compositions`, over a backend's blocks, and
`RoundState` in `Inversion/Model.lean`, over the integer model (where `d` is an integer).

* `d`: twice the "half-delta" of Bernstein and Yang, so `d = 2 delta` with `delta` a half
  integer. With the parity of `g`, it decides which branch each divstep takes (the definition
  of `divstep`, section 1 of the book), and it stays small.
* `f`, `g`: the divstep state, starting at `(p, x)`. Divsteps never increase the larger
  magnitude (Lemma 3), so `abs(f), abs(g) <= p < 2^255`; they are signed, hence five words.
* `u`, `v`: the coefficients, starting at `(0, v0)`. They are unsigned: `amontred` returns a
  nonnegative value below `2 p`, hence below `2^256` (Lemma 10). They are not reduced below `p`
  between rounds, since the next round only needs them below `2^256`.

The invariant after round `i` is `(f, g) = x * 2^(5 i - 562) * (u, v) (mod p)`. After ten rounds
(590 divsteps, Theorem 5) `g = 0` and `f = +-1`, so `x * (+-u) = 2^512`. The tenth round
therefore computes only `u`, folds in the sign of `f`, and reduces strictly.

### Values inside a round

* Entries of `M`: the transition matrix of 59 divsteps, `2^59 (f', g') = M (f, g)` (Lemma 1).
  Which branch each step takes depends only on `d` and the low bits of `f` and `g` (Lemma 2),
  so `divstep59` computes `M` exactly from the low words, in batches of 20, 20, and 19
  (Corollary 8). Each row's absolute values sum to at most `2^59` and each entry lies in
  `(-2^59, 2^59]` (Lemma 3); `2^59` itself occurs, as for `g = 0`. In Lean the matrix is
  `M 59 s : Mat2` and the block's result is `Divstep59Result`.
* `w_f`, `w_g`: the packed words of a batch of `k <= 20` steps (Lemma 6); `abs(w) < 2^63`, and
  after each step `abs(w_g) < 2^62` (Lemma 6'), so the sum `w_g +- w_f` that the next step halves
  fits a signed word. In Lean, `packedStart` builds the start and `DivstepState.f`, `.g` are
  the words.
* `(m, s)`: the sign-magnitude form of each entry, computed once per round by `sign_mag` and
  used by both row blocks. The Rust result is `[m00, m01, m10, m11, s00, s01, s10, s11]`. In
  Lean, `SignMag`, `signMask`, and `SignMagRep`.
* Row of `f`, `g`: `e * f + e' * g` for a row `(e, e')` of `M`. It is divisible by `2^59` and
  below `2^314` in absolute value (Lemma 9), so `fg_row` computes it in five words and shifts
  it right by 59 exactly.
* `t`: a row of `u`, `v`, `e * u + e' * v`, from `uv_row`. Since `abs(e) + abs(e') <= 2^59` and
  `u, v < 2^256`, `abs(t) < 2^315` (Lemma 10); this is the hypothesis `htv` of
  `amontredBlock_spec`.
* `s`, `w`, `amontred(t)`: the almost-Montgomery reduction by one word (`amontredZ`). Adding
  `2^61 p` makes `s = t + 2^61 p` nonnegative; `w = s * inv mod 2^64` makes `s + w p` divisible
  by `2^64`; the result `(s + w p) / 2^64` is congruent to `t * 2^-64`, below `2 p` and below
  `2^256`. "Almost" means it is not reduced below `p`. In Lean, `amontred` and
  `amontred_spec`.
* `sign`: the sign of the last `f`, which is `+-1`. The low word of `m00 f + m01 g` equals the
  low word of `2^59 f'`, so its top bit is the sign of `f'`. The Rust takes that bit as a mask
  and xors it into the row's masks. The Lean model keeps the whole low word (`signWordOf`) and
  tests it against `2^63` (`finalU`).

### Names of the matrix entries

The same four entries carry four sets of names (tabulated in the recap): the book's `u_n`,
`v_n`, `q_n`, `r_n`, the Rust's `m00`, `m01`, `m10`, `m11`, `Mat2`'s `a`, `b`, `c`, `d`, and
`Mat`'s `m11`, `m12`, `m21`, `m22`, each in reading order, rows first.

`Mat2` is in `Inversion/Divstep.lean`. `Mat` (`Inversion/Hull.lean`) is the rational matrix of
the termination certificate's step maps rather than the integer transition matrix, with the same
layout. Names that mean different things in different places:

* `m11` is the bottom-right entry in the Rust and in `Divstep59Result`, and the top-left entry
  in `Mat`.
* `d` is the half-delta (`State.d`) and the bottom-right entry of a `Mat2` (`(M n s).d`); both
  occur in `InvertBlocks.Spec`.
* `u` and `v` are the first row of the matrix in the book's Lemmas 1 to 7, and the coefficients
  of the rounds everywhere else.
* `t` is `-w_f` in Lemma 7 and a row of `u`, `v` in Lemma 10.
* `w` is a packed word in Lemmas 6 and 7 and the Montgomery multiplier in Lemma 10.
* `s` is `t + 2^61 p` in Lemma 10, a sign mask (`s00` to `s11`) in the Rust, and a divstep
  `State` in the Lean.
-/

/-!
## Recap: the inversion's values and names

The values of the previous section in tables, for reference. "Range" is the bound proved by
the lemma cited there; "encoding" is as defined under "Encodings" there.

### Inputs, constants, and output

| value | range               | encoding | Rust      | Lean           |
|-------|---------------------|----------|-----------|----------------|
| `p`   | `2^254 < p < 2^255` | 4 limbs  | `modulus` | `F.modulus`    |
| `inv` | `-p^-1 mod 2^64`    | 1 word   | `inv`     | `F.inv`        |
| `x`   | `< p`               | 4 limbs  | `x`       | `x`            |
| `v0`  | `2^562 mod p`       | 4 limbs  | `v0`      | `v0`           |
| `z`   | `< p`               | 4 limbs  | result    | `montInvModel` |

### The state of the rounds

| value    | range              | encoding    | Rust     | Lean                        |
|----------|--------------------|-------------|----------|-----------------------------|
| `d`      | odd, starts at `1` | signed word | `d`      | `InvertState.d`, `State.d`  |
| `f`, `g` | `abs <= p`         | 5 words     | `f`, `g` | `InvertState.f`, `.g`       |
| `u`, `v` | `< 2 p`            | 4 limbs     | `u`, `v` | `InvertState.u`, `.v`       |

### Values inside a round

| value           | range               | encoding        | Rust            | Lean              |
|-----------------|---------------------|-----------------|-----------------|-------------------|
| entries of `M`  | `(-2^59, 2^59]`     | signed words    | `m00` to `m11`  | `Divstep59Result` |
| `w_f`, `w_g`    | `abs < 2^63`        | signed words    | in `divstep59`  | `DivstepState`    |
| `(m, s)`        | `m <= 2^59`, mask   | 2 words         | `sign_mag`      | `SignMag`         |
| row of `f`, `g` | `abs < 2^314`       | 5 words         | `fg_row`        | `updateFG`        |
| `t`             | `abs < 2^315`       | 5 words         | `tu`, `tv`      | `Signed5`         |
| `s`             | `0 <= s < 2^317`    | not stored      | in `amontred`   | `amontredZ`       |
| `w`             | `< 2^64`            | 1 word          | in `amontred`   | `amontredZ`       |
| `amontred(t)`   | `< 2 p`             | 4 limbs         | `amontred`      | `amontred`        |
| `sign`          | mask                | 1 word          | `sign`          | `signWordOf`      |

### Names of the matrix entries

| position     | book, Lemma 1 | Rust, assembly | `Mat2` | `Mat` |
|--------------|---------------|----------------|--------|-------|
| top left     | `u_n`         | `m00`          | `a`    | `m11` |
| top right    | `v_n`         | `m01`          | `b`    | `m12` |
| bottom left  | `q_n`         | `m10`          | `c`    | `m21` |
| bottom right | `r_n`         | `m11`          | `d`    | `m22` |
-/
