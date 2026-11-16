# Verifying the crate's Pasta assembly routines

The `PastaCurves` library separates shared arithmetic definitions and lemmas from
architecture-specific models: one module per architecture that has an assembly backend, at present
`AArch64`, `X86_64`, and `ARMv7EM`, each with the same roles (see the layout below). Each holds the
architecture's instruction semantics, the generated transcription of every block of its backend's
source file, the Rust compositions around those blocks, the vector checks, a proof of every block,
and the entry theorems at the two fields. The block structure follows each backend's source: on
x86-64, for instance, the squaring is split into `squareLo`, the exact eight-limb square, and
`squareHi`, its Montgomery reduction, which `X86_64/Compositions.lean` composes into `sqrMont`,
and `from_mont` is a block of its own, proved for every four-limb input. The vectors of the shared
`Vectors.lean` were recorded on AArch64 hardware, so on any other architecture
(`X86_64/Vectors.lean`, `ARMv7EM/Vectors.lean`) they are cross-backend checks rather than
hardware captures of that architecture. The AArch64 and x86-64 `Checks.lean` run the shared
boundary-input checks of `ArithmeticChecks.lean` on their routines.

Every architecture uses `scripts/gen.py`, the command line of the compiler in `scripts/asm2lean/` (a
shared Rust `asm!` front end, a lifter per architecture, one IR, and the transcription and skeleton
back ends) on the crate's configuration in `scripts/pasta/`; `--check` compares generated output
without rewriting it. For ARMv7E-M, the lifter validates the Thumb-2 operand and flag behavior,
and the architecture's configuration factors the repeated conversion rounds and the divstep loop
into checked helpers, requiring each split to reconstruct, node for node, the instructions it
splits. The x86-64 lifter checks operand bindings and output order, read-only input-limb
addresses, register initialization, and CF/OF availability. Ordinary arithmetic models CF but
invalidates unmodeled OF; `imul` and shifts invalidate unmodeled flags. Any subsequent read of an
invalid flag is rejected. `adcx` and `adox` carry chains are tracked independently. The
instruction rules follow Intel's SDM (see the readable entries for
[ADCX](https://www.felixcloutier.com/x86/adcx),
[ADOX](https://www.felixcloutier.com/x86/adox),
[MULX](https://www.felixcloutier.com/x86/mulx), and
[shifts](https://www.felixcloutier.com/x86/sal:sar:shl:shr)).
These are straight-line value models, not proofs of pointer validity or register allocation.
Generator validation tests run as part of `scripts/check.sh`.

## Goal

A machine-checked proof that the crate's routines compute all operations exposed by `src/asm/mod.rs`,
on the Pasta fields, under the operand contracts they actually have.

The proof is about the instruction streams of the crate's own inline `asm!` blocks, not about
a re-derivation of the algorithms:

- `src/asm/aarch64.rs`: `add`, `sub`, `mul`, `square`, and the inversion's `divstep59`, `sign_mag`,
  `fg_row`, `de_row`, `amontred`, and `cond_sub`.
- `src/asm/x86_64.rs`: `add`, `sub`, `mul`, `square_lo`, `square_hi`, `from_mont`, and the
  inversion's `divstep59`, `sign_mag`, `fg_row`, `de_row`, `amontred`, and `cond_sub`.
- `src/asm/armv7em.rs`: 32-bit word primitives, Montgomery multiplication and conversion,
  conditional subtraction, and the packed divstep loop; squaring and inversion compose or
  delegate those blocks. ARMv7E-M does not implement Montgomery addition or subtraction.

The crate's `invert` is a Rust composition of blocks on AArch64, x86-64, and ARMv7E-M, and is
modelled as such. The portable Rust blocks of `invert` (`src/inversion/portable.rs`), which the
remaining targets run, are translated to Lean by Aeneas, with the driver they run under
(`PastaCurves/Portable/`). The translation is run, as compiled code, on the blocks' known answers
and against the model. Each of its six blocks is proved to meet its contract. The translated driver is proved to compute the
composition over the blocks that it runs, so it returns the inverse whenever they meet their
contracts. Since the portable blocks meet them at either of the crate's fields, the translation is
proved to return the Montgomery inverse there.

The generic compositions that run the entry points over a backend's blocks (`src/asm/entry.rs`) are
translated too (`PastaCurves/Glue/`). That translation is checked on the supported reference vectors
and entry-point known answers. It is also proved to meet the entry points' contracts over any
backend whose corresponding blocks meet theirs; each backend's implemented operations are proved
to meet those contracts. So `sqr_n_mul`, which composes blocks, is covered through the translation
rather than mirrored by hand.

The multiplication and squaring blocks are transcriptions of Semolina v0.1.4's
`mul_mont_pasta` and of the squaring loop body of its `sqr_n_mul_mont_pasta`, and the addition
and subtraction blocks were imported from zakura-pasta-curves (see the crate README for the
history).

## Trust story

What is trusted, beyond Lean's kernel and standard axioms:

1. The instruction semantics modules (`PastaCurves/Semantics.lean`,
   `PastaCurves/AArch64/Semantics.lean`, `PastaCurves/ARMv7EM/Semantics.lean`, and
   `PastaCurves/X86_64/Semantics.lean`): the supported AArch64, ARMv7E-M, and x86-64 instructions
   modelled over registers and flags (the carry alone for
   the Montgomery blocks' chains, the four flags for the inversion's conditional instructions).
   Small, reviewable, and cross-checked by executing the model on vectors produced by the real
   binary.
2. The transcription of the inline blocks into Lean. Generated by `scripts/gen.py` from
   `src/asm/aarch64.rs`, `src/asm/armv7em.rs`, and `src/asm/x86_64.rs`, whose front end reads the
   inline assembly and operand declarations, expanding the macros that a block invokes for a
   repeated step;
   CI regenerates and diffs.
3. The Rust mirrored in `PastaCurves/AArch64/Compositions.lean`, AArch64's `from_mont` as the
   multiplication block at one; in `PastaCurves/X86_64/Compositions.lean`, the split square; and in
   `PastaCurves/Compositions.lean`, the inversion's driver `invert` over a record of a backend's
   six blocks, which each backend's module instantiates, with the canonicity check `is_canonical`
   that its entry point asserts. Short, and checked against the Rust by inspection.
4. The reference vectors: outputs of the real assembly on an Apple M-series machine, at
   pasta_curves commit `8ad85e9fab7929f6236960e472f432a4bd9ccd74`, embedded as kernel-checked
   examples (`decide +kernel`). These are concrete closed facts that any independent run of
   the binary reproduces. The vectors were produced by the assembly routines the inline blocks
   transcribe, and by Semolina's `from_mont_pasta`, whose value `fromMont` computes as a
   multiplication by `1`; the examples cover the operands inside the proved contracts, where
   the blocks compute the same values.
5. The field constants in `Fields.lean`, the modulus limbs and `inv` of the two Pasta base fields:
   `decide` checks that the limbs encode the primes as pasta_curves states them, and every vector is
   checked with them. `FieldTypes.lean`, generated from `src/fields/fp.rs` and `src/fields/fq.rs`,
   checks that they are the field types' `MODULUS` and `INV`, and that the field types' `R`, `R2`,
   and `R3` are the powers of `R` that they name. It also checks that their `E0`, the inversion's
   starting `e`, is `2^562 mod p`. The moduli are prime: the kernel checks a Pratt certificate for
   each (`Primality.lean`, with the checker in `Pratt.lean`).
6. The Aeneas translations (`PastaCurves/Portable/`, `PastaCurves/Glue/`, and
   `PastaCurves/ARMv7EM/Compositions/`): Charon and Aeneas at the pinned revision, and the model of
   Rust's operations in Aeneas' Lean library. CI regenerates and diffs them. The one external
   function that the glue calls, `black_box`, is defined in `Glue/FunsExternal.lean` as the identity,
   which is its value. The ARMv7E-M assembly externals in
   `ARMv7EM/Compositions/FunsExternal.lean` call the generated instruction transcriptions for
   `batch`, `from_mont`, and `cond_sub` directly; the ARM composition proofs connect their word-array
   adapters and generated Rust methods to those same checked blocks.
   Each architecture's `Backend.lean` record names the methods its Rust implementation runs;
   `blocksOf` (`Glue/Blocks.lean`) builds records where applicable. The records are checked
   against the implementations by inspection.

   Aeneas' library declares axioms of its own, for opaque Rust items, and two of its tests cite
   `sorryAx`. The axiom census (`scripts/check_export_axioms.py`) checks that no declaration
   outside Aeneas' library depends on any of them, directly or through Aeneas' declarations, so
   the package's theorems rest on the standard axioms alone. nanoda is told of them, since its
   strict mode rejects any declared axiom it is not told of.

Not modelled formally: the compiler's handling of the blocks' operands, that is, the allocation of
registers to the placeholders and the `asm!` options (which the generator checks against the
operands, as described below), read-only pointer
loads, pointer validity, the compiler's `readonly` handling, and the Rust mirrored by hand (item 3).
The model treats the operands as limb inputs and outputs. Those aspects were reviewed by hand.

For x86-64, CF/OF availability is checked by the generator as described above.

## Layout

```
PastaCurves.lean                         root module, imports everything below
PastaCurves/Semantics.lean               shared 64-bit arithmetic and limb representation
PastaCurves/Compositions.lean            shared checks and contracts; the `invert` driver
PastaCurves/Fields.lean                  the two fields and facts about their constants
PastaCurves/Pratt.lean                   Pratt certificates: the checker and its soundness
PastaCurves/Primality.lean               the two Pasta primes, certified
PastaCurves/FieldTypes.lean              GENERATED: the field types' constants, checked
PastaCurves/KnownAnswers.lean            GENERATED: the backend tests' known answers, checked
PastaCurves/Spec.lean                    shared arithmetic and limb lemmas
PastaCurves/Tactic/WordStep.lean         `word_step`, one instruction's step of a generated block proof
PastaCurves/Vectors.lean                 GENERATED: the reference vectors inside the contracts
../test-vectors/pasta_mul-armv8-vectors.txt   the hardware outputs the vectors are generated from
PastaCurves/VectorCheck.lean             a backend's routines, and the vectors it fails
PastaCurves/ArithmeticChecks.lean        boundary-input arithmetic checks over a backend's routines
PastaCurves/Entry.lean                   `EntrySpecs`: the entry points on a backend's records
PastaCurves/Words.lean                   the model's values as the translations' words and arrays
PastaCurves/Inversion/Divstep.lean       half-delta divsteps on integers: the step matrix and its bounds
PastaCurves/Inversion/Packed.lean        divsteps on packed words: the batch equals the true matrix
PastaCurves/Inversion/Divstep59.lean     the 59-step block on low words: three batches and their product
PastaCurves/Inversion/Round.lean         the round arithmetic: five-word `updateFG`, `amontred`, `updateDE`, `finalD`
PastaCurves/Inversion/Termination.lean   the termination bound (Theorem 5) as a proposition
PastaCurves/Inversion/Model.lean         the rounds, `montInvModel`, the round invariant (Lemma 11), and Theorem 12
PastaCurves/Inversion/Hull.lean          convex regions by half-planes, inclusions by Farkas certificates
PastaCurves/Inversion/HullBound.lean     the termination bound from a certificate (the hull-light argument)
PastaCurves/Inversion/HullData.lean      GENERATED: the certificate's half-planes and Farkas records
PastaCurves/Inversion/HullCert.lean      the certificate checked by the kernel; `terminationBound_256`
PastaCurves/Inversion/Correctness.lean   Theorem 12 unconditionally: `montInv_correct`
PastaCurves/Inversion/SignMag.lean       the sign-magnitude form of a matrix entry; the row identities on words
PastaCurves/Inversion/PackedWords.lean   the packed step on words, its packing and decoder, and its batch iteration
PastaCurves/Inversion/Composition.lean   `InvertBlocks.Spec`, and `invert` equals the model over blocks that meet it
PastaCurves/Inversion/Vectors.lean       known answers for the six blocks, from the round model
PastaCurves/Inversion/VectorCheck.lean   a backend's inversion blocks, and the answers they fail
PastaCurves/AArch64.lean                 AArch64 umbrella module
PastaCurves/AArch64/Semantics.lean       AArch64 instruction semantics
PastaCurves/AArch64/Transcription.lean   GENERATED: the blocks and their factored rounds
PastaCurves/AArch64/Compositions.lean    `from_mont` as `mul` by one; the inversion blocks' record
PastaCurves/AArch64/Vectors.lean         the AArch64 blocks on the vectors, kernel-checked
PastaCurves/AArch64/Checks.lean          the shared arithmetic checks on the AArch64 routines
PastaCurves/AArch64/Spec.lean            proofs about the AArch64 blocks
PastaCurves/AArch64/Spec/*.lean          the block proofs, one file per block, imported by Spec.lean
PastaCurves/AArch64/Entry.lean           the AArch64 blocks' contracts; `EntrySpecs` on the backend
PastaCurves/AArch64/Backend.lean         the AArch64 blocks as the translated glue's backend
PastaCurves/ARMv7EM.lean                 ARMv7E-M umbrella module
PastaCurves/ARMv7EM/Semantics.lean       ARMv7E-M instruction and 32-bit word semantics
PastaCurves/ARMv7EM/Transcription.lean   GENERATED: the ARMv7E-M instruction-level blocks
PastaCurves/ARMv7EM/Compositions.lean    umbrella for the ARMv7E-M composition model and translation
PastaCurves/ARMv7EM/Compositions/Model.lean  the ARMv7E-M Rust-level proof models
PastaCurves/ARMv7EM/Compositions/Types.lean  GENERATED by Aeneas: ARMv7E-M translation types
PastaCurves/ARMv7EM/Compositions/Funs.lean   GENERATED by Aeneas: ARMv7E-M backend functions
PastaCurves/ARMv7EM/Compositions/FunsExternal.lean  computational models for opaque assembly calls
PastaCurves/ARMv7EM/Compositions/{Mul,Invert,Shared}/*.lean  translated Rust method proofs
PastaCurves/ARMv7EM/Vectors.lean         the ARMv7E-M routines on the shared vectors
PastaCurves/ARMv7EM/Spec.lean            umbrella for the ARMv7E-M instruction proofs
PastaCurves/ARMv7EM/Spec/**/*.lean       checked block traces and arithmetic/composition proofs
PastaCurves/ARMv7EM/Entry.lean           the proved ARMv7E-M entry compositions at the two fields
PastaCurves/ARMv7EM/Backend.lean         the proved ARMv7E-M blocks as a glue backend
PastaCurves/X86_64.lean                  x86-64 umbrella module
PastaCurves/X86_64/Semantics.lean        x86-64 instruction semantics and eight-word product
PastaCurves/X86_64/Transcription.lean    GENERATED: the blocks and their factored rounds
PastaCurves/X86_64/Compositions.lean     split square; the inversion blocks' record
PastaCurves/X86_64/Vectors.lean          the x86-64 blocks on the vectors, kernel-checked
PastaCurves/X86_64/Checks.lean           the shared arithmetic checks on the x86-64 routines
PastaCurves/X86_64/Spec.lean             proofs about the x86-64 blocks
PastaCurves/X86_64/Spec/*.lean           the block proofs, one file per block, and Arithmetic.lean
PastaCurves/X86_64/Entry.lean            the x86-64 blocks' contracts; `EntrySpecs` on the backend
PastaCurves/X86_64/Backend.lean          the x86-64 blocks as the translated glue's backend
PastaCurves/FlagConventions.lean         the AArch64 and x86-64 subtraction flags and selections correspond
PastaCurves/Portable/Types.lean          GENERATED by Aeneas: the `InvertBlocks` trait, as a record
PastaCurves/Portable/Funs.lean           GENERATED by Aeneas: the portable blocks and the driver
PastaCurves/Portable/Words.lean          the translated record of the portable blocks, `blocks`
PastaCurves/Portable/Vectors.lean        the translation on the known answers, as compiled tests
PastaCurves/Portable/Spec.lean           the helpers' step lemmas, and `cond_sub`'s contract
PastaCurves/Portable/Divstep59.lean      proofs about the translated `divstep59`
PastaCurves/Portable/SignMag.lean        the translated `sign_mag` meets its contract
PastaCurves/Portable/Row.lean            the translated rows meet their contracts
PastaCurves/Portable/Amontred.lean       the translated `amontred` meets its contract
PastaCurves/Portable/Driver.lean         the translated driver computes the composition of its blocks
PastaCurves/Portable/Entry.lean          the portable blocks at the two fields: their record, `invert`
PastaCurves/Glue/Types.lean              GENERATED by Aeneas: the trait `MontgomeryBlocks`
PastaCurves/Glue/Funs.lean               GENERATED by Aeneas: the entry points' generic compositions
PastaCurves/Glue/FunsExternal.lean       the one external function that they call, `black_box`
PastaCurves/Glue/Vectors.lean            the compositions on the vectors and known answers, as tests
PastaCurves/Glue/Blocks.lean             `blocksOf`: a backend's record, from its block models
PastaCurves/Glue/Spec.lean               the compositions meet the entry points' contracts
scripts/gen.py                           the generator's command line
scripts/asm2lean/rust.py                 the Rust inline-assembly front end and validation
scripts/asm2lean/block.py                the block driver, shared by every instruction set
scripts/asm2lean/isa/__init__.py         what an instruction set provides, and the lifter base
scripts/asm2lean/isa/aarch64.py          AArch64 lifting, its checks, and its IR nodes
scripts/asm2lean/isa/armv7em.py          ARMv7E-M lifting, its checks, and its 32-bit IR nodes
scripts/asm2lean/isa/x86_64.py           x86-64 lifting, flag validation, and its IR nodes
scripts/asm2lean/ir.py                   the IR: nodes, programs, liveness
scripts/asm2lean/reroll.py               folding unrolled rounds, matched up to renaming
scripts/asm2lean/validate.py             checking a rerolled block computes the flat block's results
scripts/asm2lean/lean.py, skeleton.py    the transcription and the proof skeletons
scripts/asm2lean/specs.py                the skeleton and Spec-coverage checks
scripts/pasta/                           the crate's blocks, data modules, and Spec manifest
scripts/test_*.py                        the generator's tests, run by check.sh
scripts/check.sh                         regenerate and diff, skeleton check, generator tests (CI)
scripts/build.sh                         lake build, failing on warnings outside Aeneas' library (CI)
scripts/test_build.sh                    check that build.sh rejects a warning (CI)
scripts/gen_aeneas.sh                    regenerate, or check, the Aeneas translations (CI)
scripts/fetch_aeneas.sh                  fetch the Charon and Aeneas that gen_aeneas.sh runs (CI)
scripts/check_nanoda.sh                  re-check the build with an independent kernel (CI)
scripts/check_export_coverage.py         the export roots reach every module, run by check_nanoda.sh
scripts/check_export_axioms.py           the axiom census of the export, run by check_nanoda.sh
```

Shared declarations use namespace `PastaCurves`; architecture declarations use their respective
`PastaCurves.<Architecture>` namespace. Handwritten ARMv7E-M translation proofs are grouped under
`PastaCurves.ARMv7EM.Compositions.{Mul,Invert,Shared}`. Generated Aeneas translations use
`pasta_curves`, the crate's name, as Aeneas names it. Shared modules do not import
architecture-specific modules, so backends reuse them without depending on one another.
`Glue/Vectors.lean` checks the translated glue over the supported vector backends and is imported
only by the root module. The package is built with Lake from this directory
(`scripts/build.sh`), with Mathlib and Aeneas' Lean library pinned in `lake-manifest.json`.
`scripts/ci.sh` at the repository root runs these checks together with the crate's.

`scripts/gen_aeneas.sh` regenerates or checks the Aeneas translations. It runs the Charon and
Aeneas binaries from the release that Aeneas' CI built at the pinned revision.
`scripts/fetch_aeneas.sh` fetches them into `work/aeneas/` and checks them against the recorded
SHA-256 digests. When the pin moves to another release, `gen_aeneas.sh` fails until
`fetch_aeneas.sh` is run again; it never fetches by itself.

## Value representation

Registers are natural numbers, reduced modulo `2^64` by every instruction, and the carry is
a natural number that is `0` or `1`; each instruction is a function on `Nat` (`mulLo a b =
a * b % 2^64`, `umulh a b = a * b / 2^64`, `addc a b c = ((a+b+c) % 2^64, (a+b+c) / 2^64)`,
`subc` with AArch64's carry-means-no-borrow convention, `cselLo`). Two reasons over
`UInt64`. The reference vectors become kernel-checked theorems (`by decide +kernel`, the
kernel's GMP-accelerated `Nat` arithmetic). And the proofs stay in `Nat`: `omega`
understands `%` and `/` by literals, so each carry chain is closed with no `toNat`
plumbing, the products `a * b` being the only atoms it does not see through; the limb
recompositions that relate those atoms are `ring` facts.

## Program shape and proof method

The generator emits each block as a chain of `let`s from the input limbs (and modulus limbs
and `inv`) to the output limbs. A block is read from the `asm!` template lines, with the
operand placeholders (`{r0}`, `{inv}`) as register names, so the transcription reads as the
block does; the `in` and `inout` operands bind the argument limbs and `inv`, and the named
`out` and the `inout` operands are the result limbs. The Rust compiler allocates a register
for every operand and substitutes it for the placeholder, so every register a block uses is a
declared input, output, or scratch. A register-only block declares `options(pure, nomem,
nostack)`: it touches no memory and no stack. The x86-64 blocks that read an argument through a
pointer operand declare `readonly` instead of `nomem`, and read only the argument's words through
it. The generator requires the options to match the operands, and every pointer to be read
(the inline assembly RFC,
<https://rust-lang.github.io/rfcs/2873-inline-asm.html>, and the Reference's chapter,
<https://doc.rust-lang.org/reference/inline-assembly.html>, define the operand forms).

Where Semolina's own generator (`pasta_mul-armv8.pl`, a perl template in the blst style)
emitted a loop body a fixed number of times, the transcription keeps that structure: the
multiplication's round becomes one definition (`mulMontRound` over `MulMontAcc`, the five
accumulator limbs, the quotient, and the two reduction terms carried between rounds), and
the block is its prologue, one call per round, and its epilogue. The generator finds the
rounds at the one instruction that ends each of them, checks that the rounds are identical
except for the operand holding the round's `rhs` limb, and computes the carried registers by
liveness.

The proofs in `Spec.lean` follow the chains instruction by instruction, and their mechanical part is
generated too (`scripts/gen.py --skeleton <arch>:<block>`). The skeleton unfolds the block in the
hypothesis that names its result. Then, for every instruction, one `word_step` (the tactic of
`Tactic/WordStep.lean`) extracts that instruction's `let`s from the hypothesis under unique names,
records the defining equation of the result (by `rfl`, in `%`/`/` form), makes the locals opaque
with `clear_value`, and proves the result's bound below `2^64` by the lemma that the generator
names. The lines after the step derive from the equation the other facts that later steps need (the
carry-chain equation `x + 2^64 * c = a + b + cin`, a carry's bound by `1`, the product decomposition
`lo + 2^64 * hi = a * b`), each an instance of one lemma, clearing the `%`/`/` equation when nothing
later needs it. The hand-written parts are the theorem statements and the `-- BEGIN ... -- END`
annotation blocks between instructions. An annotation block states the Montgomery round invariant
that holds at that point and derives it from the facts it names. `scripts/gen.py --check-spec`
strips the annotation blocks and requires the rest of the proof to be the current skeleton, so an
edit to an `asm!` block regenerates the skeleton and the check fails loudly until the annotations
are moved.

Three measurements fixed this shape. `omega` given a whole reduction round at once, or
given the cancellation fact with its `%` terms still in it, runs for minutes without
finishing; given one carry step at a time, or linear equations only, it answers in
milliseconds. So each round block first rewrites the cancellation fact
`(t0 + p0 * q % 2^64) % 2^64 = 0` to the linear `t0 + lo = 2^64 * c`, using the ghost low
limb and the carry of `subs xzr, t0, #1`; it then shows that neither `adc` wraps; and the
invariant is a linear combination of the instruction equations, proved by `omega` in a
context cleared down to exactly those equations. `clear_value` reverts every later local
and re-checks the reverted context, so extracting all the `let`s up front made each clear
quadratic in the chain's length; extracting one instruction at a time, with the rest of the
chain still folded in the hypothesis, keeps every clear cheap. With the rounds factored out,
the round lemma is proved once over a 33-instruction body and applied three times, and no
proof needs more than the default budget.

The squaring block's proof keeps the flat chain: its four cancellation steps thread a computed
quotient rather than a loaded limb, so they are not factored into a round. The arithmetic that
would otherwise run over the whole context is in three lemmas over free variables (the
eight-limb schoolbook identity, one cancellation step, applied four times, and the final
conditional subtraction), so every `omega` runs in a small context.

## Theorems

With `p = p0 + 2^64 p1 + 2^192 · 2^62` (the modulus limbs are `[p0, p1, 0, 2^62]`, as the
code assumes) and `inv · p0 ≡ −1 (mod 2^64)`:

* `mulMontRound_spec` (proved): one round of the multiplication block, on any accumulator
  carrying the quotient and reduction terms the previous round left: the result carries the
  same relations, and `2^64 · acc' = acc + q · p + 2^64 · (lhs · b)`, under two bounds that
  keep the five-limb accumulator below `2^320` while the reduction's low terms are added
  and after the round's product is folded in.
* `mulMont_spec` (proved): the block's output is below `p` and `output · 2^256 ≡ lhs · rhs
  (mod p)` under two arithmetic conditions: `lhs · (rhs_i + 1) + p + 3 · 2^254 + 2^128 ≤
  2^320` for each of `rhs_1`, `rhs_2`, `rhs_3` (the five-limb accumulator, which enters each
  round below `lhs + p`, stays below `2^320` through the round's `adc`s), and `lhs · rhs <
  2^256 · p` (the final candidate is below `2 · p`, so the carry out of its fourth limb,
  which the block drops, is `0`, and one conditional subtraction reduces it). Its two
  corollaries are the operand contracts: `mulMont_spec_of_lhs_lt`, for `lhs < p` and any
  four-limb `rhs`, and `mulMont_spec_of_rhs_lt`, for any four-limb `lhs` and `rhs < p` whose
  limbs 1 to 3 are at most `2^64 − 3`. Whether `2^64 − 2` can wrap is not settled by these
  conditions.
* `fromMont_spec` (proved): AArch64's `from_mont` is the multiplication block with `1` as
  its right operand, a canonical operand with limbs 1 to 3 zero, so for every four-limb
  `value` the output is below `p` and `output · 2^256 ≡ value (mod p)`; the theorem is
  `mulMont_spec_of_rhs_lt` at `1`.
* `sqrMont_spec` (proved): if `a < p`, the squaring block's output is below `p` and
  `output · 2^256 ≡ a² (mod p)`. The block forms the eight-limb square exactly, reduces its
  low half by four Montgomery cancellation steps, adds the high half, and reduces once
  conditionally. The candidate is below `2 · p`, so the carry the block drops there is `0`.
* `addMod_spec` (proved): for operands whose sum fits in four limbs (`lhs + rhs < 2^256`, so
  the carry that the block drops is `0`), the addition block returns the sum when that is
  below `p` and the sum minus `p` otherwise. Its corollaries: `addMod_spec_of_rhs_lt`, for
  `lhs < 2^255` and `rhs < p`, keeps the result below `2^255`, so a left operand can
  accumulate additions without reduction and still enter the multiplication under
  `mulMont_spec_of_rhs_lt`; `addMod_spec_of_lt`, for canonical operands, gives a canonical
  result. Both have `output ≡ lhs + rhs (mod p)`.
* `subMod_spec` (proved): for `rhs ≤ lhs + p` (so the difference plus `p` is nonnegative), the
  subtraction block returns `lhs − rhs` when `lhs ≥ rhs` and `lhs − rhs + p` otherwise. Its
  corollaries `subMod_spec_of_rhs_lt` (for `lhs < 2^255` and `rhs < p`, a result below
  `2^255`) and `subMod_spec_of_lt` (canonical operands, a canonical result) both have
  `output + rhs ≡ lhs (mod p)`.

The inversion's blocks, each equated with the word-level function of the shared layer
(`PastaCurves/Inversion/`) that the round model composes:

* `condSubBlock_spec` (proved): the conditional subtraction block returns its input when that
  is below `p` and the input minus `p` otherwise, so a value below `2 · p` becomes canonical.
* `amontredBlock_spec` (proved): the reduction block computes the round's `amontred`, one word
  of Montgomery reduction of a five-word value below `2^315` in magnitude after adding
  `2^61 · p`. By Lemma 10 the result is below `2 · p` and below `2^256`, so the top carry that the
  block drops is `0`.
* `signMagBlock_spec` (proved): the block converts the four matrix entries from two's
  complement to magnitudes and sign masks.
* `deRowBlock_spec` and `fgRowBlock_spec` (proved): one matrix row of the `d`, `e` combination,
  the exact `a · d + b · e` in five words, and of the `f`, `g` update, the exact
  `(a · f + b · g) / 2^59`, each under the row bound `|a| + |b| ≤ 2^63` (and, for `fg_row`, `f`
  and `g` below `2^256` in magnitude). Both are stated over the sign-magnitude representation,
  which admits a zero entry with an all-ones mask, since the last round's masking by the sign word
  produces one.
* `divstepRound_spec` and `divstepLast_spec` (proved): one packed half-delta divstep of the
  `divstep59` macro is the recurrence of `Inversion/Packed.lean`, under the no-wrap bound on
  the packed `g` (Lemma 6′).
* `divstep59Block_spec` (proved): for `f` and `two_delta` odd and `|two_delta| < 2^61`, the
  block's three batches compute `divstep59`'s `two_delta` and the entries of the 59-step matrix
  modulo `2^64`. The assembly's negated decoder is absorbed by the recurrence's sign symmetry.
* `invert_eq_model` (proved, in `Inversion/Composition.lean`): the composition `invert` of
  `PastaCurves/Compositions.lean`, which mirrors the Rust driver over a record of the six blocks,
  computes `montInvModel` for any backend whose blocks meet `InvertBlocks.Spec`, the record of
  the six statements above. Each round carries the model's state by those statements, whose
  bounds come from the model's round invariant. Each architecture's `invertBlocks_spec`
  instantiates the record for its backend's blocks.

The crate's Montgomery entry points, in `Glue/Spec.lean`: `add_with_spec`, `sub_with_spec`,
`mul_with_spec`, `square_with_spec`, `sqr_n_mul_with_spec`, and `from_mont_with_spec` (all proved)
are about Aeneas' translation of the generic compositions in `src/asm/entry.rs`. Each holds at
either of the crate's fields (a `PastaField`, that is `pallasBase` or `vestaBase`), over any backend
whose corresponding blocks meet `AddSubBlocksSpec` (addition and subtraction) or `MulBlocksSpec`
(multiplication, squaring, and conversion) at that field. Backends implementing all five operations
also meet `BlocksSpec`, which combines the two. Each theorem holds under the condition that its
entry point checks in a debug build. For `mul` that is `mulContract`. For the
others —the value operands of `square`, `sqr_n_mul`, `add`, and `sub`— it is `isCanonical`. The
translated checks compute those Booleans (`is_canonical_spec` and `mul_contract_spec`), and
`isCanonical_iff` and `mulContract_iff` relate them to the arithmetic conditions of the theorems
above.

* `sqr_n_mul` keeps its accumulator canonical through the squarings, with
  `output · 2^(256 (2^i − 1)) ≡ a^(2^i) (mod p)` after `i` of them, so its multiplication is under
  the first contract. Its result has `output · 2^(256 · 2^count) ≡ a^(2^count) · rhs (mod p)` for
  any four-limb `rhs`.
* `from_mont` checks nothing and holds for every input.

`montgomeryBlocks_spec`, in each architecture's `Entry.lean`, proves the backend's available
Montgomery operations at either field: AArch64 and x86-64 meet `BlocksSpec`, while ARMv7E-M meets
`MulBlocksSpec`. Its translated addition and subtraction methods panic and have no correctness
specification. AArch64 converts from Montgomery form by multiplying by one (`AArch64/Spec/Mul.lean`);
x86-64 proves its standalone conversion block (`X86_64/Spec/FromMont.lean`); ARMv7E-M proves its
factored conversion instruction routines (`ARMv7EM/Spec/FromMont/`) and the generated Rust method
that calls them.
`invert_entry_spec` proves the crate's `invert` for the proved architecture records. The input is
canonical, as the entry point asserts, and `e0` is `2^562 mod p`, as its contract requires. The
result is canonical. For `x = 0` it is `0`, and otherwise it is the Montgomery inverse, with
`x · result ≡ R^2 (mod p)`. The primality of the modulus is the field's `prime`, and the
termination bound is `terminationBound_256`.

`EntrySpecs` (`Entry.lean`) bundles the seven entry-point statements over a backend's two records,
and `EntrySpecs.of` proves them from the records' contracts. The AArch64 and x86-64 `entrySpecs`
instantiate it from `montgomeryBlocks_spec` and `invertBlocks_spec`, so a backend that meets
`BlocksSpec` states only its records' contracts. ARMv7E-M, which meets `MulBlocksSpec` only,
states its entry theorems in its own `Entry.lean`.

## Status

The generated assembly transcriptions, their checked proof skeletons, and the translated Rust
compositions are in the root import closure. `scripts/check.sh` requires a checked skeleton for
every registered assembly routine; there are no routines listed as unproved. At either Pasta field,
the entry theorems establish the asserted contracts for each backend's implemented operations:
addition and subtraction on AArch64 and x86-64, and multiplication, squaring, repeated squaring,
conversion out of Montgomery form, and inversion on all three backends:

* AArch64 and x86-64 prove their instruction blocks, including the inversion's six blocks, and
  their field-level compositions. AArch64's conversion uses multiplication by one; x86-64 proves
  its standalone conversion block.
* ARMv7E-M proves the 32-bit primitives and the factored Montgomery, conditional-subtraction,
  inversion multiply-accumulate, and packed divstep instruction routines. The checked conversion
  rounds and final reduction compose into `fromMontBlock`; the three checked divstep batches
  compose into `divstep59Block` under the signed-32-bit state bound. Proofs connect the generated
  Rust multiplication, squaring, Montgomery-conversion, and inversion methods and their external
  models to these instruction results. Its generated addition and subtraction methods panic; no
  entry theorem claims that they compute modular sums or differences.
* The translated portable inversion blocks and driver satisfy the shared inversion model. For
  ARMv7E-M, the inversion driver proof uses success at the Pasta moduli rather than asserting that
  its `amontred` method succeeds for arbitrary moduli, where the generated method can fail.
* The generic Aeneas-translated entry compositions meet their contracts over backend records
  whose methods satisfy the corresponding block contracts; the architecture entry theorems
  establish those premises.

The proof covers the modelled instructions and translated Rust behavior under their stated operand
contracts. Compiler register allocation, pointer validity, inline-assembly declarations, and the
hand-reviewed boundaries in the trust story above remain outside the formal model.

## What the transcription's differential testing does not cover

Beside the proofs, the transcriptions were tested against the machine code: each generated
definition, evaluated in Python from its Lean text, was compared with the routine executed natively
(AArch64), under Rosetta 2 and an executable model of Intel's SDM (x86-64), and under an
interpreter of the ARMv7-M Architecture Reference Manual over the compiled object (ARMv7E-M). No
case differed. The testing leaves these gaps, each with what would close it:

* **ARMv7E-M was executed without debug assertions.** The interpreted object was built without
  them, while CI's release tests keep them, and they change the code around each `asm!` block,
  register allocation included. Interpreting an object built with
  `CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true` would close it.
* **ARMv7E-M was not executed on a core.** The interpreter was written from the Architecture
  Reference Manual (DDI 0403E) for the testing, independently of `Semantics.lean`; a misreading
  of the manual shared by both would go unseen. Running the blocks on a Cortex-M4 or M7, or under
  `qemu-system-arm`, would close it.
* **x86-64's `adcx` and `adox` were not executed by a processor.** Rosetta 2 lacks ADX, so the
  runs under it emulated both instructions from the SDM. For them, the comparison is with the SDM
  model only, on exactly the two independent carry chains the generator keeps apart. Running the
  routines on an x86-64 processor with ADX would close it.
* **x86-64's `amontred` never negated zero.** `neg {t0}` sets the carry flag unless its operand
  is zero; no input of the testing made that operand zero, so the case in which it clears the
  carry was not compared with the machine. A vector constructed to make it zero would close it.
