# Verifying the crate's Pasta assembly routines

The Lake package of the machine-checked proofs about the crate's assembly routines. What is
proved, what is trusted, and how the proofs are made are in the book's
[Formal verification](../book/src/design/formal-verification.md) page.

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
PastaCurves/Inversion/Divstep.lean       half-delta divsteps on integers: the step matrix and its bounds
PastaCurves/Inversion/Packed.lean        divsteps on packed words: the batch equals the true matrix
PastaCurves/Inversion/Divstep59.lean     the 59-step block on low words: three batches and their product
PastaCurves/Inversion/Round.lean         the round arithmetic: five-word `updateFG`, `amontred`, `updateUV`, `finalU`
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
PastaCurves/AArch64.lean                 AArch64 umbrella module
PastaCurves/AArch64/Semantics.lean       AArch64 instruction semantics
PastaCurves/AArch64/Transcription.lean   GENERATED: the blocks and their factored rounds
PastaCurves/AArch64/Compositions.lean    compositions of the AArch64 blocks
PastaCurves/AArch64/Vectors.lean         the AArch64 blocks on the vectors, kernel-checked
PastaCurves/AArch64/Spec.lean            proofs about the AArch64 blocks and compositions
PastaCurves/AArch64/Spec/*.lean          the block proofs, one file per block, imported by Spec.lean
PastaCurves/AArch64/Entry.lean           proofs about the AArch64 entry points at the two fields
PastaCurves/X86_64.lean                  x86-64 umbrella module
PastaCurves/X86_64/Semantics.lean        x86-64 instruction semantics and eight-word product
PastaCurves/X86_64/Transcription.lean    GENERATED: all six x86-64 assembly blocks
PastaCurves/X86_64/Compositions.lean     split square, repeated squaring, backend contracts
PastaCurves/X86_64/Vectors.lean          the x86-64 blocks on the vectors, kernel-checked
PastaCurves/X86_64/Checks.lean           additional kernel-checked arithmetic examples
PastaCurves/X86_64/Spec.lean             proofs about the x86-64 blocks and compositions
PastaCurves/X86_64/Spec/*.lean           the block proofs, one file per block, and Arithmetic.lean
PastaCurves/X86_64/Entry.lean            proofs about the x86-64 entry points at the two fields
scripts/gen.py                           shared bindings, vectors, skeletons, checks, and CLI
scripts/asm_source.py                    shared Rust inline-assembly parser and validation
scripts/gen_aarch64.py                   AArch64 decoding, round factoring, and proof-fact hooks
scripts/gen_x86_64.py                    x86-64 decoding, flag validation, and proof-fact hooks
scripts/test_*.py                        the generator's tests, run by check.sh
scripts/check.sh                         regenerate and diff, skeleton check, generator tests (CI)
scripts/check_nanoda.sh                  re-check the build with an independent kernel (CI)
scripts/check_export_coverage.py         the export roots reach every module, run by check_nanoda.sh
scripts/check_export_axioms.py           the axiom census of the export, run by check_nanoda.sh
```

Shared declarations use namespace `PastaCurves`; architecture declarations use `PastaCurves.AArch64`
and `PastaCurves.X86_64`. Shared modules do not import architecture-specific modules, so both models
reuse them without depending on one another. The package is built with Lake from this directory
(`lake build`), with Mathlib pinned in `lake-manifest.json`. `scripts/ci.sh` at the repository root
runs these checks together with the crate's.
