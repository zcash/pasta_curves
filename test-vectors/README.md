# Hardware reference vectors

`pasta_mul-armv8-vectors.txt` records outputs of Semolina v0.1.4's `mul_mont_pasta`,
`sqr_mont_pasta`, and `from_mont_pasta`, the assembled routines that the crate's inline blocks
transcribe, as vendored by pasta_curves at commit `8ad85e9fab7929f6236960e472f432a4bd9ccd74` (the
head of zcash/pasta_curves#100), run on an Apple M-series machine. One vector per line: the
routine (`MUL`, `SQR`, `FROM`), the field (`Fp`, `Fq`), the operands, and the output, each
256-bit value as 64 big-endian hex digits.

`dump-asm-vectors.patch` is the test that produced them, against that pasta_curves commit; it
prints the vectors to standard output, one test per field, so run the two tests one at a time:

    git apply dump-asm-vectors.patch
    cargo test --release --features aarch64-asm dump_asm_vectors -- --nocapture --test-threads=1

The patch prints a newline before its first vector: under `--nocapture` the test harness prints
its `test ... ` line without one, and the run that produced the committed file lost the first
vector of each field, the squaring of `0`, that way; those two lines were re-added by hand.

The operands are, per field: seventeen singles (eleven fixed values, among them `0`, `1`, `R`,
`R^2`, `R^3`, `p - 1`, and limb patterns at the carry boundaries, and six random values below
`2^254`), each converted out of Montgomery form, each squared, and every ordered pair multiplied;
and six unreduced values (`2^256 - 1`, `p`, `p + 1`, and patterns of all-ones limbs), each
multiplied in both orders with the fixed singles and with each other. The 180 vectors outside the
crate's contracts are among those multiplications with an unreduced operand; the others fall within
one of the two contracts. `src/asm/tests.rs` says which of them are run.
