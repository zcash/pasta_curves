# Assembly backends

Assembly backends for the crate's Pasta (Pallas and Vesta) field arithmetic. The module
currently provides an AArch64 backend for Montgomery multiplication, squaring, a
repeated-squaring chain, and conversion out of Montgomery form.

## Provenance

The routines are transcriptions of the Pasta Montgomery routines of Supranational's
[Semolina](https://github.com/supranational/semolina) v0.1.4
([`src/mach-o/pasta_mul-armv8.S`](https://github.com/supranational/semolina/blob/v0.1.4/src/mach-o/pasta_mul-armv8.S)).
`src/asm/aarch64.rs` carries multiplication and squaring as register-renamed inline `asm!`
blocks of `mul_mont_pasta` and of the squaring loop body of `sqr_n_mul_mont_pasta`, with the
same instructions. The repeated-squaring chain and the conversion out of Montgomery form are
compositions of those blocks. The blocks were ported and adapted in
[zakura-core/common](https://github.com/zakura-core/common) and then in
[zcash/pasta_curves#100](https://github.com/zcash/pasta_curves/pull/100). The `asm` module
imports them from that pull request at commit `efc0c69533f491743162f3263acfb6c23603ad91`, which
reaches the chain and the conversion through assembled routines instead.

## Usage

The module provides a backend for `target_arch = "aarch64"` and, in part, for
`target_arch = "x86_64"`: `add`, `sub`, and `from_mont` are register-only and available on
every x86-64 target (MULX needs BMI2 for `from_mont`), while `mul`, `square`, and the routines
built on them read limbs through pointers and so require 64-bit pointers, plus MULX and
ADCX/ADOX (BMI2 and ADX: Intel Broadwell / AMD Zen or newer). Elsewhere the `asm` module is
absent. Nothing is assembled at build time: the blocks are compiled by the Rust toolchain, so no
C toolchain is needed, and the module adds no dependency.

Field elements and moduli are `[u64; 4]`, least significant limb first, and `inv` is
`-modulus[0]^-1 mod 2^64`. The routines take the modulus and `inv` as arguments, so one
implementation serves both fields, but they rely on the shape the two Pasta moduli share:
`modulus[2] = 0` and `modulus[3] = 2^62`. The module documentation states the operand contract
of each entry point.

## Testing

On AArch64, `cargo test --release` runs known-answer tests of the four entry points for both
fields; on other targets there is nothing to test.

## Formal verification

`lean/` holds a Lean 4 development that models the routines formally and contributes to assuring
their correctness: an instruction-level model, with proofs for the multiplication block under
each of its two operand contracts and for the conversion out of Montgomery form for any input,
and with the squaring block transcribed and checked against hardware vectors but not yet proved.
The transcription is generated from the module's own inline blocks, CI regenerates and diffs it,
and an independent implementation of the Lean kernel re-checks the build. See
[`lean/README.md`](../../lean/README.md).
