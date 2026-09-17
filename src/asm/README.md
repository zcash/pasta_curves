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

The module currently provides a backend only on `target_arch = "aarch64"`; elsewhere the `asm`
module is absent. Nothing is assembled at build time: the blocks are compiled by the Rust
toolchain, so no C toolchain is needed, and the module adds no dependency.

Field elements and moduli are `[u64; 4]`, least significant limb first, and `inv` is
`-modulus[0]^-1 mod 2^64`. The routines take the modulus and `inv` as arguments, so one
implementation serves both fields, but they rely on the shape the two Pasta moduli share:
`modulus[2] = 0` and `modulus[3] = 2^62`. The module documentation states the operand contract
of each entry point.

## Testing

On AArch64, `cargo test --release` runs known-answer tests of the four entry points for both
fields; on other targets there is nothing to test.
