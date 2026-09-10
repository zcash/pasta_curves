# Apple AArch64 assembly backend

Apple AArch64 assembly backend for the crate's Pasta (Pallas and Vesta) field arithmetic:
Montgomery multiplication, squaring, a fused repeated-squaring chain, and conversion out of
Montgomery form, for the `aarch64-apple-*` targets.

## Provenance

The routines are transcriptions of the Pasta Montgomery routines of Supranational's
[Semolina](https://github.com/supranational/semolina) v0.1.4
([`src/mach-o/pasta_mul-armv8.S`](https://github.com/supranational/semolina/blob/v0.1.4/src/mach-o/pasta_mul-armv8.S)).
`src/asm/pasta_mul-armv8.S` keeps the fused repeated-squaring chain and the conversion out of
Montgomery form, with their shared reduction helper, as assembled routines; `src/asm/mod.rs`
carries multiplication and squaring as register-renamed inline `asm!` blocks of the same
instructions.
They were ported and adapted in [zakura-core/common](https://github.com/zakura-core/common) and
then in [zcash/pasta_curves#100](https://github.com/zcash/pasta_curves/pull/100); the `asm`
module imports them from that pull request at commit `efc0c69533f491743162f3263acfb6c23603ad91`.

## Usage

The module is compiled only on `target_arch = "aarch64"` with `target_vendor = "apple"`;
elsewhere the crate has its portable arithmetic alone. Building on Apple AArch64 assembles
`src/asm/pasta_mul-armv8.S` through the `cc` crate, so a C toolchain is required there.

Field elements and moduli are `[u64; 4]`, least significant limb first, and `inv` is
`-modulus[0]^-1 mod 2^64`. The routines take the modulus and `inv` as arguments, so one
implementation serves both fields, but they rely on the shape the two Pasta moduli share:
`modulus[2] = 0` and `modulus[3] = 2^62`. The module documentation states the operand contract
of each entry point.

## Testing

On Apple AArch64, `cargo test --release` runs known-answer tests of the four entry points for
both fields; on other targets there is nothing to test.
