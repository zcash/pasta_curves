//! The module's entry points, re-exported at its root. Each runs a generic composition of
//! `crate::montgomery`, the matching `_with` function, over the blocks of the target's backend,
//! which implements [`MontgomeryBlocks`](crate::montgomery::MontgomeryBlocks).

pub use crate::limbs::Limbs;
use crate::montgomery::{add_with, from_mont_with, mul_with, sqr_n_mul_with, square_with, sub_with};

/// The blocks that the entry points run: the backend of the target architecture.
#[cfg(target_arch = "aarch64")]
type Selected = super::aarch64::Backend;
/// The blocks that the entry points run: the backend of the target architecture.
#[cfg(target_arch = "x86_64")]
type Selected = super::x86_64::Backend;

/// Adds two residues for a Pasta modulus and conditionally subtracts the modulus.
///
/// Outputs are canonical.
///
/// # Safety
///
/// Both inputs must be canonical. This is debug-asserted, and under that precondition the
/// machine-checked proofs in `lean/` establish the result (`add_with_spec` and the backend's
/// `montgomeryBlocks_spec`).
///
/// `modulus` must be either the Pallas or Vesta field modulus. Any other values will
/// cause undefined results.
#[inline(always)]
pub fn add(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
    add_with::<Selected>(lhs, rhs, modulus)
}

/// Subtracts two residues for a Pasta modulus, adding the modulus back on underflow.
///
/// Outputs are canonical.
///
/// # Safety
///
/// Both inputs must be canonical. This is debug-asserted, and under that precondition the
/// machine-checked proofs in `lean/` establish the result (`sub_with_spec` and the backend's
/// `montgomeryBlocks_spec`).
///
/// `modulus` must be either the Pallas or Vesta field modulus. Any other values will
/// cause undefined results.
#[inline(always)]
pub fn sub(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
    sub_with::<Selected>(lhs, rhs, modulus)
}

/// Multiplies two Montgomery residues for a Pasta modulus.
///
/// # Safety
///
/// Either `lhs` is canonical (below the modulus) and `rhs` is any four-limb value, or `rhs` is
/// canonical with each of its limbs 1 to 3 at most `2^64 - 3` and `lhs` is any four-limb value.
/// This is debug-asserted, and under that precondition the machine-checked proofs in `lean/`
/// establish the result (`mul_with_spec` and the backend's `montgomeryBlocks_spec`, from
/// `mulMont_spec_of_lhs_lt` and `mulMont_spec_of_rhs_lt`).
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
#[inline(always)]
pub fn mul(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    mul_with::<Selected>(lhs, rhs, modulus, inv)
}

/// Squares a canonical Montgomery residue for a Pasta modulus.
///
/// Outputs are canonical.
///
/// # Safety
///
/// The input of `square` must be canonical. This is debug-asserted, and under that precondition the
/// machine-checked proofs in `lean/` establish the result (`square_with_spec` and the backend's
/// `montgomeryBlocks_spec`).
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
// On x86-64 the backend always inlines the squaring blocks, which take nearly every register.
// So the entry point keeps them behind a call boundary, for the reason given on the
// multiplication block in `x86_64.rs`.
#[cfg_attr(target_arch = "aarch64", inline(always))]
#[cfg_attr(target_arch = "x86_64", inline(never))]
pub fn square(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    square_with::<Selected>(value, modulus, inv)
}

/// Squares a canonical Montgomery residue `count` times, then multiplies the
/// result by `rhs`.
///
/// A `count` of zero is just the multiplication. The squarings keep the value canonical, so the
/// multiplication is under its contract with a canonical `lhs`, and any four-limb `rhs` is
/// accepted. The accumulator stays in registers across the squarings: on AArch64 each step is
/// an inline block that the compiler inlines, and on x86-64 the squaring blocks are always
/// inlined into one loop, with the multiplication called once at the end.
///
/// # Safety
///
/// `value` must be canonical. This is debug-asserted, and under that precondition the
/// machine-checked proofs in `lean/` establish the result (`sqr_n_mul_with_spec` and the backend's
/// `montgomeryBlocks_spec`).
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
// On x86-64 the loop is kept behind a call boundary, as `square` is.
#[cfg_attr(target_arch = "aarch64", inline)]
#[cfg_attr(target_arch = "x86_64", inline(never))]
pub fn sqr_n_mul(value: &Limbs, count: usize, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    sqr_n_mul_with::<Selected>(value, count, rhs, modulus, inv)
}

/// Converts a Montgomery residue into its canonical integer, `value * 2^-256 mod p`: a
/// Montgomery multiplication by one.
///
/// Any four-limb `value` is accepted, and the machine-checked proofs in `lean/` establish the
/// result (`from_mont_with_spec` and the backend's `montgomeryBlocks_spec`). On AArch64 the
/// conversion is the multiplication block with `1` as its right operand, which is canonical with
/// limbs 1 to 3 zero and so inside the multiplication's contract for any left operand. On x86-64 it
/// is a dedicated block.
///
/// # Safety
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
#[inline]
pub fn from_mont(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    from_mont_with::<Selected>(value, modulus, inv)
}
