//! The module's entry points, re-exported at its root. Each runs a generic composition, the
//! matching `_with` function, over the blocks of the target's backend, which implements
//! [`MontgomeryBlocks`].

use core::hint::black_box;

pub use crate::limbs::Limbs;
pub(crate) use crate::limbs::is_canonical;
use crate::limbs::is_canonical_word;

/// The condition that `mul` asserts: a canonical `lhs`, or a canonical `rhs` whose limbs 1 to 3
/// are at most `2^64 - 3`, combined as opaque words (see [`is_canonical_word`]) so that the
/// check runs the same instructions whatever the operands.
#[inline(always)]
pub(crate) fn mul_contract(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> bool {
    let rhs_limbs_ok = black_box(u64::from(rhs[1] <= u64::MAX - 2))
        & black_box(u64::from(rhs[2] <= u64::MAX - 2))
        & black_box(u64::from(rhs[3] <= u64::MAX - 2));
    (is_canonical_word(lhs, modulus) | (is_canonical_word(rhs, modulus) & rhs_limbs_ok)) == 1
}

/// The Montgomery arithmetic of a backend: the blocks that the entry points run. Each method
/// has the contract of the entry point of the same name, which states it and debug-asserts it.
/// As with `crate::inversion::InvertBlocks`, one generic composition runs over any backend's
/// blocks.
pub(crate) trait MontgomeryBlocks {
    /// Adds two canonical residues and conditionally subtracts the modulus, as [`add`].
    fn add(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs;

    /// Subtracts two canonical residues, adding the modulus back on underflow, as [`sub`].
    fn sub(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs;

    /// Multiplies two Montgomery residues under either of the contracts of [`mul`].
    fn mul(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs;

    /// Squares a canonical Montgomery residue, as [`square`].
    fn square(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs;

    /// Converts any four-limb Montgomery residue into its canonical integer, as [`from_mont`].
    fn from_mont(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs;
}

/// The blocks that the entry points run: the backend of the target architecture.
#[cfg(target_arch = "aarch64")]
type Selected = super::aarch64::Backend;
/// The blocks that the entry points run: the backend of the target architecture.
#[cfg(target_arch = "x86_64")]
type Selected = super::x86_64::Backend;

/// [`add`] over the blocks `B`, with its debug assertions.
#[inline(always)]
pub(crate) fn add_with<B: MontgomeryBlocks>(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
    debug_assert!(
        is_canonical(lhs, modulus),
        "pasta_curves::asm::add requires a canonical lhs"
    );
    debug_assert!(
        is_canonical(rhs, modulus),
        "pasta_curves::asm::add requires a canonical rhs"
    );
    B::add(lhs, rhs, modulus)
}

/// [`sub`] over the blocks `B`, with its debug assertions.
#[inline(always)]
pub(crate) fn sub_with<B: MontgomeryBlocks>(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
    debug_assert!(
        is_canonical(lhs, modulus),
        "pasta_curves::asm::sub requires a canonical lhs"
    );
    debug_assert!(
        is_canonical(rhs, modulus),
        "pasta_curves::asm::sub requires a canonical rhs"
    );
    B::sub(lhs, rhs, modulus)
}

/// [`mul`] over the blocks `B`, with its debug assertion.
#[inline(always)]
pub(crate) fn mul_with<B: MontgomeryBlocks>(
    lhs: &Limbs,
    rhs: &Limbs,
    modulus: &Limbs,
    inv: u64,
) -> Limbs {
    debug_assert!(
        mul_contract(lhs, rhs, modulus),
        "pasta_curves::asm::mul requires a canonical lhs, or a canonical rhs with limbs 1 to 3 \
         at most 2^64 - 3"
    );
    B::mul(lhs, rhs, modulus, inv)
}

/// [`square`] over the blocks `B`, with its debug assertion.
#[inline(always)]
pub(crate) fn square_with<B: MontgomeryBlocks>(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    debug_assert!(
        is_canonical(value, modulus),
        "pasta_curves::asm::square requires a canonical input"
    );
    B::square(value, modulus, inv)
}

/// [`sqr_n_mul`] over the blocks `B`: the input's debug assertion, then [`square_with`] `count`
/// times and [`mul_with`], each with its own.
#[inline(always)]
pub(crate) fn sqr_n_mul_with<B: MontgomeryBlocks>(
    value: &Limbs,
    count: usize,
    rhs: &Limbs,
    modulus: &Limbs,
    inv: u64,
) -> Limbs {
    debug_assert!(
        is_canonical(value, modulus),
        "pasta_curves::asm::sqr_n_mul requires a canonical value"
    );
    let mut acc = *value;
    for _ in 0..count {
        acc = square_with::<B>(&acc, modulus, inv);
    }
    mul_with::<B>(&acc, rhs, modulus, inv)
}

/// [`from_mont`] over the blocks `B`. It asserts nothing, since it accepts every input.
#[inline(always)]
pub(crate) fn from_mont_with<B: MontgomeryBlocks>(
    value: &Limbs,
    modulus: &Limbs,
    inv: u64,
) -> Limbs {
    B::from_mont(value, modulus, inv)
}

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
