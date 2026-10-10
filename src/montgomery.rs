//! The Montgomery arithmetic of the fields: one generic composition for each entry point, over the
//! blocks of a backend, which implements [`MontgomeryBlocks`]. Each `_with` function runs its entry
//! point's composition, with that entry point's debug assertions, over any backend's blocks. The
//! assembly backends' entry points (`crate::asm`) run them over the blocks of the target. Where
//! there is no assembly backend, the field types run them over the portable blocks (`portable`).
//!
//! Aeneas translates the compositions (`lean/PastaCurves/Glue/`), and their proofs hold over any
//! backend whose blocks meet their contracts.

use core::hint::black_box;

use crate::limbs::{Limbs, is_canonical, is_canonical_word};

pub(crate) mod portable;

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
    /// Adds two canonical residues and conditionally subtracts the modulus, as `add`.
    fn add(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs;

    /// Subtracts two canonical residues, adding the modulus back on underflow, as `sub`.
    fn sub(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs;

    /// Multiplies two Montgomery residues under either of the contracts of `mul`.
    fn mul(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs;

    /// Squares a canonical Montgomery residue, as `square`.
    fn square(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs;

    /// Converts any four-limb Montgomery residue into its canonical integer, as `from_mont`.
    fn from_mont(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs;
}

/// `add` over the blocks `B`, with its debug assertions.
#[inline(always)]
pub(crate) fn add_with<B: MontgomeryBlocks>(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
    debug_assert!(
        is_canonical(lhs, modulus),
        "pasta_curves::montgomery::add_with requires a canonical lhs"
    );
    debug_assert!(
        is_canonical(rhs, modulus),
        "pasta_curves::montgomery::add_with requires a canonical rhs"
    );
    B::add(lhs, rhs, modulus)
}

/// `sub` over the blocks `B`, with its debug assertions.
#[inline(always)]
pub(crate) fn sub_with<B: MontgomeryBlocks>(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
    debug_assert!(
        is_canonical(lhs, modulus),
        "pasta_curves::montgomery::sub_with requires a canonical lhs"
    );
    debug_assert!(
        is_canonical(rhs, modulus),
        "pasta_curves::montgomery::sub_with requires a canonical rhs"
    );
    B::sub(lhs, rhs, modulus)
}

/// `mul` over the blocks `B`, with its debug assertion.
#[inline(always)]
pub(crate) fn mul_with<B: MontgomeryBlocks>(
    lhs: &Limbs,
    rhs: &Limbs,
    modulus: &Limbs,
    inv: u64,
) -> Limbs {
    debug_assert!(
        mul_contract(lhs, rhs, modulus),
        "pasta_curves::montgomery::mul_with requires a canonical lhs, or a canonical rhs with \
         limbs 1 to 3 at most 2^64 - 3"
    );
    B::mul(lhs, rhs, modulus, inv)
}

/// `square` over the blocks `B`, with its debug assertion.
#[inline(always)]
pub(crate) fn square_with<B: MontgomeryBlocks>(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    debug_assert!(
        is_canonical(value, modulus),
        "pasta_curves::montgomery::square_with requires a canonical input"
    );
    B::square(value, modulus, inv)
}

/// `sqr_n_mul` over the blocks `B`: the input's debug assertion, then [`square_with`] `count`
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
        "pasta_curves::montgomery::sqr_n_mul_with requires a canonical value"
    );
    let mut acc = *value;
    for _ in 0..count {
        acc = square_with::<B>(&acc, modulus, inv);
    }
    mul_with::<B>(&acc, rhs, modulus, inv)
}

/// `from_mont` over the blocks `B`. It asserts nothing, since it accepts every input.
#[inline(always)]
pub(crate) fn from_mont_with<B: MontgomeryBlocks>(
    value: &Limbs,
    modulus: &Limbs,
    inv: u64,
) -> Limbs {
    B::from_mont(value, modulus, inv)
}
