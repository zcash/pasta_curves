//! The module's entry points, re-exported at its root.

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

/// Adds two residues for a Pasta modulus and conditionally subtracts the modulus.
///
/// Outputs are canonical.
///
/// # Safety
///
/// Both inputs must be canonical. This is debug-asserted, and under that precondition the
/// machine-checked proofs in `lean/` establish the result (`add_entry_spec`).
///
/// `modulus` must be either the Pallas or Vesta field modulus. Any other values will
/// cause undefined results.
#[inline(always)]
pub fn add(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
    debug_assert!(
        is_canonical(lhs, modulus),
        "pasta_curves::asm::add requires a canonical lhs"
    );
    debug_assert!(
        is_canonical(rhs, modulus),
        "pasta_curves::asm::add requires a canonical rhs"
    );

    #[cfg(target_arch = "aarch64")]
    {
        super::aarch64::add(lhs, rhs, modulus)
    }

    #[cfg(target_arch = "x86_64")]
    {
        super::x86_64::add(lhs, rhs, modulus)
    }
}

/// Subtracts two residues for a Pasta modulus, adding the modulus back on underflow.
///
/// Outputs are canonical.
///
/// # Safety
///
/// Both inputs must be canonical. This is debug-asserted, and under that precondition the
/// machine-checked proofs in `lean/` establish the result (`sub_entry_spec`).
///
/// `modulus` must be either the Pallas or Vesta field modulus. Any other values will
/// cause undefined results.
#[inline(always)]
pub fn sub(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
    debug_assert!(
        is_canonical(lhs, modulus),
        "pasta_curves::asm::sub requires a canonical lhs"
    );
    debug_assert!(
        is_canonical(rhs, modulus),
        "pasta_curves::asm::sub requires a canonical rhs"
    );

    #[cfg(target_arch = "aarch64")]
    {
        super::aarch64::sub(lhs, rhs, modulus)
    }

    #[cfg(target_arch = "x86_64")]
    {
        super::x86_64::sub(lhs, rhs, modulus)
    }
}

/// Multiplies two Montgomery residues for a Pasta modulus.
///
/// # Safety
///
/// Either `lhs` is canonical (below the modulus) and `rhs` is any four-limb value, or
/// `rhs` is canonical with each of its limbs 1 to 3 at most `2^64 - 3` and `lhs` is any
/// four-limb value. This is debug-asserted, and under that precondition the machine-checked
/// proofs in `lean/` establish the result (`mul_entry_spec`, from `mulMont_spec_of_lhs_lt`
/// and `mulMont_spec_of_rhs_lt`).
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
#[inline(always)]
pub fn mul(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    debug_assert!(
        mul_contract(lhs, rhs, modulus),
        "pasta_curves::asm::mul requires a canonical lhs, or a canonical rhs with limbs 1 to 3 \
         at most 2^64 - 3"
    );

    #[cfg(target_arch = "aarch64")]
    {
        super::aarch64::mul(lhs, rhs, modulus, inv)
    }

    #[cfg(target_arch = "x86_64")]
    {
        super::x86_64::mul(lhs, rhs, modulus, inv)
    }
}

/// Squares a canonical Montgomery residue for a Pasta modulus.
///
/// Outputs are canonical.
///
/// # Safety
///
/// The input of `square` must be canonical. This is debug-asserted, and under that
/// precondition the machine-checked proofs in `lean/` establish the result
/// (`square_entry_spec`).
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
#[inline(always)]
pub fn square(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    debug_assert!(
        is_canonical(value, modulus),
        "pasta_curves::asm::square requires a canonical input"
    );

    #[cfg(target_arch = "aarch64")]
    {
        super::aarch64::square(value, modulus, inv)
    }

    #[cfg(target_arch = "x86_64")]
    {
        super::x86_64::square(value, modulus, inv)
    }
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
/// machine-checked proofs in `lean/` establish the result (`sqrNMul_entry_spec`).
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
#[inline]
pub fn sqr_n_mul(value: &Limbs, count: usize, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    debug_assert!(
        is_canonical(value, modulus),
        "pasta_curves::asm::sqr_n_mul requires a canonical value"
    );

    // On aarch64, `square` and `mul` can be inlined and optimised by Rust.
    #[cfg(target_arch = "aarch64")]
    {
        let mut acc = *value;
        for _ in 0..count {
            acc = square(&acc, modulus, inv);
        }
        mul(&acc, rhs, modulus, inv)
    }

    // On x86_64, `square` and `mul` can't be inlined due to register pressure, so the backend
    // has its own loop over the always-inlined squaring blocks.
    #[cfg(target_arch = "x86_64")]
    {
        super::x86_64::sqr_n_mul(value, count, rhs, modulus, inv)
    }
}

/// Converts a Montgomery residue into its canonical integer, `value * 2^-256 mod p`: a
/// Montgomery multiplication by one.
///
/// Any four-limb `value` is accepted, and the machine-checked proofs in `lean/` establish the
/// result (`fromMont_entry_spec`). On AArch64 the conversion is the multiplication block with
/// `1` as its right operand, which is canonical with limbs 1 to 3 zero and so inside the
/// multiplication's contract for any left operand. On x86-64 it is a dedicated block.
///
/// # Safety
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
#[inline]
pub fn from_mont(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    // On aarch64, `mul` can be inlined and optimised by Rust.
    #[cfg(target_arch = "aarch64")]
    {
        mul(value, &[1, 0, 0, 0], modulus, inv)
    }

    // On x86_64, `mul` can't be inlined due to register pressure, so we use a dedicated
    // register-only assembly implementation instead.
    #[cfg(target_arch = "x86_64")]
    {
        super::x86_64::from_mont(value, modulus, inv)
    }
}
