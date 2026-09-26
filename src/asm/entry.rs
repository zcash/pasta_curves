// Copyright the pasta-asm contributors.
// SPDX-License-Identifier: Apache-2.0

//! The module's entry points, re-exported at its root.

/// Four little-endian 64-bit limbs, least significant first: a field element
/// (in Montgomery form, or canonical after [`from_mont`]) or a modulus.
pub type Limbs = [u64; 4];

/// Whether `value < modulus` as little-endian 256-bit integers.
#[inline(always)]
pub(crate) fn is_canonical(value: &Limbs, modulus: &Limbs) -> bool {
    for i in (0..4).rev() {
        if value[i] != modulus[i] {
            return value[i] < modulus[i];
        }
    }
    false
}

/// Adds two residues for a Pasta modulus and conditionally subtracts the modulus.
///
/// Outputs are canonical.
///
/// # Safety
///
/// Both inputs must be canonical. This is debug-asserted, and under it the machine-checked
/// proofs in `lean/` establish the result (`add_entry_spec`).
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
/// Both inputs must be canonical. This is debug-asserted, and under it the machine-checked
/// proofs in `lean/` establish the result (`sub_entry_spec`).
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
/// four-limb value. This is debug-asserted, and under it the machine-checked proofs in
/// `lean/` establish the result (`mul_entry_spec`, from `mulMont_spec_of_lhs_lt` and
/// `mulMont_spec_of_rhs_lt`).
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
#[inline(always)]
pub fn mul(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    debug_assert!(
        is_canonical(lhs, modulus)
            || (is_canonical(rhs, modulus) && rhs[1..].iter().all(|&limb| limb <= u64::MAX - 2)),
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
/// The input of `square` must be canonical. This is debug-asserted, and under it the
/// machine-checked proofs in `lean/` establish the result (`square_entry_spec`).
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
/// result by the canonical Montgomery residue `rhs`.
///
/// A `count` of zero is just the multiplication. For a canonical `value`, the machine-checked
/// proofs in `lean/` establish the result (`sqrNMul_entry_spec`).
///
/// Each step is one of the inline blocks, which the compiler inlines, so the accumulator
/// stays in registers throughout.
///
/// # Safety
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
#[inline]
pub fn sqr_n_mul(value: &Limbs, count: usize, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    // On aarch64, `square` and `mul` can be inlined and optimised by Rust.
    #[cfg(target_arch = "aarch64")]
    {
        let mut acc = *value;
        for _ in 0..count {
            acc = square(&acc, modulus, inv);
        }
        mul(&acc, rhs, modulus, inv)
    }

    // On x86_64, `square` and `mul` can't be inlined due to register pressure, so we need
    // a separate fused assembly implementation.
    #[cfg(target_arch = "x86_64")]
    {
        super::x86_64::sqr_n_mul(value, count, rhs, modulus, inv)
    }
}

/// Converts a Montgomery residue into its canonical integer, `value * 2^-256 mod p`, as a
/// Montgomery multiplication by one.
///
/// Any four-limb `value` is accepted: `1` is canonical with limbs 1 to 3 zero, so it is a
/// right operand inside the multiplication's contract for any left operand; the
/// machine-checked proofs in `lean/` establish the result (`fromMont_entry_spec`).
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
