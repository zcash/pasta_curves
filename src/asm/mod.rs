// Copyright the pasta-asm contributors.
// SPDX-License-Identifier: Apache-2.0

// The crate denies unsafe code by default; the assembly backend is the one place that allows
// it, and its safety argument is the operand contracts stated on each routine.
#![allow(unsafe_code)]
// The routines are not yet reached from the field types.
#![allow(dead_code)]

//! Assembly backends for the Pasta fields.
//!
//! # Availability
//!
//! The module provides a backend for `target_arch = "aarch64"` and, in part,
//! for `target_arch = "x86_64"`: `add`, `sub`, and `from_mont` are
//! register-only and work on every x86-64 target, while `mul`, `square`, and
//! the routines built on them read limbs through pointers, so they require
//! 64-bit pointers (the x32 ABI's 32-bit pointers would break them; see the
//! x86-64 module's docs for why registers alone cannot serve there) and a CPU
//! with BMI2 and ADX (MULX, ADCX/ADOX: Intel Broadwell / AMD Zen or newer) at
//! run time — neither is checked. `from_mont` uses MULX (BMI2) alone. Elsewhere
//! the `asm` module is absent. Nothing is assembled at build time: the blocks
//! are compiled by the Rust toolchain, so no C toolchain is needed, and the
//! module adds no dependency.
//!
//! # Provenance
//!
//! The routines are transcriptions of the Pasta Montgomery routines of
//! Supranational's [Semolina] v0.1.4. See `src/asm/README.md` for the history
//! of the transcription.
//!
//! [Semolina]: https://github.com/supranational/semolina

#[cfg(any(target_arch = "aarch64", doc))]
mod aarch64;

#[cfg(any(target_arch = "x86_64", doc))]
mod x86_64;

// The tests use std only to catch the debug assertions they check, so a release test build
// stays free of it.
#[cfg(all(test, debug_assertions))]
extern crate std;

#[cfg(test)]
mod tests;

/// Four little-endian 64-bit limbs, least significant first: a field element
/// (in Montgomery form, or canonical after [`from_mont`]) or a modulus.
pub type Limbs = [u64; 4];

/// Whether `value < modulus` as little-endian 256-bit integers.
#[inline(always)]
fn is_canonical(value: &Limbs, modulus: &Limbs) -> bool {
    for i in (0..4).rev() {
        if value[i] != modulus[i] {
            return value[i] < modulus[i];
        }
    }
    false
}

/// Adds two residues for a Pasta modulus and conditionally subtracts the modulus.
///
/// # Safety
///
/// Both inputs must be canonical; this is debug-asserted.
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
        aarch64::add(lhs, rhs, modulus)
    }

    #[cfg(target_arch = "x86_64")]
    {
        x86_64::add(lhs, rhs, modulus)
    }
}

/// Subtracts two residues for a Pasta modulus, adding the modulus back on underflow.
///
/// # Safety
///
/// Both inputs must be canonical; this is debug-asserted.
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
        aarch64::sub(lhs, rhs, modulus)
    }

    #[cfg(target_arch = "x86_64")]
    {
        x86_64::sub(lhs, rhs, modulus)
    }
}

/// Multiplies two Montgomery residues for a Pasta modulus.
///
/// # Safety
///
/// Either `lhs` is canonical (below the modulus) and `rhs` is any four-limb value, or
/// `rhs` is canonical with each of its limbs 1 to 3 at most `2^64 - 3` and `lhs` is any
/// four-limb value. This is the contract that the machine-checked proofs in `lean/`
/// establish (`mulMont_spec_of_lhs_lt` and `mulMont_spec_of_rhs_lt`), and is
/// debug-asserted.
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_pointer_width = "64")
))]
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
        aarch64::mul(lhs, rhs, modulus, inv)
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    {
        x86_64::mul(lhs, rhs, modulus, inv)
    }
}

/// Squares a canonical Montgomery residue for a Pasta modulus.
///
/// Outputs are canonical.
///
/// # Safety
///
/// The input of `square` must be canonical, a contract that the proofs do not cover; this
/// is debug-asserted.
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_pointer_width = "64")
))]
#[inline(always)]
pub fn square(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    debug_assert!(
        is_canonical(value, modulus),
        "pasta_curves::asm::square requires a canonical input"
    );

    #[cfg(target_arch = "aarch64")]
    {
        aarch64::square(value, modulus, inv)
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    {
        x86_64::square(value, modulus, inv)
    }
}

/// Squares a canonical Montgomery residue `count` times, then multiplies the
/// result by the canonical Montgomery residue `rhs`.
///
/// A `count` of zero is just the multiplication.
///
/// Each step is one of the inline blocks, which the compiler inlines, so the accumulator
/// stays in registers throughout.
///
/// # Safety
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_pointer_width = "64")
))]
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
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    {
        x86_64::sqr_n_mul(value, count, rhs, modulus, inv)
    }
}

/// Converts a Montgomery residue into its canonical integer, `value * 2^-256 mod p`, as a
/// Montgomery multiplication by one.
///
/// Any four-limb `value` is accepted: `1` is canonical with limbs 1 to 3 zero, so it is a
/// right operand inside the multiplication's contract for any left operand
/// (`mulMont_spec_of_rhs_lt` in `lean/`).
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
        x86_64::from_mont(value, modulus, inv)
    }
}
