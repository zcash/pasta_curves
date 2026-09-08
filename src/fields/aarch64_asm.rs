//! Private bindings to the Apple AArch64 Pasta field backend.
//!
//! # Operand range
//!
//! `mul` needs a canonical `rhs`; `square` and `from_mont` need a canonical
//! `value`. `lhs` in `mul` may be an unreduced 256-bit value only if every
//! `rhs` limb is at most `2^64 - 4`. With both operands canonical the
//! routines are always safe.
//!
//! The `lhs` caveat exists because `mul` keeps a five-limb accumulator rather
//! than the textbook six, so the chain folding in the high cross-products can
//! wrap. That needs `lhs[3]` and some `rhs` limb both within 3 of `2^64`, so
//! a canonical `lhs`, which has `lhs[3] <= 2^62`, rules it out on its own, as
//! does the limb cap. Outside the range the backend is wrong rather than
//! merely non-canonical: with `M = 2^64 - 1`, `lhs = [0, M, M, M]` against the
//! canonical `rhs = [M, M, 0, 0]` returns a value `2^128` below the true
//! residue, where `montgomery_reduce` returns it.
//!
//! `from_u512` is the only caller that passes an unreduced `lhs`, always
//! against `R2` or `R3`. Every limb of those is far under the cap, which
//! `aarch64_asm_vectors_respect_the_documented_operand_range` checks rather
//! than leaves to inspection.

type Limbs = [u64; 4];

extern "C" {
    fn pasta_curves_mul_mont_pasta(
        out: *mut Limbs,
        lhs: *const Limbs,
        rhs: *const Limbs,
        modulus: *const Limbs,
        inv: u64,
    );
    fn pasta_curves_sqr_mont_pasta(
        out: *mut Limbs,
        value: *const Limbs,
        modulus: *const Limbs,
        inv: u64,
    );
    fn pasta_curves_from_mont_pasta(
        out: *mut Limbs,
        value: *const Limbs,
        modulus: *const Limbs,
        inv: u64,
    );
}

/// Multiplies two canonical Montgomery residues for a Pasta modulus.
#[inline]
pub(super) fn mul(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    let mut out = Limbs::default();
    // SAFETY: All pointers refer to four initialized `u64` limbs for the
    // duration of the call. The backend writes exactly four limbs to `out`.
    unsafe {
        pasta_curves_mul_mont_pasta(&mut out, lhs, rhs, modulus, inv);
    }
    out
}

/// Converts a canonical Montgomery residue into its canonical integer.
#[inline]
pub(super) fn from_mont(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    let mut out = Limbs::default();
    // SAFETY: All pointers refer to four initialized `u64` limbs for the
    // duration of the call. The backend writes exactly four limbs to `out`.
    unsafe {
        pasta_curves_from_mont_pasta(&mut out, value, modulus, inv);
    }
    out
}

/// Squares a canonical Montgomery residue for a Pasta modulus.
#[inline]
pub(super) fn square(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    let mut out = Limbs::default();
    // SAFETY: All pointers refer to four initialized `u64` limbs for the
    // duration of the call. The backend writes exactly four limbs to `out`.
    unsafe {
        pasta_curves_sqr_mont_pasta(&mut out, value, modulus, inv);
    }
    out
}
