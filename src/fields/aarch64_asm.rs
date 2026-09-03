//! Private bindings to the Apple AArch64 Pasta field backend.

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
