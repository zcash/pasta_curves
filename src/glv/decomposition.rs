//! Scalar decomposition arithmetic shared by the GLV recodings and native multiplication.

use ff::PrimeField;
use group::Group;

use super::GlvConstants;
use crate::arithmetic::{mac, sbb};

/// The magnitude of a decomposition half is below `2^HALF_BITS`: the bound
/// [`decompose`] guarantees, on which the recodings of the halves rely.
#[cfg(any(feature = "glv", test))]
pub(crate) const HALF_BITS: usize = 127;

/// The halves [`decompose`] splits a scalar into: `k1` for $P$ and `k2` for
/// $\phi(P)$.
#[cfg(feature = "glv")]
pub(crate) const HALVES: usize = 2;

/// GLV split: `k = k1 + k2 * lambda (mod n)`, each half returned as
/// `(is_negative, magnitude)`.
///
/// Babai rounding bounds the halves by `|k1| <= (BASIS_1_A + BASIS_2_A) / 2 + 1`
/// and `|k2| <= (BASIS_1_B_ABS + BASIS_2_B) / 2 + 1`, where the `+ 1` covers
/// `ROUNDING_COEFF_1` and `ROUNDING_COEFF_2` being 384-bit approximations of the
/// exact rounding coefficients. On both
/// curves that is below `2^126.3` for `k1` and `2^126.8` for `k2`, so strictly
/// below `2^127` ([`HALF_BITS`]), the bound [`signed_halves`], [`super::wnaf_digits`]
/// and the MSM's Booth digits rely on; `crate::glv_eisenstein` relies on the
/// tighter one.
#[cfg(any(feature = "glv", test))]
pub(crate) fn decompose<C>(k: &C::Scalar) -> ((bool, u128), (bool, u128))
where
    C: Group + GlvConstants,
{
    let (k1, k2) = split_limbs::<C>(k);
    (signed_halves(k1), signed_halves(k2))
}

/// Constant-time GLV split into signed coefficients for native multiplication.
/// Both coefficients fit strictly within `(-2^127, 2^127)` for every canonical
/// scalar. The signed conversion has no scalar-dependent branching or checks.
#[inline]
pub(crate) fn decompose_signed<C>(k: &C::Scalar) -> (i128, i128)
where
    C: Group + GlvConstants,
{
    let (k1, k2) = split_limbs::<C>(k);
    // The proven bound makes the lower half of each two's-complement value
    // precisely its i128 representation; no conditional magnitude conversion.
    (low_u128(k1) as i128, low_u128(k2) as i128)
}

/// Split into two 256-bit two's-complement halves using the per-curve basis.
#[inline]
fn split_limbs<C>(k: &C::Scalar) -> ([u64; 4], [u64; 4])
where
    C: Group + GlvConstants,
{
    let kl = scalar_limbs(k);
    let c1 = round_mul_shift(&C::ROUNDING_COEFF_1, &kl);
    let c2 = round_mul_shift(&C::ROUNDING_COEFF_2, &kl);
    // k1 = k - c1*BASIS_1_A - c2*BASIS_2_A   (two's complement over 256 bits)
    let k1 = sub256(
        sub256(kl, mul_u128(c1, C::BASIS_1_A)),
        mul_u128(c2, C::BASIS_2_A),
    );
    // k2 = c1*BASIS_1_B_ABS - c2*BASIS_2_B   (v1.b < 0, v2.b > 0)
    let k2 = sub256(mul_u128(c1, C::BASIS_1_B_ABS), mul_u128(c2, C::BASIS_2_B));
    (k1, k2)
}

/// The four little-endian limbs of a Pasta scalar. (Pasta scalars have a
/// 32-byte little-endian representation; the four 8-byte reads cover it
/// exactly.)
pub(super) fn scalar_limbs<F>(k: &F) -> [u64; 4]
where
    F: PrimeField,
{
    let bytes = k.to_repr();
    let bytes: &[u8] = bytes.as_ref();
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        *limb = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().expect("8 bytes"));
    }
    limbs
}

/// `round((g * k) / 2^384)` for a 5-limb `g` and 4-limb `k` — the Babai
/// coefficient. Fits `u128` (at most ~128 bits by construction).
pub(super) fn round_mul_shift(g: &[u64; 5], k: &[u64; 4]) -> u128 {
    let mut prod = [0u64; 9];
    schoolbook_mul(g, k, &mut prod);
    // Bits >= 384 live in limbs 6..; round on bit 383 (top bit of limb 5).
    let round = prod[5] >> 63;
    (u128::from(prod[6]) | (u128::from(prod[7]) << 64)).wrapping_add(u128::from(round))
}

/// 256-bit product of two `u128`s, as little-endian limbs.
pub(super) fn mul_u128(a: u128, b: u128) -> [u64; 4] {
    let mut prod = [0u64; 4];
    schoolbook_mul(
        &[a as u64, (a >> 64) as u64],
        &[b as u64, (b >> 64) as u64],
        &mut prod,
    );
    prod
}

/// Schoolbook multiply of `a` by `b` into `prod`, which must be zeroed and
/// hold exactly `a.len() + b.len()` limbs. Constant-time: a fixed loop
/// structure with explicit carry propagation.
pub(super) fn schoolbook_mul(a: &[u64], b: &[u64], prod: &mut [u64]) {
    debug_assert_eq!(prod.len(), a.len() + b.len());
    for (i, &ai) in a.iter().enumerate() {
        let mut carry = 0u64;
        for (j, &bj) in b.iter().enumerate() {
            let (limb, c) = mac(prod[i + j], ai, bj, carry);
            prod[i + j] = limb;
            carry = c;
        }
        // First write to prod[i + b.len()] on each outer iteration.
        prod[i + b.len()] = carry;
    }
}

/// 256-bit wrapping subtraction (two's complement).
pub(super) fn sub256(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let (d0, borrow) = sbb(a[0], b[0], 0);
    let (d1, borrow) = sbb(a[1], b[1], borrow);
    let (d2, borrow) = sbb(a[2], b[2], borrow);
    let (d3, _) = sbb(a[3], b[3], borrow);
    [d0, d1, d2, d3]
}

/// Interprets a 256-bit two's-complement value as `(is_negative, magnitude)`,
/// taking the low 128 bits of the magnitude.
///
/// GLV decomposition guarantees `|x| < 2^127` for the values reached here
/// (asserted in debug builds, here and in [`super::wnaf_digits`], and checked by the
/// `decompose` tests), so the high limbs of the magnitude are always zero and
/// no information is lost.
#[cfg(any(feature = "glv", test))]
fn signed_halves(x: [u64; 4]) -> (bool, u128) {
    // Guard the truncation itself: the discarded limbs must be the sign
    // extension of bit 127. Values of 2^128 or more would otherwise be
    // silently truncated before `wnaf_digits`' magnitude assertion could
    // observe them.
    let ext = if x[1] >> 63 == 0 { 0 } else { u64::MAX };
    debug_assert!(
        x[2] == ext && x[3] == ext,
        "GLV half does not fit in 128 bits"
    );
    let low = low_u128(x);
    if x[3] >> 63 == 0 {
        (false, low)
    } else {
        // Two's-complement negation commutes with truncation to the low 128
        // bits, and the magnitude lives entirely there.
        (true, (!low).wrapping_add(1))
    }
}

#[inline]
fn low_u128(x: [u64; 4]) -> u128 {
    u128::from(x[0]) | (u128::from(x[1]) << 64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fp, Fq, pallas, vesta};
    use ff::{Field, PrimeField, WithSmallOrderMulGroup};
    use proptest::prelude::*;

    fn signed<F: PrimeField>(x: i128) -> F {
        let bits = x as u128;
        let mask = (bits >> 127).wrapping_neg();
        let magnitude = (bits ^ mask).wrapping_sub(mask);
        let positive = F::from_u128(magnitude);
        if x < 0 { -positive } else { positive }
    }

    fn check<C>(scalar: C::Scalar)
    where
        C: Group + GlvConstants,
        C::Scalar: PrimeField + WithSmallOrderMulGroup<3> + core::fmt::Debug,
    {
        let (k1, k2) = decompose_signed::<C>(&scalar);
        assert!(k1.unsigned_abs() < (1u128 << HALF_BITS));
        assert!(k2.unsigned_abs() < (1u128 << HALF_BITS));
        assert_eq!(
            signed::<C::Scalar>(k1) + signed::<C::Scalar>(k2) * C::Scalar::ZETA,
            scalar
        );

        let ((negative1, magnitude1), (negative2, magnitude2)) = decompose::<C>(&scalar);
        let reference1 = if negative1 {
            -(magnitude1 as i128)
        } else {
            magnitude1 as i128
        };
        let reference2 = if negative2 {
            -(magnitude2 as i128)
        } else {
            magnitude2 as i128
        };
        assert_eq!((k1, k2), (reference1, reference2));
    }

    fn edges<C>()
    where
        C: Group + GlvConstants,
        C::Scalar: PrimeField + WithSmallOrderMulGroup<3> + core::fmt::Debug,
    {
        check::<C>(C::Scalar::ZERO);
        check::<C>(C::Scalar::ONE);
        check::<C>(-C::Scalar::ONE);
        check::<C>(C::Scalar::ZETA);
        check::<C>(-C::Scalar::ZETA);
        check::<C>(-C::Scalar::from(2u64));
        for i in 0..256u64 {
            let k = C::Scalar::from(i).square().square() + C::Scalar::ZETA;
            check::<C>(k);
        }
    }

    #[test]
    fn canonical_edges() {
        edges::<vesta::Point>();
        edges::<pallas::Point>();
    }

    proptest! {
        #[test]
        fn full_width_fp(limbs in any::<[u64; 4]>()) {
            // Field arithmetic maps all random inputs to canonical scalars.
            let k = Fp::from(limbs[0]) + Fp::from(limbs[1]).square()
                + Fp::from(limbs[2]).square().square() + Fp::from(limbs[3]);
            check::<vesta::Point>(k);
        }

        #[test]
        fn full_width_fq(limbs in any::<[u64; 4]>()) {
            let k = Fq::from(limbs[0]) + Fq::from(limbs[1]).square()
                + Fq::from(limbs[2]).square().square() + Fq::from(limbs[3]);
            check::<pallas::Point>(k);
        }
    }

    #[test]
    fn signed_matches_sign_magnitude() {
        for i in 0..1024u64 {
            let p = (Fp::from(0x9e37_79b9_7f4a_7c15 + i).square() + Fp::from(i)).square();
            let q = (Fq::from(0x9e37_79b9_7f4a_7c15 + i).square() + Fq::from(i)).square();
            check::<vesta::Point>(p);
            check::<pallas::Point>(q);
        }
    }
}
