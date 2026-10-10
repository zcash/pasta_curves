//! The Montgomery arithmetic in portable Rust. Where there is no assembly backend, on a target
//! without one or with the assembly disabled, it provides the blocks of `MontgomeryBlocks`. On
//! every target, its `const fn`s are what the field types' own `const` arithmetic runs.
//!
//! Each function takes the modulus, and where it reduces, the Montgomery constant `inv`, as
//! arguments. The field types pass their own constants, and every function is inlined, so the
//! constants fold as they would if the field types wrote them directly.

use crate::arithmetic::{adc, mac, sbb};
use crate::limbs::Limbs;

// The blocks, and the conversion that only they use, where they are the field types' arithmetic.
if_asm_unsupported! {
    use super::MontgomeryBlocks;

    /// The portable blocks, for the generic compositions of `crate::montgomery`.
    pub(crate) struct Backend;

    impl MontgomeryBlocks for Backend {
        #[inline(always)]
        fn add(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
            add(lhs, rhs, modulus)
        }

        #[inline(always)]
        fn sub(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
            sub(lhs, rhs, modulus)
        }

        #[inline(always)]
        fn mul(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
            mul(lhs, rhs, modulus, inv)
        }

        #[inline(always)]
        fn square(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
            square(value, modulus, inv)
        }

        #[inline(always)]
        fn from_mont(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
            from_mont(value, modulus, inv)
        }
    }

    /// The canonical integer `value / 2^256 mod p` of a Montgomery residue.
    #[cfg_attr(not(feature = "uninline-portable"), inline(always))]
    pub(crate) const fn from_mont(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
        montgomery_reduce(
            value[0], value[1], value[2], value[3], 0, 0, 0, 0, modulus, inv,
        )
    }
}

/// `lhs + rhs` with the modulus subtracted when that does not borrow: the sum of two canonical
/// residues, reduced.
#[cfg_attr(not(feature = "uninline-portable"), inline(always))]
pub(crate) const fn add(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
    let (d0, carry) = adc(lhs[0], rhs[0], 0);
    let (d1, carry) = adc(lhs[1], rhs[1], carry);
    let (d2, carry) = adc(lhs[2], rhs[2], carry);
    let (d3, _) = adc(lhs[3], rhs[3], carry);

    // Attempt to subtract the modulus, to ensure the value
    // is smaller than the modulus.
    sub(&[d0, d1, d2, d3], modulus, modulus)
}

/// `lhs - rhs`, with the modulus added back when the subtraction borrows.
#[cfg_attr(not(feature = "uninline-portable"), inline(always))]
pub(crate) const fn sub(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
    let (d0, borrow) = sbb(lhs[0], rhs[0], 0);
    let (d1, borrow) = sbb(lhs[1], rhs[1], borrow);
    let (d2, borrow) = sbb(lhs[2], rhs[2], borrow);
    let (d3, borrow) = sbb(lhs[3], rhs[3], borrow);

    // If underflow occurred on the final limb, borrow = 0xfff...fff, otherwise
    // borrow = 0x000...000. Thus, we use it as a mask to conditionally add the modulus.
    let (d0, carry) = adc(d0, modulus[0] & borrow, 0);
    let (d1, carry) = adc(d1, modulus[1] & borrow, carry);
    let (d2, carry) = adc(d2, modulus[2] & borrow, carry);
    let (d3, _) = adc(d3, modulus[3] & borrow, carry);

    [d0, d1, d2, d3]
}

/// The Montgomery product `lhs * rhs / 2^256 mod p`.
#[cfg_attr(not(feature = "uninline-portable"), inline(always))]
pub(crate) const fn mul(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    let u = mul_unreduced(lhs, rhs);
    montgomery_reduce(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], modulus, inv)
}

/// The Montgomery square `value^2 / 2^256 mod p`.
#[cfg_attr(not(feature = "uninline-portable"), inline(always))]
pub(crate) const fn square(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    let u = square_unreduced(value);
    montgomery_reduce(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], modulus, inv)
}

/// The Montgomery reduction of the eight-limb value `r`, with one final conditional subtraction
/// of the modulus.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "uninline-portable"), inline(always))]
pub(crate) const fn montgomery_reduce(
    r0: u64,
    r1: u64,
    r2: u64,
    r3: u64,
    r4: u64,
    r5: u64,
    r6: u64,
    r7: u64,
    modulus: &Limbs,
    inv: u64,
) -> Limbs {
    // The Montgomery reduction here is based on Algorithm 14.32 in
    // Handbook of Applied Cryptography
    // <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.

    let k = r0.wrapping_mul(inv);
    let (_, carry) = mac(r0, k, modulus[0], 0);
    let (r1, carry) = mac(r1, k, modulus[1], carry);
    let (r2, carry) = mac(r2, k, modulus[2], carry);
    let (r3, carry) = mac(r3, k, modulus[3], carry);
    let (r4, carry2) = adc(r4, 0, carry);

    let k = r1.wrapping_mul(inv);
    let (_, carry) = mac(r1, k, modulus[0], 0);
    let (r2, carry) = mac(r2, k, modulus[1], carry);
    let (r3, carry) = mac(r3, k, modulus[2], carry);
    let (r4, carry) = mac(r4, k, modulus[3], carry);
    let (r5, carry2) = adc(r5, carry2, carry);

    let k = r2.wrapping_mul(inv);
    let (_, carry) = mac(r2, k, modulus[0], 0);
    let (r3, carry) = mac(r3, k, modulus[1], carry);
    let (r4, carry) = mac(r4, k, modulus[2], carry);
    let (r5, carry) = mac(r5, k, modulus[3], carry);
    let (r6, carry2) = adc(r6, carry2, carry);

    let k = r3.wrapping_mul(inv);
    let (_, carry) = mac(r3, k, modulus[0], 0);
    let (r4, carry) = mac(r4, k, modulus[1], carry);
    let (r5, carry) = mac(r5, k, modulus[2], carry);
    let (r6, carry) = mac(r6, k, modulus[3], carry);
    let (r7, _) = adc(r7, carry2, carry);

    // Result may be within MODULUS of the correct value
    sub(&[r4, r5, r6, r7], modulus, modulus)
}

/// `lhs * rhs` as the unreduced eight-limb product.
#[cfg_attr(not(feature = "uninline-portable"), inline)]
pub(crate) const fn mul_unreduced(lhs: &Limbs, rhs: &Limbs) -> [u64; 8] {
    // Schoolbook multiplication

    let (r0, carry) = mac(0, lhs[0], rhs[0], 0);
    let (r1, carry) = mac(0, lhs[0], rhs[1], carry);
    let (r2, carry) = mac(0, lhs[0], rhs[2], carry);
    let (r3, r4) = mac(0, lhs[0], rhs[3], carry);

    let (r1, carry) = mac(r1, lhs[1], rhs[0], 0);
    let (r2, carry) = mac(r2, lhs[1], rhs[1], carry);
    let (r3, carry) = mac(r3, lhs[1], rhs[2], carry);
    let (r4, r5) = mac(r4, lhs[1], rhs[3], carry);

    let (r2, carry) = mac(r2, lhs[2], rhs[0], 0);
    let (r3, carry) = mac(r3, lhs[2], rhs[1], carry);
    let (r4, carry) = mac(r4, lhs[2], rhs[2], carry);
    let (r5, r6) = mac(r5, lhs[2], rhs[3], carry);

    let (r3, carry) = mac(r3, lhs[3], rhs[0], 0);
    let (r4, carry) = mac(r4, lhs[3], rhs[1], carry);
    let (r5, carry) = mac(r5, lhs[3], rhs[2], carry);
    let (r6, r7) = mac(r6, lhs[3], rhs[3], carry);

    [r0, r1, r2, r3, r4, r5, r6, r7]
}

/// `value^2` as the unreduced eight-limb product.
#[cfg_attr(not(feature = "uninline-portable"), inline)]
pub(crate) const fn square_unreduced(value: &Limbs) -> [u64; 8] {
    let (r1, carry) = mac(0, value[0], value[1], 0);
    let (r2, carry) = mac(0, value[0], value[2], carry);
    let (r3, r4) = mac(0, value[0], value[3], carry);

    let (r3, carry) = mac(r3, value[1], value[2], 0);
    let (r4, r5) = mac(r4, value[1], value[3], carry);

    let (r5, r6) = mac(r5, value[2], value[3], 0);

    let r7 = r6 >> 63;
    let r6 = (r6 << 1) | (r5 >> 63);
    let r5 = (r5 << 1) | (r4 >> 63);
    let r4 = (r4 << 1) | (r3 >> 63);
    let r3 = (r3 << 1) | (r2 >> 63);
    let r2 = (r2 << 1) | (r1 >> 63);
    let r1 = r1 << 1;

    let (r0, carry) = mac(0, value[0], value[0], 0);
    let (r1, carry) = adc(0, r1, carry);
    let (r2, carry) = mac(r2, value[1], value[1], carry);
    let (r3, carry) = adc(0, r3, carry);
    let (r4, carry) = mac(r4, value[2], value[2], carry);
    let (r5, carry) = adc(0, r5, carry);
    let (r6, carry) = mac(r6, value[3], value[3], carry);
    let (r7, _) = adc(0, r7, carry);

    [r0, r1, r2, r3, r4, r5, r6, r7]
}
