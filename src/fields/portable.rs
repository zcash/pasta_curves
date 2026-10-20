//! Shared portable squaring for the Pasta fields.
//!
//! `Fp` and `Fq` have the same representation (four little-endian `u64`
//! limbs in Montgomery form with `R = 2^256`). Other arithmetic retains
//! each field's existing Montgomery reduction wrapper.

// Copyright (c) 2020-2021 The Electric Coin Company

use crate::arithmetic::sbb;
use crate::arithmetic::{adc, mac};

#[cfg(target_arch = "x86_64")]
const PASTA_MODULUS_TOP_SHIFT: u32 = 62;
#[cfg(target_arch = "x86_64")]
const PASTA_MODULUS_TOP_OVERFLOW_SHIFT: u32 = u64::BITS - PASTA_MODULUS_TOP_SHIFT;

/// Squares a canonical element, returning the canonical square.
#[cfg(target_arch = "x86_64")]
#[cfg_attr(not(feature = "uninline-portable"), inline)]
pub(super) const fn square(value: &[u64; 4], modulus: &[u64; 4], inv: u64) -> [u64; 4] {
    canonicalize(
        &montgomery_reduce_low_lazy(&square_wide(value), modulus, inv),
        modulus,
    )
}

/// Squares `value`, returning the unreduced 512-bit product.
#[cfg_attr(not(feature = "uninline-portable"), inline)]
pub(super) const fn square_wide(value: &[u64; 4]) -> [u64; 8] {
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

/// Montgomery-reduces a 512-bit value `t < R * modulus` by cancelling its low
/// half first, returning a result below `2 * modulus`.
///
/// The Montgomery quotient `Q` depends only on the low 256 bits of `t`, and
/// `(t + Q * modulus) / R == (t_lo + Q * modulus) / R + t_hi` exactly, so the
/// four cancellation rounds can run over four live limbs instead of eight and
/// the high half is added once at the end. This is how the assembly backend's
/// squaring and its `mul_by_1` helper are structured. The value produced is
/// the same as the classical reduction's, limb for limb; only the dependency
/// graph differs.
///
/// `(t_lo + Q * modulus) / R <= modulus` and `t_hi < modulus`, so the sum is
/// below `2 * modulus` and fits in four limbs for either Pasta modulus.
#[cfg_attr(not(feature = "uninline-portable"), inline(always))]
pub(super) const fn montgomery_reduce_low_lazy(
    t: &[u64; 8],
    modulus: &[u64; 4],
    inv: u64,
) -> [u64; 4] {
    debug_assert!(modulus[2] == 0);

    let [r0, r1, r2, r3, t4, t5, t6, t7] = *t;

    // Each round chooses k so that the lowest live limb plus k * modulus[0]
    // vanishes modulo 2^64, adds k * modulus, and drops that limb. The
    // carry out of the top limb becomes the new top limb. Both Pasta moduli
    // have a zero third limb, so that product is only a carry propagation.
    let k = r0.wrapping_mul(inv);
    let (_, carry) = mac(r0, k, modulus[0], 0);
    let (r0, carry) = mac(r1, k, modulus[1], carry);
    let (r1, carry) = adc(r2, 0, carry);
    let (r2, r3) = mac(r3, k, modulus[3], carry);

    let k = r0.wrapping_mul(inv);
    let (_, carry) = mac(r0, k, modulus[0], 0);
    let (r0, carry) = mac(r1, k, modulus[1], carry);
    let (r1, carry) = adc(r2, 0, carry);
    let (r2, r3) = mac(r3, k, modulus[3], carry);

    let k = r0.wrapping_mul(inv);
    let (_, carry) = mac(r0, k, modulus[0], 0);
    let (r0, carry) = mac(r1, k, modulus[1], carry);
    let (r1, carry) = adc(r2, 0, carry);
    let (r2, r3) = mac(r3, k, modulus[3], carry);

    let k = r0.wrapping_mul(inv);
    let (_, carry) = mac(r0, k, modulus[0], 0);
    let (r0, carry) = mac(r1, k, modulus[1], carry);
    let (r1, carry) = adc(r2, 0, carry);
    let (r2, r3) = mac(r3, k, modulus[3], carry);

    // Add the high half; the sum is below 2 * modulus, so no carry out.
    let (r0, carry) = adc(r0, t4, 0);
    let (r1, carry) = adc(r1, t5, carry);
    let (r2, carry) = adc(r2, t6, carry);
    let (r3, _) = adc(r3, t7, carry);

    [r0, r1, r2, r3]
}

/// Subtracts `modulus` from a value below `2 * modulus` if that does not
/// underflow, which makes the value canonical.
#[cfg_attr(not(feature = "uninline-portable"), inline(always))]
pub(super) const fn canonicalize(value: &[u64; 4], modulus: &[u64; 4]) -> [u64; 4] {
    let (d0, borrow) = sbb(value[0], modulus[0], 0);
    let (d1, borrow) = sbb(value[1], modulus[1], borrow);
    let (d2, borrow) = sbb(value[2], modulus[2], borrow);
    let (d3, borrow) = sbb(value[3], modulus[3], borrow);

    let (d0, carry) = adc(d0, modulus[0] & borrow, 0);
    let (d1, carry) = adc(d1, modulus[1] & borrow, carry);
    let (d2, carry) = adc(d2, modulus[2] & borrow, carry);
    let (d3, _) = adc(d3, modulus[3] & borrow, carry);

    [d0, d1, d2, d3]
}

/// Adds `k * 2^62` to `value` and `carry`, returning the low and high limbs.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
const fn mac_pasta_modulus_top(value: u64, k: u64, carry: u64) -> (u64, u64) {
    let (low, carry) = adc(value, k << PASTA_MODULUS_TOP_SHIFT, carry);
    (low, (k >> PASTA_MODULUS_TOP_OVERFLOW_SHIFT) + carry)
}

/// Squares and lazily reduces while interleaving independent upper-half work.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
const fn square_reduce_lazy_pasta(value: &[u64; 4], modulus: &[u64; 4], inv: u64) -> [u64; 4] {
    debug_assert!(modulus[2] == 0);
    debug_assert!(modulus[3] == 1 << PASTA_MODULUS_TOP_SHIFT);

    let [a0, a1, a2, a3] = *value;

    // Form the low off-diagonal limbs first. The remaining two products do
    // not affect the low half, so LLVM can overlap them with its reduction.
    let (r1, carry) = mac(0, a0, a1, 0);
    let (r2, carry) = mac(0, a0, a2, carry);
    let (r3, r4) = mac(0, a0, a3, carry);
    let (r3, cross_carry) = mac(r3, a1, a2, 0);

    let d3 = (r3 << 1) | (r2 >> 63);
    let d2 = (r2 << 1) | (r1 >> 63);
    let d1 = r1 << 1;

    let (t0, carry) = mac(0, a0, a0, 0);
    let (t1, carry) = adc(0, d1, carry);
    let (t2, carry) = mac(d2, a1, a1, carry);
    let (t3, square_carry) = adc(0, d3, carry);

    let k = t0.wrapping_mul(inv);
    let (_, carry) = mac(t0, k, modulus[0], 0);
    let (q0, carry) = mac(t1, k, modulus[1], carry);
    let (q1, carry) = adc(t2, 0, carry);
    let (q2, q3) = mac_pasta_modulus_top(t3, k, carry);

    let k = q0.wrapping_mul(inv);
    let (_, carry) = mac(q0, k, modulus[0], 0);
    let (q0, carry) = mac(q1, k, modulus[1], carry);
    let (q1, carry) = adc(q2, 0, carry);
    let (q2, q3) = mac_pasta_modulus_top(q3, k, carry);

    let k = q0.wrapping_mul(inv);
    let (_, carry) = mac(q0, k, modulus[0], 0);
    let (q0, carry) = mac(q1, k, modulus[1], carry);
    let (q1, carry) = adc(q2, 0, carry);
    let (q2, q3) = mac_pasta_modulus_top(q3, k, carry);

    let k = q0.wrapping_mul(inv);
    let (_, carry) = mac(q0, k, modulus[0], 0);
    let (q0, carry) = mac(q1, k, modulus[1], carry);
    let (q1, carry) = adc(q2, 0, carry);
    let (q2, q3) = mac_pasta_modulus_top(q3, k, carry);

    // Complete the upper off-diagonal limbs and add the diagonal products.
    let (r4, r5) = mac(r4, a1, a3, cross_carry);
    let (r5, r6) = mac(r5, a2, a3, 0);
    let r7 = r6 >> 63;
    let r6 = (r6 << 1) | (r5 >> 63);
    let r5 = (r5 << 1) | (r4 >> 63);
    let r4 = (r4 << 1) | (r3 >> 63);

    let (t4, carry) = mac(r4, a2, a2, square_carry);
    let (t5, carry) = adc(0, r5, carry);
    let (t6, carry) = mac(r6, a3, a3, carry);
    let (t7, _) = adc(0, r7, carry);

    let (q0, carry) = adc(q0, t4, 0);
    let (q1, carry) = adc(q1, t5, carry);
    let (q2, carry) = adc(q2, t6, carry);
    let (q3, _) = adc(q3, t7, carry);

    [q0, q1, q2, q3]
}

/// Squares a canonical `value` `n` times, keeping the accumulator below `2 * modulus`.
///
/// A Montgomery reduction accepts an input below `R * m` and returns a value below
/// `input / R + m`. Write `b_i = acc_i / R` and `m / R = 1/4 + e`, where `0 < e < 2^-130`
/// for both Pasta moduli. Then `b_0 < 1/3` and, by induction,
/// `b_i < 1/2 - 1 / (2 * (i + 3))` for every `i < 2^32`: the step needs
/// `e <= (i + 2) / (4 * (i + 3)^2 * (i + 4))`, which holds for all `u32` counts. So every
/// accumulator stays below `R / 2 < 2 * m`, and its square stays below `R * m`.
///
/// The result must be canonicalized by the caller, either with
/// [`canonicalize`] or by multiplying it by a canonical value using the
/// portable multiplier. The latter product is below `2 * m^2 < R * m`.
/// The loop count is public; no branch depends on the field value.
#[cfg_attr(target_arch = "x86_64", inline(never))]
#[cfg_attr(
    all(not(target_arch = "x86_64"), not(feature = "uninline-portable")),
    inline
)]
pub(super) fn sqr_n_lazy(value: &[u64; 4], n: u32, modulus: &[u64; 4], inv: u64) -> [u64; 4] {
    let mut acc = *value;
    for _ in 0..n {
        #[cfg(target_arch = "x86_64")]
        {
            acc = square_reduce_lazy_pasta(&acc, modulus, inv);
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            acc = montgomery_reduce_low_lazy(&square_wide(&acc), modulus, inv);
        }
    }
    acc
}
