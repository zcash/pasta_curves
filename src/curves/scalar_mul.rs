//! Constant-time scalar multiplication on the native curves, in radix 27.
//!
//! The point is moved to [`Projective`] coordinates, whose complete formulas
//! let every scalar run the same sequence of operations.
//!
//! The module also holds the dispatch of the curves' `Mul` impls
//! (`mul_curve_specific!`), which uses this ladder on Pallas and Vesta except on
//! 32-bit ARM (`if_projective_mul!`), and a binary ladder otherwise.

// On 32-bit ARM the dispatch keeps the binary ladder, so only the tests use the
// radix-27 one there.
#![cfg_attr(target_arch = "arm", allow(dead_code))]

use ff::Field;
use subtle::{ConditionallySelectable, ConstantTimeEq};

use crate::curves::projective::{PASTA_B, Projective};

/// The base of the recoding: the scalar is written in ternary.
const TERNARY: u8 = 3;

/// Ternary digits per digit of the ladder: three trits make one digit in radix
/// $3^3 = 27$, consumed by three triplings and one addition.
const TRITS_PER_DIGIT: usize = 3;

/// The radix of the ladder's digits, $3^3$.
const RADIX: usize = 27;

/// The bits of the scalar that are read. Scalars are below $2^{255}$, so the top
/// bit of their 32-byte encoding is always zero and is ignored.
const SCALAR_BITS: u32 = 255;

/// The ternary digits needed for `SCALAR_BITS` bits: $3^{160} < 2^{255} \le 3^{161}$.
const TRITS: usize = 161;

/// `TRITS` rounded up to a multiple of `TRITS_PER_DIGIT`, with zero padding on top.
const PADDED_TRITS: usize = TRITS.div_ceil(TRITS_PER_DIGIT) * TRITS_PER_DIGIT;

/// The table holds $1P, \dots, 26P$, one entry per nonzero digit.
const TABLE_LEN: usize = RADIX - 1;

/// The scalar is recoded from its encoding read as `u32` limbs.
const LIMBS: usize = 8;

/// The mask that keeps the low `SCALAR_BITS - 224` bits of the top limb.
const TOP_LIMB_MASK: u32 = (1 << (SCALAR_BITS - 224)) - 1;

const _: () = assert!(RADIX == (TERNARY as usize).pow(TRITS_PER_DIGIT as u32));
// The table is filled by pairs, then its last entry by one doubling.
const _: () = assert!(TABLE_LEN % 2 == 0);
const _: () = assert!(LIMBS * 32 == 256);

/// Returns `by` times `p`, for the low 255 bits of the little-endian `by`,
/// with a fixed sequence of operations.
pub(crate) fn multiply<F: Field>(
    p: &Projective<F, PASTA_B>,
    by: &[u8; 32],
) -> Projective<F, PASTA_B> {
    // `mut` only to clear the digits afterwards, with `zeroize`.
    #[cfg_attr(not(feature = "zeroize"), allow(unused_mut))]
    let mut digits = to_radix_3(by);
    // Multiples 1P through 26P. Each pair doubles an earlier entry and
    // adds P, using complete formulas even when P is the identity.
    // The table occupies `TABLE_LEN` points of 96 bytes, 2496 bytes, on the
    // stack.
    let mut table = [*p; TABLE_LEN];
    for i in (1..TABLE_LEN - 1).step_by(2) {
        table[i] = table[i >> 1].double();
        table[i + 1] = table[i].add(p);
    }
    table[TABLE_LEN - 1] = table[(TABLE_LEN - 1) >> 1].double();

    let mut acc = Projective::identity();
    // A fixed `PADDED_TRITS / TRITS_PER_DIGIT` (54) rounds, each scanning every
    // table entry and doing three triplings and one addition.
    for trits in digits.chunks_exact(TRITS_PER_DIGIT).rev() {
        let digit = trits
            .iter()
            .rev()
            .fold(0, |digit, &trit| digit * TERNARY + trit);
        let mut addend = Projective::identity();
        for (i, entry) in table.iter().enumerate() {
            addend = Projective::conditional_select(&addend, entry, digit.ct_eq(&(i as u8 + 1)));
        }
        acc = acc.triple().triple().triple().add(&addend);
        #[cfg(feature = "zeroize")]
        zeroize::Zeroize::zeroize(&mut addend);
    }
    // The digits and the table are derived from the secret scalar. With `zeroize`,
    // overwrite them before the stack frame is released; copies made by moves or
    // spills are out of reach.
    #[cfg(feature = "zeroize")]
    {
        zeroize::Zeroize::zeroize(&mut digits);
        zeroize::Zeroize::zeroize(&mut table);
    }
    acc
}

/// Recodes the low 255 bits of a little-endian scalar into 161 little-endian
/// ternary digits. The final zero digit makes the length divisible by three.
/// The high bit is ignored, as in the previous group multiply implementations.
pub(crate) fn to_radix_3(by: &[u8; 32]) -> [u8; PADDED_TRITS] {
    let mut limbs = [0u32; LIMBS];
    for (limb, bytes) in limbs.iter_mut().zip(by.chunks_exact(4)) {
        *limb = u32::from_le_bytes(bytes.try_into().unwrap());
    }
    limbs[LIMBS - 1] &= TOP_LIMB_MASK;

    let mut digits = [0u8; PADDED_TRITS];
    for digit in &mut digits[..TRITS] {
        let mut remainder = 0u64;
        // The dividend fits in 34 bits because the incoming remainder is < 3.
        for limb in limbs.iter_mut().rev() {
            let dividend = (remainder << u32::BITS) | u64::from(*limb);
            let quotient = dividend / u64::from(TERNARY);
            remainder = dividend - quotient * u64::from(TERNARY);
            *limb = quotient as u32;
        }
        *digit = remainder as u8;
    }
    digits
}

/// Selects the radix-27 scalar multiplication in projective coordinates, except on
/// 32-bit ARM.
/// Keystone measurements favor the existing Jacobian path; retain it on other
/// 32-bit ARM targets until measurements support enabling it.
macro_rules! if_projective_mul {
    ($projective:expr, $jacobian:expr $(,)?) => {{
        #[cfg(not(target_arch = "arm"))]
        {
            $projective
        }
        #[cfg(target_arch = "arm")]
        {
            $jacobian
        }
    }};
}

macro_rules! mul_curve_specific {
    ($name:ident, $base:ident, special_a0_b5, jacobian, $point:expr, $scalar:expr) => {{
        let point = crate::curves::projective::Projective::<
            $base,
            { crate::curves::projective::PASTA_B },
        >::from_jacobian($point.x, $point.y, $point.z);
        #[cfg_attr(not(feature = "zeroize"), allow(unused_mut))]
        let mut repr = $scalar.to_repr();
        let (x, y, z) = crate::curves::scalar_mul::multiply(&point, &repr).to_jacobian();
        #[cfg(feature = "zeroize")]
        zeroize::Zeroize::zeroize(&mut repr);
        $name { x, y, z }
    }};
    ($name:ident, $base:ident, special_a0_b5, affine, $point:expr, $scalar:expr) => {{
        let point = crate::curves::projective::Projective::<
            $base,
            { crate::curves::projective::PASTA_B },
        >::from_affine($point.x, $point.y, $point.is_identity());
        #[cfg_attr(not(feature = "zeroize"), allow(unused_mut))]
        let mut repr = $scalar.to_repr();
        let (x, y, z) = crate::curves::scalar_mul::multiply(&point, &repr).to_jacobian();
        #[cfg(feature = "zeroize")]
        zeroize::Zeroize::zeroize(&mut repr);
        $name { x, y, z }
    }};
    ($name:ident, $base:ident, general, $coord:ident, $point:expr, $scalar:expr) => {{
        let mut acc = $name::identity();

        // This is a simple double-and-add implementation of point
        // multiplication, moving from most significant to least
        // significant bit of the scalar.
        //
        // We don't use `PrimeFieldBits::to_le_bits` here, because that would
        // force users of this crate to depend on `bitvec` where they otherwise
        // might not need to.
        //
        // NOTE: We skip the leading bit because it's always unset (we are turning
        // the 32-byte repr into 256 bits, and $scalar::NUM_BITS = 255).
        for bit in $scalar
            .to_repr()
            .iter()
            .rev()
            .flat_map(|byte| (0..8).rev().map(move |i| Choice::from((byte >> i) & 1u8)))
            .skip(1)
        {
            acc = acc.double();
            acc = $name::conditional_select(&acc, &(acc + $point), bit);
        }

        acc
    }};
}

#[cfg(test)]
mod tests;
