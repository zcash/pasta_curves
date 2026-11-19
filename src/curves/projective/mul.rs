//! Constant-time scalar multiplication on the projective model, in radix 27.
//!
//! The scalar is recoded into balanced radix-27 digits, and the point's
//! multiples are combined by triplings and additions whose complete formulas
//! let every scalar run the same sequence of operations.

// On 32-bit ARM the dispatch keeps the binary ladder, so only the tests use the
// radix-27 one there.
#![cfg_attr(target_arch = "arm", allow(dead_code))]

use core::mem::size_of;

use ff::Field;
use subtle::{Choice, ConditionallyNegatable, ConditionallySelectable, ConstantTimeEq};

use super::{PASTA_B, Projective};

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

/// The digits of the ladder, after padding.
const DIGITS: usize = PADDED_TRITS / TRITS_PER_DIGIT;

/// The largest magnitude of a signed digit. Digits are balanced, in
/// $[-13, 13]$, which is again a complete set of residues modulo 27.
const HALF_RADIX: u8 = (RADIX / 2) as u8;

/// The table holds $1P, \dots, 13P$, one entry per nonzero digit magnitude; a
/// negative digit selects its magnitude and negates it. It lives on the stack:
/// 13 points of three 32-byte coordinates, 1248 bytes, checked below.
const TABLE_LEN: usize = HALF_RADIX as usize;

/// The scalar is recoded from its encoding read as `u32` limbs.
const LIMBS: usize = 8;

/// The mask that keeps the low `SCALAR_BITS - 224` bits of the top limb.
const TOP_LIMB_MASK: u32 = (1 << (SCALAR_BITS - 224)) - 1;

const _: () = assert!(RADIX == (TERNARY as usize).pow(TRITS_PER_DIGIT as u32));
// The table's size on the stack, for either field.
const _: () = assert!(TABLE_LEN * size_of::<Projective<crate::Fp, PASTA_B>>() == 1248);
const _: () = assert!(TABLE_LEN * size_of::<Projective<crate::Fq, PASTA_B>>() == 1248);
// The table is filled by pairs after its first entry.
const _: () = assert!(TABLE_LEN % 2 == 1);
// A balanced expansion with `DIGITS` digits covers every integer of magnitude at
// most (27^54 - 1) / 2, about 2^255.8, so it needs no extra digit for a scalar
// below 2^255: 27^54 = 3^162 > 2 * 3^161 > 2^256.
const _: () = assert!(DIGITS == 54 && PADDED_TRITS == 2 * 81);
const _: () = assert!(LIMBS * 32 == 256);

/// Returns `by` times `p`, for the low 255 bits of the little-endian `by`,
/// with a fixed sequence of operations.
pub(crate) fn multiply<F: Field>(
    p: &Projective<F, PASTA_B>,
    by: &[u8; 32],
) -> Projective<F, PASTA_B> {
    // `mut` only to clear the digits afterwards, with `zeroize`.
    #[cfg_attr(not(feature = "zeroize"), allow(unused_mut))]
    let mut digits = to_signed_radix_27(by);
    // Multiples 1P through 13P. Each pair doubles an earlier entry and adds P,
    // using complete formulas even when P is the identity.
    let mut table = [*p; TABLE_LEN];
    for i in (1..TABLE_LEN).step_by(2) {
        table[i] = table[i >> 1].double();
        table[i + 1] = table[i].add(p);
    }

    let mut acc = Projective::identity();
    // A fixed `DIGITS` (54) rounds, each scanning every table entry for the
    // digit's magnitude, negating the result for a negative digit, and doing
    // three triplings and one addition.
    for &digit in digits.iter().rev() {
        let negative = (digit as u8) >> 7;
        let magnitude = ((digit as u8) ^ 0u8.wrapping_sub(negative)).wrapping_add(negative);
        let mut addend = Projective::identity();
        for (i, entry) in table.iter().enumerate() {
            addend =
                Projective::conditional_select(&addend, entry, magnitude.ct_eq(&(i as u8 + 1)));
        }
        addend.conditional_negate(Choice::from(negative));
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

/// Recodes the low 255 bits of a little-endian scalar into `DIGITS` balanced
/// digits in radix 27, least significant first, each in $[-13, 13]$.
///
/// The unsigned digits come from three trits each; a digit above 13 becomes
/// itself minus 27 and carries one into the next. Every step is a fixed sequence
/// of byte operations, without branches on the scalar.
pub(crate) fn to_signed_radix_27(by: &[u8; 32]) -> [i8; DIGITS] {
    #[cfg_attr(not(feature = "zeroize"), allow(unused_mut))]
    let mut trits = to_radix_3(by);
    let mut digits = [0i8; DIGITS];
    let mut carry = 0u8;
    for (digit, trits) in digits.iter_mut().zip(trits.chunks_exact(TRITS_PER_DIGIT)) {
        let unsigned = trits
            .iter()
            .rev()
            .fold(0, |digit, &trit| digit * TERNARY + trit)
            + carry;
        // `unsigned` is at most 27; it is above 13 exactly when 13 - unsigned wraps.
        carry = HALF_RADIX.wrapping_sub(unsigned) >> 7;
        *digit = unsigned.wrapping_sub(carry * RADIX as u8) as i8;
    }
    #[cfg(feature = "zeroize")]
    zeroize::Zeroize::zeroize(&mut trits);
    digits
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

#[cfg(test)]
mod tests;
