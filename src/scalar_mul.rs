//! Constant-time scalar multiplication on the native curves, in radix 27.
//!
//! The point is moved to [`Projective`] coordinates, whose complete formulas
//! let every scalar run the same sequence of operations.

use ff::Field;
use subtle::{ConditionallySelectable, ConstantTimeEq};

use crate::curves::projective::Projective;

/// Returns `by` times `p`, for the low 255 bits of the little-endian `by`,
/// with a fixed sequence of operations.
pub(crate) fn multiply<F: Field>(p: &Projective<F>, by: &[u8; 32]) -> Projective<F> {
    let digits = to_radix_3(by);
    // Multiples 1P through 26P. Each pair doubles an earlier entry and
    // adds P, using complete formulas even when P is the identity.
    // This stack table occupies 2496 bytes for either Pasta field.
    let mut table = [*p; 26];
    for i in (1..25).step_by(2) {
        table[i] = table[i >> 1].double();
        table[i + 1] = table[i].add(p);
    }
    table[25] = table[12].double();

    let mut acc = Projective::identity();
    // Three trits per radix-27 digit: a fixed 54 rounds, each scanning
    // every table entry and doing three triplings and one addition.
    for trits in digits.chunks_exact(3).rev() {
        let digit = trits[0] + 3 * trits[1] + 9 * trits[2];
        let mut addend = Projective::identity();
        for (i, entry) in table.iter().enumerate() {
            addend = Projective::conditional_select(&addend, entry, digit.ct_eq(&(i as u8 + 1)));
        }
        acc = acc.triple().triple().triple().add(&addend);
    }
    acc
}

/// Recodes the low 255 bits of a little-endian scalar into 161 little-endian
/// ternary digits. The final zero digit makes the length divisible by three.
/// The high bit is ignored, as in the previous group multiply implementations.
pub(crate) fn to_radix_3(by: &[u8; 32]) -> [u8; 162] {
    let mut limbs = [0u32; 8];
    for i in 0..8 {
        limbs[i] = u32::from_le_bytes([by[4 * i], by[4 * i + 1], by[4 * i + 2], by[4 * i + 3]]);
    }
    limbs[7] &= 0x7fff_ffff;

    let mut digits = [0u8; 162];
    for digit in &mut digits[..161] {
        let mut remainder = 0u64;
        // The dividend fits in 34 bits because the incoming remainder is < 3.
        for limb in limbs.iter_mut().rev() {
            let dividend = (remainder << 32) | u64::from(*limb);
            let quotient = dividend / 3;
            remainder = dividend - quotient * 3;
            *limb = quotient as u32;
        }
        *digit = remainder as u8;
    }
    digits
}

#[cfg(test)]
mod tests;
