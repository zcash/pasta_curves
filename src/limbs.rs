//! The limb type of the assembly backends and of the inversion, and the operand checks of their
//! debug assertions.

use core::hint::black_box;

/// Four little-endian 64-bit limbs, least significant first: a field element
/// (in Montgomery form, or canonical after `from_mont`) or a modulus.
pub type Limbs = [u64; 4];

/// `1` when `value < modulus` as little-endian 256-bit integers, else `0`: the borrow out of
/// the four-limb subtraction `value - modulus`, computed limb by limb. The debug assertions
/// must leave the routines' timing as it is, so the check has no data-dependent branch, and
/// each borrow passes through [`black_box`], the barrier that `subtle` uses, which keeps the
/// optimizer from turning the chain or its callers' combinations back into branches: an
/// opaque word can only be combined arithmetically.
#[inline(always)]
pub(crate) fn is_canonical_word(value: &Limbs, modulus: &Limbs) -> u64 {
    let mut borrow = 0;
    for (v, m) in value.iter().zip(modulus) {
        let (difference, underflow) = v.overflowing_sub(*m);
        let (_, borrow_underflow) = difference.overflowing_sub(borrow);
        borrow = black_box(u64::from(underflow | borrow_underflow));
    }
    borrow
}

/// Whether `value < modulus` as little-endian 256-bit integers; see [`is_canonical_word`].
#[inline(always)]
pub(crate) fn is_canonical(value: &Limbs, modulus: &Limbs) -> bool {
    is_canonical_word(value, modulus) == 1
}

/// The borrow chain of `is_canonical` decides `value < modulus` at the limb boundaries.
#[test]
fn is_canonical_borrow_chain() {
    let m = [5, 0, 0, 7];
    assert!(is_canonical(&[4, 0, 0, 7], &m));
    assert!(!is_canonical(&m, &m));
    assert!(!is_canonical(&[6, 0, 0, 7], &m));
    // A borrow out of the low limbs is absorbed by a larger top limb, and forced by a smaller one.
    assert!(is_canonical(&[u64::MAX, u64::MAX, u64::MAX, 6], &m));
    assert!(!is_canonical(&[0, 0, 0, 8], &m));
    // A middle limb decides when the top limbs agree.
    assert!(!is_canonical(&[0, 1, 0, 7], &m));
    assert!(is_canonical(&[u64::MAX, 0, 0, 6], &m));
    assert!(is_canonical(&[0, 0, 0, 0], &m));
}
