use core::ops::{Add, AddAssign};

use ff::Field;

use crate::arithmetic::adc;

pub(super) mod sealed {
    pub trait Sealed {}
}

/// A trait for fields with a Montgomery representation, enabling deferred reduction.
pub trait MontgomeryRepr: sealed::Sealed + Field {
    /// Performs Montgomery reduction on the given 512-bit limbs.
    fn mont_reduce(limbs: [u64; 8]) -> Self;

    /// Multiplies `rhs` by `self`, returning the unreduced 512-bit Montgomery product.
    fn mul_unreduced(&self, rhs: &Self) -> MontProduct<Self>;

    /// Squares this element, returning the unreduced 512-bit Montgomery product.
    fn square_unreduced(&self) -> MontProduct<Self>;

    /// Computes the inner product of two slices, using deferred Montgomery
    /// reduction internally.
    ///
    /// Accumulates unreduced products and reduces only when the 512-bit
    /// accumulator's headroom is exhausted, minimizing the number of
    /// expensive Montgomery reductions.
    ///
    /// If the slices have different lengths, the extra elements of the longer
    /// slice are ignored.
    fn inner_product(a: &[Self], b: &[Self]) -> Self {
        let mut total = Self::ZERO;
        let mut acc = MontProduct::<Self>::ZERO;

        for (x, y) in a.iter().zip(b.iter()) {
            acc += x.mul_unreduced(y);
            if acc.needs_reduction() {
                total += acc.reduce();
                acc = MontProduct::<Self>::ZERO;
            }
        }

        total + acc.reduce()
    }
}

/// An unreduced 512-bit Montgomery product over field `F`.
///
/// This stores the raw result of a schoolbook multiplication before Montgomery
/// reduction. Multiple `MontProduct` values can be accumulated via [`Add`] before
/// performing a single expensive [`reduce`](MontProduct::reduce) call.
///
/// # Overflow safety
///
/// Each product of two reduced field elements fits in 510 bits (since both
/// inputs are less than the field modulus $m < 2^{255}$). The 512-bit
/// representation provides 2 bits of headroom, allowing up to
/// [`MAX_NUM_ADD`](MontProduct::MAX_NUM_ADD) terms to be summed without
/// overflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MontProduct<F: MontgomeryRepr> {
    pub(crate) limbs: [u64; 8],
    _marker: core::marker::PhantomData<F>,
}

impl<F: MontgomeryRepr> MontProduct<F> {
    /// Maximum number of unreduced products that can be safely accumulated
    /// via addition without overflow. See the [struct-level documentation](Self)
    /// for the overflow bound derivation.
    pub const MAX_NUM_ADD: usize = 4;

    /// The zero (additive identity) unreduced product.
    pub const ZERO: Self = MontProduct {
        limbs: [0; 8],
        _marker: core::marker::PhantomData,
    };

    /// Creates a `MontProduct` from raw 512-bit limbs.
    pub(crate) const fn from_limbs(limbs: [u64; 8]) -> Self {
        MontProduct {
            limbs,
            _marker: core::marker::PhantomData,
        }
    }

    /// Performs Montgomery reduction, returning the field element.
    pub fn reduce(self) -> F {
        F::mont_reduce(self.limbs)
    }

    /// Returns `true` if the accumulator's headroom is exhausted and a
    /// [`reduce`](Self::reduce) is needed before further accumulation.
    ///
    /// A single product fits in 510 bits, so the top 2 bits of the
    /// 512-bit representation serve as overflow indicators.
    fn needs_reduction(&self) -> bool {
        self.limbs[7] >> 62 != 0
    }
}

impl<'a, 'b, F: MontgomeryRepr> Add<&'b MontProduct<F>> for &'a MontProduct<F> {
    type Output = MontProduct<F>;

    #[inline]
    fn add(self, rhs: &'b MontProduct<F>) -> MontProduct<F> {
        let (d0, carry) = adc(self.limbs[0], rhs.limbs[0], 0);
        let (d1, carry) = adc(self.limbs[1], rhs.limbs[1], carry);
        let (d2, carry) = adc(self.limbs[2], rhs.limbs[2], carry);
        let (d3, carry) = adc(self.limbs[3], rhs.limbs[3], carry);
        let (d4, carry) = adc(self.limbs[4], rhs.limbs[4], carry);
        let (d5, carry) = adc(self.limbs[5], rhs.limbs[5], carry);
        let (d6, carry) = adc(self.limbs[6], rhs.limbs[6], carry);
        let (d7, carry) = adc(self.limbs[7], rhs.limbs[7], carry);
        debug_assert!(carry == 0, "MontProduct addition overflow");
        MontProduct::from_limbs([d0, d1, d2, d3, d4, d5, d6, d7])
    }
}

impl<'b, F: MontgomeryRepr> Add<&'b MontProduct<F>> for MontProduct<F> {
    type Output = MontProduct<F>;

    #[inline]
    fn add(self, rhs: &'b MontProduct<F>) -> MontProduct<F> {
        &self + rhs
    }
}

impl<'a, F: MontgomeryRepr> Add<MontProduct<F>> for &'a MontProduct<F> {
    type Output = MontProduct<F>;

    #[inline]
    fn add(self, rhs: MontProduct<F>) -> MontProduct<F> {
        self + &rhs
    }
}

impl<F: MontgomeryRepr> Add<MontProduct<F>> for MontProduct<F> {
    type Output = MontProduct<F>;

    #[inline]
    fn add(self, rhs: MontProduct<F>) -> MontProduct<F> {
        &self + &rhs
    }
}

impl<F: MontgomeryRepr> AddAssign<MontProduct<F>> for MontProduct<F> {
    #[inline]
    fn add_assign(&mut self, rhs: MontProduct<F>) {
        *self = &*self + &rhs;
    }
}

impl<'b, F: MontgomeryRepr> AddAssign<&'b MontProduct<F>> for MontProduct<F> {
    #[inline]
    fn add_assign(&mut self, rhs: &'b MontProduct<F>) {
        *self = &*self + rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::MontProduct;
    use crate::{Fp, Fq};
    use ff::Field;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;
    use std::vec::Vec;

    const SEED: [u8; 16] = [
        0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ];

    fn fp_from_limbs(limbs: [u64; 4]) -> Fp {
        Fp(limbs)
    }

    fn fq_from_limbs(limbs: [u64; 4]) -> Fq {
        Fq(limbs)
    }

    macro_rules! mont_product_tests {
        ($F:ty, $mod:ident, $from_limbs:ident, $a_limbs:expr, $b_limbs:expr) => {
            mod $mod {
                use super::*;

                #[test]
                fn mul_unreduced_roundtrip() {
                    let mut rng = XorShiftRng::from_seed(SEED);
                    for _ in 0..100 {
                        let a = <$F>::random(&mut rng);
                        let b = <$F>::random(&mut rng);
                        assert_eq!(a.mul_unreduced(&b).reduce(), a * b);
                    }
                }

                #[test]
                fn square_unreduced_roundtrip() {
                    let mut rng = XorShiftRng::from_seed(SEED);
                    for _ in 0..100 {
                        let a = <$F>::random(&mut rng);
                        assert_eq!(a.square_unreduced().reduce(), a.square());
                    }
                }

                #[test]
                fn inner_product_small() {
                    use super::super::MontgomeryRepr;
                    let mut rng = XorShiftRng::from_seed(SEED);
                    let a: Vec<$F> = (0..4).map(|_| <$F>::random(&mut rng)).collect();
                    let b: Vec<$F> = (0..4).map(|_| <$F>::random(&mut rng)).collect();

                    let eager: $F = a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum();
                    let lazy = <$F>::inner_product(&a, &b);

                    assert_eq!(eager, lazy);
                }

                #[test]
                fn inner_product_large() {
                    use super::super::MontgomeryRepr;
                    let mut rng = XorShiftRng::from_seed(SEED);
                    let len = 100;
                    let a: Vec<$F> = (0..len).map(|_| <$F>::random(&mut rng)).collect();
                    let b: Vec<$F> = (0..len).map(|_| <$F>::random(&mut rng)).collect();

                    let eager: $F = a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum();
                    let lazy = <$F>::inner_product(&a, &b);

                    assert_eq!(eager, lazy);
                }

                #[test]
                fn zero() {
                    assert_eq!(MontProduct::<$F>::ZERO.reduce(), <$F>::zero());
                }

                /// Exercises the `needs_reduction` overflow bug.
                ///
                /// `needs_reduction` checks `limbs[7] >> 62 != 0`, triggering
                /// at 2^510. But `R * p > 2^510` by only ~2^353, so when
                /// products have top limbs slightly below `>> 62`'s per-product
                /// threshold, 5 products can accumulate before the check fires.
                /// At that point the accumulator exceeds `R * p`, and
                /// `montgomery_reduce` (which does a single conditional
                /// subtraction) returns a non-canonical result off by `p`.
                #[test]
                fn inner_product_needs_reduction_overflow() {
                    use super::super::MontgomeryRepr;

                    // These elements have raw Montgomery limbs with top limb
                    // ~0x3F, so each product has limbs[7] ~0x0F. This means
                    // needs_reduction stays false for the first 4 products and
                    // only fires after the 5th — by which point the 512-bit
                    // accumulator has overflowed R * p.
                    let a = $from_limbs($a_limbs);
                    let b = $from_limbs($b_limbs);

                    let a_arr = [a; 100];
                    let b_arr = [b; 100];

                    let eager: $F = a_arr.iter().zip(b_arr.iter()).map(|(x, y)| *x * *y).sum();
                    let lazy = <$F>::inner_product(&a_arr, &b_arr);

                    // Note: Fp's Debug uses to_repr() which canonicalizes,
                    // so both sides print identically even when limbs differ.
                    // Compare the raw Montgomery limbs directly.
                    assert_eq!(
                        eager, lazy,
                        "inner_product returned non-canonical result \
                         (limbs differ by a multiple of the modulus)"
                    );
                }
            }
        };
    }

    mont_product_tests!(
        Fp,
        fp,
        fp_from_limbs,
        // Montgomery limbs with top limb ~0x3F, found by searching for pairs whose
        // product accumulates 5 times before needs_reduction triggers.
        [
            0x0361524c2cc0f859u64,
            0xae68690a78bc7175,
            0xe66cd36e68ef8f5f,
            0x3fa6524a713b7e05
        ],
        [
            0x637e0edc5b6e4ae7u64,
            0xa859890cd670f668,
            0x27460f22403d1f83,
            0x3f6208144fbaecc0
        ]
    );
    mont_product_tests!(
        Fq,
        fq,
        fq_from_limbs,
        [
            0x31d0b6640589f877u64,
            0xf87f43fdf6062541,
            0xb7d6467b2f5a522a,
            0x3eb025240950fd13
        ],
        [
            0xba26d85135e8579au64,
            0x0fa34266ccfdba9b,
            0xade9b2b4efdd35f8,
            0x3caaf0e81fb797fa
        ]
    );
}
