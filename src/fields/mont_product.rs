use core::ops::{Add, AddAssign};

use crate::arithmetic::adc;

pub(super) mod sealed {
    pub trait Sealed {}
}

/// A trait for fields with a Montgomery representation, enabling deferred reduction.
pub trait MontgomeryRepr: sealed::Sealed + Sized {
    /// Performs Montgomery reduction on the given 512-bit limbs.
    fn mont_reduce(limbs: [u64; 8]) -> Self;

    /// Multiplies `rhs` by `self`, returning the unreduced 512-bit Montgomery product.
    fn mul_unreduced(&self, rhs: &Self) -> MontProduct<Self>;

    /// Squares this element, returning the unreduced 512-bit Montgomery product.
    fn square_unreduced(&self) -> MontProduct<Self>;
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

    macro_rules! mont_product_tests {
        ($F:ty, $mod:ident) => {
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
                fn inner_product() {
                    let mut rng = XorShiftRng::from_seed(SEED);
                    let n = MontProduct::<$F>::MAX_NUM_ADD;
                    let a: Vec<$F> = (0..n).map(|_| <$F>::random(&mut rng)).collect();
                    let b: Vec<$F> = (0..n).map(|_| <$F>::random(&mut rng)).collect();

                    let eager: $F = a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum();

                    let lazy = a
                        .iter()
                        .zip(b.iter())
                        .fold(MontProduct::<$F>::ZERO, |acc, (x, y)| {
                            acc + x.mul_unreduced(y)
                        })
                        .reduce();

                    assert_eq!(eager, lazy);
                }

                #[test]
                fn long_inner_product_chunked() {
                    let mut rng = XorShiftRng::from_seed(SEED);
                    let n = MontProduct::<$F>::MAX_NUM_ADD;
                    let len = 4 * n;
                    let a: Vec<$F> = (0..len).map(|_| <$F>::random(&mut rng)).collect();
                    let b: Vec<$F> = (0..len).map(|_| <$F>::random(&mut rng)).collect();

                    let eager: $F = a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum();

                    let lazy: $F = a
                        .chunks(n)
                        .zip(b.chunks(n))
                        .map(|(ac, bc)| {
                            ac.iter()
                                .zip(bc.iter())
                                .fold(MontProduct::<$F>::ZERO, |acc, (x, y)| {
                                    acc + x.mul_unreduced(y)
                                })
                                .reduce()
                        })
                        .sum();

                    assert_eq!(eager, lazy);
                }

                #[test]
                fn zero() {
                    assert_eq!(MontProduct::<$F>::ZERO.reduce(), <$F>::zero());
                }
            }
        };
    }

    mont_product_tests!(Fp, fp);
    mont_product_tests!(Fq, fq);
}
