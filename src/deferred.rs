//! Deferred normalization for field arithmetic.
//!
//! This module provides the [`DeferredField`] trait, which enables accumulating
//! multiple unreduced products before performing a single expensive reduction.
//! This is useful for operations like inner products where many multiplications
//! feed into a sum.

use core::fmt::Debug;

/// A trait for fields that support deferred reduction of products.
///
/// Instead of reducing each multiplication result immediately, callers
/// accumulate products into an [`Accumulator`](Self::Accumulator) via
/// [`mul_accumulate`](Self::mul_accumulate) and
/// [`square_accumulate`](Self::square_accumulate), then perform a single
/// reduction at the end with [`reduce`](Self::reduce).
pub trait DeferredField: ff::Field {
    /// A wide accumulator for unreduced products.
    type Accumulator: Copy + Clone + Debug + Default;

    /// Multiplies `a` by `b` and adds the result into `acc`.
    fn mul_accumulate(acc: &mut Self::Accumulator, a: &Self, b: &Self);

    /// Squares `a` and adds the result into `acc`.
    fn square_accumulate(acc: &mut Self::Accumulator, a: &Self);

    /// Reduces the accumulator to a canonical field element.
    fn reduce(acc: Self::Accumulator) -> Self;
}

#[cfg(test)]
mod tests {
    use super::DeferredField;
    use ff::Field;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;
    use std::vec::Vec;

    const SEED: [u8; 16] = [
        0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ];

    fn inner_product<F: DeferredField>(a: &[F], b: &[F]) -> F {
        let mut acc = F::Accumulator::default();
        for (x, y) in a.iter().zip(b.iter()) {
            F::mul_accumulate(&mut acc, x, y);
        }
        F::reduce(acc)
    }

    macro_rules! deferred_field_tests {
        ($F:ty, $mod:ident, $adversarial_a:expr, $adversarial_b:expr) => {
            mod $mod {
                use super::*;

                #[test]
                fn accumulate_roundtrip() {
                    let mut rng = XorShiftRng::from_seed(SEED);
                    for _ in 0..100 {
                        let a = <$F>::random(&mut rng);
                        let b = <$F>::random(&mut rng);
                        let mut acc = <$F as DeferredField>::Accumulator::default();
                        <$F>::mul_accumulate(&mut acc, &a, &b);
                        assert_eq!(<$F>::reduce(acc), a * b);
                    }
                }

                #[test]
                fn square_accumulate_roundtrip() {
                    let mut rng = XorShiftRng::from_seed(SEED);
                    for _ in 0..100 {
                        let a = <$F>::random(&mut rng);
                        let mut acc = <$F as DeferredField>::Accumulator::default();
                        <$F>::square_accumulate(&mut acc, &a);
                        assert_eq!(<$F>::reduce(acc), a.square());
                    }
                }

                #[test]
                fn test_inner_product() {
                    let mut rng = XorShiftRng::from_seed(SEED);
                    for len in [0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 100, 255, 256, 10_000] {
                        let a: Vec<$F> = (0..len).map(|_| <$F>::random(&mut rng)).collect();
                        let b: Vec<$F> = (0..len).map(|_| <$F>::random(&mut rng)).collect();

                        let eager: $F = a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum();
                        let lazy = inner_product(&a, &b);

                        assert_eq!(eager, lazy, "mismatch at len={len}");
                    }
                }

                #[test]
                fn reduce_zero() {
                    assert_eq!(
                        <$F>::reduce(<$F as DeferredField>::Accumulator::default()),
                        <$F>::ZERO,
                    );
                }

                #[test]
                fn square_vs_mul() {
                    let mut rng = XorShiftRng::from_seed(SEED);
                    for _ in 0..100 {
                        let a = <$F>::random(&mut rng);
                        let mut sq_acc = <$F as DeferredField>::Accumulator::default();
                        <$F>::square_accumulate(&mut sq_acc, &a);
                        let mut mul_acc = <$F as DeferredField>::Accumulator::default();
                        <$F>::mul_accumulate(&mut mul_acc, &a, &a);
                        assert_eq!(
                            <$F>::reduce(sq_acc),
                            <$F>::reduce(mul_acc),
                            "square_accumulate and mul_accumulate(a, a) diverged",
                        );
                    }
                }

                #[test]
                fn mixed_accumulate() {
                    let mut rng = XorShiftRng::from_seed(SEED);
                    for _ in 0..20 {
                        let a = <$F>::random(&mut rng);
                        let b = <$F>::random(&mut rng);
                        let c = <$F>::random(&mut rng);
                        let mut acc = <$F as DeferredField>::Accumulator::default();
                        <$F>::mul_accumulate(&mut acc, &a, &b);
                        <$F>::square_accumulate(&mut acc, &c);
                        assert_eq!(<$F>::reduce(acc), a * b + c.square());
                    }
                }

                /// Regression: elements with top limb ~0x3F whose products have
                /// limbs[7] ~0x0F. These adversarial elements exercise the
                /// partial-reduction path in the lazy Product accumulator.
                #[test]
                fn regression_overflow() {
                    let a = $adversarial_a;
                    let b = $adversarial_b;
                    let a_arr = [a; 100];
                    let b_arr = [b; 100];

                    let eager: $F = a_arr.iter().zip(b_arr.iter()).map(|(x, y)| *x * *y).sum();
                    let lazy = inner_product(&a_arr, &b_arr);

                    assert_eq!(eager, lazy, "inner_product returned non-canonical result");
                }
            }
        };
    }

    deferred_field_tests!(
        crate::Fp,
        fp,
        crate::Fp([
            0x0361524c2cc0f859u64,
            0xae68690a78bc7175,
            0xe66cd36e68ef8f5f,
            0x3fa6524a713b7e05,
        ]),
        crate::Fp([
            0x7a1c5e3b9d204f61u64,
            0xc48e0b71a2d5f389,
            0xd9f247a0856c13be,
            0x3d8a19f5e6c7b042,
        ])
    );
    deferred_field_tests!(
        crate::Fq,
        fq,
        crate::Fq([
            0x31d0b6640589f877u64,
            0xf87f43fdf6062541,
            0xb7d6467b2f5a522a,
            0x3eb025240950fd13,
        ]),
        crate::Fq([
            0x5e9a3c71f8b20d46u64,
            0xa3d1e6f504879c2b,
            0xcb45a8d2e1f36790,
            0x3c47d2a8b10e5f93,
        ])
    );
}
