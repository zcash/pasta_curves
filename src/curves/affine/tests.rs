//! The affine model.

use crate::curves::{EpAffine, EqAffine, IsoEpAffine, IsoEqAffine};
use core::mem::size_of;

/// Two base field elements of 32 bytes, with no padding, so that tables of
/// points cost what their coordinates do.
#[test]
fn affine_points_are_64_bytes() {
    assert_eq!(size_of::<EpAffine>(), 64);
    assert_eq!(size_of::<EqAffine>(), 64);
    assert_eq!(size_of::<IsoEpAffine>(), 64);
    assert_eq!(size_of::<IsoEqAffine>(), 64);
}

#[cfg(feature = "zeroize")]
#[test]
fn affine_points_zeroize_to_identity() {
    use group::CurveAffine;
    use zeroize::Zeroize;

    let mut p = EpAffine::generator();
    p.zeroize();
    assert!(bool::from(p.is_identity()));

    let mut q = EqAffine::generator();
    q.zeroize();
    assert!(bool::from(q.is_identity()));
}

/// The unchecked constructor and its inverse `raw_xy`, against the checked
/// `from_xy`.
#[cfg(feature = "alloc")]
mod unchecked {
    use crate::arithmetic::CurveAffine;
    use crate::curves::{Ep, EpAffine, Eq, EqAffine};
    use crate::{Fp, Fq};
    use group::{Curve, CurveAffine as _, Group};
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    // Built in a `const` context, so that `from_xy_unchecked` staying `const` is
    // pinned by the build rather than by its signature.
    const PALLAS_GENERATOR: EpAffine = EpAffine::from_xy_unchecked(
        Fp::neg(&Fp::from_raw([1, 0, 0, 0])),
        Fp::from_raw([2, 0, 0, 0]),
    );
    const VESTA_GENERATOR: EqAffine = EqAffine::from_xy_unchecked(
        Fq::neg(&Fq::from_raw([1, 0, 0, 0])),
        Fq::from_raw([2, 0, 0, 0]),
    );

    /// Rebuilds `p` from its coordinates through `unchecked`, and checks that
    /// the result is on the curve and equal to what `from_xy` produces.
    fn assert_agrees_with_from_xy<C: CurveAffine>(p: C, unchecked: impl Fn(C::Base, C::Base) -> C) {
        let coords = p.coordinates().unwrap();
        let (x, y) = (*coords.x(), *coords.y());
        let q = unchecked(x, y);
        assert!(bool::from(q.is_on_curve()));
        assert!(q == C::from_xy(x, y).unwrap());
        assert!(q == p);
    }

    #[test]
    fn from_xy_unchecked_generator() {
        assert!(bool::from(PALLAS_GENERATOR.is_on_curve()));
        assert!(bool::from(VESTA_GENERATOR.is_on_curve()));
        assert!(PALLAS_GENERATOR == <EpAffine as group::CurveAffine>::generator());
        assert!(VESTA_GENERATOR == <EqAffine as group::CurveAffine>::generator());
        assert_agrees_with_from_xy(PALLAS_GENERATOR, EpAffine::from_xy_unchecked);
        assert_agrees_with_from_xy(VESTA_GENERATOR, EqAffine::from_xy_unchecked);
    }

    #[test]
    fn from_xy_unchecked_random_points() {
        let mut rng = XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);
        for _ in 0..100 {
            let p = Ep::try_random(&mut rng).unwrap().to_affine();
            assert_agrees_with_from_xy(p, EpAffine::from_xy_unchecked);
            let q = Eq::try_random(&mut rng).unwrap().to_affine();
            assert_agrees_with_from_xy(q, EqAffine::from_xy_unchecked);
        }
    }

    #[test]
    fn from_xy_unchecked_zero_is_identity() {
        let p = EpAffine::from_xy_unchecked(Fp::zero(), Fp::zero());
        assert!(bool::from(p.is_identity()));
        assert!(bool::from(p.is_on_curve()));
        assert!(p == EpAffine::from_xy(Fp::zero(), Fp::zero()).unwrap());
        let q = EqAffine::from_xy_unchecked(Fq::zero(), Fq::zero());
        assert!(bool::from(q.is_identity()));
        assert!(bool::from(q.is_on_curve()));
        assert!(q == EqAffine::from_xy(Fq::zero(), Fq::zero()).unwrap());
    }

    #[test]
    fn from_xy_unchecked_does_not_check() {
        // (1, 1) is not on y^2 = x^3 + 5: the unchecked constructor accepts it,
        // and the checked one rejects it.
        let p = EpAffine::from_xy_unchecked(Fp::one(), Fp::one());
        assert!(!bool::from(p.is_on_curve()));
        assert!(bool::from(
            EpAffine::from_xy(Fp::one(), Fp::one()).is_none()
        ));
    }

    /// `raw_xy` and `from_xy_unchecked` are inverse bijections between the
    /// affine values and pairs of base field elements: one way on points of
    /// the curve, including the identity, and the other way on any pair, on
    /// the curve or not.
    #[cfg(feature = "glv")]
    mod raw_xy_pbt {
        use group::Group;
        use proptest::prelude::*;

        use super::*;
        use crate::arithmetic::CurveExt;
        use crate::glv::conformance::scalar_strategy;

        macro_rules! raw_xy_inverse {
            ($mod_name:ident, $curve:ty, $affine:ty, $base:ty) => {
                mod $mod_name {
                    use super::*;

                    type Scalar = <$curve as CurveExt>::ScalarExt;

                    #[test]
                    fn identity_round_trips() {
                        let identity = <$affine>::identity();
                        assert_eq!(identity.raw_xy(), (<$base>::zero(), <$base>::zero()));
                        let (x, y) = identity.raw_xy();
                        assert!(<$affine>::from_xy_unchecked(x, y) == identity);
                    }

                    proptest! {
                        #[test]
                        fn from_xy_unchecked_inverts_raw_xy(k in scalar_strategy::<Scalar>()) {
                            let p = (<$curve>::generator() * k).to_affine();
                            let (x, y) = p.raw_xy();
                            prop_assert!(<$affine>::from_xy_unchecked(x, y) == p);
                        }

                        #[test]
                        fn raw_xy_inverts_from_xy_unchecked(
                            x in scalar_strategy::<$base>(),
                            y in scalar_strategy::<$base>(),
                        ) {
                            prop_assert_eq!(<$affine>::from_xy_unchecked(x, y).raw_xy(), (x, y));
                        }
                    }
                }
            };
        }

        raw_xy_inverse!(pallas, Ep, EpAffine, Fp);
        raw_xy_inverse!(vesta, Eq, EqAffine, Fq);
    }
}
