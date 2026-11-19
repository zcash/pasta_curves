//! The affine model.

use crate::curves::params::{CurveParams, Pallas, Vesta};
use crate::curves::{Ep, EpAffine, Eq, EqAffine, IsoEpAffine, IsoEqAffine};
use core::mem::size_of;
use ff::Field;
use group::{Curve, CurveAffine as _, Group, GroupEncoding};
use proptest::prelude::*;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

/// A random point, from a seed that proptest draws and shrinks.
fn point<G: Group>(seed: [u8; 16]) -> G {
    G::random(&mut XorShiftRng::from_seed(seed))
}

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

macro_rules! model_tests {
    ($name:ident, $point:ty, $affine:ty, $params:ty) => {
        mod $name {
            use super::*;

            type Affine = $affine;
            type Base = <$params as CurveParams>::Base;

            /// Whether `p` satisfies $y^2 = x^3 + b$.
            fn on_curve(p: &Affine) -> bool {
                p.y.square() == p.x.square() * p.x + <$params as CurveParams>::B
            }

            proptest! {
                /// The affine point of a Jacobian point is on the curve and maps
                /// back to it; negation is $(x, -y)$; adding two affine points
                /// agrees with the Jacobian sum.
                #[test]
                fn chart_negation_and_addition(a in any::<[u8; 16]>(), b in any::<[u8; 16]>()) {
                    let (p, q) = (point::<$point>(a), point::<$point>(b));
                    let (pa, qa): (Affine, Affine) = (p.to_affine(), q.to_affine());
                    prop_assert!(on_curve(&pa));
                    prop_assert_eq!(<$point>::from(pa), p);
                    let neg = -pa;
                    prop_assert_eq!((neg.x, neg.y), (pa.x, -pa.y));
                    prop_assert_eq!(pa + qa, p + q);
                }

                /// The compressed encoding round-trips, and a point and its opposite
                /// differ only in the sign bit.
                #[test]
                fn encoding_round_trips(a in any::<[u8; 16]>()) {
                    let pa: Affine = point::<$point>(a).to_affine();
                    let bytes = pa.to_bytes();
                    prop_assert_eq!(Affine::from_bytes(&bytes).unwrap(), pa);
                    let mut flipped = bytes;
                    flipped[31] ^= 0x80;
                    prop_assert_eq!((-pa).to_bytes(), flipped);
                }
            }

            /// The identity is stored as $(0, 0)$, which is not on the curve
            /// because $b \neq 0$, and is encoded as zero.
            #[test]
            fn identity_is_stored_as_the_origin() {
                let o = Affine::identity();
                assert_eq!((o.x, o.y), (Base::ZERO, Base::ZERO));
                assert!(!on_curve(&o));
                assert_eq!(o.to_bytes(), [0; 32]);
                assert_eq!(Affine::from_bytes(&[0; 32]).unwrap(), o);
                assert_eq!(<$point>::from(o), <$point>::identity());
            }
        }
    };
}

model_tests!(pallas, Ep, EpAffine, Pallas);
model_tests!(vesta, Eq, EqAffine, Vesta);

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
