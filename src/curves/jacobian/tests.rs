//! The Jacobian model.

use crate::curves::params::{CurveParams, Pallas, Vesta};
use crate::curves::{Ep, EpAffine, Eq, EqAffine, IsoEp, IsoEq};
use core::mem::size_of;
use ff::Field;
use group::{Curve, Group};
use proptest::prelude::*;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

/// A random point, from a seed that proptest draws and shrinks.
fn point<G: Group>(seed: [u8; 16]) -> G {
    G::random(&mut XorShiftRng::from_seed(seed))
}

/// Three base field elements of 32 bytes, with no padding, so that tables of
/// points cost what their coordinates do.
#[test]
fn jacobian_points_are_96_bytes() {
    assert_eq!(size_of::<Ep>(), 96);
    assert_eq!(size_of::<Eq>(), 96);
    assert_eq!(size_of::<IsoEp>(), 96);
    assert_eq!(size_of::<IsoEq>(), 96);
}

#[cfg(feature = "zeroize")]
#[test]
fn jacobian_points_zeroize_to_identity() {
    use group::Group;
    use zeroize::Zeroize;

    let mut p = Ep::generator();
    p.zeroize();
    assert!(bool::from(p.is_identity()));

    let mut q = Eq::generator();
    q.zeroize();
    assert!(bool::from(q.is_identity()));
}

macro_rules! model_tests {
    ($name:ident, $point:ty, $affine:ty, $params:ty) => {
        mod $name {
            use super::*;

            type Point = $point;
            type Base = <$params as CurveParams>::Base;

            /// Whether `p` satisfies the Jacobian equation $Y^2 = X^3 + b Z^6$.
            fn on_curve(p: &Point) -> bool {
                let z2 = p.z.square();
                p.y.square() == p.x.square() * p.x + <$params as CurveParams>::B * z2.square() * z2
            }

            /// The representative $(\lambda^2 X : \lambda^3 Y : \lambda Z)$ of the same
            /// point.
            fn rescaled(p: &Point, lambda: Base) -> Point {
                let lambda2 = lambda.square();
                Point {
                    x: p.x * lambda2,
                    y: p.y * lambda2 * lambda,
                    z: p.z * lambda,
                }
            }

            proptest! {
                /// Doubling, addition, mixed addition and subtraction land on the
                /// curve's Jacobian equation.
                #[test]
                fn operations_stay_on_the_curve(a in any::<[u8; 16]>(), b in any::<[u8; 16]>()) {
                    let (p, q) = (point::<Point>(a), point::<Point>(b));
                    let q_affine: $affine = q.to_affine();
                    for r in [p.double(), p + q, p + q_affine, p - q] {
                        prop_assert!(on_curve(&r));
                    }
                }

                /// Every representative of a class is the same point, and the
                /// operations do not depend on which one they are given.
                #[test]
                fn rescaling_is_the_same_point(
                    a in any::<[u8; 16]>(),
                    b in any::<[u8; 16]>(),
                    lambda in any::<u64>(),
                ) {
                    let lambda = Base::from(lambda | 1);
                    let (p, q) = (point::<Point>(a), point::<Point>(b));
                    let scaled = rescaled(&p, lambda);
                    prop_assert_eq!(scaled, p);
                    prop_assert_eq!(scaled + q, p + q);
                    prop_assert_eq!(q + scaled, q + p);
                    prop_assert_eq!(scaled.double(), p.double());
                }

                /// The cases the incomplete addition formula handles apart: equal
                /// points, given by different representatives, and opposite points.
                #[test]
                fn exceptional_additions(a in any::<[u8; 16]>(), lambda in any::<u64>()) {
                    let lambda = Base::from(lambda | 1) + Base::ONE;
                    let p = point::<Point>(a);
                    prop_assert_eq!(p + p, p.double());
                    prop_assert_eq!(p + rescaled(&p, lambda), p.double());
                    prop_assert!(bool::from((p + (-p)).is_identity()));
                    prop_assert!(bool::from((p + rescaled(&-p, lambda)).is_identity()));
                }
            }

            /// Every triple with $Z = 0$ is the identity, not only the stored
            /// $(0, 0, 0)$.
            #[test]
            fn any_z_zero_is_the_identity() {
                let g = Point::generator();
                for (x, y) in [(0, 0), (1, 1), (13, 21)] {
                    let o = Point {
                        x: Base::from(x),
                        y: Base::from(y),
                        z: Base::ZERO,
                    };
                    assert!(bool::from(o.is_identity()));
                    assert_eq!(o, Point::identity());
                    assert_eq!(o + g, g);
                    assert_eq!(g + o, g);
                    assert!(bool::from(o.double().is_identity()));
                }
            }
        }
    };
}

model_tests!(pallas, Ep, EpAffine, Pallas);
model_tests!(vesta, Eq, EqAffine, Vesta);
