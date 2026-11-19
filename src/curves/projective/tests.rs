use super::Projective;
use ff::Field;
use group::{Curve, CurveAffine, Group};

// The original six-product cross-product tail of RCB Algorithm 7 is retained
// here as an independent reference for the five-product formula.
fn six_product_add<F: Field>(p: &Projective<F>, q: &Projective<F>) -> Projective<F> {
    let by_15 = |x: F| {
        let x2 = x + x;
        let x3 = x2 + x;
        let x6 = x3 + x3;
        x6 + x6 + x3
    };
    let t0 = p.x * q.x;
    let t1 = p.y * q.y;
    let t2 = p.z * q.z;
    let t3 = (p.x + p.y) * (q.x + q.y) - t0 - t1;
    let t4 = (p.y + p.z) * (q.y + q.z) - t1 - t2;
    let y3 = (p.x + p.z) * (q.x + q.z) - t0 - t2;
    let t0 = t0 + t0 + t0;
    let t2 = by_15(t2);
    let z3 = t1 + t2;
    let t1 = t1 - t2;
    let y3 = by_15(y3);
    let x3 = t4 * y3;
    let x3 = t3 * t1 - x3;
    let y3 = t1 * z3 + y3 * t0;
    let z3 = z3 * t4 + t0 * t3;
    Projective {
        x: x3,
        y: y3,
        z: z3,
    }
}

// The tripling as first written, with both isogenies inlined, kept as a reference
// that the factored form computes the same coordinates.
fn inlined_triple<F: Field>(p: &Projective<F>) -> Projective<F> {
    // Two complete degree-three isogenies and the isomorphism returning
    // to the original curve. Here bz² means 5z².
    let y2 = p.y.square();
    let z2 = p.z.square();
    let bz2 = z2 + z2 + z2 + z2 + z2;
    let v = y2 - bz2;
    let b4z2 = (bz2 + bz2) + (bz2 + bz2);
    let x = p.x * (v + b4z2);
    let y = p.y * (v - (b4z2 + b4z2));
    let z = p.z * v;

    let y2 = y.square();
    let z2 = z.square();
    let bz2 = z2 + z2 + z2 + z2 + z2;
    let b3z2 = bz2 + bz2 + bz2;
    let b9z2 = b3z2 + b3z2 + b3z2;
    let b27z2 = b9z2 + b9z2 + b9z2;
    let w = y2 + b27z2;
    let b108z2 = (b27z2 + b27z2) + (b27z2 + b27z2);
    let x = x * (w - b108z2);
    let y = y * (w + b108z2 + b108z2);
    let z = z * w;
    let z = z + z + z;
    let z = z + z + z;
    Projective {
        x: x + x + x,
        y,
        z: z + z + z,
    }
}

/// Whether `p` lies on $Y^2 Z = X^3 + b Z^3$.
fn on_curve<F: Field>(p: &Projective<F>, b: F) -> bool {
    p.y.square() * p.z == p.x.square() * p.x + b * p.z.square() * p.z
}

/// Whether `p` and `q` are the same projective point.
pub(crate) fn assert_same<F: Field>(p: Projective<F>, q: Projective<F>) {
    assert_eq!(bool::from(p.z.is_zero()), bool::from(q.z.is_zero()));
    if !bool::from(p.z.is_zero()) {
        assert!(bool::from((p.x * q.z).ct_eq(&(q.x * p.z))));
        assert!(bool::from((p.y * q.z).ct_eq(&(q.y * p.z))));
    }
}

/// Three base field elements of 32 bytes, with no padding, as the Jacobian
/// point type, so the radix-27 table costs what its coordinates do.
#[test]
fn projective_points_are_96_bytes() {
    assert_eq!(core::mem::size_of::<Projective<crate::Fp>>(), 96);
    assert_eq!(core::mem::size_of::<Projective<crate::Fq>>(), 96);
}

macro_rules! curve_tests {
    ($name:ident, $point:ty, $affine:ty, $field:ty, $scalar:ty) => {
        pub(crate) mod $name {
            use super::*;

            pub(crate) fn h(p: $affine) -> Projective<$field> {
                let (x, y) = p.raw_xy_for_tests();
                Projective::from_affine(x, y, p.is_identity())
            }

            #[test]
            fn complete_formulas_and_coordinates() {
                let identity = Projective::<$field>::identity();
                let g = <$point>::generator();
                let p = h(g.to_affine());
                let q = h(g.double().to_affine());
                let r = h((g.double() + g).to_affine());
                let neg = h((-g).to_affine());
                let inputs = [identity, p, q, r, neg];
                for point in [<$point>::identity(), g, g.double(), g.double() + g, -g] {
                    let point_h = h(point.to_affine());
                    assert_same(point_h.triple(), h((point.double() + point).to_affine()));
                    assert_same(point_h.add(&p), h((point + g).to_affine()));
                }
                for a in inputs {
                    assert_same(a.double(), a.add(&a));
                    assert_same(a.triple(), a.double().add(&a));
                    for b in inputs {
                        assert_same(a.add(&b), six_product_add(&a, &b));
                    }
                }
                assert_same(p.add(&neg), identity);
                assert_same(p.add(&identity), p);
                assert_same(identity.add(&p), p);
                let (x, y, z) = identity.to_jacobian();
                assert_eq!((x, y, z), (<$field>::ZERO, <$field>::ZERO, <$field>::ZERO));
                for (zero_x, zero_y) in [
                    (<$field>::ZERO, <$field>::ZERO),
                    (<$field>::ONE, <$field>::ONE),
                    (<$field>::from(13), <$field>::from(21)),
                ] {
                    let arbitrary = Projective::from_jacobian(zero_x, zero_y, <$field>::ZERO);
                    assert_eq!(arbitrary.to_jacobian(), (x, y, z));
                    assert_same(arbitrary.triple(), identity);
                    assert_same(arbitrary.add(&p), p);
                }
                let factor = <$field>::from(7);
                let factor2 = factor.square();
                let factor3 = factor2 * factor;
                let scaled = Projective::from_jacobian(p.x * factor2, p.y * factor3, factor);
                assert_same(scaled, p);
                assert_same(scaled.triple(), p.triple());
                assert_same(scaled.add(&q), p.add(&q));
                let affine_identity = <$affine>::identity();
                assert_same(h(affine_identity), identity);
            }
            #[test]
            fn tripling_factors_through_the_3_isogeny() {
                let five = <$field>::from(5);
                let image = -(five * <$field>::from(27));
                let twisted = five * <$field>::from(729);
                let g = <$point>::generator();
                for point in [<$point>::identity(), g, -g, g.double(), g.double() + g] {
                    let p = h(point.to_affine());
                    let once = p.isogeny_3(|x| x * five);
                    assert!(on_curve(&once, image));
                    let twice = once.isogeny_3(|x| x * image);
                    assert!(on_curve(&twice, twisted));
                    assert!(on_curve(&twice.iota(), five));
                    let (t, r) = (p.triple(), inlined_triple(&p));
                    assert_eq!((t.x, t.y, t.z), (r.x, r.y, r.z));
                }
            }
        }
    };
}

curve_tests!(
    pallas,
    crate::pallas::Point,
    crate::pallas::Affine,
    crate::Fp,
    crate::Fq
);
curve_tests!(
    vesta,
    crate::vesta::Point,
    crate::vesta::Affine,
    crate::Fq,
    crate::Fp
);
