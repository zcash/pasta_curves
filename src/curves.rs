//! This module contains implementations for the Pallas and Vesta elliptic curve
//! groups.

use core::cmp;
use core::fmt;
use core::iter::Sum;
use core::ops::{Add, Mul, Neg, Sub};

#[cfg(feature = "alloc")]
use alloc::boxed::Box;

use ff::{Field, PrimeField};
use group::{
    Curve as _, CurveAffine as _, Group as _, GroupEncoding,
    cofactor::{CofactorCurve, CofactorGroup},
    prime::{PrimeCurve, PrimeGroup},
};
use rand::TryRng;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

#[cfg(feature = "alloc")]
use ff::WithSmallOrderMulGroup;

use super::{Fp, Fq};

#[macro_use]
mod affine;
#[macro_use]
mod jacobian;
#[macro_use]
mod maps;

// Keep this in sync with if_homogeneous_mul!; ARM tests still exercise the formulas.
#[cfg(any(not(target_arch = "arm"), test))]
pub(crate) mod projective;

#[cfg(feature = "alloc")]
use crate::arithmetic::{Coordinates, CurveAffine, CurveExt, CurveExtUnchecked, VartimeField};

macro_rules! new_curve_impl {
    (($($privacy:tt)*), $name:ident, $name_affine:ident, $iso:ident, $base:ident, $scalar:ident,
     $curve_id:literal, $a_raw:expr, $b_raw:expr, $curve_type:ident,
     $unchecked_ctor:ident, $same_scalar_mul:ident) => {
        new_jacobian_impl!(
            ($($privacy)*), $name, $name_affine, $iso, $base, $scalar, $curve_id, $a_raw, $b_raw,
            $curve_type, $same_scalar_mul
        );
        new_affine_impl!(
            ($($privacy)*), $name, $name_affine, $base, $scalar, $curve_type,
            $unchecked_ctor
        );
        new_maps_impl!($name, $name_affine, $base);
    };
}

#[cfg(test)]
impl EpAffine {
    pub(crate) fn raw_xy_for_tests(&self) -> (Fp, Fp) {
        (self.x, self.y)
    }
}

#[cfg(test)]
impl EqAffine {
    pub(crate) fn raw_xy_for_tests(&self) -> (Fq, Fq) {
        (self.x, self.y)
    }
}

macro_rules! mul_curve_specific {
    ($name:ident, $base:ident, special_a0_b5, jacobian, $point:expr, $scalar:expr) => {{
        let point = crate::curves::projective::Projective::<$base>::from_jacobian(
            $point.x, $point.y, $point.z,
        );
        let (x, y, z) = crate::scalar_mul::multiply(&point, &$scalar.to_repr()).to_jacobian();
        $name { x, y, z }
    }};
    ($name:ident, $base:ident, special_a0_b5, affine, $point:expr, $scalar:expr) => {{
        let point = crate::curves::projective::Projective::<$base>::from_affine(
            $point.x,
            $point.y,
            $point.is_identity(),
        );
        let (x, y, z) = crate::scalar_mul::multiply(&point, &$scalar.to_repr()).to_jacobian();
        $name { x, y, z }
    }};
    ($name:ident, $base:ident, general, $coord:ident, $point:expr, $scalar:expr) => {{
        let mut acc = $name::identity();

        // This is a simple double-and-add implementation of point
        // multiplication, moving from most significant to least
        // significant bit of the scalar.
        //
        // We don't use `PrimeFieldBits::to_le_bits` here, because that would
        // force users of this crate to depend on `bitvec` where they otherwise
        // might not need to.
        //
        // NOTE: We skip the leading bit because it's always unset (we are turning
        // the 32-byte repr into 256 bits, and $scalar::NUM_BITS = 255).
        for bit in $scalar
            .to_repr()
            .iter()
            .rev()
            .flat_map(|byte| (0..8).rev().map(move |i| Choice::from((byte >> i) & 1u8)))
            .skip(1)
        {
            acc = acc.double();
            acc = $name::conditional_select(&acc, &(acc + $point), bit);
        }

        acc
    }};
}

new_curve_impl!(
    (pub),
    Ep,
    EpAffine,
    IsoEp,
    Fp,
    Fq,
    "pallas",
    [0, 0, 0, 0],
    [5, 0, 0, 0],
    special_a0_b5,
    with_unchecked_ctor,
    glv
);
new_curve_impl!(
    (pub),
    Eq,
    EqAffine,
    IsoEq,
    Fq,
    Fp,
    "vesta",
    [0, 0, 0, 0],
    [5, 0, 0, 0],
    special_a0_b5,
    with_unchecked_ctor,
    glv
);
new_curve_impl!(
    (pub(crate)),
    IsoEp,
    IsoEpAffine,
    Ep,
    Fp,
    Fq,
    "iso-pallas",
    [
        0x92bb4b0b657a014b,
        0xb74134581a27a59f,
        0x49be2d7258370742,
        0x18354a2eb0ea8c9c,
    ],
    [1265, 0, 0, 0],
    general,
    without_unchecked_ctor,
    native
);
new_curve_impl!(
    (pub(crate)),
    IsoEq,
    IsoEqAffine,
    Eq,
    Fq,
    Fp,
    "iso-vesta",
    [
        0xc515ad7242eaa6b1,
        0x9673928c7d01b212,
        0x81639c4d96f78773,
        0x267f9b2ee592271a,
    ],
    [1265, 0, 0, 0],
    general,
    without_unchecked_ctor,
    native
);

#[cfg(test)]
#[test]
fn native_scalar_multiplication_matches_jacobian_fallback() {
    macro_rules! check {
        ($name:ident, $base:ident, $scalar:ident) => {{
            let generator = $name::generator();
            let mut scalar = -$scalar::ONE;
            for _ in 0..16 {
                for point in [
                    $name::identity(),
                    generator,
                    -generator,
                    generator.double() + generator,
                ] {
                    let affine = point.to_affine();
                    let jacobian_product =
                        mul_curve_specific!($name, $base, general, jacobian, point, scalar);
                    let affine_product =
                        mul_curve_specific!($name, $base, general, affine, affine, scalar);
                    assert_eq!(point * scalar, jacobian_product);
                    assert_eq!(affine * scalar, affine_product);
                    assert_eq!(jacobian_product, affine_product);
                }
                scalar = scalar.square() + $scalar::ONE;
            }
        }};
    }
    check!(Ep, Fp, Fq);
    check!(Eq, Fq, Fp);
}

impl Ep {
    /// Z = -13
    pub const Z: Fp = Fp::from_raw([
        0x992d30ecfffffff4,
        0x224698fc094cf91b,
        0x0000000000000000,
        0x4000000000000000,
    ]);

    /// `(F::ROOT_OF_UNITY.invert().unwrap() * z).sqrt().unwrap()`
    pub const THETA: Fp = Fp::from_raw([
        0xca330bcc09ac318e,
        0x51f64fc4dc888857,
        0x4647aef782d5cdc8,
        0x0f7bdb65814179b4,
    ]);
}

impl Eq {
    /// Z = -13
    pub const Z: Fq = Fq::from_raw([
        0x8c46eb20fffffff4,
        0x224698fc0994a8dd,
        0x0000000000000000,
        0x4000000000000000,
    ]);

    /// `(F::ROOT_OF_UNITY.invert().unwrap() * z).sqrt().unwrap()`
    pub const THETA: Fq = Fq::from_raw([
        0x632cae9872df1b5d,
        0x38578ccadf03ac27,
        0x53c3808d9e2f2357,
        0x2b3483a1ee9a382f,
    ]);
}

/// The size of each point type: two base field elements of 32 bytes for the
/// affine types, three for the Jacobian ones, with no padding, so that tables of
/// points cost what their coordinates do.
#[cfg(test)]
mod size_tests {
    use super::{Ep, EpAffine, Eq, EqAffine, IsoEp, IsoEpAffine, IsoEq, IsoEqAffine};
    use core::mem::size_of;

    #[test]
    fn affine_points_are_64_bytes() {
        assert_eq!(size_of::<EpAffine>(), 64);
        assert_eq!(size_of::<EqAffine>(), 64);
        assert_eq!(size_of::<IsoEpAffine>(), 64);
        assert_eq!(size_of::<IsoEqAffine>(), 64);
    }

    #[test]
    fn jacobian_points_are_96_bytes() {
        assert_eq!(size_of::<Ep>(), 96);
        assert_eq!(size_of::<Eq>(), 96);
        assert_eq!(size_of::<IsoEp>(), 96);
        assert_eq!(size_of::<IsoEq>(), 96);
    }
}

#[cfg(all(test, feature = "zeroize"))]
mod zeroize_tests {
    use group::{CurveAffine, Group};
    use zeroize::Zeroize;

    use super::{Ep, EpAffine, Eq, EqAffine};

    #[test]
    fn jacobian_points_zeroize_to_identity() {
        let mut p = Ep::generator();
        p.zeroize();
        assert!(bool::from(p.is_identity()));

        let mut q = Eq::generator();
        q.zeroize();
        assert!(bool::from(q.is_identity()));
    }

    #[test]
    fn affine_points_zeroize_to_identity() {
        let mut p = EpAffine::generator();
        p.zeroize();
        assert!(bool::from(p.is_identity()));

        let mut q = EqAffine::generator();
        q.zeroize();
        assert!(bool::from(q.is_identity()));
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::{Ep, EpAffine, Eq, EqAffine, Fp, Fq};
    use crate::arithmetic::{CurveAffine, CurveExt};
    use group::{Curve, CurveAffine as _, Group};
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;
    use std::vec::Vec;

    mod normalization;

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

    fn check_vartime_normalization<C: CurveExt>(points: &[C]) {
        for point in points {
            assert_eq!(point.to_affine_vartime(), point.to_affine());
        }

        let mut vartime = vec![C::Affine::generator(); points.len()];
        let mut constant_time = vec![C::Affine::generator(); points.len()];
        C::batch_normalize_vartime(points, &mut vartime);
        C::batch_normalize(points, &mut constant_time);
        assert_eq!(vartime, constant_time);
        for (point, affine) in points.iter().zip(vartime.iter()) {
            assert_eq!(*affine, point.to_affine());
        }
    }

    fn check_curve_normalization<C: CurveExt>() {
        let generator = C::generator();
        let double = generator.double();
        let triple = double + generator;
        let identity = C::identity();

        check_vartime_normalization::<C>(&[]);
        check_vartime_normalization(&[identity]);
        check_vartime_normalization(&[generator]);
        check_vartime_normalization(&[identity, identity, identity]);
        check_vartime_normalization(&[
            identity, generator, double, identity, triple, generator, identity,
        ]);
        check_vartime_normalization(&[generator, double, triple]);

        let mut rng = XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);
        for len in [1, 2, 3, 7, 16, 31] {
            let points: Vec<C> = (0..len)
                .map(|i| {
                    if i % 3 == 0 {
                        C::identity()
                    } else {
                        C::try_random(&mut rng).unwrap().double()
                    }
                })
                .collect();
            check_vartime_normalization(&points);
        }
    }

    #[test]
    fn vartime_normalization_matches_group_curve() {
        check_curve_normalization::<Ep>();
        check_curve_normalization::<Eq>();
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
