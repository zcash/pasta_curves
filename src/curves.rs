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
    /// Constants used for computing the isogeny from IsoEp to Ep.
    pub const ISOGENY_CONSTANTS: [Fp; 13] = [
        Fp::from_raw([
            0x775f6034aaaaaaab,
            0x4081775473d8375b,
            0xe38e38e38e38e38e,
            0x0e38e38e38e38e38,
        ]),
        Fp::from_raw([
            0x8cf863b02814fb76,
            0x0f93b82ee4b99495,
            0x267c7ffa51cf412a,
            0x3509afd51872d88e,
        ]),
        Fp::from_raw([
            0x0eb64faef37ea4f7,
            0x380af066cfeb6d69,
            0x98c7d7ac3d98fd13,
            0x17329b9ec5253753,
        ]),
        Fp::from_raw([
            0xeebec06955555580,
            0x8102eea8e7b06eb6,
            0xc71c71c71c71c71c,
            0x1c71c71c71c71c71,
        ]),
        Fp::from_raw([
            0xc47f2ab668bcd71f,
            0x9c434ac1c96b6980,
            0x5a607fcce0494a79,
            0x1d572e7ddc099cff,
        ]),
        Fp::from_raw([
            0x2aa3af1eae5b6604,
            0xb4abf9fb9a1fc81c,
            0x1d13bf2a7f22b105,
            0x325669becaecd5d1,
        ]),
        Fp::from_raw([
            0x5ad985b5e38e38e4,
            0x7642b01ad461bad2,
            0x4bda12f684bda12f,
            0x1a12f684bda12f68,
        ]),
        Fp::from_raw([
            0xc67c31d8140a7dbb,
            0x07c9dc17725cca4a,
            0x133e3ffd28e7a095,
            0x1a84d7ea8c396c47,
        ]),
        Fp::from_raw([
            0x02e2be87d225b234,
            0x1765e924f7459378,
            0x303216cce1db9ff1,
            0x3fb98ff0d2ddcadd,
        ]),
        Fp::from_raw([
            0x93e53ab371c71c4f,
            0x0ac03e8e134eb3e4,
            0x7b425ed097b425ed,
            0x025ed097b425ed09,
        ]),
        Fp::from_raw([
            0x5a28279b1d1b42ae,
            0x5941a3a4a97aa1b3,
            0x0790bfb3506defb6,
            0x0c02c5bcca0e6b7f,
        ]),
        Fp::from_raw([
            0x4d90ab820b12320a,
            0xd976bbfabbc5661d,
            0x573b3d7f7d681310,
            0x17033d3c60c68173,
        ]),
        Fp::from_raw([
            0x992d30ecfffffde5,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
        ]),
    ];

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
    /// Constants used for computing the isogeny from IsoEq to Eq.
    pub const ISOGENY_CONSTANTS: [Fq; 13] = [
        Fq::from_raw([
            0x43cd42c800000001,
            0x0205dd51cfa0961a,
            0x8e38e38e38e38e39,
            0x38e38e38e38e38e3,
        ]),
        Fq::from_raw([
            0x8b95c6aaf703bcc5,
            0x216b8861ec72bd5d,
            0xacecf10f5f7c09a2,
            0x1d935247b4473d17,
        ]),
        Fq::from_raw([
            0xaeac67bbeb586a3d,
            0xd59d03d23b39cb11,
            0xed7ee4a9cdf78f8f,
            0x18760c7f7a9ad20d,
        ]),
        Fq::from_raw([
            0xfb539a6f0000002b,
            0xe1c521a795ac8356,
            0x1c71c71c71c71c71,
            0x31c71c71c71c71c7,
        ]),
        Fq::from_raw([
            0xb7284f7eaf21a2e9,
            0xa3ad678129b604d3,
            0x1454798a5b5c56b2,
            0x0a2de485568125d5,
        ]),
        Fq::from_raw([
            0xf169c187d2533465,
            0x30cd6d53df49d235,
            0x0c621de8b91c242a,
            0x14735171ee542778,
        ]),
        Fq::from_raw([
            0x6bef1642aaaaaaab,
            0x5601f4709a8adcb3,
            0xda12f684bda12f68,
            0x12f684bda12f684b,
        ]),
        Fq::from_raw([
            0x8bee58e5fb81de63,
            0x21d910aefb03b31d,
            0xd6767887afbe04d1,
            0x2ec9a923da239e8b,
        ]),
        Fq::from_raw([
            0x4986913ab4443034,
            0x97a3ca5c24e9ea63,
            0x66d1466e9de10e64,
            0x19b0d87e16e25788,
        ]),
        Fq::from_raw([
            0x8f64842c55555533,
            0x8bc32d36fb21a6a3,
            0x425ed097b425ed09,
            0x1ed097b425ed097b,
        ]),
        Fq::from_raw([
            0x58dfecce86b2745e,
            0x06a767bfc35b5bac,
            0x9e7eb64f890a820c,
            0x2f44d6c801c1b8bf,
        ]),
        Fq::from_raw([
            0xd43d449776f99d2f,
            0x926847fb9ddd76a1,
            0x252659ba2b546c7e,
            0x3d59f455cafc7668,
        ]),
        Fq::from_raw([
            0x8c46eb20fffffde5,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
        ]),
    ];

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
