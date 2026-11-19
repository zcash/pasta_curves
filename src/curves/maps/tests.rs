//! The maps between the Jacobian and affine models: `to_affine` and the batch
//! normalizations, constant-time and variable-time.

use crate::arithmetic::CurveExt;
use crate::curves::{Ep, Eq};
use group::CurveAffine as _;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use std::vec::Vec;

mod normalization;

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
        0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
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
