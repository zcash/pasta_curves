use super::*;
use crate::{pallas, vesta};
use ff::{Field, PrimeField, WithSmallOrderMulGroup};
use group::CurveAffine;

const BATCH_SIZES: [usize; 15] = [0, 1, 2, 3, 7, 8, 15, 16, 17, 63, 64, 65, 127, 128, 129];

fn batch_mul_same_scalar_matches_native<C: CurveExt>() {
    let full_width = (C::ScalarExt::from(0x9E37_79B9_7F4A_7C15u64).square()
        + C::ScalarExt::from(0x0123_4567_89AB_CDEFu64))
    .square();
    let scalars = [
        C::ScalarExt::ZERO,
        C::ScalarExt::ONE,
        -C::ScalarExt::ONE,
        C::ScalarExt::from(2),
        C::ScalarExt::ZETA,
        -C::ScalarExt::ZETA,
        C::ScalarExt::ZETA + C::ScalarExt::ONE,
        C::ScalarExt::from(u64::MAX),
        C::ScalarExt::from_u128((1u128 << 127) - 1),
        C::ScalarExt::from_u128(1u128 << 127),
        full_width,
    ];

    for size in BATCH_SIZES {
        let projective: alloc::vec::Vec<C> = (0..size)
            .map(|i| {
                if size > 2 && i == size / 2 {
                    C::identity()
                } else {
                    C::generator()
                        * (C::ScalarExt::from(i as u64 + 1).square()
                            + C::ScalarExt::from(0xDEAD_BEEFu64))
                }
            })
            .collect();
        let points: alloc::vec::Vec<C::AffineExt> =
            projective.iter().copied().map(C::AffineExt::from).collect();
        let mut output = alloc::vec![C::identity(); size];

        for scalar in scalars {
            C::batch_mul_same_scalar_vartime(&points, &scalar, &mut output);
            for ((point, output), expected) in
                points.iter().zip(output.iter()).zip(projective.iter())
            {
                assert_eq!(*output, *point * scalar);
                assert_eq!(*output, *expected * scalar);
            }
        }
    }
}

#[test]
fn batch_mul_same_scalar_pallas() {
    batch_mul_same_scalar_matches_native::<pallas::Point>();
}

#[test]
fn batch_mul_same_scalar_vesta() {
    batch_mul_same_scalar_matches_native::<vesta::Point>();
}

#[test]
#[should_panic]
fn batch_mul_same_scalar_length_mismatch_panics() {
    let points = [pallas::Affine::generator()];
    let mut output = [];
    pallas::Point::batch_mul_same_scalar_vartime(&points, &pallas::Scalar::ONE, &mut output);
}

#[cfg(not(feature = "glv"))]
#[test]
fn unsupported_fixed_base_backend_declines() {
    assert!(pallas::Point::try_prepare_fixed_base_sum(&[]).is_none());
    assert!(vesta::Point::try_prepare_fixed_base_sum(&[]).is_none());
}
