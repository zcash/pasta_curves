use alloc::vec::Vec;

use ff::{Field, FromUniformBytes, PrimeField};
use group::CurveAffine as _;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

use super::{MsmCurve, WINDOWS};
use crate::arithmetic::CurveExt;
use crate::{pallas, vesta};

fn check<C: MsmCurve>()
where
    C::ScalarExt: FromUniformBytes<64>,
{
    let mut rng = XorShiftRng::seed_from_u64(0x6c76);
    let bases: Vec<C::AffineExt> = (0..5)
        .map(|_| C::random(&mut rng).to_affine())
        .chain(core::iter::once(C::identity().to_affine()))
        .collect();
    let table = C::try_prepare_fixed_base_sum(&bases).expect("Pasta has a fixed-base backend");
    assert_eq!(table.bases(), bases.len());
    let empty = C::try_prepare_fixed_base_sum(&[]).expect("Pasta supports empty fixed-base sums");
    assert_eq!(empty.bases(), 0);
    assert_eq!(empty.sum_vartime(&[]), C::identity());
    assert_eq!(empty.sum_batch_vartime(&[&[], &[]]), [C::identity(); 2]);

    let mut edges = Vec::from([
        C::ScalarExt::ZERO,
        C::ScalarExt::ONE,
        -C::ScalarExt::ONE,
        C::ScalarExt::from(63),
        C::ScalarExt::from(64),
        C::ScalarExt::from(65),
        -C::ScalarExt::from(64),
        C::ScalarExt::from(u64::MAX),
        C::ScalarExt::from_u128(u128::MAX),
    ]);
    edges.extend((0..64).map(|_| C::ScalarExt::random(&mut rng)));

    for chunk in edges.chunks(bases.len()) {
        let expected: C = chunk.iter().zip(&bases).map(|(k, p)| *p * k).sum();
        assert_eq!(table.sum_vartime(chunk), expected);
    }
    // Fewer scalars than bases leaves the rest at zero.
    for len in 0..=bases.len() {
        let scalars = &edges[edges.len() - len..];
        let expected: C = scalars.iter().zip(&bases).map(|(k, p)| *p * k).sum();
        assert_eq!(table.sum_vartime(scalars), expected, "{len} scalars");
    }
    assert_eq!(WINDOWS, 19);

    // Exercise carries through every signed window independently of the
    // GLV decomposition's distribution of half-scalars.
    let positioned = super::FixedBaseSum::<C>::new(&bases[..1]);
    for half in [63, 64, 65, (1u128 << 127) - 1, u128::MAX] {
        for negative in [false, true] {
            for endo in [false, true] {
                let mut actual = C::identity();
                positioned.for_each_term(0, negative, half, endo, |point, negate| {
                    if negate {
                        actual -= point;
                    } else {
                        actual += point;
                    }
                });
                let mut expected = bases[0] * C::ScalarExt::from_u128(half);
                if endo {
                    expected = expected.endo();
                }
                if negative {
                    expected = -expected;
                }
                assert_eq!(actual, expected);
            }
        }
    }
}

#[test]
fn fixed_base_sum_matches_native_mul() {
    check::<pallas::Point>();
    check::<vesta::Point>();
}

/// The batched sums equal the per-sum ones, including identity bases,
/// zero and short scalar vectors, and pairs that hit the equal-x fallback.
fn check_batch<C: MsmCurve>()
where
    C::ScalarExt: FromUniformBytes<64>,
{
    let mut rng = XorShiftRng::seed_from_u64(0x6261);
    let p = C::random(&mut rng).to_affine();
    // A repeated base with equal or opposite scalars makes identical or
    // negated terms, so the trees meet doublings and cancellations.
    let bases = [
        p,
        p,
        C::random(&mut rng).to_affine(),
        C::identity().to_affine(),
    ];
    let table = C::try_prepare_fixed_base_sum(&bases).expect("Pasta has a fixed-base backend");
    let k = C::ScalarExt::random(&mut rng);
    let mut columns: Vec<Vec<C::ScalarExt>> = Vec::from([
        Vec::new(),
        Vec::from([C::ScalarExt::ZERO]),
        Vec::from([k, k]),
        Vec::from([k, -k]),
        Vec::from([k, k, k, k]),
        Vec::from([C::ScalarExt::ONE, -C::ScalarExt::ONE, C::ScalarExt::ONE]),
    ]);
    columns.extend((0..20).map(|i| {
        (0..(i % 5))
            .map(|_| C::ScalarExt::random(&mut rng))
            .collect()
    }));
    for count in [0, 1, 2, 7, columns.len()] {
        let slices: Vec<&[C::ScalarExt]> = columns[..count].iter().map(Vec::as_slice).collect();
        let expected: Vec<C> = slices
            .iter()
            .map(|s| s.iter().zip(&bases).map(|(k, p)| *p * k).sum())
            .collect();
        for (scalars, expected) in slices.iter().zip(&expected) {
            assert_eq!(table.sum_vartime(scalars), *expected);
        }
        assert_eq!(table.sum_batch_vartime(&slices), expected, "{count} sums");
    }
}

#[test]
fn fixed_base_sum_batch_matches_single() {
    check_batch::<pallas::Point>();
    check_batch::<vesta::Point>();
}

#[test]
#[should_panic(expected = "at most one scalar per prepared base")]
fn fixed_base_sum_rejects_extra_scalars() {
    let table = pallas::Point::try_prepare_fixed_base_sum(&[pallas::Affine::generator()])
        .expect("Pasta has a fixed-base backend");
    let _ = table.sum_vartime(&[pallas::Scalar::ONE, pallas::Scalar::ONE]);
}
