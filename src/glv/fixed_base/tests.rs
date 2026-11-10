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

fn check_dispatch<C: MsmCurve>()
where
    C::ScalarExt: FromUniformBytes<64>,
{
    use crate::arithmetic::PreparedFixedBaseSum;

    let mut rng = XorShiftRng::seed_from_u64(0x702);
    let bases: Vec<_> = (0..7)
        .map(|_| C::random(&mut rng).to_affine())
        .chain(core::iter::once(C::identity().to_affine()))
        .collect();
    let table = super::FixedBaseSum::<C>::new(&bases);
    let columns: Vec<Vec<_>> = (0..19)
        .map(|i| {
            (0..(i % (bases.len() + 1)))
                .map(|_| C::ScalarExt::random(&mut rng))
                .collect()
        })
        .collect();
    let slices: Vec<_> = columns.iter().map(Vec::as_slice).collect();
    let expected: Vec<C> = columns
        .iter()
        .map(|scalars| scalars.iter().zip(&bases).map(|(k, p)| *p * k).sum())
        .collect();
    assert_eq!(table.sum_batch_vartime(&slices), expected);

    #[cfg(all(
        feature = "asm",
        feature = "sqrt-table",
        target_arch = "x86_64",
        not(pasta_curves_noasm)
    ))]
    let table = {
        let mut table = table;
        assert_eq!(
            table.points29.is_some(),
            crate::arithmetic::avx2_sqrt::available(),
        );
        // Force the same unsupported-CPU branch without changing global CPU
        // detection or requiring AVX2 on the machine running the tests.
        table.points29 = None;
        table
    };
    // Also runs with no asm, no sqrt-table, and pasta_curves_noasm, where
    // neither the vector backend nor its private coordinate adapters exist.
    assert_eq!(table.sum_batch_vartime(&slices), expected);
}

#[test]
fn fixed_base_runtime_and_feature_disabled_fallback() {
    check_dispatch::<pallas::Point>();
    check_dispatch::<vesta::Point>();
}

#[cfg(all(
    feature = "asm",
    feature = "sqrt-table",
    target_arch = "x86_64",
    not(pasta_curves_noasm)
))]
fn check_radix29_reducer<C: MsmCurve>() {
    use crate::arithmetic::avx2_chord::Point29;

    let mut rng = XorShiftRng::seed_from_u64(0x70229);
    let p = C::random(&mut rng).to_affine();
    let mut q = C::random(&mut rng).to_affine();
    while C::affine_xy(&p).0 == C::affine_xy(&q).0 {
        q = C::random(&mut rng).to_affine();
    }
    // A single bucket crosses both sides of the 32-addition threshold;
    // lengths not divisible by eight cover vector padding and odd tree tails.
    for len in [0, 1, 2, 3, 61, 62, 63, 64, 65, 66, 67, 129, 257] {
        for exceptional in 0..5 {
            let mut inputs: Vec<_> = (0..len).map(|_| C::random(&mut rng).to_affine()).collect();
            if len >= 4 {
                match exceptional {
                    1 => inputs[..2].copy_from_slice(&[p, p]),
                    2 => inputs[..2].copy_from_slice(&[p, -p]),
                    // The first vector level succeeds, then its outputs
                    // meet equal-x doubling or cancellation one level later.
                    3 => inputs[..4].copy_from_slice(&[p, q, p, q]),
                    4 => inputs[..4].copy_from_slice(&[p, q, -p, -q]),
                    _ => {}
                }
            }
            let expected: C = inputs.iter().map(|p| C::from(*p)).sum();
            let points = inputs
                .iter()
                .map(|point| {
                    let (x, y) = C::affine_xy(point);
                    Point29 {
                        x: C::to29(&x),
                        y: C::to29(&y),
                    }
                })
                .collect();
            // Empty buckets on both sides ensure offsets stay aligned after
            // cancellations and after the vector-to-scalar transition.
            let sums = super::super::msm::reduce_affine_buckets29::<C>(
                points,
                Vec::from([0, 0, len, len]),
            )
            .expect("valid points have complete affine sums");
            let actual: Vec<_> = sums
                .into_iter()
                .map(|p| p.map_or_else(C::identity, |p| C::from(C::affine_unchecked(p.x, p.y))))
                .collect();
            assert_eq!(
                actual,
                [C::identity(), expected, C::identity()],
                "len={len}, case={exceptional}"
            );
        }
    }
}

#[test]
#[cfg(all(
    feature = "asm",
    feature = "sqrt-table",
    target_arch = "x86_64",
    not(pasta_curves_noasm)
))]
fn fixed_base_common_radix29_thresholds_and_exceptions() {
    check_radix29_reducer::<pallas::Point>();
    check_radix29_reducer::<vesta::Point>();
}
