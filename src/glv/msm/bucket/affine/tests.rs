use ff::Field;

use super::super::{affine_xy, bucket_point};
use super::*;
use crate::glv::GlvParams;
use crate::{pallas, vesta};

fn batch_invert_and_add_two_lanes_matches_individual<A>()
where
    A: CurveAffine<Base: VartimeField>,
{
    let f = |n: usize| <A::Base as From<u64>>::from(u64::try_from(n).unwrap());
    const LENGTHS: [usize; 13] = [0, 1, 2, 3, 31, 32, 33, 63, 64, 65, 255, 256, 257];

    for length in LENGTHS {
        let mut additions = (0..length)
            .map(|index| PendingAffineAddition {
                output: index,
                x_sum: f(index + 3),
                numerator: f(index + 2),
                denominator: f(index + 1),
            })
            .collect::<Vec<_>>();
        let mut points = (0..length)
            .map(|index| Coordinates::<A> {
                x: f(index + 5),
                y: f(index + 7),
            })
            .collect::<Vec<_>>();
        let expected = additions
            .iter()
            .zip(&points)
            .map(|(addition, left)| {
                let slope = addition.numerator * addition.denominator.invert().unwrap();
                let x = slope.square() - addition.x_sum;
                let y = slope * (left.x - x) - left.y;
                Coordinates::<A> { x, y }
            })
            .collect::<Vec<_>>();

        batch_invert_and_add(&mut additions, &mut points)
            .expect("nonzero denominators must be invertible");
        for (index, (actual, expected)) in points.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.x, expected.x,
                "two-lane affine x mismatch at length {length}, index {index}"
            );
            assert_eq!(
                actual.y, expected.y,
                "two-lane affine y mismatch at length {length}, index {index}"
            );
        }
    }
}

#[test]
fn batch_invert_and_add_two_lanes() {
    batch_invert_and_add_two_lanes_matches_individual::<pallas::Affine>();
    batch_invert_and_add_two_lanes_matches_individual::<vesta::Affine>();
}

fn assert_affine_bucket_results_match_native<C>(
    case: &str,
    reduced: Vec<Option<Coordinates<C::AffineExt>>>,
    source: &[Vec<C>],
) where
    C: GlvParams,
{
    assert_eq!(reduced.len(), source.len());
    for (actual, bucket) in reduced.into_iter().zip(source) {
        let actual = bucket_point::<C>(actual);
        let expected = bucket.iter().copied().sum::<C>();
        assert_eq!(actual, expected, "affine reduction mismatch in case {case}");
    }
}

fn assert_batch_affine_buckets_match_native<C>(
    case: &str,
    source: &[Vec<C>],
    scratch: &mut ReduceScratch<C::AffineExt>,
) where
    C: GlvParams,
{
    let mut points = Vec::new();
    let mut offsets = Vec::with_capacity(source.len() + 1);
    offsets.push(0);
    for bucket in source {
        for point in bucket {
            let affine = C::AffineExt::from(*point);
            let (x, y) = affine_xy::<C>(&affine);
            points.push(Coordinates { x, y });
        }
        offsets.push(points.len());
    }

    reduce_affine_buckets_reusing::<_, false>(&mut points, &mut offsets, scratch)
        .unwrap_or_else(|| panic!("valid curve points must reduce in case {case}"));
    // Affine coordinates are unique and an empty bucket is `None`, so equality
    // with the plain sum of each bucket pins the reduced points exactly.
    let reduced = scratch.buckets.clone();
    assert_affine_bucket_results_match_native::<C>(case, reduced, source);
}

fn batch_affine_buckets_match_native<C>()
where
    C: GlvParams,
{
    let generator = C::generator();
    let two = generator.double();
    let three = two + generator;
    let four = three + generator;
    let five = four + generator;
    let cases = [
        ("empty bucket", alloc::vec![Vec::new()]),
        ("singleton bucket", alloc::vec![alloc::vec![generator]]),
        (
            "first-level doubling",
            alloc::vec![alloc::vec![generator, generator]],
        ),
        (
            "first-level inverse pair",
            alloc::vec![alloc::vec![generator, -generator]],
        ),
        // The first level adds G + 2G, then pairs the resulting 3G
        // with the carried 3G. The fallback therefore first occurs at
        // the second level, after the first level has been committed.
        (
            "second-level doubling",
            alloc::vec![alloc::vec![generator, two, three]],
        ),
        // As above, but the carried point is -3G, so the second-level
        // exceptional pair cancels to the identity.
        (
            "second-level inverse pair",
            alloc::vec![alloc::vec![generator, two, -three]],
        ),
        (
            "odd bucket",
            alloc::vec![alloc::vec![generator, two, three, four, five]],
        ),
        (
            "one exceptional pair in a shared batch",
            alloc::vec![
                Vec::new(),
                alloc::vec![generator],
                alloc::vec![generator, two],
                alloc::vec![generator, generator],
                alloc::vec![three, four, five],
            ],
        ),
    ];

    let mut scratch = ReduceScratch::default();
    for (case, source) in cases {
        assert_batch_affine_buckets_match_native::<C>(case, &source, &mut scratch);
    }
    // Revisit a short cancellation after larger batches have populated scratch.
    assert_batch_affine_buckets_match_native::<C>(
        "reused cancellation",
        &[alloc::vec![generator, -generator], Vec::new()],
        &mut scratch,
    );

    // Powers of two keep every intermediate pair distinct, so this
    // exercises successful in-place reduction without its fallback.
    let clean_source = [
        Vec::new(),
        alloc::vec![generator],
        alloc::vec![generator, two],
        alloc::vec![generator, two, four],
        (0..9)
            .map(|bit| generator * C::ScalarExt::from(1u64 << bit))
            .collect(),
    ];
    let mut clean_points = Vec::new();
    let mut clean_offsets = Vec::with_capacity(clean_source.len() + 1);
    clean_offsets.push(0);
    for bucket in &clean_source {
        for point in bucket {
            let affine = C::AffineExt::from(*point);
            let (x, y) = affine_xy::<C>(&affine);
            clean_points.push(Coordinates { x, y });
        }
        clean_offsets.push(clean_points.len());
    }
    let clean_reduced = reduce_affine_buckets(clean_points, clean_offsets)
        .expect("distinct points should use the in-place reducer");
    assert_affine_bucket_results_match_native::<C>(
        "direct successful in-place reduction",
        clean_reduced,
        &clean_source,
    );
}

#[test]
fn affine_buckets() {
    batch_affine_buckets_match_native::<pallas::Point>();
    batch_affine_buckets_match_native::<vesta::Point>();
}

#[test]
fn failed_inversion_does_not_change_points() {
    for zero_at in [0, 1, 2, 4, 8] {
        let mut points = (0..9)
            .map(|i| Coordinates::<pallas::Affine> {
                x: crate::Fp::from(i),
                y: crate::Fp::from(i + 1),
            })
            .collect::<Vec<_>>();
        let original = points.clone();
        let mut pending = (0..9)
            .map(|i| PendingAffineAddition {
                output: i,
                x_sum: crate::Fp::ONE,
                numerator: crate::Fp::ONE,
                denominator: if i == zero_at {
                    crate::Fp::ZERO
                } else {
                    crate::Fp::ONE
                },
            })
            .collect::<Vec<_>>();
        assert!(batch_invert_and_add(&mut pending, &mut points).is_none());
        for (point, original) in points.iter().zip(original) {
            assert_eq!(point.x, original.x);
            assert_eq!(point.y, original.y);
        }
    }
}
