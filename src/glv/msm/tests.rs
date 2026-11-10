use super::*;

/// Deterministic full-width scalars for known-answer tests.
#[cfg(feature = "multicore")]
pub(crate) fn scalars<F: PrimeField>(n: u64) -> impl Iterator<Item = F> {
    (0..n).map(|i| {
        (F::from(0x9E37_79B9_7F4A_7C15u64 + i).square() + F::from(0x0123_4567_89AB_CDEFu64))
            .square()
            + F::from(i)
    })
}

fn batch_invert_and_add_two_lanes_matches_individual<F>()
where
    F: Field + From<u64>,
{
    const LENGTHS: [usize; 13] = [0, 1, 2, 3, 31, 32, 33, 63, 64, 65, 255, 256, 257];

    for length in LENGTHS {
        let mut additions = (0..length)
            .map(|index| PendingAffineAddition {
                output: index,
                x_sum: F::from(u64::try_from(index + 3).unwrap()),
                numerator: F::from(u64::try_from(index + 2).unwrap()),
                denominator: F::from(u64::try_from(index + 1).unwrap()),
            })
            .collect::<Vec<_>>();
        let mut points = (0..length)
            .map(|index| AffinePoint {
                x: F::from(u64::try_from(index + 5).unwrap()),
                y: F::from(u64::try_from(index + 7).unwrap()),
            })
            .collect::<Vec<_>>();
        let expected = additions
            .iter()
            .zip(&points)
            .map(|(addition, left)| {
                let slope = addition.numerator * addition.denominator.invert().unwrap();
                let x = slope.square() - addition.x_sum;
                let y = slope * (left.x - x) - left.y;
                AffinePoint { x, y }
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
    batch_invert_and_add_two_lanes_matches_individual::<crate::Fp>();
    batch_invert_and_add_two_lanes_matches_individual::<crate::Fq>();
}

#[test]
fn multiexp_backend_selection() {
    assert_eq!(default_multiexp_window_bits(31), Some(3));
    assert_eq!(default_multiexp_window_bits(32), Some(4));
    assert_eq!(default_multiexp_window_bits(54), Some(4));
    assert_eq!(default_multiexp_window_bits(55), Some(5));
    assert_eq!(default_multiexp_window_bits(2_980), Some(8));
    assert_eq!(default_multiexp_window_bits(2_981), Some(9));
    assert_eq!(default_multiexp_window_bits(8_103), Some(9));
    assert_eq!(default_multiexp_window_bits(8_104), Some(10));
    assert_eq!(default_multiexp_window_bits(u32::MAX as usize), Some(23));

    assert_eq!(estimated_signed_booth_work(2_150, 256, 8, 1), Some(79_654));
    assert_eq!(
        estimated_signed_booth_work(4_300, GLV_COMPONENT_BITS, 8, 1),
        Some(73_016)
    );
    assert_eq!(estimated_signed_booth_work(5_678, 256, 9, 1), Some(179_762));
    assert_eq!(
        estimated_signed_booth_work(11_356, GLV_COMPONENT_BITS, 9, 1),
        Some(178_146)
    );

    assert_eq!(multiexp_window_bits::<pallas::Point>(2_150, 1), Some(8));
    assert_eq!(multiexp_window_bits::<pallas::Point>(2_990, 1), Some(9));
    assert_eq!(multiexp_window_bits::<pallas::Point>(3_924, 1), Some(9));
    assert_eq!(multiexp_window_bits::<pallas::Point>(3_925, 1), Some(10));
    assert_eq!(multiexp_window_bits::<pallas::Point>(5_678, 1), Some(10));
    assert_eq!(multiexp_window_bits::<pallas::Point>(5_678, 8), Some(9));

    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(255, 1), None);
    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(2_150, 1), Some(9));
    assert_eq!(glv_multiexp_window_bits::<vesta::Point>(5_678, 1), Some(11));

    assert_eq!(estimated_signed_booth_work(2_150, 256, 8, 8), Some(12_670));
    assert_eq!(
        estimated_signed_booth_work(4_300, GLV_COMPONENT_BITS, 8, 8),
        Some(9_288)
    );
    assert_eq!(estimated_signed_booth_work(5_678, 256, 9, 8), Some(25_336));
    assert_eq!(
        estimated_signed_booth_work(11_356, GLV_COMPONENT_BITS, 9, 8),
        Some(23_916)
    );
    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(2_150, 8), Some(8));
    assert_eq!(glv_multiexp_window_bits::<vesta::Point>(5_678, 8), Some(9));

    // At sufficiently large sizes with whole waves on every worker,
    // doubling the term count outweighs the shorter components at both
    // candidate widths, and an overflowing term count is never planned.
    assert_eq!(
        glv_multiexp_window_bits::<pallas::Point>(1_000_000, 8),
        None
    );
    assert_eq!(
        glv_multiexp_window_bits::<pallas::Point>(usize::MAX, 8),
        None
    );
}

#[test]
fn multiexp_component_guard_returns_none_out_of_bounds() {
    let in_bounds = GLV_COMPONENT_BITS - 1;
    let out_of_bounds = 1u128 << GLV_COMPONENT_BITS;

    assert!(checked_signed_magnitudes(((false, 0), (true, 1))).is_some());
    assert!(checked_signed_magnitudes(((false, 1u128 << in_bounds), (false, 0))).is_some());
    assert!(checked_signed_magnitudes(((false, out_of_bounds), (false, 0))).is_none());
    assert!(checked_signed_magnitudes(((false, 0), (true, out_of_bounds))).is_none());
}

#[cfg(feature = "multicore")]
fn parallel_multiexp_matches_serial<C: MsmCurve>() {
    const TEST_TERMS: usize = 64;
    const WINDOW_BITS: [usize; 6] = [1, 3, 6, 8, 9, 10];

    let generator = C::generator();
    let mut points = (0..TEST_TERMS)
        .map(|i| generator * C::ScalarExt::from(i as u64 + 1))
        .collect::<Vec<_>>();
    let mut test_scalars = scalars::<C::ScalarExt>(TEST_TERMS as u64).collect::<Vec<_>>();

    // Exercise identity terms, zero scalars, and cancellation alongside
    // the deterministic full-width inputs.
    points[0] = C::identity();
    test_scalars[0] = C::ScalarExt::ONE;
    points[1] = generator;
    test_scalars[1] = C::ScalarExt::ZERO;
    points[2] = generator;
    test_scalars[2] = C::ScalarExt::ONE;
    points[3] = -generator;
    test_scalars[3] = C::ScalarExt::ONE;

    let mut bases = alloc::vec![C::AffineExt::identity(); TEST_TERMS];
    C::batch_normalize(&points, &mut bases);
    let components = test_scalars
        .iter()
        .map(decompose::<C>)
        .map(checked_signed_magnitudes)
        .collect::<Option<Vec<_>>>()
        .expect("valid scalar decompositions fit the GLV component bound");
    let bases = multiexp_bases::<C>(&bases);

    for window_bits in WINDOW_BITS {
        assert_eq!(
            multiexp_parallel::<C>(&components, &bases, window_bits),
            multiexp_serial::<C>(&components, &bases, window_bits),
            "parallel and serial MSMs differ at window width {window_bits}",
        );
    }
}

#[cfg(feature = "multicore")]
#[test]
fn parallel_multiexp_matches_serial_pallas() {
    parallel_multiexp_matches_serial::<pallas::Point>();
}

#[cfg(feature = "multicore")]
#[test]
fn parallel_multiexp_matches_serial_vesta() {
    parallel_multiexp_matches_serial::<vesta::Point>();
}

fn assert_affine_bucket_results_match_native<C: MsmCurve>(
    case: &str,
    reduced: Vec<Option<AffinePoint<C::Base>>>,
    source: &[Vec<C>],
) {
    assert_eq!(reduced.len(), source.len());
    for (actual, bucket) in reduced.into_iter().zip(source) {
        let actual = match actual {
            Some(point) => C::from(C::affine_unchecked(point.x, point.y)),
            None => C::identity(),
        };
        let expected = bucket.iter().copied().sum::<C>();
        assert_eq!(actual, expected, "affine reduction mismatch in case {case}");
    }
}

fn assert_batch_affine_buckets_match_native<C: MsmCurve>(case: &str, source: &[Vec<C>]) {
    let mut points = Vec::new();
    let mut offsets = Vec::with_capacity(source.len() + 1);
    offsets.push(0);
    for bucket in source {
        for point in bucket {
            let affine = C::AffineExt::from(*point);
            let (x, y) = C::affine_xy(&affine);
            points.push(AffinePoint { x, y });
        }
        offsets.push(points.len());
    }

    let reduced = reduce_affine_buckets(points.clone(), offsets.clone())
        .unwrap_or_else(|| panic!("valid curve points must reduce in case {case}"));
    let in_place = reduce_affine_buckets_in_place(points.clone(), offsets.clone())
        .or_else(|| reduce_affine_buckets(points, offsets))
        .unwrap_or_else(|| panic!("valid curve points must reduce in case {case}"));

    for reduced in [reduced, in_place] {
        assert_affine_bucket_results_match_native::<C>(case, reduced, source);
    }
}

fn batch_affine_buckets_match_native<C: MsmCurve>() {
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

    for (case, source) in cases {
        assert_batch_affine_buckets_match_native::<C>(case, &source);
    }

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
            let (x, y) = C::affine_xy(&affine);
            clean_points.push(AffinePoint { x, y });
        }
        clean_offsets.push(clean_points.len());
    }
    let clean_reduced = reduce_affine_buckets_in_place(clean_points, clean_offsets)
        .expect("distinct points should use the in-place reducer");
    assert_affine_bucket_results_match_native::<C>(
        "direct successful in-place reduction",
        clean_reduced,
        &clean_source,
    );

    let affine = C::AffineExt::from(generator);
    let (x, y) = C::affine_xy(&affine);
    let duplicate = AffinePoint { x, y };
    assert!(
        reduce_affine_buckets_in_place(alloc::vec![duplicate, duplicate], alloc::vec![0, 2],)
            .is_none()
    );
}

fn batch_inversion_zero_denominator_is_failure_atomic_for<F: Field>() {
    let two = F::ONE.double();
    let three = two + F::ONE;
    let four = two.double();
    let five = four + F::ONE;
    let cases = [
        ("single first lane", alloc::vec![F::ZERO]),
        ("first lane", alloc::vec![F::ZERO, F::ONE]),
        ("second lane", alloc::vec![F::ONE, F::ZERO]),
        (
            "odd trailing first lane",
            alloc::vec![F::ONE, F::ONE, F::ZERO],
        ),
        (
            "late first lane after prefix staging",
            alloc::vec![two, three, four, five, F::ZERO],
        ),
        (
            "late second lane after prefix staging",
            alloc::vec![two, three, four, F::ZERO],
        ),
    ];

    for (case, denominators) in cases {
        let mut additions = denominators
            .into_iter()
            .enumerate()
            .map(|(output, denominator)| PendingAffineAddition {
                output,
                x_sum: F::ZERO,
                numerator: F::ONE,
                denominator,
            })
            .collect::<Vec<_>>();
        let mut points = alloc::vec![
            AffinePoint {
                x: F::ONE,
                y: F::ONE,
            };
            additions.len()
        ];
        let original_points = points.clone();

        assert!(
            batch_invert_and_add(&mut additions, &mut points).is_none(),
            "zero denominator was accepted in case {case}"
        );
        for (actual, original) in points.iter().zip(original_points) {
            assert_eq!(actual.x, original.x, "x changed in case {case}");
            assert_eq!(actual.y, original.y, "y changed in case {case}");
        }
    }
}

#[test]
fn batch_inversion_zero_denominator_is_failure_atomic() {
    batch_inversion_zero_denominator_is_failure_atomic_for::<crate::Fp>();
    batch_inversion_zero_denominator_is_failure_atomic_for::<crate::Fq>();
}

macro_rules! glv_tests {
    ($mod_name:ident, $curve:ty) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn batch_affine_buckets() {
                batch_affine_buckets_match_native::<$curve>();
            }
        }
    };
}

glv_tests!(pallas_glv, pallas::Point);
glv_tests!(vesta_glv, vesta::Point);
