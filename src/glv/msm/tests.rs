//! Differential checks for Booth recoding and the failure-atomic affine reducer.
use super::*;

// Preserved pre-port reducer: an allocating, failure-atomic arithmetic oracle.
fn original_reduce_affine_buckets<F: VartimeField, const COMPLETE: bool>(
    mut points: Vec<AffinePoint<F>>,
    mut offsets: Vec<usize>,
) -> Option<Vec<Option<AffinePoint<F>>>> {
    debug_assert!(!offsets.is_empty());
    let mut next_offsets = Vec::with_capacity(offsets.len());
    let mut pending = Vec::with_capacity(points.len() / 2);

    while offsets.windows(2).any(|range| range[1] - range[0] > 1) {
        pending.clear();
        for range in offsets.windows(2) {
            for input in (range[0]..range[1] - (range[1] - range[0]) % 2).step_by(2) {
                let left = points[input];
                let right = points[input + 1];
                let dx = right.x - left.x;
                let (numerator, denominator) = if COMPLETE && dx.is_zero_vartime() {
                    if !(right.y - left.y).is_zero_vartime() || left.y.is_zero_vartime() {
                        continue;
                    }
                    let square = left.x.square();
                    (square.double() + square, left.y.double())
                } else {
                    (right.y - left.y, dx)
                };
                pending.push(PendingAffineAddition {
                    output: input,
                    x_sum: left.x + right.x,
                    numerator,
                    denominator,
                });
            }
        }

        if batch_invert_and_add(&mut pending, &mut points).is_none() {
            if !COMPLETE {
                return original_reduce_affine_buckets::<F, true>(points, offsets);
            }
            return None;
        }

        next_offsets.clear();
        next_offsets.push(0);
        let mut additions = pending.iter().peekable();
        let mut output = 0;
        for range in offsets.windows(2) {
            let end_pairs = range[1] - (range[1] - range[0]) % 2;
            for input in (range[0]..end_pairs).step_by(2) {
                if additions
                    .peek()
                    .is_some_and(|addition| addition.output == input)
                {
                    points[output] = points[input];
                    output += 1;
                    additions.next();
                }
            }
            if end_pairs != range[1] {
                points[output] = points[end_pairs];
                output += 1;
            }
            next_offsets.push(output);
        }
        points.truncate(output);
        core::mem::swap(&mut offsets, &mut next_offsets);
    }

    Some(
        offsets
            .windows(2)
            .map(|range| (range[0] != range[1]).then(|| points[range[0]]))
            .collect(),
    )
}

fn batch_invert_and_add_two_lanes_matches_individual<F>()
where
    F: VartimeField + From<u64>,
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
fn multiexp_component_guard_returns_none_out_of_bounds() {
    let in_bounds = GLV_COMPONENT_BITS - 1;
    let out_of_bounds = 1u128 << GLV_COMPONENT_BITS;

    assert!(checked_signed_magnitudes(((false, 0), (true, 1))).is_some());
    assert!(checked_signed_magnitudes(((false, 1u128 << in_bounds), (false, 0))).is_some());
    assert!(checked_signed_magnitudes(((false, out_of_bounds), (false, 0))).is_none());
    assert!(checked_signed_magnitudes(((false, 0), (true, out_of_bounds))).is_none());
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

fn assert_batch_affine_buckets_match_native<C: MsmCurve>(
    case: &str,
    source: &[Vec<C>],
    scratch: &mut ReduceScratch<C::Base>,
) {
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

    let original = original_reduce_affine_buckets::<_, false>(points.clone(), offsets.clone())
        .unwrap_or_else(|| panic!("original reducer must succeed in case {case}"));
    let reduced = reduce_affine_buckets_reusing::<_, false>(&mut points, &mut offsets, scratch)
        .unwrap_or_else(|| panic!("valid curve points must reduce in case {case}"));
    for (actual, expected) in reduced.iter().zip(&original) {
        assert_eq!(
            actual.map(|point| (point.x, point.y)),
            expected.map(|point| (point.x, point.y)),
            "reused reducer differs from original in case {case}"
        );
    }
    assert_affine_bucket_results_match_native::<C>(case, reduced, source);
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
            let (x, y) = C::affine_xy(&affine);
            clean_points.push(AffinePoint { x, y });
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
            .map(|i| AffinePoint {
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

#[test]
fn booth_digits_reconstruct_components() {
    for width in 1..=16 {
        let mut magnitudes = alloc::vec![0, 1, (1u128 << 127) - 1];
        for bit in 1..127 {
            magnitudes.extend([(1u128 << bit) - 1, 1u128 << bit, (1u128 << bit) + 1]);
        }
        for magnitude in magnitudes {
            for negative in [false, true] {
                let component = SignedMagnitude {
                    magnitude,
                    negative,
                };
                let mut reconstructed = crate::Fq::ZERO;
                let radix = crate::Fq::from(1u64 << width);
                for window in (0..GLV_COMPONENT_BITS / width + 1).rev() {
                    let digit = booth_digit(component, width, window);
                    assert!(digit.magnitude <= 1 << (width - 1));
                    let value = crate::Fq::from(u64::try_from(digit.magnitude).unwrap());
                    reconstructed =
                        reconstructed * radix + if digit.negative { -value } else { value };
                }
                let value = crate::Fq::from_u128(magnitude);
                assert_eq!(reconstructed, if negative { -value } else { value });
            }
        }
    }
}

fn forced_window<C: MsmCurve>(terms: usize, width: usize) {
    let scalars = (0..terms)
        .map(|i| {
            let i = u64::try_from(i).unwrap();
            match i % 8 {
                0 => C::ScalarExt::ZERO,
                1 | 2 => C::ScalarExt::ONE,
                3 => -C::ScalarExt::ONE,
                _ => (C::ScalarExt::from(i + 0x9e3779b97f4a7c15).square() + C::ScalarExt::from(i))
                    .square(),
            }
        })
        .collect::<Vec<_>>();
    // Equal and opposite bases, as well as identities, force exceptional
    // reductions independently of which backend the cost planner would pick.
    let generator = C::generator();
    let bases = (0..terms)
        .map(|i| {
            match i % 4 {
                0 => C::identity(),
                1 => generator,
                2 => -generator,
                _ => generator.double(),
            }
            .to_affine_vartime()
        })
        .collect::<Vec<_>>();
    let expected = scalars
        .iter()
        .zip(&bases)
        .fold(C::identity(), |sum, (scalar, base)| sum + *base * scalar);
    let components = scalars
        .iter()
        .map(decompose::<C>)
        .map(checked_signed_magnitudes)
        .collect::<Option<Vec<_>>>()
        .unwrap();
    let bases = multiexp_bases::<C>(&bases);
    assert_eq!(
        multiexp_serial::<C>(&components, &bases, width),
        Some(expected),
        "{terms} terms at width {width}"
    );
    #[cfg(feature = "multicore")]
    assert_eq!(
        multiexp_parallel::<C>(&components, &bases, width),
        Some(expected),
        "{terms} terms at width {width}, parallel"
    );
}

#[test]
fn all_window_widths() {
    for width in [1, 3, 6, 8, 9, 10] {
        forced_window::<pallas::Point>(64, width);
        forced_window::<vesta::Point>(64, width);
    }
}

#[test]
fn cached_scratch_alternating_curves_sizes_and_widths() {
    for (terms, width) in [
        (300, 7),
        (0, 1),
        (17, 3),
        (300, 4),
        (64, 9),
        (1, 2),
        (5, 10),
        (257, 6),
    ] {
        forced_window::<pallas::Point>(terms, width);
        forced_window::<vesta::Point>(terms, width);
    }
}

#[cfg(feature = "multicore")]
#[test]
fn nested_parallel_scratch_is_independent() {
    // One worker guarantees nested Rayon jobs re-enter the checkout thread.
    for workers in [1, 2] {
        let pool = maybe_rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            with_window_scratch::<crate::Fp, _>(|outer| {
                outer.counts.extend([123, 456]);
                let outer_pointer = outer.counts.as_ptr().addr();
                maybe_rayon::join(
                    || {
                        with_window_scratch::<crate::Fp, _>(|nested| {
                            assert_ne!(nested.counts.as_ptr().addr(), outer_pointer);
                            nested.counts.clear();
                            nested.counts.push(789);
                            forced_window::<pallas::Point>(65, 4);
                            assert_eq!(nested.counts, [789]);
                        });
                    },
                    || {
                        forced_window::<vesta::Point>(17, 3);
                        forced_window::<pallas::Point>(33, 6);
                    },
                );
                assert_eq!(outer.counts, [123, 456]);
            });
            forced_window::<pallas::Point>(5, 2);
            forced_window::<vesta::Point>(0, 8);
        });
    }
}

#[cfg(feature = "multicore")]
#[test]
fn tls_scratch_reuses_capacity_and_drops_oversized_buffers() {
    // Isolate this cache from other tests and the test runner's worker threads.
    std::thread::spawn(|| {
        let capacity = with_window_scratch::<crate::Fp, _>(|scratch| {
            scratch.assignments.reserve(128);
            scratch.assignments.capacity()
        });
        with_window_scratch::<crate::Fp, _>(|scratch| {
            assert_eq!(scratch.assignments.capacity(), capacity);
            scratch.assignments.reserve(
                MAX_CACHED_WINDOW_SCRATCH_BYTES / core::mem::size_of::<WindowAssignment>() + 1,
            );
            assert!(scratch.capacity_bytes() > MAX_CACHED_WINDOW_SCRATCH_BYTES);
        });
        with_window_scratch::<crate::Fp, _>(|scratch| {
            assert_eq!(scratch.capacity_bytes(), 0);
        });
    })
    .join()
    .unwrap();
}

#[test]
fn planner_boundaries() {
    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(255, 1), None);
    assert!(glv_multiexp_window_bits::<pallas::Point>(2048, 1).is_some());
    assert_eq!(default_multiexp_window_bits(31), Some(3));
    assert_eq!(default_multiexp_window_bits(32), Some(4));
    assert_eq!(default_multiexp_window_bits(54), Some(4));
    assert_eq!(default_multiexp_window_bits(55), Some(5));
    assert_eq!(estimated_signed_booth_costs(usize::MAX, 256, 8, 1), None);
    assert_eq!(estimated_signed_booth_costs(2048, 256, 0, 1), None);
}

#[test]
fn full_width_identity_declines_noncanonical_montgomery_residues() {
    let bases = alloc::vec![pallas::Affine::identity(); 4096];
    let mut scalars = alloc::vec![crate::Fq::ZERO; bases.len()];
    scalars[0] = crate::Fq([u64::MAX; 4]);
    assert_eq!(
        try_multiexp_full_width_is_identity::<pallas::Point>(&scalars, &bases),
        None
    );
    let bases = alloc::vec![vesta::Affine::identity(); 4096];
    let mut scalars = alloc::vec![crate::Fp::ZERO; bases.len()];
    scalars[0] = crate::Fp([u64::MAX; 4]);
    assert_eq!(
        try_multiexp_full_width_is_identity::<vesta::Point>(&scalars, &bases),
        None
    );
}
