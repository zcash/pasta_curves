//! Differential checks for Booth recoding and the failure-atomic affine reducer.
use super::affine::*;
use super::buckets::*;
use super::lift::*;
use super::plan::*;
use super::recode::*;
use super::*;
use crate::arithmetic::VartimeField;
use alloc::vec::Vec;
use ff::PrimeField;

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

    let reduced = reduce_affine_buckets(points, offsets)
        .unwrap_or_else(|| panic!("valid curve points must reduce in case {case}"));
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

fn forced_windows<C: MsmCurve>() {
    let scalars = (0..64)
        .map(|i| {
            (C::ScalarExt::from(i + 0x9e3779b97f4a7c15).square() + C::ScalarExt::from(i)).square()
        })
        .collect::<Vec<_>>();
    // Equal and opposite bases, as well as identities, force exceptional
    // reductions independently of which backend the cost planner would pick.
    let generator = C::generator();
    let bases = (0..64)
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
    let terms = glv_lift::<C>(&scalars, &bases).unwrap();
    for width in [1, 3, 6, 8, 9, 10] {
        for parallel in [false, true] {
            assert_eq!(
                bucket_multiexp::<C, _>(&terms, width, parallel, &|| ()),
                Some(expected),
                "width {width}, parallel {parallel}"
            );
        }
    }
}

#[test]
fn all_window_widths() {
    forced_windows::<pallas::Point>();
    forced_windows::<vesta::Point>();
}

/// Reducing a window in blocks, with each block's buckets carried into the
/// next, gives the same buckets as one block. Equal and opposite bases make
/// carried points double and cancel.
fn blocks_match_single_block<C: MsmCurve>() {
    let generator = C::generator();
    let scalars = (0..64u64)
        .map(|i| {
            (C::ScalarExt::from(i + 0x9e3779b97f4a7c15).square() + C::ScalarExt::from(i)).square()
        })
        .collect::<Vec<_>>();
    let bases = (0..64)
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
    let terms = glv_lift::<C>(&scalars, &bases).unwrap();
    let as_points = |buckets: Vec<Option<AffinePoint<C::Base>>>| {
        buckets
            .into_iter()
            .map(|bucket| match bucket {
                Some(point) => C::from(C::affine_unchecked(point.x, point.y)),
                None => C::identity(),
            })
            .collect::<Vec<_>>()
    };

    for width in [1, 3, 6] {
        for window in 0..GLV_COMPONENT_BITS / width + 1 {
            let whole =
                as_points(fill_window_in_blocks(&terms, width, window, terms.len()).unwrap());
            for block_terms in [1, 2, 3, 5, 16, 63] {
                let blocked =
                    as_points(fill_window_in_blocks(&terms, width, window, block_terms).unwrap());
                assert_eq!(
                    blocked, whole,
                    "width {width}, window {window}, blocks of {block_terms} terms"
                );
            }
        }
    }
}

#[test]
fn window_blocks() {
    blocks_match_single_block::<pallas::Point>();
    blocks_match_single_block::<vesta::Point>();
}

#[test]
fn planner_boundaries() {
    // Serial: Strauss up to SERIAL_STRAUSS_MAX_TERMS, then measured bucket widths.
    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(24, 1), None);
    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(25, 1), Some(5));
    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(127, 1), Some(5));
    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(128, 1), Some(6));
    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(255, 1), Some(6));
    assert!(glv_multiexp_window_bits::<pallas::Point>(2048, 1).is_some());
    // Parallel: below MIN_GLV_MULTIEXP_TERMS the planner still declines.
    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(255, 2), None);
    assert_eq!(default_multiexp_window_bits(31), Some(3));
    assert_eq!(default_multiexp_window_bits(32), Some(4));
    assert_eq!(default_multiexp_window_bits(54), Some(4));
    assert_eq!(default_multiexp_window_bits(55), Some(5));
    assert_eq!(estimated_signed_booth_work(usize::MAX, 256, 8, 1), None);
    assert_eq!(estimated_signed_booth_work(2048, 256, 0, 1), None);
    // Serial: 33 windows of 2048 visits, 2 * 128 bucket additions per window,
    // and 32 shifts of 8 doublings.
    assert_eq!(
        estimated_signed_booth_work(2048, 256, 8, 1),
        Some(BoothWork(33 * 2048 + 33 * 2 * 128 + 32 * 8))
    );
    // Four workers: 9 waves of (2048 + 2 * 128), plus the highest window's
    // shift in each wave, 8 * (32 + 28 + ... + 0).
    assert_eq!(
        estimated_signed_booth_work(2048, 256, 8, 4),
        Some(BoothWork(
            9 * (2048 + 2 * 128) + 8 * (32 + 28 + 24 + 20 + 16 + 12 + 8 + 4)
        ))
    );
}

/// The planner's choices, pinned. Below 256 terms the serial rows are the
/// measured choices; the other bucket widths and declines are the cost
/// model's, checked against an independent reimplementation of it.
#[test]
fn plan_table() {
    use Plan::*;
    let buckets = |window_bits, parallel| Buckets {
        window_bits,
        parallel,
    };
    let table = [
        // (terms, threads, plan)
        (0, 1, Strauss),
        (24, 1, Strauss),
        (25, 1, buckets(5, false)),
        (127, 1, buckets(5, false)),
        (128, 1, buckets(6, false)),
        (255, 1, buckets(6, false)),
        (256, 1, buckets(7, false)),
        (2048, 1, buckets(9, false)),
        (1 << 16, 1, buckets(13, false)),
        (1 << 20, 1, buckets(15, false)),
        (2, 2, Strauss),
        (3, 2, ParallelStrauss { jobs: 1 }),
        (66, 2, ParallelStrauss { jobs: 1 }),
        (67, 2, Decline),
        (255, 2, Decline),
        (256, 2, buckets(7, true)),
        (2048, 2, buckets(8, true)),
        (1 << 20, 2, buckets(14, true)),
        (2, 4, Strauss),
        (3, 4, ParallelStrauss { jobs: 2 }),
        (66, 4, ParallelStrauss { jobs: 2 }),
        (256, 4, buckets(7, true)),
        (1 << 20, 4, Decline),
        (3, 8, ParallelStrauss { jobs: 3 }),
        (10, 8, ParallelStrauss { jobs: 4 }),
        (256, 8, buckets(6, true)),
        (1 << 16, 8, Decline),
        (1 << 16, 16, buckets(12, true)),
        (1 << 20, 16, buckets(14, true)),
        (1 << 16, 32, Decline),
        (256, 64, Decline),
        (2048, 64, Decline),
    ];
    for (terms, threads, expected) in table {
        assert_eq!(
            plan::<pallas::Point>(terms, threads),
            expected,
            "{terms} terms on {threads} threads"
        );
    }
}

/// The thread count the planner sees is the pool it runs in, so a parallel
/// plan is chosen whenever the caller's pool has more than one thread.
#[cfg(all(
    feature = "multicore",
    not(all(target_arch = "wasm32", not(target_feature = "atomics")))
))]
#[test]
fn planner_sees_the_pool() {
    for threads in [1, 2, 4] {
        let pool = maybe_rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(pool.install(current_num_threads), threads);
    }
}

/// Runs `plan` on a two-thread pool, holding the first two jobs until both
/// have started. Jobs that ran one after another would never both start, so
/// the wait times out and the test fails.
#[cfg(all(
    feature = "multicore",
    not(all(target_arch = "wasm32", not(target_feature = "atomics")))
))]
fn assert_jobs_overlap<C: MsmCurve>(plan: Plan, terms: usize) {
    use core::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use std::time::{Duration, Instant};

    let scalars = (0..terms as u64)
        .map(|i| C::ScalarExt::from(i + 0x9e3779b97f4a7c15).square())
        .collect::<Vec<_>>();
    let bases = (1..=terms as u64)
        .map(|i| (C::generator() * C::ScalarExt::from(i)).to_affine_vartime())
        .collect::<Vec<_>>();
    let expected = scalars
        .iter()
        .zip(&bases)
        .fold(C::identity(), |sum, (scalar, base)| sum + *base * scalar);

    let started = AtomicUsize::new(0);
    let on_job = || {
        if started.fetch_add(1, SeqCst) < 2 {
            let deadline = Instant::now() + Duration::from_secs(30);
            while started.load(SeqCst) < 2 {
                assert!(Instant::now() < deadline, "{plan:?}: jobs did not overlap");
                std::thread::yield_now();
            }
        }
    };
    let pool = maybe_rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let actual = pool.install(|| execute_observed::<C>(plan, &scalars, &bases, &on_job));
    assert_eq!(actual, Some(expected), "{plan:?}");
}

#[cfg(all(
    feature = "multicore",
    not(all(target_arch = "wasm32", not(target_feature = "atomics")))
))]
#[test]
fn parallel_plans_run_concurrently() {
    assert_jobs_overlap::<pallas::Point>(Plan::ParallelStrauss { jobs: 2 }, 32);
    assert_jobs_overlap::<pallas::Point>(
        Plan::Buckets {
            window_bits: 6,
            parallel: true,
        },
        64,
    );
}

mod every_plan {
    use super::*;
    use crate::test_scalars::from_uniform_limbs;
    use ff::FromUniformBytes;
    use proptest::prelude::*;

    fn plan_strategy() -> impl Strategy<Value = Plan> {
        prop_oneof![
            Just(Plan::Strauss),
            (1usize..=5).prop_map(|jobs| Plan::ParallelStrauss { jobs }),
            (1usize..=12, any::<bool>()).prop_map(|(window_bits, parallel)| Plan::Buckets {
                window_bits,
                parallel
            }),
        ]
    }

    /// One term: a scalar kind (zero, one, minus one, shared, random) and a
    /// base kind (identity, shared, negated shared, random), with the bytes
    /// that a random scalar or base is drawn from.
    fn term_strategy() -> impl Strategy<Value = (u8, u8, [u64; 8])> {
        (0u8..8, 0u8..6, proptest::array::uniform8(any::<u64>()))
    }

    /// The law every strategy obeys: the plan evaluates to the sum of the
    /// products, computed here by native scalar multiplication.
    fn check<C: MsmCurve>(plan: Plan, terms: &[(u8, u8, [u64; 8])]) -> Result<(), TestCaseError>
    where
        C::ScalarExt: FromUniformBytes<64>,
    {
        let shared_scalar = C::ScalarExt::from(0x1234_5678_9abc_def0);
        let shared_base = C::generator() * C::ScalarExt::from(7);
        let mut scalars = Vec::with_capacity(terms.len());
        let mut bases = Vec::with_capacity(terms.len());
        for (scalar_kind, base_kind, limbs) in terms {
            scalars.push(match scalar_kind {
                0 => C::ScalarExt::ZERO,
                1 => C::ScalarExt::ONE,
                2 => -C::ScalarExt::ONE,
                3 => shared_scalar,
                _ => from_uniform_limbs::<C::ScalarExt, 8>(*limbs),
            });
            bases.push(
                match base_kind {
                    0 => C::identity(),
                    1 => shared_base,
                    2 => -shared_base,
                    _ => C::generator() * from_uniform_limbs::<C::ScalarExt, 8>(*limbs).square(),
                }
                .to_affine_vartime(),
            );
        }
        let expected = scalars
            .iter()
            .zip(&bases)
            .fold(C::identity(), |sum, (scalar, base)| sum + *base * scalar);
        prop_assert_eq!(execute::<C>(plan, &scalars, &bases), Some(expected));
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn pallas(plan in plan_strategy(), terms in proptest::collection::vec(term_strategy(), 0..48)) {
            check::<pallas::Point>(plan, &terms)?;
        }

        #[test]
        fn vesta(plan in plan_strategy(), terms in proptest::collection::vec(term_strategy(), 0..48)) {
            check::<vesta::Point>(plan, &terms)?;
        }
    }
}
