use super::*;
use crate::glv::msm::calibration::DEFAULT;
use crate::{pallas, vesta};

/// Unit weights, so that the work estimates count operations.
const UNIT_PLANNER_WEIGHTS: PlannerWeights = PlannerWeights {
    point_visit: 1,
    bucket_addition: 1,
    doubling: 1,
};

#[test]
fn planner_boundaries() {
    // Serial: Straus up to serial_straus_max_terms, then measured bucket widths.
    assert_eq!(plan_with::<pallas::Point>(24, 1, &DEFAULT), Plan::Straus);
    assert_eq!(
        glv_multiexp_window_bits::<pallas::Point>(25, 1, &DEFAULT),
        Some(5)
    );
    assert_eq!(
        glv_multiexp_window_bits::<pallas::Point>(127, 1, &DEFAULT),
        Some(5)
    );
    assert_eq!(
        glv_multiexp_window_bits::<pallas::Point>(128, 1, &DEFAULT),
        Some(6)
    );
    assert_eq!(
        glv_multiexp_window_bits::<pallas::Point>(255, 1, &DEFAULT),
        Some(6)
    );
    assert_eq!(
        glv_multiexp_window_bits::<pallas::Point>(256, 1, &DEFAULT),
        Some(6)
    );
    assert_eq!(
        glv_multiexp_window_bits::<vesta::Point>(256, 1, &DEFAULT),
        Some(6)
    );
    assert!(glv_multiexp_window_bits::<pallas::Point>(2048, 1, &DEFAULT).is_some());
    // Parallel: below min_glv_multiexp_terms the planner still declines.
    assert_eq!(
        glv_multiexp_window_bits::<pallas::Point>(255, 2, &DEFAULT),
        None
    );
    assert_eq!(default_multiexp_window_bits(31), Some(3));
    assert_eq!(default_multiexp_window_bits(32), Some(4));
    assert_eq!(default_multiexp_window_bits(54), Some(4));
    assert_eq!(default_multiexp_window_bits(55), Some(5));
    assert_eq!(
        estimated_signed_booth_work(usize::MAX, 256, 8, 1, UNIT_PLANNER_WEIGHTS),
        None
    );
    assert_eq!(
        estimated_signed_booth_work(2048, 256, 0, 1, UNIT_PLANNER_WEIGHTS),
        None
    );
    // Serial: 33 windows of 2048 visits, 2 * 128 bucket additions per window,
    // and 32 shifts of 8 doublings.
    assert_eq!(
        estimated_signed_booth_work(2048, 256, 8, 1, UNIT_PLANNER_WEIGHTS),
        Some(BoothWork(33 * 2048 + 33 * 2 * 128 + 32 * 8))
    );
    // Four workers: 9 waves of (2048 + 2 * 128), plus the highest window's
    // shift in each wave, 8 * (32 + 28 + ... + 0).
    assert_eq!(
        estimated_signed_booth_work(2048, 256, 8, 4, UNIT_PLANNER_WEIGHTS),
        Some(BoothWork(
            9 * (2048 + 2 * 128) + 8 * (32 + 28 + 24 + 20 + 16 + 12 + 8 + 4)
        ))
    );
}

#[test]
fn weighted_planner_matches_calibrated_cells() {
    for (terms, threads, expected) in [
        (255, 1, Some(6)),
        (255, 8, None),
        (7_150, 6, Some(11)),
        (7_150, 12, Some(11)),
        (2_049, 12, Some(8)),
        (4_096, 6, Some(11)),
        (16_384, 10, Some(10)),
        (32_768, 10, Some(13)),
        (256, 1, Some(6)),
        (256, 4, Some(7)),
        (512, 4, Some(8)),
        (1_024, 1, Some(8)),
        (1_024, 6, Some(8)),
        (4_096, 1, Some(10)),
        (7_150, 2, Some(11)),
        (8_192, 4, Some(11)),
        (8_192, 6, Some(11)),
        (16_384, 3, Some(11)),
        (32_768, 2, Some(13)),
        (32_768, 4, Some(11)),
        (65_536, 2, Some(13)),
        (32_768, 8, Some(11)),
        (65_536, 8, Some(12)),
    ] {
        assert_eq!(
            glv_multiexp_window_bits::<pallas::Point>(terms, threads, &DEFAULT),
            expected,
            "{terms} terms on {threads} workers, Pallas"
        );
        assert_eq!(
            glv_multiexp_window_bits::<vesta::Point>(terms, threads, &DEFAULT),
            expected,
            "{terms} terms on {threads} workers, Vesta"
        );
    }
    for threads in [1, 3, 8] {
        for terms in [2_600, 8_192, 65_536, 1_000_000] {
            let default = default_multiexp_window_bits(terms).unwrap();
            if let Some(width) = glv_multiexp_window_bits::<pallas::Point>(terms, threads, &DEFAULT)
            {
                assert!((default..=default + DEFAULT.glv_extra_window_bits).contains(&width));
            }
        }
    }
    assert_eq!(
        glv_multiexp_window_bits::<pallas::Point>(usize::MAX, 8, &DEFAULT),
        None
    );
    assert_eq!(
        estimated_signed_booth_work(usize::MAX, 256, 8, 1, DEFAULT.glv_weights),
        None
    );
    assert_eq!(
        estimated_signed_booth_work(2048, 256, 0, 1, DEFAULT.generic_weights),
        None
    );
}

/// The routes at the boundaries of the decision table in the module
/// documentation, on one thread and, with `multicore`, on several.
#[test]
fn plan_routes_the_boundary_sizes() {
    fn check(terms: usize, threads: usize, expected: Plan) {
        assert_eq!(
            plan_with::<pallas::Point>(terms, threads, &DEFAULT),
            expected,
            "{terms} terms on {threads} threads, Pallas"
        );
        assert_eq!(
            plan_with::<vesta::Point>(terms, threads, &DEFAULT),
            expected,
            "{terms} terms on {threads} threads, Vesta"
        );
    }

    for terms in [0, 1, 2, 3, 16, 17, 24] {
        check(terms, 1, Plan::Straus);
    }
    for (terms, window_bits) in [
        (25, 5),
        (66, 5),
        (67, 5),
        (127, 5),
        (128, 6),
        (255, 6),
        (256, 6),
        (257, 6),
    ] {
        check(terms, 1, Plan::Buckets { window_bits });
    }

    #[cfg(feature = "multicore")]
    for (threads, max_jobs, window_bits) in [(2, 1, 6), (8, 4, 6), (16, 8, 8)] {
        for terms in [0, 1, 2] {
            check(terms, threads, Plan::Straus);
        }
        for terms in [3, 16, 17, 24, 25, 66] {
            let jobs = max_jobs.min(terms);
            check(terms, threads, Plan::StrausLeaves { jobs });
        }
        for terms in [67, 127, 128, 255] {
            check(terms, threads, Plan::Decline);
        }
        for terms in [256, 257] {
            check(terms, threads, Plan::Buckets { window_bits });
        }
    }
}
