use super::*;
use crate::{pallas, vesta};

#[test]
fn planner_boundaries() {
    // Serial: Straus up to SERIAL_STRAUS_MAX_TERMS, then measured bucket widths.
    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(24, 1), None);
    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(25, 1), Some(5));
    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(127, 1), Some(5));
    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(128, 1), Some(6));
    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(255, 1), Some(6));
    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(256, 1), Some(6));
    assert_eq!(glv_multiexp_window_bits::<vesta::Point>(256, 1), Some(6));
    assert!(glv_multiexp_window_bits::<pallas::Point>(2048, 1).is_some());
    // Parallel: below MIN_GLV_MULTIEXP_TERMS the planner still declines.
    assert_eq!(glv_multiexp_window_bits::<pallas::Point>(255, 2), None);
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

    // Beyond the measured grids the weighted model extrapolates to GLV at
    // a width within its candidate range; an overflowing term count is
    // never planned.
    let default = multiexp_window_bits::<pallas::Point>(1_000_000, 8).unwrap();
    let width = glv_multiexp_window_bits::<pallas::Point>(1_000_000, 8).unwrap();
    assert!((default..=default + GLV_EXTRA_WINDOW_BITS).contains(&width));
    assert_eq!(
        glv_multiexp_window_bits::<pallas::Point>(usize::MAX, 8),
        None
    );
}

/// The planner's decisions at cells of the timing grids
/// (`glv_vs_generic_timing_grid`): Apple M4 with the asm backend on 1-16
/// workers, and two AMD Zen 4 hosts with portable and
/// `-C target-cpu=native` builds on 1-8 workers. Unless noted, every
/// pinned decision measured within 3% of the fastest backend and width on
/// every grid that ran the cell.
#[test]
fn glv_multiexp_plan_matches_measured_cells() {
    fn plan(terms: usize, threads: usize) -> Option<usize> {
        glv_multiexp_window_bits::<pallas::Point>(terms, threads)
    }

    assert_eq!(plan(255, 1), Some(6));
    assert_eq!(plan(255, 8), None);

    // Cells where the unweighted model chose the generic ladder although
    // GLV measured 1.3-1.8x faster: the weights price GLV's batched affine
    // point visits below the generic ladder's projective ones. The 7,150-
    // term cells are the final MSM of a 64-bundle Orchard batch.
    assert_eq!(plan(7_150, 6), Some(11));
    assert_eq!(plan(7_150, 12), Some(11));
    assert_eq!(plan(2_049, 12), Some(8));
    assert_eq!(plan(4_096, 6), Some(11));
    // GLV at w = 10 beats the generic ladder 1.3x here, though w = 13
    // measured faster still: the model's wave count does not capture it.
    assert_eq!(plan(16_384, 10), Some(10));
    assert_eq!(plan(32_768, 10), Some(13));

    // Cells measured within 3% of the best on all five grids.
    for (terms, threads, width) in [
        (256, 1, 6),
        (256, 4, 7),
        (512, 4, 8),
        (1_024, 1, 8),
        (1_024, 6, 8),
        (4_096, 1, 10),
        (7_150, 2, 11),
        (8_192, 4, 11),
        (8_192, 6, 11),
        (16_384, 3, 11),
        (32_768, 2, 13),
        (32_768, 4, 11),
        (65_536, 2, 13),
    ] {
        assert_eq!(
            plan(terms, threads),
            Some(width),
            "{terms} terms on {threads}"
        );
    }

    // The grids disagree at the largest sizes on eight workers: GLV
    // measured 1.3x faster on M4, the generic ladder 2-9% faster on Zen 4.
    // The planner keeps GLV there.
    assert_eq!(plan(32_768, 8), Some(11));
    assert_eq!(plan(65_536, 8), Some(12));

    // The planner widens at most GLV_EXTRA_WINDOW_BITS past the generic
    // default, and both curves plan identically.
    for terms in [2_600usize, 8_192, 65_536] {
        for threads in [1usize, 3, 8] {
            let default = multiexp_window_bits::<pallas::Point>(terms, threads).unwrap();
            if let Some(width) = plan(terms, threads) {
                assert!((default..=default + GLV_EXTRA_WINDOW_BITS).contains(&width));
            }
            assert_eq!(
                plan(terms, threads),
                glv_multiexp_window_bits::<vesta::Point>(terms, threads)
            );
        }
    }
}
