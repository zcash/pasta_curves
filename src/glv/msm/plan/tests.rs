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
            glv_multiexp_window_bits::<pallas::Point>(terms, threads),
            expected,
            "{terms} terms on {threads} workers, Pallas"
        );
        assert_eq!(
            glv_multiexp_window_bits::<vesta::Point>(terms, threads),
            expected,
            "{terms} terms on {threads} workers, Vesta"
        );
    }
    for threads in [1, 3, 8] {
        for terms in [2_600, 8_192, 65_536, 1_000_000] {
            let default = multiexp_window_bits::<pallas::Point>(terms, threads).unwrap();
            if let Some(width) = glv_multiexp_window_bits::<pallas::Point>(terms, threads) {
                assert!((default..=default + GLV_EXTRA_WINDOW_BITS).contains(&width));
            }
        }
    }
    assert_eq!(
        glv_multiexp_window_bits::<pallas::Point>(usize::MAX, 8),
        None
    );
    assert_eq!(
        estimated_signed_booth_work(usize::MAX, 256, 8, 1, GLV_PLANNER_WEIGHTS),
        None
    );
    assert_eq!(
        estimated_signed_booth_work(2048, 256, 0, 1, GENERIC_PLANNER_WEIGHTS),
        None
    );
}
