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
