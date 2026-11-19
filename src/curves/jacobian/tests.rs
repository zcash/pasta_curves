//! The Jacobian model.

use crate::curves::{Ep, Eq, IsoEp, IsoEq};
use core::mem::size_of;

/// Three base field elements of 32 bytes, with no padding, so that tables of
/// points cost what their coordinates do.
#[test]
fn jacobian_points_are_96_bytes() {
    assert_eq!(size_of::<Ep>(), 96);
    assert_eq!(size_of::<Eq>(), 96);
    assert_eq!(size_of::<IsoEp>(), 96);
    assert_eq!(size_of::<IsoEq>(), 96);
}

#[cfg(feature = "zeroize")]
#[test]
fn jacobian_points_zeroize_to_identity() {
    use group::Group;
    use zeroize::Zeroize;

    let mut p = Ep::generator();
    p.zeroize();
    assert!(bool::from(p.is_identity()));

    let mut q = Eq::generator();
    q.zeroize();
    assert!(bool::from(q.is_identity()));
}
