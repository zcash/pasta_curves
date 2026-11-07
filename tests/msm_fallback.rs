// The default hook needs `CurveExt`, which requires `alloc`.
#![cfg(all(feature = "alloc", not(feature = "glv")))]

use pasta_curves::{arithmetic::CurveExt, pallas, vesta};

fn check_default<C: CurveExt>() {
    assert_eq!(C::try_multiexp_vartime(&[], &[]), None);
    assert_eq!(
        C::try_multiexp_full_width_is_identity_vartime(&[], &[]),
        None
    );
    assert_eq!(
        C::try_multiexp_full_width_is_identity_vartime(
            &[C::ScalarExt::from(1)],
            &[C::generator().into()],
        ),
        None,
    );
    assert_eq!(
        C::try_multiexp_vartime(&[C::ScalarExt::from(1)], &[C::generator().into()]),
        None,
    );
}

#[test]
fn disabled_backend_declines() {
    check_default::<pallas::Point>();
    check_default::<vesta::Point>();
}
