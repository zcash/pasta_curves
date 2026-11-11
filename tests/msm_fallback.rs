// The default hook needs `CurveExt`, which requires `alloc`.
#![cfg(all(feature = "alloc", not(feature = "glv")))]

use pasta_curves::{arithmetic::CurveExt, pallas, vesta};

fn check_default<C: CurveExt>() {
    assert_eq!(C::try_multiexp_vartime(&[], &[]), None);
    assert_eq!(
        C::try_multiexp_vartime(&[C::ScalarExt::from(1)], &[C::generator().into()]),
        None,
    );
    // Mismatched lengths panic without the backend too, so the contract does
    // not depend on which features are enabled.
    let mismatched =
        std::panic::catch_unwind(|| C::try_multiexp_vartime(&[C::ScalarExt::from(1)], &[]));
    assert!(mismatched.is_err());
}

#[test]
fn disabled_backend_declines() {
    check_default::<pallas::Point>();
    check_default::<vesta::Point>();
}
