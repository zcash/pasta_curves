#![cfg(feature = "glv")]

use ff::Field;
use pasta_curves::{arithmetic::CurveExt, pallas, vesta};
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

fn check<C: CurveExt>() {
    let mut rng = XorShiftRng::from_seed([47; 16]);
    let n = 4096;
    let bases = (0..n)
        .map(|_| C::random(&mut rng).to_affine())
        .collect::<Vec<_>>();
    let mut scalars = (0..n)
        .map(|_| C::ScalarExt::random(&mut rng))
        .collect::<Vec<_>>();
    scalars[0] = C::ScalarExt::ZERO;
    scalars[1] = -C::ScalarExt::ONE;
    let mut accepted = 0;
    let mut cancellations = 0;
    for len in [256, 512, 1024, 4096] {
        let expected = scalars[..len]
            .iter()
            .zip(&bases[..len])
            .fold(C::identity(), |sum, (scalar, base)| sum + *base * *scalar);
        if let Some(actual) =
            C::try_multiexp_full_width_is_identity_vartime(&scalars[..len], &bases[..len])
        {
            accepted += 1;
            assert_eq!(actual, bool::from(expected.is_identity()));
            let mut cancelling_bases = bases[..len].to_vec();
            let mut cancelling_scalars = scalars[..len].to_vec();
            cancelling_bases.push((-expected).to_affine());
            cancelling_scalars.push(C::ScalarExt::ONE);
            if let Some(actual) = C::try_multiexp_full_width_is_identity_vartime(
                &cancelling_scalars,
                &cancelling_bases,
            ) {
                assert!(actual);
                cancellations += 1;
            }
        }
    }
    assert!(accepted > 0, "must exercise optimized identity arithmetic");
    assert!(
        cancellations > 0,
        "must exercise optimized cancellation checks"
    );
    assert!(C::try_multiexp_full_width_is_identity_vartime(&scalars[..1], &[]).is_none());
}

#[test]
fn montgomery_identity_checks_match_ordinary_msm_on_both_curves() {
    check::<pallas::Point>();
    check::<vesta::Point>();
}
