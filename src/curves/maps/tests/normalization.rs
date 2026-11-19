use super::*;
use ff::Field;

fn check_two_chain_normalization<C: CurveExt>() {
    let generator = C::generator();
    let identity = C::identity();
    let mut rng = XorShiftRng::from_seed([
        0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ]);
    for len in [0, 1, 2, 3, 7, 16, 31, 32, 33, 63, 64, 65, 128, 129] {
        check_vartime_normalization(&vec![identity; len]);
        check_vartime_normalization(&vec![generator; len]);
        let points: Vec<C> = (0..len)
            .map(|_| C::try_random(&mut rng).unwrap().double())
            .collect();
        check_vartime_normalization(&points);

        // Exercise either lane consisting entirely of identities, mixed
        // lanes, and first/last identities around the crossover.
        for stride in [2, 3] {
            for offset in 0..stride {
                let mixed: Vec<C> = points
                    .iter()
                    .enumerate()
                    .map(|(i, p)| if i % stride == offset { identity } else { *p })
                    .collect();
                check_vartime_normalization(&mixed);
            }
        }

        // Equivalent Jacobian representatives with z = 0, 1, -1 (the
        // maximum field value), and random z. Repeated affine
        // points must normalize identically regardless of their scale.
        let (x, y, _) = generator.to_affine().to_curve().jacobian_coordinates();
        let scales = [
            C::Base::ZERO,
            C::Base::ONE,
            -C::Base::ONE,
            C::Base::random(&mut rng),
        ];
        let scaled: Vec<C> = (0..len)
            .map(|i| {
                let z = scales[i % scales.len()];
                let z2 = z.square();
                C::new_jacobian(x * z2, y * z2 * z, z).unwrap()
            })
            .collect();
        check_vartime_normalization(&scaled);
    }
}

#[test]
fn two_chain_normalization_matches_group_curve() {
    check_two_chain_normalization::<Ep>();
    check_two_chain_normalization::<Eq>();
}
