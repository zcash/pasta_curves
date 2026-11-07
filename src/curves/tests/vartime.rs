use super::*;
use ff::Field;

fn check_points<C: CurveExt>(points: &[C]) {
    for point in points {
        assert_eq!(point.is_identity_vartime(), bool::from(point.is_identity()));
        let affine = point.to_affine();
        assert_eq!(
            affine.is_identity_vartime(),
            bool::from(affine.is_identity())
        );
        let mut ordinary = *point;
        let mut public = *point;
        for _ in 0..8 {
            ordinary = ordinary.double();
            public = public.double_vartime();
            assert!(bool::from(public.ct_eq(&ordinary)));
            assert_eq!(public.to_affine(), ordinary.to_affine());
        }
        for rhs in points {
            let expected = *point + *rhs;
            let sum = point.add_vartime(rhs);
            let mixed = point.add_mixed_vartime(&rhs.to_affine());
            assert!(bool::from(sum.ct_eq(&expected)));
            assert!(bool::from(mixed.ct_eq(&expected)));
            assert_eq!(sum.to_affine(), expected.to_affine());
            assert_eq!(mixed.to_affine(), expected.to_affine());
        }
    }
}

fn check_curve<C: CurveExt>() {
    let generator = C::generator();
    let mut points = vec![C::identity(), generator, -generator, generator.double()];
    // Exercise equal points with different Jacobian scales and arbitrary
    // x/y coordinates at infinity, not just canonical z = 0.
    let (x, y, z) = generator.jacobian_coordinates();
    for scale in [C::Base::from(2), -C::Base::ONE] {
        let scale2 = scale.square();
        points.push(C::new_jacobian(x * scale2, y * scale2 * scale, z * scale).unwrap());
    }
    points.push(C::new_jacobian(C::Base::ONE, C::Base::from(2), C::Base::ZERO).unwrap());
    let mut rng = XorShiftRng::from_seed([0x92; 16]);
    for _ in 0..16 {
        let point = C::try_random(&mut rng).unwrap();
        points.extend([point, -point, point.double()]);
    }
    check_points(&points);
}

#[test]
fn public_operations_match_ordinary_operations() {
    check_curve::<Ep>();
    check_curve::<Eq>();
}
