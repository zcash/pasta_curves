use crate::curves::{Ep, Eq};
use crate::{Fp, Fq};
use ff::{Field, PrimeField};
use group::{Curve, Group};
use subtle::{Choice, ConditionallySelectable};

/// The dispatch's binary fallback, run on Pallas and Vesta, agrees with the
/// radix-27 multiplication they use, for Jacobian and affine inputs.
#[test]
fn native_scalar_multiplication_matches_jacobian_fallback() {
    macro_rules! check {
        ($name:ident, $base:ident, $scalar:ident) => {{
            let generator = $name::generator();
            let mut scalar = -$scalar::ONE;
            for _ in 0..16 {
                for point in [
                    $name::identity(),
                    generator,
                    -generator,
                    generator.double() + generator,
                ] {
                    let affine = point.to_affine();
                    let jacobian_product =
                        mul_curve_specific!($name, $base, general, jacobian, point, scalar);
                    let affine_product =
                        mul_curve_specific!($name, $base, general, affine, affine, scalar);
                    assert_eq!(point * scalar, jacobian_product);
                    assert_eq!(affine * scalar, affine_product);
                    assert_eq!(jacobian_product, affine_product);
                }
                scalar = scalar.square() + $scalar::ONE;
            }
        }};
    }
    check!(Ep, Fp, Fq);
    check!(Eq, Fq, Fp);
}
