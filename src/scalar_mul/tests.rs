use super::{multiply, to_radix_3};
use crate::curves::projective::Projective;
use crate::curves::projective::tests::assert_same;
use ff::{Field, PrimeField};
use group::{Curve, CurveAffine, Group};
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use subtle::ConditionallySelectable;

fn check_reconstruction(by: &[u8; 32]) -> [u8; 162] {
    let digits = to_radix_3(by);
    let mut reconstructed = [0u8; 32];
    for &digit in digits.iter().rev() {
        assert!(digit < 3);
        let mut carry = u16::from(digit);
        for byte in &mut reconstructed {
            let value = u16::from(*byte) * 3 + carry;
            *byte = value as u8;
            carry = value >> 8;
        }
        assert_eq!(carry, 0);
    }
    let mut expected = *by;
    expected[31] &= 0x7f;
    assert_eq!(reconstructed, expected);
    digits
}

#[test]
fn recoder_boundaries_and_neighbors() {
    check_reconstruction(&[0; 32]);
    check_reconstruction(&[0xff; 32]);
    let mut top_bit = [0; 32];
    top_bit[31] = 0x80;
    assert_eq!(check_reconstruction(&top_bit), [0; 162]);
    let mut power = [0u8; 32];
    power[0] = 1;
    for exponent in 0..161 {
        let digits = check_reconstruction(&power);
        assert_eq!(digits[exponent], 1);
        assert_eq!(digits.iter().map(|&d| u32::from(d)).sum::<u32>(), 1);
        for adjustment in [-1i16, 1] {
            let mut neighbor = power;
            let mut carry = adjustment;
            for byte in &mut neighbor {
                let v = i16::from(*byte) + carry;
                *byte = v as u8;
                carry = v >> 8;
            }
            check_reconstruction(&neighbor);
        }
        if exponent < 160 {
            let mut carry = 0u16;
            for byte in &mut power {
                let v = u16::from(*byte) * 3 + carry;
                *byte = v as u8;
                carry = v >> 8;
            }
            assert_eq!(carry, 0);
        }
    }
    let mut state = 0x8e83_96b3_fcb2_7a45u64;
    for _ in 0..128 {
        let mut by = [0u8; 32];
        for chunk in by.chunks_exact_mut(8) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            chunk.copy_from_slice(&state.to_le_bytes());
        }
        check_reconstruction(&by);
    }
}

fn binary<P: Group>(point: P, bytes: &[u8; 32]) -> P {
    let mut acc = P::identity();
    for bit in bytes
        .iter()
        .rev()
        .flat_map(|byte| (0..8).rev().map(move |i| (byte >> i) & 1))
        .skip(1)
    {
        acc = acc.double();
        if bit != 0 {
            acc += point;
        }
    }
    acc
}

macro_rules! curve_tests {
    ($name:ident, $point:ty, $affine:ty, $field:ty, $scalar:ty) => {
        mod $name {
            use super::*;

            use crate::curves::projective::tests::$name::h;

            #[test]
            fn identity_times_any_scalar_is_identity() {
                let identity = Projective::<$field>::identity();
                for (x, y) in [
                    (<$field>::ZERO, <$field>::ZERO),
                    (<$field>::ONE, <$field>::ONE),
                    (<$field>::from(13), <$field>::from(21)),
                ] {
                    let arbitrary = Projective::from_jacobian(x, y, <$field>::ZERO);
                    assert_same(multiply(&arbitrary, &[3; 32]), identity);
                }
                assert_same(multiply(&h(<$affine>::identity()), &[3; 32]), identity);
            }

            #[test]
            fn scalar_multiplication_matches_binary() {
                let g = <$point>::generator();
                let points = [
                    <$point>::identity(),
                    g,
                    -g,
                    g.double(),
                    g.double() + g,
                    g.double().double() + g,
                ];
                let mut scalars = vec![
                    <$scalar>::ZERO,
                    <$scalar>::ONE,
                    <$scalar>::from(2),
                    <$scalar>::from(3),
                    <$scalar>::from(8),
                    <$scalar>::from(9),
                    <$scalar>::from(26),
                    <$scalar>::from(27),
                    -<$scalar>::ONE,
                ];
                let mut power = <$scalar>::ONE;
                for _ in 0..160 {
                    scalars.push(power - <$scalar>::ONE);
                    scalars.push(power);
                    scalars.push(power + <$scalar>::ONE);
                    power *= <$scalar>::from(3);
                }
                let mut rng = XorShiftRng::from_seed([0x53; 16]);
                for _ in 0..32 {
                    scalars.push(<$scalar>::random(&mut rng));
                }
                for scalar in scalars {
                    let repr = scalar.to_repr();
                    let mut bytes = [0; 32];
                    bytes.copy_from_slice(repr.as_ref());
                    for point in points {
                        let expected = binary(point, &bytes);
                        assert_eq!(point * scalar, expected);
                        assert_eq!(point.to_affine() * scalar, expected);
                    }
                }
                // Generate the points with the independent binary oracle too,
                // rather than depending on the multiplication under test.
                for _ in 0..32 {
                    let point_scalar = <$scalar>::random(&mut rng);
                    let point = binary(g, &point_scalar.to_repr());
                    let scalar = <$scalar>::random(&mut rng);
                    let expected = binary(point, &scalar.to_repr());
                    assert_eq!(point * scalar, expected);
                    assert_eq!(point.to_affine() * scalar, expected);
                    let hp = h(point.to_affine());
                    let scale = <$field>::random(&mut rng);
                    let scale =
                        <$field>::conditional_select(&scale, &<$field>::ONE, scale.is_zero());
                    let scaled = Projective::from_jacobian(
                        hp.x * scale.square(),
                        hp.y * scale.square() * scale,
                        scale,
                    );
                    let actual = multiply(&scaled, &scalar.to_repr());
                    assert_same(actual, h(expected.to_affine()));
                    let (x, y, z) = actual.to_jacobian();
                    let z2 = z.square();
                    assert_eq!(
                        y.square(),
                        x.square() * x + <$field>::from(5) * z2.square() * z2
                    );
                }
            }
        }
    };
}

curve_tests!(
    pallas,
    crate::pallas::Point,
    crate::pallas::Affine,
    crate::Fp,
    crate::Fq
);
curve_tests!(
    vesta,
    crate::vesta::Point,
    crate::vesta::Affine,
    crate::Fq,
    crate::Fp
);
