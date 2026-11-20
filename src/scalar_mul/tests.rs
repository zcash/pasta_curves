use super::{Homogeneous, to_radix_3};
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

// The original six-product cross-product tail of RCB Algorithm 7 is retained
// here as an independent reference for the five-product formula.
fn six_product_add<F: Field>(p: &Homogeneous<F>, q: &Homogeneous<F>) -> Homogeneous<F> {
    let by_15 = |x: F| {
        let x2 = x + x;
        let x3 = x2 + x;
        let x6 = x3 + x3;
        x6 + x6 + x3
    };
    let t0 = p.x * q.x;
    let t1 = p.y * q.y;
    let t2 = p.z * q.z;
    let t3 = (p.x + p.y) * (q.x + q.y) - t0 - t1;
    let t4 = (p.y + p.z) * (q.y + q.z) - t1 - t2;
    let y3 = (p.x + p.z) * (q.x + q.z) - t0 - t2;
    let t0 = t0 + t0 + t0;
    let t2 = by_15(t2);
    let z3 = t1 + t2;
    let t1 = t1 - t2;
    let y3 = by_15(y3);
    let x3 = t4 * y3;
    let x3 = t3 * t1 - x3;
    let y3 = t1 * z3 + y3 * t0;
    let z3 = z3 * t4 + t0 * t3;
    Homogeneous {
        x: x3,
        y: y3,
        z: z3,
    }
}

fn assert_same<F: Field>(p: Homogeneous<F>, q: Homogeneous<F>) {
    assert_eq!(bool::from(p.z.is_zero()), bool::from(q.z.is_zero()));
    if !bool::from(p.z.is_zero()) {
        assert!(bool::from((p.x * q.z).ct_eq(&(q.x * p.z))));
        assert!(bool::from((p.y * q.z).ct_eq(&(q.y * p.z))));
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

            fn h(p: $affine) -> Homogeneous<$field> {
                let (x, y) = p.raw_xy_for_tests();
                Homogeneous::from_affine(x, y, p.is_identity())
            }

            #[test]
            fn complete_formulas_and_coordinates() {
                let identity = Homogeneous::<$field>::identity();
                let g = <$point>::generator();
                let p = h(g.to_affine());
                let q = h(g.double().to_affine());
                let r = h((g.double() + g).to_affine());
                let neg = h((-g).to_affine());
                let inputs = [identity, p, q, r, neg];
                for point in [<$point>::identity(), g, g.double(), g.double() + g, -g] {
                    let point_h = h(point.to_affine());
                    assert_same(point_h.triple(), h((point.double() + point).to_affine()));
                    assert_same(point_h.add(&p), h((point + g).to_affine()));
                }
                for a in inputs {
                    assert_same(a.double(), a.add(&a));
                    assert_same(a.triple(), a.double().add(&a));
                    for b in inputs {
                        assert_same(a.add(&b), six_product_add(&a, &b));
                    }
                }
                assert_same(p.add(&neg), identity);
                assert_same(p.add(&identity), p);
                assert_same(identity.add(&p), p);
                let (x, y, z) = identity.to_jacobian();
                assert_eq!((x, y, z), (<$field>::ZERO, <$field>::ZERO, <$field>::ZERO));
                for (zero_x, zero_y) in [
                    (<$field>::ZERO, <$field>::ZERO),
                    (<$field>::ONE, <$field>::ONE),
                    (<$field>::from(13), <$field>::from(21)),
                ] {
                    let arbitrary = Homogeneous::from_jacobian(zero_x, zero_y, <$field>::ZERO);
                    assert_eq!(arbitrary.to_jacobian(), (x, y, z));
                    assert_same(arbitrary.triple(), identity);
                    assert_same(arbitrary.add(&p), p);
                    assert_same(arbitrary.multiply(&[3; 32]), identity);
                }
                let factor = <$field>::from(7);
                let factor2 = factor.square();
                let factor3 = factor2 * factor;
                let scaled = Homogeneous::from_jacobian(p.x * factor2, p.y * factor3, factor);
                assert_same(scaled, p);
                assert_same(scaled.triple(), p.triple());
                assert_same(scaled.add(&q), p.add(&q));
                let affine_identity = <$affine>::identity();
                assert_same(h(affine_identity), identity);
                assert_same(h(affine_identity).multiply(&[3; 32]), identity);
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
                    let scaled = Homogeneous::from_jacobian(
                        hp.x * scale.square(),
                        hp.y * scale.square() * scale,
                        scale,
                    );
                    let actual = scaled.multiply(&scalar.to_repr());
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

// The elements where mul_small_limbs is most likely to go wrong: zero, one,
// the largest canonical element p - 1 (the largest h), its neighbours, and
// elements whose representation has a full top limb or carries across limbs.
fn small_mul_edge_elements<F: PrimeField>() -> [F; 8] {
    [
        F::ZERO,
        F::ONE,
        -F::ONE,
        -F::ONE - F::ONE,
        F::from(u64::MAX),
        F::from_u128(u128::MAX),
        F::TWO_INV,
        -F::TWO_INV,
    ]
}

fn check_small_mul<F: PrimeField + super::FormulaField>(x: F, k: u64) {
    assert_eq!(x.mul_small(k), x * F::from(k), "k = {k}");
}

fn small_mul_exhaustive<F: PrimeField + super::FormulaField>() {
    // Every k up to 2^11, which covers the formulas' constants (at most
    // 1080), and the largest k the helper accepts.
    for x in small_mul_edge_elements::<F>() {
        for k in (0..2048).chain([(1 << 32) - 1, 1 << 31, (1 << 31) - 1]) {
            check_small_mul(x, k);
        }
    }
}

#[test]
fn small_mul_matches_multiplication() {
    small_mul_exhaustive::<crate::Fp>();
    small_mul_exhaustive::<crate::Fq>();
}

proptest::proptest! {
    #[test]
    fn small_mul_matches_multiplication_fp(
        words in proptest::prelude::any::<[u64; 8]>(),
        k in 0u64..(1 << 32),
    ) {
        let x = crate::Fp::from_u512(words);
        check_small_mul(x, k);
        check_small_mul(x, k & 0x7ff);
    }

    #[test]
    fn small_mul_matches_multiplication_fq(
        words in proptest::prelude::any::<[u64; 8]>(),
        k in 0u64..(1 << 32),
    ) {
        let x = crate::Fq::from_u512(words);
        check_small_mul(x, k);
        check_small_mul(x, k & 0x7ff);
    }
}

// x3 sums three products and z3 two: the largest sums reduce_wide sees.
fn check_lazy_sum<F: PrimeField + super::FormulaField>(v: [F; 6]) {
    let ab = v[0].mul_wide(&v[1]);
    let cd = v[2].mul_wide(&v[3]);
    let ef = v[4].mul_wide(&v[5]);
    let two = super::add_wide(&ab, &cd);
    assert_eq!(F::reduce_wide(ab), v[0] * v[1]);
    assert_eq!(F::reduce_wide(two), v[0] * v[1] + v[2] * v[3]);
    assert_eq!(
        F::reduce_wide(super::add_wide(&two, &ef)),
        v[0] * v[1] + v[2] * v[3] + v[4] * v[5]
    );
}

fn lazy_sum_edges<F: PrimeField + super::FormulaField>() {
    // -1 has the largest canonical representation; its products give the
    // largest sums.
    let edges = small_mul_edge_elements::<F>();
    for &a in &edges {
        for &b in &edges {
            check_lazy_sum([a, b, a, b, a, b]);
            check_lazy_sum([a, -F::ONE, -F::ONE, b, -F::ONE, -F::ONE]);
        }
    }
}

#[test]
fn lazy_sum_matches_field_arithmetic() {
    lazy_sum_edges::<crate::Fp>();
    lazy_sum_edges::<crate::Fq>();
}

proptest::proptest! {
    #[test]
    fn lazy_sum_matches_field_arithmetic_random(
        words in proptest::prelude::any::<[[u64; 8]; 6]>(),
    ) {
        check_lazy_sum(words.map(crate::Fp::from_u512));
        check_lazy_sum(words.map(crate::Fq::from_u512));
    }
}
