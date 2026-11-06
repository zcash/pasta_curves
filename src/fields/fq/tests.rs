use super::*;

#[test]
fn low_half_square_matches_classical_and_multiplication() {
    use rand::SeedableRng;

    const ONE_SQUARED: Fq = Fq::one().square();
    assert_eq!(ONE_SQUARED, Fq::one());
    let classical =
        |t: [u64; 8]| Fq::montgomery_reduce(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    let low_half = |t: [u64; 8]| {
        Fq(portable::canonicalize(
            &portable::montgomery_reduce_low_lazy(&t, &MODULUS.0, INV),
            &MODULUS.0,
        ))
    };
    let check_square = |a: Fq| {
        let wide = a.square_unreduced();
        assert_eq!(wide, a.mul_unreduced(&a));
        assert_eq!(a.square(), classical(wide));
        assert_eq!(a.square(), low_half(wide));
        assert_eq!(a.square(), a * a);
        assert_eq!(a.square(), Field::square(&a));
    };
    // Field boundaries and canonical Montgomery limbs just below the modulus.
    for a in [
        Fq::zero(),
        Fq::one(),
        -Fq::one(),
        Fq::from(2),
        Fq::from(3),
        Fq::from_raw([u64::MAX; 4]),
        Fq([MODULUS.0[0] - 1, MODULUS.0[1], 0, MODULUS.0[3]]),
        Fq([u64::MAX, u64::MAX, u64::MAX, MODULUS.0[3] - 1]),
    ] {
        check_square(a);
    }
    for bit in 0..255 {
        let mut limbs = [0; 4];
        limbs[bit / 64] = 1 << (bit % 64);
        check_square(Fq(limbs));
    }

    // Exercise the reducer's full domain, not just square products:
    // arbitrary low limbs and a high half below the modulus.
    for lo in [[0; 4], [u64::MAX; 4]] {
        for hi in [[0; 4], [MODULUS.0[0] - 1, MODULUS.0[1], 0, MODULUS.0[3]]] {
            let t = [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]];
            assert_eq!(low_half(t), classical(t));
        }
    }
    let mut rng = rand_xorshift::XorShiftRng::from_seed([0x3c; 16]);
    for _ in 0..20_000 {
        let a = Fq::random(&mut rng);
        check_square(a);
        let t = [
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
            rng.try_next_u64().unwrap(),
            a.0[0],
            a.0[1],
            a.0[2],
            a.0[3],
        ];
        assert_eq!(low_half(t), classical(t));
    }
}
