use super::*;

#[test]
fn low_half_square_matches_classical_and_multiplication() {
    use rand::SeedableRng;

    const ONE_SQUARED: Fp = Fp::one().square();
    assert_eq!(ONE_SQUARED, Fp::one());
    let classical =
        |t: [u64; 8]| Fp::montgomery_reduce(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    let low_half = |t: [u64; 8]| {
        Fp(portable::canonicalize(
            &portable::montgomery_reduce_low_lazy(&t, &MODULUS.0, INV),
            &MODULUS.0,
        ))
    };
    let check_square = |a: Fp| {
        let wide = a.square_unreduced();
        assert_eq!(wide, a.mul_unreduced(&a));
        assert_eq!(a.square(), classical(wide));
        assert_eq!(a.square(), low_half(wide));
        assert_eq!(a.square(), a * a);
        assert_eq!(a.square(), Field::square(&a));
    };
    // Field boundaries and canonical Montgomery limbs just below the modulus.
    for a in [
        Fp::zero(),
        Fp::one(),
        -Fp::one(),
        Fp::from(2),
        Fp::from(3),
        Fp::from_raw([u64::MAX; 4]),
        Fp([MODULUS.0[0] - 1, MODULUS.0[1], 0, MODULUS.0[3]]),
        Fp([u64::MAX, u64::MAX, u64::MAX, MODULUS.0[3] - 1]),
    ] {
        check_square(a);
    }
    for bit in 0..255 {
        let mut limbs = [0; 4];
        limbs[bit / 64] = 1 << (bit % 64);
        check_square(Fp(limbs));
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
        let a = Fp::random(&mut rng);
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

impl Fp {
    // Preserve the pre-port portable operations as independent eager oracles.
    fn sqr_n_eager(&self, n: u32) -> Self {
        (0..n).fold(*self, |acc, _| acc.square_portable())
    }

    fn sqr_n_mul_eager(&self, n: u32, by: &Self) -> Self {
        self.sqr_n_eager(n).mul(by)
    }

    #[cfg(feature = "sqrt-table")]
    fn pow_by_t_minus1_over2_eager(&self) -> Self {
        let r10 = self.square_runtime();
        let r11 = r10 * self;
        let r110 = r11.square_runtime();
        let r111 = r110 * self;
        let r1001 = r111 * r10;
        let r1101 = r111 * r110;
        let ra = self.sqr_n_mul_eager(129, self);
        let rb = ra.sqr_n_mul_eager(7, &r1001);
        let rc = rb.sqr_n_mul_eager(7, &r1101);
        let rd = rc.sqr_n_mul_eager(4, &r11);
        let re = rd.sqr_n_mul_eager(6, &r111);
        let rf = re.sqr_n_mul_eager(3, &r111);
        let rg = rf.sqr_n_mul_eager(10, &r1001);
        let rh = rg.sqr_n_mul_eager(5, &r1001);
        let ri = rh.sqr_n_mul_eager(4, &r1001);
        let rj = ri.sqr_n_mul_eager(3, &r111);
        let rk = rj.sqr_n_mul_eager(4, &r1001);
        let rl = rk.sqr_n_mul_eager(5, &r11);
        let rm = rl.sqr_n_mul_eager(4, &r111);
        let rn = rm.sqr_n_mul_eager(4, &r11);
        let ro = rn.sqr_n_mul_eager(6, &r1001);
        let rp = ro.sqr_n_mul_eager(5, &r1101);
        let rq = rp.sqr_n_mul_eager(4, &r11);
        let rr = rq.sqr_n_mul_eager(7, &r111);
        let rs = rr.sqr_n_mul_eager(3, &r11);
        rs.square_runtime()
    }

    fn sqrt_eager(&self) -> CtOption<Self> {
        #[cfg(feature = "sqrt-table")]
        {
            let (is_square, res) =
                FP_TABLES.sqrt_alt_eager(self, &self.pow_by_t_minus1_over2_eager());
            CtOption::new(res, is_square)
        }
        #[cfg(not(feature = "sqrt-table"))]
        ff::helpers::sqrt_tonelli_shanks(self, T_MINUS1_OVER2)
    }
}

#[test]
fn lazy_square_and_sqrt_match_eager() {
    use rand::SeedableRng;

    let mut rng = rand_xorshift::XorShiftRng::from_seed([0x46; 16]);
    let max = [MODULUS.0[0] - 1, MODULUS.0[1], 0, MODULUS.0[3]];
    let boundaries = [
        Fp::zero(),
        Fp::one(),
        -Fp::one(),
        Fp::from_raw(max), // Integer p - 1, independently of negation.
        Fp(max),           // Largest canonical Montgomery residue.
        Fp::ROOT_OF_UNITY,
        Fp::from(5),
    ];

    for a in boundaries
        .into_iter()
        .chain((0..4_096).map(|_| Fp::random(&mut rng)))
    {
        assert_eq!(a.sqr_n_runtime(1).0, a.square_portable().0);
        assert_eq!(a.sqr_n_runtime(1).0, Field::square(&a).0);
        for n in [0, 1, 2, 3, 4, 7, 8, 11, 32, 129, 256, 512] {
            let eager = a.sqr_n_eager(n);
            assert_eq!(a.sqr_n_runtime(n).0, eager.0, "sqr_n, n = {n}");
            assert_eq!(crate::arithmetic::square_fp_n(&a, n).0, eager.0);
            for by in [Fp::zero(), Fp::one(), -Fp::one(), a] {
                let expected = a.sqr_n_mul_eager(n, &by);
                assert_eq!(a.sqr_n_mul_runtime(n, &by).0, expected.0);
                // Also exercise the new portable path when the assembly feature is enabled.
                let lazy = Fp(portable::sqr_n_lazy(&a.0, n, &MODULUS.0, INV));
                assert_eq!(lazy.mul_portable(&by).0, expected.0);
            }
        }

        #[cfg(feature = "sqrt-table")]
        {
            assert_eq!(
                a.pow_by_t_minus1_over2().0,
                a.pow_by_t_minus1_over2_eager().0
            );
            let (choice, root) = FP_TABLES.sqrt_alt(&a);
            let (old_choice, old_root) =
                FP_TABLES.sqrt_alt_eager(&a, &a.pow_by_t_minus1_over2_eager());
            assert_eq!(choice.unwrap_u8(), old_choice.unwrap_u8());
            assert_eq!(root.0, old_root.0);
        }
        let root = a.sqrt();
        let old_root = a.sqrt_eager();
        assert_eq!(root.is_some().unwrap_u8(), old_root.is_some().unwrap_u8());
        assert_eq!(
            root.unwrap_or(Fp::zero()).0,
            old_root.unwrap_or(Fp::zero()).0
        );
    }

    for nonsquare in [Fp::ROOT_OF_UNITY, Fp::from(5)] {
        assert!(bool::from(nonsquare.sqrt().is_none()));
        assert!(bool::from(nonsquare.sqrt_eager().is_none()));
    }
}

#[test]
fn lazy_square_accumulator_stays_below_two_moduli() {
    use rand::SeedableRng;

    let mut rng = rand_xorshift::XorShiftRng::from_seed([0x9d; 16]);
    let (m0, carry) = adc(MODULUS.0[0], MODULUS.0[0], 0);
    let (m1, carry) = adc(MODULUS.0[1], MODULUS.0[1], carry);
    let (m2, carry) = adc(MODULUS.0[2], MODULUS.0[2], carry);
    let (m3, carry) = adc(MODULUS.0[3], MODULUS.0[3], carry);
    assert_eq!(carry, 0);
    let two_moduli = [m0, m1, m2, m3];
    let max = Fp([MODULUS.0[0] - 1, MODULUS.0[1], 0, MODULUS.0[3]]);
    for a in [Fp::zero(), Fp::one(), -Fp::one(), max]
        .into_iter()
        .chain((0..64).map(|_| Fp::random(&mut rng)))
    {
        let mut lazy = a.0;
        let mut eager = a;
        for _ in 0..2_048 {
            lazy = portable::sqr_n_lazy(&lazy, 1, &MODULUS.0, INV);
            eager = eager.square_portable();
            assert!(lazy.iter().rev().cmp(two_moduli.iter().rev()).is_lt());
            assert_eq!(portable::canonicalize(&lazy, &MODULUS.0), eager.0);
        }
    }
}
