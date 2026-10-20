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

impl Fq {
    // Preserve the pre-port portable operations as independent eager oracles.
    fn sqr_n_eager(&self, n: u32) -> Self {
        (0..n).fold(*self, |acc, _| acc.square_portable())
    }

    fn sqr_n_mul_eager(&self, n: u32, by: &Self) -> Self {
        self.sqr_n_eager(n).mul(by)
    }

    #[cfg(feature = "sqrt-table")]
    fn pow_by_t_minus1_over2_eager(&self) -> Self {
        let s10 = self.square_runtime();
        let s11 = s10 * self;
        let s111 = s11.sqr_n_mul_eager(1, self);
        let s1001 = s111 * s10;
        let s1011 = s1001 * s10;
        let s1101 = s1011 * s10;
        let sa = self.sqr_n_mul_eager(129, self);
        let sb = sa.sqr_n_mul_eager(7, &s1001);
        let sc = sb.sqr_n_mul_eager(7, &s1101);
        let sd = sc.sqr_n_mul_eager(4, &s11);
        let se = sd.sqr_n_mul_eager(6, &s111);
        let sf = se.sqr_n_mul_eager(3, &s111);
        let sg = sf.sqr_n_mul_eager(10, &s1001);
        let sh = sg.sqr_n_mul_eager(4, &s1001);
        let si = sh.sqr_n_mul_eager(5, &s1001);
        let sj = si.sqr_n_mul_eager(5, &s1001);
        let sk = sj.sqr_n_mul_eager(3, &s1001);
        let sl = sk.sqr_n_mul_eager(4, &s1011);
        let sm = sl.sqr_n_mul_eager(4, &s1011);
        let sn = sm.sqr_n_mul_eager(5, &s11);
        let so = sn.sqr_n_mul_eager(4, self);
        let sp = so.sqr_n_mul_eager(5, &s11);
        let sq = sp.sqr_n_mul_eager(4, &s111);
        let sr = sq.sqr_n_mul_eager(5, &s1011);
        let ss = sr.sqr_n_mul_eager(3, self);
        (0..4).fold(ss, |x, _| x.square_runtime())
    }

    fn sqrt_eager(&self) -> CtOption<Self> {
        #[cfg(feature = "sqrt-table")]
        {
            let (is_square, res) =
                FQ_TABLES.sqrt_alt_eager(self, &self.pow_by_t_minus1_over2_eager());
            CtOption::new(res, is_square)
        }
        #[cfg(not(feature = "sqrt-table"))]
        ff::helpers::sqrt_tonelli_shanks(self, T_MINUS1_OVER2)
    }
}

#[test]
fn lazy_square_and_sqrt_match_eager() {
    use rand::SeedableRng;

    let mut rng = rand_xorshift::XorShiftRng::from_seed([0x71; 16]);
    let max = [MODULUS.0[0] - 1, MODULUS.0[1], 0, MODULUS.0[3]];
    let boundaries = [
        Fq::zero(),
        Fq::one(),
        -Fq::one(),
        Fq::from_raw(max), // Integer q - 1, independently of negation.
        Fq(max),           // Largest canonical Montgomery residue.
        Fq::ROOT_OF_UNITY,
        Fq::from(5),
    ];

    for a in boundaries
        .into_iter()
        .chain((0..4_096).map(|_| Fq::random(&mut rng)))
    {
        assert_eq!(a.sqr_n_runtime(1).0, a.square_portable().0);
        assert_eq!(a.sqr_n_runtime(1).0, Field::square(&a).0);
        for n in [0, 1, 2, 3, 4, 7, 8, 11, 32, 129, 256, 512] {
            let eager = a.sqr_n_eager(n);
            assert_eq!(a.sqr_n_runtime(n).0, eager.0, "sqr_n, n = {n}");
            assert_eq!(crate::arithmetic::square_fq_n(&a, n).0, eager.0);
            for by in [Fq::zero(), Fq::one(), -Fq::one(), a] {
                let expected = a.sqr_n_mul_eager(n, &by);
                assert_eq!(a.sqr_n_mul_runtime(n, &by).0, expected.0);
                // Also exercise the new portable path when the assembly feature is enabled.
                let lazy = Fq(portable::sqr_n_lazy(&a.0, n, &MODULUS.0, INV));
                assert_eq!(lazy.mul_portable(&by).0, expected.0);
            }
        }

        #[cfg(feature = "sqrt-table")]
        {
            assert_eq!(
                a.pow_by_t_minus1_over2().0,
                a.pow_by_t_minus1_over2_eager().0
            );
            let (choice, root) = FQ_TABLES.sqrt_alt(&a);
            let (old_choice, old_root) =
                FQ_TABLES.sqrt_alt_eager(&a, &a.pow_by_t_minus1_over2_eager());
            assert_eq!(choice.unwrap_u8(), old_choice.unwrap_u8());
            assert_eq!(root.0, old_root.0);
        }
        let root = a.sqrt();
        let old_root = a.sqrt_eager();
        assert_eq!(root.is_some().unwrap_u8(), old_root.is_some().unwrap_u8());
        assert_eq!(
            root.unwrap_or(Fq::zero()).0,
            old_root.unwrap_or(Fq::zero()).0
        );
    }

    for nonsquare in [Fq::ROOT_OF_UNITY, Fq::from(5)] {
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
    let max = Fq([MODULUS.0[0] - 1, MODULUS.0[1], 0, MODULUS.0[3]]);
    for a in [Fq::zero(), Fq::one(), -Fq::one(), max]
        .into_iter()
        .chain((0..64).map(|_| Fq::random(&mut rng)))
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
