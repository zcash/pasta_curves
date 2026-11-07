use super::{Fp, Fq};
use crate::arithmetic::VartimeField;
use ff::Field;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

fn check_equality<F: VartimeField>(values: &[F]) {
    for lhs in values {
        assert_eq!(lhs.is_zero_vartime(), bool::from(lhs.is_zero()));
        for rhs in values {
            assert_eq!(lhs.eq_vartime(rhs), bool::from(lhs.ct_eq(rhs)));
            assert_eq!(lhs.eq_vartime(rhs), lhs == rhs);
        }
    }
}

#[test]
fn public_equality_matches_constant_time_equality() {
    check_equality(&[Fp::ZERO, Fp::ONE, Fp::from(2), -Fp::ONE]);
    check_equality(&[Fq::ZERO, Fq::ONE, Fq::from(2), -Fq::ONE]);
    // Canonical Montgomery residues differing in each individual limb.
    let limbs = core::array::from_fn::<_, 4, _>(|i| {
        let mut limbs = [0; 4];
        limbs[i] = 1;
        limbs
    });
    check_equality(&limbs.map(Fp));
    check_equality(&limbs.map(Fq));
    let mut rng = XorShiftRng::from_seed([0x61; 16]);
    for _ in 0..64 {
        let p = Fp::try_random(&mut rng).unwrap();
        let q = Fq::try_random(&mut rng).unwrap();
        check_equality(&[Fp::ZERO, p, -p, p.square()]);
        check_equality(&[Fq::ZERO, q, -q, q.square()]);
    }
}
