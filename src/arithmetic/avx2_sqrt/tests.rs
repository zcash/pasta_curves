use super::*;
use crate::arithmetic::avx2_sqrt::Avx2Chain;
use ff::Field;
use std::{vec, vec::Vec};
use subtle::ConstantTimeEq;

const P_PALLAS: [u64; 4] = [0x992d30ed00000001, 0x224698fc094cf91b, 0, 1 << 62];
const P_VESTA: [u64; 4] = [0x8c46eb2100000001, 0x224698fc0994a8dd, 0, 1 << 62];

fn add(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut out = [0; 4];
    let mut carry = 0u128;
    for i in 0..4 {
        let t = a[i] as u128 + b[i] as u128 + carry;
        out[i] = t as u64;
        carry = t >> 64;
    }
    assert_eq!(carry, 0);
    out
}

fn sub(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut out = [0; 4];
    let mut borrow = 0i128;
    for i in 0..4 {
        let t = a[i] as i128 - b[i] as i128 - borrow;
        out[i] = t as u64;
        borrow = (t < 0) as i128;
    }
    assert_eq!(borrow, 0);
    out
}

fn lt(a: &[u64; 4], b: &[u64; 4]) -> bool {
    (0..4)
        .rev()
        .find(|&i| a[i] != b[i])
        .is_some_and(|i| a[i] < b[i])
}

fn pow2(k: usize) -> [u64; 4] {
    let mut out = [0; 4];
    out[k / 64] = 1 << (k % 64);
    out
}

/// Integers below `2p` that stress the column sums and the `2p` bound.
fn edge_values(p: &[u64; 4]) -> Vec<[u64; 4]> {
    let one = [1, 0, 0, 0];
    let two_p = add(p, p);
    let mut v = vec![
        [0; 4],
        one,
        sub(p, &one),
        *p,
        add(p, &one),
        sub(&two_p, &one),
        sub(&two_p, &[2, 0, 0, 0]),
    ];
    // Limbs 0 to 7 all at 2^29 - 1 under each top limb up to the largest
    // that stays below 2p: the maximal column sums.
    let low_ones = sub(&pow2(232), &one);
    for top in [0u64, 1, (1 << 22) - 1, 1 << 22, (1 << 23) - 1] {
        let x = add(&low_ones, &[0, 0, 0, top << 40]);
        if lt(&x, &two_p) {
            v.push(x);
        }
    }
    for k in (0..255).step_by(17) {
        if lt(&pow2(k), p) {
            v.push(sub(&two_p, &add(&pow2(k), &one)));
            v.push(sub(p, &pow2(k)));
        }
    }
    v
}

/// The radix-2^29 limbs of `v`, unnormalized as `mul` and `sqr` leave them.
#[target_feature(enable = "avx2")]
unsafe fn raw_limbs(v: &V) -> [[u64; 4]; L] {
    let mut out = [[0u64; 4]; L];
    for (dst, src) in out.iter_mut().zip(&v.0) {
        // SAFETY: the caller checked AVX2; this stores exactly the 32 bytes of `dst`.
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, *src) };
    }
    out
}

macro_rules! field_tests {
    ($name:ident, $field:ty, $p:expr) => {
        #[test]
        fn $name() {
            if !available() {
                return;
            }
            type F = $field;
            const P: [u64; 4] = $p;
            // Interprets an integer below 2p as an R'-domain residue, as
            // the arithmetic does.
            let int = |x: [u64; 4]| F::from_raw(x);
            let r_prime_inv = F::from(2).pow_vartime([261]).invert().unwrap();
            let two_p = add(&P, &P);

            let edges = edge_values(&P);
            let r_inv = F::from(2).pow_vartime([256]).invert().unwrap();
            for &value in &edges {
                let canonical = F::from_limbs_lazy(value);
                assert!(lt(&canonical.limbs(), &P));
                assert_eq!(canonical, int(value) * r_inv);
            }
            let check = |out: &V, expected: &[F; 4]| {
                // SAFETY: AVX2 was checked at the start of this test.
                let raw = unsafe { raw_limbs(out) };
                for lane in 0..4 {
                    for (i, limb) in raw.iter().enumerate() {
                        let bound = if i + 1 < L { 1 << BITS } else { 1 << 23 };
                        assert!(limb[lane] < bound, "limb {i} out of range");
                    }
                }
                // SAFETY: AVX2 was checked at the start of this test.
                let ints = unsafe { unpack(out) };
                for lane in 0..4 {
                    assert!(lt(&ints[lane], &two_p), "result not below 2p");
                    assert!(bool::from(int(ints[lane]).ct_eq(&expected[lane])));
                }
            };

            for a in edges.chunks(4) {
                let a: [[u64; 4]; 4] = core::array::from_fn(|k| a[k % a.len()]);
                // SAFETY: as above.
                let va = unsafe { pack(&a) };
                let sq = unsafe { sqr::<{ P[0] }, { P[1] }, { P[2] }, { P[3] }>(&va) };
                check(
                    &sq,
                    &core::array::from_fn(|k| int(a[k]).square() * r_prime_inv),
                );
                for b in edges.chunks(4) {
                    let b: [[u64; 4]; 4] = core::array::from_fn(|k| b[k % b.len()]);
                    let vb = unsafe { pack(&b) };
                    let m = unsafe { mul::<{ P[0] }, { P[1] }, { P[2] }, { P[3] }>(&va, &vb) };
                    check(
                        &m,
                        &core::array::from_fn(|k| int(a[k]) * int(b[k]) * r_prime_inv),
                    );
                }
            }

            // The whole chain against the scalar square root, on extreme
            // canonical Montgomery limbs.
            let canonical: Vec<[u64; 4]> = edges.into_iter().filter(|x| lt(x, &P)).collect();
            for us in canonical.chunks(8) {
                let u: [F; 8] = core::array::from_fn(|k| F::from_limbs_lazy(us[k % us.len()]));
                let got = F::sqrt_alt_avx2::<2, 8>(&u);
                let portable = F::sqrt_alt_lanes(&u);
                for (index, (u, (is_square, root))) in u.iter().zip(got).enumerate() {
                    let (want_square, want_root) = u.sqrt_alt();
                    assert_eq!(bool::from(is_square), bool::from(want_square));
                    assert!(bool::from(root.ct_eq(&want_root)));
                    assert_eq!(bool::from(portable[index].0), bool::from(want_square));
                    assert!(bool::from(portable[index].1.ct_eq(&want_root)));
                }
            }
        }
    };
}

field_tests!(pallas_base_edges, crate::Fp, P_PALLAS);
field_tests!(vesta_base_edges, crate::Fq, P_VESTA);

#[test]
fn cached_avx2_detection_matches_standard_runtime_detection() {
    let expected = std::is_x86_feature_detected!("avx2");
    assert_eq!(available(), expected);
    assert_eq!(available(), expected);
}
