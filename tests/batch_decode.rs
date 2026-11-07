#![cfg(feature = "alloc")]

use ff::{Field, PrimeField};
use group::{Curve, Group};
use pasta_curves::{arithmetic::CurveAffine, pallas, vesta};
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

fn check<C: CurveAffine<Repr = [u8; 32]>>() {
    let mut rng = XorShiftRng::from_seed([19; 16]);
    let mut encodings = vec![[0; 32], [255; 32]];
    let mut signed_zero = [0; 32];
    signed_zero[31] = 128;
    encodings.push(signed_zero);
    let max_canonical = (-C::Base::ONE).to_repr();
    let mut boundary: [u8; 32] = max_canonical.as_ref().try_into().unwrap();
    // Both Pasta moduli end in 0x01, so these low-byte increments cannot carry.
    assert_eq!(boundary[0], 0);
    for _ in 0..3 {
        encodings.push(boundary);
        let mut signed = boundary;
        signed[31] |= 128;
        encodings.push(signed);
        boundary[0] += 1;
    }
    for _ in 0..257 {
        encodings.push(C::CurveExt::random(&mut rng).to_affine().to_bytes());
        let mut bytes = [0u8; 32];
        rng.fill_bytes(&mut bytes);
        encodings.push(bytes);
    }
    // Every tail width, identity, noncanonical x, nonsquare x and both sign bits.
    for len in 0..=encodings.len() {
        let actual = C::batch_decode(&encodings[..len]);
        let expected = encodings[..len]
            .iter()
            .map(|bytes| Option::<C>::from(C::from_bytes(bytes)))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "batch length {len}");
    }
}

#[test]
fn batch_decode_matches_scalar_pallas_and_vesta() {
    check::<pallas::Affine>();
    check::<vesta::Affine>();
}
