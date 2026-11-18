use alloc::vec::Vec;

use ff::Field;
use group::{Curve, Group};
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

use super::CombSum;
use crate::arithmetic::PreparedFixedBaseSum;
use crate::glv::GlvParams;
use crate::{pallas, vesta};

/// The comb against native multiplication: single and batched sums, omitted
/// trailing scalars, an identity base, edge scalars, and group sizes and
/// strides at and around their boundaries (a stride of 1 stores every row; a
/// stride past the scalar's bit length stores one).
fn matches_native<C: GlvParams>() {
    let mut rng = XorShiftRng::seed_from_u64(0xc0b);
    let mut bases: Vec<C::AffineExt> = (0..10).map(|_| C::random(&mut rng).to_affine()).collect();
    bases.push(C::identity().to_affine());
    let mut scalars: Vec<C::ScalarExt> = alloc::vec![
        C::ScalarExt::ZERO,
        C::ScalarExt::ONE,
        -C::ScalarExt::ONE,
        C::ScalarExt::from(2),
        C::ScalarExt::from(u64::MAX),
    ];
    scalars.extend((0..6).map(|_| C::ScalarExt::random(&mut rng)));

    for (group_bits, stride) in [(1, 8), (3, 1), (4, 5), (7, 8), (8, 4), (11, 8), (16, 300)] {
        let comb = CombSum::<C>::new(&bases, group_bits, stride);
        assert_eq!(comb.num_bases(), bases.len());
        let native = |s: &[C::ScalarExt]| -> C { s.iter().zip(&bases).map(|(k, p)| *p * k).sum() };
        for len in [0, 1, 7, bases.len()] {
            let s = &scalars[..len];
            assert_eq!(
                comb.sum_vartime(s),
                native(s),
                "{group_bits} x {stride}, {len} scalars"
            );
        }
        let batch: Vec<&[C::ScalarExt]> = alloc::vec![&scalars, &scalars[..1], &[], &scalars[3..]];
        let expected: Vec<C> = batch.iter().map(|s| native(s)).collect();
        let mut output = alloc::vec![C::generator(); batch.len()];
        comb.sum_batch_vartime(&batch, &mut output);
        assert_eq!(output, expected, "{group_bits} x {stride}");
    }

    let empty = CombSum::<C>::new(&[], 8, 8);
    assert_eq!(empty.sum_vartime(&[]), C::identity());
}

#[test]
fn comb_matches_native() {
    matches_native::<pallas::Point>();
    matches_native::<vesta::Point>();
}

#[test]
#[should_panic(expected = "at most one scalar per prepared base")]
fn comb_rejects_extra_scalars() {
    let comb = CombSum::<pallas::Point>::new(&[pallas::Point::generator().to_affine()], 8, 8);
    comb.sum_vartime(&[pallas::Scalar::ONE, pallas::Scalar::ONE]);
}

#[test]
#[should_panic(expected = "a comb groups 1 to 16 bases")]
fn comb_rejects_wide_groups() {
    CombSum::<pallas::Point>::new(&[], 17, 8);
}

/// `try_prepare_fixed_base_sum` picks the positioned tables below
/// `COMB_MIN_BASES` bases and the comb from it, in balanced groups of at most
/// `COMB_MAX_GROUP_BITS` bases; both give the native sum.
#[test]
fn prepare_picks_the_strategy() {
    use crate::arithmetic::CurveExt;

    let mut rng = XorShiftRng::seed_from_u64(0x5e1);
    for (n, strategy) in [
        (7, "FixedBaseSum"),
        (8, "CombSum { bases: 8, group_bits: 8,"),
        (11, "CombSum { bases: 11, group_bits: 11,"),
        (12, "CombSum { bases: 12, group_bits: 6,"),
    ] {
        let bases: Vec<pallas::Affine> = (0..n)
            .map(|_| pallas::Point::random(&mut rng).to_affine())
            .collect();
        let prepared = pallas::Point::try_prepare_fixed_base_sum(&bases).unwrap();
        let debug = alloc::format!("{prepared:?}");
        assert!(debug.starts_with(strategy), "{n} bases: {debug}");
        let scalars: Vec<pallas::Scalar> =
            (0..n).map(|_| pallas::Scalar::random(&mut rng)).collect();
        let expected: pallas::Point = scalars.iter().zip(&bases).map(|(k, p)| *p * k).sum();
        assert_eq!(prepared.sum_vartime(&scalars), expected, "{n} bases");
    }
}

/// The bytes of a comb that `prepare_fixed_base_sum` would build over
/// `bases` bases, from the group sizes it chooses.
fn comb_bytes<C: GlvParams>(bases: usize) -> usize {
    let groups = bases.div_ceil(super::super::COMB_MAX_GROUP_BITS);
    let group_bits = bases.div_ceil(groups);
    let rows = super::scalar_bits::<C>().div_ceil(super::super::COMB_STRIDE);
    (0..bases)
        .step_by(group_bits)
        .map(|start| ((1 << group_bits.min(bases - start)) - 1) * rows)
        .sum::<usize>()
        * core::mem::size_of::<C::AffineExt>()
}

/// Pins the comb's memory. It is not monotone in the number of bases: a
/// 12th base splits one group of 11 into two of 6.
fn check_table_size<C: GlvParams>() {
    let mut rng = XorShiftRng::seed_from_u64(0x51e);
    for (n, bytes) in [(8usize, 1_044_480), (11, 8_384_512), (12, 516_096)] {
        let bases: Vec<C::AffineExt> = (0..n).map(|_| C::random(&mut rng).to_affine()).collect();
        let groups = n.div_ceil(super::super::COMB_MAX_GROUP_BITS);
        let comb = CombSum::<C>::new(&bases, n.div_ceil(groups), super::super::COMB_STRIDE);
        let measured = comb.tables.capacity() * core::mem::size_of::<C::AffineExt>();
        assert_eq!(measured, bytes, "{n} bases");
        assert_eq!(comb_bytes::<C>(n), bytes, "{n} bases");
    }

    // A comb over the 2,050 bases of an Orchard verifier's MSM (halo2's 2^11
    // generators with W and U), as `prepare_fixed_base_sum` would build it:
    // 1.45 GiB, ten times the positioned tables; computed, not allocated.
    assert_eq!(comb_bytes::<C>(2_050), 1_559_580_672);
}

#[test]
fn comb_table_size() {
    check_table_size::<pallas::Point>();
    check_table_size::<vesta::Point>();
}
