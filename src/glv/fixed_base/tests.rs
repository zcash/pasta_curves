use alloc::vec::Vec;

#[cfg(feature = "glv")]
use ff::{Field, FromUniformBytes, PrimeField};
#[cfg(feature = "glv")]
use group::CurveAffine as _;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

use super::{Booth7, BoothTable, FixedBaseSum, GlvParams, MAGNITUDES, WINDOWS};
#[cfg(feature = "glv")]
use crate::arithmetic::CurveExt;
#[cfg(feature = "glv")]
use crate::glv::Recoding;
use crate::{pallas, vesta};

#[cfg(feature = "glv")]
fn check<C: GlvParams>()
where
    C::ScalarExt: FromUniformBytes<64>,
{
    let mut rng = XorShiftRng::seed_from_u64(0x6c76);
    let bases: Vec<C::AffineExt> = (0..5)
        .map(|_| C::random(&mut rng).to_affine())
        .chain(core::iter::once(C::identity().to_affine()))
        .collect();
    let table = C::try_prepare_fixed_base_sum(&bases).expect("Pasta has a fixed-base backend");
    assert_eq!(table.num_bases(), bases.len());
    let empty = C::try_prepare_fixed_base_sum(&[]).expect("Pasta supports empty fixed-base sums");
    assert_eq!(empty.num_bases(), 0);
    assert_eq!(empty.sum_vartime(&[]), C::identity());
    let mut output = [C::generator(); 2];
    empty.sum_batch_vartime(&[&[], &[]], &mut output);
    assert_eq!(output, [C::identity(); 2]);

    let mut edges = Vec::from([
        C::ScalarExt::ZERO,
        C::ScalarExt::ONE,
        -C::ScalarExt::ONE,
        C::ScalarExt::from(63),
        C::ScalarExt::from(64),
        C::ScalarExt::from(65),
        -C::ScalarExt::from(64),
        C::ScalarExt::from(u64::MAX),
        C::ScalarExt::from_u128(u128::MAX),
    ]);
    edges.extend((0..64).map(|_| C::ScalarExt::random(&mut rng)));

    for chunk in edges.chunks(bases.len()) {
        let expected: C = chunk.iter().zip(&bases).map(|(k, p)| *p * k).sum();
        assert_eq!(table.sum_vartime(chunk), expected);
    }
    // Fewer scalars than bases leaves the rest at zero.
    for len in 0..=bases.len() {
        let scalars = &edges[edges.len() - len..];
        let expected: C = scalars.iter().zip(&bases).map(|(k, p)| *p * k).sum();
        assert_eq!(table.sum_vartime(scalars), expected, "{len} scalars");
    }
    assert_eq!(WINDOWS, 19);

    // Exercise carries through every signed window independently of the
    // GLV decomposition's distribution of half-scalars.
    let positioned = super::FixedBaseSum::<C, Booth7>::new(&bases[..1]);
    let zero = (false, 0u128).into();
    for half in [63, 64, 65, (1u128 << 127) - 1, u128::MAX] {
        for negative in [false, true] {
            for endo in [false, true] {
                let signed = (negative, half).into();
                let digits = if endo { (zero, signed) } else { (signed, zero) };
                let mut actual = C::identity();
                for (window, table) in positioned.tables.iter().enumerate() {
                    <Booth7 as Recoding<C>>::column_terms(
                        table,
                        &digits,
                        window,
                        |point, negate| {
                            if negate {
                                actual -= point;
                            } else {
                                actual += point;
                            }
                        },
                    );
                }
                let mut expected = bases[0] * C::ScalarExt::from_u128(half);
                if endo {
                    expected = expected.endo();
                }
                if negative {
                    expected = -expected;
                }
                assert_eq!(actual, expected);
            }
        }
    }
}

#[cfg(feature = "glv")]
#[test]
fn fixed_base_sum_matches_native_mul() {
    check::<pallas::Point>();
    check::<vesta::Point>();
}

/// The batched sums equal the per-sum ones, including identity bases,
/// zero and short scalar vectors, and pairs that hit the equal-x fallback.
#[cfg(feature = "glv")]
fn check_batch<C: GlvParams>()
where
    C::ScalarExt: FromUniformBytes<64>,
{
    let mut rng = XorShiftRng::seed_from_u64(0x6261);
    let p = C::random(&mut rng).to_affine();
    // A repeated base with equal or opposite scalars makes identical or
    // negated terms, so the trees meet doublings and cancellations.
    let bases = [
        p,
        p,
        C::random(&mut rng).to_affine(),
        C::identity().to_affine(),
    ];
    let table = C::try_prepare_fixed_base_sum(&bases).expect("Pasta has a fixed-base backend");
    let k = C::ScalarExt::random(&mut rng);
    let mut columns: Vec<Vec<C::ScalarExt>> = Vec::from([
        Vec::new(),
        Vec::from([C::ScalarExt::ZERO]),
        Vec::from([k, k]),
        Vec::from([k, -k]),
        Vec::from([k, k, k, k]),
        Vec::from([C::ScalarExt::ONE, -C::ScalarExt::ONE, C::ScalarExt::ONE]),
    ]);
    columns.extend((0..20).map(|i| {
        (0..(i % 5))
            .map(|_| C::ScalarExt::random(&mut rng))
            .collect()
    }));
    for count in [0, 1, 2, 7, columns.len()] {
        let slices: Vec<&[C::ScalarExt]> = columns[..count].iter().map(Vec::as_slice).collect();
        let expected: Vec<C> = slices
            .iter()
            .map(|s| s.iter().zip(&bases).map(|(k, p)| *p * k).sum())
            .collect();
        for (scalars, expected) in slices.iter().zip(&expected) {
            assert_eq!(table.sum_vartime(scalars), *expected);
        }
        let mut output = Vec::from_iter(core::iter::repeat_n(C::generator(), count));
        table.sum_batch_vartime(&slices, &mut output);
        assert_eq!(output, expected, "{count} sums");
    }
}

#[cfg(feature = "glv")]
#[test]
fn fixed_base_sum_batch_matches_single() {
    check_batch::<pallas::Point>();
    check_batch::<vesta::Point>();
}

/// The bytes a prepared table holds per base: `WINDOWS` tables of
/// `MAGNITUDES` affine points.
const BYTES_PER_BASE: usize = 77_824;

/// Pins the memory of prepared tables for the shapes callers use, so that a
/// change of window width or layout shows up as a test failure rather than as
/// a silent change of footprint.
fn check_table_size<C: GlvParams>() {
    assert_eq!(
        WINDOWS * MAGNITUDES * core::mem::size_of::<C::AffineExt>(),
        BYTES_PER_BASE,
    );
    assert_eq!(
        core::mem::size_of::<BoothTable<C>>(),
        MAGNITUDES * core::mem::size_of::<C::AffineExt>(),
        "a table holds its points with no padding",
    );

    // Orchard's instance commitment sums over seven bases: prepare them and
    // measure what is actually allocated.
    let mut rng = XorShiftRng::seed_from_u64(0x7a62);
    let bases: Vec<C::AffineExt> = (0..7).map(|_| C::random(&mut rng).to_affine()).collect();
    let table = FixedBaseSum::<C, Booth7>::new(&bases);
    assert_eq!(table.tables.len(), bases.len() * WINDOWS);
    let bytes = table.tables.capacity() * core::mem::size_of::<BoothTable<C>>();
    assert_eq!(bytes, 544_768, "seven bases, 532 KiB");
    assert_eq!(bytes, 7 * BYTES_PER_BASE);

    // Preparing halo2's 2^11 generators with W and U, the 2,050 bases of an
    // Orchard verifier's MSM, would hold 152 MiB; computed, not allocated.
    assert_eq!(2_050 * BYTES_PER_BASE, 159_539_200);
}

#[test]
fn fixed_base_table_size() {
    check_table_size::<pallas::Point>();
    check_table_size::<vesta::Point>();
}

#[cfg(feature = "glv")]
#[test]
#[should_panic(expected = "one output per sum")]
fn fixed_base_sum_batch_rejects_mismatched_output() {
    let table = pallas::Point::try_prepare_fixed_base_sum(&[pallas::Affine::generator()])
        .expect("Pasta has a fixed-base backend");
    let mut output = [pallas::Point::from(pallas::Affine::generator()); 1];
    table.sum_batch_vartime(&[&[], &[]], &mut output);
}

#[cfg(feature = "glv")]
#[test]
#[should_panic(expected = "at most one scalar per prepared base")]
fn fixed_base_sum_rejects_extra_scalars() {
    let table = pallas::Point::try_prepare_fixed_base_sum(&[pallas::Affine::generator()])
        .expect("Pasta has a fixed-base backend");
    let _ = table.sum_vartime(&[pallas::Scalar::ONE, pallas::Scalar::ONE]);
}

/// The shared `Recoding` laws, for the fixed-base sums' signed-Booth
/// recoding.
mod pbt {
    use group::Group;
    use proptest::prelude::*;

    use super::super::Booth7;
    use crate::glv::{conformance as law, conformance::scalar_strategy};

    macro_rules! booth_pbt {
        ($mod_name:ident, $curve:ty) => {
            mod $mod_name {
                use super::*;

                type Scalar = <$curve as crate::arithmetic::CurveExt>::ScalarExt;

                proptest! {
                    #[test]
                    fn recoding_laws(
                        s in scalar_strategy::<Scalar>(),
                        a in scalar_strategy::<Scalar>(),
                        b in scalar_strategy::<Scalar>(),
                    ) {
                        use ff::Field;
                        let p = <$curve>::generator() * (s + Scalar::ONE);
                        let ps = [p, p.double(), p + <$curve>::generator()];
                        law::agrees_with_mul::<$curve, Booth7>(&p, &a);
                        law::additive_in_scalar::<$curve, Booth7>(&p, &a, &b);
                        law::zero_and_negation::<$curve, Booth7>(&p, &a);
                        law::batch_matches_solo::<$curve, Booth7>(&ps, &a);
                        law::recoding_is_reusable::<$curve, Booth7>(&ps, &a);
                        law::sum_of_products_matches_muls::<$curve, Booth7>(
                            &ps,
                            &[a, b, Scalar::from(3)],
                        );
                        law::positioned_matches_muls::<$curve, Booth7>(
                            &ps,
                            &[a, b, Scalar::from(3)],
                        );
                    }
                }
            }
        };
    }

    booth_pbt!(pallas, crate::pallas::Point);
    booth_pbt!(vesta, crate::vesta::Point);
}
