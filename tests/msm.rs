#![cfg(feature = "glv")]

use ff::Field;
use pasta_curves::{arithmetic::CurveExt, pallas, vesta};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::test_runner::{Config, TestCaseError, TestRunner};
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

/// A fresh seed per run, so every run checks different inputs. The seed is
/// printed (and shown by the test harness on failure); set `MSM_TEST_SEED` to
/// those 32 hex digits to replay a failing run.
///
/// `rand` is built without OS entropy, so the seed comes from `RandomState`,
/// which std seeds from the operating system once per process.
fn seed(case: &str) -> [u8; 16] {
    use std::hash::{BuildHasher, Hasher};

    let value = match std::env::var("MSM_TEST_SEED") {
        Ok(hex) => u128::from_str_radix(&hex, 16).expect("MSM_TEST_SEED: 32 hex digits"),
        Err(_) => {
            let state = std::collections::hash_map::RandomState::new();
            let half = |tag: u64| {
                let mut hasher = state.build_hasher();
                hasher.write_u64(tag);
                u128::from(hasher.finish())
            };
            (half(0) << 64) | half(1)
        }
    };
    eprintln!("{case}: MSM_TEST_SEED={value:032x}");
    value.to_be_bytes()
}

const MAX_SIZE: usize = 5000;
const BOUNDARY_SIZES: [usize; 10] = [255, 256, 257, 511, 512, 1023, 1024, 2048, 4096, MAX_SIZE];

fn sizes() -> impl Iterator<Item = usize> {
    (0..=128).chain(BOUNDARY_SIZES)
}

fn prefix_sums<C: CurveExt>(products: impl Iterator<Item = C>) -> Vec<C> {
    let mut sums = Vec::with_capacity(products.size_hint().0 + 1);
    let mut sum = C::identity();
    sums.push(sum);
    for product in products {
        sum += product;
        sums.push(sum);
    }
    sums
}

fn check_prefixes<C: CurveExt>(
    case: &str,
    scalars: &[C::ScalarExt],
    bases: &[C::AffineExt],
    expected: &[C],
    sizes: impl IntoIterator<Item = usize>,
) -> Vec<usize> {
    let mut accepted = Vec::new();
    let mut declined = Vec::new();
    for size in sizes {
        match C::try_multiexp_vartime(&scalars[..size], &bases[..size]) {
            Some(actual) => {
                assert_eq!(
                    actual,
                    expected[size],
                    "{}/{case}, size {size}",
                    C::CURVE_ID
                );
                accepted.push(size);
            }
            None => {
                assert_ne!(
                    size,
                    0,
                    "{}/{case}: empty MSM must be accepted",
                    C::CURVE_ID
                );
                declined.push(size);
            }
        }
    }
    // A declined shape is not an equivalence check: do not replace it with a fallback.
    eprintln!(
        "{}/{case}: accepted {accepted:?}; declined {declined:?} (not backend coverage)",
        C::CURVE_ID
    );
    accepted
}

fn require_small_and_large<C: CurveExt>(case: &str, accepted: &[usize]) {
    assert!(
        accepted.iter().any(|size| (1..=16).contains(size)),
        "{}/{case}: no nonempty small MSM was accepted",
        C::CURVE_ID
    );
    assert!(
        accepted.iter().any(|size| *size >= 256),
        "{}/{case}: no large MSM was accepted",
        C::CURVE_ID
    );
}

fn random_prefixes<C: CurveExt>() {
    let mut rng = XorShiftRng::from_seed(seed(&format!("{}/random_prefixes", C::CURVE_ID)));
    let scalars: Vec<_> = (0..MAX_SIZE)
        .map(|_| C::ScalarExt::random(&mut rng))
        .collect();
    let bases: Vec<_> = (0..MAX_SIZE)
        .map(|_| C::AffineExt::from(C::random(&mut rng)))
        .collect();
    // Native multiplication is the oracle, not GLV. Compute each product only once,
    // then reuse its prefix sum for every size and every custom thread pool.
    let expected = prefix_sums::<C>(
        scalars
            .iter()
            .zip(&bases)
            .map(|(scalar, base)| *base * scalar),
    );

    assert_eq!(C::try_multiexp_vartime(&[], &[]), Some(C::identity()));
    for (scalar_len, base_len) in [
        (0, 1),
        (1, 0),
        (15, 16),
        (16, 15),
        (16, 17),
        (17, 16),
        (255, 256),
        (256, 255),
        (256, 257),
        (257, 256),
        (MAX_SIZE - 1, MAX_SIZE),
        (MAX_SIZE, MAX_SIZE - 1),
    ] {
        assert!(
            C::try_multiexp_vartime(&scalars[..scalar_len], &bases[..base_len]).is_none(),
            "{}: mismatched lengths {scalar_len}/{base_len} must be declined",
            C::CURVE_ID
        );
    }

    let accepted = check_prefixes::<C>("random", &scalars, &bases, &expected, sizes());
    require_small_and_large::<C>("random", &accepted);

    #[cfg(feature = "multicore")]
    {
        let mut accepted_large = false;
        for workers in [1, 2, 3, 6, 8, 32] {
            let pool = maybe_rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .expect("the test environment must support a custom worker pool");
            let case = format!("random/{workers} workers");
            let accepted = pool.install(|| {
                check_prefixes::<C>(
                    &case,
                    &scalars,
                    &bases,
                    &expected,
                    [0, 1, 16, 64, 256, 2048, MAX_SIZE],
                )
            });
            assert!(
                accepted.iter().any(|size| *size != 0),
                "{}/{case}: no nonempty MSM was accepted",
                C::CURVE_ID
            );
            // Worker counts may change which large shapes are profitable. Require
            // large coverage across the pools, not acceptance of every shape.
            accepted_large |= accepted.iter().any(|size| *size >= 256);
        }
        assert!(
            accepted_large,
            "{}: no custom pool accepted a large MSM",
            C::CURVE_ID
        );
    }
}

fn check_pattern<C: CurveExt>(case: &str, pattern: &[(C::ScalarExt, C::AffineExt)], block_sum: C) {
    let products: Vec<_> = pattern
        .iter()
        .map(|(scalar, base)| *base * scalar)
        .collect();
    assert_eq!(
        products
            .iter()
            .fold(C::identity(), |sum, product| sum + product),
        block_sum,
        "{}/{case}: fixture block sum",
        C::CURVE_ID
    );
    let scalars: Vec<_> = pattern
        .iter()
        .cycle()
        .take(MAX_SIZE)
        .map(|(s, _)| *s)
        .collect();
    let bases: Vec<_> = pattern
        .iter()
        .cycle()
        .take(MAX_SIZE)
        .map(|(_, b)| *b)
        .collect();
    // These fixtures repeat, so the native products can be reused too.
    let expected = prefix_sums::<C>(products.iter().copied().cycle().take(MAX_SIZE));
    let accepted = check_prefixes::<C>(case, &scalars, &bases, &expected, sizes());
    require_small_and_large::<C>(case, &accepted);
}

fn special_cases<C: CurveExt>() {
    let mut rng = XorShiftRng::from_seed(seed(&format!("{}/special_cases", C::CURVE_ID)));
    let p = C::random(&mut rng);
    let q = C::random(&mut rng);
    let r = C::random(&mut rng);
    let k = C::ScalarExt::random(&mut rng);
    let t = C::ScalarExt::random(&mut rng);
    let zero = C::ScalarExt::ZERO;
    let one = C::ScalarExt::ONE;
    // -1 is the maximum canonical field scalar (the scalar modulus minus one).
    let max = -one;
    let identity = C::identity();

    check_pattern::<C>("zero scalars", &[(zero, p.into())], identity);
    check_pattern::<C>("one scalars/equal bases", &[(one, p.into())], p);
    check_pattern::<C>("max scalars/equal bases", &[(max, p.into())], -p);
    check_pattern::<C>(
        "identity bases",
        &[
            (zero, identity.into()),
            (one, identity.into()),
            (max, identity.into()),
            (k, identity.into()),
        ],
        identity,
    );
    // Each complete block cancels. Odd prefixes also check nonidentity results;
    // repeated blocks exercise duplicates and cancellation in the large path.
    check_pattern::<C>(
        "mixed scalars/duplicate and opposite bases/cancellation",
        &[
            (zero, p.into()),
            (one, p.into()),
            (max, p.into()),
            (k, identity.into()),
            (k, q.into()),
            (-k, q.into()),
            (t, r.into()),
            (t, (-r).into()),
        ],
        identity,
    );
}

fn single_worker(test: impl FnOnce() + Send) {
    // A machine-wide pool can legitimately decline every large test size.
    // Pin the baseline corpus; random_prefixes separately checks other counts.
    #[cfg(feature = "multicore")]
    maybe_rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("the test environment must support a single-worker pool")
        .install(test);

    #[cfg(not(feature = "multicore"))]
    test();
}

#[test]
fn pallas_random_prefixes() {
    single_worker(random_prefixes::<pallas::Point>);
}

#[test]
fn vesta_random_prefixes() {
    single_worker(random_prefixes::<vesta::Point>);
}

#[test]
fn pallas_special_cases() {
    single_worker(special_cases::<pallas::Point>);
}

#[test]
fn vesta_special_cases() {
    single_worker(special_cases::<vesta::Point>);
}

/// How the property test draws one term's scalar. The fixed values and the
/// shared pool reach the exceptional paths: zero digits, full-width scalars,
/// and equal scalars on equal or opposite bases.
#[derive(Clone, Copy, Debug)]
enum TermScalar {
    Zero,
    One,
    MinusOne,
    Shared(usize),
    Random,
}

/// How the property test draws one term's base. Shared bases, possibly
/// negated, make bucket additions double or cancel.
#[derive(Clone, Copy, Debug)]
enum TermBase {
    Identity,
    Shared(usize, bool),
    Random,
}

/// Size of the pools of shared scalars and bases.
const SHARED: usize = 3;

/// Largest property-test MSM: past every serial crossover (24, 128 and 256
/// terms) while keeping the native-multiplication oracle fast.
const MAX_PROPERTY_SIZE: usize = 300;

fn term_strategy() -> impl Strategy<Value = (TermScalar, TermBase)> {
    let scalar = prop_oneof![
        1 => Just(TermScalar::Zero),
        1 => Just(TermScalar::One),
        1 => Just(TermScalar::MinusOne),
        2 => (0..SHARED).prop_map(TermScalar::Shared),
        5 => Just(TermScalar::Random),
    ];
    let base = prop_oneof![
        1 => Just(TermBase::Identity),
        3 => (0..SHARED, any::<bool>()).prop_map(|(i, negate)| TermBase::Shared(i, negate)),
        6 => Just(TermBase::Random),
    ];
    (scalar, base)
}

/// The law: a serial `try_multiexp_vartime` accepts every shape and returns
/// the sum of the products, computed here by native scalar multiplication.
fn check_terms<C: CurveExt>(
    seed: [u8; 16],
    terms: &[(TermScalar, TermBase)],
) -> Result<(), TestCaseError> {
    let mut rng = XorShiftRng::from_seed(seed);
    let shared_scalars: Vec<C::ScalarExt> = (0..SHARED)
        .map(|_| C::ScalarExt::random(&mut rng))
        .collect();
    let shared_bases: Vec<C> = (0..SHARED).map(|_| C::random(&mut rng)).collect();

    let mut scalars = Vec::with_capacity(terms.len());
    let mut bases = Vec::with_capacity(terms.len());
    for (scalar, base) in terms {
        scalars.push(match scalar {
            TermScalar::Zero => C::ScalarExt::ZERO,
            TermScalar::One => C::ScalarExt::ONE,
            TermScalar::MinusOne => -C::ScalarExt::ONE,
            TermScalar::Shared(i) => shared_scalars[*i],
            TermScalar::Random => C::ScalarExt::random(&mut rng),
        });
        let base = match base {
            TermBase::Identity => C::identity(),
            TermBase::Shared(i, false) => shared_bases[*i],
            TermBase::Shared(i, true) => -shared_bases[*i],
            TermBase::Random => C::random(&mut rng),
        };
        bases.push(C::AffineExt::from(base));
    }

    let expected = scalars
        .iter()
        .zip(&bases)
        .fold(C::identity(), |sum, (scalar, base)| sum + *base * scalar);
    let actual = C::try_multiexp_vartime(&scalars, &bases);
    prop_assert_eq!(
        actual,
        Some(expected),
        "{} terms on {}",
        terms.len(),
        C::CURVE_ID
    );
    Ok(())
}

/// Runs the property on one worker, so every size takes the serial planner.
/// Proptest draws a fresh RNG seed per run and reports a shrunk failing case.
fn matches_native<C: CurveExt>() {
    single_worker(|| {
        let mut runner = TestRunner::new(Config::with_cases(64));
        let strategy = (
            any::<[u8; 16]>(),
            vec(term_strategy(), 0..=MAX_PROPERTY_SIZE),
        );
        if let Err(error) = runner.run(&strategy, |(seed, terms)| check_terms::<C>(seed, &terms)) {
            panic!("{}: {error}", C::CURVE_ID);
        }
    });
}

#[test]
fn pallas_matches_native() {
    matches_native::<pallas::Point>();
}

#[test]
fn vesta_matches_native() {
    matches_native::<vesta::Point>();
}
