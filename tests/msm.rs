#![cfg(feature = "glv")]

use ff::Field;
use pasta_curves::{arithmetic::CurveExt, pallas, vesta};
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

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
    let mut rng = XorShiftRng::from_seed([0x42; 16]);
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
    let mut rng = XorShiftRng::from_seed([0x5a; 16]);
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
