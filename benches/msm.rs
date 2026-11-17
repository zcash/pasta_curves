//! Benchmarks for `CurveExt::try_multiexp_vartime` against the generic MSM a
//! caller falls back to, rather than against per-term scalar multiplication.
//!
//! Three implementations are measured on the same inputs:
//!
//! - `naive`: the sum of `base * scalar`, one full scalar multiplication per term.
//! - `generic`: `halo2_proofs` 0.4.1 `best_multiexp`, the signed-Booth bucket
//!   MSM that callers keep as their fallback.
//! - `hook-or-generic`: what a caller runs, `try_multiexp_vartime` with the
//!   generic MSM when the hook declines.
//!
//! With `multicore`, both the hook and the generic MSM use the same Rayon pool.
//! Set `RAYON_NUM_THREADS` to compare them at a fixed worker count; without
//! `multicore`, both implementations are serial.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use ff::{Field, PrimeField};
use group::{CurveAffine as _, Group};
#[cfg(feature = "multicore")]
use maybe_rayon::prelude::*;
use pasta_curves::arithmetic::{CurveAffine, CurveExt};
use pasta_curves::{pallas, vesta};
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

const SIZES: [usize; 19] = [
    16, 17, 24, 25, 32, 64, 66, 67, 96, 127, 128, 192, 255, 256, 257, 1024, 4096, 16384, 65536,
];

/// Above this size the naive baseline only adds run time.
const MAX_NAIVE_SIZE: usize = 4096;

#[path = "support/generic_msm.rs"]
mod generic;

fn naive<C: CurveExt>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> C {
    scalars
        .iter()
        .zip(bases)
        .fold(C::identity(), |acc, (scalar, base)| acc + *base * scalar)
}

fn hook_or_generic<C: CurveExt>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> C {
    C::try_multiexp_vartime(scalars, bases)
        .unwrap_or_else(|| generic::best_multiexp::<C::AffineExt>(scalars, bases))
}

/// A fresh seed per run, so repeated runs measure different inputs. Set
/// `MSM_BENCH_SEED` to the 32 hex digits a run prints to replay its inputs.
///
/// `rand` is built without OS entropy, so the seed comes from `RandomState`,
/// which std seeds from the operating system once per process.
fn seed() -> [u8; 16] {
    use std::hash::{BuildHasher, Hasher};

    if let Ok(hex) = std::env::var("MSM_BENCH_SEED") {
        let value = u128::from_str_radix(&hex, 16).expect("MSM_BENCH_SEED: 32 hex digits");
        return value.to_be_bytes();
    }
    let state = std::collections::hash_map::RandomState::new();
    let half = |tag: u64| {
        let mut hasher = state.build_hasher();
        hasher.write_u64(tag);
        hasher.finish()
    };
    ((u128::from(half(0)) << 64) | u128::from(half(1))).to_be_bytes()
}

fn msm_bench<C: CurveExt>(c: &mut Criterion, name: &str) {
    let max = *SIZES.iter().max().unwrap();
    let seed = seed();
    eprintln!(
        "msm/{name}: MSM_BENCH_SEED={:032x}",
        u128::from_be_bytes(seed)
    );
    let mut rng = XorShiftRng::from_seed(seed);
    let scalars: Vec<C::ScalarExt> = (0..max).map(|_| C::ScalarExt::random(&mut rng)).collect();
    let projective: Vec<C> = (0..max).map(|_| C::random(&mut rng)).collect();
    let mut bases = vec![C::AffineExt::identity(); max];
    C::batch_normalize(&projective, &mut bases);

    let mut group = c.benchmark_group(format!("msm/{name}"));
    group.sample_size(10);
    for size in SIZES {
        let (scalars, bases) = (&scalars[..size], &bases[..size]);

        // Check the three agree once per size, and say whether the hook ran.
        let generic = generic::best_multiexp::<C::AffineExt>(scalars, bases);
        let hook = C::try_multiexp_vartime(scalars, bases);
        if let Some(hook) = hook {
            assert_eq!(hook, generic, "{name}/{size}: hook differs from generic");
        }
        if size <= MAX_NAIVE_SIZE {
            assert_eq!(naive::<C>(scalars, bases), generic, "{name}/{size}: naive");
        }
        eprintln!(
            "msm/{name}/{size}: hook {}",
            if hook.is_some() {
                "accepted"
            } else {
                "declined"
            }
        );

        group.throughput(Throughput::Elements(size as u64));
        if size <= MAX_NAIVE_SIZE {
            group.bench_function(BenchmarkId::new("naive", size), |b| {
                b.iter(|| naive::<C>(black_box(scalars), black_box(bases)))
            });
        }
        group.bench_function(BenchmarkId::new("generic", size), |b| {
            b.iter(|| generic::best_multiexp::<C::AffineExt>(black_box(scalars), black_box(bases)))
        });
        group.bench_function(BenchmarkId::new("hook-or-generic", size), |b| {
            b.iter(|| hook_or_generic::<C>(black_box(scalars), black_box(bases)))
        });
    }
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    msm_bench::<pallas::Point>(c, "Pallas");
    msm_bench::<vesta::Point>(c, "Vesta");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
