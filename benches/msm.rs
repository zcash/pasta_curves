//! Benchmarks for `CurveExt::try_multiexp_vartime` against the generic MSM a
//! caller falls back to, rather than against per-term scalar multiplication.
//!
//! Three implementations are measured on the same inputs:
//!
//! - `naive`: the sum of `base * scalar`, one full scalar multiplication per term.
//! - `generic`: the serial path of `halo2_proofs` 0.4.1 `best_multiexp`, the
//!   signed-Booth bucket MSM that callers keep as their fallback.
//! - `hook-or-generic`: what a caller runs, `try_multiexp_vartime` with the
//!   generic MSM when the hook declines.
//!
//! Build without `multicore`: the generic port below is serial, so comparing it
//! with a parallel hook would measure thread count, not the algorithm.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use ff::{Field, PrimeField};
use group::{CurveAffine as _, Group};
use pasta_curves::arithmetic::{CurveAffine, CurveExt};
use pasta_curves::{pallas, vesta};
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

const SIZES: [usize; 7] = [16, 64, 256, 1024, 4096, 16384, 65536];

/// Above this size the naive baseline only adds run time.
const MAX_NAIVE_SIZE: usize = 4096;

// Ported from halo2_proofs 0.4.1, src/arithmetic.rs (MIT OR Apache-2.0,
// Copyright the Electric Coin Company and contributors): `Bucket`,
// `booth_digit`, `booth_window_count`, `BoothBuckets` and the serial branch of
// `best_multiexp`. Kept as close to the source as possible.
mod generic {
    use super::*;

    #[derive(Clone, Copy)]
    enum Bucket<C>
    where
        C: CurveAffine,
    {
        None,
        Affine(C),
        Projective(C::CurveExt),
    }

    impl<C> Bucket<C>
    where
        C: CurveAffine,
    {
        fn add_assign(&mut self, other: C) {
            *self = match *self {
                Bucket::None => Bucket::Affine(other),
                Bucket::Affine(a) => Bucket::Projective(a + other),
                Bucket::Projective(mut a) => {
                    a += other;
                    Bucket::Projective(a)
                }
            }
        }

        fn add(self, mut other: C::CurveExt) -> C::CurveExt {
            match self {
                Bucket::None => other,
                Bucket::Affine(a) => {
                    other += a;
                    other
                }
                Bucket::Projective(a) => other + a,
            }
        }
    }

    struct BoothDigit {
        magnitude: usize,
        negative: bool,
    }

    fn booth_digit(bytes: &[u8], window_bits: usize, window: usize) -> BoothDigit {
        let window_start = window * window_bits;
        let byte_start = window_start / 8;
        let mut encoded = [0; 8];
        if byte_start < bytes.len() {
            for (slot, byte) in encoded.iter_mut().zip(&bytes[byte_start..]) {
                *slot = *byte;
            }
        }

        let bit_offset = window_start - byte_start * 8;
        let radix = 1usize << window_bits;
        let value = ((u64::from_le_bytes(encoded) >> bit_offset) as usize) & (radix - 1);
        let overlap = if window_start == 0 {
            0
        } else {
            let bit = window_start - 1;
            bytes
                .get(bit / 8)
                .map_or(0, |byte| usize::from((byte >> (bit % 8)) & 1))
        };

        if value < radix / 2 {
            BoothDigit {
                magnitude: value + overlap,
                negative: false,
            }
        } else {
            let magnitude = radix - value - overlap;
            BoothDigit {
                magnitude,
                negative: magnitude != 0,
            }
        }
    }

    fn booth_window_count<R>(reprs: &[R], window_bits: usize) -> usize
    where
        R: AsRef<[u8]>,
    {
        let repr_bytes = reprs
            .iter()
            .map(|repr| repr.as_ref().len())
            .max()
            .unwrap_or(0);
        repr_bytes * 8 / window_bits + 1
    }

    #[derive(Clone)]
    struct BoothBuckets<C>
    where
        C: CurveAffine,
    {
        c: usize,
        coeffs: Vec<Bucket<C>>,
    }

    impl<C> BoothBuckets<C>
    where
        C: CurveAffine,
    {
        fn new(c: usize) -> Self {
            Self {
                c,
                coeffs: vec![Bucket::None; 1 << (c - 1)],
            }
        }

        fn sum(
            &mut self,
            coeffs: &[<C::ScalarExt as PrimeField>::Repr],
            bases: &[C],
            i: usize,
        ) -> C::CurveExt {
            for (coeff, base) in coeffs.iter().zip(bases.iter()) {
                let digit = booth_digit(coeff.as_ref(), self.c, i);
                if digit.magnitude != 0 {
                    let base = if digit.negative { -*base } else { *base };
                    self.coeffs[digit.magnitude - 1].add_assign(base);
                }
            }
            let mut acc = C::CurveExt::identity();
            let mut sum = C::CurveExt::identity();
            self.coeffs.iter().rev().for_each(|b| {
                sum = b.add(sum);
                acc += sum;
            });
            acc
        }
    }

    /// The serial branch of halo2_proofs 0.4.1 `best_multiexp`.
    pub(super) fn best_multiexp<C>(coeffs: &[C::ScalarExt], bases: &[C]) -> C::CurveExt
    where
        C: CurveAffine,
    {
        assert_eq!(coeffs.len(), bases.len());
        let coeffs = coeffs.iter().map(PrimeField::to_repr).collect::<Vec<_>>();
        let c = if bases.len() < 4 {
            1
        } else if bases.len() < 32 {
            3
        } else {
            (f64::from(bases.len() as u32)).ln().ceil() as usize
        };
        let mut multi_buckets: Vec<BoothBuckets<C>> =
            vec![BoothBuckets::new(c); booth_window_count(&coeffs, c)];
        multi_buckets
            .iter_mut()
            .enumerate()
            .rev()
            .map(|(i, buckets)| buckets.sum(&coeffs, bases, i))
            .fold(C::CurveExt::identity(), |mut sum, bucket| {
                (0..c).for_each(|_| sum = sum.double());
                sum + bucket
            })
    }
}

fn naive<C>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> C
where
    C: CurveExt,
{
    scalars
        .iter()
        .zip(bases)
        .fold(C::identity(), |acc, (scalar, base)| acc + *base * scalar)
}

fn hook_or_generic<C>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> C
where
    C: CurveExt,
{
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

fn msm_bench<C>(c: &mut Criterion, name: &str)
where
    C: CurveExt,
{
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
