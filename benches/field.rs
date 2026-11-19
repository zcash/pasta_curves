//! Benchmarks for the fields Fp and Fq, one group per field.

use criterion::{
    BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::Measurement,
};

use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

use ff::{FromUniformBytes, PrimeField};
use pasta_curves::{Fp, Fq, arithmetic::VartimeField};

#[cfg(feature = "alloc")]
use {criterion::BatchSize, ff::BatchInvert, pasta_curves::arithmetic::VartimeBatchInvert};

/// The number of inputs each benchmark cycles through, so that no single input's
/// value or cache position decides the timing.
const SAMPLES: usize = 1000;

fn rng() -> XorShiftRng {
    XorShiftRng::from_seed([
        0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ])
}

/// Benchmarks `f` on `SAMPLES` inputs drawn by `input`, one per iteration.
fn cycle<M, T, R>(
    group: &mut BenchmarkGroup<'_, M>,
    id: &str,
    mut input: impl FnMut(&mut XorShiftRng) -> T,
    f: impl Fn(&T) -> R,
) where
    M: Measurement,
{
    let mut rng = rng();
    let v: Vec<T> = (0..SAMPLES).map(|_| input(&mut rng)).collect();
    group.bench_function(id, |b| {
        let mut count = 0;
        b.iter(|| {
            count = (count + 1) % SAMPLES;
            f(&v[count])
        });
    });
}

fn field_bench<F>(c: &mut Criterion, name: &str)
where
    F: PrimeField + VartimeField + FromUniformBytes<64>,
{
    let mut group = c.benchmark_group(name);
    let element = |rng: &mut XorShiftRng| F::random(rng);
    let pair = |rng: &mut XorShiftRng| (F::random(&mut *rng), F::random(rng));

    cycle(&mut group, "double", element, |x| x.double());
    cycle(&mut group, "add_assign", pair, |(x, y)| {
        let mut tmp = *x;
        tmp += y;
        tmp
    });
    cycle(&mut group, "sub_assign", pair, |(x, y)| {
        let mut tmp = *x;
        tmp -= y;
        tmp
    });
    cycle(&mut group, "mul_assign", pair, |(x, y)| {
        let mut tmp = *x;
        tmp *= y;
        tmp
    });
    cycle(&mut group, "square", element, |x| x.square());
    cycle(&mut group, "invert", element, |x| x.invert());
    cycle(&mut group, "invert_vartime", element, |x| {
        x.invert_vartime()
    });
    #[cfg(feature = "alloc")]
    batch_invert_bench::<F, _>(&mut group);
    cycle(&mut group, "neg", element, |x| -*x);
    cycle(
        &mut group,
        "sqrt",
        |rng| F::random(rng).square(),
        |x| x.sqrt(),
    );
    // The square root of a quotient, without inverting the denominator: the
    // core of hashing to the curves.
    cycle(&mut group, "sqrt_ratio", pair, |(num, div)| {
        F::sqrt_ratio(num, div)
    });
    cycle(&mut group, "to_repr", element, |x| x.to_repr());
    cycle(
        &mut group,
        "from_repr",
        |rng| F::random(rng).to_repr(),
        |repr| F::from_repr(*repr),
    );
    // The reduction of 64 uniform bytes, as in hashing to a scalar.
    cycle(
        &mut group,
        "from_uniform_bytes",
        |rng| {
            let mut bytes = [0u8; 64];
            rng.fill_bytes(&mut bytes);
            bytes
        },
        |bytes| F::from_uniform_bytes(bytes),
    );
}

#[cfg(feature = "alloc")]
fn batch_invert_bench<F, M>(group: &mut BenchmarkGroup<'_, M>)
where
    F: PrimeField + VartimeField,
    M: Measurement,
{
    let mut rng = rng();
    let v: Vec<F> = (0..10000).map(|_| F::random(&mut rng)).collect();

    for n in [100, 1000, 10000] {
        group.bench_with_input(format!("batch_invert/{n}"), &v[..n], |b, v| {
            b.iter_batched_ref(
                || v.to_vec(),
                |e| e.iter_mut().batch_invert(),
                BatchSize::LargeInput,
            );
        });
        group.bench_with_input(format!("batch_invert_vartime/{n}"), &v[..n], |b, v| {
            b.iter_batched_ref(
                || v.to_vec(),
                |e| e.iter_mut().batch_invert_vartime(),
                BatchSize::LargeInput,
            );
        });
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    field_bench::<Fp>(c, "Fp");
    field_bench::<Fq>(c, "Fq");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
