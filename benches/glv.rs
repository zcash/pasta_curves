//! Benchmarks for GLV scalar multiplication.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use ff::Field;
use pasta_curves::glv::{self, Decomposed, GlvParams, Table};
use pasta_curves::{pallas, vesta};

fn criterion_benchmark(c: &mut Criterion) {
    glv_bench::<pallas::Point>(c, "Pallas");
    glv_bench::<vesta::Point>(c, "Vesta");
}

fn glv_bench<C: GlvParams>(c: &mut Criterion, name: &str) {
    let mut group = c.benchmark_group(name);

    // Deterministic full-width setup (matches the crate's other benches).
    let k = (C::ScalarExt::from(0x9E37_79B9_7F4A_7C15u64).square()
        + C::ScalarExt::from(0x0123_4567_89AB_CDEFu64))
    .square();
    let p = C::generator() * (k + C::ScalarExt::ONE);
    let points: Vec<C> = (1..=50)
        .map(|i| C::generator() * (k + C::ScalarExt::from(i)))
        .collect();
    let table = Table::new(&p);
    let decomposed = Decomposed::<C>::new(&k);

    group.bench_function("native mul", |b| b.iter(|| p * k));
    group.bench_function("mul_glv one-shot", |b| b.iter(|| p.mul_glv(&k)));
    group.bench_function("table build (solo)", |b| b.iter(|| Table::new(&p)));
    // Whole-batch time; divide by 50 to compare per-point cost with the
    // solo build.
    group.bench_function("table build (batch of 50)", |b| {
        b.iter(|| Table::batch(&points));
    });
    group.bench_function("table mul (reused table)", |b| b.iter(|| table.mul(&k)));
    group.bench_function("table mul (reused table + decomposed)", |b| {
        b.iter(|| table.mul_decomposed(&decomposed));
    });

    group.finish();

    entry_points::<C>(c, name);
}

/// The four public entry points, timed the way a user calls them: nothing
/// hoisted out, because hiding the precomputation is what they are for. One
/// call shape each, so the numbers say which shape to reach for.
fn entry_points<C: GlvParams>(c: &mut Criterion, name: &str) {
    let mut group = c.benchmark_group(format!("{name} entry points"));

    let k = (C::ScalarExt::from(0x9E37_79B9_7F4A_7C15u64).square()
        + C::ScalarExt::from(0x0123_4567_89AB_CDEFu64))
    .square();
    let p = C::generator() * (k + C::ScalarExt::ONE);

    group.bench_function("mul (one point, one scalar)", |b| {
        b.iter(|| glv::mul(&p, &k))
    });

    for size in [16usize, 64, 256] {
        let points: Vec<C> = (1..=size as u64)
            .map(|i| C::generator() * (k + C::ScalarExt::from(i)))
            .collect();
        let scalars: Vec<C::ScalarExt> = (1..=size as u64)
            .map(|i| k + C::ScalarExt::from(i))
            .collect();
        let pairs: Vec<(C, C::ScalarExt)> = points
            .iter()
            .copied()
            .zip(scalars.iter().copied())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch_mul (N points, one scalar)", size),
            &size,
            |b, _| b.iter(|| glv::batch_mul(&points, &k)),
        );
        group.bench_with_input(
            BenchmarkId::new("mul_scalars (one point, N scalars)", size),
            &size,
            |b, _| b.iter(|| glv::mul_scalars(&p, &scalars)),
        );
        group.bench_with_input(
            BenchmarkId::new("mul_pairs (N points, N scalars)", size),
            &size,
            |b, _| b.iter(|| glv::mul_pairs(&pairs)),
        );
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
