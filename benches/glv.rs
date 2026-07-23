//! Benchmarks for GLV scalar multiplication.

use criterion::{criterion_group, criterion_main, Criterion};

use ff::Field;
use pasta_curves::glv::{Decomposed, GlvParams, Table};
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
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
