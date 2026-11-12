//! Benchmarks for GLV scalar multiplication with Eisenstein recoding.
//!
//! To compare with the split-wNAF recoding, run
//! `cargo bench --features glv --bench glv` and
//! `cargo bench --features glv-eisenstein --bench glv_eisenstein`: each
//! benchmark here has the same id as its counterpart there, in a group
//! named with `Eisenstein` after the curve.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use ff::Field;
use group::CurveAffine as _;
use pasta_curves::glv::GlvParams;
use pasta_curves::{glv_eisenstein, pallas, vesta};

fn criterion_benchmark(c: &mut Criterion) {
    bench::<pallas::Point>(c, "Pallas");
    bench::<vesta::Point>(c, "Vesta");
}

fn bench<C: GlvParams>(c: &mut Criterion, name: &str) {
    let mut group = c.benchmark_group(format!("{name} Eisenstein"));

    // Deterministic full-width setup (matches the crate's other benches).
    let k = (C::ScalarExt::from(0x9E37_79B9_7F4A_7C15u64).square()
        + C::ScalarExt::from(0x0123_4567_89AB_CDEFu64))
    .square();
    let p = C::generator() * (k + C::ScalarExt::ONE);
    let points: Vec<C> = (1..=50)
        .map(|i| C::generator() * (k + C::ScalarExt::from(i)))
        .collect();
    let table = glv_eisenstein::Table::new(&p);
    let recoded = glv_eisenstein::Recoded::<C>::new(&k);

    group.bench_function("mul one-shot", |b| b.iter(|| glv_eisenstein::mul(&p, &k)));
    group.bench_function("table build (solo)", |b| {
        b.iter(|| glv_eisenstein::Table::new(&p))
    });
    // Whole-batch time; divide by 50 to compare per-point cost with the
    // solo build.
    group.bench_function("table build (batch of 50)", |b| {
        b.iter(|| glv_eisenstein::Table::batch(&points))
    });
    group.bench_function("table mul (reused table)", |b| b.iter(|| table.mul(&k)));
    // Precomputation hoisted, so this isolates the ladder.
    group.bench_function("table mul (reused table + recoded)", |b| {
        b.iter(|| table.mul_recoded(&recoded))
    });
    group.bench_function("recode scalar", |b| {
        b.iter(|| glv_eisenstein::Recoded::<C>::new(&k))
    });

    group.finish();

    // One shared scalar against N points end to end, with N affine results
    // out: recode the scalar, build a table per point, ladder, and normalize.
    // Every point is fresh, so the table build belongs inside the timed
    // region.
    let mut group = c.benchmark_group(format!("{name} Eisenstein batch shared scalar"));
    for size in [16usize, 64, 96, 128, 256, 512] {
        let batch: Vec<C> = (1..=size as u64)
            .map(|i| C::generator() * (k + C::ScalarExt::from(i)))
            .collect();

        // `batch_mul` takes the projective ladder below the crate's affine
        // threshold and the batch-affine one from it, so beside the
        // projective arm below it shows where the affine ladder starts to pay.
        group.bench_with_input(BenchmarkId::new("batch_mul", size), &size, |b, _| {
            b.iter(|| glv_eisenstein::batch_mul(&batch, &k))
        });
        group.bench_with_input(
            BenchmarkId::new("projective ladder", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let recoded = glv_eisenstein::Recoded::<C>::new(&k);
                    let tables = glv_eisenstein::Table::batch(&batch);
                    let proj: Vec<C> = tables.iter().map(|t| t.mul_recoded(&recoded)).collect();
                    let mut aff = vec![C::AffineExt::identity(); proj.len()];
                    C::batch_normalize_vartime(&proj, &mut aff);
                    aff
                })
            },
        );
    }
    group.finish();

    entry_points::<C>(c, name);
}

/// The four public entry points, timed the way a user calls them: nothing
/// hoisted out, because hiding the precomputation is what they are for. One
/// call shape each, so the numbers say which shape to reach for.
fn entry_points<C: GlvParams>(c: &mut Criterion, name: &str) {
    let mut group = c.benchmark_group(format!("{name} Eisenstein entry points"));

    let k = (C::ScalarExt::from(0x9E37_79B9_7F4A_7C15u64).square()
        + C::ScalarExt::from(0x0123_4567_89AB_CDEFu64))
    .square();
    let p = C::generator() * (k + C::ScalarExt::ONE);

    group.bench_function("mul (one point, one scalar)", |b| {
        b.iter(|| glv_eisenstein::mul(&p, &k))
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
            |b, _| b.iter(|| glv_eisenstein::batch_mul(&points, &k)),
        );
        group.bench_with_input(
            BenchmarkId::new("mul_scalars (one point, N scalars)", size),
            &size,
            |b, _| b.iter(|| glv_eisenstein::mul_scalars(&p, &scalars)),
        );
        group.bench_with_input(
            BenchmarkId::new("mul_pairs (N points, N scalars)", size),
            &size,
            |b, _| b.iter(|| glv_eisenstein::mul_pairs(&pairs)),
        );
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
