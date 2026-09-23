//! Benchmarks for GLV scalar multiplication with Eisenstein recoding,
//! against the split-wNAF GLV path and the native operator.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use ff::Field;
use group::CurveAffine as _;
use pasta_curves::glv::GlvParams;
use pasta_curves::{glv, glv_eisenstein, pallas, vesta};

fn criterion_benchmark(c: &mut Criterion) {
    bench::<pallas::Point>(c, "Pallas");
    bench::<vesta::Point>(c, "Vesta");
}

fn bench<C: GlvParams>(c: &mut Criterion, name: &str) {
    let mut group = c.benchmark_group(name);

    // Deterministic full-width setup (matches the crate's other benches).
    let k = (C::ScalarExt::from(0x9E37_79B9_7F4A_7C15u64).square()
        + C::ScalarExt::from(0x0123_4567_89AB_CDEFu64))
    .square();
    let p = C::generator() * (k + C::ScalarExt::ONE);
    let points: Vec<C> = (1..=50)
        .map(|i| C::generator() * (k + C::ScalarExt::from(i)))
        .collect();

    let wnaf_table = glv::Table::new(&p);
    let wnaf_decomposed = glv::Decomposed::<C>::new(&k);
    let eis_table = glv_eisenstein::Table::new(&p);
    let eis_recoded = glv_eisenstein::Recoded::<C>::new(&k);

    group.bench_function("native mul", |b| b.iter(|| p * k));
    group.bench_function("one-shot wNAF", |b| b.iter(|| p.mul_glv(&k)));
    group.bench_function("one-shot Eisenstein", |b| {
        b.iter(|| glv_eisenstein::mul(&p, &k))
    });

    group.bench_function("table build wNAF (solo)", |b| {
        b.iter(|| glv::Table::new(&p))
    });
    group.bench_function("table build Eisenstein (solo)", |b| {
        b.iter(|| glv_eisenstein::Table::new(&p))
    });
    // Whole-batch time; divide by 50 for the per-point cost.
    group.bench_function("table build wNAF (batch of 50)", |b| {
        b.iter(|| glv::Table::batch(&points))
    });
    group.bench_function("table build Eisenstein (batch of 50)", |b| {
        b.iter(|| glv_eisenstein::Table::batch(&points))
    });

    // The scanning shape: one fixed scalar, precomputation hoisted, so this
    // isolates the ladder.
    group.bench_function("ladder wNAF (table + decomposed)", |b| {
        b.iter(|| wnaf_table.mul_decomposed(&wnaf_decomposed))
    });
    group.bench_function("ladder Eisenstein (table + recoded)", |b| {
        b.iter(|| eis_table.mul_recoded(&eis_recoded))
    });

    group.bench_function("recode scalar", |b| {
        b.iter(|| glv_eisenstein::Recoded::<C>::new(&k))
    });

    group.finish();

    // The scanning shape end to end: N ephemeral keys in, N AFFINE shared
    // secrets out, against one fixed viewing key. Affine is what the KDF
    // consumes, so the baseline pays the final normalization too (batched).
    let mut group = c.benchmark_group(format!("{name} batch key agreement"));
    for size in [16usize, 64, 128, 256, 512] {
        let batch: Vec<C> = (1..=size as u64)
            .map(|i| C::generator() * (k + C::ScalarExt::from(i)))
            .collect();

        // Both arms do the SAME end-to-end work: recode the scalar, build a
        // table per point, ladder, and hand back affine results. Every
        // ephemeral key is fresh in the scanning workload, so the table build
        // belongs inside the timed region.
        group.bench_with_input(BenchmarkId::new("wNAF + Jacobian", size), &size, |b, _| {
            b.iter(|| {
                let decomposed = glv::Decomposed::<C>::new(&k);
                let tables = glv::Table::batch(&batch);
                let proj: Vec<C> = tables
                    .iter()
                    .map(|t| t.mul_decomposed(&decomposed))
                    .collect();
                let mut aff = vec![C::AffineExt::identity(); proj.len()];
                C::batch_normalize(&proj, &mut aff);
                aff
            })
        });
        group.bench_with_input(
            BenchmarkId::new("Eisenstein + batch affine", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let recoded = glv_eisenstein::Recoded::<C>::new(&k);
                    let tables = glv_eisenstein::Table::batch(&batch);
                    glv_eisenstein::Table::batch_mul_affine(&tables, &recoded)
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("Eisenstein + projective", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let recoded = glv_eisenstein::Recoded::<C>::new(&k);
                    let tables = glv_eisenstein::Table::batch(&batch);
                    let proj: Vec<C> = tables.iter().map(|t| t.mul_recoded(&recoded)).collect();
                    let mut aff = vec![C::AffineExt::identity(); proj.len()];
                    C::batch_normalize(&proj, &mut aff);
                    aff
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
