//! Benchmarks for point operations.

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use pasta_curves::arithmetic::CurveExt;
use pasta_curves::{pallas, vesta};

fn criterion_benchmark(c: &mut Criterion) {
    point_bench::<pallas::Point>(c, "Pallas");
    point_bench::<vesta::Point>(c, "Vesta");
}

fn point_bench<C: CurveExt>(c: &mut Criterion, name: &str) {
    let mut group = c.benchmark_group(name);

    let a = C::generator();
    let b = a.double();

    group.bench_function("point doubling", |bencher| bencher.iter(|| a.double()));

    group.bench_function("point addition", |bencher| bencher.iter(|| a + b));

    group.bench_function("point subtraction", |bencher| bencher.iter(|| a - b));

    group.bench_function("point to_bytes", |bencher| bencher.iter(|| a.to_bytes()));

    let repr = a.to_bytes();
    group.bench_function("point from_bytes", |bencher| {
        bencher.iter(|| C::from_bytes(&repr))
    });

    group.bench_function("point to_affine", |bencher| bencher.iter(|| a.to_affine()));
    group.bench_function("point to_affine_vartime", |bencher| {
        bencher.iter(|| a.to_affine_vartime())
    });

    for n in [0, 1, 2, 16, 31, 32, 33, 64, 100, 1000, 10000] {
        // Use nontrivial, varying Jacobian scales instead of only z = 1.
        let mut input: Vec<C> = (0..n)
            .scan(a, |point, _| {
                *point = point.double();
                Some(*point)
            })
            .collect();
        let mut output = vec![C::Affine::default(); n];
        group.bench_function(format!("point batch_normalize/{n}"), |bencher| {
            bencher.iter(|| {
                C::batch_normalize(
                    black_box(input.as_slice()),
                    black_box(output.as_mut_slice()),
                );
                black_box(&output);
            });
        });
        group.bench_function(format!("point batch_normalize_vartime/{n}"), |bencher| {
            bencher.iter(|| {
                C::batch_normalize_vartime(
                    black_box(input.as_slice()),
                    black_box(output.as_mut_slice()),
                );
                black_box(&output);
            });
        });
        for point in input.iter_mut().step_by(3) {
            *point = C::identity();
        }
        group.bench_function(format!("point batch_normalize mixed/{n}"), |bencher| {
            bencher.iter(|| {
                C::batch_normalize(
                    black_box(input.as_slice()),
                    black_box(output.as_mut_slice()),
                );
                black_box(&output);
            });
        });
        group.bench_function(
            format!("point batch_normalize_vartime mixed/{n}"),
            |bencher| {
                bencher.iter(|| {
                    C::batch_normalize_vartime(
                        black_box(input.as_slice()),
                        black_box(output.as_mut_slice()),
                    );
                    black_box(&output);
                });
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
