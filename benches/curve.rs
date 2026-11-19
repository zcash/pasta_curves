//! Benchmarks for the curves Pallas and Vesta: point arithmetic, encoding and
//! normalization, one group per curve.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ff::Field;

use group::GroupEncoding;
use pasta_curves::arithmetic::{CurveAffine, CurveExt};
use pasta_curves::{pallas, vesta};
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

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

    // Adding an affine point, whose z = 1 saves products: the case of the
    // multi-scalar multiplications and of fixed-base tables.
    let b_affine = b.to_affine();
    group.bench_function("mixed addition", |bencher| {
        bencher.iter(|| black_box(a) + black_box(b_affine))
    });

    // Use a deterministic full-width scalar and a nontrivial projective
    // representative; prepare both input forms outside the timed region.
    let mut rng = XorShiftRng::from_seed([0x53; 16]);
    let scalar = C::Scalar::random(&mut rng);
    let point = a * C::Scalar::random(&mut rng);
    let affine = point.to_affine();
    group.bench_function("point scalar multiplication", |bencher| {
        bencher.iter(|| black_box(point) * black_box(scalar));
    });
    group.bench_function("affine scalar multiplication", |bencher| {
        bencher.iter(|| black_box(affine) * black_box(scalar));
    });

    group.bench_function("point to_bytes", |bencher| bencher.iter(|| a.to_bytes()));

    let repr = a.to_bytes();
    group.bench_function("point from_bytes", |bencher| {
        bencher.iter(|| C::from_bytes(&repr))
    });

    // Decoding a batch of public encodings, against decoding them one by one.
    for n in [1, 8, 64, 512] {
        let encodings: Vec<_> = (0..n)
            .map(|_| C::random(&mut rng).to_affine().to_bytes())
            .collect();
        group.bench_function(format!("affine batch_decode_vartime/{n}"), |bencher| {
            bencher.iter(|| C::AffineExt::batch_decode_vartime(black_box(&encodings)));
        });
        group.bench_function(format!("affine from_bytes each/{n}"), |bencher| {
            bencher.iter(|| {
                black_box(&encodings)
                    .iter()
                    .map(|bytes| Option::<C::AffineExt>::from(C::AffineExt::from_bytes(bytes)))
                    .collect::<Vec<_>>()
            });
        });
    }

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
