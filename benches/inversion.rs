//! Benchmarks for the inversion, on both fields, in one group.
//!
//! `Field::invert` runs the constant-time inversion over the inversion blocks of the backend
//! compiled into this build, which `pasta_curves::BACKEND` names; `VartimeField::invert` is
//! measured beside it as the variable-time alternative the crate also offers. The benchmark ids
//! do not carry the backend's name, so that the two runs below compare as criterion baselines;
//! the backend is printed instead.
//!
//! To measure a backend's inversion blocks against the portable ones, build the same benchmark
//! twice, since which blocks the inversion runs is fixed at compile time. `--cfg
//! pasta_curves_noasm` compiles the assembly out, so the second run takes the portable blocks:
//!
//! ```sh
//! cargo bench --features asm --bench inversion -- --save-baseline asm
//! RUSTFLAGS="--cfg pasta_curves_noasm" \
//!     cargo bench --features asm --bench inversion -- --baseline asm
//! ```
//!
//! The second run reports each benchmark's change against the first, which for `invert` is the
//! difference between the two sets of blocks and nothing else: around the blocks, `invert` only
//! compares its input to zero, in portable code either way.

use criterion::{
    BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::Measurement,
};

use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

use ff::Field;
use pasta_curves::{BACKEND, Fp, Fq, arithmetic::VartimeField};

/// The number of inputs each benchmark cycles through, as the field benchmarks use.
const SAMPLES: usize = 1000;

/// The seed the crate's benchmarks share.
const SEED: [u8; 16] = [
    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc, 0xe5,
];

fn criterion_benchmark(c: &mut Criterion) {
    // The build the measurements below come from, since the ids leave the backend out.
    println!("inversion benchmark: pasta_curves::BACKEND = {BACKEND}");

    let mut group = c.benchmark_group("inversion");
    bench_field::<_, Fp>(&mut group, "Fp");
    bench_field::<_, Fq>(&mut group, "Fq");
    group.finish();
}

/// A field's two inversions: the constant-time one, which runs this build's inversion blocks,
/// and the variable-time one.
fn bench_field<M: Measurement, F: Field + VartimeField>(
    group: &mut BenchmarkGroup<'_, M>,
    field: &str,
) {
    let mut rng = XorShiftRng::from_seed(SEED);
    let v: Vec<F> = (0..SAMPLES).map(|_| F::random(&mut rng)).collect();

    group.bench_function(format!("{field}/invert"), |b| {
        let mut count = 0;
        b.iter(|| {
            count = (count + 1) % SAMPLES;
            v[count].invert()
        });
    });
    group.bench_function(format!("{field}/invert_vartime"), |b| {
        let mut count = 0;
        b.iter(|| {
            count = (count + 1) % SAMPLES;
            v[count].invert_vartime()
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
