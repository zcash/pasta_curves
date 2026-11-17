//! The timing grid the calibration is fitted to: an ignored test that times
//! every alternative the planner chooses between, not only the one it picks.
//!
//! For each MSM size it times, on the same inputs:
//!
//! - `generic`: the generic MSM a caller falls back to, the port of
//!   `best_multiexp` from `halo2_proofs` shared with `benches/msm.rs`;
//! - `buckets` at each candidate window width, `param` being the width;
//! - `straus`, one GLV-Straus table, at the sizes where Straus competes;
//! - `leaves` at several job counts, `param` being the number of jobs, at the
//!   same sizes, when there are several threads.
//!
//! The thread count is Rayon's (`RAYON_NUM_THREADS`), so one run measures one
//! thread count; `scripts/calibrate-msm.sh` runs it at several. Repetitions
//! interleave the alternatives, so that drift during a run affects them alike,
//! and each row is the median over the repetitions.
//!
//! It exists only with the `multicore` and `asm` features (`multicore` implies
//! `glv`), so that it times the code as deployed, and records the arithmetic
//! backend in use (`portable` on a target without an assembly backend). Run it
//! with `CALIBRATION_GRID=~/grid.csv cargo test --release --features
//! "glv multicore asm" glv::msm::calibration::grid -- --ignored
//! --test-threads=1`. Optional:
//! `CALIBRATION_REPS` (default 11), `CALIBRATION_SIZES` (comma-separated term
//! counts) and `CALIBRATION_SEED` (32 hex digits).
//!
//! A grid describes the host it ran on, so grids are never committed: the test
//! refuses a `CALIBRATION_GRID` inside the checkout.

use alloc::vec::Vec;
use core::hint::black_box;
use std::io::Write;
use std::string::String;
use std::time::Instant;

use ff::{Field, PrimeField};
use group::{Curve as _, CurveAffine as _, Group};
use maybe_rayon::prelude::*;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

use crate::arithmetic::{CurveAffine, CurveExt};
use crate::glv::decompose;
use crate::glv::msm::booth::checked_signed_magnitudes;
use crate::glv::msm::bucket::{multiexp, multiexp_bases};
use crate::glv::msm::plan::default_multiexp_window_bits;
use crate::glv::msm::straus::{straus_multiexp, straus_multiexp_jobs};
use crate::pallas;

#[path = "../../../../benches/support/generic_msm.rs"]
mod generic;

/// The sizes timed when `CALIBRATION_SIZES` is not set: around the Straus
/// crossovers, the measured serial bucket widths, the modelled range, and the
/// sizes of Orchard's per-proof (2,049) and batch (7,150) MSMs.
const DEFAULT_SIZES: [usize; 25] = [
    8, 16, 17, 24, 25, 32, 48, 64, 66, 67, 96, 127, 128, 192, 255, 256, 257, 512, 1_024, 2_049,
    4_096, 7_150, 16_384, 32_768, 65_536,
];

/// Straus and its leaves are timed up to this many terms; above it the bucket
/// method wins on every machine measured so far by a wide margin.
const MAX_STRAUS_TERMS: usize = 256;

/// The repetitions per alternative when `CALIBRATION_REPS` is not set.
const DEFAULT_REPS: usize = 11;

/// One alternative the planner can choose for a size.
#[derive(Clone, Copy)]
enum Method {
    Generic,
    Buckets(usize),
    Straus,
    Leaves(usize),
}

impl Method {
    fn name(self) -> (&'static str, usize) {
        match self {
            Method::Generic => ("generic", 0),
            Method::Buckets(window_bits) => ("buckets", window_bits),
            Method::Straus => ("straus", 0),
            Method::Leaves(jobs) => ("leaves", jobs),
        }
    }
}

/// The bucket backend at a forced width: the planner's `Buckets` arm.
fn buckets(
    scalars: &[<pallas::Point as CurveExt>::ScalarExt],
    bases: &[pallas::Affine],
    window_bits: usize,
    num_threads: usize,
) -> pallas::Point {
    let components = scalars
        .iter()
        .map(decompose::<pallas::Point>)
        .map(checked_signed_magnitudes)
        .collect::<Option<Vec<_>>>()
        .expect("GLV halves are below 2^127");
    let bases = multiexp_bases::<pallas::Point>(bases);
    multiexp(&components, &bases, window_bits, num_threads).expect("valid inputs")
}

fn run(
    method: Method,
    scalars: &[<pallas::Point as CurveExt>::ScalarExt],
    bases: &[pallas::Affine],
    num_threads: usize,
) -> pallas::Point {
    match method {
        Method::Generic => generic::best_multiexp::<pallas::Affine>(scalars, bases),
        Method::Buckets(window_bits) => buckets(scalars, bases, window_bits, num_threads),
        Method::Straus => straus_multiexp::<pallas::Point>(scalars, bases),
        Method::Leaves(jobs) => straus_multiexp_jobs::<pallas::Point>(scalars, bases, jobs),
    }
}

/// The alternatives timed at `terms` terms on `num_threads` threads.
fn methods(terms: usize, num_threads: usize) -> Vec<Method> {
    let default = default_multiexp_window_bits(terms).expect("terms fit in u32");
    let low = default.saturating_sub(1).max(2);
    let mut methods: Vec<Method> = (low..=default + 3).map(Method::Buckets).collect();
    methods.push(Method::Generic);
    if terms <= MAX_STRAUS_TERMS {
        methods.push(Method::Straus);
        if num_threads > 1 && terms >= 2 {
            let mut jobs: Vec<usize> = [1, num_threads.div_ceil(2), num_threads]
                .into_iter()
                .map(|jobs| jobs.min(terms))
                .collect();
            jobs.dedup();
            methods.extend(jobs.into_iter().map(Method::Leaves));
        }
    }
    methods
}

/// Panics unless `path` lies outside the checkout, so that a grid cannot end
/// up committed.
fn assert_outside_checkout(path: &str) {
    let path = std::path::Path::new(path);
    let parent = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => std::path::Path::new("."),
    };
    let parent = std::fs::canonicalize(parent).expect("the directory of CALIBRATION_GRID exists");
    let checkout = std::fs::canonicalize(env!("CARGO_MANIFEST_DIR")).expect("the checkout");
    assert!(
        !parent.starts_with(&checkout),
        "CALIBRATION_GRID ({}) is inside the checkout; grids must stay outside it",
        parent.display()
    );
}

fn env_or<T: core::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse()
                .ok()
                .unwrap_or_else(|| panic!("{name}: {value}"))
        })
        .unwrap_or(default)
}

#[test]
#[ignore = "a calibration measurement, run by scripts/calibrate-msm.sh"]
fn timing_grid() {
    let path = std::env::var("CALIBRATION_GRID").expect("CALIBRATION_GRID: the output CSV");
    assert_outside_checkout(&path);
    let reps: usize = env_or("CALIBRATION_REPS", DEFAULT_REPS);
    let sizes: Vec<usize> = match std::env::var("CALIBRATION_SIZES") {
        Ok(list) => list
            .split(',')
            .map(|n| n.trim().parse().expect("a term count"))
            .collect(),
        Err(_) => DEFAULT_SIZES.to_vec(),
    };
    let seed: u128 = std::env::var("CALIBRATION_SEED")
        .map(|hex| u128::from_str_radix(&hex, 16).expect("CALIBRATION_SEED: 32 hex digits"))
        .unwrap_or(0x0123_4567_89ab_cdef_0123_4567_89ab_cdef);
    let num_threads = maybe_rayon::current_num_threads();

    let max = sizes.iter().copied().max().unwrap_or(0);
    let mut rng = XorShiftRng::from_seed(seed.to_be_bytes());
    let scalars: Vec<_> = (0..max)
        .map(|_| <pallas::Point as CurveExt>::ScalarExt::random(&mut rng))
        .collect();
    let projective: Vec<pallas::Point> =
        (0..max).map(|_| pallas::Point::random(&mut rng)).collect();
    let mut bases = alloc::vec![pallas::Affine::identity(); max];
    pallas::Point::batch_normalize(&projective, &mut bases);

    let fresh = std::fs::metadata(&path)
        .map(|m| m.len() == 0)
        .unwrap_or(true);
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .expect("CALIBRATION_GRID is writable");
    if fresh {
        writeln!(out, "threads,terms,method,param,median_ns,reps").expect("write");
    }
    writeln!(out, "# backend: {}", crate::BACKEND).expect("write");
    let glv = if cfg!(feature = "glv-eisenstein") {
        "glv-eisenstein"
    } else {
        "glv"
    };
    writeln!(out, "# glv: {glv}").expect("write");

    for terms in sizes {
        let (scalars, bases) = (&scalars[..terms], &bases[..terms]);
        let methods = methods(terms, num_threads);
        let expected = run(Method::Generic, scalars, bases, num_threads);
        for &method in &methods {
            assert_eq!(
                run(method, scalars, bases, num_threads),
                expected,
                "{:?} at {terms} terms",
                method.name()
            );
        }
        let mut times: Vec<Vec<u128>> = alloc::vec![Vec::with_capacity(reps); methods.len()];
        for _ in 0..reps {
            for (method, times) in methods.iter().zip(&mut times) {
                let start = Instant::now();
                black_box(run(
                    *method,
                    black_box(scalars),
                    black_box(bases),
                    num_threads,
                ));
                times.push(start.elapsed().as_nanos());
            }
        }
        for (method, mut times) in methods.into_iter().zip(times) {
            times.sort_unstable();
            let (name, param) = method.name();
            let row: String = std::format!(
                "{num_threads},{terms},{name},{param},{},{reps}",
                times[times.len() / 2]
            );
            writeln!(out, "{row}").expect("write");
        }
    }
}
