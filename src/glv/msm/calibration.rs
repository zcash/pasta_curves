//! The values of the MSM that depend on the machine: the crossovers between
//! Straus, the bucket method and the generic MSM, the serial bucket widths, the
//! planner's weights, and the window blocks. A [`Calibration`] holds them all;
//! everything else in `glv::msm` is a rule over them or fixed by the algorithm.
//!
//! [`CALIBRATION`] is the calibration in use, chosen per target at compile
//! time.
//!
//! # Providing another machine's values
//!
//! Each value is measured by timing the alternatives it chooses between, on
//! the same inputs, on the machine to calibrate, built as deployed (`glv`,
//! `multicore` and `asm`, plus any `RUSTFLAGS` the deployment uses), at the
//! thread counts the deployment runs with:
//!
//! - the Straus crossovers and the serial widths: one Straus table, Straus
//!   leaves at several job counts, and the bucket method at each width, from a
//!   handful of terms to a few hundred;
//! - the weights and the extra GLV widths: the generic MSM against the bucket
//!   method at widths around halo2's default, from a few hundred terms up;
//! - the window blocks: large MSMs, for time and for memory.
//!
//! Choose the values that minimise, over those measurements, the time lost
//! against the fastest alternative of each size and thread count, and check
//! them on an end-to-end benchmark of `CurveExt::try_multiexp_vartime`, such
//! as `benches/msm.rs`.
//!
//! Then add the values as a constant of type [`Calibration`] named after the
//! targets it covers, and select it in [`CALIBRATION`] with a `cfg` on the
//! target, keeping [`DEFAULT`] as the fallback for every other target. The
//! machine measured, the build and the measurements belong in the commit that
//! adds the values, not here. The planner's tests pin the decisions of the
//! calibration in use; a new calibration pins its own.
//!
//! # Trying other values
//!
//! Building with `--cfg pasta_curves_calibration` replaces the calibration in
//! use with the one in the environment variable `PASTA_CURVES_CALIBRATION`,
//! read at compile time (see [`overrides`] for the syntax). The values stay
//! constants, and cargo rebuilds the crate when the variable changes. Against a
//! baseline saved without the override, `benches/msm.rs` compares them:
//!
//! ```text
//! cargo bench --features glv,multicore,asm --bench msm -- \
//!   'msm/Pallas/hook-or-generic' --save-baseline current
//! RUSTFLAGS='--cfg pasta_curves_calibration' \
//! PASTA_CURVES_CALIBRATION='min_glv_multiexp_terms=64,glv_weights=5/6/5' \
//!   cargo bench --features glv,multicore,asm --bench msm -- \
//!   'msm/Pallas/hook-or-generic' --baseline current
//! ```
//!
//! The values only choose between methods that compute the same result, so an
//! override changes speed, never results.

use super::plan::PlannerWeights;

#[cfg_attr(not(any(test, pasta_curves_calibration)), allow(dead_code))]
mod overrides;

/// The machine-dependent values the planner and the bucket method read.
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
pub(super) struct Calibration {
    /// The cost model starts here for parallel MSMs and above this for serial
    /// MSMs. Serial MSMs up to here use the measured Straus and bucket choices.
    pub(super) min_glv_multiexp_terms: usize,
    /// Largest serial MSM evaluated as one GLV-Straus table, the crossover
    /// above which the bucket method is faster on one thread.
    pub(super) serial_straus_max_terms: usize,
    /// Largest GLV-Straus leaf when MSMs are split across worker threads, and
    /// the largest single Straus table on several threads.
    pub(super) straus_leaf_terms: usize,
    /// Smallest MSM worth splitting into Straus leaves across worker jobs.
    #[cfg(feature = "multicore")]
    pub(super) min_parallel_straus_terms: usize,
    /// Largest MSM split into Straus leaves across worker jobs; above it, the
    /// bucket method or the generic MSM amortises its setup better.
    #[cfg(feature = "multicore")]
    pub(super) max_parallel_straus_terms: usize,
    /// Serial window width of the bucket method below
    /// `serial_wide_window_min_terms` terms.
    pub(super) serial_narrow_window_bits: usize,
    /// Serial window width of the bucket method from
    /// `serial_wide_window_min_terms` terms through `min_glv_multiexp_terms`.
    pub(super) serial_wide_window_bits: usize,
    /// Smallest serial MSM that takes `serial_wide_window_bits`.
    pub(super) serial_wide_window_min_terms: usize,
    /// The cost of the GLV bucket method's operations: batched affine point
    /// visits and projective bucket sums.
    pub(super) glv_weights: PlannerWeights,
    /// The cost of the generic MSM's operations: projective (mixed) point
    /// visits and bucket sums.
    pub(super) generic_weights: PlannerWeights,
    /// How far past halo2's default width the GLV bucket method may widen. Its
    /// components are half as long, so fewer, wider windows often pay off.
    pub(super) glv_extra_window_bits: usize,
    /// Smallest number of terms a window reduces at a time. It bounds a
    /// window's scratch, which would otherwise grow with the input and be
    /// multiplied by the windows in flight, at a small cost in time for MSMs
    /// that need several blocks.
    pub(super) min_window_block_terms: usize,
    /// Terms per bucket in a window block. Each block after the first carries
    /// one point per nonempty bucket, so this bounds the extra additions to
    /// about `1 / (2 * window_block_terms_per_bucket)` of the block's points.
    pub(super) window_block_terms_per_bucket: usize,
}

/// The values the weighted planner was introduced with, used on every target
/// without a calibration of its own.
pub(super) const DEFAULT: Calibration = Calibration {
    min_glv_multiexp_terms: 256,
    serial_straus_max_terms: 24,
    straus_leaf_terms: 16,
    #[cfg(feature = "multicore")]
    min_parallel_straus_terms: 3,
    #[cfg(feature = "multicore")]
    max_parallel_straus_terms: 66,
    serial_narrow_window_bits: 5,
    serial_wide_window_bits: 6,
    serial_wide_window_min_terms: 128,
    glv_weights: PlannerWeights {
        point_visit: 5,
        bucket_addition: 10,
        doubling: 5,
    },
    generic_weights: PlannerWeights {
        point_visit: 7,
        bucket_addition: 5,
        doubling: 5,
    },
    glv_extra_window_bits: 2,
    min_window_block_terms: 1 << 14,
    window_block_terms_per_bucket: 8,
};

/// The calibration in use. Every target uses [`DEFAULT`].
#[cfg(not(pasta_curves_calibration))]
pub(super) const CALIBRATION: Calibration = DEFAULT;

/// The calibration in use: the one in `PASTA_CURVES_CALIBRATION`, read at
/// compile time.
#[cfg(pasta_curves_calibration)]
pub(super) const CALIBRATION: Calibration = match option_env!("PASTA_CURVES_CALIBRATION") {
    Some(spec) => overrides::parse(spec),
    None => panic!("`--cfg pasta_curves_calibration` needs PASTA_CURVES_CALIBRATION"),
};
