//! The values of the grid evaluation that depend on the machine: which grids
//! share their doublings, and how a grid is split between threads. A
//! [`Calibration`] holds them all; everything else in `glv::grid` is a rule
//! over them or fixed by the algorithm. They are not the MSM's
//! (`glv::msm::calibration`), and are calibrated apart from them.
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
//! - the thresholds: `batch_mul_scalars` against one table per point and a
//!   ladder per product, on one thread (`RAYON_NUM_THREADS=1`), at a few to a
//!   few tens of products;
//! - the blocks and the threads: `batch_mul_scalars` with `multicore`, from a
//!   few points against many scalars to thousands of points against a few,
//!   as the `batch several scalars` groups of `benches/glv.rs` and
//!   `benches/glv_eisenstein.rs` run.
//!
//! Choose the values that minimise, over those measurements, the time lost
//! against the fastest alternative of each shape. Then add them as a constant
//! of type [`Calibration`] named after the targets it covers, and select it in
//! [`CALIBRATION`] with a `cfg` on the target, keeping [`DEFAULT`] as the
//! fallback for every other target. The machine measured, the build and the
//! measurements belong in the commit that adds the values, not here.
//!
//! # Trying other values
//!
//! Building with `--cfg pasta_curves_batch_mul_scalars_calibration` replaces the
//! calibration in use with the one in the environment variable
//! `PASTA_CURVES_BATCH_MUL_SCALARS_CALIBRATION`, read at compile time: comma-separated
//! `field=value` pairs, the fields being those of [`Calibration`]. The values
//! stay constants, and cargo rebuilds the crate when the variable changes.
//! Against a baseline saved without the override, the benchmarks compare
//! them:
//!
//! ```text
//! cargo bench --features glv,multicore,asm --bench glv -- \
//!   'batch several scalars' --save-baseline current
//! RUSTFLAGS='--cfg pasta_curves_batch_mul_scalars_calibration' \
//! PASTA_CURVES_BATCH_MUL_SCALARS_CALIBRATION='min_block_products=16,max_threads=10' \
//!   cargo bench --features glv,multicore,asm --bench glv -- \
//!   'batch several scalars' --baseline current
//! ```
//!
//! The values only choose between evaluations that compute the same result,
//! so an override changes speed, never results.

// Only builds that read an override contain the parser; tests check it
// without the flag.
#[cfg(any(test, pasta_curves_batch_mul_scalars_calibration))]
mod overrides;

/// The machine-dependent values the grid evaluation reads.
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
pub(crate) struct Calibration {
    /// Fewest (point, scalar) products from which `glv::batch_mul_scalars`
    /// shares each point's doublings among the scalars; below it, one table
    /// per point and a ladder per product are cheaper.
    pub(crate) glv_min_products: usize,
    /// The same for `glv_eisenstein::batch_mul_scalars`. The shared doublings
    /// spend one field inversion per column and a few more to combine the
    /// buckets, about 135 in all whatever the grid, against roughly 400
    /// multiplications saved per product with two scalars and more with more.
    #[cfg(feature = "glv-eisenstein")]
    pub(crate) eisenstein_min_products: usize,
    /// Fewest products in a block when a grid is split between threads: below
    /// it a block's own inversions, the same whatever its size, outweigh what
    /// its thread saves. A block holds at least `min_block_products / m`
    /// points for `m` scalars, so the more scalars, the fewer points it needs.
    #[cfg(feature = "multicore")]
    pub(super) min_block_products: usize,
    /// Fewest points in a block for a grid to be split into
    /// `balanced_blocks_per_thread` blocks per thread rather than one.
    #[cfg(feature = "multicore")]
    pub(super) min_balanced_block_lanes: usize,
    /// Blocks per thread for grids large enough: Rayon's work stealing then
    /// balances threads that run at different speeds, such as performance and
    /// efficiency cores, but each extra block pays its own inversions, which
    /// only a block of many points makes up for.
    #[cfg(feature = "multicore")]
    pub(super) balanced_blocks_per_thread: usize,
    /// Most threads a grid is split between, or 0 for every thread of the
    /// pool. Where some cores are much slower than the others, leaving them
    /// out can be faster than balancing.
    #[cfg(feature = "multicore")]
    pub(super) max_threads: usize,
}

/// The calibration of every target without one of its own.
pub(crate) const DEFAULT: Calibration = Calibration {
    glv_min_products: 16,
    #[cfg(feature = "glv-eisenstein")]
    eisenstein_min_products: 16,
    #[cfg(feature = "multicore")]
    min_block_products: 8,
    #[cfg(feature = "multicore")]
    min_balanced_block_lanes: 16,
    #[cfg(feature = "multicore")]
    balanced_blocks_per_thread: 2,
    #[cfg(feature = "multicore")]
    max_threads: 0,
};

/// The calibration in use. Every target uses [`DEFAULT`].
#[cfg(not(pasta_curves_batch_mul_scalars_calibration))]
pub(crate) const CALIBRATION: Calibration = DEFAULT;

/// The calibration in use: the one in `PASTA_CURVES_BATCH_MUL_SCALARS_CALIBRATION`, read at
/// compile time.
#[cfg(pasta_curves_batch_mul_scalars_calibration)]
pub(crate) const CALIBRATION: Calibration = match option_env!(
    "PASTA_CURVES_BATCH_MUL_SCALARS_CALIBRATION"
) {
    Some(spec) => overrides::parse(spec),
    None => panic!(
        "`--cfg pasta_curves_batch_mul_scalars_calibration` needs PASTA_CURVES_BATCH_MUL_SCALARS_CALIBRATION"
    ),
};
