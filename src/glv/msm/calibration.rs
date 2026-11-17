//! The constants of the MSM that depend on the machine: the measured
//! crossovers between Straus, the bucket method and the generic MSM, the
//! planner's weights, and the window blocks. They are the only values to
//! revisit when calibrating on another machine; everything else in `glv::msm`
//! is either a rule over them or fixed by the algorithm.

use super::plan::PlannerWeights;

#[cfg(all(test, feature = "multicore"))]
mod grid;

/// The cost model starts here for parallel MSMs and above this for serial MSMs.
/// Serial MSMs through 256 terms use measured Straus and bucket choices.
pub(super) const MIN_GLV_MULTIEXP_TERMS: usize = 256;

/// Largest serial MSM evaluated as one GLV-Straus table. Measured on Pallas
/// with `benches/msm.rs`: one table beats both the generic MSM and the bucket
/// backend at 17 and 24 terms, and the bucket backend wins from 32 terms on.
pub(super) const SERIAL_STRAUS_MAX_TERMS: usize = 24;

/// Largest GLV-Straus leaf when MSMs are split across worker threads: the
/// measured crossover above which one table stopped beating the bucket backend
/// on one thread.
pub(super) const STRAUS_LEAF_TERMS: usize = 16;

/// Smallest MSM worth splitting across worker jobs.
#[cfg(feature = "multicore")]
pub(super) const MIN_PARALLEL_STRAUS_TERMS: usize = 3;

/// Largest parallel MSM split into [`STRAUS_LEAF_TERMS`]-or-smaller Straus
/// leaves. Above this conservative crossover, the generic or GLV bucket
/// backends amortize their bucket setup more effectively.
///
/// Tuned for the halo2 IPA prover, whose last rounds commit to L and R with
/// 66-term MSMs. `benches/msm.rs` brackets it with 66 and 67 terms.
#[cfg(feature = "multicore")]
pub(super) const MAX_PARALLEL_STRAUS_TERMS: usize = 66;

/// Serial window width of the bucket backend below
/// [`SERIAL_WIDE_WINDOW_MIN_TERMS`] terms, the fastest measured from 32 to 96.
pub(super) const SERIAL_NARROW_WINDOW_BITS: usize = 5;

/// Serial window width of the bucket backend from
/// [`SERIAL_WIDE_WINDOW_MIN_TERMS`] terms through [`MIN_GLV_MULTIEXP_TERMS`],
/// the fastest measured from 128 to 256.
pub(super) const SERIAL_WIDE_WINDOW_BITS: usize = 6;

/// Smallest serial MSM that takes [`SERIAL_WIDE_WINDOW_BITS`]: the measured
/// switch point between the two widths.
pub(super) const SERIAL_WIDE_WINDOW_MIN_TERMS: usize = 128;

/// The GLV ladder: affine batched point visits, projective bucket sums.
pub(super) const GLV_PLANNER_WEIGHTS: PlannerWeights = PlannerWeights {
    point_visit: 5,
    bucket_addition: 10,
    doubling: 5,
};

pub(super) const GENERIC_PLANNER_WEIGHTS: PlannerWeights = PlannerWeights {
    point_visit: 7,
    bucket_addition: 5,
    doubling: 5,
};

pub(super) const GLV_EXTRA_WINDOW_BITS: usize = 2;

/// Smallest number of terms a window reduces at a time. It bounds a window's
/// scratch, which would otherwise grow with the input and be multiplied by the
/// windows in flight, at a small measured cost in time for MSMs that need
/// several blocks, as `benches/msm.rs` does at 65,536 terms.
pub(super) const MIN_WINDOW_BLOCK_TERMS: usize = 1 << 14;

/// Terms per bucket in a window block. Each block after the first carries one
/// point per nonempty bucket, so this bounds the extra additions to about
/// `1 / (2 * WINDOW_BLOCK_TERMS_PER_BUCKET)` of the block's points.
pub(super) const WINDOW_BLOCK_TERMS_PER_BUCKET: usize = 8;
