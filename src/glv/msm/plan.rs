//! The planner: decides how an MSM is evaluated, from its numbers of terms
//! and threads alone.
//!
//! # Decisions
//!
//! [`plan`] returns a [`Plan`], in this order, the bounds being the fields of
//! the calibration in use ([`CALIBRATION`]):
//!
//! | threads | terms                                         | plan                       |
//! |---------|-----------------------------------------------|----------------------------|
//! | > 1     | `min_parallel_straus_terms` to `max_parallel_straus_terms` | `StrausLeaves` |
//! | 1       | up to `serial_straus_max_terms`               | `Straus`                   |
//! | > 1     | up to `straus_leaf_terms`                     | `Straus`                   |
//! | 1       | up to `min_glv_multiexp_terms`                | `Buckets`, serial widths   |
//! | > 1     | below `min_glv_multiexp_terms`                | `Decline`                  |
//! | any     | above                                         | modelled `Buckets` or `Decline` |
//!
//! `StrausLeaves` takes `ceil(threads / STRAUS_POOL_SHARES).min(terms)` jobs;
//! the serial widths are `serial_narrow_window_bits` below
//! `serial_wide_window_min_terms` terms and `serial_wide_window_bits` from
//! there. Several threads arise only with `multicore`.
//!
//! # Reading guide
//!
//! [`plan`] is the entry point; [`plan_with`] takes the calibration as an
//! argument. It first gives small inputs to Straus, at the measured crossovers
//! of the calibration and `STRAUS_POOL_SHARES`. Otherwise
//! [`glv_multiexp_window_bits`] picks the bucket width, or declines, on one of
//! two paths.
//!
//! **Measured, for small inputs:** up to `min_glv_multiexp_terms` terms on
//! one thread, or below it on several. On one thread the width is the one
//! [`serial_small_window_bits`] gives, from benchmarks; on several the bucket
//! backend declines at these sizes.
//!
//! **Modelled, for large inputs:** [`estimated_signed_booth_work`] estimates
//! the work on the critical path ([`BoothWork`]) of both competitors, each
//! operation weighted by its cost in that backend ([`PlannerWeights`]):
//!
//! - the generic MSM the caller falls back to (`generic_weights`):
//!   $n$ full-width terms of [`scalar_repr_bits`] bits, at the width it runs,
//!   from the schedule of `halo2_proofs`' `best_multiexp`
//!   ([`default_multiexp_window_bits`]);
//! - the bucket backend (`glv_weights`): $2n$ half-width components, at that
//!   width and up to `glv_extra_window_bits` more, keeping the cheapest.
//!
//! The backend runs when its estimate is no worse than the generic MSM's. The
//! estimates compare plans; they do not predict time.
//!
//! [`current_num_threads`] is the thread count all of this plans for.

use ff::PrimeField;

use super::calibration::{CALIBRATION, Calibration};
use crate::glv::{GlvParams, HALF_BITS, HALVES};

#[cfg(test)]
mod tests;

/// The parallel Straus path splits into at most the pool's size divided by
/// this, rounded up: the IPA prover computes its L and R commitments at the
/// same time, and each gets an equal share.
#[cfg(feature = "multicore")]
pub(super) const STRAUS_POOL_SHARES: usize = 2;

/// Bucket-backend window width for serial MSMs between the Straus crossover
/// and `c.min_glv_multiexp_terms`, the fastest width measured. The cost model
/// counts a batched affine addition as a projective one, so it would decline
/// these sizes although the bucket backend measures faster than the generic MSM
/// there.
fn serial_small_window_bits(terms: usize, c: &Calibration) -> usize {
    if terms < c.serial_wide_window_min_terms {
        c.serial_narrow_window_bits
    } else {
        c.serial_wide_window_bits
    }
}

/// Additions per bucket when a window sums its buckets with a running sum: one
/// into the running sum, one into the window's total.
const BUCKET_SUM_ADDITIONS: usize = 2;

/// Estimated critical-path work of a signed-Booth MSM, weighted by backend.
/// Values compare plans but do not predict time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct BoothWork(usize);

/// The serial work model counts total point/window visits, bucket additions,
/// and accumulator doublings. The parallel model estimates the corresponding
/// critical-worker cost when windows have independent accumulators. Both omit
/// GLV setup costs; `min_glv_multiexp_terms` handles their small-MSM
/// amortization separately. Returns `None` on overflow or a zero width.
fn estimated_signed_booth_work(
    terms: usize,
    scalar_bits: usize,
    window_bits: usize,
    num_threads: usize,
    weights: PlannerWeights,
) -> Option<BoothWork> {
    let windows = scalar_bits.checked_div(window_bits)?.checked_add(1)?;
    let bucket_shift = u32::try_from(window_bits.checked_sub(1)?).ok()?;
    let buckets = 1usize.checked_shl(bucket_shift)?;
    let visit = |count: usize| count.checked_mul(weights.point_visit);
    let bucket_adds = |count: usize| count.checked_mul(weights.bucket_addition);
    let doublings = |count: usize| count.checked_mul(weights.doubling);

    if num_threads <= 1 {
        let work = visit(terms.checked_mul(windows)?)?
            .checked_add(bucket_adds(
                buckets
                    .checked_mul(windows)?
                    .checked_mul(BUCKET_SUM_ADDITIONS)?,
            )?)?
            .checked_add(doublings(
                window_bits.checked_mul(windows.checked_sub(1)?)?,
            )?)?;
        return Some(BoothWork(work));
    }

    let workers = num_threads.min(windows);
    let waves = windows.div_ceil(workers);
    let per_window =
        visit(terms)?.checked_add(bucket_adds(buckets.checked_mul(BUCKET_SUM_ADDITIONS)?)?)?;
    let mut shift_doublings = 0usize;
    let mut window = windows.checked_sub(1)?;
    for _ in 0..waves {
        shift_doublings = shift_doublings.checked_add(window_bits.checked_mul(window)?)?;
        match window.checked_sub(workers) {
            Some(next) => window = next,
            None => break,
        }
    }
    Some(BoothWork(
        per_window
            .checked_mul(waves)?
            .checked_add(doublings(shift_doublings)?)?,
    ))
}

/// Backend-specific work weights, in fifths of a doubling. Calibrated on
/// Apple M4 and AMD Zen 4, 256–65,536 terms and 1–16 workers: GLV's
/// batched affine visits cost less than the generic projective bucket visits.
#[derive(Clone, Copy)]
pub(super) struct PlannerWeights {
    pub(super) point_visit: usize,
    pub(super) bucket_addition: usize,
    pub(super) doubling: usize,
}

/// Floors of `e^k` for `k = 4..=22`.
///
/// These preserve the established `ceil(ln(terms))` window schedule without
/// requiring floating-point transcendental functions, which are unavailable
/// in `no_std` builds. They are computed by `sage/msm_window_thresholds.sage`,
/// which prints this table so the two can be diffed.
const DEFAULT_MULTIEXP_WINDOW_THRESHOLDS: [u32; 19] = [
    54,
    148,
    403,
    1_096,
    2_980,
    8_103,
    22_026,
    59_874,
    162_754,
    442_413,
    1_202_604,
    3_269_017,
    8_886_110,
    24_154_952,
    65_659_969,
    178_482_300,
    485_165_195,
    1_318_815_734,
    3_584_912_846,
];

/// Below this many terms, `best_multiexp` uses [`TINY_MULTIEXP_WINDOW_BITS`].
const TINY_MULTIEXP_TERMS: usize = 4;

/// The window width `best_multiexp` uses below [`TINY_MULTIEXP_TERMS`] terms.
const TINY_MULTIEXP_WINDOW_BITS: usize = 1;

/// Below this many terms, and from [`TINY_MULTIEXP_TERMS`], `best_multiexp`
/// uses [`SMALL_MULTIEXP_WINDOW_BITS`]; from here on, `ceil(ln(terms))`.
const SMALL_MULTIEXP_TERMS: usize = 32;

/// The window width `best_multiexp` uses below [`SMALL_MULTIEXP_TERMS`] terms.
const SMALL_MULTIEXP_WINDOW_BITS: usize = 3;

/// The width `ceil(ln(terms))` up to the first of
/// [`DEFAULT_MULTIEXP_WINDOW_THRESHOLDS`], `e^4`; each later threshold adds
/// one bit.
const FIRST_LN_WINDOW_BITS: usize = 4;

/// The window width `best_multiexp` in `halo2_proofs` uses:
/// [`TINY_MULTIEXP_WINDOW_BITS`] below [`TINY_MULTIEXP_TERMS`],
/// [`SMALL_MULTIEXP_WINDOW_BITS`] below [`SMALL_MULTIEXP_TERMS`], and
/// `ceil(ln(terms))` above, read from [`DEFAULT_MULTIEXP_WINDOW_THRESHOLDS`].
/// `None` above `u32::MAX` terms.
fn default_multiexp_window_bits(terms: usize) -> Option<usize> {
    if terms < TINY_MULTIEXP_TERMS {
        Some(TINY_MULTIEXP_WINDOW_BITS)
    } else if terms < SMALL_MULTIEXP_TERMS {
        Some(SMALL_MULTIEXP_WINDOW_BITS)
    } else {
        let terms = u32::try_from(terms).ok()?;
        let threshold = DEFAULT_MULTIEXP_WINDOW_THRESHOLDS
            .iter()
            .position(|upper| terms <= *upper)
            .unwrap_or(DEFAULT_MULTIEXP_WINDOW_THRESHOLDS.len());
        Some(FIRST_LN_WINDOW_BITS + threshold)
    }
}

/// The bit length of a scalar's byte encoding (256 for the Pasta scalars), the
/// width the generic MSM recodes.
fn scalar_repr_bits<C>() -> Option<usize>
where
    C: GlvParams,
{
    <C::ScalarExt as PrimeField>::Repr::default()
        .as_ref()
        .len()
        .checked_mul(u8::BITS as usize)
}

/// How [`super::try_multiexp`] evaluates an MSM, decided by [`plan`] from the
/// numbers of terms and threads alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Plan {
    /// One GLV-Straus table over all terms.
    Straus,
    /// GLV-Straus leaves of at most [`Calibration::straus_leaf_terms`] terms, split into
    /// `jobs` parallel jobs, run on the calling thread when `jobs` is 1.
    #[cfg(feature = "multicore")]
    StrausLeaves {
        /// The number of jobs, at least 1 and at most the number of terms.
        jobs: usize,
    },
    /// The signed-Booth bucket backend at this window width.
    Buckets {
        /// The width of the signed-Booth windows.
        window_bits: usize,
    },
    /// Not profitable here: the caller keeps its generic MSM.
    Decline,
}

/// Decides how to evaluate an MSM of `terms` terms on `num_threads` threads,
/// with the calibration in use ([`CALIBRATION`]).
pub(super) fn plan<C>(terms: usize, num_threads: usize) -> Plan
where
    C: GlvParams,
{
    plan_with::<C>(terms, num_threads, &CALIBRATION)
}

/// [`plan`] with the calibration `c`.
fn plan_with<C>(terms: usize, num_threads: usize, c: &Calibration) -> Plan
where
    C: GlvParams,
{
    #[cfg(feature = "multicore")]
    if num_threads > 1
        && (c.min_parallel_straus_terms..=c.max_parallel_straus_terms).contains(&terms)
    {
        // Bound small-MSM jobs to a share of the pool (STRAUS_POOL_SHARES),
        // leaving room for concurrent caller work. Keep each leaf small
        // instead of building one wide table for an entire worker's share.
        return Plan::StrausLeaves {
            jobs: num_threads.div_ceil(STRAUS_POOL_SHARES).min(terms),
        };
    }
    let straus_max_terms = if num_threads <= 1 {
        c.serial_straus_max_terms
    } else {
        c.straus_leaf_terms
    };
    if terms <= straus_max_terms {
        return Plan::Straus;
    }
    match glv_multiexp_window_bits::<C>(terms, num_threads, c) {
        Some(window_bits) => Plan::Buckets { window_bits },
        None => Plan::Decline,
    }
}

/// Choose the cheapest weighted GLV width when it is no worse than generic.
/// Half-width components can favor up to two bits beyond the generic width.
/// Called by [`plan`] once Straus has declined.
fn glv_multiexp_window_bits<C>(terms: usize, num_threads: usize, c: &Calibration) -> Option<usize>
where
    C: GlvParams,
{
    let min_terms = c.min_glv_multiexp_terms;
    if terms < min_terms || (terms == min_terms && num_threads <= 1) {
        return (num_threads <= 1).then(|| serial_small_window_bits(terms, c));
    }

    let window_bits = default_multiexp_window_bits(terms)?;
    let generic = estimated_signed_booth_work(
        terms,
        scalar_repr_bits::<C>()?,
        window_bits,
        num_threads,
        c.generic_weights,
    )?;
    let glv_terms = terms.checked_mul(HALVES)?;
    let mut best: Option<(BoothWork, usize)> = None;
    for candidate in window_bits..=window_bits.checked_add(c.glv_extra_window_bits)? {
        if let Some(work) =
            estimated_signed_booth_work(glv_terms, HALF_BITS, candidate, num_threads, c.glv_weights)
        {
            if best.is_none_or(|(best_work, _)| work < best_work) {
                best = Some((work, candidate));
            }
        }
    }
    let (glv, glv_window_bits) = best?;
    (glv <= generic).then_some(glv_window_bits)
}

/// The number of threads the MSM may use: Rayon's with `multicore`, except on
/// WebAssembly without atomics, and 1 otherwise.
pub(super) fn current_num_threads() -> usize {
    #[cfg(all(
        feature = "multicore",
        not(all(target_arch = "wasm32", not(target_feature = "atomics")))
    ))]
    {
        maybe_rayon::current_num_threads()
    }
    #[cfg(any(
        not(feature = "multicore"),
        all(target_arch = "wasm32", not(target_feature = "atomics"))
    ))]
    {
        1
    }
}
