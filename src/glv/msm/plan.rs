//! The planner: estimates the work of the signed-Booth bucket MSM against the
//! generic MSM a caller falls back to, and chooses a window width or declines.
//!
//! # Reading guide
//!
//! [`glv_multiexp_window_bits`] is the entry point, called once Straus has
//! declined. It takes one of two paths.
//!
//! **Measured, for small inputs:** up to [`MIN_GLV_MULTIEXP_TERMS`] terms on
//! one thread, or below it on several. The cost model below is too coarse
//! there (it prices a batched affine addition like a projective one), so the
//! choices come from benchmarks: buckets of the width
//! [`serial_small_window_bits`] gives above [`SERIAL_STRAUS_MAX_TERMS`] (the
//! largest input Straus takes on one thread). On several threads the bucket
//! backend declines at these sizes.
//!
//! **Modelled, for large inputs:** [`estimated_signed_booth_work`] counts the
//! point operations on the critical path ([`BoothWork`]) of both competitors:
//!
//! - the generic MSM the caller falls back to:
//!   $n$ full-width terms of [`scalar_repr_bits`] bits, at the width
//!   [`multiexp_window_bits`] picks from the schedule of `halo2_proofs`'
//!   `best_multiexp` ([`default_multiexp_window_bits`]);
//! - the bucket backend: $2n$ half-width
//!   components at that width and the next.
//!
//! The backend runs only when it is estimated cheaper. The counts compare
//! plans; they do not predict time.
//!
//! [`current_num_threads`] is the thread count all of this plans for.

use ff::PrimeField;

use crate::glv::{GlvParams, HALF_BITS, HALVES};

#[cfg(test)]
mod tests;

/// The cost model starts here for parallel MSMs and above this for serial MSMs.
/// Serial MSMs through 256 terms use measured Straus and bucket choices.
const MIN_GLV_MULTIEXP_TERMS: usize = 256;

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

/// The parallel Straus path splits into at most the pool's size divided by
/// this, rounded up: the IPA prover computes its L and R commitments at the
/// same time, and each gets an equal share.
#[cfg(feature = "multicore")]
pub(super) const STRAUS_POOL_SHARES: usize = 2;

/// Serial window width of the bucket backend below
/// [`SERIAL_WIDE_WINDOW_MIN_TERMS`] terms, the fastest measured from 32 to 96.
const SERIAL_NARROW_WINDOW_BITS: usize = 5;

/// Serial window width of the bucket backend from
/// [`SERIAL_WIDE_WINDOW_MIN_TERMS`] terms through [`MIN_GLV_MULTIEXP_TERMS`],
/// the fastest measured from 128 to 256.
const SERIAL_WIDE_WINDOW_BITS: usize = 6;

/// Smallest serial MSM that takes [`SERIAL_WIDE_WINDOW_BITS`]: the measured
/// switch point between the two widths.
const SERIAL_WIDE_WINDOW_MIN_TERMS: usize = 128;

/// Bucket-backend window width for serial MSMs between
/// `SERIAL_STRAUS_MAX_TERMS` and `MIN_GLV_MULTIEXP_TERMS`, the fastest width
/// measured on Pallas. The cost model counts a batched affine addition as a
/// projective one, so it would decline these sizes although the bucket backend
/// measures faster than the generic MSM there.
fn serial_small_window_bits(terms: usize) -> usize {
    if terms < SERIAL_WIDE_WINDOW_MIN_TERMS {
        SERIAL_NARROW_WINDOW_BITS
    } else {
        SERIAL_WIDE_WINDOW_BITS
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
/// GLV setup costs; [`MIN_GLV_MULTIEXP_TERMS`] handles their small-MSM
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
struct PlannerWeights {
    point_visit: usize,
    bucket_addition: usize,
    doubling: usize,
}

/// Unit weights preserve the generic window schedule and unweighted reference estimates.
const UNIT_PLANNER_WEIGHTS: PlannerWeights = PlannerWeights {
    point_visit: 1,
    bucket_addition: 1,
    doubling: 1,
};

/// The GLV ladder: affine batched point visits, projective bucket sums.
const GLV_PLANNER_WEIGHTS: PlannerWeights = PlannerWeights {
    point_visit: 5,
    bucket_addition: 10,
    doubling: 5,
};

const GENERIC_PLANNER_WEIGHTS: PlannerWeights = PlannerWeights {
    point_visit: 7,
    bucket_addition: 5,
    doubling: 5,
};

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

/// The one schedule width that the serial planner also compares with its
/// successor. The comparison stays this narrow because the model omits cache
/// behaviour.
const SERIAL_WIDENED_WINDOW_BITS: usize = 9;

/// Selects the backend's signed-window width.
///
/// Compare [`SERIAL_WIDENED_WINDOW_BITS`] with its wider successor in serial
/// mode.
fn multiexp_window_bits<C>(terms: usize, num_threads: usize) -> Option<usize>
where
    C: GlvParams,
{
    let window_bits = default_multiexp_window_bits(terms)?;
    if num_threads != 1 || window_bits != SERIAL_WIDENED_WINDOW_BITS {
        return Some(window_bits);
    }

    let scalar_bits = scalar_repr_bits::<C>()?;
    let next_window_bits = window_bits.checked_add(1)?;
    let current = estimated_signed_booth_work(
        terms,
        scalar_bits,
        window_bits,
        num_threads,
        UNIT_PLANNER_WEIGHTS,
    );
    let next = estimated_signed_booth_work(
        terms,
        scalar_bits,
        next_window_bits,
        num_threads,
        UNIT_PLANNER_WEIGHTS,
    );
    Some(
        if matches!((current, next), (Some(current), Some(next)) if next < current) {
            next_window_bits
        } else {
            window_bits
        },
    )
}

/// Choose the cheapest weighted GLV width when it is no worse than generic.
/// Half-width components can favor up to two bits beyond the generic width.
pub(super) fn glv_multiexp_window_bits<C>(terms: usize, num_threads: usize) -> Option<usize>
where
    C: GlvParams,
{
    if terms < MIN_GLV_MULTIEXP_TERMS || (terms == MIN_GLV_MULTIEXP_TERMS && num_threads <= 1) {
        return (num_threads <= 1 && terms > SERIAL_STRAUS_MAX_TERMS)
            .then(|| serial_small_window_bits(terms));
    }

    let window_bits = multiexp_window_bits::<C>(terms, num_threads)?;
    let generic = estimated_signed_booth_work(
        terms,
        scalar_repr_bits::<C>()?,
        window_bits,
        num_threads,
        GENERIC_PLANNER_WEIGHTS,
    )?;
    let glv_terms = terms.checked_mul(HALVES)?;
    let mut best: Option<(BoothWork, usize)> = None;
    for candidate in window_bits..=window_bits.checked_add(GLV_EXTRA_WINDOW_BITS)? {
        if let Some(work) = estimated_signed_booth_work(
            glv_terms,
            HALF_BITS,
            candidate,
            num_threads,
            GLV_PLANNER_WEIGHTS,
        ) {
            if best.is_none_or(|(best_work, _)| work < best_work) {
                best = Some((work, candidate));
            }
        }
    }
    let (glv, glv_window_bits) = best?;
    (glv <= generic).then_some(glv_window_bits)
}

const GLV_EXTRA_WINDOW_BITS: usize = 2;

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
