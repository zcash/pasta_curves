//! Choosing how to evaluate an MSM.
//!
//! This is algorithm selection: every [`Plan`] except [`Plan::Decline`]
//! computes the same sum, so [`plan`] only weighs their cost, from measured
//! crossovers below 256 terms and from a cost model above. Being a pure
//! function of the term and thread counts, its choices are tested directly.

use ff::PrimeField;

use super::MsmCurve;
use super::lift::{GLV_COMPONENT_BITS, GLV_TERMS_PER_TERM};
use super::strauss::STRAUSS_LEAF_TERMS;

/// Below this, the cost model is not consulted. With several worker threads
/// the hook then declines; serial MSMs use the measured choices below instead.
pub(super) const MIN_GLV_MULTIEXP_TERMS: usize = 256;

/// Largest serial MSM evaluated as one GLV-Strauss table. Measured on Pallas
/// with `benches/msm.rs`: one table beats both the generic MSM and the bucket
/// backend at 17 and 24 terms, and the bucket backend wins from 32 terms on.
pub(super) const SERIAL_STRAUSS_MAX_TERMS: usize = 24;

/// Fastest bucket-backend width measured on Pallas for serial MSMs above
/// [`SERIAL_STRAUSS_MAX_TERMS`] and below [`SERIAL_WIDE_WINDOW_TERMS`].
pub(super) const SERIAL_NARROW_WINDOW_BITS: usize = 5;

/// Fastest width measured on Pallas from [`SERIAL_WIDE_WINDOW_TERMS`] up to
/// [`MIN_GLV_MULTIEXP_TERMS`].
pub(super) const SERIAL_WIDE_WINDOW_BITS: usize = 6;

/// Smallest serial MSM that uses [`SERIAL_WIDE_WINDOW_BITS`]: 5 bits measured
/// fastest up to 96 terms, 6 bits from 128.
pub(super) const SERIAL_WIDE_WINDOW_TERMS: usize = 128;

/// Bucket-backend window width for serial MSMs between
/// [`SERIAL_STRAUSS_MAX_TERMS`] and [`MIN_GLV_MULTIEXP_TERMS`]. The cost model
/// counts a batched affine addition as a projective one, so it would decline
/// these sizes although they are 28% to 50% faster than the generic MSM.
pub(super) fn serial_small_window_bits(terms: usize) -> usize {
    if terms < SERIAL_WIDE_WINDOW_TERMS {
        SERIAL_NARROW_WINDOW_BITS
    } else {
        SERIAL_WIDE_WINDOW_BITS
    }
}

/// Smallest MSM worth splitting across worker jobs.
pub(super) const MIN_PARALLEL_STRAUSS_TERMS: usize = 3;

/// A parallel Strauss MSM runs at most `1 / PARALLEL_STRAUSS_POOL_DIVISOR` of
/// the pool's threads as jobs, rounded up, leaving the rest to concurrent
/// caller work.
pub(super) const PARALLEL_STRAUSS_POOL_DIVISOR: usize = 2;

/// Largest parallel MSM split into [`STRAUSS_LEAF_TERMS`]-or-smaller Strauss
/// leaves. Above this conservative crossover, the generic or GLV bucket
/// backends amortize their bucket setup more effectively.
pub(super) const MAX_PARALLEL_STRAUSS_TERMS: usize = 66;

/// Estimated work of a signed-Booth MSM, in point operations on its critical
/// path. Every operation counts the same, so values compare two plans but do
/// not predict time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct BoothWork(pub(super) usize);

/// Additions per bucket to combine a window's buckets as `sum_b b * S_b` by
/// suffix sums: one into the running sum, one of the running sum into the
/// total.
pub(super) const ADDITIONS_PER_BUCKET: usize = 2;

/// The serial work model counts total point/window visits, bucket additions,
/// and accumulator doublings. The parallel model estimates the corresponding
/// critical-worker cost when windows have independent accumulators. Both omit
/// GLV setup costs; [`MIN_GLV_MULTIEXP_TERMS`] handles their small-MSM
/// amortization separately. Returns `None` on overflow or a zero width.
pub(super) fn estimated_signed_booth_work(
    terms: usize,
    scalar_bits: usize,
    window_bits: usize,
    num_threads: usize,
) -> Option<BoothWork> {
    // Signed-Booth digits need one window more than `scalar_bits / window_bits`
    // for the final carry.
    let windows = scalar_bits.checked_div(window_bits)?.checked_add(1)?;
    // Signed digits have magnitude at most `2^(window_bits - 1)`, one bucket
    // per nonzero magnitude.
    let bucket_shift = u32::try_from(window_bits.checked_sub(1)?).ok()?;
    let buckets = 1usize.checked_shl(bucket_shift)?;

    if num_threads <= 1 {
        let point_visits = terms.checked_mul(windows)?;
        let bucket_additions = buckets
            .checked_mul(windows)?
            .checked_mul(ADDITIONS_PER_BUCKET)?;
        let doublings = window_bits.checked_mul(windows.checked_sub(1)?)?;
        return Some(BoothWork(
            point_visits
                .checked_add(bucket_additions)?
                .checked_add(doublings)?,
        ));
    }

    // Parallel windows have independent accumulators. Estimate the critical
    // worker's full-window work and the shifts of its highest-cost windows.
    let workers = num_threads.min(windows);
    let waves = windows.div_ceil(workers);
    let per_window = terms.checked_add(buckets.checked_mul(ADDITIONS_PER_BUCKET)?)?;
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
            .checked_add(shift_doublings)?,
    ))
}

/// Floors of `e^k` for `k = 4..=22`.
///
/// These preserve the established `ceil(ln(terms))` window schedule without
/// requiring floating-point transcendental functions, which are unavailable
/// in `no_std` builds. They are computed by `sage/msm_window_thresholds.sage`,
/// which prints this table so the two can be diffed.
pub(super) const DEFAULT_MULTIEXP_WINDOW_THRESHOLDS: [u32; 19] = [
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

/// Below this many terms, the default schedule uses [`TINY_MSM_WINDOW_BITS`].
pub(super) const SMALL_MSM_MIN_TERMS: usize = 4;

/// Window width of the default schedule below [`SMALL_MSM_MIN_TERMS`].
pub(super) const TINY_MSM_WINDOW_BITS: usize = 1;

/// From this many terms, the default schedule is `ceil(ln(terms))`; below it,
/// [`SMALL_MSM_WINDOW_BITS`].
pub(super) const LARGE_MSM_MIN_TERMS: usize = 32;

/// Window width of the default schedule from [`SMALL_MSM_MIN_TERMS`] to
/// [`LARGE_MSM_MIN_TERMS`].
pub(super) const SMALL_MSM_WINDOW_BITS: usize = 3;

/// The width `ceil(ln(terms))` up to the first threshold, `floor(e^4)`: entry
/// `i` of [`DEFAULT_MULTIEXP_WINDOW_THRESHOLDS`] bounds width
/// `FIRST_THRESHOLD_WINDOW_BITS + i`.
pub(super) const FIRST_THRESHOLD_WINDOW_BITS: usize = 4;

/// The generic MSM's window schedule, that of halo2's `best_multiexp`: one bit
/// for tiny MSMs, three for small ones, then `ceil(ln(terms))`.
pub(super) fn default_multiexp_window_bits(terms: usize) -> Option<usize> {
    if terms < SMALL_MSM_MIN_TERMS {
        Some(TINY_MSM_WINDOW_BITS)
    } else if terms < LARGE_MSM_MIN_TERMS {
        Some(SMALL_MSM_WINDOW_BITS)
    } else {
        let terms = u32::try_from(terms).ok()?;
        let threshold = DEFAULT_MULTIEXP_WINDOW_THRESHOLDS
            .iter()
            .position(|upper| terms <= *upper)
            .unwrap_or(DEFAULT_MULTIEXP_WINDOW_THRESHOLDS.len());
        FIRST_THRESHOLD_WINDOW_BITS.checked_add(threshold)
    }
}

/// Bit length of the scalar field's byte representation.
pub(super) fn scalar_repr_bits<C>() -> Option<usize>
where
    C: MsmCurve,
{
    <C::ScalarExt as PrimeField>::Repr::default()
        .as_ref()
        .len()
        .checked_mul(u8::BITS as usize)
}

/// The one default width at which a serial MSM also estimates its successor
/// and takes the cheaper. Only one width is compared because the cost model
/// omits cache behaviour, which favours narrower windows.
pub(super) const SERIAL_SUCCESSOR_CHECK_WINDOW_BITS: usize = 9;

/// Selects the backend's signed-window width: the default schedule, widened by
/// one bit at [`SERIAL_SUCCESSOR_CHECK_WINDOW_BITS`] when the model says so.
pub(super) fn multiexp_window_bits<C>(terms: usize, num_threads: usize) -> Option<usize>
where
    C: MsmCurve,
{
    let window_bits = default_multiexp_window_bits(terms)?;
    if num_threads != 1 || window_bits != SERIAL_SUCCESSOR_CHECK_WINDOW_BITS {
        return Some(window_bits);
    }

    let scalar_bits = scalar_repr_bits::<C>()?;
    let next_window_bits = window_bits.checked_add(1)?;
    let current = estimated_signed_booth_work(terms, scalar_bits, window_bits, num_threads);
    let next = estimated_signed_booth_work(terms, scalar_bits, next_window_bits, num_threads);
    Some(
        if matches!((current, next), (Some(current), Some(next)) if next < current) {
            next_window_bits
        } else {
            window_bits
        },
    )
}

/// Choose a GLV window only when its estimated work beats the generic MSM.
/// Compare both the default width and its successor to account for worker waves.
pub(super) fn glv_multiexp_window_bits<C>(terms: usize, num_threads: usize) -> Option<usize>
where
    C: MsmCurve,
{
    if terms < MIN_GLV_MULTIEXP_TERMS {
        return (num_threads <= 1 && terms > SERIAL_STRAUSS_MAX_TERMS)
            .then(|| serial_small_window_bits(terms));
    }
    let generic = estimated_generic_work::<C>(terms, num_threads)?;
    let (glv, glv_window_bits) = booth_multiexp_estimate::<C>(terms, num_threads)?;
    (glv < generic).then_some(glv_window_bits)
}

/// The generic Signed-Booth MSM's estimated work at its default width (see
/// [`estimated_signed_booth_work`]).
pub(super) fn estimated_generic_work<C>(terms: usize, num_threads: usize) -> Option<BoothWork>
where
    C: MsmCurve,
{
    let window_bits = multiexp_window_bits::<C>(terms, num_threads)?;
    let scalar_bits = scalar_repr_bits::<C>()?;
    estimated_signed_booth_work(terms, scalar_bits, window_bits, num_threads)
}

/// The Signed-Booth GLV backend's cheapest `(work, window width)` over the
/// candidate widths `w` and `w + 1` (see [`glv_multiexp_window_bits`]).
pub(super) fn booth_multiexp_estimate<C>(
    terms: usize,
    num_threads: usize,
) -> Option<(BoothWork, usize)>
where
    C: MsmCurve,
{
    let window_bits = multiexp_window_bits::<C>(terms, num_threads)?;
    let glv_terms = terms.checked_mul(GLV_TERMS_PER_TERM)?;
    let mut best: Option<(BoothWork, usize)> = None;
    for candidate in window_bits..=window_bits.checked_add(1)? {
        if let Some(work) =
            estimated_signed_booth_work(glv_terms, GLV_COMPONENT_BITS, candidate, num_threads)
        {
            if best.is_none_or(|(best_work, _)| work < best_work) {
                best = Some((work, candidate));
            }
        }
    }
    best
}

/// How [`try_multiexp`](super::try_multiexp) evaluates an MSM.
///
/// Every strategy except [`Plan::Decline`] computes the same sum, so the
/// choice affects only speed; [`plan`] makes it and [`execute`](super::execute) carries it out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Plan {
    /// Leave the MSM to the caller's generic implementation.
    Decline,
    /// One GLV-Strauss table over all terms.
    Strauss,
    /// `jobs` worker jobs, each running [`STRAUSS_LEAF_TERMS`]-term Strauss
    /// leaves over a contiguous share of the terms.
    ParallelStrauss { jobs: usize },
    /// GLV signed-Booth buckets with `window_bits`-bit windows, evaluated one
    /// window per worker job when `parallel`, serially otherwise.
    Buckets { window_bits: usize, parallel: bool },
}

/// Chooses how to evaluate an MSM of `terms` terms with `num_threads` worker
/// threads. A pure function, so its choices can be checked directly.
pub(super) fn plan<C>(terms: usize, num_threads: usize) -> Plan
where
    C: MsmCurve,
{
    if num_threads > 1 && (MIN_PARALLEL_STRAUSS_TERMS..=MAX_PARALLEL_STRAUSS_TERMS).contains(&terms)
    {
        // Keep each leaf small instead of building one wide table for an
        // entire worker's share.
        return Plan::ParallelStrauss {
            jobs: num_threads
                .div_ceil(PARALLEL_STRAUSS_POOL_DIVISOR)
                .min(terms),
        };
    }

    let strauss_max_terms = if num_threads <= 1 {
        SERIAL_STRAUSS_MAX_TERMS
    } else {
        STRAUSS_LEAF_TERMS
    };
    if terms <= strauss_max_terms {
        return Plan::Strauss;
    }
    match glv_multiexp_window_bits::<C>(terms, num_threads) {
        Some(window_bits) => Plan::Buckets {
            window_bits,
            parallel: num_threads > 1,
        },
        None => Plan::Decline,
    }
}
