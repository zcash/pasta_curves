//! Variable-time GLV MSMs for public scalars and bases.
//!
//! With `glv`, `CurveExt::try_multiexp_vartime` offers a variable-time MSM backend
//! for public scalars and bases. It uses a shared-doubling GLV-Strauss ladder for
//! small inputs and signed-Booth GLV buckets with batched affine additions for
//! large inputs. It returns `None` when its cost model prefers the caller's
//! generic MSM; callers must keep that fallback. Enable `multicore` to evaluate
//! independent windows through the Rayon pool; WebAssembly without atomics uses
//! the serial implementation. Neither feature changes ordinary constant-time
//! scalar multiplication.
//!
//! The small-MSM ladder reuses the existing split-wNAF tables unchanged.

use alloc::vec::Vec;
use core::ops::Range;
use ff::{Field, PrimeField, WithSmallOrderMulGroup};
use group::CurveAffine as _;
#[cfg(feature = "multicore")]
use maybe_rayon::prelude::*;

use super::{Decomposed, GlvParams, Table, decompose};
use crate::arithmetic::{CurveAffine, VartimeField};
use crate::{pallas, vesta};

#[cfg(test)]
mod tests;

// Keep raw-coordinate plumbing private rather than strengthening the public
// CurveExt associated-type bounds or exposing a generic unchecked constructor.
pub(crate) trait MsmCurve: GlvParams {
    fn affine_unchecked(x: Self::Base, y: Self::Base) -> Self::AffineExt;
    fn affine_xy(point: &Self::AffineExt) -> (Self::Base, Self::Base);
}

macro_rules! impl_msm_curve {
    ($curve:ident) => {
        impl MsmCurve for $curve::Point {
            fn affine_unchecked(x: Self::Base, y: Self::Base) -> Self::AffineExt {
                $curve::Affine::from_xy_unchecked(x, y)
            }

            fn affine_xy(point: &Self::AffineExt) -> (Self::Base, Self::Base) {
                Option::<crate::arithmetic::Coordinates<$curve::Affine>>::from(point.coordinates())
                    .map(|p| (*p.x(), *p.y()))
                    .unwrap_or((Self::Base::ZERO, Self::Base::ZERO))
            }
        }
    };
}
impl_msm_curve!(pallas);
impl_msm_curve!(vesta);

/// Below this, the cost model is not consulted. With several worker threads
/// the hook then declines; serial MSMs use the measured choices below instead.
const MIN_GLV_MULTIEXP_TERMS: usize = 256;

/// Largest GLV-Strauss leaf when MSMs are split across worker threads.
const STRAUSS_LEAF_TERMS: usize = 16;

/// Largest serial MSM evaluated as one GLV-Strauss table. Measured on Pallas
/// with `benches/msm.rs`: one table beats both the generic MSM and the bucket
/// backend at 17 and 24 terms, and the bucket backend wins from 32 terms on.
const SERIAL_STRAUSS_MAX_TERMS: usize = 24;

/// Bucket-backend window width for serial MSMs between
/// `SERIAL_STRAUSS_MAX_TERMS` and `MIN_GLV_MULTIEXP_TERMS`, the fastest width
/// measured on Pallas: 5 from 32 to 96 terms, 6 from 128 to 255 terms. The cost
/// model counts a batched affine addition as a projective one, so it would
/// decline these sizes although they are 28% to 50% faster than the generic MSM.
fn serial_small_window_bits(terms: usize) -> usize {
    if terms < 128 { 5 } else { 6 }
}

/// Smallest MSM worth splitting across worker jobs.
const MIN_PARALLEL_STRAUSS_TERMS: usize = 3;

/// Largest parallel MSM split into [`STRAUSS_LEAF_TERMS`]-or-smaller Strauss
/// leaves. Above this conservative crossover, the generic or GLV bucket
/// backends amortize their bucket setup more effectively.
const MAX_PARALLEL_STRAUSS_TERMS: usize = 66;

/// Each GLV decomposition component has magnitude strictly below `2^127`.
const GLV_COMPONENT_BITS: usize = 127;

/// Estimated work of a signed-Booth MSM, in point operations on its critical
/// path. Every operation counts the same, so values compare two plans but do
/// not predict time.
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
) -> Option<BoothWork> {
    let windows = scalar_bits.checked_div(window_bits)?.checked_add(1)?;
    let bucket_shift = u32::try_from(window_bits.checked_sub(1)?).ok()?;
    let buckets = 1usize.checked_shl(bucket_shift)?;

    if num_threads <= 1 {
        let point_visits = terms.checked_mul(windows)?;
        let bucket_additions = buckets.checked_mul(windows)?.checked_mul(2)?;
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
    let per_window = terms.checked_add(buckets.checked_mul(2)?)?;
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

fn default_multiexp_window_bits(terms: usize) -> Option<usize> {
    if terms < 4 {
        Some(1)
    } else if terms < 32 {
        Some(3)
    } else {
        let terms = u32::try_from(terms).ok()?;
        let threshold = DEFAULT_MULTIEXP_WINDOW_THRESHOLDS
            .iter()
            .position(|upper| terms <= *upper)
            .unwrap_or(DEFAULT_MULTIEXP_WINDOW_THRESHOLDS.len());
        Some(4 + threshold)
    }
}

fn scalar_repr_bits<C: MsmCurve>() -> Option<usize> {
    <C::ScalarExt as PrimeField>::Repr::default()
        .as_ref()
        .len()
        .checked_mul(8)
}

/// Selects the backend's signed-window width.
///
/// Compare the nine-bit schedule with its wider successor in serial mode.
/// Keep the comparison narrow because this model omits cache behavior.
fn multiexp_window_bits<C: MsmCurve>(terms: usize, num_threads: usize) -> Option<usize> {
    let window_bits = default_multiexp_window_bits(terms)?;
    if num_threads != 1 || window_bits != 9 {
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
fn glv_multiexp_window_bits<C: MsmCurve>(terms: usize, num_threads: usize) -> Option<usize> {
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
fn estimated_generic_work<C: MsmCurve>(terms: usize, num_threads: usize) -> Option<BoothWork> {
    let window_bits = multiexp_window_bits::<C>(terms, num_threads)?;
    let scalar_bits = scalar_repr_bits::<C>()?;
    estimated_signed_booth_work(terms, scalar_bits, window_bits, num_threads)
}

/// The Signed-Booth GLV backend's cheapest `(work, window width)` over the
/// candidate widths `w` and `w + 1` (see [`glv_multiexp_window_bits`]).
fn booth_multiexp_estimate<C: MsmCurve>(
    terms: usize,
    num_threads: usize,
) -> Option<(BoothWork, usize)> {
    let window_bits = multiexp_window_bits::<C>(terms, num_threads)?;
    let glv_terms = terms.checked_mul(2)?;
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

fn current_num_threads() -> usize {
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

/// Share the doubling ladder across all scalar/base pairs, reusing the
/// upstream GLV decomposition and width-four split-wNAF tables unchanged.
fn strauss_multiexp<C: MsmCurve>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> C {
    debug_assert_eq!(scalars.len(), bases.len());

    let mut decomposed = Vec::with_capacity(scalars.len());
    let mut points = Vec::with_capacity(bases.len());
    for (scalar, base) in scalars.iter().zip(bases) {
        let scalar = Decomposed::<C>::new(scalar);
        if scalar.len == 0 || bool::from(base.is_identity()) {
            continue;
        }
        decomposed.push(scalar);
        points.push(base.to_curve());
    }
    if decomposed.is_empty() {
        return C::identity();
    }

    let tables = Table::batch(&points);
    let columns = decomposed
        .iter()
        .map(|scalar| scalar.len)
        .max()
        .expect("at least one nonzero scalar");
    let mut acc = C::identity();
    for column in (0..columns).rev() {
        if column + 1 < columns {
            acc = acc.double();
        }
        for (scalar, table) in decomposed.iter().zip(&tables) {
            Table::add_digit(&mut acc, &table.t1, scalar.digits1[column]);
            Table::add_digit(&mut acc, &table.t2, scalar.digits2[column]);
        }
    }
    acc
}

fn strauss_multiexp_leaves<C: MsmCurve>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> C {
    scalars
        .chunks(STRAUSS_LEAF_TERMS)
        .zip(bases.chunks(STRAUSS_LEAF_TERMS))
        .map(|(scalars, bases)| strauss_multiexp::<C>(scalars, bases))
        .fold(C::identity(), |left, right| left + right)
}

/// Splits the terms into `jobs` contiguous shares of Strauss leaves, run on
/// worker threads with `multicore` and one after another without it.
fn parallel_strauss_multiexp<C: MsmCurve>(
    scalars: &[C::ScalarExt],
    bases: &[C::AffineExt],
    jobs: usize,
    on_job: &(impl Fn() + Sync),
) -> C {
    let job = |job: usize| {
        on_job();
        let start = scalars.len() * job / jobs;
        let end = scalars.len() * (job + 1) / jobs;
        strauss_multiexp_leaves::<C>(&scalars[start..end], &bases[start..end])
    };
    #[cfg(feature = "multicore")]
    if jobs > 1 {
        return (0..jobs).into_par_iter().map(job).sum();
    }
    (0..jobs).map(job).sum()
}

/// How [`try_multiexp`] evaluates an MSM.
///
/// Every strategy except [`Plan::Decline`] computes the same sum, so the
/// choice affects only speed; [`plan`] makes it and [`execute`] carries it out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Plan {
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
fn plan<C: MsmCurve>(terms: usize, num_threads: usize) -> Plan {
    if num_threads > 1 && (MIN_PARALLEL_STRAUSS_TERMS..=MAX_PARALLEL_STRAUSS_TERMS).contains(&terms)
    {
        // Bound small-MSM jobs to half the pool, rounded up, leaving room for
        // concurrent caller work. Keep each leaf small instead of building
        // one wide table for an entire worker's share.
        return Plan::ParallelStrauss {
            jobs: num_threads.div_ceil(2).min(terms),
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SignedMagnitude {
    negative: bool,
    magnitude: u128,
}

impl From<(bool, u128)> for SignedMagnitude {
    fn from((negative, magnitude): (bool, u128)) -> Self {
        Self {
            negative,
            magnitude,
        }
    }
}

/// Converts decomposed halves into the representation used by the
/// Signed-Booth MSM, rejecting values outside the decomposition bound.
///
/// The large MSM does not construct a [`Decomposed`], so check the bound
/// before extracting digits. Declining lets the caller retain its generic MSM.
fn checked_signed_magnitudes(
    (first, second): ((bool, u128), (bool, u128)),
) -> Option<(SignedMagnitude, SignedMagnitude)> {
    if first.1 >> GLV_COMPONENT_BITS == 0 && second.1 >> GLV_COMPONENT_BITS == 0 {
        Some((first.into(), second.into()))
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BoothDigit {
    magnitude: usize,
    negative: bool,
}

#[derive(Clone, Copy)]
struct WindowAssignment {
    term: usize,
    signed_bucket: isize,
}

/// The MSM the bucket method evaluates: the sum over terms of
/// `scalar * point`, each scalar an integer of magnitude below
/// `2^GLV_COMPONENT_BITS`. It knows nothing about GLV; [`GlvTerms`] is the
/// implementation that [`execute`] uses.
trait SmallScalarTerms<F>: Sync {
    /// The number of terms.
    fn len(&self) -> usize;
    /// The scalar of term `term`.
    fn scalar(&self, term: usize) -> SignedMagnitude;
    /// The point of term `term`, or `None` for the identity.
    fn point(&self, term: usize) -> Option<AffinePoint<F>>;
}

/// One base with its endomorphism image, `phi(x, y) = (zeta * x, y)`, which
/// shares `y`.
#[derive(Clone, Copy)]
struct MultiexpBase<F> {
    x: F,
    endo_x: F,
    y: F,
    identity: bool,
}

/// The GLV lift of an MSM over the scalar field to one over the integers.
///
/// The curve is a module over the Eisenstein integers `Z[omega]`, with `omega`
/// acting as `phi = lambda`, and `Z[omega]` is free over `Z` with basis
/// `{1, omega}`. Restricting scalars along `Z -> Z[omega]` turns each term
/// `k * P` into the two integer terms `k1 * P` and `k2 * phi(P)`, where
/// `k = k1 + k2 * lambda` with `|k1|, |k2| < 2^127` ([`decompose`]). Term
/// `2 i` is `(k1_i, P_i)` and term `2 i + 1` is `(k2_i, phi(P_i))`.
struct GlvTerms<F> {
    components: Vec<(SignedMagnitude, SignedMagnitude)>,
    bases: Vec<MultiexpBase<F>>,
}

impl<F: Copy + Sync> SmallScalarTerms<F> for GlvTerms<F> {
    fn len(&self) -> usize {
        2 * self.bases.len()
    }

    fn scalar(&self, term: usize) -> SignedMagnitude {
        let (first, second) = self.components[term / 2];
        if term % 2 == 0 { first } else { second }
    }

    fn point(&self, term: usize) -> Option<AffinePoint<F>> {
        let base = self.bases[term / 2];
        (!base.identity).then_some(AffinePoint {
            x: if term % 2 == 0 { base.x } else { base.endo_x },
            y: base.y,
        })
    }
}

/// Lifts an MSM to [`GlvTerms`], or declines when a decomposition half is
/// out of bounds.
fn glv_lift<C: MsmCurve>(
    scalars: &[C::ScalarExt],
    bases: &[C::AffineExt],
) -> Option<GlvTerms<C::Base>> {
    let components = scalars
        .iter()
        .map(decompose::<C>)
        .map(checked_signed_magnitudes)
        .collect::<Option<Vec<_>>>()?;
    Some(GlvTerms {
        components,
        bases: multiexp_bases::<C>(bases),
    })
}

impl WindowAssignment {
    fn bucket(self) -> usize {
        self.signed_bucket.unsigned_abs() - 1
    }

    fn negative(self) -> bool {
        self.signed_bucket < 0
    }
}

/// Extracts one signed-Booth digit and folds in the component's overall sign.
fn booth_digit(component: SignedMagnitude, window_bits: usize, window: usize) -> BoothDigit {
    let window_start = window * window_bits;
    let radix = 1usize << window_bits;
    let value = if window_start < 128 {
        // Only the low window_bits bits survive; widths are below usize::BITS.
        ((component.magnitude >> window_start) as usize) & (radix - 1)
    } else {
        0
    };
    let overlap = if window_start == 0 {
        0
    } else {
        // The masked bit is either zero or one, which always fits usize.
        ((component.magnitude >> (window_start - 1)) & 1) as usize
    };

    let (magnitude, digit_negative) = if value < radix / 2 {
        (value + overlap, false)
    } else {
        let magnitude = radix - value - overlap;
        (magnitude, magnitude != 0)
    };
    BoothDigit {
        magnitude,
        negative: magnitude != 0 && (digit_negative ^ component.negative),
    }
}

#[derive(Clone, Copy)]
struct AffinePoint<F> {
    x: F,
    y: F,
}

// The left operand lives in its `points[output]` result slot. The prefix pass
// scales `numerator` in place, so the backward pass can recover the slope
// directly without keeping a separate inversion scratch field.
struct PendingAffineAddition<F> {
    output: usize,
    x_sum: F,
    numerator: F,
    denominator: F,
}

/// Independent multiplication lanes used by batch inversion.
const BATCH_INVERSION_LANES: usize = 2;

/// Batch-inverts affine denominators and immediately finishes the additions.
///
/// Each `output` identifies the slot containing the addition's left operand.
/// The outputs must be distinct. Returns `None` if the product of the
/// denominators is zero. In that case, this function has not written to
/// `points`: the prefix pass changes only disposable pending numerators, and
/// the output pass starts only after the product is successfully inverted.
/// The reducer relies on this failure-atomic behavior to retry the level.
fn batch_invert_and_add<F: VartimeField>(
    additions: &mut [PendingAffineAddition<F>],
    points: &mut [AffinePoint<F>],
) -> Option<()> {
    let Some((first, additions)) = additions.split_first_mut() else {
        return Some(());
    };

    if additions.is_empty() {
        let denominator_inverse = first.denominator.invert_vartime()?;
        let left = points[first.output];
        let slope = first.numerator * denominator_inverse;
        let x = slope.square() - first.x_sum;
        let y = slope * (left.x - x) - left.y;
        points[first.output] = AffinePoint { x, y };
        return Some(());
    }

    let (second, additions) = additions.split_first_mut().unwrap();

    // Compute two prefix lanes in lockstep. This retains one field inversion
    // for the entire batch while exposing independent multiplication chains.
    // Seeding from the first denominator in each lane removes the initial
    // multiplication by one. Pre-scale each numerator by its denominator's
    // prefix product so the backward pass recovers the slope directly. This
    // moves one multiplication off the dependent path before each square
    // without changing the total multiplication count.
    let mut lane_products = [first.denominator, second.denominator];
    for pair in additions.chunks_mut(BATCH_INVERSION_LANES) {
        for (addition, product) in pair.iter_mut().zip(&mut lane_products) {
            addition.numerator *= *product;
            *product *= addition.denominator;
        }
    }

    // Invert both lane products with one inversion. A field has no zero
    // divisors, so this product is zero exactly when at least one affine
    // denominator is zero. No output point has been written yet; `?` therefore
    // makes failure atomic with respect to `points`. Writes to
    // pending numerators are discarded by the caller on failure.
    let product = lane_products[0] * lane_products[1];
    // This MSM is already variable-time with respect to scalar digits; batch
    // inversion does not provide a constant-time guarantee.
    let product_inverse = product.invert_vartime()?;
    let mut lane_inverses = [
        lane_products[1] * product_inverse,
        lane_products[0] * product_inverse,
    ];

    // If the last pair is incomplete, consume its first lane before walking
    // the complete pairs backward.
    let complete_pairs = additions.len() / BATCH_INVERSION_LANES;
    if additions.len() % BATCH_INVERSION_LANES != 0 {
        let addition = &additions[additions.len() - 1];
        let denominator = addition.denominator;
        let slope = addition.numerator * lane_inverses[0];
        lane_inverses[0] *= denominator;

        let left = points[addition.output];
        let x = slope.square() - addition.x_sum;
        let y = slope * (left.x - x) - left.y;
        points[addition.output] = AffinePoint { x, y };
    }
    for pair in additions[..complete_pairs * BATCH_INVERSION_LANES]
        .chunks(BATCH_INVERSION_LANES)
        .rev()
    {
        let first = &pair[0];
        let second = &pair[1];
        let first_slope = first.numerator * lane_inverses[0];
        let second_slope = second.numerator * lane_inverses[1];
        lane_inverses[0] *= first.denominator;
        lane_inverses[1] *= second.denominator;

        // Complete each affine chord addition while its pending record is
        // already resident. Keep two independent field-operation lanes.
        let first_left = points[first.output];
        let second_left = points[second.output];
        let first_x = first_slope.square() - first.x_sum;
        let second_x = second_slope.square() - second.x_sum;
        let first_y = first_slope * (first_left.x - first_x) - first_left.y;
        let second_y = second_slope * (second_left.x - second_x) - second_left.y;
        points[first.output] = AffinePoint {
            x: first_x,
            y: first_y,
        };
        points[second.output] = AffinePoint {
            x: second_x,
            y: second_y,
        };
    }

    let first_left = points[first.output];
    let second_left = points[second.output];
    let first_slope = first.numerator * lane_inverses[0];
    let second_slope = second.numerator * lane_inverses[1];
    let first_x = first_slope.square() - first.x_sum;
    let second_x = second_slope.square() - second.x_sum;
    let first_y = first_slope * (first_left.x - first_x) - first_left.y;
    let second_y = second_slope * (second_left.x - second_x) - second_left.y;
    points[first.output] = AffinePoint {
        x: first_x,
        y: first_y,
    };
    points[second.output] = AffinePoint {
        x: second_x,
        y: second_y,
    };
    Some(())
}

/// Collects the terms in `range` with a nonzero digit in one Booth window,
/// each with its signed bucket, and counts the terms per bucket. This is the
/// pushforward of the window's terms along their digits.
fn window_assignments<F, T: SmallScalarTerms<F>>(
    terms: &T,
    range: Range<usize>,
    window_bits: usize,
    window: usize,
) -> (Vec<WindowAssignment>, Vec<usize>) {
    let bucket_count = 1 << (window_bits - 1);
    let mut assignments = Vec::with_capacity(range.len());
    let mut counts = alloc::vec![0usize; bucket_count];

    for term in range {
        if terms.point(term).is_none() {
            continue;
        }
        let digit = booth_digit(terms.scalar(term), window_bits, window);
        if digit.magnitude != 0 {
            let bucket = digit.magnitude - 1;
            let magnitude =
                isize::try_from(digit.magnitude).expect("Booth digit magnitude fits in isize");
            assignments.push(WindowAssignment {
                term,
                signed_bucket: if digit.negative {
                    -magnitude
                } else {
                    magnitude
                },
            });
            counts[bucket] += 1;
        }
    }

    (assignments, counts)
}

/// Reduce buckets with one Montgomery inversion per tree level. Inputs are
/// valid nonidentity points on a Pasta curve (a = 0); offsets partition them.
/// The incomplete pass writes nothing until inversion succeeds. On a zero
/// denominator, retry the unchanged level with complete formulas. Successful
/// results are then compacted in place, retaining the input allocation.
fn reduce_affine_buckets<F: VartimeField>(
    points: Vec<AffinePoint<F>>,
    offsets: Vec<usize>,
) -> Option<Vec<Option<AffinePoint<F>>>> {
    reduce_affine_buckets_inner::<F, false>(points, offsets)
}

fn reduce_affine_buckets_inner<F: VartimeField, const COMPLETE: bool>(
    mut points: Vec<AffinePoint<F>>,
    mut offsets: Vec<usize>,
) -> Option<Vec<Option<AffinePoint<F>>>> {
    debug_assert!(!offsets.is_empty());
    let mut next_offsets = Vec::with_capacity(offsets.len());
    let mut pending = Vec::with_capacity(points.len() / 2);

    while offsets.windows(2).any(|range| range[1] - range[0] > 1) {
        pending.clear();
        for range in offsets.windows(2) {
            for input in (range[0]..range[1] - (range[1] - range[0]) % 2).step_by(2) {
                let left = points[input];
                let right = points[input + 1];
                let dx = right.x - left.x;
                let (numerator, denominator) = if COMPLETE && dx.is_zero_vartime() {
                    // Equal x means equal or opposite points. Omit inverse
                    // pairs, including order-two points, from the next level.
                    if !(right.y - left.y).is_zero_vartime() || left.y.is_zero_vartime() {
                        continue;
                    }
                    let square = left.x.square();
                    (square.double() + square, left.y.double())
                } else {
                    (right.y - left.y, dx)
                };
                pending.push(PendingAffineAddition {
                    output: input,
                    x_sum: left.x + right.x,
                    numerator,
                    denominator,
                });
            }
        }

        if batch_invert_and_add(&mut pending, &mut points).is_none() {
            if !COMPLETE {
                return reduce_affine_buckets_inner::<F, true>(points, offsets);
            }
            // Complete formulas have nonzero denominators for valid inputs.
            // Decline rather than returning a partial MSM if that guard fails.
            return None;
        }

        // Pending outputs identify surviving pairs; cancelled pairs have no
        // record. A carried odd point survives unchanged. The write index is
        // never ahead of the read index, so compaction cannot clobber a result.
        next_offsets.clear();
        next_offsets.push(0);
        let mut additions = pending.iter().peekable();
        let mut output = 0;
        for range in offsets.windows(2) {
            let end_pairs = range[1] - (range[1] - range[0]) % 2;
            for input in (range[0]..end_pairs).step_by(2) {
                if additions
                    .peek()
                    .is_some_and(|addition| addition.output == input)
                {
                    points[output] = points[input];
                    output += 1;
                    additions.next();
                }
            }
            if end_pairs != range[1] {
                points[output] = points[end_pairs];
                output += 1;
            }
            next_offsets.push(output);
        }
        points.truncate(output);
        core::mem::swap(&mut offsets, &mut next_offsets);
    }

    Some(
        offsets
            .windows(2)
            .map(|range| (range[0] != range[1]).then(|| points[range[0]]))
            .collect(),
    )
}

fn sum_buckets<C: MsmCurve>(buckets: &[Option<AffinePoint<C::Base>>]) -> C {
    let mut running = C::identity();
    let mut sum = C::identity();
    for bucket in buckets.iter().rev() {
        if let Some(point) = bucket {
            running += C::affine_unchecked(point.x, point.y);
        }
        sum += running;
    }
    sum
}

/// Smallest number of terms a window reduces at a time.
const MIN_WINDOW_BLOCK_TERMS: usize = 1 << 15;

/// Terms per bucket in a window block. Each block after the first carries one
/// point per nonempty bucket, so this bounds the extra additions to about
/// `1 / WINDOW_BLOCK_TERMS_PER_BUCKET` of the block's points.
const WINDOW_BLOCK_TERMS_PER_BUCKET: usize = 16;

/// Fills and reduces one window's buckets.
///
/// The terms are processed in blocks, so the window's scratch memory (points,
/// pending additions and assignments) is bounded by the block size instead of
/// growing with the number of terms. With parallel windows this bound is what
/// keeps memory from scaling with the number of windows in flight.
fn fill_window<F: VartimeField, T: SmallScalarTerms<F>>(
    terms: &T,
    window_bits: usize,
    window: usize,
) -> Option<Vec<Option<AffinePoint<F>>>> {
    let bucket_count = 1usize << (window_bits - 1);
    let block_terms =
        MIN_WINDOW_BLOCK_TERMS.max(bucket_count.saturating_mul(WINDOW_BLOCK_TERMS_PER_BUCKET));
    fill_window_in_blocks(terms, window_bits, window, block_terms)
}

/// Reduces a window block by block. Each block's bucket results enter the next
/// block as the first point of the same buckets, so the batched affine reducer
/// also merges the blocks, including when a carried point doubles or cancels.
/// This is the monoid law `sum(A + B) = sum({sum(A)} + B)` for multisets.
fn fill_window_in_blocks<F: VartimeField, T: SmallScalarTerms<F>>(
    terms: &T,
    window_bits: usize,
    window: usize,
    block_terms: usize,
) -> Option<Vec<Option<AffinePoint<F>>>> {
    debug_assert!(block_terms > 0);
    let mut buckets: Option<Vec<Option<AffinePoint<F>>>> = None;
    for start in (0..terms.len()).step_by(block_terms) {
        let end = terms.len().min(start + block_terms);
        buckets = Some(fill_block(
            terms,
            start..end,
            window_bits,
            window,
            buckets.as_deref(),
        )?);
    }
    Some(buckets.unwrap_or_else(|| alloc::vec![None; 1 << (window_bits - 1)]))
}

/// Fills one block's buckets, each led by its carried point from the previous
/// block, and reduces them.
fn fill_block<F: VartimeField, T: SmallScalarTerms<F>>(
    terms: &T,
    range: Range<usize>,
    window_bits: usize,
    window: usize,
    carry: Option<&[Option<AffinePoint<F>>]>,
) -> Option<Vec<Option<AffinePoint<F>>>> {
    let bucket_count = 1 << (window_bits - 1);
    let (assignments, mut counts) = window_assignments(terms, range, window_bits, window);
    if let Some(carry) = carry {
        for (count, point) in counts.iter_mut().zip(carry) {
            *count += usize::from(point.is_some());
        }
    }

    let mut offsets = Vec::with_capacity(bucket_count + 1);
    offsets.push(0);
    for count in counts {
        offsets.push(offsets.last().copied().unwrap() + count);
    }

    let mut positions = offsets[..bucket_count].to_vec();
    let mut points = alloc::vec![
        AffinePoint {
            x: F::ZERO,
            y: F::ZERO,
        };
        *offsets.last().unwrap()
    ];
    if let Some(carry) = carry {
        for (bucket, point) in carry.iter().enumerate() {
            if let Some(point) = point {
                points[positions[bucket]] = *point;
                positions[bucket] += 1;
            }
        }
    }
    for assignment in assignments {
        let point = terms
            .point(assignment.term)
            .expect("only terms with a point are assigned");
        let bucket = assignment.bucket();
        let position = positions[bucket];
        points[position] = AffinePoint {
            x: point.x,
            y: if assignment.negative() {
                -point.y
            } else {
                point.y
            },
        };
        positions[bucket] += 1;
    }

    reduce_affine_buckets(points, offsets)
}

/// One window's share of the MSM: the sum over terms of `digit * point`,
/// computed as `sum_b b * S_b` over the window's buckets.
fn window_sum<C: MsmCurve, T: SmallScalarTerms<C::Base>>(
    terms: &T,
    window_bits: usize,
    window: usize,
) -> Option<C> {
    let buckets = fill_window(terms, window_bits, window)?;
    Some(sum_buckets::<C>(&buckets))
}

/// Horner's rule at `X = 2^window_bits`: given the window sums from the top
/// window down, returns `sum_j W_j * 2^(window_bits * j)`. The top window is
/// not shifted, so the fold starts from it rather than from the identity.
fn horner<C: MsmCurve>(top_down: impl Iterator<Item = Option<C>>, window_bits: usize) -> Option<C> {
    let mut acc: Option<C> = None;
    for sum in top_down {
        let sum = sum?;
        acc = Some(match acc {
            None => sum,
            Some(mut acc) => {
                for _ in 0..window_bits {
                    acc = acc.double();
                }
                acc + sum
            }
        });
    }
    Some(acc.unwrap_or_else(C::identity))
}

/// The bucket method over `terms` with `window_bits`-bit signed-Booth
/// windows. With `parallel`, the windows' sums are independent jobs (on
/// worker threads with `multicore`, one after another without it), combined
/// by one Horner fold; otherwise each window is computed as the fold reaches
/// it, so only one window's buckets exist at a time.
fn bucket_multiexp<C: MsmCurve, T: SmallScalarTerms<C::Base>>(
    terms: &T,
    window_bits: usize,
    parallel: bool,
    on_job: &(impl Fn() + Sync),
) -> Option<C> {
    let windows = GLV_COMPONENT_BITS / window_bits + 1;
    if !parallel {
        let top_down = (0..windows)
            .rev()
            .map(|window| window_sum::<C, T>(terms, window_bits, window));
        return horner(top_down, window_bits);
    }

    let job = |window| {
        on_job();
        window_sum::<C, T>(terms, window_bits, window)
    };
    #[cfg(feature = "multicore")]
    let sums: Vec<Option<C>> = (0..windows).into_par_iter().map(job).collect();
    #[cfg(not(feature = "multicore"))]
    let sums: Vec<Option<C>> = (0..windows).map(job).collect();
    horner(sums.into_iter().rev(), window_bits)
}

/// Carries out `plan`. Declines (`None`) for [`Plan::Decline`] and when an
/// arithmetic guard fails, so the caller's generic MSM runs instead.
fn execute<C: MsmCurve>(plan: Plan, scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> Option<C> {
    execute_observed(plan, scalars, bases, &|| ())
}

/// [`execute`], calling `on_job` at the start of every parallel job. Tests use
/// it to check that parallel plans really run their jobs concurrently.
fn execute_observed<C: MsmCurve>(
    plan: Plan,
    scalars: &[C::ScalarExt],
    bases: &[C::AffineExt],
    on_job: &(impl Fn() + Sync),
) -> Option<C> {
    debug_assert_eq!(scalars.len(), bases.len());
    match plan {
        Plan::Decline => None,
        Plan::Strauss => Some(strauss_multiexp::<C>(scalars, bases)),
        Plan::ParallelStrauss { jobs } => {
            Some(parallel_strauss_multiexp::<C>(scalars, bases, jobs, on_job))
        }
        Plan::Buckets {
            window_bits,
            parallel,
        } => {
            let terms = glv_lift::<C>(scalars, bases)?;
            bucket_multiexp::<C, _>(&terms, window_bits, parallel, on_job)
        }
    }
}

fn multiexp_bases<C: MsmCurve>(bases: &[C::AffineExt]) -> Vec<MultiexpBase<C::Base>> {
    bases
        .iter()
        .map(|base| {
            let (x, y) = C::affine_xy(base);
            MultiexpBase {
                x,
                endo_x: x * C::Base::ZETA,
                y,
                identity: bool::from(base.is_identity()),
            }
        })
        .collect()
}

/// Declines unprofitable input shapes and failed arithmetic guards so callers
/// can retain their generic MSM.
///
/// # Panics
///
/// Panics if `scalars` and `bases` have different lengths.
pub(crate) fn try_multiexp<C: MsmCurve>(
    scalars: &[C::ScalarExt],
    bases: &[C::AffineExt],
) -> Option<C> {
    assert_eq!(
        scalars.len(),
        bases.len(),
        "try_multiexp_vartime: scalars and bases differ in length"
    );
    execute(
        plan::<C>(scalars.len(), current_num_threads()),
        scalars,
        bases,
    )
}
