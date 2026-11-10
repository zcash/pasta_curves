use alloc::vec::Vec;
use ff::{Field, PrimeField, WithSmallOrderMulGroup};
use group::CurveAffine as _;
#[cfg(feature = "multicore")]
use maybe_rayon::prelude::*;

use super::{Decomposed, GlvParams, Table, decompose};
use crate::arithmetic::CurveAffine;
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

/// Small MSMs do not amortize GLV decomposition, affine endomorphism mapping,
/// and the two temporary vectors.
const MIN_GLV_MULTIEXP_TERMS: usize = 256;

/// Largest serial leaf in the GLV-Strauss backend. The crossover is
/// benchmark-tuned against halo2's generic Signed-Booth fallback at the
/// power-of-two sizes reached by the late IPA rounds.
const STRAUSS_LEAF_TERMS: usize = 16;

/// Smallest MSM worth splitting across an IPA side's worker budget.
#[cfg(feature = "multicore")]
const MIN_PARALLEL_STRAUSS_TERMS: usize = 3;

/// Largest parallel MSM split into [`STRAUSS_LEAF_TERMS`]-or-smaller Strauss
/// leaves. This includes a 64-term late IPA half plus its two auxiliary terms.
/// Above this, the generic or GLV bucket backends amortize their bucket setup
/// and use the worker pool more efficiently.
#[cfg(feature = "multicore")]
const MAX_PARALLEL_STRAUSS_TERMS: usize = 66;

/// Each GLV decomposition component has magnitude strictly below `2^127`.
const GLV_COMPONENT_BITS: usize = 127;

/// Estimates a Signed-Booth MSM's costs as a `(work, traffic)` pair: the
/// dominant group work, and the total point/window visits plus bucket
/// additions — the traffic the planner's shared-bandwidth floor scales,
/// since a wide pool cannot divide it away (see `plan_multiexp`).
///
/// The serial work model counts total point/window visits, bucket additions,
/// and accumulator doublings. The parallel model estimates the corresponding
/// critical-worker cost when windows have independent accumulators. Both omit
/// GLV setup costs; [`MIN_GLV_MULTIEXP_TERMS`] handles their small-MSM
/// amortization separately.
fn estimated_signed_booth_costs(
    terms: usize,
    scalar_bits: usize,
    window_bits: usize,
    num_threads: usize,
) -> Option<(usize, usize)> {
    let windows = scalar_bits.checked_div(window_bits)?.checked_add(1)?;
    let bucket_shift = u32::try_from(window_bits.checked_sub(1)?).ok()?;
    let buckets = 1usize.checked_shl(bucket_shift)?;
    let point_visits = terms.checked_mul(windows)?;
    let bucket_additions = buckets.checked_mul(windows)?.checked_mul(2)?;
    let traffic = point_visits.checked_add(bucket_additions)?;

    if num_threads <= 1 {
        let doublings = window_bits.checked_mul(windows.checked_sub(1)?)?;
        return Some((traffic.checked_add(doublings)?, traffic));
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
    Some((
        per_window
            .checked_mul(waves)?
            .checked_add(shift_doublings)?,
        traffic,
    ))
}

/// Test convenience: the work component of [`estimated_signed_booth_costs`].
#[cfg(test)]
fn estimated_signed_booth_work(
    terms: usize,
    scalar_bits: usize,
    window_bits: usize,
    num_threads: usize,
) -> Option<usize> {
    estimated_signed_booth_costs(terms, scalar_bits, window_bits, num_threads).map(|(work, _)| work)
}

/// Floors of `e^k` for `k = 4..=22`.
///
/// These preserve the established `ceil(ln(terms))` window schedule without
/// requiring floating-point transcendental functions, which are unavailable
/// in `no_std` builds.
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
        .checked_mul(u8::BITS as usize)
}

/// Selects the backend's signed-window width.
///
/// The accepted serial sweep found a wider successor useful only when the
/// default selected 9 bits. Keep that comparison narrow because the operation
/// model deliberately omits cache behavior.
fn multiexp_window_bits<C: MsmCurve>(terms: usize, num_threads: usize) -> Option<usize> {
    let window_bits = default_multiexp_window_bits(terms)?;
    if num_threads != 1 || window_bits != 9 {
        return Some(window_bits);
    }

    let scalar_bits = scalar_repr_bits::<C>()?;
    let next_window_bits = window_bits.checked_add(1)?;
    let current = estimated_signed_booth_costs(terms, scalar_bits, window_bits, num_threads)
        .map(|(work, _)| work);
    let next = estimated_signed_booth_costs(terms, scalar_bits, next_window_bits, num_threads)
        .map(|(work, _)| work);
    Some(
        if matches!((current, next), (Some(current), Some(next)) if next < current) {
            next_window_bits
        } else {
            window_bits
        },
    )
}

/// Plans a GLV multiscalar multiplication: the window width the GLV ladder
/// should use for `terms` on `num_threads` workers, or `None` when the
/// generic MSM is estimated to be cheaper.
///
/// The generic MSM runs at the default width `w` from
/// [`multiexp_window_bits`]. The GLV ladder is evaluated at both `w` and
/// `w + 1` and takes the cheaper; it is chosen when that beats the generic
/// estimate. GLV has about half as many windows as the generic ladder
/// (127-bit components), so the parallel model's wave count — windows
/// divided by workers, rounded up — is much more sensitive to the width for
/// GLV than for the generic ladder, and comparing the two at a single width
/// chosen for the generic ladder mispredicts: e.g. 8,192 terms on 3 workers
/// puts 13 GLV windows into 5 waves at `w = 10` but 12 windows into exactly
/// 4 waves at `w = 11`. Measured on Apple M4 (asm backend) and EPYC
/// (portable) across 1–16 workers and 2,150–65,536 terms, this choice
/// selected the faster backend in every one of 46 cells, where the
/// single-width comparison was wrong in 8 (M4) and 5 (EPYC) — all cases
/// where GLV was faster by 4–27% but not chosen. The width is only ever
/// widened: a narrower GLV window never measured a meaningful gain.
/// (Under the `orbits` feature, production planning goes through
/// `plan_multiexp`, which extends this Booth-versus-generic comparison with
/// the orbit backend and this becomes test-only — the measured-cell
/// contract for the Booth half of that decision, pinned by
/// `glv_multiexp_plan_matches_measured_cells`. Without the feature this is
/// the production planner, as before.)
fn glv_multiexp_window_bits<C: MsmCurve>(terms: usize, num_threads: usize) -> Option<usize> {
    if terms < MIN_GLV_MULTIEXP_TERMS {
        return None;
    }
    let (generic, _) = estimated_generic_costs::<C>(terms, num_threads)?;
    let (glv, glv_window_bits) = booth_multiexp_estimate::<C>(terms, num_threads)?;
    (glv < generic).then_some(glv_window_bits)
}

/// The generic Signed-Booth MSM's estimated `(work, traffic)` costs at its
/// default width (see [`estimated_signed_booth_costs`]).
fn estimated_generic_costs<C: MsmCurve>(
    terms: usize,
    num_threads: usize,
) -> Option<(usize, usize)> {
    let window_bits = multiexp_window_bits::<C>(terms, num_threads)?;
    let scalar_bits = scalar_repr_bits::<C>()?;
    estimated_signed_booth_costs(terms, scalar_bits, window_bits, num_threads)
}

/// The Signed-Booth GLV backend's cheapest `(work, window width)` over the
/// candidate widths `w` and `w + 1` (see [`glv_multiexp_window_bits`]).
fn booth_multiexp_estimate<C: MsmCurve>(
    terms: usize,
    num_threads: usize,
) -> Option<(usize, usize)> {
    let window_bits = multiexp_window_bits::<C>(terms, num_threads)?;
    let glv_terms = terms.checked_mul(2)?;
    let mut best: Option<(usize, usize)> = None;
    for candidate in window_bits..=window_bits.checked_add(1)? {
        if let Some((work, _)) =
            estimated_signed_booth_costs(glv_terms, GLV_COMPONENT_BITS, candidate, num_threads)
        {
            if best.is_none_or(|(best_work, _)| work < best_work) {
                best = Some((work, candidate));
            }
        }
    }
    best
}

fn current_num_threads() -> usize {
    #[cfg(feature = "multicore")]
    {
        maybe_rayon::current_num_threads()
    }
    #[cfg(not(feature = "multicore"))]
    {
        1
    }
}

/// A small MSM through one shared doubling ladder over jointly recoded GLV
/// scalars. Each point's eight orbit representatives are batch-normalized with
/// every other table, then each nonzero digit contributes one mixed addition.
///
/// This is Strauss' algorithm with the existing joint width-3 Eisenstein NAF
/// in place of two independent integer wNAFs. It retains the ~127-column GLV
/// ladder while averaging fewer additions and one table per input point.
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

#[cfg(feature = "multicore")]
fn strauss_multiexp_leaves<C: MsmCurve>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> C {
    scalars
        .chunks(STRAUSS_LEAF_TERMS)
        .zip(bases.chunks(STRAUSS_LEAF_TERMS))
        .map(|(scalars, bases)| strauss_multiexp::<C>(scalars, bases))
        .fold(C::identity(), |left, right| left + right)
}

fn planned_strauss_multiexp<C: MsmCurve>(
    scalars: &[C::ScalarExt],
    bases: &[C::AffineExt],
    num_threads: usize,
) -> Option<C> {
    #[cfg(feature = "multicore")]
    if (MIN_PARALLEL_STRAUSS_TERMS..=MAX_PARALLEL_STRAUSS_TERMS).contains(&scalars.len())
        && num_threads > 1
    {
        // Late IPA evaluates L and R together. Give this side at most half the
        // pool, so both commitments make progress at once. Each worker retains
        // the measured leaf crossover internally rather than building one wide
        // table for its entire share.
        let worker_jobs = num_threads.div_ceil(2).min(scalars.len());
        if worker_jobs == 1 {
            return Some(strauss_multiexp_leaves::<C>(scalars, bases));
        }
        return Some(
            (0..worker_jobs)
                .into_par_iter()
                .map(|job| {
                    let start = scalars.len() * job / worker_jobs;
                    let end = scalars.len() * (job + 1) / worker_jobs;
                    strauss_multiexp_leaves::<C>(&scalars[start..end], &bases[start..end])
                })
                .reduce(C::identity, |left, right| left + right),
        );
    }

    if scalars.len() <= STRAUSS_LEAF_TERMS {
        return Some(strauss_multiexp::<C>(scalars, bases));
    }

    #[cfg(not(feature = "multicore"))]
    let _ = num_threads;
    None
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
/// This check must remain on the MSM path because it does not construct a
/// [`Decomposed`], whose constructor enforces the same bound. Returning
/// `None` lets the caller use the generic MSM instead of panicking. The
/// rejection boundary is covered by
/// `multiexp_component_guard_returns_none_out_of_bounds`.
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
    component: usize,
    signed_bucket: isize,
}

#[derive(Clone, Copy)]
struct MultiexpBase<F> {
    x: F,
    endo_x: F,
    y: F,
    identity: bool,
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
    let value = if window_start < u128::BITS as usize {
        ((component.magnitude >> window_start) as usize) & (radix - 1)
    } else {
        0
    };
    let overlap = if window_start == 0 {
        0
    } else {
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
/// [`reduce_affine_buckets`] combines this failure-atomic behavior with
/// separate level staging before it retries an exceptional level.
fn batch_invert_and_add<F: Field>(
    additions: &mut [PendingAffineAddition<F>],
    points: &mut [AffinePoint<F>],
) -> Option<()> {
    let Some((first, additions)) = additions.split_first_mut() else {
        return Some(());
    };

    if additions.is_empty() {
        let denominator_inverse = Option::<F>::from(first.denominator.invert())?;
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
    let product_inverse = Option::<F>::from(product.invert())?;
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

/// Collects the nonzero signed points assigned to one Booth window.
fn window_assignments<C>(
    components: &[(SignedMagnitude, SignedMagnitude)],
    bases: &[MultiexpBase<C::Base>],
    window_bits: usize,
    window: usize,
) -> (Vec<WindowAssignment>, Vec<usize>)
where
    C: MsmCurve,
{
    let bucket_count = 1 << (window_bits - 1);
    let mut assignments = Vec::with_capacity(bases.len().saturating_mul(2));
    let mut counts = alloc::vec![0usize; bucket_count];

    for (base_index, ((first, second), base)) in components.iter().zip(bases).enumerate() {
        if base.identity {
            continue;
        }
        for (component_offset, component) in [*first, *second].into_iter().enumerate() {
            let digit = booth_digit(component, window_bits, window);
            if digit.magnitude != 0 {
                let bucket = digit.magnitude - 1;
                let magnitude =
                    isize::try_from(digit.magnitude).expect("Booth digit magnitude fits in isize");
                assignments.push(WindowAssignment {
                    component: base_index * 2 + component_offset,
                    signed_bucket: if digit.negative {
                        -magnitude
                    } else {
                        magnitude
                    },
                });
                counts[bucket] += 1;
            }
        }
    }

    (assignments, counts)
}

/// Reduces every affine bucket through shared Montgomery batch inversions.
///
/// `offsets` partitions `points` into one contiguous range per bucket. At
/// each tree level, all independent additions share one inversion. The normal
/// route uses incomplete chord additions without per-pair coordinate checks;
/// a zero batch product restarts that level with complete handling for
/// identity, doubling, and inverse pairs.
///
/// For valid points on a short-Weierstrass curve over an odd-prime field,
/// equal x-coordinates imply equal or opposite y-coordinates. Thus the only
/// zero-denominator cases omitted by the incomplete chord formula are a
/// doubling and an inverse pair. The latter includes the y = 0 overlap, which
/// is a point of order two and sums with itself to the identity.
///
/// Each level is built in `next_points` and `next_offsets`, while `points` and
/// `offsets` continue to hold its unchanged inputs. The vectors are swapped
/// only after every addition succeeds. An exceptional incomplete level can
/// therefore discard its staging and safely retry the same inputs with the
/// complete formulas.
///
/// # Invariants
///
/// - Every coordinate in `points` represents a valid, non-identity point on
///   the same short-Weierstrass curve with `a = 0`. The doubling numerator is
///   therefore `3x^2`, with no `+ a` term.
/// - `offsets` begins at zero, is non-decreasing, and ends at `points.len()`.
///   Its consecutive entries partition `points` into buckets.
///
/// The callers in this module establish these invariants by sourcing points
/// from a single Pasta curve and skipping identity inputs before building the
/// buckets. The function is generic over the field only for reuse between
/// [`crate::Fp`] and [`crate::Fq`].
fn reduce_affine_buckets<F: Field>(
    points: Vec<AffinePoint<F>>,
    offsets: Vec<usize>,
) -> Option<Vec<Option<AffinePoint<F>>>> {
    reduce_affine_buckets_inner::<F, false>(points, offsets)
}

/// Destructively compacts every incomplete affine reduction level into its
/// input buffer. This is for callers that can reconstruct the original points
/// if a zero denominator declines the incomplete formulas.
#[cfg(test)]
fn reduce_affine_buckets_in_place<F: Field>(
    mut points: Vec<AffinePoint<F>>,
    mut offsets: Vec<usize>,
) -> Option<Vec<Option<AffinePoint<F>>>> {
    debug_assert!(!offsets.is_empty());
    let bucket_count = offsets.len() - 1;
    let mut next_offsets = Vec::with_capacity(offsets.len());
    let mut pending = Vec::with_capacity(points.len() / 2);

    while offsets.windows(2).any(|range| range[1] - range[0] > 1) {
        next_offsets.clear();
        pending.clear();
        next_offsets.push(0);
        let mut output = 0;

        for range in offsets.windows(2) {
            let mut input = range[0];
            while input + 1 < range[1] {
                let left = points[input];
                let right = points[input + 1];
                points[output] = left;
                pending.push(PendingAffineAddition {
                    output,
                    x_sum: left.x + right.x,
                    numerator: right.y - left.y,
                    denominator: right.x - left.x,
                });
                input += 2;
                output += 1;
            }
            if input < range[1] {
                points[output] = points[input];
                output += 1;
            }
            next_offsets.push(output);
        }

        batch_invert_and_add(&mut pending, &mut points)?;
        points.truncate(output);
        core::mem::swap(&mut offsets, &mut next_offsets);
    }

    let mut buckets = alloc::vec![None; bucket_count];
    for (bucket, range) in buckets.iter_mut().zip(offsets.windows(2)) {
        if range[0] != range[1] {
            debug_assert_eq!(range[1] - range[0], 1);
            *bucket = Some(points[range[0]]);
        }
    }
    Some(buckets)
}

fn reduce_affine_buckets_inner<F: Field, const COMPLETE: bool>(
    mut points: Vec<AffinePoint<F>>,
    mut offsets: Vec<usize>,
) -> Option<Vec<Option<AffinePoint<F>>>> {
    debug_assert!(!offsets.is_empty());
    let bucket_count = offsets.len() - 1;
    let mut next_points = Vec::with_capacity((points.len() + bucket_count) / 2);
    let mut next_offsets = Vec::with_capacity(offsets.len());
    let mut pending = Vec::with_capacity(points.len() / 2);

    while offsets.windows(2).any(|range| range[1] - range[0] > 1) {
        next_points.clear();
        next_offsets.clear();
        pending.clear();
        next_offsets.push(0);

        for range in offsets.windows(2) {
            let bucket = &points[range[0]..range[1]];
            for pair in bucket.chunks_exact(2) {
                let left = pair[0];
                let right = pair[1];

                let dx = right.x - left.x;
                let (numerator, denominator) = if COMPLETE && dx.is_zero_vartime() {
                    // Valid curve points with the same x-coordinate have the
                    // same or opposite y-coordinate. Handle both branches
                    // before asking the batch inverter to divide.
                    if !(right.y - left.y).is_zero_vartime() || left.y.is_zero_vartime() {
                        // The points are inverses, or this is a point of order
                        // two. Their sum is the identity, which is omitted.
                        continue;
                    }
                    let x_squared = left.x.square();
                    (x_squared.double() + x_squared, left.y.double())
                } else {
                    (right.y - left.y, dx)
                };

                let output = next_points.len();
                // Preserve the left input in its result slot until the batch
                // inversion completes. This avoids duplicating its coordinates
                // in every pending addition.
                next_points.push(left);
                pending.push(PendingAffineAddition {
                    output,
                    x_sum: left.x + right.x,
                    numerator,
                    denominator,
                });
            }
            if bucket.len() % 2 == 1 {
                next_points.push(bucket[bucket.len() - 1]);
            }
            next_offsets.push(next_points.len());
        }

        if batch_invert_and_add(&mut pending, &mut next_points).is_none() {
            if !COMPLETE {
                // At least one incomplete chord had equal x-coordinates. The
                // failed batch has not overwritten any staged point, and the
                // source vectors are not swapped until success, so `points`
                // and `offsets` are still the exact inputs to this level.
                // Retry them and keep complete formulas enabled for every
                // remaining level; this prevents a later exceptional pair
                // from entering an incomplete formula too.
                return reduce_affine_buckets_inner::<F, true>(points, offsets);
            }
            // Complete formulas cannot produce a zero denominator under the
            // invariants above. If the guard nevertheless fails, propagate
            // `None` so the verifier caller uses the generic MSM instead of a
            // potentially incomplete result.
            return None;
        }

        core::mem::swap(&mut points, &mut next_points);
        core::mem::swap(&mut offsets, &mut next_offsets);
    }

    let mut buckets = alloc::vec![None; bucket_count];
    for (bucket, range) in buckets.iter_mut().zip(offsets.windows(2)) {
        if range[0] != range[1] {
            debug_assert_eq!(range[1] - range[0], 1);
            *bucket = Some(points[range[0]]);
        }
    }
    Some(buckets)
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

fn fill_window<C: MsmCurve>(
    components: &[(SignedMagnitude, SignedMagnitude)],
    bases: &[MultiexpBase<C::Base>],
    window_bits: usize,
    window: usize,
) -> Option<Vec<Option<AffinePoint<C::Base>>>> {
    let bucket_count = 1 << (window_bits - 1);
    let (assignments, counts) = window_assignments::<C>(components, bases, window_bits, window);

    let mut offsets = Vec::with_capacity(bucket_count + 1);
    offsets.push(0);
    for count in counts {
        offsets.push(offsets.last().copied().unwrap() + count);
    }

    let mut positions = offsets[..bucket_count].to_vec();
    let mut points = alloc::vec![
        AffinePoint {
            x: C::Base::ZERO,
            y: C::Base::ZERO,
        };
        *offsets.last().unwrap()
    ];
    for assignment in assignments {
        let base_index = assignment.component / 2;
        let base = bases[base_index];
        let x = if assignment.component & 1 == 0 {
            base.x
        } else {
            base.endo_x
        };
        let bucket = assignment.bucket();
        let position = positions[bucket];
        points[position] = AffinePoint {
            x,
            y: if assignment.negative() {
                -base.y
            } else {
                base.y
            },
        };
        positions[bucket] += 1;
    }

    reduce_affine_buckets(points, offsets)
}

fn multiexp_serial<C: MsmCurve>(
    components: &[(SignedMagnitude, SignedMagnitude)],
    bases: &[MultiexpBase<C::Base>],
    window_bits: usize,
) -> Option<C> {
    let window_count = GLV_COMPONENT_BITS / window_bits + 1;
    let mut acc = C::identity();

    for window in (0..window_count).rev() {
        if window + 1 != window_count {
            for _ in 0..window_bits {
                acc = acc.double();
            }
        }

        let buckets = fill_window::<C>(components, bases, window_bits, window)?;
        acc += sum_buckets::<C>(&buckets);
    }
    Some(acc)
}

/// Evaluates each Signed-Booth window independently through Rayon, then
/// combines the ordered sums with one Horner fold. This keeps the expensive
/// window reductions parallel without duplicating the shift chain.
#[cfg(feature = "multicore")]
fn parallel_windows_sum<C: MsmCurve>(
    windows: usize,
    window_bits: usize,
    window_sum: impl Fn(usize) -> Option<C> + Sync,
) -> Option<C> {
    let window_sums: Option<Vec<C>> = (0..windows).into_par_iter().map(&window_sum).collect();
    let window_sums = window_sums?;
    let mut acc = C::identity();
    for (window, sum) in window_sums.into_iter().enumerate().rev() {
        if window + 1 != windows {
            for _ in 0..window_bits {
                acc = acc.double();
            }
        }
        acc += sum;
    }
    Some(acc)
}

#[cfg(feature = "multicore")]
fn multiexp_parallel<C: MsmCurve>(
    components: &[(SignedMagnitude, SignedMagnitude)],
    bases: &[MultiexpBase<C::Base>],
    window_bits: usize,
) -> Option<C> {
    let window_count = GLV_COMPONENT_BITS / window_bits + 1;
    parallel_windows_sum::<C>(window_count, window_bits, |window| {
        let buckets = fill_window::<C>(components, bases, window_bits, window)?;
        Some(sum_buckets::<C>(&buckets))
    })
}

fn multiexp<C: MsmCurve>(
    components: &[(SignedMagnitude, SignedMagnitude)],
    bases: &[MultiexpBase<C::Base>],
    window_bits: usize,
    num_threads: usize,
) -> Option<C> {
    debug_assert_eq!(components.len(), bases.len());

    #[cfg(not(feature = "multicore"))]
    let _ = num_threads;
    #[cfg(feature = "multicore")]
    let acc = if num_threads > 1 {
        multiexp_parallel::<C>(components, bases, window_bits)
    } else {
        multiexp_serial::<C>(components, bases, window_bits)
    };
    #[cfg(not(feature = "multicore"))]
    let acc = multiexp_serial::<C>(components, bases, window_bits);
    acc
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

/// Attempts a GLV bucket multiscalar multiplication for a large Pasta MSM,
/// through whichever backend `plan_multiexp` selects (with the `orbits`
/// feature disabled, the Signed-Booth backend at
/// `glv_multiexp_window_bits`'s width).
///
/// Returns `None` when the cost model selects the generic MSM or when an
/// internal arithmetic guard fails, allowing verifier callers to use the
/// generic implementation instead of panicking.
pub(crate) fn try_multiexp<C: MsmCurve>(
    scalars: &[C::ScalarExt],
    bases: &[C::AffineExt],
) -> Option<C> {
    assert_eq!(scalars.len(), bases.len());
    let num_threads = current_num_threads();
    if let Some(result) = planned_strauss_multiexp::<C>(scalars, bases, num_threads) {
        return Some(result);
    }

    let window_bits = glv_multiexp_window_bits::<C>(scalars.len(), num_threads)?;
    let components = scalars
        .iter()
        .map(decompose::<C>)
        .map(checked_signed_magnitudes)
        .collect::<Option<Vec<_>>>()?;
    let bases = multiexp_bases::<C>(bases);
    multiexp(&components, &bases, window_bits, num_threads)
}
