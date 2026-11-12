//! The bucket method over small-scalar terms.
//!
//! For one window, `sum_i d_i Q_i = sum_b w_b * S_b`, where `S_b` sums the
//! terms whose digit falls in bucket `b` and `w_b` is its weight: the terms are pushed forward along their digits into
//! buckets, each bucket is reduced by batched affine additions, and the
//! weighted sum is taken by summation by parts. Horner's rule at
//! `2^window_bits` combines the windows.

use alloc::vec::Vec;
use core::ops::Range;
#[cfg(feature = "multicore")]
use maybe_rayon::prelude::*;

use super::MsmCurve;
use super::affine::{AffinePoint, reduce_affine_buckets};
use super::recode::Recoding;
use crate::arithmetic::VartimeField;

#[derive(Clone, Copy)]
pub(super) struct WindowAssignment {
    pub(super) term: usize,
    pub(super) signed_bucket: isize,
}

/// The MSM the bucket method evaluates: the sum over terms of
/// `scalar * point`, each scalar below `2^SCALAR_BITS` and recoded into
/// digits by [`Self::Recoding`]. It knows nothing about GLV;
/// [`GlvTerms`](super::lift::GlvTerms) is the implementation that
/// [`execute`](super::execute) uses.
pub(super) trait SmallScalarTerms<F>: Sync {
    /// The digit system the scalars are recoded with.
    type Recoding: Recoding;
    /// Every scalar is below `2^SCALAR_BITS` in magnitude.
    const SCALAR_BITS: usize;
    /// The number of terms.
    fn len(&self) -> usize;
    /// The scalar of term `term`.
    fn scalar(&self, term: usize) -> <Self::Recoding as Recoding>::Scalar;
    /// The point of term `term`, or `None` for the identity.
    fn point(&self, term: usize) -> Option<AffinePoint<F>>;
}

impl WindowAssignment {
    fn bucket(self) -> usize {
        self.signed_bucket.unsigned_abs() - 1
    }

    fn negative(self) -> bool {
        self.signed_bucket < 0
    }
}

/// Collects the terms in `range` with a nonzero digit in one Booth window,
/// each with its signed bucket, and counts the terms per bucket. This is the
/// pushforward of the window's terms along their digits.
pub(super) fn window_assignments<F, T>(
    terms: &T,
    range: Range<usize>,
    window_bits: usize,
    window: usize,
) -> (Vec<WindowAssignment>, Vec<usize>)
where
    T: SmallScalarTerms<F>,
{
    let bucket_count = T::Recoding::bucket_count(window_bits);
    let mut assignments = Vec::with_capacity(range.len());
    let mut counts = alloc::vec![0usize; bucket_count];

    for term in range {
        if terms.point(term).is_none() {
            continue;
        }
        if let Some(digit) = T::Recoding::digit(terms.scalar(term), window_bits, window) {
            let magnitude = isize::try_from(digit.bucket + 1).expect("bucket index fits in isize");
            assignments.push(WindowAssignment {
                term,
                signed_bucket: if digit.negate { -magnitude } else { magnitude },
            });
            counts[digit.bucket] += 1;
        }
    }

    (assignments, counts)
}

/// Smallest number of terms a window reduces at a time.
pub(super) const MIN_WINDOW_BLOCK_TERMS: usize = 1 << 15;

/// Terms per bucket in a window block. Each block after the first carries one
/// point per nonempty bucket, so this bounds the extra additions to about
/// `1 / WINDOW_BLOCK_TERMS_PER_BUCKET` of the block's points.
pub(super) const WINDOW_BLOCK_TERMS_PER_BUCKET: usize = 16;

/// Fills and reduces one window's buckets.
///
/// The terms are processed in blocks, so the window's scratch memory (points,
/// pending additions and assignments) is bounded by the block size instead of
/// growing with the number of terms. With parallel windows this bound is what
/// keeps memory from scaling with the number of windows in flight.
pub(super) fn fill_window<F, T>(
    terms: &T,
    window_bits: usize,
    window: usize,
) -> Option<Vec<Option<AffinePoint<F>>>>
where
    F: VartimeField,
    T: SmallScalarTerms<F>,
{
    let bucket_count = T::Recoding::bucket_count(window_bits);
    let block_terms =
        MIN_WINDOW_BLOCK_TERMS.max(bucket_count.saturating_mul(WINDOW_BLOCK_TERMS_PER_BUCKET));
    fill_window_in_blocks(terms, window_bits, window, block_terms)
}

/// Reduces a window block by block. Each block's bucket results enter the next
/// block as the first point of the same buckets, so the batched affine reducer
/// also merges the blocks, including when a carried point doubles or cancels.
/// This is the monoid law `sum(A + B) = sum({sum(A)} + B)` for multisets.
pub(super) fn fill_window_in_blocks<F, T>(
    terms: &T,
    window_bits: usize,
    window: usize,
    block_terms: usize,
) -> Option<Vec<Option<AffinePoint<F>>>>
where
    F: VartimeField,
    T: SmallScalarTerms<F>,
{
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
    Some(buckets.unwrap_or_else(|| alloc::vec![None; T::Recoding::bucket_count(window_bits)]))
}

/// Fills one block's buckets, each led by its carried point from the previous
/// block, and reduces them.
pub(super) fn fill_block<F, T>(
    terms: &T,
    range: Range<usize>,
    window_bits: usize,
    window: usize,
    carry: Option<&[Option<AffinePoint<F>>]>,
) -> Option<Vec<Option<AffinePoint<F>>>>
where
    F: VartimeField,
    T: SmallScalarTerms<F>,
{
    let bucket_count = T::Recoding::bucket_count(window_bits);
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
/// computed from the window's buckets by [`Recoding::sum_buckets`].
pub(super) fn window_sum<C, T>(terms: &T, window_bits: usize, window: usize) -> Option<C>
where
    C: MsmCurve,
    T: SmallScalarTerms<C::Base>,
{
    let buckets = fill_window(terms, window_bits, window)?;
    Some(T::Recoding::sum_buckets::<C>(&buckets))
}

/// Horner's rule at `X = 2^window_bits`: given the window sums from the top
/// window down, returns `sum_j W_j * 2^(window_bits * j)`. The top window is
/// not shifted, so the fold starts from it rather than from the identity.
pub(super) fn horner<C, I>(top_down: I, window_bits: usize) -> Option<C>
where
    C: MsmCurve,
    I: Iterator<Item = Option<C>>,
{
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

/// The bucket method over `terms` with `window_bits`-bit windows of their
/// recoding. With `parallel`, the windows' sums are independent jobs (on
/// worker threads with `multicore`, one after another without it), combined
/// by one Horner fold; otherwise each window is computed as the fold reaches
/// it, so only one window's buckets exist at a time.
pub(super) fn bucket_multiexp<C, T, J>(
    terms: &T,
    window_bits: usize,
    parallel: bool,
    on_job: &J,
) -> Option<C>
where
    C: MsmCurve,
    T: SmallScalarTerms<C::Base>,
    J: Fn() + Sync,
{
    let windows = T::Recoding::window_count(T::SCALAR_BITS, window_bits);
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
