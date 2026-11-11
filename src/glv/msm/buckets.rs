//! The bucket method over small-scalar terms.
//!
//! For one window, `sum_i d_i Q_i = sum_b b * S_b`, where `S_b` sums the terms
//! whose digit is `+-b`: the terms are pushed forward along their digits into
//! buckets, each bucket is reduced by batched affine additions, and the
//! weighted sum is taken by summation by parts. Horner's rule at
//! `2^window_bits` combines the windows.

use alloc::vec::Vec;
use core::ops::Range;
#[cfg(feature = "multicore")]
use maybe_rayon::prelude::*;

use super::MsmCurve;
use super::affine::{AffinePoint, reduce_affine_buckets};
use super::lift::{GLV_COMPONENT_BITS, SignedMagnitude};
use super::recode::booth_digit;
use crate::arithmetic::VartimeField;

#[derive(Clone, Copy)]
pub(super) struct WindowAssignment {
    pub(super) term: usize,
    pub(super) signed_bucket: isize,
}

/// The MSM the bucket method evaluates: the sum over terms of
/// `scalar * point`, each scalar an integer of magnitude below
/// `2^GLV_COMPONENT_BITS`. It knows nothing about GLV; [`GlvTerms`](super::lift::GlvTerms) is the
/// implementation that [`execute`](super::execute) uses.
pub(super) trait SmallScalarTerms<F>: Sync {
    /// The number of terms.
    fn len(&self) -> usize;
    /// The scalar of term `term`.
    fn scalar(&self, term: usize) -> SignedMagnitude;
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
pub(super) fn window_assignments<F, T: SmallScalarTerms<F>>(
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

pub(super) fn sum_buckets<C: MsmCurve>(buckets: &[Option<AffinePoint<C::Base>>]) -> C {
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
pub(super) fn fill_window<F: VartimeField, T: SmallScalarTerms<F>>(
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
pub(super) fn fill_window_in_blocks<F: VartimeField, T: SmallScalarTerms<F>>(
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
pub(super) fn fill_block<F: VartimeField, T: SmallScalarTerms<F>>(
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
pub(super) fn window_sum<C: MsmCurve, T: SmallScalarTerms<C::Base>>(
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
pub(super) fn horner<C: MsmCurve>(
    top_down: impl Iterator<Item = Option<C>>,
    window_bits: usize,
) -> Option<C> {
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
pub(super) fn bucket_multiexp<C: MsmCurve, T: SmallScalarTerms<C::Base>>(
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
