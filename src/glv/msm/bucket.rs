//! Large sums: signed-Booth buckets per window, reduced with batched affine
//! additions, then combined across windows.

mod affine;
mod cache;

use alloc::vec::Vec;
use ff::{Field, WithSmallOrderMulGroup};
use group::CurveAffine as _;
#[cfg(feature = "multicore")]
use maybe_rayon::prelude::*;

use super::booth::{GLV_COMPONENT_BITS, SignedMagnitude, booth_digit};
use crate::arithmetic::CurveAffine;
use crate::glv::{GlvParams, affine_from_xy_unchecked, private::AffineBase};
#[cfg(feature = "multicore")]
use affine::PendingAffineAddition;
use affine::{AffinePoint, ReduceScratch, reduce_affine_buckets_reusing};
pub(crate) use cache::CachedField;
use cache::Checkout;

#[cfg(test)]
mod tests;

/// Reads affine coordinates, using the identity's zero representation.
fn affine_xy<C: GlvParams>(point: &C::AffineExt) -> (AffineBase<C>, AffineBase<C>) {
    Option::<crate::arithmetic::Coordinates<C::AffineExt>>::from(point.coordinates())
        .map(|p| (*p.x(), *p.y()))
        .unwrap_or((AffineBase::<C>::ZERO, AffineBase::<C>::ZERO))
}

#[derive(Clone, Copy)]
struct WindowAssignment {
    component: usize,
    signed_bucket: isize,
}

#[derive(Clone, Copy)]
pub(super) struct MultiexpBase<F> {
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

/// Collects the nonzero signed points assigned to one Booth window.
fn window_assignments<C>(
    components: &[(SignedMagnitude, SignedMagnitude)],
    bases: &[MultiexpBase<AffineBase<C>>],
    window_bits: usize,
    window: usize,
    assignments: &mut Vec<WindowAssignment>,
    counts: &mut Vec<usize>,
) where
    C: GlvParams,
{
    let bucket_count = 1 << (window_bits - 1);
    assignments.clear();
    assignments.reserve(bases.len().saturating_mul(2));
    counts.clear();
    counts.resize(bucket_count, 0);

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
}

fn sum_buckets<C: GlvParams>(buckets: &[Option<AffinePoint<AffineBase<C>>>]) -> C {
    let mut running = C::identity();
    let mut sum = C::identity();
    for bucket in buckets.iter().rev() {
        if let Some(point) = bucket {
            running += affine_from_xy_unchecked::<C>(point.x, point.y);
        }
        sum += running;
    }
    sum
}

/// Smallest number of terms a window reduces at a time.
const MIN_WINDOW_BLOCK_TERMS: usize = 1 << 14;

/// Terms per bucket in a window block. Each block after the first carries one
/// point per nonempty bucket, so this bounds the extra additions to about
/// `1 / (2 * WINDOW_BLOCK_TERMS_PER_BUCKET)` of the block's points.
const WINDOW_BLOCK_TERMS_PER_BUCKET: usize = 8;

/// Buffers for filling and reducing one window's buckets, reused across the
/// windows a task evaluates (see [`ReduceScratch`]) and, with `multicore`,
/// across calls on the same thread (see [`cache`]).
pub(crate) struct WindowScratch<F> {
    assignments: Vec<WindowAssignment>,
    counts: Vec<usize>,
    offsets: Vec<usize>,
    positions: Vec<usize>,
    points: Vec<AffinePoint<F>>,
    reduce: ReduceScratch<F>,
}

impl<F> Default for WindowScratch<F> {
    fn default() -> Self {
        Self {
            assignments: Vec::new(),
            counts: Vec::new(),
            offsets: Vec::new(),
            positions: Vec::new(),
            points: Vec::new(),
            reduce: ReduceScratch::default(),
        }
    }
}

#[cfg(feature = "multicore")]
impl<F> WindowScratch<F> {
    fn capacity_bytes(&self) -> usize {
        use core::mem::size_of;
        self.assignments.capacity() * size_of::<WindowAssignment>()
            + (self.counts.capacity()
                + self.offsets.capacity()
                + self.positions.capacity()
                + self.reduce.next_offsets.capacity())
                * size_of::<usize>()
            + self.points.capacity() * size_of::<AffinePoint<F>>()
            + self.reduce.buckets.capacity() * size_of::<Option<AffinePoint<F>>>()
            + self.reduce.pending.capacity() * size_of::<PendingAffineAddition<F>>()
    }
}

/// Fills and reduces one window's buckets.
///
/// Scratch is bounded by `max(2^14, 8 * buckets)` terms for each window, rather
/// than by the whole input. Wider windows increase the bucket count and thus
/// the block bound; parallel windows multiply this bound by the jobs in flight.
fn fill_window<'a, C: GlvParams>(
    components: &[(SignedMagnitude, SignedMagnitude)],
    bases: &[MultiexpBase<AffineBase<C>>],
    window_bits: usize,
    window: usize,
    scratch: &'a mut WindowScratch<AffineBase<C>>,
) -> Option<&'a [Option<AffinePoint<AffineBase<C>>>]> {
    let bucket_count = 1usize << (window_bits - 1);
    let block_terms =
        MIN_WINDOW_BLOCK_TERMS.max(bucket_count.saturating_mul(WINDOW_BLOCK_TERMS_PER_BUCKET));
    fill_window_in_blocks::<C>(components, bases, window_bits, window, block_terms, scratch)
}

/// Reduces a window block by block. Each block's bucket results enter the next
/// block as the first point of the same buckets, so the batched affine reducer
/// also merges the blocks, including when a carried point doubles or cancels.
/// The result is the reused `scratch.reduce.buckets`.
fn fill_window_in_blocks<'a, C: GlvParams>(
    components: &[(SignedMagnitude, SignedMagnitude)],
    bases: &[MultiexpBase<AffineBase<C>>],
    window_bits: usize,
    window: usize,
    block_terms: usize,
    scratch: &'a mut WindowScratch<AffineBase<C>>,
) -> Option<&'a [Option<AffinePoint<AffineBase<C>>>]> {
    debug_assert!(block_terms > 0);
    // Empty buckets are both the result of an empty window and the carry into
    // the first block, which therefore needs no special case.
    scratch.reduce.buckets.clear();
    scratch.reduce.buckets.resize(1 << (window_bits - 1), None);
    for (components, bases) in components
        .chunks(block_terms)
        .zip(bases.chunks(block_terms))
    {
        fill_block::<C>(components, bases, window_bits, window, scratch)?;
    }
    Some(&scratch.reduce.buckets)
}

/// Fills one block's buckets, each led by its point carried in
/// `scratch.reduce.buckets` from the previous block, and reduces them into
/// `scratch.reduce.buckets`.
fn fill_block<C: GlvParams>(
    components: &[(SignedMagnitude, SignedMagnitude)],
    bases: &[MultiexpBase<AffineBase<C>>],
    window_bits: usize,
    window: usize,
    scratch: &mut WindowScratch<AffineBase<C>>,
) -> Option<()> {
    let bucket_count = 1 << (window_bits - 1);
    let WindowScratch {
        assignments,
        counts,
        offsets,
        positions,
        points,
        reduce,
    } = scratch;
    window_assignments::<C>(components, bases, window_bits, window, assignments, counts);
    for (count, point) in counts.iter_mut().zip(&reduce.buckets) {
        *count += usize::from(point.is_some());
    }

    offsets.clear();
    offsets.reserve(bucket_count + 1);
    offsets.push(0);
    for &count in counts.iter() {
        offsets.push(offsets.last().copied().unwrap() + count);
    }

    positions.clear();
    positions.extend_from_slice(&offsets[..bucket_count]);
    points.clear();
    points.resize(
        *offsets.last().unwrap(),
        AffinePoint {
            x: AffineBase::<C>::ZERO,
            y: AffineBase::<C>::ZERO,
        },
    );
    for (bucket, point) in reduce.buckets.iter().enumerate() {
        if let Some(point) = point {
            points[positions[bucket]] = *point;
            positions[bucket] += 1;
        }
    }
    for &assignment in assignments.iter() {
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

    reduce_affine_buckets_reusing::<AffineBase<C>, false>(points, offsets, reduce)
}

/// Combines the window sums, highest window first, by Horner's rule: one shift
/// chain of `window_bits` doublings between windows, shared by all of them.
/// Declines as soon as any window does.
fn horner<C: GlvParams>(
    window_bits: usize,
    sums_from_highest: impl IntoIterator<Item = Option<C>>,
) -> Option<C> {
    let mut acc = C::identity();
    for (index, sum) in sums_from_highest.into_iter().enumerate() {
        if index != 0 {
            for _ in 0..window_bits {
                acc = acc.double();
            }
        }
        acc += sum?;
    }
    Some(acc)
}

fn multiexp_serial<C: GlvParams>(
    components: &[(SignedMagnitude, SignedMagnitude)],
    bases: &[MultiexpBase<AffineBase<C>>],
    window_bits: usize,
) -> Option<C>
where
    AffineBase<C>: CachedField,
{
    let window_count = GLV_COMPONENT_BITS / window_bits + 1;
    let mut scratch = Checkout::<AffineBase<C>>::take();
    horner::<C>(
        window_bits,
        (0..window_count).rev().map(|window| {
            let buckets = fill_window::<C>(components, bases, window_bits, window, &mut scratch)?;
            Some(sum_buckets::<C>(buckets))
        }),
    )
}

/// Evaluates each signed-Booth window as its own Rayon task, then combines the
/// ordered sums with one Horner fold. This keeps the expensive window
/// reductions parallel, with work stealing, without duplicating the shift
/// chain. Each task checks scratch out for its window: a thread returns it
/// before starting its next window, so consecutive windows on a thread reuse
/// the same buffers.
#[cfg(feature = "multicore")]
fn multiexp_parallel<C: GlvParams>(
    components: &[(SignedMagnitude, SignedMagnitude)],
    bases: &[MultiexpBase<AffineBase<C>>],
    window_bits: usize,
) -> Option<C>
where
    AffineBase<C>: CachedField,
{
    let window_count = GLV_COMPONENT_BITS / window_bits + 1;
    let window_sums: Option<Vec<C>> = (0..window_count)
        .into_par_iter()
        .map(|window| {
            let mut scratch = Checkout::<AffineBase<C>>::take();
            let buckets = fill_window::<C>(components, bases, window_bits, window, &mut scratch)?;
            Some(sum_buckets::<C>(buckets))
        })
        .collect();
    horner::<C>(window_bits, window_sums?.into_iter().rev().map(Some))
}

pub(super) fn multiexp<C: GlvParams>(
    components: &[(SignedMagnitude, SignedMagnitude)],
    bases: &[MultiexpBase<AffineBase<C>>],
    window_bits: usize,
    num_threads: usize,
) -> Option<C>
where
    AffineBase<C>: CachedField,
{
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

pub(super) fn multiexp_bases<C: GlvParams>(
    bases: &[C::AffineExt],
) -> Vec<MultiexpBase<AffineBase<C>>> {
    bases
        .iter()
        .map(|base| {
            let (x, y) = affine_xy::<C>(base);
            MultiexpBase {
                x,
                endo_x: x * AffineBase::<C>::ZETA,
                y,
                identity: bool::from(base.is_identity()),
            }
        })
        .collect()
}
