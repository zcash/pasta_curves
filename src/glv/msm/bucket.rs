//! Large sums: signed-Booth buckets per window, reduced with batched affine
//! additions, then combined across windows.
//!
//! # Components
//!
//! The MSM computes $S = \sum_i k_i P_i$. Each term is split as in
//! [`crate::glv`], into component $c = 2i$, $(k_{i,1}, P_i)$, and component
//! $c = 2i + 1$, $(k_{i,2}, \phi(P_i))$; write $(m_c, Q_c)$ for them.
//!
//! # One window
//!
//! With the signed-Booth digits $d_{c,k}$ of [`super::booth`], $m_c = \sum_k
//! d_{c,k} 2^{kw}$, hence $S = \sum_k 2^{kw} W_k$ with
//! $W_k = \sum_c d_{c,k} Q_c$. The bucket method, usually attributed to
//! Pippenger (N. Pippenger, "On the evaluation of powers and monomials", SIAM
//! J. Comput. 9(2), 1980), computes $W_k$ without any scalar multiplication:
//!
//! 1. **Assign.** Put $\operatorname{sign}(d_{c,k}) Q_c$ into bucket
//!    $|d_{c,k}| - 1$, for every nonzero digit. Negating a point negates $y$,
//!    so signed digits need only [`bucket_count`]$(w) = 2^{w-1}$ buckets.
//! 2. **Reduce.** Sum each bucket to one point $B_j$, all buckets at once
//!    ([`affine`]). Then $W_k = \sum_j (j + 1) B_j$.
//! 3. **Running sum** ([`sum_buckets`]). From the highest bucket down, keep
//!    $R = \sum_{j' \ge j} B_{j'}$ and add $R$ into the total at each step: each
//!    $B_j$ is added $j + 1$ times, with at most $2 \cdot 2^{w-1}$ additions.
//!
//! # All windows
//!
//! [`horner`] combines the window sums from the highest window down with $w$
//! doublings between them: $S = (\cdots(W_K 2^w + W_{K-1}) 2^w + \cdots) + W_0$.
//! Serially, one [`WindowReducer`] serves every window in turn; with
//! `multicore`, each window is its own Rayon task.
//!
//! # Memory
//!
//! A window is filled and reduced in blocks of at most
//! `max(2^14, 8 * buckets)` components; each block's bucket sums enter the
//! next block as the first point of the same buckets. The buffers live in a
//! [`WindowReducer`] that is reused across windows, and with `multicore`
//! across calls on the same thread ([`cache`]).
//!
//! # Reading guide
//!
//! [`multiexp`] is the entry point, with bases prepared by [`multiexp_bases`].
//! It runs the windows with [`multiexp_serial`] or, with `multicore`,
//! `multiexp_parallel`, and combines them with [`horner`]. A [`WindowReducer`]
//! computes each window: [`WindowReducer::fill_window`] splits it into blocks
//! ([`MIN_WINDOW_BLOCK_TERMS`], [`WINDOW_BLOCK_TERMS_PER_BUCKET`]),
//! [`WindowReducer::fill_block`] assigns the digits to buckets
//! ([`window_assignments`]) and reduces them ([`affine`]), and [`sum_buckets`]
//! turns the buckets into the window sum. Reducers come from [`cache`].

mod affine;
mod cache;

use alloc::vec::Vec;
use group::CurveAffine as _;
#[cfg(feature = "multicore")]
use maybe_rayon::prelude::*;

use crate::arithmetic::VartimeField;
use crate::glv::booth::{SignedMagnitude, booth_digit, window_count};
use crate::glv::{GlvParams, affine_from_xy_unchecked, affine_xy, phi_x, private::AffineBase};
pub(in crate::glv) use affine::{AffinePoint, reduce_affine_buckets};
use affine::{ReduceScratch, reduce_affine_buckets_reusing};
pub(crate) use cache::CachedField;
use cache::Checkout;

#[cfg(test)]
mod tests;

/// One nonzero digit of a window: component `component` goes into bucket
/// `|signed_bucket| - 1`, negated when `signed_bucket < 0`.
#[derive(Clone, Copy)]
struct WindowAssignment {
    component: usize,
    signed_bucket: isize,
}

/// A base $P$ prepared for the MSM: its coordinates, and `endo_x`, the $x$ of
/// $\phi(P)$ (see [`crate::glv`]). When `identity` is set, the coordinates are
/// meaningless and the base is skipped.
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

/// The number of buckets of a window of width `window_bits`: one per nonzero
/// Booth digit magnitude, which ranges over `1..=2^(window_bits - 1)`.
fn bucket_count(window_bits: usize) -> usize {
    1 << (window_bits - 1)
}

/// Collects the nonzero signed points assigned to one Booth window.
///
/// Fills `assignments` with one entry per nonzero digit of window `window`,
/// skipping identity bases, and `counts[j]` with the number of entries in
/// bucket `j`.
fn window_assignments<F>(
    components: &[(SignedMagnitude, SignedMagnitude)],
    bases: &[MultiexpBase<F>],
    window_bits: usize,
    window: usize,
    assignments: &mut Vec<WindowAssignment>,
    counts: &mut Vec<usize>,
) {
    let bucket_count = bucket_count(window_bits);
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

/// Returns $\sum_j (j + 1) B_j$, where $B_j$ is bucket `j` (the identity for
/// `None`): the window sum, since bucket `j` holds the digits of magnitude
/// `j + 1`. Computed with a running sum from the highest bucket down.
fn sum_buckets<C>(buckets: &[Option<AffinePoint<AffineBase<C>>>]) -> C
where
    C: GlvParams,
{
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

/// The curve point a reduced bucket holds, the identity for an empty bucket.
#[cfg(test)]
fn bucket_point<C>(bucket: Option<AffinePoint<AffineBase<C>>>) -> C
where
    C: GlvParams,
{
    match bucket {
        Some(point) => C::from(affine_from_xy_unchecked::<C>(point.x, point.y)),
        None => C::identity(),
    }
}

/// Smallest number of terms a window reduces at a time. It bounds a window's
/// scratch, which would otherwise grow with the input and be multiplied by the
/// windows in flight, at a small measured cost in time for MSMs that need
/// several blocks, as `benches/msm.rs` does at 65,536 terms.
const MIN_WINDOW_BLOCK_TERMS: usize = 1 << 14;

/// Terms per bucket in a window block. Each block after the first carries one
/// point per nonempty bucket, so this bounds the extra additions to about
/// `1 / (2 * WINDOW_BLOCK_TERMS_PER_BUCKET)` of the block's points.
const WINDOW_BLOCK_TERMS_PER_BUCKET: usize = 8;

/// The buffers for filling and reducing one window's buckets, and the window
/// algorithm that uses them. One reducer serves the windows a task evaluates
/// in turn (see [`ReduceScratch`]) and, with `multicore`, the calls on the same
/// thread (see [`cache`]).
pub(crate) struct WindowReducer<F> {
    /// The nonzero digits of the current block.
    assignments: Vec<WindowAssignment>,
    /// The number of points in each bucket of the current block.
    counts: Vec<usize>,
    /// The bucket boundaries in `points`.
    offsets: Vec<usize>,
    /// Each bucket's next free slot in `points` while the block is filled.
    positions: Vec<usize>,
    /// Every bucket's points, contiguous.
    points: Vec<AffinePoint<F>>,
    /// The reduction's buffers, including the output buckets.
    reduce: ReduceScratch<F>,
}

impl<F> Default for WindowReducer<F> {
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
impl<F> WindowReducer<F> {
    /// The heap memory these buffers keep between windows. The exhaustive
    /// destructuring makes a new buffer fail to compile until it is counted.
    fn capacity_bytes(&self) -> usize {
        fn bytes<T>(buffer: &Vec<T>) -> usize {
            buffer.capacity() * core::mem::size_of::<T>()
        }
        let Self {
            assignments,
            counts,
            offsets,
            positions,
            points,
            reduce:
                ReduceScratch {
                    next_offsets,
                    pending,
                    buckets,
                },
        } = self;
        bytes(assignments)
            + bytes(counts)
            + bytes(offsets)
            + bytes(positions)
            + bytes(points)
            + bytes(next_offsets)
            + bytes(pending)
            + bytes(buckets)
    }
}

impl<F> WindowReducer<F>
where
    F: VartimeField,
{
    /// Fills and reduces one window's buckets.
    ///
    /// Returns, on `Some`, the buckets $B_j$ of window `window`: $B_j$ is the
    /// sum of the components' signed points with digit magnitude `j + 1`
    /// (`None` when that sum is the identity), so that [`sum_buckets`] gives
    /// $W_k$. `None` means the reduction declined (see
    /// [`affine::reduce_affine_buckets_reusing`]); it does not happen for valid
    /// inputs.
    ///
    /// Scratch is bounded by `max(2^14, 8 * buckets)` terms for each window, rather
    /// than by the whole input. Wider windows increase the bucket count and thus
    /// the block bound; parallel windows multiply this bound by the jobs in flight.
    fn fill_window(
        &mut self,
        components: &[(SignedMagnitude, SignedMagnitude)],
        bases: &[MultiexpBase<F>],
        window_bits: usize,
        window: usize,
    ) -> Option<&[Option<AffinePoint<F>>]> {
        let bucket_count = bucket_count(window_bits);
        let block_terms =
            MIN_WINDOW_BLOCK_TERMS.max(bucket_count.saturating_mul(WINDOW_BLOCK_TERMS_PER_BUCKET));
        self.fill_window_in_blocks(components, bases, window_bits, window, block_terms)
    }

    /// Reduces a window block by block. Each block's bucket results enter the next
    /// block as the first point of the same buckets, so the batched affine reducer
    /// also merges the blocks, including when a carried point doubles or cancels.
    /// The result is the reused `self.reduce.buckets`.
    fn fill_window_in_blocks(
        &mut self,
        components: &[(SignedMagnitude, SignedMagnitude)],
        bases: &[MultiexpBase<F>],
        window_bits: usize,
        window: usize,
        block_terms: usize,
    ) -> Option<&[Option<AffinePoint<F>>]> {
        debug_assert!(block_terms > 0);
        // Empty buckets are both the result of an empty window and the carry into
        // the first block, which therefore needs no special case.
        self.reduce.buckets.clear();
        self.reduce.buckets.resize(bucket_count(window_bits), None);
        for (components, bases) in components
            .chunks(block_terms)
            .zip(bases.chunks(block_terms))
        {
            self.fill_block(components, bases, window_bits, window)?;
        }
        Some(&self.reduce.buckets)
    }

    /// Fills one block's buckets, each led by its point carried in
    /// `self.reduce.buckets` from the previous block, and reduces them into
    /// `self.reduce.buckets`.
    fn fill_block(
        &mut self,
        components: &[(SignedMagnitude, SignedMagnitude)],
        bases: &[MultiexpBase<F>],
        window_bits: usize,
        window: usize,
    ) -> Option<()> {
        let bucket_count = bucket_count(window_bits);
        let Self {
            assignments,
            counts,
            offsets,
            positions,
            points,
            reduce,
        } = self;
        window_assignments(components, bases, window_bits, window, assignments, counts);
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
        // Placeholders, not curve points: each is overwritten exactly once below.
        points.clear();
        points.resize(
            *offsets.last().unwrap(),
            AffinePoint {
                x: F::ZERO,
                y: F::ZERO,
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
        // Each bucket received exactly its counted points, so every
        // placeholder was overwritten and no bucket spilled into the next.
        debug_assert!(
            positions
                .iter()
                .zip(&offsets[1..])
                .all(|(position, end)| position == end)
        );

        reduce_affine_buckets_reusing::<F, false>(points, offsets, reduce)
    }
}

/// Combines the window sums, highest window first, by Horner's rule: one shift
/// chain of `window_bits` doublings between windows, shared by all of them.
/// Declines as soon as any window does.
///
/// Returns $\sum_k 2^{kw} W_k$ for window sums given as $W_K, \dots, W_0$, or
/// `None` if any window is `None`.
fn horner<C>(
    window_bits: usize,
    sums_from_highest: impl IntoIterator<Item = Option<C>>,
) -> Option<C>
where
    C: GlvParams,
{
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

fn multiexp_serial<C>(
    components: &[(SignedMagnitude, SignedMagnitude)],
    bases: &[MultiexpBase<AffineBase<C>>],
    window_bits: usize,
) -> Option<C>
where
    C: GlvParams,
    AffineBase<C>: CachedField,
{
    let window_count = window_count(window_bits);
    let mut reducer = Checkout::<AffineBase<C>>::take();
    horner::<C>(
        window_bits,
        (0..window_count).rev().map(|window| {
            let buckets = reducer.fill_window(components, bases, window_bits, window)?;
            Some(sum_buckets::<C>(buckets))
        }),
    )
}

/// Evaluates each signed-Booth window as its own Rayon task, then combines the
/// ordered sums with one Horner fold. This keeps the expensive window
/// reductions parallel, with work stealing, without duplicating the shift
/// chain. Each task checks a reducer out for its window: a thread returns it
/// before starting its next window, so consecutive windows on a thread reuse
/// the same buffers.
#[cfg(feature = "multicore")]
fn multiexp_parallel<C>(
    components: &[(SignedMagnitude, SignedMagnitude)],
    bases: &[MultiexpBase<AffineBase<C>>],
    window_bits: usize,
) -> Option<C>
where
    C: GlvParams,
    AffineBase<C>: CachedField,
{
    let window_count = window_count(window_bits);
    let window_sums: Option<Vec<C>> = (0..window_count)
        .into_par_iter()
        .map(|window| {
            let mut reducer = Checkout::<AffineBase<C>>::take();
            let buckets = reducer.fill_window(components, bases, window_bits, window)?;
            Some(sum_buckets::<C>(buckets))
        })
        .collect();
    horner::<C>(window_bits, window_sums?.into_iter().rev().map(Some))
}

/// Returns $\sum_c m_c Q_c = \sum_i k_i P_i$ for the GLV components of the
/// scalars (from `glv::decompose`) and the prepared bases (from
/// [`multiexp_bases`]), in the same order; `None` only if a reduction
/// declined. Uses parallel windows when `multicore` is enabled and
/// `num_threads > 1`.
pub(super) fn multiexp<C>(
    components: &[(SignedMagnitude, SignedMagnitude)],
    bases: &[MultiexpBase<AffineBase<C>>],
    window_bits: usize,
    num_threads: usize,
) -> Option<C>
where
    C: GlvParams,
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

/// Prepares the bases: coordinates, the $x$ of $\phi(P)$, and the identity flag.
pub(super) fn multiexp_bases<C>(bases: &[C::AffineExt]) -> Vec<MultiexpBase<AffineBase<C>>>
where
    C: GlvParams,
{
    bases
        .iter()
        .map(|base| {
            let (x, y) = affine_xy::<C>(base);
            MultiexpBase {
                x,
                endo_x: phi_x::<C>(x),
                y,
                identity: bool::from(base.is_identity()),
            }
        })
        .collect()
}
