//! Every point of a batch multiplied by every one of several scalars, for any
//! [`Recoding`] whose digits read as a unit times one of a few representatives
//! ([`OrbitRecoding`]).
//!
//! A ladder per product doubles an accumulator per point *and* scalar. Here a
//! point's doublings serve every scalar. A nonzero digit $u\,r_j$ of a scalar,
//! in the column of $2^i P$, contributes $u\,2^i P$ to the scalar's bucket
//! $B_j$ for the representative $r_j$; the unit $u$ is free (a negation, or a
//! rotation by $\varphi$, one multiplication). Each product is then
//! $\sum_j r_j B_j$, by an addition chain the recoding provides
//! ([`OrbitRecoding::combine_buckets`]).
//!
//! The evaluation is the one `fixed_base` runs over its prepared tables: the
//! multiples $2^i P$ of every point are computed once (projective doublings,
//! one batch normalization), every bucket's terms are listed, and the MSM's
//! affine bucket reducer ([`reduce_affine_buckets`]) sums all the buckets of
//! the grid at once, with one shared inversion per level of its pairwise trees
//! and complete formulas for equal and opposite operands. The combination's
//! rounds go through the same reducer.

use alloc::{vec, vec::Vec};

use ff::{Field, WithSmallOrderMulGroup};
use group::CurveAffine as _;

use super::msm::{AffinePoint, reduce_affine_buckets};
use super::private::AffineBase;
use super::{GlvParams, Recoding, affine_from_xy_unchecked, affine_xy};
use crate::arithmetic::{VartimeBatchInvert, VartimeField};

/// A sum of points, or `None` for the identity, as the bucket reducer returns
/// it.
pub(crate) type Sum<F> = Option<AffinePoint<F>>;

/// A [`Recoding`] whose digits read as a unit times one of
/// [`OrbitRecoding::REPS`] representatives, which is what [`mul_grid`] needs.
///
/// Implementations must satisfy, for every recoding `k` of a scalar and every
/// point `P`, with $B_j$ the sum over the columns $i$ of the terms of
/// [`OrbitRecoding::terms`] for representative $j$: `combine_buckets` of the
/// $B_j$ is `P * k`.
pub(crate) trait OrbitRecoding<C>: Recoding<C>
where
    C: GlvParams,
{
    /// The number of representatives, hence of buckets per product.
    const REPS: usize;

    /// Calls `f(rep, rotation, negative)` for each nonzero term of column
    /// `i`, which adds $(-1)^{\text{negative}} \varphi^{\text{rotation}}(2^i P)$
    /// to the bucket of representative `rep`; nothing for a zero column and
    /// for every `i >= columns(digits)`. `rotation` is below 3.
    fn terms(digits: &Self::Digits, i: usize, f: impl FnMut(usize, usize, bool));

    /// $\sum_j r_j B_j$ for every product, `buckets` holding the
    /// [`OrbitRecoding::REPS`] buckets of one product after those of another.
    fn combine_buckets(buckets: &[Sum<AffineBase<C>>]) -> Vec<Sum<AffineBase<C>>>;
}

/// The sum of each group of points, all groups sharing the reducer's
/// inversions.
fn sum_each<F, G>(groups: impl IntoIterator<Item = G>) -> Vec<Sum<F>>
where
    F: VartimeField,
    G: IntoIterator<Item = Sum<F>>,
{
    let mut points = Vec::new();
    let mut offsets = vec![0];
    for group in groups {
        points.extend(group.into_iter().flatten());
        offsets.push(points.len());
    }
    reduce_affine_buckets(points, offsets).expect("complete affine formulas")
}

/// Two equally long columns of sums, to be added entry by entry.
pub(crate) type Columns<'a, F> = (&'a [Sum<F>], &'a [Sum<F>]);

/// `a + b` entry by entry for each `(a, b)` pair of equally long columns,
/// all in one reduction: one round of an addition chain.
pub(crate) fn add_columns<F, const N: usize>(terms: [Columns<'_, F>; N]) -> [Vec<Sum<F>>; N]
where
    F: VartimeField,
{
    let len = terms.first().map_or(0, |(a, _)| a.len());
    let sums = sum_each(
        terms
            .iter()
            .flat_map(|(a, b)| a.iter().zip(b.iter()).map(|(a, b)| [*a, *b])),
    );
    core::array::from_fn(|i| sums[i * len..(i + 1) * len].to_vec())
}

/// $-P$.
pub(crate) fn neg<F: Field>(p: Sum<F>) -> Sum<F> {
    p.map(|p| AffinePoint { x: p.x, y: -p.y })
}

/// $\varphi(P) = (\zeta x, y)$.
pub(crate) fn phi<F: WithSmallOrderMulGroup<3>>(p: Sum<F>) -> Sum<F> {
    p.map(|p| AffinePoint {
        x: p.x * F::ZETA,
        y: p.y,
    })
}

/// `k * p` for every point `p` and every recoded scalar `k`: the product of
/// `points[i]` and `digits[j]` is at index `i * digits.len() + j` of the
/// result, which is affine.
///
/// A grid of fewer than `threshold` products is evaluated with one table per
/// point, built together, and one projective ladder per product; a larger one
/// shares the doublings of each point among the scalars ([`shared_doublings`]).
/// Identity points and zero scalars are handled. Variable-time in the scalars
/// and the points.
pub(crate) fn mul_grid<C, R>(
    points: &[C],
    digits: &[R::Digits],
    threshold: usize,
) -> Vec<C::AffineExt>
where
    C: GlvParams,
    R: OrbitRecoding<C>,
{
    if points.is_empty() || digits.is_empty() {
        return Vec::new();
    }
    if points.len() * digits.len() < threshold {
        let proj: Vec<C> = R::batch_tables(points)
            .iter()
            .flat_map(|t| digits.iter().map(move |k| R::mul(t, k)))
            .collect();
        let mut affine = vec![C::AffineExt::identity(); proj.len()];
        C::batch_normalize_vartime(&proj, &mut affine);
        return affine;
    }
    shared_doublings::<C, R>(points, digits)
}

/// The shared-doubling evaluation of [`mul_grid`], whatever the size of the
/// grid.
pub(crate) fn shared_doublings<C, R>(points: &[C], digits: &[R::Digits]) -> Vec<C::AffineExt>
where
    C: GlvParams,
    R: OrbitRecoding<C>,
{
    let (n, m) = (points.len(), digits.len());
    let mut out = vec![C::AffineExt::identity(); n * m];

    // An identity point has no lane: its products are the identities `out`
    // already holds.
    let lanes: Vec<usize> = (0..n)
        .filter(|&i| !bool::from(points[i].is_identity()))
        .collect();
    let len = digits.iter().map(R::columns).max().unwrap_or(0);
    if lanes.is_empty() || len == 0 {
        return out;
    }

    // The multiples 2^col P of every lane's point, column by column: affine
    // doublings sharing one inversion per column, as the ladders do. `chain`
    // holds column col of lane l at `col * lanes.len() + l`.
    let nl = lanes.len();
    let lane_points: Vec<C> = lanes.iter().map(|&i| points[i]).collect();
    let mut base = vec![C::AffineExt::identity(); nl];
    C::batch_normalize_vartime(&lane_points, &mut base);
    let mut cur: Vec<AffinePoint<AffineBase<C>>> = base
        .iter()
        .map(|a| {
            let (x, y) = affine_xy::<C>(a);
            AffinePoint { x, y }
        })
        .collect();
    let mut chain = Vec::with_capacity(nl * len);
    let mut inverses = Vec::with_capacity(nl);
    for col in 0..len {
        chain.extend_from_slice(&cur);
        if col + 1 < len {
            for _ in 0..R::COLUMN_BITS {
                // The tangent's slope 3x^2 / 2y; 2y is not zero, since no
                // point has order two, the group's order being prime.
                inverses.clear();
                inverses.extend(cur.iter().map(|p| p.y.double()));
                inverses.iter_mut().batch_invert_vartime();
                for (p, inverse) in cur.iter_mut().zip(&inverses) {
                    let xx = p.x.square();
                    let slope = (xx.double() + xx) * inverse;
                    let x = slope.square() - p.x.double();
                    p.y = slope * (p.x - x) - p.y;
                    p.x = x;
                }
            }
        }
    }

    // Each scalar's terms, by representative: (column, rotation, negative),
    // and where each representative's terms start. They depend on the scalar
    // alone, so every lane reuses them.
    let mut terms = Vec::with_capacity(m);
    let mut starts = Vec::with_capacity(m);
    for k in digits {
        let mut t = Vec::new();
        for col in 0..len {
            R::terms(k, col, |rep, rotation, negative| {
                t.push((rep, col, rotation, negative))
            });
        }
        t.sort_by_key(|&(rep, ..)| rep);
        let mut start = vec![0; R::REPS + 1];
        for &(rep, ..) in &t {
            start[rep + 1] += 1;
        }
        for j in 0..R::REPS {
            start[j + 1] += start[j];
        }
        terms.push(t);
        starts.push(start);
    }

    // The terms of the bucket of lane l, scalar s and representative j, one
    // bucket after another in that order.
    let zeta = AffineBase::<C>::ZETA;
    let mut bucket_points = Vec::new();
    let mut offsets = Vec::with_capacity(lanes.len() * m * R::REPS + 1);
    offsets.push(0);
    for l in 0..nl {
        for (t, start) in terms.iter().zip(&starts) {
            for &(_, col, rotation, negative) in t {
                let AffinePoint { x, y } = chain[col * nl + l];
                let x = match rotation {
                    0 => x,
                    1 => x * zeta,
                    // zeta^2 x = -(1 + zeta) x, one multiplication too.
                    _ => -(x + x * zeta),
                };
                let y = if negative { -y } else { y };
                bucket_points.push(AffinePoint { x, y });
            }
            let base = offsets[offsets.len() - 1];
            offsets.extend(start[1..].iter().map(|&end| base + end));
        }
    }
    let buckets = reduce_affine_buckets(bucket_points, offsets).expect("complete affine formulas");

    let products = R::combine_buckets(&buckets);
    for (l, &i) in lanes.iter().enumerate() {
        for s in 0..m {
            if let Some(p) = products[l * m + s] {
                out[i * m + s] = affine_from_xy_unchecked::<C>(p.x, p.y);
            }
        }
    }
    out
}
