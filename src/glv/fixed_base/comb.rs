//! A cross-base comb: the second prepared fixed-base strategy, beside the
//! positioned tables of the parent module, used from `COMB_MIN_BASES` bases.
//!
//! Writing every scalar in binary, $k_i = \sum_j k_{i,j} 2^j$, turns the sum
//! into one over a bit matrix, $\sum_i \sum_j k_{i,j} \, (2^j P_i)$, with a row
//! per base and a column per bit. A table lookup can cover a block of that
//! matrix: the positioned tables cover a run of consecutive bits of one scalar,
//! the comb covers the same bit of several bases.
//!
//! # Two classical ideas, combined
//!
//! - Straus's method (1964) precomputes the subset sums of a few bases, so that
//!   one addition covers one bit column of all of them, then runs Horner's rule
//!   over every bit: about as many doublings as the scalars have bits.
//! - Lim and Lee's comb (1994) precomputes copies of a base shifted by
//!   $2^{r \cdot \mathrm{stride}}$, so that Horner's rule runs over only
//!   `stride` columns: `stride - 1` doublings, for more stored rows.
//!
//! Applied across bases, the two give this comb. The bases are grouped
//! `group_bits` at a time. For a group $P_1, \dots, P_m$ and every stored row
//! $r$ (one bit position in every `stride`), the table holds the $2^m - 1$
//! nonzero subset sums $\sum_{i \in S} 2^{r \cdot \mathrm{stride}} P_i$. The
//! bits at position $r \cdot \mathrm{stride} + c$ of the group's scalars form an
//! $m$-bit index, so one addition covers all $m$ bases, and Horner's rule over
//! the `stride` positions $c$ between stored rows supplies the rest:
//!
//! $$\sum_i k_i P_i = \sum_{c} 2^c \sum_{\text{groups}} \sum_r
//! T_r[\text{bits } r \cdot \mathrm{stride} + c].$$
//!
//! # Cost
//!
//! About $\mathrm{bits} \cdot \lceil n / m \rceil$ additions and
//! $\mathrm{stride} - 1$ doublings per sum, for
//! $\lceil \mathrm{bits} / \mathrm{stride} \rceil (2^m - 1)$ points per group:
//! wider groups trade memory for additions, shorter strides trade memory for
//! doublings. A lookup's index is a bit pattern across bases, so neither of the
//! positioned tables' symmetries applies cheaply: negation would act on a whole
//! pattern at once (halving the table at best), and $\phi$ would double the
//! group. So for a group as large as a positioned window the comb stores more
//! points; its gain comes from larger groups.
//!
//! # References
//!
//! - E. G. Straus, "Addition chains of vectors" (solution to Advanced Problem
//!   5125, posed by R. Bellman), Amer. Math. Monthly 71 (1964), 806-808.
//! - C. H. Lim and P. J. Lee, "More flexible exponentiation with
//!   precomputation", CRYPTO '94, LNCS 839, Springer (1994), 95-107.

use alloc::{vec, vec::Vec};
use core::fmt;

use ff::PrimeField;

use super::{GlvParams, REDUCE_SUMS, bucket_sums, push_term};
use crate::arithmetic::PreparedFixedBaseSum;

/// The bit length of a scalar: the rows a comb with stride 1 would store.
fn scalar_bits<C: GlvParams>() -> usize {
    C::ScalarExt::NUM_BITS as usize
}

/// Bit `position` of a scalar's little-endian byte encoding (the Pasta
/// fields' `to_repr`), zero past its end.
fn bit(repr: &[u8], position: usize) -> usize {
    repr.get(position / 8)
        .map_or(0, |byte| usize::from((byte >> (position % 8)) & 1))
}

/// A cross-base comb over fixed bases; see the module documentation.
///
/// `tables` is laid out by group, then row, then subset: group `g`'s row `r`
/// holds its `2^m - 1` nonzero subset sums, `m` the group's size, at
/// `group_offsets[g] + r * (2^m - 1) + mask - 1`, where bit `i` of `mask`
/// selects the group's `i`-th base. Every group but the last has
/// `group_bits` bases.
pub(in crate::glv) struct CombSum<C: GlvParams> {
    /// The number of prepared bases.
    bases: usize,
    /// Bases per group, the width of a table index.
    group_bits: usize,
    /// Bit positions between stored rows; `stride - 1` doublings per sum.
    stride: usize,
    /// Stored rows per group, `ceil(scalar_bits / stride)`.
    rows: usize,
    /// Per group, per row, the nonzero subset sums at index `mask - 1`.
    tables: Vec<C::AffineExt>,
    /// Where each group's tables start in `tables`.
    group_offsets: Vec<usize>,
}

impl<C: GlvParams> fmt::Debug for CombSum<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CombSum")
            .field("bases", &self.bases)
            .field("group_bits", &self.group_bits)
            .field("stride", &self.stride)
            .field("points", &self.tables.len())
            .finish()
    }
}

impl<C: GlvParams> CombSum<C> {
    /// Prepares the comb. Requires `1 <= group_bits <= 16` (a group's table
    /// has `2^group_bits - 1` points per row) and `stride >= 1`.
    pub(in crate::glv) fn new(bases: &[C::AffineExt], group_bits: usize, stride: usize) -> Self {
        assert!(
            (1..=16).contains(&group_bits),
            "a comb groups 1 to 16 bases"
        );
        assert!(stride >= 1, "a comb stores at least one row in every bit");
        let rows = scalar_bits::<C>().div_ceil(stride);
        let mut projective = Vec::new();
        let mut group_offsets = Vec::with_capacity(bases.len().div_ceil(group_bits));
        for group in bases.chunks(group_bits) {
            group_offsets.push(projective.len());
            let mut shifted: Vec<C> = group.iter().map(|base| C::from(*base)).collect();
            let mut sums = vec![C::identity(); 1 << group.len()];
            for _ in 0..rows {
                // Each subset sum is a smaller one plus the lowest base it
                // contains: one addition per entry.
                for mask in 1..sums.len() {
                    let lowest = mask.trailing_zeros() as usize;
                    sums[mask] = sums[mask & (mask - 1)] + shifted[lowest];
                }
                projective.extend_from_slice(&sums[1..]);
                for point in &mut shifted {
                    for _ in 0..stride {
                        *point = point.double();
                    }
                }
            }
        }
        let mut tables = vec![C::AffineExt::default(); projective.len()];
        C::batch_normalize_vartime(&projective, &mut tables);
        CombSum {
            bases: bases.len(),
            group_bits,
            stride,
            rows,
            tables,
            group_offsets,
        }
    }

    /// Calls `f(point)` for every term of column `column`, whose sum the
    /// Horner fold multiplies by `2^column`: per group and stored row, the
    /// subset sum indexed by bit `row * stride + column` of the group's
    /// scalars, skipped when that pattern is zero. Fewer scalars than bases
    /// leave the remaining groups out, as zero scalars would.
    fn for_each_column_term(
        &self,
        reprs: &[<C::ScalarExt as PrimeField>::Repr],
        column: usize,
        mut f: impl FnMut(C::AffineExt),
    ) {
        let bits = scalar_bits::<C>();
        for (group, scalars) in reprs.chunks(self.group_bits).enumerate() {
            let members = (self.bases - group * self.group_bits).min(self.group_bits);
            let width = (1 << members) - 1;
            let start = self.group_offsets[group];
            for row in 0..self.rows {
                let position = row * self.stride + column;
                if position >= bits {
                    break;
                }
                let mask = scalars.iter().enumerate().fold(0, |mask, (i, repr)| {
                    mask | (bit(repr.as_ref(), position) << i)
                });
                if mask != 0 {
                    f(self.tables[start + row * width + mask - 1]);
                }
            }
        }
    }

    /// The scalars' byte encodings, whose bits index the tables. Panics if
    /// there are more scalars than prepared bases.
    fn reprs(&self, scalars: &[C::ScalarExt]) -> Vec<<C::ScalarExt as PrimeField>::Repr> {
        assert!(
            scalars.len() <= self.bases,
            "at most one scalar per prepared base"
        );
        scalars.iter().map(PrimeField::to_repr).collect()
    }
}

impl<C: GlvParams> PreparedFixedBaseSum<C> for CombSum<C> {
    fn num_bases(&self) -> usize {
        self.bases
    }

    /// Horner's rule over the `stride` columns: double, then add the
    /// column's terms.
    fn sum_vartime(&self, scalars: &[C::ScalarExt]) -> C {
        let reprs = self.reprs(scalars);
        let mut acc = C::identity();
        for column in (0..self.stride).rev() {
            if column + 1 < self.stride {
                acc = acc.double();
            }
            self.for_each_column_term(&reprs, column, |point| acc += point);
        }
        acc
    }

    /// The same fold for many sums, each column of a chunk of
    /// [`REDUCE_SUMS`] sums reduced together, one bucket per sum, so that all
    /// their additions share the affine reducer's inversions.
    fn sum_batch_vartime(&self, scalars: &[&[C::ScalarExt]], output: &mut [C]) {
        assert_eq!(scalars.len(), output.len(), "one output per sum");
        for (chunk, acc) in scalars
            .chunks(REDUCE_SUMS)
            .zip(output.chunks_mut(REDUCE_SUMS))
        {
            let reprs: Vec<_> = chunk.iter().map(|s| self.reprs(s)).collect();
            acc.fill(C::identity());
            for column in (0..self.stride).rev() {
                // Each sum's terms of this column are one bucket; the buckets
                // of the whole chunk share the reducer's inversions.
                let mut points = Vec::new();
                let mut offsets = Vec::with_capacity(chunk.len() + 1);
                offsets.push(0);
                for reprs in &reprs {
                    self.for_each_column_term(reprs, column, |point| {
                        push_term::<C>(&mut points, point, false);
                    });
                    offsets.push(points.len());
                }
                // The reducer declines only if its complete formulas meet a
                // zero denominator, which valid points rule out; stay correct
                // anyway with the projective fold.
                let columns = bucket_sums::<C>(points, offsets).unwrap_or_else(|| {
                    reprs
                        .iter()
                        .map(|reprs| {
                            let mut sum = C::identity();
                            self.for_each_column_term(reprs, column, |point| sum += point);
                            sum
                        })
                        .collect()
                });
                for (acc, column_sum) in acc.iter_mut().zip(columns) {
                    if column + 1 < self.stride {
                        *acc = acc.double();
                    }
                    *acc += column_sum;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
