//! Doubling-free sums over a few fixed bases.
//!
//! Each scalar is split by GLV into two signed halves of at most
//! [`HALF_BITS`] bits, $k = k_1 + k_2\lambda$, and each half is
//! recoded into signed-Booth digits of [`WINDOW_BITS`] bits, as in the MSM
//! (`glv::booth`). The table stores, for every base $P$ and window $w$, the
//! positive multiples
//! $m \cdot 2^{w \cdot \mathrm{WINDOW\_BITS}} P$ for
//! $m \in 1..=2^{\mathrm{WINDOW\_BITS} - 1}$, already shifted into position,
//! so evaluation needs no doublings:
//!
//! - a negative digit (or a negative half) negates the stored point;
//! - the $k_2$ half uses $\phi(Q) = (\zeta x, y)$ of the stored point $Q$,
//!   one field multiplication, since $\phi(Q) = \lambda Q$.
//!
//! Storing only the $P$ multiples and deriving the $\phi(P)$ ones on the fly
//! halves the table relative to recoding full-width scalars at the same
//! window width, at the cost of one multiplication per $k_2$ addition.
//!
//! [`PreparedFixedBaseSum::sum_batch_vartime`] evaluates many independent
//! sums together. Each sum is a list of known affine points, so the sums are
//! the buckets of the MSM's affine bucket reducer (`reduce_affine_buckets`):
//! pairwise trees reduced level by level, with one shared inversion per level
//! (Montgomery's trick) and complete formulas for doublings and cancellations.
//! An affine addition then costs about half a mixed projective one.

use alloc::{boxed::Box, vec, vec::Vec};
use core::fmt;

use ff::{Field, WithSmallOrderMulGroup};

use super::{
    GlvParams, HALF_BITS, affine_from_xy_unchecked, affine_xy,
    booth::{SignedMagnitude, booth_digit, window_count},
    decompose,
    msm::{AffinePoint, reduce_affine_buckets},
    private::AffineBase,
};
use crate::arithmetic::PreparedFixedBaseSum;

/// The signed-digit window width.
///
/// Seven bits keep the table at about 76 KiB per base (19 windows of 64
/// points). Wider windows mean fewer additions per half-scalar (19 at 7 bits,
/// 22 at 6) for twice the table per extra bit.
/// Measured on Zen 4 for Orchard's seven bases, 7 bits committed instances
/// 1.36x faster than the old fixed-window path, against 1.20x at 6 bits.
const WINDOW_BITS: usize = 7;

/// Positive digit magnitudes per window, `1..=2^(WINDOW_BITS - 1)`.
const MAGNITUDES: usize = 1 << (WINDOW_BITS - 1);

/// Windows per half-scalar ([`window_count`]).
const WINDOWS: usize = window_count(WINDOW_BITS);

/// Sums reduced together by one call to the affine bucket reducer.
///
/// Each Orchard-shaped sum holds about 270 points (17 KiB), so a reduction
/// over many sums outgrows the core's L2 cache; on Zen 4, 64 or 128 sums per
/// call ran about a third slower per sum than 8 to 32. Caps of 8, 16 and 32
/// measured the same, so the smallest keeps the working set (about 136 KiB)
/// furthest from cache limits; it costs only one more shared inversion per
/// tree level. Larger batches are split.
const REDUCE_SUMS: usize = 8;

/// Builds a [`PreparedFixedBaseSum`] over `bases`.
pub(crate) fn prepare_fixed_base_sum<C: GlvParams>(
    bases: &[C::AffineExt],
) -> Box<dyn PreparedFixedBaseSum<C>> {
    Box::new(FixedBaseSum::<C>::new(bases))
}

/// The positioned multiples of each base, laid out by base, then window,
/// then magnitude.
struct FixedBaseSum<C: GlvParams> {
    bases: usize,
    points: Vec<C::AffineExt>,
}

impl<C: GlvParams> fmt::Debug for FixedBaseSum<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FixedBaseSum")
            .field("bases", &self.bases)
            .field("points", &self.points.len())
            .finish()
    }
}

impl<C: GlvParams> FixedBaseSum<C> {
    fn new(bases: &[C::AffineExt]) -> Self {
        let mut projective = Vec::with_capacity(bases.len() * WINDOWS * MAGNITUDES);
        for base in bases {
            let mut shifted = C::from(*base);
            for _ in 0..WINDOWS {
                let mut multiple = shifted;
                for _ in 0..MAGNITUDES {
                    projective.push(multiple);
                    multiple += shifted;
                }
                for _ in 0..WINDOW_BITS {
                    shifted = shifted.double();
                }
            }
        }
        let mut points = vec![C::AffineExt::default(); projective.len()];
        C::batch_normalize_vartime(&projective, &mut points);
        FixedBaseSum {
            bases: bases.len(),
            points,
        }
    }

    /// Calls `f(point, negate)` for each nonzero signed-Booth digit of
    /// `half * P_base`, or of `phi` of it when `endo`. The term is `-point`
    /// when `negate` is set.
    fn for_each_term(
        &self,
        base: usize,
        half: SignedMagnitude,
        endo: bool,
        mut f: impl FnMut(C::AffineExt, bool),
    ) {
        let windows = &self.points[base * WINDOWS * MAGNITUDES..][..WINDOWS * MAGNITUDES];
        for window in 0..WINDOWS {
            let digit = booth_digit(half, WINDOW_BITS, window);
            if digit.magnitude == 0 {
                continue;
            }
            let point = windows[window * MAGNITUDES + digit.magnitude - 1];
            let point = if endo {
                let (x, y) = affine_xy::<C>(&point);
                affine_from_xy_unchecked::<C>(x * AffineBase::<C>::ZETA, y)
            } else {
                point
            };
            f(point, digit.negative);
        }
    }

    /// Calls `f(point, negate)` for every term of `sum_i [k_i] P_i`.
    fn for_each_sum_term(&self, scalars: &[C::ScalarExt], mut f: impl FnMut(C::AffineExt, bool)) {
        assert!(
            scalars.len() <= self.bases,
            "at most one scalar per prepared base"
        );
        for (base, scalar) in scalars.iter().enumerate() {
            let (first, second) = decompose::<C>(scalar);
            // The digits of `WINDOWS` windows sum to a half only below
            // `2^HALF_BITS`, which `glv::decompose` guarantees.
            debug_assert!(
                first.1 >> HALF_BITS == 0 && second.1 >> HALF_BITS == 0,
                "a GLV half is 2^127 or larger"
            );
            self.for_each_term(base, first.into(), false, &mut f);
            self.for_each_term(base, second.into(), true, &mut f);
        }
    }
}

impl<C: GlvParams> PreparedFixedBaseSum<C> for FixedBaseSum<C> {
    fn bases(&self) -> usize {
        self.bases
    }

    fn sum_vartime(&self, scalars: &[C::ScalarExt]) -> C {
        let mut acc = C::identity();
        self.for_each_sum_term(scalars, |point, negate| {
            if negate {
                acc -= point;
            } else {
                acc += point;
            }
        });
        acc
    }

    fn sum_batch_vartime(&self, scalars: &[&[C::ScalarExt]]) -> Vec<C> {
        scalars
            .chunks(REDUCE_SUMS)
            .flat_map(|chunk| self.reduce_sums(chunk))
            .collect()
    }
}

impl<C: GlvParams> FixedBaseSum<C> {
    /// Reduces the sums of `scalars` together; see [`REDUCE_SUMS`].
    fn reduce_sums(&self, scalars: &[&[C::ScalarExt]]) -> Vec<C> {
        // One bucket per sum, laid out contiguously for the reducer.
        let mut points =
            Vec::with_capacity(scalars.iter().map(|s| s.len()).sum::<usize>() * 2 * WINDOWS);
        let mut offsets = Vec::with_capacity(scalars.len() + 1);
        offsets.push(0);
        for scalars in scalars {
            self.for_each_sum_term(scalars, |point, negate| {
                let (x, y) = affine_xy::<C>(&point);
                // Table entries of an identity base are the identity, encoded
                // as (0, 0). The reducer requires non-identity points, and
                // they contribute nothing.
                if !(x.is_zero_vartime() && y.is_zero_vartime()) {
                    points.push(AffinePoint {
                        x,
                        y: if negate { -y } else { y },
                    });
                }
            });
            offsets.push(points.len());
        }
        match reduce_affine_buckets(points, offsets) {
            Some(sums) => sums
                .into_iter()
                .map(|sum| match sum {
                    Some(p) => C::from(affine_from_xy_unchecked::<C>(p.x, p.y)),
                    None => C::identity(),
                })
                .collect(),
            // The reducer declines only if its complete formulas meet a zero
            // denominator, which valid points rule out; stay correct anyway.
            None => scalars.iter().map(|s| self.sum_vartime(s)).collect(),
        }
    }
}

#[cfg(test)]
mod tests;
