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

use ff::Field;

use super::{
    GlvParams, HALF_BITS, Recoding, affine_from_xy_unchecked, affine_xy,
    booth::{SignedMagnitude, booth_digit, window_count},
    decompose,
    msm::{AffinePoint, reduce_affine_buckets},
    phi_x,
    private::AffineBase,
};
use crate::arithmetic::PreparedFixedBaseSum;

/// The signed-digit window width.
///
/// Seven bits keep the table at about 76 KiB per base (19 windows of 64
/// points). Wider windows mean fewer additions per half-scalar (19 at 7 bits,
/// 22 at 6) for twice the table per extra bit; for Orchard's seven bases, 7
/// bits measured faster than 6.
const WINDOW_BITS: usize = 7;

/// Positive digit magnitudes per window, `1..=2^(WINDOW_BITS - 1)`.
const MAGNITUDES: usize = 1 << (WINDOW_BITS - 1);

/// Windows per half-scalar ([`window_count`]).
const WINDOWS: usize = window_count(WINDOW_BITS);

/// Sums reduced together by one call to the affine bucket reducer.
///
/// Each Orchard-shaped sum holds about 270 points (17 KiB), so a reduction
/// over many sums outgrows a core's L2 cache, and larger calls ran slower per
/// sum. Caps of 8, 16 and 32 measured the same, so the smallest keeps the
/// working set (about 136 KiB) furthest from cache limits; it costs only one
/// more shared inversion per tree level. Larger batches are split.
const REDUCE_SUMS: usize = 8;

/// The recoding of the prepared fixed-base sums: both GLV halves of a scalar
/// in signed-Booth digits of [`WINDOW_BITS`] bits (`glv::booth`), radix
/// $2^{\mathrm{WINDOW\_BITS}}$.
#[derive(Clone, Copy, Debug)]
pub(super) struct Booth7;

/// The [`Booth7`] table of a point $Q$: $m Q$ for $m \in 1..=$[`MAGNITUDES`].
/// The second half's digits index the same entries through $\phi$, applied on
/// lookup, which halves the table.
pub(super) struct BoothTable<C: GlvParams> {
    multiples: [C::AffineExt; MAGNITUDES],
}

impl<C: GlvParams> BoothTable<C> {
    /// The tables of many points, sharing one batch normalization.
    fn batch(points: &[C]) -> Vec<Self> {
        let mut projective = Vec::with_capacity(points.len() * MAGNITUDES);
        for point in points {
            let mut multiple = *point;
            for _ in 0..MAGNITUDES {
                projective.push(multiple);
                multiple += point;
            }
        }
        let mut affine = vec![C::AffineExt::default(); projective.len()];
        C::batch_normalize_vartime(&projective, &mut affine);
        affine
            .chunks_exact(MAGNITUDES)
            .map(|multiples| BoothTable {
                multiples: multiples.try_into().expect("MAGNITUDES multiples"),
            })
            .collect()
    }
}

impl<C: GlvParams> Recoding<C> for Booth7 {
    type Digits = (SignedMagnitude, SignedMagnitude);
    type Table = BoothTable<C>;
    const COLUMN_BITS: usize = WINDOW_BITS;
    const MAX_COLUMNS: usize = WINDOWS;

    fn recode(k: &C::ScalarExt) -> Self::Digits {
        let (first, second) = decompose::<C>(k);
        // The digits of `WINDOWS` windows sum to a half only below
        // `2^HALF_BITS`, which `glv::decompose` guarantees.
        debug_assert!(
            first.1 >> HALF_BITS == 0 && second.1 >> HALF_BITS == 0,
            "a GLV half is 2^127 or larger"
        );
        (first.into(), second.into())
    }

    #[cfg(test)]
    fn table(p: &C) -> Self::Table {
        let mut tables = BoothTable::batch(core::slice::from_ref(p));
        tables.pop().expect("one point, one table")
    }

    fn batch_tables(points: &[C]) -> Vec<Self::Table> {
        BoothTable::batch(points)
    }

    /// Every recoding has [`WINDOWS`] columns: a half below `2^HALF_BITS`
    /// needs them all in the worst case, and zero digits cost nothing.
    fn columns(_digits: &Self::Digits) -> usize {
        WINDOWS
    }

    fn column_terms(
        table: &Self::Table,
        digits: &Self::Digits,
        i: usize,
        mut f: impl FnMut(C::AffineExt, bool),
    ) {
        if i >= WINDOWS {
            return;
        }
        for (half, endo) in [(digits.0, false), (digits.1, true)] {
            let digit = booth_digit(half, WINDOW_BITS, i);
            if digit.magnitude == 0 {
                continue;
            }
            let point = table.multiples[digit.magnitude - 1];
            let point = if endo {
                let (x, y) = affine_xy::<C>(&point);
                affine_from_xy_unchecked::<C>(phi_x::<C>(x), y)
            } else {
                point
            };
            f(point, digit.negative);
        }
    }
}

/// Builds a [`PreparedFixedBaseSum`] over `bases`.
pub(crate) fn prepare_fixed_base_sum<C: GlvParams>(
    bases: &[C::AffineExt],
) -> Box<dyn PreparedFixedBaseSum<C>> {
    Box::new(FixedBaseSum::<C, Booth7>::new(bases))
}

/// A positioned evaluation of the recoding `R`: per base $P$, one `R` table
/// per column, column $j$'s over $2^{j \cdot \mathrm{COLUMN\_BITS}} P$, laid
/// out by base, then column. Column $j$'s digit then indexes its own table,
/// so the sum needs none of Horner's doublings: they were spent once, on
/// preparation. Pasta's backend is `FixedBaseSum<C, Booth7>`.
pub(super) struct FixedBaseSum<C: GlvParams, R: Recoding<C>> {
    bases: usize,
    tables: Vec<R::Table>,
}

impl<C: GlvParams, R: Recoding<C>> fmt::Debug for FixedBaseSum<C, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FixedBaseSum")
            .field("bases", &self.bases)
            .field("tables", &self.tables.len())
            .finish()
    }
}

impl<C: GlvParams, R: Recoding<C>> FixedBaseSum<C, R> {
    pub(super) fn new(bases: &[C::AffineExt]) -> Self {
        let mut positioned = Vec::with_capacity(bases.len() * R::MAX_COLUMNS);
        for base in bases {
            let mut shifted = C::from(*base);
            for _ in 0..R::MAX_COLUMNS {
                positioned.push(shifted);
                for _ in 0..R::COLUMN_BITS {
                    shifted = shifted.double();
                }
            }
        }
        FixedBaseSum {
            bases: bases.len(),
            tables: R::batch_tables(&positioned),
        }
    }

    /// Calls `f(point, negate)` for every term of `sum_i [k_i] P_i`.
    fn for_each_sum_term(&self, scalars: &[C::ScalarExt], mut f: impl FnMut(C::AffineExt, bool)) {
        assert!(
            scalars.len() <= self.bases,
            "at most one scalar per prepared base"
        );
        for (base, scalar) in scalars.iter().enumerate() {
            let digits = R::recode(scalar);
            debug_assert!(R::columns(&digits) <= R::MAX_COLUMNS);
            let tables = &self.tables[base * R::MAX_COLUMNS..][..R::MAX_COLUMNS];
            for (column, table) in tables.iter().enumerate().take(R::columns(&digits)) {
                R::column_terms(table, &digits, column, &mut f);
            }
        }
    }
}

impl<C, R> PreparedFixedBaseSum<C> for FixedBaseSum<C, R>
where
    C: GlvParams,
    R: Recoding<C>,
    R::Table: Send + Sync,
{
    fn num_bases(&self) -> usize {
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

    fn sum_batch_vartime(&self, scalars: &[&[C::ScalarExt]], output: &mut [C]) {
        assert_eq!(scalars.len(), output.len(), "one output per sum");
        for (scalars, output) in scalars
            .chunks(REDUCE_SUMS)
            .zip(output.chunks_mut(REDUCE_SUMS))
        {
            self.reduce_sums(scalars, output);
        }
    }
}

impl<C, R> FixedBaseSum<C, R>
where
    C: GlvParams,
    R: Recoding<C>,
    R::Table: Send + Sync,
{
    /// Reduces the sums of `scalars` together into `output`, one per sum; see
    /// [`REDUCE_SUMS`].
    fn reduce_sums(&self, scalars: &[&[C::ScalarExt]], output: &mut [C]) {
        // One bucket per sum, laid out contiguously for the reducer.
        let mut points =
            Vec::with_capacity(scalars.iter().map(|s| s.len()).sum::<usize>() * 2 * WINDOWS);
        let mut offsets = Vec::with_capacity(scalars.len() + 1);
        offsets.push(0);
        for scalars in scalars {
            self.for_each_sum_term(scalars, |point, negate| {
                push_term::<C>(&mut points, point, negate);
            });
            offsets.push(points.len());
        }
        match bucket_sums::<C>(points, offsets) {
            Some(sums) => output.copy_from_slice(&sums),
            // The reducer declines only if its complete formulas meet a zero
            // denominator, which valid points rule out and which it asserts
            // against in debug builds; stay correct anyway.
            None => {
                for (output, scalars) in output.iter_mut().zip(scalars) {
                    *output = self.sum_vartime(scalars);
                }
            }
        }
    }
}

/// Appends `point`, negated when `negate`, to the current bucket of a batched
/// affine reduction. Table entries of an identity base are the identity,
/// encoded as (0, 0); the reducer requires non-identity points, and they
/// contribute nothing, so they are skipped.
fn push_term<C: GlvParams>(
    points: &mut Vec<AffinePoint<AffineBase<C>>>,
    point: C::AffineExt,
    negate: bool,
) {
    let (x, y) = affine_xy::<C>(&point);
    if !(x.is_zero_vartime() && y.is_zero_vartime()) {
        points.push(AffinePoint {
            x,
            y: if negate { -y } else { y },
        });
    }
}

/// The sums of the buckets `points[offsets[j]..offsets[j + 1]]`, one shared
/// inversion per tree level, or `None` if the reducer declines.
fn bucket_sums<C: GlvParams>(
    points: Vec<AffinePoint<AffineBase<C>>>,
    offsets: Vec<usize>,
) -> Option<Vec<C>> {
    let sums = reduce_affine_buckets(points, offsets)?;
    Some(
        sums.into_iter()
            .map(|sum| match sum {
                Some(p) => C::from(affine_from_xy_unchecked::<C>(p.x, p.y)),
                None => C::identity(),
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests;
