//! Small sums: one Straus doubling chain shared by all terms, using the
//! split-wNAF tables of `glv`, optionally split into parallel leaves.
//!
//! # Straus's method
//!
//! A sum $S = \sum_i k_i P_i$ could be computed term by term, each with its own
//! double-and-add ladder. Straus's method merges the ladders into one. Write
//! each term as a polynomial with point coefficients,
//! $p_i(x) = \sum_j d_{i,j} P_i x^j$, where the $d_{i,j}$ are the digits of
//! $k_i$, so that $k_i P_i = p_i(2)$. Evaluating at $x = 2$ is additive in the
//! coefficients, hence $S = \left(\sum_i p_i\right)(2)$: a single Horner
//! evaluation of the sum polynomial, which doubles once per digit column and
//! adds every term's contribution to that column. All terms share the
//! doublings. With two terms, this is known as Shamir's trick.
//!
//! Here the scalars are recoded as width-4 wNAF digits ([`Decomposed`]) and
//! each base gets a table of odd multiples ([`Table`]), as in [`crate::glv`];
//! the fold itself is [`Recoding::sum_of_products`].
//! That costs about 128 doublings in all, then per term about one addition per
//! nonzero digit (one digit in five on average, over two halves) and its table,
//! all tables sharing one normalization. The cost grows with every term, while
//! the bucket method pays a fixed setup per window and amortizes it over many
//! terms, so Straus's method wins for few terms. The crossovers are the
//! planner's (`plan`).
//!
//! E. G. Straus, "Addition chains of vectors" (solution to Advanced Problem
//! 5125, posed by R. Bellman), Amer. Math. Monthly 71 (1964), 806-808.
//!
//! # Reading guide
//!
//! The planner decides whether Straus applies. [`straus_multiexp`] then runs
//! one table, or, with `multicore`, `straus_multiexp_jobs` runs leaves of at
//! most `straus_leaf_terms` terms of the calibration in use, in parallel
//! (`straus_multiexp_leaves`).

use alloc::vec::Vec;
use group::CurveAffine as _;
#[cfg(feature = "multicore")]
use maybe_rayon::prelude::*;

#[cfg(any(feature = "multicore", doc))]
use super::calibration::CALIBRATION;
use crate::glv::{Decomposed, GlvParams, Recoding, Table, Wnaf4};

/// Share the doubling ladder across all scalar/base pairs, reusing the
/// upstream GLV decomposition and width-four split-wNAF tables unchanged.
///
/// Returns $\sum_i k_i P_i$ exactly, for `scalars` and `bases` of equal length.
/// Zero scalars and identity bases are skipped.
pub(super) fn straus_multiexp<C>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> C
where
    C: GlvParams,
{
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
    <Wnaf4 as Recoding<C>>::sum_of_products(&tables, &decomposed)
}

#[cfg(feature = "multicore")]
fn straus_multiexp_leaves<C>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> C
where
    C: GlvParams,
{
    scalars
        .chunks(CALIBRATION.straus_leaf_terms)
        .zip(bases.chunks(CALIBRATION.straus_leaf_terms))
        .map(|(scalars, bases)| straus_multiexp::<C>(scalars, bases))
        .fold(C::identity(), |left, right| left + right)
}

/// Returns $\sum_i k_i P_i$, the terms split into `jobs` contiguous jobs run in
/// parallel and summed, each a run of Straus leaves of at most
/// `straus_leaf_terms` terms (see [`CALIBRATION`]); with one job, the leaves
/// run on this thread.
/// Requires `scalars.len() == bases.len()` and `1 <= jobs <= scalars.len()`.
#[cfg(feature = "multicore")]
pub(super) fn straus_multiexp_jobs<C>(
    scalars: &[C::ScalarExt],
    bases: &[C::AffineExt],
    jobs: usize,
) -> C
where
    C: GlvParams,
{
    if jobs == 1 {
        return straus_multiexp_leaves::<C>(scalars, bases);
    }
    (0..jobs)
        .into_par_iter()
        .map(|job| {
            let start = scalars.len() * job / jobs;
            let end = scalars.len() * (job + 1) / jobs;
            straus_multiexp_leaves::<C>(&scalars[start..end], &bases[start..end])
        })
        .sum()
}
