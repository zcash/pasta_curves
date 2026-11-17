//! Small sums: one Straus doubling chain shared by all terms, using the
//! split-wNAF tables of `glv`, optionally split into parallel leaves.
//!
//! # The shared doubling chain
//!
//! The scalars are recoded ([`Decomposed`]) and the bases tabulated
//! ([`Table`]) as in [`crate::glv`], where one multiplication $kP$ is Horner's
//! rule over the digit columns: double, then add the column's table entries.
//!
//! For a sum $\sum_i k_i P_i$ the doublings can be shared. Doubling is a group
//! homomorphism, so running Horner's rule once on the column-wise sum of all
//! terms' contributions gives the sum of the individual results: double once
//! per column, then add every term's entries for that column. This is Straus's
//! method (E. G. Straus, "Addition chains of vectors", Amer. Math. Monthly 71,
//! 1964). It costs about 128 doublings in all instead of 128 per term, plus one
//! addition per nonzero digit and the tables, which share one normalization.
//!
//! For few terms that beats the bucket method, whose setup does not amortize;
//! the crossovers are the constants below.
//!
//! # Reading guide
//!
//! [`planned_straus_multiexp`] is the entry point. It decides whether Straus
//! applies, the constants being the measured crossovers, and runs either one
//! [`straus_multiexp`] or, with `multicore`, leaves of at most
//! [`STRAUS_LEAF_TERMS`] terms in parallel (`straus_multiexp_leaves`).

use alloc::vec::Vec;
use group::CurveAffine as _;
#[cfg(feature = "multicore")]
use maybe_rayon::prelude::*;

use super::plan::SERIAL_STRAUS_MAX_TERMS;
use crate::glv::{Decomposed, GlvParams, Table};

/// Largest GLV-Straus leaf when MSMs are split across worker threads.
const STRAUS_LEAF_TERMS: usize = 16;

/// Smallest MSM worth splitting across worker jobs.
#[cfg(feature = "multicore")]
const MIN_PARALLEL_STRAUS_TERMS: usize = 3;

/// Largest parallel MSM split into [`STRAUS_LEAF_TERMS`]-or-smaller Straus
/// leaves. Above this conservative crossover, the generic or GLV bucket
/// backends amortize their bucket setup more effectively.
#[cfg(feature = "multicore")]
const MAX_PARALLEL_STRAUS_TERMS: usize = 66;

/// Share the doubling ladder across all scalar/base pairs, reusing the
/// upstream GLV decomposition and width-four split-wNAF tables unchanged.
///
/// Returns $\sum_i k_i P_i$ exactly, for `scalars` and `bases` of equal length.
/// Zero scalars and identity bases are skipped.
fn straus_multiexp<C>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> C
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
fn straus_multiexp_leaves<C>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> C
where
    C: GlvParams,
{
    scalars
        .chunks(STRAUS_LEAF_TERMS)
        .zip(bases.chunks(STRAUS_LEAF_TERMS))
        .map(|(scalars, bases)| straus_multiexp::<C>(scalars, bases))
        .fold(C::identity(), |left, right| left + right)
}

/// Evaluates the sum with Straus when the input is small enough, and declines
/// otherwise so that the caller can use the bucket backend.
///
/// Returns `Some` of $\sum_i k_i P_i$ when:
/// - with `multicore` and `num_threads > 1`, there are 3 to 66 terms: they are
///   split into at most `ceil(num_threads / 2)` jobs of Straus leaves of at
///   most [`STRAUS_LEAF_TERMS`] terms, computed in parallel and summed;
/// - otherwise, there are at most [`SERIAL_STRAUS_MAX_TERMS`] terms with one
///   thread, or [`STRAUS_LEAF_TERMS`] terms with more.
///
/// Returns `None` for larger inputs. Requires `scalars.len() == bases.len()`.
pub(super) fn planned_straus_multiexp<C>(
    scalars: &[C::ScalarExt],
    bases: &[C::AffineExt],
    num_threads: usize,
) -> Option<C>
where
    C: GlvParams,
{
    #[cfg(feature = "multicore")]
    if (MIN_PARALLEL_STRAUS_TERMS..=MAX_PARALLEL_STRAUS_TERMS).contains(&scalars.len())
        && num_threads > 1
    {
        // Bound small-MSM jobs to half the pool, rounded up, leaving room for
        // concurrent caller work. Keep each leaf small instead of building
        // one wide table for an entire worker's share.
        let worker_jobs = num_threads.div_ceil(2).min(scalars.len());
        if worker_jobs == 1 {
            return Some(straus_multiexp_leaves::<C>(scalars, bases));
        }
        return Some(
            (0..worker_jobs)
                .into_par_iter()
                .map(|job| {
                    let start = scalars.len() * job / worker_jobs;
                    let end = scalars.len() * (job + 1) / worker_jobs;
                    straus_multiexp_leaves::<C>(&scalars[start..end], &bases[start..end])
                })
                .sum(),
        );
    }

    let max_terms = if num_threads <= 1 {
        SERIAL_STRAUS_MAX_TERMS
    } else {
        STRAUS_LEAF_TERMS
    };
    if scalars.len() <= max_terms {
        return Some(straus_multiexp::<C>(scalars, bases));
    }
    None
}
