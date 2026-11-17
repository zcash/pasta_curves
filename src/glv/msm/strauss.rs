//! Small sums: one Strauss doubling chain shared by all terms, using the
//! split-wNAF tables of `glv`, optionally split into parallel leaves.

use alloc::vec::Vec;
use group::CurveAffine as _;
#[cfg(feature = "multicore")]
use maybe_rayon::prelude::*;

use super::plan::SERIAL_STRAUSS_MAX_TERMS;
use crate::glv::{Decomposed, GlvParams, Table};

/// Largest GLV-Strauss leaf when MSMs are split across worker threads.
const STRAUSS_LEAF_TERMS: usize = 16;

/// Smallest MSM worth splitting across worker jobs.
#[cfg(feature = "multicore")]
const MIN_PARALLEL_STRAUSS_TERMS: usize = 3;

/// Largest parallel MSM split into [`STRAUSS_LEAF_TERMS`]-or-smaller Strauss
/// leaves. Above this conservative crossover, the generic or GLV bucket
/// backends amortize their bucket setup more effectively.
#[cfg(feature = "multicore")]
const MAX_PARALLEL_STRAUSS_TERMS: usize = 66;

/// Share the doubling ladder across all scalar/base pairs, reusing the
/// upstream GLV decomposition and width-four split-wNAF tables unchanged.
fn strauss_multiexp<C: GlvParams>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> C {
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
fn strauss_multiexp_leaves<C: GlvParams>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> C {
    scalars
        .chunks(STRAUSS_LEAF_TERMS)
        .zip(bases.chunks(STRAUSS_LEAF_TERMS))
        .map(|(scalars, bases)| strauss_multiexp::<C>(scalars, bases))
        .fold(C::identity(), |left, right| left + right)
}

pub(super) fn planned_strauss_multiexp<C: GlvParams>(
    scalars: &[C::ScalarExt],
    bases: &[C::AffineExt],
    num_threads: usize,
) -> Option<C> {
    #[cfg(feature = "multicore")]
    if (MIN_PARALLEL_STRAUSS_TERMS..=MAX_PARALLEL_STRAUSS_TERMS).contains(&scalars.len())
        && num_threads > 1
    {
        // Bound small-MSM jobs to half the pool, rounded up, leaving room for
        // concurrent caller work. Keep each leaf small instead of building
        // one wide table for an entire worker's share.
        let worker_jobs = num_threads.div_ceil(2).min(scalars.len());
        if worker_jobs == 1 {
            return Some(strauss_multiexp_leaves::<C>(scalars, bases));
        }
        return Some(
            (0..worker_jobs)
                .into_par_iter()
                .map(|job| {
                    let start = scalars.len() * job / worker_jobs;
                    let end = scalars.len() * (job + 1) / worker_jobs;
                    strauss_multiexp_leaves::<C>(&scalars[start..end], &bases[start..end])
                })
                .sum(),
        );
    }

    let max_terms = if num_threads <= 1 {
        SERIAL_STRAUSS_MAX_TERMS
    } else {
        STRAUSS_LEAF_TERMS
    };
    if scalars.len() <= max_terms {
        return Some(strauss_multiexp::<C>(scalars, bases));
    }
    None
}
