//! The GLV-Strauss ladder for small MSMs.
//!
//! One doubling chain is shared by every term, and each term adds digits from
//! its own table of odd multiples, so the precomputation is per point rather
//! than per window as in the bucket method.

use alloc::vec::Vec;
use group::CurveAffine as _;
#[cfg(feature = "multicore")]
use maybe_rayon::prelude::*;

use super::MsmCurve;
use crate::glv::{Decomposed, Table};

/// Largest GLV-Strauss leaf when MSMs are split across worker threads: each
/// leaf builds one table per point and shares one doubling chain.
pub(super) const STRAUSS_LEAF_TERMS: usize = 16;

/// Share the doubling ladder across all scalar/base pairs, reusing the
/// upstream GLV decomposition and width-four split-wNAF tables unchanged.
pub(super) fn strauss_multiexp<C>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> C
where
    C: MsmCurve,
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

pub(super) fn strauss_multiexp_leaves<C>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> C
where
    C: MsmCurve,
{
    scalars
        .chunks(STRAUSS_LEAF_TERMS)
        .zip(bases.chunks(STRAUSS_LEAF_TERMS))
        .map(|(scalars, bases)| strauss_multiexp::<C>(scalars, bases))
        .fold(C::identity(), |left, right| left + right)
}

/// Splits the terms into `jobs` contiguous shares of Strauss leaves, run on
/// worker threads with `multicore` and one after another without it.
pub(super) fn parallel_strauss_multiexp<C, J>(
    scalars: &[C::ScalarExt],
    bases: &[C::AffineExt],
    jobs: usize,
    on_job: &J,
) -> C
where
    C: MsmCurve,
    J: Fn() + Sync,
{
    let job = |job: usize| {
        on_job();
        let start = scalars.len() * job / jobs;
        let end = scalars.len() * (job + 1) / jobs;
        strauss_multiexp_leaves::<C>(&scalars[start..end], &bases[start..end])
    };
    #[cfg(feature = "multicore")]
    if jobs > 1 {
        return (0..jobs).into_par_iter().map(job).sum();
    }
    (0..jobs).map(job).sum()
}
