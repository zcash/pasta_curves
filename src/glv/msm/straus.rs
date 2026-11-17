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
//! constants below.
//!
//! E. G. Straus, "Addition chains of vectors" (solution to Advanced Problem
//! 5125, posed by R. Bellman), Amer. Math. Monthly 71 (1964), 806-808.
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
use crate::glv::{Decomposed, GlvParams, Recoding, Table, Wnaf4};

/// Largest GLV-Straus leaf when MSMs are split across worker threads: the
/// measured crossover above which one table stopped beating the bucket backend
/// on one thread.
const STRAUS_LEAF_TERMS: usize = 16;

/// Smallest MSM worth splitting across worker jobs.
#[cfg(feature = "multicore")]
const MIN_PARALLEL_STRAUS_TERMS: usize = 3;

/// Largest parallel MSM split into [`STRAUS_LEAF_TERMS`]-or-smaller Straus
/// leaves. Above this conservative crossover, the generic or GLV bucket
/// backends amortize their bucket setup more effectively.
///
/// Tuned for the halo2 IPA prover, whose last rounds commit to L and R with
/// 66-term MSMs. `benches/msm.rs` brackets it with 66 and 67 terms.
#[cfg(feature = "multicore")]
const MAX_PARALLEL_STRAUS_TERMS: usize = 66;

/// The parallel Straus path splits into at most the pool's size divided by
/// this, rounded up: the IPA prover computes its L and R commitments at the
/// same time, and each gets an equal share.
#[cfg(feature = "multicore")]
const STRAUS_POOL_SHARES: usize = 2;

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
    <Wnaf4 as Recoding<C>>::sum_of_products(&tables, &decomposed)
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
/// - with `multicore` and `num_threads > 1`, there are
///   `MIN_PARALLEL_STRAUS_TERMS` to `MAX_PARALLEL_STRAUS_TERMS` terms: they
///   are split into at most `ceil(num_threads / STRAUS_POOL_SHARES)` jobs of
///   Straus leaves of at most [`STRAUS_LEAF_TERMS`] terms, computed in
///   parallel and summed;
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
        // Bound small-MSM jobs to a share of the pool (STRAUS_POOL_SHARES),
        // leaving room for concurrent caller work. Keep each leaf small
        // instead of building one wide table for an entire worker's share.
        let worker_jobs = num_threads.div_ceil(STRAUS_POOL_SHARES).min(scalars.len());
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
