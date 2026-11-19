//! Variable-time GLV MSMs for public scalars and bases.
//!
//! This is an accelerator in front of a generic MSM, not a replacement for it:
//! call `C::try_multiexp_vartime(s, b).unwrap_or_else(|| generic_msm(s, b))`.
//! Always try the hook first; its planner chooses by input shape, so callers
//! need no size threshold. Every `Some` equals any correct MSM on the same inputs.
//!
//! With `glv`, the Pallas and Vesta endomorphisms lift N full-width terms to
//! 2N half-width terms. Large sums use signed-Booth buckets with affine additions
//! sharing one inversion per batch. Small sums share a Straus doubling chain
//! using the existing split-wNAF tables. With `multicore`, large sums split
//! independent windows across threads and small sums split their terms into
//! Straus leaves. WebAssembly without atomics stays serial. Without `glv`,
//! and for other curves, the default implementation always declines.
//!
//! Both scalars and bases must be public. Generic bucket MSMs also commonly
//! skip zero digits and index buckets by digit; neither is suitable for secret
//! inputs or a constant-time MSM. Ordinary constant-time multiplication is unchanged.
//!
//! Window assignment and affine-reduction buffers are reused across windows.
//! With `multicore`, each thread also caches at most 8 MiB per field between
//! calls; checked-out buffers are unavailable to nested Rayon work.
//!
//! # Reading guide
//!
//! [`try_multiexp`] is the entry point. Two methods share the work, because
//! neither wins at every size: Straus ([`straus`]) pays one addition per
//! nonzero digit of every term but has almost no setup, while the bucket
//! method ([`bucket`]) amortizes additions over many terms at the price of
//! setting up its buckets. So `try_multiexp` asks the planner ([`plan`]) for a
//! `Plan` and matches on it: Straus for small inputs, the bucket method at the
//! planned window width, or a decline when the caller's generic MSM is
//! expected to be faster. The bucket method splits the scalars as in
//! [`crate::glv`], recodes the halves into signed digits ([`booth`]), and
//! evaluates the windows.
//!
//! # Layers
//!
//! | layer                    | role                                   | module                 |
//! |--------------------------|----------------------------------------|------------------------|
//! | strategy                 | Straus, buckets, or the generic MSM    | [`plan`]               |
//! | GLV split                | two halves below $2^{127}$ per scalar  | [`crate::glv`]         |
//! | signed-Booth digits      | $2^{w-1}$ buckets per window           | [`booth`]              |
//! | bucket method            | buckets, their sums, a running sum     | [`bucket`]             |
//! | batched affine additions | one inversion per tree level           | `bucket::affine`       |
//! | parallelism              | with `multicore`: per window, per leaf | [`bucket`], [`straus`] |
//! | memory                   | bounded blocks, reused buffers, cache  | `bucket::cache`        |
//!
//! The generic MSM this falls back to (`best_multiexp` in `halo2_proofs`) is
//! itself a signed-Booth bucket MSM. The gain over it comes from the GLV split,
//! which halves the number of windows and so the per-window bucket setup, and
//! from the batched affine additions, which make bucket sums cheaper than
//! projective additions. The planner takes over only when its estimate says
//! these win.
//!
//! The work is shared at three levels: one doubling chain across all windows
//! (Horner's rule), a running sum within a window that weights each bucket
//! without multiplying, and one inversion for all the divisions of a tree
//! level. Straus's method shares the first level across terms directly, for
//! inputs too small for buckets to pay off.
//!
//! Not covered: bases reused across calls are not precomputed, and the running
//! sum stays in projective coordinates.

mod bucket;
mod calibration;
mod plan;
mod straus;

use alloc::vec::Vec;

use super::booth;
use bucket::{CachedField, multiexp, multiexp_bases};
use plan::{Plan, current_num_threads};
use straus::straus_multiexp;
#[cfg(feature = "multicore")]
use straus::straus_multiexp_jobs;

pub(crate) use bucket::{AffinePoint, reduce_affine_buckets};

use super::{GlvParams, decompose, private::AffineBase};

/// Declines unprofitable input shapes and failed arithmetic guards so callers
/// can retain their generic MSM. Mismatched lengths also decline.
pub(crate) fn try_multiexp<C>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> Option<C>
where
    C: GlvParams,
    AffineBase<C>: CachedField,
{
    if scalars.len() != bases.len() {
        return None;
    }
    let num_threads = current_num_threads();
    let window_bits = match plan::plan::<C>(scalars.len(), num_threads) {
        Plan::Straus => return Some(straus_multiexp::<C>(scalars, bases)),
        #[cfg(feature = "multicore")]
        Plan::StrausLeaves { jobs } => {
            return Some(straus_multiexp_jobs::<C>(scalars, bases, jobs));
        }
        Plan::Buckets { window_bits } => window_bits,
        Plan::Decline => return None,
    };
    let components = scalars
        .iter()
        .map(decompose::<C>)
        .map(booth::checked_signed_magnitudes)
        .collect::<Option<Vec<_>>>();
    // `glv::decompose` keeps both halves below 2^127, so this never declines;
    // if that bound broke, release builds would still fall back safely.
    debug_assert!(components.is_some(), "a GLV half is 2^127 or larger");
    let components = components?;
    let bases = multiexp_bases::<C>(bases);
    multiexp(&components, &bases, window_bits, num_threads)
}
