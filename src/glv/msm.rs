//! Variable-time GLV MSMs for public scalars and bases.
//!
//! This is an accelerator in front of a generic MSM, not a replacement for it:
//! call `C::try_multiexp_vartime(s, b).unwrap_or_else(|| generic_msm(s, b))`.
//! Always try the hook first; its planner chooses by input shape, so callers
//! need no size threshold. Every `Some` equals any correct MSM on the same inputs.
//!
//! With `glv`, the Pallas and Vesta endomorphisms lift N full-width terms to
//! 2N half-width terms. Large sums use signed-Booth buckets with affine additions
//! sharing one inversion per batch. Small sums share a Strauss doubling chain
//! using the existing split-wNAF tables. With `multicore`, large sums split
//! independent windows across threads and small sums split their terms into
//! Strauss leaves. WebAssembly without atomics stays serial. Without `glv`,
//! and for other curves, the default implementation always declines.
//!
//! Both scalars and bases must be public. Generic bucket MSMs also commonly
//! skip zero digits and index buckets by digit; neither is suitable for secret
//! inputs or a constant-time MSM. Ordinary constant-time multiplication is unchanged.
//!
//! Window assignment and affine-reduction buffers are reused across windows.
//! With `multicore`, each thread also caches at most 8 MiB per field between
//! calls; checked-out buffers are unavailable to nested Rayon work.

mod booth;
mod bucket;
mod plan;
mod strauss;

use alloc::vec::Vec;

use booth::checked_signed_magnitudes;
use bucket::{CachedField, multiexp, multiexp_bases};
use plan::{current_num_threads, glv_multiexp_window_bits};
use strauss::planned_strauss_multiexp;

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
    if let Some(result) = planned_strauss_multiexp::<C>(scalars, bases, num_threads) {
        return Some(result);
    }
    let window_bits = glv_multiexp_window_bits::<C>(scalars.len(), num_threads)?;
    let components = scalars
        .iter()
        .map(decompose::<C>)
        .map(checked_signed_magnitudes)
        .collect::<Option<Vec<_>>>()?;
    let bases = multiexp_bases::<C>(bases);
    multiexp(&components, &bases, window_bits, num_threads)
}
