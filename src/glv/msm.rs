//! Variable-time GLV MSMs for public scalars and bases.
//!
//! With `glv`, `CurveExt::try_multiexp_vartime` offers a variable-time MSM backend
//! for public scalars and bases. It uses a shared-doubling GLV-Strauss ladder for
//! small inputs and signed-Booth GLV buckets with batched affine additions for
//! large inputs. It returns `None` when its cost model prefers the caller's
//! generic MSM; callers must keep that fallback. Enable `multicore` to evaluate
//! independent windows through the Rayon pool; WebAssembly without atomics uses
//! the serial implementation. Neither feature changes ordinary constant-time
//! scalar multiplication.
//!
//! The small-MSM ladder reuses the existing split-wNAF tables unchanged.
//!
//! # Structure
//!
//! An MSM evaluates the free module's map `(k_1, ..., k_N) -> sum k_i P_i`,
//! and every backend here factors that one map:
//!
//! ```text
//! (Z/n)^N --lift--> small integers^(2N) --recode--> signed digits --evaluate--> G
//! ```
//!
//! - [`lift`]: the GLV decomposition, restricting scalars from the Eisenstein
//!   integers to the integers, so `N` terms become `2N` with half-width scalars;
//! - [`recode`]: signed-Booth digits of each integer scalar;
//! - [`buckets`] and [`affine`]: the bucket method and its batched additions,
//!   combined across windows by Horner's rule;
//! - [`strauss`]: the alternative for small MSMs, one shared doubling chain;
//! - [`plan`](mod@plan): which of these to run, a pure function of the shape.
//!
//! [`execute`] carries out a plan; every plan but a decline computes the same
//! sum, which is the property the tests check across all plans.

use ff::Field;

use super::GlvParams;
use crate::arithmetic::CurveAffine;
use crate::{pallas, vesta};

mod affine;
mod buckets;
mod lift;
mod plan;
mod recode;
mod strauss;
#[cfg(test)]
mod tests;

use buckets::bucket_multiexp;
use lift::glv_lift;
use plan::{Plan, plan};
use strauss::{parallel_strauss_multiexp, strauss_multiexp};

// Keep raw-coordinate plumbing private rather than strengthening the public
// CurveExt associated-type bounds or exposing a generic unchecked constructor.
pub(crate) trait MsmCurve: GlvParams {
    fn affine_unchecked(x: Self::Base, y: Self::Base) -> Self::AffineExt;
    fn affine_xy(point: &Self::AffineExt) -> (Self::Base, Self::Base);
}

macro_rules! impl_msm_curve {
    ($curve:ident) => {
        impl MsmCurve for $curve::Point {
            fn affine_unchecked(x: Self::Base, y: Self::Base) -> Self::AffineExt {
                $curve::Affine::from_xy_unchecked(x, y)
            }

            fn affine_xy(point: &Self::AffineExt) -> (Self::Base, Self::Base) {
                Option::<crate::arithmetic::Coordinates<$curve::Affine>>::from(point.coordinates())
                    .map(|p| (*p.x(), *p.y()))
                    .unwrap_or((Self::Base::ZERO, Self::Base::ZERO))
            }
        }
    };
}
impl_msm_curve!(pallas);
impl_msm_curve!(vesta);

fn current_num_threads() -> usize {
    #[cfg(all(
        feature = "multicore",
        not(all(target_arch = "wasm32", not(target_feature = "atomics")))
    ))]
    {
        maybe_rayon::current_num_threads()
    }
    #[cfg(any(
        not(feature = "multicore"),
        all(target_arch = "wasm32", not(target_feature = "atomics"))
    ))]
    {
        1
    }
}

/// Carries out `plan`. Declines (`None`) for [`Plan::Decline`] and when an
/// arithmetic guard fails, so the caller's generic MSM runs instead.
fn execute<C>(plan: Plan, scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> Option<C>
where
    C: MsmCurve,
{
    execute_observed(plan, scalars, bases, &|| ())
}

/// [`execute`], calling `on_job` at the start of every parallel job. Tests use
/// it to check that parallel plans really run their jobs concurrently.
fn execute_observed<C, J>(
    plan: Plan,
    scalars: &[C::ScalarExt],
    bases: &[C::AffineExt],
    on_job: &J,
) -> Option<C>
where
    C: MsmCurve,
    J: Fn() + Sync,
{
    debug_assert_eq!(scalars.len(), bases.len());
    match plan {
        Plan::Decline => None,
        Plan::Strauss => Some(strauss_multiexp::<C>(scalars, bases)),
        Plan::ParallelStrauss { jobs } => Some(parallel_strauss_multiexp::<C, _>(
            scalars, bases, jobs, on_job,
        )),
        Plan::Buckets {
            window_bits,
            parallel,
        } => {
            let terms = glv_lift::<C>(scalars, bases)?;
            bucket_multiexp::<C, _, _>(&terms, window_bits, parallel, on_job)
        }
    }
}

/// Declines unprofitable input shapes and failed arithmetic guards so callers
/// can retain their generic MSM.
///
/// # Panics
///
/// Panics if `scalars` and `bases` have different lengths.
pub(crate) fn try_multiexp<C>(scalars: &[C::ScalarExt], bases: &[C::AffineExt]) -> Option<C>
where
    C: MsmCurve,
{
    assert_eq!(
        scalars.len(),
        bases.len(),
        "try_multiexp_vartime: scalars and bases differ in length"
    );
    execute(
        plan::<C>(scalars.len(), current_num_threads()),
        scalars,
        bases,
    )
}
