//! The GLV lift: an MSM over the scalar field as one over the integers.
//!
//! The curve is a module over the Eisenstein integers `Z[omega]`, and
//! restricting scalars along `Z -> Z[omega]` turns each term into two terms
//! with half-width integer scalars (see [`GlvTerms`]). The decomposition is a
//! section of `Z^2 -> Z/n`, not a homomorphism; its law is that the halves
//! recombine to the scalar and stay below `2^127`.

use alloc::vec::Vec;
use ff::WithSmallOrderMulGroup;
use group::CurveAffine as _;

use super::MsmCurve;
use super::affine::AffinePoint;
use super::buckets::SmallScalarTerms;
use crate::glv::decompose;

/// Each GLV decomposition component has magnitude strictly below `2^127`.
pub(super) const GLV_COMPONENT_BITS: usize = 127;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SignedMagnitude {
    pub(super) negative: bool,
    pub(super) magnitude: u128,
}

impl From<(bool, u128)> for SignedMagnitude {
    fn from((negative, magnitude): (bool, u128)) -> Self {
        Self {
            negative,
            magnitude,
        }
    }
}

/// Converts decomposed halves into the representation used by the
/// Signed-Booth MSM, rejecting values outside the decomposition bound.
///
/// The large MSM does not construct a [`Decomposed`](crate::glv::Decomposed), so check the bound
/// before extracting digits. Declining lets the caller retain its generic MSM.
pub(super) fn checked_signed_magnitudes(
    (first, second): ((bool, u128), (bool, u128)),
) -> Option<(SignedMagnitude, SignedMagnitude)> {
    if first.1 >> GLV_COMPONENT_BITS == 0 && second.1 >> GLV_COMPONENT_BITS == 0 {
        Some((first.into(), second.into()))
    } else {
        None
    }
}

/// One base with its endomorphism image, `phi(x, y) = (zeta * x, y)`, which
/// shares `y`.
#[derive(Clone, Copy)]
pub(super) struct MultiexpBase<F> {
    pub(super) x: F,
    pub(super) endo_x: F,
    pub(super) y: F,
    pub(super) identity: bool,
}

/// The GLV lift of an MSM over the scalar field to one over the integers.
///
/// The curve is a module over the Eisenstein integers `Z[omega]`, with `omega`
/// acting as `phi = lambda`, and `Z[omega]` is free over `Z` with basis
/// `{1, omega}`. Restricting scalars along `Z -> Z[omega]` turns each term
/// `k * P` into the two integer terms `k1 * P` and `k2 * phi(P)`, where
/// `k = k1 + k2 * lambda` with `|k1|, |k2| < 2^127` ([`decompose`]). Term
/// `2 i` is `(k1_i, P_i)` and term `2 i + 1` is `(k2_i, phi(P_i))`.
pub(super) struct GlvTerms<F> {
    pub(super) components: Vec<(SignedMagnitude, SignedMagnitude)>,
    pub(super) bases: Vec<MultiexpBase<F>>,
}

impl<F> SmallScalarTerms<F> for GlvTerms<F>
where
    F: Copy + Sync,
{
    fn len(&self) -> usize {
        2 * self.bases.len()
    }

    fn scalar(&self, term: usize) -> SignedMagnitude {
        let (first, second) = self.components[term / 2];
        if term % 2 == 0 { first } else { second }
    }

    fn point(&self, term: usize) -> Option<AffinePoint<F>> {
        let base = self.bases[term / 2];
        (!base.identity).then_some(AffinePoint {
            x: if term % 2 == 0 { base.x } else { base.endo_x },
            y: base.y,
        })
    }
}

/// Lifts an MSM to [`GlvTerms`], or declines when a decomposition half is
/// out of bounds.
pub(super) fn glv_lift<C>(
    scalars: &[C::ScalarExt],
    bases: &[C::AffineExt],
) -> Option<GlvTerms<C::Base>>
where
    C: MsmCurve,
{
    let components = scalars
        .iter()
        .map(decompose::<C>)
        .map(checked_signed_magnitudes)
        .collect::<Option<Vec<_>>>()?;
    Some(GlvTerms {
        components,
        bases: multiexp_bases::<C>(bases),
    })
}

pub(super) fn multiexp_bases<C>(bases: &[C::AffineExt]) -> Vec<MultiexpBase<C::Base>>
where
    C: MsmCurve,
{
    bases
        .iter()
        .map(|base| {
            let (x, y) = C::affine_xy(base);
            MultiexpBase {
                x,
                endo_x: x * C::Base::ZETA,
                y,
                identity: bool::from(base.is_identity()),
            }
        })
        .collect()
}
