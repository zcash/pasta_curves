//! The Vesta and iso-Vesta elliptic curve groups.

use super::{Eq, EqAffine, Fp, Fq};

/// The base field of the Vesta and iso-Vesta curves.
pub type Base = Fq;

/// The scalar field of the Vesta and iso-Vesta curves.
pub type Scalar = Fp;

/// A Vesta point in Jacobian coordinates.
pub type Point = Eq;

/// A Vesta point in the affine coordinate space (or the point at infinity).
pub type Affine = EqAffine;
