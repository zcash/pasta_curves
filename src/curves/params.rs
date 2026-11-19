//! The parameters of the curves: each curve is a type implementing
//! [`CurveParams`], which every model of the curve reads.

use super::{Fp, Fq};

/// The constant $b$ of Pallas and Vesta, $y^2 = x^3 + 5$. The projective model is
/// indexed by it.
pub(crate) const PASTA_B: i64 = 5;

/// A short Weierstrass curve $y^2 = x^3 + a x + b$ over `Self::Base`, and the
/// identifier of its hash-to-curve suite.
pub(crate) trait CurveParams {
    /// The base field.
    type Base;
    /// The coefficient $a$.
    const A: Self::Base;
    /// The constant $b$.
    const B: Self::Base;
    /// The curve identifier of [`crate::arithmetic::CurveExt::CURVE_ID`].
    const ID: &'static str;
}

/// Pallas, $y^2 = x^3 + 5$ over $\mathbb{F}_p$.
pub(crate) enum Pallas {}

/// Vesta, $y^2 = x^3 + 5$ over $\mathbb{F}_q$.
pub(crate) enum Vesta {}

/// The curve 3-isogenous to Pallas that hashing to Pallas maps to first.
pub(crate) enum IsoPallas {}

/// The curve 3-isogenous to Vesta that hashing to Vesta maps to first.
pub(crate) enum IsoVesta {}

impl CurveParams for Pallas {
    type Base = Fp;
    const A: Fp = Fp::zero();
    const B: Fp = Fp::from_raw([PASTA_B as u64, 0, 0, 0]);
    const ID: &'static str = "pallas";
}

impl CurveParams for Vesta {
    type Base = Fq;
    const A: Fq = Fq::zero();
    const B: Fq = Fq::from_raw([PASTA_B as u64, 0, 0, 0]);
    const ID: &'static str = "vesta";
}

impl CurveParams for IsoPallas {
    type Base = Fp;
    const A: Fp = Fp::from_raw([
        0x92bb4b0b657a014b,
        0xb74134581a27a59f,
        0x49be2d7258370742,
        0x18354a2eb0ea8c9c,
    ]);
    const B: Fp = Fp::from_raw([1265, 0, 0, 0]);
    const ID: &'static str = "iso-pallas";
}

impl CurveParams for IsoVesta {
    type Base = Fq;
    const A: Fq = Fq::from_raw([
        0xc515ad7242eaa6b1,
        0x9673928c7d01b212,
        0x81639c4d96f78773,
        0x267f9b2ee592271a,
    ]);
    const B: Fq = Fq::from_raw([1265, 0, 0, 0]);
    const ID: &'static str = "iso-vesta";
}
