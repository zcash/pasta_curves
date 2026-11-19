//! The test suite that zkcrypto/group provides for implementations of its traits,
//! run on the Pallas and Vesta point types.
//!
//! It checks the group laws on random points (identity, associativity and
//! commutativity over every order of three additions, doubling, negation, the
//! linearity of scalar multiplication), the maps between the Jacobian and affine
//! types (`to_affine`, `to_curve`, `batch_normalize`), the compressed encoding, and,
//! with `alloc`, the wNAF multiplication. Linearity does not fix which
//! multiplication is computed, so the crate's own tests also compare `Mul` with an
//! independent ladder.

use pasta_curves::{pallas, vesta};

#[test]
fn pallas_curve_tests() {
    group::tests::curve_tests::<pallas::Point>();
}

#[test]
fn vesta_curve_tests() {
    group::tests::curve_tests::<vesta::Point>();
}

#[cfg(feature = "alloc")]
#[test]
fn pallas_wnaf_tests() {
    group::tests::random_wnaf_tests::<pallas::Point>();
}

#[cfg(feature = "alloc")]
#[test]
fn vesta_wnaf_tests() {
    group::tests::random_wnaf_tests::<vesta::Point>();
}
