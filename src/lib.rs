//! Implementation of the Pallas / Vesta curve cycle.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(unknown_lints)]
#![allow(
    clippy::op_ref,
    clippy::too_many_arguments,
    clippy::suspicious_arithmetic_impl,
    clippy::same_item_push,
    clippy::upper_case_acronyms,
    clippy::unknown_clippy_lints
)]
#![deny(broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unsafe_code)]

#[macro_use]
mod macros;
mod curves;
mod fields;

pub mod arithmetic;
mod hashtocurve;
pub mod pallas;
pub mod vesta;

pub use curves::*;
pub use fields::*;

#[test]
fn test_endo_consistency() {
    use crate::arithmetic::{CurveExt, FieldExt};
    use group::Group;

    let a = pallas::Point::generator();
    assert_eq!(a * pallas::Scalar::ZETA, a.endo());
    let a = vesta::Point::generator();
    assert_eq!(a * vesta::Scalar::ZETA, a.endo());
}
