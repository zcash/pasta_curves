//! Implementation of the Pallas / Vesta curve cycle.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
// The first two concern the library's public API, so they cannot move to the `[lints]`
// table in Cargo.toml, which would also apply them to the benches. `unsafe_code` is
// denied there as well, so that it covers every target; it is repeated here because
// "this crate contains no unsafe code" is a property a reader should see in the source.
// The rest of the lint configuration lives in Cargo.toml.
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(test)]
#[macro_use]
extern crate std;

#[macro_use]
mod macros;
mod curves;
mod fields;

pub mod arithmetic;
#[cfg(feature = "deferred")]
#[cfg_attr(docsrs, doc(cfg(feature = "deferred")))]
pub mod deferred;
pub mod pallas;
pub mod vesta;

#[cfg(feature = "alloc")]
mod hashtocurve;

#[cfg(feature = "serde")]
mod serde_impl;

pub use curves::*;
pub use fields::*;

pub extern crate group;

#[cfg(feature = "alloc")]
#[test]
fn test_endo_consistency() {
    use crate::arithmetic::CurveExt;
    use group::{Group, ff::WithSmallOrderMulGroup};

    let a = pallas::Point::generator();
    assert_eq!(a * pallas::Scalar::ZETA, a.endo());
    let a = vesta::Point::generator();
    assert_eq!(a * vesta::Scalar::ZETA, a.endo());
}
