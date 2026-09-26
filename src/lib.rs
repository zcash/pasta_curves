//! Implementation of the Pallas / Vesta curve cycle.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(unknown_lints)]
#![allow(clippy::op_ref, clippy::same_item_push, clippy::upper_case_acronyms)]
#![deny(rustdoc::broken_intra_doc_links)]
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

// We cannot build the assembly on Apple x86-64 targets because they reserve `rbp`, and so have
// fewer available registers than the squaring blocks need. So the module is absent on those
// targets, as on every other target without a backend.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", not(target_vendor = "apple")),
    doc
))]
mod asm;

pub mod arithmetic;
#[cfg(feature = "deferred")]
#[cfg_attr(docsrs, doc(cfg(feature = "deferred")))]
pub mod deferred;
pub mod pallas;
pub mod vesta;

#[cfg(feature = "glv")]
#[cfg_attr(docsrs, doc(cfg(feature = "glv")))]
pub mod glv;

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
