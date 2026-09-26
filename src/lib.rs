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

// The assembly backends, and the `if_asm_supported!` and `if_asm_unsupported!` macros that gate
// code on them; declared first so that the macros are in scope in the modules below.
#[macro_use]
mod asm;
pub use asm::BACKEND;

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

#[test]
fn backend_name() {
    // The backend's condition, restated: the target, and the absence of the opt-out flag.
    let supported = cfg!(all(
        not(pasta_curves_noasm),
        any(
            target_arch = "aarch64",
            all(
                target_arch = "x86_64",
                target_pointer_width = "64",
                not(target_vendor = "apple")
            )
        )
    ));
    let expected = if !supported {
        "portable"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "x86-64"
    };
    assert_eq!(BACKEND, expected);
}

#[cfg(test)]
mod asm_gate_tests {
    use crate::BACKEND;

    if_asm_supported! {
        fn backend_name() -> &'static str { BACKEND }
    }
    if_asm_unsupported! {
        fn backend_name() -> &'static str { "portable" }
    }

    fn selected_backend() -> bool {
        if_asm_supported! {{ BACKEND != "portable" }}
        if_asm_unsupported! {{ false }}
    }

    #[test]
    fn cfg_gated_local_bindings_and_blocks() {
        if_asm_supported! { let backend: &str = backend_name(); }
        if_asm_unsupported! { let backend: &str = backend_name(); }
        if_asm_supported! {{ assert_ne!(backend, "portable"); }}
        if_asm_unsupported! {{ assert_eq!(backend, "portable"); }}
        assert_eq!(backend == "portable", !selected_backend());
    }
}
