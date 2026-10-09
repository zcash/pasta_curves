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

// The limb type of the backends and the inversion, the checks of their debug assertions, and the
// `unroll!` macro, which its users import by path.
mod limbs;

// The constant-time inversion, over the AArch64 assembly blocks or the portable ones.
mod inversion;

// The Montgomery arithmetic as generic compositions over a backend's blocks.
if_asm_supported! {
    mod montgomery;
}

// The fields' constants and known answers for the tests of the backends and the inversion.
#[cfg(test)]
mod test_fields;

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
    // The backend's condition, restated: the feature, the target, and the absence of the opt-out
    // flag.
    let supported = cfg!(all(
        feature = "asm",
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

// The point of these tests is that they compile: each macro is used in every position the backend's
// users need (an item, a `let` binding, and a block inside a function), so the assertions are
// trivially true by design.
#[cfg(feature = "asm")]
#[cfg(test)]
mod asm_gate_tests {
    if_asm_supported! {
        fn backend_name() -> &'static str { crate::BACKEND }
    }
    if_asm_unsupported! {
        fn backend_name() -> &'static str { "portable" }
    }

    fn selected_backend() -> bool {
        if_asm_supported! {{ crate::BACKEND != "portable" }}
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
