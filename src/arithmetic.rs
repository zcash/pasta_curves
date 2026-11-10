//! This module provides common utilities, traits and structures for group and
//! field arithmetic.
//!
//! This module is temporary, and the extension traits defined here are expected to be
//! upstreamed into the `ff` and `group` crates after some refactoring.

mod curves;
mod fields;

#[cfg(all(
    feature = "asm",
    feature = "sqrt-table",
    target_arch = "x86_64",
    not(pasta_curves_noasm)
))]
pub(crate) mod avx2_sqrt;

#[cfg(all(
    feature = "glv",
    feature = "asm",
    feature = "sqrt-table",
    target_arch = "x86_64",
    not(pasta_curves_noasm)
))]
pub(crate) mod avx2_chord;

#[cfg(feature = "alloc")]
pub use curves::*;
pub use fields::*;
