//! This module provides common utilities, traits and structures for group and
//! field arithmetic.
//!
//! This module is temporary, and the extension traits defined here are expected to be
//! upstreamed into the `ff` and `group` crates after some refactoring.

mod curves;
mod fields;

#[cfg(feature = "alloc")]
pub use curves::*;
pub use fields::*;

/// Squares an [`Fp`](crate::Fp) element `count` times.
///
/// Returns the input unchanged when `count` is zero. The result is canonical;
/// runtime depends on the public square count, not the field value.
#[inline]
pub fn square_fp_n(value: &crate::Fp, count: u32) -> crate::Fp {
    if count == 0 {
        *value
    } else {
        value.sqr_n_runtime(count)
    }
}

/// Squares an [`Fq`](crate::Fq) element `count` times.
///
/// Returns the input unchanged when `count` is zero. The result is canonical;
/// runtime depends on the public square count, not the field value.
#[inline]
pub fn square_fq_n(value: &crate::Fq, count: u32) -> crate::Fq {
    if count == 0 {
        *value
    } else {
        value.sqr_n_runtime(count)
    }
}
