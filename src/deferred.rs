//! Deferred normalization for field arithmetic.
//!
//! This module provides the [`DeferredField`] trait, which enables accumulating
//! multiple unreduced products before performing a single expensive reduction.
//! This is useful for operations like inner products where many multiplications
//! feed into a sum.

use core::fmt::Debug;

/// A trait for fields that support deferred reduction of products.
///
/// Instead of reducing each multiplication result immediately, callers
/// accumulate products into an [`Accumulator`](Self::Accumulator) via
/// [`mul_accumulate`](Self::mul_accumulate) and
/// [`square_accumulate`](Self::square_accumulate), then perform a single
/// reduction at the end with [`reduce`](Self::reduce).
pub trait DeferredField: ff::Field {
    /// A wide accumulator for unreduced products.
    type Accumulator: Copy + Clone + Debug + Default;

    /// Multiplies `a` by `b` and adds the result into `acc`.
    fn mul_accumulate(acc: &mut Self::Accumulator, a: &Self, b: &Self);

    /// Squares `a` and adds the result into `acc`.
    fn square_accumulate(acc: &mut Self::Accumulator, a: &Self);

    /// Reduces the accumulator to a canonical field element.
    fn reduce(acc: Self::Accumulator) -> Self;
}
