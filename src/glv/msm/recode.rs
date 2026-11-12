//! Recoding the terms' scalars into bucket digits.
//!
//! The bucket method needs, for every term and window, the bucket the term's
//! point goes into and how the point is transformed on the way. A
//! [`Recoding`] supplies that, together with the weights its buckets carry.
//! [`SignedBooth`] is the only one: signed digits in
//! `{-2^(w-1), ..., 2^(w-1)}`, a section of the evaluation map from digit
//! strings to the integers, where a negative digit only negates its point.
//!
//! A recoding over the Eisenstein integers, with digits a unit of `mu_6`
//! times an orbit representative, fits the same interface with two changes:
//! the transform becomes the unit's action `(x, y) -> (zeta^i x, +-y)`
//! instead of a negation, and [`Recoding::sum_buckets`] weights each bucket
//! by its representative instead of by `bucket + 1`.

use super::MsmCurve;
use super::affine::AffinePoint;

/// An integer scalar as a sign and a magnitude.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SignedMagnitude {
    pub(super) negative: bool,
    pub(super) magnitude: u128,
}

impl From<(bool, u128)> for SignedMagnitude {
    fn from((negative, magnitude): (bool, u128)) -> Self {
        Self {
            negative,
            magnitude,
        }
    }
}

/// Where a term's point goes in one window: into bucket `bucket`, negated
/// first when `negate`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BucketDigit {
    pub(super) bucket: usize,
    pub(super) negate: bool,
}

/// A digit system for the bucket method: how a scalar splits into windows,
/// and each window's digit into a bucket and a transform of the point.
pub(super) trait Recoding {
    /// The scalars it recodes.
    type Scalar: Copy;

    /// Buckets per window of `window_bits` bits.
    fn bucket_count(window_bits: usize) -> usize;

    /// Windows of `window_bits` bits covering scalars below `2^scalar_bits`.
    fn window_count(scalar_bits: usize, window_bits: usize) -> usize;

    /// The digit of `scalar` in `window`, or `None` when it is zero.
    fn digit(scalar: Self::Scalar, window_bits: usize, window: usize) -> Option<BucketDigit>;

    /// The window's value from its reduced buckets: the sum of each bucket's
    /// point times the bucket's weight.
    fn sum_buckets<C>(buckets: &[Option<AffinePoint<C::Base>>]) -> C
    where
        C: MsmCurve;
}

/// Signed-Booth recoding of [`SignedMagnitude`] integers. Bucket `b` holds
/// the terms whose digit is `+-(b + 1)`.
pub(super) struct SignedBooth;

impl Recoding for SignedBooth {
    type Scalar = SignedMagnitude;

    fn bucket_count(window_bits: usize) -> usize {
        1 << (window_bits - 1)
    }

    /// One window more than `scalar_bits / window_bits`, for the final carry.
    fn window_count(scalar_bits: usize, window_bits: usize) -> usize {
        scalar_bits / window_bits + 1
    }

    fn digit(scalar: SignedMagnitude, window_bits: usize, window: usize) -> Option<BucketDigit> {
        let digit = booth_digit(scalar, window_bits, window);
        (digit.magnitude != 0).then(|| BucketDigit {
            bucket: digit.magnitude - 1,
            negate: digit.negative,
        })
    }

    /// `sum_b (b + 1) * S_b` by summation by parts: a running suffix sum of the
    /// buckets, added into the total once per bucket.
    fn sum_buckets<C>(buckets: &[Option<AffinePoint<C::Base>>]) -> C
    where
        C: MsmCurve,
    {
        let mut running = C::identity();
        let mut sum = C::identity();
        for bucket in buckets.iter().rev() {
            if let Some(point) = bucket {
                running += C::affine_unchecked(point.x, point.y);
            }
            sum += running;
        }
        sum
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BoothDigit {
    pub(super) magnitude: usize,
    pub(super) negative: bool,
}

/// Extracts one signed-Booth digit and folds in the scalar's overall sign.
pub(super) fn booth_digit(
    component: SignedMagnitude,
    window_bits: usize,
    window: usize,
) -> BoothDigit {
    let window_start = window * window_bits;
    let radix = 1usize << window_bits;
    let value = if window_start < 128 {
        // Only the low window_bits bits survive; widths are below usize::BITS.
        ((component.magnitude >> window_start) as usize) & (radix - 1)
    } else {
        0
    };
    let overlap = if window_start == 0 {
        0
    } else {
        // The masked bit is either zero or one, which always fits usize.
        ((component.magnitude >> (window_start - 1)) & 1) as usize
    };

    let (magnitude, digit_negative) = if value < radix / 2 {
        (value + overlap, false)
    } else {
        let magnitude = radix - value - overlap;
        (magnitude, magnitude != 0)
    };
    BoothDigit {
        magnitude,
        negative: magnitude != 0 && (digit_negative ^ component.negative),
    }
}
