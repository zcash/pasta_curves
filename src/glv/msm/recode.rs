//! Signed-Booth recoding of the integer scalars.
//!
//! A section of the evaluation map from digit strings in
//! `{-2^(w-1), ..., 2^(w-1)}` to the integers: the digits evaluate back to
//! the scalar, and a negative digit only negates the point it multiplies.

use super::lift::SignedMagnitude;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BoothDigit {
    pub(super) magnitude: usize,
    pub(super) negative: bool,
}

/// Extracts one signed-Booth digit and folds in the component's overall sign.
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
