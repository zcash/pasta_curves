//! Signed-Booth recoding of the GLV half-scalars into window digits.

#[cfg(doc)]
use crate::glv::Decomposed;

#[cfg(test)]
mod tests;

/// Each GLV decomposition component has magnitude strictly below `2^127`.
pub(super) const GLV_COMPONENT_BITS: usize = 127;

/// The number of signed-Booth windows of width `window_bits` that the MSM
/// evaluates for a GLV half-scalar, the highest one included.
pub(super) fn window_count(window_bits: usize) -> usize {
    GLV_COMPONENT_BITS / window_bits + 1
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SignedMagnitude {
    negative: bool,
    magnitude: u128,
}

impl From<(bool, u128)> for SignedMagnitude {
    fn from((negative, magnitude): (bool, u128)) -> Self {
        Self {
            negative,
            magnitude,
        }
    }
}

/// Converts decomposed halves into the representation used by the
/// Signed-Booth MSM, rejecting values outside the decomposition bound.
///
/// The large MSM does not construct a [`Decomposed`], so check the bound
/// before extracting digits. Declining lets the caller retain its generic MSM.
pub(super) fn checked_signed_magnitudes(
    (first, second): ((bool, u128), (bool, u128)),
) -> Option<(SignedMagnitude, SignedMagnitude)> {
    if first.1 >> GLV_COMPONENT_BITS == 0 && second.1 >> GLV_COMPONENT_BITS == 0 {
        Some((first.into(), second.into()))
    } else {
        None
    }
}

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
