//! Signed-Booth recoding of the GLV half-scalars into window digits.
//!
//! The bucket MSM and the prepared fixed-base sums evaluate every half-scalar
//! $m$ window by window, in base $2^w$. A plain base-$2^w$ digit lies in
//! $[0, 2^w)$ and would need $2^w - 1$ buckets. Signed digits halve that:
//! because $-Q$ costs nothing on a curve (negate $y$), a digit $d$ with
//! $|d| \le 2^{w-1}$ needs only the $2^{w-1}$ buckets indexed by $|d|$.
//!
//! # The recoding
//!
//! For window $k$, let $v_k \in [0, 2^w)$ be the bits $[kw, kw + w)$ of $m$, and
//! $c_k$ the bit just below them ($c_0 = 0$). The digit is
//!
//! $$d_k = v_k + c_k - 2^w c_{k+1}, \qquad c_{k+1} = [v_k \ge 2^{w-1}],$$
//!
//! so $|d_k| \le 2^{w-1}$. The carry $c_{k+1}$ out of window $k$ is the top bit
//! of window $k$, which is exactly the bit below window $k + 1$. The sum
//! therefore telescopes:
//!
//! $$\sum_{k=0}^{K} d_k 2^{kw} = m - c_{K+1} 2^{(K+1)w}.$$
//!
//! With $K + 1 =$ [`window_count`]`(w)` $= \lfloor 127 / w \rfloor + 1$, the
//! last window ends at bit $(K + 1)w - 1 \ge 127$, and $c_{K+1}$ is that bit,
//! which is zero because $m < 2^{127}$. So the digits sum to $m$ exactly.
//!
//! This is the radix-$2^w$ form of Booth's recoding (A. D. Booth, "A signed
//! binary multiplication technique", Q. J. Mech. Appl. Math. 4(2), 1951).
//!
//! # Signs
//!
//! A GLV half-scalar carries its own sign ([`SignedMagnitude`]); the recoding
//! works on the magnitude and folds the sign into each digit, so that the
//! digits of $-m$ are the negated digits of $m$.
//!
//! # Reading guide
//!
//! [`checked_signed_magnitudes`] turns the halves `glv::decompose` returns into
//! [`SignedMagnitude`]s, checking the bound [`HALF_BITS`]. The bucket
//! backend then asks [`booth_digit`] for the digit of each of the
//! [`window_count`] windows a half-scalar needs.

#[cfg(doc)]
use crate::glv::Decomposed;
use crate::glv::HALF_BITS;

#[cfg(test)]
mod tests;

/// The number of signed-Booth windows of width `window_bits` that the MSM
/// evaluates for a GLV half-scalar, the highest one included.
///
/// It suffices: the last window ends at bit
/// `window_count(w) * w - 1 >= HALF_BITS`, which is zero for a
/// half-scalar below `2^127`, so no carry leaves it and the digits of windows
/// `0..window_count(w)` sum to the half-scalar exactly (see the module
/// documentation). Requires `window_bits >= 1`.
pub(super) const fn window_count(window_bits: usize) -> usize {
    HALF_BITS / window_bits + 1
}

/// A GLV half-scalar $(-1)^{\text{negative}} \cdot \text{magnitude}$, in the
/// sign-magnitude form `glv::decompose` returns. The MSM only builds values
/// with `magnitude < 2^127` (see [`checked_signed_magnitudes`]).
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
/// signed-Booth MSM, rejecting values outside the decomposition bound.
///
/// The large MSM does not construct a [`Decomposed`], so check the bound
/// before extracting digits. Declining lets the caller retain its generic MSM.
pub(super) fn checked_signed_magnitudes(
    (first, second): ((bool, u128), (bool, u128)),
) -> Option<(SignedMagnitude, SignedMagnitude)> {
    if first.1 >> HALF_BITS == 0 && second.1 >> HALF_BITS == 0 {
        Some((first.into(), second.into()))
    } else {
        None
    }
}

/// A signed digit $(-1)^{\text{negative}} \cdot \text{magnitude}$ with
/// `magnitude <= 2^(w - 1)`. Zero is never negative, so equal digits have
/// equal representations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BoothDigit {
    pub(super) magnitude: usize,
    pub(super) negative: bool,
}

/// Extracts one signed-Booth digit and folds in the component's overall sign.
///
/// Returns $\pm d_k$, where $d_k$ is the digit of window `window` of
/// `component.magnitude` defined in the module documentation, negated when
/// `component.negative`. Requires `1 <= window_bits < usize::BITS` and
/// `window * window_bits <= u128::BITS`; the MSM calls it for
/// `window < window_count(window_bits)` only.
pub(super) fn booth_digit(
    component: SignedMagnitude,
    window_bits: usize,
    window: usize,
) -> BoothDigit {
    let window_start = window * window_bits;
    let radix = 1usize << window_bits;
    let value = if window_start < u128::BITS as usize {
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
