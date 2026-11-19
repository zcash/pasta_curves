//! The dispatch of the curves' `Mul` impls (`mul_curve_specific!`): the radix-27
//! ladder of the projective model on Pallas and Vesta except on 32-bit ARM
//! (`if_projective_mul!`), and a binary ladder otherwise.

/// Selects the radix-27 scalar multiplication in projective coordinates, except on
/// 32-bit ARM.
/// Keystone measurements favor the existing Jacobian path; retain it on other
/// 32-bit ARM targets until measurements support enabling it.
macro_rules! if_projective_mul {
    ($projective:expr, $jacobian:expr $(,)?) => {{
        #[cfg(not(target_arch = "arm"))]
        {
            $projective
        }
        #[cfg(target_arch = "arm")]
        {
            $jacobian
        }
    }};
}

macro_rules! mul_curve_specific {
    // Jacobian or affine input alike: `From` picks the map into the projective
    // model from the input's type, and back to the Jacobian type.
    ($name:ident, $base:ident, special_a0_b5, $coord:ident, $point:expr, $scalar:expr) => {{
        let point = crate::curves::projective::Projective::<
            $base,
            { crate::curves::projective::PASTA_B },
        >::from($point);
        #[cfg_attr(not(feature = "zeroize"), allow(unused_mut))]
        let mut repr = $scalar.to_repr();
        let product = crate::curves::projective::mul::multiply(&point, &repr);
        #[cfg(feature = "zeroize")]
        zeroize::Zeroize::zeroize(&mut repr);
        $name::from(product)
    }};
    ($name:ident, $base:ident, general, $coord:ident, $point:expr, $scalar:expr) => {{
        let mut acc = $name::identity();

        // This is a simple double-and-add implementation of point
        // multiplication, moving from most significant to least
        // significant bit of the scalar.
        //
        // We don't use `PrimeFieldBits::to_le_bits` here, because that would
        // force users of this crate to depend on `bitvec` where they otherwise
        // might not need to.
        //
        // NOTE: We skip the leading bit because it's always unset (we are turning
        // the 32-byte repr into 256 bits, and $scalar::NUM_BITS = 255).
        for bit in $scalar
            .to_repr()
            .iter()
            .rev()
            .flat_map(|byte| (0..8).rev().map(move |i| Choice::from((byte >> i) & 1u8)))
            .skip(1)
        {
            acc = acc.double();
            acc = $name::conditional_select(&acc, &(acc + $point), bit);
        }

        acc
    }};
}

#[cfg(test)]
mod tests;
