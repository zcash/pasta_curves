use ff::{Field, PrimeField};

use super::*;

#[test]
fn multiexp_component_guard_returns_none_out_of_bounds() {
    let in_bounds = GLV_COMPONENT_BITS - 1;
    let out_of_bounds = 1u128 << GLV_COMPONENT_BITS;

    assert!(checked_signed_magnitudes(((false, 0), (true, 1))).is_some());
    assert!(checked_signed_magnitudes(((false, 1u128 << in_bounds), (false, 0))).is_some());
    assert!(checked_signed_magnitudes(((false, out_of_bounds), (false, 0))).is_none());
    assert!(checked_signed_magnitudes(((false, 0), (true, out_of_bounds))).is_none());
}

#[test]
fn booth_digits_reconstruct_components() {
    for width in 1..=16 {
        let mut magnitudes = alloc::vec![0, 1, (1u128 << 127) - 1];
        for bit in 1..127 {
            magnitudes.extend([(1u128 << bit) - 1, 1u128 << bit, (1u128 << bit) + 1]);
        }
        for magnitude in magnitudes {
            for negative in [false, true] {
                let component = SignedMagnitude {
                    magnitude,
                    negative,
                };
                let mut reconstructed = crate::Fq::ZERO;
                let radix = crate::Fq::from(1u64 << width);
                for window in (0..window_count(width)).rev() {
                    let digit = booth_digit(component, width, window);
                    assert!(digit.magnitude <= 1 << (width - 1));
                    let value = crate::Fq::from(u64::try_from(digit.magnitude).unwrap());
                    reconstructed =
                        reconstructed * radix + if digit.negative { -value } else { value };
                }
                let value = crate::Fq::from_u128(magnitude);
                assert_eq!(reconstructed, if negative { -value } else { value });
            }
        }
    }
}
