//! Private complete formulas for scalar multiplication on the native curves.
//!
//! The homogeneous addition and tripling adapt the algorithms in
//! <https://eprint.iacr.org/2026/2667> and the MIT/Apache-2.0 implementation
//! in zkcrypto/bls12_381. Public point arithmetic keeps its Jacobian representation.

use ff::Field;
use subtle::{Choice, ConditionallySelectable};

#[cfg(test)]
use subtle::ConstantTimeEq;

mod tau;

/// The field operations the formulas below use beyond `Field`: the
/// multiplication by a small public integer, for the multiples of b = 5, and
/// the unreduced products that let the portable addition reduce a sum of
/// products once.
pub(crate) trait FormulaField: Field {
    /// Returns `k * self`. `k` is a public constant below 2^32.
    fn mul_small(&self, k: u32) -> Self;

    /// Returns the unreduced 512-bit product of the representations.
    // Unused where an assembly backend multiplies.
    #[cfg_attr(feature = "asm", allow(dead_code))]
    fn mul_wide(&self, rhs: &Self) -> [u64; 8];

    /// Montgomery-reduces a sum of at most three `mul_wide` products.
    #[cfg_attr(feature = "asm", allow(dead_code))]
    fn reduce_wide(wide: [u64; 8]) -> Self;
}

// Adds two sums of products. The sum of three products of canonical
// elements is below 3p^2 < 2^510, so no carry leaves the top limb.
#[cfg_attr(not(feature = "uninline-portable"), inline(always))]
#[cfg_attr(feature = "asm", allow(dead_code))]
fn add_wide(a: &[u64; 8], b: &[u64; 8]) -> [u64; 8] {
    use crate::arithmetic::adc;
    let mut out = [0; 8];
    let mut carry = 0;
    for i in 0..8 {
        (out[i], carry) = adc(a[i], b[i], carry);
    }
    out
}

// Both Pasta moduli are p = 2^254 + c with c < 2^128. The Montgomery form is
// linear, so k times the representation of x represents k x. That product,
// below 2^288, is h 2^254 + l with l < 2^254, and 2^254 = -c (mod p) gives
// k x = l - h c (mod p). As h < 2^35, h c < 2^163 < p, so l - h c lies in
// (-p, 2^254): one addition of p, masked by the final borrow rather than
// branched on, makes it canonical.
#[cfg_attr(not(feature = "uninline-portable"), inline(always))]
fn mul_small_limbs(a: &[u64; 4], k: u32, modulus: &[u64; 4]) -> [u64; 4] {
    use crate::arithmetic::{adc, mac, sbb};
    let k = k as u64;
    let (r0, carry) = mac(0, a[0], k, 0);
    let (r1, carry) = mac(0, a[1], k, carry);
    let (r2, carry) = mac(0, a[2], k, carry);
    let (r3, r4) = mac(0, a[3], k, carry);
    let h = (r4 << 2) | (r3 >> 62);
    let r3 = r3 & ((1 << 62) - 1);
    let (s0, carry) = mac(0, h, modulus[0], 0);
    let (s1, s2) = mac(0, h, modulus[1], carry);
    let (d0, borrow) = sbb(r0, s0, 0);
    let (d1, borrow) = sbb(r1, s1, borrow);
    let (d2, borrow) = sbb(r2, s2, borrow);
    let (d3, borrow) = sbb(r3, 0, borrow);
    let (d0, carry) = adc(d0, modulus[0] & borrow, 0);
    let (d1, carry) = adc(d1, modulus[1] & borrow, carry);
    let (d2, carry) = adc(d2, modulus[2] & borrow, carry);
    let (d3, _) = adc(d3, modulus[3] & borrow, carry);
    [d0, d1, d2, d3]
}

macro_rules! small_mul {
    ($field:ident, $modulus:path) => {
        // mul_small_limbs needs the modulus shape 2^254 + c with c < 2^128.
        const _: () = assert!($modulus.0[2] == 0 && $modulus.0[3] == 1 << 62);

        impl FormulaField for crate::$field {
            #[inline(always)]
            fn mul_small(&self, k: u32) -> Self {
                crate::$field(mul_small_limbs(&self.0, k, &$modulus.0))
            }

            #[inline(always)]
            fn mul_wide(&self, rhs: &Self) -> [u64; 8] {
                self.mul_unreduced(rhs)
            }

            // Montgomery reduction is valid below p 2^256, which bounds
            // 3p^2 as 3p < 2^256, and its result is then canonical.
            #[inline(always)]
            fn reduce_wide(w: [u64; 8]) -> Self {
                crate::$field::montgomery_reduce(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])
            }
        }
    };
}
small_mul!(Fp, crate::fields::fp::MODULUS);
small_mul!(Fq, crate::fields::fq::MODULUS);

#[derive(Clone, Copy)]
pub(crate) struct Homogeneous<F: Field> {
    pub(crate) x: F,
    pub(crate) y: F,
    pub(crate) z: F,
}

impl<F: Field> Homogeneous<F> {
    pub(crate) fn identity() -> Self {
        Self {
            x: F::ZERO,
            y: F::ONE,
            z: F::ZERO,
        }
    }

    pub(crate) fn from_jacobian(x: F, y: F, z: F) -> Self {
        let z2 = z.square();
        let p = Self {
            x: x * z,
            y,
            z: z2 * z,
        };
        Self::conditional_select(&p, &Self::identity(), z.is_zero())
    }

    pub(crate) fn from_affine(x: F, y: F, infinity: Choice) -> Self {
        let p = Self { x, y, z: F::ONE };
        Self::conditional_select(&p, &Self::identity(), infinity)
    }

    pub(crate) fn to_jacobian(self) -> (F, F, F) {
        let z2 = self.z.square();
        let x = self.x * self.z;
        let y = self.y * z2;
        let zero = F::ZERO;
        (
            F::conditional_select(&x, &zero, self.z.is_zero()),
            F::conditional_select(&y, &zero, self.z.is_zero()),
            self.z,
        )
    }
}

impl<F: FormulaField> Homogeneous<F> {
    // The native curve has b = 5, hence 3b = 15.
    fn mul_by_3b(x: F) -> F {
        x.mul_small(15)
    }

    pub(crate) fn add(&self, rhs: &Self) -> Self {
        // Complete addition for a = 0 in homogeneous coordinates; the last
        // five products evaluate the cross product instead of its six terms.
        let t0 = self.x * rhs.x;
        let t1 = self.y * rhs.y;
        let t2 = self.z * rhs.z;
        let t3 = (self.x + self.y) * (rhs.x + rhs.y) - (t0 + t1);
        let t4 = (self.y + self.z) * (rhs.y + rhs.z) - (t1 + t2);
        let y3 = (self.x + self.z) * (rhs.x + rhs.z) - (t0 + t2);
        let t0 = t0.mul_small(3);
        let t2 = Self::mul_by_3b(t2);
        let z3 = t1 + t2;
        let t1 = t1 - t2;
        let y3 = Self::mul_by_3b(y3);

        let s = z3 + y3;
        let r = t1 - t4;
        if_asm!(
            {
                let m0 = s * r;
                let m1 = z3 * t4;
                let m2 = y3 * (t0 - r);
                let m3 = (t3 - s) * t1;
                let m4 = t3 * t0;
                Self {
                    x: m0 + m1 + m3,
                    y: m0 + m1 + m2,
                    z: m1 + m4,
                }
            },
            {
                // Portable: sum the unreduced products and reduce each
                // coordinate once, three reductions instead of five.
                let m0 = s.mul_wide(&r);
                let m1 = z3.mul_wide(&t4);
                let m2 = y3.mul_wide(&(t0 - r));
                let m3 = (t3 - s).mul_wide(&t1);
                let m4 = t3.mul_wide(&t0);
                let m01 = add_wide(&m0, &m1);
                Self {
                    x: F::reduce_wide(add_wide(&m01, &m3)),
                    y: F::reduce_wide(add_wide(&m01, &m2)),
                    z: F::reduce_wide(add_wide(&m1, &m4)),
                }
            },
        )
    }

    pub(crate) fn double(&self) -> Self {
        // Complete homogeneous doubling for a = 0 (RCB Algorithm 9).
        let t0 = self.y.square();
        let z3 = t0 + t0;
        let z3 = z3 + z3;
        let z3 = z3 + z3;
        let t1 = self.y * self.z;
        let t2 = Self::mul_by_3b(self.z.square());
        let x3 = t2 * z3;
        let y3 = t0 + t2;
        let z3 = t1 * z3;
        let t1 = t2 + t2;
        let t2 = t1 + t2;
        let t0 = t0 - t2;
        let y3 = x3 + t0 * y3;
        let t1 = self.x * self.y;
        let x3 = t0 * t1;
        let result = Self {
            x: x3 + x3,
            y: y3,
            z: z3,
        };
        Self::conditional_select(&result, &Self::identity(), self.z.is_zero())
    }

    pub(crate) fn triple(&self) -> Self {
        // Two complete degree-three isogenies and the isomorphism returning
        // to the original curve. Here bz² means 5z².
        let y2 = self.y.square();
        let z2 = self.z.square();
        let (bz2, b4z2, b8z2) = (z2.mul_small(5), z2.mul_small(20), z2.mul_small(40));
        let v = y2 - bz2;
        let x = self.x * (v + b4z2);
        let y = self.y * (v - b8z2);
        let z = self.z * v;

        // Here 27bz^2 = 135z^2, and 108bz^2 = 540z^2.
        let y2 = y.square();
        let z2 = z.square();
        let (b27z2, b108z2, b216z2) = (z2.mul_small(135), z2.mul_small(540), z2.mul_small(1080));
        let w = y2 + b27z2;
        let x = x * (w - b108z2);
        let y = y * (w + b216z2);
        let z = z * w;
        Self {
            x: x.mul_small(3),
            y,
            z: z.mul_small(27),
        }
    }

    #[cfg(test)]
    pub(crate) fn multiply(&self, by: &[u8; 32]) -> Self {
        let digits = to_radix_3(by);
        // Multiples 1P through 26P. Each pair doubles an earlier entry and
        // adds P, using complete formulas even when P is the identity.
        // This stack table occupies 2496 bytes for either Pasta field.
        let mut table = [*self; 26];
        for i in (1..25).step_by(2) {
            table[i] = table[i >> 1].double();
            table[i + 1] = table[i].add(self);
        }
        table[25] = table[12].double();

        let mut acc = Self::identity();
        // Three trits per radix-27 digit: a fixed 54 rounds, each scanning
        // every table entry and doing three triplings and one addition.
        for trits in digits.chunks_exact(3).rev() {
            let digit = trits[0] + 3 * trits[1] + 9 * trits[2];
            let mut addend = Self::identity();
            for (i, entry) in table.iter().enumerate() {
                addend = Self::conditional_select(&addend, entry, digit.ct_eq(&(i as u8 + 1)));
            }
            acc = acc.triple().triple().triple().add(&addend);
        }
        acc
    }
}

impl<F: Field> ConditionallySelectable for Homogeneous<F> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self {
            x: F::conditional_select(&a.x, &b.x, choice),
            y: F::conditional_select(&a.y, &b.y, choice),
            z: F::conditional_select(&a.z, &b.z, choice),
        }
    }
}

/// Recodes the low 255 bits of a little-endian scalar into 161 little-endian
/// ternary digits. The final zero digit makes the length divisible by three.
/// The high bit is ignored, as in the previous group multiply implementations.
#[cfg(test)]
pub(crate) fn to_radix_3(by: &[u8; 32]) -> [u8; 162] {
    let mut limbs = [0u32; 8];
    for i in 0..8 {
        limbs[i] = u32::from_le_bytes([by[4 * i], by[4 * i + 1], by[4 * i + 2], by[4 * i + 3]]);
    }
    limbs[7] &= 0x7fff_ffff;

    let mut digits = [0u8; 162];
    for digit in &mut digits[..161] {
        let mut remainder = 0u64;
        // The dividend fits in 34 bits because the incoming remainder is < 3.
        for limb in limbs.iter_mut().rev() {
            let dividend = (remainder << 32) | u64::from(*limb);
            let quotient = dividend / 3;
            remainder = dividend - quotient * 3;
            *limb = quotient as u32;
        }
        *digit = remainder as u8;
    }
    digits
}

#[cfg(test)]
mod tests;
