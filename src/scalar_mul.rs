//! Private complete formulas for scalar multiplication on the native curves.
//!
//! The homogeneous addition and tripling adapt the algorithms in
//! https://eprint.iacr.org/2026/2667 and the MIT/Apache-2.0 implementation
//! in zkcrypto/bls12_381. Public point arithmetic keeps its Jacobian representation.

use ff::Field;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

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

    // The native curve has b = 5, hence 3b = 15. No field-specific constants.
    fn mul_by_3b(x: F) -> F {
        let x2 = x + x;
        let x3 = x2 + x;
        let x6 = x3 + x3;
        let x12 = x6 + x6;
        x12 + x3
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
        let t0 = t0 + t0 + t0;
        let t2 = Self::mul_by_3b(t2);
        let z3 = t1 + t2;
        let t1 = t1 - t2;
        let y3 = Self::mul_by_3b(y3);

        let s = z3 + y3;
        let r = t1 - t4;
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
        let bz2 = z2 + z2 + z2 + z2 + z2;
        let v = y2 - bz2;
        let b4z2 = (bz2 + bz2) + (bz2 + bz2);
        let x = self.x * (v + b4z2);
        let y = self.y * (v - (b4z2 + b4z2));
        let z = self.z * v;

        let y2 = y.square();
        let z2 = z.square();
        let bz2 = z2 + z2 + z2 + z2 + z2;
        let b3z2 = bz2 + bz2 + bz2;
        let b9z2 = b3z2 + b3z2 + b3z2;
        let b27z2 = b9z2 + b9z2 + b9z2;
        let w = y2 + b27z2;
        let b108z2 = (b27z2 + b27z2) + (b27z2 + b27z2);
        let x = x * (w - b108z2);
        let y = y * (w + b108z2 + b108z2);
        let z = z * w;
        let z = z + z + z;
        let z = z + z + z;
        Self {
            x: x + x + x,
            y,
            z: z + z + z,
        }
    }

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
