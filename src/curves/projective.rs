//! Projective coordinates for Pallas and Vesta, with complete formulas.
//!
//! A point is a class $(X : Y : Z)$ of the projective plane, in which
//! $(X : Y : Z)$ and $(\lambda X : \lambda Y : \lambda Z)$ are the same point for
//! every nonzero $\lambda$. The curve is the zero set of the affine equation
//! $y^2 = x^3 + 5$ made homogeneous of degree three:
//!
//! $$Y^2 Z = X^3 + 5 Z^3.$$
//!
//! Where $Z \neq 0$ the point is the affine point $(X / Z, Y / Z)$. Where $Z = 0$
//! the equation forces $X = 0$, which leaves the single point $(0 : 1 : 0)$, the
//! identity.
//!
//! The formulas need no case for the identity, for doubling or for opposite
//! points: they are complete on curves without a rational point of order two
//! (Renes, Costello and Batina, <https://eprint.iacr.org/2015/1060>), which
//! Pallas and Vesta, of prime order, are. The addition and tripling adapt the
//! algorithms in <https://eprint.iacr.org/2026/2667> and the MIT/Apache-2.0
//! implementation in zkcrypto/bls12_381.
//!
//! The formulas use $b = 5$, the constant of both Pallas and Vesta; they do not
//! apply to the isogenous curves used for hashing to the curve.

use ff::Field;
use subtle::{Choice, ConditionallySelectable};

/// A point $(X : Y : Z)$ on $Y^2 Z = X^3 + 5 Z^3$; see the module documentation.
#[derive(Clone, Copy)]
pub(crate) struct Projective<F: Field> {
    pub(crate) x: F,
    pub(crate) y: F,
    pub(crate) z: F,
}

impl<F: Field> Projective<F> {
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
}

impl<F: Field> ConditionallySelectable for Projective<F> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self {
            x: F::conditional_select(&a.x, &b.x, choice),
            y: F::conditional_select(&a.y, &b.y, choice),
            z: F::conditional_select(&a.z, &b.z, choice),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
