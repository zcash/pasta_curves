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

    /// The 3-isogeny $\psi_b : E_b \to E_{-27b}$ whose kernel is the subgroup
    /// $\{O, (0 : \pm\sqrt{b} : 1)\}$, where $E_c$ is $Y^2 Z = X^3 + c Z^3$:
    ///
    /// $$\psi_b(X : Y : Z) = (X (Y^2 + 3 b Z^2) : Y (Y^2 - 9 b Z^2) : Z (Y^2 - b Z^2)).$$
    ///
    /// `times_b` multiplies by $b$. The tuple never vanishes on $E_b$, so the formula
    /// needs no case for $O$ or the kernel. The result lies on $E_{-27b}$, not on the
    /// curve of `self`. On Pallas and Vesta the kernel's nonzero points are not
    /// rational, since 5 is not a square: the curves have prime order, so no rational
    /// point of order 3.
    ///
    /// Aranha and Barreto, <https://eprint.iacr.org/2026/2667>, section 3. Costs
    /// $3M + 2S$ and the additions in `times_b`.
    fn isogeny_3(&self, times_b: impl Fn(F) -> F) -> Self {
        let y2 = self.y.square();
        let bz2 = times_b(self.z.square());
        let v = y2 - bz2;
        let b4z2 = (bz2 + bz2) + (bz2 + bz2);
        Self {
            x: self.x * (v + b4z2),
            y: self.y * (v - (b4z2 + b4z2)),
            z: self.z * v,
        }
    }

    /// The isomorphism $E_{729 \cdot 5} \to E_5$, $(X : Y : Z) \mapsto (3X : Y : 27Z)$,
    /// that is $(x, y) \mapsto (x / 9, y / 27)$, since $729 = 3^6$.
    fn iota(&self) -> Self {
        let x = self.x + self.x + self.x;
        let z = self.z + self.z + self.z;
        let z = z + z + z;
        Self {
            x,
            y: self.y,
            z: z + z + z,
        }
    }

    /// Returns $3P$ as $\iota \circ \psi_{-27b} \circ \psi_b$: a 3-isogeny, the
    /// same formula on its image curve, which is the dual up to isomorphism, and
    /// [`Self::iota`] back to the curve. Costs $6M + 4S$; complete because
    /// each step is.
    pub(crate) fn triple(&self) -> Self {
        // b = 5 for Pallas and Vesta, so the image curve has -27b = -135.
        let times_5 = |x: F| (x + x) + (x + x) + x;
        let times_minus_135 = |x: F| {
            let x5 = times_5(x);
            let x15 = x5 + x5 + x5;
            let x45 = x15 + x15 + x15;
            -(x45 + x45 + x45)
        };
        self.isogeny_3(times_5).isogeny_3(times_minus_135).iota()
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
