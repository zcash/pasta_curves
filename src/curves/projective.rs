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
//! [`Projective`] is indexed by the curve's constant $b$, so the type says which
//! curve a point lies on: Pallas and Vesta have $b = 5$ ([`PASTA_B`]), and the
//! tripling passes through two other curves, $E_{-27 b}$ and $E_{729 b}$. The
//! addition and doubling are only defined on $E_5$, whose $3b$ they hard-code;
//! nothing here applies to the isogenous curves used for hashing to the curve.

// On 32-bit ARM the scalar multiplication keeps the Jacobian ladder, so only the
// tests use this model there.
#![cfg_attr(target_arch = "arm", allow(dead_code))]

use core::ops::Neg;

use ff::Field;
use subtle::{Choice, ConditionallySelectable};

// The constant $b$ of Pallas and Vesta, defined with the curves' parameters. The
// formulas multiply by small multiples of it with additions, through the
// functions below, rather than with a field multiplication.
pub(crate) use super::params::PASTA_B;

/// The constant of $E_{-27b}$, onto which [`Projective::isogeny_3`] maps $E_b$.
const IMAGE_B: i64 = -27 * PASTA_B;

/// The scaling of the isomorphism [`Projective::iota`] from $E_{729b}$ back to
/// $E_b$: $(x, y) \mapsto (x / U^2, y / U^3)$ with $U^6 = 729$, which in
/// projective coordinates is $(X : Y : Z) \mapsto (U X : Y : U^3 Z)$.
const U: i64 = 3;

/// The constant of $E_{729b}$, onto which the second 3-isogeny maps.
const IMAGE2_B: i64 = U.pow(6) * PASTA_B;

// The second isogeny lands on E_{729b}; the addition chains below hard-code these
// multiples.
const _: () = assert!(-27 * IMAGE_B == IMAGE2_B);
const _: () = assert!(PASTA_B == 5 && 3 * PASTA_B == 15 && IMAGE_B == -135);
const _: () = assert!(U == 3 && U.pow(3) == 27);

/// $b x$, with additions: $5 x = 2 (2 x) + x$.
fn times_b<F: Field>(x: F) -> F {
    (x + x) + (x + x) + x
}

/// $3 b x$, with additions: $15 x = 2 (2 (3 x)) + 3 x$. The complete addition
/// and doubling formulas use $3b$.
fn times_3b<F: Field>(x: F) -> F {
    let x2 = x + x;
    let x3 = x2 + x;
    let x6 = x3 + x3;
    let x12 = x6 + x6;
    x12 + x3
}

/// $-27 b x$, with additions: $-135 x = -(3 (3 (3 (5 x))))$, the constant
/// [`IMAGE_B`] of the curve onto which [`Projective::isogeny_3`] maps.
fn times_minus_27b<F: Field>(x: F) -> F {
    let x5 = times_b(x);
    let x15 = x5 + x5 + x5;
    let x45 = x15 + x15 + x15;
    -(x45 + x45 + x45)
}

/// $U x$, with additions.
fn times_u<F: Field>(x: F) -> F {
    x + x + x
}

/// $U^3 x$, with additions: $27 x = 3 (3 (3 x))$.
fn times_u3<F: Field>(x: F) -> F {
    let x3 = times_u(x);
    let x9 = x3 + x3 + x3;
    x9 + x9 + x9
}

/// A point $(X : Y : Z)$ on $E_B : Y^2 Z = X^3 + B Z^3$; see the module
/// documentation.
#[derive(Clone, Copy)]
pub(crate) struct Projective<F: Field, const B: i64> {
    pub(crate) x: F,
    pub(crate) y: F,
    pub(crate) z: F,
}

impl<F: Field, const B: i64> Projective<F, B> {
    pub(crate) fn identity() -> Self {
        Self {
            x: F::ZERO,
            y: F::ONE,
            z: F::ZERO,
        }
    }

    /// The 3-isogeny $\psi_B : E_B \to E_{-27B}$ whose kernel is the subgroup
    /// $\{O, (0 : \pm\sqrt{B} : 1)\}$:
    ///
    /// $$\psi_B(X : Y : Z) = (X (Y^2 + 3 B Z^2) : Y (Y^2 - 9 B Z^2) : Z (Y^2 - B Z^2)).$$
    ///
    /// `times_b` multiplies by $B$, and `IMAGE` must be $-27 B$. The tuple never
    /// vanishes on $E_B$, so the formula needs no case for $O$ or the kernel. On
    /// Pallas and Vesta the kernel's nonzero points are not rational, since 5 is not
    /// a square: the curves have prime order, so no rational point of order 3.
    ///
    /// Aranha and Barreto, <https://eprint.iacr.org/2026/2667>, section 3. Costs
    /// $3M + 2S$ and the additions in `times_b`.
    fn isogeny_3_with<const IMAGE: i64>(&self, times_b: impl Fn(F) -> F) -> Projective<F, IMAGE> {
        let y2 = self.y.square();
        let bz2 = times_b(self.z.square());
        let v = y2 - bz2;
        let b4z2 = (bz2 + bz2) + (bz2 + bz2);
        Projective {
            x: self.x * (v + b4z2),
            y: self.y * (v - (b4z2 + b4z2)),
            z: self.z * v,
        }
    }
}

impl<F: Field> Projective<F, PASTA_B> {
    pub(crate) fn add(&self, rhs: &Self) -> Self {
        // Complete addition for a = 0 in projective coordinates; the last
        // five products evaluate the cross product instead of its six terms.
        let t0 = self.x * rhs.x;
        let t1 = self.y * rhs.y;
        let t2 = self.z * rhs.z;
        let t3 = (self.x + self.y) * (rhs.x + rhs.y) - (t0 + t1);
        let t4 = (self.y + self.z) * (rhs.y + rhs.z) - (t1 + t2);
        let y3 = (self.x + self.z) * (rhs.x + rhs.z) - (t0 + t2);
        let t0 = t0 + t0 + t0;
        let t2 = times_3b(t2);
        let z3 = t1 + t2;
        let t1 = t1 - t2;
        let y3 = times_3b(y3);

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
        // Complete projective doubling for a = 0 (RCB Algorithm 9).
        let t0 = self.y.square();
        let z3 = t0 + t0;
        let z3 = z3 + z3;
        let z3 = z3 + z3;
        let t1 = self.y * self.z;
        let t2 = times_3b(self.z.square());
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

    /// The 3-isogeny $\psi_5$ onto $E_{-135}$; see [`Projective::isogeny_3_with`].
    fn isogeny_3(&self) -> Projective<F, IMAGE_B> {
        self.isogeny_3_with(times_b)
    }

    /// Returns $3P$ as $\iota \circ \psi_{-27b} \circ \psi_b$: a 3-isogeny, the
    /// same formula on its image curve, which is the dual up to isomorphism, and
    /// [`Projective::iota`] back to the curve. Costs $6M + 4S$; complete because
    /// each step is.
    pub(crate) fn triple(&self) -> Self {
        self.isogeny_3().isogeny_3().iota()
    }
}

impl<F: Field> Projective<F, IMAGE_B> {
    /// The 3-isogeny $\psi_{-135}$ onto $E_{3645}$, $3645 = 729 \cdot 5$; see
    /// [`Projective::isogeny_3_with`].
    fn isogeny_3(&self) -> Projective<F, IMAGE2_B> {
        self.isogeny_3_with(times_minus_27b)
    }
}

impl<F: Field> Projective<F, IMAGE2_B> {
    /// The isomorphism $E_{729 b} \to E_b$, $(X : Y : Z) \mapsto (U X : Y : U^3 Z)$,
    /// that is $(x, y) \mapsto (x / 9, y / 27)$, since $729 = U^6$.
    fn iota(&self) -> Projective<F, PASTA_B> {
        Projective {
            x: times_u(self.x),
            y: self.y,
            z: times_u3(self.z),
        }
    }
}

/// The identity, which zeroizing writes over a point, as for the public point types.
#[cfg(feature = "zeroize")]
impl<F: Field, const B: i64> Default for Projective<F, B> {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(feature = "zeroize")]
impl<F: Field, const B: i64> zeroize::DefaultIsZeroes for Projective<F, B> {}

/// The opposite point, $(X : -Y : Z)$; it fixes the identity $(0 : 1 : 0)$ as a
/// class, since $(0 : -1 : 0) = (0 : 1 : 0)$.
impl<F: Field, const B: i64> Neg for &Projective<F, B> {
    type Output = Projective<F, B>;

    fn neg(self) -> Projective<F, B> {
        Projective {
            x: self.x,
            y: -self.y,
            z: self.z,
        }
    }
}

impl<F: Field, const B: i64> ConditionallySelectable for Projective<F, B> {
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
