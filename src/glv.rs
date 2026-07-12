//! GLV (Gallant–Lambert–Vanstone) scalar multiplication for the Pasta curves.
//!
//! Both Pasta curves are equipped with a cube-root endomorphism
//! φ(x, y) = (ζ·x, y) (exposed as [`CurveExt::endo`]), for which
//! φ(P) = λ·P with λ = `Scalar::ZETA`. This module uses that structure to
//! split a full-width scalar multiplication `k·P` into two half-width
//! multiplications evaluated against a shared table of odd multiples of `P`
//! and φ(P) — the classic GLV speedup described in the Halo paper.
//!
//! The API is additive: the `_glv` naming distinguishes this variable-time
//! path from the constant-time native `Mul` implementations, which are
//! unchanged.
//!
//! # Security
//!
//! **This path is variable-time with respect to the scalar** (GLV
//! decomposition plus wNAF recoding). It is intended for public or
//! per-session scalars — batched trial decryption, verification-style
//! workloads — and must not be used with long-term secret scalars.
//!
//! # Amortization
//!
//! The costs split into three independently reusable pieces:
//!
//! - [`Table`]: per *point*. [`Table::batch`] builds many tables with one
//!   shared batch normalization (a single field inversion for the whole
//!   batch).
//! - [`Decomposed`]: per *scalar*. Decomposition and wNAF recoding are
//!   hoisted so one scalar can be multiplied against many tables.
//! - [`Table::mul_decomposed`]: the remaining per-(point, scalar) work — a
//!   shared-doubling Straus ladder over the two half-width digit strings.
//!
//! One-shot use is [`MulGlv::mul_glv`].

use alloc::vec::Vec;

use ff::{PrimeField, WithSmallOrderMulGroup};
use group::prime::PrimeCurveAffine;

use crate::arithmetic::{CurveAffine, CurveExt};
use crate::{pallas, vesta};

mod private {
    /// Seals [`super::GlvParams`]: the lattice constants are curve-specific
    /// and verified in-crate; external implementations are not supported.
    pub trait Sealed {}
    impl Sealed for crate::pallas::Point {}
    impl Sealed for crate::vesta::Point {}
}

/// Per-curve GLV constants: a short basis for the lattice
/// {(a, b) : a + b·λ ≡ 0 (mod r)} and the Babai rounding coefficients
/// derived from it.
///
/// The `constants_verify_*` tests re-verify every constant against the
/// curve's own λ (= `Scalar::ZETA`) using field arithmetic only, and the
/// `decompose_reconstructs_*` tests prove the full pipeline algebraically:
/// wrong constants cannot reconstruct `k`.
///
/// This trait is sealed; it is implemented for [`pallas::Point`] and
/// [`vesta::Point`].
pub trait GlvParams: CurveExt + private::Sealed {
    /// v1 = (`V1A`, −`V1B_NEG`): first short lattice vector.
    const V1A: u128;
    /// Magnitude of v1's (negative) second component.
    const V1B_NEG: u128;
    /// v2 = (`V2A`, `V2B`): second short lattice vector.
    const V2A: u128;
    /// v2's (positive) second component.
    const V2B: u128;
    /// Babai coefficient `round(2^384·V2B / r)`, little-endian limbs.
    const G1: [u64; 5];
    /// Babai coefficient `round(2^384·V1B_NEG / r)`, little-endian limbs.
    const G2: [u64; 5];
}

impl GlvParams for pallas::Point {
    const V1A: u128 = 0x49e69d1640f049157fcae1c700000001;
    const V1B_NEG: u128 = 0x49e69d1640a899538cb1279300000000;
    const V2A: u128 = 0x49e69d1640a899538cb1279300000000;
    const V2B: u128 = 0x93cd3a2c8198e2690c7c095a00000001;
    const G1: [u64; 5] = [
        0x111f686111afc293,
        0xc35fbd4d086862e0,
        0x31f0256800000002,
        0x4f34e8b2066389a4,
        0x2,
    ];
    const G2: [u64; 5] = [
        0x4a95a2d972171db4,
        0x61afdea68480fa55,
        0x32c49e4bffffffff,
        0x279a745902a2654e,
        0x1,
    ];
}

impl GlvParams for vesta::Point {
    const V1A: u128 = 0x49e69d1640f049157fcae1c700000000;
    const V1B_NEG: u128 = 0x49e69d1640a899538cb1279300000001;
    const V2A: u128 = 0x49e69d1640a899538cb1279300000001;
    const V2B: u128 = 0x93cd3a2c8198e2690c7c095a00000001;
    const G1: [u64; 5] = [
        0x841d8d62296e1563,
        0xc35fbd4d0afe9926,
        0x31f0256800000002,
        0x4f34e8b2066389a4,
        0x2,
    ];
    const G2: [u64; 5] = [
        0x841414c24bf99a83,
        0x61afdea685cc1578,
        0x32c49e4c00000003,
        0x279a745902a2654e,
        0x1,
    ];
}

/// `round((g · k) / 2^384)` for a 5-limb `g` and 4-limb `k` — the Babai
/// coefficient. Fits `u128` (at most ~128 bits by construction).
fn round_mul_shift(g: &[u64; 5], k: &[u64; 4]) -> u128 {
    let mut prod = [0u64; 9];
    for (i, &gi) in g.iter().enumerate() {
        let mut carry = 0u128;
        for (j, &kj) in k.iter().enumerate() {
            let t = u128::from(gi) * u128::from(kj) + u128::from(prod[i + j]) + carry;
            prod[i + j] = t as u64;
            carry = t >> 64;
        }
        prod[i + 4] = prod[i + 4].wrapping_add(carry as u64);
    }
    // Bits >= 384 live in limbs 6..; round on bit 383 (top bit of limb 5).
    let round = prod[5] >> 63;
    (u128::from(prod[6]) | (u128::from(prod[7]) << 64)).wrapping_add(u128::from(round))
}

/// 256-bit product of two `u128`s, as little-endian limbs.
fn mul_u128(a: u128, b: u128) -> [u64; 4] {
    let (a0, a1) = (a as u64, (a >> 64) as u64);
    let (b0, b1) = (b as u64, (b >> 64) as u64);
    let mut out = [0u64; 4];
    let mut acc = |i: usize, v: u128| {
        let mut idx = i;
        let mut carry = v;
        while carry != 0 {
            let t = u128::from(out[idx]) + (carry & u128::from(u64::MAX));
            out[idx] = t as u64;
            carry = (carry >> 64) + (t >> 64);
            idx += 1;
        }
    };
    acc(0, u128::from(a0) * u128::from(b0));
    acc(1, u128::from(a0) * u128::from(b1));
    acc(1, u128::from(a1) * u128::from(b0));
    acc(2, u128::from(a1) * u128::from(b1));
    out
}

/// 256-bit wrapping subtraction (two's complement).
fn sub256(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut borrow = 0u64;
    for i in 0..4 {
        let (d, b1) = a[i].overflowing_sub(b[i]);
        let (d, b2) = d.overflowing_sub(borrow);
        out[i] = d;
        borrow = u64::from(b1) + u64::from(b2);
    }
    out
}

/// Interprets a 256-bit two's-complement value with |x| < 2^128 as
/// (is_negative, |x|).
fn signed_halves(x: [u64; 4]) -> (bool, u128) {
    if x[3] >> 63 == 0 {
        debug_assert!(x[2] == 0 && x[3] == 0, "positive half exceeds 128 bits");
        (false, u128::from(x[0]) | (u128::from(x[1]) << 64))
    } else {
        // Negate: !x + 1.
        let mut n = [!x[0], !x[1], !x[2], !x[3]];
        let mut carry = 1u64;
        for limb in &mut n {
            let (v, c) = limb.overflowing_add(carry);
            *limb = v;
            carry = u64::from(c);
            if carry == 0 {
                break;
            }
        }
        debug_assert!(n[2] == 0 && n[3] == 0, "negative half exceeds 128 bits");
        (true, u128::from(n[0]) | (u128::from(n[1]) << 64))
    }
}

/// GLV split: `k = k1 + k2·λ (mod r)` with |k1|, |k2| ≤ 2^128, each half
/// returned as (is_negative, magnitude).
fn decompose<C: GlvParams>(k: &C::ScalarExt) -> ((bool, u128), (bool, u128)) {
    let repr = k.to_repr();
    let bytes: &[u8] = repr.as_ref();
    debug_assert_eq!(bytes.len(), 32, "Pasta scalars have a 32-byte repr");
    let mut kl = [0u64; 4];
    for (i, limb) in kl.iter_mut().enumerate() {
        *limb = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().expect("8 bytes"));
    }
    let c1 = round_mul_shift(&C::G1, &kl);
    let c2 = round_mul_shift(&C::G2, &kl);
    // k1 = k − c1·V1A − c2·V2A   (two's complement over 256 bits)
    let k1 = sub256(sub256(kl, mul_u128(c1, C::V1A)), mul_u128(c2, C::V2A));
    // k2 = c1·V1B_NEG − c2·V2B   (v1.b = −V1B_NEG, v2.b = +V2B)
    let k2 = sub256(mul_u128(c1, C::V1B_NEG), mul_u128(c2, C::V2B));
    (signed_halves(k1), signed_halves(k2))
}

/// φ(P) on affine coordinates: (ζ·x, y). The identity maps to the identity.
fn endo_affine<A: CurveAffine>(p: &A) -> A {
    let coords = p.coordinates();
    if bool::from(coords.is_none()) {
        return A::identity();
    }
    let c = coords.unwrap();
    A::from_xy(A::Base::ZETA * *c.x(), *c.y()).unwrap()
}

/// The GLV window for one base point: the odd multiples {1, 3, 5, 7}·P and
/// {1, 3, 5, 7}·φ(P) in affine coordinates. 512 bytes per table.
///
/// Build one with [`Table::new`], or many with one shared normalization via
/// [`Table::batch`].
#[derive(Clone, Copy, Debug)]
pub struct Table<C: GlvParams> {
    /// {1, 3, 5, 7}·P
    t1: [C::AffineExt; 4],
    /// {1, 3, 5, 7}·φ(P)
    t2: [C::AffineExt; 4],
}

impl<C: GlvParams> Table<C> {
    /// Builds the window for a single non-identity point.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `p` is the identity (the table of a fixed
    /// base point has no meaningful identity form).
    pub fn new(p: &C) -> Self {
        Self::batch(core::slice::from_ref(p))
            .pop()
            .expect("one table per input point")
    }

    /// Builds [`Table`]s for a batch of non-identity points with one shared
    /// batch normalization across all 4·n odd multiples — a single field
    /// inversion for the whole batch, where building each window
    /// individually pays one inversion per point.
    ///
    /// Uses projective group operations only (no hand-rolled affine
    /// formulas), and on a prime-order curve the odd multiples of a
    /// non-identity point are never the identity, so the normalized windows
    /// are always well-formed.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if any input point is the identity.
    pub fn batch(points: &[C]) -> Vec<Table<C>> {
        let n = points.len();
        if n == 0 {
            return Vec::new();
        }
        // Odd multiples per point, projective (cheap additions, no
        // inversions), interleaved [1·P₀, 3·P₀, 5·P₀, 7·P₀, 1·P₁, ...].
        let mut proj = Vec::with_capacity(n * 4);
        for p in points {
            debug_assert!(
                !bool::from(p.is_identity()),
                "Table::batch contract: non-identity points only"
            );
            let two_p = p.double();
            let mut m = *p;
            proj.push(m);
            for _ in 1..4 {
                m += two_p;
                proj.push(m);
            }
        }
        // One inversion for the whole batch.
        let mut affine = alloc::vec![C::AffineExt::identity(); n * 4];
        C::batch_normalize(&proj, &mut affine);
        affine
            .chunks_exact(4)
            .map(|c| {
                let t1: [C::AffineExt; 4] = c.try_into().expect("chunks of 4");
                let t2 = [
                    endo_affine(&t1[0]),
                    endo_affine(&t1[1]),
                    endo_affine(&t1[2]),
                    endo_affine(&t1[3]),
                ];
                Table { t1, t2 }
            })
            .collect()
    }

    /// The base point P (= t1\[0\]) back as a projective point.
    pub fn point(&self) -> C {
        C::from(self.t1[0])
    }

    /// `k·P` for the P encoded by this table, decomposing `k` on the spot.
    ///
    /// When one scalar meets many tables, decompose once with
    /// [`Decomposed::new`] and use [`Table::mul_decomposed`] instead.
    pub fn mul(&self, k: &C::ScalarExt) -> C {
        self.mul_decomposed(&Decomposed::new(k))
    }

    /// `k·P` for the P encoded by this table, via the Straus
    /// shared-doubling ladder over the GLV split. Identical to `P * k`
    /// (tested).
    pub fn mul_decomposed(&self, k: &Decomposed<C>) -> C {
        let len = k.len1.max(k.len2);
        let mut acc = C::identity();
        for i in (0..len).rev() {
            acc = acc.double();
            let d = if i < k.len1 { k.digits1[i] } else { 0 };
            if d != 0 {
                let mut a = self.t1[(d.unsigned_abs() / 2) as usize];
                if (d < 0) ^ k.neg1 {
                    a = -a;
                }
                acc += a;
            }
            let d = if i < k.len2 { k.digits2[i] } else { 0 };
            if d != 0 {
                let mut a = self.t2[(d.unsigned_abs() / 2) as usize];
                if (d < 0) ^ k.neg2 {
                    a = -a;
                }
                acc += a;
            }
        }
        acc
    }
}

/// A scalar in GLV-decomposed, wNAF-recoded form, ready for
/// [`Table::mul_decomposed`].
///
/// Building this once per scalar hoists the decomposition and digit
/// recoding out of a loop that multiplies the same scalar against many
/// tables (e.g. one viewing key against a batch of ephemeral keys).
#[derive(Clone, Debug)]
pub struct Decomposed<C: GlvParams> {
    neg1: bool,
    digits1: [i8; 132],
    len1: usize,
    neg2: bool,
    digits2: [i8; 132],
    len2: usize,
    _curve: core::marker::PhantomData<C>,
}

impl<C: GlvParams> Decomposed<C> {
    /// Decomposes `k` and recodes both halves as width-4 wNAF digits.
    pub fn new(k: &C::ScalarExt) -> Self {
        let ((neg1, a1), (neg2, a2)) = decompose::<C>(k);
        let (digits1, len1) = wnaf_digits(a1);
        let (digits2, len2) = wnaf_digits(a2);
        Decomposed {
            neg1,
            digits1,
            len1,
            neg2,
            digits2,
            len2,
            _curve: core::marker::PhantomData,
        }
    }
}

/// One-shot GLV scalar multiplication.
///
/// Implemented for the curves carrying [`GlvParams`]. For repeated
/// multiplications against the same point or the same scalar, use
/// [`Table`] / [`Decomposed`] directly to reuse the precomputation.
pub trait MulGlv: GlvParams {
    /// `k·self` via the GLV split — variable-time in `k` (see the module
    /// docs), identical in value to `self * k` (including `self` = identity).
    fn mul_glv(&self, k: &Self::ScalarExt) -> Self;
}

impl<C: GlvParams> MulGlv for C {
    fn mul_glv(&self, k: &Self::ScalarExt) -> Self {
        if bool::from(self.is_identity()) {
            // k·O = O; the identity has no meaningful multiples table.
            return Self::identity();
        }
        Table::new(self).mul(k)
    }
}

/// Width-4 wNAF digits of a u128 magnitude, lowest position first. A
/// magnitude of at most 2^127 yields at most 129 digits; the array is sized
/// with headroom.
fn wnaf_digits(a: u128) -> ([i8; 132], usize) {
    debug_assert!(a >> 127 == 0, "magnitude must be at most 127 bits");
    let mut digits = [0i8; 132];
    let mut n = 0;
    let mut k = a;
    while k != 0 {
        if k & 1 == 1 {
            let low = (k & 0xF) as i8;
            let d = if low >= 8 { low - 16 } else { low };
            digits[n] = d;
            if d >= 0 {
                k -= d as u128;
            } else {
                k += (-d) as u128;
            }
        }
        n += 1;
        k >>= 1;
    }
    (digits, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;

    /// Deterministic full-width scalars for the known-answer tests.
    fn scalars<F: PrimeField>(n: u64) -> impl Iterator<Item = F> {
        (0..n).map(|i| {
            (F::from(0x9E37_79B9_7F4A_7C15u64 + i).square() + F::from(0x0123_4567_89AB_CDEFu64))
                .square()
                + F::from(i)
        })
    }

    /// The short-basis lattice relations, re-verified against the curve's
    /// own λ (= `Scalar::ZETA`) using field arithmetic only:
    ///   V1A − V1B_NEG·λ ≡ 0  and  V2A + V2B·λ ≡ 0  (mod r).
    fn constants_verify<C: GlvParams>() {
        let lambda = C::ScalarExt::ZETA;
        let from = C::ScalarExt::from_u128;
        assert_eq!(from(C::V1A), from(C::V1B_NEG) * lambda, "v1 not in lattice");
        assert_eq!(from(C::V2A), -(from(C::V2B) * lambda), "v2 not in lattice");
    }

    /// The φ↔λ pairing on the real curve: (ζ_base·x, y) == λ·P.
    fn endo_map_is_lambda<C: GlvParams>() {
        let g = C::generator();
        for k in scalars::<C::ScalarExt>(64) {
            let p = (g * k).to_affine();
            let via_map = endo_affine(&p);
            let via_mul = (C::from(p) * C::ScalarExt::ZETA).to_affine();
            assert_eq!(via_map, via_mul, "phi(P) must equal ZETA_scalar * P");
        }
    }

    /// The algebraic gate: k1 + k2·λ ≡ k (mod r) with both halves at most
    /// 2^127, for full-width scalars and the edge cases. Wrong GLV
    /// constants cannot pass this.
    fn decompose_reconstructs<C: GlvParams>() {
        let lambda = C::ScalarExt::ZETA;
        let check = |k: C::ScalarExt| {
            let ((neg1, a1), (neg2, a2)) = decompose::<C>(&k);
            assert!(a1 >> 127 == 0, "k1 exceeds 127 bits");
            assert!(a2 >> 127 == 0, "k2 exceeds 127 bits");
            let s1 = C::ScalarExt::from_u128(a1);
            let s1 = if neg1 { -s1 } else { s1 };
            let s2 = C::ScalarExt::from_u128(a2);
            let s2 = if neg2 { -s2 } else { s2 };
            assert_eq!(s1 + s2 * lambda, k, "decomposition must reconstruct k");
        };
        check(C::ScalarExt::ZERO);
        check(C::ScalarExt::ONE);
        check(-C::ScalarExt::ONE);
        check(lambda);
        check(-lambda);
        for k in scalars::<C::ScalarExt>(1000) {
            check(k);
        }
    }

    /// Table-based multiplication matches the group's native `Mul`.
    fn table_mul_matches_group_mul<C: GlvParams>() {
        let g = C::generator();
        for (i, k) in scalars::<C::ScalarExt>(64).enumerate() {
            let p = g * (k + C::ScalarExt::from(i as u64 + 1));
            let table = Table::new(&p);
            for k2 in scalars::<C::ScalarExt>(4) {
                assert_eq!(table.mul(&k2), p * k2, "table mul must match group mul");
            }
        }
    }

    /// One-shot `mul_glv` matches the native operator.
    fn mul_glv_matches_operator<C: GlvParams>() {
        let g = C::generator();
        for k in scalars::<C::ScalarExt>(64) {
            let p = g * (k + C::ScalarExt::ONE);
            assert_eq!(p.mul_glv(&k), p * k, "mul_glv must match operator");
        }
    }

    /// The batched table build equals the solo build, point by point.
    fn batch_tables_equal_solo<C: GlvParams>() {
        let g = C::generator();
        let points: Vec<C> = scalars::<C::ScalarExt>(16)
            .map(|k| g * (k + C::ScalarExt::ONE))
            .collect();
        let batched = Table::batch(&points);
        assert_eq!(batched.len(), points.len());
        for (p, table) in points.iter().zip(batched.iter()) {
            let solo = Table::new(p);
            assert_eq!(table.point(), solo.point());
            let k = C::ScalarExt::from(0xDEAD_BEEFu64);
            assert_eq!(
                table.mul(&k),
                solo.mul(&k),
                "batched table must act like solo"
            );
        }
    }

    /// A reused [`Decomposed`] gives the same products as decomposing
    /// per-multiplication.
    fn decomposed_reuse_matches_fresh<C: GlvParams>() {
        let g = C::generator();
        let k = scalars::<C::ScalarExt>(1).next().unwrap();
        let decomposed = Decomposed::<C>::new(&k);
        for k2 in scalars::<C::ScalarExt>(16) {
            let p = g * (k2 + C::ScalarExt::ONE);
            let table = Table::new(&p);
            assert_eq!(
                table.mul_decomposed(&decomposed),
                table.mul(&k),
                "hoisted decomposition must match fresh"
            );
        }
    }

    macro_rules! glv_tests {
        ($mod_name:ident, $curve:ty) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn constants() {
                    constants_verify::<$curve>();
                }
                #[test]
                fn endo_map() {
                    endo_map_is_lambda::<$curve>();
                }
                #[test]
                fn decompose() {
                    decompose_reconstructs::<$curve>();
                }
                #[test]
                fn table_mul() {
                    table_mul_matches_group_mul::<$curve>();
                }
                #[test]
                fn one_shot() {
                    mul_glv_matches_operator::<$curve>();
                }
                #[test]
                fn batch_build() {
                    batch_tables_equal_solo::<$curve>();
                }
                #[test]
                fn decomposed_reuse() {
                    decomposed_reuse_matches_fresh::<$curve>();
                }
            }
        };
    }

    glv_tests!(pallas_glv, pallas::Point);
    glv_tests!(vesta_glv, vesta::Point);

    /// Edge-case scalars exercised through the FULL `mul_glv` path (not just
    /// `decompose`): the additive/multiplicative identities and their
    /// negations, λ and its neighbours (the decomposition's own axis), and
    /// the half-width boundary where k1/k2 magnitudes live.
    fn edge_case_matrix<C: GlvParams>() {
        let lambda = C::ScalarExt::ZETA;
        let edge_scalars = [
            C::ScalarExt::ZERO,
            C::ScalarExt::ONE,
            -C::ScalarExt::ONE,
            C::ScalarExt::from(2),
            lambda,
            -lambda,
            lambda + C::ScalarExt::ONE,
            C::ScalarExt::from(u64::MAX),
            C::ScalarExt::from_u128((1u128 << 127) - 1),
            C::ScalarExt::from_u128(1u128 << 127),
            C::ScalarExt::from_u128((1u128 << 127) + 1),
        ];
        let g = C::generator();
        let points = [g, g * (lambda + C::ScalarExt::from(42))];
        for p in points {
            for k in edge_scalars {
                assert_eq!(p.mul_glv(&k), p * k, "mul_glv must match Mul on edges");
            }
        }
        // k·O = O for every scalar, including 0.
        let identity = C::identity();
        for k in edge_scalars {
            assert_eq!(identity.mul_glv(&k), C::identity(), "k*O must be O");
        }
    }

    #[test]
    fn edge_cases_pallas() {
        edge_case_matrix::<pallas::Point>();
    }
    #[test]
    fn edge_cases_vesta() {
        edge_case_matrix::<vesta::Point>();
    }

    /// Property-based tests: scalars are drawn as four uniform u64 limbs
    /// widened through `from_uniform_bytes` (so the whole field is reachable
    /// without modular bias), and points as `G·(s+1)`.
    mod pbt {
        use group::Group;
        use proptest::prelude::*;

        use super::*;

        fn scalar_strategy<F: PrimeField + ff::FromUniformBytes<64>>() -> impl Strategy<Value = F> {
            proptest::array::uniform4(any::<u64>()).prop_map(|limbs| {
                let mut bytes = [0u8; 64];
                for (i, l) in limbs.iter().enumerate() {
                    bytes[i * 8..(i + 1) * 8].copy_from_slice(&l.to_le_bytes());
                }
                F::from_uniform_bytes(&bytes)
            })
        }

        macro_rules! glv_pbt {
            ($mod_name:ident, $curve:ty) => {
                mod $mod_name {
                    use super::*;

                    type Scalar = <$curve as CurveExt>::ScalarExt;

                    proptest! {
                        /// ∀ P ≠ O, k: P.mul_glv(k) == P * k.
                        #[test]
                        fn mul_glv_matches_mul(
                            s in scalar_strategy::<Scalar>(),
                            k in scalar_strategy::<Scalar>(),
                        ) {
                            let p = <$curve>::generator() * (s + Scalar::ONE);
                            prop_assert_eq!(p.mul_glv(&k), p * k);
                        }

                        /// ∀ k: the GLV split reconstructs k with half-width parts.
                        #[test]
                        fn decompose_reconstructs(k in scalar_strategy::<Scalar>()) {
                            let ((neg1, a1), (neg2, a2)) = decompose::<$curve>(&k);
                            prop_assert!(a1 >> 127 == 0);
                            prop_assert!(a2 >> 127 == 0);
                            let s1 = Scalar::from_u128(a1);
                            let s1 = if neg1 { -s1 } else { s1 };
                            let s2 = Scalar::from_u128(a2);
                            let s2 = if neg2 { -s2 } else { s2 };
                            prop_assert_eq!(s1 + s2 * Scalar::ZETA, k);
                        }

                        /// ∀ points: batched tables act identically to solo tables.
                        #[test]
                        fn batch_equals_solo(
                            seeds in proptest::collection::vec(scalar_strategy::<Scalar>(), 1..8),
                            k in scalar_strategy::<Scalar>(),
                        ) {
                            let points: alloc::vec::Vec<$curve> = seeds
                                .iter()
                                .map(|s| <$curve>::generator() * (*s + Scalar::ONE))
                                .collect();
                            let batched = Table::batch(&points);
                            for (p, table) in points.iter().zip(batched.iter()) {
                                prop_assert_eq!(table.mul(&k), Table::new(p).mul(&k));
                                prop_assert_eq!(table.mul(&k), *p * k);
                            }
                        }

                        /// ∀ k reused across points: hoisted decomposition == fresh.
                        #[test]
                        fn decomposed_reuse(
                            s in scalar_strategy::<Scalar>(),
                            k in scalar_strategy::<Scalar>(),
                        ) {
                            let p = <$curve>::generator() * (s + Scalar::ONE);
                            let table = Table::new(&p);
                            let hoisted = Decomposed::<$curve>::new(&k);
                            prop_assert_eq!(table.mul_decomposed(&hoisted), table.mul(&k));
                        }
                    }
                }
            };
        }

        glv_pbt!(pallas_pbt, pallas::Point);
        glv_pbt!(vesta_pbt, vesta::Point);
    }
}
