//! GLV (Gallant–Lambert–Vanstone) scalar multiplication for the Pasta curves.
//!
//! GLV is a **re-encoding** of the scalar, in two stages, and only the first
//! is what the technique is named for.
//!
//! The **split**. Both Pasta curves carry a cube-root endomorphism
//! $\phi(x, y) = (\zeta x, y)$ (exposed as [`CurveExt::endo`]), for which
//! $\phi(P) = \lambda P$ with $\lambda$ = [`Scalar::ZETA`]. Because
//! $\lambda^2 + \lambda + 1 = 0$, a full-width scalar can be rewritten as
//! $k \equiv k_1 + k_2\lambda \pmod n$ with both halves near $\sqrt{n}$, so
//! that $k P = k_1 P + k_2 \phi(P)$. Running both halves over one shared
//! column loop halves the doublings. That is the whole of what the split
//! buys, and it is independent of what comes next.
//!
//! The **digit expansion**. Turning that re-encoded scalar into columns a
//! ladder can walk is a separate choice, and GLV says nothing about it. More
//! than one answer works, and this crate carries two:
//!
//! - *This module*: two **independent** width-4 wNAF digit strings, one per
//!   half, sharing a column index. An addition is paid whenever either string
//!   is nonzero, so their densities add.
//! - *`crate::glv_eisenstein`* (feature `glv-eisenstein`): one **joint**
//!   width-3 NAF over the Eisenstein integers. It uses the fact that
//!   $(k_1, k_2)$ is not two unrelated integers but the single element
//!   $k_1 + k_2\omega$ of $\mathbb{Z}[\omega]$, which the scalar field
//!   receives by $\omega \mapsto \lambda$. One digit string instead of two,
//!   so an addition is paid once per column rather than twice.
//!
//! # Using it
//!
//! Both modules expose the same two entry points, so choosing a recoding is a
//! change of import and nothing else:
//!
//! - [`mul`] for one point against one scalar;
//! - [`batch_mul`] for many points against one shared scalar, which is the
//!   wallet-scanning shape and the one `glv_eisenstein` optimises hardest.
//!
//! For the shapes those two do not cover, one point against many scalars, or
//! many of each, build the pieces and reuse them: a [`Table`] per point, a
//! [`Decomposed`] per scalar, and [`Table::mul_decomposed`] for each pair.
//!
//! This module evaluates its two half-width multiplications against a shared
//! table of odd multiples of $P$ and $\phi(P)$.
//!
//! This path is variable-time in the scalar (GLV decomposition plus wNAF
//! recoding); the `_glv` naming distinguishes it from the native `Mul`
//! implementations, which are unchanged.
//!
//! [`Scalar::ZETA`]: ff::WithSmallOrderMulGroup::ZETA
//!
//! # References
//!
//! - R. P. Gallant, R. J. Lambert, S. A. Vanstone, "Faster Point Multiplication
//!   on Elliptic Curves with Efficient Endomorphisms", CRYPTO 2001.
//!   <https://www.iacr.org/archive/crypto2001/21390189.pdf>
//! - S. Bowe, J. Grigg, D. Hopwood, "Halo: Recursive Proof Composition without
//!   a Trusted Setup", <https://eprint.iacr.org/2019/1021> (see the GLV section).
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
//! One-shot use is [`GlvParams::mul_glv`].

use alloc::vec::Vec;

use ff::PrimeField;
#[cfg(test)]
use ff::WithSmallOrderMulGroup;
use group::CurveAffine as _;

use crate::arithmetic::{CurveExt, VartimeField, mac, sbb};
use crate::{pallas, vesta};

mod private {
    /// Seals [`super::GlvParams`]: the lattice constants are curve-specific
    /// and verified in-crate; external implementations are not supported.
    pub trait Sealed {}
    impl Sealed for crate::pallas::Point {}
    impl Sealed for crate::vesta::Point {}
}

/// How a GLV-split scalar is expanded into digit columns, together with the
/// table of points those digits index.
///
/// `decompose` answers the first half of a GLV multiplication, turning `k`
/// into a pair of half-width integers. It says nothing about the second half,
/// which is how that pair becomes digits, and there is more than one answer:
/// [`Wnaf4`] recodes the two halves as independent width-4 wNAFs, while
/// `crate::glv_eisenstein::EisensteinNaf3` recodes them jointly as one
/// width-3 NAF over the Eisenstein integers. This trait is the shape they
/// share, so the laws below can be stated once.
///
/// Implementations must satisfy, for every `k` and every non-identity `P`:
///
/// - `mul(&table(P), &recode(k)) == P * k`, the identity included;
/// - `batch_tables(ps)[i] == table(&ps[i])`, so batching is only a shared
///   inversion and never a different answer;
/// - `recode` is a function of `k` alone, so a recoding may be built once and
///   reused across points.
///
/// # Both tables are the same construction
///
/// The two recodings look unlike each other and are not. A digit set carries
/// a group of cheap symmetries, the table stores one point per ORBIT, and the
/// lookup applies the group element it dropped. When the action is free the
/// table is exactly `|digits| / |G|` points:
///
/// |                  | group `G`   | digits                       | stored |
/// |------------------|-------------|------------------------------|--------|
/// | [`Wnaf4`]        | $\{\pm 1\}$ | $\pm 1, \pm 3, \pm 5, \pm 7$ | 4      |
/// | `EisensteinNaf3` | $\mu_6$     | the 48 odd classes           | 8      |
///
/// Freeness is what makes the saving exactly $|G|$: $-d \neq d$ for odd $d$,
/// just as no nonidentity unit fixes an odd class. So the familiar
/// signed-digit trick, storing only positive multiples and negating on
/// lookup, is the $|G| = 2$ case of what
/// `crate::glv_eisenstein` does with the six curve automorphisms.
///
/// Internal: it exists to pin the contract and to let one conformance suite
/// run against both recodings, not as a surface for callers.
// `recode`, `table` and `batch_tables` are the conformance suite's surface:
// production code reaches those operations through the inherent methods, and
// generic trait methods that are never instantiated cost nothing.
#[allow(dead_code)]
pub(crate) trait Recoding<C>
where
    C: GlvParams,
{
    /// The recoded scalar: the digit columns the ladder walks.
    type Digits;

    /// The precomputed multiples of `P` that the digits index.
    type Table;

    /// Recodes a scalar, independently of any point.
    fn recode(k: &C::ScalarExt) -> Self::Digits;

    /// Builds the table for one point.
    fn table(p: &C) -> Self::Table;

    /// Builds tables for many points, sharing one field inversion.
    fn batch_tables(points: &[C]) -> Vec<Self::Table>;

    /// Number of digit columns in a recoding.
    ///
    /// Hidden: the per-column interface exists so that [`Recoding::mul`] can
    /// be written once, not for callers to drive a ladder by hand.
    #[doc(hidden)]
    fn columns(digits: &Self::Digits) -> usize;

    /// Adds column `i`'s contribution to the accumulator, which is nothing at
    /// all for a zero column.
    #[doc(hidden)]
    fn add_column(table: &Self::Table, digits: &Self::Digits, i: usize, acc: &mut C);

    /// Walks the ladder: the point the digits name, against that table.
    ///
    /// This is Horner's rule, right to left, and it is the same fold for
    /// every recoding: an accumulator doubled once per column, with that
    /// column's contribution added in. Only [`Recoding::add_column`] differs,
    /// so the loop is provided here rather than written per implementation.
    fn mul(table: &Self::Table, digits: &Self::Digits) -> C {
        let len = Self::columns(digits);
        let mut acc = C::identity();
        for i in (0..len).rev() {
            // `acc` is still the identity on the first iteration; skip the
            // wasted doubling.
            if i + 1 < len {
                acc = acc.double();
            }
            Self::add_column(table, digits, i, &mut acc);
        }
        acc
    }
}

/// The laws every [`Recoding`] must satisfy, as functions the per-curve
/// property tests in this module and in [`crate::glv_eisenstein`] both run.
///
/// Stating them once is the point of the trait: the two recodings share no
/// ladder code, so without a common suite nothing forces them to mean the
/// same thing by "reuse gives the same answer".
#[cfg(test)]
pub(crate) mod conformance {
    use ff::PrimeField;
    use proptest::prelude::*;

    use super::{GlvParams, Recoding};

    /// Scalars drawn as four uniform `u64` limbs widened through
    /// `from_uniform_bytes`, so the whole field is reachable without modular
    /// bias. Shared with [`crate::glv_eisenstein`]'s property tests.
    pub(crate) fn scalar_strategy<F>() -> impl Strategy<Value = F>
    where
        F: PrimeField + ff::FromUniformBytes<64>,
    {
        proptest::array::uniform4(any::<u64>()).prop_map(|limbs| {
            let mut bytes = [0u8; 64];
            for (i, l) in limbs.iter().enumerate() {
                bytes[i * 8..(i + 1) * 8].copy_from_slice(&l.to_le_bytes());
            }
            F::from_uniform_bytes(&bytes)
        })
    }

    /// The defining law: the ladder computes `k * P`.
    pub(crate) fn agrees_with_mul<C, R>(p: &C, k: &C::ScalarExt)
    where
        C: GlvParams,
        R: Recoding<C>,
    {
        assert_eq!(R::mul(&R::table(p), &R::recode(k)), *p * *k);
    }

    /// Multiplication is additive in the scalar, so the recoding cannot be
    /// merely self-consistent: it has to respect the group structure.
    pub(crate) fn additive_in_scalar<C, R>(p: &C, a: &C::ScalarExt, b: &C::ScalarExt)
    where
        C: GlvParams,
        R: Recoding<C>,
    {
        let t = R::table(p);
        let sum = R::mul(&t, &R::recode(&(*a + *b)));
        let parts = R::mul(&t, &R::recode(a)) + R::mul(&t, &R::recode(b));
        assert_eq!(sum, parts, "mul is not additive in the scalar");
    }

    /// Zero and negation, which the additive law alone does not pin.
    pub(crate) fn zero_and_negation<C, R>(p: &C, k: &C::ScalarExt)
    where
        C: GlvParams,
        R: Recoding<C>,
    {
        let t = R::table(p);
        assert!(bool::from(
            R::mul(&t, &R::recode(&<C::ScalarExt as ff::Field>::ZERO)).is_identity()
        ));
        let both = R::mul(&t, &R::recode(k)) + R::mul(&t, &R::recode(&(-*k)));
        assert!(bool::from(both.is_identity()), "k*P + (-k)*P is not O");
    }

    /// Batching is a shared inversion, never a different answer.
    pub(crate) fn batch_matches_solo<C, R>(points: &[C], k: &C::ScalarExt)
    where
        C: GlvParams,
        R: Recoding<C>,
    {
        let digits = R::recode(k);
        for (t, p) in R::batch_tables(points).iter().zip(points) {
            assert_eq!(R::mul(t, &digits), R::mul(&R::table(p), &digits));
        }
    }

    /// A recoding is a function of `k` alone, so it survives reuse across
    /// points. This is what makes `Recoded`/`Decomposed` worth exposing.
    pub(crate) fn recoding_is_reusable<C, R>(points: &[C], k: &C::ScalarExt)
    where
        C: GlvParams,
        R: Recoding<C>,
    {
        let shared = R::recode(k);
        for p in points {
            assert_eq!(R::mul(&R::table(p), &shared), *p * *k);
        }
    }
}

/// The recoding [`crate::glv`] performs: two independent width-4 wNAF digit
/// strings, one per GLV half, sharing a column index.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Wnaf4;

impl<C> Recoding<C> for Wnaf4
where
    C: GlvParams,
{
    type Digits = Decomposed<C>;
    type Table = Table<C>;

    fn recode(k: &C::ScalarExt) -> Self::Digits {
        Decomposed::new(k)
    }

    fn table(p: &C) -> Self::Table {
        Table::new(p)
    }

    fn batch_tables(points: &[C]) -> Vec<Self::Table> {
        Table::batch(points)
    }

    fn columns(digits: &Self::Digits) -> usize {
        digits.len
    }

    fn add_column(table: &Self::Table, digits: &Self::Digits, i: usize, acc: &mut C) {
        Table::add_digit(acc, &table.t1, digits.digits1[i]);
        Table::add_digit(acc, &table.t2, digits.digits2[i]);
    }
}

/// Per-curve GLV constants: a short basis for the lattice
/// $\{(a, b) : a + b\lambda \equiv 0 \pmod n\}$ — where $n$ is the order of the
/// group (equivalently the scalar field modulus) and $\lambda$ =
/// [`Scalar::ZETA`] — together with the Babai rounding coefficients derived
/// from that basis.
///
/// [`Scalar::ZETA`]: ff::WithSmallOrderMulGroup::ZETA
///
/// The base field is required to implement [`VartimeField`], so that
/// `crate::glv_eisenstein`'s batch-affine ladder can reach the safegcd
/// inversion.
///
/// This trait is sealed; it is implemented for [`pallas::Point`] and
/// [`vesta::Point`].
pub trait GlvParams: CurveExt<Base: VartimeField> + private::Sealed {
    /// First short lattice vector `v1 = (V1A, -V1B_NEG)`.
    const V1A: u128;
    /// Magnitude of `v1`'s (negative) second component.
    const V1B_NEG: u128;
    /// Second short lattice vector `v2 = (V2A, V2B)`.
    const V2A: u128;
    /// `v2`'s (positive) second component.
    const V2B: u128;
    /// Babai coefficient `round(2^384 * V2B / n)`, little-endian limbs.
    const G1: [u64; 5];
    /// Babai coefficient `round(2^384 * V1B_NEG / n)`, little-endian limbs.
    const G2: [u64; 5];

    /// One-shot `k * self` via the GLV split — variable-time in `k` (see the
    /// module docs), identical in value to `self * k` (including `self` =
    /// identity).
    ///
    /// For repeated multiplications against the same point or the same
    /// scalar, use [`Table`] / [`Decomposed`] directly to reuse the
    /// precomputation.
    fn mul_glv(&self, k: &Self::ScalarExt) -> Self {
        if bool::from(self.is_identity()) {
            // k*O = O. Identity tables work (see [`Table::batch`]), but
            // building one still costs a field inversion; short-circuit.
            return Self::identity();
        }
        Table::new(self).mul(k)
    }
}

/// These constants are computed by `sage/glv_constants.sage`, which prints
/// this impl body verbatim.
///
/// The `constants` test (see the module's test suite) re-verifies the short
/// basis against Pallas's own $\lambda$ = [`Scalar::ZETA`] using field
/// arithmetic alone, and the Babai coefficients `G1`/`G2` against their
/// defining rounding using limb arithmetic alone; the `decompose` tests prove
/// that the decomposition reconstructs `k`. A wrong constant cannot pass them.
///
/// [`Scalar::ZETA`]: ff::WithSmallOrderMulGroup::ZETA
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

/// As for Pallas, these constants are computed by `sage/glv_constants.sage`,
/// and the `constants` and `decompose` tests re-verify them against Vesta's
/// own $\lambda$ = [`Scalar::ZETA`] and the Babai coefficients' defining
/// rounding.
///
/// [`Scalar::ZETA`]: ff::WithSmallOrderMulGroup::ZETA
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

/// Schoolbook multiply of `a` by `b` into `prod`, which must be zeroed and
/// hold exactly `a.len() + b.len()` limbs. Constant-time: a fixed loop
/// structure with explicit carry propagation.
fn schoolbook_mul(a: &[u64], b: &[u64], prod: &mut [u64]) {
    debug_assert_eq!(prod.len(), a.len() + b.len());
    for (i, &ai) in a.iter().enumerate() {
        let mut carry = 0u64;
        for (j, &bj) in b.iter().enumerate() {
            let (limb, c) = mac(prod[i + j], ai, bj, carry);
            prod[i + j] = limb;
            carry = c;
        }
        // First write to prod[i + b.len()] on each outer iteration.
        prod[i + b.len()] = carry;
    }
}

/// `round((g * k) / 2^384)` for a 5-limb `g` and 4-limb `k` — the Babai
/// coefficient. Fits `u128` (at most ~128 bits by construction).
fn round_mul_shift(g: &[u64; 5], k: &[u64; 4]) -> u128 {
    let mut prod = [0u64; 9];
    schoolbook_mul(g, k, &mut prod);
    // Bits >= 384 live in limbs 6..; round on bit 383 (top bit of limb 5).
    let round = prod[5] >> 63;
    (u128::from(prod[6]) | (u128::from(prod[7]) << 64)).wrapping_add(u128::from(round))
}

/// 256-bit product of two `u128`s, as little-endian limbs.
fn mul_u128(a: u128, b: u128) -> [u64; 4] {
    let mut prod = [0u64; 4];
    schoolbook_mul(
        &[a as u64, (a >> 64) as u64],
        &[b as u64, (b >> 64) as u64],
        &mut prod,
    );
    prod
}

/// 256-bit wrapping subtraction (two's complement).
fn sub256(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let (d0, borrow) = sbb(a[0], b[0], 0);
    let (d1, borrow) = sbb(a[1], b[1], borrow);
    let (d2, borrow) = sbb(a[2], b[2], borrow);
    let (d3, _) = sbb(a[3], b[3], borrow);
    [d0, d1, d2, d3]
}

/// Interprets a 256-bit two's-complement value as `(is_negative, magnitude)`,
/// taking the low 128 bits of the magnitude.
///
/// GLV decomposition guarantees `|x| < 2^127` for the values reached here
/// (asserted in debug builds, here and in [`wnaf_digits`], and checked by the
/// `decompose` tests), so the high limbs of the magnitude are always zero and
/// no information is lost.
fn signed_halves(x: [u64; 4]) -> (bool, u128) {
    // Guard the truncation itself: the discarded limbs must be the sign
    // extension of bit 127. Values of 2^128 or more would otherwise be
    // silently truncated before `wnaf_digits`' magnitude assertion could
    // observe them.
    let ext = if x[1] >> 63 == 0 { 0 } else { u64::MAX };
    debug_assert!(
        x[2] == ext && x[3] == ext,
        "GLV half does not fit in 128 bits"
    );
    let low = u128::from(x[0]) | (u128::from(x[1]) << 64);
    if x[3] >> 63 == 0 {
        (false, low)
    } else {
        // Two's-complement negation commutes with truncation to the low 128
        // bits, and the magnitude lives entirely there.
        (true, (!low).wrapping_add(1))
    }
}

/// The four little-endian limbs of a Pasta scalar. (Pasta scalars have a
/// 32-byte little-endian representation; the four 8-byte reads cover it
/// exactly.)
fn scalar_limbs<F>(k: &F) -> [u64; 4]
where
    F: PrimeField,
{
    let bytes = k.to_repr();
    let bytes: &[u8] = bytes.as_ref();
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        *limb = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().expect("8 bytes"));
    }
    limbs
}

/// GLV split: `k = k1 + k2 * lambda (mod n)` with `|k1|`, `|k2|` strictly
/// below `2^127`, each half returned as `(is_negative, magnitude)`.
pub(crate) fn decompose<C>(k: &C::ScalarExt) -> ((bool, u128), (bool, u128))
where
    C: GlvParams,
{
    let kl = scalar_limbs(k);
    let c1 = round_mul_shift(&C::G1, &kl);
    let c2 = round_mul_shift(&C::G2, &kl);
    // k1 = k - c1*V1A - c2*V2A   (two's complement over 256 bits)
    let k1 = sub256(sub256(kl, mul_u128(c1, C::V1A)), mul_u128(c2, C::V2A));
    // k2 = c1*V1B_NEG - c2*V2B   (v1.b = -V1B_NEG, v2.b = +V2B)
    let k2 = sub256(mul_u128(c1, C::V1B_NEG), mul_u128(c2, C::V2B));
    (signed_halves(k1), signed_halves(k2))
}

/// The GLV window for one base point: the odd multiples `{1, 3, 5, 7} * P` and
/// `{1, 3, 5, 7} * phi(P)` in affine coordinates. 512 bytes per table.
///
/// Build one with [`Table::new`], or many with one shared normalization via
/// [`Table::batch`].
#[derive(Clone, Copy, Debug)]
pub struct Table<C>
where
    C: GlvParams,
{
    /// `{1, 3, 5, 7} * P`
    t1: [C::AffineExt; 4],
    /// `{1, 3, 5, 7} * phi(P)`
    t2: [C::AffineExt; 4],
}

impl<C> Table<C>
where
    C: GlvParams,
{
    /// Builds the window for a single point (with no heap allocation, but
    /// one field inversion; amortize that with [`Table::batch`]).
    pub fn new(p: &C) -> Self {
        let proj = Self::window_proj(p);
        let mut affine = [C::AffineExt::identity(); 8];
        C::batch_normalize(&proj, &mut affine);
        Self::from_window(&affine)
    }

    /// Builds [`Table`]s for a batch of points with one shared
    /// batch normalization across all `8 * n` window entries — a single field
    /// inversion for the whole batch, where building each window individually
    /// pays one inversion per point.
    ///
    /// Identity inputs produce identity tables and may be mixed with
    /// non-identity points in the same batch.
    pub fn batch(points: &[C]) -> Vec<Table<C>> {
        let n = points.len();
        if n == 0 {
            return Vec::new();
        }
        let mut proj = Vec::with_capacity(n * 8);
        for p in points {
            proj.extend_from_slice(&Self::window_proj(p));
        }
        // One inversion for the whole batch.
        let mut affine = alloc::vec![C::AffineExt::identity(); n * 8];
        C::batch_normalize(&proj, &mut affine);
        affine.chunks_exact(8).map(Self::from_window).collect()
    }

    /// The eight projective window entries for one point:
    /// `[1P, 3P, 5P, 7P, 1phi(P), 3phi(P), 5phi(P), 7phi(P)]`. Projective
    /// group operations only (cheap additions and endomorphism, no
    /// inversions); the endomorphism multiples are taken via
    /// [`CurveExt::endo`] so they ride along in the caller's normalization.
    fn window_proj(p: &C) -> [C; 8] {
        let two_p = p.double();
        let mut w = [*p; 8];
        for i in 1..4 {
            w[i] = w[i - 1] + two_p;
        }
        for i in 0..4 {
            w[i + 4] = w[i].endo();
        }
        w
    }

    /// Assembles a table from one normalized 8-entry window.
    fn from_window(w: &[C::AffineExt]) -> Self {
        Table {
            t1: w[..4].try_into().expect("four P multiples"),
            t2: w[4..8].try_into().expect("four phi(P) multiples"),
        }
    }

    /// The base point P (= t1\[0\]) back as a projective point.
    #[cfg(test)]
    fn point(&self) -> C {
        C::from(self.t1[0])
    }

    /// `k * P` for the P encoded by this table, decomposing `k` on the spot.
    ///
    /// When one scalar meets many tables, decompose once with
    /// [`Decomposed::new`] and use [`Table::mul_decomposed`] instead.
    pub fn mul(&self, k: &C::ScalarExt) -> C {
        self.mul_decomposed(&Decomposed::new(k))
    }

    /// `k * P` for the P encoded by this table, via the Straus
    /// shared-doubling ladder over the GLV split. Identical to `P * k`
    /// (tested).
    pub fn mul_decomposed(&self, k: &Decomposed<C>) -> C {
        <Wnaf4 as Recoding<C>>::mul(self, k)
    }

    /// Adds `d * B` to `acc`, where `table` holds `{1, 3, 5, 7} * B` and `d`
    /// is a signed odd wNAF digit (zero adds nothing).
    fn add_digit(acc: &mut C, table: &[C::AffineExt; 4], d: i8) {
        if d != 0 {
            let mut a = table[(d.unsigned_abs() / 2) as usize];
            if d < 0 {
                a = -a;
            }
            *acc += a;
        }
    }
}

/// A scalar in GLV-decomposed, wNAF-recoded form, ready for
/// [`Table::mul_decomposed`].
///
/// Building this once per scalar hoists the decomposition and digit
/// recoding out of a loop that multiplies the same scalar against many
/// tables (e.g. one viewing key against a batch of ephemeral keys).
#[derive(Clone, Debug)]
pub struct Decomposed<C>
where
    C: GlvParams,
{
    digits1: [i8; MAX_WNAF_DIGITS],
    digits2: [i8; MAX_WNAF_DIGITS],
    /// Digit positions in use: the longer of the two halves' wNAF lengths.
    /// Both arrays are zero beyond their own half's length.
    len: usize,
    _curve: core::marker::PhantomData<C>,
}

impl<C> Decomposed<C>
where
    C: GlvParams,
{
    /// Decomposes `k` and recodes both halves as width-4 wNAF digits, with
    /// each half's sign folded into its digits.
    pub fn new(k: &C::ScalarExt) -> Self {
        let ((neg1, a1), (neg2, a2)) = decompose::<C>(k);
        let (digits1, len1) = wnaf_digits(a1, neg1);
        let (digits2, len2) = wnaf_digits(a2, neg2);
        Decomposed {
            digits1,
            digits2,
            len: len1.max(len2),
            _curve: core::marker::PhantomData,
        }
    }
}

/// Upper bound on the number of width-4 wNAF digits of a decomposition half:
/// an n-bit magnitude yields at most n + 1 digits, and [`decompose`] bounds
/// the halves below `2^127`.
const MAX_WNAF_DIGITS: usize = 128;

/// Width-4 wNAF digits of a u128 magnitude, lowest position first, with the
/// half's overall sign folded into the digits when `negate` is set.
fn wnaf_digits(a: u128, negate: bool) -> ([i8; MAX_WNAF_DIGITS], usize) {
    debug_assert!(a >> 127 == 0, "magnitude must be at most 127 bits");
    let mut digits = [0i8; MAX_WNAF_DIGITS];
    let mut n = 0;
    let mut k = a;
    while k != 0 {
        if k & 1 == 1 {
            let low = (k & 0xF) as i8;
            let d = if low >= 8 { low - 16 } else { low };
            digits[n] = if negate { -d } else { d };
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

/// One-shot `k * p` through the split wNAF recoding: variable-time in `k`,
/// identical in value to `p * k` (including `p` = identity).
///
/// The same call shape as `crate::glv_eisenstein::mul`, so the two
/// recodings are interchangeable at the call site.
pub fn mul<C>(p: &C, k: &C::ScalarExt) -> C
where
    C: GlvParams,
{
    p.mul_glv(k)
}

/// `k * p` for every `p`, building the per-point tables with one shared
/// field inversion and recoding the scalar once, then returning affine
/// results.
///
/// The same call shape as `crate::glv_eisenstein::batch_mul`. That one is
/// faster on a large batch, because it also shares an inversion across the
/// ladder itself; this one exists so a caller can pick the recoding without
/// changing anything else. Identity inputs are handled.
pub fn batch_mul<C>(points: &[C], k: &C::ScalarExt) -> Vec<C::AffineExt>
where
    C: GlvParams,
{
    if points.is_empty() {
        return Vec::new();
    }
    let decomposed = Decomposed::new(k);
    let proj: Vec<C> = Table::batch(points)
        .iter()
        .map(|t| t.mul_decomposed(&decomposed))
        .collect();
    let mut affine = alloc::vec![C::AffineExt::identity(); proj.len()];
    C::batch_normalize(&proj, &mut affine);
    affine
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::arithmetic::adc;
    use ff::Field;

    #[test]
    fn integer_multiplication_carry_boundaries() {
        assert_eq!(
            mul_u128(u128::MAX, u128::MAX),
            [1, 0, u64::MAX - 1, u64::MAX]
        );

        let pallas_scalar_max = [
            0x8c46eb2100000000,
            0x224698fc0994a8dd,
            0,
            0x4000000000000000,
        ];
        assert_eq!(
            round_mul_shift(&pallas::Point::G1, &pallas_scalar_max),
            0x93cd3a2c8198e2690c7c095a00000001
        );
        assert_eq!(
            round_mul_shift(&pallas::Point::G2, &pallas_scalar_max),
            0x49e69d1640a899538cb1279300000000
        );

        let vesta_scalar_max = [
            0x992d30ed00000000,
            0x224698fc094cf91b,
            0,
            0x4000000000000000,
        ];
        assert_eq!(
            round_mul_shift(&vesta::Point::G1, &vesta_scalar_max),
            0x93cd3a2c8198e2690c7c095a00000001
        );
        assert_eq!(
            round_mul_shift(&vesta::Point::G2, &vesta_scalar_max),
            0x49e69d1640a899538cb1279300000001
        );
    }

    /// Deterministic full-width scalars for the known-answer tests.
    fn scalars<F>(n: u64) -> impl Iterator<Item = F>
    where
        F: PrimeField,
    {
        (0..n).map(|i| {
            (F::from(0x9E37_79B9_7F4A_7C15u64 + i).square() + F::from(0x0123_4567_89AB_CDEFu64))
                .square()
                + F::from(i)
        })
    }

    /// Checks `g == round(2^384 * v / n)` for the curve's scalar modulus `n`,
    /// using limb arithmetic only: `g` is that rounding if and only if
    /// `|2^384 * v - g * n| < n/2` (an exact tie is impossible: `n` is odd,
    /// so `n/2` is not an integer).
    fn babai_coefficient_verify<C>(g: &[u64; 5], v: u128)
    where
        C: GlvParams,
    {
        // n = (n - 1) + 1, with n - 1 read out of the field type as -1.
        // n is odd, so n - 1 is even and adding the 1 back cannot carry.
        let mut n = scalar_limbs(&-C::ScalarExt::ONE);
        n[0] += 1;

        // 2^384 * v occupies limbs 6..8 of a 9-limb value.
        let mut target = [0u64; 9];
        target[6] = v as u64;
        target[7] = (v >> 64) as u64;

        let mut gn = [0u64; 9];
        schoolbook_mul(g, &n, &mut gn);

        // residual = 2^384 * v - g*n, two's complement over 9 limbs; negate
        // to a magnitude if the subtraction borrows.
        let mut residual = [0u64; 9];
        let mut borrow = 0;
        for (r, (&t, &m)) in residual.iter_mut().zip(target.iter().zip(gn.iter())) {
            let (limb, b) = sbb(t, m, borrow);
            *r = limb;
            borrow = b;
        }
        if borrow != 0 {
            let mut carry = 1;
            for limb in residual.iter_mut() {
                let (l, c) = adc(!*limb, 0, carry);
                *limb = l;
                carry = c;
            }
        }

        // |residual| < n/2 requires it to fit four limbs in the first place.
        assert!(
            residual[4..].iter().all(|&l| l == 0),
            "Babai residual far exceeds n"
        );
        // |residual| < n/2 <=> 2*|residual| < n, checked by subtraction:
        // n - 2*|residual| must not borrow (and equality is impossible, as
        // n is odd and the doubled value even).
        let mut doubled = [0u64; 5];
        doubled[0] = residual[0] << 1;
        for i in 1..5 {
            doubled[i] = (residual[i] << 1) | (residual[i - 1] >> 63);
        }
        let n5 = [n[0], n[1], n[2], n[3], 0];
        let mut borrow = 0;
        for (&ni, &di) in n5.iter().zip(doubled.iter()) {
            let (_, b) = sbb(ni, di, borrow);
            borrow = b;
        }
        assert!(borrow == 0, "g is not round(2^384 * v / n)");
    }

    /// The short-basis lattice relations, re-verified against the curve's
    /// own lambda (= [`Scalar::ZETA`]) using field arithmetic only:
    ///   V1A - V1B_NEG*lambda == 0  and  V2A + V2B*lambda == 0  (mod n),
    /// plus the Babai coefficients G1/G2 against their defining rounding.
    ///
    /// [`Scalar::ZETA`]: ff::WithSmallOrderMulGroup::ZETA
    fn constants_verify<C>()
    where
        C: GlvParams,
    {
        let lambda = C::ScalarExt::ZETA;
        let from = C::ScalarExt::from_u128;
        assert_eq!(from(C::V1A), from(C::V1B_NEG) * lambda, "v1 not in lattice");
        assert_eq!(from(C::V2A), -(from(C::V2B) * lambda), "v2 not in lattice");
        babai_coefficient_verify::<C>(&C::G1, C::V2B);
        babai_coefficient_verify::<C>(&C::G2, C::V1B_NEG);
    }

    /// The endomorphism / lambda pairing on the real curve, on the same
    /// projective `endo` the table build relies on: `phi(P) == ZETA * P`.
    fn endo_map_is_lambda<C>()
    where
        C: GlvParams,
    {
        let g = C::generator();
        for k in scalars::<C::ScalarExt>(64) {
            let p = g * k;
            assert_eq!(
                p.endo(),
                p * C::ScalarExt::ZETA,
                "phi(P) must equal ZETA_scalar * P"
            );
        }
    }

    /// The algebraic gate: k1 + k2*lambda == k (mod n) with both halves at most
    /// 2^127, for full-width scalars and the edge cases. Wrong GLV
    /// constants cannot pass this.
    fn decompose_reconstructs<C>()
    where
        C: GlvParams,
    {
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
    fn table_mul_matches_group_mul<C>()
    where
        C: GlvParams,
    {
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
    fn mul_glv_matches_operator<C>()
    where
        C: GlvParams,
    {
        let g = C::generator();
        for k in scalars::<C::ScalarExt>(64) {
            let p = g * (k + C::ScalarExt::ONE);
            assert_eq!(p.mul_glv(&k), p * k, "mul_glv must match operator");
        }
    }

    /// The batched table build equals the solo build, point by point.
    fn batch_tables_equal_solo<C>()
    where
        C: GlvParams,
    {
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

    /// Identity tables work both alone and alongside non-identity tables.
    fn identity_tables<C>()
    where
        C: GlvParams,
    {
        let identity = C::identity();
        let generator = C::generator();
        let k = C::ScalarExt::from(0xDEAD_BEEFu64);

        let solo = Table::new(&identity);
        assert_eq!(solo.point(), identity);
        assert_eq!(solo.mul(&k), identity);

        let batched = Table::batch(&[identity, generator]);
        assert_eq!(batched.len(), 2);
        assert_eq!(batched[0].point(), identity);
        assert_eq!(batched[0].mul(&k), identity);
        assert_eq!(batched[1].point(), generator);
        assert_eq!(batched[1].mul(&k), generator * k);
    }

    /// A reused [`Decomposed`] gives the same products as decomposing
    /// per-multiplication.
    fn decomposed_reuse_matches_fresh<C>()
    where
        C: GlvParams,
    {
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
                fn identity_table() {
                    identity_tables::<$curve>();
                }
                #[test]
                fn decomposed_reuse() {
                    decomposed_reuse_matches_fresh::<$curve>();
                }
            }
        };
    }

    /// Known-answer vectors. Generated by `sage/glv_test_vectors.sage`;
    /// regenerate with that script rather than editing the tables by hand.
    ///
    /// The property tests check both recodings against the crate's own `Mul`
    /// and against each other, which cannot catch a fault the reference
    /// shares with them. These expected points come from Sage's own curve
    /// arithmetic instead, so a wrong answer would have to be wrong the same
    /// way in two implementations that share no code.
    pub(crate) mod vectors {
        use super::*;
        use crate::arithmetic::CurveAffine;

        /// `k * G` on Pallas: `(k, x, y)` as little-endian limbs.
        ///
        /// Generated by `sage/glv_test_vectors.sage`; do not edit by hand.
        #[rustfmt::skip]
        pub(crate) const PALLAS_VECTORS: [([u64; 4], [u64; 4], [u64; 4]); 15] = [
            // one
            ([0x1, 0, 0, 0],
             [0x992d30ed00000000, 0x224698fc094cf91b, 0, 0x4000000000000000],
             [0x2, 0, 0, 0]),
            // two
            ([0x2, 0, 0, 0],
             [0x1303c567b0000003, 0xefee2ee4411acfc, 0, 0x1c00000000000000],
             [0x8aea5cdf3bfffffc, 0x17076ec9563fb75e, 0, 0x2b00000000000000]),
            // minus one
            ([0x8c46eb2100000000, 0x224698fc0994a8dd, 0, 0x4000000000000000],
             [0x992d30ed00000000, 0x224698fc094cf91b, 0, 0x4000000000000000],
             [0x992d30ecffffffff, 0x224698fc094cf91b, 0, 0x4000000000000000]),
            // zeta
            ([0x2aa9d2e050aa0e4f, 0xfed467d47c033af, 0x511db4d81cf70f5a, 0x6819a58283e528e],
             [0x7b7fd22f0201b548, 0x5270d29d19fc7d2, 0xd3552a23a8554e50, 0x2d33357cb532458e],
             [0x2, 0, 0, 0]),
            // zeta squared
            ([0x619d1840af55f1b1, 0x1259527ec1d4752e, 0xaee24b27e308f0a6, 0x397e65a7d7c1ad71],
             [0x1dad5ebdfdfe4aba, 0x1d1f8bd237ad3149, 0x2caad5dc57aab1b0, 0x12ccca834acdba71],
             [0x2, 0, 0, 0]),
            // half the order
            ([0xc623759080000000, 0x11234c7e04ca546e, 0, 0x2000000000000000],
             [0xc0ba6527af70acf7, 0x7865416c09f327e5, 0xdae60f42ce13f6e9, 0xc376da060916888],
             [0xdf56887b807cc21b, 0xc36d942037670d7a, 0xf372497ad0968cb2, 0x17a9db4ec26a7523]),
            // 2^127
            ([0, 0x8000000000000000, 0, 0],
             [0x69a0fa0803dc4843, 0x58095e73b31f2e48, 0xedb41d2ecba5a33a, 0x3ddc2602361790f9],
             [0x559d859b845787b0, 0x2b1850bd5b3a0a14, 0x1b879f9cb5b9bb17, 0x2350a7f5001193da]),
            // 2^127 - 1
            ([0xffffffffffffffff, 0x7fffffffffffffff, 0, 0],
             [0xe376fcaecc6e0b05, 0xef120698d47c1742, 0x7d3af336e2089900, 0x10a37079455f743c],
             [0x63cdf51f2000aaeb, 0x2760f1367385dddf, 0x5029035ce8bcd2ca, 0x1366e8c2c9e573b4]),
            // random 0
            ([0xfdce6ea5a6ee4db7, 0xe00504d671811e3c, 0xf46923051583328e, 0x132104c4fb5f15b9],
             [0xc5a2f8ce7f192802, 0x61a8bbd4b4157510, 0xe24cf6288aff7877, 0x36e72cbf397928e9],
             [0xc4f2bfd744116faa, 0x620550f383930a9a, 0x6e40ba8899fc5b9d, 0xf1bd4438df0f2f1]),
            // random 1
            ([0x3ba008c0de374107, 0x3d4f6399eff48ca1, 0x2741b50b6f2d327, 0x3466e07974bc6868],
             [0x15dd40b4a9534ffc, 0xcd1181f2522a39a7, 0x80a56e4d3ddab432, 0x3be381077241156],
             [0xd3ff1da7a01b8ba, 0x36787ce533f2e649, 0x2d37c7431c9e0ca4, 0x1ad718c0bdda49dc]),
            // random 2
            ([0xb9faec4a77f959e, 0x4a6646e8be79c3fd, 0xfc45ce72a8439fd9, 0x28bfc4ac5f6e87be],
             [0xc87d533e6cc32527, 0x2dd7b64d61bfa377, 0x8224c19237036c08, 0x1c66d4d7e6a4c046],
             [0x402cc8d9b43aa24d, 0x52167e42edd74051, 0x3fc68125dc81b9fe, 0x3f3dd7eeb2f0baaf]),
            // random 3
            ([0x9e33d7e44025bd7f, 0x2d843c477ac3f944, 0x5c06921f84629441, 0x17d3821f6f833913],
             [0xb9282277e7adf4b3, 0x87acf7994d6c96a1, 0x4ae061fe136d210, 0x3c558c05793b55da],
             [0x5e477c7f3878eaae, 0x5f51312749a96544, 0x16572e4d06f43514, 0xdf6fa226772dd76]),
            // random 4
            ([0xc2009548b79ee5a5, 0x6a5bbd72608485aa, 0xd32fb6749afa42f2, 0x22f26e04cc8ddf09],
             [0xbae02a480b04e1ad, 0xf40b69c9554360b2, 0xb338a6fd47453147, 0xf9551ebf1430b05],
             [0xa1b9bb8cd8674785, 0x12021f374a955125, 0x9bbd5f64a584d825, 0x164b2df90f887220]),
            // random 5
            ([0xfd35e85fa0c427c, 0xae1f07216ee0204e, 0x1b8cfd042cc3edac, 0xdd86b087824bf97],
             [0xdbfae910ff170147, 0xfb7b9eb17dfd914e, 0x380fd68cf0827828, 0x3502eeb171a67154],
             [0x453be311d72de1e8, 0x98fcf277dbdcd0f0, 0x2a5054a7b3100313, 0xa780105092304e4]),
            // random 6
            ([0x48e890686e74115a, 0x52bf1077a5c0e79f, 0xa0362cc2cf016b36, 0x3e6e5e5aa12aa594],
             [0x2954114ff466aaa6, 0x63ecae7bffcc7e43, 0x81a07bcc95b6e679, 0x12937ca56e65dfc9],
             [0x3dbedece4191483a, 0x21601d3f3dad50be, 0xd20b6c187dae7cb9, 0x10e4724b9863c7ec]),
        ];

        /// `k * G` on Vesta: `(k, x, y)` as little-endian limbs.
        ///
        /// Generated by `sage/glv_test_vectors.sage`; do not edit by hand.
        #[rustfmt::skip]
        pub(crate) const VESTA_VECTORS: [([u64; 4], [u64; 4], [u64; 4]); 15] = [
            // one
            ([0x1, 0, 0, 0],
             [0x8c46eb2100000000, 0x224698fc0994a8dd, 0, 0x4000000000000000],
             [0x2, 0, 0, 0]),
            // two
            ([0x2, 0, 0, 0],
             [0xed5f06de70000003, 0xefee2ee443109e0, 0, 0x1c00000000000000],
             [0xda3fa5fa2bfffffc, 0x17076ec9566fe174, 0, 0x2b00000000000000]),
            // minus one
            ([0x992d30ed00000000, 0x224698fc094cf91b, 0, 0x4000000000000000],
             [0x8c46eb2100000000, 0x224698fc0994a8dd, 0, 0x4000000000000000],
             [0x8c46eb20ffffffff, 0x224698fc0994a8dd, 0, 0x4000000000000000]),
            // zeta
            ([0x7b7fd22f0201b547, 0x5270d29d19fc7d2, 0xd3552a23a8554e50, 0x2d33357cb532458e],
             [0x2aa9d2e050aa0e50, 0xfed467d47c033af, 0x511db4d81cf70f5a, 0x6819a58283e528e],
             [0x2, 0, 0, 0]),
            // zeta squared
            ([0x1dad5ebdfdfe4ab9, 0x1d1f8bd237ad3149, 0x2caad5dc57aab1b0, 0x12ccca834acdba71],
             [0x619d1840af55f1b2, 0x1259527ec1d4752e, 0xaee24b27e308f0a6, 0x397e65a7d7c1ad71],
             [0x2, 0, 0, 0]),
            // half the order
            ([0xcc96987680000000, 0x11234c7e04a67c8d, 0, 0x2000000000000000],
             [0xb7df5b85a62e2cfb, 0xcdbe894f14c1bfb6, 0xf84fa0cc8fc9ffcb, 0x27855ad5b23eb036],
             [0x31f46767c566e7bf, 0x9f5cb835b7497d36, 0x3574ede136d59f54, 0x374f5576a6c9724f]),
            // 2^127
            ([0, 0x8000000000000000, 0, 0],
             [0x20f60677de11721d, 0xcc6aab1817a91d34, 0x69d2b790ea086318, 0x18331218993f554],
             [0x2352a52f0dc66700, 0x7a9034244b36ec4f, 0x9dfa93931879d470, 0x34f773d6e8d4141b]),
            // 2^127 - 1
            ([0xffffffffffffffff, 0x7fffffffffffffff, 0, 0],
             [0x3d56ce7288ee429e, 0x7d350382db76ed56, 0x534a6b0135e6f614, 0x3ec431bb0b76576f],
             [0x850930c1f035bdc4, 0xa6c91b9983f0d42d, 0x8f948bad2db9013a, 0x3ae7e35ca6b8a0c0]),
            // random 0
            ([0xfdce6ea5a6ee4db7, 0xe00504d671811e3c, 0xf46923051583328e, 0x132104c4fb5f15b9],
             [0xaac6a2ae8fb5c79f, 0xd35e8ac32674b100, 0x34ef8fccd7ea93ee, 0x1f21a986bad2e01f],
             [0xf77bf9b100e9b612, 0x7150b9cbd3f2571a, 0x65be2eb1070d6e47, 0x31e117224aa896b]),
            // random 1
            ([0x3ba008c0de374107, 0x3d4f6399eff48ca1, 0x2741b50b6f2d327, 0x3466e07974bc6868],
             [0x4605280a640545ac, 0x3a3450783f49a169, 0xb9162adccb7ccfb5, 0x24b316f325a25ecb],
             [0xacd1abc6702201f6, 0x40d33fa23e5225f, 0x1aecb0cd0efd5611, 0xfacbb27fe219331]),
            // random 2
            ([0xb9faec4a77f959e, 0x4a6646e8be79c3fd, 0xfc45ce72a8439fd9, 0x28bfc4ac5f6e87be],
             [0x8a03c367583e1f15, 0x220e6e7b15e1fb3f, 0x285b1109fed53939, 0x20a0d7b9fa392d8c],
             [0xcbccd57b89032cf8, 0x8d8d8243473efb44, 0x85faece5b5f66d72, 0x3d88861ba90c2ad6]),
            // random 3
            ([0x9e33d7e44025bd7f, 0x2d843c477ac3f944, 0x5c06921f84629441, 0x17d3821f6f833913],
             [0x86d8e5c0048fbeef, 0x5331a3de05e659e5, 0x5dfaf9749765eb73, 0x78c8965235b4429],
             [0x35da430bcae34aaa, 0x133ca03691efbd6, 0x62f8012d4d4cff11, 0x386a5219f0326b5a]),
            // random 4
            ([0xc2009548b79ee5a5, 0x6a5bbd72608485aa, 0xd32fb6749afa42f2, 0x22f26e04cc8ddf09],
             [0xeed46cbc4252d60f, 0x6b2ca66990a4972b, 0x6dbaf78e21bfaa31, 0x3fa797508b012682],
             [0x88975f8b89893189, 0x7f559b69ce057b64, 0x982859e12302e425, 0x14ad45dbd098e139]),
            // random 5
            ([0xfd35e85fa0c427c, 0xae1f07216ee0204e, 0x1b8cfd042cc3edac, 0xdd86b087824bf97],
             [0xf38547b2c7f747de, 0xe0e0bab944057631, 0xa4caecd2b371f1ea, 0x2b9e6b4838115fb2],
             [0x954ba7c20e83682e, 0x27f2152fb43f4afd, 0x8fd88cb1bfca6780, 0x284a4fe1798d0955]),
            // random 6
            ([0x48e890686e74115a, 0x52bf1077a5c0e79f, 0xa0362cc2cf016b36, 0x3e6e5e5aa12aa594],
             [0xd65e2d301e77b640, 0xf4ee1e16056f88c1, 0x4dbf063d2fb8b7f4, 0x31d09821f60f49a1],
             [0xb8ebcd9c64f7d106, 0x52784e86422e9029, 0x7fb292b5ba3202c1, 0x192c4e7f8d935b7a]),
        ];

        /// One shared scalar for the Pallas batch below.
        ///
        /// Generated by `sage/glv_test_vectors.sage`; do not edit by hand.
        #[rustfmt::skip]
        pub(crate) const PALLAS_BATCH_K: [u64; 4] =
            [0xc99ba0549081818e, 0xc9f76e5dc64b091a, 0xd893c97c7bbda5f2, 0x31cfe3d2aebbab79];

        /// A batch against that scalar: `(s, x, y)` with `P = s*G`
        /// and `k*P = (x, y)`.
        ///
        /// Generated by `sage/glv_test_vectors.sage`; do not edit by hand.
        #[rustfmt::skip]
        pub(crate) const PALLAS_BATCH: [([u64; 4], [u64; 4], [u64; 4]); 12] = [
            ([0x983863f621af356a, 0xfcdb5f0259fc14f9, 0xef9b2f8ac0096464, 0x2f9d8832912722dc],
             [0x6241b62a2e51ef89, 0xe7273968dbc0b1f4, 0xb42fc9858c0655d1, 0x16ae323ccbc38a1a],
             [0x9788b18474a23a1f, 0x417ff59ae9d54cb, 0x51a547faebf45732, 0x1634def4ee6a05ca]),
            ([0x17c68b0404270e9, 0xdaa1f84b200a990a, 0xdac643e9f076baed, 0xdcb616172687715],
             [0x65f8bdabca56a221, 0xac94b13951bb2fb8, 0xeaaf4fa7bcdcaccf, 0x14f58da7308b0e77],
             [0x91bb04b1ed174cd6, 0x14d659e33e509c92, 0xc8ec740705f1e8d6, 0x37061cc79f9f8d5d]),
            ([0xd06b50747cb4c21f, 0x2bc4e50938b7b5ae, 0xd110865bfea78db4, 0x34155318a4e07bc6],
             [0xce92a457796e895c, 0xeb0905e6544b07a9, 0xefc9ea67744a17f2, 0x228bb1afff7b58cb],
             [0xaf849d0ab1824605, 0xe4c17946a2355b49, 0xabb8f031bd5c6cc4, 0xbb9f15b2b7af710]),
            ([0x96f2f83cb79661c9, 0xba483a6ce84c5d0f, 0x38dc269d0ad25316, 0x3545bafaa9b903e2],
             [0x3748945dbc22dff, 0xd1972b459ee1d0a2, 0x3ff9014a812ba094, 0x1ff16517b937b78c],
             [0xfa0f7b4fa394b4dc, 0xfc1f707339c072cf, 0x3c1fbbd65ffdd4c8, 0x37d4996eb4c7b7b3]),
            ([0xf44ced56be715e08, 0x13815e9c87b16830, 0xee29cfabcf965fbf, 0x378c8b975840e2e1],
             [0xc7d1792018cc02a3, 0x98f67af807c255b4, 0x46e9406d0a3526e6, 0x4d06af86dffb8de],
             [0x5ad1ecf3871b5385, 0xb64972ece41feeff, 0xd96165862fe5ca3b, 0x22d153f535965db]),
            ([0xd2e4810c256b20de, 0x43e21617e1d0aa4e, 0x6dc00bd4a2914131, 0x3022a6ba6444cd30],
             [0x320397d96071d649, 0xc583a12e409237ef, 0x1670a9066289f301, 0x28d948266593cf3a],
             [0xa165d4d1a4905119, 0xce8893ebcf854f08, 0x1d1d30c4884942a, 0x2eb1adba5567b824]),
            ([0x8c0560e48d7ac65d, 0x9b5eabebc8782989, 0xca66c21efedfd1e3, 0x218f0fdc5d191d57],
             [0xe2a170603c672aba, 0x91aacdb78cc4bd26, 0x9b1b76083b3b8972, 0xebd74674a1e29da],
             [0xa061180aa4fab0fb, 0x6d8e41cb648e4bb5, 0x8346c90fa05e3b49, 0x27ce37a8b6a8449d]),
            ([0xf412cb010ebf204e, 0x5f8218762d58e3ee, 0x61121bf49fed1964, 0x1a62341c77c3e3a0],
             [0xf3527a30afcd4c0, 0x626d1837461e21d7, 0x4e4e9d7ed9b54e5a, 0xc9b50ea2a4ea337],
             [0xc922d4f18348d536, 0xabcab4165af17ebe, 0x8e000d9f2f786545, 0x16512d43d9414ec3]),
            ([0xc63062b8a8716fb9, 0xd09dc7fcbf8ee155, 0xbca98ba0003bf1df, 0x39f0ac94228fa367],
             [0x538a64ba8f2b1edd, 0x79de939d955e9cf0, 0x95b19f17ae76f027, 0x4dc4e3a9c6b9780],
             [0x3641a6b0e1caa59, 0x283875c800778a47, 0xb9eba6b59b450797, 0x31f5892701016420]),
            ([0x842a35eaae197d62, 0x1ce7073dc73d4e7c, 0xead3cd19432b1ad4, 0x35c136b547d0d61f],
             [0x76b5ff7786074669, 0x4cc1f45996cf7675, 0x8055d1f1860397c6, 0x21f9f074adeb0254],
             [0x935d0c4e249d6e9e, 0x557591922afc2cdd, 0x706f058d2c647e63, 0xb730502404eeb7e]),
            ([0x1ad56f3fc6ee0205, 0x6363874972ffc0b9, 0xa397bcb4d426c79a, 0x2bb2346de32f977],
             [0xa8d1775bef2b1922, 0x1e8e044a8ea65df8, 0x8e6ad6f1c518391, 0x32fe1e98b923a511],
             [0x9bc195cd216019cf, 0x80c18c7c46ce6f35, 0xd0f4e6f05e3e88cf, 0x2c9a09619d17f88]),
            ([0x5b515f0a484d6a78, 0x9fda63c0cc9ae2bb, 0x4e43b31603ad1f38, 0x33579d6dff2a6276],
             [0xe356891d1d502c47, 0x5887ffbb32ee476, 0x127ee4655dabb6a9, 0x119b86fa32cc036b],
             [0x2384c64d3842a74d, 0x31243be15a08cd7d, 0x9eb169cdcb362563, 0x8577331e089c74f]),
        ];

        /// One shared scalar for the Vesta batch below.
        ///
        /// Generated by `sage/glv_test_vectors.sage`; do not edit by hand.
        #[rustfmt::skip]
        pub(crate) const VESTA_BATCH_K: [u64; 4] =
            [0xc99ba0549081818e, 0xc9f76e5dc64b091a, 0xd893c97c7bbda5f2, 0x31cfe3d2aebbab79];

        /// A batch against that scalar: `(s, x, y)` with `P = s*G`
        /// and `k*P = (x, y)`.
        ///
        /// Generated by `sage/glv_test_vectors.sage`; do not edit by hand.
        #[rustfmt::skip]
        pub(crate) const VESTA_BATCH: [([u64; 4], [u64; 4], [u64; 4]); 12] = [
            ([0x983863f621af356a, 0xfcdb5f0259fc14f9, 0xef9b2f8ac0096464, 0x2f9d8832912722dc],
             [0x807e4cb552b72e49, 0x91ef77dd3b2df14f, 0x78e5346964c6ffcd, 0xc22c59598025cc9],
             [0x9c810379f5c38508, 0x73c7061b10b1ffb9, 0x19c923679a44ee77, 0x1e53bd2e20458be1]),
            ([0x17c68b0404270e9, 0xdaa1f84b200a990a, 0xdac643e9f076baed, 0xdcb616172687715],
             [0x37fa85ac7f0fc001, 0x445e1b7f3be80d01, 0x5fa335b1e10a8024, 0x380e354528d22b8c],
             [0x64ba140e40ccc22c, 0x648f5c3af5ddbbf9, 0x67fb54968965789c, 0x3da46a162992f77a]),
            ([0xd06b50747cb4c21f, 0x2bc4e50938b7b5ae, 0xd110865bfea78db4, 0x34155318a4e07bc6],
             [0x8449a173239d4d55, 0xb1e6e067567866f3, 0x45cab55f22c1c066, 0x1e382634c7c0a70d],
             [0xfb8f266afe573d7c, 0x461d9a28d235b7d1, 0x38fbb8bbb697fe9f, 0x306a1703a2bb10e0]),
            ([0x96f2f83cb79661c9, 0xba483a6ce84c5d0f, 0x38dc269d0ad25316, 0x3545bafaa9b903e2],
             [0x82316890a801e94c, 0x62163c03c58eafe1, 0xfc4e73846135d473, 0x636e66dbf7ded4c],
             [0x39024542c993398e, 0x14b9ada1778380e1, 0xc3139402ae9d0011, 0x236d85ac60b10b0a]),
            ([0xf44ced56be715e08, 0x13815e9c87b16830, 0xee29cfabcf965fbf, 0x378c8b975840e2e1],
             [0x836bab68400f1da4, 0x990d8f522fc4655, 0x688edc3e58a6bf1a, 0xd0b79c2d6bbe2fe],
             [0x109aa3c2d8f0f4f9, 0x98df15a25ab2af87, 0x265ec2242020fe0, 0x32191ef8dfb18a50]),
            ([0xd2e4810c256b20de, 0x43e21617e1d0aa4e, 0x6dc00bd4a2914131, 0x3022a6ba6444cd30],
             [0x4bc6f68a9404753b, 0x869e27a3310bb303, 0xbcaaf2f2ee472a57, 0x12928e72f3d2fdb1],
             [0xa089304abd16e9dd, 0x73f0179af00c4a7f, 0x5283aa97ae019195, 0xbe1bb0d8a621184]),
            ([0x8c0560e48d7ac65d, 0x9b5eabebc8782989, 0xca66c21efedfd1e3, 0x218f0fdc5d191d57],
             [0x3d99f8859911ae81, 0x6e04f808a46ed3b2, 0x50a07a178d72dc6c, 0x3f8b89fcb7e74e5],
             [0x27536f94efe07602, 0x67e59408768a37, 0xdcbe966d00ddc25, 0xdc9bce302ad9164]),
            ([0xf412cb010ebf204e, 0x5f8218762d58e3ee, 0x61121bf49fed1964, 0x1a62341c77c3e3a0],
             [0xbaf196bae295bae5, 0xac4fa4832a96ba06, 0x409a3aca7c6ca62a, 0x11ebee29b772704b],
             [0x9f30d9ad5dc3b302, 0xa4992b6979d08d2a, 0x94ba306bebeb5718, 0x27d0bf9d0114263e]),
            ([0xc63062b8a8716fb9, 0xd09dc7fcbf8ee155, 0xbca98ba0003bf1df, 0x39f0ac94228fa367],
             [0xa94fbfaf84b1ff5a, 0x5c73dc2b0e6ed8c4, 0x2d60b6808d378b8, 0x38226d91051a7a88],
             [0xda867c579e5c5b52, 0x771cc4491dd85719, 0x9b036d11caaf6fad, 0x15925f238d3266dd]),
            ([0x842a35eaae197d62, 0x1ce7073dc73d4e7c, 0xead3cd19432b1ad4, 0x35c136b547d0d61f],
             [0xe9cf0078cd01b055, 0x41426b5a9533930b, 0x34fbef371f7d07bb, 0x121df8307ee6ba84],
             [0x699514ef507ae213, 0x3ca9a07067dac05a, 0x666100d79c168926, 0x1703894f17e31b23]),
            ([0x1ad56f3fc6ee0205, 0x6363874972ffc0b9, 0xa397bcb4d426c79a, 0x2bb2346de32f977],
             [0xca9108707cd91e90, 0x870b7e96096d594b, 0xfa4bf75d188f1bff, 0xe93f8f4ad4b3689],
             [0x606362ac8fc8f6be, 0x22c42359229fcaa7, 0xc303f812a7c1d63d, 0x30b349231d8a0e16]),
            ([0x5b515f0a484d6a78, 0x9fda63c0cc9ae2bb, 0x4e43b31603ad1f38, 0x33579d6dff2a6276],
             [0x81f11edf8ddcccea, 0xb98d02137b783af1, 0x286501502fac2ac1, 0x1e6394a50e6e9c0f],
             [0x4741fe59a199b4c0, 0x89e2fb64dcad99ab, 0xfc13ff850dc8f9ee, 0x12ccc7a3fbc898d8]),
        ];

        /// A field element from little-endian 64-bit limbs.
        pub(crate) fn from_limbs<F>(l: &[u64; 4]) -> F
        where
            F: PrimeField,
        {
            let mut repr = F::Repr::default();
            let bytes = repr.as_mut();
            for (i, limb) in l.iter().enumerate() {
                bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
            }
            F::from_repr(repr).expect("vector limb value is in range")
        }

        /// Every vector, against the native ladder, `mul_glv`, and a recoding
        /// driven through the [`Recoding`] trait.
        pub(crate) fn check<C, R>(vs: &[([u64; 4], [u64; 4], [u64; 4])])
        where
            C: GlvParams,
            C::Base: PrimeField,
            R: Recoding<C>,
        {
            for (k, x, y) in vs {
                let k: C::ScalarExt = from_limbs(k);
                let g = C::generator();
                let want = C::AffineExt::from_xy_unchecked(from_limbs(x), from_limbs(y));
                assert!(bool::from(want.is_on_curve()), "vector is off the curve");
                assert_eq!((g * k).to_affine(), want, "native Mul missed a vector");
                assert_eq!(g.mul_glv(&k).to_affine(), want, "mul_glv missed a vector");
                assert_eq!(
                    R::mul(&R::table(&g), &R::recode(&k)).to_affine(),
                    want,
                    "recoding missed a vector"
                );
            }
        }

        /// A batch vector set, against the non-GLV reference: every lane's
        /// expected point is reproduced by the native `Mul`, and by `glv`'s
        /// batched tables driven with one shared recoding.
        pub(crate) fn check_batch<C, R>(k: &[u64; 4], vs: &[([u64; 4], [u64; 4], [u64; 4])])
        where
            C: GlvParams,
            C::Base: PrimeField,
            R: Recoding<C>,
        {
            let k: C::ScalarExt = from_limbs(k);
            let g = C::generator();
            let points: Vec<C> = vs
                .iter()
                .map(|(s, ..)| g * from_limbs::<C::ScalarExt>(s))
                .collect();
            let want: Vec<C::AffineExt> = vs
                .iter()
                .map(|(_, x, y)| C::AffineExt::from_xy_unchecked(from_limbs(x), from_limbs(y)))
                .collect();

            for (p, w) in points.iter().zip(&want) {
                assert!(bool::from(w.is_on_curve()), "vector is off the curve");
                assert_eq!((*p * k).to_affine(), *w, "native Mul missed a batch lane");
            }

            let digits = R::recode(&k);
            for (t, w) in R::batch_tables(&points).iter().zip(&want) {
                assert_eq!(
                    R::mul(t, &digits).to_affine(),
                    *w,
                    "batched table missed a lane"
                );
            }
        }

        #[test]
        fn pallas_wnaf() {
            check::<pallas::Point, Wnaf4>(&PALLAS_VECTORS);
        }

        #[test]
        fn vesta_wnaf() {
            check::<vesta::Point, Wnaf4>(&VESTA_VECTORS);
        }

        /// The free entry points agree with the vectors too, so the shape a
        /// caller actually reaches for is covered.
        #[test]
        fn free_functions_match_vectors() {
            use group::Curve as _;
            for (k, x, y) in PALLAS_VECTORS.iter() {
                let k: <pallas::Point as CurveExt>::ScalarExt = from_limbs(k);
                let g = <pallas::Point as group::Group>::generator();
                let want = <pallas::Point as CurveExt>::AffineExt::from_xy_unchecked(
                    from_limbs(x),
                    from_limbs(y),
                );
                assert_eq!(super::super::mul(&g, &k).to_affine(), want);
                assert_eq!(super::super::batch_mul(&[g], &k), alloc::vec![want]);
                #[cfg(feature = "glv-eisenstein")]
                {
                    assert_eq!(
                        crate::glv_eisenstein::mul(&g, &k).to_affine(),
                        want,
                        "the two modules' free `mul` must agree"
                    );
                    assert_eq!(
                        crate::glv_eisenstein::batch_mul(&[g], &k),
                        alloc::vec![want]
                    );
                }
            }
        }

        #[test]
        fn pallas_wnaf_batch() {
            check_batch::<pallas::Point, Wnaf4>(&PALLAS_BATCH_K, &PALLAS_BATCH);
        }

        #[test]
        fn vesta_wnaf_batch() {
            check_batch::<vesta::Point, Wnaf4>(&VESTA_BATCH_K, &VESTA_BATCH);
        }
    }

    glv_tests!(pallas_glv, pallas::Point);
    glv_tests!(vesta_glv, vesta::Point);

    /// Edge-case scalars exercised through the FULL `mul_glv` path (not just
    /// `decompose`): the additive/multiplicative identities and their
    /// negations, lambda and its neighbours (the decomposition's own axis), and
    /// the half-width boundary where k1/k2 magnitudes live.
    fn edge_case_matrix<C>()
    where
        C: GlvParams,
    {
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
        // k*O = O for every scalar, including 0.
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

    /// Loads a Pasta scalar from its four little-endian limbs.
    fn scalar_from_limbs<F>(limbs: [u64; 4]) -> F
    where
        F: PrimeField,
    {
        let mut bytes = [0u8; 32];
        for (chunk, limb) in bytes.chunks_exact_mut(8).zip(limbs.iter()) {
            chunk.copy_from_slice(&limb.to_le_bytes());
        }
        let mut repr = F::Repr::default();
        repr.as_mut().copy_from_slice(&bytes);
        F::from_repr(repr).unwrap()
    }

    /// The lattice-constructed Babai-boundary scalars, computed by
    /// `sage/glv_boundary_scalars.sage` (which prints these constants
    /// verbatim); provenance in [`babai_boundary_witness`].
    const PALLAS_BOUNDARY_SCALAR: [u64; 4] = [
        0xf1616cb5a3632910,
        0xa487c2df3b0d145f,
        0xd70a3d98c2549413,
        0x3d70a3d70a3d70a3,
    ];
    const VESTA_BOUNDARY_SCALAR: [u64; 4] = [
        0x17b30ff8ae506c98,
        0xecc8ab77c7c0d84f,
        0xd70a3d86799d8e38,
        0x3d70a3d70a3d70a3,
    ];

    /// A scalar constructed (by lattice reduction over the joint residues
    /// `G1*k mod 2^384`, `G2*k mod 2^384` — see
    /// `sage/glv_boundary_scalars.sage`) to sit on the Babai rounding
    /// boundary: flipping bit 127 of `G2` — a corruption that the suite
    /// predating `babai_coefficient_verify` provably accepted, since it
    /// leaves the `round_mul_shift` known-answer test unmoved and shifts
    /// `c2` for only ~2^-16 of random scalars — moves `c2` by one *here*
    /// and pushes `|k2|` past the half-width bound that `wnaf_digits` and
    /// `MAX_WNAF_DIGITS` rely on.
    ///
    /// With the shipped constants the witness must behave like any other
    /// scalar; the second half of the test pins its boundary geometry.
    fn babai_boundary_witness<C>(limbs: [u64; 4])
    where
        C: GlvParams,
    {
        let k = scalar_from_limbs::<C::ScalarExt>(limbs);
        assert_eq!(
            scalar_limbs(&k),
            limbs,
            "witness must be a canonical scalar"
        );

        // In bounds, reconstructs, and multiplies correctly as shipped.
        let ((neg1, a1), (neg2, a2)) = decompose::<C>(&k);
        assert!(
            a1 >> 127 == 0 && a2 >> 127 == 0,
            "witness must be in bounds"
        );
        let s1 = C::ScalarExt::from_u128(a1);
        let s2 = C::ScalarExt::from_u128(a2);
        let (s1, s2) = (if neg1 { -s1 } else { s1 }, if neg2 { -s2 } else { s2 });
        assert_eq!(s1 + s2 * C::ScalarExt::ZETA, k, "witness must reconstruct");
        assert_eq!(C::generator().mul_glv(&k), C::generator() * k);

        // The boundary geometry under the bit flip.
        let mut g2_bad = C::G2;
        g2_bad[1] ^= 1 << 63;
        let kl = scalar_limbs(&k);
        let c1 = round_mul_shift(&C::G1, &kl);
        let c2 = round_mul_shift(&C::G2, &kl);
        assert_eq!(
            round_mul_shift(&g2_bad, &kl),
            c2 + 1,
            "witness must straddle the rounding boundary"
        );
        let k2_bad = sub256(mul_u128(c1, C::V1B_NEG), mul_u128(c2 + 1, C::V2B));
        let mag = if k2_bad[3] >> 63 == 1 {
            sub256([0; 4], k2_bad)
        } else {
            k2_bad
        };
        assert!(
            mag[2] == 0 && mag[3] == 0,
            "witness |k2'| stays below 2^128"
        );
        let mag = u128::from(mag[0]) | (u128::from(mag[1]) << 64);
        assert!(
            mag >> 127 == 1,
            "flipped G2 must push |k2| past 2^127 at this scalar"
        );
    }

    #[test]
    fn babai_boundary_pallas() {
        babai_boundary_witness::<pallas::Point>(PALLAS_BOUNDARY_SCALAR);
    }
    #[test]
    fn babai_boundary_vesta() {
        babai_boundary_witness::<vesta::Point>(VESTA_BOUNDARY_SCALAR);
    }

    /// Native (constant-time) `Mul` against the whole GLV pipeline at the
    /// boundary scalars — nothing but two multiplications and an equality.
    /// Native multiplication never reads the GLV constants, so the sides
    /// only diverge if the GLV path regresses, and at these scalars it
    /// does so for exactly the suite-invisible corruption identified
    /// above (`G2[1] ^= 1 << 63`): the decomposition half leaves its
    /// 2^127 bound and the pipeline panics on a bound assertion in debug
    /// builds — how tests run. (Any corruption small enough to evade the
    /// known-answer tests keeps `|k2| < 2^128`, which the wNAF ladder
    /// still multiplies correctly, so release products stay numerically
    /// right; the broken invariant is the observable, not a wrong point.)
    /// On the pre-`babai_coefficient_verify` code, this test alone
    /// detects the flip; nothing else in that suite did.
    fn native_vs_glv_boundary<C>(limbs: [u64; 4])
    where
        C: GlvParams,
    {
        let k = scalar_from_limbs::<C::ScalarExt>(limbs);
        let p = C::generator() * (k + C::ScalarExt::ONE);
        assert_eq!(p.mul_glv(&k), p * k, "GLV must agree with native Mul");
        assert_eq!(C::generator().mul_glv(&k), C::generator() * k);
    }

    #[test]
    fn native_vs_glv_boundary_pallas() {
        native_vs_glv_boundary::<pallas::Point>(PALLAS_BOUNDARY_SCALAR);
    }
    #[test]
    fn native_vs_glv_boundary_vesta() {
        native_vs_glv_boundary::<vesta::Point>(VESTA_BOUNDARY_SCALAR);
    }

    /// Property-based tests: scalars are drawn as four uniform u64 limbs
    /// widened through `from_uniform_bytes` (so the whole field is reachable
    /// without modular bias), and points as `G*(s+1)`.
    mod pbt {
        use group::Group;
        use proptest::prelude::*;

        use super::*;
        use crate::glv::conformance::scalar_strategy;

        macro_rules! glv_pbt {
            ($mod_name:ident, $curve:ty) => {
                mod $mod_name {
                    use super::*;

                    type Scalar = <$curve as CurveExt>::ScalarExt;

                    proptest! {
                        /// The shared `Recoding` laws, for this module's
                        /// width-4 wNAF recoding.
                        #[test]
                        fn recoding_laws(
                            s in scalar_strategy::<Scalar>(),
                            a in scalar_strategy::<Scalar>(),
                            b in scalar_strategy::<Scalar>(),
                        ) {
                            use crate::glv::conformance as law;
                            let p = <$curve>::generator() * (s + Scalar::ONE);
                            let ps = [p, p.double(), p + <$curve>::generator()];
                            law::agrees_with_mul::<$curve, Wnaf4>(&p, &a);
                            law::additive_in_scalar::<$curve, Wnaf4>(&p, &a, &b);
                            law::zero_and_negation::<$curve, Wnaf4>(&p, &a);
                            law::batch_matches_solo::<$curve, Wnaf4>(&ps, &a);
                            law::recoding_is_reusable::<$curve, Wnaf4>(&ps, &a);
                        }

                        /// For all P != O, k: P.mul_glv(k) == P * k.
                        #[test]
                        fn mul_glv_matches_mul(
                            s in scalar_strategy::<Scalar>(),
                            k in scalar_strategy::<Scalar>(),
                        ) {
                            let p = <$curve>::generator() * (s + Scalar::ONE);
                            prop_assert_eq!(p.mul_glv(&k), p * k);
                        }

                        /// For all k: the GLV split reconstructs k with half-width parts.
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

                        /// For all points: batched tables act identically to solo tables.
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

                        /// For all k reused across points: hoisted decomposition == fresh.
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
