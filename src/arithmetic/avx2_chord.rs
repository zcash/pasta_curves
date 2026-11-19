//! Four-way AVX2 affine chord additions for the fixed-base sums.
//!
//! The fixed-base sums (`glv::fixed_base`) reduce lists of affine points pairwise,
//! one tree level at a time, sharing one inversion per level (Montgomery's
//! trick). Each addition costs six field multiplications, independent across
//! the additions of a level, so [`chord_level`] runs eight at a time: two
//! vectors of four lanes, in the radix-2^29 arithmetic of
//! [`super::avx2_sqrt`] (availability and the AVX2 guarantee are documented
//! there). Every safe entry checks CPU and OS support at runtime; only AVX2 is used.
//!
//! Public inputs only: the production entry is
//! [`PreparedFixedBaseSum::sum_batch_vartime`](super::PreparedFixedBaseSum::sum_batch_vartime),
//! through `glv::fixed_base` and its radix-29 bucket reducer. Neither ordinary
//! `Field::invert` nor native scalar multiplication calls this module.
//!
//! Points stay in radix 2^29 ([`Point29`], canonical Montgomery limbs below
//! `p`) for the whole reduction, so a level loads and stores limbs without
//! base conversion, and fixed tables can be stored in this form once.
//!
//! # Domain
//!
//! [`mul`] computes `a b / 2^261`, while the scalar fields store `x 2^256`.
//! Scaling one operand of each product by `2^5` (a shift, [`shl5`]) makes it
//! the scalar Montgomery product, so values never leave the scalar fields'
//! Montgomery domain and need no conversion multiplications. Each value
//! carries a scale `s`: it is held as `s x 2^256 mod p`, lazily reduced, and
//! [`mul`] of scales `s1` and `s2` has scale `s1 s2 / 32`.
//!
//! # Bounds
//!
//! [`mul`] is exact whenever limbs 0 to 7 of both inputs are below 2^29 and
//! both values are below 2^261; its result is below `a b / 2^261 + p`. Both
//! moduli are `2^254 + d` with `d < 2^126`, so `p / 2^261` exceeds `1/128` by
//! a factor below `1 + 2^-128`, which the stated bounds absorb.
//! [`chord_level`] states the bound of each step. All values stay below
//! `96 p < 2^261`. Subtractions add a
//! multiple of `p` whose limbs dominate the subtrahend's ([`kp`]), and every
//! level's outputs are reduced to canonical form ([`canonical`]). The tests
//! exercise each helper at these limits.
//!
//! The reused `avx2_sqrt::mul` also accepts these operands. For normalized inputs
//! below 2^261 all nine limbs are below 2^29, including limb 8. Each column has at
//! most nine input products and five modulus terms, each below 2^58, plus a carry
//! below 2^35: even the conservative sum is below 2^63. Thus `vpmuludq` truncates no
//! operand, `vpaddq` loses no carry, and the broader chord bounds do not rely on the
//! sqrt caller's 2p bound.
#![allow(unsafe_code)]
#![allow(unused_unsafe)]

use core::arch::x86_64::*;

use super::avx2_sqrt::{BITS, L, MASK, V, mul, pack, pasta_limbs29};
use alloc::vec::Vec;
use ff::Field;

/// Additions in flight: two vectors of four lanes.
const G: usize = 2;
/// Lanes per chunk.
pub(crate) const LANES: usize = 4 * G;

/// `c p` in radix 2^29, with limbs 0 to 7 raised to at least `2^30 - 2` by
/// borrowing from the next limb, so it limb-wise dominates any normalized
/// subtrahend whose limb 8 is at most `c 2^22 - 2`.
///
/// Each limb is computed in one expression, adding the `2^30` it receives
/// before subtracting the `2` it lends: limbs 5 to 7 of both Pasta moduli are
/// zero, so borrowing first would underflow (a compile error in `const`
/// evaluation with overflow checks on, as in debug builds).
const fn kp(p: &[u64; L], c: u64) -> [u64; L] {
    let mut k = [0u64; L];
    let mut i = 0;
    while i < L {
        let received = if i < L - 1 { 1 << (BITS + 1) } else { 0 };
        let lent = if i > 0 { 2 } else { 0 };
        k[i] = c * p[i] + received - lent;
        i += 1;
    }
    k
}

/// Propagates carries so limbs 0 to 7 are below 2^29.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn normalize(mut v: [__m256i; L]) -> V {
    // SAFETY: the caller establishes AVX2 and the documented operand bounds.
    unsafe {
        let mask = _mm256_set1_epi64x(MASK as i64);
        for i in 0..L - 1 {
            let carry = _mm256_srli_epi64::<BITS>(v[i]);
            v[i] = _mm256_and_si256(v[i], mask);
            v[i + 1] = _mm256_add_epi64(v[i + 1], carry);
        }
        V(v)
    }
}

/// `32 a`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn shl5(a: &V) -> V {
    // SAFETY: the caller establishes AVX2 and the documented operand bounds.
    unsafe { normalize(core::array::from_fn(|i| _mm256_slli_epi64::<5>(a.0[i]))) }
}

/// `a + k - b`, for `k` from [`kp`] dominating `b`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sub(a: &V, b: &V, k: &[u64; L]) -> V {
    // SAFETY: the caller establishes AVX2 and the documented operand bounds.
    unsafe {
        normalize(core::array::from_fn(|i| {
            _mm256_sub_epi64(
                _mm256_add_epi64(a.0[i], _mm256_set1_epi64x(k[i] as i64)),
                b.0[i],
            )
        }))
    }
}

/// The four 64-bit lanes of each limb vector.
///
/// A bit-cast rather than `_mm256_storeu_si256`: when AVX2 is enabled only
/// per function (the forced x86-64 backend), the store intrinsic was left as
/// an out-of-line call.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn lanes_of(v: &V) -> [[u64; 4]; L] {
    // SAFETY: the caller establishes AVX2 and the documented operand bounds.
    unsafe {
        // SAFETY: `__m256i` and `[u64; 4]` are both 32 bytes of plain data with
        // no invalid bit patterns.
        v.0.map(|limb| unsafe { core::mem::transmute::<__m256i, [u64; 4]>(limb) })
    }
}

/// Unpacks four values below 2^261 to 320-bit integers.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn unpack_wide(v: &V) -> [[u64; 5]; 4] {
    // SAFETY: the caller establishes AVX2 and the documented operand bounds.
    unsafe {
        let limbs = lanes_of(v);
        let mut out = [[0u64; 5]; 4];
        for (lane, out) in out.iter_mut().enumerate() {
            for (i, limb) in limbs.iter().enumerate() {
                let bit = i * BITS as usize;
                let (w, s) = (bit / 64, bit % 64);
                let x = limb[lane];
                out[w] |= x << s;
                if s > 0 && w + 1 < 5 {
                    out[w + 1] |= x >> (64 - s);
                }
            }
        }
        out
    }
}

/// Reduces `x < 2^260` modulo `p` (for `2^254 < p < 2^254 + 2^128`):
/// `q = floor(x / 2^254)` is `floor(x / p)` or one more.
#[inline]
fn reduce_wide(x: &[u64; 5], p: &[u64; 4]) -> [u64; 4] {
    let q = (x[4] << 2) | (x[3] >> 62);
    // x - q p, with q p = q (p0 + p1 2^64) + q 2^254.
    let mut out = [0u64; 4];
    let mut borrow = 0i128;
    let qp = [
        u128::from(q) * u128::from(p[0]),
        u128::from(q) * u128::from(p[1]),
        u128::from(q) * u128::from(p[2]),
        u128::from(q) * u128::from(p[3]),
    ];
    let mut carry = 0u128;
    let mut qpw = [0u64; 5];
    for i in 0..4 {
        let t = qp[i] + carry;
        qpw[i] = t as u64;
        carry = t >> 64;
    }
    qpw[4] = carry as u64;
    let mut word4 = 0i128;
    for i in 0..5 {
        let t = i128::from(x[i]) - i128::from(qpw[i]) + borrow;
        if i < 4 {
            out[i] = t as u64;
        } else {
            word4 = t;
        }
        borrow = t >> 64;
    }
    if word4 < 0 {
        // One subtraction too many: add `p` back.
        let mut carry = 0u128;
        for i in 0..4 {
            let t = u128::from(out[i]) + u128::from(p[i]) + carry;
            out[i] = t as u64;
            carry = t >> 64;
        }
    }
    out
}

/// A Pasta base field whose batched chord additions run in AVX2.
///
/// Public inputs only; used by the explicit variable-time fixed-base batch sum.
pub(crate) trait Avx2ChordField: crate::arithmetic::VartimeField {
    /// Canonical Montgomery limbs.
    fn chord_limbs(&self) -> [u64; 4];
    /// The element with canonical Montgomery limbs `limbs`.
    fn from_chord_limbs(limbs: [u64; 4]) -> Self;
    /// [`chord_level`] for this field. All input points must be public.
    fn chord_level(
        n: usize,
        pair: impl Fn(usize) -> (usize, usize, usize),
        points: &[Point29],
        next: &mut [Point29],
        scratch: &mut ChordScratch,
    ) -> Option<()>;
    /// The modulus in radix 2^29.
    fn modulus29() -> [u64; L];
}

#[cfg(test)]
std::thread_local! {
    static INVERT_LANES_VARTIME_CALLS: core::cell::Cell<usize> =
        const { core::cell::Cell::new(0) };
}

/// Replaces eight Montgomery values (below 2^256) with `32` times their
/// inverses, sharing one inversion; `None` if any is zero.
///
/// Variable-time; public inputs only. Called only by the field adapter for
/// `PreparedFixedBaseSum::sum_batch_vartime`, never by ordinary `Field::invert`.
fn invert_lanes_vartime<F: Avx2ChordField>(
    lanes: &mut [[u64; 4]; LANES],
    modulus: &[u64; 4],
) -> Option<()> {
    #[cfg(test)]
    INVERT_LANES_VARTIME_CALLS.with(|calls| calls.set(calls.get() + 1));
    let values: [F; LANES] = core::array::from_fn(|i| {
        F::from_chord_limbs(reduce_wide(
            &[lanes[i][0], lanes[i][1], lanes[i][2], lanes[i][3], 0],
            modulus,
        ))
    });
    let mut prefix = [F::ONE; LANES];
    let mut acc = F::ONE;
    for (prefix, value) in prefix.iter_mut().zip(&values) {
        *prefix = acc;
        acc *= value;
    }
    // Only public chord denominators reach this variable-time inversion.
    let mut inverse = acc.invert_vartime()?;
    let scale = F::ONE.double().double().double().double().double();
    for i in (0..LANES).rev() {
        lanes[i] = (inverse * prefix[i] * scale).chord_limbs();
        inverse *= values[i];
    }
    Some(())
}

/// Per-chunk buffers of [`chord_level`] (about 144 bytes per addition),
/// reused across the levels of one reduction.
#[derive(Default)]
pub(crate) struct ChordScratch {
    scaled: Vec<[V; G]>,
    denominators: Vec<[V; G]>,
}

impl core::fmt::Debug for ChordScratch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ChordScratch")
            .field("chunks", &self.scaled.capacity())
            .finish()
    }
}

/// A canonical field element (Montgomery form, below `p`) in radix 2^29.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct Limbs29(pub(crate) [u32; L]);

/// An affine point with [`Limbs29`] coordinates.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct Point29 {
    pub(crate) x: Limbs29,
    pub(crate) y: Limbs29,
}

impl Limbs29 {
    /// Converts canonical Montgomery limbs.
    #[inline]
    pub(crate) fn from_limbs(x: &[u64; 4]) -> Self {
        let mut out = [0u32; L];
        for (i, limb) in out.iter_mut().enumerate() {
            let bit = i * BITS as usize;
            let (w, s) = (bit / 64, bit % 64);
            let mut v = x[w] >> s;
            if s + BITS as usize > 64 && w + 1 < 4 {
                v |= x[w + 1] << (64 - s);
            }
            *limb = (v & MASK) as u32;
        }
        Limbs29(out)
    }

    /// The canonical Montgomery limbs.
    #[inline]
    pub(crate) fn to_limbs(self) -> [u64; 4] {
        let mut out = [0u64; 4];
        for (i, &limb) in self.0.iter().enumerate() {
            let bit = i * BITS as usize;
            let (w, s) = (bit / 64, bit % 64);
            let x = u64::from(limb);
            out[w] |= x << s;
            if s + BITS as usize > 64 && w + 1 < 4 {
                out[w + 1] |= x >> (64 - s);
            }
        }
        out
    }

    /// `p - self` for nonzero `self`, and zero for zero.
    #[inline]
    pub(crate) fn neg(&self, p: &[u64; L]) -> Self {
        if self.0.iter().all(|&l| l == 0) {
            return *self;
        }
        let mut out = [0u32; L];
        let mut borrow = 0i64;
        for i in 0..L {
            let t = p[i] as i64 - i64::from(self.0[i]) + borrow;
            out[i] = (t & MASK as i64) as u32;
            borrow = t >> BITS;
        }
        Limbs29(out)
    }
}

/// `p` in radix 2^29.
pub(crate) const fn modulus29(p: &[u64; 4]) -> [u64; L] {
    pasta_limbs29(p)
}

/// Loads one coordinate of four points.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load4(a: [&Limbs29; 4]) -> V {
    // SAFETY: the caller establishes AVX2 and the documented operand bounds.
    unsafe {
        V(core::array::from_fn(|i| {
            _mm256_set_epi64x(
                i64::from(a[3].0[i]),
                i64::from(a[2].0[i]),
                i64::from(a[1].0[i]),
                i64::from(a[0].0[i]),
            )
        }))
    }
}

/// Stores four canonical values (as [`canonical`] leaves them).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store4(v: &V) -> [Limbs29; 4] {
    // SAFETY: the caller establishes AVX2 and the documented operand bounds.
    unsafe {
        let limbs = lanes_of(v);
        core::array::from_fn(|lane| Limbs29(core::array::from_fn(|i| limbs[i][lane] as u32)))
    }
}

/// `a + b`, for normalized `a`, `b`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn add(a: &V, b: &V) -> V {
    // SAFETY: the caller establishes AVX2 and the documented operand bounds.
    unsafe { normalize(core::array::from_fn(|i| _mm256_add_epi64(a.0[i], b.0[i]))) }
}

/// Reduces a normalized value below `8p` (limb 8 at most `2^25`: `8p` exceeds
/// `2^257`) to `[0, p)`.
///
/// `q = floor(v / 2^254)` is `limb 8 >> 22`; `v - q p` lies in `[-p, p)`
/// because `2^254 < p < 2^254 + 2^128`, so at most one `p` is added back.
/// Limbs 0 to 7 get `2^32` added (borrowed as 8 from the next limb), so they
/// stay non-negative for the logical carry shifts; only limb 8 carries the
/// sign.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn canonical(v: &V, p: &[u64; L]) -> V {
    // SAFETY: the caller establishes AVX2 and the documented operand bounds.
    unsafe {
        let q = _mm256_srli_epi64::<22>(v.0[8]);
        let mut t: [__m256i; L] = core::array::from_fn(|i| {
            let qp = match i {
                0 => q,
                8 => _mm256_slli_epi64::<22>(q),
                _ => _mm256_mul_epu32(q, _mm256_set1_epi64x(p[i] as i64)),
            };
            let offset: i64 = match i {
                0 => 1 << 32,
                8 => -8,
                _ => (1 << 32) - 8,
            };
            _mm256_sub_epi64(_mm256_add_epi64(v.0[i], _mm256_set1_epi64x(offset)), qp)
        });
        t = normalize(t).0;
        let negative = _mm256_cmpgt_epi64(_mm256_setzero_si256(), t[8]);
        for (i, t) in t.iter_mut().enumerate() {
            *t = _mm256_add_epi64(
                *t,
                _mm256_and_si256(negative, _mm256_set1_epi64x(p[i] as i64)),
            );
        }
        normalize(t)
    }
}

/// One level of chord additions over radix-2^29 points: for each `i < n`,
/// `pair(i) = (left, right, output)` and `next[output] = points[left] +
/// points[right]`. The x-coordinates of each pair must differ; otherwise
/// returns `None` without writing `next`.
///
/// Variable-time; public inputs only, including the lane-inversion callback.
///
/// Bounds (scales as in the module docs): differences below `3p`, their
/// 32-scaled forms below `96p`; lane products, scaled numerators and running
/// inverses stay below `4.01p` (a bound `Bp` becomes at most
/// `96 B p^2 / 2^261 + p`, whose fixed point exceeds `4p` by less than
/// `2^-120 p`); slope below `1.126p`; `x3` below `4.32p`, `x - x3` below `6p`,
/// `y3` below `4.69p`; outputs are reduced to canonical.
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn chord_level<const P0: u64, const P1: u64, const P2: u64, const P3: u64>(
    n: usize,
    pair: impl Fn(usize) -> (usize, usize, usize),
    points: &[Point29],
    next: &mut [Point29],
    scratch: &mut ChordScratch,
    one: &Limbs29,
    thirty_two: &Limbs29,
    invert: impl FnOnce(&mut [[u64; 4]; LANES]) -> Option<()>,
) -> Option<()> {
    // SAFETY: the caller establishes AVX2 and the documented operand bounds.
    unsafe {
        let p = const { pasta_limbs29(&[P0, P1, P2, P3]) };
        let k2 = const { kp(&pasta_limbs29(&[P0, P1, P2, P3]), 2) };
        let k3 = const { kp(&pasta_limbs29(&[P0, P1, P2, P3]), 3) };
        let k5 = const { kp(&pasta_limbs29(&[P0, P1, P2, P3]), 5) };
        macro_rules! mul {
            ($a:expr, $b:expr) => {
                mul::<P0, P1, P2, P3>($a, $b)
            };
        }
        let chunks = n.div_ceil(LANES);
        let pad = Point29 {
            x: *one,
            y: Limbs29::default(),
        };
        let zero = Point29::default();
        // Lane `l` of vector `g` in chunk `c` is addition `c * LANES + 4 g + l`;
        // padding pairs `(one, 0) + (0, 0)` have difference `-1`, a unit.
        let operands = |i: usize| -> (&Point29, &Point29) {
            if i < n {
                let (left, right, _) = pair(i);
                (&points[left], &points[right])
            } else {
                (&pad, &zero)
            }
        };

        let mut lane_products = [load4([thirty_two; 4]); G];
        let ChordScratch {
            scaled,
            denominators,
        } = scratch;
        scaled.clear();
        denominators.clear();
        scaled.reserve(chunks);
        denominators.reserve(chunks);
        for chunk in 0..chunks {
            let base = chunk * LANES;
            let mut s = lane_products;
            let mut d = lane_products;
            for g in 0..G {
                let first = base + 4 * g;
                let ops = [
                    operands(first),
                    operands(first + 1),
                    operands(first + 2),
                    operands(first + 3),
                ];
                let lx = load4([&ops[0].0.x, &ops[1].0.x, &ops[2].0.x, &ops[3].0.x]);
                let ly = load4([&ops[0].0.y, &ops[1].0.y, &ops[2].0.y, &ops[3].0.y]);
                let rx = load4([&ops[0].1.x, &ops[1].1.x, &ops[2].1.x, &ops[3].1.x]);
                let ry = load4([&ops[0].1.y, &ops[1].1.y, &ops[2].1.y, &ops[3].1.y]);
                let num32 = shl5(&sub(&ry, &ly, &k2));
                d[g] = shl5(&sub(&rx, &lx, &k2));
                s[g] = mul!(&num32, &lane_products[g]);
                lane_products[g] = mul!(&lane_products[g], &d[g]);
            }
            scaled.push(s);
            denominators.push(d);
        }

        let mut inverses = [[0u64; 4]; LANES];
        let modulus = [P0, P1, P2, P3];
        for g in 0..G {
            let u = unpack_wide(&lane_products[g]);
            for l in 0..4 {
                inverses[4 * g + l] = reduce_wide(&u[l], &modulus);
            }
        }
        invert(&mut inverses)?;
        let mut j: [V; G] =
            core::array::from_fn(|g| pack(&core::array::from_fn(|l| inverses[4 * g + l])));

        for chunk in (0..chunks).rev() {
            let base = chunk * LANES;
            for (g, jg) in j.iter_mut().enumerate() {
                let slope = mul!(&scaled[chunk][g], &*jg);
                *jg = mul!(&*jg, &denominators[chunk][g]);
                let slope32 = shl5(&slope);
                let first = base + 4 * g;
                let ops = [
                    operands(first),
                    operands(first + 1),
                    operands(first + 2),
                    operands(first + 3),
                ];
                let lx = load4([&ops[0].0.x, &ops[1].0.x, &ops[2].0.x, &ops[3].0.x]);
                let ly = load4([&ops[0].0.y, &ops[1].0.y, &ops[2].0.y, &ops[3].0.y]);
                let rx = load4([&ops[0].1.x, &ops[1].1.x, &ops[2].1.x, &ops[3].1.x]);
                let x_sum = add(&lx, &rx);
                let x3 = sub(&mul!(&slope, &slope32), &x_sum, &k3);
                let y3 = sub(&mul!(&slope32, &sub(&lx, &x3, &k5)), &ly, &k2);
                let xs = store4(&canonical(&x3, &p));
                let ys = store4(&canonical(&y3, &p));
                for l in 0..4 {
                    let i = base + 4 * g + l;
                    if i < n {
                        next[pair(i).2] = Point29 { x: xs[l], y: ys[l] };
                    }
                }
            }
        }
        Some(())
    }
}

// The scalar fields keep canonical Montgomery limbs. This adapter is private
// to the public-input fixed-base reducer; ordinary field operations do not change.
macro_rules! impl_chord_field {
    ($field:ident, $p0:expr, $p1:expr) => {
        impl Avx2ChordField for crate::$field {
            fn chord_limbs(&self) -> [u64; 4] {
                self.0
            }

            fn from_chord_limbs(limbs: [u64; 4]) -> Self {
                Self(limbs)
            }

            fn chord_level(
                n: usize,
                pair: impl Fn(usize) -> (usize, usize, usize),
                points: &[Point29],
                next: &mut [Point29],
                scratch: &mut ChordScratch,
            ) -> Option<()> {
                if !super::avx2_sqrt::available() {
                    return None;
                }
                // SAFETY: runtime detection checks AVX2 and enabled XMM/YMM
                // state, even on no_std and builds without AVX2 target features.
                unsafe {
                    chord_level::<$p0, $p1, 0, { 1 << 62 }>(
                        n,
                        pair,
                        points,
                        next,
                        scratch,
                        &Limbs29::from_limbs(&Self::ONE.0),
                        &Limbs29::from_limbs(&Self::from(32).0),
                        |lanes| invert_lanes_vartime::<Self>(lanes, &[$p0, $p1, 0, 1 << 62]),
                    )
                }
            }

            fn modulus29() -> [u64; L] {
                modulus29(&[$p0, $p1, 0, 1 << 62])
            }
        }
    };
}
impl_chord_field!(Fp, 0x992d30ed00000001, 0x224698fc094cf91b);
impl_chord_field!(Fq, 0x8c46eb2100000001, 0x224698fc0994a8dd);

#[cfg(test)]
mod tests;
