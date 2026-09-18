//! Variable-time modular inversion for the Pasta base fields, using 62-bit
//! divsteps (safegcd).
//!
//! Portions adapted from libsecp256k1's `src/modinv64_impl.h`:
//! Copyright (c) 2020 Peter Dettman. Distributed under the MIT software
//! license; see this crate's `LICENSE-MIT` file.
//!
//! This is a near-verbatim port of libsecp256k1's signed-62 variable-time
//! modular inversion, specialized to the Pasta base fields:
//!
//! - The coefficient `e` is seeded with $R^2 \bmod m$ (rather than $1$), so a
//!   Montgomery-form input $X = aR$ yields the Montgomery-form inverse
//!   $a^{-1}R$ directly: no conversion into or out of Montgomery form, and no
//!   Montgomery reduction anywhere.
//! - Both moduli are $[m_0, m_1, 2, 0, 64]$ in signed radix $2^{62}$, so the
//!   `k*m` corrections in the coefficient updates need only two general
//!   multiplications (limbs 0 and 1); limb 2 is `k << 1`, limb 3 vanishes,
//!   and limb 4 is `k << 6`. The first `f,g` update exploits the same shape
//!   (the initial `f` *is* the modulus).
//! - `f,g` are updated before `d,e` each batch, so the terminal batch computes
//!   only the one coefficient row that becomes the inverse, and the first
//!   batch (where `d = 0`) skips every product against `d`.
//!
//! The scaled invariants maintained across batches are
//! $X d \equiv R^2 f \pmod m$ and $X e \equiv R^2 g \pmod m$; at termination
//! $g = 0$ and $f = \sigma \in \{-1, 1\}$, so $\sigma d \equiv R^2 X^{-1}
//! \equiv a^{-1} R \pmod m$.
//!
//! Every inner-loop invariant inherited from upstream (`VERIFY_CHECK`) is kept
//! live in unit tests — including `--release` test runs — via the `verify!`
//! macro below, and compiled out of production builds.
//!
//! # Warning
//!
//! This inversion is **variable-time in the value being inverted**: its
//! trailing-zero counts, divstep branches, batch count, and active limb length
//! all depend on the input. It must only reach values whose timing is
//! acceptable to leak. See the crate changelog entry for the posture of this
//! crate's inversion call sites.
//!
//! # References
//!
//! - Daniel J. Bernstein, Bo-Yin Yang: "Fast constant-time gcd computation and
//!   modular inversion" <https://eprint.iacr.org/2019/266>
//! - libsecp256k1, `src/modinv64_impl.h` (the ported implementation) and
//!   `doc/safegcd_implementation.md`
//!   <https://github.com/bitcoin-core/secp256k1>
//! - Thomas Pornin: "Optimized Binary GCD for Modular Inversion"
//!   <https://eprint.iacr.org/2020/972> (the constant-time alternative)

/// Mask of the low 62 bits of a word.
const MASK62: u64 = (1u64 << 62) - 1;

/// Per-field constants for the divstep inversion, in signed radix $2^{62}$.
///
/// The kernels hardcode the shared sparse shape of both Pasta moduli
/// (`MODULUS[2] == 2`, `MODULUS[3] == 0`, `MODULUS[4] == 64`); the const
/// assertions below pin that assumption to the constants.
pub(crate) trait InvParams {
    /// The field modulus $m$, radix $2^{62}$, least-significant limb first.
    const MODULUS: [i64; 5];
    /// $m^{-1} \bmod 2^{62}$ (note: *not* the Montgomery `INV`, which is
    /// $-m^{-1} \bmod 2^{64}$).
    const MU: u64;
    /// $R^2 = 2^{512} \bmod m$, radix $2^{62}$: the initial value of `e`.
    const R2: [i64; 5];
}

/// [`InvParams`] for the Pallas base field (`Fp`).
pub(crate) struct FpParams;

impl InvParams for FpParams {
    const MODULUS: [i64; 5] = [0x192d_30ed_0000_0001, 0x091a_63f0_2533_e46e, 2, 0, 0x40];
    const MU: u64 = 0x26d2_cf13_0000_0001;
    const R2: [i64; 5] = [
        0x0c78_ecb3_0000_000f,
        0x1f4c_36f6_2c37_839e,
        0x397a_99bc_3c95_d18d,
        0x1b50_6bde_e72d_c51d,
        9,
    ];
}

/// [`InvParams`] for the Vesta base field (`Fq`).
pub(crate) struct FqParams;

impl InvParams for FqParams {
    const MODULUS: [i64; 5] = [0x0c46_eb21_0000_0001, 0x091a_63f0_2652_a376, 2, 0, 0x40];
    const MU: u64 = 0x33b9_14df_0000_0001;
    const R2: [i64; 5] = [
        0x3c96_78ff_0000_000f,
        0x1eed_0cf6_2468_5b8f,
        0x3ae2_3100_4ccf_5906,
        0x1b50_6bdf_33f6_aa5f,
        9,
    ];
}

/// The kernels bake in the sparse modulus shape; pin it (and the low-limb
/// inverse relation) at compile time for both parameter sets.
const _: () = {
    assert!(FpParams::MODULUS[2] == 2 && FpParams::MODULUS[3] == 0 && FpParams::MODULUS[4] == 64);
    assert!(FqParams::MODULUS[2] == 2 && FqParams::MODULUS[3] == 0 && FqParams::MODULUS[4] == 64);
    assert!(FpParams::MU.wrapping_mul(FpParams::MODULUS[0] as u64) & MASK62 == 1);
    assert!(FqParams::MU.wrapping_mul(FqParams::MODULUS[0] as u64) & MASK62 == 1);
};

/// A signed multi-word integer in radix $2^{62}$, least-significant limb
/// first. In canonical form all limbs below the active length are in
/// $[0, 2^{62})$ and the top active limb carries the sign, so the sign of the
/// value is the sign of that limb.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Signed62([i64; 5]);

/// The transition matrix for one batch of 62 divsteps. Row sums are bounded
/// by $|u| + |v| \le 2^{62}$ and $|q| + |r| \le 2^{62}$, and the determinant
/// is exactly $2^{62}$.
#[derive(Clone, Copy, Debug)]
struct Trans2x2 {
    u: i64,
    v: i64,
    q: i64,
    r: i64,
}

/// Computes the wider cancellation multiplier used after a negative-eta
/// divstep swap on AArch64.
///
/// Keeping this out of line prevents the AArch64 backend from speculatively
/// evaluating both cancellation formulas before selecting one.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn negative_eta_multiplier(f: u64, g: u64, mask: u64) -> u64 {
    f.wrapping_mul(g)
        .wrapping_mul(f.wrapping_mul(f).wrapping_sub(2))
        & mask
}

/// An upstream `VERIFY_CHECK`: live in unit tests (including `--release`
/// runs, where `debug_assert!` would be inert) and in debug builds; compiled
/// out of production builds. Keep every check ported from upstream on this
/// macro so the user's `cargo test --release` exercises them.
macro_rules! verify {
    ($cond:expr_2021 $(, $($arg:tt)+)?) => {
        if cfg!(any(test, debug_assertions)) {
            assert!($cond $(, $($arg)+)?);
        }
    };
}

/// Value-level bound checks, too expensive for unconditional inlining: a port
/// of upstream's `secp256k1_modinv64_mul_cmp_62` machinery, active only where
/// `verify!` is.
#[cfg(any(test, debug_assertions))]
mod checks {
    use super::{MASK62, Signed62};
    use core::cmp::Ordering;

    /// Computes `a * factor` (using `alen` active limbs of `a`) as a 5-limb
    /// signed-62 value with all but the top limb normalized.
    #[allow(clippy::needless_range_loop)]
    pub(super) fn mul_62(a: &Signed62, alen: usize, factor: i64) -> Signed62 {
        let mut c: i128 = 0;
        let mut r = [0i64; 5];
        for i in 0..4 {
            if i < alen {
                c += i128::from(a.0[i]) * i128::from(factor);
            }
            r[i] = (c as u64 & MASK62) as i64;
            c >>= 62;
        }
        if 4 < alen {
            c += i128::from(a.0[4]) * i128::from(factor);
        }
        assert!(c == i128::from(c as i64), "mul_62 top limb must fit i64");
        r[4] = c as i64;
        Signed62(r)
    }

    /// Compares `a` (with `alen` active limbs) against `b * factor`.
    pub(super) fn mul_cmp_62(a: &Signed62, alen: usize, b: &Signed62, factor: i64) -> Ordering {
        let am = mul_62(a, alen, 1); // normalizes all but the top limb of a
        let bm = mul_62(b, 5, factor);
        for i in 0..4 {
            assert!(am.0[i] >> 62 == 0, "mul_cmp operand not normalized");
            assert!(bm.0[i] >> 62 == 0, "mul_cmp operand not normalized");
        }
        for i in (0..5).rev() {
            if am.0[i] < bm.0[i] {
                return Ordering::Less;
            }
            if am.0[i] > bm.0[i] {
                return Ordering::Greater;
            }
        }
        Ordering::Equal
    }

    /// Asserts $-2m < x < m$ (the coefficient range invariant).
    pub(super) fn assert_coeff_range<P: super::InvParams>(x: &Signed62, what: &str) {
        let m = Signed62(P::MODULUS);
        assert!(
            mul_cmp_62(x, 5, &m, -2) == Ordering::Greater,
            "{} must exceed -2m",
            what
        );
        assert!(
            mul_cmp_62(x, 5, &m, 1) == Ordering::Less,
            "{} must be below m",
            what
        );
    }

    /// Asserts $-m < x \le m$ / $-m < x < m$ (the `f`/`g` range invariants).
    pub(super) fn assert_fg_range<P: super::InvParams>(f: &Signed62, g: &Signed62, len: usize) {
        let m = Signed62(P::MODULUS);
        assert!(
            mul_cmp_62(f, len, &m, -1) == Ordering::Greater,
            "f must exceed -m"
        );
        assert!(
            mul_cmp_62(f, len, &m, 1) != Ordering::Greater,
            "f must not exceed m"
        );
        assert!(
            mul_cmp_62(g, len, &m, -1) == Ordering::Greater,
            "g must exceed -m"
        );
        assert!(
            mul_cmp_62(g, len, &m, 1) == Ordering::Less,
            "g must be below m"
        );
    }
}

/// Repacks a canonical 4x64-bit little-endian field representation (a value
/// in $[0, m)$) into signed radix $2^{62}$.
fn pack62(x: &[u64; 4]) -> Signed62 {
    let out = Signed62([
        (x[0] & MASK62) as i64,
        (((x[0] >> 62) | (x[1] << 2)) & MASK62) as i64,
        (((x[1] >> 60) | (x[2] << 4)) & MASK62) as i64,
        (((x[2] >> 58) | (x[3] << 6)) & MASK62) as i64,
        (x[3] >> 56) as i64,
    ]);
    verify!(
        out.0[4] >= 0 && out.0[4] <= 0x7f,
        "packed top limb out of range"
    );
    out
}

/// Repacks a canonical signed-62 value in $[0, m)$ (a [`normalize_62`]
/// output) back into the 4x64-bit little-endian field representation.
fn unpack62(v: &Signed62) -> [u64; 4] {
    let v0 = v.0[0] as u64;
    let v1 = v.0[1] as u64;
    let v2 = v.0[2] as u64;
    let v3 = v.0[3] as u64;
    let v4 = v.0[4] as u64;
    verify!(
        v0 <= MASK62 && v1 <= MASK62 && v2 <= MASK62 && v3 <= MASK62 && v4 <= 0x7f,
        "unpack input must be canonical and reduced"
    );
    [
        v0 | (v1 << 62),
        (v1 >> 2) | (v2 << 60),
        (v2 >> 4) | (v3 << 58),
        (v3 >> 6) | (v4 << 56),
    ]
}

/// Computes the transition matrix for 62 variable-time divsteps, reading only
/// the bottom limbs of `f` and `g` (which must be odd / the current `eta`).
/// Returns the matrix and the updated `eta`.
///
/// This is upstream's `secp256k1_modinv64_divsteps_62_var`: trailing-zero
/// counts batch the halvings of `g`, and small modular-inverse formulas cancel
/// up to 4 bits (`eta >= 0`) or 6 bits (`eta < 0`, after the swap) of `g` per
/// inner iteration.
#[allow(clippy::many_single_char_names)]
fn divsteps_62_var(mut eta: i64, f0: u64, g0: u64) -> (Trans2x2, i64) {
    let mut u: u64 = 1;
    let mut v: u64 = 0;
    let mut q: u64 = 0;
    let mut r: u64 = 1;
    let mut f = f0;
    let mut g = g0;
    let mut i: u32 = 62;

    loop {
        // Use a sentinel bit to count zeros only up to i.
        let zeros = (g | (u64::MAX << i)).trailing_zeros();
        // Perform zeros divsteps at once; they all just divide g by two.
        g >>= zeros;
        u <<= zeros;
        v <<= zeros;
        eta -= i64::from(zeros);
        i -= zeros;
        // We're done once we've performed 62 divsteps.
        if i == 0 {
            break;
        }
        verify!(f & 1 == 1, "f must be odd");
        verify!(g & 1 == 1, "g must be odd after shifting out its zeros");
        verify!(
            u.wrapping_mul(f0).wrapping_add(v.wrapping_mul(g0)) == f << (62 - i),
            "top row of T must reproduce f"
        );
        verify!(
            q.wrapping_mul(f0).wrapping_add(r.wrapping_mul(g0)) == g << (62 - i),
            "bottom row of T must reproduce g"
        );
        // eta starts at -1 and moves by at most one per divstep, and a full
        // inversion performs at most 744 divsteps.
        verify!((-745..=745).contains(&eta), "eta out of range");
        let limit;
        let m;
        let w;
        if eta < 0 {
            // If eta is negative, negate it and replace f,g with g,-f.
            eta = -eta;
            let tmp = f;
            f = g;
            g = tmp.wrapping_neg();
            let tmp = u;
            u = q;
            q = tmp.wrapping_neg();
            let tmp = v;
            v = r;
            r = tmp.wrapping_neg();
            // Use a formula to cancel out up to 6 bits of g, as f is odd:
            // f*(f*f - 2) is the inverse of -f modulo 64.
            limit = if eta as u32 + 1 > i {
                i
            } else {
                eta as u32 + 1
            };
            verify!((1..=62).contains(&limit), "limit out of range");
            m = (u64::MAX >> (64 - limit)) & 63;
            #[cfg(target_arch = "aarch64")]
            {
                w = negative_eta_multiplier(f, g, m);
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                w = f
                    .wrapping_mul(g)
                    .wrapping_mul(f.wrapping_mul(f).wrapping_sub(2))
                    & m;
            }
        } else {
            // A simpler formula that cancels up to 4 bits of g:
            // f + (((f + 1) & 4) << 1) is the inverse of f modulo 16.
            limit = if eta as u32 + 1 > i {
                i
            } else {
                eta as u32 + 1
            };
            verify!((1..=62).contains(&limit), "limit out of range");
            m = (u64::MAX >> (64 - limit)) & 15;
            let w0 = f.wrapping_add((f.wrapping_add(1) & 4) << 1);
            w = w0.wrapping_neg().wrapping_mul(g) & m;
        }
        g = g.wrapping_add(f.wrapping_mul(w));
        q = q.wrapping_add(u.wrapping_mul(w));
        r = r.wrapping_add(v.wrapping_mul(w));
        verify!(g & m == 0, "the masked low bits of g must cancel");
    }
    let t = Trans2x2 {
        u: u as i64,
        v: v as i64,
        q: q as i64,
        r: r as i64,
    };
    // The determinant of T must be a power of two: this guarantees the
    // matrix-vector products below preserve the relative sizes of f and g.
    verify!(
        i128::from(t.u) * i128::from(t.r) - i128::from(t.v) * i128::from(t.q) == 1 << 62,
        "determinant of T must be 2^62"
    );
    #[cfg(any(test, debug_assertions))]
    {
        assert!(
            t.u.unsigned_abs() + t.v.unsigned_abs() <= 1 << 62,
            "|u| + |v| must not exceed 2^62"
        );
        assert!(
            t.q.unsigned_abs() + t.r.unsigned_abs() <= 1 << 62,
            "|q| + |r| must not exceed 2^62"
        );
    }
    (t, eta)
}

/// Applies the transition matrix to the full-width `f` and `g` (over `len`
/// active limbs), dividing exactly by $2^{62}$. Upstream's
/// `secp256k1_modinv64_update_fg_62_var`.
#[allow(clippy::many_single_char_names)]
fn update_fg_62_var(len: usize, f: &mut Signed62, g: &mut Signed62, t: &Trans2x2) {
    let u = t.u;
    let v = t.v;
    let q = t.q;
    let r = t.r;
    verify!((1..=5).contains(&len), "active length out of range");
    let mut fi = f.0[0];
    let mut gi = g.0[0];
    let mut cf = i128::from(u) * i128::from(fi) + i128::from(v) * i128::from(gi);
    let mut cg = i128::from(q) * i128::from(fi) + i128::from(r) * i128::from(gi);
    // The bottom 62 bits of t*[f,g] are zero by construction of t.
    verify!(cf as u64 & MASK62 == 0, "low bits of new f must vanish");
    verify!(cg as u64 & MASK62 == 0, "low bits of new g must vanish");
    cf >>= 62;
    cg >>= 62;
    for j in 1..len {
        fi = f.0[j];
        gi = g.0[j];
        cf += i128::from(u) * i128::from(fi) + i128::from(v) * i128::from(gi);
        cg += i128::from(q) * i128::from(fi) + i128::from(r) * i128::from(gi);
        f.0[j - 1] = (cf as u64 & MASK62) as i64;
        cf >>= 62;
        g.0[j - 1] = (cg as u64 & MASK62) as i64;
        cg >>= 62;
    }
    verify!(cf == i128::from(cf as i64), "f tail must fit one limb");
    verify!(cg == i128::from(cg as i64), "g tail must fit one limb");
    f.0[len - 1] = cf as i64;
    g.0[len - 1] = cg as i64;
}

/// [`update_fg_62_var`] specialized to the first batch, where `f` is still
/// the modulus `[m0, m1, 2, 0, 64]`: the three sparse `f` limbs become shifts
/// of the matrix entries. Emits exactly the limbs the generic update would.
#[allow(clippy::many_single_char_names)]
fn update_fg_62_first<P: InvParams>(f: &mut Signed62, g: &mut Signed62, t: &Trans2x2) {
    let u = t.u;
    let v = t.v;
    let q = t.q;
    let r = t.r;
    verify!(f.0 == P::MODULUS, "first f,g update requires f = m");
    let m0 = P::MODULUS[0];
    let m1 = P::MODULUS[1];
    // Limb 0.
    let mut cf = i128::from(u) * i128::from(m0) + i128::from(v) * i128::from(g.0[0]);
    let mut cg = i128::from(q) * i128::from(m0) + i128::from(r) * i128::from(g.0[0]);
    verify!(cf as u64 & MASK62 == 0, "low bits of new f must vanish");
    verify!(cg as u64 & MASK62 == 0, "low bits of new g must vanish");
    cf >>= 62;
    cg >>= 62;
    // Limb 1.
    cf += i128::from(u) * i128::from(m1) + i128::from(v) * i128::from(g.0[1]);
    cg += i128::from(q) * i128::from(m1) + i128::from(r) * i128::from(g.0[1]);
    f.0[0] = (cf as u64 & MASK62) as i64;
    cf >>= 62;
    g.0[0] = (cg as u64 & MASK62) as i64;
    cg >>= 62;
    // Limb 2: m2 = 2.
    cf += (i128::from(u) << 1) + i128::from(v) * i128::from(g.0[2]);
    cg += (i128::from(q) << 1) + i128::from(r) * i128::from(g.0[2]);
    f.0[1] = (cf as u64 & MASK62) as i64;
    cf >>= 62;
    g.0[1] = (cg as u64 & MASK62) as i64;
    cg >>= 62;
    // Limb 3: m3 = 0.
    cf += i128::from(v) * i128::from(g.0[3]);
    cg += i128::from(r) * i128::from(g.0[3]);
    f.0[2] = (cf as u64 & MASK62) as i64;
    cf >>= 62;
    g.0[2] = (cg as u64 & MASK62) as i64;
    cg >>= 62;
    // Limb 4: m4 = 64.
    cf += (i128::from(u) << 6) + i128::from(v) * i128::from(g.0[4]);
    cg += (i128::from(q) << 6) + i128::from(r) * i128::from(g.0[4]);
    f.0[3] = (cf as u64 & MASK62) as i64;
    cf >>= 62;
    g.0[3] = (cg as u64 & MASK62) as i64;
    cg >>= 62;
    verify!(cf == i128::from(cf as i64), "f tail must fit one limb");
    verify!(cg == i128::from(cg as i64), "g tail must fit one limb");
    f.0[4] = cf as i64;
    g.0[4] = cg as i64;
}

/// Applies the transition matrix to the coefficients `d`, `e` modulo $m$,
/// dividing exactly by $2^{62}$. Upstream's
/// `secp256k1_modinv64_update_de_62`, with the sparse modulus limbs 2 and 4
/// applied as shifts and limb 3 skipped.
///
/// Maintains $-2m < d, e < m$: with the sign corrections `(u & sd) + (v & se)`
/// and the $\mu$-derived cancellation term, the numerator of each row stays
/// within $(-2^{63} m, 2^{62} m)$, so the exact shift by 62 lands back in
/// $(-2m, m)$. The bound depends only on the ranges of `d` and `e`, never on
/// their values.
#[allow(clippy::many_single_char_names)]
fn update_de_62<P: InvParams>(d: &mut Signed62, e: &mut Signed62, t: &Trans2x2) {
    let u = t.u;
    let v = t.v;
    let q = t.q;
    let r = t.r;
    let d0 = d.0[0];
    let d1 = d.0[1];
    let d2 = d.0[2];
    let d3 = d.0[3];
    let d4 = d.0[4];
    let e0 = e.0[0];
    let e1 = e.0[1];
    let e2 = e.0[2];
    let e3 = e.0[3];
    let e4 = e.0[4];
    #[cfg(any(test, debug_assertions))]
    {
        checks::assert_coeff_range::<P>(d, "d");
        checks::assert_coeff_range::<P>(e, "e");
    }
    // [md, me] start as zero; plus [u, q] if d is negative; plus [v, r] if e
    // is negative.
    let sd = d4 >> 63;
    let se = e4 >> 63;
    let mut md = (u & sd) + (v & se);
    let mut me = (q & sd) + (r & se);
    // Begin computing t*[d, e].
    let mut cd = i128::from(u) * i128::from(d0) + i128::from(v) * i128::from(e0);
    let mut ce = i128::from(q) * i128::from(d0) + i128::from(r) * i128::from(e0);
    // Correct md, me so that t*[d, e] + m*[md, me] has 62 zero bottom bits.
    md = md.wrapping_sub((P::MU.wrapping_mul(cd as u64).wrapping_add(md as u64) & MASK62) as i64);
    me = me.wrapping_sub((P::MU.wrapping_mul(ce as u64).wrapping_add(me as u64) & MASK62) as i64);
    // Limb 0 of the modulus contribution (general multiplication).
    cd += i128::from(P::MODULUS[0]) * i128::from(md);
    ce += i128::from(P::MODULUS[0]) * i128::from(me);
    verify!(cd as u64 & MASK62 == 0, "low bits of new d must vanish");
    verify!(ce as u64 & MASK62 == 0, "low bits of new e must vanish");
    cd >>= 62;
    ce >>= 62;
    // Limb 1 (general multiplication by m1).
    cd += i128::from(u) * i128::from(d1) + i128::from(v) * i128::from(e1);
    ce += i128::from(q) * i128::from(d1) + i128::from(r) * i128::from(e1);
    cd += i128::from(P::MODULUS[1]) * i128::from(md);
    ce += i128::from(P::MODULUS[1]) * i128::from(me);
    d.0[0] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    e.0[0] = (ce as u64 & MASK62) as i64;
    ce >>= 62;
    // Limb 2: m2 = 2, so the modulus contribution is a shift.
    cd += i128::from(u) * i128::from(d2) + i128::from(v) * i128::from(e2);
    ce += i128::from(q) * i128::from(d2) + i128::from(r) * i128::from(e2);
    cd += i128::from(md) << 1;
    ce += i128::from(me) << 1;
    d.0[1] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    e.0[1] = (ce as u64 & MASK62) as i64;
    ce >>= 62;
    // Limb 3: m3 = 0, no modulus contribution.
    cd += i128::from(u) * i128::from(d3) + i128::from(v) * i128::from(e3);
    ce += i128::from(q) * i128::from(d3) + i128::from(r) * i128::from(e3);
    d.0[2] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    e.0[2] = (ce as u64 & MASK62) as i64;
    ce >>= 62;
    // Limb 4: m4 = 64, so the modulus contribution is a shift.
    cd += i128::from(u) * i128::from(d4) + i128::from(v) * i128::from(e4);
    ce += i128::from(q) * i128::from(d4) + i128::from(r) * i128::from(e4);
    cd += i128::from(md) << 6;
    ce += i128::from(me) << 6;
    d.0[3] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    e.0[3] = (ce as u64 & MASK62) as i64;
    ce >>= 62;
    verify!(cd == i128::from(cd as i64), "d tail must fit one limb");
    verify!(ce == i128::from(ce as i64), "e tail must fit one limb");
    d.0[4] = cd as i64;
    e.0[4] = ce as i64;
    #[cfg(any(test, debug_assertions))]
    {
        checks::assert_coeff_range::<P>(d, "new d");
        checks::assert_coeff_range::<P>(e, "new e");
    }
}

/// [`update_de_62`] specialized to the first batch, where `d = 0` (so every
/// product against `d` vanishes and both sign masks are zero, since the
/// initial `e = R^2` satisfies $0 \le e < m$). Returns the new `(d, e)`.
#[allow(clippy::many_single_char_names)]
fn update_de_62_first<P: InvParams>(e: &Signed62, t: &Trans2x2) -> (Signed62, Signed62) {
    // d = 0, so the u and q columns of the matrix contribute nothing.
    let v = t.v;
    let r = t.r;
    let e0 = e.0[0];
    let e1 = e.0[1];
    let e2 = e.0[2];
    let e3 = e.0[3];
    let e4 = e.0[4];
    #[cfg(any(test, debug_assertions))]
    {
        use core::cmp::Ordering;
        assert!(
            checks::mul_cmp_62(e, 5, &Signed62([0; 5]), 0) != Ordering::Less,
            "first-batch e must be non-negative"
        );
        checks::assert_coeff_range::<P>(e, "e");
    }
    // d = 0 and e >= 0, so both sign corrections vanish.
    let mut cd = i128::from(v) * i128::from(e0);
    let mut ce = i128::from(r) * i128::from(e0);
    let mut md = (P::MU.wrapping_mul(cd as u64) & MASK62) as i64;
    md = md.wrapping_neg();
    let mut me = (P::MU.wrapping_mul(ce as u64) & MASK62) as i64;
    me = me.wrapping_neg();
    cd += i128::from(P::MODULUS[0]) * i128::from(md);
    ce += i128::from(P::MODULUS[0]) * i128::from(me);
    verify!(cd as u64 & MASK62 == 0, "low bits of new d must vanish");
    verify!(ce as u64 & MASK62 == 0, "low bits of new e must vanish");
    cd >>= 62;
    ce >>= 62;
    let mut d_out = [0i64; 5];
    let mut e_out = [0i64; 5];
    // Limb 1.
    cd += i128::from(v) * i128::from(e1) + i128::from(P::MODULUS[1]) * i128::from(md);
    ce += i128::from(r) * i128::from(e1) + i128::from(P::MODULUS[1]) * i128::from(me);
    d_out[0] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    e_out[0] = (ce as u64 & MASK62) as i64;
    ce >>= 62;
    // Limb 2: m2 = 2.
    cd += i128::from(v) * i128::from(e2) + (i128::from(md) << 1);
    ce += i128::from(r) * i128::from(e2) + (i128::from(me) << 1);
    d_out[1] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    e_out[1] = (ce as u64 & MASK62) as i64;
    ce >>= 62;
    // Limb 3: m3 = 0.
    cd += i128::from(v) * i128::from(e3);
    ce += i128::from(r) * i128::from(e3);
    d_out[2] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    e_out[2] = (ce as u64 & MASK62) as i64;
    ce >>= 62;
    // Limb 4: m4 = 64.
    cd += i128::from(v) * i128::from(e4) + (i128::from(md) << 6);
    ce += i128::from(r) * i128::from(e4) + (i128::from(me) << 6);
    d_out[3] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    e_out[3] = (ce as u64 & MASK62) as i64;
    ce >>= 62;
    verify!(cd == i128::from(cd as i64), "d tail must fit one limb");
    verify!(ce == i128::from(ce as i64), "e tail must fit one limb");
    d_out[4] = cd as i64;
    e_out[4] = ce as i64;
    let d_out = Signed62(d_out);
    let e_out = Signed62(e_out);
    #[cfg(any(test, debug_assertions))]
    {
        checks::assert_coeff_range::<P>(&d_out, "new d");
        checks::assert_coeff_range::<P>(&e_out, "new e");
    }
    (d_out, e_out)
}

/// The terminal coefficient update: only the top row of the matrix, since
/// after the final batch only `d` feeds the result and `e` is dead.
#[allow(clippy::many_single_char_names)]
fn update_d_only_62<P: InvParams>(d: &mut Signed62, e: &Signed62, u: i64, v: i64) {
    let d0 = d.0[0];
    let d1 = d.0[1];
    let d2 = d.0[2];
    let d3 = d.0[3];
    let d4 = d.0[4];
    #[cfg(any(test, debug_assertions))]
    {
        checks::assert_coeff_range::<P>(d, "d");
        checks::assert_coeff_range::<P>(e, "e");
    }
    let sd = d4 >> 63;
    let se = e.0[4] >> 63;
    let mut md = (u & sd) + (v & se);
    let mut cd = i128::from(u) * i128::from(d0) + i128::from(v) * i128::from(e.0[0]);
    md = md.wrapping_sub((P::MU.wrapping_mul(cd as u64).wrapping_add(md as u64) & MASK62) as i64);
    cd += i128::from(P::MODULUS[0]) * i128::from(md);
    verify!(cd as u64 & MASK62 == 0, "low bits of new d must vanish");
    cd >>= 62;
    cd += i128::from(u) * i128::from(d1) + i128::from(v) * i128::from(e.0[1]);
    cd += i128::from(P::MODULUS[1]) * i128::from(md);
    d.0[0] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    cd += i128::from(u) * i128::from(d2) + i128::from(v) * i128::from(e.0[2]);
    cd += i128::from(md) << 1;
    d.0[1] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    cd += i128::from(u) * i128::from(d3) + i128::from(v) * i128::from(e.0[3]);
    d.0[2] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    cd += i128::from(u) * i128::from(d4) + i128::from(v) * i128::from(e.0[4]);
    cd += i128::from(md) << 6;
    d.0[3] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    verify!(cd == i128::from(cd as i64), "d tail must fit one limb");
    d.0[4] = cd as i64;
    #[cfg(any(test, debug_assertions))]
    checks::assert_coeff_range::<P>(d, "new d");
}

/// Normalizes `r` from $(-2m, m)$ to $[0, m)$, negating first if `sign` is
/// negative. Upstream's `secp256k1_modinv64_normalize_62` (the mask-based
/// sequence, kept for 1:1 auditability even though this path is
/// variable-time).
fn normalize_62<P: InvParams>(r: &mut Signed62, sign: i64) {
    #[cfg(any(test, debug_assertions))]
    checks::assert_coeff_range::<P>(r, "normalize input");
    let mut r0 = r.0[0];
    let mut r1 = r.0[1];
    let mut r2 = r.0[2];
    let mut r3 = r.0[3];
    let mut r4 = r.0[4];

    // Add the modulus if the input is negative, bringing r to (-m, m); then
    // negate if requested (still (-m, m)). All limbs remain within i64.
    let mut cond_add = r4 >> 63;
    r0 += P::MODULUS[0] & cond_add;
    r1 += P::MODULUS[1] & cond_add;
    r2 += P::MODULUS[2] & cond_add;
    r3 += P::MODULUS[3] & cond_add;
    r4 += P::MODULUS[4] & cond_add;
    let cond_negate = sign >> 63;
    r0 = (r0 ^ cond_negate) - cond_negate;
    r1 = (r1 ^ cond_negate) - cond_negate;
    r2 = (r2 ^ cond_negate) - cond_negate;
    r3 = (r3 ^ cond_negate) - cond_negate;
    r4 = (r4 ^ cond_negate) - cond_negate;
    // Propagate the carries.
    r1 += r0 >> 62;
    r0 &= MASK62 as i64;
    r2 += r1 >> 62;
    r1 &= MASK62 as i64;
    r3 += r2 >> 62;
    r2 &= MASK62 as i64;
    r4 += r3 >> 62;
    r3 &= MASK62 as i64;
    // Add the modulus again if the result is still negative, bringing r to
    // [0, m), and propagate once more.
    cond_add = r4 >> 63;
    r0 += P::MODULUS[0] & cond_add;
    r1 += P::MODULUS[1] & cond_add;
    r2 += P::MODULUS[2] & cond_add;
    r3 += P::MODULUS[3] & cond_add;
    r4 += P::MODULUS[4] & cond_add;
    r1 += r0 >> 62;
    r0 &= MASK62 as i64;
    r2 += r1 >> 62;
    r1 &= MASK62 as i64;
    r3 += r2 >> 62;
    r2 &= MASK62 as i64;
    r4 += r3 >> 62;
    r3 &= MASK62 as i64;

    r.0 = [r0, r1, r2, r3, r4];
    #[cfg(any(test, debug_assertions))]
    {
        use core::cmp::Ordering;
        assert!(
            r0 >> 62 == 0 && r1 >> 62 == 0 && r2 >> 62 == 0 && r3 >> 62 == 0 && r4 >> 62 == 0,
            "normalize output must be canonical"
        );
        assert!(
            checks::mul_cmp_62(r, 5, &Signed62([0; 5]), 0) != Ordering::Less,
            "normalize output must be non-negative"
        );
        assert!(
            checks::mul_cmp_62(r, 5, &Signed62(P::MODULUS), 1) == Ordering::Less,
            "normalize output must be reduced"
        );
    }
}

/// Whether the `len` active limbs of `g` are all zero (bottom limb checked
/// first as a fast path, as upstream does).
fn is_zero(g: &Signed62, len: usize) -> bool {
    if g.0[0] != 0 {
        return false;
    }
    let mut cond = 0i64;
    for limb in &g.0[1..len] {
        cond |= limb;
    }
    cond == 0
}

/// Shrinks the active length of `f` and `g` by one when both top limbs are
/// pure sign extensions of the limb below, folding their signs down.
fn shrink_len(f: &mut Signed62, g: &mut Signed62, len: &mut usize) {
    let l = *len;
    let fn_ = f.0[l - 1];
    let gn = g.0[l - 1];
    let mut cond = ((l as i64) - 2) >> 63;
    cond |= fn_ ^ (fn_ >> 63);
    cond |= gn ^ (gn >> 63);
    if cond == 0 {
        f.0[l - 2] = (f.0[l - 2] as u64 | (fn_ as u64) << 62) as i64;
        g.0[l - 2] = (g.0[l - 2] as u64 | (gn as u64) << 62) as i64;
        *len = l - 1;
    }
}

/// Inverts a nonzero field element given by its canonical 4x64-bit
/// little-endian internal (Montgomery) representation, returning the
/// representation of the inverse — see the module docs for why the result is
/// already in Montgomery form. Returns `None` for zero.
pub(crate) fn invert<P: InvParams>(x: &[u64; 4]) -> Option<[u64; 4]> {
    invert_counted::<P>(x).map(|(limbs, _)| limbs)
}

/// [`invert`], additionally reporting how many 62-divstep batches ran (used
/// by tests to pin the batch-count distribution).
pub(crate) fn invert_counted<P: InvParams>(x: &[u64; 4]) -> Option<([u64; 4], u32)> {
    if x == &[0u64; 4] {
        return None;
    }
    let mut f = Signed62(P::MODULUS);
    let mut g = pack62(x);
    let mut d;
    let mut e = Signed62(P::R2);
    let mut eta: i64 = -1;
    let mut len: usize = 5;

    // Batch 1, specialized: f = m is sparse, d = 0, and 0 <= e = R^2 < m.
    //
    // This batch can never terminate the algorithm: g' = (q*m + r*g)/2^62 = 0
    // with 0 < g < m would need m | r*g, hence (gcd(g, m) = 1) m | r, hence
    // (|r| <= 2^62 < m) r = 0, and then q*m = 2^62*g' = 0 forces q = 0 —
    // making det T = u*r - v*q = 0, contradicting det T = 2^62. Even if this
    // reasoning were somehow violated, the next batch's divsteps on g0 = 0
    // produce the exact identity matrix scaled by 2^62, and batch 2
    // terminates correctly.
    let (t, new_eta) = divsteps_62_var(eta, f.0[0] as u64, g.0[0] as u64);
    eta = new_eta;
    update_fg_62_first::<P>(&mut f, &mut g, &t);
    verify!(
        !is_zero(&g, len),
        "the first divstep batch cannot terminate"
    );
    let (nd, ne) = update_de_62_first::<P>(&e, &t);
    d = nd;
    e = ne;
    #[cfg(any(test, debug_assertions))]
    checks::assert_fg_range::<P>(&f, &g, len);
    shrink_len(&mut f, &mut g, &mut len);

    // 12 * 62 = 744 divsteps upper-bound the requirement of
    // floor((49 * 255 + 57) / 17) = 738 for 255-bit inputs (Bernstein-Yang;
    // upstream documents the analogous 741 for 256-bit inputs).
    for batch in 2..=12u32 {
        let (t, new_eta) = divsteps_62_var(eta, f.0[0] as u64, g.0[0] as u64);
        eta = new_eta;
        // Update f,g first: if this batch terminates, only the top row of the
        // coefficient update is needed.
        update_fg_62_var(len, &mut f, &mut g, &t);
        if is_zero(&g, len) {
            update_d_only_62::<P>(&mut d, &e, t.u, t.v);
            #[cfg(any(test, debug_assertions))]
            {
                use core::cmp::Ordering;
                let one = Signed62([1, 0, 0, 0, 0]);
                assert!(
                    checks::mul_cmp_62(&f, len, &one, 1) == Ordering::Equal
                        || checks::mul_cmp_62(&f, len, &one, -1) == Ordering::Equal,
                    "f must be +/-1 at termination"
                );
            }
            // The sign of f lives in its top *active* limb.
            normalize_62::<P>(&mut d, f.0[len - 1]);
            return Some((unpack62(&d), batch));
        }
        update_de_62::<P>(&mut d, &mut e, &t);
        #[cfg(any(test, debug_assertions))]
        checks::assert_fg_range::<P>(&f, &g, len);
        shrink_len(&mut f, &mut g, &mut len);
    }
    panic!("safegcd inversion exceeded its 744-divstep bound");
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use core::cmp::Ordering;
    use ff::{Field, PrimeField, WithSmallOrderMulGroup};
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;

    const SEED: [u8; 16] = [
        0x62, 0x64, 0x69, 0x76, 0x73, 0x74, 0x65, 0x70, 0x73, 0x36, 0x32, 0x76, 0x61, 0x72, 0x21,
        0x21,
    ];

    /// Fixed regression vectors: internal little-endian representations with
    /// their exact expected 62-divstep batch counts, produced by an
    /// independently validated simulation of this exact algorithm. If a count
    /// drifts, the port has deviated from the validated control flow:
    /// investigate the divergence, do not re-pin the count.
    const PALLAS_VECTORS: &[([u64; 4], u32)] = &[
        (
            [
                0x34786d38fffffffd,
                0x992c350be41914ad,
                0xffffffffffffffff,
                0x3fffffffffffffff,
            ],
            7,
        ),
        ([0x00000000000005d1, 0, 0, 0], 8),
        (
            [
                0xe096c0a18679d7ae,
                0x2cc4e34bc6b6f06a,
                0x20eddc12b5a7661d,
                0x0c3c5e66537210ad,
            ],
            8,
        ),
        ([1, 0, 0, 0], 9),
        (
            [
                0xa162a7d34ad63d62,
                0xba71beb748b1fa25,
                0x68dc1330fab3847b,
                0x2dc205e082f2c197,
            ],
            10,
        ),
        (
            [
                0x7ceb4d8e9534361d,
                0x8d7879adc97a8e79,
                0x19ed0538ef7b15eb,
                0x35fb58b0ee06c5b4,
            ],
            10,
        ),
    ];

    const VESTA_VECTORS: &[([u64; 4], u32)] = &[
        (
            [
                0x5b2b3e9cfffffffd,
                0x992c350be3420567,
                0xffffffffffffffff,
                0x3fffffffffffffff,
            ],
            7,
        ),
        (
            [
                0x3ea12df55a259593,
                0xd9c58668e724391e,
                0xfcf9d370dd78552a,
                0x1f7fd517b4f7efeb,
            ],
            8,
        ),
        (
            [
                0x4973c2bd762bc27a,
                0x9085d079ecab3a12,
                0x9f66128174a6731a,
                0x3e7853d1fb18fcca,
            ],
            8,
        ),
        ([1, 0, 0, 0], 9),
        ([0x00000ba5f061296c, 0, 0, 0], 10),
        (
            [
                0xc00e79606e554fa8,
                0x13f9947f444f41d3,
                0xd5a780db63e83468,
                0x337e275425a385f3,
            ],
            10,
        ),
    ];

    // ---- signed-62 value helpers (test-side big arithmetic) ----

    /// `a + m`, preserving canonical form.
    fn s62_add_m(a: &Signed62, m: &[i64; 5]) -> Signed62 {
        let mut r = [0i64; 5];
        let mut carry: i64 = 0;
        for i in 0..4 {
            let s = a.0[i] + m[i] + carry;
            r[i] = s & MASK62 as i64;
            carry = s >> 62;
        }
        r[4] = a.0[4] + m[4] + carry;
        Signed62(r)
    }

    /// `a - m`, preserving canonical form.
    fn s62_sub_m(a: &Signed62, m: &[i64; 5]) -> Signed62 {
        let mut r = [0i64; 5];
        let mut borrow: i64 = 0;
        for i in 0..4 {
            let s = a.0[i] - m[i] + borrow;
            r[i] = s & MASK62 as i64;
            borrow = s >> 62;
        }
        r[4] = a.0[4] - m[4] + borrow;
        Signed62(r)
    }

    /// `-a`, preserving canonical form.
    fn s62_neg(a: &Signed62) -> Signed62 {
        let mut r = [0i64; 5];
        let mut borrow: i64 = 0;
        for i in 0..4 {
            let s = -a.0[i] + borrow;
            r[i] = s & MASK62 as i64;
            borrow = s >> 62;
        }
        r[4] = -a.0[4] + borrow;
        Signed62(r)
    }

    /// Reduces a value in `(-2m, 2m)` to `[0, m)`.
    fn s62_mod_m(a: &Signed62, m: &[i64; 5]) -> Signed62 {
        let zero = Signed62([0; 5]);
        let ms = Signed62(*m);
        let mut x = *a;
        while checks::mul_cmp_62(&x, 5, &zero, 0) == Ordering::Less {
            x = s62_add_m(&x, m);
        }
        while checks::mul_cmp_62(&x, 5, &ms, 1) != Ordering::Less {
            x = s62_sub_m(&x, m);
        }
        x
    }

    /// Widens a `len`-limb value to five canonical limbs (sign-preserving).
    fn s62_widen(s: &Signed62, len: usize) -> Signed62 {
        let mut r = [0i64; 5];
        r[..len].copy_from_slice(&s.0[..len]);
        for i in (len - 1)..4 {
            let carry = r[i] >> 62;
            r[i] &= MASK62 as i64;
            r[i + 1] += carry;
        }
        Signed62(r)
    }

    // ---- upstream-ordered reference driver with generic kernels ----

    /// Generic-modulus port of upstream `secp256k1_modinv64_update_de_62`:
    /// multiplies by every modulus limb (including the zero one), applies no
    /// first-batch or terminal specialization.
    #[allow(clippy::many_single_char_names)]
    fn ref_update_de(d: &mut Signed62, e: &mut Signed62, t: &Trans2x2, m: &[i64; 5], mu: u64) {
        let (u, v, q, r) = (t.u, t.v, t.q, t.r);
        let sd = d.0[4] >> 63;
        let se = e.0[4] >> 63;
        let mut md = (u & sd) + (v & se);
        let mut me = (q & sd) + (r & se);
        let mut cd = i128::from(u) * i128::from(d.0[0]) + i128::from(v) * i128::from(e.0[0]);
        let mut ce = i128::from(q) * i128::from(d.0[0]) + i128::from(r) * i128::from(e.0[0]);
        md = md.wrapping_sub((mu.wrapping_mul(cd as u64).wrapping_add(md as u64) & MASK62) as i64);
        me = me.wrapping_sub((mu.wrapping_mul(ce as u64).wrapping_add(me as u64) & MASK62) as i64);
        cd += i128::from(m[0]) * i128::from(md);
        ce += i128::from(m[0]) * i128::from(me);
        assert!(cd as u64 & MASK62 == 0 && ce as u64 & MASK62 == 0);
        cd >>= 62;
        ce >>= 62;
        let d_in = *d;
        let e_in = *e;
        for i in 1..5 {
            cd += i128::from(u) * i128::from(d_in.0[i]) + i128::from(v) * i128::from(e_in.0[i]);
            ce += i128::from(q) * i128::from(d_in.0[i]) + i128::from(r) * i128::from(e_in.0[i]);
            cd += i128::from(m[i]) * i128::from(md);
            ce += i128::from(m[i]) * i128::from(me);
            d.0[i - 1] = (cd as u64 & MASK62) as i64;
            cd >>= 62;
            e.0[i - 1] = (ce as u64 & MASK62) as i64;
            ce >>= 62;
        }
        assert!(cd == i128::from(cd as i64) && ce == i128::from(ce as i64));
        d.0[4] = cd as i64;
        e.0[4] = ce as i64;
    }

    /// Generic-modulus copy of [`normalize_62`].
    fn ref_normalize(r: &mut Signed62, sign: i64, m: &[i64; 5]) {
        let mut l = r.0;
        let mut cond_add = l[4] >> 63;
        for i in 0..5 {
            l[i] += m[i] & cond_add;
        }
        let cond_negate = sign >> 63;
        for i in 0..5 {
            l[i] = (l[i] ^ cond_negate) - cond_negate;
        }
        for i in 0..4 {
            l[i + 1] += l[i] >> 62;
            l[i] &= MASK62 as i64;
        }
        cond_add = l[4] >> 63;
        for i in 0..5 {
            l[i] += m[i] & cond_add;
        }
        for i in 0..4 {
            l[i + 1] += l[i] >> 62;
            l[i] &= MASK62 as i64;
        }
        r.0 = l;
    }

    /// One recorded batch of a reference run: the transition matrix and the
    /// full state *before* the batch's updates.
    #[derive(Clone, Copy)]
    struct BatchRec {
        t: Trans2x2,
        d_in: Signed62,
        e_in: Signed62,
        f_in: Signed62,
        g_in: Signed62,
        len: usize,
        terminal: bool,
    }

    struct RefRun {
        out: [u64; 4],
        batches: u32,
        trace: [BatchRec; 12],
    }

    /// Upstream-ordered reference driver: d,e updated with the generic kernel
    /// *every* batch (including the terminal one) before the g == 0 check —
    /// the ordering `secp256k1_modinv64_var` uses. Output must be limb-exact
    /// equal to the production driver's.
    fn ref_invert(x: &[u64; 4], m: &[i64; 5], mu: u64, e_seed: &[i64; 5]) -> Option<RefRun> {
        if x == &[0u64; 4] {
            return None;
        }
        let dummy = BatchRec {
            t: Trans2x2 {
                u: 1,
                v: 0,
                q: 0,
                r: 1,
            },
            d_in: Signed62([0; 5]),
            e_in: Signed62([0; 5]),
            f_in: Signed62([0; 5]),
            g_in: Signed62([0; 5]),
            len: 5,
            terminal: false,
        };
        let mut trace = [dummy; 12];
        let mut f = Signed62(*m);
        let mut g = pack62(x);
        let mut d = Signed62([0; 5]);
        let mut e = Signed62(*e_seed);
        let mut eta: i64 = -1;
        let mut len: usize = 5;
        for batch in 1..=12u32 {
            let (t, new_eta) = divsteps_62_var(eta, f.0[0] as u64, g.0[0] as u64);
            eta = new_eta;
            let rec = &mut trace[(batch - 1) as usize];
            *rec = BatchRec {
                t,
                d_in: d,
                e_in: e,
                f_in: f,
                g_in: g,
                len,
                terminal: false,
            };
            ref_update_de(&mut d, &mut e, &t, m, mu);
            update_fg_62_var(len, &mut f, &mut g, &t);
            if is_zero(&g, len) {
                trace[(batch - 1) as usize].terminal = true;
                ref_normalize(&mut d, f.0[len - 1], m);
                return Some(RefRun {
                    out: unpack62(&d),
                    batches: batch,
                    trace,
                });
            }
            shrink_len(&mut f, &mut g, &mut len);
        }
        panic!("reference driver exceeded 12 batches");
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn checked_mode_is_active() {
        // T24: the verify! strategy relies on cfg(test) being live even in
        // --release test runs; a refactor to debug_assert! would disarm it.
        assert!(cfg!(any(test, debug_assertions)));
    }

    macro_rules! modinv_field_tests {
        ($F:path, $mod:ident, $P:ty, $fermat_exp:expr_2021, $vectors:expr_2021) => {
            mod $mod {
                use super::*;

                fn modulus62() -> [i64; 5] {
                    <$P as InvParams>::MODULUS
                }

                fn m64() -> [u64; 4] {
                    unpack62(&Signed62(modulus62()))
                }

                fn fermat_invert(x: &$F) -> $F {
                    x.pow_vartime(&$fermat_exp)
                }

                fn rnd_elt(rng: &mut XorShiftRng) -> $F {
                    <$F>::from_raw([
                        rng.next_u64(),
                        rng.next_u64(),
                        rng.next_u64(),
                        rng.next_u64(),
                    ])
                }

                /// Lifts a signed-62 value in (-2m, 2m) into the field by the
                /// identity representation map (its canonical residue becomes
                /// the internal representation).
                fn lift(s: &Signed62) -> $F {
                    let m = modulus62();
                    $F(unpack62(&s62_mod_m(s, &m)))
                }

                fn lift_fg(s: &Signed62, len: usize) -> $F {
                    lift(&s62_widen(s, len))
                }

                /// Asserts the scaled invariants Xd = R^2 f and Xe = R^2 g
                /// (mod m) at every recorded batch boundary of a reference
                /// run. Field multiplication divides both sides by the same
                /// R, so representation-level products compare exactly.
                fn assert_trace_congruences(x_repr: &[u64; 4], run: &RefRun) {
                    let xf = $F(*x_repr);
                    let r2f = $F(unpack62(&Signed62(<$P as InvParams>::R2)));
                    for rec in run.trace[..run.batches as usize].iter() {
                        assert_eq!(
                            xf * lift(&rec.d_in),
                            r2f * lift_fg(&rec.f_in, rec.len),
                            "Xd = R^2 f violated"
                        );
                        assert_eq!(
                            xf * lift(&rec.e_in),
                            r2f * lift_fg(&rec.g_in, rec.len),
                            "Xe = R^2 g violated"
                        );
                    }
                }

                fn ref_run(x: &[u64; 4]) -> RefRun {
                    ref_invert(
                        x,
                        &modulus62(),
                        <$P as InvParams>::MU,
                        &<$P as InvParams>::R2,
                    )
                    .expect("nonzero input")
                }

                #[test]
                fn constants_cross_check() {
                    // T1: the radix-62 constants are computed, not trusted.
                    // Rebuild m from field arithmetic: (-1).to_repr() + 1.
                    let m_minus_1 = (-<$F>::ONE).to_repr();
                    let mut limbs = [0u64; 4];
                    for (i, chunk) in m_minus_1.as_ref().chunks(8).enumerate() {
                        limbs[i] = u64::from_le_bytes(chunk.try_into().unwrap());
                    }
                    let mut carry = 1u64;
                    for l in limbs.iter_mut() {
                        let (s, c) = l.overflowing_add(carry);
                        *l = s;
                        carry = u64::from(c);
                    }
                    assert_eq!(limbs, m64(), "MODULUS mismatch vs field constants");

                    // The representation of the field value 2^256 is
                    // 2^256 * R = R^2 (mod m): exactly the R2 constant.
                    let r_val = <$F>::from_u128(1u128 << 64).square().square();
                    assert_eq!(
                        unpack62(&Signed62(<$P as InvParams>::R2)),
                        r_val.0,
                        "R2 mismatch vs field constants"
                    );

                    // mu * m = 1 (mod 2^62), and the sparse shape the kernels
                    // hardcode.
                    let m62 = modulus62();
                    assert_eq!(
                        <$P as InvParams>::MU.wrapping_mul(m62[0] as u64) & MASK62,
                        1
                    );
                    assert_eq!(&m62[2..], &[2, 0, 64]);
                    // R2 is canonical and reduced.
                    let r2 = Signed62(<$P as InvParams>::R2);
                    for l in r2.0 {
                        assert!(l >= 0 && (l as u64) <= MASK62);
                    }
                    assert!(
                        checks::mul_cmp_62(&r2, 5, &Signed62(m62), 1) == Ordering::Less,
                        "R2 must be below m"
                    );
                }

                #[test]
                fn pack_unpack_roundtrip() {
                    // T2
                    let m = m64();
                    let mut cases = [[0u64; 4]; 16];
                    cases[1] = [1, 0, 0, 0];
                    // m - 1:
                    cases[2] = {
                        let mut t = m;
                        t[0] -= 1;
                        t
                    };
                    // single bits (all below both moduli, which exceed 2^254):
                    let bits = [1u32, 61, 62, 63, 124, 186, 248, 254];
                    for (i, b) in bits.iter().enumerate() {
                        let mut t = [0u64; 4];
                        t[(b / 64) as usize] = 1u64 << (b % 64);
                        cases[3 + i] = t;
                    }
                    cases[11] = [MASK62, 0, 0, 0];
                    cases[12] = [u64::MAX, 0, 0, 0];
                    cases[13] = [0, u64::MAX, 0, 0];
                    cases[14] = [0xaaaa_aaaa_aaaa_aaaa, 0x5555_5555_5555_5555, 0, 0];
                    cases[15] = [u64::MAX, u64::MAX, u64::MAX, 0x3fff_ffff_ffff_ffff];
                    for c in cases {
                        assert_eq!(unpack62(&pack62(&c)), c);
                    }
                    let mut rng = XorShiftRng::from_seed(SEED);
                    for _ in 0..1024 {
                        let c = rnd_elt(&mut rng).0;
                        let p = pack62(&c);
                        for l in p.0 {
                            assert!(l >= 0 && (l as u64) <= MASK62);
                        }
                        assert_eq!(unpack62(&p), c);
                    }
                }

                #[test]
                fn divsteps_matrix_properties() {
                    // T3
                    let mut rng = XorShiftRng::from_seed(SEED);
                    for _ in 0..10_000 {
                        let f0 = rng.next_u64() | 1;
                        let g0 = rng.next_u64();
                        let eta_in = (rng.next_u64() % 61) as i64 - 30;
                        let (t, eta_out) = divsteps_62_var(eta_in, f0, g0);
                        assert_eq!(
                            i128::from(t.u) * i128::from(t.r) - i128::from(t.v) * i128::from(t.q),
                            1 << 62
                        );
                        assert!(t.u.unsigned_abs() + t.v.unsigned_abs() <= 1 << 62);
                        assert!(t.q.unsigned_abs() + t.r.unsigned_abs() <= 1 << 62);
                        // After 62 divsteps both rows are divisible by 2^62.
                        assert_eq!(
                            (t.u as u64)
                                .wrapping_mul(f0)
                                .wrapping_add((t.v as u64).wrapping_mul(g0))
                                & MASK62,
                            0
                        );
                        assert_eq!(
                            (t.q as u64)
                                .wrapping_mul(f0)
                                .wrapping_add((t.r as u64).wrapping_mul(g0))
                                & MASK62,
                            0
                        );
                        assert!(eta_out >= eta_in - 62 && eta_out <= eta_in.abs() + 62);
                    }
                }

                #[test]
                fn divsteps_zero_g_identity() {
                    // T10: with g0 = 0 the whole batch is halvings, giving
                    // the identity matrix scaled by 2^62; update_fg then
                    // keeps f and shifts g down one limb.
                    let mut rng = XorShiftRng::from_seed(SEED);
                    for _ in 0..64 {
                        let f0 = rng.next_u64() | 1;
                        let (t, _) = divsteps_62_var(-1, f0, 0);
                        assert_eq!(t.u, 1 << 62);
                        assert_eq!(t.v, 0);
                        assert_eq!(t.q, 0);
                        assert_eq!(t.r, 1);
                        let mut f = pack62(&rnd_elt(&mut rng).0);
                        let f_before = f;
                        let mut g = pack62(&rnd_elt(&mut rng).0);
                        g.0[0] = 0;
                        let g_before = g;
                        update_fg_62_var(5, &mut f, &mut g, &t);
                        assert_eq!(f, f_before);
                        assert_eq!(
                            g.0,
                            [
                                g_before.0[1],
                                g_before.0[2],
                                g_before.0[3],
                                g_before.0[4],
                                0
                            ]
                        );
                    }
                }

                #[test]
                fn update_de_sparse_matches_generic() {
                    // T4 + T8: the production sparse-modulus kernel is
                    // limb-exact against the generic reference over real
                    // trajectory states and synthetic boundary states, and
                    // its outputs respect the (-2m, m) range invariant.
                    let m = modulus62();
                    let mu = <$P as InvParams>::MU;
                    let mut rng = XorShiftRng::from_seed(SEED);
                    let check_state = |d: &Signed62, e: &Signed62, t: &Trans2x2| {
                        let mut d_prod = *d;
                        let mut e_prod = *e;
                        update_de_62::<$P>(&mut d_prod, &mut e_prod, t);
                        let mut d_ref = *d;
                        let mut e_ref = *e;
                        ref_update_de(&mut d_ref, &mut e_ref, t, &m, mu);
                        assert_eq!(d_prod, d_ref, "d limbs diverge from generic kernel");
                        assert_eq!(e_prod, e_ref, "e limbs diverge from generic kernel");
                        checks::assert_coeff_range::<$P>(&d_prod, "d output");
                        checks::assert_coeff_range::<$P>(&e_prod, "e output");
                    };
                    // Real trajectory states.
                    for _ in 0..50 {
                        let x = rnd_elt(&mut rng).0;
                        let run = ref_run(&x);
                        for rec in run.trace[..run.batches as usize].iter() {
                            check_state(&rec.d_in, &rec.e_in, &rec.t);
                        }
                    }
                    // Synthetic boundary states under real matrices.
                    let zero = Signed62([0; 5]);
                    let one = Signed62([1, 0, 0, 0, 0]);
                    let m_s = Signed62(m);
                    let m_minus_1 = s62_neg(&s62_sub_m(&one, &m)); // -(1 - m)
                    let neg_one = s62_neg(&one);
                    let neg_m = s62_neg(&m_s);
                    let neg_2m_plus_1 = s62_add_m(&s62_neg(&s62_add_m(&m_s, &m)), &one.0);
                    let boundary = [zero, one, m_minus_1, neg_one, neg_m, neg_2m_plus_1];
                    let sample = ref_run(&rnd_elt(&mut rng).0);
                    for rec in sample.trace[..sample.batches as usize].iter() {
                        for d in &boundary {
                            for e in &boundary {
                                check_state(d, e, &rec.t);
                            }
                        }
                    }
                }

                #[test]
                fn update_fg_first_matches_generic() {
                    // T5
                    let m = modulus62();
                    let mut rng = XorShiftRng::from_seed(SEED);
                    for _ in 0..1000 {
                        let x = rnd_elt(&mut rng).0;
                        let g0 = pack62(&x);
                        let (t, _) = divsteps_62_var(-1, m[0] as u64, g0.0[0] as u64);
                        let mut f_prod = Signed62(m);
                        let mut g_prod = g0;
                        update_fg_62_first::<$P>(&mut f_prod, &mut g_prod, &t);
                        let mut f_ref = Signed62(m);
                        let mut g_ref = g0;
                        update_fg_62_var(5, &mut f_ref, &mut g_ref, &t);
                        assert_eq!(f_prod, f_ref);
                        assert_eq!(g_prod, g_ref);
                    }
                }

                #[test]
                fn update_de_first_matches_generic() {
                    // T6
                    let m = modulus62();
                    let mu = <$P as InvParams>::MU;
                    let mut rng = XorShiftRng::from_seed(SEED);
                    for i in 0..1000 {
                        // The real first-batch e is R2; also cover random
                        // e in [0, m).
                        let e = if i == 0 {
                            Signed62(<$P as InvParams>::R2)
                        } else {
                            pack62(&rnd_elt(&mut rng).0)
                        };
                        let g0 = pack62(&rnd_elt(&mut rng).0);
                        let (t, _) = divsteps_62_var(-1, m[0] as u64, g0.0[0] as u64);
                        let (d_prod, e_prod) = update_de_62_first::<$P>(&e, &t);
                        let mut d_ref = Signed62([0; 5]);
                        let mut e_ref = e;
                        ref_update_de(&mut d_ref, &mut e_ref, &t, &m, mu);
                        assert_eq!(d_prod, d_ref);
                        assert_eq!(e_prod, e_ref);
                    }
                }

                #[test]
                fn update_d_only_matches_generic_row0() {
                    // T7: the terminal kernel equals the d-row of the generic
                    // update on real terminal states.
                    let m = modulus62();
                    let mu = <$P as InvParams>::MU;
                    let mut rng = XorShiftRng::from_seed(SEED);
                    let mut seen_terminal = 0u32;
                    for _ in 0..200 {
                        let x = rnd_elt(&mut rng).0;
                        let run = ref_run(&x);
                        let rec = &run.trace[(run.batches - 1) as usize];
                        assert!(rec.terminal);
                        let mut d_prod = rec.d_in;
                        update_d_only_62::<$P>(&mut d_prod, &rec.e_in, rec.t.u, rec.t.v);
                        let mut d_ref = rec.d_in;
                        let mut e_ref = rec.e_in;
                        ref_update_de(&mut d_ref, &mut e_ref, &rec.t, &m, mu);
                        assert_eq!(d_prod, d_ref);
                        seen_terminal += 1;
                    }
                    assert_eq!(seen_terminal, 200);
                }

                #[test]
                fn normalize_quadrants() {
                    // T9: all sign/range quadrants, against reference
                    // reduction, catching in particular any wrong-sign-limb
                    // regression.
                    let m = modulus62();
                    let m_s = Signed62(m);
                    let one = Signed62([1, 0, 0, 0, 0]);
                    let mut rng = XorShiftRng::from_seed(SEED);
                    let mut cases: [Signed62; 24] = [Signed62([0; 5]); 24];
                    cases[0] = s62_add_m(&s62_neg(&s62_add_m(&m_s, &m)), &one.0); // -2m + 1
                    cases[1] = s62_neg(&s62_add_m(&m_s, &one.0)); // -m - 1
                    cases[2] = s62_neg(&m_s); // -m
                    cases[3] = s62_add_m(&s62_neg(&m_s), &one.0); // -m + 1
                    cases[4] = s62_neg(&one); // -1
                    cases[5] = Signed62([0; 5]); // 0
                    cases[6] = one; // 1
                    cases[7] = s62_sub_m(&s62_add_m(&Signed62([0; 5]), &m), &one.0); // m - 1
                    for i in 8..24 {
                        // random values across (-2m, m): canonical in [0, m)
                        // minus 0, 1, or 2 copies of m.
                        let mut x = pack62(&rnd_elt(&mut rng).0);
                        for _ in 0..(i % 3) {
                            x = s62_sub_m(&x, &m);
                        }
                        cases[i] = x;
                    }
                    for c in &cases {
                        for sign in [1i64, -1] {
                            let mut got = *c;
                            normalize_62::<$P>(&mut got, sign);
                            let expected = if sign < 0 {
                                s62_mod_m(&s62_neg(c), &m)
                            } else {
                                s62_mod_m(c, &m)
                            };
                            assert_eq!(got, expected, "normalize mismatch");
                        }
                    }
                }

                #[test]
                fn invert_matches_fermat_50k() {
                    // T11
                    let mut rng = XorShiftRng::from_seed(SEED);
                    for _ in 0..50_000 {
                        let x = rnd_elt(&mut rng);
                        if bool::from(x.is_zero()) {
                            continue;
                        }
                        let (out, batches) = invert_counted::<$P>(&x.0).expect("nonzero");
                        assert!(batches <= 12);
                        assert_eq!(out, fermat_invert(&x).0);
                    }
                }

                #[test]
                fn invert_matches_reference_driver() {
                    // T12: limb-exact against the upstream-ordered generic
                    // driver, whose per-batch states also satisfy the scaled
                    // congruence invariants.
                    let mut rng = XorShiftRng::from_seed(SEED);
                    for i in 0..2000 {
                        let x = rnd_elt(&mut rng).0;
                        let (out, batches) = invert_counted::<$P>(&x).expect("nonzero");
                        let run = ref_run(&x);
                        assert_eq!(out, run.out, "production diverges from reference");
                        assert_eq!(batches, run.batches, "batch count diverges");
                        if i < 200 {
                            assert_trace_congruences(&x, &run);
                        }
                    }
                    for (repr, _) in $vectors {
                        let (out, batches) = invert_counted::<$P>(repr).expect("nonzero");
                        let run = ref_run(repr);
                        assert_eq!(out, run.out);
                        assert_eq!(batches, run.batches);
                        assert_trace_congruences(repr, &run);
                    }
                }

                #[test]
                fn invert_fixed_edge_values() {
                    // T13: field-value edges. `invert()` here is already the
                    // divstep implementation under test.
                    let r_val = <$F>::from_u128(1u128 << 64).square().square();
                    let mut edge = [
                        <$F>::ONE,
                        -<$F>::ONE,
                        <$F>::from(2u64),
                        <$F>::TWO_INV,
                        <$F>::MULTIPLICATIVE_GENERATOR,
                        <$F>::ROOT_OF_UNITY,
                        <$F>::ZETA,
                        r_val,
                        r_val.square(),
                    ]
                    .to_vec();
                    for k in 2u64..=64 {
                        edge.push(<$F>::from(k));
                    }
                    for x in edge {
                        let inv = x.invert().unwrap();
                        assert_eq!(x * inv, <$F>::ONE);
                        assert_eq!(inv, fermat_invert(&x));
                    }
                }

                #[test]
                fn invert_fixed_edge_reprs() {
                    // T14: raw representation edges (trailing-zero, eta, and
                    // length-shrink stress). All values below m.
                    let m = m64();
                    let mut reprs = std::vec::Vec::new();
                    reprs.push([1u64, 0, 0, 0]);
                    reprs.push([2, 0, 0, 0]);
                    reprs.push([3, 0, 0, 0]);
                    let mut m_minus = m;
                    m_minus[0] -= 1;
                    reprs.push(m_minus); // m - 1
                    m_minus[0] -= 1;
                    reprs.push(m_minus); // m - 2
                    for k in [1u32, 61, 62, 63, 123, 124, 186, 247, 254] {
                        let mut t = [0u64; 4];
                        t[(k / 64) as usize] = 1u64 << (k % 64);
                        reprs.push(t);
                    }
                    reprs.push([MASK62, 0, 0, 0]); // 2^62 - 1
                    for k in [1u64, 3, 0x5d1] {
                        // k << 200
                        reprs.push([0, 0, 0, k << 8]);
                    }
                    let mut rng = XorShiftRng::from_seed(SEED);
                    for _ in 0..16 {
                        reprs.push([
                            0,
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64() & 0x3fff_ffff_ffff_ffff,
                        ]);
                    }
                    for repr in reprs {
                        let x = $F(repr);
                        let inv = x.invert().unwrap();
                        assert_eq!(x * inv, <$F>::ONE, "repr {:x?}", repr);
                        assert_eq!(inv, fermat_invert(&x), "repr {:x?}", repr);
                    }
                }

                #[test]
                fn invert_zero_is_none() {
                    // T15
                    assert!(invert::<$P>(&[0u64; 4]).is_none());
                    assert!(bool::from(<$F>::ZERO.invert().is_none()));
                    assert!(bool::from(<$F>::ONE.invert().is_some()));
                }

                #[test]
                fn batch_count_distribution_pinned() {
                    // T16: exact batch counts for the searched vectors. See
                    // the vector table's comment: on mismatch, investigate;
                    // do not re-pin.
                    for (repr, expected) in $vectors {
                        let (out, batches) = invert_counted::<$P>(repr).expect("nonzero");
                        assert_eq!(batches, *expected, "batch count drifted for {:x?}", repr);
                        assert_eq!($F(*repr) * $F(out), <$F>::ONE);
                    }
                }

                #[test]
                fn invert_self_consistency_200k() {
                    // T17: x * x^-1 == 1 is a complete correctness check for
                    // inversion; also bounds the batch-count distribution.
                    let mut rng = XorShiftRng::from_seed(SEED);
                    let mut histogram = [0u32; 13];
                    for _ in 0..200_000 {
                        let x = rnd_elt(&mut rng);
                        if bool::from(x.is_zero()) {
                            continue;
                        }
                        let (out, batches) = invert_counted::<$P>(&x.0).expect("nonzero");
                        histogram[batches as usize] += 1;
                        assert_eq!($F(out) * x, <$F>::ONE);
                    }
                    for (batches, count) in histogram.iter().enumerate() {
                        assert!(
                            *count == 0 || (5..=12).contains(&batches),
                            "unexpected batch count {} ({} times)",
                            batches,
                            count
                        );
                    }
                    // Structured families exercise the long tails.
                    for j in 0..252u32 {
                        let mut repr = [0u64; 4];
                        repr[(j / 64) as usize] = 1u64 << (j % 64);
                        repr[0] |= 5;
                        let x = $F(repr);
                        let (out, batches) = invert_counted::<$P>(&x.0).expect("nonzero");
                        assert!(batches <= 12);
                        assert_eq!($F(out) * x, <$F>::ONE);
                    }
                }

                #[test]
                fn termination_bound_constants() {
                    // T18: Bernstein-Yang iteration bound for 255-bit inputs,
                    // as used by the driver's 12-batch cap.
                    assert_eq!((49 * 255 + 57) / 17, 738);
                    assert!(738 <= 12 * 62);
                }

                #[test]
                fn r2_seeding_equivalence() {
                    // T19: seeding e with R^2 is exactly a classic (e = 1)
                    // inversion followed by two R2 Montgomery multiplications.
                    let m = modulus62();
                    let one_seed = [1i64, 0, 0, 0, 0];
                    let r2f = $F(unpack62(&Signed62(<$P as InvParams>::R2)));
                    let mut rng = XorShiftRng::from_seed(SEED);
                    for _ in 0..1000 {
                        let x = rnd_elt(&mut rng);
                        if bool::from(x.is_zero()) {
                            continue;
                        }
                        let classic = ref_invert(&x.0, &m, <$P as InvParams>::MU, &one_seed)
                            .expect("nonzero")
                            .out;
                        let production = invert::<$P>(&x.0).expect("nonzero");
                        assert_eq!(($F(classic) * r2f * r2f).0, production);
                    }
                }

                proptest::proptest! {
                    #[test]
                    fn pbt_inverse(a in proptest::prelude::any::<[u64; 4]>()) {
                        // T25
                        let x = <$F>::from_raw(a);
                        if !bool::from(x.is_zero()) {
                            let inv = x.invert().unwrap();
                            proptest::prop_assert_eq!(inv, fermat_invert(&x));
                            proptest::prop_assert_eq!(x * inv, <$F>::ONE);
                            proptest::prop_assert_eq!(inv.invert().unwrap(), x);
                        }
                    }
                }
            }
        };
    }

    modinv_field_tests!(
        crate::fields::Fp,
        fp,
        FpParams,
        [
            0x992d_30ec_ffff_ffff,
            0x2246_98fc_094c_f91b,
            0x0000_0000_0000_0000,
            0x4000_0000_0000_0000,
        ],
        PALLAS_VECTORS
    );

    modinv_field_tests!(
        crate::fields::Fq,
        fq,
        FqParams,
        [
            0x8c46_eb20_ffff_ffff,
            0x2246_98fc_0994_a8dd,
            0x0000_0000_0000_0000,
            0x4000_0000_0000_0000,
        ],
        VESTA_VECTORS
    );
}
