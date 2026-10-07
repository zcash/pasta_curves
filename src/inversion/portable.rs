//! The inversion's six blocks in portable Rust, for the targets without the AArch64 assembly
//! blocks.
//!
//! Each function computes what its block contract in [`InvertBlocks`] states, in the shape that the
//! Lean word-level model gives it (`lean/PastaCurves/Inversion/`) rather than instruction for
//! instruction after the assembly. The packed divstep is the recurrence of `Packed.lean` on words,
//! with the branch taken by masks; the decoder is the formula of `Packed.lean`'s `unpack`; the rows
//! and the reduction are limb arithmetic modulo `2^320`, with `u128` accumulators over `u64 × u64`
//! products, which 64-bit targets compile to their widening multiplication. The blocks are not
//! modelled in Lean: the crate's tests check them against the same known answers as the assembly
//! blocks. A translation via Aeneas would be possible; the code stays within the subset of Rust
//! that Aeneas could translate (no `unsafe`, explicit wrapping arithmetic, plain loops).
//!
//! There is no data-dependent branch or memory access: the divstep selects its two outcomes by a
//! mask, and the rows fold each matrix entry's sign in by a conditional negation under its mask.

use super::InvertBlocks;
use crate::limbs::Limbs;

/// The mask of a word's sign, read as two's complement: all ones for a negative word, else zero.
#[inline(always)]
fn sign_mask(x: u64) -> u64 {
    ((x as i64) >> 63) as u64
}

/// One packed divstep: the recurrence of `Packed.lean` on the packed state `(d, f, g)`, in
/// two's-complement words. When `d > 0` and `g` is odd, the step is `(2 - d, g, (g - f) / 2)`;
/// otherwise it is `(2 + d, f, (g + (g mod 2) f) / 2)`, the divisions rounding down. `d` stays
/// far from `-2^63`, so the sign of `-d` decides `d > 0`.
#[inline(always)]
fn divstep(d: u64, f: u64, g: u64) -> (u64, u64, u64) {
    let odd = 0u64.wrapping_sub(g & 1);
    let swap = odd & sign_mask(0u64.wrapping_sub(d));
    let d2 = (swap & 2u64.wrapping_sub(d)) | (!swap & d.wrapping_add(2));
    let f2 = (swap & g) | (!swap & f);
    let sum = (swap & g.wrapping_sub(f)) | (!swap & g.wrapping_add(odd & f));
    (d2, f2, ((sum as i64) >> 1) as u64)
}

/// The coefficient pair in the upper bits of a packed word after `k` steps, as `Packed.lean`'s
/// `unpack` reads it: negate the word, take the upper part rounded to the nearest at bit
/// `41 - k`, and split it at bit 21, with the first coefficient taken in `(-2^k, 2^k]`.
#[inline(always)]
fn unpack(k: u32, w: u64) -> (u64, u64) {
    let t = 0u64.wrapping_sub(w) as i64;
    let up = t.wrapping_add(1i64 << (40 - k)) >> (41 - k);
    let v = up.wrapping_add((1i64 << 20) - 1) >> 21;
    (up.wrapping_sub(v << 21) as u64, v as u64)
}

/// `k` packed divsteps from the low words `f` and `g` at `d`, then the batch's matrix read from
/// the two words: `[d', m00, m01, m10, m11]`, the entries as two's-complement words. The packed
/// words start as the low 20 bits of `f` and `g` with the identity's rows `(1, 0)` and `(0, 1)`
/// negated at bits 41 and 62.
#[inline(always)]
fn batch(k: u32, d: u64, f: u64, g: u64) -> [u64; 5] {
    let mut d = d;
    let mut pf = (f & 0xfffff) | 0xfffffe0000000000;
    let mut pg = (g & 0xfffff) | 0xc000000000000000;
    for _ in 0..k {
        (d, pf, pg) = divstep(d, pf, pg);
    }
    let (m00, m01) = unpack(k, pf);
    let (m10, m11) = unpack(k, pg);
    [d, m00, m01, m10, m11]
}

/// The next low word after a batch: the row `(a, b)` applied to the low words, modulo `2^64`,
/// shifted down by the batch's `k` steps (`Divstep59.lean`'s `nextLow`).
#[inline(always)]
fn next_low(k: u32, a: u64, b: u64, f: u64, g: u64) -> u64 {
    a.wrapping_mul(f).wrapping_add(b.wrapping_mul(g)) >> k
}

/// The product of two matrices given as `[m00, m01, m10, m11]`, each entry modulo `2^64`.
#[inline(always)]
fn mat_mul(m: [u64; 4], n: [u64; 4]) -> [u64; 4] {
    [
        m[0].wrapping_mul(n[0])
            .wrapping_add(m[1].wrapping_mul(n[2])),
        m[0].wrapping_mul(n[1])
            .wrapping_add(m[1].wrapping_mul(n[3])),
        m[2].wrapping_mul(n[0])
            .wrapping_add(m[3].wrapping_mul(n[2])),
        m[2].wrapping_mul(n[1])
            .wrapping_add(m[3].wrapping_mul(n[3])),
    ]
}

/// `x` negated under the mask `s`, modulo `2^320`: `-x` for the all-ones mask, `x` for zero.
#[inline(always)]
fn negate(x: &[u64; 5], s: u64) -> [u64; 5] {
    let mut out = [0u64; 5];
    let mut carry = s & 1;
    for i in 0..5 {
        let (word, overflow) = (x[i] ^ s).overflowing_add(carry);
        out[i] = word;
        carry = u64::from(overflow);
    }
    out
}

/// The row combination `a x + b y` modulo `2^320`, for `a` and `b` given as magnitudes `m0`,
/// `m1` and sign masks `s0`, `s1`. Each column's two products and the carry stay below `2^128`
/// because `m0 + m1 ≤ 2^63`, which the row bound of the contracts gives.
#[inline(always)]
fn row(x: &[u64; 5], y: &[u64; 5], m0: u64, m1: u64, s0: u64, s1: u64) -> [u64; 5] {
    let x = negate(x, s0);
    let y = negate(y, s1);
    let mut out = [0u64; 5];
    let mut carry: u128 = 0;
    for i in 0..5 {
        let column = carry + u128::from(x[i]) * u128::from(m0) + u128::from(y[i]) * u128::from(m1);
        out[i] = column as u64;
        carry = column >> 64;
    }
    out
}

/// `x + y` modulo `2^320`.
#[inline(always)]
fn add5(x: &[u64; 5], y: &[u64; 5]) -> [u64; 5] {
    let mut out = [0u64; 5];
    let mut carry = 0u64;
    for i in 0..5 {
        let (sum, overflow1) = x[i].overflowing_add(y[i]);
        let (sum, overflow2) = sum.overflowing_add(carry);
        out[i] = sum;
        carry = u64::from(overflow1 | overflow2);
    }
    out
}

/// The portable inversion blocks, for the generic driver of `crate::inversion`.
pub(crate) struct Backend;

impl InvertBlocks for Backend {
    /// Three batches of 20, 20, and 19 packed divsteps, the next low words computed between
    /// them, and the batches' matrices multiplied with the newest on the left.
    #[inline]
    fn divstep59(d: u64, f0: u64, g0: u64) -> [u64; 5] {
        let [d1, a1, b1, c1, e1] = batch(20, d, f0, g0);
        let f1 = next_low(20, a1, b1, f0, g0);
        let g1 = next_low(20, c1, e1, f0, g0);
        let [d2, a2, b2, c2, e2] = batch(20, d1, f1, g1);
        let f2 = next_low(20, a2, b2, f1, g1);
        let g2 = next_low(20, c2, e2, f1, g1);
        let [d3, a3, b3, c3, e3] = batch(19, d2, f2, g2);
        let [m00, m01, m10, m11] = mat_mul(
            [a3, b3, c3, e3],
            mat_mul([a2, b2, c2, e2], [a1, b1, c1, e1]),
        );
        [d3, m00, m01, m10, m11]
    }

    /// Each entry's sign mask, and its magnitude as the entry negated under that mask.
    #[inline]
    fn sign_mag(m00: u64, m01: u64, m10: u64, m11: u64) -> [u64; 8] {
        let [s00, s01, s10, s11] = [
            sign_mask(m00),
            sign_mask(m01),
            sign_mask(m10),
            sign_mask(m11),
        ];
        [
            (m00 ^ s00).wrapping_sub(s00),
            (m01 ^ s01).wrapping_sub(s01),
            (m10 ^ s10).wrapping_sub(s10),
            (m11 ^ s11).wrapping_sub(s11),
            s00,
            s01,
            s10,
            s11,
        ]
    }

    /// The row combination modulo `2^320`, then the shift right by 59 of the five words, the
    /// top word arithmetically, which rounds the signed value down.
    #[inline]
    fn fg_row(f: &[u64; 5], g: &[u64; 5], m0: u64, m1: u64, s0: u64, s1: u64) -> [u64; 5] {
        let t = row(f, g, m0, m1, s0, s1);
        [
            (t[0] >> 59) | (t[1] << 5),
            (t[1] >> 59) | (t[2] << 5),
            (t[2] >> 59) | (t[3] << 5),
            (t[3] >> 59) | (t[4] << 5),
            ((t[4] as i64) >> 59) as u64,
        ]
    }

    /// The row combination of the four-word values with a zero sign word.
    #[inline]
    fn uv_row(u: &Limbs, v: &Limbs, m0: u64, m1: u64, s0: u64, s1: u64) -> [u64; 5] {
        let u = [u[0], u[1], u[2], u[3], 0];
        let v = [v[0], v[1], v[2], v[3], 0];
        row(&u, &v, m0, m1, s0, s1)
    }

    /// Adds `2^61 p`, then adds the multiple `w p` of the modulus that clears the low word and
    /// drops that word. The words compute modulo `2^320`; the carry out of the top word is
    /// dropped, since the result is below `2^256` (Lemma 10 of `book/src/design/inversion.md`).
    #[inline]
    fn amontred(t: &[u64; 5], modulus: &Limbs, inv: u64) -> Limbs {
        let p = modulus;
        let p61 = [
            p[0] << 61,
            (p[0] >> 3) | (p[1] << 61),
            (p[1] >> 3) | (p[2] << 61),
            (p[2] >> 3) | (p[3] << 61),
            p[3] >> 3,
        ];
        let s = add5(t, &p61);
        let w = s[0].wrapping_mul(inv);
        // The low word cancels: `s[0] + w p[0] ≡ 0 (mod 2^64)`.
        let carry = (u128::from(s[0]) + u128::from(w) * u128::from(p[0])) >> 64;
        let column1 = carry + u128::from(s[1]) + u128::from(w) * u128::from(p[1]);
        let column2 = (column1 >> 64) + u128::from(s[2]) + u128::from(w) * u128::from(p[2]);
        let column3 = (column2 >> 64) + u128::from(s[3]) + u128::from(w) * u128::from(p[3]);
        let column4 = (column3 >> 64) + u128::from(s[4]);
        [
            column1 as u64,
            column2 as u64,
            column3 as u64,
            column4 as u64,
        ]
    }

    /// The four-word subtraction of the modulus, kept unless it borrows out of the top word.
    #[inline]
    fn cond_sub(value: &Limbs, modulus: &Limbs) -> Limbs {
        let mut difference = [0u64; 4];
        let mut borrow = 0u64;
        for i in 0..4 {
            let (word, underflow1) = value[i].overflowing_sub(modulus[i]);
            let (word, underflow2) = word.overflowing_sub(borrow);
            difference[i] = word;
            borrow = u64::from(underflow1 | underflow2);
        }
        let keep = 0u64.wrapping_sub(borrow);
        let mut out = [0u64; 4];
        for i in 0..4 {
            out[i] = (keep & value[i]) | (!keep & difference[i]);
        }
        out
    }
}

/// The generic checks of `crate::inversion` over the portable blocks.
#[cfg(test)]
mod tests {
    use super::Backend;
    use crate::inversion::tests as checks;

    #[test]
    fn sign_mag_known_answers() {
        checks::sign_mag_known_answers::<Backend>();
    }

    #[test]
    fn divstep59_known_answers() {
        checks::divstep59_known_answers::<Backend>();
    }

    #[test]
    fn fg_row_known_answers() {
        checks::fg_row_known_answers::<Backend>();
    }

    #[test]
    fn uv_row_known_answers() {
        checks::uv_row_known_answers::<Backend>();
    }

    #[test]
    fn amontred_known_answers() {
        checks::amontred_known_answers::<Backend>();
    }

    #[test]
    fn cond_sub_known_answers() {
        checks::cond_sub_known_answers::<Backend>();
    }

    #[test]
    fn invert_known_answers() {
        checks::invert_known_answers::<Backend>();
    }

    #[test]
    fn invert_small_and_near_modulus() {
        checks::invert_small_and_near_modulus::<Backend>();
    }

    #[test]
    fn invert_random() {
        checks::invert_random::<Backend>();
    }

    #[test]
    fn invert_is_the_inverse() {
        checks::invert_is_the_inverse::<Backend>();
    }

    #[test]
    fn invert_is_multiplicative() {
        checks::invert_is_multiplicative::<Backend>();
    }

    #[test]
    fn invert_commutes_with_negation() {
        checks::invert_commutes_with_negation::<Backend>();
    }
}
