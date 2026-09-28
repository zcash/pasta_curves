//! The constant-time inversion: one Rust driver over the six inline blocks of a backend, which
//! `src/asm/aarch64.rs` provides. `lean/PastaCurves/Compositions.lean` mirrors the driver over a
//! record of the blocks.

use super::Limbs;
use super::aarch64::{amontred, cond_sub, divstep59, fg_row, sign_mag, de_row};

/// The constant-time inversion in Montgomery form: for a canonical `x`, the
/// canonical `z` with `x · z ≡ 2^512 (mod p)`, and `0` for `x = 0`.
///
/// The composition of the blocks that `lean/PastaCurves/Inversion/Model.lean`'s
/// `montInvModel` specifies. From `(two_delta, f, g, d, e) = (1, p, x, 0, e0)`, with
/// `e0 = 2^562 mod p`, nine rounds each run `divstep59` on the low words of
/// `f` and `g`, take the matrix's sign-magnitude form, update `f` and `g` by
/// its two rows, and combine `d` and `e` by the two rows, each combination
/// reduced by `amontred`. The invariant after round `i` is
/// `(f, g) ≡ x · 2^(5i - 562) · (d, e) (mod p)`. The tenth round computes only
/// `d`, with the sign of the new `f` (which is `±1`, `g` being `0`) folded into
/// the row's masks, and reduces strictly. That sign is the top bit of the low
/// word of `u · f + v · g`, since that sum is `2^59 · f'`.
#[inline]
pub(crate) fn invert(x: &Limbs, modulus: &Limbs, inv: u64, e0: &Limbs) -> Limbs {
    let mut two_delta: u64 = 1;
    let mut f = [modulus[0], modulus[1], modulus[2], modulus[3], 0];
    let mut g = [x[0], x[1], x[2], x[3], 0];
    let mut d: Limbs = [0; 4];
    let mut e: Limbs = *e0;
    for _ in 0..9 {
        let [two_delta_new, u, v, q, r] = divstep59(two_delta, f[0], g[0]);
        two_delta = two_delta_new;
        let [u, v, q, r, su, sv, sq, sr] = sign_mag(u, v, q, r);
        let f_new = fg_row(&f, &g, u, v, su, sv);
        g = fg_row(&f, &g, q, r, sq, sr);
        f = f_new;
        let td = de_row(&d, &e, u, v, su, sv);
        let te = de_row(&d, &e, q, r, sq, sr);
        d = amontred(&td, modulus, inv);
        e = amontred(&te, modulus, inv);
    }
    let [_, u, v, q, r] = divstep59(two_delta, f[0], g[0]);
    let sign = ((f[0].wrapping_mul(u).wrapping_add(g[0].wrapping_mul(v))) as i64 >> 63) as u64;
    let [u, v, _, _, su, sv, _, _] = sign_mag(u, v, q, r);
    let t = de_row(&d, &e, u, v, su ^ sign, sv ^ sign);
    cond_sub(&amontred(&t, modulus, inv), modulus)
}
