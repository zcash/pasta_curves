//! The constant-time inversion: one Rust driver over the six inline blocks of a backend, which
//! `src/asm/aarch64.rs` provides. `lean/PastaCurves/Compositions.lean` mirrors the driver over a
//! record of the blocks.

use super::Limbs;
use super::aarch64::{amontred, cond_sub, divstep59, fg_row, sign_mag, uv_row};

/// The constant-time inversion in Montgomery form: for a canonical `x`, the
/// canonical `z` with `x · z ≡ 2^512 (mod p)`, and `0` for `x = 0`.
///
/// The composition of the blocks that `lean/PastaCurves/Inversion/Model.lean`'s
/// `montInvModel` specifies. From `(d, f, g, u, v) = (1, p, x, 0, v0)`, with
/// `v0 = 2^562 mod p`, nine rounds each run `divstep59` on the low words of
/// `f` and `g`, take the matrix's sign-magnitude form, update `f` and `g` by
/// its two rows, and combine `u` and `v` by the two rows, each combination
/// reduced by `amontred`. The invariant after round `i` is
/// `(f, g) ≡ x · 2^(5i - 562) · (u, v) (mod p)`. The tenth round computes only
/// `u`, with the sign of the new `f` (which is `±1`, `g` being `0`) folded into
/// the row's masks, and reduces strictly. That sign is the top bit of the low
/// word of `m00 · f + m01 · g`, since that sum is `2^59 · f'`.
#[inline]
pub(crate) fn invert(x: &Limbs, modulus: &Limbs, inv: u64, v0: &Limbs) -> Limbs {
    let mut d: u64 = 1;
    let mut f = [modulus[0], modulus[1], modulus[2], modulus[3], 0];
    let mut g = [x[0], x[1], x[2], x[3], 0];
    let mut u: Limbs = [0; 4];
    let mut v: Limbs = *v0;
    for _ in 0..9 {
        let [d2, m00, m01, m10, m11] = divstep59(d, f[0], g[0]);
        d = d2;
        let [m00, m01, m10, m11, s00, s01, s10, s11] = sign_mag(m00, m01, m10, m11);
        let f2 = fg_row(&f, &g, m00, m01, s00, s01);
        g = fg_row(&f, &g, m10, m11, s10, s11);
        f = f2;
        let tu = uv_row(&u, &v, m00, m01, s00, s01);
        let tv = uv_row(&u, &v, m10, m11, s10, s11);
        u = amontred(&tu, modulus, inv);
        v = amontred(&tv, modulus, inv);
    }
    let [_, m00, m01, m10, m11] = divstep59(d, f[0], g[0]);
    let sign = ((f[0].wrapping_mul(m00).wrapping_add(g[0].wrapping_mul(m01))) as i64 >> 63) as u64;
    let [m00, m01, _, _, s00, s01, _, _] = sign_mag(m00, m01, m10, m11);
    let t = uv_row(&u, &v, m00, m01, s00 ^ sign, s01 ^ sign);
    cond_sub(&amontred(&t, modulus, inv), modulus)
}
