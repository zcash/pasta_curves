//! The constant-time inversion: one Rust driver over the six blocks of a backend, which implements
//! [`InvertBlocks`]. `lean/PastaCurves/Compositions.lean` mirrors the driver over a record of the
//! blocks, and `lean/PastaCurves/Inversion/Composition.lean` proves it equal to the algorithm's
//! model for any blocks that meet the record of their specifications.
//!
//! [`invert`] runs the AArch64 assembly blocks where the AArch64 backend is compiled, and the
//! same blocks in portable Rust (`portable`) everywhere else.

// The inversion is not yet reached from the field types.
#![allow(dead_code)]

use crate::limbs::Limbs;

if_asm_supported! {
    // The portable blocks serve wherever the assembly blocks do not; beside the AArch64 backend
    // they are compiled for their tests and the documentation only.
    #[cfg(any(not(target_arch = "aarch64"), test, doc))]
    mod portable;
}
if_asm_unsupported! {
    mod portable;
}

/// The six blocks of the inversion, as a backend provides them. The trait mirrors the Lean
/// record `InvertBlocks`, and each method's contract restates in words the block specification
/// that `InvertBlocks.Spec` requires of it. A backend that meets the contracts gets the driver's
/// correctness from the shared proof.
pub(crate) trait InvertBlocks {
    /// Fifty-nine half-delta divsteps on the low words of `f` and `g`. `two_delta = 2δ`, as a
    /// two's-complement word, and `f0` and `g0` are the low 64 bits of `f` and `g`, `f` odd.
    /// Returns `[two_delta', u, v, q, r]`: the new `two_delta`, and the 59-step transition matrix
    /// `M` with `2^59 (f', g') = M (f, g)`, as two's-complement words.
    fn divstep59(two_delta: u64, f0: u64, g0: u64) -> [u64; 5];

    /// The sign-magnitude form of a transition matrix whose entries are below `2^63` in magnitude:
    /// `[u, v, q, r, su, sv, sq, sr]`, each entry's magnitude, then its sign as a mask,
    /// all ones for a negative entry and zero otherwise.
    fn sign_mag(u: u64, v: u64, q: u64, r: u64) -> [u64; 8];

    /// The row `(a f + b g) / 2^59` of the `f`, `g` update, rounded down, in five two's-complement
    /// words. `f` and `g` are five-word values below `2^256` in magnitude, and `a` and `b` are
    /// given as the magnitudes `m0`, `m1` with the sign masks `s0`, `s1`, with `|a| + |b| ≤ 2^63`.
    fn fg_row(f: &[u64; 5], g: &[u64; 5], m0: u64, m1: u64, s0: u64, s1: u64) -> [u64; 5];

    /// The row `a d + b e` of the `d`, `e` combination, exact, in five two's-complement words. `d`
    /// and `e` are four-word values, and `a` and `b` are given as for [`fg_row`](Self::fg_row).
    fn de_row(d: &Limbs, e: &Limbs, m0: u64, m1: u64, s0: u64, s1: u64) -> [u64; 5];

    /// The almost-Montgomery reduction of a five-word value `t` with `|t| < 2^315`: adds `2^61 p`,
    /// then performs one word of Montgomery reduction. The result is below `2p` and below `2^256`,
    /// and congruent to `t / 2^64` modulo `p`.
    fn amontred(t: &[u64; 5], modulus: &Limbs, inv: u64) -> Limbs;

    /// Subtracts the modulus from `value` unless the subtraction borrows: `value mod p` for
    /// `value < 2p`.
    fn cond_sub(value: &Limbs, modulus: &Limbs) -> Limbs;
}

/// The constant-time inversion in Montgomery form over the blocks `B`: for a canonical `x`, the
/// canonical `z` with `x · z ≡ 2^512 (mod p)`, and `0` for `x = 0`.
///
/// The composition of the blocks that `lean/PastaCurves/Inversion/Model.lean`'s `montInvModel`
/// specifies. From `(two_delta, f, g, d, e) = (1, p, x, 0, e0)`, with `e0 = 2^562 mod p`, nine rounds each
/// run `divstep59` on the low words of `f` and `g`, take the matrix's sign-magnitude form, update
/// `f` and `g` by its two rows, and combine `d` and `e` by the two rows, each combination reduced
/// by `amontred`. The invariant after round `i` is `(f, g) ≡ x · 2^(5i - 562) · (d, e) (mod p)`.
/// The tenth round computes only `d`, with the sign of the new `f` (which is `±1`, `g` being `0`)
/// folded into the row's masks, and reduces strictly. That sign is the top bit of the low word of
/// `u · f + v · g`, since that sum is `2^59 · f'`.
#[inline]
pub(crate) fn invert_with<B: InvertBlocks>(
    x: &Limbs,
    modulus: &Limbs,
    inv: u64,
    e0: &Limbs,
) -> Limbs {
    let mut two_delta: u64 = 1;
    let mut f = [modulus[0], modulus[1], modulus[2], modulus[3], 0];
    let mut g = [x[0], x[1], x[2], x[3], 0];
    let mut d: Limbs = [0; 4];
    let mut e: Limbs = *e0;
    for _ in 0..9 {
        let [two_delta_new, u, v, q, r] = B::divstep59(two_delta, f[0], g[0]);
        two_delta = two_delta_new;
        let [u, v, q, r, su, sv, sq, sr] = B::sign_mag(u, v, q, r);
        let f_new = B::fg_row(&f, &g, u, v, su, sv);
        g = B::fg_row(&f, &g, q, r, sq, sr);
        f = f_new;
        let td = B::de_row(&d, &e, u, v, su, sv);
        let te = B::de_row(&d, &e, q, r, sq, sr);
        d = B::amontred(&td, modulus, inv);
        e = B::amontred(&te, modulus, inv);
    }
    let [_, u, v, q, r] = B::divstep59(two_delta, f[0], g[0]);
    let sign = ((f[0].wrapping_mul(u).wrapping_add(g[0].wrapping_mul(v))) as i64 >> 63) as u64;
    let [u, v, _, _, su, sv, _, _] = B::sign_mag(u, v, q, r);
    let t = B::de_row(&d, &e, u, v, su ^ sign, sv ^ sign);
    B::cond_sub(&B::amontred(&t, modulus, inv), modulus)
}

if_asm_supported! {
    /// The blocks that [`invert`] runs: the AArch64 assembly blocks where the AArch64 backend is
    /// compiled, and the portable blocks elsewhere.
    #[cfg(target_arch = "aarch64")]
    type Selected = crate::asm::aarch64::Backend;
    /// The blocks that [`invert`] runs: the AArch64 assembly blocks where the AArch64 backend is
    /// compiled, and the portable blocks elsewhere.
    #[cfg(not(target_arch = "aarch64"))]
    type Selected = portable::Backend;
}
if_asm_unsupported! {
    /// The blocks that [`invert`] runs: the AArch64 assembly blocks where the AArch64 backend is
    /// compiled, and the portable blocks elsewhere.
    type Selected = portable::Backend;
}

/// Inverts a canonical Montgomery residue for a Pasta modulus, in constant time.
///
/// Returns the canonical `z` with `x * z ≡ 2^512 (mod p)`: for `x` the Montgomery form of a nonzero
/// residue `X`, `z` is the Montgomery form of `X^-1`; for `x = 0` it is `0`, so a caller that needs
/// an optional inverse checks for zero separately. The algorithm is the serial variant of
/// Bernstein, Chen, Harrison, Huang, Maxwell, Wang, Wuille, and Yang, "Accelerating and verifying
/// constant-time modular inversion" (EUROCRYPT 2026), as in s2n-bignum's `bignum_montinv_p256`: 590
/// half-delta divsteps in ten rounds of 59, computed on packed words, with the coefficients reduced
/// by one Montgomery word per round. It runs a fixed sequence of six blocks, in register-only
/// assembly on AArch64 and in portable Rust on every other target (and on AArch64 with the assembly
/// disabled), with no data-dependent branch or memory access in either, so its timing does not
/// depend on `x`. The design and the correctness argument are in `book/src/design/inversion.md`.
///
/// Outputs are canonical.
///
/// # Safety
///
/// `x` must be canonical. This is debug-asserted. Under that precondition the machine-checked
/// proofs in `lean/` establish the result, from `montInv_spec` on words and the six block proofs.
/// For the assembly blocks the theorem is `AArch64.invert_entry_spec`; for the portable blocks it
/// is `Portable.invert_entry_spec`, about Aeneas' translation of `invert_with` over them. Both sets
/// of blocks are also tested against the same known answers.
///
/// `modulus` must be either the Pallas or Vesta field modulus, `inv` must be correctly derived
/// from it, and `e0` must be `2^562 mod p`, the starting value of the coefficient `e`, which
/// compensates the ten one-word Montgomery reductions (`2^562 = 2^(512 + 5 * 10)`). Any other
/// values will cause undefined results.
#[inline]
pub(crate) fn invert(x: &Limbs, modulus: &Limbs, inv: u64, e0: &Limbs) -> Limbs {
    debug_assert!(
        crate::limbs::is_canonical(x, modulus),
        "pasta_curves::inversion::invert requires a canonical input"
    );
    invert_with::<Selected>(x, modulus, inv, e0)
}

/// Known answers for every block and for the inversion, and the checks that run them over any
/// backend's blocks; a backend's own test module calls the checks with its blocks.
#[cfg(test)]
pub(crate) mod tests {
    use super::{InvertBlocks, invert_with};
    use crate::limbs::Limbs;
    use crate::test_fields::{FIELDS, Field, ZERO, p_minus_1, sub_limbs};

    /// The four entries of a transition matrix as words, then the expected magnitudes and masks,
    /// from the round model on both fields.
    pub(crate) const SIGN_MAG_VECTORS: [[u64; 12]; 6] = [
        [
            0xffce000000000000,
            0x004a000000000000,
            0xffffffffffffffe5,
            0xffffffffffffffff,
            0x0032000000000000,
            0x004a000000000000,
            0x000000000000001b,
            0x0000000000000001,
            0xffffffffffffffff,
            0x0000000000000000,
            0xffffffffffffffff,
            0xffffffffffffffff,
        ],
        [
            0x0000000000000000,
            0x0000008000000000,
            0xfffffffffff00000,
            0x00000058b6db6db7,
            0x0000000000000000,
            0x0000008000000000,
            0x0000000000100000,
            0x00000058b6db6db7,
            0x0000000000000000,
            0x0000000000000000,
            0xffffffffffffffff,
            0x0000000000000000,
        ],
        [
            0x0800000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000001,
            0x0800000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000001,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ],
        [
            0xffffffffc5706190,
            0x000000004e1e1b64,
            0xfffffffffe7f182c,
            0xffffffffdf08984b,
            0x000000003a8f9e70,
            0x000000004e1e1b64,
            0x000000000180e7d4,
            0x0000000020f767b5,
            0xffffffffffffffff,
            0x0000000000000000,
            0xffffffffffffffff,
            0xffffffffffffffff,
        ],
        [
            0xffffffffb32e679c,
            0xffffffffb4ff6206,
            0xffffffffe38509c2,
            0xffffffffc9886ce5,
            0x000000004cd19864,
            0x000000004b009dfa,
            0x000000001c7af63e,
            0x000000003677931b,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
        ],
        [
            0x8000000000000001,
            0x7fffffffffffffff,
            0x0000000000000000,
            0xffffffffffffffff,
            0x7fffffffffffffff,
            0x7fffffffffffffff,
            0x0000000000000000,
            0x0000000000000001,
            0xffffffffffffffff,
            0x0000000000000000,
            0x0000000000000000,
            0xffffffffffffffff,
        ],
    ];

    /// `(two_delta, f0, g0)` and the expected `(two_delta_new, u, v, q, r)`, from the integer divstep
    /// recurrence run on the words as integers, which the block agrees with by locality.
    pub(crate) const DIVSTEP59_VECTORS: [[u64; 8]; 6] = [
        [
            0x0000000000000001,
            0x992d30ed00000001,
            0x2a5f8c1b7e3d9046,
            0x0000000000000001,
            0xffffffffd94098a0,
            0x000000001c166090,
            0xffffffffc77ed7e2,
            0xfffffffff41aad25,
        ],
        [
            0x0000000000000001,
            0xffffffffffffffff,
            0x8000000000000001,
            0x0000000000000073,
            0xfc00000000000000,
            0x0400000000000000,
            0xffffffffffffffff,
            0xffffffffffffffff,
        ],
        [
            0xfffffffffffffffb,
            0x1234567890abcdef,
            0xfedcba0987654321,
            0x000000000000000d,
            0xfffffffee6d31a00,
            0x000000014cbb7a00,
            0xfffffffff5435e89,
            0x00000000056bfef9,
        ],
        [
            0x0000000000000011,
            0x0000000000000001,
            0x0000000000000000,
            0x0000000000000087,
            0x0800000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000001,
        ],
        [
            0xfffffffffffffb65,
            0xdeadbeefcafef00d,
            0x0123456789abcdef,
            0xfffffffffffffbdb,
            0x0800000000000000,
            0x0000000000000000,
            0x025c39e1b6d34515,
            0x0000000000000001,
        ],
        [
            0x0000000000000007,
            0xc8f1e2d3b4a59687,
            0x1e2d3c4b5a697887,
            0x0000000000000007,
            0xffffffffb89e1e30,
            0xfffffffe50f081d0,
            0xffffffffff43c3c5,
            0xffffffffdede823b,
        ],
    ];

    /// `f`, `g` (five words each), a row's magnitudes and masks, then the expected row of the
    /// update, from the round model on both fields.
    pub(crate) const FG_ROW_VECTORS: [[u64; 19]; 12] = [
        [
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0xd83bd700ffffffe5,
            0x628ddd6b04e1ba16,
            0xfffffffffffffffc,
            0x3fffffffffffffff,
            0x0000000000000000,
            0x0032000000000000,
            0x004a000000000000,
            0xffffffffffffffff,
            0x0000000000000000,
            0x66d2cf12ffffffff,
            0xddb96703f6b306e4,
            0xffffffffffffffff,
            0x00bfffffffffffff,
            0x0000000000000000,
        ],
        [
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0xd83bd700ffffffe5,
            0x628ddd6b04e1ba16,
            0xfffffffffffffffc,
            0x3fffffffffffffff,
            0x0000000000000000,
            0x000000000000001b,
            0x0000000000000001,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0xffffffffffffff20,
            0xffffffffffffffff,
        ],
        [
            0x66d2cf12ffffffff,
            0xddb96703f6b306e4,
            0xffffffffffffffff,
            0x00bfffffffffffff,
            0x0000000000000000,
            0xffffffffff900000,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0x0000000000000000,
            0x0000008000000000,
            0x0000000000000000,
            0x0000000000000000,
            0xfffffffffffffff9,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
        ],
        [
            0x66d2cf12ffffffff,
            0xddb96703f6b306e4,
            0xffffffffffffffff,
            0x00bfffffffffffff,
            0x0000000000000000,
            0xffffffffff900000,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0x0000000000100000,
            0x00000058b6db6db7,
            0xffffffffffffffff,
            0x0000000000000000,
            0xf81299f237325a5d,
            0x0000000000448d31,
            0x0000000000000000,
            0xfffffffffffe8000,
            0xffffffffffffffff,
        ],
        [
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0800000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0000000000000000,
        ],
        [
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000001,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ],
        [
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0800000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0000000000000000,
        ],
        [
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000001,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ],
        [
            0xdb241905fa248697,
            0xfffffffffffffff9,
            0x872517d55c1331ff,
            0xfffffffffffffff4,
            0xffffffffffffffff,
            0x599ff9f880e4b2a4,
            0xfffffffffffffffe,
            0xeb5679967b925eff,
            0xfffffffffffffffc,
            0xffffffffffffffff,
            0x000000003a8f9e70,
            0x000000004e1e1b64,
            0xffffffffffffffff,
            0x0000000000000000,
            0x0000001cdd290a4d,
            0x3d11b618fe078000,
            0x00000035e519a92f,
            0x0000000000000000,
            0x0000000000000000,
        ],
        [
            0xdb241905fa248697,
            0xfffffffffffffff9,
            0x872517d55c1331ff,
            0xfffffffffffffff4,
            0xffffffffffffffff,
            0x599ff9f880e4b2a4,
            0xfffffffffffffffe,
            0xeb5679967b925eff,
            0xfffffffffffffffc,
            0xffffffffffffffff,
            0x000000000180e7d4,
            0x0000000020f767b5,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0x00000007f42196ae,
            0xc132e71048cda000,
            0x0000000ed9e178b1,
            0x0000000000000000,
            0x0000000000000000,
        ],
        [
            0xe71681d6d322b145,
            0x000000000000577d,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x74b8fd762c36d47e,
            0x00000000000014ae,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x000000004cd19864,
            0x000000004b009dfa,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xfffbf5fa922aef71,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
        ],
        [
            0xe71681d6d322b145,
            0x000000000000577d,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x74b8fd762c36d47e,
            0x00000000000014ae,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x000000001c7af63e,
            0x000000003677931b,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xfffe3bb7def33478,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
        ],
    ];

    /// `d`, `e` (four words each), a row's magnitudes and masks, then the expected five-word row
    /// combination, from the round model on both fields; the last rows are the final round's, with
    /// the sign of `f` folded into the masks.
    pub(crate) const DE_ROW_VECTORS: [[u64; 17]; 22] = [
        [
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x9a5f583ce5084635,
            0x4f417e233776c195,
            0x74634b1a733f7785,
            0x1c51de5ea66f0f25,
            0x0032000000000000,
            0x004a000000000000,
            0xffffffffffffffff,
            0x0000000000000000,
            0x4b52000000000000,
            0xf53e9f8f819a3464,
            0x8c88e8ee762e0853,
            0x60d3a4b3b5a55058,
            0x00082faa475c1c1a,
        ],
        [
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x9a5f583ce5084635,
            0x4f417e233776c195,
            0x74634b1a733f7785,
            0x1c51de5ea66f0f25,
            0x000000000000001b,
            0x0000000000000001,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0x65a0a7c31af7b9cb,
            0xb0be81dcc8893e6a,
            0x8b9cb4e58cc0887a,
            0xe3ae21a15990f0da,
            0xffffffffffffffff,
        ],
        [
            0xc6c552cd2258cd61,
            0xb4f028949f9dad38,
            0x3834c1a749676b4a,
            0x25cdda675f548eb8,
            0x993d30ed00000001,
            0xb4002bcf181cf91b,
            0x000224698fc094cf,
            0x4000000000000000,
            0x0000000000000000,
            0x0000008000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000008000000000,
            0x0e7c8dcc9e987680,
            0xe04a67da0015e78c,
            0x00000000011234c7,
            0x0000002000000000,
        ],
        [
            0xc6c552cd2258cd61,
            0xb4f028949f9dad38,
            0x3834c1a749676b4a,
            0x25cdda675f548eb8,
            0x993d30ed00000001,
            0xb4002bcf181cf91b,
            0x000224698fc094cf,
            0x4000000000000000,
            0x0000000000100000,
            0x00000058b6db6db7,
            0xffffffffffffffff,
            0x0000000000000000,
            0xc47fbd36e0cb6db7,
            0x34c05fd325d0d03d,
            0x6d486e24e511ab96,
            0x198a0ab7153a88b6,
            0x000000162db47e90,
        ],
        [
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x9a5f583ce5084635,
            0x4f417e233776c195,
            0x74634b1a733f7785,
            0x1c51de5ea66f0f25,
            0x0800000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ],
        [
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x9a5f583ce5084635,
            0x4f417e233776c195,
            0x74634b1a733f7785,
            0x1c51de5ea66f0f25,
            0x0000000000000000,
            0x0000000000000001,
            0x0000000000000000,
            0x0000000000000000,
            0x9a5f583ce5084635,
            0x4f417e233776c195,
            0x74634b1a733f7785,
            0x1c51de5ea66f0f25,
            0x0000000000000000,
        ],
        [
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x993130ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0800000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0800000000000000,
            0xdcc9698768000000,
            0x011234c7e04a67c8,
            0x0000000000000000,
            0x0200000000000000,
        ],
        [
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x993130ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0x0000000000000001,
            0x0000000000000000,
            0x0000000000000000,
            0x993130ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0000000000000000,
        ],
        [
            0x48cb952368f0cbda,
            0x63a7041d70d5fc08,
            0xc5673a829079038d,
            0x27df43973ee24a3f,
            0x6dcd9761281f03dc,
            0x04f780e32ffca047,
            0x664f5e8fa4f2dbe5,
            0x209079863e8aa1ad,
            0x000000003a8f9e70,
            0x000000004e1e1b64,
            0xffffffffffffffff,
            0x0000000000000000,
            0x1737410aa35dfa90,
            0xb8e66c17d71f7ca3,
            0xd8211d46aeb7b8ab,
            0xd6e9e255bb861dfb,
            0x0000000000d0e5bc,
        ],
        [
            0x48cb952368f0cbda,
            0x63a7041d70d5fc08,
            0xc5673a829079038d,
            0x27df43973ee24a3f,
            0x6dcd9761281f03dc,
            0x04f780e32ffca047,
            0x664f5e8fa4f2dbe5,
            0x209079863e8aa1ad,
            0x000000000180e7d4,
            0x0000000020f767b5,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xc0b26c8bf7e63aec,
            0xbfc45c1c4733ea1c,
            0x267edd41b4b9a6a1,
            0xf2dd9e86b1ca270f,
            0xfffffffffb928537,
        ],
        [
            0xe6271d6294fbc898,
            0x3e2f8c089c7ceef4,
            0x767a844ff1a42a71,
            0x460ec71de713dfb1,
            0xd641284a6ae5037b,
            0xc3b6ebf1abb53dea,
            0xee9e6be06bcf421d,
            0x09541103d50b15e9,
            0x000000004cd19864,
            0x000000004b009dfa,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0x9ee787ac8aab8f82,
            0x7c279d554451297d,
            0x4a202205c178e33b,
            0xa20ab2aa3e30dfa2,
            0xffffffffe83e9a60,
        ],
        [
            0xe6271d6294fbc898,
            0x3e2f8c089c7ceef4,
            0x767a844ff1a42a71,
            0x460ec71de713dfb1,
            0xd641284a6ae5037b,
            0xc3b6ebf1abb53dea,
            0xee9e6be06bcf421d,
            0x09541103d50b15e9,
            0x000000001c7af63e,
            0x000000003677931b,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0x72dda29c687f5c37,
            0x3d5ddd77e2361e16,
            0x27f0f1b1b7be7085,
            0xb23361418edf7439,
            0xfffffffff638a4c3,
        ],
        [
            0x493586b8db6db6ca,
            0x860d0369a70b9cb1,
            0x6db6db6db6db6db4,
            0x36db6db6db6db6db,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0800000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x5000000000000000,
            0x8a49ac35c6db6db6,
            0xa430681b4d385ce5,
            0xdb6db6db6db6db6d,
            0x01b6db6db6db6db6,
        ],
        [
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0xb4becc383c4c0001,
            0x22460fe1a55cd3e7,
            0x0000000000000000,
            0x3fff000000000000,
            0x0800000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0800000000000000,
            0xdcc9698768000000,
            0x011234c7e04a67c8,
            0x0000000000000000,
            0x0200000000000000,
        ],
        [
            0x6ec45e40fffffe25,
            0xb0ff453accc4c098,
            0x0d0acc8786d45ce5,
            0x1257ca108c691d71,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0800000000000000,
            0x0000000000000000,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xd800000000000000,
            0x3c89dd0df800000e,
            0xd27805d62999d9fb,
            0x7797a99bc3c95d18,
            0xff6d41af7b9cb714,
        ],
        [
            0x2a68d2ac000001dc,
            0x714753c13c883883,
            0xf2f53378792ba31a,
            0x2da835ef7396e28e,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0800000000000000,
            0x0000000000000000,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0x2000000000000000,
            0xe6acb96a9ffffff1,
            0x2c75c561f61bbe3b,
            0x886856643c36a2e7,
            0xfe92be50846348eb,
        ],
        [
            0x48434e89d6a56baa,
            0x2089cebea655c8a7,
            0x8391299eb98be0ad,
            0x20b8eb4c2df91a89,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0800000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x5000000000000000,
            0x3a421a744eb52b5d,
            0x69044e75f532ae45,
            0x4c1c894cf5cc5f05,
            0x0105c75a616fc8d4,
        ],
        [
            0x2a076bc0db6db6ca,
            0x860d0369a22a37c6,
            0x6db6db6db6db6db4,
            0x36db6db6db6db6db,
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
            0x0800000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x5000000000000000,
            0x31503b5e06db6db6,
            0xa430681b4d1151be,
            0xdb6db6db6db6db6d,
            0x01b6db6db6db6db6,
        ],
        [
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
            0xe8d0ba05537c0001,
            0x22460fe1a5a4828a,
            0x0000000000000000,
            0x3fff000000000000,
            0x0800000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0800000000000000,
            0xec62375908000000,
            0x011234c7e04ca546,
            0x0000000000000000,
            0x0200000000000000,
        ],
        [
            0x61b3735c000001dc,
            0x6e4e03c0fcf03909,
            0xf5c46200999eb20c,
            0x2da835ef99fb552f,
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
            0x0800000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0xe000000000000000,
            0x4b0d9b9ae000000e,
            0x6372701e07e781c8,
            0x7fae231004ccf590,
            0x016d41af7ccfdaa9,
        ],
        [
            0x2a9377c4fffffe25,
            0xb3f8953b0ca46fd4,
            0x0a3b9dff66614df3,
            0x1257ca106604aad0,
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
            0x0800000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x2800000000000000,
            0xa1549bbe27fffff1,
            0x9d9fc4a9d865237e,
            0x8051dceffb330a6f,
            0x0092be5083302556,
        ],
        [
            0xca612b52ba19e957,
            0x8954aaeb1967c172,
            0xac4ce37d081b0c2b,
            0x0a79ccdfa3a30d0e,
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
            0x0800000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0xb800000000000000,
            0x9653095a95d0cf4a,
            0x5c4aa55758cb3e0b,
            0x7562671be840d861,
            0x0053ce66fd1d1868,
        ],
    ];

    /// A five-word row combination, the modulus, `inv`, then the expected reduction, from the round
    /// model on both fields.
    pub(crate) const AMONTRED_VECTORS: [[u64; 14]; 22] = [
        [
            0x4b52000000000000,
            0xf53e9f8f819a3464,
            0x8c88e8ee762e0853,
            0x60d3a4b3b5a55058,
            0x00082faa475c1c1a,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0xadb482ad66b03465,
            0xa4b9d87ba80679cc,
            0x60d3a4b3b5a55058,
            0x2d33afaa475c1c1a,
        ],
        [
            0x65a0a7c31af7b9cb,
            0xb0be81dcc8893e6a,
            0x8b9cb4e58cc0887a,
            0xe3ae21a15990f0da,
            0xffffffffffffffff,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0x7806e5a0c4c00001,
            0xadeb423bad68faee,
            0x23ae21a15990f0da,
            0x400eda4af942118d,
        ],
        [
            0x0000008000000000,
            0x0e7c8dcc9e987680,
            0xe04a67da0015e78c,
            0x00000000011234c7,
            0x0000002000000000,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0x012d30ed08000001,
            0x029100c4e61662a3,
            0x00000000011234c8,
            0x4000000000000000,
        ],
        [
            0xc47fbd36e0cb6db7,
            0x34c05fd325d0d03d,
            0x6d486e24e511ab96,
            0x198a0ab7153a88b6,
            0x000000162db47e90,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0xcd5f403284675db7,
            0x722dece6c4bee381,
            0x598a0ab7153a88b6,
            0x092489633581a322,
        ],
        [
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
        ],
        [
            0x9a5f583ce5084635,
            0x4f417e233776c195,
            0x74634b1a733f7785,
            0x1c51de5ea66f0f25,
            0x0000000000000000,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0xba537c393b400001,
            0x96a1efbc6530f748,
            0xdc51de5ea66f0f25,
            0x3ff125b506bdee72,
        ],
        [
            0x0800000000000000,
            0xdcc9698768000000,
            0x011234c7e04a67c8,
            0x0000000000000000,
            0x0200000000000000,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
        ],
        [
            0x993130ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0xb4becc383c4c0001,
            0x22460fe1a55cd3e7,
            0x0000000000000000,
            0x3fff000000000000,
        ],
        [
            0x1737410aa35dfa90,
            0xb8e66c17d71f7ca3,
            0xd8211d46aeb7b8ab,
            0xd6e9e255bb861dfb,
            0x0000000000d0e5bc,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0x6e3d45a481a7dcc4,
            0xf643efa7a0c6948d,
            0xd6e9e255bb861dfb,
            0x38452d9157f96718,
        ],
        [
            0xc0b26c8bf7e63aec,
            0xbfc45c1c4733ea1c,
            0x267edd41b4b9a6a1,
            0xf2dd9e86b1ca270f,
            0xfffffffffb928537,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0xdaef3d96915ccd46,
            0x3178b9732fd6b26a,
            0xf2dd9e86b1ca270f,
            0x147e97fbfd98f67c,
        ],
        [
            0x9ee787ac8aab8f82,
            0x7c279d554451297d,
            0x4a202205c178e33b,
            0xa20ab2aa3e30dfa2,
            0xffffffffe83e9a60,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0x55fb4a4f35158971,
            0x67231724fb0356b4,
            0x220ab2aa3e30dfa2,
            0x362baceb4593b680,
        ],
        [
            0x72dda29c687f5c37,
            0x3d5ddd77e2361e16,
            0x27f0f1b1b7be7085,
            0xb23361418edf7439,
            0xfffffffff638a4c3,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0xb4bb2db5fe7889f4,
            0x30a4e02f8b124676,
            0xf23361418edf7439,
            0x104003139c18cdb5,
        ],
        [
            0x5000000000000000,
            0x8a49ac35c6db6db6,
            0xa430681b4d385ce5,
            0xdb6db6db6db6db6d,
            0x01b6db6db6db6db6,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0x8398bdd8b6db6db7,
            0xbbc0f148939d4828,
            0xdb6db6db6db6db6d,
            0x2db6db6db6db6db6,
        ],
        [
            0x0800000000000000,
            0xdcc9698768000000,
            0x011234c7e04a67c8,
            0x0000000000000000,
            0x0200000000000000,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
        ],
        [
            0xd800000000000000,
            0x3c89dd0df800000e,
            0xd27805d62999d9fb,
            0x7797a99bc3c95d18,
            0xff6d41af7b9cb714,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0x8c78ecb30000000f,
            0xd7d30dbd8b0de0e7,
            0x7797a99bc3c95d18,
            0x096d41af7b9cb714,
        ],
        [
            0x2000000000000000,
            0xe6acb96a9ffffff1,
            0x2c75c561f61bbe3b,
            0x886856643c36a2e7,
            0xfe92be50846348eb,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0x0cb44439fffffff2,
            0x4a738b3e7e3f1834,
            0x886856643c36a2e7,
            0x3692be50846348eb,
        ],
        [
            0x5000000000000000,
            0x3a421a744eb52b5d,
            0x69044e75f532ae45,
            0x4c1c894cf5cc5f05,
            0x0105c75a616fc8d4,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ecffffffff,
            0x33912c173eb52b5e,
            0x8094d7a33b979988,
            0x4c1c894cf5cc5f05,
            0x2d05c75a616fc8d4,
        ],
        [
            0x5000000000000000,
            0x31503b5e06db6db6,
            0xa430681b4d1151be,
            0xdb6db6db6db6db6d,
            0x01b6db6db6db6db6,
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
            0x8c46eb20ffffffff,
            0x81c0fd04b6db6db7,
            0xbbc0f14893a785d6,
            0xdb6db6db6db6db6d,
            0x2db6db6db6db6db6,
        ],
        [
            0x0800000000000000,
            0xec62375908000000,
            0x011234c7e04ca546,
            0x0000000000000000,
            0x0200000000000000,
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
            0x8c46eb20ffffffff,
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
        ],
        [
            0xe000000000000000,
            0x4b0d9b9ae000000e,
            0x6372701e07e781c8,
            0x7fae231004ccf590,
            0x016d41af7ccfdaa9,
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
            0x8c46eb20ffffffff,
            0xfc9678ff0000000f,
            0x67bb433d891a16e3,
            0x7fae231004ccf590,
            0x096d41af7ccfdaa9,
        ],
        [
            0x2800000000000000,
            0xa1549bbe27fffff1,
            0x9d9fc4a9d865237e,
            0x8051dceffb330a6f,
            0x0092be5083302556,
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
            0x8c46eb20ffffffff,
            0x8fb07221fffffff2,
            0xba8b55be807a91f9,
            0x8051dceffb330a6f,
            0x3692be5083302556,
        ],
        [
            0xb800000000000000,
            0x9653095a95d0cf4a,
            0x5c4aa55758cb3e0b,
            0x7562671be840d861,
            0x0053ce66fd1d1868,
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
            0x8c46eb20ffffffff,
            0xe5c6fb7bddd0cf4b,
            0x65ee805e3b7d0d89,
            0x7562671be840d861,
            0x1253ce66fd1d1868,
        ],
    ];

    /// A reduction below `2p`, the modulus, then the expected canonical value, from the final
    /// rounds of the round model on both fields.
    pub(crate) const COND_SUB_VECTORS: [[u64; 12]; 10] = [
        [
            0x8398bdd8b6db6db7,
            0xbbc0f148939d4828,
            0xdb6db6db6db6db6d,
            0x2db6db6db6db6db6,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x8398bdd8b6db6db7,
            0xbbc0f148939d4828,
            0xdb6db6db6db6db6d,
            0x2db6db6db6db6db6,
        ],
        [
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ],
        [
            0x8c78ecb30000000f,
            0xd7d30dbd8b0de0e7,
            0x7797a99bc3c95d18,
            0x096d41af7b9cb714,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x8c78ecb30000000f,
            0xd7d30dbd8b0de0e7,
            0x7797a99bc3c95d18,
            0x096d41af7b9cb714,
        ],
        [
            0x0cb44439fffffff2,
            0x4a738b3e7e3f1834,
            0x886856643c36a2e7,
            0x3692be50846348eb,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x0cb44439fffffff2,
            0x4a738b3e7e3f1834,
            0x886856643c36a2e7,
            0x3692be50846348eb,
        ],
        [
            0x33912c173eb52b5e,
            0x8094d7a33b979988,
            0x4c1c894cf5cc5f05,
            0x2d05c75a616fc8d4,
            0x992d30ed00000001,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
            0x33912c173eb52b5e,
            0x8094d7a33b979988,
            0x4c1c894cf5cc5f05,
            0x2d05c75a616fc8d4,
        ],
        [
            0x81c0fd04b6db6db7,
            0xbbc0f14893a785d6,
            0xdb6db6db6db6db6d,
            0x2db6db6db6db6db6,
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
            0x81c0fd04b6db6db7,
            0xbbc0f14893a785d6,
            0xdb6db6db6db6db6d,
            0x2db6db6db6db6db6,
        ],
        [
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ],
        [
            0xfc9678ff0000000f,
            0x67bb433d891a16e3,
            0x7fae231004ccf590,
            0x096d41af7ccfdaa9,
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
            0xfc9678ff0000000f,
            0x67bb433d891a16e3,
            0x7fae231004ccf590,
            0x096d41af7ccfdaa9,
        ],
        [
            0x8fb07221fffffff2,
            0xba8b55be807a91f9,
            0x8051dceffb330a6f,
            0x3692be5083302556,
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
            0x8fb07221fffffff2,
            0xba8b55be807a91f9,
            0x8051dceffb330a6f,
            0x3692be5083302556,
        ],
        [
            0xe5c6fb7bddd0cf4b,
            0x65ee805e3b7d0d89,
            0x7562671be840d861,
            0x1253ce66fd1d1868,
            0x8c46eb2100000001,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
            0xe5c6fb7bddd0cf4b,
            0x65ee805e3b7d0d89,
            0x7562671be840d861,
            0x1253ce66fd1d1868,
        ],
    ];

    /// `sign_mag` against its known answers.
    pub(crate) fn sign_mag_known_answers<B: InvertBlocks>() {
        for row in SIGN_MAG_VECTORS {
            let [u, v, q, r] = [row[0], row[1], row[2], row[3]];
            assert_eq!(B::sign_mag(u, v, q, r), row[4..12]);
        }
    }

    /// `divstep59` against its known answers.
    pub(crate) fn divstep59_known_answers<B: InvertBlocks>() {
        for [two_delta, f0, g0, two_delta_new, u, v, q, r] in DIVSTEP59_VECTORS {
            assert_eq!(B::divstep59(two_delta, f0, g0), [two_delta_new, u, v, q, r]);
        }
    }

    /// `fg_row` against its known answers.
    pub(crate) fn fg_row_known_answers<B: InvertBlocks>() {
        for row in FG_ROW_VECTORS {
            let f: [u64; 5] = row[0..5].try_into().expect("five words");
            let g: [u64; 5] = row[5..10].try_into().expect("five words");
            assert_eq!(
                B::fg_row(&f, &g, row[10], row[11], row[12], row[13]),
                row[14..19]
            );
        }
    }

    /// `de_row` against its known answers.
    pub(crate) fn de_row_known_answers<B: InvertBlocks>() {
        for row in DE_ROW_VECTORS {
            let d: Limbs = row[0..4].try_into().expect("four words");
            let e: Limbs = row[4..8].try_into().expect("four words");
            assert_eq!(
                B::de_row(&d, &e, row[8], row[9], row[10], row[11]),
                row[12..17]
            );
        }
    }

    /// `amontred` against its known answers.
    pub(crate) fn amontred_known_answers<B: InvertBlocks>() {
        for row in AMONTRED_VECTORS {
            let t: [u64; 5] = row[0..5].try_into().expect("five words");
            let modulus: Limbs = row[5..9].try_into().expect("four words");
            assert_eq!(B::amontred(&t, &modulus, row[9]), row[10..14]);
        }
    }

    /// `cond_sub` against its known answers.
    pub(crate) fn cond_sub_known_answers<B: InvertBlocks>() {
        for row in COND_SUB_VECTORS {
            let x: Limbs = row[0..4].try_into().expect("four words");
            let modulus: Limbs = row[4..8].try_into().expect("four words");
            assert_eq!(B::cond_sub(&x, &modulus), row[8..12]);
        }
    }

    /// `z = invert(x)` is canonical and is the Montgomery inverse of a nonzero `x`: the Montgomery
    /// product `x · z` by the field type's portable arithmetic is `R`, inverting `z` gives `x`
    /// back, and `z` is the inverse that the field type's portable arithmetic gives.
    fn check_inverse<B: InvertBlocks>(f: &Field, x: &Limbs) {
        let z = invert_with::<B>(x, &f.modulus, f.inv, &f.e0);
        // Canonical: below the modulus, comparing the limbs from the top.
        assert!(z.iter().rev().lt(f.modulus.iter().rev()), "{x:x?}");
        assert_eq!((f.portable_mul)(x, &z), f.r, "{x:x?}");
        assert_eq!(invert_with::<B>(&z, &f.modulus, f.inv, &f.e0), *x, "{x:x?}");
        assert_eq!(z, (f.portable_inverse)(x), "{x:x?}");
    }

    /// `invert` reproduces the integer model of the algorithm on the recorded inputs, which
    /// include `0`.
    pub(crate) fn invert_known_answers<B: InvertBlocks>() {
        for f in FIELDS {
            for (x, z) in &f.inversions {
                assert_eq!(invert_with::<B>(x, &f.modulus, f.inv, &f.e0), *z);
                if *x != ZERO {
                    check_inverse::<B>(f, x);
                }
            }
        }
    }

    /// The small values `1` to `256` and their negatives `p - 1` down to `p - 256`, the powers
    /// of two up to `2^253`, and `R`, `R^2`, and `R^3`.
    pub(crate) fn invert_small_and_near_modulus<B: InvertBlocks>() {
        for f in FIELDS {
            for k in 1..=256u64 {
                check_inverse::<B>(f, &[k, 0, 0, 0]);
                check_inverse::<B>(f, &sub_limbs(&f.modulus, &[k, 0, 0, 0]));
            }
            for k in 0..254 {
                let mut x = ZERO;
                x[k / 64] = 1 << (k % 64);
                check_inverse::<B>(f, &x);
            }
            for x in [f.r, f.r2, f.r3] {
                check_inverse::<B>(f, &x);
            }
        }
    }

    /// Random inputs: uniform values below `2^64`; uniform values below `2^254`; values between
    /// `2^254` and `p`, whose top limb is `2^62` and whose limb 1 is below the modulus's; and
    /// values within a random 64-bit distance below `p - 1`.
    pub(crate) fn invert_random<B: InvertBlocks>() {
        use rand::{Rng, SeedableRng};
        use rand_xorshift::XorShiftRng;

        let mut rng = XorShiftRng::from_seed([0x5a; 16]);
        let mut next = || rng.next_u64();
        for f in FIELDS {
            let pm1 = p_minus_1(f);
            for _ in 0..128 {
                let below_2_64 = [next(), 0, 0, 0];
                check_inverse::<B>(f, &below_2_64);
                let below_2_254 = [next(), next(), next(), next() >> 2];
                check_inverse::<B>(f, &below_2_254);
                let above_2_254 = [next(), next() % f.modulus[1], 0, 1 << 62];
                check_inverse::<B>(f, &above_2_254);
                let near_p = sub_limbs(&pm1, &[next(), 0, 0, 0]);
                check_inverse::<B>(f, &near_p);
            }
        }
    }

    /// The entry point `invert` runs the selected blocks, and in a debug build its assertion
    /// fires on a non-canonical input.
    #[test]
    fn invert_entry_point() {
        for f in FIELDS {
            for (x, z) in &f.inversions {
                assert_eq!(super::invert(x, &f.modulus, f.inv, &f.e0), *z);
            }
            #[cfg(all(debug_assertions, panic = "unwind"))]
            {
                let panic = std::panic::catch_unwind(|| {
                    super::invert(&f.modulus, &f.modulus, f.inv, &f.e0)
                })
                .expect_err("the debug assertion of invert's contract did not fire");
                let message = panic
                    .downcast_ref::<&str>()
                    .expect("the assertion's message is a string literal");
                assert!(message.contains("requires a canonical input"), "{message}");
            }
        }
    }
}
