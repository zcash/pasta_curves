//! Private Apple AArch64 backend for the Pasta fields.
//!
//! Montgomery multiplication and squaring are implemented as inline `asm!`
//! blocks below; the fused repeated-squaring chain and the canonical-form
//! conversion remain in `src/asm/pasta_mul-armv8.S` and are reached through
//! `extern "C"`.
//!
//! The inline blocks are register-renamed transcriptions of the upstream
//! Semolina v0.1.4 routines (`mul_mont_pasta`, and the squaring loop body of
//! the vendored `sqr_n_mul_mont_pasta`), with rhs limbs and the modulus
//! constants supplied in registers instead of loaded from memory. The
//! per-instruction comments are carried over from the removed assembly
//! routines. Because the operands are
//! ordinary register operands and the blocks are declared
//! `options(pure, nomem, nostack)`, LLVM inlines the wrappers into callers
//! and keeps field values in registers between operations — there is no
//! call, pointer, or ABI-clobber traffic per field operation.
//!
//! Like the assembly file, the arithmetic relies on the shared Pasta modulus
//! shape: `modulus[2] = 0` and `modulus[3] = 2^62` (materialized inline as an
//! immediate). Only `modulus[0]`, `modulus[1]`, and `inv` vary between Fp
//! and Fq, so a single implementation serves both fields.
//!
//! Canonicity contract (same as the assembly): `rhs` in `mul` and the input
//! of `square` must be canonical (below the modulus). `lhs` in `mul` may be
//! an unreduced 256-bit value **only if every `rhs` limb is at most
//! `2^64 - 4`**; with both operands canonical the routines are always safe.
//! Outputs are canonical.
//!
//! The `lhs` caveat exists because `mul` keeps a five-limb accumulator (one
//! word fewer than textbook CIOS). The only carry chain that can wrap is the
//! one folding in the high cross-products: its tail computes
//! `acc4 + high(lhs[3] * rhs_limb) + carry` with `acc4 <= 2`, which reaches
//! `2^64` only when `high(lhs[3] * rhs_limb) >= 2^64 - 3`, i.e. when
//! `lhs[3]` and some `rhs` limb are both within 3 of `2^64`. A canonical
//! `lhs` has `lhs[3] <= 2^62`, and any `rhs` limb `<= 2^64 - 4` caps the
//! high product at `2^64 - 5`, so either condition alone rules the wrap out.
//! No current caller passes an unreduced `lhs`: `from_u512`, the one place
//! that produces unreduced values, deliberately uses the portable
//! multiplication instead. The
//! `aarch64_asm_mul_unreduced_lhs_matches_portable` tests in `fp.rs` and
//! `fq.rs` still pin the unreduced-`lhs` behaviour against the `R2`/`R3`
//! constants in case a future caller relies on it.
//!
//! There are no branches and no memory accesses inside the blocks, so the
//! code is constant-time.

use core::arch::asm;

type Limbs = [u64; 4];

extern "C" {
    fn pasta_curves_sqr_n_mul_mont_pasta(
        out: *mut Limbs,
        value: *const Limbs,
        count: usize,
        rhs: *const Limbs,
        modulus: *const Limbs,
        inv: u64,
    );
    fn pasta_curves_from_mont_pasta(
        out: *mut Limbs,
        value: *const Limbs,
        modulus: *const Limbs,
        inv: u64,
    );
}

/// Multiplies two Montgomery residues for a Pasta modulus. `rhs` must be
/// canonical. `lhs` may be unreduced only if every `rhs` limb is at most
/// `2^64 - 4`; see the module docs for the carry-chain bound behind this.
#[inline(always)]
pub(super) fn mul(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    let (o0, o1, o2, o3): (u64, u64, u64, u64);
    // SAFETY: straight-line register-only arithmetic; no memory access, no
    // stack use, and outputs depend only on the declared inputs.
    unsafe {
        asm!(
            // Round 0: lhs * rhs[0], then cancel the low limb.
            // Form the five-limb product a * b[0] in r0..r4.
            "mul {r0}, {a0}, {b0}",             // r0 = low(a[0] * b[0]).
            "mul {r1}, {a1}, {b0}",             // r1 = low(a[1] * b[0]).
            "mul {r2}, {a2}, {b0}",             // r2 = low(a[2] * b[0]).
            "mul {r3}, {a3}, {b0}",             // r3 = low(a[3] * b[0]).

            "umulh {t0}, {a0}, {b0}",           // t0 = high(a[0] * b[0]).
            "umulh {t1}, {a1}, {b0}",           // t1 = high(a[1] * b[0]).
            "mul {q}, {inv}, {r0}",             // q = r0 * inv mod 2^64.
            "umulh {t2}, {a2}, {b0}",           // t2 = high(a[2] * b[0]).
            "umulh {t3}, {a3}, {b0}",           // t3 = high(a[3] * b[0]).
            "adds {r1}, {r1}, {t0}",            // Add high(a[0] * b[0]) into limb 1.
            // low(q * p[0]) cancels r0 and is discarded by the limb shift.
            "adcs {r2}, {r2}, {t1}",            // Add high(a[1] * b[0]) and carry.
            "mul {t1}, {p1}, {q}",              // t1 = low(q * p[1]).
            "adcs {r3}, {r3}, {t2}",            // Add high(a[2] * b[0]) and carry.
            // q * p[2] is zero because p[2] = 0.
            "adc {r4}, xzr, {t3}",              // Finish a * b[0] with its fifth limb.
            "lsl {t3}, {q}, #62",               // t3 = low(q * p[3]).
            // Carry from r0 + low(q*p[0]) is one exactly when r0 is nonzero.
            "subs xzr, {r0}, #1",               // Set that carry without computing the zero sum.
            "umulh {t0}, {p0}, {q}",            // t0 = high(q * p[0]).
            "adcs {r1}, {r1}, {t1}",            // Add low(q * p[1]) and cancellation carry.
            "umulh {t1}, {p1}, {q}",            // t1 = high(q * p[1]).
            "adcs {r2}, {r2}, xzr",             // Propagate carry; p[2]'s product is zero.
            // high(q * p[2]) is zero.
            "adcs {r3}, {r3}, {t3}",            // Add low(q * p[3]) and carry.
            "lsr {t3}, {q}, #2",                // t3 = high(q * p[3]).
            "adc {r4}, {r4}, xzr",              // Propagate carry into the fifth limb.

            // Drop the cancelled low limb: (a*b[0] + q*p) / 2^64.
            "adds {r0}, {r1}, {t0}",            // New limb 0 includes high(q * p[0]).
            "mul {t0}, {a0}, {b1}",             // t0 = low(a[0] * b[1]).
            "adcs {r1}, {r2}, {t1}",            // New limb 1 includes high(q * p[1]).
            "mul {t1}, {a1}, {b1}",             // t1 = low(a[1] * b[1]).
            "adcs {r2}, {r3}, xzr",             // New limb 2; p[2] contributes zero.
            "mul {t2}, {a2}, {b1}",             // t2 = low(a[2] * b[1]).
            "adcs {r3}, {r4}, {t3}",            // New limb 3 includes high(q * p[3]).
            "mul {t3}, {a3}, {b1}",             // t3 = low(a[3] * b[1]).
            "adc {r4}, xzr, xzr",               // Capture the reduction carry as limb 4.

            // Round 1: add a * b[1] to the reduced accumulator.
            "adds {r0}, {r0}, {t0}",            // Add low(a[0] * b[1]) to limb 0.
            "umulh {t0}, {a0}, {b1}",           // t0 = high(a[0] * b[1]).
            "adcs {r1}, {r1}, {t1}",            // Add low(a[1] * b[1]) and carry.
            "umulh {t1}, {a1}, {b1}",           // t1 = high(a[1] * b[1]).
            "adcs {r2}, {r2}, {t2}",            // Add low(a[2] * b[1]) and carry.
            "mul {q}, {inv}, {r0}",             // q = current limb 0 * inv mod 2^64.
            "umulh {t2}, {a2}, {b1}",           // t2 = high(a[2] * b[1]).
            "adcs {r3}, {r3}, {t3}",            // Add low(a[3] * b[1]) and carry.
            "umulh {t3}, {a3}, {b1}",           // t3 = high(a[3] * b[1]).
            "adc {r4}, {r4}, xzr",              // Propagate multiplication carry to limb 4.

            "adds {r1}, {r1}, {t0}",            // Add high(a[0] * b[1]) to limb 1.
            // low(q * p[0]) cancels r0.
            "adcs {r2}, {r2}, {t1}",            // Add high(a[1] * b[1]) and carry.
            "mul {t1}, {p1}, {q}",              // t1 = low(q * p[1]).
            "adcs {r3}, {r3}, {t2}",            // Add high(a[2] * b[1]) and carry.
            // low(q * p[2]) is zero.
            "adc {r4}, {r4}, {t3}",             // Add high(a[3] * b[1]) and final carry.
            "lsl {t3}, {q}, #62",               // t3 = low(q * p[3]).
            "subs xzr, {r0}, #1",               // Set the low-limb cancellation carry.
            "umulh {t0}, {p0}, {q}",            // t0 = high(q * p[0]).
            "adcs {r1}, {r1}, {t1}",            // Add low(q * p[1]) and cancellation carry.
            "umulh {t1}, {p1}, {q}",            // t1 = high(q * p[1]).
            "adcs {r2}, {r2}, xzr",             // Propagate carry across zero p[2].
            // high(q * p[2]) is zero.
            "adcs {r3}, {r3}, {t3}",            // Add low(q * p[3]) and carry.
            "lsr {t3}, {q}, #2",                // t3 = high(q * p[3]).
            "adc {r4}, {r4}, xzr",              // Propagate carry to limb 4.

            // Shift after round 1 while starting a * b[2].
            "adds {r0}, {r1}, {t0}",            // New limb 0 includes high(q * p[0]).
            "mul {t0}, {a0}, {b2}",             // t0 = low(a[0] * b[2]).
            "adcs {r1}, {r2}, {t1}",            // New limb 1 includes high(q * p[1]).
            "mul {t1}, {a1}, {b2}",             // t1 = low(a[1] * b[2]).
            "adcs {r2}, {r3}, xzr",             // New limb 2; p[2] contributes zero.
            "mul {t2}, {a2}, {b2}",             // t2 = low(a[2] * b[2]).
            "adcs {r3}, {r4}, {t3}",            // New limb 3 includes high(q * p[3]).
            "mul {t3}, {a3}, {b2}",             // t3 = low(a[3] * b[2]).
            "adc {r4}, xzr, xzr",               // Capture the reduction carry as limb 4.

            // Round 2: add a * b[2] and cancel the resulting low limb.
            "adds {r0}, {r0}, {t0}",            // Add low(a[0] * b[2]) to limb 0.
            "umulh {t0}, {a0}, {b2}",           // t0 = high(a[0] * b[2]).
            "adcs {r1}, {r1}, {t1}",            // Add low(a[1] * b[2]) and carry.
            "umulh {t1}, {a1}, {b2}",           // t1 = high(a[1] * b[2]).
            "adcs {r2}, {r2}, {t2}",            // Add low(a[2] * b[2]) and carry.
            "mul {q}, {inv}, {r0}",             // q = current limb 0 * inv mod 2^64.
            "umulh {t2}, {a2}, {b2}",           // t2 = high(a[2] * b[2]).
            "adcs {r3}, {r3}, {t3}",            // Add low(a[3] * b[2]) and carry.
            "umulh {t3}, {a3}, {b2}",           // t3 = high(a[3] * b[2]).
            "adc {r4}, {r4}, xzr",              // Propagate multiplication carry to limb 4.

            "adds {r1}, {r1}, {t0}",            // Add high(a[0] * b[2]) to limb 1.
            // low(q * p[0]) cancels r0.
            "adcs {r2}, {r2}, {t1}",            // Add high(a[1] * b[2]) and carry.
            "mul {t1}, {p1}, {q}",              // t1 = low(q * p[1]).
            "adcs {r3}, {r3}, {t2}",            // Add high(a[2] * b[2]) and carry.
            // low(q * p[2]) is zero.
            "adc {r4}, {r4}, {t3}",             // Add high(a[3] * b[2]) and final carry.
            "lsl {t3}, {q}, #62",               // t3 = low(q * p[3]).
            "subs xzr, {r0}, #1",               // Set the low-limb cancellation carry.
            "umulh {t0}, {p0}, {q}",            // t0 = high(q * p[0]).
            "adcs {r1}, {r1}, {t1}",            // Add low(q * p[1]) and cancellation carry.
            "umulh {t1}, {p1}, {q}",            // t1 = high(q * p[1]).
            "adcs {r2}, {r2}, xzr",             // Propagate carry across zero p[2].
            // high(q * p[2]) is zero.
            "adcs {r3}, {r3}, {t3}",            // Add low(q * p[3]) and carry.
            "lsr {t3}, {q}, #2",                // t3 = high(q * p[3]).
            "adc {r4}, {r4}, xzr",              // Propagate carry to limb 4.

            // Shift after round 2 while starting a * b[3].
            "adds {r0}, {r1}, {t0}",            // New limb 0 includes high(q * p[0]).
            "mul {t0}, {a0}, {b3}",             // t0 = low(a[0] * b[3]).
            "adcs {r1}, {r2}, {t1}",            // New limb 1 includes high(q * p[1]).
            "mul {t1}, {a1}, {b3}",             // t1 = low(a[1] * b[3]).
            "adcs {r2}, {r3}, xzr",             // New limb 2; p[2] contributes zero.
            "mul {t2}, {a2}, {b3}",             // t2 = low(a[2] * b[3]).
            "adcs {r3}, {r4}, {t3}",            // New limb 3 includes high(q * p[3]).
            "mul {t3}, {a3}, {b3}",             // t3 = low(a[3] * b[3]).
            "adc {r4}, xzr, xzr",               // Capture the reduction carry as limb 4.

            // Round 3: add a * b[3] and perform the last Montgomery cancellation.
            "adds {r0}, {r0}, {t0}",            // Add low(a[0] * b[3]) to limb 0.
            "umulh {t0}, {a0}, {b3}",           // t0 = high(a[0] * b[3]).
            "adcs {r1}, {r1}, {t1}",            // Add low(a[1] * b[3]) and carry.
            "umulh {t1}, {a1}, {b3}",           // t1 = high(a[1] * b[3]).
            "adcs {r2}, {r2}, {t2}",            // Add low(a[2] * b[3]) and carry.
            "mul {q}, {inv}, {r0}",             // q = current limb 0 * inv mod 2^64.
            "umulh {t2}, {a2}, {b3}",           // t2 = high(a[2] * b[3]).
            "adcs {r3}, {r3}, {t3}",            // Add low(a[3] * b[3]) and carry.
            "umulh {t3}, {a3}, {b3}",           // t3 = high(a[3] * b[3]).
            "adc {r4}, {r4}, xzr",              // Propagate multiplication carry to limb 4.

            "adds {r1}, {r1}, {t0}",            // Add high(a[0] * b[3]) to limb 1.
            // low(q * p[0]) cancels r0.
            "adcs {r2}, {r2}, {t1}",            // Add high(a[1] * b[3]) and carry.
            "mul {t1}, {p1}, {q}",              // t1 = low(q * p[1]).
            "adcs {r3}, {r3}, {t2}",            // Add high(a[2] * b[3]) and carry.
            // low(q * p[2]) is zero.
            "adc {r4}, {r4}, {t3}",             // Add high(a[3] * b[3]) and final carry.
            "lsl {t3}, {q}, #62",               // t3 = low(q * p[3]).
            "subs xzr, {r0}, #1",               // Set the low-limb cancellation carry.
            "umulh {t0}, {p0}, {q}",            // t0 = high(q * p[0]).
            "adcs {r1}, {r1}, {t1}",            // Add low(q * p[1]) and cancellation carry.
            "umulh {t1}, {p1}, {q}",            // t1 = high(q * p[1]).
            "adcs {r2}, {r2}, xzr",             // Propagate carry across zero p[2].
            // high(q * p[2]) is zero.
            "adcs {r3}, {r3}, {t3}",            // Add low(q * p[3]) and carry.
            "lsr {t3}, {q}, #2",                // t3 = high(q * p[3]).
            "adc {r4}, {r4}, xzr",              // Propagate carry to limb 4.

            // Shift out the fourth cancelled limb. r4 records any 257th bit.
            "adds {r0}, {r1}, {t0}",            // Final candidate limb 0.
            "adcs {r1}, {r2}, {t1}",            // Final candidate limb 1.
            "adcs {r2}, {r3}, xzr",             // Final candidate limb 2.
            "adcs {r3}, {r4}, {t3}",            // Final candidate limb 3.
            "adc {r4}, xzr, xzr",               // Final candidate carry limb.

            // Subtract the five-limb value p = [p0,p1,0,p3,0].
            "mov {q}, #0x4000000000000000",     // Materialize p3 = 2^62.
            "subs {t0}, {r0}, {p0}",            // Tentative result limb 0 = candidate - p[0].
            "sbcs {t1}, {r1}, {p1}",            // Tentative result limb 1 minus p[1].
            "sbcs {t2}, {r2}, xzr",             // Tentative result limb 2; p[2] is zero.
            "sbcs {t3}, {r3}, {q}",             // Tentative result limb 3 minus p[3].
            "sbcs xzr, {r4}, xzr",              // Include the carry limb in the comparison.

            // `lo` means subtraction borrowed, so retain the original candidate.
            "csel {r0}, {r0}, {t0}, lo",        // Select canonical output limb 0.
            "csel {r1}, {r1}, {t1}, lo",        // Select canonical output limb 1.
            "csel {r2}, {r2}, {t2}, lo",        // Select canonical output limb 2.
            "csel {r3}, {r3}, {t3}, lo",        // Select canonical output limb 3.
            a0 = in(reg) lhs[0],
            a1 = in(reg) lhs[1],
            a2 = in(reg) lhs[2],
            a3 = in(reg) lhs[3],
            b0 = in(reg) rhs[0],
            b1 = in(reg) rhs[1],
            b2 = in(reg) rhs[2],
            b3 = in(reg) rhs[3],
            p0 = in(reg) modulus[0],
            p1 = in(reg) modulus[1],
            inv = in(reg) inv,
            q = out(reg) _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            r0 = out(reg) o0,
            r1 = out(reg) o1,
            r2 = out(reg) o2,
            r3 = out(reg) o3,
            r4 = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [o0, o1, o2, o3]
}

/// Squares a canonical Montgomery residue for a Pasta modulus.
#[inline(always)]
pub(super) fn square(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    let mut a0 = value[0];
    let mut a1 = value[1];
    let mut a2 = value[2];
    let mut a3 = value[3];
    // SAFETY: straight-line register-only arithmetic; no memory access, no
    // stack use, and outputs depend only on the declared inputs.
    unsafe {
        asm!(
            // 512-bit square: cross products, doubling, diagonals.
            // Square a (in a0..a3) exactly as in sqr_mont above; the 512-bit
            // product limbs A0..A7 map to z0,z1,z2,z3,z4,z5,z6,z7.

            "mul {z1}, {a1}, {a0}",             // z1 = low(a[1] * a[0]).
            "umulh {w1}, {a1}, {a0}",           // w1 = high(a[1] * a[0]).
            "mul {z2}, {a2}, {a0}",             // z2 = low(a[2] * a[0]).
            "umulh {w2}, {a2}, {a0}",           // w2 = high(a[2] * a[0]).
            "mul {z3}, {a3}, {a0}",             // z3 = low(a[3] * a[0]).
            "umulh {z4}, {a3}, {a0}",           // z4 = high(a[3] * a[0]).

            "adds {z2}, {z2}, {w1}",            // Fold high(a[1] * a[0]) into product limb 2.
            "mul {w0}, {a2}, {a1}",             // w0 = low(a[2] * a[1]).
            "umulh {w1}, {a2}, {a1}",           // w1 = high(a[2] * a[1]).
            "adcs {z3}, {z3}, {w2}",            // Fold high(a[2] * a[0]) into product limb 3.
            "mul {w2}, {a3}, {a1}",             // w2 = low(a[3] * a[1]).
            "umulh {w3}, {a3}, {a1}",           // w3 = high(a[3] * a[1]).
            "adc {z4}, {z4}, xzr",              // Propagate carry into product limb 4.

            "mul {z5}, {a3}, {a2}",             // z5 = low(a[3] * a[2]).
            "umulh {z6}, {a3}, {a2}",           // z6 = high(a[3] * a[2]).

            "adds {w1}, {w1}, {w2}",            // Combine terms contributing to product limb 4.
            "mul {z0}, {a0}, {a0}",             // z0 = low(a[0]^2), product limb 0.
            "adc {w2}, {w3}, xzr",              // Combine terms contributing to product limb 5.

            "adds {z3}, {z3}, {w0}",            // Add low(a[2] * a[1]) into product limb 3.
            "umulh {a0}, {a0}, {a0}",           // a0 = high(a[0]^2).
            "adcs {z4}, {z4}, {w1}",            // Accumulate cross terms into product limb 4.
            "mul {w1}, {a1}, {a1}",             // w1 = low(a[1]^2).
            "adcs {z5}, {z5}, {w2}",            // Accumulate cross terms into product limb 5.
            "umulh {a1}, {a1}, {a1}",           // a1 = high(a[1]^2).
            "adc {z6}, {z6}, xzr",              // Propagate carry into product limb 6.

            "adds {z1}, {z1}, {z1}",            // Double cross-term product limb 1.
            "mul {w2}, {a2}, {a2}",             // w2 = low(a[2]^2).
            "adcs {z2}, {z2}, {z2}",            // Double cross-term product limb 2.
            "umulh {a2}, {a2}, {a2}",           // a2 = high(a[2]^2).
            "adcs {z3}, {z3}, {z3}",            // Double cross-term product limb 3.
            "mul {w3}, {a3}, {a3}",             // w3 = low(a[3]^2).
            "adcs {z4}, {z4}, {z4}",            // Double cross-term product limb 4.
            "umulh {a3}, {a3}, {a3}",           // a3 = high(a[3]^2).
            "adcs {z5}, {z5}, {z5}",            // Double cross-term product limb 5.
            "adcs {z6}, {z6}, {z6}",            // Double cross-term product limb 6.
            "adc {z7}, xzr, xzr",               // Capture the doubled cross-term carry in limb 7.

            "mul {q}, {inv}, {z0}",             // q = product limb 0 * inv mod 2^64.

            // Add diagonal squares to obtain a^2 in z0..z7.
            "adds {z1}, {z1}, {a0}",            // Add high(a[0]^2) to product limb 1.
            "adcs {z2}, {z2}, {w1}",            // Add low(a[1]^2) to product limb 2.
            "adcs {z3}, {z3}, {a1}",            // Add high(a[1]^2) to product limb 3.
            "adcs {z4}, {z4}, {w2}",            // Add low(a[2]^2) to product limb 4.
            "adcs {z5}, {z5}, {a2}",            // Add high(a[2]^2) to product limb 5.
            "adcs {z6}, {z6}, {w3}",            // Add low(a[3]^2) to product limb 6.
            "adc {z7}, {z7}, {a3}",             // Add high(a[3]^2) to product limb 7.

            // Montgomery cancellation 0 on the low half, as in the shared helper.
            // low(q * p[0]) cancels z0 and is discarded by the limb shift.
            "mul {w1}, {p1}, {q}",              // w1 = low(q * p[1]).
            // q * p[2] is zero because p[2] = 0.
            "lsl {w3}, {q}, #62",               // w3 = low(q * p[3]).
            // Carry from z0 + low(q*p[0]) is one exactly when z0 is nonzero.
            "subs xzr, {z0}, #1",               // Set that cancellation carry.
            "umulh {w0}, {p0}, {q}",            // w0 = high(q * p[0]).
            "adcs {z1}, {z1}, {w1}",            // Add low(q * p[1]) and cancellation carry.
            "umulh {w1}, {p1}, {q}",            // w1 = high(q * p[1]).
            "adcs {z2}, {z2}, xzr",             // Propagate carry across zero p[2].
            // high(q * p[2]) is zero.
            "adcs {z3}, {z3}, {w3}",            // Add low(q * p[3]) and carry.
            "lsr {w3}, {q}, #2",                // w3 = high(q * p[3]).
            "adc {cy}, xzr, xzr",               // Save the carry above limb 3.

            // Shift out cancelled limb 0 and start cancellation 1.
            "adds {z0}, {z1}, {w0}",            // New limb 0 includes high(q * p[0]).
            "adcs {z1}, {z2}, {w1}",            // New limb 1 includes high(q * p[1]).
            "adcs {z2}, {z3}, xzr",             // New limb 2; p[2] contributes zero.
            "mul {q}, {inv}, {z0}",             // Next q = new limb 0 * inv mod 2^64.
            "adc {z3}, {cy}, {w3}",             // New limb 3 includes high(q * p[3]).
            // low(q * p[0]) cancels z0 and is discarded.
            "mul {w1}, {p1}, {q}",              // w1 = low(next q * p[1]).
            // next q * p[2] is zero.
            "lsl {w3}, {q}, #62",               // w3 = low(next q * p[3]).
            "subs xzr, {z0}, #1",               // Set the low-limb cancellation carry.
            "umulh {w0}, {p0}, {q}",            // w0 = high(next q * p[0]).
            "adcs {z1}, {z1}, {w1}",            // Add low(next q * p[1]) and carry.
            "umulh {w1}, {p1}, {q}",            // w1 = high(next q * p[1]).
            "adcs {z2}, {z2}, xzr",             // Propagate carry across zero p[2].
            // high(next q * p[2]) is zero.
            "adcs {z3}, {z3}, {w3}",            // Add low(next q * p[3]) and carry.
            "lsr {w3}, {q}, #2",                // w3 = high(next q * p[3]).
            "adc {cy}, xzr, xzr",               // Save the carry above limb 3.

            // Shift out cancelled limb 1 and start cancellation 2.
            "adds {z0}, {z1}, {w0}",            // New limb 0 includes high(q * p[0]).
            "adcs {z1}, {z2}, {w1}",            // New limb 1 includes high(q * p[1]).
            "adcs {z2}, {z3}, xzr",             // New limb 2; p[2] contributes zero.
            "mul {q}, {inv}, {z0}",             // Next q = new limb 0 * inv mod 2^64.
            "adc {z3}, {cy}, {w3}",             // New limb 3 includes high(q * p[3]).
            // low(q * p[0]) cancels z0 and is discarded.
            "mul {w1}, {p1}, {q}",              // w1 = low(next q * p[1]).
            // next q * p[2] is zero.
            "lsl {w3}, {q}, #62",               // w3 = low(next q * p[3]).
            "subs xzr, {z0}, #1",               // Set the low-limb cancellation carry.
            "umulh {w0}, {p0}, {q}",            // w0 = high(next q * p[0]).
            "adcs {z1}, {z1}, {w1}",            // Add low(next q * p[1]) and carry.
            "umulh {w1}, {p1}, {q}",            // w1 = high(next q * p[1]).
            "adcs {z2}, {z2}, xzr",             // Propagate carry across zero p[2].
            // high(next q * p[2]) is zero.
            "adcs {z3}, {z3}, {w3}",            // Add low(next q * p[3]) and carry.
            "lsr {w3}, {q}, #2",                // w3 = high(next q * p[3]).
            "adc {cy}, xzr, xzr",               // Save the carry above limb 3.

            // Shift out cancelled limb 2 and start cancellation 3.
            "adds {z0}, {z1}, {w0}",            // New limb 0 includes high(q * p[0]).
            "adcs {z1}, {z2}, {w1}",            // New limb 1 includes high(q * p[1]).
            "adcs {z2}, {z3}, xzr",             // New limb 2; p[2] contributes zero.
            "mul {q}, {inv}, {z0}",             // Final q = new limb 0 * inv mod 2^64.
            "adc {z3}, {cy}, {w3}",             // New limb 3 includes high(q * p[3]).
            // low(q * p[0]) cancels z0 and is discarded.
            "mul {w1}, {p1}, {q}",              // w1 = low(final q * p[1]).
            // final q * p[2] is zero.
            "lsl {w3}, {q}, #62",               // w3 = low(final q * p[3]).
            "subs xzr, {z0}, #1",               // Set the low-limb cancellation carry.
            "umulh {w0}, {p0}, {q}",            // w0 = high(final q * p[0]).
            "adcs {z1}, {z1}, {w1}",            // Add low(final q * p[1]) and carry.
            "umulh {w1}, {p1}, {q}",            // w1 = high(final q * p[1]).
            "adcs {z2}, {z2}, xzr",             // Propagate carry across zero p[2].
            // high(final q * p[2]) is zero.
            "adcs {z3}, {z3}, {w3}",            // Add low(final q * p[3]) and carry.
            "lsr {w3}, {q}, #2",                // w3 = high(final q * p[3]).
            "adc {cy}, xzr, xzr",               // Save the carry above limb 3.

            // Shift out cancelled limb 3 to finish dividing the low half by R.
            "adds {z0}, {z1}, {w0}",            // Reduced limb 0 includes high(q * p[0]).
            "adcs {z1}, {z2}, {w1}",            // Reduced limb 1 includes high(q * p[1]).
            "adcs {z2}, {z3}, xzr",             // Reduced limb 2; p[2] contributes zero.
            "adc {z3}, {cy}, {w3}",             // Reduced limb 3 includes high(q * p[3]).
            // Add the upper product half; the sum stays below 2p (see above), so no
            // carry escapes and no conditional subtraction is needed mid-loop.
            "adds {a0}, {z0}, {z4}",            // Next-iteration a[0].
            "adcs {a1}, {z1}, {z5}",            // Next-iteration a[1].
            "adcs {a2}, {z2}, {z6}",            // Next-iteration a[2].
            "adc {a3}, {z3}, {z7}",             // Next-iteration a[3].

            // Conditional subtraction of p = [p0, p1, 0, 2^62]. The input is
            // canonical, so the candidate is below 1.25p < 2^255: no bit 256
            // exists and a four-limb comparison suffices.
            "mov {q}, #0x4000000000000000",     // Materialize p3 = 2^62.
            "subs {z0}, {a0}, {p0}",            // Tentative limb 0 = candidate - p0.
            "sbcs {z1}, {a1}, {p1}",            // Tentative limb 1 minus p1.
            "sbcs {z2}, {a2}, xzr",             // Tentative limb 2; p2 is zero.
            "sbcs {z3}, {a3}, {q}",             // Tentative limb 3 minus p3.
            // `lo` means the subtraction borrowed, so retain the original
            // candidate.
            "csel {a0}, {a0}, {z0}, lo",        // Select canonical output limb 0.
            "csel {a1}, {a1}, {z1}, lo",        // Select canonical output limb 1.
            "csel {a2}, {a2}, {z2}, lo",        // Select canonical output limb 2.
            "csel {a3}, {a3}, {z3}, lo",        // Select canonical output limb 3.
            a0 = inout(reg) a0,
            a1 = inout(reg) a1,
            a2 = inout(reg) a2,
            a3 = inout(reg) a3,
            p0 = in(reg) modulus[0],
            p1 = in(reg) modulus[1],
            inv = in(reg) inv,
            q = out(reg) _,
            cy = out(reg) _,
            z0 = out(reg) _,
            z1 = out(reg) _,
            z2 = out(reg) _,
            z3 = out(reg) _,
            z4 = out(reg) _,
            z5 = out(reg) _,
            z6 = out(reg) _,
            z7 = out(reg) _,
            w0 = out(reg) _,
            w1 = out(reg) _,
            w2 = out(reg) _,
            w3 = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [a0, a1, a2, a3]
}

/// Squares a canonical Montgomery residue `count` times, then multiplies the
/// result by the canonical Montgomery residue `rhs`, keeping the accumulator
/// in registers throughout.
#[inline]
pub(super) fn sqr_n_mul(
    value: &Limbs,
    count: usize,
    rhs: &Limbs,
    modulus: &Limbs,
    inv: u64,
) -> Limbs {
    // The assembly decrements the count before testing it, so a zero count
    // would wrap around and effectively never terminate.
    assert!(count >= 1);
    let mut out = Limbs::default();
    // SAFETY: All pointers refer to four initialized `u64` limbs for the
    // duration of the call. The backend writes exactly four limbs to `out`.
    unsafe {
        pasta_curves_sqr_n_mul_mont_pasta(&mut out, value, count, rhs, modulus, inv);
    }
    out
}

/// Converts a canonical Montgomery residue into its canonical integer.
#[inline]
pub(super) fn from_mont(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    let mut out = Limbs::default();
    // SAFETY: All pointers refer to four initialized `u64` limbs for the
    // duration of the call. The backend writes exactly four limbs to `out`.
    unsafe {
        pasta_curves_from_mont_pasta(&mut out, value, modulus, inv);
    }
    out
}
