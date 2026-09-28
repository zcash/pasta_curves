// Copyright Supranational LLC (the Montgomery routines, transcribed from Semolina v0.1.4).
// Copyright Amazon.com, Inc. or its affiliates (the inversion blocks, adapted from s2n-bignum's
// `bignum_montinv_p256` at `ec62054cc1864839d44b1acc6e6a3f9eff5b6e68`, licensed Apache-2.0 OR
// ISC OR MIT-0).
// Copyright the zakura-core and pasta_curves contributors (the transcription and wrappers).

//! AArch64 backend for the Pasta fields.
//!
//! The inline blocks are register-renamed transcriptions of the upstream
//! Semolina v0.1.4 routines (`mul_mont_pasta`, and the squaring loop body of
//! `sqr_n_mul_mont_pasta`), with rhs limbs and the modulus constants supplied
//! in registers instead of loaded from memory, and of the blocks of
//! s2n-bignum's `bignum_montinv_p256` (its `divstep59` macro), with the
//! P-256 constants replaced by the Pasta ones. The per-instruction comments
//! are carried over from the assembly routines they transcribe. Because the
//! operands are ordinary register operands and the blocks are declared
//! `options(pure, nomem, nostack)`, LLVM inlines the wrappers into callers and
//! keeps field values in registers between operations — there is no call,
//! pointer, or ABI-clobber traffic per field operation.
//!
//! The arithmetic relies on the shared Pasta modulus shape: `modulus[2] = 0`
//! and `modulus[3] = 2^62` (materialized inline as an immediate). Only
//! `modulus[0]`, `modulus[1]`, and `inv` vary between Fp and Fq, so a single
//! implementation serves both fields.
//!
//! There are no branches and no memory accesses inside the blocks, and the
//! repeated-squaring loop branches only on its public count, so the code
//! should be constant-time, unless behaviour of the Rust toolchain or
//! platform introduces an unexpected obstacle to that.

use core::arch::asm;

use super::{Limbs, is_canonical, mul_contract};

/// Adds two residues for a Pasta modulus and conditionally subtracts the modulus.
///
/// Like [`mul`], the block hardcodes the Pasta modulus shape
/// (`modulus[2] == 0`). Both inputs must be canonical (debug-asserted; a
/// violation yields an incorrect residue): the top carry of the addition is
/// dropped and only one subtraction is attempted, both justified by
/// `2p < 2^256`. Unreduced values, such as an unreduced `lhs` that `mul`
/// accepts when `rhs` is canonical, must be reduced before reaching this
/// path. Keeping both carry chains in one block avoids materializing carries
/// between Rust operations.
#[inline(always)]
pub(super) fn add(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
    debug_assert!(
        is_canonical(lhs, modulus),
        "pasta_curves::asm::add requires a canonical lhs"
    );
    debug_assert!(
        is_canonical(rhs, modulus),
        "pasta_curves::asm::add requires a canonical rhs"
    );
    let [mut r0, mut r1, mut r2, mut r3] = *lhs;
    // SAFETY: register-only arithmetic with declared inputs and outputs;
    // no memory or stack access and no data-dependent control flow.
    unsafe {
        asm!(
            "adds {r0}, {r0}, {b0}",
            "adcs {r1}, {r1}, {b1}",
            "adcs {r2}, {r2}, {b2}",
            "adc {r3}, {r3}, {b3}",
            "subs {t0}, {r0}, {p0}",
            "sbcs {t1}, {r1}, {p1}",
            "sbcs {t2}, {r2}, xzr",
            "sbcs {t3}, {r3}, {p3}",
            "csel {r0}, {t0}, {r0}, cs",
            "csel {r1}, {t1}, {r1}, cs",
            "csel {r2}, {t2}, {r2}, cs",
            "csel {r3}, {t3}, {r3}, cs",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            b0 = in(reg) rhs[0],
            b1 = in(reg) rhs[1],
            b2 = in(reg) rhs[2],
            b3 = in(reg) rhs[3],
            p0 = in(reg) modulus[0],
            p1 = in(reg) modulus[1],
            p3 = in(reg) modulus[3],
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [r0, r1, r2, r3]
}

/// Subtracts two residues for a Pasta modulus, adding the modulus back on
/// underflow.
///
/// Like [`add`] and [`mul`], the block hardcodes the Pasta
/// modulus shape (`modulus[2] == 0`). Canonical inputs (debug-asserted)
/// guarantee a canonical result: the difference lies strictly between `-p`
/// and `p`, so one conditional addition suffices, and the final carry is
/// discarded after wrapping modulo `2^256`.
#[inline(always)]
pub(super) fn sub(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
    debug_assert!(
        is_canonical(lhs, modulus),
        "pasta_curves::asm::sub requires a canonical lhs"
    );
    debug_assert!(
        is_canonical(rhs, modulus),
        "pasta_curves::asm::sub requires a canonical rhs"
    );
    let [mut r0, mut r1, mut r2, mut r3] = *lhs;
    // SAFETY: register-only arithmetic with declared inputs and outputs;
    // no memory or stack access and no data-dependent control flow.
    unsafe {
        asm!(
            "subs {r0}, {r0}, {b0}",
            "sbcs {r1}, {r1}, {b1}",
            "sbcs {r2}, {r2}, {b2}",
            "sbcs {r3}, {r3}, {b3}",
            "csel {t0}, {p0}, xzr, cc",
            "csel {t1}, {p1}, xzr, cc",
            "csel {t3}, {p3}, xzr, cc",
            "adds {r0}, {r0}, {t0}",
            "adcs {r1}, {r1}, {t1}",
            "adcs {r2}, {r2}, xzr",
            "adc {r3}, {r3}, {t3}",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            b0 = in(reg) rhs[0],
            b1 = in(reg) rhs[1],
            b2 = in(reg) rhs[2],
            b3 = in(reg) rhs[3],
            p0 = in(reg) modulus[0],
            p1 = in(reg) modulus[1],
            p3 = in(reg) modulus[3],
            t0 = out(reg) _,
            t1 = out(reg) _,
            t3 = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [r0, r1, r2, r3]
}

/// Multiplies two Montgomery residues for a Pasta modulus.
///
/// # Safety
///
/// Either `lhs` is canonical and `rhs` is any four-limb value, or `rhs` is canonical with
/// limbs 1 to 3 at most `2^64 - 3` and `lhs` is any four-limb value.
///
/// Two things can go wrong outside the contract. First, `mul` keeps a
/// five-limb accumulator (one word fewer than textbook CIOS; the module's
/// README describes the form), and the carry chain folding in the high
/// cross-products can wrap: its tail computes
/// `acc4 + high(lhs[3] * rhs_limb) + carry` with `acc4 <= 2`, which reaches
/// `2^64` only when `high(lhs[3] * rhs_limb) >= 2^64 - 3`. A canonical `lhs`
/// has `lhs[3] <= 2^62`, and a `rhs` limb at most `2^64 - 3` caps the high
/// product at `2^64 - 4`, so either condition alone rules the wrap out.
/// Whether the chain can wrap with a `rhs` limb of `2^64 - 2` is not settled
/// by the proofs. Second, `mul` keeps only four limbs of its final candidate
/// `(lhs * rhs + m * p) / R`, where `m < R` is the Montgomery cancellation
/// factor. The candidate is below `2p < R` whenever `lhs * rhs < R * p`, which
/// a canonical `lhs` (with `rhs < R`) or a canonical `rhs` (with `lhs < R`)
/// gives, so under either contract the dropped fifth limb is zero. With both
/// operands unreduced the candidate can reach `R`, and the result is then an
/// incorrect residue that still looks canonical.
#[inline(always)]
pub(crate) fn mul(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    debug_assert!(
        mul_contract(lhs, rhs, modulus),
        "pasta_curves::asm::mul requires a canonical lhs, or a canonical rhs with limbs 1 to 3 \
         at most 2^64 - 3"
    );
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

            // Shift out the fourth cancelled limb. Either contract gives
            // lhs*rhs < R*p, and m < R gives m*p < R*p. Thus the candidate
            // (lhs*rhs + m*p)/R is below 2p < R, so no fifth limb exists.
            "adds {r0}, {r1}, {t0}",            // Final candidate limb 0.
            "adcs {r1}, {r2}, {t1}",            // Final candidate limb 1.
            "adcs {r2}, {r3}, xzr",             // Final candidate limb 2.
            "adcs {r3}, {r4}, {t3}",            // Final candidate limb 3.

            // Subtract p = [p0,p1,0,p3].
            "mov {q}, #0x4000000000000000",     // Materialize p3 = 2^62.
            "subs {t0}, {r0}, {p0}",            // Tentative result limb 0 = candidate - p[0].
            "sbcs {t1}, {r1}, {p1}",            // Tentative result limb 1 minus p[1].
            "sbcs {t2}, {r2}, xzr",             // Tentative result limb 2; p[2] is zero.
            "sbcs {t3}, {r3}, {q}",             // Tentative result limb 3 minus p[3].

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
///
/// The input's canonicity is debug-asserted.
#[inline(always)]
pub(crate) fn square(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    debug_assert!(
        is_canonical(value, modulus),
        "pasta_curves::asm::square requires a canonical input"
    );
    let mut a0 = value[0];
    let mut a1 = value[1];
    let mut a2 = value[2];
    let mut a3 = value[3];
    // SAFETY: straight-line register-only arithmetic; no memory access, no
    // stack use, and outputs depend only on the declared inputs.
    unsafe {
        asm!(
            // 512-bit square: cross products, doubling, diagonals.
            // Square a (in a0..a3); the 512-bit product limbs A0..A7 map to
            // z0,z1,z2,z3,z4,z5,z6,z7.

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

            // Montgomery cancellation 0 on the low half.
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
            // Add the upper product half. A canonical input's square is below
            // R*p, so, as for `mul`'s candidate, the sum stays below 2p: no carry
            // escapes and no conditional subtraction is needed mid-loop.
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

/// One packed half-delta divstep, the step body of s2n-bignum's `divstep59`.
///
/// The words `pf` and `pg` carry the low bits of `f` and `g` with a row of the
/// transition matrix in their upper bits, and `two_delta = 2δ`.
/// The flags hold the parity test of `pg` from the previous step (`tst pg, #1`
/// before the first step of a batch). A step swaps and subtracts when `g` is
/// odd and `two_delta` is positive, adds `f` to `g` when `g` is odd and `two_delta` is not,
/// and does nothing to `g` when it is even; then it halves `g` and adds two
/// to `two_delta`. The parity of the next `g` is tested before the halving, on bit 1,
/// so that the test does not wait for the shift. The `last` arm omits that
/// test, since nothing follows the last step of a batch.
macro_rules! divstep {
    (core) => {
        concat!(
            "csel {t}, {pf}, xzr, ne\n",           // t = g odd ? f : 0.
            "ccmp {two_delta}, xzr, #8, ne\n",     // g odd: flags of two_delta - 0; else N set.
            "cneg {two_delta}, {two_delta}, ge\n", // g odd and two_delta >= 0: two_delta = -two_delta.
            "cneg {t}, {t}, ge\n",                 // g odd and two_delta >= 0: t = -f.
            "csel {pf}, {pg}, {pf}, ge\n",         // g odd and two_delta >= 0: f = g.
            "add {pg}, {pg}, {t}\n",               // g = g + t.
            "add {two_delta}, {two_delta}, #2\n",  // two_delta = two_delta + 2.
        )
    };
    () => {
        concat!(
            divstep!(core),
            "tst {pg}, #2\n",                      // The parity of the halved g.
            "asr {pg}, {pg}, #1\n",                // g = g / 2.
        )
    };
    (last) => {
        concat!(
            divstep!(core),
            "asr {pg}, {pg}, #1\n",                // g = g / 2.
        )
    };
}

/// Fifty-nine half-delta divsteps on the low words of `f` and `g`, returning
/// the new `two_delta` and the transition matrix.
///
/// `two_delta = 2δ`, as a two's-complement word (`1` at the
/// start), and `f0` and `g0` are the low 64 bits of `f` and `g`, `f` odd. The
/// result is `[two_delta', u, v, q, r]`: the new `two_delta`, and the 59-step
/// transition matrix `M` with `2^59 (f', g') = M (f, g)`, as two's-complement
/// words. The block is s2n-bignum's `divstep59` macro on named registers: three
/// batches of 20, 20, and 19 steps, each on two words that pack the low 20 bits
/// of `f` and `g` with a row of the batch's matrix in the upper bits (`u` at
/// bit 41 and `v` at bit 62 at the start, halved with each step), whose
/// matrices are read back out of the upper bits, negated, and multiplied
/// together. Between batches the next low words are the matrix rows applied
/// to the current ones, shifted right by the batch's step count.
#[inline(always)]
pub(super) fn divstep59(mut two_delta: u64, f0: u64, g0: u64) -> [u64; 5] {
    let (u, v, q, r): (u64, u64, u64, u64);
    // SAFETY: straight-line register-only arithmetic; no memory access, no
    // stack use, and outputs depend only on the declared inputs.
    unsafe {
        asm!(
            // Batch 1: pack the low 20 bits with the identity row (-2^41, -2^62).
            "and {pf}, {f}, #0xfffff",
            "orr {pf}, {pf}, #0xfffffe0000000000",
            "and {pg}, {g}, #0xfffff",
            "orr {pg}, {pg}, #0xc000000000000000",
            "tst {pg}, #1",
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(last),
            // Read the negated matrix of batch 1 out of the upper bits: the
            // row's first entry as the 21-bit field at bit 21 after adding
            // 2^20, the second as the arithmetic shift by 42 after adding
            // 2^20 + 2^41.
            "add {a00}, {pf}, #0x100, lsl #12",
            "sbfx {a00}, {a00}, #21, #21",
            "mov {a11}, #0x100000",
            "add {a11}, {a11}, {a11}, lsl #21",
            "add {a01}, {pf}, {a11}",
            "asr {a01}, {a01}, #42",
            "add {a10}, {pg}, #0x100, lsl #12",
            "sbfx {a10}, {a10}, #21, #21",
            "add {a11}, {pg}, {a11}",
            "asr {a11}, {a11}, #42",
            // The next low words: the rows applied to the current ones,
            // shifted right by 20.
            "mul {t}, {a00}, {f}",
            "mul {t2}, {a01}, {g}",
            "mul {f}, {a10}, {f}",
            "mul {g}, {a11}, {g}",
            "add {pf}, {t}, {t2}",
            "add {pg}, {f}, {g}",
            "asr {f}, {pf}, #20",
            "asr {g}, {pg}, #20",
            // Batch 2.
            "and {pf}, {f}, #0xfffff",
            "orr {pf}, {pf}, #0xfffffe0000000000",
            "and {pg}, {g}, #0xfffff",
            "orr {pg}, {pg}, #0xc000000000000000",
            "tst {pg}, #1",
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(last),
            "add {b00}, {pf}, #0x100, lsl #12",
            "sbfx {b00}, {b00}, #21, #21",
            "mov {b11}, #0x100000",
            "add {b11}, {b11}, {b11}, lsl #21",
            "add {b01}, {pf}, {b11}",
            "asr {b01}, {b01}, #42",
            "add {b10}, {pg}, #0x100, lsl #12",
            "sbfx {b10}, {b10}, #21, #21",
            "add {b11}, {pg}, {b11}",
            "asr {b11}, {b11}, #42",
            "mul {t}, {b00}, {f}",
            "mul {t2}, {b01}, {g}",
            "mul {f}, {b10}, {f}",
            "mul {g}, {b11}, {g}",
            "add {pf}, {t}, {t2}",
            "add {pg}, {f}, {g}",
            "asr {f}, {pf}, #20",
            "asr {g}, {pg}, #20",
            // Batch 3, of 19 steps.
            "and {pf}, {f}, #0xfffff",
            "orr {pf}, {pf}, #0xfffffe0000000000",
            "and {pg}, {g}, #0xfffff",
            "orr {pg}, {pg}, #0xc000000000000000",
            "tst {pg}, #1",
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            // The product of the negated matrices of batches 2 and 1, which is
            // the product of the matrices themselves, into (a00, a01, c10, c11).
            "mul {f}, {b00}, {a00}",
            "mul {g}, {b00}, {a01}",
            "mul {t}, {b10}, {a00}",
            "mul {t2}, {b10}, {a01}",
            "madd {a00}, {b01}, {a10}, {f}",
            "madd {a01}, {b01}, {a11}, {g}",
            "madd {c10}, {b11}, {a10}, {t}",
            "madd {c11}, {b11}, {a11}, {t2}",
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(),
            divstep!(last),
            // The negated matrix of batch 3, whose rows sit one bit higher
            // after 19 steps.
            "add {b00}, {pf}, #0x100, lsl #12",
            "sbfx {b00}, {b00}, #22, #21",
            "mov {b11}, #0x100000",
            "add {b11}, {b11}, {b11}, lsl #21",
            "add {b01}, {pf}, {b11}",
            "asr {b01}, {b01}, #43",
            "add {b10}, {pg}, #0x100, lsl #12",
            "sbfx {b10}, {b10}, #22, #21",
            "add {b11}, {pg}, {b11}",
            "asr {b11}, {b11}, #43",
            // The 59-step matrix: minus the negated matrix of batch 3 times
            // the product of batches 2 and 1.
            "mneg {f}, {b00}, {a00}",
            "mneg {g}, {b00}, {a01}",
            "mneg {pf}, {b10}, {a00}",
            "mneg {pg}, {b10}, {a01}",
            "msub {u}, {b01}, {c10}, {f}",
            "msub {v}, {b01}, {c11}, {g}",
            "msub {q}, {b11}, {c10}, {pf}",
            "msub {r}, {b11}, {c11}, {pg}",
            two_delta = inout(reg) two_delta,
            f = inout(reg) f0 => _,
            g = inout(reg) g0 => _,
            pf = out(reg) _,
            pg = out(reg) _,
            t = out(reg) _,
            t2 = out(reg) _,
            a00 = out(reg) _,
            a01 = out(reg) _,
            a10 = out(reg) _,
            a11 = out(reg) _,
            b00 = out(reg) _,
            b01 = out(reg) _,
            b10 = out(reg) _,
            b11 = out(reg) _,
            c10 = out(reg) _,
            c11 = out(reg) _,
            u = out(reg) u,
            v = out(reg) v,
            q = out(reg) q,
            r = out(reg) r,
            options(pure, nomem, nostack),
        );
    }
    [two_delta, u, v, q, r]
}

/// The sign-magnitude form of a transition matrix: each entry's magnitude,
/// and its sign as a mask (all ones for a negative entry, else zero).
///
/// The row blocks multiply by a negative entry `m` as `|m| · ((x ^ mask) + 1)`,
/// that is by the magnitude times the complement of `x`, with the `+ |m|` folded
/// into the initial carry, so they take the entries in this form. The result is
/// `[u, v, q, r, su, sv, sq, sr]`, magnitudes then masks. The entry
/// `-2^63` has no magnitude as a word; the 59-step matrices' entries are below
/// `2^59` in magnitude.
#[inline(always)]
pub(super) fn sign_mag(mut u: u64, mut v: u64, mut q: u64, mut r: u64) -> [u64; 8] {
    let (su, sv, sq, sr): (u64, u64, u64, u64);
    // SAFETY: register-only arithmetic with declared inputs and outputs;
    // no memory or stack access and no data-dependent control flow.
    unsafe {
        asm!(
            "cmp {u}, xzr",
            "csetm {su}, mi",
            "cneg {u}, {u}, mi",
            "cmp {v}, xzr",
            "csetm {sv}, mi",
            "cneg {v}, {v}, mi",
            "cmp {q}, xzr",
            "csetm {sq}, mi",
            "cneg {q}, {q}, mi",
            "cmp {r}, xzr",
            "csetm {sr}, mi",
            "cneg {r}, {r}, mi",
            u = inout(reg) u,
            v = inout(reg) v,
            q = inout(reg) q,
            r = inout(reg) r,
            su = out(reg) su,
            sv = out(reg) sv,
            sq = out(reg) sq,
            sr = out(reg) sr,
            options(pure, nomem, nostack),
        );
    }
    [u, v, q, r, su, sv, sq, sr]
}

/// One row of the update of `f` and `g` by a transition matrix:
/// `(m0 · f + m1 · g) / 2^59`, an exact division, on five-word signed values.
///
/// `f` and `g` are five words each, the top word being the sign word (zero or
/// all ones, since the rounds keep them below `2^256` in magnitude). `m0` and
/// `m1` are the row's magnitudes and `s0` and `s1` its sign masks, from
/// `sign_mag`. A negative entry multiplies the complement of its operand under
/// the mask, with the magnitude added as the initial carry. The 320-bit sum is
/// accumulated digit by digit with a two-word carry and shifted right by 59 as
/// it is stored, the top word arithmetically. It is s2n-bignum's `f`/`g` update
/// on named registers, one row at a time.
#[inline(always)]
pub(super) fn fg_row(f: &[u64; 5], g: &[u64; 5], m0: u64, m1: u64, s0: u64, s1: u64) -> [u64; 5] {
    let (t0, t1, t2, t3, t4): (u64, u64, u64, u64, u64);
    // SAFETY: register-only arithmetic with declared inputs and outputs;
    // no memory or stack access and no data-dependent control flow.
    unsafe {
        asm!(
            // The initial carry: the magnitude of each negative entry.
            "and {lo}, {m0}, {s0}",
            "and {w}, {m1}, {s1}",
            "add {t0}, {lo}, {w}",
            // Digit 0.
            "eor {w}, {f0}, {s0}",
            "mul {lo}, {w}, {m0}",
            "umulh {w}, {w}, {m0}",
            "adds {t0}, {t0}, {lo}",
            "adc {t1}, xzr, {w}",
            "eor {w}, {g0}, {s1}",
            "mul {lo}, {w}, {m1}",
            "umulh {w}, {w}, {m1}",
            "adds {t0}, {t0}, {lo}",
            "adc {t1}, {t1}, {w}",
            // Digit 1, then the shifted digit 0.
            "eor {w}, {f1}, {s0}",
            "mul {lo}, {w}, {m0}",
            "umulh {w}, {w}, {m0}",
            "adds {t1}, {t1}, {lo}",
            "adc {t2}, xzr, {w}",
            "eor {w}, {g1}, {s1}",
            "mul {lo}, {w}, {m1}",
            "umulh {w}, {w}, {m1}",
            "adds {t1}, {t1}, {lo}",
            "adc {t2}, {t2}, {w}",
            "extr {t0}, {t1}, {t0}, #59",
            // Digit 2, then the shifted digit 1.
            "eor {w}, {f2}, {s0}",
            "mul {lo}, {w}, {m0}",
            "umulh {w}, {w}, {m0}",
            "adds {t2}, {t2}, {lo}",
            "adc {t3}, xzr, {w}",
            "eor {w}, {g2}, {s1}",
            "mul {lo}, {w}, {m1}",
            "umulh {w}, {w}, {m1}",
            "adds {t2}, {t2}, {lo}",
            "adc {t3}, {t3}, {w}",
            "extr {t1}, {t2}, {t1}, #59",
            // Digits 3 and 4: the sign word contributes minus the magnitude when
            // the complemented operand is negative.
            "eor {w}, {f3}, {s0}",
            "eor {t4}, {f4}, {s0}",
            "and {t4}, {t4}, {m0}",
            "neg {t4}, {t4}",
            "mul {lo}, {w}, {m0}",
            "umulh {w}, {w}, {m0}",
            "adds {t3}, {t3}, {lo}",
            "adc {t4}, {t4}, {w}",
            "eor {w}, {g3}, {s1}",
            "eor {lo}, {g4}, {s1}",
            "and {lo}, {lo}, {m1}",
            "sub {t4}, {t4}, {lo}",
            "mul {lo}, {w}, {m1}",
            "umulh {w}, {w}, {m1}",
            "adds {t3}, {t3}, {lo}",
            "adc {t4}, {t4}, {w}",
            "extr {t2}, {t3}, {t2}, #59",
            "extr {t3}, {t4}, {t3}, #59",
            "asr {t4}, {t4}, #59",
            f0 = in(reg) f[0],
            f1 = in(reg) f[1],
            f2 = in(reg) f[2],
            f3 = in(reg) f[3],
            f4 = in(reg) f[4],
            g0 = in(reg) g[0],
            g1 = in(reg) g[1],
            g2 = in(reg) g[2],
            g3 = in(reg) g[3],
            g4 = in(reg) g[4],
            m0 = in(reg) m0,
            m1 = in(reg) m1,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            lo = out(reg) _,
            w = out(reg) _,
            t0 = out(reg) t0,
            t1 = out(reg) t1,
            t2 = out(reg) t2,
            t3 = out(reg) t3,
            t4 = out(reg) t4,
            options(pure, nomem, nostack),
        );
    }
    [t0, t1, t2, t3, t4]
}

/// One row of the combination of `d` and `e` by a transition matrix,
/// `m0 · d + m1 · e`, as a five-word signed value for `amontred`.
///
/// `d` and `e` are four unsigned words each, below `2^256`. `m0`, `m1`, `s0`,
/// and `s1` are the row's magnitudes and sign masks from `sign_mag`; the last
/// round folds the sign of `f` into the masks before calling. The accumulation
/// is that of `fg_row` without the shift, and with the top word contributing
/// minus the magnitude under the mask, since the operands' sign words are zero.
/// It is s2n-bignum's coefficient accumulation on named registers, one row at a
/// time.
#[inline(always)]
pub(super) fn de_row(d: &Limbs, e: &Limbs, m0: u64, m1: u64, s0: u64, s1: u64) -> [u64; 5] {
    let (t0, t1, t2, t3, t4): (u64, u64, u64, u64, u64);
    // SAFETY: register-only arithmetic with declared inputs and outputs;
    // no memory or stack access and no data-dependent control flow.
    unsafe {
        asm!(
            // The initial carry: the magnitude of each negative entry.
            "and {lo}, {m0}, {s0}",
            "and {w}, {m1}, {s1}",
            "add {t0}, {lo}, {w}",
            // Digit 0.
            "eor {w}, {d0}, {s0}",
            "mul {lo}, {w}, {m0}",
            "umulh {w}, {w}, {m0}",
            "adds {t0}, {t0}, {lo}",
            "adc {t1}, xzr, {w}",
            "eor {w}, {e0}, {s1}",
            "mul {lo}, {w}, {m1}",
            "umulh {w}, {w}, {m1}",
            "adds {t0}, {t0}, {lo}",
            "adc {t1}, {t1}, {w}",
            // Digit 1.
            "eor {w}, {d1}, {s0}",
            "mul {lo}, {w}, {m0}",
            "umulh {w}, {w}, {m0}",
            "adds {t1}, {t1}, {lo}",
            "adc {t2}, xzr, {w}",
            "eor {w}, {e1}, {s1}",
            "mul {lo}, {w}, {m1}",
            "umulh {w}, {w}, {m1}",
            "adds {t1}, {t1}, {lo}",
            "adc {t2}, {t2}, {w}",
            // Digit 2.
            "eor {w}, {d2}, {s0}",
            "mul {lo}, {w}, {m0}",
            "umulh {w}, {w}, {m0}",
            "adds {t2}, {t2}, {lo}",
            "adc {t3}, xzr, {w}",
            "eor {w}, {e2}, {s1}",
            "mul {lo}, {w}, {m1}",
            "umulh {w}, {w}, {m1}",
            "adds {t2}, {t2}, {lo}",
            "adc {t3}, {t3}, {w}",
            // Digits 3 and 4: the top word is unsigned, so the sign word's
            // contribution is minus the magnitude under the mask.
            "eor {w}, {d3}, {s0}",
            "and {t4}, {s0}, {m0}",
            "neg {t4}, {t4}",
            "mul {lo}, {w}, {m0}",
            "umulh {w}, {w}, {m0}",
            "adds {t3}, {t3}, {lo}",
            "adc {t4}, {t4}, {w}",
            "eor {w}, {e3}, {s1}",
            "and {lo}, {s1}, {m1}",
            "sub {t4}, {t4}, {lo}",
            "mul {lo}, {w}, {m1}",
            "umulh {w}, {w}, {m1}",
            "adds {t3}, {t3}, {lo}",
            "adc {t4}, {t4}, {w}",
            d0 = in(reg) d[0],
            d1 = in(reg) d[1],
            d2 = in(reg) d[2],
            d3 = in(reg) d[3],
            e0 = in(reg) e[0],
            e1 = in(reg) e[1],
            e2 = in(reg) e[2],
            e3 = in(reg) e[3],
            m0 = in(reg) m0,
            m1 = in(reg) m1,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            lo = out(reg) _,
            w = out(reg) _,
            t0 = out(reg) t0,
            t1 = out(reg) t1,
            t2 = out(reg) t2,
            t3 = out(reg) t3,
            t4 = out(reg) t4,
            options(pure, nomem, nostack),
        );
    }
    [t0, t1, t2, t3, t4]
}

/// The almost-Montgomery reduction of a five-word signed value by one word:
/// `(s + w · p) / 2^64` for `s = t + 2^61 · p` and `w = s · inv mod 2^64`, as
/// four words.
///
/// For `|t| < 2^315`, which the row combinations satisfy, the result is below
/// `2p` and below `2^256` (Lemma 10 of `book/src/design/inversion.md`), so
/// unlike P-256's `amontred` in s2n-bignum there is no top carry to capture
/// and no conditional subtraction. Adding `2^61 · p` for the Pasta shape adds
/// `p0 << 61`, `(p1 : p0) >> 3`, `p1 >> 3`, and `2^59` at words 0, 1, 2, and 4;
/// adding `w · p` is the cancellation step of `mul`, with the low word's carry
/// taken from `s0` being nonzero.
#[inline(always)]
pub(super) fn amontred(t: &[u64; 5], modulus: &Limbs, inv: u64) -> Limbs {
    let (o0, o1, o2, o3): (u64, u64, u64, u64);
    // SAFETY: register-only arithmetic with declared inputs and outputs;
    // no memory or stack access and no data-dependent control flow.
    unsafe {
        asm!(
            // s = t + 2^61 p, in five words.
            "lsl {w}, {p0}, #61",
            "adds {t0}, {t0}, {w}",
            "extr {w}, {p1}, {p0}, #3",
            "adcs {t1}, {t1}, {w}",
            "lsr {w}, {p1}, #3",
            "adcs {t2}, {t2}, {w}",
            "adcs {t3}, {t3}, xzr",
            "mov {w}, #0x800000000000000",
            "adc {t4}, {t4}, {w}",
            // The multiple of p that clears the low word, and its products.
            "mul {q}, {t0}, {inv}",
            "umulh {hi}, {q}, {p0}",
            "mul {lo}, {q}, {p1}",
            "umulh {w}, {q}, {p1}",
            "lsl {l3}, {q}, #62",
            "lsr {h3}, {q}, #2",
            // s + q p: the low word cancels, with a carry exactly when s0 is
            // nonzero; the high halves and q * 2^254 in one chain, low(q p1) in
            // another.
            "subs xzr, {t0}, #1",
            "adcs {t1}, {t1}, {hi}",
            "adcs {t2}, {t2}, {w}",
            "adcs {t3}, {t3}, {l3}",
            "adc {t4}, {t4}, {h3}",
            "adds {t1}, {t1}, {lo}",
            "adcs {t2}, {t2}, xzr",
            "adcs {t3}, {t3}, xzr",
            "adc {t4}, {t4}, xzr",
            t0 = inout(reg) t[0] => _,
            t1 = inout(reg) t[1] => o0,
            t2 = inout(reg) t[2] => o1,
            t3 = inout(reg) t[3] => o2,
            t4 = inout(reg) t[4] => o3,
            p0 = in(reg) modulus[0],
            p1 = in(reg) modulus[1],
            inv = in(reg) inv,
            q = out(reg) _,
            w = out(reg) _,
            lo = out(reg) _,
            hi = out(reg) _,
            l3 = out(reg) _,
            h3 = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [o0, o1, o2, o3]
}

/// Subtracts the modulus from `value` unless the subtraction borrows.
///
/// For `value < 2p` the result is `value mod p`; the inversion applies it to the final
/// round's `amontred` output, which is below `2p`. It is the tail of the
/// Montgomery blocks, with the modulus shape `modulus[2] = 0`,
/// `modulus[3] = 2^62`.
#[inline(always)]
pub(super) fn cond_sub(value: &Limbs, modulus: &Limbs) -> Limbs {
    let [mut r0, mut r1, mut r2, mut r3] = *value;
    // SAFETY: register-only arithmetic with declared inputs and outputs;
    // no memory or stack access and no data-dependent control flow.
    unsafe {
        asm!(
            "mov {q}, #0x4000000000000000",
            "subs {t0}, {r0}, {p0}",
            "sbcs {t1}, {r1}, {p1}",
            "sbcs {t2}, {r2}, xzr",
            "sbcs {t3}, {r3}, {q}",
            "csel {r0}, {r0}, {t0}, lo",
            "csel {r1}, {r1}, {t1}, lo",
            "csel {r2}, {r2}, {t2}, lo",
            "csel {r3}, {r3}, {t3}, lo",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            p0 = in(reg) modulus[0],
            p1 = in(reg) modulus[1],
            q = out(reg) _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [r0, r1, r2, r3]
}

/// The backend's inversion blocks, for the generic driver of `crate::inversion`.
pub(crate) struct Backend;

impl crate::inversion::InvertBlocks for Backend {
    #[inline(always)]
    fn divstep59(two_delta: u64, f0: u64, g0: u64) -> [u64; 5] {
        divstep59(two_delta, f0, g0)
    }

    #[inline(always)]
    fn sign_mag(u: u64, v: u64, q: u64, r: u64) -> [u64; 8] {
        sign_mag(u, v, q, r)
    }

    #[inline(always)]
    fn fg_row(f: &[u64; 5], g: &[u64; 5], m0: u64, m1: u64, s0: u64, s1: u64) -> [u64; 5] {
        fg_row(f, g, m0, m1, s0, s1)
    }

    #[inline(always)]
    fn de_row(d: &Limbs, e: &Limbs, m0: u64, m1: u64, s0: u64, s1: u64) -> [u64; 5] {
        de_row(d, e, m0, m1, s0, s1)
    }

    #[inline(always)]
    fn amontred(t: &[u64; 5], modulus: &Limbs, inv: u64) -> Limbs {
        amontred(t, modulus, inv)
    }

    #[inline(always)]
    fn cond_sub(value: &Limbs, modulus: &Limbs) -> Limbs {
        cond_sub(value, modulus)
    }
}

#[cfg(test)]
mod tests {
    use super::Backend;
    use crate::inversion::tests::DIVSTEP59_VECTORS;
    use core::arch::asm;

    /// One step of the `divstep!` macro on a packed state, for tracing the block step by step:
    /// the parity test that precedes a batch's first step, then the step without the test that
    /// would feed a next one.
    fn divstep_once(mut two_delta: u64, mut pf: u64, mut pg: u64) -> [u64; 3] {
        // SAFETY: register-only arithmetic with declared inputs and outputs.
        unsafe {
            asm!(
                "tst {pg}, #1",
                divstep!(last),
                two_delta = inout(reg) two_delta,
                pf = inout(reg) pf,
                pg = inout(reg) pg,
                t = out(reg) _,
                options(pure, nomem, nostack),
            );
        }
        [two_delta, pf, pg]
    }

    /// The packed state `(two_delta, pf, pg)` before each of the first batch's twenty steps on the
    /// first vector of `DIVSTEP59_VECTORS`, and after the last, from the integer recurrence on
    /// the packed state (`Inversion/Packed.lean`'s `packedDivsteps`).
    const DIVSTEP_TRACE: [[u64; 3]; 21] = [
        [0x0000000000000001, 0xfffffe0000000001, 0xc0000000000d9046],
        [0x0000000000000003, 0xfffffe0000000001, 0xe00000000006c823],
        [0xffffffffffffffff, 0xe00000000006c823, 0xf000010000036411],
        [0x0000000000000001, 0xe00000000006c823, 0xe80000800005161a],
        [0x0000000000000003, 0xe00000000006c823, 0xf400004000028b0d],
        [0xffffffffffffffff, 0xf400004000028b0d, 0x0a00001ffffde175],
        [0x0000000000000001, 0xf400004000028b0d, 0xff00003000003641],
        [0x0000000000000001, 0xff00003000003641, 0x057ffff7fffed59a],
        [0x0000000000000003, 0xff00003000003641, 0x02bffffbffff6acd],
        [0xffffffffffffffff, 0x02bffffbffff6acd, 0x01dfffe5ffff9a46],
        [0x0000000000000001, 0x02bffffbffff6acd, 0x00effff2ffffcd23],
        [0x0000000000000001, 0x00effff2ffffcd23, 0xff17fffb8000312b],
        [0x0000000000000001, 0xff17fffb8000312b, 0xff14000440003204],
        [0x0000000000000003, 0xff17fffb8000312b, 0xff8a000220001902],
        [0x0000000000000005, 0xff17fffb8000312b, 0xffc5000110000c81],
        [0xfffffffffffffffd, 0xffc5000110000c81, 0x00568002c7ffedab],
        [0xffffffffffffffff, 0xffc5000110000c81, 0x000dc001ebfffd16],
        [0x0000000000000001, 0xffc5000110000c81, 0x0006e000f5fffe8b],
        [0x0000000000000001, 0x0006e000f5fffe8b, 0x0020effff2fff905],
        [0x0000000000000001, 0x0020effff2fff905, 0x000d07ff7e7ffd3d],
        [0x0000000000000001, 0x000d07ff7e7ffd3d, 0xfff60bffc5c0021c],
    ];

    #[test]
    fn divstep_trace() {
        let [two_delta, f0, g0, ..] = DIVSTEP59_VECTORS[0];
        let packed = [two_delta, (f0 & 0xfffff) | 0xfffffe0000000000, (g0 & 0xfffff) | 0xc000000000000000];
        assert_eq!(packed, DIVSTEP_TRACE[0]);
        for (i, pair) in DIVSTEP_TRACE.windows(2).enumerate() {
            let [two_delta, pf, pg] = pair[0];
            assert_eq!(divstep_once(two_delta, pf, pg), pair[1], "step {}", i + 1);
        }
    }

    // The inversion's checks, over this backend's blocks.

    #[test]
    fn sign_mag_known_answers() {
        crate::inversion::tests::sign_mag_known_answers::<Backend>();
    }

    #[test]
    fn divstep59_known_answers() {
        crate::inversion::tests::divstep59_known_answers::<Backend>();
    }

    #[test]
    fn fg_row_known_answers() {
        crate::inversion::tests::fg_row_known_answers::<Backend>();
    }

    #[test]
    fn de_row_known_answers() {
        crate::inversion::tests::de_row_known_answers::<Backend>();
    }

    #[test]
    fn amontred_known_answers() {
        crate::inversion::tests::amontred_known_answers::<Backend>();
    }

    #[test]
    fn cond_sub_known_answers() {
        crate::inversion::tests::cond_sub_known_answers::<Backend>();
    }

    #[test]
    fn invert_known_answers() {
        crate::inversion::tests::invert_known_answers::<Backend>();
    }

    #[test]
    fn invert_small_and_near_modulus() {
        crate::inversion::tests::invert_small_and_near_modulus::<Backend>();
    }

    #[test]
    fn invert_random() {
        crate::inversion::tests::invert_random::<Backend>();
    }

}
