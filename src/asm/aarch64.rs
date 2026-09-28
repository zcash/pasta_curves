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
/// transition matrix in their upper bits, and `d` is the doubled half-delta.
/// The flags hold the parity test of `pg` from the previous step (`tst pg, #1`
/// before the first step of a batch). A step swaps and subtracts when `g` is
/// odd and `d` is positive, adds `f` to `g` when `g` is odd and `d` is not,
/// and does nothing to `g` when it is even; then it halves `g` and adds two
/// to `d`. The parity of the next `g` is tested before the halving, on bit 1,
/// so that the test does not wait for the shift. The `last` arm omits that
/// test, since nothing follows the last step of a batch.
macro_rules! divstep {
    (core) => {
        concat!(
            "csel {t}, {pf}, xzr, ne\n",   // t = g odd ? f : 0.
            "ccmp {d}, xzr, #8, ne\n",     // g odd: flags of d - 0; else N set.
            "cneg {d}, {d}, ge\n",         // g odd and d >= 0: d = -d.
            "cneg {t}, {t}, ge\n",         // g odd and d >= 0: t = -f.
            "csel {pf}, {pg}, {pf}, ge\n", // g odd and d >= 0: f = g.
            "add {pg}, {pg}, {t}\n",       // g = g + t.
            "add {d}, {d}, #2\n",          // d = d + 2.
        )
    };
    () => {
        concat!(
            divstep!(core),
            "tst {pg}, #2\n",              // The parity of the halved g.
            "asr {pg}, {pg}, #1\n",        // g = g / 2.
        )
    };
    (last) => {
        concat!(
            divstep!(core),
            "asr {pg}, {pg}, #1\n",        // g = g / 2.
        )
    };
}

/// Fifty-nine half-delta divsteps on the low words of `f` and `g`, returning
/// the new `d` and the transition matrix.
///
/// `d` is the doubled half-delta as a two's-complement word (`1` at the
/// start), and `f0` and `g0` are the low 64 bits of `f` and `g`, `f` odd. The
/// result is `[d', m00, m01, m10, m11]`: the new `d`, and the 59-step
/// transition matrix `M` with `2^59 (f', g') = M (f, g)`, as two's-complement
/// words. The block is s2n-bignum's `divstep59` macro on named registers: three
/// batches of 20, 20, and 19 steps, each on two words that pack the low 20 bits
/// of `f` and `g` with a row of the batch's matrix in the upper bits (`u` at
/// bit 41 and `v` at bit 62 at the start, halved with each step), whose
/// matrices are read back out of the upper bits, negated, and multiplied
/// together. Between batches the next low words are the matrix rows applied
/// to the current ones, shifted right by the batch's step count.
#[inline(always)]
// Called only by its test until the inversion's driver composes it.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn divstep59(mut d: u64, f0: u64, g0: u64) -> [u64; 5] {
    let (m00, m01, m10, m11): (u64, u64, u64, u64);
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
            "msub {m00}, {b01}, {c10}, {f}",
            "msub {m01}, {b01}, {c11}, {g}",
            "msub {m10}, {b11}, {c10}, {pf}",
            "msub {m11}, {b11}, {c11}, {pg}",
            d = inout(reg) d,
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
            m00 = out(reg) m00,
            m01 = out(reg) m01,
            m10 = out(reg) m10,
            m11 = out(reg) m11,
            options(pure, nomem, nostack),
        );
    }
    [d, m00, m01, m10, m11]
}

/// The sign-magnitude form of a transition matrix: each entry's magnitude,
/// and its sign as a mask (all ones for a negative entry, else zero).
///
/// The row blocks multiply by a negative entry `m` as `|m| · ((x ^ mask) + 1)`,
/// that is by the magnitude times the complement of `x`, with the `+ |m|` folded
/// into the initial carry, so they take the entries in this form. The result is
/// `[m00, m01, m10, m11, s00, s01, s10, s11]`, magnitudes then masks. The entry
/// `-2^63` has no magnitude as a word; the 59-step matrices' entries are below
/// `2^59` in magnitude.
#[inline(always)]
// Called only by its test until the inversion's driver composes it.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn sign_mag(mut m00: u64, mut m01: u64, mut m10: u64, mut m11: u64) -> [u64; 8] {
    let (s00, s01, s10, s11): (u64, u64, u64, u64);
    // SAFETY: register-only arithmetic with declared inputs and outputs;
    // no memory or stack access and no data-dependent control flow.
    unsafe {
        asm!(
            "cmp {m00}, xzr",
            "csetm {s00}, mi",
            "cneg {m00}, {m00}, mi",
            "cmp {m01}, xzr",
            "csetm {s01}, mi",
            "cneg {m01}, {m01}, mi",
            "cmp {m10}, xzr",
            "csetm {s10}, mi",
            "cneg {m10}, {m10}, mi",
            "cmp {m11}, xzr",
            "csetm {s11}, mi",
            "cneg {m11}, {m11}, mi",
            m00 = inout(reg) m00,
            m01 = inout(reg) m01,
            m10 = inout(reg) m10,
            m11 = inout(reg) m11,
            s00 = out(reg) s00,
            s01 = out(reg) s01,
            s10 = out(reg) s10,
            s11 = out(reg) s11,
            options(pure, nomem, nostack),
        );
    }
    [m00, m01, m10, m11, s00, s01, s10, s11]
}

#[cfg(test)]
mod tests {
    use super::{divstep59, sign_mag};
    use core::arch::asm;

    /// The four entries of a transition matrix as words, then the expected magnitudes and
    /// masks, from the round model on both fields.
    const SIGN_MAG_VECTORS: [[u64; 12]; 6] = [
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

    #[test]
    fn sign_mag_known_answers() {
        for row in SIGN_MAG_VECTORS {
            let [m00, m01, m10, m11] = [row[0], row[1], row[2], row[3]];
            assert_eq!(sign_mag(m00, m01, m10, m11), row[4..12]);
        }
    }

    /// One step of the `divstep!` macro on a packed state, for tracing the block step by step:
    /// the parity test that precedes a batch's first step, then the step without the test that
    /// would feed a next one.
    fn divstep_once(mut d: u64, mut pf: u64, mut pg: u64) -> [u64; 3] {
        // SAFETY: register-only arithmetic with declared inputs and outputs.
        unsafe {
            asm!(
                "tst {pg}, #1",
                divstep!(last),
                d = inout(reg) d,
                pf = inout(reg) pf,
                pg = inout(reg) pg,
                t = out(reg) _,
                options(pure, nomem, nostack),
            );
        }
        [d, pf, pg]
    }

    /// The packed state `(d, pf, pg)` before each of the first batch's twenty steps on the
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
        let [d, f0, g0, ..] = DIVSTEP59_VECTORS[0];
        let packed = [d, (f0 & 0xfffff) | 0xfffffe0000000000, (g0 & 0xfffff) | 0xc000000000000000];
        assert_eq!(packed, DIVSTEP_TRACE[0]);
        for (i, pair) in DIVSTEP_TRACE.windows(2).enumerate() {
            let [d, pf, pg] = pair[0];
            assert_eq!(divstep_once(d, pf, pg), pair[1], "step {}", i + 1);
        }
    }

    /// `(d, f0, g0)` and the expected `(d', m00, m01, m10, m11)`, from the integer divstep
    /// recurrence run on the words as integers, which the block agrees with by locality.
    const DIVSTEP59_VECTORS: [[u64; 8]; 6] = [
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

    #[test]
    fn divstep59_known_answers() {
        for [d, f0, g0, d2, m00, m01, m10, m11] in DIVSTEP59_VECTORS {
            assert_eq!(divstep59(d, f0, g0), [d2, m00, m01, m10, m11]);
        }
    }
}
