// Copyright Supranational LLC (the Montgomery routines, transcribed from Semolina v0.1.4).
// Copyright the zakura-core and pasta-asm contributors (the transcription and wrappers).
// SPDX-License-Identifier: Apache-2.0

//! x86-64 backend for the Pasta fields.
//!
//! Modular addition, subtraction, Montgomery multiplication, and squaring are
//! implemented as inline `asm!` blocks. Multiplication uses MULX (BMI2) with
//! ADCX/ADOX dual carry chains (ADX) in the multiplication rows. Two negative
//! scheduling results are pinned here so they are not retried on this
//! microarchitecture family: routing squaring
//! through the multiplication measured 2–5% *slower* (run-dependent) than
//! the dedicated squaring below (21.0 vs 20.0–20.7 ns on Skylake-X —
//! mirroring the AArch64 backend, whose inline square also beats its
//! multiplication; an earlier contrary reading came from a benchmark cell
//! in which the inherent
//! portable `square` shadowed `Field::square`), and merging each
//! Montgomery step's two carry sweeps into interleaved ADCX/ADOX chains
//! (staging all five `q*p` operands flag-free, `TEST` to clear both
//! chains) measured ~10% slower than the two short sequential sweeps
//! (22.3 vs 20.2 ns) despite the shorter nominal dependency length.
//!
//! The round structure is a transcription of the AArch64 backend
//! (`aarch64.rs`), which is itself the upstream Semolina
//! `mul_mont_pasta`: a five-limb CIOS accumulator, one Montgomery
//! cancellation per round, and the shared Pasta modulus shape —
//! `modulus[2] = 0` and `modulus[3] = 2^62` — materialized as shifts, so
//! only `modulus[0]`, `modulus[1]`, and `inv` distinguish Fp from Fq.
//! Because the mathematical structure is identical, the AArch64 module's
//! bounds analysis carries over verbatim; see its module docs for the
//! five-limb no-wrap argument.
//!
//! Unlike the AArch64 blocks, `mul` and `square` address their operand limbs
//! through pointers (`readonly` memory operands) rather than individual
//! registers. This is a necessity, not a scheduling choice: the operands
//! that the AArch64 backend pins in registers — eight operand limbs plus
//! `modulus[0]`, `modulus[1]`, and `inv` — together with the five-limb
//! accumulator, the three staging registers, and MULX's implicit RDX come to
//! twenty simultaneously-live 64-bit registers, and an all-registers
//! transcription of exactly that operand set makes the compiler refuse with
//! "inline assembly requires more registers than available". The memory
//! operands are also why `mul` and `square` — and the public routines
//! composed from them — gate on 64-bit pointers: their blocks bind pointers
//! to registers and use them as full-width addresses, which the x32 ABI's
//! 32-bit pointers would break. `add` and `sub` are register-only and are
//! available on every x86-64 target. The pointers reference the caller's own
//! arrays: there is no packed parameter block to build and no spill stores,
//! only loads that are expected to hit L1. The modulus and inverse travel as
//! separate arguments, as on AArch64: `inv` is bound to a register of its own
//! (measured slightly faster than loading it through memory each round) and
//! `modulus` is read through a pointer to the caller's array. `add` and `sub`
//! fit the budget in register-only form and keep the AArch64 blocks' `nomem`
//! contract.
//!
//! Canonicity contract (same as the AArch64 backend): `rhs` in `mul` and
//! the input of `square` must be canonical (below the modulus) — the
//! five-limb accumulator drops the candidate's would-be fifth limb, and
//! for `rhs >= R - p` the result would be an incorrect residue that still
//! looks canonical. Both routines debug-assert that precondition, and with
//! both operands canonical they are always safe. `lhs` in `mul` may be an
//! unreduced 256-bit value only if every `rhs` limb is at most `2^64 - 4`
//! (the accumulator no-wrap bound) — a condition that is *not* asserted.
//! [`PrimeField::from_repr`](ff::PrimeField::from_repr) passes its decoded,
//! potentially unreduced value as `lhs`, but its `R2` multiplier satisfies
//! the limb bound. `from_u512` continues to use the portable path. The
//! `x86_64_asm_mul_unreduced_lhs_near_modulus_rhs_matches_portable` tests
//! in `fp.rs`/`fq.rs` pin the allowance. Outputs are canonical.
//!
//! The block is straight-line: no branches, no data-dependent memory
//! addresses, and a CMOV-based final conditional subtraction, so the code is
//! constant-time.
//!
//! ISA requirement: `mul` and `square` use MULX (BMI2) and ADCX/ADOX (ADX:
//! Intel Broadwell / AMD Zen or newer). They are not runtime-checked: code
//! built where they are available uses these instructions unconditionally,
//! and running it on an older CPU faults with an illegal instruction. `add`
//! and `sub` use baseline x86-64 instructions only.

use core::arch::asm;

use super::{Limbs, is_canonical};

const PASTA_HIGH_LIMB: u64 = 1 << 62;

/// Adds two canonical residues and conditionally subtracts the modulus.
///
/// Like [`mul`], this hardcodes the Pasta modulus shape (`modulus[2] == 0`).
/// Both inputs must be canonical (debug-asserted). Their sum is below
/// `2 * modulus < 2^256`, so the top carry can be discarded and one
/// conditional subtraction produces a canonical result.
///
/// A register-only counterpart of the AArch64 backend's `add`: all operands
/// arrive in registers and the block declares `nomem`. x86-64's two-operand
/// `sub` destroys its destination, so staging the AArch64 block's four
/// subtraction results in temporaries would take fifteen simultaneously-live
/// registers; the subtraction instead happens in place and the modulus is
/// conditionally added back — the shape of the AArch64 `sub` — with the
/// modulus limbs zeroed in place when no borrow occurred, keeping the block
/// at twelve registers.
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
            "add {r0}, {b0}",
            "adc {r1}, {b1}",
            "adc {r2}, {b2}",
            "adc {r3}, {b3}",
            // Subtract the modulus in place; the canonical sum is below 2p,
            // so the top carry is zero and the tentative difference lies in
            // (-p, p).
            "sub {r0}, {p0}",
            "sbb {r1}, {p1}",
            "sbb {r2}, 0",
            "sbb {r3}, {p3}",
            // A borrow means the sum was below the modulus: add it back.
            // The modulus limbs are zeroed in place when no borrow occurred
            // (MOV and CMOV preserve the borrow flag), which keeps the
            // block at twelve registers.
            "mov {z}, 0",
            "cmovnc {p0}, {z}",
            "cmovnc {p1}, {z}",
            "cmovnc {p3}, {z}",
            "add {r0}, {p0}",
            "adc {r1}, {p1}",
            "adc {r2}, 0",
            "adc {r3}, {p3}",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            b0 = in(reg) rhs[0],
            b1 = in(reg) rhs[1],
            b2 = in(reg) rhs[2],
            b3 = in(reg) rhs[3],
            p0 = inout(reg) modulus[0] => _,
            p1 = inout(reg) modulus[1] => _,
            p3 = inout(reg) modulus[3] => _,
            z = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [r0, r1, r2, r3]
}

/// Subtracts two canonical residues, adding the modulus back on underflow.
///
/// Like [`add`] and [`mul`], this hardcodes the Pasta modulus shape
/// (`modulus[2] == 0`). The difference lies strictly between `-modulus` and
/// `modulus`, so one conditional addition produces a canonical result.
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
            "sub {r0}, {b0}",
            "sbb {r1}, {b1}",
            "sbb {r2}, {b2}",
            "sbb {r3}, {b3}",
            // A borrow means the difference went below zero: add the modulus
            // back. The modulus limbs are zeroed in place when no borrow
            // occurred (MOV and CMOV preserve the borrow flag), which keeps
            // the block at twelve registers.
            "mov {z}, 0",
            "cmovnc {p0}, {z}",
            "cmovnc {p1}, {z}",
            "cmovnc {p3}, {z}",
            "add {r0}, {p0}",
            "adc {r1}, {p1}",
            "adc {r2}, 0",
            "adc {r3}, {p3}",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            b0 = in(reg) rhs[0],
            b1 = in(reg) rhs[1],
            b2 = in(reg) rhs[2],
            b3 = in(reg) rhs[3],
            p0 = inout(reg) modulus[0] => _,
            p1 = inout(reg) modulus[1] => _,
            p3 = inout(reg) modulus[3] => _,
            z = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [r0, r1, r2, r3]
}

/// Multiplies two Montgomery residues for a Pasta modulus. `rhs` must be
/// canonical (debug-asserted; a violation yields an incorrect residue, see
/// the module docs). `lhs` may be unreduced only if every `rhs` limb is at
/// most `2^64 - 4`; see the AArch64 module docs for the carry-chain bound
/// behind this.
// Keep the assembly behind a call boundary. It consumes nearly every x86-64
// register; forcing it into a register-heavy caller can make allocation
// impossible instead of merely causing spills.
#[inline(never)]
pub(super) fn mul(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    debug_assert!(
        is_canonical(rhs, modulus),
        "pasta_curves::asm::mul requires a canonical rhs"
    );
    let (o0, o1, o2, o3): (u64, u64, u64, u64);
    // SAFETY: straight-line arithmetic reading only the twelve words
    // behind the three passed references (`readonly`): four limbs each from
    // `lhs` and `rhs` (addressed through the base-plus-displacement form
    // that AArch64's `mul` provides in registers) and the four modulus
    // limbs, with `inv` arriving in a register of its own. No stack use, and
    // outputs depend only on the declared inputs.
    //
    // Register roles: the five-limb accumulator lives in {ae}/{be}/{ce}/
    // {de}/{ee} and its window rotates down one register per round (the
    // register cancelled by the round's Montgomery step becomes the next
    // round's fifth limb), so after four rounds the candidate sits in
    // {ee},{ae},{be},{ce}. {s1}/{s2}/{s3} stage multiplier halves and
    // shifted `q * modulus[3]` terms so no flag-writing instruction lands
    // inside a carry chain. RDX is the implicit MULX source: each round's
    // `rhs` limb, then the round's Montgomery factor `q`.
    unsafe {
        asm!(
            // Round 0: initialize the accumulator with lhs * rhs[0].
            "mov rdx, qword ptr [{b}]",          // rdx = b[0].
            // ae = low(a[0]*b[0]); be = high.
            "mulx {be}, {ae}, qword ptr [{a}]",
            "mulx {ce}, {s1}, qword ptr [{a} + 8]",
            // Fold low(a[1]*b[0]) into limb 1.
            "add {be}, {s1}",
            "mulx {de}, {s1}, qword ptr [{a} + 16]",
            "adc {ce}, {s1}",                    // Fold low(a[2]*b[0]) and carry.
            "mulx {ee}, {s1}, qword ptr [{a} + 24]",
            "adc {de}, {s1}",                    // Fold low(a[3]*b[0]) and carry.
            "adc {ee}, 0",                       // Fifth limb of lhs * b[0].

            // Montgomery step 0: q = limb0 * inv; add q*p; shift one limb.
            "mov rdx, {ae}",
            "imul rdx, {inv}",                   // rdx = q (low 64 bits only).
            // s1 = low(q*p[1]); s2 = high (kept).
            "mulx {s2}, {s1}, qword ptr [{p} + 8]",
            "mov {s3}, rdx",
            // s3 = low(q*p[3]); p[2] contributes nothing.
            "shl {s3}, 62",
            // low(q*p[0]) cancels limb 0; its carry is one exactly when the
            // limb is nonzero, which NEG leaves in CF.
            "neg {ae}",                          // CF = (limb0 != 0); ae is dead.
            // Add low(q*p[1]) and the cancellation carry.
            "adc {be}, {s1}",
            "adc {ce}, 0",                       // Propagate across zero p[2].
            "adc {de}, {s3}",                    // Add low(q*p[3]) and carry.
            "adc {ee}, 0",                       // Propagate into the fifth limb.
            // s1 = high(q*p[0]); the low half is spent.
            "mulx {s1}, {s3}, qword ptr [{p}]",
            "mov {s3}, rdx",
            "shr {s3}, 2",                       // s3 = high(q*p[3]).
            // Next round's fifth limb (MOV keeps flags).
            "mov {ae}, 0",
            // New limbs include the high halves of q*p.
            "add {be}, {s1}",
            "adc {ce}, {s2}",
            "adc {de}, 0",                       // p[2] contributes zero.
            "adc {ee}, {s3}",
            // Capture the reduction carry as limb 4.
            "adc {ae}, 0",

            // Round 1: accumulator window is [be,ce,de,ee,ae]; add lhs*b[1]
            // on dual carry chains (CF: low halves, OF: high halves).
            "mov rdx, qword ptr [{b} + 8]",      // rdx = b[1].
            "xor {s1}, {s1}",                    // Clear CF and OF.
            "mulx {s2}, {s1}, qword ptr [{a}]",
            "adcx {be}, {s1}",
            "adox {ce}, {s2}",
            "mulx {s2}, {s1}, qword ptr [{a} + 8]",
            "adcx {ce}, {s1}",
            "adox {de}, {s2}",
            "mulx {s2}, {s1}, qword ptr [{a} + 16]",
            "adcx {de}, {s1}",
            "adox {ee}, {s2}",
            "mulx {s2}, {s1}, qword ptr [{a} + 24]",
            "adcx {ee}, {s1}",
            "adox {ae}, {s2}",
            "mov {s1}, 0",
            // Close both carry chains into limb 4.
            "adcx {ae}, {s1}",
            "adox {ae}, {s1}",

            // Montgomery step 1.
            "mov rdx, {be}",
            "imul rdx, {inv}",
            "mulx {s2}, {s1}, qword ptr [{p} + 8]",
            "mov {s3}, rdx",
            "shl {s3}, 62",
            "neg {be}",
            "adc {ce}, {s1}",
            "adc {de}, 0",
            "adc {ee}, {s3}",
            "adc {ae}, 0",
            "mulx {s1}, {s3}, qword ptr [{p}]",
            "mov {s3}, rdx",
            "shr {s3}, 2",
            "mov {be}, 0",
            "add {ce}, {s1}",
            "adc {de}, {s2}",
            "adc {ee}, 0",
            "adc {ae}, {s3}",
            "adc {be}, 0",

            // Round 2: window [ce,de,ee,ae,be]; add lhs*b[2].
            "mov rdx, qword ptr [{b} + 16]",
            "xor {s1}, {s1}",
            "mulx {s2}, {s1}, qword ptr [{a}]",
            "adcx {ce}, {s1}",
            "adox {de}, {s2}",
            "mulx {s2}, {s1}, qword ptr [{a} + 8]",
            "adcx {de}, {s1}",
            "adox {ee}, {s2}",
            "mulx {s2}, {s1}, qword ptr [{a} + 16]",
            "adcx {ee}, {s1}",
            "adox {ae}, {s2}",
            "mulx {s2}, {s1}, qword ptr [{a} + 24]",
            "adcx {ae}, {s1}",
            "adox {be}, {s2}",
            "mov {s1}, 0",
            "adcx {be}, {s1}",
            "adox {be}, {s1}",

            // Montgomery step 2.
            "mov rdx, {ce}",
            "imul rdx, {inv}",
            "mulx {s2}, {s1}, qword ptr [{p} + 8]",
            "mov {s3}, rdx",
            "shl {s3}, 62",
            "neg {ce}",
            "adc {de}, {s1}",
            "adc {ee}, 0",
            "adc {ae}, {s3}",
            "adc {be}, 0",
            "mulx {s1}, {s3}, qword ptr [{p}]",
            "mov {s3}, rdx",
            "shr {s3}, 2",
            "mov {ce}, 0",
            "add {de}, {s1}",
            "adc {ee}, {s2}",
            "adc {ae}, 0",
            "adc {be}, {s3}",
            "adc {ce}, 0",

            // Round 3: window [de,ee,ae,be,ce]; add lhs*b[3].
            "mov rdx, qword ptr [{b} + 24]",
            "xor {s1}, {s1}",
            "mulx {s2}, {s1}, qword ptr [{a}]",
            "adcx {de}, {s1}",
            "adox {ee}, {s2}",
            "mulx {s2}, {s1}, qword ptr [{a} + 8]",
            "adcx {ee}, {s1}",
            "adox {ae}, {s2}",
            "mulx {s2}, {s1}, qword ptr [{a} + 16]",
            "adcx {ae}, {s1}",
            "adox {be}, {s2}",
            "mulx {s2}, {s1}, qword ptr [{a} + 24]",
            "adcx {be}, {s1}",
            "adox {ce}, {s2}",
            "mov {s1}, 0",
            "adcx {ce}, {s1}",
            "adox {ce}, {s1}",

            // Montgomery step 3. Canonical rhs bounds the candidate below
            // 2p < R, so the final shift produces no fifth limb (see the
            // AArch64 module docs); the shift's carry adc is omitted.
            "mov rdx, {de}",
            "imul rdx, {inv}",
            "mulx {s2}, {s1}, qword ptr [{p} + 8]",
            "mov {s3}, rdx",
            "shl {s3}, 62",
            "neg {de}",
            "adc {ee}, {s1}",
            "adc {ae}, 0",
            "adc {be}, {s3}",
            "adc {ce}, 0",
            "mulx {s1}, {s3}, qword ptr [{p}]",
            "mov {s3}, rdx",
            "shr {s3}, 2",
            "add {ee}, {s1}",                    // Final candidate limb 0.
            "adc {ae}, {s2}",                    // Final candidate limb 1.
            "adc {be}, 0",                       // Final candidate limb 2.
            "adc {ce}, {s3}",                    // Final candidate limb 3.

            // Conditional subtraction of p = [p0, p1, 0, 2^62].
            "movabs rdx, {p3}",                  // Materialize p[3] = 2^62.
            "mov {s1}, {ee}",
            "mov {s2}, {ae}",
            "mov {s3}, {be}",
            "mov {de}, {ce}",
            // Tentative limb 0 = candidate - p[0].
            "sub {s1}, qword ptr [{p}]",
            "sbb {s2}, qword ptr [{p} + 8]",     // Tentative limb 1 minus p[1].
            "sbb {s3}, 0",                       // Limb 2; p[2] is zero.
            "sbb {de}, rdx",                     // Tentative limb 3 minus p[3].
            // No borrow (CF clear) means the candidate is at least p, so the
            // subtracted value is the canonical output.
            "cmovnc {ee}, {s1}",
            "cmovnc {ae}, {s2}",
            "cmovnc {be}, {s3}",
            "cmovnc {ce}, {de}",
            a = in(reg) lhs.as_ptr(),
            b = in(reg) rhs.as_ptr(),
            p = in(reg) modulus.as_ptr(),
            inv = in(reg) inv,
            p3 = const PASTA_HIGH_LIMB,
            ae = out(reg) o1,
            be = out(reg) o2,
            ce = out(reg) o3,
            de = out(reg) _,
            ee = out(reg) o0,
            s1 = out(reg) _,
            s2 = out(reg) _,
            s3 = out(reg) _,
            out("rdx") _,
            options(pure, readonly, nostack),
        );
    }
    [o0, o1, o2, o3]
}

/// Squares a canonical Montgomery residue for a Pasta modulus (the input's
/// canonicity is debug-asserted).
///
/// A transcription of the AArch64 backend's dedicated squaring: the 512-bit
/// square as cross products, one doubling pass, and the diagonals (ten MULX
/// against the multiplication's sixteen), then four Montgomery
/// cancellations on a rotating four-limb window with a carried fifth limb,
/// the high product half folded in (the sum stays below `2p`, so no carry
/// escapes — see the AArch64 module's bounds), and a CMOV conditional
/// subtraction. Measured 2–5% ahead of squaring through [`mul`] on
/// Skylake-X (20.0–20.7 vs 21.0 ns across runs), mirroring the AArch64
/// backend's own square-over-mul margin.
///
/// Kept behind a call boundary for the register-allocation reason documented
/// on [`mul`].
#[inline(never)]
pub(super) fn square(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    debug_assert!(
        is_canonical(value, modulus),
        "pasta_curves::asm::square requires a canonical input"
    );
    let (o0, o1, o2, o3): (u64, u64, u64, u64);
    // SAFETY: straight-line arithmetic reading only the eight words
    // behind the two passed references (`readonly`): the input's four limbs
    // and the four modulus limbs, with `inv` arriving in a register of its
    // own. No stack use, and outputs depend only on the declared inputs.
    // The input pointer's register is reclaimed as the reduction's carry
    // limb once phase 1 has consumed the last load.
    unsafe {
        asm!(
            // Phase 1: the 512-bit square in z0..z7.
            // Cross products a[i]*a[j] (i < j), accumulated as they stream.
            "xor {z5:e}, {z5:e}",
            "xor {z6:e}, {z6:e}",
            "xor {z7:e}, {z7:e}",
            "mov rdx, qword ptr [{a}]",
            "mulx {t1}, {z1}, qword ptr [{a} + 8]",  // a0*a1.
            "mulx {t2}, {z2}, qword ptr [{a} + 16]", // a0*a2.
            "mulx {z4}, {z3}, qword ptr [{a} + 24]", // a0*a3.
            "add {z2}, {t1}",                        // Fold high(a0*a1).
            // Fold high(a0*a2) and carry.
            "adc {z3}, {t2}",
            "adc {z4}, 0",
            "mov rdx, qword ptr [{a} + 8]",
            "mulx {t2}, {t1}, qword ptr [{a} + 16]", // a1*a2.
            "add {z3}, {t1}",
            "adc {z4}, {t2}",
            "adc {z5}, 0",
            "mulx {t2}, {t1}, qword ptr [{a} + 24]", // a1*a3.
            "add {z4}, {t1}",
            "adc {z5}, {t2}",
            "adc {z6}, 0",
            "mov rdx, qword ptr [{a} + 16]",
            "mulx {t2}, {t1}, qword ptr [{a} + 24]", // a2*a3.
            "add {z5}, {t1}",
            "adc {z6}, {t2}",
            "adc {z7}, 0",
            // Double the cross products. The doubled sum is below 2^512, so
            // no carry leaves z7.
            "add {z1}, {z1}",
            "adc {z2}, {z2}",
            "adc {z3}, {z3}",
            "adc {z4}, {z4}",
            "adc {z5}, {z5}",
            "adc {z6}, {z6}",
            "adc {z7}, {z7}",
            // Add the diagonal squares in one carry chain (MOV and MULX
            // preserve flags).
            "mov rdx, qword ptr [{a}]",
            "mulx {t2}, {z0}, rdx",                  // z0 = low(a0^2).
            "add {z1}, {t2}",                        // High(a0^2).
            "mov rdx, qword ptr [{a} + 8]",
            "mulx {t2}, {t1}, rdx",
            "adc {z2}, {t1}",
            "adc {z3}, {t2}",
            "mov rdx, qword ptr [{a} + 16]",
            "mulx {t2}, {t1}, rdx",
            "adc {z4}, {t1}",
            "adc {z5}, {t2}",
            "mov rdx, qword ptr [{a} + 24]",
            "mulx {t2}, {t1}, rdx",
            "adc {z6}, {t1}",
            "adc {z7}, {t2}",                        // a^2 < 2^510: no carry out.

            // Phase 2: four Montgomery cancellations on the low half, the
            // same two-sweep step as [`mul`]'s. The window rotates down one
            // register per step; {a} (its loads are done) serves as the
            // first carried fifth limb.
            // Step 0: window [z0, z1, z2, z3], carry into {a}.
            "mov rdx, {z0}",
            "imul rdx, {inv}",                       // rdx = q.
            "mulx {t2}, {t1}, qword ptr [{p} + 8]",  // t1/t2 = low/high(q*p1).
            "mov {a}, rdx",
            "shl {a}, 62",                           // low(q*p3); p2 is zero.
            "neg {z0}",                              // CF = (limb0 != 0).
            "adc {z1}, {t1}",
            "adc {z2}, 0",
            "adc {z3}, {a}",
            "mov {a}, 0",
            "adc {a}, 0",                            // Carry above limb 3.
            // t1 = high(q*p0); low is spent.
            "mulx {t1}, {z0}, qword ptr [{p}]",
            "mov {z0}, rdx",
            "shr {z0}, 2",                           // high(q*p3).
            "add {z1}, {t1}",                        // New limb 0.
            "adc {z2}, {t2}",                        // New limb 1 += high(q*p1).
            "adc {z3}, 0",                           // New limb 2.
            "adc {a}, {z0}",                         // New limb 3 += high(q*p3).
            // Step 1: window [z1, z2, z3, a], carry into z0.
            "mov rdx, {z1}",
            "imul rdx, {inv}",
            "mulx {t2}, {t1}, qword ptr [{p} + 8]",
            "mov {z0}, rdx",
            "shl {z0}, 62",
            "neg {z1}",
            "adc {z2}, {t1}",
            "adc {z3}, 0",
            "adc {a}, {z0}",
            "mov {z0}, 0",
            "adc {z0}, 0",
            "mulx {t1}, {z1}, qword ptr [{p}]",
            "mov {z1}, rdx",
            "shr {z1}, 2",
            "add {z2}, {t1}",
            "adc {z3}, {t2}",
            "adc {a}, 0",
            "adc {z0}, {z1}",
            // Step 2: window [z2, z3, a, z0], carry into z1.
            "mov rdx, {z2}",
            "imul rdx, {inv}",
            "mulx {t2}, {t1}, qword ptr [{p} + 8]",
            "mov {z1}, rdx",
            "shl {z1}, 62",
            "neg {z2}",
            "adc {z3}, {t1}",
            "adc {a}, 0",
            "adc {z0}, {z1}",
            "mov {z1}, 0",
            "adc {z1}, 0",
            "mulx {t1}, {z2}, qword ptr [{p}]",
            "mov {z2}, rdx",
            "shr {z2}, 2",
            "add {z3}, {t1}",
            "adc {a}, {t2}",
            "adc {z0}, 0",
            "adc {z1}, {z2}",
            // Step 3: window [z3, a, z0, z1], carry into z2.
            "mov rdx, {z3}",
            "imul rdx, {inv}",
            "mulx {t2}, {t1}, qword ptr [{p} + 8]",
            "mov {z2}, rdx",
            "shl {z2}, 62",
            "neg {z3}",
            "adc {a}, {t1}",
            "adc {z0}, 0",
            "adc {z1}, {z2}",
            "mov {z2}, 0",
            "adc {z2}, 0",
            "mulx {t1}, {z3}, qword ptr [{p}]",
            "mov {z3}, rdx",
            "shr {z3}, 2",
            "add {a}, {t1}",
            "adc {z0}, {t2}",
            "adc {z1}, 0",
            "adc {z2}, {z3}",

            // Fold in the high product half; the sum stays below 2p, so no
            // carry escapes and a four-limb conditional subtraction suffices.
            "add {a}, {z4}",
            "adc {z0}, {z5}",
            "adc {z1}, {z6}",
            "adc {z2}, {z7}",
            "movabs rdx, {p3}",                      // p3 = 2^62.
            "mov {t1}, {a}",
            "mov {t2}, {z0}",
            "mov {z3}, {z1}",
            "mov {z4}, {z2}",
            "sub {t1}, qword ptr [{p}]",
            "sbb {t2}, qword ptr [{p} + 8]",
            "sbb {z3}, 0",
            "sbb {z4}, rdx",
            "cmovnc {a}, {t1}",
            "cmovnc {z0}, {t2}",
            "cmovnc {z1}, {z3}",
            "cmovnc {z2}, {z4}",
            a = inout(reg) value.as_ptr() => o0,
            p = in(reg) modulus.as_ptr(),
            inv = in(reg) inv,
            p3 = const PASTA_HIGH_LIMB,
            z0 = out(reg) o1,
            z1 = out(reg) o2,
            z2 = out(reg) o3,
            z3 = out(reg) _,
            z4 = out(reg) _,
            z5 = out(reg) _,
            z6 = out(reg) _,
            z7 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            out("rdx") _,
            options(pure, readonly, nostack),
        );
    }
    [o0, o1, o2, o3]
}
