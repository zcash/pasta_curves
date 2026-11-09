// Copyright Supranational LLC (the Montgomery routines, transcribed from Semolina v0.1.4).
// Copyright the zakura-core and pasta_curves contributors (the transcription and wrappers).

//! x86-64 backend for the Pasta fields.
//!
//! Modular addition, subtraction, Montgomery multiplication, and squaring are
//! implemented as inline `asm!` blocks. Multiplication uses MULX (BMI2) with
//! ADCX/ADOX dual carry chains (ADX) in the multiplication rows. Two negative
//! scheduling results are pinned here so they are not retried on this
//! microarchitecture family: routing squaring
//! through the multiplication measured 1–5% *slower* (run-dependent) than
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
//! The arithmetic is that of the AArch64 backend (`aarch64.rs`), which is
//! itself Semolina's `mul_mont_pasta`. It keeps a five-limb CIOS
//! accumulator (the module's README describes the form) and makes one
//! Montgomery cancellation per round. It relies on the shared Pasta modulus
//! shape, `modulus[2] = 0` and `modulus[3] = 2^62`, materialized as shifts,
//! so only `modulus[0]`, `modulus[1]`, and `inv` distinguish Fp from Fq.
//! The instruction schedule is x86-64's own: two carry chains, and a final
//! round fused with the conditional subtraction. The bounds are the same,
//! though. The AArch64 `mul`'s documentation gives the five-limb no-wrap
//! argument, and the x86-64 proofs establish it for this transcription
//! under the same hypotheses.
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
//! operands are also why the backend requires 64-bit pointers: the
//! multiplication and squaring blocks bind pointers to registers and use
//! them as full-width addresses, which the x32 ABI's 32-bit pointers would
//! break, so the module has no backend on that target although `add`, `sub`,
//! and `from_mont` are register-only. The pointers reference the caller's own
//! arrays: there is no packed parameter block to build and no spill stores,
//! only loads that are expected to hit L1. The modulus and inverse travel as
//! separate arguments, as on AArch64: `inv` is bound to a register of its own
//! (measured slightly faster than loading it through memory each round) and
//! `modulus` is read through a pointer to the caller's array. `add` and `sub`
//! fit the budget in register-only form and keep the AArch64 blocks' `nomem`
//! contract.
//!
//! Operand contracts (the same as the AArch64 backend's): `mul` is exact for
//! a canonical `lhs` with any `rhs`, or for a canonical `rhs` whose limbs 1
//! to 3 are at most `2^64 - 3` with any `lhs`. The public entry point
//! debug-asserts that disjunction, and the Lean proofs establish both
//! contracts (`mulMont_spec_of_lhs_lt` and `mulMont_spec_of_rhs_lt`).
//! Outside both, the accumulator can wrap, or the dropped fifth limb of the
//! final candidate can be nonzero, and the result is then an incorrect
//! residue that still looks canonical. `square` needs a canonical input,
//! which its entry point debug-asserts. Outputs are canonical.
//!
//! The blocks are straight-line: no branches, no data-dependent memory
//! addresses, and CMOV-based final conditional subtractions, so the code
//! should be constant-time, unless behaviour of the Rust toolchain or
//! platform introduces an unexpected obstacle to that.
//!
//! ISA requirement: `mul` and `square` use MULX (BMI2) and ADCX/ADOX (ADX:
//! Intel Broadwell / AMD Zen or newer). `from_mont` uses MULX (BMI2) alone.
//! None of these are runtime-checked: the backend uses the instructions
//! unconditionally, and running it on an older CPU faults with an illegal
//! instruction; build with `--cfg pasta_curves_noasm` for such CPUs. `add` and
//! `sub` use baseline x86-64 instructions only.

use core::arch::asm;

use super::Limbs;
use crate::inversion::InvertBlocks;

const PASTA_HIGH_LIMB: u64 = 1 << 62;

/// Adds two canonical residues and conditionally subtracts the modulus.
///
/// Like [`mul`], this hardcodes the Pasta modulus shape (`modulus[2] == 0`). Both inputs
/// must be canonical, which the entry point in `entry.rs` debug-asserts. Their sum is
/// below `2 * modulus < 2^256`, so the top carry can be discarded and one conditional
/// subtraction produces a canonical result.
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
            // [-p, p).
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
/// Like [`add`] and [`mul`], this hardcodes the Pasta modulus shape (`modulus[2] == 0`).
/// Both inputs must be canonical, which the entry point in `entry.rs` debug-asserts. The
/// difference then lies strictly between `-modulus` and `modulus`, so one conditional
/// addition produces a canonical result.
#[inline(always)]
pub(super) fn sub(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
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

/// Multiplies two Montgomery residues for a Pasta modulus, under the same two contracts
/// as the AArch64 block, which the entry point in `entry.rs` debug-asserts: a canonical
/// `lhs` with any `rhs`, or a canonical `rhs` whose limbs 1 to 3 are at most `2^64 - 3`
/// with any `lhs`. The instruction schedule differs from AArch64's (two carry chains, and
/// a final round fused with the conditional subtraction), but each round is the same
/// five-limb CIOS step, and the Lean proof establishes both contracts for this
/// transcription under the same round bounds as the AArch64 proof
/// (`mulMont_spec_of_lhs_lt` and `mulMont_spec_of_rhs_lt` in `X86_64/Spec/Mul.lean`).
// Keep the assembly behind a call boundary. It consumes nearly every x86-64
// register; forcing it into a register-heavy caller can make allocation
// impossible instead of merely causing spills.
#[inline(never)]
pub(super) fn mul(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
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

            // Montgomery step 3. Either contract bounds the candidate below
            // 2p < R, so the final shift produces no fifth limb (see the
            // AArch64 `mul`'s documentation); the shift's carry adc is omitted.
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

/// Phase one of squaring: the 512-bit product as cross products, one
/// doubling pass, and the diagonals (ten MULX against the multiplication's
/// sixteen). Always inlined so that the repeated-squaring loop of
/// `sqr_n_mul` in `entry.rs` can pass the accumulator in registers. The
/// block occupies fifteen simultaneous registers, which is the most the
/// compiler will allocate to one block in both the release and dev profiles.
///
/// # Safety
///
/// The block is straight-line register-only arithmetic with declared
/// inputs and outputs; no memory or stack access.
#[inline(always)]
fn square_lo(value: Limbs) -> [u64; 8] {
    let (z0, z1, z2, z3, z4, z5, z6, z7): (u64, u64, u64, u64, u64, u64, u64, u64);
    let [a0, a1, a2, a3] = value;
    // SAFETY: straight-line register-only arithmetic with declared inputs
    // and outputs; no memory or stack access, and outputs depend only on
    // the declared inputs.
    unsafe {
        asm!(
            // Phase 1: the 512-bit square in z0..z7.
            // Cross products a[i]*a[j] (i < j), accumulated as they stream.
            "xor {z5:e}, {z5:e}",
            "xor {z6:e}, {z6:e}",
            "xor {z7:e}, {z7:e}",
            "mov rdx, {a0}",
            "mulx {t1}, {z1}, {a1}",              // a0*a1.
            "mulx {t2}, {z2}, {a2}",              // a0*a2.
            "mulx {z4}, {z3}, {a3}",              // a0*a3.
            "add {z2}, {t1}",                        // Fold high(a0*a1).
            // Fold high(a0*a2) and carry.
            "adc {z3}, {t2}",
            "adc {z4}, 0",
            "mov rdx, {a1}",
            "mulx {t2}, {t1}, {a2}",              // a1*a2.
            "add {z3}, {t1}",
            "adc {z4}, {t2}",
            "adc {z5}, 0",
            "mulx {t2}, {t1}, {a3}",              // a1*a3.
            "add {z4}, {t1}",
            "adc {z5}, {t2}",
            "adc {z6}, 0",
            "mov rdx, {a2}",
            "mulx {t2}, {t1}, {a3}",              // a2*a3.
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
            "mov rdx, {a0}",
            "mulx {t2}, {z0}, rdx",                  // z0 = low(a0^2).
            "add {z1}, {t2}",                        // High(a0^2).
            "mov rdx, {a1}",
            "mulx {t2}, {t1}, rdx",
            "adc {z2}, {t1}",
            "adc {z3}, {t2}",
            "mov rdx, {a2}",
            "mulx {t2}, {t1}, rdx",
            "adc {z4}, {t1}",
            "adc {z5}, {t2}",
            "mov rdx, {a3}",
            "mulx {t2}, {t1}, rdx",
            "adc {z6}, {t1}",
            "adc {z7}, {t2}",                        // a^2 < 2^510: no carry out.
            a0 = in(reg) a0,
            a1 = in(reg) a1,
            a2 = in(reg) a2,
            a3 = in(reg) a3,
            z0 = out(reg) z0,
            z1 = out(reg) z1,
            z2 = out(reg) z2,
            z3 = out(reg) z3,
            z4 = out(reg) z4,
            z5 = out(reg) z5,
            z6 = out(reg) z6,
            z7 = out(reg) z7,
            t1 = out(reg) _,
            t2 = out(reg) _,
            out("rdx") _,
            options(pure, nomem, nostack),
        );
    }
    [z0, z1, z2, z3, z4, z5, z6, z7]
}

/// Phase two of squaring: four Montgomery cancellations on the rotating
/// four-limb window of the low product half, the high product half folded
/// in (the sum stays below `2p`, so no carry escapes; see `square` in
/// [`Backend`]), and a CMOV conditional subtraction. Always inlined like
/// [`square_lo`], for the same loop-accumulator reason.
///
/// # Safety
///
/// `product` must be the square of a canonical value. The memory operand
/// reads the four modulus limbs.
#[inline(always)]
fn square_hi(product: [u64; 8], modulus: &Limbs, inv: u64) -> Limbs {
    let (o0, o1, o2, o3): (u64, u64, u64, u64);
    // SAFETY: straight-line arithmetic reading only the four modulus words
    // behind the passed reference (`readonly`), with `inv` arriving in a
    // register of its own. No stack use, and outputs depend only on the
    // declared inputs.
    unsafe {
        asm!(
            // Phase 2: four Montgomery cancellations on the low half, the
            // same two-sweep step as [`mul`]'s. The window rotates down one
            // register per step; the fifth limb alternates between the two
            // otherwise-dead high product limbs.
            // Step 0: window [z0, z1, z2, z3], carry into a.
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
            // Window limbs: each z register is both an input (a product limb)
            // and a rewritten accumulator position; z3 and z4 end as scratch.
            a = out(reg) o0,
            p = in(reg) modulus.as_ptr(),
            inv = in(reg) inv,
            p3 = const PASTA_HIGH_LIMB,
            z0 = inout(reg) product[0] => o1,
            z1 = inout(reg) product[1] => o2,
            z2 = inout(reg) product[2] => o3,
            z3 = inout(reg) product[3] => _,
            z4 = inout(reg) product[4] => _,
            z5 = in(reg) product[5],
            z6 = in(reg) product[6],
            z7 = in(reg) product[7],
            t1 = out(reg) _,
            t2 = out(reg) _,
            out("rdx") _,
            options(pure, readonly, nostack),
        );
    }
    [o0, o1, o2, o3]
}

/// Converts a Montgomery residue into its canonical integer, as a
/// multiplication by one: the four Montgomery cancellations of the squaring
/// step on the rotating window of `value` itself, then an in-place
/// conditional subtraction of the modulus.
///
/// A multiplication by one has no product rows: `value` is its own
/// accumulator, so the whole reduction fits registers without memory
/// operands (`nomem`), unlike [`mul`]. The result's bounds follow the
/// squaring step's: the input value plays the role a canonical right
/// operand's product plays there, so the final candidate stays below `2p`
/// for any four-limb input and the four-limb conditional subtraction
/// suffices.
///
/// # Safety
///
/// `modulus` must be a Pasta modulus and `inv` its derived inverse. Any
/// four-limb `value` is accepted.
#[inline(always)]
pub(super) fn from_mont(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    let (o0, o1, o2, o3): (u64, u64, u64, u64);
    // SAFETY: straight-line register-only arithmetic with declared inputs
    // and outputs; no memory or stack access, and outputs depend only on
    // the declared inputs.
    unsafe {
        asm!(
            // Four Montgomery cancellations on the rotating window of
            // value itself; the window rotates down one register per step
            // and the fifth limb is carried in a scratch register.
            // Step 0: window [z0, z1, z2, z3], carry into a.
            "mov rdx, {z0}",
            "imul rdx, {inv}",                       // rdx = q.
            "mulx {t2}, {t1}, {p1}",                 // t1/t2 = low/high(q*p1).
            "mov {a}, rdx",
            "shl {a}, 62",                           // low(q*p3); p2 is zero.
            "neg {z0}",                              // CF = (limb0 != 0).
            "adc {z1}, {t1}",
            "adc {z2}, 0",
            "adc {z3}, {a}",
            "mov {a}, 0",
            "adc {a}, 0",                            // Carry above limb 3.
            // t1 = high(q*p0); low is spent.
            "mulx {t1}, {z0}, {p0}",
            "mov {z0}, rdx",
            "shr {z0}, 2",                           // high(q*p3).
            "add {z1}, {t1}",                        // New limb 0.
            "adc {z2}, {t2}",                        // New limb 1 += high(q*p1).
            "adc {z3}, 0",                           // New limb 2.
            "adc {a}, {z0}",                         // New limb 3 += high(q*p3).
            // Step 1: window [z1, z2, z3, a], carry into z0.
            "mov rdx, {z1}",
            "imul rdx, {inv}",
            "mulx {t2}, {t1}, {p1}",
            "mov {z0}, rdx",
            "shl {z0}, 62",
            "neg {z1}",
            "adc {z2}, {t1}",
            "adc {z3}, 0",
            "adc {a}, {z0}",
            "mov {z0}, 0",
            "adc {z0}, 0",
            "mulx {t1}, {z1}, {p0}",
            "mov {z1}, rdx",
            "shr {z1}, 2",
            "add {z2}, {t1}",
            "adc {z3}, {t2}",
            "adc {a}, 0",
            "adc {z0}, {z1}",
            // Step 2: window [z2, z3, a, z0], carry into z1.
            "mov rdx, {z2}",
            "imul rdx, {inv}",
            "mulx {t2}, {t1}, {p1}",
            "mov {z1}, rdx",
            "shl {z1}, 62",
            "neg {z2}",
            "adc {z3}, {t1}",
            "adc {a}, 0",
            "adc {z0}, {z1}",
            "mov {z1}, 0",
            "adc {z1}, 0",
            "mulx {t1}, {z2}, {p0}",
            "mov {z2}, rdx",
            "shr {z2}, 2",
            "add {z3}, {t1}",
            "adc {a}, {t2}",
            "adc {z0}, 0",
            "adc {z1}, {z2}",
            // Step 3: window [z3, a, z0, z1], carry into z2.
            "mov rdx, {z3}",
            "imul rdx, {inv}",
            "mulx {t2}, {t1}, {p1}",
            "mov {z2}, rdx",
            "shl {z2}, 62",
            "neg {z3}",
            "adc {a}, {t1}",
            "adc {z0}, 0",
            "adc {z1}, {z2}",
            "mov {z2}, 0",
            "adc {z2}, 0",
            "mulx {t1}, {z3}, {p0}",
            "mov {z3}, rdx",
            "shr {z3}, 2",
            "add {a}, {t1}",
            "adc {z0}, {t2}",
            "adc {z1}, 0",
            "adc {z2}, {z3}",

            // Conditional subtraction of p = [p0, p1, 0, 2^62]. The
            // candidate sits in [a, z0, z1, z2]; the top limb is staged
            // through z4. The candidate stays below 2p, so no carry escapes
            // and four limbs suffice.
            "movabs rdx, {p3}",                      // Materialize p3 = 2^62.
            "mov {t1}, {a}",
            "mov {t2}, {z0}",
            "mov {z3}, {z1}",
            "mov {z4}, {z2}",
            "sub {t1}, {p0}",
            "sbb {t2}, {p1}",
            "sbb {z3}, 0",
            "sbb {z4}, rdx",
            "cmovnc {a}, {t1}",
            "cmovnc {z0}, {t2}",
            "cmovnc {z1}, {z3}",
            "cmovnc {z2}, {z4}",
            a = out(reg) o0,
            z0 = inout(reg) value[0] => o1,
            z1 = inout(reg) value[1] => o2,
            z2 = inout(reg) value[2] => o3,
            z3 = inout(reg) value[3] => _,
            z4 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            p0 = in(reg) modulus[0],
            p1 = in(reg) modulus[1],
            inv = in(reg) inv,
            p3 = const PASTA_HIGH_LIMB,
            out("rdx") _,
            options(pure, nomem, nostack),
        );
    }
    [o0, o1, o2, o3]
}

/// One packed half-delta divstep, the step body of s2n-bignum's x86-64
/// `divstep59`.
///
/// The words `pf` and `pg` carry the low bits of `f` and `g` with a row of the
/// transition matrix in their upper bits, and `td = 2δ`. A step swaps and
/// subtracts when `g` is odd and `td` is nonnegative, adds `f` to `g` when `g` is
/// odd and `td` is negative, and does nothing to `g` when it is even; then it
/// halves `g` and adds two to `td`. As in s2n-bignum, the step is a mask `k`,
/// which is `-2` to swap and `0` otherwise: `t` is `f` complemented under `k`
/// (or `0` when `g` is even), which is `-f` for odd `f` since `k` keeps bit 0,
/// and `td` is complemented under it likewise, which negates the odd `td`.
/// Each arm but `first` finishes a step, and each arm but `last` starts the
/// next one: the sign of the new `td` and the parity of the new `g`, tested on
/// bit 1 before the halving, set its mask. `first` starts a batch's first step
/// from the parity of `pg` and the sign of `td`, re-creating the zero register
/// `z` that the conditional moves take.
macro_rules! divstep {
    (start) => {
        concat!(
            "mov {t}, {pf}\n",       // t = f.
            "mov {k}, -2\n",         // k = -2: swap, unless...
        )
    };
    (decide) => {
        concat!(
            "cmove {k}, {z}\n",      // ...g is even, when k = 0
            "cmove {t}, {z}\n",      // and t = 0.
            "xor {t}, {k}\n",        // t = -f when swapping.
            "xor {td}, {k}\n",       // td = -td when swapping.
            "bt {k}, 63\n",          // CF = swap.
            "cmovb {pf}, {pg}\n",    // f = g when swapping.
        )
    };
    (first) => {
        concat!(
            "xor {z:e}, {z:e}\n",
            divstep!(start),
            "test {td}, {td}\n",
            "cmovs {k}, {z}\n",      // ...td is negative, when k = 0, or...
            "test {pg}, 1\n",
            divstep!(decide),
        )
    };
    () => {
        concat!(
            "mov {k}, -2\n",         // The next step's k = -2: swap, unless...
            "add {td}, 2\n",         // td = td + 2.
            "lea {pg}, [{pg} + {t}]\n", // g = g + t, leaving the flags of td.
            "cmovs {k}, {z}\n",      // ...the new td is negative, when k = 0, or...
            "mov {t}, {pf}\n",       // t = the next step's f.
            "test {pg}, 2\n",        // The parity of the halved g.
            "cmove {k}, {z}\n",      // ...g is even, when k = 0
            "cmove {t}, {z}\n",      // and t = 0.
            "sar {pg}, 1\n",         // g = g / 2.
            "xor {t}, {k}\n",
            "xor {td}, {k}\n",
            "bt {k}, 63\n",
            "cmovb {pf}, {pg}\n",
        )
    };
    (last) => {
        concat!(
            "add {td}, 2\n",         // td = td + 2.
            "lea {pg}, [{pg} + {t}]\n", // g = g + t.
            "sar {pg}, 1\n",         // g = g / 2.
        )
    };
}

/// Fifty-nine half-delta divsteps on the low words of `f` and `g`, returning
/// the new `two_delta` and the transition matrix.
///
/// `two_delta = 2δ`, as a two's-complement word (`1` at the start), and `f0`
/// and `g0` are the low 64 bits of `f` and `g`, `f` odd. The result is
/// `[two_delta', u, v, q, r]`: the new `two_delta`, and the 59-step transition
/// matrix `M` with `2^59 (f', g') = M (f, g)`, as two's-complement words. The
/// steps are s2n-bignum's x86-64 `divstep59` macro on named registers, and the
/// arithmetic around them is the AArch64 block's: three batches of 20, 20, and
/// 19 steps, each on two words that pack the low 20 bits of `f` and `g` with a
/// row of the batch's matrix in the upper bits, whose matrices are read back
/// out of the upper bits, negated, and multiplied together. Between batches the
/// next low words are the matrix rows applied to the current ones, shifted
/// right by the batch's step count.
///
/// s2n-bignum keeps a batch's matrix on the stack while the next batch runs;
/// this block is `nostack` and holds it in registers instead, thirteen for
/// the whole block. The steps'
/// constants `2` and `-2` are immediates, so a step needs six registers. A
/// packed word's top bits are set with `movabs` through `pg` before `pg` is
/// formed, and with two `bts` for `pg`, so that packing needs no register
/// beyond the two words.
#[inline(always)]
pub(super) fn divstep59(mut two_delta: u64, f0: u64, g0: u64) -> [u64; 5] {
    let (u, v, q, r): (u64, u64, u64, u64);
    // SAFETY: straight-line register-only arithmetic; no memory access, no
    // stack use, and outputs depend only on the declared inputs.
    unsafe {
        asm!(
            // Batch 1: pack the low 20 bits with the identity's rows, as -2^41
            // in pf and -2^62 in pg.
            "movabs {pg}, 0xfffffe0000000000",
            "mov {pf}, {x}",
            "and {pf}, 0xfffff",
            "or {pf}, {pg}",
            "mov {pg}, {y}",
            "and {pg}, 0xfffff",
            "bts {pg}, 63",
            "bts {pg}, 62",
            divstep!(first),
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
            // Read the negated matrix of batch 1 out of the upper bits into
            // (a0, a1; a2, a3): the row's first entry as the 21-bit field at bit 21
            // after adding 2^20, the second as the arithmetic shift by 42 after
            // adding 2^20 + 2^41.
            "lea {a0}, [{pf} + 0x100000]",
            "shl {a0}, 22",
            "sar {a0}, 43",
            "movabs {h}, 0x20000100000",
            "lea {a1}, [{pf} + {h}]",
            "sar {a1}, 42",
            "lea {a2}, [{pg} + 0x100000]",
            "shl {a2}, 22",
            "sar {a2}, 43",
            "lea {a3}, [{pg} + {h}]",
            "sar {a3}, 42",
            // The next low words: the rows applied to the current ones, shifted
            // right by 20.
            "mov {pf}, {a0}",
            "imul {pf}, {x}",
            "mov {pg}, {a1}",
            "imul {pg}, {y}",
            "add {pf}, {pg}",
            "imul {x}, {a2}",
            "imul {y}, {a3}",
            "add {y}, {x}",
            "sar {pf}, 20",
            "sar {y}, 20",
            "mov {x}, {pf}",
            // Batch 2.
            "movabs {pg}, 0xfffffe0000000000",
            "mov {pf}, {x}",
            "and {pf}, 0xfffff",
            "or {pf}, {pg}",
            "mov {pg}, {y}",
            "and {pg}, 0xfffff",
            "bts {pg}, 63",
            "bts {pg}, 62",
            divstep!(first),
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
            // The negated matrix of batch 2 into (t, k; h, z), the second column
            // first, while h still holds the constant.
            "movabs {h}, 0x20000100000",
            "lea {k}, [{pf} + {h}]",
            "sar {k}, 42",
            "lea {z}, [{pg} + {h}]",
            "sar {z}, 42",
            "lea {t}, [{pf} + 0x100000]",
            "shl {t}, 22",
            "sar {t}, 43",
            "lea {h}, [{pg} + 0x100000]",
            "shl {h}, 22",
            "sar {h}, 43",
            "mov {pf}, {t}",
            "imul {pf}, {x}",
            "mov {pg}, {k}",
            "imul {pg}, {y}",
            "add {pf}, {pg}",
            "mov {pg}, {h}",
            "imul {pg}, {x}",
            "imul {y}, {z}",
            "add {y}, {pg}",
            "sar {pf}, 20",
            "sar {y}, 20",
            "mov {x}, {pf}",
            // Batch 3, of 19 steps.
            "movabs {pg}, 0xfffffe0000000000",
            "mov {pf}, {x}",
            "and {pf}, 0xfffff",
            "or {pf}, {pg}",
            "mov {pg}, {y}",
            "and {pg}, 0xfffff",
            "bts {pg}, 63",
            "bts {pg}, 62",
            // The product of the negated matrices of batches 2 and 1, which is the
            // product of the matrices themselves, into (x, a0; y, a3), freeing t,
            // k, and z for the steps.
            "mov {x}, {t}",
            "imul {x}, {a0}",
            "mov {y}, {k}",
            "imul {y}, {a2}",
            "add {x}, {y}",
            "mov {y}, {h}",
            "imul {y}, {a0}",
            "mov {a0}, {z}",
            "imul {a0}, {a2}",
            "add {y}, {a0}",
            "mov {a0}, {t}",
            "imul {a0}, {a1}",
            "mov {a2}, {k}",
            "imul {a2}, {a3}",
            "add {a0}, {a2}",
            "mov {a2}, {h}",
            "imul {a2}, {a1}",
            "imul {a3}, {z}",
            "add {a3}, {a2}",
            divstep!(first),
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
            // The negated matrix of batch 3, whose rows sit one bit higher after
            // 19 steps, into (t, a1; k, a2).
            "movabs {h}, 0x20000100000",
            "lea {a1}, [{pf} + {h}]",
            "sar {a1}, 43",
            "lea {a2}, [{pg} + {h}]",
            "sar {a2}, 43",
            "lea {t}, [{pf} + 0x100000]",
            "shl {t}, 21",
            "sar {t}, 43",
            "lea {k}, [{pg} + 0x100000]",
            "shl {k}, 21",
            "sar {k}, 43",
            // The 59-step matrix: minus the negated matrix of batch 3 times the
            // product of batches 2 and 1.
            "mov {pf}, {t}",
            "imul {pf}, {x}",
            "mov {pg}, {a1}",
            "imul {pg}, {y}",
            "add {pf}, {pg}",
            "neg {pf}",
            "mov {pg}, {t}",
            "imul {pg}, {a0}",
            "mov {h}, {a1}",
            "imul {h}, {a3}",
            "add {pg}, {h}",
            "neg {pg}",
            "mov {z}, {k}",
            "imul {z}, {x}",
            "mov {t}, {a2}",
            "imul {t}, {y}",
            "add {z}, {t}",
            "neg {z}",
            "imul {k}, {a0}",
            "imul {a2}, {a3}",
            "add {k}, {a2}",
            "neg {k}",
            td = inout(reg) two_delta,
            x = inout(reg) f0 => _,
            y = inout(reg) g0 => _,
            pf = out(reg) u,
            pg = out(reg) v,
            t = out(reg) _,
            k = out(reg) r,
            z = out(reg) q,
            a0 = out(reg) _,
            a1 = out(reg) _,
            a2 = out(reg) _,
            a3 = out(reg) _,
            h = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [two_delta, u, v, q, r]
}

/// The sign-magnitude form of a transition matrix: each entry's magnitude,
/// and its sign as a mask (all ones for a negative entry, else zero).
///
/// The row blocks multiply by a negative entry `m` as `|m| · ((x ^ mask) + 1)`,
/// so they take the entries in this form. The result is `[u, v, q, r, su, sv,
/// sq, sr]`, magnitudes then masks. Each entry is split as s2n-bignum's x86-64
/// `bignum_montinv_p256` splits it: the mask is the arithmetic shift of the
/// entry by 63, and the magnitude is the entry xored with the mask minus the
/// mask.
#[inline(always)]
pub(super) fn sign_mag(mut u: u64, mut v: u64, mut q: u64, mut r: u64) -> [u64; 8] {
    let (su, sv, sq, sr): (u64, u64, u64, u64);
    // SAFETY: register-only arithmetic with declared inputs and outputs;
    // no memory or stack access and no data-dependent control flow.
    unsafe {
        asm!(
            "mov {su}, {u}",
            "sar {su}, 63",
            "xor {u}, {su}",
            "sub {u}, {su}",
            "mov {sv}, {v}",
            "sar {sv}, 63",
            "xor {v}, {sv}",
            "sub {v}, {sv}",
            "mov {sq}, {q}",
            "sar {sq}, 63",
            "xor {q}, {sq}",
            "sub {q}, {sq}",
            "mov {sr}, {r}",
            "sar {sr}, 63",
            "xor {r}, {sr}",
            "sub {r}, {sr}",
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
/// `f` and `g` are five words each, the top word being the sign word. `m0` and
/// `m1` are the row's magnitudes and `s0` and `s1` its sign masks, from
/// [`sign_mag`]. A negative entry multiplies the complement of its operand
/// under the mask, with the magnitude added as the initial carry, so the
/// initial carry is `(m0 & s0) + (m1 & s1)`. Each column's two products and
/// the carry stay below `2^128` because `m0 + m1 <= 2^63`, so the high word of
/// a column is accumulated directly in the next column's word, which starts at
/// zero, and needs no register of its own. The top column is needed only
/// modulo `2^64`, so it takes the low products of two-operand IMUL. The
/// 320-bit sum is then shifted right by 59 in place with SHRD, the top word
/// arithmetically.
///
/// The operand words are read through pointers, and the products use the
/// one-operand MUL into RAX and RDX, as s2n-bignum's x86-64
/// `bignum_montinv_p256` forms them. The block uses thirteen registers; with
/// its ten operand words in registers it would need twenty-one.
#[inline(always)]
pub(super) fn fg_row(f: &[u64; 5], g: &[u64; 5], m0: u64, m1: u64, s0: u64, s1: u64) -> [u64; 5] {
    let (t0, t1, t2, t3, t4): (u64, u64, u64, u64, u64);
    // SAFETY: straight-line arithmetic reading only the ten words behind the
    // passed references (`readonly`); no stack use, and outputs depend only on
    // the declared inputs.
    unsafe {
        asm!(
            // The initial carry, and zero high words.
            "mov {t0}, {m0}",
            "and {t0}, {s0}",
            "mov rax, {m1}",
            "and rax, {s1}",
            "add {t0}, rax",
            "xor {t1:e}, {t1:e}",
            "xor {t2:e}, {t2:e}",
            "xor {t3:e}, {t3:e}",
            "xor {t4:e}, {t4:e}",
            // Column 0.
            "mov rax, qword ptr [{x}]",
            "xor rax, {s0}",
            "mul {m0}",
            "add {t0}, rax",
            "adc {t1}, rdx",
            "mov rax, qword ptr [{y}]",
            "xor rax, {s1}",
            "mul {m1}",
            "add {t0}, rax",
            "adc {t1}, rdx",
            // Column 1.
            "mov rax, qword ptr [{x} + 8]",
            "xor rax, {s0}",
            "mul {m0}",
            "add {t1}, rax",
            "adc {t2}, rdx",
            "mov rax, qword ptr [{y} + 8]",
            "xor rax, {s1}",
            "mul {m1}",
            "add {t1}, rax",
            "adc {t2}, rdx",
            // Column 2.
            "mov rax, qword ptr [{x} + 16]",
            "xor rax, {s0}",
            "mul {m0}",
            "add {t2}, rax",
            "adc {t3}, rdx",
            "mov rax, qword ptr [{y} + 16]",
            "xor rax, {s1}",
            "mul {m1}",
            "add {t2}, rax",
            "adc {t3}, rdx",
            // Column 3.
            "mov rax, qword ptr [{x} + 24]",
            "xor rax, {s0}",
            "mul {m0}",
            "add {t3}, rax",
            "adc {t4}, rdx",
            "mov rax, qword ptr [{y} + 24]",
            "xor rax, {s1}",
            "mul {m1}",
            "add {t3}, rax",
            "adc {t4}, rdx",
            // Column 4, modulo 2^64.
            "mov rax, qword ptr [{x} + 32]",
            "xor rax, {s0}",
            "imul rax, {m0}",
            "add {t4}, rax",
            "mov rax, qword ptr [{y} + 32]",
            "xor rax, {s1}",
            "imul rax, {m1}",
            "add {t4}, rax",
            // The exact division by 2^59.
            "shrd {t0}, {t1}, 59",
            "shrd {t1}, {t2}, 59",
            "shrd {t2}, {t3}, 59",
            "shrd {t3}, {t4}, 59",
            "sar {t4}, 59",
            x = in(reg) f.as_ptr(),
            y = in(reg) g.as_ptr(),
            m0 = in(reg) m0,
            m1 = in(reg) m1,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            t0 = out(reg) t0,
            t1 = out(reg) t1,
            t2 = out(reg) t2,
            t3 = out(reg) t3,
            t4 = out(reg) t4,
            out("rax") _,
            out("rdx") _,
            options(pure, readonly, nostack),
        );
    }
    [t0, t1, t2, t3, t4]
}

/// One row of the combination of `d` and `e` by a transition matrix:
/// `m0 · d + m1 · e`, as a five-word signed value for [`amontred`].
///
/// `d` and `e` are four words, nonnegative, and `m0`, `m1`, `s0`, and `s1` are
/// as for [`fg_row`]. The block is that of [`fg_row`] without the shift, and
/// with an implicit top word of zero in `d` and `e`: complemented under a mask,
/// that word is the mask itself, so the top column multiplies the masks.
#[inline(always)]
pub(super) fn de_row(d: &Limbs, e: &Limbs, m0: u64, m1: u64, s0: u64, s1: u64) -> [u64; 5] {
    let (t0, t1, t2, t3, t4): (u64, u64, u64, u64, u64);
    // SAFETY: straight-line arithmetic reading only the eight words behind the
    // passed references (`readonly`); no stack use, and outputs depend only on
    // the declared inputs.
    unsafe {
        asm!(
            // The initial carry, and zero high words.
            "mov {t0}, {m0}",
            "and {t0}, {s0}",
            "mov rax, {m1}",
            "and rax, {s1}",
            "add {t0}, rax",
            "xor {t1:e}, {t1:e}",
            "xor {t2:e}, {t2:e}",
            "xor {t3:e}, {t3:e}",
            "xor {t4:e}, {t4:e}",
            // Column 0.
            "mov rax, qword ptr [{x}]",
            "xor rax, {s0}",
            "mul {m0}",
            "add {t0}, rax",
            "adc {t1}, rdx",
            "mov rax, qword ptr [{y}]",
            "xor rax, {s1}",
            "mul {m1}",
            "add {t0}, rax",
            "adc {t1}, rdx",
            // Column 1.
            "mov rax, qword ptr [{x} + 8]",
            "xor rax, {s0}",
            "mul {m0}",
            "add {t1}, rax",
            "adc {t2}, rdx",
            "mov rax, qword ptr [{y} + 8]",
            "xor rax, {s1}",
            "mul {m1}",
            "add {t1}, rax",
            "adc {t2}, rdx",
            // Column 2.
            "mov rax, qword ptr [{x} + 16]",
            "xor rax, {s0}",
            "mul {m0}",
            "add {t2}, rax",
            "adc {t3}, rdx",
            "mov rax, qword ptr [{y} + 16]",
            "xor rax, {s1}",
            "mul {m1}",
            "add {t2}, rax",
            "adc {t3}, rdx",
            // Column 3.
            "mov rax, qword ptr [{x} + 24]",
            "xor rax, {s0}",
            "mul {m0}",
            "add {t3}, rax",
            "adc {t4}, rdx",
            "mov rax, qword ptr [{y} + 24]",
            "xor rax, {s1}",
            "mul {m1}",
            "add {t3}, rax",
            "adc {t4}, rdx",
            // Column 4, modulo 2^64: the complemented top words are the masks.
            "mov rax, {s0}",
            "imul rax, {m0}",
            "add {t4}, rax",
            "mov rax, {s1}",
            "imul rax, {m1}",
            "add {t4}, rax",
            x = in(reg) d.as_ptr(),
            y = in(reg) e.as_ptr(),
            m0 = in(reg) m0,
            m1 = in(reg) m1,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            t0 = out(reg) t0,
            t1 = out(reg) t1,
            t2 = out(reg) t2,
            t3 = out(reg) t3,
            t4 = out(reg) t4,
            out("rax") _,
            out("rdx") _,
            options(pure, readonly, nostack),
        );
    }
    [t0, t1, t2, t3, t4]
}

/// The almost-Montgomery reduction of a five-word signed value by one word:
/// `(s + w · p) / 2^64` for `s = t + 2^61 · p` and `w = s · inv mod 2^64`, as
/// four words.
///
/// For `|t| < 2^315`, which the row combinations satisfy, the result is below
/// `2p` and below `2^256` (Lemma 10 of `book/src/design/inversion.md`), so, as
/// in the AArch64 block and unlike P-256's `amontred` in s2n-bignum, there is
/// no top carry to capture and no conditional subtraction. Adding `2^61 · p`
/// for the Pasta shape adds `p0 << 61`, `(p1 : p0) >> 3`, `p1 >> 3`, and
/// `2^59` at words 0, 1, 2, and 4. All four are formed before the addition,
/// since the shifts write the flags. `w` is formed in RDX, as in
/// [`from_mont`], so that MULX, which leaves the flags alone, takes it as its
/// implicit multiplicand. Adding `w · p` is the cancellation step of `mul`:
/// the low word cancels, with a carry exactly when `s0` is nonzero, which is
/// the carry that `neg` sets; the high halves and `w · 2^254` go in one chain
/// and the low half of `w · p1` in another.
///
/// The modulus limbs are read through a pointer, as in [`mul`], which keeps
/// the block at thirteen registers.
#[inline(always)]
pub(super) fn amontred(t: &[u64; 5], modulus: &Limbs, inv: u64) -> Limbs {
    let (o0, o1, o2, o3): (u64, u64, u64, u64);
    // SAFETY: straight-line arithmetic reading only the modulus words behind
    // the passed reference (`readonly`); no stack use, and outputs depend only
    // on the declared inputs.
    unsafe {
        asm!(
            // s = t + 2^61 p, in five words.
            "mov {w0}, qword ptr [{p}]",
            "shl {w0}, 61",
            "mov {w2}, qword ptr [{p} + 8]",
            "mov {w1}, qword ptr [{p}]",
            "shrd {w1}, {w2}, 3",
            "shr {w2}, 3",
            "movabs {w3}, 0x800000000000000",
            "add {t0}, {w0}",
            "adc {t1}, {w1}",
            "adc {t2}, {w2}",
            "adc {t3}, 0",
            "adc {t4}, {w3}",
            // w = s0 · inv, and the products of w with the modulus.
            "mov rdx, {t0}",
            "imul rdx, {inv}",
            "mulx {w0}, {w1}, qword ptr [{p}]",
            "mulx {w2}, {w3}, qword ptr [{p} + 8]",
            "mov {w1}, rdx",
            "shl {w1}, 62",
            "mov {w4}, rdx",
            "shr {w4}, 2",
            // s + w p.
            "neg {t0}",
            "adc {t1}, {w0}",
            "adc {t2}, {w2}",
            "adc {t3}, {w1}",
            "adc {t4}, {w4}",
            "add {t1}, {w3}",
            "adc {t2}, 0",
            "adc {t3}, 0",
            "adc {t4}, 0",
            t0 = inout(reg) t[0] => _,
            t1 = inout(reg) t[1] => o0,
            t2 = inout(reg) t[2] => o1,
            t3 = inout(reg) t[3] => o2,
            t4 = inout(reg) t[4] => o3,
            p = in(reg) modulus.as_ptr(),
            inv = in(reg) inv,
            w0 = out(reg) _,
            w1 = out(reg) _,
            w2 = out(reg) _,
            w3 = out(reg) _,
            w4 = out(reg) _,
            out("rdx") _,
            options(pure, readonly, nostack),
        );
    }
    [o0, o1, o2, o3]
}

/// Subtracts the modulus from `value` unless the subtraction borrows.
///
/// For `value < 2p` the result is `value mod p`; the inversion applies it to
/// the final round's `amontred` output, which is below `2p`. Like [`add`], it
/// hardcodes the Pasta modulus shape (`modulus[2] == 0`). The differences are
/// formed in temporaries, since MOV leaves the borrow flag alone, and CMOVNC
/// takes them when the last subtraction did not borrow.
#[inline(always)]
pub(super) fn cond_sub(value: &Limbs, modulus: &Limbs) -> Limbs {
    let [mut r0, mut r1, mut r2, mut r3] = *value;
    // SAFETY: register-only arithmetic with declared inputs and outputs;
    // no memory or stack access and no data-dependent control flow.
    unsafe {
        asm!(
            "mov {t0}, {r0}",
            "sub {t0}, {p0}",
            "mov {t1}, {r1}",
            "sbb {t1}, {p1}",
            "mov {t2}, {r2}",
            "sbb {t2}, 0",
            "mov {t3}, {r3}",
            "sbb {t3}, {p3}",
            "cmovnc {r0}, {t0}",
            "cmovnc {r1}, {t1}",
            "cmovnc {r2}, {t2}",
            "cmovnc {r3}, {t3}",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
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

/// The backend's blocks: its Montgomery arithmetic, for the entry points in
/// `entry.rs`, and its inversion blocks, for the generic driver of
/// `crate::inversion`.
pub(crate) struct Backend;

impl super::entry::MontgomeryBlocks for Backend {
    #[inline(always)]
    fn add(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
        add(lhs, rhs, modulus)
    }

    #[inline(always)]
    fn sub(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
        sub(lhs, rhs, modulus)
    }

    /// The multiplication block, behind its own call boundary.
    #[inline(always)]
    fn mul(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
        mul(lhs, rhs, modulus, inv)
    }

    /// Squares a canonical Montgomery residue: [`square_hi`] after
    /// [`square_lo`].
    ///
    /// A transcription of the AArch64 backend's dedicated squaring: the
    /// 512-bit square as cross products, one doubling pass, and the
    /// diagonals (ten MULX against the multiplication's sixteen), then four
    /// Montgomery cancellations on a rotating four-limb window with a
    /// carried fifth limb, the high product half folded in, and a CMOV
    /// conditional subtraction. A canonical input's square is below `R * p`,
    /// so, as for [`mul`]'s candidate, the folded sum stays below `2p` and
    /// no carry escapes. Measured 1–5% ahead of squaring through [`mul`] on
    /// Skylake-X (20.0–20.7 vs 21.0 ns across runs), mirroring the AArch64
    /// backend's own square-over-mul margin.
    ///
    /// Always inlined, so that the repeated-squaring loop of `sqr_n_mul`
    /// keeps the accumulator in registers, with no call boundary or
    /// return-value traffic per iteration. The ~250 squarings of a field
    /// inversion would otherwise each pay a call's cost. The entry points
    /// `square` and `sqr_n_mul` are the call boundaries instead, for the
    /// register-allocation reason documented on [`mul`].
    #[inline(always)]
    fn square(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
        square_hi(square_lo(*value), modulus, inv)
    }

    /// The dedicated conversion block, rather than AArch64's multiplication by one.
    #[inline(always)]
    fn from_mont(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
        from_mont(value, modulus, inv)
    }
}

/// The inversion's blocks, in this backend's assembly.
impl InvertBlocks for Backend {
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

    #[test]
    fn invert_is_the_inverse() {
        crate::inversion::tests::invert_is_the_inverse::<Backend>();
    }

    #[test]
    fn invert_is_multiplicative() {
        crate::inversion::tests::invert_is_multiplicative::<Backend>();
    }

    #[test]
    fn invert_commutes_with_negation() {
        crate::inversion::tests::invert_commutes_with_negation::<Backend>();
    }
}
