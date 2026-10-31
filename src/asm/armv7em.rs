// Copyright the pasta_curves contributors.
// Licensed under either the MIT license or the Apache License, Version 2.0.

#![cfg_attr(target_arch = "arm", allow(dead_code))]

//! Thumb-2 Montgomery arithmetic for the Pasta fields.
//!
//! Field elements enter as four little-endian `u64` limbs and are split into eight
//! `u32` words. Montgomery multiplication scans one word of the right operand
//! per round in a ten-word rolling accumulator. The reduction uses the Pasta
//! modulus shape `[1, p1, p2, p3, 0, 0, 0, 2^30]`: it cancels the low word with
//! `q = -acc[0] mod 2^32`, propagates carries across the zero words, then
//! shifts the accumulator by one word. Two high words retain the carries until
//! the final conditional subtraction. Squaring uses the same multiplication
//! schedule; addition and subtraction use eight-word carry chains. Conversion
//! out of Montgomery form uses a dedicated, unrolled eight-round reduction.
//!
//! The Thumb-2 blocks implement word addition, subtraction, and
//! multiply-accumulate (`UMAAL`). Their Rust compositions use fixed-bound loops
//! and mask-based reductions, with no operand-dependent branches or indices.
//! Timing still depends on the compiler and the target's instruction behavior.

use core::arch::asm;

use crate::{inversion::InvertBlocks, limbs::Limbs};

const WORDS: usize = 8;
const ACC_WORDS: usize = WORDS + 2;
type Words = [u32; WORDS];

#[inline(always)]
fn to_words(value: &Limbs) -> Words {
    let mut words = [0; WORDS];
    let mut i = 0;
    while i < 4 {
        words[2 * i] = value[i] as u32;
        words[2 * i + 1] = (value[i] >> 32) as u32;
        i += 1;
    }
    words
}

#[inline(always)]
fn from_words(words: Words) -> Limbs {
    let mut value = [0; 4];
    let mut i = 0;
    while i < 4 {
        value[i] = u64::from(words[2 * i]) | (u64::from(words[2 * i + 1]) << 32);
        i += 1;
    }
    value
}

/// Returns `a + b + carry` and its carry bit.
#[cfg(target_arch = "arm")]
#[inline(always)]
fn adc(a: u32, b: u32, carry: u32) -> (u32, u32) {
    let sum: u32;
    let carry_out: u32;
    // SAFETY: this is register-only arithmetic. `out(reg)` prevents either
    // output from overlapping an input, and the block has no control flow or
    // memory access.
    unsafe {
        asm!(
            "adds {sum}, {a}, {b}",
            "mov {carry_out}, #0",
            "adc {carry_out}, {carry_out}, #0",
            "adds {sum}, {sum}, {carry}",
            "adc {carry_out}, {carry_out}, #0",
            a = in(reg) a,
            b = in(reg) b,
            carry = in(reg) carry,
            sum = out(reg) sum,
            carry_out = out(reg) carry_out,
            options(pure, nomem, nostack),
        );
    }
    (sum, carry_out)
}

/// Returns `a + b` and its carry bit.
#[cfg(target_arch = "arm")]
#[inline(always)]
fn add_word(a: u32, b: u32) -> (u32, u32) {
    let sum: u32;
    let carry: u32;
    // SAFETY: this is register-only arithmetic with distinct outputs.
    unsafe {
        asm!(
            "adds {sum}, {a}, {b}",
            "mov {carry}, #0",
            "adc {carry}, {carry}, #0",
            a = in(reg) a,
            b = in(reg) b,
            sum = out(reg) sum,
            carry = out(reg) carry,
            options(pure, nomem, nostack),
        );
    }
    (sum, carry)
}

/// Returns `a - b - borrow` and its borrow bit.
#[cfg(target_arch = "arm")]
#[inline(always)]
fn sbb(a: u32, b: u32, borrow: u32) -> (u32, u32) {
    let difference: u32;
    let borrow_out: u32;
    // SAFETY: this is register-only arithmetic. `rsbs` sets C to `!borrow`
    // for the 0/1 input, which is exactly the carry-in that `sbcs` needs.
    // Distinct outputs prevent invalid overlap, and the block has no control
    // flow or memory access.
    unsafe {
        asm!(
            "rsbs {borrow}, {borrow}, #0",
            "sbcs {difference}, {a}, {b}",
            "mov {borrow_out}, #0",
            "adc {borrow_out}, {borrow_out}, #0",
            "eor {borrow_out}, {borrow_out}, #1",
            a = in(reg) a,
            b = in(reg) b,
            borrow = inout(reg) borrow => _,
            difference = out(reg) difference,
            borrow_out = out(reg) borrow_out,
            options(pure, nomem, nostack),
        );
    }
    (difference, borrow_out)
}

/// Returns `acc + a * b + carry` and the high word.
#[cfg(target_arch = "arm")]
#[inline(always)]
fn mac_word(acc: u32, a: u32, b: u32, carry: u32) -> (u32, u32) {
    let mut low = acc;
    let mut high = carry;
    // SAFETY: this is register-only arithmetic. UMAAL adds its two 32-bit
    // destination inputs independently, matching `acc + carry` rather than a
    // 64-bit accumulator. The destinations are distinct, as UMAAL requires.
    // The exact maximum is `(2^32 - 1)^2 + 2 * (2^32 - 1) = 2^64 - 1`, so no
    // overflow is lost.
    unsafe {
        asm!(
            "umaal {low}, {high}, {a}, {b}",
            low = inout(reg) low,
            high = inout(reg) high,
            a = in(reg) a,
            b = in(reg) b,
            options(pure, nomem, nostack),
        );
    }
    (low, high)
}

/// Adds `a + b * c + carry` as two 64-bit words for the inversion's
/// almost-Montgomery reduction.
#[inline(always)]
pub(crate) fn inversion_mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let r0 = a as u32;
    let r1 = (a >> 32) as u32;
    let b0 = b as u32;
    let b1 = (b >> 32) as u32;
    let c0 = c as u32;
    let c1 = (c >> 32) as u32;
    let r2 = carry as u32;
    let r3 = (carry >> 32) as u32;
    let (r0, r1, r2, r3) = inversion_mac_words(r0, r1, b0, b1, c0, c1, r2, r3);

    (
        u64::from(r0) | (u64::from(r1) << 32),
        u64::from(r2) | (u64::from(r3) << 32),
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn inversion_mac_words(
    r0: u32,
    r1: u32,
    b0: u32,
    b1: u32,
    c0: u32,
    c1: u32,
    r2: u32,
    r3: u32,
) -> (u32, u32, u32, u32) {
    let mut r0 = r0;
    let mut r1 = r1;
    let mut r2 = r2;
    let mut r3 = r3;
    let mut c0 = c0;

    // SAFETY: this fixed block accesses registers only. The inout destinations
    // remain distinct from operands needed later, and every UMAAL has distinct
    // destination registers. ADDS and ADC intentionally communicate carry via flags.
    #[expect(unused_assignments, reason = "c0 is an assembly carry temporary")]
    unsafe {
        asm!(
            "umaal {r0}, {r2}, {b0}, {c0}",
            "umaal {r1}, {r3}, {b0}, {c1}",
            "umaal {r1}, {r2}, {b1}, {c0}",
            "mov {c0}, #0",
            "adds {r2}, {r2}, {r3}",
            "adc {c0}, {c0}, #0",
            "mov {r3}, #0",
            "umaal {r2}, {r3}, {b1}, {c1}",
            "add {r3}, {r3}, {c0}",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            c0 = inout(reg) c0,
            b0 = in(reg) b0,
            b1 = in(reg) b1,
            c1 = in(reg) c1,
            options(pure, nomem, nostack),
        );
    }

    (r0, r1, r2, r3)
}

/// Adds one eight-word product to a rolling accumulator. The two high words
/// retain overflow until the round's division by `2^32`.
#[inline(always)]
fn mac_product(acc: &mut [u32; ACC_WORDS], multiplicand: &Words, multiplier: u32) {
    let mut carry = 0;
    let mut i = 0;
    while i < WORDS {
        (acc[i], carry) = mac_word(acc[i], multiplicand[i], multiplier, carry);
        i += 1;
    }
    (acc[WORDS], carry) = add_word(acc[WORDS], carry);
    let (top, overflow) = add_word(acc[WORDS + 1], carry);
    acc[WORDS + 1] = top;
    debug_assert!(overflow == 0, "rolling multiplication accumulator overflow");
}

/// Adds `q * modulus` for a Pasta modulus. In 32-bit words both moduli are
/// `[1, a, b, c, 0, 0, 0, 2^30]`, so three multiplications can be omitted.
#[inline(always)]
fn mac_sparse_modulus(acc: &mut [u32; ACC_WORDS], modulus: &Words, q: u32) {
    debug_assert_eq!(modulus[0], 1);
    debug_assert_eq!(modulus[4], 0);
    debug_assert_eq!(modulus[5], 0);
    debug_assert_eq!(modulus[6], 0);
    debug_assert_eq!(modulus[7], 1 << 30);

    let mut carry;
    (acc[0], carry) = adc(acc[0], q, 0);
    let mut i = 1;
    while i < 4 {
        (acc[i], carry) = mac_word(acc[i], modulus[i], q, carry);
        i += 1;
    }
    while i < 7 {
        (acc[i], carry) = add_word(acc[i], carry);
        i += 1;
    }
    (acc[7], carry) = mac_word(acc[7], modulus[7], q, carry);
    (acc[WORDS], carry) = add_word(acc[WORDS], carry);
    let (top, overflow) = add_word(acc[WORDS + 1], carry);
    acc[WORDS + 1] = top;
    debug_assert!(overflow == 0, "rolling reduction accumulator overflow");
}

/// Drops the zero low word after a Montgomery round.
#[inline(always)]
fn shift_round(acc: &mut [u32; ACC_WORDS]) {
    debug_assert_eq!(acc[0], 0);
    acc[0] = acc[1];
    acc[1] = acc[2];
    acc[2] = acc[3];
    acc[3] = acc[4];
    acc[4] = acc[5];
    acc[5] = acc[6];
    acc[6] = acc[7];
    acc[7] = acc[8];
    acc[8] = acc[9];
    acc[9] = 0;
}

#[inline(always)]
fn select_reduced(candidate: Words, modulus: Words, high_nonzero: u32) -> Words {
    let mut reduced = [0; WORDS];
    let mut borrow = 0;
    let mut i = 0;
    while i < WORDS {
        (reduced[i], borrow) = sbb(candidate[i], modulus[i], borrow);
        i += 1;
    }

    // Select the subtraction when the candidate is at least the modulus. A
    // nonzero ninth word also forces subtraction for addition. For
    // multiplication it is zero under the entry point's operand contracts.
    let subtract = high_nonzero | (borrow ^ 1);
    let mask = 0u32.wrapping_sub(subtract);
    i = 0;
    while i < WORDS {
        reduced[i] = (reduced[i] & mask) | (candidate[i] & !mask);
        i += 1;
    }
    reduced
}

/// Multiplies two Montgomery residues using an eight-word rolling CIOS
/// accumulator specialized to the sparse Pasta moduli.
///
/// This has the same operand contract as `asm::mul`: either `lhs` is canonical,
/// or `rhs` is canonical with its upper three 64-bit limbs at most
/// `u64::MAX - 2`. Two high accumulator words model every carry explicitly;
/// after eight rounds the second must be zero and the first is the ninth word
/// passed to the constant-time final reduction.
#[inline(never)]
fn mul(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    let lhs = to_words(lhs);
    let rhs = to_words(rhs);
    let modulus = to_words(modulus);
    let inv = inv as u32;
    let mut acc = [0; ACC_WORDS];

    let mut i = 0;
    while i < WORDS {
        mac_product(&mut acc, &lhs, rhs[i]);

        // `inv = -modulus[0]^-1 (mod 2^64)`, so its low word is the
        // corresponding inverse modulo 2^32.
        let q = acc[0].wrapping_mul(inv);
        mac_sparse_modulus(&mut acc, &modulus, q);
        shift_round(&mut acc);
        i += 1;
    }

    let mut candidate = [0; WORDS];
    candidate.copy_from_slice(&acc[..WORDS]);
    debug_assert!(acc[WORDS + 1] == 0, "unmodeled final accumulator overflow");
    let high = acc[WORDS];
    let high_nonzero = (high | high.wrapping_neg()) >> 31;
    from_words(select_reduced(candidate, modulus, high_nonzero))
}

// One round of the conversion out of Montgomery form. The register names rotate
// between invocations so the nine-word accumulator does not need to move.
macro_rules! from_mont_round {
    ($a0:ident, $a1:ident, $a2:ident, $a3:ident, $a4:ident, $a5:ident, $a6:ident, $a7:ident, $a8:ident, $tmp:ident) => {
        concat!(
            // q = -a0. The carry into modulus word one is one exactly when a0
            // was nonzero, because a0 + q is then 2^32.
            concat!("rsbs.w ", stringify!($a0), ", ", stringify!($a0), ", #0\n"),
            concat!("sbc.w r12, ", stringify!($a0), ", ", stringify!($a0), "\n"),
            "and.w r12, r12, #1\n",
            // The only nontrivial modulus words below the top word.
            concat!("ldr.w ", stringify!($tmp), ", [r1, #4]\n"),
            concat!("umaal ", stringify!($a1), ", r12, ", stringify!($tmp), ", ", stringify!($a0), "\n"),
            concat!("ldr.w ", stringify!($tmp), ", [r1, #8]\n"),
            concat!("umaal ", stringify!($a2), ", r12, ", stringify!($tmp), ", ", stringify!($a0), "\n"),
            concat!("ldr.w ", stringify!($tmp), ", [r1, #12]\n"),
            concat!("umaal ", stringify!($a3), ", r12, ", stringify!($tmp), ", ", stringify!($a0), "\n"),
            // Modulus words four through six are zero.
            concat!("mov.w ", stringify!($tmp), ", #0\n"),
            concat!("adds.w ", stringify!($a4), ", ", stringify!($a4), ", r12\n"),
            concat!("adc.w r12, ", stringify!($tmp), ", ", stringify!($tmp), "\n"),
            concat!("adds.w ", stringify!($a5), ", ", stringify!($a5), ", r12\n"),
            concat!("adc.w r12, ", stringify!($tmp), ", ", stringify!($tmp), "\n"),
            concat!("adds.w ", stringify!($a6), ", ", stringify!($a6), ", r12\n"),
            concat!("adc.w r12, ", stringify!($tmp), ", ", stringify!($tmp), "\n"),
            // The top modulus word is 2^30. The final carry becomes the new
            // ninth accumulator word after the implicit division by 2^32.
            concat!("mov.w ", stringify!($tmp), ", #0x40000000\n"),
            concat!("umaal ", stringify!($a7), ", r12, ", stringify!($tmp), ", ", stringify!($a0), "\n"),
            concat!("mov.w ", stringify!($tmp), ", #0\n"),
            concat!("adds.w ", stringify!($a8), ", ", stringify!($a8), ", r12\n"),
            concat!("adc.w ", stringify!($tmp), ", ", stringify!($tmp), ", ", stringify!($tmp), "\n"),
        )
    };
}

// Reduce an arbitrary four-limb input with eight unrolled Montgomery rounds.
// Rotating the register names avoids moving the nine-word accumulator between
// rounds. Every load, store, and instruction is independent of the input.
#[inline(never)]
fn from_mont(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    debug_assert_eq!(inv as u32, u32::MAX);
    debug_assert_eq!(modulus[0] as u32, 1);
    debug_assert_eq!(modulus[2], 0);
    debug_assert_eq!(modulus[3], 1 << 62);

    let (word0, word1, word2, word3, word4, word5, word6, word7): (
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
    );
    // SAFETY: this fixed block only reads the four limbs of `value` and
    // `modulus`, and returns eight initialized words. It is a pure function of
    // those inputs, and the compiler preserves its clobbered AAPCS registers.
    unsafe {
        core::arch::asm!(
            // r0 starts as the input pointer, then becomes the round temporary.
            // Load the input into a nine-word accumulator without using r6 or r7,
            // which Rust reserves on this Thumb target.
            "ldr.w r2, [r0, #0]",
            "ldr.w r3, [r0, #4]",
            "ldr.w r4, [r0, #8]",
            "ldr.w r5, [r0, #12]",
            "ldr.w r8, [r0, #16]",
            "ldr.w r9, [r0, #20]",
            "ldr.w r10, [r0, #24]",
            "ldr.w r11, [r0, #28]",
            "mov.w lr, #0",
            // Rotate operand names rather than moving the physical registers.
            from_mont_round!(r2, r3, r4, r5, r8, r9, r10, r11, lr, r0),
            from_mont_round!(r3, r4, r5, r8, r9, r10, r11, lr, r0, r2),
            from_mont_round!(r4, r5, r8, r9, r10, r11, lr, r0, r2, r3),
            from_mont_round!(r5, r8, r9, r10, r11, lr, r0, r2, r3, r4),
            from_mont_round!(r8, r9, r10, r11, lr, r0, r2, r3, r4, r5),
            from_mont_round!(r9, r10, r11, lr, r0, r2, r3, r4, r5, r8),
            from_mont_round!(r10, r11, lr, r0, r2, r3, r4, r5, r8, r9),
            from_mont_round!(r11, lr, r0, r2, r3, r4, r5, r8, r9, r10),
            // The result is at most the modulus. Subtract it once, then add it
            // back under the all-ones borrow mask. r10 is the bounded ninth word.
            "subs.w lr, lr, #1",
            "ldr.w r11, [r1, #4]",
            "sbcs.w r0, r0, r11",
            "ldr.w r11, [r1, #8]",
            "sbcs.w r2, r2, r11",
            "ldr.w r11, [r1, #12]",
            "sbcs.w r3, r3, r11",
            "mov.w r11, #0",
            "sbcs.w r4, r4, r11",
            "sbcs.w r5, r5, r11",
            "sbcs.w r8, r8, r11",
            "mov.w r11, #0x40000000",
            "sbcs.w r9, r9, r11",
            "sbc.w r12, r10, r10",
            "and.w r11, r12, #1",
            "adds.w lr, lr, r11",
            "ldr.w r11, [r1, #4]",
            "and.w r11, r11, r12",
            "adcs.w r0, r0, r11",
            "ldr.w r11, [r1, #8]",
            "and.w r11, r11, r12",
            "adcs.w r2, r2, r11",
            "ldr.w r11, [r1, #12]",
            "and.w r11, r11, r12",
            "adcs.w r3, r3, r11",
            "mov.w r11, #0",
            "adcs.w r4, r4, r11",
            "adcs.w r5, r5, r11",
            "adcs.w r8, r8, r11",
            "and.w r11, r12, #0x40000000",
            "adc.w r9, r9, r11",
            inout("r0") value.as_ptr() => word1,
            in("r1") modulus.as_ptr(),
            lateout("r2") word2,
            lateout("r3") word3,
            lateout("r4") word4,
            lateout("r5") word5,
            lateout("r8") word6,
            lateout("r9") word7,
            lateout("r10") _,
            lateout("r11") _,
            lateout("r12") _,
            lateout("r14") word0,
            options(pure, readonly, nostack),
        );
    }
    [
        u64::from(word0) | (u64::from(word1) << 32),
        u64::from(word2) | (u64::from(word3) << 32),
        u64::from(word4) | (u64::from(word5) << 32),
        u64::from(word6) | (u64::from(word7) << 32),
    ]
}

#[inline(always)]
fn divstep59(two_delta: u64, f0: u64, g0: u64) -> [u64; 5] {
    let [d1, u1, v1, q1, r1] = batch(20, two_delta, f0, g0);
    let f1 = crate::inversion::portable::next_low(20, u1, v1, f0, g0);
    let g1 = crate::inversion::portable::next_low(20, q1, r1, f0, g0);
    let [d2, u2, v2, q2, r2] = batch(20, d1, f1, g1);
    let f2 = crate::inversion::portable::next_low(20, u2, v2, f1, g1);
    let g2 = crate::inversion::portable::next_low(20, q2, r2, f1, g1);
    let [d3, u3, v3, q3, r3] = batch(19, d2, f2, g2);
    let matrix = crate::inversion::portable::mat_mul(
        [u3, v3, q3, r3],
        crate::inversion::portable::mat_mul([u2, v2, q2, r2], [u1, v1, q1, r1]),
    );
    [d3, matrix[0], matrix[1], matrix[2], matrix[3]]
}

#[inline(always)]
fn batch(k: u32, mut delta: u64, f: u64, g: u64) -> [u64; 5] {
    let mut pf = (f & 0xfffff) | 0xfffffe0000000000;
    let mut pg = (g & 0xfffff) | 0xc000000000000000;
    let mut pflo = pf as u32;
    let mut pfhi = (pf >> 32) as u32;
    let mut pglo = pg as u32;
    let mut pghi = (pg >> 32) as u32;
    // The divstep state `two_delta` stays within a signed 32-bit word on every
    // reachable input (the bound under which the Lean proofs compose the three
    // batches into `divstep59Block`), so the block keeps it in one register and
    // sign-extends it back.
    let mut d = delta as u32;
    let remaining = k;
    // SAFETY: the block reads and writes only its register operands (no
    // memory, no stack), and returns all computed state. The count is public
    // (20 or 19), and the loop's branches do not depend on the packed values.
    unsafe {
        asm!(
            "2:",
            "and {odd}, {pglo}, #1",
            "rsb.w {odd}, {odd}, #0",
            "rsbs {tmp}, {d}, #0",
            "asr.w {swap}, {tmp}, #31",
            "and.w {swap}, {swap}, {odd}",
            "and.w {alo}, {pflo}, {odd}",
            "and.w {ahi}, {pfhi}, {odd}",
            "eor.w {alo}, {alo}, {swap}",
            "eor.w {ahi}, {ahi}, {swap}",
            "and.w {tmp}, {swap}, #1",
            "adds {alo}, {alo}, {tmp}",
            "mov.w {tmp}, #0",
            "adc.w {ahi}, {ahi}, {tmp}",
            "rsb.w {tmp}, {d}, #2",
            "add.w {d}, {d}, #2",
            "eor.w {tmp}, {tmp}, {d}",
            "and.w {tmp}, {tmp}, {swap}",
            "eor.w {d}, {d}, {tmp}",
            "eor.w {tmp}, {pflo}, {pglo}",
            "and.w {tmp}, {tmp}, {swap}",
            "eor.w {pflo}, {pflo}, {tmp}",
            "eor.w {tmp}, {pfhi}, {pghi}",
            "and.w {tmp}, {tmp}, {swap}",
            "eor.w {pfhi}, {pfhi}, {tmp}",
            "adds {pglo}, {pglo}, {alo}",
            "adc.w {pghi}, {pghi}, {ahi}",
            "lsr.w {tmp}, {pglo}, #1",
            "orr.w {pglo}, {tmp}, {pghi}, lsl #31",
            "asr.w {pghi}, {pghi}, #1",
            "subs {remaining}, {remaining}, #1",
            "bne 2b",
            pflo = inout(reg) pflo,
            pfhi = inout(reg) pfhi,
            pglo = inout(reg) pglo,
            pghi = inout(reg) pghi,
            d = inout(reg) d,
            remaining = inout(reg) remaining => _,
            odd = out(reg) _,
            swap = out(reg) _,
            alo = out(reg) _,
            ahi = out(reg) _,
            tmp = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    pf = u64::from(pflo) | (u64::from(pfhi) << 32);
    pg = u64::from(pglo) | (u64::from(pghi) << 32);
    delta = (d as i32 as i64) as u64;
    let (u, v) = crate::inversion::portable::unpack(k, pf);
    let (q, r) = crate::inversion::portable::unpack(k, pg);
    [delta, u, v, q, r]
}

#[inline(always)]
fn amontred(t: &[u64; 5], modulus: &Limbs, inv: u64) -> Limbs {
    let p = modulus;
    let s = crate::inversion::portable::add5(t, &crate::inversion::portable::shifted_modulus(p));
    let w = s[0].wrapping_mul(inv);
    let (_, carry) = inversion_mac(s[0], w, p[0], 0);
    let (r0, carry) = inversion_mac(s[1], w, p[1], carry);
    debug_assert!(p[2] == 0, "Pasta modulus has a zero third limb");
    let (r1, overflow) = s[2].overflowing_add(carry);
    let carry = u64::from(overflow);
    let (r2, carry) = inversion_mac(s[3], w, p[3], carry);
    [r0, r1, r2, s[4].wrapping_add(carry)]
}

// Propagate the borrow through one 32-bit word and save its difference.
macro_rules! cond_sub_word {
    (first, $offset:literal) => {
        concat!("ldr r2, [{value}, #", stringify!($offset), "]\n",
                "ldr r3, [{modulus}, #", stringify!($offset), "]\n",
                "subs r2, r2, r3\n",
                "str r2, [{out}, #", stringify!($offset), "]")
    };
    ($offset:literal) => {
        concat!("ldr r2, [{value}, #", stringify!($offset), "]\n",
                "ldr r3, [{modulus}, #", stringify!($offset), "]\n",
                "sbcs r2, r2, r3\n",
                "str r2, [{out}, #", stringify!($offset), "]")
    };
}

// Choose the original word on borrow, or retain its difference otherwise.
macro_rules! cond_sub_select_word {
    ($offset:literal) => {
        concat!("ldr r2, [{out}, #", stringify!($offset), "]\n",
                "ldr r3, [{value}, #", stringify!($offset), "]\n",
                "eor r3, r3, r2\n",
                "and r3, r3, {keep}\n",
                "eor r2, r2, r3\n",
                "str r2, [{out}, #", stringify!($offset), "]")
    };
}

#[inline(always)]
fn cond_sub(value: &Limbs, modulus: &Limbs) -> Limbs {
    let mut difference = [0u64; 4];
    // SAFETY: subtraction and masked selection use only fixed word offsets.
    // The final borrow becomes an all-ones mask without a branch.
    unsafe {
        asm!(
            cond_sub_word!(first, 0),
            cond_sub_word!(4),
            cond_sub_word!(8),
            cond_sub_word!(12),
            cond_sub_word!(16),
            cond_sub_word!(20),
            cond_sub_word!(24),
            cond_sub_word!(28),
            "mov.w {keep}, #0",
            "sbc.w {keep}, {keep}, {keep}",
            cond_sub_select_word!(0),
            cond_sub_select_word!(4),
            cond_sub_select_word!(8),
            cond_sub_select_word!(12),
            cond_sub_select_word!(16),
            cond_sub_select_word!(20),
            cond_sub_select_word!(24),
            cond_sub_select_word!(28),
            value = in(reg) value.as_ptr(),
            modulus = in(reg) modulus.as_ptr(),
            out = in(reg) difference.as_mut_ptr(),
            keep = out(reg) _,
            out("r2") _,
            out("r3") _,
            options(nostack),
        );
    }
    difference
}

/// The backend's blocks, for the entry points in `entry.rs`.
pub(crate) struct Backend;

impl super::entry::MontgomeryBlocks for Backend {
    #[inline(always)]
    fn add(_: &Limbs, _: &Limbs, _: &Limbs) -> Limbs {
        unreachable!("Portable impl used for this backend")
    }

    #[inline(always)]
    fn sub(_: &Limbs, _: &Limbs, _: &Limbs) -> Limbs {
        unreachable!("Portable impl used for this backend")
    }

    #[inline(always)]
    fn mul(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
        mul(lhs, rhs, modulus, inv)
    }

    #[inline(always)]
    fn square(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
        mul(value, value, modulus, inv)
    }

    #[inline(always)]
    fn from_mont(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
        from_mont(value, modulus, inv)
    }
}

impl InvertBlocks for Backend {
    #[inline]
    fn divstep59(two_delta: u64, f0: u64, g0: u64) -> [u64; 5] {
        divstep59(two_delta, f0, g0)
    }

    #[inline]
    fn sign_mag(u: u64, v: u64, q: u64, r: u64) -> [u64; 8] {
        <crate::inversion::portable::Backend as InvertBlocks>::sign_mag(u, v, q, r)
    }

    #[inline]
    fn fg_row(f: &[u64; 5], g: &[u64; 5], m0: u64, m1: u64, s0: u64, s1: u64) -> [u64; 5] {
        <crate::inversion::portable::Backend as InvertBlocks>::fg_row(f, g, m0, m1, s0, s1)
    }

    #[inline]
    fn de_row(d: &Limbs, e: &Limbs, m0: u64, m1: u64, s0: u64, s1: u64) -> [u64; 5] {
        <crate::inversion::portable::Backend as InvertBlocks>::de_row(d, e, m0, m1, s0, s1)
    }

    #[inline]
    fn amontred(t: &[u64; 5], modulus: &Limbs, inv: u64) -> Limbs {
        amontred(t, modulus, inv)
    }

    #[inline]
    fn cond_sub(value: &Limbs, modulus: &Limbs) -> Limbs {
        cond_sub(value, modulus)
    }
}

#[cfg(test)]
mod tests {
    use super::Backend;

    // The inversion's checks, over this backend's blocks. `sign_mag`, `fg_row`, and `de_row` are
    // the portable blocks, but are checked here too, so that every block `invert` runs on this
    // backend is.

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
