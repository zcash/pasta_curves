// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from the pre-generated Mach-O assembly in Semolina v0.1.4:
// https://github.com/supranational/semolina/blob/v0.1.4/src/mach-o/pasta_mul-armv8.S
//
// Only the fused repeated-squaring chains, the conversion out of Montgomery
// form, and their shared reduction helper are retained here.
// Single Montgomery multiplication and squaring live as inline `asm!`
// transcriptions of the same upstream routines in
// src/fields/aarch64_asm.rs. Symbols are renamed for this crate. The
// implementation relies on both Pasta moduli having limb 2 equal to zero
// and limb 3 equal to 2^62.

// All field elements and moduli are four little-endian 64-bit limbs. `inv` is
// -modulus[0]^-1 mod 2^64. The routines implement Montgomery reduction with
// R = 2^256. Registers named as limbs below always list the least-significant
// limb first.
//
// AArch64 procedure-call argument registers:
//
// - sqr_n:      x0 = out, x1 = value, x2 = count, x3 = modulus, x4 = inv
// - sqr_n_mul:  x0 = out, x1 = value, x2 = count, x3 = rhs, x4 = modulus,
//               x5 = inv
// - from_mont:  x0 = out, x1 = value, x2 = modulus, x3 = inv
//
// The code has no secret-dependent branches or memory accesses. The chains
// branch only on their public counts. Conditional reductions use `csel`
// after a full-width subtraction.

.text

#ifdef __APPLE__
#define C_SYMBOL(name) _##name
#define PRIVATE_SYMBOL(name) .private_extern C_SYMBOL(name)
#else
#define C_SYMBOL(name) name
#define PRIVATE_SYMBOL(name) .hidden C_SYMBOL(name)
#endif

.globl C_SYMBOL(pasta_curves_from_mont_pasta)
PRIVATE_SYMBOL(pasta_curves_from_mont_pasta)

.align 5
C_SYMBOL(pasta_curves_from_mont_pasta):
    .long 3573752639        // `paciasp`: sign LR with the entry stack pointer.
    stp x29,x30,[sp,#-16]! // Allocate frame; save frame pointer and signed LR.
    add x29,sp,#0           // Set the frame pointer to the new stack top.

    mov x4,x3               // Move inv into the helper's expected register.
    ldp x10,x11,[x1]        // Load Montgomery residue limbs 0 and 1.
    ldp x12,x13,[x1,#16]    // Load Montgomery residue limbs 2 and 3.

    // Montgomery-reduce an implicit zero upper half: value * R^-1 mod p.
    bl L$pasta_curves_mul_by_1_mont_pasta // Reduce x10..x13 by R.
    ldr x30,[x29,#8]        // Restore signed LR clobbered by `bl`.

    // The helper leaves p in x5..x8; one subtraction canonicalizes the result.
    subs x14,x10,x5         // Tentative result limb 0 = candidate - p[0].
    sbcs x15,x11,x6         // Tentative result limb 1 minus p[1].
    sbcs x16,x12,x7         // Tentative result limb 2 minus p[2] = 0.
    sbcs x17,x13,x8         // Tentative result limb 3 minus p[3].

    // If subtraction borrowed (`lo`), keep the original candidate.
    csel x10,x10,x14,lo     // Select canonical integer limb 0.
    csel x11,x11,x15,lo     // Select canonical integer limb 1.
    csel x12,x12,x16,lo     // Select canonical integer limb 2.
    csel x13,x13,x17,lo     // Select canonical integer limb 3.

    stp x10,x11,[x0]        // Store canonical integer limbs 0 and 1.
    stp x12,x13,[x0,#16]    // Store canonical integer limbs 2 and 3.

    ldr x29,[sp],#16        // Restore frame pointer and entry stack pointer.
    .long 3573752767        // `autiasp`: authenticate LR with entry SP.
    ret                     // Return through the authenticated address.

.align 5
L$pasta_curves_mul_by_1_mont_pasta:
    // Entry: x10..x13 = t, x2 = p, x4 = inv. Each iteration chooses q_i
    // so the current low limb plus q_i*p[0] is zero modulo 2^64, adds
    // q_i*p, and discards that zero limb. On exit, x10..x13 contain
    // (t + Q*p) / R, where Q has the four q_i as little-endian limbs.

    // Montgomery cancellation 0.
    mul x3,x4,x10           // q = t[0] * inv mod 2^64.
    ldp x5,x6,[x2]          // Load modulus limbs p[0] and p[1].
    ldp x7,x8,[x2,#16]      // Load p[2] = 0 and p[3] = 2^62.
    // low(q * p[0]) cancels x10 and is discarded.
    mul x15,x6,x3           // x15 = low(q * p[1]).
    // q * p[2] is zero because p[2] = 0.
    lsl x17,x3,#62          // x17 = low(q * p[3]).
    // Carry from x10 + low(q*p[0]) is one exactly when x10 is nonzero.
    subs xzr,x10,#1         // Set that cancellation carry.
    umulh x14,x5,x3         // x14 = high(q * p[0]).
    adcs x11,x11,x15        // Add low(q * p[1]) and cancellation carry.
    umulh x15,x6,x3         // x15 = high(q * p[1]).
    adcs x12,x12,xzr        // Propagate carry across zero p[2].
    // high(q * p[2]) is zero.
    adcs x13,x13,x17        // Add low(q * p[3]) and carry.
    lsr x17,x3,#2           // x17 = high(q * p[3]).
    adc x9,xzr,xzr          // Save the carry above limb 3.

    // Shift out cancelled limb 0 and start cancellation 1.
    adds x10,x11,x14        // New limb 0 includes high(q * p[0]).
    adcs x11,x12,x15        // New limb 1 includes high(q * p[1]).
    adcs x12,x13,xzr        // New limb 2; p[2] contributes zero.
    mul x3,x4,x10           // Next q = new limb 0 * inv mod 2^64.
    adc x13,x9,x17          // New limb 3 includes high(q * p[3]).
    // low(q * p[0]) cancels x10 and is discarded.
    mul x15,x6,x3           // x15 = low(next q * p[1]).
    // next q * p[2] is zero.
    lsl x17,x3,#62          // x17 = low(next q * p[3]).
    subs xzr,x10,#1         // Set the low-limb cancellation carry.
    umulh x14,x5,x3         // x14 = high(next q * p[0]).
    adcs x11,x11,x15        // Add low(next q * p[1]) and carry.
    umulh x15,x6,x3         // x15 = high(next q * p[1]).
    adcs x12,x12,xzr        // Propagate carry across zero p[2].
    // high(next q * p[2]) is zero.
    adcs x13,x13,x17        // Add low(next q * p[3]) and carry.
    lsr x17,x3,#2           // x17 = high(next q * p[3]).
    adc x9,xzr,xzr          // Save the carry above limb 3.

    // Shift out cancelled limb 1 and start cancellation 2.
    adds x10,x11,x14        // New limb 0 includes high(q * p[0]).
    adcs x11,x12,x15        // New limb 1 includes high(q * p[1]).
    adcs x12,x13,xzr        // New limb 2; p[2] contributes zero.
    mul x3,x4,x10           // Next q = new limb 0 * inv mod 2^64.
    adc x13,x9,x17          // New limb 3 includes high(q * p[3]).
    // low(q * p[0]) cancels x10 and is discarded.
    mul x15,x6,x3           // x15 = low(next q * p[1]).
    // next q * p[2] is zero.
    lsl x17,x3,#62          // x17 = low(next q * p[3]).
    subs xzr,x10,#1         // Set the low-limb cancellation carry.
    umulh x14,x5,x3         // x14 = high(next q * p[0]).
    adcs x11,x11,x15        // Add low(next q * p[1]) and carry.
    umulh x15,x6,x3         // x15 = high(next q * p[1]).
    adcs x12,x12,xzr        // Propagate carry across zero p[2].
    // high(next q * p[2]) is zero.
    adcs x13,x13,x17        // Add low(next q * p[3]) and carry.
    lsr x17,x3,#2           // x17 = high(next q * p[3]).
    adc x9,xzr,xzr          // Save the carry above limb 3.

    // Shift out cancelled limb 2 and start cancellation 3.
    adds x10,x11,x14        // New limb 0 includes high(q * p[0]).
    adcs x11,x12,x15        // New limb 1 includes high(q * p[1]).
    adcs x12,x13,xzr        // New limb 2; p[2] contributes zero.
    mul x3,x4,x10           // Final q = new limb 0 * inv mod 2^64.
    adc x13,x9,x17          // New limb 3 includes high(q * p[3]).
    // low(q * p[0]) cancels x10 and is discarded.
    mul x15,x6,x3           // x15 = low(final q * p[1]).
    // final q * p[2] is zero.
    lsl x17,x3,#62          // x17 = low(final q * p[3]).
    subs xzr,x10,#1         // Set the low-limb cancellation carry.
    umulh x14,x5,x3         // x14 = high(final q * p[0]).
    adcs x11,x11,x15        // Add low(final q * p[1]) and carry.
    umulh x15,x6,x3         // x15 = high(final q * p[1]).
    adcs x12,x12,xzr        // Propagate carry across zero p[2].
    // high(final q * p[2]) is zero.
    adcs x13,x13,x17        // Add low(final q * p[3]) and carry.
    lsr x17,x3,#2           // x17 = high(final q * p[3]).
    adc x9,xzr,xzr          // Save the carry above limb 3.

    // Shift out cancelled limb 3 to finish division by R = 2^256.
    adds x10,x11,x14        // Output limb 0 includes high(q * p[0]).
    adcs x11,x12,x15        // Output limb 1 includes high(q * p[1]).
    adcs x12,x13,xzr        // Output limb 2; p[2] contributes zero.
    adc x13,x9,x17          // Output limb 3 includes high(q * p[3]).

    ret                     // Return to from_mont through x30.

.globl C_SYMBOL(pasta_curves_sqr_n_mul_mont_pasta)
PRIVATE_SYMBOL(pasta_curves_sqr_n_mul_mont_pasta)

// Squares the input `count` times (count must be at least 1), then multiplies
// by `rhs`, keeping the accumulator in registers throughout. Intermediate
// squarings skip the final conditional subtraction, so mid-loop residues may
// exceed the modulus; they stay below 2p (the map t -> p + t^2/2^256 keeps
// [0, 2p) essentially invariant since 4p exceeds 2^256 by only ~2^-129 * 4p),
// so no limb-4 carry can be lost. The closing multiplication by a canonical
// `rhs` yields a value below rhs/2 + p < 2p, which one conditional
// subtraction (with no fifth limb, as the sum fits 256 bits) canonicalizes.
//
// The loop count is consumed with a decrement-then-test, so `count` = 0 would
// wrap; callers must pass at least 1.

.align 5
C_SYMBOL(pasta_curves_sqr_n_mul_mont_pasta):
    stp x29,x30,[sp,#-80]! // Allocate frame; save frame pointer and LR.
    add x29,sp,#0           // Set the frame pointer to the new stack top.
    stp x19,x20,[sp,#16]    // Preserve callee-saved scratch x19 and x20.
    stp x21,x22,[sp,#32]    // Preserve callee-saved scratch x21 and x22.
    stp x23,x24,[sp,#48]    // Preserve callee-saved modulus limbs 0 and 1.

    ldp x6,x7,[x1]          // Load value limbs a[0] and a[1].
    ldp x8,x9,[x1,#16]      // Load value limbs a[2] and a[3].

    ldp x23,x24,[x4]        // Keep p[0] and p[1] resident across the loop.
    // p[2] = 0 and p[3] = 2^62 are implicit throughout.

L$pasta_curves_sqr_n_mul_loop:
    sub x2,x2,#1            // Consume one squaring from the count.

    // Square a (in x6..x9) exactly as the inline `square` in
    // src/fields/aarch64_asm.rs does; the 512-bit
    // product limbs A0..A7 map to x10,x11,x12,x13,x14,x15,x16,x17.

    mul x11,x7,x6           // x11 = low(a[1] * a[0]).
    umulh x20,x7,x6         // x20 = high(a[1] * a[0]).
    mul x12,x8,x6           // x12 = low(a[2] * a[0]).
    umulh x21,x8,x6         // x21 = high(a[2] * a[0]).
    mul x13,x9,x6           // x13 = low(a[3] * a[0]).
    umulh x14,x9,x6         // x14 = high(a[3] * a[0]).

    adds x12,x12,x20        // Fold high(a[1] * a[0]) into product limb 2.
    mul x19,x8,x7           // x19 = low(a[2] * a[1]).
    umulh x20,x8,x7         // x20 = high(a[2] * a[1]).
    adcs x13,x13,x21        // Fold high(a[2] * a[0]) into product limb 3.
    mul x21,x9,x7           // x21 = low(a[3] * a[1]).
    umulh x22,x9,x7         // x22 = high(a[3] * a[1]).
    adc x14,x14,xzr         // Propagate carry into product limb 4.

    mul x15,x9,x8           // x15 = low(a[3] * a[2]).
    umulh x16,x9,x8         // x16 = high(a[3] * a[2]).

    adds x20,x20,x21        // Combine terms contributing to product limb 4.
    mul x10,x6,x6           // x10 = low(a[0]^2), product limb 0.
    adc x21,x22,xzr         // Combine terms contributing to product limb 5.

    adds x13,x13,x19        // Add low(a[2] * a[1]) into product limb 3.
    umulh x6,x6,x6          // x6 = high(a[0]^2).
    adcs x14,x14,x20        // Accumulate cross terms into product limb 4.
    mul x20,x7,x7           // x20 = low(a[1]^2).
    adcs x15,x15,x21        // Accumulate cross terms into product limb 5.
    umulh x7,x7,x7          // x7 = high(a[1]^2).
    adc x16,x16,xzr         // Propagate carry into product limb 6.

    adds x11,x11,x11        // Double cross-term product limb 1.
    mul x21,x8,x8           // x21 = low(a[2]^2).
    adcs x12,x12,x12        // Double cross-term product limb 2.
    umulh x8,x8,x8          // x8 = high(a[2]^2).
    adcs x13,x13,x13        // Double cross-term product limb 3.
    mul x22,x9,x9           // x22 = low(a[3]^2).
    adcs x14,x14,x14        // Double cross-term product limb 4.
    umulh x9,x9,x9          // x9 = high(a[3]^2).
    adcs x15,x15,x15        // Double cross-term product limb 5.
    adcs x16,x16,x16        // Double cross-term product limb 6.
    adc x17,xzr,xzr         // Capture the doubled cross-term carry in limb 7.

    mul x4,x5,x10           // q = product limb 0 * inv mod 2^64.

    // Add diagonal squares to obtain a^2 in x10..x17.
    adds x11,x11,x6         // Add high(a[0]^2) to product limb 1.
    adcs x12,x12,x20        // Add low(a[1]^2) to product limb 2.
    adcs x13,x13,x7         // Add high(a[1]^2) to product limb 3.
    adcs x14,x14,x21        // Add low(a[2]^2) to product limb 4.
    adcs x15,x15,x8         // Add high(a[2]^2) to product limb 5.
    adcs x16,x16,x22        // Add low(a[3]^2) to product limb 6.
    adc x17,x17,x9          // Add high(a[3]^2) to product limb 7.

    // Montgomery cancellation 0 on the low half, as in the shared helper.
    // low(q * p[0]) cancels x10 and is discarded by the limb shift.
    mul x20,x24,x4          // x20 = low(q * p[1]).
    // q * p[2] is zero because p[2] = 0.
    lsl x22,x4,#62          // x22 = low(q * p[3]).
    // Carry from x10 + low(q*p[0]) is one exactly when x10 is nonzero.
    subs xzr,x10,#1         // Set that cancellation carry.
    umulh x19,x23,x4        // x19 = high(q * p[0]).
    adcs x11,x11,x20        // Add low(q * p[1]) and cancellation carry.
    umulh x20,x24,x4        // x20 = high(q * p[1]).
    adcs x12,x12,xzr        // Propagate carry across zero p[2].
    // high(q * p[2]) is zero.
    adcs x13,x13,x22        // Add low(q * p[3]) and carry.
    lsr x22,x4,#2           // x22 = high(q * p[3]).
    adc x1,xzr,xzr          // Save the carry above limb 3.

    // Shift out cancelled limb 0 and start cancellation 1.
    adds x10,x11,x19        // New limb 0 includes high(q * p[0]).
    adcs x11,x12,x20        // New limb 1 includes high(q * p[1]).
    adcs x12,x13,xzr        // New limb 2; p[2] contributes zero.
    mul x4,x5,x10           // Next q = new limb 0 * inv mod 2^64.
    adc x13,x1,x22          // New limb 3 includes high(q * p[3]).
    // low(q * p[0]) cancels x10 and is discarded.
    mul x20,x24,x4          // x20 = low(next q * p[1]).
    // next q * p[2] is zero.
    lsl x22,x4,#62          // x22 = low(next q * p[3]).
    subs xzr,x10,#1         // Set the low-limb cancellation carry.
    umulh x19,x23,x4        // x19 = high(next q * p[0]).
    adcs x11,x11,x20        // Add low(next q * p[1]) and carry.
    umulh x20,x24,x4        // x20 = high(next q * p[1]).
    adcs x12,x12,xzr        // Propagate carry across zero p[2].
    // high(next q * p[2]) is zero.
    adcs x13,x13,x22        // Add low(next q * p[3]) and carry.
    lsr x22,x4,#2           // x22 = high(next q * p[3]).
    adc x1,xzr,xzr          // Save the carry above limb 3.

    // Shift out cancelled limb 1 and start cancellation 2.
    adds x10,x11,x19        // New limb 0 includes high(q * p[0]).
    adcs x11,x12,x20        // New limb 1 includes high(q * p[1]).
    adcs x12,x13,xzr        // New limb 2; p[2] contributes zero.
    mul x4,x5,x10           // Next q = new limb 0 * inv mod 2^64.
    adc x13,x1,x22          // New limb 3 includes high(q * p[3]).
    // low(q * p[0]) cancels x10 and is discarded.
    mul x20,x24,x4          // x20 = low(next q * p[1]).
    // next q * p[2] is zero.
    lsl x22,x4,#62          // x22 = low(next q * p[3]).
    subs xzr,x10,#1         // Set the low-limb cancellation carry.
    umulh x19,x23,x4        // x19 = high(next q * p[0]).
    adcs x11,x11,x20        // Add low(next q * p[1]) and carry.
    umulh x20,x24,x4        // x20 = high(next q * p[1]).
    adcs x12,x12,xzr        // Propagate carry across zero p[2].
    // high(next q * p[2]) is zero.
    adcs x13,x13,x22        // Add low(next q * p[3]) and carry.
    lsr x22,x4,#2           // x22 = high(next q * p[3]).
    adc x1,xzr,xzr          // Save the carry above limb 3.

    // Shift out cancelled limb 2 and start cancellation 3.
    adds x10,x11,x19        // New limb 0 includes high(q * p[0]).
    adcs x11,x12,x20        // New limb 1 includes high(q * p[1]).
    adcs x12,x13,xzr        // New limb 2; p[2] contributes zero.
    mul x4,x5,x10           // Final q = new limb 0 * inv mod 2^64.
    adc x13,x1,x22          // New limb 3 includes high(q * p[3]).
    // low(q * p[0]) cancels x10 and is discarded.
    mul x20,x24,x4          // x20 = low(final q * p[1]).
    // final q * p[2] is zero.
    lsl x22,x4,#62          // x22 = low(final q * p[3]).
    subs xzr,x10,#1         // Set the low-limb cancellation carry.
    umulh x19,x23,x4        // x19 = high(final q * p[0]).
    adcs x11,x11,x20        // Add low(final q * p[1]) and carry.
    umulh x20,x24,x4        // x20 = high(final q * p[1]).
    adcs x12,x12,xzr        // Propagate carry across zero p[2].
    // high(final q * p[2]) is zero.
    adcs x13,x13,x22        // Add low(final q * p[3]) and carry.
    lsr x22,x4,#2           // x22 = high(final q * p[3]).
    adc x1,xzr,xzr          // Save the carry above limb 3.

    // Shift out cancelled limb 3 to finish dividing the low half by R.
    adds x10,x11,x19        // Reduced limb 0 includes high(q * p[0]).
    adcs x11,x12,x20        // Reduced limb 1 includes high(q * p[1]).
    adcs x12,x13,xzr        // Reduced limb 2; p[2] contributes zero.
    adc x13,x1,x22          // Reduced limb 3 includes high(q * p[3]).
    // Add the upper product half; the sum stays below 2p (see above), so no
    // carry escapes and no conditional subtraction is needed mid-loop.
    adds x6,x10,x14         // Next-iteration a[0].
    adcs x7,x11,x15         // Next-iteration a[1].
    adcs x8,x12,x16         // Next-iteration a[2].
    adc x9,x13,x17          // Next-iteration a[3].

    cbnz x2,L$pasta_curves_sqr_n_mul_loop // Square again if count remains.

    // Multiply the accumulator (x6..x9) by rhs, as the inline `mul` in
    // src/fields/aarch64_asm.rs does, but
    // with the five-limb product in x10..x14 and temporaries in x19..x22.
    ldr x1,[x3]             // Load rhs limb b[0].

    mul x10,x6,x1           // x10 = low(a[0] * b[0]).
    mul x11,x7,x1           // x11 = low(a[1] * b[0]).
    mul x12,x8,x1           // x12 = low(a[2] * b[0]).
    mul x13,x9,x1           // x13 = low(a[3] * b[0]).

    umulh x19,x6,x1         // x19 = high(a[0] * b[0]).
    umulh x20,x7,x1         // x20 = high(a[1] * b[0]).
    mul x4,x5,x10           // q = x10 * inv mod 2^64.
    umulh x21,x8,x1         // x21 = high(a[2] * b[0]).
    umulh x22,x9,x1         // x22 = high(a[3] * b[0]).
    adds x11,x11,x19        // Add high(a[0] * b[0]) into limb 1.
    // low(q * p[0]) cancels x10 and is discarded by the limb shift.
    adcs x12,x12,x20        // Add high(a[1] * b[0]) and carry.
    mul x20,x24,x4          // x20 = low(q * p[1]).
    adcs x13,x13,x21        // Add high(a[2] * b[0]) and carry.
    // q * p[2] is zero because p[2] = 0.
    adc x14,xzr,x22         // Finish a * b[0] with its fifth limb.
    lsl x22,x4,#62          // x22 = low(q * p[3]).
    ldr x1,[x3,8*1]         // Load rhs limb b[1] for the next round.
    // Carry from x10 + low(q*p[0]) is one exactly when x10 is nonzero.
    subs xzr,x10,#1         // Set that carry without computing the zero sum.
    umulh x19,x23,x4        // x19 = high(q * p[0]).
    adcs x11,x11,x20        // Add low(q * p[1]) and cancellation carry.
    umulh x20,x24,x4        // x20 = high(q * p[1]).
    adcs x12,x12,xzr        // Propagate carry; p[2]'s product is zero.
    // high(q * p[2]) is zero.
    adcs x13,x13,x22        // Add low(q * p[3]) and carry.
    lsr x22,x4,#2           // x22 = high(q * p[3]).
    adc x14,x14,xzr         // Propagate carry into the fifth limb.

    // Drop the cancelled low limb: (a*b[0] + q*p) / 2^64.
    adds x10,x11,x19        // New limb 0 includes high(q * p[0]).
    mul x19,x6,x1           // x19 = low(a[0] * b[1]).
    adcs x11,x12,x20        // New limb 1 includes high(q * p[1]).
    mul x20,x7,x1           // x20 = low(a[1] * b[1]).
    adcs x12,x13,xzr        // New limb 2; p[2] contributes zero.
    mul x21,x8,x1           // x21 = low(a[2] * b[1]).
    adcs x13,x14,x22        // New limb 3 includes high(q * p[3]).
    mul x22,x9,x1           // x22 = low(a[3] * b[1]).
    adc x14,xzr,xzr         // Capture the reduction carry as limb 4.

    // Round 1: add a * b[1] to the reduced accumulator.
    adds x10,x10,x19        // Add low(a[0] * b[1]) to limb 0.
    umulh x19,x6,x1         // x19 = high(a[0] * b[1]).
    adcs x11,x11,x20        // Add low(a[1] * b[1]) and carry.
    umulh x20,x7,x1         // x20 = high(a[1] * b[1]).
    adcs x12,x12,x21        // Add low(a[2] * b[1]) and carry.
    mul x4,x5,x10           // q = current limb 0 * inv mod 2^64.
    umulh x21,x8,x1         // x21 = high(a[2] * b[1]).
    adcs x13,x13,x22        // Add low(a[3] * b[1]) and carry.
    umulh x22,x9,x1         // x22 = high(a[3] * b[1]).
    adc x14,x14,xzr         // Propagate multiplication carry to limb 4.

    adds x11,x11,x19        // Add high(a[0] * b[1]) to limb 1.
    // low(q * p[0]) cancels x10.
    adcs x12,x12,x20        // Add high(a[1] * b[1]) and carry.
    mul x20,x24,x4          // x20 = low(q * p[1]).
    adcs x13,x13,x21        // Add high(a[2] * b[1]) and carry.
    // low(q * p[2]) is zero.
    adc x14,x14,x22         // Add high(a[3] * b[1]) and final carry.
    lsl x22,x4,#62          // x22 = low(q * p[3]).
    ldr x1,[x3,8*2]         // Load rhs limb b[2] for the next round.
    subs xzr,x10,#1         // Set the low-limb cancellation carry.
    umulh x19,x23,x4        // x19 = high(q * p[0]).
    adcs x11,x11,x20        // Add low(q * p[1]) and cancellation carry.
    umulh x20,x24,x4        // x20 = high(q * p[1]).
    adcs x12,x12,xzr        // Propagate carry across zero p[2].
    // high(q * p[2]) is zero.
    adcs x13,x13,x22        // Add low(q * p[3]) and carry.
    lsr x22,x4,#2           // x22 = high(q * p[3]).
    adc x14,x14,xzr         // Propagate carry to limb 4.

    // Shift after round 1 while starting a * b[2].
    adds x10,x11,x19        // New limb 0 includes high(q * p[0]).
    mul x19,x6,x1           // x19 = low(a[0] * b[2]).
    adcs x11,x12,x20        // New limb 1 includes high(q * p[1]).
    mul x20,x7,x1           // x20 = low(a[1] * b[2]).
    adcs x12,x13,xzr        // New limb 2; p[2] contributes zero.
    mul x21,x8,x1           // x21 = low(a[2] * b[2]).
    adcs x13,x14,x22        // New limb 3 includes high(q * p[3]).
    mul x22,x9,x1           // x22 = low(a[3] * b[2]).
    adc x14,xzr,xzr         // Capture the reduction carry as limb 4.

    // Round 2: add a * b[2] and cancel the resulting low limb.
    adds x10,x10,x19        // Add low(a[0] * b[2]) to limb 0.
    umulh x19,x6,x1         // x19 = high(a[0] * b[2]).
    adcs x11,x11,x20        // Add low(a[1] * b[2]) and carry.
    umulh x20,x7,x1         // x20 = high(a[1] * b[2]).
    adcs x12,x12,x21        // Add low(a[2] * b[2]) and carry.
    mul x4,x5,x10           // q = current limb 0 * inv mod 2^64.
    umulh x21,x8,x1         // x21 = high(a[2] * b[2]).
    adcs x13,x13,x22        // Add low(a[3] * b[2]) and carry.
    umulh x22,x9,x1         // x22 = high(a[3] * b[2]).
    adc x14,x14,xzr         // Propagate multiplication carry to limb 4.

    adds x11,x11,x19        // Add high(a[0] * b[2]) to limb 1.
    // low(q * p[0]) cancels x10.
    adcs x12,x12,x20        // Add high(a[1] * b[2]) and carry.
    mul x20,x24,x4          // x20 = low(q * p[1]).
    adcs x13,x13,x21        // Add high(a[2] * b[2]) and carry.
    // low(q * p[2]) is zero.
    adc x14,x14,x22         // Add high(a[3] * b[2]) and final carry.
    lsl x22,x4,#62          // x22 = low(q * p[3]).
    ldr x1,[x3,8*3]         // Load rhs limb b[3] for the last round.
    subs xzr,x10,#1         // Set the low-limb cancellation carry.
    umulh x19,x23,x4        // x19 = high(q * p[0]).
    adcs x11,x11,x20        // Add low(q * p[1]) and cancellation carry.
    umulh x20,x24,x4        // x20 = high(q * p[1]).
    adcs x12,x12,xzr        // Propagate carry across zero p[2].
    // high(q * p[2]) is zero.
    adcs x13,x13,x22        // Add low(q * p[3]) and carry.
    lsr x22,x4,#2           // x22 = high(q * p[3]).
    adc x14,x14,xzr         // Propagate carry to limb 4.

    // Shift after round 2 while starting a * b[3].
    adds x10,x11,x19        // New limb 0 includes high(q * p[0]).
    mul x19,x6,x1           // x19 = low(a[0] * b[3]).
    adcs x11,x12,x20        // New limb 1 includes high(q * p[1]).
    mul x20,x7,x1           // x20 = low(a[1] * b[3]).
    adcs x12,x13,xzr        // New limb 2; p[2] contributes zero.
    mul x21,x8,x1           // x21 = low(a[2] * b[3]).
    adcs x13,x14,x22        // New limb 3 includes high(q * p[3]).
    mul x22,x9,x1           // x22 = low(a[3] * b[3]).
    adc x14,xzr,xzr         // Capture the reduction carry as limb 4.

    // Round 3: add a * b[3] and perform the last Montgomery cancellation.
    adds x10,x10,x19        // Add low(a[0] * b[3]) to limb 0.
    umulh x19,x6,x1         // x19 = high(a[0] * b[3]).
    adcs x11,x11,x20        // Add low(a[1] * b[3]) and carry.
    umulh x20,x7,x1         // x20 = high(a[1] * b[3]).
    adcs x12,x12,x21        // Add low(a[2] * b[3]) and carry.
    mul x4,x5,x10           // q = current limb 0 * inv mod 2^64.
    umulh x21,x8,x1         // x21 = high(a[2] * b[3]).
    adcs x13,x13,x22        // Add low(a[3] * b[3]) and carry.
    umulh x22,x9,x1         // x22 = high(a[3] * b[3]).
    adc x14,x14,xzr         // Propagate multiplication carry to limb 4.

    adds x11,x11,x19        // Add high(a[0] * b[3]) to limb 1.
    // low(q * p[0]) cancels x10.
    adcs x12,x12,x20        // Add high(a[1] * b[3]) and carry.
    mul x20,x24,x4          // x20 = low(q * p[1]).
    adcs x13,x13,x21        // Add high(a[2] * b[3]) and carry.
    // low(q * p[2]) is zero.
    adc x14,x14,x22         // Add high(a[3] * b[3]) and final carry.
    lsl x22,x4,#62          // x22 = low(q * p[3]).
    subs xzr,x10,#1         // Set the low-limb cancellation carry.
    umulh x19,x23,x4        // x19 = high(q * p[0]).
    adcs x11,x11,x20        // Add low(q * p[1]) and cancellation carry.
    umulh x20,x24,x4        // x20 = high(q * p[1]).
    adcs x12,x12,xzr        // Propagate carry across zero p[2].
    // high(q * p[2]) is zero.
    adcs x13,x13,x22        // Add low(q * p[3]) and carry.
    lsr x22,x4,#2           // x22 = high(q * p[3]).
    adc x14,x14,xzr         // Propagate carry to limb 4.

    // Shift out the fourth cancelled limb; the result fits 256 bits.
    adds x10,x11,x19        // Final candidate limb 0.
    adcs x11,x12,x20        // Final candidate limb 1.
    adcs x12,x13,xzr        // Final candidate limb 2.
    adc x13,x14,x22         // Final candidate limb 3.

L$pasta_curves_sqr_n_canonicalize:
    // Subtract p = [p0,p1,0,2^62]; the upper limbs are materialized inline.
    subs x19,x10,x23        // Tentative result limb 0 = candidate - p[0].
    mov x22,#1<<62          // Materialize p[3] = 2^62.
    sbcs x20,x11,x24        // Tentative result limb 1 minus p[1].
    sbcs x21,x12,xzr        // Tentative result limb 2; p[2] is zero.
    sbcs x22,x13,x22        // Tentative result limb 3 minus p[3].

    // `lo` means subtraction borrowed, so retain the original candidate.
    csel x10,x10,x19,lo     // Select canonical output limb 0.
    csel x11,x11,x20,lo     // Select canonical output limb 1.
    csel x12,x12,x21,lo     // Select canonical output limb 2.
    csel x13,x13,x22,lo     // Select canonical output limb 3.

    stp x10,x11,[x0]        // Store output limbs 0 and 1.
    stp x12,x13,[x0,#16]    // Store output limbs 2 and 3.

    ldp x19,x20,[x29,#16]   // Restore callee-saved x19 and x20.
    ldp x21,x22,[x29,#32]   // Restore callee-saved x21 and x22.
    ldp x23,x24,[x29,#48]   // Restore callee-saved x23 and x24.
    ldr x29,[sp],#80        // Restore frame pointer and release the frame.
    ret                     // Return; x30 still holds the caller's address.

.globl C_SYMBOL(pasta_curves_sqr_n_mont_pasta)
PRIVATE_SYMBOL(pasta_curves_sqr_n_mont_pasta)

// Squares the input `count` times (count must be at least 1), retaining the
// lazy accumulator in registers across the entire chain and canonicalizing
// only once. This is a separate emitted loop so the existing fused
// square-and-multiply path remains instruction-for-instruction unchanged.

.align 5
C_SYMBOL(pasta_curves_sqr_n_mont_pasta):
    stp x29,x30,[sp,#-80]! // Allocate frame; save frame pointer and LR.
    add x29,sp,#0           // Set the frame pointer to the new stack top.
    stp x19,x20,[sp,#16]    // Preserve callee-saved scratch x19 and x20.
    stp x21,x22,[sp,#32]    // Preserve callee-saved scratch x21 and x22.
    stp x23,x24,[sp,#48]    // Preserve callee-saved modulus limbs 0 and 1.

    mov x5,x4               // Move inv into the loop's expected register.
    mov x4,x3               // Move modulus into the expected register.

    ldp x6,x7,[x1]          // Load value limbs a[0] and a[1].
    ldp x8,x9,[x1,#16]      // Load value limbs a[2] and a[3].

    ldp x23,x24,[x4]        // Keep p[0] and p[1] resident across the loop.
    // p[2] = 0 and p[3] = 2^62 are implicit throughout.

L$pasta_curves_sqr_n_loop:
    sub x2,x2,#1            // Consume one squaring from the count.

    // Square a (in x6..x9) exactly as the inline `square` in
    // src/fields/aarch64_asm.rs does; the 512-bit
    // product limbs A0..A7 map to x10,x11,x12,x13,x14,x15,x16,x17.

    mul x11,x7,x6           // x11 = low(a[1] * a[0]).
    umulh x20,x7,x6         // x20 = high(a[1] * a[0]).
    mul x12,x8,x6           // x12 = low(a[2] * a[0]).
    umulh x21,x8,x6         // x21 = high(a[2] * a[0]).
    mul x13,x9,x6           // x13 = low(a[3] * a[0]).
    umulh x14,x9,x6         // x14 = high(a[3] * a[0]).

    adds x12,x12,x20        // Fold high(a[1] * a[0]) into product limb 2.
    mul x19,x8,x7           // x19 = low(a[2] * a[1]).
    umulh x20,x8,x7         // x20 = high(a[2] * a[1]).
    adcs x13,x13,x21        // Fold high(a[2] * a[0]) into product limb 3.
    mul x21,x9,x7           // x21 = low(a[3] * a[1]).
    umulh x22,x9,x7         // x22 = high(a[3] * a[1]).
    adc x14,x14,xzr         // Propagate carry into product limb 4.

    mul x15,x9,x8           // x15 = low(a[3] * a[2]).
    umulh x16,x9,x8         // x16 = high(a[3] * a[2]).

    adds x20,x20,x21        // Combine terms contributing to product limb 4.
    mul x10,x6,x6           // x10 = low(a[0]^2), product limb 0.
    adc x21,x22,xzr         // Combine terms contributing to product limb 5.

    adds x13,x13,x19        // Add low(a[2] * a[1]) into product limb 3.
    umulh x6,x6,x6          // x6 = high(a[0]^2).
    adcs x14,x14,x20        // Accumulate cross terms into product limb 4.
    mul x20,x7,x7           // x20 = low(a[1]^2).
    adcs x15,x15,x21        // Accumulate cross terms into product limb 5.
    umulh x7,x7,x7          // x7 = high(a[1]^2).
    adc x16,x16,xzr         // Propagate carry into product limb 6.

    adds x11,x11,x11        // Double cross-term product limb 1.
    mul x21,x8,x8           // x21 = low(a[2]^2).
    adcs x12,x12,x12        // Double cross-term product limb 2.
    umulh x8,x8,x8          // x8 = high(a[2]^2).
    adcs x13,x13,x13        // Double cross-term product limb 3.
    mul x22,x9,x9           // x22 = low(a[3]^2).
    adcs x14,x14,x14        // Double cross-term product limb 4.
    umulh x9,x9,x9          // x9 = high(a[3]^2).
    adcs x15,x15,x15        // Double cross-term product limb 5.
    adcs x16,x16,x16        // Double cross-term product limb 6.
    adc x17,xzr,xzr         // Capture the doubled cross-term carry in limb 7.

    mul x4,x5,x10           // q = product limb 0 * inv mod 2^64.

    // Add diagonal squares to obtain a^2 in x10..x17.
    adds x11,x11,x6         // Add high(a[0]^2) to product limb 1.
    adcs x12,x12,x20        // Add low(a[1]^2) to product limb 2.
    adcs x13,x13,x7         // Add high(a[1]^2) to product limb 3.
    adcs x14,x14,x21        // Add low(a[2]^2) to product limb 4.
    adcs x15,x15,x8         // Add high(a[2]^2) to product limb 5.
    adcs x16,x16,x22        // Add low(a[3]^2) to product limb 6.
    adc x17,x17,x9          // Add high(a[3]^2) to product limb 7.

    // Montgomery cancellation 0 on the low half, as in the shared helper.
    // low(q * p[0]) cancels x10 and is discarded by the limb shift.
    mul x20,x24,x4          // x20 = low(q * p[1]).
    // q * p[2] is zero because p[2] = 0.
    lsl x22,x4,#62          // x22 = low(q * p[3]).
    // Carry from x10 + low(q*p[0]) is one exactly when x10 is nonzero.
    subs xzr,x10,#1         // Set that cancellation carry.
    umulh x19,x23,x4        // x19 = high(q * p[0]).
    adcs x11,x11,x20        // Add low(q * p[1]) and cancellation carry.
    umulh x20,x24,x4        // x20 = high(q * p[1]).
    adcs x12,x12,xzr        // Propagate carry across zero p[2].
    // high(q * p[2]) is zero.
    adcs x13,x13,x22        // Add low(q * p[3]) and carry.
    lsr x22,x4,#2           // x22 = high(q * p[3]).
    adc x1,xzr,xzr          // Save the carry above limb 3.

    // Shift out cancelled limb 0 and start cancellation 1.
    adds x10,x11,x19        // New limb 0 includes high(q * p[0]).
    adcs x11,x12,x20        // New limb 1 includes high(q * p[1]).
    adcs x12,x13,xzr        // New limb 2; p[2] contributes zero.
    mul x4,x5,x10           // Next q = new limb 0 * inv mod 2^64.
    adc x13,x1,x22          // New limb 3 includes high(q * p[3]).
    // low(q * p[0]) cancels x10 and is discarded.
    mul x20,x24,x4          // x20 = low(next q * p[1]).
    // next q * p[2] is zero.
    lsl x22,x4,#62          // x22 = low(next q * p[3]).
    subs xzr,x10,#1         // Set the low-limb cancellation carry.
    umulh x19,x23,x4        // x19 = high(next q * p[0]).
    adcs x11,x11,x20        // Add low(next q * p[1]) and carry.
    umulh x20,x24,x4        // x20 = high(next q * p[1]).
    adcs x12,x12,xzr        // Propagate carry across zero p[2].
    // high(next q * p[2]) is zero.
    adcs x13,x13,x22        // Add low(next q * p[3]) and carry.
    lsr x22,x4,#2           // x22 = high(next q * p[3]).
    adc x1,xzr,xzr          // Save the carry above limb 3.

    // Shift out cancelled limb 1 and start cancellation 2.
    adds x10,x11,x19        // New limb 0 includes high(q * p[0]).
    adcs x11,x12,x20        // New limb 1 includes high(q * p[1]).
    adcs x12,x13,xzr        // New limb 2; p[2] contributes zero.
    mul x4,x5,x10           // Next q = new limb 0 * inv mod 2^64.
    adc x13,x1,x22          // New limb 3 includes high(q * p[3]).
    // low(q * p[0]) cancels x10 and is discarded.
    mul x20,x24,x4          // x20 = low(next q * p[1]).
    // next q * p[2] is zero.
    lsl x22,x4,#62          // x22 = low(next q * p[3]).
    subs xzr,x10,#1         // Set the low-limb cancellation carry.
    umulh x19,x23,x4        // x19 = high(next q * p[0]).
    adcs x11,x11,x20        // Add low(next q * p[1]) and carry.
    umulh x20,x24,x4        // x20 = high(next q * p[1]).
    adcs x12,x12,xzr        // Propagate carry across zero p[2].
    // high(next q * p[2]) is zero.
    adcs x13,x13,x22        // Add low(next q * p[3]) and carry.
    lsr x22,x4,#2           // x22 = high(next q * p[3]).
    adc x1,xzr,xzr          // Save the carry above limb 3.

    // Shift out cancelled limb 2 and start cancellation 3.
    adds x10,x11,x19        // New limb 0 includes high(q * p[0]).
    adcs x11,x12,x20        // New limb 1 includes high(q * p[1]).
    adcs x12,x13,xzr        // New limb 2; p[2] contributes zero.
    mul x4,x5,x10           // Final q = new limb 0 * inv mod 2^64.
    adc x13,x1,x22          // New limb 3 includes high(q * p[3]).
    // low(q * p[0]) cancels x10 and is discarded.
    mul x20,x24,x4          // x20 = low(final q * p[1]).
    // final q * p[2] is zero.
    lsl x22,x4,#62          // x22 = low(final q * p[3]).
    subs xzr,x10,#1         // Set the low-limb cancellation carry.
    umulh x19,x23,x4        // x19 = high(final q * p[0]).
    adcs x11,x11,x20        // Add low(final q * p[1]) and carry.
    umulh x20,x24,x4        // x20 = high(final q * p[1]).
    adcs x12,x12,xzr        // Propagate carry across zero p[2].
    // high(final q * p[2]) is zero.
    adcs x13,x13,x22        // Add low(final q * p[3]) and carry.
    lsr x22,x4,#2           // x22 = high(final q * p[3]).
    adc x1,xzr,xzr          // Save the carry above limb 3.

    // Shift out cancelled limb 3 to finish dividing the low half by R.
    adds x10,x11,x19        // Reduced limb 0 includes high(q * p[0]).
    adcs x11,x12,x20        // Reduced limb 1 includes high(q * p[1]).
    adcs x12,x13,xzr        // Reduced limb 2; p[2] contributes zero.
    adc x13,x1,x22          // Reduced limb 3 includes high(q * p[3]).
    // Add the upper product half; the sum stays below 2p (see above), so no
    // carry escapes and no conditional subtraction is needed mid-loop.
    adds x6,x10,x14         // Next-iteration a[0].
    adcs x7,x11,x15         // Next-iteration a[1].
    adcs x8,x12,x16         // Next-iteration a[2].
    adc x9,x13,x17          // Next-iteration a[3].

    cbnz x2,L$pasta_curves_sqr_n_loop // Square again if count remains.

    // The loop leaves the lazy accumulator in x6..x9. Move it into the
    // shared canonicalization/output registers without a dummy multiply.
    mov x10,x6
    mov x11,x7
    mov x12,x8
    mov x13,x9
    b L$pasta_curves_sqr_n_canonicalize
