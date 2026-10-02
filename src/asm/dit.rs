//! Arm's data-independent timing (`FEAT_DIT`) around a secret computation: an example for the
//! maintainers to evaluate, compiled only with `--cfg pasta_curves_dit` on an AArch64 target whose
//! CPU baseline has the feature (`target_feature = "dit"`).
//!
//! # What DIT is
//!
//! `PSTATE.DIT` is one bit of the processor state, like the condition flags: not memory, not a
//! build setting. With it set, the architecture requires that "the execution time of a
//! data-independent-time sequence of code must be independent of all data-independent-time values",
//! for an enumerated list of instructions that includes every instruction of the blocks; with it
//! clear, it promises nothing. Apple: certain instructions "may take a different amount of time to
//! run depending on the data values", and with DIT on "the processor uses the longer, worst-case
//! amount of time". On Apple M3, setting DIT also effectively disables the data memory-dependent
//! prefetcher that GoFetch (USENIX Security 2024) exploits; on M1 and M2 it does not.
//!
//! The bit belongs to the running thread: the operating system saves and restores it on a context
//! switch, a new thread starts with it clear, and user code sets it with `msr dit, #1`. Linux sets
//! it while running kernel code only. Nothing in the crate sets it otherwise, so the blocks'
//! constant-timeness, which the Lean model proves relative to the DIT instruction list, is today a
//! hypothesis that the hardware is not asked to honour.
//!
//! # Where to set it
//!
//! Not inside an `asm!` block: the blocks are `pure`, which promises no side effect and lets the
//! compiler move, merge, or remove them, and writing `PSTATE` is a side effect. Around a secret
//! computation instead, saving the caller's value and restoring it after, as Apple's
//! `timingsafe_enable_if_supported` / `timingsafe_restore_if_supported` and Go's
//! `crypto/subtle.WithDataIndependentTiming` do. This example wraps the inversion; the other
//! candidates, each field operation (too fine: millions per proof) or the caller's whole operation
//! (a signature, a proof: the granularity Apple and Go choose), are discussed in the pull request.
//!
//! # Ordering
//!
//! A guard that only executes `msr dit, #1`, the computation, and `msr dit, <saved>` does not
//! work: the blocks are `pure` and `nomem`, so nothing orders them after the first `asm!` or
//! before the second, and the compiler may move them out of the window. [`with_dit`] threads the
//! secret input through the `asm!` that sets DIT and the result through the one that restores it,
//! as `inout` operands, so the computation depends on the first and the second depends on the
//! computation. Work that depends only on public values (the modulus, constants) may still move
//! out of the window, which is harmless.
//!
//! # Detection
//!
//! `msr dit` is an undefined instruction on a CPU without `FEAT_DIT`. `std` detects the feature
//! at run time (`is_aarch64_feature_detected!("dit")`), but the crate is `no_std`, and a
//! user-space read of `ID_AA64PFR0_EL1` traps on macOS. This example therefore relies on the
//! target: `target_feature = "dit"` promises a CPU with the feature. It holds by default on
//! `aarch64-apple-darwin`, and elsewhere is opted into with `-C target-feature=+dit`.

use core::arch::asm;

use crate::limbs::Limbs;

/// Runs `f` on `x` with `PSTATE.DIT` set, and restores the caller's value of the bit after. The
/// input passes through the `asm!` that sets DIT and the result through the one that restores
/// it, so that `f`'s secret-dependent work is ordered between them (see the module's docs).
#[inline(always)]
pub(crate) fn with_dit(x: Limbs, f: impl FnOnce(Limbs) -> Limbs) -> Limbs {
    let [mut x0, mut x1, mut x2, mut x3] = x;
    let saved: u64;
    // SAFETY: `mrs`/`msr` of DIT touch only `PSTATE.DIT`, which the target guarantees to exist
    // (`target_feature = "dit"`); the `isb` makes the new value effective before the next
    // instruction. The limbs pass through unchanged, named in a comment of the template. No
    // memory or stack is used, and the flags are preserved.
    unsafe {
        asm!(
            "mrs {saved}, dit",
            "msr dit, #1",
            "isb",
            "// the input passes through: {x0} {x1} {x2} {x3}",
            saved = out(reg) saved,
            x0 = inout(reg) x0,
            x1 = inout(reg) x1,
            x2 = inout(reg) x2,
            x3 = inout(reg) x3,
            options(nostack, preserves_flags),
        );
    }
    let [mut z0, mut z1, mut z2, mut z3] = f([x0, x1, x2, x3]);
    // SAFETY: as above; `msr dit, <saved>` writes back bit 24 of the value read before.
    unsafe {
        asm!(
            "msr dit, {saved}",
            "// the result passes through: {z0} {z1} {z2} {z3}",
            saved = in(reg) saved,
            z0 = inout(reg) z0,
            z1 = inout(reg) z1,
            z2 = inout(reg) z2,
            z3 = inout(reg) z3,
            options(nostack, preserves_flags),
        );
    }
    [z0, z1, z2, z3]
}

/// The current value of `PSTATE.DIT`, as `0` or `1`, for the tests in `src/inversion.rs`.
#[cfg(test)]
pub(crate) fn dit() -> u64 {
    let value: u64;
    // SAFETY: reads `PSTATE.DIT`, which the target guarantees to exist.
    unsafe { asm!("mrs {}, dit", out(reg) value, options(nomem, nostack, preserves_flags)) };
    (value >> 24) & 1
}

/// Sets `PSTATE.DIT` to `on`, for the tests in `src/inversion.rs`.
#[cfg(test)]
pub(crate) fn set_dit(on: bool) {
    // SAFETY: writes `PSTATE.DIT`, which the target guarantees to exist.
    unsafe {
        if on {
            asm!("msr dit, #1", "isb", options(nomem, nostack, preserves_flags));
        } else {
            asm!("msr dit, #0", "isb", options(nomem, nostack, preserves_flags));
        }
    }
}
