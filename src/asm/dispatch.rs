//! Cached x86-64 CPU detection, including in `no_std` builds.

use core::arch::x86_64::{__cpuid, __cpuid_count};
use core::sync::atomic::{AtomicU8, Ordering};

// Zero means not yet detected, one means portable, and two means assembly.
static BACKEND: AtomicU8 = AtomicU8::new(0);

#[inline(always)]
pub(super) fn available() -> bool {
    let backend = BACKEND.load(Ordering::Relaxed);
    backend == 2 || (backend == 0 && detect())
}

#[cold]
fn detect() -> bool {
    // SAFETY: CPUID is available on every x86-64 CPU. Check the maximum basic leaf before
    // querying leaf seven, so CPUs predating that leaf cannot report unrelated feature bits.
    // The intrinsics were unsafe on the MSRV and are safe on newer toolchains.
    #[allow(unused_unsafe)]
    let supported = unsafe {
        __cpuid(0).eax >= 7 && {
            let features = __cpuid_count(7, 0).ebx;
            features & ((1 << 8) | (1 << 19)) == (1 << 8) | (1 << 19)
        }
    };
    // Only this scalar is shared; concurrent first callers may safely repeat the same detection.
    BACKEND.store(if supported { 2 } else { 1 }, Ordering::Relaxed);
    supported
}
