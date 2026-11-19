// The crate denies unsafe code by default; the assembly backend is the one place that allows
// it, and its safety argument is the operand contracts stated on each routine.
#![allow(unsafe_code)]

//! Assembly backends for the Pasta fields: AArch64, x86-64, and ARMv7E-M.
//!
//! # Availability
//!
//! With the `asm` feature enabled, the ARMv7E-M backend is intended for
//! `thumbv7em-none-eabihf`, where it uses rolling UMAAL multiplication and
//! squaring and dedicated Montgomery conversion; field addition and subtraction
//! remain portable. The AArch64 backend provides all inversion blocks in
//! assembly; the ARM backend uses its `divstep59`, `amontred`, and `cond_sub`
//! blocks alongside portable Rust for the other three.
//!
//! Some ARM targets match the gate but lack the required Thumb-2 or DSP
//! instructions. Pass `--cfg pasta_curves_noasm` through `RUSTFLAGS` or
//! `rustflags` in `.cargo/config.toml` to select portable arithmetic on those
//! targets, or on x86-64 CPUs without BMI2 and ADX. On x86-64, the x32 ABI is
//! excluded because some routines read limbs through pointers, and Apple
//! targets reserve `rbp`, leaving too few registers for the squaring blocks;
//! BMI2 and ADX are required at run time and are not checked.
//!
//! Code that uses the backend does not repeat the gate: it declares its uses
//! under `if_asm_supported!` and its portable fallback under
//! `if_asm_unsupported!`; the first expands to its items exactly where the
//! module has a backend, and the second exactly where it does not. [`BACKEND`]
//! names the result.
//!
//! Nothing is assembled at build time: the blocks are compiled by the Rust
//! toolchain, so no C toolchain is needed. On x86-64, `asm` also enables
//! `cpufeatures` for AVX2 helper dispatch. Hosted targets detect AVX2 at runtime;
//! SGX, freestanding and UEFI targets require compile-time AVX2 guarantees.
//!
//! # Timing
//!
//! The blocks have no data-dependent branch or memory access, and a release
//! build runs nothing else. So the routines' timing should not depend on their
//! operands, unless behaviour of the Rust toolchain or platform introduces an
//! unexpected obstacle to that. A debug build also runs the assertions' checks,
//! and debug mode carries no constant-time guarantee. The checks are written
//! without data-dependent branches, and pass their words through
//! `core::hint::black_box`, as `subtle` does. An inspection of the output of
//! one toolchain (AArch64, Rust 1.96.1) found only the assertions' own branches
//! left, but that is best effort, which the compiler owes nothing to.
//!
//! # Provenance
//!
//! The AArch64 and x86-64 Montgomery routines are transcriptions of
//! Supranational's [Semolina] v0.1.4. The ARMv7E-M implementation is original.
//! See `src/asm/README.md` for the history of the transcriptions.
//!
//! [Semolina]: https://github.com/supranational/semolina

/// Declares items, a local binding, or a block only where this module has a backend.
///
/// The expansion carries the target, presence of the `asm` feature, and absence of
/// `--cfg pasta_curves_noasm`, so code that uses the backend does not repeat the backend
/// condition. Write an extra pair of braces to cfg-gate arbitrary statements in a block:
/// `if_asm_supported! {{ ... }}`. A binding used after the macro must instead be
/// written without the extra braces: `if_asm_supported! { let value = expression; }`.
macro_rules! if_asm_supported {
    (@cfg $($tokens:tt)+) => {
        // The x86-64 multiplication family addresses limbs through pointers, so the backend needs
        // 64-bit pointers; and Apple x86-64 targets reserve `rbp`, and so have fewer available
        // registers than the squaring blocks need.
        #[cfg(all(
            feature = "asm",
            not(pasta_curves_noasm),
            any(
                target_arch = "aarch64",
                all(
                    target_arch = "x86_64",
                    target_pointer_width = "64",
                    not(target_vendor = "apple"),
                ),
                all(
                    target_arch = "arm",
                    any(
                        target_os = "none",
                        all(test, pasta_curves_armv7em_test, target_os = "linux"),
                    ),
                    target_abi = "eabihf",
                    target_pointer_width = "32",
                    target_endian = "little",
                )
            )
        ))]
        $($tokens)+
    };
    ({ $($body:tt)* }) => {
        if_asm_supported! { @cfg { $($body)* } }
    };
    (let $name:ident $(: $ty:ty)? = $value:expr;) => {
        if_asm_supported! { @cfg let $name $(: $ty)? = $value; }
    };
    ($($item:item)*) => { $(
        if_asm_supported! { @cfg $item }
    )* };
}

/// Declares items, a local binding, or a block where this module has no backend.
/// This is the complement of `if_asm_supported!` for portable fallbacks.
macro_rules! if_asm_unsupported {
    (@cfg $($tokens:tt)+) => {
        #[cfg(not(all(
            feature = "asm",
            not(pasta_curves_noasm),
            any(
                target_arch = "aarch64",
                all(
                    target_arch = "x86_64",
                    target_pointer_width = "64",
                    not(target_vendor = "apple"),
                ),
                all(
                    target_arch = "arm",
                    any(
                        target_os = "none",
                        all(test, pasta_curves_armv7em_test, target_os = "linux"),
                    ),
                    target_abi = "eabihf",
                    target_pointer_width = "32",
                    target_endian = "little",
                )
            )
        )))]
        $($tokens)+
    };
    ({ $($body:tt)* }) => {
        if_asm_unsupported! { @cfg { $($body)* } }
    };
    (let $name:ident $(: $ty:ty)? = $value:expr;) => {
        if_asm_unsupported! { @cfg let $name $(: $ty)? = $value; }
    };
    ($($item:item)*) => { $(
        if_asm_unsupported! { @cfg $item }
    )* };
}

if_asm_supported! {
    /// The assembly backend compiled into this build: `"aarch64"`, `"x86-64"`, or `"armv7em"`
    /// where the crate has one, and `"portable"` where it does not. Intended for diagnostics only,
    /// such as logs and benchmark labels; it is not a stable interface.
    pub const BACKEND: &str = if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else if cfg!(target_arch = "arm") {
        "armv7em"
    } else {
        "x86-64"
    };

    #[cfg(any(target_arch = "aarch64", doc))]
    pub(crate) mod aarch64;

    #[cfg(any(target_arch = "x86_64", doc))]
    mod x86_64;

    #[cfg(any(target_arch = "arm", doc))]
    pub(crate) mod armv7em;

    // The tests use std only to catch the debug assertions they check, so a release test
    // build stays free of it.
    #[cfg(all(test, debug_assertions))]
    extern crate std;

    #[cfg(test)]
    mod tests;

    mod entry;
    pub use entry::*;
}

if_asm_unsupported! {
    /// The assembly backend compiled into this build: `"aarch64"`, `"x86-64"`, or `"armv7em"`
    /// where the crate has one, and `"portable"` where it does not. Intended for diagnostics only,
    /// such as logs and benchmark labels; it is not a stable interface.
    pub const BACKEND: &str = "portable";
}
