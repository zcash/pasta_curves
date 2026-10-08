// The crate denies unsafe code by default; the assembly backend is the one place that allows
// it, and its safety argument is the operand contracts stated on each routine.
#![allow(unsafe_code)]

//! Assembly backends for the Pasta fields.
//!
//! # Availability
//!
//! When the `asm` feature flag is enabled, the module provides a backend for
//! `target_arch = "aarch64"`, and for `target_arch = "x86_64"` with 64-bit
//! pointers. On x86-64, `add`, `sub`, and `from_mont` are register-only, while
//! `mul`, `square`, and the routines built on them read limbs through pointers;
//! the x32 ABI's 32-bit pointers would break them (see the x86-64 module's docs
//! for why registers alone cannot serve there), so the module has no backend on
//! that target. They also need, at run time, a CPU with BMI2 and ADX (MULX,
//! ADCX/ADOX: Intel Broadwell / AMD Zen or newer); neither is checked.
//! `from_mont` uses MULX (BMI2) alone. Apple x86-64 targets are excluded
//! altogether: they reserve `rbp`, and so have fewer available registers than
//! the squaring blocks need. The AArch64 backend also provides the six blocks
//! of the constant-time inversion, which `crate::inversion` runs.
//!
//! On every other target, that is any target other than AArch64 and non-Apple
//! x86-64 with 64-bit pointers, the module has no backend. The same holds on
//! any target when the compiler is passed `--cfg pasta_curves_noasm` (through
//! `RUSTFLAGS`, or `rustflags` in `.cargo/config.toml`), which is how to build
//! for old x86-64 CPUs without BMI2 and ADX. Code that uses the backend does
//! not repeat these conditions: it declares its uses of the backend under
//! `if_asm_supported!` and its portable fallback under `if_asm_unsupported!`;
//! the first expands to its items exactly where the module has a backend, and
//! the second exactly where it does not. [`BACKEND`] names the result.
//!
//! Nothing is assembled at build time: the blocks are compiled by the Rust
//! toolchain, so no C toolchain is needed, and the module adds no dependency.
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
//! The routines are transcriptions of the Pasta Montgomery routines of
//! Supranational's [Semolina] v0.1.4. See `src/asm/README.md` for the history
//! of the transcription.
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
    /// The assembly backend compiled into this build: `"aarch64"` or `"x86-64"` where the crate
    /// has one, and `"portable"` where it does not. Intended for diagnostics only, such as logs
    /// and benchmark labels; it is not a stable interface.
    pub const BACKEND: &str = if cfg!(target_arch = "aarch64") { "aarch64" } else { "x86-64" };

    #[cfg(any(target_arch = "aarch64", doc))]
    pub(crate) mod aarch64;

    #[cfg(any(target_arch = "x86_64", doc))]
    mod x86_64;

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
    /// The assembly backend compiled into this build: `"aarch64"` or `"x86-64"` where the crate
    /// has one, and `"portable"` where it does not. Intended for diagnostics only, such as logs
    /// and benchmark labels; it is not a stable interface.
    pub const BACKEND: &str = "portable";
}
