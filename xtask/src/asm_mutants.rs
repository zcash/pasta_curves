#![cfg_attr(
    not(all(target_arch = "aarch64", target_vendor = "apple")),
    allow(dead_code)
)]

use std::process;

mod arch;
mod instructions;
mod mutations;
mod runner;

pub fn run() {
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    let res = runner::run::<arch::AArch64Apple>();

    #[cfg(not(all(target_arch = "aarch64", target_vendor = "apple")))]
    let res: Result<(), _> = Err("asm mutation execution requires Apple AArch64");

    if let Err(error) = res {
        eprintln!("asm-mutants: {error}");
        process::exit(2);
    }
}
