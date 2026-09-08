#![cfg_attr(
    not(all(target_arch = "aarch64", target_vendor = "apple")),
    allow(dead_code)
)]

use std::process;

mod arch;
mod instructions;
mod mutations;
mod runner;
mod survivors;

pub fn run(verify_list_only: bool, list_sites: bool) {
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    let res = runner::run::<arch::AArch64Apple>(verify_list_only, list_sites);

    #[cfg(not(all(target_arch = "aarch64", target_vendor = "apple")))]
    let res: Result<(), _> = {
        let _ = (verify_list_only, list_sites);
        Err("asm mutation execution requires Apple AArch64")
    };

    if let Err(error) = res {
        eprintln!("asm-mutants: {error}");
        process::exit(2);
    }
}
