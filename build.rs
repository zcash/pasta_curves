use std::env;

fn main() {
    println!("cargo:rerun-if-changed=src/asm/pasta_mul-armv8.S");

    // The assembly is Mach-O AArch64. Elsewhere the crate is empty (see the
    // crate-level `cfg` in `src/lib.rs`) and there is nothing to assemble.
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_vendor = env::var("CARGO_CFG_TARGET_VENDOR").unwrap();
    if target_arch == "aarch64" && target_vendor == "apple" {
        cc::Build::new()
            .file("src/asm/pasta_mul-armv8.S")
            .compile("pasta_aarch64_asm");
    }
}
